from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import threading
import time
import uuid
import warnings
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_float, as_int, load_json, normalize_ranks, write_json
from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    require_within,
    validate_json_mapping,
)
from .capture_merge import merge_trace_shards_with_loader


class TraceRecorder:
    """Process-local recorder with fork-safe and race-safe shard writes."""

    def __init__(
        self,
        output_path: str,
        *,
        workload: Optional[Mapping[str, Any]] = None,
        system: Optional[Mapping[str, Any]] = None,
        limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    ) -> None:
        workload_snapshot = _snapshot_json_mapping(workload, "workload", limits=limits)
        system_snapshot = _snapshot_json_mapping(system, "system", limits=limits)
        self._limits = limits
        self._requested_output_path = output_path
        self._recorder_id = uuid.uuid4().hex
        self._trace_root = _configured_trace_root()
        self.output_path = _resolve_output_path(
            output_path,
            recorder_id=self._recorder_id,
            trace_root=self._trace_root,
        )
        self._direct_claim: Optional[Tuple[Path, Path]] = None
        self._claim_finalizer: Optional[weakref.finalize[..., TraceRecorder]] = None
        if self._trace_root is None and os.environ.get("COMMCANARY_TRACE_SHARDED") != "1":
            self._direct_claim = _claim_direct_output_path(
                Path(self.output_path),
                recorder_id=self._recorder_id,
            )
            target, claim_path = self._direct_claim
            self._claim_finalizer = weakref.finalize(
                self,
                _release_direct_output_claim,
                target,
                claim_path,
                self._recorder_id,
            )
        self.workload = workload_snapshot
        self.system = system_snapshot
        self.events: List[JsonDict] = []
        self._start_ns = time.perf_counter_ns()
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._generation = 0
        self._last_saved_generation = -1
        self._closed = False
        self._session_id = os.environ.get("COMMCANARY_CAPTURE_SESSION_ID", str(uuid.uuid4()))
        _RECORDERS.add(self)

    @classmethod
    def from_env(
        cls,
        *,
        workload: Optional[Mapping[str, Any]] = None,
        system: Optional[Mapping[str, Any]] = None,
        limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    ) -> "TraceRecorder":
        output_path = os.environ.get("COMMCANARY_TRACE_DIR") or os.environ.get("COMMCANARY_TRACE_OUT")
        if not output_path:
            output_path = "commcanary.trace.json"
        return cls(output_path, workload=workload, system=system, limits=limits)

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc is None:
                self.save()
                return
            try:
                self.save()
            except Exception as save_error:
                _report_suppressed_save_error(exc, save_error)
        finally:
            self.close()

    def elapsed_us(self) -> float:
        return (time.perf_counter_ns() - self._start_ns) / 1000.0

    def record_collective(
        self,
        *,
        op: str,
        bytes: int,
        ranks: List[int],
        phase: str = "unknown",
        group: str = "default",
        start_us: Optional[float] = None,
        rank_arrival_us: Optional[Mapping[Any, float]] = None,
        arrival_skew_us: Optional[float] = None,
        compute_before_us: float = 0.0,
        compute_overlap_us: float = 0.0,
        compute_pressure: float = 0.5,
        concurrent_groups: int = 1,
        collective_id: Optional[str] = None,
        sender_rank: Optional[int] = None,
        receiver_rank: Optional[int] = None,
        tag: Optional[str] = None,
        channel: Optional[str] = None,
        message_sequence: Optional[int] = None,
        observed_exposed_us: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._require_open()
        with self._lock:
            self._ensure_current_process_locked()
            self._require_event_capacity_locked()
        if not isinstance(op, str) or not op.strip():
            raise SchemaError("op must be a non-empty string")
        parsed_bytes = as_int(bytes)
        if parsed_bytes <= 0:
            raise SchemaError("bytes must be positive")
        try:
            require_within(len(ranks), self._limits.max_ranks, label="trace recorder ranks")
        except (JsonResourceError, TypeError) as exc:
            raise SchemaError(str(exc)) from exc
        parsed_ranks = normalize_ranks(ranks)
        parsed_start = self.elapsed_us() if start_us is None else as_float(start_us)
        parsed_before = as_float(compute_before_us)
        parsed_overlap = as_float(compute_overlap_us)
        parsed_pressure = as_float(compute_pressure)
        parsed_groups = as_int(concurrent_groups)
        if parsed_start < 0.0:
            raise SchemaError("start_us must be non-negative")
        if parsed_before < 0.0 or parsed_overlap < 0.0 or parsed_pressure < 0.0:
            raise SchemaError("compute timing and pressure values must be non-negative")
        if parsed_groups <= 0:
            raise SchemaError("concurrent_groups must be positive")
        parsed_observed: Optional[float] = None
        if observed_exposed_us is not None:
            parsed_observed = as_float(observed_exposed_us)
            if parsed_observed < 0.0:
                raise SchemaError("observed_exposed_us must be non-negative")

        metadata_snapshot: Optional[JsonDict] = None
        if metadata is not None:
            metadata_snapshot = _snapshot_json_mapping(metadata, "metadata", limits=self._limits)

        parsed_sender = as_int(sender_rank) if sender_rank is not None else None
        parsed_receiver = as_int(receiver_rank) if receiver_rank is not None else None
        if parsed_sender is not None and parsed_sender not in parsed_ranks:
            raise SchemaError("sender_rank must be one of ranks")
        if parsed_receiver is not None and parsed_receiver not in parsed_ranks:
            raise SchemaError("receiver_rank must be one of ranks")
        if parsed_sender is not None and parsed_receiver is not None and parsed_sender == parsed_receiver:
            raise SchemaError("sender_rank and receiver_rank must differ")
        parsed_message_sequence = as_int(message_sequence) if message_sequence is not None else None
        if parsed_message_sequence is not None and parsed_message_sequence < 0:
            raise SchemaError("message_sequence must be non-negative")
        parsed_tag = str(tag) if tag is not None else None
        parsed_channel = str(channel) if channel is not None else None
        if parsed_tag == "" or parsed_channel == "":
            raise SchemaError("tag and channel must be non-empty when provided")

        parsed_arrivals: Optional[JsonDict] = None
        parsed_skew: Optional[float] = None
        if rank_arrival_us is not None:
            if not isinstance(rank_arrival_us, Mapping) or not rank_arrival_us:
                raise SchemaError("rank_arrival_us must be a non-empty mapping")
            expected = {str(rank) for rank in parsed_ranks}
            parsed_arrivals = {}
            for key, value in rank_arrival_us.items():
                rank_key = str(as_int(key))
                if rank_key not in expected:
                    raise SchemaError("rank_arrival_us contains a rank outside ranks")
                parsed_value = as_float(value)
                if parsed_value < 0.0:
                    raise SchemaError("rank_arrival_us values must be non-negative")
                if rank_key in parsed_arrivals:
                    raise SchemaError("rank_arrival_us contains duplicate rank keys")
                parsed_arrivals[rank_key] = parsed_value
        elif arrival_skew_us is not None:
            parsed_skew = as_float(arrival_skew_us)
            if parsed_skew < 0.0:
                raise SchemaError("arrival_skew_us must be non-negative")
            if len(parsed_ranks) == 1 and parsed_skew > 0.001:
                raise SchemaError("a one-rank collective cannot have positive arrival skew")

        with self._lock:
            self._ensure_current_process_locked()
            self._require_event_capacity_locked()
            sequence = len(self.events)
            event: JsonDict = {
                "id": f"event-{sequence:06d}",
                "capture_session_id": self._session_id,
                "collective_seq": sequence,
                "recorder_rank": _rank_label(),
                "phase": str(phase),
                "op": op,
                "bytes": parsed_bytes,
                "ranks": parsed_ranks,
                "group": str(group),
                "start_us": round(parsed_start, 9),
                "compute_before_us": round(parsed_before, 9),
                "compute_overlap_us": round(parsed_overlap, 9),
                "compute_pressure": round(parsed_pressure, 6),
                "concurrent_groups": parsed_groups,
            }
            if collective_id is not None:
                collective_text = str(collective_id)
                if not collective_text:
                    raise SchemaError("collective_id must not be empty")
                event["collective_id"] = collective_text
            if parsed_sender is not None:
                event["sender_rank"] = parsed_sender
            if parsed_receiver is not None:
                event["receiver_rank"] = parsed_receiver
            if parsed_tag is not None:
                event["tag"] = parsed_tag
            if parsed_channel is not None:
                event["channel"] = parsed_channel
            if parsed_message_sequence is not None:
                event["message_sequence"] = parsed_message_sequence
            if parsed_arrivals is not None:
                event["rank_arrival_us"] = parsed_arrivals
                if set(parsed_arrivals) != {str(rank) for rank in parsed_ranks}:
                    event["partial_rank_arrival"] = True
            elif parsed_skew is not None:
                event["arrival_skew_us"] = parsed_skew
            if parsed_observed is not None:
                event["observed_exposed_us"] = round(parsed_observed, 9)
            if metadata_snapshot:
                event["metadata"] = metadata_snapshot
            self.events.append(event)
            self._generation += 1

    def to_trace(self) -> JsonDict:
        self._require_open()
        with self._lock:
            self._ensure_current_process_locked()
            return self._to_trace_locked(copy.deepcopy(self.events))

    def _to_trace_locked(self, events: List[JsonDict]) -> JsonDict:
        return {
            "format": TRACE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workload": copy.deepcopy(self.workload),
            "system": {
                **copy.deepcopy(self.system),
                "pid": self._pid,
                "rank": _rank_label(),
                "capture_session_id": self._session_id,
                "recorder_id": self._recorder_id,
            },
            "events": events,
        }

    def save(self) -> None:
        self._require_open()
        with self._lock:
            self._ensure_current_process_locked()
            generation = self._generation
            snapshot = copy.deepcopy(self.events)
            trace = self._to_trace_locked(snapshot)
            output_path = self.output_path
            trace_root = self._trace_root
        validate_trace(trace, allow_partial_arrivals=True, limits=self._limits)
        with self._save_lock:
            if generation <= self._last_saved_generation:
                return
            if trace_root is not None:
                _require_path_below_root(Path(output_path), trace_root)
            write_json(output_path, trace)
            self._last_saved_generation = generation

    def close(self) -> None:
        """Release direct-output ownership; a closed recorder cannot be reused."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            finalizer = self._claim_finalizer
            self._claim_finalizer = None
            self._direct_claim = None
        if finalizer is not None and finalizer.alive:
            finalizer()

    def _require_open(self) -> None:
        if self._closed:
            raise SchemaError("trace recorder is closed")

    def _ensure_current_process_locked(self) -> None:
        if os.getpid() != self._pid:
            self._reset_for_current_process_locked()

    def _require_event_capacity_locked(self) -> None:
        try:
            next_count = checked_add(len(self.events), 1, label="trace recorder events")
            require_within(
                next_count,
                self._limits.max_capture_events,
                label="trace recorder events",
            )
            require_within(
                next_count,
                self._limits.max_stored_events,
                label="trace recorder stored events",
            )
        except JsonResourceError as exc:
            raise SchemaError(str(exc)) from exc

    def reset_after_fork(self) -> None:
        self._reset_after_fork_in_child()

    def _reset_after_fork_in_child(self) -> None:
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        with self._lock:
            self._reset_for_current_process_locked()

    def _reset_for_current_process_locked(self) -> None:
        if self._claim_finalizer is not None:
            # A forked child must not remove the parent's direct-output claim.
            self._claim_finalizer.detach()
            self._claim_finalizer = None
            self._direct_claim = None
        self._pid = os.getpid()
        self._start_ns = time.perf_counter_ns()
        self.events = []
        self._generation = 0
        self._last_saved_generation = -1
        self._closed = False
        self._recorder_id = uuid.uuid4().hex
        self._trace_root = _configured_trace_root()
        self.output_path = _resolve_output_path(
            self._requested_output_path,
            force_shard=True,
            recorder_id=self._recorder_id,
            trace_root=self._trace_root,
        )


class NullRecorder:
    def record_collective(self, **kwargs: Any) -> None:
        return None

    def save(self) -> None:
        return None

    def close(self) -> None:
        return None


_AUTO_RECORDER: Optional[Any] = None
_AUTO_RECORDER_LOCK = threading.Lock()
_AUTO_RECORDER_SIGNATURE: Optional[Tuple[Optional[str], str, Optional[str], str, bool]] = None
_AUTO_RECORDER_ATEXIT_REGISTERED = False
_RECORDERS: "weakref.WeakSet[TraceRecorder]" = weakref.WeakSet()
_DIRECT_OUTPUT_CLAIMS: Dict[Path, str] = {}
_DIRECT_OUTPUT_CLAIM_LOCK = threading.Lock()


def _claim_direct_output_path(
    output_path: Path,
    *,
    recorder_id: str,
) -> Tuple[Path, Path]:
    """Claim a non-sharded output in this process and across processes."""

    try:
        target = output_path.expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError) as exc:
        raise SchemaError(f"cannot resolve direct trace output {output_path}: {exc}") from exc
    claim_path = target.with_name(f".{target.name}.commcanary.lock")
    with _DIRECT_OUTPUT_CLAIM_LOCK:
        if target in _DIRECT_OUTPUT_CLAIMS:
            raise SchemaError(f"direct trace output already has an active recorder: {target}")
        try:
            descriptor = os.open(
                claim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise SchemaError(
                f"direct trace output is claimed by another process: {target}; "
                f"remove stale claim {claim_path} only after confirming no recorder is active"
            ) from exc
        except OSError as exc:
            raise SchemaError(f"cannot claim direct trace output {target}: {exc}") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(f"{os.getpid()} {recorder_id}\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(claim_path)
            except OSError:
                pass
            raise
        _DIRECT_OUTPUT_CLAIMS[target] = recorder_id
    return target, claim_path


def _release_direct_output_claim(
    target: Path,
    claim_path: Path,
    recorder_id: str,
) -> None:
    """Release a claim only when it still belongs to this recorder."""

    try:
        with _DIRECT_OUTPUT_CLAIM_LOCK:
            if _DIRECT_OUTPUT_CLAIMS.get(target) == recorder_id:
                _DIRECT_OUTPUT_CLAIMS.pop(target, None)
            try:
                owner = claim_path.read_text(encoding="ascii")
            except (OSError, UnicodeError):
                return
            if owner.strip().endswith(recorder_id):
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
    except Exception:
        # Finalizers may run during interpreter teardown. Ownership cleanup is
        # best effort there; normal close paths surface claim errors earlier.
        pass


def get_recorder() -> Any:
    global _AUTO_RECORDER, _AUTO_RECORDER_ATEXIT_REGISTERED, _AUTO_RECORDER_SIGNATURE
    output_path = os.environ.get("COMMCANARY_TRACE_DIR") or os.environ.get("COMMCANARY_TRACE_OUT")
    signature = _auto_recorder_environment_signature()
    with _AUTO_RECORDER_LOCK:
        if (
            _AUTO_RECORDER is not None
            and _AUTO_RECORDER_SIGNATURE == signature
            and not (isinstance(_AUTO_RECORDER, TraceRecorder) and _AUTO_RECORDER._closed)
        ):
            return _AUTO_RECORDER
        previous = _AUTO_RECORDER
        if isinstance(previous, TraceRecorder) and not previous._closed:
            previous.save()
            previous.close()
        if not output_path:
            _AUTO_RECORDER = NullRecorder()
            _AUTO_RECORDER_SIGNATURE = signature
            if not _AUTO_RECORDER_ATEXIT_REGISTERED:
                atexit.register(_close_auto_recorder_at_exit)
                _AUTO_RECORDER_ATEXIT_REGISTERED = True
            return _AUTO_RECORDER
        workload = {"name": os.environ.get("COMMCANARY_WORKLOAD_NAME", "instrumented-workload")}
        _AUTO_RECORDER = TraceRecorder(output_path, workload=workload)
        _AUTO_RECORDER_SIGNATURE = signature
        if not _AUTO_RECORDER_ATEXIT_REGISTERED:
            atexit.register(_close_auto_recorder_at_exit)
            _AUTO_RECORDER_ATEXIT_REGISTERED = True
        return _AUTO_RECORDER


def _auto_recorder_environment_signature() -> Tuple[Optional[str], str, Optional[str], str, bool]:
    output_path = os.environ.get("COMMCANARY_TRACE_DIR") or os.environ.get("COMMCANARY_TRACE_OUT")
    return (
        output_path,
        os.environ.get("COMMCANARY_WORKLOAD_NAME", "instrumented-workload"),
        os.environ.get("COMMCANARY_CAPTURE_SESSION_ID"),
        _rank_label(),
        os.environ.get("COMMCANARY_TRACE_SHARDED") == "1",
    )


def _close_auto_recorder_at_exit() -> None:
    recorder = _AUTO_RECORDER
    if isinstance(recorder, TraceRecorder) and not recorder._closed:
        try:
            recorder.save()
        finally:
            recorder.close()


def record_collective(
    *,
    op: str,
    ranks: List[int],
    byte_count: Optional[int] = None,
    bytes: Optional[int] = None,
    phase: str = "unknown",
    group: str = "default",
    start_us: Optional[float] = None,
    rank_arrival_us: Optional[Mapping[Any, float]] = None,
    arrival_skew_us: Optional[float] = None,
    compute_before_us: float = 0.0,
    compute_overlap_us: float = 0.0,
    compute_pressure: float = 0.5,
    concurrent_groups: int = 1,
    collective_id: Optional[str] = None,
    sender_rank: Optional[int] = None,
    receiver_rank: Optional[int] = None,
    tag: Optional[str] = None,
    channel: Optional[str] = None,
    message_sequence: Optional[int] = None,
    observed_exposed_us: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record one collective through the environment-managed recorder.

    ``byte_count`` is the preferred spelling. The historical ``bytes=`` keyword
    remains supported, but callers must provide exactly one spelling.
    """

    if byte_count is None and bytes is None:
        raise SchemaError("record_collective requires byte_count (or the compatible bytes keyword)")
    if byte_count is not None and bytes is not None:
        raise SchemaError("record_collective accepts only one of byte_count and bytes")
    resolved_byte_count = byte_count if byte_count is not None else bytes
    if resolved_byte_count is None:
        raise SchemaError("record_collective requires a byte count")
    get_recorder().record_collective(
        op=op,
        bytes=resolved_byte_count,
        ranks=ranks,
        phase=phase,
        group=group,
        start_us=start_us,
        rank_arrival_us=rank_arrival_us,
        arrival_skew_us=arrival_skew_us,
        compute_before_us=compute_before_us,
        compute_overlap_us=compute_overlap_us,
        compute_pressure=compute_pressure,
        concurrent_groups=concurrent_groups,
        collective_id=collective_id,
        sender_rank=sender_rank,
        receiver_rank=receiver_rank,
        tag=tag,
        channel=channel,
        message_sequence=message_sequence,
        observed_exposed_us=observed_exposed_us,
        metadata=metadata,
    )


def merge_trace_shards(
    shard_dir: str,
    *,
    workload_name: str,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Merge capture shards while preserving the historical loader patch seam."""

    return merge_trace_shards_with_loader(
        shard_dir,
        workload_name=workload_name,
        limits=limits,
        load_trace=load_json,
    )


def _resolve_output_path(
    output_path: str,
    *,
    force_shard: bool = False,
    recorder_id: Optional[str] = None,
    trace_root: Optional[Path] = None,
) -> str:
    recorder_suffix = recorder_id or uuid.uuid4().hex
    configured_root = trace_root if trace_root is not None else _configured_trace_root()
    if configured_root is not None:
        rank_component = _rank_filename_component(_rank_label())
        candidate = configured_root / (f"rank-{rank_component}-pid-{os.getpid()}-rec-{recorder_suffix}.trace.json")
        _require_path_below_root(candidate, configured_root)
        try:
            configured_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SchemaError(f"cannot create trace directory {configured_root}: {exc}") from exc
        _require_path_below_root(candidate, configured_root)
        return str(candidate)
    if force_shard or os.environ.get("COMMCANARY_TRACE_SHARDED") == "1":
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(
            target.with_name(
                f"{target.stem}.rank-{_rank_filename_component(_rank_label())}"
                f"-pid-{os.getpid()}-rec-{recorder_suffix}{target.suffix}"
            )
        )
    return output_path


def _configured_trace_root() -> Optional[Path]:
    trace_dir = os.environ.get("COMMCANARY_TRACE_DIR")
    if not trace_dir:
        return None
    try:
        return Path(trace_dir).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SchemaError(f"cannot resolve trace directory {trace_dir!r}: {exc}") from exc


def _require_path_below_root(path: Path, root: Path) -> Path:
    """Resolve *path* and require it to remain a child of frozen *root*."""

    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SchemaError(f"cannot resolve capture shard path {path}: {exc}") from exc
    if resolved_path == root or root not in resolved_path.parents:
        raise SchemaError(f"capture shard path {resolved_path} escapes configured trace directory {root}")
    return resolved_path


def _rank_filename_component(label: str) -> str:
    """Return a bounded ASCII filename component without exposing unsafe text."""

    if label == "unknown":
        return label
    if 1 <= len(label) <= 20 and label.isascii() and label.isdecimal():
        return label
    digest = hashlib.sha256(label.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
    return f"label-{digest}"


def _snapshot_json_mapping(
    value: Optional[Mapping[str, Any]],
    label: str,
    *,
    limits: ResourceLimits,
) -> JsonDict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    try:
        validate_json_mapping(value, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"{label} must be JSON serializable: {exc}") from exc
    snapshot = copy.deepcopy(dict(value))
    _validate_json_serializable(snapshot, label)
    return snapshot


def _validate_json_serializable(value: Any, label: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise SchemaError(f"{label} must be JSON serializable: {exc}") from exc


def _report_suppressed_save_error(workload_error: BaseException, save_error: Exception) -> None:
    try:
        detail = str(save_error)
    except Exception:
        detail = type(save_error).__name__
    message = f"CommCanary failed to save its capture while handling an exception: {detail}"
    try:
        setattr(workload_error, "commcanary_save_error", save_error)
    except Exception:
        pass
    add_note = getattr(workload_error, "add_note", None)
    if callable(add_note):
        try:
            add_note(message)
        except Exception:
            pass
    else:
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        except Exception:
            # Warning filters may promote warnings to exceptions. The workload
            # exception must remain authoritative even in that configuration.
            pass


def _rank_label() -> str:
    for name in ("COMMCANARY_RANK", "RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value != "":
            return str(value)
    return "unknown"


def _reset_auto_recorder_after_fork() -> None:
    global _AUTO_RECORDER_LOCK, _AUTO_RECORDER_SIGNATURE
    global _DIRECT_OUTPUT_CLAIM_LOCK, _DIRECT_OUTPUT_CLAIMS
    _AUTO_RECORDER_LOCK = threading.Lock()
    _DIRECT_OUTPUT_CLAIM_LOCK = threading.Lock()
    _DIRECT_OUTPUT_CLAIMS = {}
    auto_recorder = _AUTO_RECORDER
    if auto_recorder is not None and hasattr(auto_recorder, "reset_after_fork"):
        auto_recorder.reset_after_fork()
    for recorder in list(_RECORDERS):
        if recorder is not _AUTO_RECORDER:
            recorder._reset_after_fork_in_child()
    _AUTO_RECORDER_SIGNATURE = _auto_recorder_environment_signature()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_auto_recorder_after_fork)


__all__ = [
    "NullRecorder",
    "TraceRecorder",
    "get_recorder",
    "merge_trace_shards",
    "record_collective",
]
