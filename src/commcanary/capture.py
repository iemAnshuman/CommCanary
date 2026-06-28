from __future__ import annotations

import atexit
import copy
import glob
import os
import threading
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .schema import (
    TRACE_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    load_json,
    normalize_ranks,
    validate_trace,
    write_json,
)


class TraceRecorder:
    """Process-local recorder with fork-safe and race-safe shard writes."""

    def __init__(
        self,
        output_path: str,
        *,
        workload: Optional[Mapping[str, Any]] = None,
        system: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._requested_output_path = output_path
        self._recorder_id = uuid.uuid4().hex
        self.output_path = _resolve_output_path(output_path, recorder_id=self._recorder_id)
        self.workload = copy.deepcopy(dict(workload or {}))
        self.system = copy.deepcopy(dict(system or {}))
        self.events: List[JsonDict] = []
        self._start_ns = time.perf_counter_ns()
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._generation = 0
        self._last_saved_generation = -1
        self._session_id = os.environ.get("COMMCANARY_CAPTURE_SESSION_ID", str(uuid.uuid4()))
        _RECORDERS.add(self)

    @classmethod
    def from_env(
        cls,
        *,
        workload: Optional[Mapping[str, Any]] = None,
        system: Optional[Mapping[str, Any]] = None,
    ) -> "TraceRecorder":
        output_path = os.environ.get("COMMCANARY_TRACE_DIR") or os.environ.get("COMMCANARY_TRACE_OUT")
        if not output_path:
            output_path = "commcanary.trace.json"
        return cls(output_path, workload=workload, system=system)

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.save()

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
        observed_exposed_us: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(op, str) or not op.strip():
            raise SchemaError("op must be a non-empty string")
        parsed_bytes = as_int(bytes)
        if parsed_bytes <= 0:
            raise SchemaError("bytes must be positive")
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
            if parsed_arrivals is not None:
                event["rank_arrival_us"] = parsed_arrivals
                if set(parsed_arrivals) != {str(rank) for rank in parsed_ranks}:
                    event["partial_rank_arrival"] = True
            elif parsed_skew is not None:
                event["arrival_skew_us"] = parsed_skew
            if parsed_observed is not None:
                event["observed_exposed_us"] = round(parsed_observed, 9)
            if metadata:
                if not isinstance(metadata, Mapping):
                    raise SchemaError("metadata must be an object")
                event["metadata"] = copy.deepcopy(dict(metadata))
            self.events.append(event)
            self._generation += 1

    def to_trace(self) -> JsonDict:
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
        with self._lock:
            self._ensure_current_process_locked()
            generation = self._generation
            snapshot = copy.deepcopy(self.events)
            trace = self._to_trace_locked(snapshot)
            output_path = self.output_path
        validate_trace(trace, allow_partial_arrivals=True)
        with self._save_lock:
            if generation <= self._last_saved_generation:
                return
            write_json(output_path, trace)
            self._last_saved_generation = generation

    def _ensure_current_process_locked(self) -> None:
        if os.getpid() != self._pid:
            self._reset_for_current_process_locked()

    def reset_after_fork(self) -> None:
        self._reset_after_fork_in_child()

    def _reset_after_fork_in_child(self) -> None:
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        with self._lock:
            self._reset_for_current_process_locked()

    def _reset_for_current_process_locked(self) -> None:
        self._pid = os.getpid()
        self._start_ns = time.perf_counter_ns()
        self.events = []
        self._generation = 0
        self._last_saved_generation = -1
        self._recorder_id = uuid.uuid4().hex
        self.output_path = _resolve_output_path(
            self._requested_output_path,
            force_shard=True,
            recorder_id=self._recorder_id,
        )


class NullRecorder:
    def record_collective(self, **kwargs: Any) -> None:
        return None

    def save(self) -> None:
        return None


_AUTO_RECORDER: Optional[Any] = None
_AUTO_RECORDER_LOCK = threading.Lock()
_RECORDERS: "weakref.WeakSet[TraceRecorder]" = weakref.WeakSet()


def get_recorder() -> Any:
    global _AUTO_RECORDER
    output_path = os.environ.get("COMMCANARY_TRACE_DIR") or os.environ.get("COMMCANARY_TRACE_OUT")
    with _AUTO_RECORDER_LOCK:
        if _AUTO_RECORDER is not None and not (
            isinstance(_AUTO_RECORDER, NullRecorder) and output_path
        ):
            return _AUTO_RECORDER
        if not output_path:
            _AUTO_RECORDER = NullRecorder()
            return _AUTO_RECORDER
        workload = {"name": os.environ.get("COMMCANARY_WORKLOAD_NAME", "instrumented-workload")}
        _AUTO_RECORDER = TraceRecorder(output_path, workload=workload)
        atexit.register(_AUTO_RECORDER.save)
        return _AUTO_RECORDER


def record_collective(**kwargs: Any) -> None:
    get_recorder().record_collective(**kwargs)


def merge_trace_shards(shard_dir: str, *, workload_name: str) -> JsonDict:
    shard_paths = _trace_shard_paths(shard_dir)
    buckets: Dict[Tuple[Any, ...], List[JsonDict]] = {}
    systems: List[JsonDict] = []
    workload: JsonDict = {"name": workload_name}
    strict_sharded_merge = len(shard_paths) > 1
    seen_occurrences: Dict[Tuple[str, str, str], str] = {}
    session_ids: Set[str] = set()
    canonical_workload: Optional[JsonDict] = None

    for shard_path in shard_paths:
        trace = load_json(shard_path)
        validate_trace(trace, allow_partial_arrivals=True)
        trace_workload = dict(trace.get("workload", {}))
        if canonical_workload is None:
            canonical_workload = trace_workload
        elif trace_workload != canonical_workload:
            raise SchemaError("trace shards contain conflicting workload metadata")
        workload.update(trace_workload)

        system = trace.get("system", {})
        if not isinstance(system, Mapping):
            raise SchemaError("trace shard system metadata must be an object")
        system_session = system.get("capture_session_id")
        if system_session:
            session_ids.add(str(system_session))
        clock_offset_us = _clock_offset_us(system)
        clock_alignment = "explicit_offset_us" if clock_offset_us is not None else "uncalibrated"
        systems.append({**system, "clock_alignment": clock_alignment, "clock_offset_us": clock_offset_us})
        shard_name = Path(shard_path).name

        for event in trace.get("events", []):
            session = event.get("capture_session_id")
            collective_id = event.get("collective_id")
            if session:
                session_ids.add(str(session))
            if system_session and session and str(system_session) != str(session):
                raise SchemaError("event capture_session_id conflicts with shard system metadata")
            if strict_sharded_merge:
                if not session:
                    raise SchemaError("sharded capture merge requires each event to include capture_session_id")
                if not collective_id:
                    raise SchemaError("sharded capture merge requires each event to include a stable collective_id")
                occurrence_key = (str(session), str(collective_id), shard_name)
                if occurrence_key in seen_occurrences:
                    raise SchemaError(
                        f"collective_id {collective_id!r} was reused in shard {shard_name}; "
                        "use a globally unique occurrence id"
                    )
                seen_occurrences[occurrence_key] = str(event.get("id", "unknown"))

            copied = dict(event)
            copied["shard"] = shard_name
            system_rank = system.get("rank")
            copied["_shard_rank"] = (
                system_rank
                if system_rank not in (None, "", "unknown")
                else event.get("recorder_rank", "unknown")
            )
            copied["_clock_calibrated"] = clock_offset_us is not None
            copied["_strict_sharded_merge"] = strict_sharded_merge
            copied["_aligned_start_us"] = as_float(event.get("start_us"), 0.0) + (clock_offset_us or 0.0)
            buckets.setdefault(_coalesce_key(copied), []).append(copied)

    if len(session_ids) > 1:
        raise SchemaError("trace shard directory must contain exactly one capture session")
    if strict_sharded_merge and len(session_ids) != 1:
        raise SchemaError("trace shard directory must contain exactly one capture session")

    events = [_coalesce_events(group) for group in buckets.values()]
    _reject_unordered_uncalibrated_domains(events, systems)
    events.sort(key=lambda event: as_float(event.get("start_us"), 0.0))
    merged: JsonDict = {
        "format": TRACE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": workload,
        "system": {
            "capture_mode": "sharded",
            "shards": len(shard_paths),
            "capture_session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
            "shard_systems": systems,
        },
        "events": events,
    }
    validate_trace(merged)
    return merged


def _resolve_output_path(
    output_path: str,
    *,
    force_shard: bool = False,
    recorder_id: Optional[str] = None,
) -> str:
    recorder_suffix = recorder_id or uuid.uuid4().hex
    trace_dir = os.environ.get("COMMCANARY_TRACE_DIR")
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(
            trace_dir,
            f"rank-{_rank_label()}-pid-{os.getpid()}-rec-{recorder_suffix}.trace.json",
        )
    if force_shard or os.environ.get("COMMCANARY_TRACE_SHARDED") == "1":
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(
            target.with_name(
                f"{target.stem}.rank-{_rank_label()}-pid-{os.getpid()}-rec-{recorder_suffix}{target.suffix}"
            )
        )
    return output_path


def _rank_label() -> str:
    for name in ("COMMCANARY_RANK", "RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value != "":
            return str(value)
    return "unknown"


def _trace_shard_paths(shard_dir: str) -> List[str]:
    patterns = ("*.trace.json", "*.trace.*.json", "*.rank-*-pid-*.json")
    paths = set()
    for pattern in patterns:
        paths.update(glob.glob(os.path.join(shard_dir, pattern)))
    return sorted(paths)


def _reject_unordered_uncalibrated_domains(events: List[JsonDict], systems: List[JsonDict]) -> None:
    if len(events) < 2 or len(systems) < 2:
        return
    if all(system.get("clock_alignment") == "explicit_offset_us" for system in systems):
        return
    rank_domains = [set(normalize_ranks(event.get("ranks"))) for event in events]
    for left_index, left in enumerate(rank_domains):
        for right in rank_domains[left_index + 1:]:
            if left and right and left != right:
                raise SchemaError(
                    "cannot globally order different collective rank domains from uncalibrated rank clocks"
                )


def _clock_offset_us(system: Mapping[str, Any]) -> Optional[float]:
    if "clock_offset_us" in system:
        return as_float(system.get("clock_offset_us"))
    calibration = system.get("clock_calibration")
    if isinstance(calibration, Mapping) and "offset_us" in calibration:
        return as_float(calibration.get("offset_us"))
    return None


def _reset_auto_recorder_after_fork() -> None:
    global _AUTO_RECORDER_LOCK
    _AUTO_RECORDER_LOCK = threading.Lock()
    if hasattr(_AUTO_RECORDER, "reset_after_fork"):
        _AUTO_RECORDER.reset_after_fork()
    for recorder in list(_RECORDERS):
        if recorder is not _AUTO_RECORDER:
            recorder._reset_after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_auto_recorder_after_fork)


def _coalesce_key(event: Mapping[str, Any]) -> Tuple[Any, ...]:
    session = event.get("capture_session_id")
    collective_id = event.get("collective_id")
    if session is None or collective_id is None:
        return ("uncoalesced", event.get("shard"), event.get("id"))
    return session, collective_id


def _coalesce_events(events: List[JsonDict]) -> JsonDict:
    if len(events) == 1:
        event = dict(events[0])
        if event.get("_strict_sharded_merge") and len(normalize_ranks(event.get("ranks"))) > 1:
            raise SchemaError("missing rank contributions for sharded collective")
        event["start_us"] = round(as_float(event.get("_aligned_start_us"), event.get("start_us", 0.0)), 9)
        for key in ("_shard_rank", "_aligned_start_us", "_clock_calibrated", "_strict_sharded_merge"):
            event.pop(key, None)
        return event

    identity_event = events[0]
    ranks = normalize_ranks(identity_event.get("ranks"))
    expected_ranks = {str(rank) for rank in ranks}
    merged_op, point_to_point = _validate_collective_identity(events, ranks)
    recorder_ranks = _recorder_ranks(events, expected_ranks)
    calibrated_values = {bool(event.get("_clock_calibrated")) for event in events}
    if len(calibrated_values) > 1:
        raise SchemaError("cannot merge mixed calibrated and uncalibrated rank clocks")
    calibrated = calibrated_values == {True}
    starts = [as_float(event.get("_aligned_start_us"), 0.0) for event in events]
    if calibrated:
        timeline_event = min(events, key=lambda event: as_float(event.get("_aligned_start_us"), 0.0))
        coalesced_start_us = min(starts)
    else:
        canonical_rank = min(as_int(rank) for rank in recorder_ranks)
        timeline_event = _event_for_recorder_rank(events, canonical_rank)
        coalesced_start_us = as_float(timeline_event.get("_aligned_start_us"), 0.0)

    coalesced: JsonDict = {
        key: value
        for key, value in timeline_event.items()
        if not key.startswith("_") and key not in {"shard", "rank_arrival_us", "recorder_rank"}
    }
    coalesced["id"] = f"collective-{timeline_event.get('collective_id', timeline_event.get('collective_seq', 0))}"
    coalesced["op"] = merged_op
    coalesced["start_us"] = round(coalesced_start_us, 9)
    if point_to_point is not None:
        coalesced["sender_rank"] = point_to_point["sender_rank"]
        coalesced["receiver_rank"] = point_to_point["receiver_rank"]

    raw_map_events = [event for event in events if isinstance(event.get("rank_arrival_us"), Mapping)]
    if raw_map_events:
        if len(raw_map_events) != len(events):
            raise SchemaError("cannot merge mixed rank_arrival_us and scalar arrival records")
        full_map = _identical_full_arrival_map(events, expected_ranks)
        if full_map is not None:
            earliest = min(full_map.values())
            coalesced["rank_arrival_us"] = {
                rank: round(value - earliest, 9)
                for rank, value in sorted(full_map.items(), key=lambda item: int(item[0]))
            }
        elif calibrated:
            absolute_arrivals: Dict[str, float] = {}
            for event in events:
                start_us = as_float(event.get("_aligned_start_us"), 0.0)
                for raw_rank, raw_offset in _own_rank_arrival_map(event).items():
                    rank_key = str(as_int(raw_rank))
                    if rank_key in absolute_arrivals:
                        raise SchemaError(f"duplicate arrival record for rank {rank_key}")
                    absolute_arrivals[rank_key] = start_us + as_float(raw_offset)
            if set(absolute_arrivals) != expected_ranks:
                missing = ", ".join(sorted(expected_ranks - set(absolute_arrivals), key=int))
                raise SchemaError(f"missing arrival records for ranks: {missing}")
            earliest = min(absolute_arrivals.values())
            coalesced["rank_arrival_us"] = {
                rank: round(value - earliest, 9)
                for rank, value in sorted(absolute_arrivals.items(), key=lambda item: int(item[0]))
            }
        else:
            local_skews = []
            for event in events:
                values = [as_float(value) for value in _own_rank_arrival_map(event).values()]
                if values:
                    local_skews.append(max(values) - min(values))
            coalesced["arrival_skew_us"] = round(max(local_skews, default=0.0), 9)
            coalesced["arrival_skew_unknown"] = True
    else:
        scalar_skew = max(as_float(event.get("arrival_skew_us"), 0.0) for event in events)
        if calibrated:
            coalesced["arrival_skew_us"] = round(max(scalar_skew, max(starts) - min(starts)), 9)
        else:
            coalesced["arrival_skew_us"] = round(scalar_skew, 9)
            coalesced["arrival_skew_unknown"] = True

    coalesced["merged_shards"] = sorted(str(event.get("shard")) for event in events)
    coalesced["recorder_ranks"] = sorted(recorder_ranks, key=int)
    _aggregate_compute_fields(coalesced, events)
    _aggregate_observed_field(coalesced, events)
    return coalesced


def _validate_collective_identity(events: List[JsonDict], ranks: List[int]) -> Tuple[str, Optional[JsonDict]]:
    first = events[0]
    session = first.get("capture_session_id")
    collective_id = first.get("collective_id")
    if not session:
        raise SchemaError("merged collective is missing capture_session_id")
    if not collective_id:
        raise SchemaError("merged collective is missing collective_id")
    expected = {
        "capture_session_id": session,
        "collective_id": collective_id,
        "phase": first.get("phase", "unknown"),
        "bytes": as_int(first.get("bytes")),
        "group": first.get("group", "default"),
        "ranks": tuple(ranks),
        "concurrent_groups": as_int(first.get("concurrent_groups"), 1),
    }
    ops = {str(first.get("op"))}
    for event in events[1:]:
        actual = {
            "capture_session_id": event.get("capture_session_id"),
            "collective_id": event.get("collective_id"),
            "phase": event.get("phase", "unknown"),
            "bytes": as_int(event.get("bytes")),
            "group": event.get("group", "default"),
            "ranks": tuple(normalize_ranks(event.get("ranks"))),
            "concurrent_groups": as_int(event.get("concurrent_groups"), 1),
        }
        if actual != expected:
            raise SchemaError("conflicting records share the same collective identity")
        ops.add(str(event.get("op")))
    if len(ops) == 1:
        return next(iter(ops)), None
    if ops == {"send", "recv"}:
        senders = [
            as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown")))
            for event in events
            if event.get("op") == "send"
        ]
        receivers = [
            as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown")))
            for event in events
            if event.get("op") == "recv"
        ]
        if len(senders) == 1 and len(receivers) == 1 and senders[0] != receivers[0]:
            return "point_to_point", {"sender_rank": senders[0], "receiver_rank": receivers[0]}
    raise SchemaError("conflicting records share the same collective identity")


def _recorder_ranks(events: List[JsonDict], expected_ranks: Set[str]) -> Set[str]:
    recorder_ranks: Set[str] = set()
    for event in events:
        raw_rank = event.get("_shard_rank", event.get("recorder_rank", "unknown"))
        rank_key = str(as_int(raw_rank))
        if rank_key not in expected_ranks:
            raise SchemaError(f"recorder rank {rank_key} is not part of the collective ranks")
        if rank_key in recorder_ranks:
            raise SchemaError(f"duplicate contribution from recorder rank {rank_key}")
        recorder_ranks.add(rank_key)
    if recorder_ranks != expected_ranks:
        missing = ", ".join(sorted(expected_ranks - recorder_ranks, key=int))
        raise SchemaError(f"missing recorder contributions for ranks: {missing}")
    return recorder_ranks


def _event_for_recorder_rank(events: List[JsonDict], rank: int) -> JsonDict:
    rank_key = str(rank)
    for event in events:
        raw_rank = event.get("_shard_rank", event.get("recorder_rank", "unknown"))
        if str(as_int(raw_rank)) == rank_key:
            return event
    raise SchemaError(f"missing recorder contribution for rank {rank_key}")


def _own_rank_arrival_map(event: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_arrivals = event.get("rank_arrival_us", {})
    if not isinstance(raw_arrivals, Mapping):
        raise SchemaError("rank_arrival_us must be an object")
    rank_key = str(as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown"))))
    if {str(key) for key in raw_arrivals} != {rank_key}:
        raise SchemaError("partial rank_arrival_us must contain exactly the recorder's own rank")
    return raw_arrivals


def _identical_full_arrival_map(
    events: List[JsonDict], expected_ranks: Set[str]
) -> Optional[Dict[str, float]]:
    full_maps: List[Dict[str, float]] = []
    for event in events:
        raw_arrivals = event.get("rank_arrival_us", {})
        if not isinstance(raw_arrivals, Mapping):
            return None
        if {str(key) for key in raw_arrivals} != expected_ranks:
            return None
        full_maps.append({str(key): as_float(value) for key, value in raw_arrivals.items()})
    first = full_maps[0]
    for full_map in full_maps[1:]:
        if full_map != first:
            raise SchemaError("full rank_arrival_us maps must be identical across shards")
    return first


def _aggregate_compute_fields(coalesced: JsonDict, events: List[JsonDict]) -> None:
    by_rank: Dict[str, Dict[str, float]] = {}
    for event in events:
        rank_key = str(as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown"))))
        by_rank[rank_key] = {
            "compute_before_us": round(as_float(event.get("compute_before_us"), 0.0), 9),
            "compute_overlap_us": round(as_float(event.get("compute_overlap_us"), 0.0), 9),
            "compute_pressure": round(as_float(event.get("compute_pressure"), 0.5), 6),
        }
    ordered = {
        rank: by_rank[rank]
        for rank in sorted(by_rank, key=int)
    }
    coalesced["compute_by_rank"] = ordered
    before_values = [values["compute_before_us"] for values in ordered.values()]
    overlap_values = [values["compute_overlap_us"] for values in ordered.values()]
    pressure_values = [values["compute_pressure"] for values in ordered.values()]
    coalesced["compute_before_us"] = round(max(before_values), 9) if before_values else 0.0
    coalesced["compute_overlap_us"] = round(min(overlap_values), 9) if overlap_values else 0.0
    coalesced["compute_pressure"] = round(max(pressure_values), 6) if pressure_values else 0.5
    if len(set(before_values)) > 1 or len(set(overlap_values)) > 1 or len(set(pressure_values)) > 1:
        coalesced["compute_fields_uncertain"] = True


def _aggregate_observed_field(coalesced: JsonDict, events: List[JsonDict]) -> None:
    flags = ["observed_exposed_us" in event for event in events]
    if any(flags) and not all(flags):
        raise SchemaError("observed_exposed_us must be present on every rank contribution or none")
    if flags and all(flags):
        coalesced["observed_exposed_us"] = round(
            max(as_float(event.get("observed_exposed_us")) for event in events), 9
        )
