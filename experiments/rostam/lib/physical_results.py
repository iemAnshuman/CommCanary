"""Strict adapters from physical producer output to analyzer measurements."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, fields
from functools import partial
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

from ..harness import ContractError, strict_json_loads

MICRO_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.micro-measurement.v1"
FULL_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.full-measurement.v1"
PARAM_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.param-measurement.v1"
OVERLAP_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.overlap-measurement.v1"
CAPTURE_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.capture-measurement.v1"

MICRO_PRODUCER_SCHEMA = "commcanary.rostam.physical.micro-producer.v1"
FULL_PRODUCER_SCHEMA = "commcanary.rostam.physical.full-producer.v1"
PARAM_PRODUCER_SCHEMA = "commcanary.rostam.physical.param-producer.v1"
OVERLAP_PRODUCER_SCHEMA = "commcanary.rostam.physical.overlap-producer.v1"
CAPTURE_PRODUCER_SCHEMA = "commcanary.rostam.physical.capture-producer.v1"

MICRO_STDOUT_SCHEMA = "commcanary.rostam.microbench_tp8.stdout.v1"
FULL_STDOUT_SCHEMA = "commcanary.rostam.workload_tp8.stdout.v1"
OVERLAP_STDOUT_SCHEMA = "commcanary.rostam.overlap_replay.stdout.v1"


@dataclass(frozen=True)
class ParamTraceLimits:
    """Bound one PARAM trace before decoding or starting torch.

    Defaults intentionally match the corresponding public CommCanary
    ``ResourceLimits`` defaults.  They are committed here as part of the
    standalone experiment contract because local campaign inspection must work
    before the exact CommCanary wheel is installed.
    """

    max_input_bytes: int = 64 * 1024 * 1024
    max_json_depth: int = 64
    max_json_items: int = 2_000_000
    max_json_number_chars: int = 1024
    max_param_entries: int = 2_000_000

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field.name} must be an integer")
            if value < 1:
                raise ValueError(f"{field.name} must be positive")


DEFAULT_PARAM_TRACE_LIMITS = ParamTraceLimits()

PHYSICAL_SCHEMA_PAIRS = {
    MICRO_MEASUREMENT_SCHEMA: MICRO_PRODUCER_SCHEMA,
    FULL_MEASUREMENT_SCHEMA: FULL_PRODUCER_SCHEMA,
    PARAM_MEASUREMENT_SCHEMA: PARAM_PRODUCER_SCHEMA,
    OVERLAP_MEASUREMENT_SCHEMA: OVERLAP_PRODUCER_SCHEMA,
    CAPTURE_MEASUREMENT_SCHEMA: CAPTURE_PRODUCER_SCHEMA,
}

_RAW_LATENCY_FIELDS = {
    "schema",
    "rank",
    "world_size",
    "timings_us",
    "metrics",
}
_RAW_STDOUT_CONTRACTS = {
    MICRO_PRODUCER_SCHEMA: (
        MICRO_STDOUT_SCHEMA,
        _RAW_LATENCY_FIELDS | {"dtype", "msg_sizes_bytes"},
    ),
    FULL_PRODUCER_SCHEMA: (
        FULL_STDOUT_SCHEMA,
        _RAW_LATENCY_FIELDS
        | {
            "tokens",
            "layers",
            "hidden",
            "gemm_m_rank0",
            "gemm_n",
            "dtype",
            "msg_sizes_bytes",
            "inject_skew",
        },
    ),
    OVERLAP_PRODUCER_SCHEMA: (
        OVERLAP_STDOUT_SCHEMA,
        _RAW_LATENCY_FIELDS,
    ),
}
_PARAM_LATENCY_RE = re.compile(
    r"Replayed\s+([A-Za-z0-9_]+)\s+in block \[[^\]]*\]\.\.\.\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*us"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^a-[0-9]{6}$")


class PhysicalResultError(ContractError):
    """Raised when physical output cannot satisfy its declared contract."""


@dataclass
class _JSONContainer:
    opening: int
    has_content: bool = False
    commas: int = 0


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalResultError(f"{field} must be an object")
    return value


def _strict(value: Mapping[str, Any], field: str, fields: Iterable[str]) -> None:
    expected = set(fields)
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise PhysicalResultError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise PhysicalResultError(f"{field} has unknown fields: {', '.join(unknown)}")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalResultError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PhysicalResultError(f"{field} must be a finite non-negative number")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0, maximum: int = 10_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PhysicalResultError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return cast(int, value)


def _text(value: Any, field: str, *, nullable: bool = False, maximum: int = 4096) -> Optional[str]:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        suffix = " or null" if nullable else ""
        raise PhysicalResultError(f"{field} must be a non-empty NUL-free string{suffix}")
    return value


def _reject_param_json_constant(value: str) -> None:
    raise PhysicalResultError(f"non-standard JSON constant {value!r} is not allowed")


def _param_json_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicalResultError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _bounded_param_json_int(value: str, *, max_chars: int) -> int:
    if len(value) > max_chars:
        raise PhysicalResultError(f"JSON numeric token exceeds max_json_number_chars={max_chars}")
    return int(value)


def _bounded_param_json_float(value: str, *, max_chars: int) -> float:
    if len(value) > max_chars:
        raise PhysicalResultError(f"JSON numeric token exceeds max_json_number_chars={max_chars}")
    result = float(value)
    if not math.isfinite(result):
        raise PhysicalResultError(f"JSON number {value!r} is outside the finite float range")
    return result


def _preflight_param_json(raw: bytes, *, limits: ParamTraceLimits) -> None:
    """Scan structural bytes before decoding, without copying the document."""

    first = next((value for value in raw if value not in b" \t\r\n"), None)
    if first != ord("["):
        raise PhysicalResultError("PARAM trace JSON root must be an array")

    containers: List[_JSONContainer] = []
    in_string = False
    escaped = False
    root_started = False
    root_closed = False
    comma_count = 0
    nonempty_containers = 0
    for value in raw:
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
            if containers:
                containers[-1].has_content = True
            continue
        if value in (ord("["), ord("{")):
            if not containers:
                if root_started or root_closed or value != ord("["):
                    raise PhysicalResultError("PARAM trace must contain one JSON array document")
                root_started = True
            else:
                containers[-1].has_content = True
            containers.append(_JSONContainer(value))
            if len(containers) > limits.max_json_depth:
                raise PhysicalResultError(f"JSON nesting exceeds max_json_depth={limits.max_json_depth}")
            continue
        if value == ord(",") and containers:
            containers[-1].commas += 1
            comma_count += 1
            if comma_count + nonempty_containers > limits.max_json_items:
                raise PhysicalResultError(f"JSON item count exceeds max_json_items={limits.max_json_items}")
            if len(containers) == 1 and containers[-1].commas + 1 > limits.max_param_entries:
                raise PhysicalResultError(f"PARAM trace entries exceed max_param_entries={limits.max_param_entries}")
            continue
        if value in (ord("]"), ord("}")):
            if not containers:
                continue
            container = containers.pop()
            expected_close = ord("]") if container.opening == ord("[") else ord("}")
            if value != expected_close:
                raise PhysicalResultError("PARAM trace has mismatched JSON containers")
            if container.has_content:
                nonempty_containers += 1
                if comma_count + nonempty_containers > limits.max_json_items:
                    raise PhysicalResultError(f"JSON item count exceeds max_json_items={limits.max_json_items}")
                if not containers and container.commas + 1 > limits.max_param_entries:
                    raise PhysicalResultError(
                        f"PARAM trace entries exceed max_param_entries={limits.max_param_entries}"
                    )
            if not containers:
                root_closed = True
            continue
        if value not in b" \t\r\n:" and containers:
            containers[-1].has_content = True


def _load_bounded_param_json(path: str, *, limits: ParamTraceLimits) -> Any:
    try:
        with open(path, "rb") as handle:
            raw_bytes = handle.read(limits.max_input_bytes + 1)
    except OSError as exc:
        raise PhysicalResultError(f"cannot read PARAM trace: {exc}") from exc
    if len(raw_bytes) > limits.max_input_bytes:
        raise PhysicalResultError(f"PARAM trace exceeds max_input_bytes={limits.max_input_bytes}")
    _preflight_param_json(raw_bytes, limits=limits)
    try:
        return json.loads(
            raw_bytes,
            object_pairs_hook=_param_json_object,
            parse_constant=_reject_param_json_constant,
            parse_float=partial(_bounded_param_json_float, max_chars=limits.max_json_number_chars),
            parse_int=partial(_bounded_param_json_int, max_chars=limits.max_json_number_chars),
        )
    except PhysicalResultError:
        raise
    except (UnicodeError, ValueError, OverflowError, RecursionError, MemoryError) as exc:
        raise PhysicalResultError(f"cannot decode PARAM trace: {exc}") from exc


def _samples(raw: Any, field: str = "timings_us") -> Tuple[float, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 1_000_000:
        raise PhysicalResultError(f"{field} must contain 1..1000000 samples")
    return tuple(_finite(value, f"{field}[{index}]") for index, value in enumerate(raw))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        lower = ordered[:middle]
        upper = ordered[middle + 1 :]
    else:
        lower = ordered[:middle]
        upper = ordered[middle:]
    if not lower or not upper:
        return 0.0
    return float(statistics.median(upper) - statistics.median(lower))


def validate_physical_layout(parameters: Any) -> Tuple[int, Tuple[int, ...]]:
    """Fail before launch for operations/layouts the current runners cannot honor."""

    data = _object(parameters, "workload.parameters")
    if data.get("operation") != "all_reduce":
        raise PhysicalResultError("unsupported operation: physical runners currently prove all_reduce only")
    world_size = _integer(data.get("world_size"), "workload.parameters.world_size", minimum=1, maximum=1024)
    ranks_raw = data.get("global_ranks")
    if not isinstance(ranks_raw, list):
        raise PhysicalResultError("workload.parameters.global_ranks must be an array")
    ranks = tuple(_integer(rank, "workload.parameters.global_ranks[]", maximum=world_size - 1) for rank in ranks_raw)
    if ranks != tuple(range(world_size)):
        raise PhysicalResultError(
            "unsupported process-group layout: current physical runners require exactly the dense world ranks"
        )
    return world_size, ranks


def _validate_trace_world(
    ranks_raw: Any,
    *,
    field: str,
    world_size: int,
) -> Tuple[int, ...]:
    expected = tuple(range(world_size))
    if not isinstance(ranks_raw, list):
        raise PhysicalResultError(f"{field} must be the full world ranks {list(expected)!r}")
    try:
        ranks = tuple(_integer(rank, f"{field}[]", maximum=2**63 - 1) for rank in ranks_raw)
    except PhysicalResultError as exc:
        raise PhysicalResultError(f"{field} must be the full world ranks {list(expected)!r}") from exc
    if ranks != expected:
        raise PhysicalResultError(f"{field} must be the full world ranks {list(expected)!r}")
    return ranks


def validate_param_trace(
    raw: Any,
    *,
    world_size: int,
    limits: ParamTraceLimits = DEFAULT_PARAM_TRACE_LIMITS,
) -> Dict[str, int]:
    """Validate the all-reduce PARAM subset before starting torch.distributed.

    The overlap runner deliberately supports only full-world groups today.  A
    non-world group is rejected rather than aliased to ``dist.group.WORLD``.
    Request/wait ownership, dtype, sizes, and pending operations are checked in
    one deterministic pass.
    """

    world_size = _integer(world_size, "world_size", minimum=1, maximum=1024)
    if not isinstance(raw, list) or not raw:
        raise PhysicalResultError("PARAM trace must be a non-empty array")
    if len(raw) > limits.max_param_entries:
        raise PhysicalResultError(f"PARAM trace entries exceed max_param_entries={limits.max_param_entries}")
    groups: Dict[int, Tuple[int, ...]] = {}
    pending: Dict[int, int] = {}
    issued_requests: Set[int] = set()
    comm_count = 0
    wait_count = 0
    allowed_dtypes = {"float32", "float", "bfloat16", "float16", "half", "float64", "double"}
    for index, entry_raw in enumerate(raw):
        entry = _object(entry_raw, f"trace[{index}]")
        compute = entry.get("compute")
        if compute is not None:
            if entry.get("comms") is not None:
                raise PhysicalResultError(f"trace[{index}] cannot mix compute and communication operations")
            if compute != "gemm":
                raise PhysicalResultError(f"trace[{index}] has unsupported compute operation {compute!r}")
            _integer(entry.get("mm_dim"), f"trace[{index}].mm_dim", minimum=1)
            _integer(entry.get("count"), f"trace[{index}].count", minimum=1)
            dtype = entry.get("dtype", "float32")
            if dtype not in allowed_dtypes:
                raise PhysicalResultError(f"trace[{index}] has unsupported dtype {dtype!r}")
            continue
        comms = entry.get("comms")
        if comms == "init":
            pg_id = _integer(entry.get("pg_id"), f"trace[{index}].pg_id", maximum=2**63 - 1)
            if pg_id in groups:
                raise PhysicalResultError(f"trace[{index}] has duplicate pg_id")
            ranks = _validate_trace_world(
                entry.get("global_ranks"),
                field=f"trace[{index}].global_ranks",
                world_size=world_size,
            )
            if "world_size" in entry:
                observed_world_size = _integer(
                    entry["world_size"], f"trace[{index}].world_size", minimum=1, maximum=1024
                )
                if observed_world_size != world_size:
                    raise PhysicalResultError(f"trace[{index}].world_size disagrees with the replay world")
            groups[pg_id] = ranks
            continue
        if comms == "all_reduce":
            pg_id = _integer(entry.get("pg_id"), f"trace[{index}].pg_id", maximum=2**63 - 1)
            if pg_id not in groups:
                raise PhysicalResultError(f"trace[{index}] references an uninitialized process group")
            if "global_ranks" in entry:
                _validate_trace_world(
                    entry["global_ranks"],
                    field=f"trace[{index}].global_ranks",
                    world_size=world_size,
                )
            if "world_size" in entry:
                observed_world_size = _integer(
                    entry["world_size"], f"trace[{index}].world_size", minimum=1, maximum=1024
                )
                if observed_world_size != world_size:
                    raise PhysicalResultError(f"trace[{index}].world_size disagrees with the replay world")
            request = _integer(entry.get("req"), f"trace[{index}].req", maximum=2**63 - 1)
            if request in issued_requests:
                raise PhysicalResultError(f"trace[{index}] has duplicate request id")
            size = _integer(entry.get("in_msg_size"), f"trace[{index}].in_msg_size", minimum=1, maximum=2**63 - 1)
            out_size = entry.get("out_msg_size", size)
            if _integer(out_size, f"trace[{index}].out_msg_size", minimum=1, maximum=2**63 - 1) != size:
                raise PhysicalResultError(f"trace[{index}] all_reduce input/output sizes differ")
            dtype = entry.get("dtype")
            if dtype not in allowed_dtypes:
                raise PhysicalResultError(f"trace[{index}] has unsupported dtype {dtype!r}")
            pending[request] = pg_id
            issued_requests.add(request)
            comm_count += 1
            continue
        if comms == "wait":
            request = _integer(entry.get("req"), f"trace[{index}].req", maximum=2**63 - 1)
            if request not in pending:
                raise PhysicalResultError(f"trace[{index}] waits for an unknown or already-completed request")
            del pending[request]
            wait_count += 1
            continue
        raise PhysicalResultError(f"trace[{index}] has unsupported communication operation {comms!r}")
    if not groups:
        raise PhysicalResultError("PARAM trace contains no process-group initialization")
    if not comm_count:
        raise PhysicalResultError("PARAM trace contains no measurable all_reduce")
    if wait_count and pending:
        raise PhysicalResultError(f"PARAM trace leaves {len(pending)} request(s) pending")
    if wait_count not in {0, comm_count}:
        raise PhysicalResultError("PARAM trace mixes blocking and explicit-wait collectives")
    return {"process_groups": len(groups), "collectives": comm_count, "waits": wait_count}


def validate_overlap_trace(
    raw: Any,
    *,
    world_size: int,
    limits: ParamTraceLimits = DEFAULT_PARAM_TRACE_LIMITS,
) -> Dict[str, int]:
    """Validate the explicit-wait subset required by the overlap runner."""

    audit = validate_param_trace(raw, world_size=world_size, limits=limits)
    if audit["waits"] != audit["collectives"]:
        raise PhysicalResultError("overlap trace requires exactly one explicit wait for every all_reduce request")
    return audit


def load_validated_param_trace(
    path: str,
    *,
    world_size: int,
    require_explicit_waits: bool = False,
    limits: ParamTraceLimits = DEFAULT_PARAM_TRACE_LIMITS,
) -> Tuple[List[Mapping[str, Any]], Dict[str, int]]:
    """Load strict JSON and return the trace only after complete validation."""

    raw = _load_bounded_param_json(path, limits=limits)
    audit = (
        validate_overlap_trace(raw, world_size=world_size, limits=limits)
        if require_explicit_waits
        else validate_param_trace(raw, world_size=world_size, limits=limits)
    )
    return cast(List[Mapping[str, Any]], raw), audit


def load_and_validate_param_trace(path: str, *, world_size: int) -> Dict[str, int]:
    _, audit = load_validated_param_trace(path, world_size=world_size)
    return audit


def _runtime(raw: Any) -> Dict[str, Any]:
    data = _object(raw, "runtime")
    _strict(
        data,
        "runtime",
        (
            "hostname",
            "job_id",
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "runtime_nccl_version_code",
        ),
    )
    return {
        "hostname": _text(data["hostname"], "runtime.hostname", maximum=256),
        "job_id": _text(data["job_id"], "runtime.job_id", nullable=True, maximum=256),
        "python_version": _text(data["python_version"], "runtime.python_version", maximum=64),
        "torch_version": _text(data["torch_version"], "runtime.torch_version", maximum=128),
        "torch_cuda_version": _text(
            data["torch_cuda_version"], "runtime.torch_cuda_version", nullable=True, maximum=64
        ),
        "runtime_nccl_version_code": _integer(
            data["runtime_nccl_version_code"],
            "runtime.runtime_nccl_version_code",
            minimum=1,
            maximum=99_999,
        ),
    }


def validate_expected_runtime(runtime: Mapping[str, Any], expected: Any) -> None:
    expectation = _object(expected, "configuration.expected_runtime")
    for field in ("python_version", "torch_version", "runtime_nccl_version_code"):
        if expectation.get(field) != runtime.get(field):
            raise PhysicalResultError(
                f"runtime {field} mismatch: expected {expectation.get(field)!r}, observed {runtime.get(field)!r}"
            )


def _last_json_object(stdout: str) -> Mapping[str, Any]:
    candidate: Optional[Mapping[str, Any]] = None
    for line in stdout.splitlines():
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            parsed = strict_json_loads(text)
        except ContractError:
            continue
        if isinstance(parsed, Mapping):
            candidate = parsed
    if candidate is None:
        raise PhysicalResultError("producer stdout contains no JSON object")
    return candidate


def _message_sizes(raw: Any, field: str) -> List[int]:
    if not isinstance(raw, list) or not raw:
        raise PhysicalResultError(f"{field} must contain at least one message size")
    return [_integer(value, f"{field}[]", minimum=1, maximum=2**63 - 1) for value in raw]


def _torch_payload(
    stdout: str,
    world_size: int,
    producer_schema: str,
) -> Tuple[Mapping[str, Any], Tuple[float, ...]]:
    payload = _last_json_object(stdout)
    contract = _RAW_STDOUT_CONTRACTS.get(producer_schema)
    if contract is None:
        raise PhysicalResultError(f"producer {producer_schema!r} has no raw stdout contract")
    expected_schema, fields = contract
    if payload.get("schema") != expected_schema:
        raise PhysicalResultError(
            f"producer {producer_schema!r} requires raw stdout schema {expected_schema!r}, "
            f"observed {payload.get('schema')!r}"
        )
    _strict(payload, "producer stdout", fields)
    rank = _integer(payload["rank"], "producer stdout.rank", maximum=max(0, world_size - 1))
    observed_world_size = _integer(payload["world_size"], "producer stdout.world_size", minimum=1, maximum=1024)
    if rank != 0 or observed_world_size != world_size:
        raise PhysicalResultError("producer stdout does not belong to rank 0 of the declared world")
    samples = _samples(payload["timings_us"])
    metrics = _object(payload["metrics"], "producer stdout.metrics")
    _strict(metrics, "producer stdout.metrics", ("median_us", "iqr_us", "count"))
    count = _integer(metrics["count"], "producer stdout.metrics.count", minimum=1, maximum=1_000_000)
    if count != len(samples):
        raise PhysicalResultError("producer stdout metric count disagrees with timing samples")
    if abs(_finite(metrics["median_us"], "producer stdout.metrics.median_us") - _median(samples)) > 0.001:
        raise PhysicalResultError("producer stdout median disagrees with timing samples")
    if abs(_finite(metrics["iqr_us"], "producer stdout.metrics.iqr_us") - _iqr(samples)) > 0.001:
        raise PhysicalResultError("producer stdout IQR disagrees with timing samples")
    if producer_schema == MICRO_PRODUCER_SCHEMA:
        _text(payload["dtype"], "producer stdout.dtype", maximum=32)
        _message_sizes(payload["msg_sizes_bytes"], "producer stdout.msg_sizes_bytes")
    elif producer_schema == FULL_PRODUCER_SCHEMA:
        _text(payload["dtype"], "producer stdout.dtype", maximum=32)
        for field in ("tokens", "layers", "hidden", "gemm_m_rank0", "gemm_n"):
            _integer(payload[field], f"producer stdout.{field}", minimum=1)
        _message_sizes(payload["msg_sizes_bytes"], "producer stdout.msg_sizes_bytes")
        _finite(payload["inject_skew"], "producer stdout.inject_skew")
    return payload, samples


def _param_samples(stdout: str, stderr: str) -> Tuple[float, ...]:
    samples: List[float] = []
    for line in (stdout + "\n" + stderr).splitlines():
        if "[Warm-up]" in line or "compute-fill" in line:
            continue
        for operation, raw_value in _PARAM_LATENCY_RE.findall(line):
            if operation != "all_reduce":
                raise PhysicalResultError(f"PARAM emitted unsupported replay operation {operation!r}")
            samples.append(_finite(float(raw_value), "PARAM latency"))
    if not samples:
        raise PhysicalResultError("PARAM output contains no per-all_reduce latency samples")
    return tuple(samples)


def _base_measurement(
    *,
    attempt_id: str,
    parameters: Mapping[str, Any],
    samples: Sequence[float],
    wall_time_s: float,
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(attempt_id, str) or not _ATTEMPT_RE.fullmatch(attempt_id):
        raise PhysicalResultError("attempt_id must use the canonical a-NNNNNN form")
    world_size, ranks = validate_physical_layout(parameters)
    if not samples:
        raise PhysicalResultError("physical measurement cannot have an empty sample set")
    values = tuple(_finite(item, "samples_us[]") for item in samples)
    return {
        "attempt_id": attempt_id,
        "operation": "all_reduce",
        "world_size": world_size,
        "global_ranks": list(ranks),
        "value_us": _median(values),
        "samples_us": list(values),
        "iqr_us": _iqr(values),
        "count": len(values),
        "wall_time_s": _finite(wall_time_s, "wall_time_s"),
        "runtime": dict(runtime),
    }


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PhysicalResultError(f"{field} must be a lowercase SHA-256")
    return value


def _artifact_map(raw: Any) -> Dict[str, Any]:
    data = _object(raw, "artifacts")
    if not data:
        raise PhysicalResultError("capture measurement requires at least one artifact")
    result: Dict[str, Any] = {}
    for artifact_id, raw_reference in sorted(data.items()):
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", artifact_id):
            raise PhysicalResultError(f"invalid capture artifact id {artifact_id!r}")
        reference = _object(raw_reference, f"artifacts.{artifact_id}")
        _strict(reference, f"artifacts.{artifact_id}", ("path", "sha256", "size_bytes"))
        path = reference["path"]
        if not isinstance(path, str) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise PhysicalResultError(f"artifacts.{artifact_id}.path must be a contained relative path")
        result[artifact_id] = {
            "path": path,
            "sha256": _sha256(reference["sha256"], f"artifacts.{artifact_id}.sha256"),
            "size_bytes": _integer(
                reference["size_bytes"],
                f"artifacts.{artifact_id}.size_bytes",
                maximum=2**63 - 1,
            ),
        }
    return result


def adapt_physical_measurement(
    *,
    measurement_schema: str,
    producer_schema: str,
    attempt_id: str,
    parameters: Any,
    stdout: str,
    stderr: str,
    wall_time_s: float,
    runtime: Any,
    trace_sha256: Optional[str] = None,
    artifacts: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert one physical producer result to its strict committed schema."""

    expected_producer = PHYSICAL_SCHEMA_PAIRS.get(measurement_schema)
    if expected_producer is None:
        raise PhysicalResultError(f"unsupported physical measurement schema {measurement_schema!r}")
    if producer_schema != expected_producer:
        raise PhysicalResultError(f"measurement schema {measurement_schema!r} requires producer {expected_producer!r}")
    parameter_object = _object(parameters, "workload.parameters")
    runtime_object = _runtime(runtime)
    if measurement_schema == CAPTURE_MEASUREMENT_SCHEMA:
        duration_us = _finite(wall_time_s, "wall_time_s") * 1_000_000.0
        measurement = _base_measurement(
            attempt_id=attempt_id,
            parameters=parameter_object,
            samples=(duration_us,),
            wall_time_s=wall_time_s,
            runtime=runtime_object,
        )
        measurement["artifacts"] = _artifact_map(artifacts)
        return measurement
    if measurement_schema == PARAM_MEASUREMENT_SCHEMA:
        samples = _param_samples(stdout, stderr)
        payload = None
    else:
        world_size, _ = validate_physical_layout(parameter_object)
        payload, samples = _torch_payload(stdout, world_size, producer_schema)
    measurement = _base_measurement(
        attempt_id=attempt_id,
        parameters=parameter_object,
        samples=samples,
        wall_time_s=wall_time_s,
        runtime=runtime_object,
    )
    if measurement_schema == MICRO_MEASUREMENT_SCHEMA:
        assert payload is not None
        measurement.update(
            {
                "dtype": _text(payload["dtype"], "producer stdout.dtype", maximum=32),
                "message_sizes_bytes": _message_sizes(payload["msg_sizes_bytes"], "producer stdout.msg_sizes_bytes"),
            }
        )
    elif measurement_schema == FULL_MEASUREMENT_SCHEMA:
        assert payload is not None
        measurement.update(
            {
                "dtype": _text(payload["dtype"], "producer stdout.dtype", maximum=32),
                "layers": _integer(payload["layers"], "producer stdout.layers", minimum=1),
                "tokens": _integer(payload["tokens"], "producer stdout.tokens", minimum=1),
                "hidden": _integer(payload["hidden"], "producer stdout.hidden", minimum=1),
                "gemm_m": _integer(payload["gemm_m_rank0"], "producer stdout.gemm_m_rank0", minimum=1),
                "gemm_n": _integer(payload["gemm_n"], "producer stdout.gemm_n", minimum=1),
            }
        )
    elif measurement_schema in {PARAM_MEASUREMENT_SCHEMA, OVERLAP_MEASUREMENT_SCHEMA}:
        measurement.update(
            {
                "replay_mode": _text(parameter_object.get("replay_mode"), "workload.parameters.replay_mode"),
                "trace_sha256": _sha256(trace_sha256, "trace_sha256"),
            }
        )
    return measurement
