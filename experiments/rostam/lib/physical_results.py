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

from ..decision_gate_schedule import (
    REPLICATED_ORDER_METHOD,
    REPRESENTATION_IDS,
    representation_order,
)
from ..harness import ContractError, strict_json_loads

MICRO_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.micro-measurement.v1"
FULL_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.full-measurement.v1"
PARAM_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.param-measurement.v1"
OVERLAP_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.overlap-measurement.v1"
CAPTURE_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.capture-measurement.v1"
QUALIFICATION_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.qualification-measurement.v1"
DECISION_GATE_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.decision-gate-measurement.v1"
DECISION_GATE_REPLICATED_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.decision-gate-measurement.v2"

MICRO_PRODUCER_SCHEMA = "commcanary.rostam.physical.micro-producer.v1"
FULL_PRODUCER_SCHEMA = "commcanary.rostam.physical.full-producer.v1"
PARAM_PRODUCER_SCHEMA = "commcanary.rostam.physical.param-producer.v1"
OVERLAP_PRODUCER_SCHEMA = "commcanary.rostam.physical.overlap-producer.v1"
CAPTURE_PRODUCER_SCHEMA = "commcanary.rostam.physical.capture-producer.v1"
QUALIFICATION_PRODUCER_SCHEMA = "commcanary.rostam.physical.qualification-producer.v1"
DECISION_GATE_PRODUCER_SCHEMA = "commcanary.rostam.physical.decision-gate-producer.v1"
DECISION_GATE_REPLICATED_PRODUCER_SCHEMA = "commcanary.rostam.physical.decision-gate-producer.v2"

MICRO_STDOUT_SCHEMA = "commcanary.rostam.microbench_tp8.stdout.v1"
FULL_STDOUT_SCHEMA = "commcanary.rostam.workload_tp8.stdout.v1"
OVERLAP_STDOUT_SCHEMA = "commcanary.rostam.overlap_replay.stdout.v1"
REFERENCE_EXECUTION_STDOUT_SCHEMA = "commcanary.reference-execution.stdout.v1"
QUALIFICATION_STDOUT_SCHEMA = "commcanary.rostam.qualification-comparison.stdout.v1"
DECISION_GATE_STDOUT_SCHEMA = "commcanary.rostam.decision-gate.stdout.v1"
DECISION_GATE_REPLICATED_STDOUT_SCHEMA = "commcanary.rostam.decision-gate.stdout.v2"
SOURCE_TIMING_SEMANTICS = "maximum-rank-unprofiled-whole-program-duration"
DECISION_GATE_TIMING_SEMANTICS = "maximum-rank-cuda-event-whole-program-duration"
DECISION_GATE_ORDER_METHOD = "iteration-rotated-latin-cycle.v1"
DECISION_GATE_REPLICATED_ORDER_METHOD = REPLICATED_ORDER_METHOD
DECISION_GATE_STRATIFIED_METHOD = "first-observed-per-collective-shape.v1"
_DECISION_GATE_REPRESENTATIONS = REPRESENTATION_IDS
_DECISION_GATE_REPRESENTATION_CONTRACTS = {
    "source": ("ground_truth", "direct-source-issue-rank-work-wait"),
    "exact_work": ("product_candidate", "verified-materialization-issue-rank-work-wait"),
    "stratified": ("kill_condition_baseline", DECISION_GATE_STRATIFIED_METHOD),
    "isolated": ("incumbent_baseline", "full-message-sequence-blocking-all-reduce-no-compute"),
    "no_overlap": ("causal_ablation", "blocking-all-reduce-then-exact-rank-work"),
    "no_rank_skew": ("causal_ablation", "issue-rank-zero-work-on-every-rank-wait"),
}
_DECISION_GATE_REPLICATED_REPRESENTATION_CONTRACTS = {
    **_DECISION_GATE_REPRESENTATION_CONTRACTS,
    "exact_work": ("positive_conformance_control", "verified-materialization-issue-rank-work-wait"),
}


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
    QUALIFICATION_MEASUREMENT_SCHEMA: QUALIFICATION_PRODUCER_SCHEMA,
    DECISION_GATE_MEASUREMENT_SCHEMA: DECISION_GATE_PRODUCER_SCHEMA,
    DECISION_GATE_REPLICATED_MEASUREMENT_SCHEMA: DECISION_GATE_REPLICATED_PRODUCER_SCHEMA,
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


def _summary_samples(raw: Any, field: str) -> Tuple[float, ...]:
    summary = _object(raw, field)
    _strict(
        summary,
        field,
        ("semantics", "timings_us", "count", "median_us", "max_us"),
    )
    _text(summary["semantics"], f"{field}.semantics", maximum=160)
    samples = _samples(summary["timings_us"], f"{field}.timings_us")
    count = _integer(summary["count"], f"{field}.count", minimum=1, maximum=1_000_000)
    if count != len(samples):
        raise PhysicalResultError(f"{field}.count disagrees with its timing samples")
    if abs(_finite(summary["median_us"], f"{field}.median_us") - _median(samples)) > 0.001:
        raise PhysicalResultError(f"{field}.median_us disagrees with its timing samples")
    if abs(_finite(summary["max_us"], f"{field}.max_us") - max(samples)) > 0.001:
        raise PhysicalResultError(f"{field}.max_us disagrees with its timing samples")
    return samples


def _qualification_source_capture(
    raw: Any,
    *,
    world_size: int,
    parameters: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Tuple[float, ...]]:
    source = _object(raw, "qualification producer stdout.source_capture")
    _strict(
        source,
        "qualification producer stdout.source_capture",
        (
            "evidence_sha256",
            "stdout_sha256",
            "diagnostic_id",
            "scheduler",
            "execution_semantics",
            "timing_semantics",
            "rank_timings_us",
            "timings_us",
            "metrics",
        ),
    )
    for field in ("evidence_sha256", "stdout_sha256"):
        _sha256(source[field], f"qualification producer stdout.source_capture.{field}")
    if source["diagnostic_id"] != parameters.get("expected_source_capture_diagnostic_id"):
        raise PhysicalResultError("qualification source diagnostic id disagrees with the frozen workload")
    scheduler = _object(source["scheduler"], "qualification producer stdout.source_capture.scheduler")
    _strict(
        scheduler,
        "qualification producer stdout.source_capture.scheduler",
        ("job_id", "node", "partition"),
    )
    expected_scheduler = {
        "job_id": parameters.get("expected_source_job_id"),
        "node": parameters.get("expected_source_node"),
        "partition": parameters.get("expected_source_partition"),
    }
    if dict(scheduler) != expected_scheduler:
        raise PhysicalResultError("qualification source scheduler identity disagrees with the frozen workload")
    if source["execution_semantics"] != "async-all-reduce-then-gemm-then-explicit-wait":
        raise PhysicalResultError("qualification source execution semantics are unsupported")
    if source["timing_semantics"] != SOURCE_TIMING_SEMANTICS:
        raise PhysicalResultError("qualification source timing semantics are unsupported")

    iterations = _integer(parameters.get("iterations"), "workload.parameters.iterations", minimum=1)
    rank_timings_raw = source["rank_timings_us"]
    if not isinstance(rank_timings_raw, list) or len(rank_timings_raw) != world_size:
        raise PhysicalResultError("qualification source rank timings do not cover every rank")
    rank_timings = tuple(
        _samples(
            values,
            f"qualification producer stdout.source_capture.rank_timings_us[{rank}]",
        )
        for rank, values in enumerate(rank_timings_raw)
    )
    if any(len(values) != iterations for values in rank_timings):
        raise PhysicalResultError("qualification source rank timing count disagrees with iterations")
    samples = _samples(
        source["timings_us"],
        "qualification producer stdout.source_capture.timings_us",
    )
    if len(samples) != iterations:
        raise PhysicalResultError("qualification source timing count disagrees with iterations")
    recomputed = tuple(
        max(rank_timings[rank][iteration] for rank in range(world_size)) for iteration in range(iterations)
    )
    if samples != recomputed:
        raise PhysicalResultError("qualification source timings disagree with rank timings")

    metrics = _object(source["metrics"], "qualification producer stdout.source_capture.metrics")
    _strict(
        metrics,
        "qualification producer stdout.source_capture.metrics",
        ("count", "median_us", "iqr_us", "min_us", "max_us"),
    )
    if _integer(metrics["count"], "qualification source count", minimum=1) != len(samples):
        raise PhysicalResultError("qualification source metric count disagrees with timing samples")
    expected_metrics = {
        "median_us": _median(samples),
        "iqr_us": _iqr(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }
    for field, expected in expected_metrics.items():
        if abs(_finite(metrics[field], f"qualification source {field}") - expected) > 0.001:
            raise PhysicalResultError(f"qualification source {field} disagrees with timing samples")
    return source, samples


def _qualification_payload(
    stdout: str,
    *,
    world_size: int,
    parameters: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Tuple[float, ...], Mapping[str, Any], Tuple[float, ...]]:
    envelope = _last_json_object(stdout)
    _strict(
        envelope,
        "qualification producer stdout",
        ("schema", "source_capture", "execution", "claims"),
    )
    if envelope["schema"] != QUALIFICATION_STDOUT_SCHEMA:
        raise PhysicalResultError(f"qualification producer stdout requires schema {QUALIFICATION_STDOUT_SCHEMA!r}")
    claims = _object(envelope["claims"], "qualification producer stdout.claims")
    if dict(claims) != {
        "single_configuration_timing_comparison": "diagnostic",
        "physical_fidelity": "unproven",
        "multi_configuration_ranking": "not_measured",
        "qualification_verdict": "not_issued",
    }:
        raise PhysicalResultError("qualification comparison claims exceed the diagnostic boundary")
    source, source_samples = _qualification_source_capture(
        envelope["source_capture"],
        world_size=world_size,
        parameters=parameters,
    )

    payload = _object(envelope["execution"], "qualification producer stdout.execution")
    _strict(
        payload,
        "qualification producer stdout.execution",
        (
            "schema",
            "request_id",
            "materialization_id",
            "program_sha256",
            "executor",
            "runtime",
            "rank_tensor_bytes",
            "rank_compute_operations_per_pass",
            "correctness_validation",
            "rank_samples",
            "program_makespan",
            "aggregate",
            "claims",
        ),
    )
    if payload["schema"] != REFERENCE_EXECUTION_STDOUT_SCHEMA:
        raise PhysicalResultError(f"qualification execution requires schema {REFERENCE_EXECUTION_STDOUT_SCHEMA!r}")
    for field in ("request_id", "materialization_id", "program_sha256"):
        observed = _sha256(payload[field], f"qualification producer stdout.{field}")
        expected = parameters.get(f"expected_{field}")
        if observed != expected:
            raise PhysicalResultError(f"qualification producer stdout.{field} disagrees with the frozen workload")

    executor = _object(payload["executor"], "qualification producer stdout.executor")
    _strict(
        executor,
        "qualification producer stdout.executor",
        (
            "name",
            "claim",
            "device",
            "backend",
            "world_size",
            "iterations",
            "warmup",
            "distributed_timeout_seconds",
        ),
    )
    expected_executor = {
        "name": "commcanary.torch-distributed-reference.v3",
        "claim": "reference-implementation-not-yet-physically-conformance-validated",
        "device": parameters.get("device"),
        "backend": parameters.get("backend"),
        "world_size": world_size,
        "iterations": parameters.get("iterations"),
        "warmup": parameters.get("warmup"),
        "distributed_timeout_seconds": parameters.get("distributed_timeout_seconds"),
    }
    if dict(executor) != expected_executor:
        raise PhysicalResultError("qualification producer executor contract disagrees with the frozen workload")

    runtime = _object(payload["runtime"], "qualification producer stdout.runtime")
    _strict(
        runtime,
        "qualification producer stdout.runtime",
        ("torch_version", "torch_cuda_version", "runtime_nccl_version_code", "distributed_backend"),
    )
    _text(runtime["torch_version"], "qualification producer stdout.runtime.torch_version", maximum=128)
    _text(
        runtime["torch_cuda_version"],
        "qualification producer stdout.runtime.torch_cuda_version",
        nullable=True,
        maximum=64,
    )
    _integer(
        runtime["runtime_nccl_version_code"],
        "qualification producer stdout.runtime.runtime_nccl_version_code",
        minimum=1,
        maximum=99_999,
    )
    if runtime["distributed_backend"] != parameters.get("backend"):
        raise PhysicalResultError("qualification producer distributed backend disagrees with the frozen workload")

    for field in ("rank_tensor_bytes", "rank_compute_operations_per_pass"):
        values = payload[field]
        if not isinstance(values, list) or len(values) != world_size:
            raise PhysicalResultError(f"qualification producer stdout.{field} must cover every rank")
        parsed_values = [
            _integer(
                value,
                f"qualification producer stdout.{field}[{rank}]",
                maximum=2**63 - 1,
            )
            for rank, value in enumerate(values)
        ]
        expected_values = parameters.get(f"expected_{field}")
        if parsed_values != expected_values:
            raise PhysicalResultError(f"qualification producer stdout.{field} disagrees with the frozen workload")

    correctness = _object(
        payload["correctness_validation"],
        "qualification producer stdout.correctness_validation",
    )
    _strict(
        correctness,
        "qualification producer stdout.correctness_validation",
        ("status", "semantics", "checks_per_rank", "total_check_count"),
    )
    if correctness["status"] != "passed" or correctness["semantics"] != (
        "untimed-deterministic-communication-data-check"
    ):
        raise PhysicalResultError("qualification producer correctness validation did not pass")
    checks = correctness["checks_per_rank"]
    if not isinstance(checks, list) or len(checks) != world_size:
        raise PhysicalResultError("qualification producer correctness checks must cover every rank")
    parsed_checks = [
        _integer(value, f"qualification producer correctness checks[{rank}]", maximum=1_000_000)
        for rank, value in enumerate(checks)
    ]
    if parsed_checks != parameters.get("expected_correctness_checks_per_rank"):
        raise PhysicalResultError("qualification producer correctness checks disagree with the frozen workload")
    if _integer(
        correctness["total_check_count"],
        "qualification producer total_check_count",
        maximum=1_000_000,
    ) != sum(parsed_checks):
        raise PhysicalResultError("qualification producer correctness check total is inconsistent")

    makespan = _object(payload["program_makespan"], "qualification producer stdout.program_makespan")
    _strict(
        makespan,
        "qualification producer stdout.program_makespan",
        ("semantics", "rank_timings_us", "timings_us", "count", "median_us", "max_us"),
    )
    if makespan["semantics"] != "maximum-rank-whole-program-wall-clock":
        raise PhysicalResultError("qualification producer program makespan semantics are unsupported")
    rank_timings = _object(
        makespan["rank_timings_us"],
        "qualification producer stdout.program_makespan.rank_timings_us",
    )
    expected_rank_keys = {str(rank) for rank in range(world_size)}
    if set(rank_timings) != expected_rank_keys:
        raise PhysicalResultError("qualification producer program timings do not cover the dense rank domain")
    iterations = _integer(parameters.get("iterations"), "workload.parameters.iterations", minimum=1)
    parsed_rank_timings = {
        rank: _samples(
            rank_timings[str(rank)],
            f"qualification producer stdout.program_makespan.rank_timings_us[{rank}]",
        )
        for rank in range(world_size)
    }
    if any(len(values) != iterations for values in parsed_rank_timings.values()):
        raise PhysicalResultError("qualification producer rank timing count disagrees with iterations")
    samples = _samples(
        makespan["timings_us"],
        "qualification producer stdout.program_makespan.timings_us",
    )
    if len(samples) != iterations:
        raise PhysicalResultError("qualification producer makespan sample count disagrees with iterations")
    recomputed = tuple(
        max(parsed_rank_timings[rank][iteration] for rank in range(world_size)) for iteration in range(iterations)
    )
    if any(abs(observed - expected) > 0.001 for observed, expected in zip(samples, recomputed)):
        raise PhysicalResultError("qualification producer makespan samples disagree with rank timings")
    count = _integer(makespan["count"], "qualification producer stdout.program_makespan.count", minimum=1)
    if count != len(samples):
        raise PhysicalResultError("qualification producer makespan count is inconsistent")
    if abs(_finite(makespan["median_us"], "qualification producer makespan median") - _median(samples)) > 0.001:
        raise PhysicalResultError("qualification producer makespan median is inconsistent")
    if abs(_finite(makespan["max_us"], "qualification producer makespan maximum") - max(samples)) > 0.001:
        raise PhysicalResultError("qualification producer makespan maximum is inconsistent")

    aggregate_samples = _summary_samples(payload["aggregate"], "qualification producer stdout.aggregate")
    rank_samples = _object(payload["rank_samples"], "qualification producer stdout.rank_samples")
    if set(rank_samples) != expected_rank_keys or any(not isinstance(rank_samples[key], list) for key in rank_samples):
        raise PhysicalResultError("qualification producer operation samples do not cover every rank")
    grouped_samples: Dict[Tuple[int, int, int, str], Dict[int, float]] = {}
    for rank in range(world_size):
        raw_samples = cast(List[Any], rank_samples[str(rank)])
        for index, raw_sample in enumerate(raw_samples):
            sample = _object(
                raw_sample,
                f"qualification producer stdout.rank_samples[{rank}][{index}]",
            )
            _strict(
                sample,
                f"qualification producer stdout.rank_samples[{rank}][{index}]",
                ("duration_us", "iteration", "operation", "rank", "request", "sequence"),
            )
            observed_rank = _integer(
                sample["rank"],
                f"qualification producer operation sample rank[{rank}][{index}]",
                maximum=world_size - 1,
            )
            if observed_rank != rank:
                raise PhysicalResultError("qualification producer operation sample ownership is inconsistent")
            iteration = _integer(
                sample["iteration"],
                f"qualification producer operation sample iteration[{rank}][{index}]",
                maximum=iterations - 1,
            )
            sequence = _integer(
                sample["sequence"],
                f"qualification producer operation sample sequence[{rank}][{index}]",
                maximum=10_000_000,
            )
            request = _integer(
                sample["request"],
                f"qualification producer operation sample request[{rank}][{index}]",
                maximum=10_000_000,
            )
            operation = cast(
                str,
                _text(
                    sample["operation"],
                    f"qualification producer operation sample operation[{rank}][{index}]",
                    maximum=64,
                ),
            )
            if operation != parameters.get("operation"):
                raise PhysicalResultError("qualification producer operation sample disagrees with the frozen workload")
            key = (iteration, sequence, request, operation)
            by_rank = grouped_samples.setdefault(key, {})
            if rank in by_rank:
                raise PhysicalResultError("qualification producer operation sample duplicates one rank")
            by_rank[rank] = _finite(
                sample["duration_us"],
                f"qualification producer operation sample duration[{rank}][{index}]",
            )
    operations_per_pass = _integer(
        parameters.get("expected_communication_operations_per_pass"),
        "workload.parameters.expected_communication_operations_per_pass",
        minimum=1,
        maximum=1_000_000,
    )
    if len(grouped_samples) != operations_per_pass * iterations:
        raise PhysicalResultError("qualification producer operation sample count disagrees with the frozen workload")
    expected_ranks = set(range(world_size))
    if any(set(by_rank) != expected_ranks for by_rank in grouped_samples.values()):
        raise PhysicalResultError("qualification producer operation sample is missing a participating rank")
    recomputed_aggregate = tuple(max(grouped_samples[key].values()) for key in sorted(grouped_samples))
    if len(aggregate_samples) != len(recomputed_aggregate) or any(
        abs(observed - expected) > 0.001 for observed, expected in zip(aggregate_samples, recomputed_aggregate)
    ):
        raise PhysicalResultError("qualification producer aggregate samples disagree with rank operation samples")
    claims = _object(payload["claims"], "qualification producer stdout.claims")
    if dict(claims) != {
        "physical_execution": "self_reported_reference_executor",
        "physical_fidelity": "unproven",
        "qualification_verdict": "not_issued",
    }:
        raise PhysicalResultError("qualification producer claims exceed the diagnostic boundary")
    return payload, samples, source, source_samples


def _decision_gate_reference(raw: Any, field: str, expected: Mapping[str, str]) -> Dict[str, str]:
    reference = _object(raw, field)
    _strict(reference, field, expected)
    result: Dict[str, str] = {}
    for name, expected_value in expected.items():
        value = reference[name]
        if name.endswith("_id") or name.endswith("sha256"):
            result[name] = _sha256(value, f"{field}.{name}")
        else:
            result[name] = _text(value, f"{field}.{name}") or ""
        if expected_value and result[name] != expected_value:
            raise PhysicalResultError(f"{field}.{name} disagrees with the frozen workload")
    return result


def _decision_gate_representation(
    raw: Any,
    *,
    representation: str,
    world_size: int,
    iterations: int,
    expected_event_count: int,
    expected_template_count: int,
    contracts: Mapping[str, Tuple[str, str]],
) -> Tuple[Dict[str, Any], Tuple[float, ...]]:
    field = f"decision-gate producer stdout.representations.{representation}"
    value = _object(raw, field)
    _strict(
        value,
        field,
        (
            "category",
            "semantics",
            "executed_event_count",
            "template_count",
            "rank_timings_us",
            "timings_us",
            "metrics",
        ),
    )
    expected_category, expected_semantics = contracts[representation]
    if value["category"] != expected_category or value["semantics"] != expected_semantics:
        raise PhysicalResultError(f"{field} semantics disagree with the declared representation")
    if _integer(value["executed_event_count"], f"{field}.executed_event_count", minimum=1) != expected_event_count:
        raise PhysicalResultError(f"{field}.executed_event_count disagrees with the frozen program")
    if _integer(value["template_count"], f"{field}.template_count", minimum=1) != expected_template_count:
        raise PhysicalResultError(f"{field}.template_count disagrees with the frozen program")
    raw_rank_timings = value["rank_timings_us"]
    if not isinstance(raw_rank_timings, list) or len(raw_rank_timings) != world_size:
        raise PhysicalResultError(f"{field}.rank_timings_us must cover the launched world")
    rank_timings = [
        _samples(raw_values, f"{field}.rank_timings_us[{rank}]") for rank, raw_values in enumerate(raw_rank_timings)
    ]
    if any(len(values) != iterations for values in rank_timings):
        raise PhysicalResultError(f"{field}.rank_timings_us sample count disagrees with iterations")
    samples = _samples(value["timings_us"], f"{field}.timings_us")
    if len(samples) != iterations:
        raise PhysicalResultError(f"{field}.timings_us sample count disagrees with iterations")
    maxima = tuple(max(rank_timings[rank][iteration] for rank in range(world_size)) for iteration in range(iterations))
    if samples != maxima:
        raise PhysicalResultError(f"{field}.timings_us disagree with max-rank timings")
    metrics = _object(value["metrics"], f"{field}.metrics")
    _strict(metrics, f"{field}.metrics", ("count", "median_us", "iqr_us", "min_us", "max_us"))
    expected_metrics = {
        "count": len(samples),
        "median_us": _median(samples),
        "iqr_us": _iqr(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }
    if _integer(metrics["count"], f"{field}.metrics.count", minimum=1) != expected_metrics["count"]:
        raise PhysicalResultError(f"{field}.metrics.count disagrees with retained timings")
    for name in ("median_us", "iqr_us", "min_us", "max_us"):
        if abs(_finite(metrics[name], f"{field}.metrics.{name}") - expected_metrics[name]) > 0.001:
            raise PhysicalResultError(f"{field}.metrics.{name} disagrees with retained timings")
    return dict(value), samples


def _decision_gate_payload(
    stdout: str,
    *,
    world_size: int,
    parameters: Mapping[str, Any],
    measurement_schema: str,
    repetition: Optional[int],
) -> Tuple[Mapping[str, Any], Tuple[float, ...]]:
    replicated = measurement_schema == DECISION_GATE_REPLICATED_MEASUREMENT_SCHEMA
    if replicated:
        if repetition is None or isinstance(repetition, bool) or not 0 <= repetition <= 999:
            raise PhysicalResultError("replicated decision-gate cell repetition is invalid")
        stdout_schema = DECISION_GATE_REPLICATED_STDOUT_SCHEMA
        order_method = DECISION_GATE_REPLICATED_ORDER_METHOD
        contracts = _DECISION_GATE_REPLICATED_REPRESENTATION_CONTRACTS
    else:
        stdout_schema = DECISION_GATE_STDOUT_SCHEMA
        order_method = DECISION_GATE_ORDER_METHOD
        contracts = _DECISION_GATE_REPRESENTATION_CONTRACTS
    payload = _last_json_object(stdout)
    _strict(
        payload,
        "decision-gate producer stdout",
        (
            "schema",
            "request",
            "materialization",
            "policy",
            "execution",
            "runtime",
            "correctness",
            "representations",
            "claims",
        ),
    )
    if payload["schema"] != stdout_schema:
        raise PhysicalResultError(f"decision-gate producer stdout requires schema {stdout_schema!r}")
    request = _decision_gate_reference(
        payload["request"],
        "decision-gate producer stdout.request",
        {
            "format": "commcanary.qualification_request.v2",
            "request_id": _text(parameters.get("expected_request_id"), "workload.parameters.expected_request_id") or "",
        },
    )
    materialization = _decision_gate_reference(
        payload["materialization"],
        "decision-gate producer stdout.materialization",
        {
            "materialization_id": _text(
                parameters.get("expected_materialization_id"),
                "workload.parameters.expected_materialization_id",
            )
            or "",
            "program_sha256": _text(
                parameters.get("expected_program_sha256"),
                "workload.parameters.expected_program_sha256",
            )
            or "",
        },
    )
    policy = _decision_gate_reference(
        payload["policy"],
        "decision-gate producer stdout.policy",
        {
            "format": "commcanary.qualification_policy.v1",
            "policy_id": _text(parameters.get("expected_policy_id"), "workload.parameters.expected_policy_id") or "",
        },
    )
    execution = _object(payload["execution"], "decision-gate producer stdout.execution")
    execution_fields = {
        "world_size",
        "iterations",
        "warmup",
        "timing_semantics",
        "order_method",
        "representation_order_by_iteration",
        "source_event_count",
        "stratified_method",
        "stratified_source_event_indices",
    }
    if replicated:
        execution_fields.add("configuration_repetition")
    _strict(execution, "decision-gate producer stdout.execution", execution_fields)
    iterations = _integer(parameters.get("iterations"), "workload.parameters.iterations", minimum=1, maximum=1000)
    warmup = _integer(parameters.get("warmup"), "workload.parameters.warmup", maximum=100)
    source_event_count = _integer(
        parameters.get("expected_source_event_count"),
        "workload.parameters.expected_source_event_count",
        minimum=1,
    )
    if (
        execution["world_size"] != world_size
        or execution["iterations"] != iterations
        or execution["warmup"] != warmup
        or execution["timing_semantics"] != DECISION_GATE_TIMING_SEMANTICS
        or execution["order_method"] != order_method
        or execution["source_event_count"] != source_event_count
        or execution["stratified_method"] != DECISION_GATE_STRATIFIED_METHOD
    ):
        raise PhysicalResultError("decision-gate execution contract disagrees with the frozen workload")
    if replicated and execution["configuration_repetition"] != repetition:
        raise PhysicalResultError("decision-gate configuration repetition disagrees with its manifest repetition")
    selected = execution["stratified_source_event_indices"]
    expected_selected = parameters.get("expected_stratified_source_event_indices")
    if not isinstance(selected, list) or selected != expected_selected or not selected:
        raise PhysicalResultError("decision-gate stratified selection disagrees with the frozen workload")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < source_event_count
        for index in selected
    ):
        raise PhysicalResultError("decision-gate stratified selection contains an invalid source index")
    raw_orders = execution["representation_order_by_iteration"]
    if not isinstance(raw_orders, list) or len(raw_orders) != iterations:
        raise PhysicalResultError("decision-gate representation order inventory is incomplete")
    for iteration, order in enumerate(raw_orders):
        expected_order = list(
            representation_order(
                iteration,
                configuration_repetition=repetition if replicated else None,
            )
        )
        if order != expected_order:
            raise PhysicalResultError(f"decision-gate representation order disagrees at iteration {iteration}")

    runtime = _object(payload["runtime"], "decision-gate producer stdout.runtime")
    _strict(
        runtime,
        "decision-gate producer stdout.runtime",
        ("torch_version", "torch_cuda_version", "runtime_nccl_version_code", "distributed_backend"),
    )
    _text(runtime["torch_version"], "decision-gate producer stdout.runtime.torch_version", maximum=128)
    _text(runtime["torch_cuda_version"], "decision-gate producer stdout.runtime.torch_cuda_version", maximum=128)
    _integer(
        runtime["runtime_nccl_version_code"],
        "decision-gate producer stdout.runtime.runtime_nccl_version_code",
        minimum=1,
        maximum=99999,
    )
    if runtime["distributed_backend"] != "nccl":
        raise PhysicalResultError("decision-gate producer must use NCCL")

    correctness = _object(payload["correctness"], "decision-gate producer stdout.correctness")
    _strict(
        correctness,
        "decision-gate producer stdout.correctness",
        ("status", "semantics", "checks_per_rank", "total_check_count"),
    )
    expected_checks = parameters.get("expected_correctness_checks_per_rank")
    if (
        correctness["status"] != "passed"
        or correctness["semantics"] != "one-source-value-sum-check-per-collective-shape"
        or correctness["checks_per_rank"] != expected_checks
        or not isinstance(expected_checks, list)
        or len(expected_checks) != world_size
    ):
        raise PhysicalResultError("decision-gate correctness inventory disagrees with the frozen workload")
    parsed_checks = [
        _integer(value, f"decision-gate correctness checks_per_rank[{rank}]", minimum=1)
        for rank, value in enumerate(expected_checks)
    ]
    if correctness["total_check_count"] != sum(parsed_checks):
        raise PhysicalResultError("decision-gate correctness total is inconsistent")

    raw_representations = _object(payload["representations"], "decision-gate producer stdout.representations")
    if set(raw_representations) != set(_DECISION_GATE_REPRESENTATIONS):
        raise PhysicalResultError("decision-gate representation inventory is not closed")
    normalized: Dict[str, Any] = {}
    source_samples: Optional[Tuple[float, ...]] = None
    for representation in _DECISION_GATE_REPRESENTATIONS:
        if representation == "stratified":
            event_count = len(selected)
            template_count = len(selected)
        else:
            event_count = source_event_count
            template_count = len(selected) if representation == "isolated" else source_event_count
        normalized[representation], samples = _decision_gate_representation(
            raw_representations[representation],
            representation=representation,
            world_size=world_size,
            iterations=iterations,
            expected_event_count=event_count,
            expected_template_count=template_count,
            contracts=contracts,
        )
        if representation == "source":
            source_samples = samples
    claims = _object(payload["claims"], "decision-gate producer stdout.claims")
    if dict(claims) != {
        "physical_execution": "same_allocation_self_reported",
        "physical_decision_fidelity": "not_analyzed",
        "qualification_verdict": "policy_bound_not_issued",
    }:
        raise PhysicalResultError("decision-gate claims exceed the pre-analysis boundary")
    if source_samples is None:  # pragma: no cover - closed inventory above
        raise PhysicalResultError("decision-gate source timings are missing")
    return {
        **dict(payload),
        "request": request,
        "materialization": materialization,
        "policy": policy,
        "representations": normalized,
    }, source_samples


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
    repetition: Optional[int] = None,
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
        source_capture = None
        source_samples = None
    elif measurement_schema == QUALIFICATION_MEASUREMENT_SCHEMA:
        world_size, _ = validate_physical_layout(parameter_object)
        payload, samples, source_capture, source_samples = _qualification_payload(
            stdout,
            world_size=world_size,
            parameters=parameter_object,
        )
    elif measurement_schema in {DECISION_GATE_MEASUREMENT_SCHEMA, DECISION_GATE_REPLICATED_MEASUREMENT_SCHEMA}:
        world_size, _ = validate_physical_layout(parameter_object)
        payload, samples = _decision_gate_payload(
            stdout,
            world_size=world_size,
            parameters=parameter_object,
            measurement_schema=measurement_schema,
            repetition=repetition,
        )
        source_capture = None
        source_samples = None
    else:
        world_size, _ = validate_physical_layout(parameter_object)
        payload, samples = _torch_payload(stdout, world_size, producer_schema)
        source_capture = None
        source_samples = None
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
    elif measurement_schema == QUALIFICATION_MEASUREMENT_SCHEMA:
        assert payload is not None
        assert source_capture is not None
        assert source_samples is not None
        correctness = _object(payload["correctness_validation"], "qualification producer correctness validation")
        source_scheduler = _object(
            source_capture["scheduler"],
            "qualification producer source scheduler",
        )
        if str(runtime_object["hostname"]).split(".", 1)[0] != source_scheduler["node"]:
            raise PhysicalResultError("qualification comparison is not same-node")
        replay_median = _median(samples)
        source_median = _median(source_samples)
        if source_median <= 0.0:
            raise PhysicalResultError("qualification source median must be positive")
        signed_error = replay_median - source_median
        relative_error_pct = signed_error / source_median * 100.0
        measurement.update(
            {
                "replay_mode": _text(
                    parameter_object.get("replay_mode"),
                    "workload.parameters.replay_mode",
                ),
                "request_id": _sha256(payload["request_id"], "qualification request_id"),
                "materialization_id": _sha256(
                    payload["materialization_id"],
                    "qualification materialization_id",
                ),
                "program_sha256": _sha256(
                    payload["program_sha256"],
                    "qualification program_sha256",
                ),
                "correctness_check_count": _integer(
                    correctness["total_check_count"],
                    "qualification correctness_check_count",
                    maximum=1_000_000,
                ),
                "rank_compute_operations_per_pass": list(payload["rank_compute_operations_per_pass"]),
                "rank_tensor_bytes": list(payload["rank_tensor_bytes"]),
                "source_samples_us": list(source_samples),
                "source_value_us": source_median,
                "source_iqr_us": _iqr(source_samples),
                "source_timing_semantics": source_capture["timing_semantics"],
                "source_capture_diagnostic_id": source_capture["diagnostic_id"],
                "source_capture_evidence_sha256": source_capture["evidence_sha256"],
                "source_capture_stdout_sha256": source_capture["stdout_sha256"],
                "source_capture_job_id": source_scheduler["job_id"],
                "source_capture_node": source_scheduler["node"],
                "signed_median_error_us": signed_error,
                "relative_median_error_pct": relative_error_pct,
                "absolute_relative_median_error_pct": abs(relative_error_pct),
                "comparison_claims": {
                    "single_configuration_timing_comparison": "diagnostic",
                    "physical_fidelity": "unproven",
                    "multi_configuration_ranking": "not_measured",
                    "qualification_verdict": "not_issued",
                },
            }
        )
    elif measurement_schema in {DECISION_GATE_MEASUREMENT_SCHEMA, DECISION_GATE_REPLICATED_MEASUREMENT_SCHEMA}:
        assert payload is not None
        payload_runtime = _object(payload["runtime"], "decision-gate producer runtime")
        for field in ("torch_version", "torch_cuda_version", "runtime_nccl_version_code"):
            if payload_runtime[field] != runtime_object[field]:
                raise PhysicalResultError(f"decision-gate producer runtime.{field} disagrees with the cell runtime")
        correctness = _object(payload["correctness"], "decision-gate producer correctness")
        measurement.update(
            {
                "request": dict(_object(payload["request"], "decision-gate producer request")),
                "materialization": dict(_object(payload["materialization"], "decision-gate producer materialization")),
                "policy": dict(_object(payload["policy"], "decision-gate producer policy")),
                "execution": dict(_object(payload["execution"], "decision-gate producer execution")),
                "correctness_check_count": _integer(
                    correctness["total_check_count"],
                    "decision-gate correctness total_check_count",
                    minimum=1,
                ),
                "representations": dict(_object(payload["representations"], "decision-gate producer representations")),
                "decision_claims": dict(_object(payload["claims"], "decision-gate producer claims")),
            }
        )
    return measurement
