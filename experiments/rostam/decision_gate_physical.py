#!/usr/bin/env python3
"""Run the predeclared same-allocation physical decision-fidelity gate.

Every rank verifies the policy-bound qualification request and materialization
before importing PyTorch.  The runner then interleaves a trace-derived
reference, the exact materialization control, two practical baselines, and two causal
ablations inside one process group.  All representations use the same CUDA
event timing method and allocation; rank 0 emits one strict JSON document for
the manifest-owned physical adapter.

This runner does not decide whether CommCanary passes the product gate.  It
retains raw per-rank samples and explicitly leaves decision fidelity unanalyzed
until every frozen configuration has one selected terminal attempt.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import socket
import statistics
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from commcanary.artifacts import load_json, validate_qualification_policy
from commcanary.artifacts.dtypes import dtype_size_bytes
from commcanary.execution import (
    QualificationExecutionPlan,
    distributed_execution_environment,
    preflight_qualification_execution,
)
from commcanary.services import verify_qualification_request

from .decision_gate_schedule import (
    REPLICATED_ORDER_METHOD,
    REPRESENTATION_IDS,
    representation_order,
    warmup_representation_order,
)
from .harness import canonical_sha256, utc_timestamp
from .lib.cell_entrypoint import CellEntrypointError, _run_bounded_probe
from .qualification_physical import stage_qualification_inputs

DECISION_GATE_STDOUT_SCHEMA = "commcanary.rostam.decision-gate.stdout.v1"
DECISION_GATE_REPLICATED_STDOUT_SCHEMA = "commcanary.rostam.decision-gate.stdout.v2"
DECISION_GATE_TIMING_SEMANTICS = "maximum-rank-cuda-event-whole-program-duration"
DECISION_GATE_ORDER_METHOD = "iteration-rotated-latin-cycle.v1"
DECISION_GATE_REPLICATED_ORDER_METHOD = REPLICATED_ORDER_METHOD
STRATIFIED_METHOD = "first-observed-per-collective-shape.v1"
REPRESENTATION_METADATA = {
    "source": ("ground_truth", "direct-source-issue-rank-work-wait"),
    "exact_work": ("product_candidate", "verified-materialization-issue-rank-work-wait"),
    "stratified": ("kill_condition_baseline", STRATIFIED_METHOD),
    "isolated": ("incumbent_baseline", "full-message-sequence-blocking-all-reduce-no-compute"),
    "no_overlap": ("causal_ablation", "blocking-all-reduce-then-exact-rank-work"),
    "no_rank_skew": ("causal_ablation", "issue-rank-zero-work-on-every-rank-wait"),
}
REPLICATED_REPRESENTATION_METADATA = {
    **REPRESENTATION_METADATA,
    "source": ("trace_derived_reference", "direct-source-issue-rank-work-wait"),
    "exact_work": ("exact_materialization_control", "verified-materialization-issue-rank-work-wait"),
}
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 300
_MAX_PROC_MAPS_BYTES = 4 * 1024 * 1024
_CYCLE_TELEMETRY_SCHEMA = "commcanary.rostam.decision-gate-cycle-telemetry.v1"
_CYCLE_TELEMETRY_METHOD = "bounded-between-six-row-cycles.v1"
_CYCLE_TELEMETRY_GPU_FIELDS = (
    "index",
    "uuid",
    "performance_state",
    "temperature_c",
    "power_draw_w",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "throttle_reasons_active",
    "ecc_corrected_volatile_total",
    "ecc_uncorrected_volatile_total",
)


@dataclass(frozen=True)
class GateEvent:
    """One source-bound all-reduce and its exact per-rank GEMM work."""

    request: int
    source_event_index: int
    pg_id: int
    ranks: Tuple[int, ...]
    elements: int
    dtype: str
    recipes: Tuple[Tuple[Tuple[str, int, int, int], ...], ...]

    @property
    def stratum(self) -> Tuple[int, Tuple[int, ...], int, str]:
        return self.pg_id, self.ranks, self.elements, self.dtype


@dataclass(frozen=True)
class CompiledOp:
    """One schema-free instruction compiled before warmup and timing."""

    kind: str
    request: int
    pg_id: Optional[int] = None
    recipe: Optional[Tuple[str, int, int, int]] = None


@dataclass(frozen=True, eq=False)
class RuntimeOp:
    """One instruction bound to preallocated runtime objects."""

    kind: str
    request: int
    group: Any = None
    tensor: Any = None
    operands: Optional[Tuple[Any, Any, Any]] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--qualification-policy", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--replay-program", type=Path, required=True)
    parser.add_argument("--expected-request-id", required=True)
    parser.add_argument("--expected-materialization-id", required=True)
    parser.add_argument("--expected-program-sha256", required=True)
    parser.add_argument("--expected-policy-id", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--configuration-repetition", type=int)
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS,
    )
    return parser


def _strict_positive(value: int, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise SystemExit(f"{field} must be an integer in [1, {maximum}]")
    return value


def _required_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{field} must be an integer")
    return int(value)


def _recipe_tuple(
    value: Any, *, ranks: Tuple[int, ...], field: str
) -> Tuple[Tuple[Tuple[str, int, int, int], ...], ...]:
    if not isinstance(value, Mapping) or set(value) != {str(rank) for rank in ranks}:
        raise SystemExit(f"{field} must cover exactly ranks {list(ranks)!r}")
    result: List[Tuple[Tuple[str, int, int, int], ...]] = []
    for rank in ranks:
        raw_operations = value[str(rank)]
        if not isinstance(raw_operations, list):
            raise SystemExit(f"{field}[{rank}] must be an array")
        operations: List[Tuple[str, int, int, int]] = []
        for index, operation in enumerate(raw_operations):
            if not isinstance(operation, Mapping) or operation.get("op") != "gemm":
                raise SystemExit(f"{field}[{rank}][{index}] must be a GEMM")
            dtype = operation.get("dtype")
            if not isinstance(dtype, str):
                raise SystemExit(f"{field}[{rank}][{index}].dtype must be a string")
            dimensions = []
            for dimension in ("m", "n", "k"):
                raw = operation.get(dimension)
                if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                    raise SystemExit(f"{field}[{rank}][{index}].{dimension} must be positive")
                dimensions.append(raw)
            operations.append((dtype, dimensions[0], dimensions[1], dimensions[2]))
        result.append(tuple(operations))
    return tuple(result)


def source_events(trace: Mapping[str, Any], *, world_size: int) -> Tuple[GateEvent, ...]:
    raw_events = trace.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise SystemExit("decision-gate source trace must contain events")
    expected_ranks = tuple(range(world_size))
    result: List[GateEvent] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise SystemExit(f"decision-gate source event {index} must be an object")
        ranks_raw = raw.get("ranks")
        if not isinstance(ranks_raw, list) or tuple(ranks_raw) != expected_ranks:
            raise SystemExit(f"decision-gate source event {index} must use the dense launched world")
        if raw.get("op") != "all_reduce" or raw.get("reduction_op") != "sum":
            raise SystemExit(f"decision-gate source event {index} must be SUM all_reduce")
        dtype = raw.get("dtype")
        byte_count = raw.get("bytes")
        if not isinstance(dtype, str) or isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise SystemExit(f"decision-gate source event {index} lacks exact dtype/bytes")
        element_size = dtype_size_bytes(dtype)
        if byte_count <= 0 or byte_count % element_size:
            raise SystemExit(f"decision-gate source event {index} bytes are not dtype aligned")
        result.append(
            GateEvent(
                request=index,
                source_event_index=index,
                pg_id=0,
                ranks=expected_ranks,
                elements=byte_count // element_size,
                dtype=dtype,
                recipes=_recipe_tuple(
                    raw.get("compute_recipe_by_rank"),
                    ranks=expected_ranks,
                    field=f"decision-gate source event {index}.compute_recipe_by_rank",
                ),
            )
        )
    return tuple(result)


def plan_events(plan: QualificationExecutionPlan) -> Tuple[GateEvent, ...]:
    groups = dict(plan.groups)
    result: List[GateEvent] = []
    entries = plan.entries
    index = 0
    while index < len(entries):
        entry = entries[index]
        if entry.get("comms") == "init":
            index += 1
            continue
        if index + 2 >= len(entries):
            raise SystemExit("decision-gate materialization ends inside an issue/work/wait region")
        compute = entries[index + 1]
        wait = entries[index + 2]
        if (
            entry.get("comms") != "all_reduce"
            or entry.get("reduction_op") != "sum"
            or compute.get("compute") != "gemm_recipe"
            or wait.get("comms") != "wait"
            or compute.get("overlap_request") != entry.get("req")
            or wait.get("req") != entry.get("req")
        ):
            raise SystemExit("decision-gate materialization is outside the exact all-reduce/work/wait domain")
        request = _required_integer(entry.get("req"), "decision-gate materialization request")
        source_index = _required_integer(
            entry.get("source_event_index"),
            "decision-gate materialization source_event_index",
        )
        pg_id = _required_integer(entry.get("pg_id"), "decision-gate materialization pg_id")
        elements = _required_integer(
            entry.get("in_msg_size"),
            "decision-gate materialization in_msg_size",
        )
        ranks = groups.get(pg_id)
        if ranks is None or list(ranks) != entry.get("global_ranks"):
            raise SystemExit("decision-gate materialization process-group identity is inconsistent")
        if elements <= 0 or entry.get("out_msg_size") != elements:
            raise SystemExit("decision-gate materialization all-reduce shape is inconsistent")
        dtype = entry.get("dtype")
        if not isinstance(dtype, str):
            raise SystemExit("decision-gate materialization dtype is missing")
        result.append(
            GateEvent(
                request=request,
                source_event_index=source_index,
                pg_id=pg_id,
                ranks=ranks,
                elements=elements,
                dtype=dtype,
                recipes=_recipe_tuple(
                    compute.get("recipe_by_rank"),
                    ranks=ranks,
                    field=f"decision-gate materialization request {request}.recipe_by_rank",
                ),
            )
        )
        index += 3
    if not result:
        raise SystemExit("decision-gate materialization contains no executable events")
    return tuple(result)


def compile_event_program(
    events: Sequence[GateEvent],
    *,
    rank: int,
    mode: str = "overlap",
) -> Tuple[CompiledOp, ...]:
    """Compile one representation without retaining mappings or schema rows."""

    if mode not in {"overlap", "isolated", "no_overlap", "no_rank_skew"}:
        raise SystemExit(f"unsupported decision-gate compile mode {mode!r}")
    compiled: List[CompiledOp] = []
    for event in events:
        if rank not in event.ranks:
            raise SystemExit(f"decision-gate rank {rank} is outside request {event.request}")
        recipe_rank = event.ranks[0] if mode == "no_rank_skew" else rank
        recipe_index = event.ranks.index(recipe_rank)
        collective_kind = "collective_blocking" if mode in {"isolated", "no_overlap"} else "collective_start"
        compiled.append(
            CompiledOp(
                kind=collective_kind,
                request=event.request,
                pg_id=event.pg_id,
            )
        )
        if mode != "isolated":
            compiled.extend(
                CompiledOp(kind="gemm", request=event.request, recipe=recipe) for recipe in event.recipes[recipe_index]
            )
        if collective_kind == "collective_start":
            compiled.append(CompiledOp(kind="collective_wait", request=event.request))
    return tuple(compiled)


def matching_source_and_exact_programs(
    source: Sequence[GateEvent],
    materialized: Sequence[GateEvent],
    *,
    rank: int,
) -> Tuple[CompiledOp, ...]:
    """Compile both evaluated paths and refuse any instruction-level mismatch."""

    if tuple(source) != tuple(materialized):
        raise SystemExit("decision-gate source and exact-work event programs disagree")
    source_program = compile_event_program(source, rank=rank)
    exact_program = compile_event_program(materialized, rank=rank)
    if source_program != exact_program:
        raise SystemExit("decision-gate source and exact-work compiled programs disagree")
    return source_program


def _bind_runtime_program(
    program: Sequence[CompiledOp],
    *,
    groups: Mapping[int, Any],
    communication: Mapping[int, Any],
    gemms: Mapping[Tuple[str, int, int, int], Tuple[Any, Any, Any]],
) -> Tuple[RuntimeOp, ...]:
    """Bind every lookup before a program can enter warmup or timing."""

    bound: List[RuntimeOp] = []
    tensor_owners: Dict[int, int] = {}
    for operation in program:
        if operation.kind in {"collective_start", "collective_blocking"}:
            if operation.pg_id is None or operation.pg_id not in groups or operation.request not in communication:
                raise SystemExit("decision-gate compiled collective has an unbound runtime object")
            tensor = communication[operation.request]
            tensor_identity = id(tensor)
            prior_owner = tensor_owners.setdefault(tensor_identity, operation.request)
            if prior_owner != operation.request:
                raise SystemExit("decision-gate distinct requests alias one communication buffer")
            bound.append(
                RuntimeOp(
                    kind=operation.kind,
                    request=operation.request,
                    group=groups[operation.pg_id],
                    tensor=tensor,
                )
            )
        elif operation.kind == "gemm":
            if operation.recipe is None or operation.recipe not in gemms:
                raise SystemExit("decision-gate compiled GEMM has no preallocated operands")
            bound.append(
                RuntimeOp(
                    kind=operation.kind,
                    request=operation.request,
                    operands=gemms[operation.recipe],
                )
            )
        elif operation.kind == "collective_wait":
            bound.append(RuntimeOp(kind=operation.kind, request=operation.request))
        else:
            raise SystemExit(f"unsupported compiled decision-gate operation {operation.kind!r}")
    return tuple(bound)


def _run_runtime_program(
    program: Sequence[RuntimeOp],
    *,
    dist: Any,
    torch: Any,
    wait_callback: Optional[Any] = None,
) -> None:
    """Execute a prebound program without schema access or mapping traversal."""

    pending: Dict[int, Any] = {}
    for operation in program:
        if operation.kind == "gemm":
            if operation.operands is None:  # pragma: no cover - guarded by binding
                raise SystemExit("decision-gate GEMM operands disappeared")
            left, right, output = operation.operands
            torch.mm(left, right, out=output)
        elif operation.kind == "collective_blocking":
            dist.all_reduce(operation.tensor, op=dist.ReduceOp.SUM, group=operation.group)
        elif operation.kind == "collective_start":
            if operation.request in pending:
                raise SystemExit("decision-gate compiled program reused a pending request")
            pending[operation.request] = dist.all_reduce(
                operation.tensor,
                op=dist.ReduceOp.SUM,
                group=operation.group,
                async_op=True,
            )
        elif operation.kind == "collective_wait":
            work = pending.pop(operation.request, None)
            if work is None:
                raise SystemExit("decision-gate compiled wait has no pending request")
            work.wait()
            if wait_callback is not None:
                wait_callback(operation.request)
        else:  # pragma: no cover - guarded by binding
            raise SystemExit(f"unsupported bound decision-gate operation {operation.kind!r}")
    if pending:
        raise SystemExit("decision-gate compiled program retained pending requests")


def stratified_indices(events: Sequence[GateEvent]) -> Tuple[int, ...]:
    """Select the first source event in every collective-shape stratum."""

    selected: List[int] = []
    seen = set()
    for index, event in enumerate(events):
        if event.stratum not in seen:
            seen.add(event.stratum)
            selected.append(index)
    return tuple(selected)


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
    return float(statistics.median(upper) - statistics.median(lower))


def result_payload(
    *,
    request: Mapping[str, Any],
    materialization_id: str,
    program_sha256: str,
    policy: Mapping[str, Any],
    world_size: int,
    iterations: int,
    warmup: int,
    source_event_count: int,
    selected_indices: Sequence[int],
    gathered: Sequence[Mapping[str, Any]],
    correctness_checks_per_rank: Sequence[int],
    runtime: Mapping[str, Any],
    configuration_repetition: Optional[int] = None,
    backend_smoke_checks_per_rank: Optional[Sequence[int]] = None,
    source_output_commitment_sha256_by_rank: Optional[Sequence[str]] = None,
    exact_work_output_commitment_sha256_by_rank: Optional[Sequence[str]] = None,
    telemetry_checkpoints: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if len(gathered) != world_size:
        raise SystemExit("decision-gate timing inventory does not cover the launched world")
    if configuration_repetition is not None and configuration_repetition < 0:
        raise SystemExit("configuration_repetition must be non-negative")
    representations: Dict[str, Any] = {}
    for representation in REPRESENTATION_IDS:
        by_rank: List[List[float]] = []
        for expected_rank, raw in enumerate(gathered):
            if raw.get("rank") != expected_rank:
                raise SystemExit(f"decision-gate timing owner disagrees at rank {expected_rank}")
            raw_timings = raw.get("timings_us")
            if not isinstance(raw_timings, Mapping):
                raise SystemExit(f"decision-gate rank {expected_rank} timings must be an object")
            values = raw_timings.get(representation)
            if not isinstance(values, list) or len(values) != iterations:
                raise SystemExit(f"decision-gate rank {expected_rank} {representation} sample count is invalid")
            parsed = [float(value) for value in values]
            if any(not math.isfinite(value) or value < 0.0 for value in parsed):
                raise SystemExit(f"decision-gate rank {expected_rank} {representation} timings are invalid")
            by_rank.append(parsed)
        maxima = [max(by_rank[rank][iteration] for rank in range(world_size)) for iteration in range(iterations)]
        rounded_by_rank = [[round(value, 3) for value in values] for values in by_rank]
        rounded_maxima = [round(value, 3) for value in maxima]
        metadata = REPRESENTATION_METADATA if configuration_repetition is None else REPLICATED_REPRESENTATION_METADATA
        category, semantics = metadata[representation]
        if representation == "stratified":
            executed_events = len(selected_indices)
            template_count = len(selected_indices)
        else:
            executed_events = source_event_count
            template_count = len(selected_indices) if representation == "isolated" else source_event_count
        representations[representation] = {
            "category": category,
            "semantics": semantics,
            "executed_event_count": executed_events,
            "template_count": template_count,
            "rank_timings_us": rounded_by_rank,
            "timings_us": rounded_maxima,
            "metrics": {
                "count": len(rounded_maxima),
                "median_us": round(_median(rounded_maxima), 3),
                "iqr_us": round(_iqr(rounded_maxima), 3),
                "min_us": round(min(rounded_maxima), 3),
                "max_us": round(max(rounded_maxima), 3),
            },
        }
    measured_orders = [
        list(
            representation_order(
                index,
                configuration_repetition=configuration_repetition,
            )
        )
        for index in range(iterations)
    ]
    execution: Dict[str, Any] = {
        "world_size": world_size,
        "iterations": iterations,
        "warmup": warmup,
        "timing_semantics": DECISION_GATE_TIMING_SEMANTICS,
        "order_method": (
            DECISION_GATE_ORDER_METHOD if configuration_repetition is None else DECISION_GATE_REPLICATED_ORDER_METHOD
        ),
        "representation_order_by_iteration": measured_orders,
        "source_event_count": source_event_count,
        "stratified_method": STRATIFIED_METHOD,
        "stratified_source_event_indices": list(selected_indices),
    }
    if configuration_repetition is not None:
        if telemetry_checkpoints is None:
            raise SystemExit("replicated decision-gate cycle telemetry is incomplete")
        warmup_orders = [
            list(
                warmup_representation_order(
                    index,
                    configuration_repetition=configuration_repetition,
                )
            )
            for index in range(warmup)
        ]
        execution["configuration_repetition"] = configuration_repetition
        execution["representation_order_by_warmup"] = warmup_orders
        execution["representation_schedule_sha256"] = canonical_sha256(
            {
                "warmup": warmup_orders,
                "measured": measured_orders,
            }
        )
    if configuration_repetition is None:
        correctness = {
            "status": "passed",
            "semantics": "one-source-value-sum-check-per-collective-shape",
            "checks_per_rank": list(correctness_checks_per_rank),
            "total_check_count": sum(correctness_checks_per_rank),
        }
    else:
        if (
            backend_smoke_checks_per_rank is None
            or source_output_commitment_sha256_by_rank is None
            or exact_work_output_commitment_sha256_by_rank is None
            or len(backend_smoke_checks_per_rank) != world_size
            or len(correctness_checks_per_rank) != world_size
            or len(source_output_commitment_sha256_by_rank) != world_size
            or len(exact_work_output_commitment_sha256_by_rank) != world_size
            or list(source_output_commitment_sha256_by_rank) != list(exact_work_output_commitment_sha256_by_rank)
        ):
            raise SystemExit("replicated decision-gate correctness evidence is incomplete")
        correctness = {
            "status": "passed",
            "semantics": "complete-source-and-exact-work-output-commitment-comparison",
            "checks_per_rank": list(correctness_checks_per_rank),
            "total_check_count": sum(correctness_checks_per_rank),
            "source_output_commitment_sha256_by_rank": list(source_output_commitment_sha256_by_rank),
            "exact_work_output_commitment_sha256_by_rank": list(exact_work_output_commitment_sha256_by_rank),
            "backend_smoke_check": {
                "status": "passed",
                "semantics": "one-blocking-sum-check-per-collective-shape",
                "checks_per_rank": list(backend_smoke_checks_per_rank),
                "total_check_count": sum(backend_smoke_checks_per_rank),
            },
        }
    return {
        "schema": (
            DECISION_GATE_STDOUT_SCHEMA if configuration_repetition is None else DECISION_GATE_REPLICATED_STDOUT_SCHEMA
        ),
        "request": {
            "format": request["format"],
            "request_id": request["request_id"],
        },
        "materialization": {
            "materialization_id": materialization_id,
            "program_sha256": program_sha256,
        },
        "policy": {
            "format": policy["format"],
            "policy_id": policy["policy_id"],
        },
        "execution": execution,
        "runtime": dict(runtime),
        "correctness": correctness,
        **({} if telemetry_checkpoints is None else {"telemetry_checkpoints": dict(telemetry_checkpoints)}),
        "representations": representations,
        "claims": {
            "physical_execution": "same_allocation_self_reported",
            "physical_decision_fidelity": "not_analyzed",
            "qualification_verdict": "policy_bound_not_issued",
        },
    }


def _torch_dtype_map(torch: Any) -> Mapping[str, Any]:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }


def _selected_nccl_library() -> Path:
    raw = os.environ.get("LD_LIBRARY_PATH")
    if raw is None or not raw or os.pathsep in raw:
        raise SystemExit("decision-gate runtime requires one explicit NCCL library directory")
    directory = Path(raw)
    if not directory.is_absolute() or not directory.is_dir():
        raise SystemExit("decision-gate NCCL library directory is invalid")
    for name in ("libnccl.so.2", "libnccl.so"):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit("decision-gate runtime cannot find the selected NCCL library")


def _verify_exclusively_loaded_nccl(library_path: Path, proc_maps_path: Path) -> None:
    try:
        with proc_maps_path.open("rb") as handle:
            data = handle.read(_MAX_PROC_MAPS_BYTES + 1)
    except OSError as exc:
        raise SystemExit("decision-gate runtime cannot inspect its loaded NCCL library") from exc
    if len(data) > _MAX_PROC_MAPS_BYTES:
        raise SystemExit("decision-gate runtime library map exceeds the supported limit")
    mapped = set()
    for line in data.decode("utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        candidate = Path(fields[5])
        if candidate.name.startswith("libnccl.so"):
            mapped.add(candidate.resolve())
    if mapped != {library_path.resolve()}:
        raise SystemExit("decision-gate runtime loaded an unexpected NCCL library")


def _runtime_nccl_version_code(
    library_path: Path,
    *,
    proc_maps_path: Path = Path("/proc/self/maps"),
) -> int:
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise SystemExit("decision-gate runtime cannot load the selected NCCL library") from exc
    value = ctypes.c_int()
    try:
        status = library.ncclGetVersion(ctypes.byref(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise SystemExit("decision-gate runtime cannot query the selected NCCL library") from exc
    if status != 0 or value.value <= 0:
        raise SystemExit("decision-gate runtime reported an invalid NCCL version")
    _verify_exclusively_loaded_nccl(library_path, proc_maps_path)
    return int(value.value)


def _normalized_torch_version(torch: Any) -> str:
    raw = getattr(torch, "__version__", None)
    if raw is None:
        raise SystemExit("decision-gate runtime did not report a PyTorch version")
    version = str(raw).split("+", 1)[0]
    if not version:
        raise SystemExit("decision-gate runtime reported an invalid PyTorch version")
    return version


def _cycle_telemetry_snapshot(label: str) -> Dict[str, Any]:
    """Capture one bounded between-cycle GPU, node, ECC, and Xid snapshot."""

    query_prefix = "index,uuid,pstate,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,"
    query_suffix = ",ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total"
    raw_gpu: Optional[str] = None
    for throttle_field in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        try:
            raw_gpu = _run_bounded_probe(
                (
                    "nvidia-smi",
                    f"--query-gpu={query_prefix}{throttle_field}{query_suffix}",
                    "--format=csv,noheader,nounits",
                )
            )
            break
        except CellEntrypointError:
            continue
    if raw_gpu is None:
        raise SystemExit("decision-gate cycle telemetry cannot query GPU throttle/ECC state")
    try:
        parsed_rows = list(csv.reader(raw_gpu.splitlines(), strict=True))
    except csv.Error as exc:
        raise SystemExit("decision-gate cycle GPU telemetry is not valid CSV") from exc
    if not parsed_rows:
        raise SystemExit("decision-gate cycle GPU telemetry is empty")
    gpus: List[Dict[str, Any]] = []
    for row_index, row in enumerate(parsed_rows):
        fields = [value.strip() for value in row]
        if len(fields) != len(_CYCLE_TELEMETRY_GPU_FIELDS):
            raise SystemExit(f"decision-gate cycle GPU row {row_index} has an invalid field count")
        try:
            index = int(fields[0])
            temperature_c = int(fields[3])
            power_draw_w = float(fields[4])
            sm_clock_mhz = int(fields[5])
            memory_clock_mhz = int(fields[6])
            throttle_bits = int(fields[7], 16)
            ecc_corrected = int(fields[8])
            ecc_uncorrected = int(fields[9])
        except ValueError as exc:
            raise SystemExit(f"decision-gate cycle GPU row {row_index} has invalid numeric telemetry") from exc
        if (
            index != row_index
            or not fields[1]
            or not fields[2]
            or not fields[7].lower().startswith("0x")
            or not -50 <= temperature_c <= 200
            or not math.isfinite(power_draw_w)
            or power_draw_w < 0.0
            or min(sm_clock_mhz, memory_clock_mhz, throttle_bits, ecc_corrected, ecc_uncorrected) < 0
        ):
            raise SystemExit(f"decision-gate cycle GPU row {row_index} is outside the supported domain")
        gpus.append(
            {
                "index": index,
                "uuid": fields[1],
                "performance_state": fields[2],
                "temperature_c": temperature_c,
                "power_draw_w": power_draw_w,
                "sm_clock_mhz": sm_clock_mhz,
                "memory_clock_mhz": memory_clock_mhz,
                "throttle_reasons_active": f"0x{throttle_bits:016x}",
                "ecc_corrected_volatile_total": ecc_corrected,
                "ecc_uncorrected_volatile_total": ecc_uncorrected,
            }
        )
    node = socket.gethostname().split(".", 1)[0]
    try:
        raw_node = _run_bounded_probe(("scontrol", "show", "node", "--oneliner", node))
        raw_kernel = _run_bounded_probe(("journalctl", "--dmesg", "--boot", "--no-pager", "--grep", "NVRM.*Xid"))
    except CellEntrypointError as exc:
        raise SystemExit(f"decision-gate cycle telemetry probe failed: {exc}") from exc
    state_fields = [part.split("=", 1)[1] for part in raw_node.split() if part.startswith("State=")]
    if len(state_fields) != 1 or not state_fields[0]:
        raise SystemExit("decision-gate cycle node telemetry lacks one parsed State")
    xid_lines = [line.strip() for line in raw_kernel.splitlines() if "NVRM" in line and "Xid" in line]
    return {
        "label": label,
        "captured_at": utc_timestamp(),
        "gpus": gpus,
        "node_state": {
            "method": "scontrol show node --oneliner HOSTNAME",
            "node": node,
            "state": state_fields[0],
        },
        "xid": {
            "method": "journalctl --dmesg --boot --no-pager --grep NVRM.*Xid",
            "event_count": len(xid_lines),
            "window_sha256": hashlib.sha256("\n".join(xid_lines).encode("utf-8")).hexdigest(),
        },
    }


def _initialize_representation_signatures(
    events: Sequence[GateEvent],
    *,
    rank: int,
    communication: Mapping[int, Any],
) -> None:
    """Fill every request with a request/rank/lane-specific exact signature."""

    supported_dtypes = {"float16", "bfloat16", "float32", "float64"}
    if len(events) > 8:
        raise SystemExit("decision-gate representation signature domain supports at most eight requests")
    for event_index, event in enumerate(events):
        if event.dtype not in supported_dtypes:
            raise SystemExit(
                f"decision-gate representation validation does not support signature dtype {event.dtype!r}"
            )
        rank_index = event.ranks.index(rank)
        tensor = communication[event.request]
        period = min(4, event.elements)
        for lane in range(period):
            # Multiples of four keep all inputs exact in bfloat16 over the
            # bounded eight-event/four-lane decision-gate domain.
            value = 4 + event_index * 8 + rank_index * 4 + lane * 64
            tensor[lane::period].fill_(value)


def _expected_representation_output(
    event: GateEvent,
    *,
    event_index: int,
    tensor: Any,
) -> Any:
    expected = tensor.new_empty(tensor.shape)
    period = min(4, event.elements)
    rank_sum = len(event.ranks) * (len(event.ranks) - 1) // 2
    for lane in range(period):
        value = len(event.ranks) * (4 + event_index * 8 + lane * 64) + rank_sum * 4
        expected[lane::period].fill_(value)
    return expected


def _tensor_commitment(tensor: Any, *, torch: Any) -> str:
    byte_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, int(byte_view.numel()), 64 * 1024):
        chunk = byte_view[offset : offset + 64 * 1024].cpu().tolist()
        digest.update(bytes(chunk))
    return digest.hexdigest()


def _validate_complete_representation(
    label: str,
    program: Sequence[RuntimeOp],
    events: Sequence[GateEvent],
    *,
    rank: int,
    communication: Mapping[int, Any],
    dist: Any,
    torch: Any,
) -> Tuple[int, str]:
    """Execute and validate every result through the actual representation loop."""

    _initialize_representation_signatures(events, rank=rank, communication=communication)
    event_by_request = {event.request: (index, event) for index, event in enumerate(events)}
    if len(event_by_request) != len(events):
        raise SystemExit("decision-gate representation validation requires unique requests")
    commitments: List[Tuple[int, str]] = []

    def validate_wait(request: int) -> None:
        item = event_by_request.get(request)
        if item is None:
            raise SystemExit(f"decision-gate {label} waited for an unknown request {request}")
        event_index, event = item
        tensor = communication[request]
        expected = _expected_representation_output(event, event_index=event_index, tensor=tensor)
        if not bool(tensor.eq(expected).all().item()):
            raise SystemExit(f"decision-gate {label} correctness failed for request {request}")
        commitments.append((request, _tensor_commitment(tensor, torch=torch)))

    _run_runtime_program(
        program,
        dist=dist,
        torch=torch,
        wait_callback=validate_wait,
    )
    if len(commitments) != len(events):
        raise SystemExit(f"decision-gate {label} did not validate every request")
    commitment = canonical_sha256(
        {
            "outputs": [{"request": request, "sha256": digest} for request, digest in commitments],
        }
    )
    return len(commitments), commitment


def _execute(
    *,
    plan: QualificationExecutionPlan,
    source: Tuple[GateEvent, ...],
    selected_indices: Tuple[int, ...],
    rank: int,
    world_size: int,
    local_rank: int,
    iterations: int,
    warmup: int,
    timeout_seconds: int,
    configuration_repetition: int,
) -> Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.distributed as dist  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("decision-gate physical execution requires target-compatible PyTorch") from exc

    if not torch.cuda.is_available():
        raise SystemExit("decision-gate requested CUDA but torch.cuda.is_available() is false")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
    try:
        if int(dist.get_rank()) != rank or int(dist.get_world_size()) != world_size:
            raise SystemExit("decision-gate initialized rank domain disagrees with the launch")
        groups = {
            pg_id: dist.new_group(ranks=list(ranks), timeout=timedelta(seconds=timeout_seconds))
            for pg_id, ranks in plan.groups
        }
        dtype_map = _torch_dtype_map(torch)
        communication = {
            event.request: torch.zeros(event.elements, device="cuda", dtype=dtype_map[event.dtype]) for event in source
        }
        gemms: Dict[Tuple[str, int, int, int], Tuple[Any, Any, Any]] = {}
        for event in source:
            for recipes in event.recipes:
                for recipe in recipes:
                    if recipe not in gemms:
                        dtype, m, n, k = recipe
                        gemms[recipe] = (
                            torch.randn((m, k), device="cuda", dtype=dtype_map[dtype]),
                            torch.randn((k, n), device="cuda", dtype=dtype_map[dtype]),
                            torch.empty((m, n), device="cuda", dtype=dtype_map[dtype]),
                        )

        selected = tuple(source[index] for index in selected_indices)
        materialized = plan_events(plan)
        shared_compiled = matching_source_and_exact_programs(source, materialized, rank=rank)
        compiled_programs = {
            "source": shared_compiled,
            "exact_work": shared_compiled,
            "stratified": compile_event_program(selected, rank=rank),
            "isolated": compile_event_program(source, rank=rank, mode="isolated"),
            "no_overlap": compile_event_program(source, rank=rank, mode="no_overlap"),
            "no_rank_skew": compile_event_program(source, rank=rank, mode="no_rank_skew"),
        }
        shared_runtime = _bind_runtime_program(
            shared_compiled,
            groups=groups,
            communication=communication,
            gemms=gemms,
        )
        runtime_programs = {
            representation: (
                shared_runtime
                if representation in {"source", "exact_work"}
                else _bind_runtime_program(
                    program,
                    groups=groups,
                    communication=communication,
                    gemms=gemms,
                )
            )
            for representation, program in compiled_programs.items()
        }

        def run_representation(representation: str) -> None:
            program = runtime_programs.get(representation)
            if program is None:  # pragma: no cover - closed constant vocabulary
                raise SystemExit(f"unsupported decision-gate representation {representation!r}")
            _run_runtime_program(program, dist=dist, torch=torch)

        telemetry_snapshots: List[Mapping[str, Any]] = []

        def capture_telemetry(label: str) -> None:
            dist.barrier()
            payload: List[Any] = [None]
            if rank == 0:
                payload[0] = _cycle_telemetry_snapshot(label)
            dist.broadcast_object_list(payload, src=0)
            if not isinstance(payload[0], Mapping) or payload[0].get("label") != label:
                raise SystemExit("decision-gate cycle telemetry broadcast is inconsistent")
            telemetry_snapshots.append(dict(payload[0]))
            dist.barrier()

        backend_checks = 0
        seen_strata = set()
        for event in source:
            if event.stratum in seen_strata:
                continue
            seen_strata.add(event.stratum)
            tensor = communication[event.request]
            local_rank_index = event.ranks.index(rank)
            tensor.fill_(local_rank_index + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=groups[event.pg_id])
            expected = len(event.ranks) * (len(event.ranks) + 1) // 2
            if not bool(tensor.eq(expected).all().item()):
                raise SystemExit(f"decision-gate backend smoke check failed for request {event.request}")
            backend_checks += 1

        representation_checks = 0
        commitments: Dict[str, str] = {}
        for label in ("source", "exact_work"):
            torch.cuda.synchronize()
            dist.barrier()
            check_count, commitment = _validate_complete_representation(
                label,
                runtime_programs[label],
                source,
                rank=rank,
                communication=communication,
                dist=dist,
                torch=torch,
            )
            torch.cuda.synchronize()
            dist.barrier()
            representation_checks += check_count
            commitments[label] = commitment
        if commitments["source"] != commitments["exact_work"]:
            raise SystemExit("decision-gate source and exact-work output commitments disagree")

        timings: Dict[str, List[float]] = {name: [] for name in REPRESENTATION_IDS}
        capture_telemetry("before_warmup")
        for warmup_index in range(warmup):
            order = warmup_representation_order(
                warmup_index,
                configuration_repetition=configuration_repetition,
            )
            for representation in order:
                for tensor in communication.values():
                    tensor.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                run_representation(representation)
                torch.cuda.synchronize()
                dist.barrier()
        capture_telemetry("before_measured_cycle_1")
        for iteration in range(iterations):
            order = representation_order(
                iteration,
                configuration_repetition=configuration_repetition,
            )
            for representation in order:
                for tensor in communication.values():
                    tensor.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                started = torch.cuda.Event(enable_timing=True)
                ended = torch.cuda.Event(enable_timing=True)
                started.record()
                run_representation(representation)
                ended.record()
                ended.synchronize()
                timings[representation].append(float(started.elapsed_time(ended) * 1000.0))
                dist.barrier()
            if (iteration + 1) % 6 == 0:
                capture_telemetry(f"after_measured_cycle_{(iteration + 1) // 6}")
        capture_telemetry("final")

        gathered: List[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "timings_us": timings})
        gathered_checks: List[Any] = [None] * world_size
        dist.all_gather_object(
            gathered_checks,
            {
                "rank": rank,
                "backend_smoke_check_count": backend_checks,
                "representation_check_count": representation_checks,
                "source_output_commitment_sha256": commitments["source"],
                "exact_work_output_commitment_sha256": commitments["exact_work"],
            },
        )
        normalized_backend_checks = []
        normalized_representation_checks = []
        source_commitments = []
        exact_commitments = []
        for expected_rank, raw in enumerate(gathered_checks):
            if not isinstance(raw, Mapping) or raw.get("rank") != expected_rank:
                raise SystemExit("decision-gate correctness inventory is inconsistent")
            backend_count = raw.get("backend_smoke_check_count")
            representation_count = raw.get("representation_check_count")
            source_commitment = raw.get("source_output_commitment_sha256")
            exact_commitment = raw.get("exact_work_output_commitment_sha256")
            if (
                backend_count != backend_checks
                or representation_count != representation_checks
                or not isinstance(source_commitment, str)
                or not isinstance(exact_commitment, str)
                or source_commitment != exact_commitment
            ):
                raise SystemExit("decision-gate correctness inventory is inconsistent")
            normalized_backend_checks.append(backend_count)
            normalized_representation_checks.append(representation_count)
            source_commitments.append(source_commitment)
            exact_commitments.append(exact_commitment)
        correctness = {
            "backend_smoke_checks_per_rank": normalized_backend_checks,
            "representation_checks_per_rank": normalized_representation_checks,
            "source_output_commitment_sha256_by_rank": source_commitments,
            "exact_work_output_commitment_sha256_by_rank": exact_commitments,
        }
        runtime = {
            "torch_version": _normalized_torch_version(torch),
            "torch_cuda_version": str(torch.version.cuda),
            "runtime_nccl_version_code": _runtime_nccl_version_code(_selected_nccl_library()),
            "distributed_backend": str(dist.get_backend()),
        }
        telemetry = {
            "schema": _CYCLE_TELEMETRY_SCHEMA,
            "method": _CYCLE_TELEMETRY_METHOD,
            "snapshots": [dict(snapshot) for snapshot in telemetry_snapshots],
        }
        return gathered, correctness, runtime, telemetry
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args: argparse.Namespace) -> int:
    iterations = _strict_positive(args.iterations, "iterations", maximum=1000)
    if (
        isinstance(args.warmup, bool)
        or not isinstance(args.warmup, int)
        or not 1 <= args.warmup <= 100
        or args.warmup % 6
    ):
        raise SystemExit("replicated warmup must be a positive multiple of six in [1, 100]")
    if iterations % 6:
        raise SystemExit("replicated iterations must contain complete six-row cycles")
    timeout_seconds = _strict_positive(
        args.distributed_timeout_seconds,
        "distributed-timeout-seconds",
        maximum=3600,
    )
    configuration_repetition = args.configuration_repetition
    if configuration_repetition is None:
        raise SystemExit("new decision-gate v1 physical execution is prohibited; v1 is analysis-only")
    if isinstance(configuration_repetition, bool) or not 0 <= configuration_repetition <= 999:
        raise SystemExit("configuration-repetition must be an integer in [0, 999]")
    rank, world_size, local_rank = distributed_execution_environment(os.environ)
    sources = {
        "request_manifest": args.request_manifest,
        "source_trace": args.source_trace,
        "canary": args.canary,
        "fidelity": args.fidelity,
        "qualification_policy": args.qualification_policy,
        "materialization_manifest": args.materialization_manifest,
        "replay_program": args.replay_program,
    }
    request_directory, materialization_directory = stage_qualification_inputs(
        sources,
        rank=rank,
        workspace=Path.cwd(),
    )
    request = verify_qualification_request(str(request_directory))
    policy = load_json(str(request_directory / "qualification-policy.json"))
    validate_qualification_policy(policy)
    plan = preflight_qualification_execution(
        str(request_directory),
        str(materialization_directory),
        world_size=world_size,
        iterations=iterations,
        warmup=args.warmup,
        distributed_timeout_seconds=timeout_seconds,
    )
    if (
        request["request_id"] != args.expected_request_id
        or plan.materialization_id != args.expected_materialization_id
        or plan.program_sha256 != args.expected_program_sha256
        or policy["policy_id"] != args.expected_policy_id
    ):
        raise SystemExit("decision-gate immutable artifact identities disagree with the frozen workload")
    trace = load_json(str(request_directory / "source.trace.json"))
    source = source_events(trace, world_size=world_size)
    materialized = plan_events(plan)
    if source != materialized:
        raise SystemExit("decision-gate source and materialized event programs disagree")
    selected_indices = stratified_indices(source)
    gathered, correctness, runtime, telemetry = _execute(
        plan=plan,
        source=source,
        selected_indices=selected_indices,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        iterations=iterations,
        warmup=args.warmup,
        timeout_seconds=timeout_seconds,
        configuration_repetition=configuration_repetition,
    )
    if rank == 0:
        payload = result_payload(
            request=request,
            materialization_id=plan.materialization_id,
            program_sha256=plan.program_sha256,
            policy=policy,
            world_size=world_size,
            iterations=iterations,
            warmup=args.warmup,
            source_event_count=len(source),
            selected_indices=selected_indices,
            gathered=gathered,
            correctness_checks_per_rank=correctness["representation_checks_per_rank"],
            runtime=runtime,
            configuration_repetition=configuration_repetition,
            backend_smoke_checks_per_rank=correctness["backend_smoke_checks_per_rank"],
            source_output_commitment_sha256_by_rank=correctness["source_output_commitment_sha256_by_rank"],
            exact_work_output_commitment_sha256_by_rank=correctness["exact_work_output_commitment_sha256_by_rank"],
            telemetry_checkpoints=telemetry,
        )
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
