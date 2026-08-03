#!/usr/bin/env python3
"""Run the predeclared same-allocation physical decision-fidelity gate.

Every rank verifies the policy-bound qualification request and materialization
before importing PyTorch.  The runner then interleaves the source program, the
lossless exact-work materialization, two practical baselines, and two causal
ablations inside one process group.  All representations use the same CUDA
event timing method and allocation; rank 0 emits one strict JSON document for
the manifest-owned physical adapter.

This runner does not decide whether CommCanary passes the product gate.  It
retains raw per-rank samples and explicitly leaves decision fidelity unanalyzed
until every frozen configuration has one selected terminal attempt.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
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
)
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
    "exact_work": ("positive_conformance_control", "verified-materialization-issue-rank-work-wait"),
}
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 300
_MAX_PROC_MAPS_BYTES = 4 * 1024 * 1024


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
    parser.add_argument("--warmup", type=int, default=5)
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
    execution: Dict[str, Any] = {
        "world_size": world_size,
        "iterations": iterations,
        "warmup": warmup,
        "timing_semantics": DECISION_GATE_TIMING_SEMANTICS,
        "order_method": (
            DECISION_GATE_ORDER_METHOD if configuration_repetition is None else DECISION_GATE_REPLICATED_ORDER_METHOD
        ),
        "representation_order_by_iteration": [
            list(
                representation_order(
                    index,
                    configuration_repetition=configuration_repetition,
                )
            )
            for index in range(iterations)
        ],
        "source_event_count": source_event_count,
        "stratified_method": STRATIFIED_METHOD,
        "stratified_source_event_indices": list(selected_indices),
    }
    if configuration_repetition is not None:
        execution["configuration_repetition"] = configuration_repetition
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
        "correctness": {
            "status": "passed",
            "semantics": "one-source-value-sum-check-per-collective-shape",
            "checks_per_rank": list(correctness_checks_per_rank),
            "total_check_count": sum(correctness_checks_per_rank),
        },
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
) -> Tuple[Sequence[Mapping[str, Any]], Sequence[int], Mapping[str, Any]]:
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

        def compute(recipes: Sequence[Tuple[str, int, int, int]]) -> None:
            for recipe in recipes:
                left, right, output = gemms[recipe]
                torch.mm(left, right, out=output)

        def event_program(events: Sequence[GateEvent], mode: str) -> None:
            for event in events:
                tensor = communication[event.request]
                if mode in {"isolated", "no_overlap"}:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=groups[event.pg_id])
                    if mode == "no_overlap":
                        compute(event.recipes[rank])
                    continue
                work = dist.all_reduce(
                    tensor,
                    op=dist.ReduceOp.SUM,
                    group=groups[event.pg_id],
                    async_op=True,
                )
                if mode == "no_rank_skew":
                    compute(event.recipes[0])
                else:
                    compute(event.recipes[rank])
                work.wait()

        def exact_program() -> None:
            pending: Dict[int, Any] = {}
            for entry in plan.entries:
                if entry.get("compute") == "gemm_recipe":
                    compute(
                        _recipe_tuple(entry["recipe_by_rank"], ranks=tuple(entry["global_ranks"]), field="plan")[rank]
                    )
                    continue
                comms = entry.get("comms")
                if comms == "init":
                    continue
                if comms == "wait":
                    request = int(entry["req"])
                    pending.pop(request).wait()
                    continue
                request = int(entry["req"])
                pending[request] = dist.all_reduce(
                    communication[request],
                    op=dist.ReduceOp.SUM,
                    group=groups[int(entry["pg_id"])],
                    async_op=True,
                )
            if pending:
                raise SystemExit("decision-gate exact replay retained pending requests")

        selected = tuple(source[index] for index in selected_indices)

        def run_representation(representation: str) -> None:
            if representation == "source":
                event_program(source, "source")
            elif representation == "exact_work":
                exact_program()
            elif representation == "stratified":
                event_program(selected, "stratified")
            elif representation == "isolated":
                event_program(source, "isolated")
            elif representation == "no_overlap":
                event_program(source, "no_overlap")
            elif representation == "no_rank_skew":
                event_program(source, "no_rank_skew")
            else:  # pragma: no cover - closed constant vocabulary
                raise SystemExit(f"unsupported decision-gate representation {representation!r}")

        checks = 0
        expected = world_size * (world_size + 1) // 2
        seen_strata = set()
        for event in source:
            if event.stratum in seen_strata:
                continue
            seen_strata.add(event.stratum)
            tensor = communication[event.request]
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=groups[event.pg_id])
            if not bool(tensor.eq(expected).all().item()):
                raise SystemExit(f"decision-gate correctness check failed for request {event.request}")
            checks += 1

        timings: Dict[str, List[float]] = {name: [] for name in REPRESENTATION_IDS}
        for pass_index in range(-warmup, iterations):
            order_index = pass_index if pass_index >= 0 else pass_index + warmup
            order = representation_order(
                order_index,
                configuration_repetition=configuration_repetition,
            )
            for representation in order:
                for tensor in communication.values():
                    tensor.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                if pass_index < 0:
                    run_representation(representation)
                    torch.cuda.synchronize()
                else:
                    started = torch.cuda.Event(enable_timing=True)
                    ended = torch.cuda.Event(enable_timing=True)
                    started.record()
                    run_representation(representation)
                    ended.record()
                    ended.synchronize()
                    timings[representation].append(float(started.elapsed_time(ended) * 1000.0))
                dist.barrier()

        gathered: List[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "timings_us": timings})
        gathered_checks: List[Any] = [None] * world_size
        dist.all_gather_object(gathered_checks, {"rank": rank, "check_count": checks})
        normalized_checks = []
        for expected_rank, raw in enumerate(gathered_checks):
            if not isinstance(raw, Mapping) or raw.get("rank") != expected_rank or raw.get("check_count") != checks:
                raise SystemExit("decision-gate correctness inventory is inconsistent")
            normalized_checks.append(checks)
        runtime = {
            "torch_version": _normalized_torch_version(torch),
            "torch_cuda_version": str(torch.version.cuda),
            "runtime_nccl_version_code": _runtime_nccl_version_code(_selected_nccl_library()),
            "distributed_backend": str(dist.get_backend()),
        }
        return gathered, normalized_checks, runtime
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args: argparse.Namespace) -> int:
    iterations = _strict_positive(args.iterations, "iterations", maximum=1000)
    if isinstance(args.warmup, bool) or not isinstance(args.warmup, int) or not 0 <= args.warmup <= 100:
        raise SystemExit("warmup must be an integer in [0, 100]")
    timeout_seconds = _strict_positive(
        args.distributed_timeout_seconds,
        "distributed-timeout-seconds",
        maximum=3600,
    )
    configuration_repetition = args.configuration_repetition
    if configuration_repetition is not None and (
        isinstance(configuration_repetition, bool) or not 0 <= configuration_repetition <= 999
    ):
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
    gathered, correctness, runtime = _execute(
        plan=plan,
        source=source,
        selected_indices=selected_indices,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        iterations=iterations,
        warmup=args.warmup,
        timeout_seconds=timeout_seconds,
        configuration_repetition=configuration_repetition or 0,
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
            correctness_checks_per_rank=correctness,
            runtime=runtime,
            configuration_repetition=configuration_repetition,
        )
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
