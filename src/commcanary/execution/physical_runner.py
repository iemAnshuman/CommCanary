"""Digest-bound physical execution of the qualified Chakra graph subset.

The importable planning boundary is dependency-light. Torch is imported only
by :func:`main`, after every source, projection, executable, and output guard
that can be checked without a GPU has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..artifacts.chakra import decode_chakra_execution_trace, encode_chakra_subgraph
from ..artifacts.dtypes import dtype_size_bytes, require_canonical_dtype
from ..artifacts.io import SENSITIVE_JSON_POLICY, atomic_write_json
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.physical_canary import validate_chakra_projection
from ..artifacts.physical_execution import (
    SETUP_COST_PHASES,
    telemetry_assessment,
    validate_physical_execution_measurement,
)
from ..errors import SchemaError
from ..formats import PHYSICAL_EXECUTION_MEASUREMENT_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits, decode_bounded_json_bytes
from .telemetry import capture_cycle_telemetry

_RUNNER_PROTOCOL = "chakra-et-collective-graph.v1"
_COMPUTE_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_TORCH_DTYPES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}


@dataclass(frozen=True)
class GemmRecipe:
    rank: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class CompiledAllReduce:
    node_id: int
    dtype: str
    numel: int
    tensor_bytes: int


@dataclass(frozen=True)
class CompiledGemm:
    node_id: int
    dtype: str
    recipes: Tuple[GemmRecipe, ...]
    executed_flops: int


CompiledOperation = Union[CompiledAllReduce, CompiledGemm]


@dataclass(frozen=True)
class CompiledRegion:
    region_id: str
    operations: Tuple[CompiledOperation, ...]


@dataclass(frozen=True)
class PhysicalProgram:
    source_et_sha256: str
    executable_sha256: str
    projection_id: str
    selected_region_ids: Tuple[str, ...]
    selected_node_ids: Tuple[int, ...]
    regions: Tuple[CompiledRegion, ...]
    world_size: int
    executed_collectives: int
    executed_flops: int
    workspace_bytes_max_rank: int
    communication_only: bool


def prepare_physical_program(
    source_et: bytes,
    executable_et: bytes,
    projection: Mapping[str, Any],
    selected_region_ids: Sequence[str],
    *,
    world_size: int,
    max_workspace_bytes: int,
    communication_only: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> PhysicalProgram:
    """Validate and compile one exact executable without importing Torch."""

    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 2:
        raise SchemaError("physical runner world_size must be an integer of at least two")
    if not isinstance(max_workspace_bytes, int) or isinstance(max_workspace_bytes, bool) or max_workspace_bytes < 1:
        raise SchemaError("physical runner max_workspace_bytes must be positive")
    source = decode_chakra_execution_trace(source_et, limits=limits)
    executable = decode_chakra_execution_trace(executable_et, limits=limits)
    validate_chakra_projection(projection, source, limits=limits)
    if executable.metadata_raw != source.metadata_raw:
        raise SchemaError("physical executable Chakra metadata does not match its source")

    requested_regions = _sorted_unique_region_ids(selected_region_ids)
    projection_regions = projection.get("regions")
    if not isinstance(projection_regions, list):
        raise SchemaError("Chakra projection regions must be an array")
    region_by_id = {str(region["region_id"]): region for region in projection_regions}
    unknown = sorted(set(requested_regions).difference(region_by_id))
    if unknown:
        raise SchemaError(f"physical runner selected unknown regions {unknown[:10]}")

    source_order = {node_id: index for index, node_id in enumerate(source.node_ids)}
    ordered_region_ids = tuple(
        str(region["region_id"]) for region in projection_regions if str(region["region_id"]) in set(requested_regions)
    )
    selected_nodes: Set[int] = set()
    for region_id in ordered_region_ids:
        raw_node_ids = tuple(int(value) for value in region_by_id[region_id]["node_ids"])
        node_ids = {raw_node_ids[0]} if communication_only else set(raw_node_ids)
        overlap = selected_nodes.intersection(node_ids)
        if overlap:
            raise SchemaError(
                "physical runner requires non-overlapping execution regions; "
                f"region {region_id!r} repeats Node IDs {sorted(overlap)[:10]}"
            )
        selected_nodes.update(node_ids)
    expected_et, closure = encode_chakra_subgraph(source, selected_nodes)
    if expected_et != executable.source_raw:
        raise SchemaError("physical executable bytes do not equal the selected source Chakra subgraph")
    if executable.node_ids != closure:
        raise SchemaError("physical executable node order does not equal its dependency closure")

    projected_nodes = projection.get("nodes")
    if not isinstance(projected_nodes, list):
        raise SchemaError("Chakra projection nodes must be an array")
    node_by_id = {int(node["node_id"]): node for node in projected_nodes}
    if communication_only and any(node_by_id[node_id]["operation"] != "all_reduce" for node_id in closure):
        raise SchemaError("communication-only executable has a non-collective dependency")
    compiled_regions: List[CompiledRegion] = []
    allreduce_count = 0
    total_flops = 0
    expected_ranks = tuple(range(world_size))
    for region_id in ordered_region_ids:
        raw_node_ids = tuple(int(value) for value in region_by_id[region_id]["node_ids"] if int(value) in set(closure))
        if raw_node_ids != tuple(sorted(raw_node_ids, key=source_order.__getitem__)):
            raise SchemaError(f"physical runner region {region_id!r} does not use source node order")
        operations: List[CompiledOperation] = []
        for node_id in raw_node_ids:
            row = node_by_id[node_id]
            execution = row["execution"]
            operation = row["operation"]
            dtype = require_canonical_dtype(execution["dtype"], label=f"physical runner Node.id={node_id} dtype")
            if dtype not in _TORCH_DTYPES:
                raise SchemaError(f"physical runner Node.id={node_id} dtype {dtype!r} is not executable")
            if operation == "all_reduce":
                ranks = tuple(int(value) for value in execution["ranks"])
                if ranks != expected_ranks:
                    raise SchemaError(
                        f"physical runner all-reduce Node.id={node_id} ranks must equal {list(expected_ranks)}"
                    )
                tensor_bytes = int(row["tensor_bytes"])
                operations.append(
                    CompiledAllReduce(
                        node_id=node_id,
                        dtype=dtype,
                        numel=tensor_bytes // dtype_size_bytes(dtype),
                        tensor_bytes=tensor_bytes,
                    )
                )
                allreduce_count += 1
            elif operation == "gemm":
                if dtype not in _COMPUTE_DTYPES:
                    raise SchemaError(f"physical runner GEMM Node.id={node_id} dtype {dtype!r} is unsupported")
                recipes = tuple(
                    GemmRecipe(
                        rank=int(recipe["rank"]),
                        m=int(recipe["m"]),
                        n=int(recipe["n"]),
                        k=int(recipe["k"]),
                    )
                    for recipe in execution["rank_recipes"]
                )
                if tuple(recipe.rank for recipe in recipes) != expected_ranks:
                    raise SchemaError(
                        f"physical runner GEMM Node.id={node_id} recipes must cover ranks {list(expected_ranks)}"
                    )
                executed_flops = int(row["executed_flops"])
                operations.append(
                    CompiledGemm(
                        node_id=node_id,
                        dtype=dtype,
                        recipes=recipes,
                        executed_flops=executed_flops,
                    )
                )
                total_flops += executed_flops
            else:  # Projection validation closes this branch.
                raise SchemaError(f"physical runner Node.id={node_id} operation is unsupported")
        if not operations or not isinstance(operations[0], CompiledAllReduce):
            raise SchemaError(f"physical runner region {region_id!r} must begin with one all-reduce")
        if sum(isinstance(operation, CompiledAllReduce) for operation in operations) != 1:
            raise SchemaError(f"physical runner region {region_id!r} must contain exactly one all-reduce")
        if not communication_only and not any(isinstance(operation, CompiledGemm) for operation in operations):
            raise SchemaError(f"physical runner region {region_id!r} must contain overlapping GEMM work")
        compiled_regions.append(CompiledRegion(region_id=region_id, operations=tuple(operations)))

    workspace = _workspace_bytes(compiled_regions, world_size=world_size)
    if workspace > max_workspace_bytes:
        raise SchemaError(
            f"physical runner workspace bytes={workspace} exceeds max_workspace_bytes={max_workspace_bytes}"
        )
    return PhysicalProgram(
        source_et_sha256=source.source_sha256,
        executable_sha256=executable.source_sha256,
        projection_id=str(projection["projection_id"]),
        selected_region_ids=ordered_region_ids,
        selected_node_ids=closure,
        regions=tuple(compiled_regions),
        world_size=world_size,
        executed_collectives=allreduce_count,
        executed_flops=total_flops,
        workspace_bytes_max_rank=workspace,
        communication_only=communication_only,
    )


#: Distinct collective buffers held per dtype.  Collectives are issued with
#: ``async_op=True`` and drained at the end of a region, so two in-flight
#: all-reduces sharing one buffer would reduce into overlapping memory: the
#: results are undefined and NCCL may serialise them, destroying the very
#: compute/communication overlap the canary exists to measure.  Concurrency is
#: therefore bounded by the number of slots rather than by chance.
COLLECTIVE_BUFFER_SLOTS = 4

#: How injected per-rank skew is realised.  It delays the *issue* of a rank's
#: collective on the host; it is not a GPU-side arrival guarantee, and callers
#: must not read it as one.  ``time.sleep`` was previously used here, which
#: cannot resolve the single-digit microsecond skews these traces carry -- its
#: granularity is tens to hundreds of microseconds, so requested skew was
#: silently inflated by an order of magnitude.  A monotonic spin is accurate at
#: this scale at the cost of burning one core for the duration.
RANK_SKEW_MECHANISM = "host_issue_monotonic_spin"

#: Skew below this is not reliably distinguishable from scheduling noise.
#: Measured on A100/nasrin0 (job 186046) the spin carries a fixed ~0.35 us
#: overhead from its own clock reads: 0.5 us requested measured 0.82 us (+64%),
#: 5 us measured 5.35 us (+7.0%), 20 us measured 20.35 us (+1.7%), 100 us
#: measured 100.36 us (+0.4%). The floor is therefore set where the relative
#: error first falls under 10 percent rather than at the smallest value the
#: clock can name.
MINIMUM_RESOLVABLE_SKEW_US = 4.0


def _spin_us(duration_us: float) -> None:
    """Busy-wait for ``duration_us`` microseconds on the monotonic clock."""

    if duration_us <= 0.0:
        return
    deadline = time.perf_counter_ns() + int(duration_us * 1000.0)
    while time.perf_counter_ns() < deadline:
        pass


def _sorted_unique_region_ids(values: Sequence[str]) -> Tuple[str, ...]:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise SchemaError("physical runner selected regions must be non-empty strings")
    result = tuple(sorted(values))
    if len(result) != len(set(result)):
        raise SchemaError("physical runner selected regions must be unique")
    return result


def _workspace_bytes(regions: Sequence[CompiledRegion], *, world_size: int) -> int:
    maxima: List[Dict[Tuple[str, str], int]] = [dict() for _ in range(world_size)]
    for region in regions:
        for operation in region.operations:
            if isinstance(operation, CompiledAllReduce):
                key = (operation.dtype, "collective")
                for rank in range(world_size):
                    maxima[rank][key] = max(maxima[rank].get(key, 0), operation.numel * COLLECTIVE_BUFFER_SLOTS)
                continue
            for recipe in operation.recipes:
                for kind, elements in (
                    ("gemm_a", recipe.m * recipe.k),
                    ("gemm_b", recipe.k * recipe.n),
                    ("gemm_c", recipe.m * recipe.n),
                ):
                    key = (operation.dtype, kind)
                    maxima[recipe.rank][key] = max(maxima[recipe.rank].get(key, 0), elements)
    return max(
        sum(elements * dtype_size_bytes(dtype) for (dtype, _kind), elements in rank_maxima.items())
        for rank_maxima in maxima
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-et", required=True)
    parser.add_argument("--executable-et", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--selected-region", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner-oci-digest", required=True)
    parser.add_argument("--subject-sha256", required=True)
    parser.add_argument("--perturbation-id", required=True)
    parser.add_argument("--role", choices=("trace_derived_reference", "reduced_decision_canary"), required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-workspace-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=180)
    parser.add_argument("--disable-overlap", action="store_true")
    parser.add_argument("--rank-skew-us", type=float, default=0.0)
    parser.add_argument("--communication-only", action="store_true")
    return parser


class _CostClock:
    """Wall-clock accounting that separates workload cost from instrumentation.

    A reduction ratio is only meaningful against the cost a user actually pays.
    This runner shells out to ``nvidia-smi`` once per measured iteration for
    telemetry; a customer's canary does not. Instrumentation time is therefore
    accumulated on its own and excluded from both setup and measured totals,
    and the excluded amount is published so the exclusion is auditable rather
    than merely asserted.

    Every duration here is rank 0's view. Ranks are barriered around each phase,
    so rank 0's wall clock is the run's wall clock.
    """

    def __init__(self) -> None:
        self._phases: Dict[str, int] = {}
        self._instrumentation_ns = 0

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0) + (time.perf_counter_ns() - started)

    @contextmanager
    def instrumentation(self) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self._instrumentation_ns += time.perf_counter_ns() - started

    def seconds(self, name: str) -> float:
        return self._phases.get(name, 0) / 1_000_000_000.0

    @property
    def instrumentation_seconds(self) -> float:
        return self._instrumentation_ns / 1_000_000_000.0


def cost_accounting(clock: "_CostClock", *, iterations: int) -> Dict[str, Any]:
    """Split one run's wall clock into setup and measured time.

    Reduction claims are made both ways and this is what makes that possible.
    ``total_seconds`` is what a user pays per run; ``measured_seconds`` divided
    by ``iterations`` is the steady-state cost that amortizes once the canary
    runs long enough to bury its own setup. Publishing only the second is how
    an eight-second job with six seconds of ``init_process_group`` gets
    reported as a fast canary.
    """

    setup = {name: clock.seconds(name) for name in SETUP_COST_PHASES}
    setup_total = math.fsum(setup.values())
    measured = clock.seconds("measured")
    return {
        "setup_seconds": {**setup, "total": setup_total},
        "measured_seconds": measured,
        "instrumentation_seconds": clock.instrumentation_seconds,
        "total_seconds": setup_total + measured,
        "steady_state_seconds_per_iteration": measured / iterations,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmups < 0 or args.iterations < 2:
        raise SchemaError("physical runner requires non-negative warmups and at least two measured iterations")
    if 0.0 < args.rank_skew_us < MINIMUM_RESOLVABLE_SKEW_US:
        raise SchemaError(
            f"physical runner rank-skew-us below {MINIMUM_RESOLVABLE_SKEW_US} cannot be resolved by "
            f"{RANK_SKEW_MECHANISM}; request zero or a resolvable value rather than a skew it cannot deliver"
        )
    if args.rank_skew_us < 0.0 or not math.isfinite(args.rank_skew_us):
        raise SchemaError("physical runner rank-skew-us must be finite and non-negative")
    _sha256(args.subject_sha256, "physical runner subject-sha256")
    _oci_digest(args.runner_oci_digest)
    declared_digest = os.environ.get("COMMCANARY_RUNNER_OCI_DIGEST")
    if declared_digest is not None and declared_digest != args.runner_oci_digest:
        raise SchemaError("physical runner OCI digest disagrees with its container environment")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical execution evidence: {output}")

    clock = _CostClock()
    with clock.phase("program_preparation"):
        source_raw = _bounded_bytes(Path(args.source_et))
        executable_raw = _bounded_bytes(Path(args.executable_et))
        projection_raw = _bounded_bytes(Path(args.projection))
        projection_value = decode_bounded_json_bytes(projection_raw)
        if not isinstance(projection_value, Mapping):
            raise SchemaError("physical runner projection must be a JSON object")

        world_size_environment = os.environ.get("WORLD_SIZE")
        if world_size_environment is None or not world_size_environment.isdigit():
            raise SchemaError("physical runner requires torchrun WORLD_SIZE")
        program = prepare_physical_program(
            source_raw,
            executable_raw,
            projection_value,
            args.selected_region,
            world_size=int(world_size_environment),
            max_workspace_bytes=args.max_workspace_bytes,
            communication_only=bool(args.communication_only),
        )

    with clock.phase("runtime_initialization"):
        import torch  # type: ignore[import-not-found]
        import torch.distributed as dist  # type: ignore[import-not-found]

        local_rank = _environment_int("LOCAL_RANK")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=args.distributed_timeout_seconds),
        )
        rank = dist.get_rank()
        if dist.get_world_size() != program.world_size:
            raise RuntimeError("initialized process-group size does not match the compiled physical program")

    telemetry: List[Dict[str, Any]] = []
    with clock.phase("allocation_and_correctness"):
        runtime = _TorchProgramRuntime(torch, dist, program, rank=rank, local_rank=local_rank)
        if rank == 0:
            with clock.instrumentation():
                telemetry.append(capture_cycle_telemetry("before_correctness"))
        dist.barrier()
        correctness = runtime.validate_complete_program(disable_overlap=args.disable_overlap)
        gathered_correctness: List[Any] = [None for _ in range(program.world_size)]
        dist.all_gather_object(gathered_correctness, correctness)
        if not all(row.get("passed") is True for row in gathered_correctness):
            raise RuntimeError("physical runner complete-program correctness validation failed")

    with clock.phase("warmups"):
        for _ in range(args.warmups):
            dist.barrier()
            runtime.execute_pass(
                disable_overlap=args.disable_overlap,
                rank_skew_us=args.rank_skew_us,
                timed=False,
            )
        dist.barrier()
    if rank == 0:
        with clock.instrumentation():
            telemetry.append(capture_cycle_telemetry("before_measured_cycle_1"))
    dist.barrier()

    samples: List[Dict[str, Any]] = []
    for iteration in range(args.iterations):
        with clock.phase("measured"):
            dist.barrier()
            local_sample = runtime.execute_pass(
                disable_overlap=args.disable_overlap,
                rank_skew_us=args.rank_skew_us,
                timed=True,
            )
            gathered: List[Any] = [None for _ in range(program.world_size)]
            dist.all_gather_object(gathered, local_sample)
            if rank == 0:
                samples.append(
                    {
                        "iteration": iteration,
                        "cuda_seconds_by_rank": [float(row["cuda_seconds"]) for row in gathered],
                        "host_seconds_by_rank": [float(row["host_seconds"]) for row in gathered],
                        "physical_runtime_seconds": max(float(row["cuda_seconds"]) for row in gathered),
                        "peak_memory_bytes_by_rank": [int(row["peak_memory_bytes"]) for row in gathered],
                    }
                )
        if rank == 0:
            with clock.instrumentation():
                telemetry.append(capture_cycle_telemetry(f"after_measured_cycle_{iteration + 1}"))
        dist.barrier()

    if rank == 0:
        with clock.instrumentation():
            telemetry.append(capture_cycle_telemetry("final"))
        measurement = _measurement(
            args=args,
            program=program,
            samples=samples,
            correctness=gathered_correctness,
            telemetry=telemetry,
            torch=torch,
            cost=cost_accounting(clock, iterations=args.iterations),
        )
        validate_physical_execution_measurement(measurement)
        policy = replace(
            SENSITIVE_JSON_POLICY,
            artifact_label="physical execution evidence",
            overwrite=False,
        )
        atomic_write_json(output, measurement, indent=2, policy=policy)
        print(json.dumps({"measurement_id": measurement["measurement_id"], "output": str(output)}, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


class _TorchProgramRuntime:
    def __init__(self, torch: Any, dist: Any, program: PhysicalProgram, *, rank: int, local_rank: int) -> None:
        self.torch = torch
        self.dist = dist
        self.program = program
        self.rank = rank
        self.device = torch.device("cuda", local_rank)
        self.collective_buffers: Dict[str, List[Any]] = {}
        self.gemm_buffers: Dict[Tuple[str, str], Any] = {}
        self._allocate()

    def _allocate(self) -> None:
        maxima: Dict[Tuple[str, str], int] = {}
        for region in self.program.regions:
            for operation in region.operations:
                if isinstance(operation, CompiledAllReduce):
                    key = (operation.dtype, "collective")
                    maxima[key] = max(maxima.get(key, 0), operation.numel)
                else:
                    recipe = operation.recipes[self.rank]
                    for kind, elements in (
                        ("gemm_a", recipe.m * recipe.k),
                        ("gemm_b", recipe.k * recipe.n),
                        ("gemm_c", recipe.m * recipe.n),
                    ):
                        key = (operation.dtype, kind)
                        maxima[key] = max(maxima.get(key, 0), elements)
        for (dtype, kind), elements in maxima.items():
            if kind == "collective":
                pool = []
                for _ in range(COLLECTIVE_BUFFER_SLOTS):
                    slot_tensor = self.torch.empty(elements, dtype=self._dtype(dtype), device=self.device)
                    slot_tensor.zero_()
                    pool.append(slot_tensor)
                self.collective_buffers[dtype] = pool
                continue
            tensor = self.torch.empty(elements, dtype=self._dtype(dtype), device=self.device)
            if kind != "gemm_c":
                tensor.fill_(0.125)
            self.gemm_buffers[(dtype, kind)] = tensor

    def _dtype(self, name: str) -> Any:
        return getattr(self.torch, _TORCH_DTYPES[name])

    def validate_complete_program(self, *, disable_overlap: bool) -> Dict[str, Any]:
        commitments: List[Dict[str, Any]] = []
        with self.torch.inference_mode():
            for region_index, region in enumerate(self.program.regions):
                self._execute_region(
                    region,
                    disable_overlap=disable_overlap,
                    rank_skew_us=0.0,
                    validation_commitments=commitments,
                    region_index=region_index,
                )
        self.torch.cuda.synchronize(self.device)
        passed = all(row["passed"] for row in commitments) and len(commitments) == len(self.program.selected_node_ids)
        return {
            "rank": self.rank,
            "passed": passed,
            "validated_node_ids": [row["node_id"] for row in commitments],
            "commitment_sha256": hashlib.sha256(canonical_json_bytes(commitments)).hexdigest(),
        }

    def execute_pass(self, *, disable_overlap: bool, rank_skew_us: float, timed: bool) -> Dict[str, Any]:
        self.torch.cuda.reset_peak_memory_stats(self.device)
        started = self.torch.cuda.Event(enable_timing=True)
        ended = self.torch.cuda.Event(enable_timing=True)
        host_started = time.perf_counter_ns()
        started.record()
        with self.torch.inference_mode():
            for region_index, region in enumerate(self.program.regions):
                self._execute_region(
                    region,
                    disable_overlap=disable_overlap,
                    rank_skew_us=rank_skew_us,
                    validation_commitments=None,
                    region_index=region_index,
                )
        ended.record()
        ended.synchronize()
        host_seconds = (time.perf_counter_ns() - host_started) / 1_000_000_000.0
        cuda_seconds = started.elapsed_time(ended) / 1000.0
        return {
            "cuda_seconds": cuda_seconds,
            "host_seconds": host_seconds,
            "peak_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "timed": timed,
        }

    def _execute_region(
        self,
        region: CompiledRegion,
        *,
        disable_overlap: bool,
        rank_skew_us: float,
        validation_commitments: Optional[List[Dict[str, Any]]],
        region_index: int,
    ) -> None:
        outstanding: List[Optional[Tuple[CompiledAllReduce, Any, Any, Optional[float]]]] = []
        probes: List[Optional[Dict[str, Any]]] = []
        slot_owner: Dict[Tuple[str, int], int] = {}
        slot_cursor: Dict[str, int] = {}
        for operation_index, operation in enumerate(region.operations):
            if isinstance(operation, CompiledAllReduce):
                # Claim a distinct buffer slot. If its previous collective is
                # still in flight the buffer cannot be reused yet, so that work
                # is settled first; concurrency is bounded, never aliased.
                slot = slot_cursor.get(operation.dtype, 0) % COLLECTIVE_BUFFER_SLOTS
                slot_cursor[operation.dtype] = slot + 1
                slot_key = (operation.dtype, slot)
                previous = slot_owner.get(slot_key)
                if previous is not None:
                    self._settle(outstanding, previous, validation_commitments, probes)
                buffer = self.collective_buffers[operation.dtype][slot][: operation.numel]
                expected = None
                if validation_commitments is not None:
                    signature = 1.0 + self.rank + region_index / 32.0
                    buffer.fill_(signature)
                    cast_signature = float(buffer[0].item())
                    gathered_signatures: List[Any] = [None for _ in range(self.program.world_size)]
                    self.dist.all_gather_object(gathered_signatures, cast_signature)
                    expected = sum(float(value) for value in gathered_signatures)
                if rank_skew_us and self.rank:
                    _spin_us(rank_skew_us * self.rank)
                work = self.dist.all_reduce(buffer, op=self.dist.ReduceOp.SUM, async_op=True)
                slot_owner[slot_key] = len(outstanding)
                outstanding.append((operation, buffer, work, expected))
                probes.append(None)
                if disable_overlap:
                    self._settle(outstanding, len(outstanding) - 1, validation_commitments, probes)
            else:
                recipe = operation.recipes[self.rank]
                a = self.gemm_buffers[(operation.dtype, "gemm_a")][: recipe.m * recipe.k].view(recipe.m, recipe.k)
                b = self.gemm_buffers[(operation.dtype, "gemm_b")][: recipe.k * recipe.n].view(recipe.k, recipe.n)
                c = self.gemm_buffers[(operation.dtype, "gemm_c")][: recipe.m * recipe.n].view(recipe.m, recipe.n)
                expected_gemm = None
                if validation_commitments is not None:
                    signature = min(0.125, math.sqrt(1024.0 / recipe.k))
                    a.fill_(signature)
                    b.fill_(signature * (1.0 + operation_index / 64.0))
                    # Read one element back rather than reusing `signature`, so
                    # the expectation carries the dtype's rounding. `a` and `b`
                    # are two-dimensional views, so this must index both axes:
                    # `a[0]` is a row of `recipe.k` elements and `.item()` on it
                    # raises for every GEMM wider than one column.
                    expected_gemm = float(a[0, 0].item()) * float(b[0, 0].item()) * recipe.k
                self.torch.mm(a, b, out=c)
                if validation_commitments is not None:
                    if expected_gemm is None:
                        raise RuntimeError("physical runner lost a GEMM correctness expectation")
                    self.torch.cuda.synchronize(self.device)
                    actual = float(c[0, 0].item())
                    passed = math.isfinite(actual) and math.isclose(
                        actual, float(expected_gemm), rel_tol=0.08, abs_tol=1e-5
                    )
                    validation_commitments.append(
                        {"node_id": operation.node_id, "operation": "gemm", "actual": actual, "passed": passed}
                    )
        for index in range(len(outstanding)):
            self._settle(outstanding, index, validation_commitments, probes)
        # Collective probes are appended after the region's GEMM probes and in
        # program order, independent of the order slot pressure forced waits in.
        if validation_commitments is not None:
            validation_commitments.extend(probe for probe in probes if probe is not None)

    def _settle(
        self,
        outstanding: List[Optional[Tuple[CompiledAllReduce, Any, Any, Optional[float]]]],
        index: int,
        validation_commitments: Optional[List[Dict[str, Any]]],
        probes: List[Optional[Dict[str, Any]]],
    ) -> None:
        """Wait for one in-flight collective and record its correctness probe."""

        entry = outstanding[index]
        if entry is None:
            return
        outstanding[index] = None
        operation, buffer, work, expected = entry
        work.wait()
        if validation_commitments is not None:
            if expected is None:
                raise RuntimeError("physical runner lost an all-reduce correctness expectation")
            self.torch.cuda.synchronize(self.device)
            actual = float(buffer[0].item())
            passed = math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=0.001, abs_tol=0.001)
            probes[index] = {
                "node_id": operation.node_id,
                "operation": "all_reduce",
                "actual": actual,
                "passed": passed,
            }


def _measurement(
    *,
    args: Any,
    program: PhysicalProgram,
    samples: Sequence[Mapping[str, Any]],
    correctness: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    torch: Any,
    cost: Mapping[str, Any],
) -> Dict[str, Any]:
    runtimes = [float(row["physical_runtime_seconds"]) for row in samples]
    peak_memory = max(max(int(value) for value in row["peak_memory_bytes_by_rank"]) for row in samples)
    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "nccl": list(torch.cuda.nccl.version()),
        "world_size": program.world_size,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "nccl_configuration": {
            name: os.environ.get(name)
            for name in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_NET", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
        },
    }
    raw: Dict[str, Any] = {
        "format": PHYSICAL_EXECUTION_MEASUREMENT_FORMAT,
        "role": args.role,
        "runner": {"oci_digest": args.runner_oci_digest, "execution_protocol": _RUNNER_PROTOCOL},
        "subject_sha256": args.subject_sha256,
        "perturbation_id": args.perturbation_id,
        "source_et_sha256": program.source_et_sha256,
        "executable_sha256": program.executable_sha256,
        "projection_id": program.projection_id,
        "selected_region_ids": list(program.selected_region_ids),
        "selected_node_ids": list(program.selected_node_ids),
        "execution": {
            "world_size": program.world_size,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "disable_overlap": bool(args.disable_overlap),
            "rank_skew_mechanism": RANK_SKEW_MECHANISM,
            "collective_buffer_slots": COLLECTIVE_BUFFER_SLOTS,
            "communication_only": program.communication_only,
            "rank_skew_us": args.rank_skew_us,
            "workspace_bytes_max_rank": program.workspace_bytes_max_rank,
        },
        "correctness": list(correctness),
        "cost": dict(cost),
        "samples": list(samples),
        "physical_metrics": {
            "physical_runtime_seconds": statistics.median(runtimes),
            "gpu_seconds": statistics.median(runtimes) * program.world_size,
            "gpu_count": program.world_size,
            "executed_collectives": program.executed_collectives,
            "executed_flops": program.executed_flops,
            "peak_memory_bytes": peak_memory,
        },
        "environment": environment,
        "environment_sha256": hashlib.sha256(canonical_json_bytes(environment)).hexdigest(),
        "telemetry": list(telemetry),
        "telemetry_assessment": telemetry_assessment(telemetry, world_size=program.world_size),
    }
    raw["measurement_id"] = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    return raw


def _bounded_bytes(path: Path, *, limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(limits.max_input_bytes + 1)
    if len(raw) > limits.max_input_bytes:
        raise SchemaError(f"physical runner input exceeds max_input_bytes={limits.max_input_bytes}: {path}")
    return raw


def _environment_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None or not value.isdigit():
        raise SchemaError(f"physical runner requires integer environment variable {name}")
    return int(value)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(f"{label} must be a lowercase SHA-256")
    return value


def _oci_digest(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SchemaError("physical runner OCI digest must use sha256:<64 lowercase hex>")
    _sha256(value.partition(":")[2], "physical runner OCI digest")
    return value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompiledAllReduce",
    "CompiledGemm",
    "CompiledRegion",
    "GemmRecipe",
    "PHYSICAL_EXECUTION_MEASUREMENT_FORMAT",
    "PhysicalProgram",
    "main",
    "prepare_physical_program",
    "validate_physical_execution_measurement",
]
