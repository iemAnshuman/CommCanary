"""Fail-closed torch.distributed reference execution for materialized requests.

This module deliberately imports PyTorch only after the portable request,
materialization, complete replay program, distributed rank domain, operation
lifetimes, and allocation budgets have passed pure preflight.  Its output is a
bound diagnostic from a reference executor, not yet a qualification
observation or verdict.
"""

from __future__ import annotations

import copy
import hashlib
import math
import statistics
import struct
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from itertools import product
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from ..artifacts.dtypes import (
    dtype_size_bytes,
    require_param_compute_dtype,
    require_param_dtype,
)
from ..artifacts.qualification_program import qualification_compute_tensor_elements
from ..artifacts.wire import (
    SUPPORTED_REDUCTION_OPS,
    JsonDict,
    as_float,
    as_int,
    normalize_ranks,
)
from ..errors import CommCanaryError, SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
)
from ..workflows.qualification import load_verified_qualification_materialization

REFERENCE_EXECUTION_SCHEMA = "commcanary.reference-execution.stdout.v1"
REFERENCE_EXECUTOR = "commcanary.torch-distributed-reference.v2"
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 300

_COLLECTIVES = {
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
}
_POINT_TO_POINT = {"send", "recv"}
_REDUCTION_COLLECTIVES = {"all_reduce", "reduce_scatter"}
_VALIDATION_LANE_PERIOD = 8
_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
_INTEGER_DTYPES: Mapping[str, Tuple[int, bool]] = {
    "int8": (8, True),
    "uint8": (8, False),
    "int16": (16, True),
    "int32": (32, True),
    "int64": (64, True),
}
_REDUCTION_OUTCOME_INDEX = {
    "sum": 0,
    "avg": 1,
    "min": 2,
    "max": 3,
    "product": 4,
}
ValidationNumber = Union[int, float]


@dataclass(frozen=True)
class _ReductionProbe:
    values: Tuple[ValidationNumber, ...]
    outcomes: Tuple[ValidationNumber, ...]


@dataclass(frozen=True)
class QualificationExecutionPlan:
    """Completely preflighted execution inputs and bounded per-rank work."""

    request_id: str
    materialization_id: str
    program_sha256: str
    world_size: int
    iterations: int
    warmup: int
    distributed_timeout_seconds: int
    groups: Tuple[Tuple[int, Tuple[int, ...]], ...]
    entries: Tuple[Mapping[str, Any], ...]
    communication_entries_per_pass: int
    compute_operations_per_pass: int
    rank_compute_operations_per_pass: Tuple[int, ...]
    observation_samples: int
    rank_correctness_checks: Tuple[int, ...]
    rank_tensor_bytes: Tuple[int, ...]


def distributed_execution_environment(environ: Mapping[str, str]) -> Tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` from a torchrun environment."""

    try:
        rank = int(environ["RANK"])
        world_size = int(environ["WORLD_SIZE"])
        local_rank = int(environ.get("LOCAL_RANK", str(rank)))
    except (KeyError, ValueError) as exc:
        raise CommCanaryError("RANK, WORLD_SIZE, and LOCAL_RANK must be valid integers") from exc
    if world_size <= 0 or rank < 0 or rank >= world_size or local_rank < 0:
        raise CommCanaryError("distributed rank environment is outside the declared world")
    return rank, world_size, local_rank


def preflight_qualification_execution(
    request_directory: str,
    materialization_directory: str,
    *,
    world_size: int,
    iterations: int = 1,
    warmup: int = 1,
    distributed_timeout_seconds: int = DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> QualificationExecutionPlan:
    """Verify and bound an execution completely before importing PyTorch."""

    world = as_int(world_size)
    measured_passes = as_int(iterations)
    warmup_passes = as_int(warmup)
    timeout_seconds = as_int(distributed_timeout_seconds)
    if world <= 0 or world > limits.max_ranks:
        raise SchemaError(f"world_size must be in [1, {limits.max_ranks}]")
    if measured_passes <= 0:
        raise SchemaError("iterations must be positive")
    if warmup_passes < 0:
        raise SchemaError("warmup must be non-negative")
    if timeout_seconds <= 0:
        raise SchemaError("distributed_timeout_seconds must be positive")
    if timeout_seconds > limits.max_execution_timeout_seconds:
        raise SchemaError(
            "distributed_timeout_seconds="
            f"{timeout_seconds} exceeds max_execution_timeout_seconds="
            f"{limits.max_execution_timeout_seconds}"
        )

    verified_materialization = load_verified_qualification_materialization(
        request_directory,
        materialization_directory,
        limits=limits,
    )
    materialization = verified_materialization.manifest
    raw_entries = verified_materialization.program_entries
    if not raw_entries:
        raise SchemaError("qualification replay program must be a non-empty array")
    if not all(isinstance(entry, Mapping) for entry in raw_entries):
        raise SchemaError("qualification replay program entries must be objects")
    entries = tuple(copy.deepcopy(dict(entry)) for entry in raw_entries)

    groups: Dict[int, Tuple[int, ...]] = {}
    pending: Dict[int, Tuple[str, Tuple[int, ...], int]] = {}
    expected_overlap_wait: Optional[int] = None
    issued_requests: set[int] = set()
    source_event_indices: set[int] = set()
    communication_entries = 0
    compute_operations = 0
    rank_compute_operations = [0 for _ in range(world)]
    rank_correctness_checks = [0 for _ in range(world)]
    sample_width = 0
    tensor_specs: List[set[Tuple[str, Tuple[Any, ...], int, str]]] = [set() for _ in range(world)]

    for index, entry in enumerate(entries):
        if expected_overlap_wait is not None:
            if entry.get("comms") != "wait" or as_int(entry.get("req"), -1) != expected_overlap_wait:
                raise SchemaError(
                    f"replay program entry {index} must wait immediately after "
                    f"source-overlap compute for request {expected_overlap_wait}"
                )
        compute = entry.get("compute")
        if compute is not None:
            if compute != "gemm_recipe" or entry.get("comms") is not None:
                raise SchemaError(f"replay program entry {index} has unsupported compute operation")
            compute_ranks = tuple(normalize_ranks(entry.get("global_ranks")))
            if any(rank >= world for rank in compute_ranks):
                raise SchemaError(f"replay program entry {index} compute rank is outside world_size={world}")
            if entry.get("compute_phase") != "source-bound-overlap":
                raise SchemaError(
                    f"replay program entry {index} has unsupported compute_phase {entry.get('compute_phase')!r}"
                )
            overlap_request = _nonnegative_int(
                entry.get("overlap_request"),
                f"replay program entry {index} overlap_request",
            )
            pending_item = pending.get(overlap_request)
            if pending_item is None:
                raise SchemaError(f"replay program entry {index} overlaps an unknown or completed request")
            if compute_ranks != pending_item[1]:
                raise SchemaError(f"replay program entry {index} overlap ranks do not match the pending collective")
            source_event_index = _nonnegative_int(
                entry.get("source_event_index"),
                f"replay program entry {index} source_event_index",
            )
            if source_event_index != pending_item[2]:
                raise SchemaError(f"replay program entry {index} source event does not match the pending collective")
            recipes = _rank_compute_recipes(
                entry.get("recipe_by_rank"),
                ranks=compute_ranks,
                index=index,
                limits=limits,
            )
            expected_overlap_wait = overlap_request
            for rank in compute_ranks:
                rank_count = len(recipes[rank])
                rank_compute_operations[rank] = _checked_bounded_add(
                    rank_compute_operations[rank],
                    rank_count,
                    limit=limits.max_param_compute_operations,
                    label=f"qualification rank {rank} compute operations per pass",
                )
                compute_operations = _checked_bounded_add(
                    compute_operations,
                    rank_count,
                    limit=limits.max_param_compute_operations,
                    label="qualification compute operations per pass",
                )
                for operation in recipes[rank]:
                    dtype = str(operation["dtype"])
                    m = as_int(operation["m"])
                    n = as_int(operation["n"])
                    k = as_int(operation["k"])
                    left_elements, right_elements, output_elements = qualification_compute_tensor_elements(operation)
                    key = (m, n, k, dtype)
                    tensor_specs[rank].add(("gemm-a", key, left_elements, dtype))
                    tensor_specs[rank].add(("gemm-b", key, right_elements, dtype))
                    tensor_specs[rank].add(("gemm-out", key, output_elements, dtype))
            continue

        comms = entry.get("comms")
        if comms == "init":
            pg_id = _nonnegative_int(entry.get("pg_id"), f"replay program entry {index} pg_id")
            if pg_id in groups:
                raise SchemaError(f"replay program entry {index} duplicates process group {pg_id}")
            ranks = tuple(normalize_ranks(entry.get("global_ranks")))
            if any(rank >= world for rank in ranks):
                raise SchemaError(f"replay program entry {index} process-group rank is outside world_size={world}")
            declared_world = _positive_int(
                entry.get("world_size"),
                f"replay program entry {index} world_size",
            )
            if declared_world != len(ranks):
                raise SchemaError(f"replay program entry {index} world_size does not match process-group membership")
            groups[pg_id] = ranks
            continue
        if comms == "wait":
            request = _nonnegative_int(entry.get("req"), f"replay program entry {index} req")
            pending_item = pending.get(request)
            if pending_item is None:
                raise SchemaError(f"replay program entry {index} waits for an unknown or completed request")
            if (
                _nonnegative_int(
                    entry.get("source_event_index"),
                    f"replay program entry {index} source_event_index",
                )
                != pending_item[2]
            ):
                raise SchemaError(f"replay program entry {index} source event does not match the pending collective")
            del pending[request]
            if expected_overlap_wait == request:
                expected_overlap_wait = None
            continue
        if comms not in _COLLECTIVES and comms not in _POINT_TO_POINT:
            raise SchemaError(f"replay program entry {index} has unsupported communication {comms!r}")

        pg_id = _nonnegative_int(entry.get("pg_id"), f"replay program entry {index} pg_id")
        group_ranks = groups.get(pg_id)
        if group_ranks is None:
            raise SchemaError(f"replay program entry {index} references an uninitialized process group")
        if "global_ranks" in entry and tuple(normalize_ranks(entry.get("global_ranks"))) != group_ranks:
            raise SchemaError(f"replay program entry {index} process-group membership changed")
        if "world_size" in entry and as_int(entry.get("world_size")) != len(group_ranks):
            raise SchemaError(f"replay program entry {index} world_size changed")
        if comms == "broadcast":
            if "root" not in entry:
                raise SchemaError(f"replay program entry {index} broadcast is missing its source-bound root")
            root = _nonnegative_int(
                entry.get("root"),
                f"replay program entry {index} broadcast root",
            )
            if root not in group_ranks:
                raise SchemaError(f"replay program entry {index} broadcast root is outside its process group")
        elif "root" in entry:
            raise SchemaError(f"replay program entry {index} root is only valid for broadcast")
        if comms in _REDUCTION_COLLECTIVES:
            reduction_op = entry.get("reduction_op")
            if not isinstance(reduction_op, str) or reduction_op not in SUPPORTED_REDUCTION_OPS:
                raise SchemaError(
                    f"replay program entry {index} {comms} requires a source-bound "
                    f"reduction_op in {sorted(SUPPORTED_REDUCTION_OPS)!r}"
                )
        elif "reduction_op" in entry:
            raise SchemaError(f"replay program entry {index} reduction_op is only valid for reduction collectives")
        request = _nonnegative_int(entry.get("req"), f"replay program entry {index} req")
        if request in issued_requests:
            raise SchemaError(f"replay program entry {index} reuses request id {request}")
        issued_requests.add(request)
        dtype = require_param_dtype(
            entry.get("dtype"),
            label=f"replay program entry {index} communication dtype",
        )
        in_elements = _positive_int(
            entry.get("in_msg_size"),
            f"replay program entry {index} in_msg_size",
        )
        out_elements = _positive_int(
            entry.get("out_msg_size"),
            f"replay program entry {index} out_msg_size",
        )
        _validate_message_shape(
            str(comms),
            in_elements=in_elements,
            out_elements=out_elements,
            group_size=len(group_ranks),
            index=index,
        )
        communication_entries += 1

        if comms in _COLLECTIVES:
            if pending:
                raise SchemaError(f"replay program entry {index} issues a collective while another request is pending")
            source_event_index = _nonnegative_int(
                entry.get("source_event_index"),
                f"replay program entry {index} source_event_index",
            )
            if source_event_index in source_event_indices:
                raise SchemaError(f"replay program entry {index} duplicates source_event_index {source_event_index}")
            source_event_indices.add(source_event_index)
            pending[request] = (str(comms), group_ranks, source_event_index)
            sample_width = _checked_add(
                sample_width,
                len(group_ranks),
                label="qualification sample width",
            )
            for rank in group_ranks:
                rank_correctness_checks[rank] = _checked_add(
                    rank_correctness_checks[rank],
                    1,
                    label=f"qualification rank {rank} correctness checks",
                )
                _add_communication_tensor_specs(
                    tensor_specs[rank],
                    operation=str(comms),
                    request=request,
                    pg_id=pg_id,
                    in_elements=in_elements,
                    out_elements=out_elements,
                    dtype=dtype,
                )
        else:
            if pending:
                raise SchemaError(
                    f"replay program entry {index} issues point-to-point work while a collective is pending"
                )
            source = _nonnegative_int(
                entry.get("src_rank"),
                f"replay program entry {index} src_rank",
            )
            destination = _nonnegative_int(
                entry.get("dst_rank"),
                f"replay program entry {index} dst_rank",
            )
            if source not in group_ranks or destination not in group_ranks or source == destination:
                raise SchemaError(f"replay program entry {index} has invalid point-to-point endpoints")
            active_rank = source if comms == "send" else destination
            sample_width = _checked_add(sample_width, 1, label="qualification sample width")
            if comms == "recv":
                rank_correctness_checks[destination] = _checked_add(
                    rank_correctness_checks[destination],
                    1,
                    label=f"qualification rank {destination} correctness checks",
                )
            tensor_specs[active_rank].add(
                (
                    str(comms),
                    (request, pg_id, in_elements, dtype, source, destination),
                    in_elements,
                    dtype,
                )
            )

    if not groups:
        raise SchemaError("qualification replay program initializes no process groups")
    covered_ranks = tuple(sorted({rank for ranks in groups.values() for rank in ranks}))
    if covered_ranks != tuple(range(world)):
        raise SchemaError(
            "qualification replay process groups must cover exactly the launched "
            f"rank domain 0..{world - 1}; observed {list(covered_ranks)}"
        )
    if pending:
        raise SchemaError(f"qualification replay program leaves {len(pending)} collective requests pending")
    if communication_entries == 0:
        raise SchemaError("qualification replay program contains no measurable communications")
    declared_event_count = as_int(materialization["compute_work"]["event_count"])
    if source_event_indices != set(range(declared_event_count)):
        raise SchemaError("qualification replay program source-event indices do not match its compute-work audit")
    if compute_operations != as_int(materialization["program"]["compute_operation_count"]):
        raise SchemaError("qualification replay program rank-GEMM count does not match its materialization")
    declared_rank_counts = materialization["compute_work"]["rank_operation_counts"]
    expected_rank_keys = {str(rank) for rank in range(world)}
    if set(declared_rank_counts) != expected_rank_keys:
        raise SchemaError("qualification materialization rank-operation map does not match the launched rank domain")
    expected_rank_counts = tuple(as_int(declared_rank_counts[str(rank)]) for rank in range(world))
    if tuple(rank_compute_operations) != expected_rank_counts:
        raise SchemaError("qualification replay program per-rank GEMM recipes do not match its materialization")

    timed_and_warmup_passes = _checked_add(
        measured_passes,
        warmup_passes,
        label="qualification timed and warmup passes",
    )
    passes = _checked_add(
        timed_and_warmup_passes,
        1,
        label="qualification execution passes including correctness validation",
    )
    total_communications = _checked_multiply(
        communication_entries,
        passes,
        label="qualification communication operations",
    )
    _require_within(
        total_communications,
        limits.max_replay_events,
        label="qualification communication operations",
    )
    total_compute_operations = _checked_multiply(
        compute_operations,
        timed_and_warmup_passes,
        label="qualification execution compute operations",
    )
    _require_within(
        total_compute_operations,
        limits.max_execution_compute_operations,
        label="qualification execution compute operations",
    )
    observation_samples = _checked_multiply(
        sample_width,
        measured_passes,
        label="qualification observation samples",
    )
    _require_within(
        observation_samples,
        limits.max_execution_observation_samples,
        label="qualification observation samples",
    )

    rank_tensor_bytes = tuple(
        _planned_tensor_bytes(specs, limits=limits, rank=rank) for rank, specs in enumerate(tensor_specs)
    )
    _validate_correctness_probe_support(entries, groups)
    return QualificationExecutionPlan(
        request_id=str(materialization["request"]["request_id"]),
        materialization_id=str(materialization["materialization_id"]),
        program_sha256=str(materialization["program"]["sha256"]),
        world_size=world,
        iterations=measured_passes,
        warmup=warmup_passes,
        distributed_timeout_seconds=timeout_seconds,
        groups=tuple(sorted(groups.items())),
        entries=entries,
        communication_entries_per_pass=communication_entries,
        compute_operations_per_pass=compute_operations,
        rank_compute_operations_per_pass=tuple(rank_compute_operations),
        observation_samples=observation_samples,
        rank_correctness_checks=tuple(rank_correctness_checks),
        rank_tensor_bytes=rank_tensor_bytes,
    )


def execute_qualification_materialization(
    request_directory: str,
    materialization_directory: str,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: str = "cuda",
    backend: str = "nccl",
    iterations: int = 1,
    warmup: int = 1,
    distributed_timeout_seconds: int = DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Optional[JsonDict]:
    """Execute a verified plan and return one rank-aggregated diagnostic on rank 0."""

    if device not in {"cuda", "cpu"}:
        raise SchemaError("execution device must be 'cuda' or 'cpu'")
    if backend not in {"nccl", "gloo"}:
        raise SchemaError("execution backend must be 'nccl' or 'gloo'")
    if device == "cpu" and backend == "nccl":
        raise SchemaError("NCCL execution requires device='cuda'")
    if rank < 0 or rank >= world_size or local_rank < 0:
        raise SchemaError("execution rank is outside the declared world")

    plan = preflight_qualification_execution(
        request_directory,
        materialization_directory,
        world_size=world_size,
        iterations=iterations,
        warmup=warmup,
        distributed_timeout_seconds=distributed_timeout_seconds,
        limits=limits,
    )
    try:
        import torch  # type: ignore[import-not-found]
        import torch.distributed as dist  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CommCanaryError(
            "execute-materialization requires a target-compatible PyTorch installation; "
            "install PyTorch separately for the target CUDA/NCCL or CPU/Gloo environment"
        ) from exc

    if device == "cuda":
        if not torch.cuda.is_available():
            raise CommCanaryError("CUDA execution requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(local_rank)
    process_group_timeout = timedelta(seconds=plan.distributed_timeout_seconds)
    dist.init_process_group(backend, timeout=process_group_timeout)
    try:
        actual_rank = int(dist.get_rank())
        actual_world_size = int(dist.get_world_size())
        actual_backend = str(dist.get_backend())
        if actual_rank != rank or actual_world_size != world_size:
            raise CommCanaryError(
                "initialized distributed rank domain does not match the verified launch "
                f"(rank={actual_rank}, world_size={actual_world_size}; "
                f"expected rank={rank}, world_size={world_size})"
            )
        if actual_backend != backend:
            raise CommCanaryError(
                f"initialized distributed backend {actual_backend!r} does not match requested {backend!r}"
            )
        group_handles: Dict[int, Any] = {}
        for pg_id, ranks in plan.groups:
            # Every encoded pg_id is an explicit process-group identity. Reusing
            # WORLD for a dense membership would collapse distinct encoded
            # groups onto one communicator.
            group_handles[pg_id] = dist.new_group(
                ranks=list(ranks),
                timeout=process_group_timeout,
            )

        dtype_map = _torch_dtype_map(torch)
        buffers = _allocate_runtime_buffers(
            plan,
            rank=rank,
            device=device,
            torch=torch,
            dtype_map=dtype_map,
        )
        use_cuda_events = device == "cuda"
        local_correctness = _validate_runtime_communications(
            plan,
            rank=rank,
            group_handles=group_handles,
            buffers=buffers,
            dist=dist,
        )
        gathered_correctness: List[Any] = [None] * world_size
        dist.all_gather_object(gathered_correctness, local_correctness)
        correctness_checks = _validated_correctness_checks(
            plan,
            gathered_correctness,
        )

        def replay_once(
            iteration: int,
            *,
            measure: bool,
        ) -> Tuple[List[JsonDict], Optional[float]]:
            _reset_runtime_buffers(buffers, torch=torch)
            if device == "cuda":
                torch.cuda.synchronize()
            dist.barrier()
            program_started = time.perf_counter() if measure else None
            pending: Dict[int, Tuple[Any, Any, int, str]] = {}
            deferred_events: List[Tuple[Any, Any, int, str]] = []
            samples: List[JsonDict] = []
            for sequence, entry in enumerate(plan.entries):
                if entry.get("compute") == "gemm_recipe":
                    compute_ranks = normalize_ranks(entry["global_ranks"])
                    if rank not in compute_ranks:
                        continue
                    for operation in entry["recipe_by_rank"][str(rank)]:
                        operands = buffers["gemm"][
                            (
                                as_int(operation["m"]),
                                as_int(operation["n"]),
                                as_int(operation["k"]),
                                str(operation["dtype"]),
                            )
                        ]
                        torch.mm(operands[0], operands[1], out=operands[2])
                    continue
                comms = entry.get("comms")
                if comms in {"init", "wait"}:
                    if comms == "wait":
                        request = as_int(entry["req"])
                        pending_item = pending.pop(request, None)
                        if pending_item is not None:
                            work, pending_started, issue_sequence, operation = pending_item
                            work.wait()
                            if use_cuda_events:
                                ended = torch.cuda.Event(enable_timing=True)
                                ended.record()
                                if measure:
                                    deferred_events.append(
                                        (
                                            pending_started,
                                            ended,
                                            issue_sequence,
                                            operation,
                                        )
                                    )
                            elif measure:
                                samples.append(
                                    _sample(
                                        rank=rank,
                                        iteration=iteration,
                                        sequence=issue_sequence,
                                        request=request,
                                        operation=operation,
                                        duration_us=(time.perf_counter() - pending_started) * 1_000_000.0,
                                    )
                                )
                    continue
                if comms not in _COLLECTIVES and comms not in _POINT_TO_POINT:
                    continue
                pg_id = as_int(entry["pg_id"])
                ranks = dict(plan.groups)[pg_id]
                if comms in _COLLECTIVES and rank not in ranks:
                    continue
                if comms == "send" and rank != as_int(entry["src_rank"]):
                    continue
                if comms == "recv" and rank != as_int(entry["dst_rank"]):
                    continue
                started: Any
                if use_cuda_events:
                    started = torch.cuda.Event(enable_timing=True)
                    started.record()
                else:
                    started = time.perf_counter()
                work = _issue_runtime_operation(
                    entry,
                    rank=rank,
                    group=group_handles[pg_id],
                    buffers=buffers,
                    dist=dist,
                )
                request = as_int(entry["req"])
                if comms in _COLLECTIVES:
                    pending[request] = (work, started, sequence, str(comms))
                else:
                    work.wait()
                    if use_cuda_events:
                        ended = torch.cuda.Event(enable_timing=True)
                        ended.record()
                        if measure:
                            deferred_events.append((started, ended, sequence, str(comms)))
                    elif measure:
                        samples.append(
                            _sample(
                                rank=rank,
                                iteration=iteration,
                                sequence=sequence,
                                request=request,
                                operation=str(comms),
                                duration_us=(time.perf_counter() - started) * 1_000_000.0,
                            )
                        )
            if pending:
                raise CommCanaryError(
                    f"reference executor retained {len(pending)} pending requests after verified replay"
                )
            if device == "cuda":
                torch.cuda.synchronize()
            program_duration_us = (
                None if program_started is None else (time.perf_counter() - program_started) * 1_000_000.0
            )
            if measure:
                for started, ended, sequence, operation in deferred_events:
                    request = as_int(plan.entries[sequence]["req"])
                    samples.append(
                        _sample(
                            rank=rank,
                            iteration=iteration,
                            sequence=sequence,
                            request=request,
                            operation=operation,
                            duration_us=started.elapsed_time(ended) * 1000.0,
                        )
                    )
            dist.barrier()
            return (
                sorted(
                    samples,
                    key=lambda item: (
                        item["iteration"],
                        item["sequence"],
                        item["rank"],
                    ),
                ),
                program_duration_us,
            )

        for warmup_index in range(plan.warmup):
            replay_once(-(warmup_index + 1), measure=False)
        local_samples: List[JsonDict] = []
        local_program_makespans_us: List[float] = []
        for iteration in range(plan.iterations):
            iteration_samples, program_duration_us = replay_once(
                iteration,
                measure=True,
            )
            if program_duration_us is None or not math.isfinite(program_duration_us) or program_duration_us < 0.0:
                raise CommCanaryError(
                    f"reference executor produced an invalid program makespan at iteration {iteration}"
                )
            local_samples.extend(iteration_samples)
            local_program_makespans_us.append(program_duration_us)

        gathered: List[Any] = [None] * world_size
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "samples": local_samples,
                "program_makespans_us": local_program_makespans_us,
            },
        )
        if rank != 0:
            return None
        rank_samples, rank_program_makespans_us = _validated_rank_measurements(
            plan,
            gathered,
        )
        return _execution_payload(
            plan,
            rank_samples=rank_samples,
            rank_program_makespans_us=rank_program_makespans_us,
            device=device,
            backend=backend,
            torch=torch,
            dist=dist,
            correctness_checks=correctness_checks,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _validated_rank_measurements(
    plan: QualificationExecutionPlan,
    gathered: List[Any],
) -> Tuple[Dict[str, List[JsonDict]], Dict[str, List[float]]]:
    if len(gathered) != plan.world_size:
        raise CommCanaryError("reference executor measurements did not cover the launched world")
    rank_samples: Dict[str, List[JsonDict]] = {}
    rank_program_makespans_us: Dict[str, List[float]] = {}
    for rank, raw in enumerate(gathered):
        if not isinstance(raw, Mapping) or as_int(raw.get("rank")) != rank:
            raise CommCanaryError(f"reference executor measurement ownership disagrees at rank {rank}")
        raw_samples = raw.get("samples")
        raw_makespans = raw.get("program_makespans_us")
        if not isinstance(raw_samples, list):
            raise CommCanaryError(f"reference executor rank {rank} returned malformed samples")
        if not isinstance(raw_makespans, list) or len(raw_makespans) != plan.iterations:
            raise CommCanaryError(f"reference executor rank {rank} returned a malformed program-makespan inventory")
        makespans = [as_float(value) for value in raw_makespans]
        if any(not math.isfinite(value) or value < 0.0 for value in makespans):
            raise CommCanaryError(f"reference executor rank {rank} returned an invalid program makespan")
        rank_samples[str(rank)] = copy.deepcopy(raw_samples)
        rank_program_makespans_us[str(rank)] = [round(value, 3) for value in makespans]
    return rank_samples, rank_program_makespans_us


def _execution_payload(
    plan: QualificationExecutionPlan,
    *,
    rank_samples: Mapping[str, Any],
    rank_program_makespans_us: Mapping[str, Any],
    device: str,
    backend: str,
    torch: Any,
    dist: Any,
    correctness_checks: Tuple[int, ...],
) -> JsonDict:
    flattened: List[JsonDict] = []
    for rank in range(plan.world_size):
        raw_samples = rank_samples.get(str(rank))
        if not isinstance(raw_samples, list):
            raise CommCanaryError(f"reference executor did not receive samples from rank {rank}")
        for raw in raw_samples:
            if not isinstance(raw, Mapping):
                raise CommCanaryError(f"reference executor rank {rank} returned a malformed sample")
            if as_int(raw.get("rank")) != rank:
                raise CommCanaryError(f"reference executor sample ownership disagrees with gathered rank {rank}")
            flattened.append(dict(raw))
    if len(flattened) != plan.observation_samples:
        raise CommCanaryError(
            f"reference executor retained {len(flattened)} samples, expected {plan.observation_samples}"
        )
    grouped: Dict[Tuple[int, int, int, str], Dict[int, float]] = {}
    plan_groups = dict(plan.groups)
    for sample in flattened:
        iteration = as_int(sample["iteration"])
        sequence = as_int(sample["sequence"])
        request = as_int(sample["request"])
        operation = str(sample["operation"])
        sample_rank = as_int(sample["rank"])
        if iteration < 0 or iteration >= plan.iterations:
            raise CommCanaryError(f"reference executor sample iteration {iteration} is outside the measured plan")
        if sequence < 0 or sequence >= len(plan.entries):
            raise CommCanaryError(f"reference executor sample sequence {sequence} is outside the verified program")
        entry = plan.entries[sequence]
        if entry.get("comms") != operation or as_int(entry.get("req")) != request:
            raise CommCanaryError(
                f"reference executor sample at sequence {sequence} does not match the verified operation"
            )
        group_ranks = plan_groups[as_int(entry["pg_id"])]
        if operation in _COLLECTIVES:
            expected_ranks = set(group_ranks)
        elif operation == "send":
            expected_ranks = {as_int(entry["src_rank"])}
        elif operation == "recv":
            expected_ranks = {as_int(entry["dst_rank"])}
        else:
            raise CommCanaryError(f"reference executor sample at sequence {sequence} is not a measurable operation")
        if sample_rank not in expected_ranks:
            raise CommCanaryError(
                f"reference executor rank {sample_rank} did not participate in operation at sequence {sequence}"
            )
        duration = as_float(sample["duration_us"])
        if not math.isfinite(duration) or duration < 0.0:
            raise CommCanaryError(f"reference executor sample at sequence {sequence} has an invalid duration")
        key = (
            iteration,
            sequence,
            request,
            operation,
        )
        rank_durations = grouped.setdefault(key, {})
        if sample_rank in rank_durations:
            raise CommCanaryError(f"reference executor operation at sequence {sequence} duplicates rank {sample_rank}")
        rank_durations[sample_rank] = duration
    expected_groups = plan.communication_entries_per_pass * plan.iterations
    if len(grouped) != expected_groups:
        raise CommCanaryError(f"reference executor aggregated {len(grouped)} operations, expected {expected_groups}")
    for (_iteration, sequence, _request, operation), rank_durations in grouped.items():
        entry = plan.entries[sequence]
        if operation in _COLLECTIVES:
            expected_ranks = set(plan_groups[as_int(entry["pg_id"])])
        elif operation == "send":
            expected_ranks = {as_int(entry["src_rank"])}
        else:
            expected_ranks = {as_int(entry["dst_rank"])}
        if set(rank_durations) != expected_ranks:
            raise CommCanaryError(
                f"reference executor operation at sequence {sequence} has "
                f"rank samples {sorted(rank_durations)}, expected {sorted(expected_ranks)}"
            )
    timings = [round(max(grouped[key].values()), 3) for key in sorted(grouped)]
    if not timings:
        raise CommCanaryError("reference execution produced no measured communications")
    runtime_nccl = _runtime_nccl_version_code(torch) if backend == "nccl" else None
    program_makespans_us = [
        round(
            max(as_float(rank_program_makespans_us[str(rank)][iteration]) for rank in range(plan.world_size)),
            3,
        )
        for iteration in range(plan.iterations)
    ]
    return {
        "schema": REFERENCE_EXECUTION_SCHEMA,
        "request_id": plan.request_id,
        "materialization_id": plan.materialization_id,
        "program_sha256": plan.program_sha256,
        "executor": {
            "name": REFERENCE_EXECUTOR,
            "claim": "reference-implementation-not-yet-physically-conformance-validated",
            "device": device,
            "backend": backend,
            "world_size": plan.world_size,
            "iterations": plan.iterations,
            "warmup": plan.warmup,
            "distributed_timeout_seconds": plan.distributed_timeout_seconds,
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": (None if getattr(torch.version, "cuda", None) is None else str(torch.version.cuda)),
            "runtime_nccl_version_code": runtime_nccl,
            "distributed_backend": str(dist.get_backend()),
        },
        "rank_tensor_bytes": list(plan.rank_tensor_bytes),
        "rank_compute_operations_per_pass": list(plan.rank_compute_operations_per_pass),
        "correctness_validation": {
            "status": "passed",
            "semantics": "untimed-deterministic-communication-data-check",
            "checks_per_rank": list(correctness_checks),
            "total_check_count": sum(correctness_checks),
        },
        "rank_samples": rank_samples,
        "program_makespan": {
            "semantics": "maximum-rank-whole-program-wall-clock",
            "rank_timings_us": rank_program_makespans_us,
            "timings_us": program_makespans_us,
            "count": len(program_makespans_us),
            "median_us": round(
                float(statistics.median(program_makespans_us)),
                3,
            ),
            "max_us": round(max(program_makespans_us), 3),
        },
        "aggregate": {
            "semantics": "maximum-participating-rank-issue-to-explicit-wait",
            "timings_us": timings,
            "count": len(timings),
            "median_us": round(float(statistics.median(timings)), 3),
            "max_us": round(max(timings), 3),
        },
        "claims": {
            "physical_execution": "self_reported_reference_executor",
            "physical_fidelity": "unproven",
            "qualification_verdict": "not_issued",
        },
    }


def _runtime_nccl_version_code(torch: Any) -> int:
    """Return NCCL's integer version code without discarding tuple results."""

    try:
        raw_version = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError) as exc:
        raise CommCanaryError("NCCL execution could not report its runtime version") from exc
    if isinstance(raw_version, int) and not isinstance(raw_version, bool):
        if raw_version <= 0:
            raise CommCanaryError("NCCL execution reported a non-positive runtime version code")
        return raw_version
    if isinstance(raw_version, (tuple, list)) and len(raw_version) == 3:
        if not all(isinstance(component, int) and not isinstance(component, bool) for component in raw_version):
            raise CommCanaryError("NCCL execution reported non-integer runtime version components")
        major, minor, patch = (int(component) for component in raw_version)
        if major <= 0 or minor < 0 or minor >= 100 or patch < 0 or patch >= 100:
            raise CommCanaryError("NCCL execution reported invalid runtime version components")
        return major * 10_000 + minor * 100 + patch
    raise CommCanaryError("NCCL execution reported an unsupported runtime version representation")


def _torch_dtype_map(torch: Any) -> Dict[str, Any]:
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


def _allocate_runtime_buffers(
    plan: QualificationExecutionPlan,
    *,
    rank: int,
    device: str,
    torch: Any,
    dtype_map: Mapping[str, Any],
) -> Dict[str, MutableMapping[Any, Any]]:
    buffers: Dict[str, MutableMapping[Any, Any]] = {
        "gemm": {},
        "communication": {},
    }
    groups = dict(plan.groups)
    for entry in plan.entries:
        if entry.get("compute") == "gemm_recipe":
            if rank not in normalize_ranks(entry["global_ranks"]):
                continue
            for operation in entry["recipe_by_rank"][str(rank)]:
                key = (
                    as_int(operation["m"]),
                    as_int(operation["n"]),
                    as_int(operation["k"]),
                    str(operation["dtype"]),
                )
                if key in buffers["gemm"]:
                    continue
                dtype = dtype_map[key[3]]
                buffers["gemm"][key] = (
                    torch.randn(key[0], key[2], device=device, dtype=dtype),
                    torch.randn(key[2], key[1], device=device, dtype=dtype),
                    torch.empty(key[0], key[1], device=device, dtype=dtype),
                )
            continue
        comms = entry.get("comms")
        if comms not in _COLLECTIVES and comms not in _POINT_TO_POINT:
            continue
        pg_id = as_int(entry["pg_id"])
        ranks = groups[pg_id]
        if comms in _COLLECTIVES and rank not in ranks:
            continue
        if comms == "send" and rank != as_int(entry["src_rank"]):
            continue
        if comms == "recv" and rank != as_int(entry["dst_rank"]):
            continue
        key = _communication_buffer_key(entry)
        if key in buffers["communication"]:
            continue
        dtype = dtype_map[str(entry["dtype"])]
        input_tensor = torch.ones(as_int(entry["in_msg_size"]), device=device, dtype=dtype)
        output_tensor = (
            input_tensor
            if comms in {"all_reduce", "broadcast", "send", "recv"}
            else torch.empty(as_int(entry["out_msg_size"]), device=device, dtype=dtype)
        )
        buffers["communication"][key] = (input_tensor, output_tensor)
    return buffers


def _reset_runtime_buffers(buffers: Mapping[str, Mapping[Any, Any]], *, torch: Any) -> None:
    del torch
    for input_tensor, output_tensor in buffers["communication"].values():
        input_tensor.fill_(1)
        if output_tensor is not input_tensor:
            output_tensor.zero_()


def _cast_integer(value: int, dtype: str) -> int:
    bits, signed = _INTEGER_DTYPES[dtype]
    modulus = 1 << bits
    result = value % modulus
    if signed and result >= 1 << (bits - 1):
        result -= modulus
    return result


def _cast_float(value: float, dtype: str) -> float:
    if dtype == "float64":
        return float(value)
    if dtype == "float32":
        return float(struct.unpack(">f", struct.pack(">f", value))[0])
    if dtype == "float16":
        return float(struct.unpack(">e", struct.pack(">e", value))[0])
    if dtype == "bfloat16":
        bits = struct.unpack(">I", struct.pack(">f", value))[0]
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        return float(struct.unpack(">f", struct.pack(">I", rounded & 0xFFFF0000))[0])
    raise CommCanaryError(f"unsupported floating correctness dtype {dtype!r}")


def _reduction_outcomes(
    values: Sequence[ValidationNumber],
    *,
    dtype: str,
) -> Tuple[ValidationNumber, ...]:
    if dtype in _FLOAT_DTYPES:
        floating = [float(value) for value in values]
        total = math.fsum(floating)
        multiplied = math.prod(floating)
        return (
            _cast_float(total, dtype),
            _cast_float(total / len(floating), dtype),
            _cast_float(min(floating), dtype),
            _cast_float(max(floating), dtype),
            _cast_float(multiplied, dtype),
        )
    if dtype in _INTEGER_DTYPES:
        integers = [int(value) for value in values]
        integer_total = sum(integers)
        return (
            _cast_integer(integer_total, dtype),
            _cast_integer(math.trunc(integer_total / len(integers)), dtype),
            min(integers),
            max(integers),
            _cast_integer(math.prod(integers), dtype),
        )
    raise CommCanaryError(f"dtype {dtype!r} cannot carry an injective reduction correctness probe")


@lru_cache(maxsize=None)
def _reduction_probe_candidates(
    dtype: str,
    group_size: int,
    reduction_op: str,
) -> Tuple[_ReductionProbe, ...]:
    if group_size < 2:
        raise CommCanaryError("a one-rank group cannot distinguish collective reduction operators")
    outcome_index = _REDUCTION_OUTCOME_INDEX[reduction_op]
    raw_candidates: List[_ReductionProbe] = []
    if dtype in _FLOAT_DTYPES:
        for tag in range(32):
            offset = tag / 64.0
            raw_values = [0.5 + offset, 1.5 + offset, *([1.0] * (group_size - 2))]
            values = tuple(_cast_float(value, dtype) for value in raw_values)
            outcomes = _reduction_outcomes(values, dtype=dtype)
            if len(set(outcomes)) == len(_REDUCTION_OUTCOME_INDEX):
                raw_candidates.append(_ReductionProbe(values=values, outcomes=outcomes))
    elif dtype in _INTEGER_DTYPES:
        _bits, signed = _INTEGER_DTYPES[dtype]
        candidate_values = range(-8, 16) if signed else range(0, 24)
        seen_values = set()
        for first, second, repeated in product(candidate_values, repeat=3):
            integer_values = (first, second) if group_size == 2 else (first, second, *([repeated] * (group_size - 2)))
            values = tuple(_cast_integer(value, dtype) for value in integer_values)
            if values in seen_values:
                continue
            seen_values.add(values)
            outcomes = _reduction_outcomes(values, dtype=dtype)
            if len(set(outcomes)) == len(_REDUCTION_OUTCOME_INDEX):
                raw_candidates.append(_ReductionProbe(values=values, outcomes=outcomes))
    else:
        raise CommCanaryError(f"dtype {dtype!r} cannot carry an injective reduction correctness probe")
    by_expected: Dict[ValidationNumber, _ReductionProbe] = {}
    for candidate in raw_candidates:
        by_expected.setdefault(candidate.outcomes[outcome_index], candidate)
    return tuple(by_expected[key] for key in sorted(by_expected))


def _stable_validation_offset(*parts: object, modulus: int) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def _reduction_probe(
    *,
    request_id: str,
    request: int,
    group_ranks: Tuple[int, ...],
    destination_rank: Optional[int],
    lane: int,
    dtype: str,
    reduction_op: str,
) -> _ReductionProbe:
    candidates = _reduction_probe_candidates(dtype, len(group_ranks), reduction_op)
    required = len(group_ranks) if destination_rank is not None else 1
    if len(candidates) < required:
        raise CommCanaryError(
            f"dtype {dtype!r} has only {len(candidates)} distinguishable {reduction_op} probes "
            f"for a {len(group_ranks)}-rank reduce_scatter"
        )
    destination_index = 0 if destination_rank is None else group_ranks.index(destination_rank)
    offset = _stable_validation_offset(
        request_id,
        request,
        lane,
        dtype,
        reduction_op,
        modulus=len(candidates),
    )
    return candidates[(offset + destination_index) % len(candidates)]


def _routing_value(
    *,
    request_id: str,
    request: int,
    source_rank: int,
    destination_rank: Optional[int],
    lane: int,
    dtype: str,
) -> int:
    modulus = _routing_modulus(dtype)
    destination = -1 if destination_rank is None else destination_rank
    base = _stable_validation_offset(request_id, request, modulus=modulus)
    return (base + 17 * source_rank + 31 * destination + 7 * lane) % modulus


def _routing_modulus(dtype: str) -> int:
    if dtype == "bool":
        raise CommCanaryError("bool tensors cannot carry injective communication routing signatures")
    if dtype == "int8":
        return 127
    if dtype == "uint8":
        return 251
    if dtype in {"float16", "bfloat16"}:
        return 127
    return 32_749


def _validate_correctness_probe_support(
    entries: Tuple[Mapping[str, Any], ...],
    groups: Mapping[int, Tuple[int, ...]],
) -> None:
    for index, entry in enumerate(entries):
        comms = entry.get("comms")
        if comms not in _COLLECTIVES and comms not in _POINT_TO_POINT:
            continue
        dtype = str(entry["dtype"])
        try:
            capacity = _routing_modulus(dtype)
        except CommCanaryError as exc:
            raise SchemaError(
                f"replay program entry {index} cannot be checked by the reference executor: {exc}"
            ) from exc
        group_ranks = groups[as_int(entry["pg_id"])]
        if comms in {"all_gather", "all_to_all"} and len(group_ranks) > capacity:
            raise SchemaError(f"replay program entry {index} group size exceeds the {dtype} routing-signature capacity")
        if comms not in _REDUCTION_COLLECTIVES:
            continue
        reduction_op = str(entry["reduction_op"])
        try:
            candidates = _reduction_probe_candidates(dtype, len(group_ranks), reduction_op)
        except CommCanaryError as exc:
            raise SchemaError(f"replay program entry {index} cannot distinguish its reduction operator: {exc}") from exc
        required = len(group_ranks) if comms == "reduce_scatter" else 1
        if len(candidates) < required:
            raise SchemaError(
                f"replay program entry {index} dtype {dtype!r} cannot encode {required} "
                f"distinct {reduction_op} reduce-scatter shard signatures"
            )


def _fill_tensor_pattern(tensor: Any, *, length: int, value_for_lane: Any) -> None:
    period = min(_VALIDATION_LANE_PERIOD, length)
    for lane in range(period):
        tensor[lane::period].fill_(value_for_lane(lane))


def _tensor_pattern_matches(tensor: Any, *, length: int, value_for_lane: Any) -> bool:
    period = min(_VALIDATION_LANE_PERIOD, length)
    return all(_tensor_all_equal(tensor[lane::period], value_for_lane(lane)) for lane in range(period))


def _initialize_validation_buffers(
    plan: QualificationExecutionPlan,
    *,
    rank: int,
    buffers: Mapping[str, Mapping[Any, Any]],
) -> None:
    groups = dict(plan.groups)
    for entry in plan.entries:
        comms = entry.get("comms")
        if comms not in _COLLECTIVES and comms not in _POINT_TO_POINT:
            continue
        group_ranks = groups[as_int(entry["pg_id"])]
        if comms in _COLLECTIVES and rank not in group_ranks:
            continue
        if comms == "send" and rank != as_int(entry["src_rank"]):
            continue
        if comms == "recv" and rank != as_int(entry["dst_rank"]):
            continue
        input_tensor, output_tensor = buffers["communication"][_communication_buffer_key(entry)]
        input_tensor.zero_()
        if output_tensor is not input_tensor:
            output_tensor.zero_()
        request = as_int(entry["req"])
        dtype = str(entry["dtype"])
        if comms in _REDUCTION_COLLECTIVES:
            source_index = group_ranks.index(rank)
            destinations = (None,) if comms == "all_reduce" else tuple(group_ranks)
            segment_length = as_int(entry["in_msg_size"]) if comms == "all_reduce" else as_int(entry["out_msg_size"])
            for destination_index, destination in enumerate(destinations):
                segment = input_tensor.narrow(0, destination_index * segment_length, segment_length)

                def reduction_value(lane: int, *, destination_rank: Optional[int] = destination) -> ValidationNumber:
                    probe = _reduction_probe(
                        request_id=plan.request_id,
                        request=request,
                        group_ranks=group_ranks,
                        destination_rank=destination_rank,
                        lane=lane,
                        dtype=dtype,
                        reduction_op=str(entry["reduction_op"]),
                    )
                    return probe.values[source_index]

                _fill_tensor_pattern(segment, length=segment_length, value_for_lane=reduction_value)
            continue
        if comms == "all_to_all":
            segment_length = as_int(entry["in_msg_size"]) // len(group_ranks)
            for destination_index, destination in enumerate(group_ranks):
                segment = input_tensor.narrow(0, destination_index * segment_length, segment_length)
                _fill_tensor_pattern(
                    segment,
                    length=segment_length,
                    value_for_lane=lambda lane, destination_rank=destination: _routing_value(
                        request_id=plan.request_id,
                        request=request,
                        source_rank=rank,
                        destination_rank=destination_rank,
                        lane=lane,
                        dtype=dtype,
                    ),
                )
            continue
        if comms == "all_gather":
            source_rank = rank
            destination_rank = None
        elif comms == "broadcast":
            source_rank = as_int(entry["root"])
            destination_rank = None
            if rank != source_rank:
                continue
        elif comms == "send":
            source_rank = as_int(entry["src_rank"])
            destination_rank = as_int(entry["dst_rank"])
        elif comms == "recv":
            continue
        else:
            raise CommCanaryError(f"correctness initialization reached unsupported operation {comms!r}")
        length = as_int(entry["in_msg_size"])
        _fill_tensor_pattern(
            input_tensor,
            length=length,
            value_for_lane=lambda lane: _routing_value(
                request_id=plan.request_id,
                request=request,
                source_rank=source_rank,
                destination_rank=destination_rank,
                lane=lane,
                dtype=dtype,
            ),
        )


def _validate_runtime_communications(
    plan: QualificationExecutionPlan,
    *,
    rank: int,
    group_handles: Mapping[int, Any],
    buffers: Mapping[str, Mapping[Any, Any]],
    dist: Any,
) -> JsonDict:
    """Run one untimed deterministic data check over every communication."""

    groups = dict(plan.groups)
    _initialize_validation_buffers(plan, rank=rank, buffers=buffers)

    pending: Dict[int, Tuple[Any, Mapping[str, Any]]] = {}
    checks = 0
    failure_count = 0
    failures: List[str] = []
    for sequence, entry in enumerate(plan.entries):
        comms = entry.get("comms")
        if comms == "wait":
            request = as_int(entry["req"])
            pending_item = pending.pop(request, None)
            if pending_item is None:
                continue
            work, issued_entry = pending_item
            work.wait()
            checks += 1
            if not _validation_output_matches(
                issued_entry,
                request_id=plan.request_id,
                rank=rank,
                group_ranks=groups[as_int(issued_entry["pg_id"])],
                buffers=buffers,
            ):
                failure_count += 1
                if len(failures) < 8:
                    failures.append(f"sequence {sequence} request {request} {issued_entry['comms']}")
            continue
        if comms not in _COLLECTIVES and comms not in _POINT_TO_POINT:
            continue
        pg_id = as_int(entry["pg_id"])
        group_ranks = groups[pg_id]
        if comms in _COLLECTIVES and rank not in group_ranks:
            continue
        if comms == "send" and rank != as_int(entry["src_rank"]):
            continue
        if comms == "recv" and rank != as_int(entry["dst_rank"]):
            continue
        work = _issue_runtime_operation(
            entry,
            rank=rank,
            group=group_handles[pg_id],
            buffers=buffers,
            dist=dist,
        )
        if comms in _COLLECTIVES:
            pending[as_int(entry["req"])] = (work, entry)
            continue
        work.wait()
        if comms == "recv":
            checks += 1
            if not _validation_output_matches(
                entry,
                request_id=plan.request_id,
                rank=rank,
                group_ranks=group_ranks,
                buffers=buffers,
            ):
                failure_count += 1
                if len(failures) < 8:
                    failures.append(f"sequence {sequence} request {entry['req']} recv")
    if pending:
        raise CommCanaryError(f"correctness validation retained {len(pending)} pending requests after verified replay")
    return {
        "rank": rank,
        "check_count": checks,
        "failure_count": failure_count,
        "failures": failures,
    }


def _validation_output_matches(
    entry: Mapping[str, Any],
    *,
    request_id: str,
    rank: int,
    group_ranks: Tuple[int, ...],
    buffers: Mapping[str, Mapping[Any, Any]],
) -> bool:
    comms = str(entry["comms"])
    _input_tensor, output_tensor = buffers["communication"][_communication_buffer_key(entry)]
    request = as_int(entry["req"])
    dtype = str(entry["dtype"])
    if comms in _REDUCTION_COLLECTIVES:
        destination_rank = None if comms == "all_reduce" else rank
        length = as_int(entry["out_msg_size"])
        outcome_index = _REDUCTION_OUTCOME_INDEX[str(entry["reduction_op"])]
        return _tensor_pattern_matches(
            output_tensor,
            length=length,
            value_for_lane=lambda lane: _reduction_probe(
                request_id=request_id,
                request=request,
                group_ranks=group_ranks,
                destination_rank=destination_rank,
                lane=lane,
                dtype=dtype,
                reduction_op=str(entry["reduction_op"]),
            ).outcomes[outcome_index],
        )
    if comms in {"broadcast", "recv"}:
        source_rank = as_int(entry["root"] if comms == "broadcast" else entry["src_rank"])
        destination_rank = None if comms == "broadcast" else as_int(entry["dst_rank"])
        length = as_int(entry["out_msg_size"])
        return _tensor_pattern_matches(
            output_tensor,
            length=length,
            value_for_lane=lambda lane: _routing_value(
                request_id=request_id,
                request=request,
                source_rank=source_rank,
                destination_rank=destination_rank,
                lane=lane,
                dtype=dtype,
            ),
        )
    if comms in {"all_gather", "all_to_all"}:
        segment_length = (
            as_int(entry["in_msg_size"]) if comms == "all_gather" else as_int(entry["out_msg_size"]) // len(group_ranks)
        )
        destination_rank = None if comms == "all_gather" else rank
        for source_index, source_rank in enumerate(group_ranks):
            segment = output_tensor.narrow(0, source_index * segment_length, segment_length)
            if not _tensor_pattern_matches(
                segment,
                length=segment_length,
                value_for_lane=lambda lane, source=source_rank: _routing_value(
                    request_id=request_id,
                    request=request,
                    source_rank=source,
                    destination_rank=destination_rank,
                    lane=lane,
                    dtype=dtype,
                ),
            ):
                return False
        return True
    raise CommCanaryError(f"correctness validation reached unsupported result operation {comms!r}")


def _validation_expected_output_values(
    entry: Mapping[str, Any],
    *,
    request_id: str,
    rank: int,
    group_ranks: Tuple[int, ...],
) -> Tuple[ValidationNumber, ...]:
    """Return the pure expected tensor for small tests and independent adapters."""

    comms = str(entry["comms"])
    request = as_int(entry["req"])
    dtype = str(entry["dtype"])
    output_size = as_int(entry["out_msg_size"])
    values: List[ValidationNumber] = []
    if comms in _REDUCTION_COLLECTIVES:
        destination_rank = None if comms == "all_reduce" else rank
        outcome_index = _REDUCTION_OUTCOME_INDEX[str(entry["reduction_op"])]
        for index in range(output_size):
            lane = index % min(_VALIDATION_LANE_PERIOD, output_size)
            values.append(
                _reduction_probe(
                    request_id=request_id,
                    request=request,
                    group_ranks=group_ranks,
                    destination_rank=destination_rank,
                    lane=lane,
                    dtype=dtype,
                    reduction_op=str(entry["reduction_op"]),
                ).outcomes[outcome_index]
            )
        return tuple(values)
    if comms in {"broadcast", "recv"}:
        source_rank = as_int(entry["root"] if comms == "broadcast" else entry["src_rank"])
        destination_rank = None if comms == "broadcast" else as_int(entry["dst_rank"])
        for index in range(output_size):
            lane = index % min(_VALIDATION_LANE_PERIOD, output_size)
            values.append(
                _routing_value(
                    request_id=request_id,
                    request=request,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    lane=lane,
                    dtype=dtype,
                )
            )
        return tuple(values)
    if comms in {"all_gather", "all_to_all"}:
        segment_length = as_int(entry["in_msg_size"]) if comms == "all_gather" else output_size // len(group_ranks)
        destination_rank = None if comms == "all_gather" else rank
        for source_rank in group_ranks:
            for index in range(segment_length):
                lane = index % min(_VALIDATION_LANE_PERIOD, segment_length)
                values.append(
                    _routing_value(
                        request_id=request_id,
                        request=request,
                        source_rank=source_rank,
                        destination_rank=destination_rank,
                        lane=lane,
                        dtype=dtype,
                    )
                )
        return tuple(values)
    raise CommCanaryError(f"expected-value model reached unsupported operation {comms!r}")


def _tensor_all_equal(tensor: Any, expected: ValidationNumber) -> bool:
    return bool(tensor.eq(expected).all().item())


def _validated_correctness_checks(
    plan: QualificationExecutionPlan,
    gathered: List[Any],
) -> Tuple[int, ...]:
    if len(gathered) != plan.world_size:
        raise CommCanaryError("correctness validation did not cover the launched world")
    checks: List[int] = []
    failures: List[str] = []
    for rank, raw in enumerate(gathered):
        if not isinstance(raw, Mapping) or as_int(raw.get("rank")) != rank:
            raise CommCanaryError(f"correctness validation result ownership disagrees at rank {rank}")
        check_count = as_int(raw.get("check_count"))
        if check_count != plan.rank_correctness_checks[rank]:
            raise CommCanaryError(
                f"correctness validation rank {rank} retained {check_count} checks, "
                f"expected {plan.rank_correctness_checks[rank]}"
            )
        failure_count = as_int(raw.get("failure_count"))
        raw_failures = raw.get("failures")
        if (
            failure_count < 0
            or not isinstance(raw_failures, list)
            or not all(isinstance(item, str) for item in raw_failures)
            or len(raw_failures) > min(failure_count, 8)
        ):
            raise CommCanaryError(f"correctness validation rank {rank} returned malformed failures")
        if failure_count:
            detail = ", ".join(raw_failures) if raw_failures else "details omitted"
            failures.append(f"rank {rank}: {failure_count} failures ({detail})")
        checks.append(check_count)
    if failures:
        raise CommCanaryError("reference executor communication correctness validation failed: " + "; ".join(failures))
    return tuple(checks)


def _issue_runtime_operation(
    entry: Mapping[str, Any],
    *,
    rank: int,
    group: Any,
    buffers: Mapping[str, Mapping[Any, Any]],
    dist: Any,
) -> Any:
    comms = str(entry["comms"])
    input_tensor, output_tensor = buffers["communication"][_communication_buffer_key(entry)]
    if comms == "all_reduce":
        return dist.all_reduce(
            input_tensor,
            op=_torch_reduction_op(dist, entry),
            group=group,
            async_op=True,
        )
    if comms == "broadcast":
        source = as_int(entry["root"])
        return dist.broadcast(input_tensor, src=source, group=group, async_op=True)
    if comms == "all_gather":
        return dist.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=True)
    if comms == "reduce_scatter":
        return dist.reduce_scatter_tensor(
            output_tensor,
            input_tensor,
            op=_torch_reduction_op(dist, entry),
            group=group,
            async_op=True,
        )
    if comms == "all_to_all":
        return dist.all_to_all_single(output_tensor, input_tensor, group=group, async_op=True)
    if comms == "send":
        return dist.isend(
            input_tensor,
            dst=as_int(entry["dst_rank"]),
            group=group,
        )
    if comms == "recv":
        return dist.irecv(
            input_tensor,
            src=as_int(entry["src_rank"]),
            group=group,
        )
    raise CommCanaryError(f"reference executor reached unsupported operation {comms!r} on rank {rank}")


def _torch_reduction_op(dist: Any, entry: Mapping[str, Any]) -> Any:
    reduction_op = str(entry["reduction_op"])
    attribute = {
        "avg": "AVG",
        "max": "MAX",
        "min": "MIN",
        "product": "PRODUCT",
        "sum": "SUM",
    }[reduction_op]
    try:
        return getattr(dist.ReduceOp, attribute)
    except AttributeError as exc:
        raise CommCanaryError(
            f"target PyTorch does not expose ReduceOp.{attribute} required by the source program"
        ) from exc


def _communication_buffer_key(entry: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        as_int(entry["req"]),
        str(entry["comms"]),
        as_int(entry["pg_id"]),
        as_int(entry["in_msg_size"]),
        as_int(entry["out_msg_size"]),
        str(entry["dtype"]),
        entry.get("src_rank"),
        entry.get("dst_rank"),
    )


def _sample(
    *,
    rank: int,
    iteration: int,
    sequence: int,
    request: int,
    operation: str,
    duration_us: float,
) -> JsonDict:
    if not math.isfinite(duration_us) or duration_us < 0.0:
        raise CommCanaryError("reference executor produced a non-finite or negative duration")
    return {
        "rank": rank,
        "iteration": iteration,
        "sequence": sequence,
        "request": request,
        "operation": operation,
        "duration_us": round(duration_us, 3),
    }


def _validate_message_shape(
    operation: str,
    *,
    in_elements: int,
    out_elements: int,
    group_size: int,
    index: int,
) -> None:
    valid = False
    if operation in {"all_reduce", "broadcast", "send", "recv"}:
        valid = in_elements == out_elements
    elif operation == "all_to_all":
        valid = in_elements == out_elements and in_elements % group_size == 0
    elif operation == "all_gather":
        valid = out_elements == in_elements * group_size
    elif operation == "reduce_scatter":
        valid = in_elements == out_elements * group_size
    if not valid:
        raise SchemaError(f"replay program entry {index} has invalid {operation} input/output element counts")


def _rank_compute_recipes(
    value: Any,
    *,
    ranks: Tuple[int, ...],
    index: int,
    limits: ResourceLimits,
) -> Dict[int, Tuple[JsonDict, ...]]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"replay program entry {index} recipe_by_rank must be an object")
    expected_keys = {str(rank) for rank in ranks}
    if set(value) != expected_keys:
        raise SchemaError(f"replay program entry {index} recipe_by_rank must match global_ranks")
    result: Dict[int, Tuple[JsonDict, ...]] = {}
    for rank in ranks:
        raw_operations = value[str(rank)]
        if not isinstance(raw_operations, list):
            raise SchemaError(f"replay program entry {index} recipe_by_rank[{rank}] must be an array")
        if len(raw_operations) > limits.max_param_entries:
            raise SchemaError(
                f"replay program entry {index} recipe_by_rank[{rank}] exceeds "
                f"max_param_entries={limits.max_param_entries}"
            )
        operations: List[JsonDict] = []
        for operation_index, raw_operation in enumerate(raw_operations):
            label = f"replay program entry {index} recipe_by_rank[{rank}][{operation_index}]"
            if not isinstance(raw_operation, Mapping):
                raise SchemaError(f"{label} must be an object")
            expected_fields = {"op", "dtype", "m", "n", "k"}
            if set(raw_operation) != expected_fields:
                raise SchemaError(f"{label} fields do not match the exact GEMM recipe")
            if raw_operation.get("op") != "gemm":
                raise SchemaError(f"{label} op must be 'gemm'")
            dtype = require_param_compute_dtype(
                raw_operation.get("dtype"),
                label=f"{label} dtype",
            )
            operation: JsonDict = {"op": "gemm", "dtype": dtype}
            for dimension in ("m", "n", "k"):
                parsed = _positive_int(
                    raw_operation.get(dimension),
                    f"{label} {dimension}",
                )
                if parsed > limits.max_param_gemm_dim:
                    raise SchemaError(
                        f"{label} {dimension}={parsed} exceeds max_param_gemm_dim={limits.max_param_gemm_dim}"
                    )
                operation[dimension] = parsed
            operations.append(operation)
        result[rank] = tuple(operations)
    return result


def _add_communication_tensor_specs(
    specs: set[Tuple[str, Tuple[Any, ...], int, str]],
    *,
    operation: str,
    request: int,
    pg_id: int,
    in_elements: int,
    out_elements: int,
    dtype: str,
) -> None:
    key = (request, operation, pg_id, in_elements, out_elements, dtype)
    specs.add(("communication-input", key, in_elements, dtype))
    if operation not in {"all_reduce", "broadcast"}:
        specs.add(("communication-output", key, out_elements, dtype))


def _planned_tensor_bytes(
    specs: set[Tuple[str, Tuple[Any, ...], int, str]],
    *,
    limits: ResourceLimits,
    rank: int,
) -> int:
    total = 0
    for _kind, _key, elements, dtype in sorted(specs, key=repr):
        size = _checked_multiply(
            elements,
            dtype_size_bytes(dtype),
            label=f"qualification rank {rank} tensor bytes",
        )
        _require_within(
            size,
            limits.max_execution_tensor_bytes,
            label=f"qualification rank {rank} tensor bytes",
        )
        total = _checked_bounded_add(
            total,
            size,
            limit=limits.max_execution_total_tensor_bytes,
            label=f"qualification rank {rank} total tensor bytes",
        )
    return total


def _positive_int(value: Any, label: str) -> int:
    parsed = as_int(value)
    if parsed <= 0:
        raise SchemaError(f"{label} must be positive")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    parsed = as_int(value)
    if parsed < 0:
        raise SchemaError(f"{label} must be non-negative")
    return parsed


def _checked_add(left: int, right: int, *, label: str) -> int:
    try:
        return checked_add(left, right, label=label)
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def _checked_multiply(left: int, right: int, *, label: str) -> int:
    try:
        return checked_multiply(left, right, label=label)
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def _require_within(value: int, limit: int, *, label: str) -> int:
    try:
        return require_within(value, limit, label=label)
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def _checked_bounded_add(
    left: int,
    right: int,
    *,
    limit: int,
    label: str,
) -> int:
    return _require_within(
        _checked_add(left, right, label=label),
        limit,
        label=label,
    )


__all__ = [
    "DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS",
    "REFERENCE_EXECUTION_SCHEMA",
    "QualificationExecutionPlan",
    "distributed_execution_environment",
    "execute_qualification_materialization",
    "preflight_qualification_execution",
]
