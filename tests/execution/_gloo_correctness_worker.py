"""Four-process worker for the real Gloo correctness-oracle test."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, List, Mapping

import torch
import torch.distributed as dist

from commcanary.execution.qualification import (
    QualificationExecutionPlan,
    _allocate_runtime_buffers,
    _communication_buffer_key,
    _torch_dtype_map,
    _validate_runtime_communications,
    _validated_correctness_checks,
    _validation_expected_output_values,
    _validation_output_matches,
)


def _communication(
    request: int,
    operation: str,
    *,
    group_size: int,
    reduction_op: str | None = None,
) -> Dict[str, Any]:
    input_size = 16
    output_size = 16
    if operation == "all_gather":
        output_size = input_size * group_size
    elif operation == "reduce_scatter":
        input_size = output_size * group_size
    entry: Dict[str, Any] = {
        "comms": operation,
        "req": request,
        "pg_id": 0,
        "global_ranks": list(range(group_size)),
        "in_msg_size": input_size,
        "out_msg_size": output_size,
        "dtype": "float32",
    }
    if reduction_op is not None:
        entry["reduction_op"] = reduction_op
    if operation == "broadcast":
        entry["root"] = 1
    return entry


def _plan(group_size: int) -> QualificationExecutionPlan:
    entries: List[Mapping[str, Any]] = []
    request = 1
    for reduction_op in ("avg", "max", "min", "product", "sum"):
        entries.extend(
            (
                _communication(request, "all_reduce", group_size=group_size, reduction_op=reduction_op),
                {"comms": "wait", "req": request},
            )
        )
        request += 1
    for operation, reduction_op in (
        ("all_gather", None),
        ("reduce_scatter", "sum"),
        ("all_to_all", None),
        ("broadcast", None),
    ):
        entries.extend(
            (
                _communication(request, operation, group_size=group_size, reduction_op=reduction_op),
                {"comms": "wait", "req": request},
            )
        )
        request += 1
    return QualificationExecutionPlan(
        request_id="gloo-correctness-conformance",
        materialization_id="0" * 64,
        program_sha256="1" * 64,
        world_size=group_size,
        iterations=1,
        warmup=0,
        distributed_timeout_seconds=60,
        groups=((0, tuple(range(group_size))),),
        entries=tuple(entries),
        communication_entries_per_pass=9,
        compute_operations_per_pass=0,
        rank_compute_operations_per_pass=(0,) * group_size,
        observation_samples=9 * group_size,
        rank_correctness_checks=(9,) * group_size,
        rank_tensor_bytes=(0,) * group_size,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        plan = _plan(world_size)
        group = dist.new_group(ranks=list(range(world_size)), timeout=timedelta(seconds=60))
        buffers = _allocate_runtime_buffers(
            plan,
            rank=rank,
            device="cpu",
            torch=torch,
            dtype_map=_torch_dtype_map(torch),
        )
        local = _validate_runtime_communications(
            plan,
            rank=rank,
            group_handles={0: group},
            buffers=buffers,
            dist=dist,
        )
        gathered: List[Any] = [None] * world_size
        dist.all_gather_object(gathered, local)
        assert _validated_correctness_checks(plan, gathered) == (9,) * world_size

        all_to_all = next(entry for entry in plan.entries if entry.get("comms") == "all_to_all")
        _input, output = buffers["communication"][_communication_buffer_key(all_to_all)]
        segment_length = int(all_to_all["out_msg_size"]) // world_size
        if rank in {0, 3}:
            foreign_rank, target_source, foreign_source = (3, 0, 2) if rank == 0 else (0, 2, 0)
            foreign = _validation_expected_output_values(
                all_to_all,
                request_id=plan.request_id,
                rank=foreign_rank,
                group_ranks=tuple(range(world_size)),
            )
            foreign_chunk = foreign[foreign_source * segment_length : (foreign_source + 1) * segment_length]
            output.narrow(0, target_source * segment_length, segment_length).copy_(
                torch.tensor(foreign_chunk, dtype=output.dtype)
            )
            assert not _validation_output_matches(
                all_to_all,
                request_id=plan.request_id,
                rank=rank,
                group_ranks=tuple(range(world_size)),
                buffers=buffers,
            )
        if rank == 0:
            print("gloo correctness oracle passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
