"""Two-process worker for the optional real Gloo correctness-oracle test."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, List, Mapping

import torch
import torch.distributed as dist

from commcanary.execution.qualification import (
    QualificationExecutionPlan,
    _allocate_runtime_buffers,
    _torch_dtype_map,
    _validate_runtime_communications,
    _validated_correctness_checks,
)


def _communication(
    request: int,
    operation: str,
    *,
    reduction_op: str | None = None,
) -> Dict[str, Any]:
    input_size = 16
    output_size = 16
    if operation == "all_gather":
        output_size = input_size * 2
    elif operation == "reduce_scatter":
        input_size = output_size * 2
    entry: Dict[str, Any] = {
        "comms": operation,
        "req": request,
        "pg_id": 0,
        "global_ranks": [0, 1],
        "in_msg_size": input_size,
        "out_msg_size": output_size,
        "dtype": "float32",
    }
    if reduction_op is not None:
        entry["reduction_op"] = reduction_op
    if operation == "broadcast":
        entry["root"] = 1
    return entry


def _plan() -> QualificationExecutionPlan:
    entries: List[Mapping[str, Any]] = []
    request = 1
    for reduction_op in ("avg", "max", "min", "product", "sum"):
        entries.extend(
            (
                _communication(request, "all_reduce", reduction_op=reduction_op),
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
                _communication(request, operation, reduction_op=reduction_op),
                {"comms": "wait", "req": request},
            )
        )
        request += 1
    return QualificationExecutionPlan(
        request_id="gloo-correctness-conformance",
        materialization_id="0" * 64,
        program_sha256="1" * 64,
        world_size=2,
        iterations=1,
        warmup=0,
        distributed_timeout_seconds=60,
        groups=((0, (0, 1)),),
        entries=tuple(entries),
        communication_entries_per_pass=9,
        compute_operations_per_pass=0,
        rank_compute_operations_per_pass=(0, 0),
        observation_samples=18,
        rank_correctness_checks=(9, 9),
        rank_tensor_bytes=(0, 0),
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        plan = _plan()
        group = dist.new_group(ranks=[0, 1], timeout=timedelta(seconds=60))
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
        gathered: List[Any] = [None, None]
        dist.all_gather_object(gathered, local)
        assert _validated_correctness_checks(plan, gathered) == (9, 9)
        if rank == 0:
            print("gloo correctness oracle passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
