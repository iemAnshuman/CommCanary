#!/usr/bin/env python3
"""Pure torch.distributed TP8 microbenchmark for the Rostam physical experiment.

This file intentionally imports torch lazily so ``python microbench_tp8.py --help``
works on machines without torch. It measures back-to-back NCCL all_reduce calls
over bf16-sized communication buffers, matching the message-size cycling and
communication dtype of ``workload_tp8.py`` while removing all interleaved
compute. Its stdout contract is microbenchmark-specific and contains no
synthetic workload-shape fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

MICRO_STDOUT_SCHEMA = "commcanary.rostam.microbench_tp8.stdout.v1"


def _parse_size(value: str) -> int:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("message size cannot be empty")
    suffix = text[-1].upper()
    scale = 1
    number = text
    if suffix in ("K", "M", "G"):
        number = text[:-1]
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[suffix]
    try:
        parsed = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("message size must be positive")
    return int(parsed * scale)


def _parse_size_list(value: str) -> List[int]:
    sizes = [_parse_size(item) for item in value.split(",") if item.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one message size is required")
    return sizes


def _dtype_name(value: str) -> str:
    if value not in ("bf16", "fp16", "fp32"):
        raise argparse.ArgumentTypeError("dtype must be one of: bf16, fp16, fp32")
    return value


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def _iqr(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = sorted(float(item) for item in values)
    mid = len(xs) // 2
    if len(xs) % 2:
        lower = xs[:mid]
        upper = xs[mid + 1 :]
    else:
        lower = xs[:mid]
        upper = xs[mid:]
    if not lower or not upper:
        return 0.0
    return float(statistics.median(upper) - statistics.median(lower))


def _result_payload(
    *,
    rank: int,
    world_size: int,
    dtype: str,
    message_sizes: Sequence[int],
    timings_us: Sequence[float],
) -> Dict[str, object]:
    rounded = [round(value, 3) for value in timings_us]
    return {
        "schema": MICRO_STDOUT_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "dtype": dtype,
        "msg_sizes_bytes": list(message_sizes),
        "timings_us": rounded,
        "metrics": {
            "median_us": round(_median(rounded), 3),
            "iqr_us": round(_iqr(rounded), 3),
            "count": len(rounded),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Torch distributed TP8 all_reduce microbenchmark for CommCanary Rostam experiments."
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", type=_dtype_name, default="bf16")
    parser.add_argument("--msg-sizes", type=_parse_size_list, default=_parse_size_list("64K,128K,256K"))
    parser.add_argument("--warmup", type=int, default=20)
    return parser


def _load_torch() -> Tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.distributed as dist  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "microbench_tp8.py requires torch for execution. "
            "Install via experiments/rostam/setup.sh; --help works without torch."
        ) from exc
    return torch, dist


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))


def _run_collective(dist: Any, comm_buffer: Any) -> None:
    dist.all_reduce(comm_buffer, op=dist.ReduceOp.SUM)


def run(args: argparse.Namespace) -> int:
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("iters must be positive and warmup must be non-negative")

    torch, dist = _load_torch()
    local_rank = _local_rank()
    rank = _rank()
    world_size = _world_size()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    dtype = _torch_dtype(torch, args.dtype)
    element_size = torch.tensor([], dtype=dtype).element_size()
    comm_buffers = [
        torch.empty((max(1, math.ceil(size / element_size)),), device=device, dtype=dtype) for size in args.msg_sizes
    ]
    for buffer in comm_buffers:
        buffer.fill_(rank + 1)

    for warmup_index in range(args.warmup):
        comm_buffer = comm_buffers[warmup_index % len(comm_buffers)]
        _run_collective(dist, comm_buffer)
    torch.cuda.synchronize(device)
    dist.barrier()

    latencies_us: List[float] = []
    for iteration in range(args.iters):
        comm_buffer = comm_buffers[iteration % len(comm_buffers)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_collective(dist, comm_buffer)
        end.record()
        end.synchronize()
        latencies_us.append(float(start.elapsed_time(end) * 1000.0))

    torch.cuda.synchronize(device)
    dist.barrier()

    if rank == 0:
        result = _result_payload(
            rank=rank,
            world_size=world_size,
            dtype=args.dtype,
            message_sizes=args.msg_sizes,
            timings_us=latencies_us,
        )
        print(json.dumps(result, sort_keys=True), flush=True)

    dist.destroy_process_group()
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
