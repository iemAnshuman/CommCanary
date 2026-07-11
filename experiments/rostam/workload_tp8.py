#!/usr/bin/env python3
"""Decode-like TP8 workload for the Rostam physical experiment.

This file intentionally imports torch lazily so ``python workload_tp8.py --help``
works on machines without torch. The default GEMM is a small bf16 matrix
multiply, ``gemm_m x hidden`` by ``hidden x gemm_n``. On an A100-SXM4, the
default ``gemm_m=16, hidden=8192, gemm_n=8192`` is intended to land in the
rough 200-400 us per-layer band after calibration, but the exact value depends
on the installed torch/CUDA stack and should be checked on Rostam.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Sequence, Tuple

WORKLOAD_STDOUT_SCHEMA = "commcanary.rostam.workload_tp8.stdout.v1"


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
    tokens: int,
    layers: int,
    hidden: int,
    gemm_m: int,
    gemm_n: int,
    dtype: str,
    message_sizes: Sequence[int],
    inject_skew: float,
    timings_us: Sequence[float],
) -> Dict[str, object]:
    rounded = [round(value, 3) for value in timings_us]
    return {
        "schema": WORKLOAD_STDOUT_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "tokens": tokens,
        "layers": layers,
        "hidden": hidden,
        "gemm_m_rank0": gemm_m,
        "gemm_n": gemm_n,
        "dtype": dtype,
        "msg_sizes_bytes": list(message_sizes),
        "inject_skew": inject_skew,
        "timings_us": rounded,
        "metrics": {
            "median_us": round(_median(rounded), 3),
            "iqr_us": round(_iqr(rounded), 3),
            "count": len(rounded),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Torch distributed TP8 decode-like workload for CommCanary Rostam experiments."
    )
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--gemm-m", type=int, default=16)
    parser.add_argument("--gemm-n", type=int, default=None)
    parser.add_argument("--dtype", type=_dtype_name, default="bf16")
    parser.add_argument("--msg-sizes", type=_parse_size_list, default=_parse_size_list("64K,128K,256K"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile")
    parser.add_argument("--inject-skew", type=float, default=0.0)
    return parser


def _load_torch() -> Tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.distributed as dist  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "workload_tp8.py requires torch for execution. "
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


def _scaled_gemm_m(base: int, rank: int, world_size: int, inject_skew: float) -> int:
    if world_size <= 1 or inject_skew == 0.0:
        return max(1, base)
    position = (rank / float(world_size - 1) - 0.5) * 2.0
    factor = 1.0 + inject_skew * position
    return max(1, int(round(base * factor)))


def _run_layer(torch: Any, dist: Any, activation: Any, weight: Any, comm_buffer: Any) -> None:
    torch.matmul(activation, weight)
    dist.all_reduce(comm_buffer, op=dist.ReduceOp.SUM)


def run(args: argparse.Namespace) -> int:
    if args.layers <= 0 or args.tokens <= 0 or args.hidden <= 0 or args.gemm_m <= 0:
        raise SystemExit("layers, tokens, hidden, and gemm-m must be positive")
    if args.gemm_n is not None and args.gemm_n <= 0:
        raise SystemExit("gemm-n must be positive")
    if args.inject_skew < 0:
        raise SystemExit("--inject-skew must be non-negative")

    torch, dist = _load_torch()
    local_rank = _local_rank()
    rank = _rank()
    world_size = _world_size()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    dtype = _torch_dtype(torch, args.dtype)
    gemm_m = _scaled_gemm_m(args.gemm_m, rank, world_size, args.inject_skew)
    gemm_n = args.gemm_n if args.gemm_n is not None else args.hidden
    activation = torch.randn((gemm_m, args.hidden), device=device, dtype=dtype)
    weight = torch.randn((args.hidden, gemm_n), device=device, dtype=dtype)
    comm_buffers = [
        torch.empty(
            (max(1, math.ceil(size / torch.tensor([], dtype=dtype).element_size())),), device=device, dtype=dtype
        )
        for size in args.msg_sizes
    ]
    for buffer in comm_buffers:
        buffer.fill_(rank + 1)

    def token_step(token_index: int) -> None:
        for layer in range(args.layers):
            comm_buffer = comm_buffers[(token_index * args.layers + layer) % len(comm_buffers)]
            _run_layer(torch, dist, activation, weight, comm_buffer)

    for warmup_index in range(args.warmup):
        token_step(warmup_index)
    torch.cuda.synchronize(device)
    dist.barrier()

    activities = None
    if args.profile and rank == 0:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    profiler_context = (
        # record_shapes=True is required on torch 2.2: the collective metadata
        # (Collective name, In/Out msg nelems, ...) rides the input-capture
        # path in this torch generation; without it, record_param_comms
        # events carry no args and import-kineto correctly fails closed.
        torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False)
        if activities is not None
        else nullcontext()
    )

    latencies_us: List[float] = []
    with profiler_context as prof:
        for token_index in range(args.tokens):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            token_step(token_index)
            end.record()
            end.synchronize()
            latencies_us.append(float(start.elapsed_time(end) * 1000.0))
            if prof is not None and hasattr(prof, "step"):
                prof.step()

    dist.barrier()
    if args.profile and rank == 0 and prof is not None:
        prof.export_chrome_trace(args.profile)

    if rank == 0:
        result = _result_payload(
            rank=rank,
            world_size=world_size,
            tokens=args.tokens,
            layers=args.layers,
            hidden=args.hidden,
            gemm_m=gemm_m,
            gemm_n=gemm_n,
            dtype=args.dtype,
            message_sizes=args.msg_sizes,
            inject_skew=args.inject_skew,
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
