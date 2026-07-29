#!/usr/bin/env python3
"""Multi-rank CUDA-profile producer with real compute/communication overlap.

Unlike ``workload_tp8.py``, this producer issues each all-reduce
asynchronously, launches the layer GEMM, and waits explicitly afterward. Every
rank records one Kineto profile into a new, shared, otherwise-empty directory.
The output is a capture diagnostic for constructing a future frozen campaign;
it is not compatible with the historical blocking-workload producer schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

if __package__:
    from . import workload_tp8 as _workload
else:
    import workload_tp8 as _workload  # type: ignore[import-not-found,no-redef]

_dtype_name = _workload._dtype_name
_iqr = _workload._iqr
_load_torch = _workload._load_torch
_local_rank = _workload._local_rank
_median = _workload._median
_parse_size_list = _workload._parse_size_list
_rank = _workload._rank
_scaled_gemm_m = _workload._scaled_gemm_m
_torch_dtype = _workload._torch_dtype
_world_size = _workload._world_size

OVERLAP_CAPTURE_STDOUT_SCHEMA = "commcanary.rostam.workload-overlap-capture.stdout.v2"
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 300
DEFAULT_MEASUREMENT_ITERATIONS = 5
DEFAULT_PROFILE_WARMUP_PROGRAMS = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile every rank of an asynchronous NCCL/GEMM overlap workload.")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--gemm-m", type=int, default=16)
    parser.add_argument("--gemm-n", type=int, default=None)
    parser.add_argument("--dtype", type=_dtype_name, default="bf16")
    parser.add_argument("--msg-sizes", type=_parse_size_list, default=_parse_size_list("64K,128K,256K"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--measurement-iterations",
        type=int,
        default=DEFAULT_MEASUREMENT_ITERATIONS,
        help="unprofiled whole-program measurements to collect before the separate Kineto capture",
    )
    parser.add_argument(
        "--profile-warmup-programs",
        type=int,
        default=DEFAULT_PROFILE_WARMUP_PROGRAMS,
        help="whole programs used to warm Kineto before one active captured program",
    )
    parser.add_argument("--inject-skew", type=float, default=0.0)
    parser.add_argument("--profile-directory", required=True)
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS,
    )
    return parser


def _run_overlap_layer(
    torch: Any,
    dist: Any,
    activation: Any,
    weight: Any,
    comm_buffer: Any,
) -> None:
    work = dist.all_reduce(
        comm_buffer,
        op=dist.ReduceOp.SUM,
        async_op=True,
    )
    torch.matmul(activation, weight)
    work.wait()


def _profile_path(directory: Path, *, rank: int) -> Path:
    if directory.is_symlink():
        raise SystemExit("profile directory must not be a symlink")
    try:
        directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create profile directory: {exc}") from exc
    if not directory.is_dir() or directory.is_symlink():
        raise SystemExit("profile directory must be a real directory")
    path = directory / f"rank-{rank:05d}.json"
    if path.exists() or path.is_symlink():
        raise SystemExit(f"profile output already exists: {path.name}")
    return path


def _profile_identity(path: Path, *, rank: int) -> Dict[str, object]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise SystemExit(f"cannot inspect rank {rank} profile: {exc}") from exc
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"rank {rank} profile is not a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SystemExit(f"cannot hash rank {rank} profile: {exc}") from exc
    return {
        "rank": rank,
        "filename": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": stat_result.st_size,
    }


def _validate_profile_inventory(
    profiles: Sequence[Any],
    *,
    world_size: int,
) -> List[Dict[str, object]]:
    if len(profiles) != world_size:
        raise SystemExit("profile inventory does not cover the launched world")
    normalized: List[Dict[str, object]] = []
    for expected_rank, raw in enumerate(profiles):
        if not isinstance(raw, Mapping) or raw.get("rank") != expected_rank:
            raise SystemExit(f"profile inventory disagrees at rank {expected_rank}")
        filename = raw.get("filename")
        digest = raw.get("sha256")
        size_bytes = raw.get("size_bytes")
        if filename != f"rank-{expected_rank:05d}.json":
            raise SystemExit(f"profile inventory filename disagrees at rank {expected_rank}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"profile inventory digest is invalid at rank {expected_rank}")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise SystemExit(f"profile inventory size is invalid at rank {expected_rank}")
        normalized.append(dict(raw))
    return normalized


def _max_rank_timings(
    rank_timings: Sequence[Any],
    *,
    world_size: int,
    measurement_iterations: int,
) -> Tuple[List[List[float]], List[float]]:
    if len(rank_timings) != world_size:
        raise SystemExit("rank timing inventory does not cover the launched world")
    normalized: List[List[float]] = []
    for rank, raw in enumerate(rank_timings):
        if not isinstance(raw, list) or len(raw) != measurement_iterations:
            raise SystemExit(f"rank {rank} timing count disagrees with measurement iterations")
        values = [float(value) for value in raw]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise SystemExit(f"rank {rank} timings contain an invalid duration")
        normalized.append(values)
    maxima = [
        max(normalized[rank][iteration] for rank in range(world_size)) for iteration in range(measurement_iterations)
    ]
    return normalized, maxima


def _result_payload(
    *,
    world_size: int,
    tokens: int,
    layers: int,
    hidden: int,
    gemm_m_rank0: int,
    gemm_n: int,
    dtype: str,
    message_sizes: Sequence[int],
    inject_skew: float,
    warmup_programs: int,
    measurement_iterations: int,
    profile_warmup_programs: int,
    distributed_timeout_seconds: int,
    profiles: Sequence[Any],
    rank_timings: Sequence[Any],
) -> Dict[str, object]:
    profile_inventory = _validate_profile_inventory(profiles, world_size=world_size)
    normalized_timings, maxima = _max_rank_timings(
        rank_timings,
        world_size=world_size,
        measurement_iterations=measurement_iterations,
    )
    rounded_maxima = [round(value, 3) for value in maxima]
    rounded_by_rank = [[round(value, 3) for value in values] for values in normalized_timings]
    return {
        "schema": OVERLAP_CAPTURE_STDOUT_SCHEMA,
        "rank": 0,
        "world_size": world_size,
        "execution_semantics": "async-all-reduce-then-gemm-then-explicit-wait",
        "timing_semantics": "maximum-rank-unprofiled-whole-program-duration",
        "tokens": tokens,
        "layers": layers,
        "hidden": hidden,
        "gemm_m_rank0": gemm_m_rank0,
        "gemm_n": gemm_n,
        "dtype": dtype,
        "msg_sizes_bytes": list(message_sizes),
        "inject_skew": inject_skew,
        "warmup_programs": warmup_programs,
        "measurement_iterations": measurement_iterations,
        "profile_warmup_programs": profile_warmup_programs,
        "profiled_programs": 1,
        "distributed_timeout_seconds": distributed_timeout_seconds,
        "profiles": profile_inventory,
        "rank_timings_us": rounded_by_rank,
        "timings_us": rounded_maxima,
        "metrics": {
            "median_us": round(_median(rounded_maxima), 3),
            "iqr_us": round(_iqr(rounded_maxima), 3),
            "count": len(rounded_maxima),
        },
    }


def _positive_arguments(args: argparse.Namespace) -> None:
    if args.layers <= 0 or args.tokens <= 0 or args.hidden <= 0 or args.gemm_m <= 0:
        raise SystemExit("layers, tokens, hidden, and gemm-m must be positive")
    if args.gemm_n is not None and args.gemm_n <= 0:
        raise SystemExit("gemm-n must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.measurement_iterations <= 0 or args.measurement_iterations > 1000:
        raise SystemExit("--measurement-iterations must be in [1, 1000]")
    if args.profile_warmup_programs <= 0 or args.profile_warmup_programs > 100:
        raise SystemExit("--profile-warmup-programs must be in [1, 100]")
    if args.inject_skew < 0.0:
        raise SystemExit("--inject-skew must be non-negative")
    if args.distributed_timeout_seconds <= 0 or args.distributed_timeout_seconds > 3600:
        raise SystemExit("--distributed-timeout-seconds must be in [1, 3600]")


def run(args: argparse.Namespace) -> int:
    _positive_arguments(args)
    torch, dist = _load_torch()
    local_rank = _local_rank()
    rank = _rank()
    world_size = _world_size()
    if world_size < 2 or rank < 0 or rank >= world_size or local_rank < 0:
        raise SystemExit("overlap capture requires a valid multi-rank launch")

    torch.cuda.set_device(local_rank)
    process_group_timeout = timedelta(seconds=args.distributed_timeout_seconds)
    dist.init_process_group(backend="nccl", timeout=process_group_timeout)
    try:
        profile_directory = Path(args.profile_directory)
        profile_path = _profile_path(profile_directory, rank=rank)
        dist.barrier()
        unexpected = sorted(path.name for path in profile_directory.iterdir())
        if unexpected:
            raise SystemExit(f"profile directory was not empty before capture: {unexpected!r}")
        dist.barrier()

        device = torch.device("cuda", local_rank)
        dtype = _torch_dtype(torch, args.dtype)
        gemm_m = _scaled_gemm_m(args.gemm_m, rank, world_size, args.inject_skew)
        gemm_n = args.gemm_n if args.gemm_n is not None else args.hidden
        activation = torch.randn((gemm_m, args.hidden), device=device, dtype=dtype)
        weight = torch.randn((args.hidden, gemm_n), device=device, dtype=dtype)
        element_size = torch.tensor([], dtype=dtype).element_size()
        comm_buffers = [
            torch.zeros(
                (max(1, math.ceil(size / element_size)),),
                device=device,
                dtype=dtype,
            )
            for size in args.msg_sizes
        ]

        def token_step(token_index: int) -> None:
            for layer in range(args.layers):
                comm_buffer = comm_buffers[(token_index * args.layers + layer) % len(comm_buffers)]
                _run_overlap_layer(torch, dist, activation, weight, comm_buffer)

        def program_step() -> None:
            for token_index in range(args.tokens):
                token_step(token_index)

        for _ in range(args.warmup):
            program_step()
        torch.cuda.synchronize(device)
        dist.barrier()

        latencies_us: List[float] = []
        for _ in range(args.measurement_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            program_step()
            end.record()
            end.synchronize()
            latencies_us.append(float(start.elapsed_time(end) * 1000.0))
        dist.barrier()

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=args.profile_warmup_programs,
                active=1,
                repeat=1,
            ),
        ) as profiler:
            for _ in range(args.profile_warmup_programs + 1):
                program_step()
                profiler.step()
        torch.cuda.synchronize(device)
        profiler.export_chrome_trace(str(profile_path))
        dist.barrier()

        local_profile = _profile_identity(profile_path, rank=rank)
        gathered_profiles: List[Any] = [None] * world_size
        gathered_timings: List[Any] = [None] * world_size
        dist.all_gather_object(gathered_profiles, local_profile)
        dist.all_gather_object(gathered_timings, latencies_us)
        if rank == 0:
            result = _result_payload(
                world_size=world_size,
                tokens=args.tokens,
                layers=args.layers,
                hidden=args.hidden,
                gemm_m_rank0=gemm_m,
                gemm_n=gemm_n,
                dtype=args.dtype,
                message_sizes=args.msg_sizes,
                inject_skew=args.inject_skew,
                warmup_programs=args.warmup,
                measurement_iterations=args.measurement_iterations,
                profile_warmup_programs=args.profile_warmup_programs,
                distributed_timeout_seconds=args.distributed_timeout_seconds,
                profiles=gathered_profiles,
                rank_timings=gathered_timings,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
