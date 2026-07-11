#!/usr/bin/env python3
"""Overlap-aware reference replayer for CommCanary PARAM-format traces.

PARAM's commsTraceReplay hardwires blocking execution (``self.is_blocking =
True``), so it cannot express compute/communication concurrency no matter
what the trace says. This reference replayer consumes the same exported
basic-format trace (init / comm / gemm / wait entries) and honors the
overlap structure: collectives are issued with ``async_op=True`` and only
awaited at their explicit ``wait`` entry, so the GEMM entries in between
genuinely execute while the collective is in flight.

Per-collective latency = issue-to-completion time (CUDA events on GPU,
perf_counter on CPU), i.e. the exposed completion latency including any
interference from concurrent compute. Rank 0 prints an overlap-specific,
strict stdout JSON payload with no synthetic workload placeholders.

Trace validation, including exact dense-world membership and every request /
wait lifetime, runs before torch is imported or a process group is initialized.
This also keeps ``--help`` usable on machines without torch.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

OVERLAP_STDOUT_SCHEMA = "commcanary.rostam.overlap_replay.stdout.v1"


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def _iqr(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if len(ordered) < 2:
        return 0.0
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[mid + (len(ordered) % 2) :]
    if not lower or not upper:
        return 0.0
    return _median(upper) - _median(lower)


def _result_payload(
    *,
    rank: int,
    world_size: int,
    timings_us: Sequence[float],
) -> Dict[str, object]:
    rounded = [round(value, 3) for value in timings_us]
    if not rounded:
        raise ValueError("overlap replay produced no measured collectives")
    return {
        "schema": OVERLAP_STDOUT_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "timings_us": rounded,
        "metrics": {
            "median_us": round(_median(rounded), 3),
            "iqr_us": round(_iqr(rounded), 3),
            "count": len(rounded),
        },
    }


def _distributed_environment(environ: Mapping[str, str]) -> Tuple[int, int, int]:
    try:
        rank = int(environ["RANK"])
        world_size = int(environ["WORLD_SIZE"])
        local_rank = int(environ.get("LOCAL_RANK", str(rank)))
    except (KeyError, ValueError) as exc:
        raise SystemExit("RANK, WORLD_SIZE, and LOCAL_RANK must be valid integers") from exc
    if world_size <= 0 or rank < 0 or rank >= world_size or local_rank < 0:
        raise SystemExit("distributed rank environment is outside the declared world")
    return rank, world_size, local_rank


def _prepare_replay(
    trace_path: str,
    *,
    world_size: int,
    iterations: int,
    warmup: int,
) -> Tuple[List[Mapping[str, Any]], Dict[str, int]]:
    """Pure preflight used before torch or distributed state exists."""

    if iterations <= 0 or warmup < 0:
        raise SystemExit("iters must be positive and warmup must be non-negative")
    if not __package__:
        repository_root = str(Path(__file__).resolve().parents[2])
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
    from experiments.rostam.lib.physical_results import load_validated_param_trace

    return load_validated_param_trace(
        trace_path,
        world_size=world_size,
        require_explicit_waits=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--iters", type=int, default=1, help="measured replay passes")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup passes")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    return parser


def run(args: argparse.Namespace) -> int:
    rank, world_size, local_rank = _distributed_environment(os.environ)
    entries, _ = _prepare_replay(
        args.trace_path,
        world_size=world_size,
        iterations=args.iters,
        warmup=args.warmup,
    )

    import torch  # type: ignore[import-not-found]
    import torch.distributed as dist  # type: ignore[import-not-found]

    _DTYPES = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float64": torch.float64,
        "double": torch.float64,
    }

    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(args.backend)

    # Preflight proved that every init entry is exactly dense WORLD. Reuse is
    # therefore an explicit supported-layout decision, never a length alias.
    groups = {}
    for entry in entries:
        if entry.get("comms") == "init":
            groups[entry["pg_id"]] = dist.group.WORLD

    # preallocated buffers per (pg_id, nelems, dtype) and gemm operands per
    # (dim, dtype); allocation must never sit inside the timed loop
    comm_buffers = {}
    gemm_operands = {}
    for entry in entries:
        if entry.get("comms") in ("init", "wait", None):
            if entry.get("compute") == "gemm":
                gemm_key = (entry["mm_dim"], entry.get("dtype", "float32"))
                if gemm_key not in gemm_operands:
                    dtype = _DTYPES[gemm_key[1]]
                    gemm_operands[gemm_key] = (
                        torch.randn(gemm_key[0], gemm_key[0], device=args.device, dtype=dtype),
                        torch.randn(gemm_key[0], gemm_key[0], device=args.device, dtype=dtype),
                    )
            continue
        comm_key = (entry["pg_id"], entry["in_msg_size"], entry["dtype"])
        if comm_key not in comm_buffers:
            dtype = _DTYPES[entry["dtype"]]
            comm_buffers[comm_key] = torch.ones(entry["in_msg_size"], device=args.device, dtype=dtype)

    use_events = args.device == "cuda"

    def replay_once(measure: bool) -> List[float]:
        pending = {}  # req -> (work, start marker)
        latencies: List[float] = []
        event_pairs = []  # collected after ONE sync at pass end: per-op
        # synchronization would block host run-ahead and distort pipelining
        for entry in entries:
            if entry.get("compute") == "gemm":
                a, b = gemm_operands[(entry["mm_dim"], entry.get("dtype", "float32"))]
                for _ in range(entry["count"]):
                    torch.mm(a, b)
                continue
            comms = entry.get("comms")
            if comms == "init":
                continue
            if comms == "all_reduce":
                buf = comm_buffers[(entry["pg_id"], entry["in_msg_size"], entry["dtype"])]
                if use_events:
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start = time.perf_counter()
                work = dist.all_reduce(buf, group=groups[entry["pg_id"]], async_op=True)
                pending[entry["req"]] = (work, start)
                continue
            if comms == "wait":
                work, start = pending.pop(entry["req"])
                work.wait()
                if use_events:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    if measure:
                        event_pairs.append((start, end))
                else:
                    if measure:
                        latencies.append((time.perf_counter() - start) * 1_000_000.0)
                continue
        if pending:
            raise SystemExit(f"{len(pending)} collectives were never awaited; malformed trace")
        if args.device == "cuda":
            torch.cuda.synchronize()
        for start, end in event_pairs:
            latencies.append(start.elapsed_time(end) * 1000.0)
        dist.barrier()
        return latencies

    for _ in range(args.warmup):
        replay_once(measure=False)
    timed: List[float] = []
    for _ in range(args.iters):
        timed.extend(replay_once(measure=True))

    if rank == 0:
        result = _result_payload(rank=rank, world_size=world_size, timings_us=timed)
        print(json.dumps(result, sort_keys=True), flush=True)

    dist.destroy_process_group()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
