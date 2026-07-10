#!/usr/bin/env python3
"""Overlap-aware reference replayer for CommCanary PARAM-format traces.

PARAM's commsTraceReplay hardwires blocking execution (``self.is_blocking =
True``), so it cannot express compute/communication concurrency no matter
what the trace says. This ~180-line replayer consumes the SAME exported
basic-format trace (init / comm / gemm / wait entries) and honors the
overlap structure: collectives are issued with ``async_op=True`` and only
awaited at their explicit ``wait`` entry, so the GEMM entries in between
genuinely execute while the collective is in flight.

Per-collective latency = issue-to-completion time (CUDA events on GPU,
perf_counter on CPU), i.e. the exposed completion latency INCLUDING any
interference from concurrent compute. Rank 0 prints the same stdout JSON
schema as workload_tp8.py / microbench_tp8.py so the result wrappers parse
all three identically.

This file intentionally imports torch lazily so ``--help`` works on
machines without torch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import List


def _median(values) -> float:
    return float(statistics.median(list(values)))


def _iqr(values) -> float:
    ordered = sorted(values)
    if len(ordered) < 2:
        return 0.0
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[mid + (len(ordered) % 2):]
    if not lower or not upper:
        return 0.0
    return _median(upper) - _median(lower)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--iters", type=int, default=1, help="measured replay passes")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup passes")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    return parser


def run(args: argparse.Namespace) -> int:
    import os

    import torch
    import torch.distributed as dist

    _DTYPES = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float64": torch.float64,
        "double": torch.float64,
    }

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(args.backend)

    with open(args.trace_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    # process groups from init entries; the world group is reused as-is
    groups = {}
    for entry in entries:
        if entry.get("comms") == "init":
            ranks = entry["global_ranks"]
            if len(ranks) != world_size:
                raise SystemExit(
                    f"trace process group {entry['pg_id']} has {len(ranks)} ranks "
                    f"but the replay world is {world_size}; launch with a "
                    "matching --nproc_per_node"
                )
            groups[entry["pg_id"]] = dist.group.WORLD

    # preallocated buffers per (pg_id, nelems, dtype) and gemm operands per
    # (dim, dtype); allocation must never sit inside the timed loop
    comm_buffers = {}
    gemm_operands = {}
    for entry in entries:
        if entry.get("comms") in ("init", "wait", None):
            if entry.get("compute") == "gemm":
                key = (entry["mm_dim"], entry.get("dtype", "float32"))
                if key not in gemm_operands:
                    dtype = _DTYPES.get(key[1], torch.float32)
                    gemm_operands[key] = (
                        torch.randn(key[0], key[0], device=args.device, dtype=dtype),
                        torch.randn(key[0], key[0], device=args.device, dtype=dtype),
                    )
            continue
        if entry["comms"] != "all_reduce":
            raise SystemExit(
                f"unsupported comm {entry['comms']!r}: this reference replayer "
                "covers the experiment's all_reduce traces; extend it for more"
            )
        key = (entry["pg_id"], entry["in_msg_size"], entry["dtype"])
        if key not in comm_buffers:
            dtype = _DTYPES.get(entry["dtype"], torch.float32)
            comm_buffers[key] = torch.ones(entry["in_msg_size"], device=args.device, dtype=dtype)

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
        result = {
            "schema": "commcanary.rostam.workload_tp8.stdout.v1",
            "rank": rank,
            "world_size": world_size,
            "tokens": args.iters,
            "layers": 1,
            "hidden": 0,
            "gemm_m_rank0": 0,
            "gemm_n": 0,
            "dtype": "trace",
            "msg_sizes_bytes": [],
            "inject_skew": 0.0,
            "timings_us": [round(value, 3) for value in timed],
            "metrics": {
                "median_us": round(_median(timed), 3) if timed else 0.0,
                "iqr_us": round(_iqr(timed), 3) if timed else 0.0,
                "count": len(timed),
            },
        }
        print(json.dumps(result, sort_keys=True), flush=True)

    dist.destroy_process_group()
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
