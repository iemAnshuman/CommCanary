"""Local, simulator-only de-risking probe for phase-representative reduction.

This does **not** produce physical evidence.  Replay here is the deterministic
simulator, so every number below is the model checking its own arithmetic.  Its
only job is to answer, before any node-hours are spent, whether repetition
reduction can deliver a large compression ratio while preserving pairwise
configuration rankings on a serving-shaped trace.

Run:  .venv/bin/python -m experiments.phase_reduction_local
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from commcanary.behavior_config import (
    behavior_replay_arguments,
    parse_behavior_configurations,
)
from commcanary.formats import TRACE_FORMAT
from commcanary.replay.core import replay_canary
from commcanary.services._ranking import RANKING_METRICS, ranking_relation
from commcanary.services.compile import compile_trace
from commcanary.services.phase_reduction import phase_representative_reduction

RANKS = list(range(4))
STEP_BOUNDARY_GAP_RATIO = 6.0


def _event(
    event_id: int,
    *,
    iteration_index: int,
    phase: str,
    op: str,
    size: int,
    start_us: float,
    skew_us: float,
    compute_before_us: float,
    compute_overlap_us: float,
    compute_pressure: float,
    concurrent_groups: int,
) -> Dict[str, Any]:
    last = len(RANKS) - 1
    return {
        "id": f"{phase}-{event_id:06d}",
        "iteration_index": iteration_index,
        "op": op,
        "bytes": size,
        "dtype": "bfloat16",
        "reduction_op": "sum",
        "ranks": list(RANKS),
        "group": "tp0",
        "phase": phase,
        "start_us": start_us,
        "rank_arrival_us": {str(rank): skew_us * rank / last for rank in RANKS},
        "compute_before_us": compute_before_us,
        "compute_overlap_us": compute_overlap_us,
        "compute_pressure": compute_pressure,
        "concurrent_groups": concurrent_groups,
    }


def serving_trace(*, steps: int, layers: int, prefill_every: int, seed: int) -> Dict[str, Any]:
    """A serving-shaped trace: repeated decode steps punctuated by prefills.

    Jitter is applied per event so that no two iterations are byte-identical.
    Reduction therefore has to cluster on bucketed structure, not on equality,
    which is the situation a captured trace actually presents.
    """

    rng = random.Random(seed)
    events: List[Dict[str, Any]] = []
    now = 0.0
    event_id = 0
    for step in range(steps):
        prefill = prefill_every > 0 and step % prefill_every == 0
        for layer in range(layers):
            if prefill:
                size, before, overlap, pressure, groups = 256 * 1024, 58.0, 34.0, 0.81, 2
            else:
                size, before, overlap, pressure, groups = 64 * 1024, 26.0, 17.0, 0.64, 1
            # A step boundary carries work no layer performs -- scheduling,
            # sampling, detokenisation -- so the first collective of a step sees
            # a materially larger preceding compute gap than an interior layer.
            if layer == 0:
                before = before * STEP_BOUNDARY_GAP_RATIO
            events.append(
                _event(
                    event_id,
                    iteration_index=step,
                    phase="prefill" if prefill else "decode",
                    op="all_reduce",
                    size=size,
                    start_us=now,
                    skew_us=8.0 + rng.uniform(-1.0, 1.2),
                    compute_before_us=before + rng.uniform(-2.0, 2.0),
                    compute_overlap_us=overlap + rng.uniform(-2.0, 2.5),
                    compute_pressure=pressure,
                    concurrent_groups=groups,
                )
            )
            event_id += 1
            now += before + rng.uniform(-3.0, 4.0)
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "serving-shaped-probe", "notes": "synthetic; simulator-only probe"},
        "system": {"world_size": len(RANKS)},
        "events": events,
    }


def ranking_relations(
    trace: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[str, str, str], str]:
    """Pairwise ranking relations, computed the way the ddmin oracle computes them."""

    canary = compile_trace(trace, timing_sample_limit=128)
    metrics: Dict[str, Mapping[str, Any]] = {}
    for config in configs:
        name = config["name"]
        report = replay_canary(canary, backend_label=name, **behavior_replay_arguments(config))
        metrics[name] = report["metrics"]
    names = sorted(metrics)
    relations: Dict[Tuple[str, str, str], str] = {}
    for metric in RANKING_METRICS:
        for left_index in range(len(names)):
            for right_index in range(left_index + 1, len(names)):
                left, right = names[left_index], names[right_index]
                relations[(metric, left, right)] = ranking_relation(
                    float(metrics[left].get(metric, 0.0)),
                    float(metrics[right].get(metric, 0.0)),
                    0.001,
                )
    return relations


def _strip_declaration(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Same trace with the producer's segmentation removed."""

    return {
        **trace,
        "events": [
            {key: value for key, value in event.items() if key != "iteration_index"} for event in trace["events"]
        ],
    }


def _run(label: str, trace: Mapping[str, Any], configs: Sequence[Mapping[str, Any]], **kwargs: Any) -> bool:
    started = time.monotonic()
    reduced = phase_representative_reduction(trace, **kwargs)
    elapsed = time.monotonic() - started
    record = reduced["workload"]["reduction"]

    source_relations = ranking_relations(trace, configs)
    reduced_relations = ranking_relations(reduced, configs)
    agreed = sum(1 for key, value in source_relations.items() if reduced_relations.get(key) == value)
    total = len(source_relations)

    print(f"--- {label} " + "-" * max(0, 62 - len(label)))
    print(f"  segmentation source    {record['segmentation_source']}")
    print(
        f"  events                 {record['original_events']} -> {record['reduced_events']}"
        f"   ({record['original_events'] / max(record['reduced_events'], 1):.1f}x)"
    )
    print(f"  iterations             {record['original_iterations']} -> {record['reduced_iterations']}")
    print(
        f"  weights sum to         {record['weight_total_iterations']} iterations"
        f"   (source has {record['original_iterations']})"
    )
    print(f"  folded clusters        {len(record['folded_clusters'])}")
    print(f"  reduction wall time    {elapsed * 1000:.1f} ms")
    print(f"  pairwise decisions     {agreed}/{total} preserved   ({100.0 * agreed / max(total, 1):.1f}%)")
    print(f"  verdict                {'PRESERVED' if agreed == total else 'DIVERGED'}")
    print()
    return agreed == total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--prefill-every", type=int, default=32)
    parser.add_argument("--max-representatives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    trace = serving_trace(
        steps=args.steps,
        layers=args.layers,
        prefill_every=args.prefill_every,
        seed=args.seed,
    )
    configs = parse_behavior_configurations(None)
    stripped = _strip_declaration(trace)

    print("=" * 68)
    print("phase-representative reduction - SIMULATOR ONLY, not physical evidence")
    print("=" * 68)
    print(f"workload    {args.steps} steps x {args.layers} layers, prefill every {args.prefill_every}")
    print(f"configs     {len(configs)}: {', '.join(sorted(c['name'] for c in configs))}")
    print(f"metrics     {len(RANKING_METRICS)}: {', '.join(RANKING_METRICS)}")
    print()

    results = []
    results.append(
        _run(
            "declared segmentation (producer emits iteration_index)",
            trace,
            configs,
            max_representatives=args.max_representatives,
        )
    )
    results.append(
        _run(
            "inferred segmentation (compute-gap heuristic)",
            stripped,
            configs,
            infer_boundaries=True,
            max_representatives=args.max_representatives,
        )
    )
    results.append(
        _run(
            "explicit period (operator supplies layers-per-step)",
            stripped,
            configs,
            period=args.layers,
            max_representatives=args.max_representatives,
        )
    )

    print("--- refusal path " + "-" * 46)
    try:
        phase_representative_reduction(stripped)
    except Exception as exc:  # noqa: BLE001 - the refusal is the assertion
        print(f"  undeclared segmentation refused: {type(exc).__name__}")
        print(f"  {str(exc)[:150]}")
    else:
        print("  FAILED: undeclared segmentation was not refused")
        return 1
    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
