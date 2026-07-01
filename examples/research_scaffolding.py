from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from commcanary.baselines import (
    frequency_representative_baseline_trace,
    random_sampling_baseline_trace,
)
from commcanary.compiler import compile_trace, synthesize_behavioral_canary, verify_canary_behavior
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, write_json


RANKING_CONFIGS = [
    {
        "name": "isolated-fast-no-overlap",
        "latency_floor_us": 7.0,
        "overlap_efficiency": 0.0,
        "seed": 1,
    },
    {
        "name": "workload-overlap-friendly",
        "latency_floor_us": 8.0,
        "overlap_efficiency": 1.0,
        "seed": 1,
    },
]


def adversarial_decode_trace() -> Dict[str, object]:
    """Synthetic ranking inversion: isolated latency favors A, workload overlap favors B."""

    tail_indices = {10, 30, 50, 70, 90}
    events: List[Dict[str, object]] = []
    for index in range(100):
        tail = index in tail_indices
        events.append(
            {
                "id": f"event-{index}",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "group": "tp",
                "gap_us": 500.0 if (index - 1) in tail_indices else 1.0,
                "rank_arrival_us": {"0": 0.0, "1": 500.0 if tail else 0.0},
                "compute_overlap_us": 10000.0 if tail else 0.0,
                "compute_pressure": 1.0 if tail else 0.5,
            }
        )
    return {
        "format": TRACE_FORMAT,
        "workload": {
            "name": "adversarial-decode-ranking-inversion",
            "notes": "A tiny synthetic decode loop where queue reset gaps and high-overlap tail windows matter.",
        },
        "events": events,
    }


def isolated_collective_trace() -> Dict[str, object]:
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "isolated-collective-baseline"},
        "events": [
            {
                "id": "isolated-0",
                "phase": "microbenchmark",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "group": "tp",
                "gap_us": 1.0,
                "rank_arrival_us": {"0": 0.0, "1": 0.0},
                "compute_overlap_us": 0.0,
                "compute_pressure": 0.5,
            }
        ],
    }


def rank_order(canary: Dict[str, object]) -> List[str]:
    reports = []
    for config in RANKING_CONFIGS:
        config_args = {key: value for key, value in config.items() if key != "name"}
        reports.append((config["name"], replay_canary(canary, backend_label=config["name"], **config_args)))
    return [name for name, report in sorted(reports, key=lambda item: item[1]["metrics"]["median_us"])]


def main() -> None:
    out = Path("out/research_scaffold")
    out.mkdir(parents=True, exist_ok=True)

    workload_trace = adversarial_decode_trace()
    isolated_trace = isolated_collective_trace()
    write_json(str(out / "adversarial_decode.trace.json"), workload_trace)
    write_json(str(out / "isolated_collective.trace.json"), isolated_trace)

    isolated = compile_trace(isolated_trace, require_lossless_timing=True)
    full = compile_trace(workload_trace, timing_sample_limit=128, require_lossless_timing=True)
    too_small = compile_trace(workload_trace, timing_sample_limit=2)
    random_baseline_trace = random_sampling_baseline_trace(workload_trace, sample_count=8, seed=11)
    frequency_baseline_trace = frequency_representative_baseline_trace(workload_trace)
    random_baseline = compile_trace(random_baseline_trace, timing_sample_limit=16)
    frequency_baseline = compile_trace(frequency_baseline_trace, timing_sample_limit=16)
    exact_small = compile_trace(workload_trace, timing_sample_limit=32, require_lossless_timing=True)
    behavior_search = synthesize_behavioral_canary(
        workload_trace,
        min_timing_sample_limit=2,
        max_timing_sample_limit=32,
        behavior_configurations=RANKING_CONFIGS,
        ranking_tie_tolerance_us=0.0,
    )

    write_json(str(out / "isolated.canary.json"), isolated)
    write_json(str(out / "full_upper_bound.canary.json"), full)
    write_json(str(out / "too_small.canary.json"), too_small)
    write_json(str(out / "random_sampling_baseline.trace.json"), random_baseline_trace)
    write_json(str(out / "frequency_representative_baseline.trace.json"), frequency_baseline_trace)
    write_json(str(out / "random_sampling_baseline.canary.json"), random_baseline)
    write_json(str(out / "frequency_representative_baseline.canary.json"), frequency_baseline)
    write_json(str(out / "verified_small.canary.json"), exact_small)
    write_json(str(out / "behavior_search.canary.json"), behavior_search)

    lossy_verification = verify_canary_behavior(
        workload_trace,
        too_small,
        configurations=RANKING_CONFIGS,
        relative_tolerance_pct=1000.0,
        absolute_tolerance_us=1000.0,
        hidden_tolerance_points=100.0,
        tail_recall_threshold=0.0,
        ranking_tie_tolerance_us=0.0,
    )
    random_baseline_verification = verify_canary_behavior(
        workload_trace,
        random_baseline,
        configurations=RANKING_CONFIGS,
        relative_tolerance_pct=1000.0,
        absolute_tolerance_us=1000.0,
        hidden_tolerance_points=100.0,
        tail_recall_threshold=0.0,
        ranking_tie_tolerance_us=0.0,
    )
    frequency_baseline_verification = verify_canary_behavior(
        workload_trace,
        frequency_baseline,
        configurations=RANKING_CONFIGS,
        relative_tolerance_pct=1000.0,
        absolute_tolerance_us=1000.0,
        hidden_tolerance_points=100.0,
        tail_recall_threshold=0.0,
        ranking_tie_tolerance_us=0.0,
    )
    exact_verification = verify_canary_behavior(
        workload_trace,
        exact_small,
        configurations=RANKING_CONFIGS,
        ranking_tie_tolerance_us=0.0,
    )
    behavior_search_verification = verify_canary_behavior(
        workload_trace,
        behavior_search,
        configurations=RANKING_CONFIGS,
        ranking_tie_tolerance_us=0.0,
    )
    write_json(str(out / "too_small.behavior.json"), lossy_verification)
    write_json(str(out / "random_sampling_baseline.behavior.json"), random_baseline_verification)
    write_json(str(out / "frequency_representative_baseline.behavior.json"), frequency_baseline_verification)
    write_json(str(out / "verified_small.behavior.json"), exact_verification)
    write_json(str(out / "behavior_search.behavior.json"), behavior_search_verification)

    print("isolated ranking:", " > ".join(rank_order(isolated)))
    print("full workload ranking:", " > ".join(rank_order(full)))
    print("too-small canary status:", lossy_verification["status"], lossy_verification["configuration_ranking_status"])
    print(
        "random baseline status:",
        random_baseline_verification["status"],
        random_baseline_verification["source_verified_status"],
        random_baseline_verification["configuration_ranking_status"],
    )
    print(
        "frequency baseline status:",
        frequency_baseline_verification["status"],
        frequency_baseline_verification["source_verified_status"],
        frequency_baseline_verification["configuration_ranking_status"],
    )
    print("verified canary status:", exact_verification["status"], exact_verification["configuration_ranking_status"])
    print(
        "behavior-search canary:",
        behavior_search["compiler"]["behavior_search"]["selected_timing_sample_limit"],
        "samples,",
        behavior_search_verification["status"],
        behavior_search_verification["configuration_ranking_status"],
    )
    print(out)


if __name__ == "__main__":
    main()
