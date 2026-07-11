"""Shared constructed test inputs.

Literal golden artifacts remain in ``tests/fixtures``; this module is only for
programmatically constructed inputs reused across capability suites.
"""

from __future__ import annotations

from commcanary import TRACE_FORMAT


def small_trace():
    ranks = [0, 1, 2, 3]
    events = []
    for index in range(6):
        events.append(
            {
                "id": f"decode-{index}",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 128 * 1024,
                "ranks": ranks,
                "group": "tp0",
                "start_us": index * 40.0,
                "rank_arrival_us": {"0": 0.0, "1": 2.0, "2": 4.0, "3": 10.0 + (index % 2)},
                "compute_overlap_us": 15.0,
            }
        )
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "unit"},
        "events": events,
    }


def adversarial_ranking_trace():
    tail_indices = {10, 30, 50, 70, 90}
    events = []
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
    return {"format": TRACE_FORMAT, "workload": {"name": "ranking-inversion"}, "events": events}


def adversarial_ranking_configs():
    return [
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


def two_group_refinement_trace():
    trace = adversarial_ranking_trace()
    events = list(trace["events"])
    for index in range(80):
        events.append(
            {
                "id": f"quiet-{index}",
                "phase": "quiet",
                "op": "all_reduce",
                "bytes": 2048,
                "ranks": [0, 1],
                "group": "tp",
                "gap_us": 1000.0 + (index % 17) * 3.0,
                "rank_arrival_us": {"0": 0.0, "1": (index % 11) * 0.1},
                "compute_overlap_us": 0.0,
                "compute_pressure": 0.5,
            }
        )
    return {"format": TRACE_FORMAT, "workload": {"name": "two-group-refinement"}, "events": events}
