"""Shared constructed test inputs.

Literal golden artifacts remain in ``tests/fixtures``; this module is only for
programmatically constructed inputs reused across capability suites.
"""

from __future__ import annotations

from commcanary import TRACE_FORMAT
from commcanary.artifacts import qualification_policy_sha256
from commcanary.formats import QUALIFICATION_POLICY_FORMAT


def qualification_policy():
    """Return an explicit deterministic policy for qualification tests."""

    policy = {
        "format": QUALIFICATION_POLICY_FORMAT,
        "primary_metric": "program_median_us",
        "repetitions": {"warmup": 5, "minimum_measured": 20},
        "regression": {
            "relative_threshold_pct": 5.0,
            "absolute_threshold_us": 2.0,
            "threshold_combination": "larger_of_absolute_or_relative",
        },
        "uncertainty": {
            "method": "percentile_bootstrap_median_difference",
            "confidence": 0.95,
            "bootstrap_resamples": 1000,
            "seed": 7411,
        },
        "noise": {"max_relative_iqr_pct": 20.0},
        "environment": {
            "max_clock_variation_pct": 3.0,
            "require_same_gpu_count": True,
            "require_same_topology_class": True,
        },
        "outcome_policy": {
            "incomplete_measurement": "inconclusive",
            "unstable_measurement": "inconclusive",
            "environment_mismatch": "incomparable",
            "baseline_correctness_failure": "incomparable",
            "candidate_correctness_failure": "fail",
            "confidence_interval_crosses_threshold": "inconclusive",
        },
    }
    policy["policy_id"] = qualification_policy_sha256(policy)
    return policy


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


def qualification_trace():
    """Return a small Kineto-derived trace with exact per-rank compute work."""

    trace = small_trace()
    trace["workload"].update(
        {
            "import_source": "pytorch-kineto",
            "skipped_empty_events": 0,
        }
    )
    trace["system"] = {"source_format": "pytorch-kineto"}
    for event_index, event in enumerate(trace["events"]):
        event["dtype"] = "float32"
        event["reduction_op"] = "sum"
        nelems = event["bytes"] // 4
        event["metadata"] = {
            "kineto_in_msg_nelems": nelems,
            "kineto_out_msg_nelems": nelems,
            "kineto_in_split_sizes": [],
            "kineto_out_split_sizes": [],
            "kineto_message_shape_status": "derived",
            "kineto_message_shape_method": "record-param-comms-in-out-nelems.v1",
            "kineto_compute_recipe_status": "derived",
            "kineto_compute_recipe_method": "explicit-wait-linked-contiguous-gemm.v1",
        }
        event["compute_recipe_by_rank"] = {
            str(rank): [
                {
                    "op": "gemm",
                    "dtype": "bfloat16",
                    "m": rank + 2,
                    "n": 8,
                    "k": 8,
                    "source_kernel_count": 1,
                    "source_kernel_duration_us": round(
                        1.0 + rank * 0.1 + event_index * 0.01,
                        3,
                    ),
                }
            ]
            for rank in event["ranks"]
        }
    return trace


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
