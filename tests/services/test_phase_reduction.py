"""Contract tests for phase-representative repetition reduction."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pytest

from commcanary.errors import SchemaError
from commcanary.formats import TRACE_FORMAT
from commcanary.services.phase_reduction import (
    DEFAULT_OVERLAP_BUCKETS,
    REDUCTION_METHOD,
    cluster_iterations,
    declared_boundaries,
    inferred_boundaries,
    phase_representative_reduction,
    rebase_timeline,
    segment_iterations,
)

RANKS = [0, 1, 2, 3]
STEP_GAP_RATIO = 6.0


def event(
    *,
    iteration: Optional[int],
    layer: int,
    start_us: float,
    size: int = 65536,
    overlap: float = 17.0,
    before: float = 26.0,
    pressure: float = 0.64,
    groups: int = 1,
    phase: str = "decode",
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": f"{phase}-{start_us:.0f}-{layer}",
        "op": "all_reduce",
        "bytes": size,
        "dtype": "bfloat16",
        "reduction_op": "sum",
        "ranks": list(RANKS),
        "group": "tp0",
        "phase": phase,
        "start_us": start_us,
        "rank_arrival_us": {str(rank): 2.0 * rank for rank in RANKS},
        "compute_before_us": before * (STEP_GAP_RATIO if layer == 0 else 1.0),
        "compute_overlap_us": overlap,
        "compute_pressure": pressure,
        "concurrent_groups": groups,
    }
    if iteration is not None:
        record["iteration_index"] = iteration
    return record


def trace(*, steps: int = 12, layers: int = 4, declare: bool = True, prefill_every: int = 0) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    now = 0.0
    for step in range(steps):
        prefill = prefill_every > 0 and step % prefill_every == 0
        for layer in range(layers):
            events.append(
                event(
                    iteration=step if declare else None,
                    layer=layer,
                    start_us=now,
                    size=262144 if prefill else 65536,
                    overlap=34.0 if prefill else 17.0,
                    phase="prefill" if prefill else "decode",
                )
            )
            now += 30.0
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "phase-fixture"},
        "system": {"world_size": len(RANKS)},
        "events": events,
    }


def test_declared_segmentation_is_used_when_the_capture_provides_it() -> None:
    document = trace(steps=8, layers=4)
    iterations, source = segment_iterations(document["events"])

    assert source == "declared_iteration_index"
    assert len(iterations) == 8
    assert all(len(iteration.events) == 4 for iteration in iterations)
    assert declared_boundaries(document["events"]) == [0, 4, 8, 12, 16, 20, 24, 28]


def test_undeclared_segmentation_is_refused_rather_than_inferred_from_structure() -> None:
    document = trace(steps=8, layers=4, declare=False)

    with pytest.raises(SchemaError, match="cannot determine the repetition unit"):
        segment_iterations(document["events"])

    with pytest.raises(SchemaError, match="cannot determine the repetition unit"):
        phase_representative_reduction(document)


def test_partially_declared_segmentation_is_refused() -> None:
    document = trace(steps=4, layers=4)
    del document["events"][5]["iteration_index"]

    with pytest.raises(SchemaError, match="partially declared segmentation"):
        declared_boundaries(document["events"])


def test_explicit_period_and_gap_inference_agree_with_the_declaration() -> None:
    document = trace(steps=8, layers=4)
    stripped = [{key: value for key, value in item.items() if key != "iteration_index"} for item in document["events"]]

    declared, _ = segment_iterations(document["events"])
    explicit, explicit_source = segment_iterations(stripped, period=4)
    inferred, inferred_source = segment_iterations(stripped, infer_boundaries=True)

    assert explicit_source == "explicit_period"
    assert inferred_source == "inferred_compute_gap"
    starts = [iteration.start for iteration in declared]
    assert [iteration.start for iteration in explicit] == starts
    assert [iteration.start for iteration in inferred] == starts
    assert inferred_boundaries(stripped, gap_ratio=4.0) == starts


def test_events_without_declared_overlap_are_refused() -> None:
    document = trace(steps=4, layers=4)
    del document["events"][2]["compute_pressure"]

    with pytest.raises(SchemaError, match="refuses to assume zero concurrency"):
        phase_representative_reduction(document)


def test_reduction_keeps_weights_and_shrinks_the_event_stream() -> None:
    document = trace(steps=12, layers=4)
    reduced = phase_representative_reduction(document)
    record = reduced["workload"]["reduction"]

    assert record["method"] == REDUCTION_METHOD
    assert record["segmentation_source"] == "declared_iteration_index"
    assert record["original_iterations"] == 12
    assert record["reduced_iterations"] < record["original_iterations"]
    assert record["reduced_events"] < record["original_events"]
    # Every source iteration is still represented by exactly one weight.
    assert record["weight_total_iterations"] == record["original_iterations"]
    assert sum(row["weight_iterations"] for row in record["representatives"]) == 12
    assert reduced["workload"]["reduction_method"] == REDUCTION_METHOD
    assert "not source-verified" in reduced["workload"]["notes"]


def test_distinct_phases_do_not_collapse_into_one_representative() -> None:
    document = trace(steps=12, layers=4, prefill_every=4)
    iterations, _ = segment_iterations(document["events"])
    clusters = cluster_iterations(iterations, overlap_buckets=DEFAULT_OVERLAP_BUCKETS)

    signatures = {cluster.signature for cluster in clusters}
    assert len(signatures) >= 2, "prefill and decode iterations must not share a signature"
    phases = set()
    for cluster in clusters:
        phases.add(iterations[cluster.representative].events[0]["phase"])
    assert phases == {"decode", "prefill"}


def test_budget_preserves_total_weight_and_keeps_the_tail_phase() -> None:
    """Percentiles are proportions, so the mix has to survive the budget.

    A rare phase that carries the tail must not be rounded away, and the weights
    must still account for every source iteration however few are retained.
    """

    document = trace(steps=12, layers=4, prefill_every=4)
    reduced = phase_representative_reduction(document, max_representatives=2)
    record = reduced["workload"]["reduction"]

    assert record["reduced_iterations"] <= 2
    assert record["weight_total_iterations"] == record["original_iterations"] == 12
    assert any(row["role"] == "tail" for row in record["representatives"]), (
        "the phase carrying the longest iterations must survive the budget"
    )
    retained_phases = {document["events"][row["source_event_start"]]["phase"] for row in record["representatives"]}
    assert "prefill" in retained_phases, "the rare phase must still appear in the artifact"


def test_retained_iterations_are_rebased_onto_a_contiguous_timeline() -> None:
    document = trace(steps=12, layers=4)
    reduced = phase_representative_reduction(document)
    record = reduced["workload"]["reduction"]

    assert record["timeline"] == "rebased_to_per_cluster_median_cadence"
    starts = [item["start_us"] for item in reduced["events"]]
    assert starts[0] == 0.0
    assert starts == sorted(starts)
    # Cadence must close the holes left by dropped iterations rather than
    # preserving absolute offsets, which would present fictitious idle to a
    # replay that models queueing from inter-arrival time.
    assert max(starts) < max(item["start_us"] for item in document["events"])


def test_rebase_timeline_preserves_intra_iteration_offsets() -> None:
    document = trace(steps=4, layers=4)
    iterations, _ = segment_iterations(document["events"])
    rebased = rebase_timeline(iterations, [0, 2], cadence_us=1000.0)

    assert len(rebased) == 8
    first = [item["start_us"] for item in rebased[:4]]
    second = [item["start_us"] for item in rebased[4:]]
    assert first == [0.0, 30.0, 60.0, 90.0]
    assert second == [1000.0, 1030.0, 1060.0, 1090.0]


def test_reduction_is_deterministic() -> None:
    document = trace(steps=12, layers=4, prefill_every=4)
    assert phase_representative_reduction(document) == phase_representative_reduction(document)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_representatives": 0}, "max_representatives must be positive"),
        ({"overlap_buckets": 1}, "overlap_buckets must be at least 2"),
        ({"period": 0}, "iteration period must be positive"),
        ({"period": 10_000}, "iteration period cannot exceed"),
    ],
)
def test_reduction_rejects_out_of_range_arguments(kwargs: Mapping[str, Any], match: str) -> None:
    document = trace(steps=4, layers=4)
    if "period" in kwargs:
        document["events"] = [
            {key: value for key, value in item.items() if key != "iteration_index"} for item in document["events"]
        ]

    with pytest.raises(SchemaError, match=match):
        phase_representative_reduction(document, **kwargs)


def test_empty_trace_and_empty_segmentation_are_refused() -> None:
    document = trace(steps=4, layers=4)
    document["events"] = []

    with pytest.raises(SchemaError, match="cannot reduce an empty trace"):
        phase_representative_reduction(document)
    with pytest.raises(SchemaError, match="cannot segment an empty trace"):
        segment_iterations([])


def test_gap_inference_refuses_a_trace_without_positive_compute() -> None:
    document = trace(steps=4, layers=4, declare=False)
    for item in document["events"]:
        item["compute_before_us"] = 0.0

    with pytest.raises(SchemaError, match="median compute_before_us is not positive"):
        inferred_boundaries(document["events"], gap_ratio=4.0)
