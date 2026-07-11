from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import commcanary.compiler as compiler_module
from commcanary.artifacts import canonical_json_bytes
from commcanary.behavior_config import (
    BehaviorConfiguration,
    behavior_replay_arguments,
    parse_behavior_configurations,
    preflight_behavior_ranking_work,
)
from commcanary.compare import (
    ComparisonReasonCode,
    ComparisonThresholdPolicy,
    compare_reports,
    comparison_reason_codes,
)
from commcanary.compiler import (
    BehaviorSearchSizeKey,
    TimingPriorityTier,
    _behavior_search_size_key,
    _grouped_event_summary,
    _important_timing_indices,
    _update_size_metrics,
    compile_trace,
)
from commcanary.errors import SchemaError
from commcanary.operation_identity import OperationIdentity
from commcanary.replay import ReplayAccumulator, _ReplaySampleValues
from commcanary.resources import ResourceLimits

ROOT = Path(__file__).resolve().parents[1]


def _report_fixture() -> dict[str, Any]:
    return json.loads((ROOT / "tests/fixtures/contracts/report.valid.json").read_text(encoding="utf-8"))


def test_public_behavior_configuration_is_immutable_typed_and_detached() -> None:
    raw = [
        {"name": " first ", "iterations": "2", "seed": "3"},
        {"name": "second", "bandwidth_gbps": 99},
    ]
    parsed = parse_behavior_configurations(raw)

    assert isinstance(parsed, tuple)
    assert all(isinstance(config, BehaviorConfiguration) for config in parsed)
    assert parsed[0].name == "first"
    assert parsed[0].iterations == 2
    assert behavior_replay_arguments(parsed[0])["seed"] == 3
    raw[0]["seed"] = 999
    assert parsed[0].seed == 3
    with pytest.raises(FrozenInstanceError):
        parsed[0].seed = 4  # type: ignore[misc]
    with pytest.raises(SchemaError, match="at least two"):
        parse_behavior_configurations([])


def test_behavior_ranking_work_preflights_entire_pairwise_matrix() -> None:
    configs = parse_behavior_configurations([{"name": "a"}, {"name": "b"}, {"name": "c"}])
    work = preflight_behavior_ranking_work(configs, candidate_count=5)
    assert work.configuration_count == 3
    assert work.configuration_pairs == 3
    assert work.metric_count == 4
    assert work.comparisons == 60

    with pytest.raises(SchemaError, match="behavior ranking comparisons=60 exceeds limit=59"):
        preflight_behavior_ranking_work(
            configs,
            candidate_count=5,
            limits=ResourceLimits(max_behavior_ranking_comparisons=59),
        )


def test_operation_identity_has_named_non_flag_projections() -> None:
    operation = {
        "id": "event-7",
        "phase": "decode",
        "op": "point_to_point",
        "bytes": 1024,
        "ranks": [0, 1],
        "group": "pp",
        "sender_rank": 0,
        "receiver_rank": 1,
        "tag": "kv",
        "channel": "pipe",
        "message_sequence": 7,
        "concurrent_groups": 2,
        "capture_session_id": "session-a",
        "collective_id": "collective-7",
    }
    identity = OperationIdentity.from_mapping(operation)

    assert identity.compression_key() == (
        "decode",
        "point_to_point",
        1024,
        (0, 1),
        "pp",
        0,
        1,
        "kv",
        "pipe",
        7,
        2,
        False,
    )
    assert identity.scheduler_ordering_key() == (
        "decode",
        "point_to_point",
        1024,
        "pp",
        (0, 1),
        0,
        1,
        "kv",
        "pipe",
    )
    assert identity.scheduler_resource_label() == "p2p:pp:0->1:channel=pipe:tag=kv:seq=7"
    assert identity.capture_coalescing_key() == ("session-a", "collective-7")
    assert identity.baseline_shape_key()[0] == "decode"
    assert identity.isolated_baseline_shape_key()[0] == "*"
    assert identity.noise_identity([0.0, 1.1234567894], occurrence=9).to_wire() == {
        "phase": "decode",
        "op": "point_to_point",
        "ranks": [0, 1],
        "group": "pp",
        "arrival_offsets_us": [0.0, 1.123456789],
        "occurrence": 9,
        "sender_rank": 0,
        "receiver_rank": 1,
        "message_sequence": 7,
        "tag": "kv",
        "channel": "pipe",
    }

    uncoalesced = OperationIdentity.from_mapping(
        {"id": "event", "op": "all_reduce", "bytes": 4, "ranks": [0], "shard": "rank-0"}
    )
    assert uncoalesced.capture_coalescing_key() == ("uncoalesced", "rank-0", "event")


def test_threshold_policy_matches_legacy_surface_and_codes_stay_on_evaluations() -> None:
    baseline = _report_fixture()
    candidate = copy.deepcopy(baseline)
    candidate["metrics"]["p99_us"] = 20.0
    candidate["metrics"]["max_us"] = 20.0

    legacy = compare_reports(baseline, candidate)
    typed = compare_reports(baseline, candidate, threshold_policy=ComparisonThresholdPolicy())
    legacy.pop("created_at")
    typed.pop("created_at")

    assert typed == legacy
    assert "reason_codes" not in typed
    assert ComparisonReasonCode.OVERALL_P99.value == "overall.p99"
    assert ComparisonReasonCode.OVERALL_P99.value in comparison_reason_codes(typed)
    with pytest.raises(FrozenInstanceError):
        policy = ComparisonThresholdPolicy()
        policy.p99_threshold_pct = 1.0  # type: ignore[misc]
    with pytest.raises(SchemaError, match="p99_threshold_pct must be non-negative"):
        ComparisonThresholdPolicy(p99_threshold_pct=-1.0)


def test_behavior_search_uses_one_effective_size_key_for_uniform_and_refined_limits() -> None:
    uniform = {
        "compiler": {
            "canary_bytes": 100,
            "stored_recursive_timing_records": 4,
            "canary_events": 2,
            "timing_sample_limit": 8,
            "timing_group_count": 3,
        }
    }
    refined = copy.deepcopy(uniform)
    refined["compiler"]["timing_sample_limits_by_group"] = {"1": 2}

    assert _behavior_search_size_key(uniform) == BehaviorSearchSizeKey(100, 4, 2, 24)
    assert _behavior_search_size_key(refined) == BehaviorSearchSizeKey(100, 4, 2, 18)
    assert _behavior_search_size_key(refined) < _behavior_search_size_key(uniform)


def test_named_timing_priority_tiers_preserve_characterized_anchor_order() -> None:
    samples = [
        {
            "gap_us": 10.0,
            "arrival_skew_us": 0.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.2,
            "observed_exposed_us": 5.0,
        },
        {
            "gap_us": 0.0,
            "arrival_skew_us": 1.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.3,
            "observed_exposed_us": 6.0,
        },
        {
            "gap_us": 0.0,
            "arrival_skew_us": 20.0,
            "compute_overlap_us": 5.0,
            "compute_pressure": 0.9,
            "observed_exposed_us": 40.0,
        },
        {
            "gap_us": 5.0,
            "arrival_skew_us": 2.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.2,
            "observed_exposed_us": 7.0,
        },
        {
            "gap_us": 100.0,
            "arrival_skew_us": 3.0,
            "compute_overlap_us": 10.0,
            "compute_pressure": 1.0,
            "observed_exposed_us": 60.0,
        },
        {
            "gap_us": 8.0,
            "arrival_skew_us": 0.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.1,
            "observed_exposed_us": 4.0,
        },
        {
            "gap_us": 8.0,
            "arrival_skew_us": 30.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.8,
            "observed_exposed_us": 50.0,
        },
        {
            "gap_us": 8.0,
            "arrival_skew_us": 0.0,
            "compute_overlap_us": 20.0,
            "compute_pressure": 0.8,
            "observed_exposed_us": 10.0,
        },
        {
            "gap_us": 1.0,
            "arrival_skew_us": 0.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.1,
            "observed_exposed_us": 3.0,
        },
        {
            "gap_us": 50.0,
            "arrival_skew_us": 0.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.1,
            "observed_exposed_us": 55.0,
        },
        {
            "gap_us": 2.0,
            "arrival_skew_us": 5.0,
            "compute_overlap_us": 2.0,
            "compute_pressure": 0.5,
            "observed_exposed_us": 8.0,
        },
        {
            "gap_us": 2.0,
            "arrival_skew_us": 0.0,
            "compute_overlap_us": 0.0,
            "compute_pressure": 0.1,
            "observed_exposed_us": 2.0,
        },
    ]

    assert TimingPriorityTier.OBSERVED_TAIL > TimingPriorityTier.BACKLOG_TRANSITION
    assert _important_timing_indices(samples, 5) == [0, 4, 11]
    assert _important_timing_indices(samples, 8) == [0, 1, 4, 9, 10, 11]


def test_grouped_event_summary_uses_componentwise_medians_for_offsets_and_scalars() -> None:
    samples = [
        {
            "gap_us": 1.0,
            "arrival_offsets_us": [0.0, 0.0],
            "arrival_skew_us": 0.0,
            "compute_before_us": 2.0,
            "compute_overlap_us": 3.0,
            "compute_pressure": 0.2,
        },
        {
            "gap_us": 5.0,
            "arrival_offsets_us": [0.0, 100.0],
            "arrival_skew_us": 100.0,
            "compute_before_us": 4.0,
            "compute_overlap_us": 5.0,
            "compute_pressure": 0.4,
        },
        {
            "gap_us": 9.0,
            "arrival_offsets_us": [0.0, 200.0],
            "arrival_skew_us": 200.0,
            "compute_before_us": 6.0,
            "compute_overlap_us": 7.0,
            "compute_pressure": 0.6,
        },
    ]
    assert _grouped_event_summary(samples) == {
        "gap_us": 5.0,
        "arrival_skew_us": 100.0,
        "arrival_offsets_us": [0.0, 100.0],
        "compute_before_us": 4.0,
        "compute_overlap_us": 5.0,
        "compute_pressure": 0.4,
    }


def test_size_accounting_is_exact_and_nonconvergence_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = {
        "format": "commcanary.trace.v1",
        "events": [{"op": "all_reduce", "bytes": 4, "ranks": [0, 1], "gap_us": 1.0}],
    }
    canary = compile_trace(trace)
    assert canary["compiler"]["canary_bytes"] == len(canonical_json_bytes(canary))

    cycling = {"compiler": {"source_bytes": 10}}

    def fake_size(document: Any) -> int:
        declared = document["compiler"].get("canary_bytes", 0)
        return 11 if declared == 10 else 10

    monkeypatch.setattr(compiler_module, "_json_size", fake_size)
    with pytest.raises(SchemaError, match="did not converge"):
        _update_size_metrics(cycling)


def test_replay_aggregates_share_one_detached_full_precision_sample() -> None:
    raw = {
        "phase": "decode",
        "op": "all_reduce",
        "exposed_us": 1.23456,
        "arrival_skew_us": 0.12345,
        "avg_rank_wait_us": 0.061725,
        "hidden_us": 0.11111,
        "total_us": 1.34567,
    }
    values = _ReplaySampleValues.from_mapping(raw)
    raw["exposed_us"] = 999.0
    assert values.exposed_us == 1.23456
    assert values.to_wire()["exposed_us"] == 1.235

    accumulator = ReplayAccumulator(include_samples=True)
    accumulator.add(values.payload)
    assert accumulator.samples == [values.to_wire()]
    assert accumulator.metrics()["mean_us"] == 1.235
