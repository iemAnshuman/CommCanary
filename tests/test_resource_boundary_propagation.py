from __future__ import annotations

import copy
import dataclasses
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from commcanary.capture import TraceRecorder, merge_trace_shards
from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary, verify_report_against_canary
from commcanary.resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, validate_json_mapping
from commcanary.schema import SchemaError, replay_protocol_sha256, validate_comparison, validate_report
from tests.builders import small_trace


def _json_item_count(value: Any) -> int:
    if isinstance(value, dict):
        return len(value) + sum(_json_item_count(child) for child in value.values())
    if isinstance(value, list):
        return len(value) + sum(_json_item_count(child) for child in value)
    return 0


def _canary_and_report() -> tuple[dict[str, Any], dict[str, Any]]:
    canary = compile_trace(small_trace())
    return canary, replay_canary(canary, seed=3)


def _record_one(recorder: TraceRecorder, *, metadata: dict[str, Any] | None = None) -> None:
    recorder.record_collective(
        op="all_reduce",
        bytes=16,
        ranks=[0],
        rank_arrival_us={"0": 0.0},
        metadata=metadata,
    )


def test_in_memory_json_preflight_requires_an_object_root() -> None:
    with pytest.raises(JsonResourceError, match="root must be an object"):
        validate_json_mapping([])  # type: ignore[arg-type]


def test_report_json_preflight_precedes_protocol_hash_allocation() -> None:
    _, report = _canary_and_report()
    report["workload"]["oversized"] = "x" * 257
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_string_bytes=256)

    with mock.patch(
        "commcanary.artifacts.report.replay_protocol_sha256",
        side_effect=AssertionError("hashing must not begin"),
    ) as protocol_hash:
        with pytest.raises(SchemaError, match="max_json_string_bytes=256"):
            validate_report(report, limits=limits)
    protocol_hash.assert_not_called()


def test_protocol_hash_preflights_in_memory_extensions() -> None:
    _, report = _canary_and_report()
    protocol = copy.deepcopy(report["replay_protocol"])
    protocol["oversized"] = "x" * 257
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_string_bytes=256)

    with mock.patch(
        "commcanary.artifacts.wire.canonical_json_bytes",
        side_effect=AssertionError("encoding must not begin"),
    ) as encode:
        with pytest.raises(SchemaError, match="max_json_string_bytes=256"):
            replay_protocol_sha256(protocol, limits=limits)
    encode.assert_not_called()


def test_report_replay_count_has_an_exact_resource_boundary() -> None:
    _, report = _canary_and_report()
    event_count = report["metrics"]["count"]

    validate_report(
        report,
        limits=dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_replay_events=event_count),
    )
    with pytest.raises(SchemaError, match=rf"report replay events={event_count} exceeds limit={event_count - 1}"):
        validate_report(
            report,
            limits=dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_replay_events=event_count - 1),
        )


def test_compare_and_report_verification_forward_the_same_replay_budget() -> None:
    canary, report = _canary_and_report()
    event_count = report["metrics"]["count"]
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_replay_events=event_count - 1)

    with mock.patch(
        "commcanary.comparison.core._breakdown_deltas",
        side_effect=AssertionError("comparison expansion must not begin"),
    ) as breakdown:
        with pytest.raises(SchemaError, match="report replay events"):
            compare_reports(report, copy.deepcopy(report), limits=limits)
    breakdown.assert_not_called()

    with mock.patch(
        "commcanary.verification.report.replay_canary",
        side_effect=AssertionError("verification replay must not begin"),
    ) as replay:
        with pytest.raises(SchemaError, match="report replay events"):
            verify_report_against_canary(report, canary, limits=limits)
    replay.assert_not_called()


def test_comparison_item_boundary_precedes_policy_evaluation_allocation() -> None:
    _, report = _canary_and_report()
    comparison = compare_reports(report, copy.deepcopy(report))
    base_items = _json_item_count(comparison)
    validate_comparison(
        comparison,
        limits=dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_items=base_items),
    )

    oversized = copy.deepcopy(comparison)
    oversized["extension"] = {"items": [None, None, None]}
    oversized_items = _json_item_count(oversized)
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_items=oversized_items - 1)
    with mock.patch(
        "commcanary.artifacts.comparison._comparison_policy_evaluations",
        side_effect=AssertionError("policy evaluation must not begin"),
    ) as evaluate:
        with pytest.raises(SchemaError, match=rf"max_json_items={oversized_items - 1}"):
            validate_comparison(oversized, limits=limits)
    evaluate.assert_not_called()


def test_recorder_rejects_oversized_metadata_before_copying_or_appending() -> None:
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_items=2)
    with tempfile.TemporaryDirectory() as tmp:
        recorder = TraceRecorder(str(Path(tmp) / "trace.json"), limits=limits)
        try:
            with mock.patch(
                "commcanary.adapters.capture.copy.deepcopy",
                side_effect=AssertionError("metadata copy must not begin"),
            ) as deepcopy:
                with pytest.raises(SchemaError, match="max_json_items=2"):
                    _record_one(recorder, metadata={"items": [1, 2]})
            deepcopy.assert_not_called()
            assert recorder.events == []
        finally:
            recorder.close()


def test_recorder_rejects_the_event_after_its_exact_capacity_before_metadata_copy() -> None:
    limits = dataclasses.replace(
        DEFAULT_RESOURCE_LIMITS,
        max_capture_events=2,
        max_stored_events=2,
    )
    with tempfile.TemporaryDirectory() as tmp:
        recorder = TraceRecorder(str(Path(tmp) / "trace.json"), limits=limits)
        try:
            _record_one(recorder)
            _record_one(recorder)
            recorder.save()
            assert len(recorder.events) == 2

            with mock.patch(
                "commcanary.adapters.capture.copy.deepcopy",
                side_effect=AssertionError("metadata copy must not begin"),
            ) as deepcopy:
                with pytest.raises(SchemaError, match="trace recorder events=3 exceeds limit=2"):
                    _record_one(recorder, metadata={"safe": True})
            deepcopy.assert_not_called()
            assert len(recorder.events) == 2
        finally:
            recorder.close()


def test_recorder_constructor_preflights_metadata_before_claiming_output() -> None:
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_items=2)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "trace.json"
        with mock.patch(
            "commcanary.adapters.capture.copy.deepcopy",
            side_effect=AssertionError("workload copy must not begin"),
        ) as deepcopy:
            with pytest.raises(SchemaError, match="max_json_items=2"):
                TraceRecorder(
                    str(output),
                    workload={"items": [1, 2]},
                    limits=limits,
                )
        deepcopy.assert_not_called()
        assert not (Path(tmp) / ".trace.json.commcanary.lock").exists()


def test_capture_shard_discovery_stops_at_limit_plus_one() -> None:
    def discovered_paths(pattern: str) -> Any:
        yield f"{pattern}-one"
        yield f"{pattern}-two"
        raise AssertionError("discovery must stop once the limit is exceeded")

    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_capture_shards=1)
    with mock.patch(
        "commcanary.adapters.capture_merge.glob.iglob",
        side_effect=discovered_paths,
    ) as discover:
        with pytest.raises(SchemaError, match="capture shards=2 exceeds limit=1"):
            merge_trace_shards("unused", workload_name="bounded", limits=limits)
    assert discover.call_count == 1
