from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from jsonschema import Draft202012Validator

from commcanary.artifacts.serving_measurement import (
    build_serving_measurement,
    decision_metric,
    meets_latency_budget,
    serving_measurement_id,
    summarize_serving_run,
    validate_serving_measurement,
)
from commcanary.errors import SchemaError

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "schemas" / "commcanary.serving_measurement.v1.schema.json"
PACKAGED = ROOT / "src" / "commcanary" / "schemas" / "commcanary.serving_measurement.v1.schema.json"

TRACE_ID = "a" * 64


def _records(count: int = 20, *, output_length: int = 120) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index in range(count):
        arrival = index * 100_000
        first_token = arrival + 40_000
        records.append(
            {
                "request_id": f"r-{index:06d}",
                "outcome": "completed",
                "arrival_offset_us": arrival,
                "first_token_offset_us": first_token,
                "completion_offset_us": first_token + 300_000,
                "input_length": 200,
                "output_length": output_length,
            }
        )
    return records


def _measurement(**overrides: Any) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "traffic_trace_id": TRACE_ID,
        "configuration_id": "nccl-2.20.5-tree-ll",
        "window_start_us": 500_000,
        "window_end_us": 2_000_000,
        "records": _records(),
    }
    parameters.update(overrides)
    return build_serving_measurement(**parameters)


def test_schema_mirror_is_byte_identical_and_accepts_a_built_measurement() -> None:
    assert PUBLISHED.read_bytes() == PACKAGED.read_bytes()
    schema = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(_measurement())) == []


def test_throughput_uses_the_window_not_the_run() -> None:
    # 15 arrivals fall in a 1.5 s window at 120 output tokens each.
    summary = _measurement()["summary"]
    assert summary["requests_in_window"] == 15
    assert summary["requests_excluded_by_window"] == 5
    assert summary["window_seconds"] == pytest.approx(1.5)
    assert summary["sustained_output_tokens_per_second"] == pytest.approx(15 * 120 / 1.5)


def test_a_faster_ramp_cannot_inflate_the_decision_metric() -> None:
    """Requests outside the window contribute nothing, however fast they were.

    This is the property the window exists for: a configuration that warms up
    sooner produces more early completions, and counting them would score the
    ramp rather than the steady state.
    """

    baseline = _measurement()
    early = _records()
    for record in early[:5]:
        record["completion_offset_us"] = record["first_token_offset_us"] + 1_000
    assert decision_metric(_measurement(records=early)) == decision_metric(baseline)


def test_failed_requests_are_counted_without_inventing_timings() -> None:
    records = _records()
    records.append(
        {
            "request_id": "r-000099",
            "outcome": "failed",
            "arrival_offset_us": 900_000,
            "first_token_offset_us": None,
            "completion_offset_us": None,
            "input_length": 200,
            "output_length": 0,
        }
    )
    summary = _measurement(records=records)["summary"]
    assert summary["failed_requests"] == 1
    assert summary["completed_requests"] == 15
    # A failure contributes no tokens and no latency observation.
    assert summary["output_tokens"] == 15 * 120
    assert summary["time_to_first_token_us"]["count"] == 15


def test_single_token_completions_are_excluded_from_inter_token_latency() -> None:
    summary = _measurement(records=_records(output_length=1))["summary"]
    assert summary["single_token_completions"] == 15
    # There is no gap between successive tokens when there is one token.
    assert summary["inter_token_latency_us"] == {
        "count": 0,
        "mean_us": None,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    assert summary["time_to_first_token_us"]["count"] == 15


def test_latency_budget_gates_the_throughput_number() -> None:
    measurement = _measurement()
    observed = measurement["summary"]["time_to_first_token_us"]["p99"]
    assert meets_latency_budget(measurement, p99_time_to_first_token_us=observed)
    assert not meets_latency_budget(measurement, p99_time_to_first_token_us=observed - 1.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(traffic_trace_id="not-a-digest"), "traffic_trace_id must be a lowercase SHA-256"),
        (lambda m: m.update(configuration_id="  "), "configuration_id must be a non-empty string"),
        (lambda m: m.update(window={"start_us": 10, "end_us": 10}), "window must be a positive interval"),
        (lambda m: m.update(format="commcanary.serving_measurement.v2"), "format is unsupported"),
        (lambda m: m.update(operator="anshuman"), "serving measurement fields are not closed"),
    ],
)
def test_measurement_level_refusals(mutate: Any, message: str) -> None:
    measurement = _measurement()
    mutate(measurement)
    measurement["measurement_id"] = serving_measurement_id(measurement)
    with pytest.raises(SchemaError, match=message):
        validate_serving_measurement(measurement)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r: r.update(first_token_offset_us=r["arrival_offset_us"] - 1),
            "produced a token before it arrived",
        ),
        (
            lambda r: r.update(completion_offset_us=r["first_token_offset_us"] - 1),
            "completed before its first token",
        ),
        (lambda r: r.update(outcome="cancelled"), "outcome is unsupported"),
        (lambda r: r.update(output_length=0), "output_length must be a positive integer"),
        (lambda r: r.update(request_id=""), "request_id must be non-empty"),
    ],
)
def test_request_level_refusals(mutate: Any, message: str) -> None:
    records = _records()
    mutate(records[7])
    with pytest.raises(SchemaError, match=message):
        build_serving_measurement(
            traffic_trace_id=TRACE_ID,
            configuration_id="c",
            window_start_us=0,
            window_end_us=2_000_000,
            records=records,
        )


def test_a_failed_request_may_not_declare_tokens_or_timings() -> None:
    records = _records()
    records[3].update(outcome="failed", output_length=0)
    with pytest.raises(SchemaError, match="failed but declares token timings"):
        build_serving_measurement(
            traffic_trace_id=TRACE_ID,
            configuration_id="c",
            window_start_us=0,
            window_end_us=2_000_000,
            records=records,
        )


def test_duplicate_identifiers_are_refused_but_an_empty_window_is_recorded() -> None:
    records = _records()
    records[4]["request_id"] = records[3]["request_id"]
    with pytest.raises(SchemaError, match="request identifiers must be unique"):
        build_serving_measurement(
            traffic_trace_id=TRACE_ID,
            configuration_id="c",
            window_start_us=0,
            window_end_us=2_000_000,
            records=records,
        )
    # An empty window records a run that produced nothing. That is kept as
    # evidence; what it cannot do is supply a number to rank.
    empty = build_serving_measurement(
        traffic_trace_id=TRACE_ID,
        configuration_id="c",
        window_start_us=5_000_000,
        window_end_us=6_000_000,
        records=_records(),
    )
    assert empty["summary"]["completed_requests"] == 0
    validate_serving_measurement(empty)
    with pytest.raises(SchemaError, match="cannot supply a decision metric"):
        decision_metric(empty)


def test_summary_must_recompute_from_the_records_it_claims() -> None:
    measurement = _measurement()
    measurement["summary"] = {**measurement["summary"], "output_tokens": 1}
    measurement["measurement_id"] = serving_measurement_id(measurement)
    with pytest.raises(SchemaError, match="summary does not recompute"):
        validate_serving_measurement(measurement)

    resealed = _measurement()
    resealed["configuration_id"] = "renamed"
    with pytest.raises(SchemaError, match="measurement_id does not match"):
        validate_serving_measurement(resealed)


def test_summarize_refuses_an_inverted_window() -> None:
    with pytest.raises(SchemaError, match="window must be a positive interval"):
        summarize_serving_run(_records(), window_start_us=10, window_end_us=5)
