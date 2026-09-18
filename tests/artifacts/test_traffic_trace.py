from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from jsonschema import Draft202012Validator

from commcanary.artifacts.traffic_trace import (
    requests_due_by,
    require_measured_traffic_trace,
    traffic_trace_from_observation,
    traffic_trace_id,
    validate_traffic_trace,
)
from commcanary.errors import SchemaError
from commcanary.resources import ResourceLimits
from commcanary.services.traffic_synthesis import synthesize_traffic_trace

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "schemas" / "commcanary.traffic_trace.v1.schema.json"
PACKAGED = ROOT / "src" / "commcanary" / "schemas" / "commcanary.traffic_trace.v1.schema.json"

INPUT_DISTRIBUTION = {"median_tokens": 240, "sigma": 0.85, "minimum_tokens": 8, "maximum_tokens": 4096}
OUTPUT_DISTRIBUTION = {"median_tokens": 180, "sigma": 0.95, "minimum_tokens": 1, "maximum_tokens": 2048}


def _trace(**overrides: Any) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "name": "precheck",
        "duration_us": 30_000_000,
        "mean_interarrival_us": 250_000,
        "input_distribution": INPUT_DISTRIBUTION,
        "output_distribution": OUTPUT_DISTRIBUTION,
        "seed": 314159,
    }
    parameters.update(overrides)
    return synthesize_traffic_trace(**parameters)


def test_schema_mirror_is_byte_identical_and_accepts_both_sources() -> None:
    assert PUBLISHED.read_bytes() == PACKAGED.read_bytes()
    schema = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    synthetic = _trace()
    measured = traffic_trace_from_observation(
        name="observed",
        observed=[
            {"arrival_offset_us": 0, "input_length": 12, "output_length": 5},
            {"arrival_offset_us": 900, "input_length": 30, "output_length": 9},
        ],
    )
    assert list(validator.iter_errors(synthetic)) == []
    assert list(validator.iter_errors(measured)) == []


def test_synthesis_is_deterministic_and_seed_separates_the_two_axes() -> None:
    assert _trace()["trace_id"] == _trace()["trace_id"]
    assert _trace(seed=7)["trace_id"] != _trace()["trace_id"]

    # Changing only the arrival rate must not reshuffle the length sequence, or
    # a rate sweep would silently vary two things at once.
    slow = _trace(mean_interarrival_us=500_000)
    fast = _trace(mean_interarrival_us=250_000)
    shared = min(len(slow["requests"]), len(fast["requests"]))
    assert shared > 0
    assert [request["input_length"] for request in slow["requests"][:shared]] == [
        request["input_length"] for request in fast["requests"][:shared]
    ]


def test_deterministic_arrivals_are_evenly_spaced_and_declare_no_seed() -> None:
    trace = _trace(poisson=False)
    assert trace["arrival_model"] == {
        "kind": "deterministic",
        "seed": None,
        "mean_interarrival_us": 250_000,
    }
    offsets = [request["arrival_offset_us"] for request in trace["requests"]]
    assert offsets == [250_000 * (index + 1) for index in range(len(offsets))]


def test_arrivals_stay_inside_the_declared_duration() -> None:
    trace = _trace(duration_us=5_000_000)
    assert trace["summary"]["last_arrival_offset_us"] <= 5_000_000
    assert trace["summary"]["request_count"] == len(trace["requests"])


def test_synthetic_traffic_cannot_qualify_but_measured_traffic_can() -> None:
    with pytest.raises(SchemaError, match="synthetic traffic cannot qualify"):
        require_measured_traffic_trace(_trace())
    require_measured_traffic_trace(
        traffic_trace_from_observation(
            name="observed",
            observed=[{"arrival_offset_us": 0, "input_length": 4, "output_length": 4}],
        )
    )


def test_validation_refuses_tampering_reordering_and_open_fields() -> None:
    trace = _trace()

    dropped = {**trace, "requests": trace["requests"][:-1]}
    with pytest.raises(SchemaError, match="summary disagrees"):
        validate_traffic_trace(dropped)

    retimed = json.loads(json.dumps(trace))
    retimed["requests"][3]["arrival_offset_us"] = 0
    retimed["summary"] = {**retimed["summary"]}
    with pytest.raises(SchemaError, match="arrives before its predecessor"):
        validate_traffic_trace(retimed)

    renamed = json.loads(json.dumps(trace))
    renamed["requests"][2]["request_id"] = "r-999999"
    with pytest.raises(SchemaError, match="must be identified"):
        validate_traffic_trace(renamed)

    with pytest.raises(SchemaError, match="fields are not closed"):
        validate_traffic_trace({**trace, "operator": "anshuman"})

    relabelled = {**trace, "name": "renamed"}
    assert relabelled["trace_id"] != traffic_trace_id(relabelled)
    with pytest.raises(SchemaError, match="trace_id does not match"):
        validate_traffic_trace(relabelled)


def test_replayed_models_must_not_claim_a_generator_that_never_ran() -> None:
    measured = traffic_trace_from_observation(
        name="observed",
        observed=[{"arrival_offset_us": 0, "input_length": 4, "output_length": 4}],
    )
    invented = json.loads(json.dumps(measured))
    invented["arrival_model"]["mean_interarrival_us"] = 1_000
    invented["trace_id"] = traffic_trace_id(invented)
    with pytest.raises(SchemaError, match="replayed arrival_model must not declare"):
        validate_traffic_trace(invented)


def test_request_count_is_bounded_and_an_empty_window_is_refused() -> None:
    with pytest.raises(SchemaError, match="exceeds limit"):
        synthesize_traffic_trace(
            name="too-many",
            duration_us=30_000_000,
            mean_interarrival_us=1_000,
            input_distribution=INPUT_DISTRIBUTION,
            output_distribution=OUTPUT_DISTRIBUTION,
            seed=1,
            poisson=False,
            limits=ResourceLimits(max_traffic_requests=16),
        )
    with pytest.raises(SchemaError, match="produced no requests"):
        synthesize_traffic_trace(
            name="too-short",
            duration_us=1_000,
            mean_interarrival_us=250_000,
            input_distribution=INPUT_DISTRIBUTION,
            output_distribution=OUTPUT_DISTRIBUTION,
            seed=1,
            poisson=False,
        )


def test_requests_due_by_follows_wall_clock_not_an_iteration_counter() -> None:
    trace = _trace(poisson=False, mean_interarrival_us=1_000_000, duration_us=10_000_000)
    assert requests_due_by(trace, 0) == []
    assert len(requests_due_by(trace, 3_000_000)) == 3
    assert len(requests_due_by(trace, 10_000_000)) == trace["summary"]["request_count"]


def _mutate(trace: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Return a trace with fields replaced and its identity resealed.

    Resealing keeps each case pointed at the rule under test; without it every
    mutation would trip the trace_id check first and prove nothing.
    """

    mutated = json.loads(json.dumps(trace))
    mutated.update(overrides)
    mutated["trace_id"] = traffic_trace_id(mutated)
    return mutated


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "commcanary.traffic_trace.v2"}, "format is unsupported"),
        ({"name": "   "}, "name must be a non-empty string"),
        ({"source": "estimated"}, "source is unsupported"),
        ({"requests": []}, "must contain at least one request"),
        ({"requests": ["r-000000"]}, "request 0 must be an object"),
        ({"summary": []}, "summary must be an object"),
        ({"arrival_model": "poisson"}, "arrival_model must be an object"),
        ({"length_model": "lognormal"}, "length_model must be an object"),
    ],
)
def test_closed_field_refusals(overrides: Dict[str, Any], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        validate_traffic_trace(_mutate(_trace(), **overrides))


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"kind": "markov", "seed": 1, "mean_interarrival_us": 10}, "arrival_model kind is unsupported"),
        ({"kind": "deterministic", "seed": 5, "mean_interarrival_us": 10}, "must not declare a seed"),
        ({"kind": "poisson", "seed": -1, "mean_interarrival_us": 10}, "seed must be a non-negative integer"),
        ({"kind": "poisson", "seed": 1, "mean_interarrival_us": 0}, "must be a positive integer"),
    ],
)
def test_arrival_model_refusals(model: Dict[str, Any], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        validate_traffic_trace(_mutate(_trace(), arrival_model=model))


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            {"kind": "pareto", "seed": 1, "input": INPUT_DISTRIBUTION, "output": OUTPUT_DISTRIBUTION},
            "length_model kind is unsupported",
        ),
        (
            {"kind": "lognormal", "seed": 1, "input": "wide", "output": OUTPUT_DISTRIBUTION},
            "length_model input must be an object",
        ),
        (
            {"kind": "replayed", "seed": 3, "input": None, "output": None},
            "replayed length_model must not declare",
        ),
    ],
)
def test_length_model_refusals(model: Dict[str, Any], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        validate_traffic_trace(_mutate(_trace(), length_model=model))


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        ({**INPUT_DISTRIBUTION, "minimum_tokens": 5000}, "minimum_tokens exceeds maximum_tokens"),
        ({**INPUT_DISTRIBUTION, "median_tokens": 1}, "median_tokens is outside"),
        ({**INPUT_DISTRIBUTION, "sigma": "wide"}, "sigma must be a number"),
        ({**INPUT_DISTRIBUTION, "sigma": 0.0}, "sigma must be finite and positive"),
        ({**INPUT_DISTRIBUTION, "median_tokens": 0}, "median_tokens must be a positive integer"),
    ],
)
def test_length_distribution_refusals(distribution: Dict[str, Any], message: str) -> None:
    model = {"kind": "lognormal", "seed": 1, "input": distribution, "output": OUTPUT_DISTRIBUTION}
    with pytest.raises(SchemaError, match=message):
        validate_traffic_trace(_mutate(_trace(), length_model=model))


def test_request_bound_and_empty_observation_are_refused() -> None:
    trace = _trace()
    with pytest.raises(SchemaError, match="request count exceeds limit"):
        validate_traffic_trace(trace, limits=ResourceLimits(max_traffic_requests=2))
    with pytest.raises(SchemaError, match="measured traffic requires at least one"):
        traffic_trace_from_observation(name="empty", observed=[])


def test_resource_constraints_are_reported_as_schema_refusals() -> None:
    with pytest.raises(SchemaError, match="violates JSON resource constraints"):
        validate_traffic_trace(_trace(), limits=ResourceLimits(max_json_items=4))
