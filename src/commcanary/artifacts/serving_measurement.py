"""One sustained serving run measured the way a serving team decides upgrades.

The existing application measurement is batch shaped: a fixed loop of
``batch_size`` prompts, timed end to end. That answers "how long does one batch
take", which is not the question anyone gates a driver upgrade on. This artifact
answers the question they do ask -- sustained output throughput at a fixed p99
time-to-first-token budget -- and keeps the per-request evidence that produced
it.

Three properties make the number trustworthy rather than merely present.

**The traffic is bound.** ``traffic_trace_id`` names the exact request sequence
that was issued. Two runs comparing throughput are comparing the same work or
they are comparing nothing.

**The window is declared.** A sustained run ramps: the queue fills, caches warm,
clocks boost. Throughput computed over the whole run mixes the ramp into the
steady state and quietly rewards whichever configuration ramps fastest. Only
requests that *arrive* inside ``[window_start_us, window_end_us]`` count, the
window is part of the frozen policy rather than chosen after seeing results,
and the excluded counts are published.

**Inter-token latency is refused where it does not exist.** ITL is the gap
between successive output tokens, so a single-token completion has none. Such
requests are counted in ``single_token_completions`` and excluded from the ITL
distribution rather than contributing a fabricated zero.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Sequence, Set

from ..errors import SchemaError
from ..formats import SERVING_MEASUREMENT_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping
from ..statistics import percentile
from .json_codec import canonical_json_bytes

#: Percentiles published for every latency distribution. Fixed rather than
#: configurable: a policy that can choose its own percentile after seeing the
#: data is not a predeclared policy.
PUBLISHED_QUANTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))

REQUEST_OUTCOMES = ("completed", "failed")


def _closed(value: Mapping[str, Any], expected: Set[str], label: str) -> None:
    if set(value) != expected:
        raise SchemaError(f"{label} fields are not closed")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError(f"{label} must be a positive integer")
    return int(value)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SchemaError(f"{label} must be a lowercase SHA-256")
    return value


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    """Summarize one latency distribution, or declare it empty.

    An empty distribution returns nulls rather than zeros. Zero is a latency;
    "there were no observations" is not, and a downstream gate that cannot tell
    the two apart will pass a run that measured nothing.
    """

    if not values:
        return {"count": 0, "mean_us": None, **{name: None for name, _ in PUBLISHED_QUANTILES}}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_us": math.fsum(ordered) / len(ordered),
        **{name: percentile(ordered, q) for name, q in PUBLISHED_QUANTILES},
    }


def _request_latencies(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    time_to_first_token: List[float] = []
    inter_token: List[float] = []
    end_to_end: List[float] = []
    for record in records:
        if record.get("outcome") != "completed":
            continue
        arrival = float(record["arrival_offset_us"])
        first_token = float(record["first_token_offset_us"])
        completion = float(record["completion_offset_us"])
        output_length = int(record["output_length"])
        time_to_first_token.append(first_token - arrival)
        end_to_end.append(completion - arrival)
        if output_length > 1:
            inter_token.append((completion - first_token) / (output_length - 1))
    return {
        "time_to_first_token_us": time_to_first_token,
        "inter_token_latency_us": inter_token,
        "end_to_end_latency_us": end_to_end,
    }


def summarize_serving_run(
    records: Sequence[Mapping[str, Any]],
    *,
    window_start_us: int,
    window_end_us: int,
) -> Dict[str, Any]:
    """Recompute the decision metric and its supporting distributions.

    ``sustained_output_tokens_per_second`` is the decision metric. It divides
    output tokens produced by *in-window* completed requests by the window's
    own duration, not by the run's, so a configuration cannot improve its score
    by finishing the ramp sooner.
    """

    if window_end_us <= window_start_us:
        raise SchemaError("serving measurement window must be a positive interval")
    in_window = [record for record in records if window_start_us <= int(record["arrival_offset_us"]) <= window_end_us]
    completed = [record for record in in_window if record.get("outcome") == "completed"]
    latencies = _request_latencies(completed)
    output_tokens = sum(int(record["output_length"]) for record in completed)
    window_seconds = (window_end_us - window_start_us) / 1_000_000.0
    single_token = sum(1 for record in completed if int(record["output_length"]) == 1)
    return {
        "requests_in_window": len(in_window),
        "requests_excluded_by_window": len(records) - len(in_window),
        "completed_requests": len(completed),
        "failed_requests": len(in_window) - len(completed),
        "single_token_completions": single_token,
        "output_tokens": output_tokens,
        "window_seconds": window_seconds,
        "sustained_output_tokens_per_second": output_tokens / window_seconds,
        "time_to_first_token_us": _distribution(latencies["time_to_first_token_us"]),
        "inter_token_latency_us": _distribution(latencies["inter_token_latency_us"]),
        "end_to_end_latency_us": _distribution(latencies["end_to_end_latency_us"]),
    }


def serving_measurement_id(measurement: Mapping[str, Any]) -> str:
    """Return the canonical identity of a serving measurement."""

    return hashlib.sha256(
        canonical_json_bytes({key: value for key, value in measurement.items() if key != "measurement_id"})
    ).hexdigest()


def _validate_records(records: Any, *, limits: ResourceLimits) -> List[Mapping[str, Any]]:
    if not isinstance(records, list) or not records:
        raise SchemaError("serving measurement must contain at least one request record")
    if len(records) > limits.max_traffic_requests:
        raise SchemaError(f"serving measurement request count exceeds limit={limits.max_traffic_requests}")
    validated: List[Mapping[str, Any]] = []
    for index, raw in enumerate(records):
        record = _mapping(raw, f"serving request {index}")
        _closed(
            record,
            {
                "request_id",
                "outcome",
                "arrival_offset_us",
                "first_token_offset_us",
                "completion_offset_us",
                "input_length",
                "output_length",
            },
            f"serving request {index}",
        )
        if not isinstance(record.get("request_id"), str) or not record["request_id"]:
            raise SchemaError(f"serving request {index} request_id must be non-empty")
        outcome = record.get("outcome")
        if outcome not in REQUEST_OUTCOMES:
            raise SchemaError(f"serving request {index} outcome is unsupported")
        arrival = _non_negative_int(record.get("arrival_offset_us"), f"serving request {index} arrival_offset_us")
        _positive_int(record.get("input_length"), f"serving request {index} input_length")
        if outcome == "failed":
            # A failed request produced no tokens and has no timings to assert.
            if record.get("first_token_offset_us") is not None or record.get("completion_offset_us") is not None:
                raise SchemaError(f"serving request {index} failed but declares token timings")
            if record.get("output_length") != 0:
                raise SchemaError(f"serving request {index} failed but declares produced output")
            validated.append(record)
            continue
        first_token = _non_negative_int(
            record.get("first_token_offset_us"), f"serving request {index} first_token_offset_us"
        )
        completion = _non_negative_int(
            record.get("completion_offset_us"), f"serving request {index} completion_offset_us"
        )
        _positive_int(record.get("output_length"), f"serving request {index} output_length")
        if first_token < arrival:
            raise SchemaError(f"serving request {index} produced a token before it arrived")
        if completion < first_token:
            raise SchemaError(f"serving request {index} completed before its first token")
        validated.append(record)
    identifiers = [record["request_id"] for record in validated]
    if len(set(identifiers)) != len(identifiers):
        raise SchemaError("serving measurement request identifiers must be unique")
    return validated


def validate_serving_measurement(
    measurement: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Validate one sustained serving run fail-closed."""

    try:
        validate_json_mapping(measurement, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"serving measurement violates JSON resource constraints: {exc}") from exc

    _closed(
        measurement,
        {
            "format",
            "measurement_id",
            "traffic_trace_id",
            "configuration_id",
            "window",
            "requests",
            "summary",
        },
        "serving measurement",
    )
    if measurement.get("format") != SERVING_MEASUREMENT_FORMAT:
        raise SchemaError("serving measurement format is unsupported")
    _sha256(measurement.get("traffic_trace_id"), "serving measurement traffic_trace_id")
    configuration = measurement.get("configuration_id")
    if not isinstance(configuration, str) or not configuration.strip():
        raise SchemaError("serving measurement configuration_id must be a non-empty string")

    window = _mapping(measurement.get("window"), "serving measurement window")
    _closed(window, {"start_us", "end_us"}, "serving measurement window")
    start = _non_negative_int(window.get("start_us"), "serving measurement window start_us")
    end = _non_negative_int(window.get("end_us"), "serving measurement window end_us")
    if end <= start:
        raise SchemaError("serving measurement window must be a positive interval")

    records = _validate_records(measurement.get("requests"), limits=limits)
    expected = summarize_serving_run(records, window_start_us=start, window_end_us=end)
    summary = _mapping(measurement.get("summary"), "serving measurement summary")
    if dict(summary) != expected:
        raise SchemaError("serving measurement summary does not recompute from its request records")

    if measurement.get("measurement_id") != serving_measurement_id(measurement):
        raise SchemaError("serving measurement_id does not match canonical content")


def build_serving_measurement(
    *,
    traffic_trace_id: str,
    configuration_id: str,
    window_start_us: int,
    window_end_us: int,
    records: Sequence[Mapping[str, Any]],
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Dict[str, Any]:
    """Assemble and seal one serving measurement from observed requests."""

    measurement: Dict[str, Any] = {
        "format": SERVING_MEASUREMENT_FORMAT,
        "traffic_trace_id": traffic_trace_id,
        "configuration_id": configuration_id,
        "window": {"start_us": window_start_us, "end_us": window_end_us},
        "requests": [dict(record) for record in records],
        "summary": summarize_serving_run(records, window_start_us=window_start_us, window_end_us=window_end_us),
    }
    measurement["measurement_id"] = serving_measurement_id(measurement)
    validate_serving_measurement(measurement, limits=limits)
    return measurement


def decision_metric(measurement: Mapping[str, Any]) -> float:
    """Return the value a ranking is built from: sustained output throughput.

    A run with no completed request in the window is recorded -- a run that
    failed is evidence, and refusing to store it would delete exactly the
    evidence worth keeping -- but it has no throughput to rank. Its zero is the
    absence of a measurement, not a slow one, so it is refused here rather than
    silently sorted last.
    """

    summary = _mapping(measurement.get("summary"), "serving measurement summary")
    if int(summary.get("completed_requests", 0)) < 1:
        raise SchemaError(
            "serving measurement completed no request in its window; it records a failed run "
            "and cannot supply a decision metric"
        )
    value = summary.get("sustained_output_tokens_per_second")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SchemaError("serving measurement decision metric is not a finite number")
    return float(value)


def meets_latency_budget(measurement: Mapping[str, Any], *, p99_time_to_first_token_us: float) -> bool:
    """Report whether the run stayed inside its declared p99 TTFT budget.

    The decision metric is throughput *at* a latency budget, so a throughput
    number from a run that blew the budget is not comparable and must not be
    ranked against one that held it.
    """

    summary = _mapping(measurement.get("summary"), "serving measurement summary")
    observed = _mapping(summary.get("time_to_first_token_us"), "serving TTFT distribution").get("p99")
    if observed is None:
        raise SchemaError("serving measurement has no observed p99 time to first token")
    return float(observed) <= p99_time_to_first_token_us


__all__ = [
    "PUBLISHED_QUANTILES",
    "REQUEST_OUTCOMES",
    "build_serving_measurement",
    "decision_metric",
    "meets_latency_budget",
    "serving_measurement_id",
    "summarize_serving_run",
    "validate_serving_measurement",
]
