"""Frozen request traffic for a sustained serving reference workload.

A reference workload is only a reference if the work driving it is fixed. This
artifact carries the exact request sequence: when each request arrives, how
long its prompt is, and how many tokens it must produce. Two runs that bind the
same ``trace_id`` issued the same work, so a ranking difference between them is
attributable to the substrate rather than to the traffic.

Arrival offsets are integer microseconds rather than float seconds. A sustained
run is twenty minutes of arrivals and the offsets are compared, summed, and
content-addressed; integers make those operations exact and keep the canonical
identity stable across producers.

Synthetic traffic is marked as such. It can drive a sensitivity precheck, where
the question is only whether a change class separates beyond noise. It cannot
qualify a canary, for the same reason a synthetic oracle corpus cannot: an
invented request mix is a guess about the workload, and a decision claim made
against a guess is a claim about the guess.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Sequence, Set

from ..errors import SchemaError
from ..formats import TRAFFIC_TRACE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping
from .json_codec import canonical_json_bytes

TRAFFIC_SOURCES = ("synthetic", "measured")
ARRIVAL_KINDS = ("poisson", "deterministic", "replayed")
LENGTH_KINDS = ("lognormal", "replayed")

#: A request identifier is positional and zero padded so lexical order is
#: arrival order; a producer that emits them out of order is rejected rather
#: than silently reordered.
REQUEST_ID_WIDTH = 6


def request_identifier(index: int) -> str:
    """Return the canonical identifier for the request at ``index``."""

    return f"r-{index:0{REQUEST_ID_WIDTH}d}"


def _closed(value: Mapping[str, Any], expected: Set[str], label: str) -> None:
    if set(value) != expected:
        raise SchemaError(f"{label} fields are not closed")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError(f"{label} must be a positive integer")
    return int(value)


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")
    return int(value)


def _finite_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SchemaError(f"{label} must be finite and positive")
    return number


def validate_length_distribution(model: Mapping[str, Any], label: str) -> None:
    _closed(model, {"median_tokens", "sigma", "minimum_tokens", "maximum_tokens"}, label)
    median = _positive_int(model.get("median_tokens"), f"{label} median_tokens")
    _finite_positive_float(model.get("sigma"), f"{label} sigma")
    minimum = _positive_int(model.get("minimum_tokens"), f"{label} minimum_tokens")
    maximum = _positive_int(model.get("maximum_tokens"), f"{label} maximum_tokens")
    if minimum > maximum:
        raise SchemaError(f"{label} minimum_tokens exceeds maximum_tokens")
    if not minimum <= median <= maximum:
        raise SchemaError(f"{label} median_tokens is outside [minimum_tokens, maximum_tokens]")


def _validate_arrival_model(model: Any) -> None:
    if not isinstance(model, Mapping):
        raise SchemaError("traffic trace arrival_model must be an object")
    _closed(model, {"kind", "seed", "mean_interarrival_us"}, "traffic trace arrival_model")
    kind = model.get("kind")
    if kind not in ARRIVAL_KINDS:
        raise SchemaError("traffic trace arrival_model kind is unsupported")
    seed = model.get("seed")
    interval = model.get("mean_interarrival_us")
    if kind == "replayed":
        # A replayed trace took its offsets from an observation. Declaring a
        # generator seed or rate for it would assert a model that never ran.
        if seed is not None or interval is not None:
            raise SchemaError("replayed arrival_model must not declare a seed or mean interarrival")
        return
    if kind == "poisson":
        _non_negative_int(seed, "traffic trace arrival_model seed")
    elif seed is not None:
        raise SchemaError("deterministic arrival_model must not declare a seed")
    _positive_int(interval, "traffic trace arrival_model mean_interarrival_us")


def _validate_length_model(model: Any) -> None:
    if not isinstance(model, Mapping):
        raise SchemaError("traffic trace length_model must be an object")
    _closed(model, {"kind", "seed", "input", "output"}, "traffic trace length_model")
    kind = model.get("kind")
    if kind not in LENGTH_KINDS:
        raise SchemaError("traffic trace length_model kind is unsupported")
    if kind == "replayed":
        if model.get("seed") is not None or model.get("input") is not None or model.get("output") is not None:
            raise SchemaError("replayed length_model must not declare a seed or distributions")
        return
    _non_negative_int(model.get("seed"), "traffic trace length_model seed")
    for field in ("input", "output"):
        distribution = model.get(field)
        if not isinstance(distribution, Mapping):
            raise SchemaError(f"traffic trace length_model {field} must be an object")
        validate_length_distribution(distribution, f"traffic trace length_model {field}")


def summarize_requests(requests: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Return the recomputable summary bound into a traffic trace.

    Every field is an integer count or an exact microsecond offset. Nothing
    here is a rate: a reader that wants requests per second can divide, and a
    stored float would be one more value to keep byte-stable for no gain.
    """

    return {
        "request_count": len(requests),
        "first_arrival_offset_us": int(requests[0]["arrival_offset_us"]),
        "last_arrival_offset_us": int(requests[-1]["arrival_offset_us"]),
        "total_input_tokens": sum(int(request["input_length"]) for request in requests),
        "total_output_tokens": sum(int(request["output_length"]) for request in requests),
    }


def traffic_trace_id(trace: Mapping[str, Any]) -> str:
    """Return the canonical identity of a traffic trace, excluding the field itself."""

    return hashlib.sha256(
        canonical_json_bytes({key: value for key, value in trace.items() if key != "trace_id"})
    ).hexdigest()


def validate_traffic_trace(
    trace: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Validate one frozen request sequence fail-closed."""

    try:
        validate_json_mapping(trace, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"traffic trace violates JSON resource constraints: {exc}") from exc

    _closed(
        trace,
        {"format", "trace_id", "name", "source", "arrival_model", "length_model", "requests", "summary"},
        "traffic trace",
    )
    if trace.get("format") != TRAFFIC_TRACE_FORMAT:
        raise SchemaError("traffic trace format is unsupported")
    name = trace.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SchemaError("traffic trace name must be a non-empty string")
    if trace.get("source") not in TRAFFIC_SOURCES:
        raise SchemaError("traffic trace source is unsupported")
    _validate_arrival_model(trace.get("arrival_model"))
    _validate_length_model(trace.get("length_model"))

    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        raise SchemaError("traffic trace must contain at least one request")
    if len(requests) > limits.max_traffic_requests:
        raise SchemaError(f"traffic trace request count exceeds limit={limits.max_traffic_requests}")

    previous_offset = -1
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise SchemaError(f"traffic trace request {index} must be an object")
        _closed(
            request, {"request_id", "arrival_offset_us", "input_length", "output_length"}, f"traffic request {index}"
        )
        expected_id = request_identifier(index)
        if request.get("request_id") != expected_id:
            raise SchemaError(f"traffic trace request {index} must be identified {expected_id!r}")
        offset = _non_negative_int(request.get("arrival_offset_us"), f"traffic request {index} arrival_offset_us")
        if offset < previous_offset:
            raise SchemaError(f"traffic trace request {index} arrives before its predecessor")
        previous_offset = offset
        _positive_int(request.get("input_length"), f"traffic request {index} input_length")
        _positive_int(request.get("output_length"), f"traffic request {index} output_length")

    summary = trace.get("summary")
    if not isinstance(summary, Mapping):
        raise SchemaError("traffic trace summary must be an object")
    expected_summary = summarize_requests(requests)
    if dict(summary) != expected_summary:
        raise SchemaError("traffic trace summary disagrees with its requests")

    if trace.get("trace_id") != traffic_trace_id(trace):
        raise SchemaError("traffic trace trace_id does not match canonical content")


def require_measured_traffic_trace(trace: Mapping[str, Any]) -> None:
    """Refuse synthetic traffic on a path that issues a qualification claim.

    Mirrors the oracle-corpus rule: a synthetic input can exercise a workflow
    but cannot stand behind a decision claim about a real deployment.
    """

    validate_traffic_trace(trace)
    if trace.get("source") != "measured":
        raise SchemaError("synthetic traffic cannot qualify a canary; it may drive a sensitivity precheck only")


def traffic_trace_from_observation(
    *,
    name: str,
    observed: Sequence[Mapping[str, Any]],
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Dict[str, Any]:
    """Freeze an observed request sequence as measured traffic.

    ``observed`` supplies ``arrival_offset_us``, ``input_length`` and
    ``output_length`` per request in arrival order. Identifiers and the summary
    are assigned here so a producer cannot declare an order the offsets do not
    support.
    """

    if not observed:
        raise SchemaError("measured traffic requires at least one observed request")
    requests: List[Dict[str, Any]] = []
    for index, row in enumerate(observed):
        requests.append(
            {
                "request_id": request_identifier(index),
                "arrival_offset_us": _non_negative_int(
                    row.get("arrival_offset_us"), f"observed request {index} arrival_offset_us"
                ),
                "input_length": _positive_int(row.get("input_length"), f"observed request {index} input_length"),
                "output_length": _positive_int(row.get("output_length"), f"observed request {index} output_length"),
            }
        )
    trace: Dict[str, Any] = {
        "format": TRAFFIC_TRACE_FORMAT,
        "name": name,
        "source": "measured",
        "arrival_model": {"kind": "replayed", "seed": None, "mean_interarrival_us": None},
        "length_model": {"kind": "replayed", "seed": None, "input": None, "output": None},
        "requests": requests,
        "summary": summarize_requests(requests),
    }
    trace["trace_id"] = traffic_trace_id(trace)
    validate_traffic_trace(trace, limits=limits)
    return trace


def requests_due_by(trace: Mapping[str, Any], elapsed_us: int) -> List[Dict[str, Any]]:
    """Return the requests whose arrival offset has passed ``elapsed_us``.

    The driver uses this to feed a sustained run from wall-clock progress
    instead of from an iteration counter, which is what makes the run a load
    test rather than a batch loop.
    """

    _non_negative_int(elapsed_us, "elapsed_us")
    requests = trace["requests"]
    return [dict(request) for request in requests if int(request["arrival_offset_us"]) <= elapsed_us]


__all__ = [
    "ARRIVAL_KINDS",
    "LENGTH_KINDS",
    "REQUEST_ID_WIDTH",
    "TRAFFIC_SOURCES",
    "request_identifier",
    "require_measured_traffic_trace",
    "requests_due_by",
    "summarize_requests",
    "traffic_trace_from_observation",
    "traffic_trace_id",
    "validate_length_distribution",
    "validate_traffic_trace",
]
