"""Trace artifact contract validation."""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, require_within
from .wire import (
    MAX_TIME_US,
    as_float,
    as_int,
    normalize_arrival_offsets,
    normalize_ranks,
    require_format,
    require_optional_mapping,
    validate_arrival_keys,
    validate_nonempty_string,
    validate_op,
    validate_point_to_point_metadata,
    validate_skew_matches_offsets,
)


def validate_trace(
    trace: Mapping[str, Any],
    *,
    allow_partial_arrivals: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    require_format(trace, TRACE_FORMAT, "trace")
    require_optional_mapping(trace, "workload", "trace")
    require_optional_mapping(trace, "system", "trace")
    events = trace.get("events")
    if not isinstance(events, list):
        raise SchemaError("trace must contain an 'events' list")
    try:
        require_within(
            len(events),
            limits.max_stored_events,
            label="stored trace events",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise SchemaError(f"trace event {index} must be an object")
        if "op" not in event:
            raise SchemaError(f"trace event {index} is missing 'op'")
        validate_op(event.get("op"), f"trace event {index}", custom=event.get("custom_op") is True)
        for text_key in ("phase", "group"):
            if text_key in event:
                validate_nonempty_string(event.get(text_key), f"trace event {index} {text_key}")
        if "bytes" not in event:
            raise SchemaError(f"trace event {index} is missing 'bytes'")
        if as_int(event.get("bytes")) <= 0:
            raise SchemaError(f"trace event {index} bytes must be positive")
        ranks = normalize_ranks(event.get("ranks"))
        if len(ranks) > limits.max_ranks:
            raise SchemaError(f"trace event {index} rank count exceeds resource policy limit={limits.max_ranks}")
        if "rank_count" in event and as_int(event.get("rank_count")) != len(ranks):
            raise SchemaError(f"trace event {index} rank_count must match ranks")
        if (
            allow_partial_arrivals
            and event.get("partial_rank_arrival")
            and isinstance(event.get("rank_arrival_us"), Mapping)
        ):
            validate_arrival_keys(
                event.get("rank_arrival_us", {}),
                ranks,
                f"trace event {index} rank_arrival_us",
                allow_subset=True,
            )
            for value in event.get("rank_arrival_us", {}).values():
                if as_float(value) < 0.0:
                    raise SchemaError(f"trace event {index} rank_arrival_us values must be non-negative")
        else:
            offsets = normalize_arrival_offsets(event, ranks)
            if "arrival_skew_us" in event and event.get("rank_arrival_us") is not None:
                validate_skew_matches_offsets(
                    as_float(event.get("arrival_skew_us")),
                    offsets,
                    f"trace event {index}",
                )
        for numeric_key in (
            "start_us",
            "gap_us",
            "compute_before_us",
            "compute_overlap_us",
            "compute_pressure",
            "observed_exposed_us",
        ):
            if numeric_key in event:
                numeric_value = as_float(event.get(numeric_key))
                if numeric_value < 0.0:
                    raise SchemaError(f"trace event {index} {numeric_key} must be non-negative")
                if numeric_key.endswith("_us") and numeric_value > MAX_TIME_US:
                    raise SchemaError(f"trace event {index} {numeric_key} exceeds maximum supported duration")
        if "concurrent_groups" in event and as_int(event.get("concurrent_groups")) <= 0:
            raise SchemaError(f"trace event {index} concurrent_groups must be positive")
        validate_point_to_point_metadata(event, ranks, f"trace event {index}")


__all__ = ["validate_trace"]
