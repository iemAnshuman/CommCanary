"""Trace artifact contract validation."""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, require_within
from .dtypes import require_canonical_dtype, require_param_compute_dtype
from .wire import (
    MAX_TIME_US,
    as_float,
    as_int,
    normalize_arrival_offsets,
    normalize_ranks,
    require_format,
    require_optional_mapping,
    validate_arrival_keys,
    validate_broadcast_metadata,
    validate_nonempty_string,
    validate_op,
    validate_point_to_point_metadata,
    validate_reduction_metadata,
    validate_sha256,
    validate_skew_matches_offsets,
)


def validate_trace(
    trace: Mapping[str, Any],
    *,
    allow_partial_arrivals: bool = False,
    require_known_overlap: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    require_format(trace, TRACE_FORMAT, "trace")
    require_optional_mapping(trace, "workload", "trace")
    require_optional_mapping(trace, "system", "trace")
    _validate_kineto_source_profiles(trace.get("system"), limits=limits)
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
        if "dtype" in event:
            require_canonical_dtype(event.get("dtype"), label=f"trace event {index} dtype")
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
        overlap_unknown = event.get("compute_overlap_unknown")
        if overlap_unknown is not None and not isinstance(overlap_unknown, bool):
            raise SchemaError(f"trace event {index} compute_overlap_unknown must be a boolean")
        if overlap_unknown is True and "compute_overlap_us" in event:
            raise SchemaError(
                f"trace event {index} cannot carry both compute_overlap_us and compute_overlap_unknown=true"
            )
        if overlap_unknown is False and "compute_overlap_us" not in event:
            raise SchemaError(f"trace event {index} compute_overlap_unknown=false requires compute_overlap_us")
        if require_known_overlap and ("compute_overlap_us" not in event or overlap_unknown is True):
            raise SchemaError(
                f"trace event {index} has unknown compute overlap; "
                "a measured or deliberately constructed compute_overlap_us value is required"
            )
        _validate_compute_recipe(event, ranks, index=index, limits=limits)
        if "concurrent_groups" in event and as_int(event.get("concurrent_groups")) <= 0:
            raise SchemaError(f"trace event {index} concurrent_groups must be positive")
        validate_broadcast_metadata(event, ranks, f"trace event {index}")
        validate_point_to_point_metadata(event, ranks, f"trace event {index}")
        validate_reduction_metadata(event, f"trace event {index}")


def _validate_compute_recipe(
    event: Mapping[str, Any],
    ranks: list[int],
    *,
    index: int,
    limits: ResourceLimits,
) -> None:
    recipe = event.get("compute_recipe")
    by_rank = event.get("compute_recipe_by_rank")
    if recipe is not None and by_rank is not None:
        raise SchemaError(f"trace event {index} cannot carry both compute_recipe and compute_recipe_by_rank")
    if recipe is not None:
        _validate_compute_recipe_operations(
            recipe,
            label=f"trace event {index} compute_recipe",
            limits=limits,
        )
    if by_rank is None:
        return
    if not isinstance(by_rank, Mapping):
        raise SchemaError(f"trace event {index} compute_recipe_by_rank must be an object")
    expected_keys = {str(rank) for rank in ranks}
    if set(by_rank) != expected_keys:
        raise SchemaError(f"trace event {index} compute_recipe_by_rank keys must match ranks")
    for rank in ranks:
        _validate_compute_recipe_operations(
            by_rank[str(rank)],
            label=f"trace event {index} compute_recipe_by_rank[{rank}]",
            limits=limits,
        )


def _validate_compute_recipe_operations(
    value: Any,
    *,
    label: str,
    limits: ResourceLimits,
) -> None:
    if not isinstance(value, list):
        raise SchemaError(f"{label} must be an array")
    try:
        require_within(
            len(value),
            limits.max_param_entries,
            label=f"{label} operations",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    expected_fields = {
        "op",
        "dtype",
        "m",
        "n",
        "k",
        "source_kernel_count",
        "source_kernel_duration_us",
    }
    for operation_index, operation in enumerate(value):
        operation_label = f"{label}[{operation_index}]"
        if not isinstance(operation, Mapping):
            raise SchemaError(f"{operation_label} must be an object")
        if set(operation) != expected_fields:
            raise SchemaError(f"{operation_label} fields do not match the supported GEMM recipe")
        if operation.get("op") != "gemm":
            raise SchemaError(f"{operation_label} op must be 'gemm'")
        canonical_dtype = require_canonical_dtype(
            operation.get("dtype"),
            label=f"{operation_label} dtype",
        )
        require_param_compute_dtype(
            canonical_dtype,
            label=f"{operation_label} dtype",
        )
        for dimension in ("m", "n", "k"):
            parsed = as_int(operation.get(dimension))
            if parsed <= 0:
                raise SchemaError(f"{operation_label} {dimension} must be positive")
            if parsed > limits.max_param_gemm_dim:
                raise SchemaError(
                    f"{operation_label} {dimension} exceeds max_param_gemm_dim={limits.max_param_gemm_dim}"
                )
        if as_int(operation.get("source_kernel_count")) <= 0:
            raise SchemaError(f"{operation_label} source_kernel_count must be positive")
        duration_us = as_float(operation.get("source_kernel_duration_us"))
        if duration_us <= 0.0:
            raise SchemaError(f"{operation_label} source_kernel_duration_us must be positive")
        if duration_us > MAX_TIME_US:
            raise SchemaError(f"{operation_label} source_kernel_duration_us exceeds maximum supported duration")


def _validate_kineto_source_profiles(
    system: Any,
    *,
    limits: ResourceLimits,
) -> None:
    if not isinstance(system, Mapping) or "kineto_source_profiles" not in system:
        return
    if system.get("source_format") != "pytorch-kineto":
        raise SchemaError("trace system.kineto_source_profiles requires source_format='pytorch-kineto'")
    profiles = system.get("kineto_source_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise SchemaError("trace system.kineto_source_profiles must be a non-empty list")
    try:
        require_within(
            len(profiles),
            limits.max_capture_shards,
            label="Kineto source profiles",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    seen = set()
    source_ranks = []
    for index, profile in enumerate(profiles):
        label = f"trace system.kineto_source_profiles[{index}]"
        if not isinstance(profile, Mapping):
            raise SchemaError(f"{label} must be an object")
        unexpected = set(profile) - {"sha256", "size_bytes", "rank"}
        if unexpected:
            raise SchemaError(f"{label} contains unsupported keys: {sorted(unexpected)!r}")
        validate_sha256(profile.get("sha256"), f"{label}.sha256")
        if as_int(profile.get("size_bytes")) <= 0:
            raise SchemaError(f"{label}.size_bytes must be positive")
        rank = None
        if "rank" in profile:
            rank = as_int(profile.get("rank"))
            if rank < 0:
                raise SchemaError(f"{label}.rank must be non-negative")
            if rank in source_ranks:
                raise SchemaError(f"{label}.rank duplicates an earlier source rank")
            source_ranks.append(rank)
        identity = (profile.get("sha256"), profile.get("size_bytes"), rank)
        if identity in seen:
            raise SchemaError(f"{label} duplicates an earlier source profile identity")
        seen.add(identity)
    if len(source_ranks) != len(profiles):
        if len(profiles) != 1 or source_ranks:
            raise SchemaError("trace system.kineto_source_profiles may omit rank only for one unranked profile")
        if "kineto_rank" in system or "kineto_imported_ranks" in system:
            raise SchemaError("unranked Kineto source profile conflicts with distributed rank metadata")
        return
    if "kineto_imported_ranks" in system:
        imported_ranks = normalize_ranks(system.get("kineto_imported_ranks"))
        if sorted(source_ranks) != imported_ranks:
            raise SchemaError("trace system Kineto source-profile ranks must match kineto_imported_ranks")
        return
    if "kineto_rank" not in system:
        raise SchemaError("ranked Kineto source profile requires kineto_rank ownership metadata")
    if len(source_ranks) != 1 or source_ranks[0] != as_int(system.get("kineto_rank")):
        raise SchemaError("trace system Kineto source-profile rank must match kineto_rank")


__all__ = ["validate_trace"]
