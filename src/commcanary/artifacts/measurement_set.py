"""Validation for tool-neutral repeated measurement sets."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..errors import SchemaError
from ..formats import MEASUREMENT_SET_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping

MEASUREMENT_DIRECTIONS = ("lower_is_better", "higher_is_better")
MEASUREMENT_ROLES = ("reference", "proxy")


def validate_measurement_set(
    measurement_set: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Validate one closed repeated-measurement mapping."""

    try:
        validate_json_mapping(measurement_set, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"measurement set violates JSON resource constraints: {exc}") from exc
    expected = {"format", "name", "role", "metric", "direction", "measurements"}
    if set(measurement_set) != expected:
        raise SchemaError("measurement set fields are not closed")
    if measurement_set.get("format") != MEASUREMENT_SET_FORMAT:
        raise SchemaError("measurement set format is unsupported")
    for field in ("name", "metric"):
        value = measurement_set.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"measurement set {field} must be a non-empty string")
    if measurement_set.get("role") not in MEASUREMENT_ROLES:
        raise SchemaError("measurement set role is unsupported")
    if measurement_set.get("direction") not in MEASUREMENT_DIRECTIONS:
        raise SchemaError("measurement set direction is unsupported")

    measurements = measurement_set.get("measurements")
    if not isinstance(measurements, Mapping) or len(measurements) < 2:
        raise SchemaError("measurement set must contain at least two configurations")
    if len(measurements) > limits.max_behavior_configurations:
        raise SchemaError(
            "measurement set configuration count "
            f"{len(measurements)} exceeds limit={limits.max_behavior_configurations}"
        )
    reference_values = measurement_set["role"] == "reference"
    sample_count = 0
    for configuration, raw_samples in measurements.items():
        if not isinstance(configuration, str) or not configuration.strip():
            raise SchemaError("measurement set configuration names must be non-empty strings")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise SchemaError(f"measurement set configuration {configuration!r} must contain samples")
        sample_count += len(raw_samples)
        if sample_count > limits.max_execution_observation_samples:
            raise SchemaError(f"measurement set sample count exceeds limit={limits.max_execution_observation_samples}")
        for index, raw_value in enumerate(raw_samples):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise SchemaError(f"measurement set configuration {configuration!r} sample {index} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise SchemaError(f"measurement set configuration {configuration!r} sample {index} must be finite")
            if reference_values and value <= 0.0:
                raise SchemaError(
                    f"reference measurement set configuration {configuration!r} sample {index} must be positive"
                )


__all__ = [
    "MEASUREMENT_DIRECTIONS",
    "MEASUREMENT_ROLES",
    "validate_measurement_set",
]
