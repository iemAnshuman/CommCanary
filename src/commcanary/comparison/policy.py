"""Typed comparison policy and stable evaluation reason codes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Optional

from ..artifacts.wire import JsonDict, as_float
from ..errors import SchemaError


class ComparisonReasonCode(str, Enum):
    """Stable fixed codes stored in ``comparison.evaluations[].metric``."""

    OVERALL_MEDIAN = "overall.median"
    OVERALL_P95 = "overall.p95"
    OVERALL_P99 = "overall.p99"
    OVERALL_HIDDEN_DROP = "overall.communication_hidden_pct_drop"
    UNCERTAIN_BASELINE_COMPUTE = "uncertainty.baseline.compute_fields"
    UNCERTAIN_CANDIDATE_COMPUTE = "uncertainty.candidate.compute_fields"


@dataclass(frozen=True)
class ComparisonThresholdPolicy:
    """Validated immutable threshold policy serialized by comparison v2."""

    p99_threshold_pct: float = 15.0
    p95_threshold_pct: float = 10.0
    median_threshold_pct: float = 8.0
    p99_absolute_threshold_us: float = 1.0
    p95_absolute_threshold_us: float = 1.0
    median_absolute_threshold_us: float = 1.0
    hidden_drop_threshold_points: float = 5.0
    breakdown_threshold_pct: float = 15.0
    breakdown_absolute_threshold_us: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            parsed = _non_negative_threshold(getattr(self, field.name), field.name)
            object.__setattr__(self, field.name, parsed)

    @classmethod
    def from_legacy_arguments(
        cls,
        *,
        p99_threshold_pct: float,
        p95_threshold_pct: float,
        median_threshold_pct: float,
        p99_absolute_threshold_us: float,
        p95_absolute_threshold_us: float,
        median_absolute_threshold_us: float,
        hidden_drop_threshold_points: float,
        breakdown_threshold_pct: Optional[float],
        breakdown_absolute_threshold_us: Optional[float],
    ) -> "ComparisonThresholdPolicy":
        """Parse the original keyword surface without changing its defaults."""

        parsed_p99 = _non_negative_threshold(p99_threshold_pct, "p99_threshold_pct")
        parsed_p99_absolute = _non_negative_threshold(
            p99_absolute_threshold_us,
            "p99_absolute_threshold_us",
        )
        return cls(
            p99_threshold_pct=parsed_p99,
            p95_threshold_pct=p95_threshold_pct,
            median_threshold_pct=median_threshold_pct,
            p99_absolute_threshold_us=parsed_p99_absolute,
            p95_absolute_threshold_us=p95_absolute_threshold_us,
            median_absolute_threshold_us=median_absolute_threshold_us,
            hidden_drop_threshold_points=hidden_drop_threshold_points,
            breakdown_threshold_pct=(parsed_p99 if breakdown_threshold_pct is None else breakdown_threshold_pct),
            breakdown_absolute_threshold_us=(
                parsed_p99_absolute if breakdown_absolute_threshold_us is None else breakdown_absolute_threshold_us
            ),
        )

    def to_wire(self) -> JsonDict:
        """Return the exact comparison-v2 threshold object."""

        return {field.name: getattr(self, field.name) for field in fields(self)}


def _non_negative_threshold(value: Any, name: str) -> float:
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed


__all__ = ["ComparisonReasonCode", "ComparisonThresholdPolicy"]
