"""Shared deterministic statistics used by compilation and replay."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

from .errors import SchemaError


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"expected finite numeric value, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"expected numeric value, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise SchemaError(f"expected finite numeric value, got {value!r}")
    return parsed


def median(values: Iterable[float]) -> float:
    """Return the linearly interpolated 50th percentile, or zero if empty."""

    return percentile(values, 50.0)


def percentile(values: Iterable[float], q: float) -> float:
    """Return CommCanary's stable linear-interpolated percentile."""

    return percentile_from_sorted(sorted(_finite_float(value) for value in values), q)


def percentile_from_sorted(ordered: Sequence[float], q: float) -> float:
    """Return a percentile from an already sorted sequence.

    This deliberately preserves the original wire-metric algorithm: the
    fractional position is ``(n - 1) * q / 100`` and adjacent values are
    linearly interpolated.
    """

    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latencies(values: Iterable[float]) -> Dict[str, Any]:
    """Produce the exact rounded latency summary used on artifact wires."""

    data = sorted(_finite_float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "median_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
            "mean_us": 0.0,
        }
    return {
        "count": len(data),
        "median_us": round(percentile_from_sorted(data, 50.0), 3),
        "p95_us": round(percentile_from_sorted(data, 95.0), 3),
        "p99_us": round(percentile_from_sorted(data, 99.0), 3),
        "max_us": round(data[-1], 3),
        "mean_us": round(sum(data) / len(data), 3),
    }


__all__ = ["median", "percentile", "percentile_from_sorted", "summarize_latencies"]
