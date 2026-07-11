"""Comparison artifact contract and threshold-policy evaluation."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from ..errors import SchemaError
from ..formats import COMPARE_FORMAT
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
    validate_json_mapping,
)
from .wire import JsonDict, as_float, as_int, require_format


def validate_comparison(
    comparison: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    require_format(comparison, COMPARE_FORMAT, "comparison")
    _validate_comparison_json_resources(comparison, limits=limits)
    verdict = comparison.get("verdict")
    if verdict not in {"pass", "warn", "fail"}:
        raise SchemaError("comparison verdict must be pass, warn or fail")
    reasons = comparison.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        raise SchemaError("comparison must contain non-empty reasons")
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        raise SchemaError("comparison reasons must be non-empty strings")

    thresholds = comparison.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise SchemaError("comparison thresholds must be an object")
    for key, value in thresholds.items():
        if as_float(value) < 0.0:
            raise SchemaError(f"comparison threshold {key} must be non-negative")

    compatibility = comparison.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise SchemaError("comparison compatibility must be an object")
    compatible = compatibility.get("compatible")
    if not isinstance(compatible, bool):
        raise SchemaError("comparison compatibility.compatible must be a boolean")
    compatibility_reasons = compatibility.get("reasons", [])
    if not isinstance(compatibility_reasons, list):
        raise SchemaError("comparison compatibility.reasons must be a list")
    if compatible and compatibility_reasons:
        raise SchemaError("comparison compatibility cannot be true with mismatch reasons")
    if not compatible and not compatibility_reasons:
        raise SchemaError("comparison compatibility mismatch requires reasons")

    baseline_metrics = _comparison_metrics(comparison, "baseline")
    candidate_metrics = _comparison_metrics(comparison, "candidate")
    delta = comparison.get("delta")
    if not isinstance(delta, Mapping):
        raise SchemaError("comparison delta must be an object")
    for metric in ("median", "p95", "p99"):
        base_value = as_float(baseline_metrics.get(f"{metric}_us"))
        candidate_value = as_float(candidate_metrics.get(f"{metric}_us"))
        expected_pct = _comparison_round_optional(_comparison_pct_delta(base_value, candidate_value))
        if delta.get(f"{metric}_pct") != expected_pct:
            raise SchemaError(f"comparison delta.{metric}_pct does not match embedded metrics")
        if delta.get(f"{metric}_relative_status") != _comparison_delta_status(base_value, candidate_value):
            raise SchemaError(f"comparison delta.{metric}_relative_status does not match embedded metrics")
        expected_abs = round(candidate_value - base_value, 3)
        if abs(as_float(delta.get(f"{metric}_absolute_us")) - expected_abs) > 0.001:
            raise SchemaError(f"comparison delta.{metric}_absolute_us does not match embedded metrics")
    hidden_delta = round(
        as_float(candidate_metrics.get("communication_hidden_pct"))
        - as_float(baseline_metrics.get("communication_hidden_pct")),
        2,
    )
    if abs(as_float(delta.get("communication_hidden_pct_points")) - hidden_delta) > 0.01:
        raise SchemaError("comparison delta.communication_hidden_pct_points does not match embedded metrics")

    expected_evaluations = _comparison_policy_evaluations(comparison, limits=limits)
    evaluations = comparison.get("evaluations")
    if evaluations != expected_evaluations:
        raise SchemaError("comparison evaluations do not match embedded metrics and thresholds")
    derived_verdict = _comparison_verdict(comparison, expected_evaluations)
    if comparison.get("derived_verdict") != derived_verdict:
        raise SchemaError("comparison derived_verdict does not match policy evaluations")
    if verdict != derived_verdict:
        raise SchemaError("comparison verdict does not match policy evaluations")


def comparison_policy_evaluations(
    comparison: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> List[JsonDict]:
    _validate_comparison_json_resources(comparison, limits=limits)
    return _comparison_policy_evaluations(comparison, limits=limits)


def _comparison_policy_evaluations(
    comparison: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> List[JsonDict]:
    _preflight_comparison_evaluations(comparison, limits=limits)
    thresholds = comparison.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise SchemaError("comparison thresholds must be an object")
    baseline_metrics = _comparison_metrics(comparison, "baseline")
    candidate_metrics = _comparison_metrics(comparison, "candidate")
    evaluations = [
        _comparison_relative_evaluation(
            f"overall.{metric}",
            as_float(baseline_metrics.get(f"{metric}_us")),
            as_float(candidate_metrics.get(f"{metric}_us")),
            as_float(thresholds.get(f"{metric}_threshold_pct")),
            as_float(thresholds.get(f"{metric}_absolute_threshold_us")),
        )
        for metric in ("median", "p95", "p99")
    ]
    base_hidden = as_float(baseline_metrics.get("communication_hidden_pct"))
    candidate_hidden = as_float(candidate_metrics.get("communication_hidden_pct"))
    hidden_drop = max(0.0, base_hidden - candidate_hidden)
    hidden_threshold = as_float(thresholds.get("hidden_drop_threshold_points"))
    evaluations.append(
        {
            "metric": "overall.communication_hidden_pct_drop",
            "baseline": round(base_hidden, 2),
            "candidate": round(candidate_hidden, 2),
            "drop_points": round(hidden_drop, 2),
            "threshold_points": hidden_threshold,
            "threshold_result": _comparison_points_result(hidden_drop, hidden_threshold),
        }
    )

    breakdown = comparison.get("breakdown_delta", {})
    if not isinstance(breakdown, Mapping):
        raise SchemaError("comparison breakdown_delta must be an object")
    breakdown_threshold = as_float(thresholds.get("breakdown_threshold_pct"))
    breakdown_absolute_threshold = as_float(thresholds.get("breakdown_absolute_threshold_us"))
    for scope, key in (("phase", "by_phase"), ("operation", "by_op")):
        rows = breakdown.get(key, [])
        if not isinstance(rows, list):
            raise SchemaError(f"comparison breakdown_delta.{key} must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise SchemaError(f"comparison breakdown_delta.{key} rows must be objects")
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise SchemaError(f"comparison breakdown_delta.{key} rows must contain a name")
            for metric in ("median", "p95", "p99"):
                base = as_float(row.get(f"baseline_{metric}_us"))
                candidate = as_float(row.get(f"candidate_{metric}_us"))
                expected_pct = _comparison_round_optional(_comparison_pct_delta(base, candidate))
                if row.get(f"{metric}_pct") != expected_pct:
                    raise SchemaError(f"comparison breakdown_delta.{key}.{metric}_pct does not match row metrics")
                if row.get(f"{metric}_relative_status") != _comparison_delta_status(base, candidate):
                    raise SchemaError(
                        f"comparison breakdown_delta.{key}.{metric}_relative_status does not match row metrics"
                    )
                expected_abs = round(candidate - base, 3)
                if abs(as_float(row.get(f"{metric}_absolute_us")) - expected_abs) > 0.001:
                    raise SchemaError(
                        f"comparison breakdown_delta.{key}.{metric}_absolute_us does not match row metrics"
                    )
                evaluation = _comparison_relative_evaluation(
                    f"{scope}.{metric}",
                    base,
                    candidate,
                    breakdown_threshold,
                    breakdown_absolute_threshold,
                )
                evaluation["name"] = name
                evaluations.append(evaluation)

    uncertainty = comparison.get("uncertainty", {})
    if uncertainty is not None and not isinstance(uncertainty, Mapping):
        raise SchemaError("comparison uncertainty must be an object")
    uncertainty = uncertainty or {}
    for label in ("baseline", "candidate"):
        count = as_int(uncertainty.get(f"{label}_compute_fields_uncertain_events"), 0)
        if count < 0:
            raise SchemaError("comparison uncertainty counts must be non-negative")
        if count:
            evaluations.append(
                {
                    "metric": f"uncertainty.{label}.compute_fields",
                    "count": count,
                    "threshold_result": "warn",
                }
            )
    return evaluations


def derive_comparison_verdict(
    comparison: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> str:
    _validate_comparison_json_resources(comparison, limits=limits)
    evaluations = _comparison_policy_evaluations(comparison, limits=limits)
    return _comparison_verdict(comparison, evaluations)


def _comparison_verdict(comparison: Mapping[str, Any], evaluations: List[JsonDict]) -> str:
    if any(evaluation.get("threshold_result") == "fail" for evaluation in evaluations):
        return "fail"
    compatibility = comparison.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise SchemaError("comparison compatibility must be an object")
    if any(evaluation.get("threshold_result") == "warn" for evaluation in evaluations):
        return "warn"
    if compatibility.get("compatible") is False:
        return "warn"
    return "pass"


def _validate_comparison_json_resources(
    comparison: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> None:
    try:
        validate_json_mapping(comparison, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"comparison violates JSON resource constraints: {exc}") from exc


def _preflight_comparison_evaluations(
    comparison: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> None:
    breakdown = comparison.get("breakdown_delta", {})
    row_count = 0
    if isinstance(breakdown, Mapping):
        for key in ("by_phase", "by_op"):
            rows = breakdown.get(key, [])
            if isinstance(rows, list):
                try:
                    row_count = checked_add(row_count, len(rows), label="comparison breakdown rows")
                except JsonResourceError as exc:
                    raise SchemaError(str(exc)) from exc
    try:
        evaluation_count = checked_add(
            6,
            checked_multiply(row_count, 3, label="comparison policy evaluations"),
            label="comparison policy evaluations",
        )
        require_within(
            evaluation_count,
            limits.max_json_items,
            label="comparison policy evaluations",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def _comparison_metrics(comparison: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = comparison.get(key)
    if not isinstance(section, Mapping):
        raise SchemaError(f"comparison {key}.metrics must be an object")
    metrics = section.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SchemaError(f"comparison {key}.metrics must be an object")
    return metrics


def _comparison_relative_evaluation(
    metric: str,
    baseline: float,
    candidate: float,
    threshold_pct: float,
    absolute_threshold_us: float,
) -> JsonDict:
    delta_pct = _comparison_pct_delta(baseline, candidate)
    return {
        "metric": metric,
        "baseline": round(baseline, 3),
        "candidate": round(candidate, 3),
        "absolute_delta": round(candidate - baseline, 3),
        "relative_delta_pct": _comparison_round_optional(delta_pct),
        "relative_status": _comparison_delta_status(baseline, candidate),
        "threshold_pct": threshold_pct,
        "absolute_threshold_us": absolute_threshold_us,
        "threshold_result": _comparison_relative_result(
            delta_pct,
            baseline,
            candidate,
            threshold_pct,
            absolute_threshold_us,
        ),
    }


def _comparison_relative_result(
    delta_pct: Optional[float],
    baseline: float,
    candidate: float,
    threshold_pct: float,
    absolute_threshold_us: float,
) -> str:
    if _comparison_exceeds_relative_threshold(delta_pct, baseline, candidate, threshold_pct, absolute_threshold_us):
        return "fail"
    if _comparison_exceeds_relative_threshold(
        delta_pct,
        baseline,
        candidate,
        threshold_pct / 2.0,
        absolute_threshold_us / 2.0,
    ):
        return "warn"
    return "pass"


def _comparison_points_result(value: float, threshold: float) -> str:
    if value > threshold:
        return "fail"
    if value > threshold / 2.0:
        return "warn"
    return "pass"


def _comparison_pct_delta(base: float, candidate: float) -> Optional[float]:
    if base == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - base) / base * 100.0


def _comparison_round_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 2)


def _comparison_delta_status(base: float, candidate: float) -> str:
    if base == 0.0 and candidate == 0.0:
        return "both_zero"
    if base == 0.0:
        return "new_nonzero_regression" if candidate > 0.0 else "undefined"
    return "finite"


def _comparison_exceeds_relative_threshold(
    delta_pct: Optional[float],
    base: float,
    candidate: float,
    threshold_pct: float,
    absolute_threshold_us: float,
) -> bool:
    if candidate - base <= absolute_threshold_us:
        return False
    if delta_pct is None:
        return candidate > base
    return delta_pct > threshold_pct


__all__ = ["comparison_policy_evaluations", "derive_comparison_verdict", "validate_comparison"]
