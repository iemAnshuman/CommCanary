from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional

from .schema import (
    COMPARE_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    comparison_policy_evaluations,
    derive_comparison_verdict,
    validate_comparison,
    validate_report,
)


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    p99_threshold_pct: float = 15.0,
    p95_threshold_pct: float = 10.0,
    median_threshold_pct: float = 8.0,
    p99_absolute_threshold_us: float = 1.0,
    p95_absolute_threshold_us: float = 1.0,
    median_absolute_threshold_us: float = 1.0,
    hidden_drop_threshold_points: float = 5.0,
    breakdown_threshold_pct: Optional[float] = None,
    breakdown_absolute_threshold_us: Optional[float] = None,
    require_compatible: bool = True,
) -> JsonDict:
    validate_report(baseline)
    validate_report(candidate)
    p99_threshold_pct = _non_negative_threshold(p99_threshold_pct, "p99_threshold_pct")
    p95_threshold_pct = _non_negative_threshold(p95_threshold_pct, "p95_threshold_pct")
    median_threshold_pct = _non_negative_threshold(median_threshold_pct, "median_threshold_pct")
    p99_absolute_threshold_us = _non_negative_threshold(p99_absolute_threshold_us, "p99_absolute_threshold_us")
    p95_absolute_threshold_us = _non_negative_threshold(p95_absolute_threshold_us, "p95_absolute_threshold_us")
    median_absolute_threshold_us = _non_negative_threshold(
        median_absolute_threshold_us, "median_absolute_threshold_us"
    )
    hidden_drop_threshold_points = _non_negative_threshold(
        hidden_drop_threshold_points, "hidden_drop_threshold_points"
    )
    if breakdown_threshold_pct is None:
        breakdown_threshold_pct = p99_threshold_pct
    breakdown_threshold_pct = _non_negative_threshold(
        breakdown_threshold_pct, "breakdown_threshold_pct"
    )
    if breakdown_absolute_threshold_us is None:
        breakdown_absolute_threshold_us = p99_absolute_threshold_us
    breakdown_absolute_threshold_us = _non_negative_threshold(
        breakdown_absolute_threshold_us, "breakdown_absolute_threshold_us"
    )
    compatibility = _compatibility(baseline, candidate)
    if require_compatible and not compatibility["compatible"]:
        raise SchemaError("; ".join(compatibility["reasons"]))

    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    base_median = as_float(base_metrics.get("median_us"))
    cand_median = as_float(cand_metrics.get("median_us"))
    base_p95 = as_float(base_metrics.get("p95_us"))
    cand_p95 = as_float(cand_metrics.get("p95_us"))
    base_p99 = as_float(base_metrics.get("p99_us"))
    cand_p99 = as_float(cand_metrics.get("p99_us"))
    median_delta = _pct_delta(base_median, cand_median)
    p95_delta = _pct_delta(base_p95, cand_p95)
    p99_delta = _pct_delta(base_p99, cand_p99)
    hidden_delta = as_float(cand_metrics.get("communication_hidden_pct")) - as_float(
        base_metrics.get("communication_hidden_pct")
    )
    phase_deltas = _breakdown_deltas(baseline.get("by_phase", []), candidate.get("by_phase", []))
    op_deltas = _breakdown_deltas(baseline.get("by_op", []), candidate.get("by_op", []))

    verdict = "pass"
    reasons: List[str] = []
    if _exceeds_relative_threshold(p99_delta, base_p99, cand_p99, p99_threshold_pct, p99_absolute_threshold_us):
        verdict = "fail"
        reasons.append(_regression_reason("p99", p99_delta, base_p99, cand_p99, p99_threshold_pct))
    if _exceeds_relative_threshold(p95_delta, base_p95, cand_p95, p95_threshold_pct, p95_absolute_threshold_us):
        verdict = "fail"
        reasons.append(_regression_reason("p95", p95_delta, base_p95, cand_p95, p95_threshold_pct))
    if _exceeds_relative_threshold(
        median_delta, base_median, cand_median, median_threshold_pct, median_absolute_threshold_us
    ):
        verdict = "fail"
        reasons.append(_regression_reason("median", median_delta, base_median, cand_median, median_threshold_pct))
    hidden_drop = max(0.0, -hidden_delta)
    if hidden_drop > hidden_drop_threshold_points:
        verdict = "fail"
        reasons.append(
            "communication hidden percentage dropped "
            f"{hidden_drop:.2f} points, exceeding {hidden_drop_threshold_points:.2f}"
        )
    breakdown_failures = (
        _breakdown_regression_reasons("phase", phase_deltas, breakdown_threshold_pct, breakdown_absolute_threshold_us)
        + _breakdown_regression_reasons(
            "operation", op_deltas, breakdown_threshold_pct, breakdown_absolute_threshold_us
        )
    )
    if breakdown_failures:
        verdict = "fail"
        reasons.extend(breakdown_failures)
    if verdict == "pass" and (
        _exceeds_relative_threshold(
            p99_delta, base_p99, cand_p99, p99_threshold_pct / 2.0, p99_absolute_threshold_us / 2.0
        )
        or _exceeds_relative_threshold(
            p95_delta, base_p95, cand_p95, p95_threshold_pct / 2.0, p95_absolute_threshold_us / 2.0
        )
        or _exceeds_relative_threshold(
            median_delta,
            base_median,
            cand_median,
            median_threshold_pct / 2.0,
            median_absolute_threshold_us / 2.0,
        )
        or hidden_drop > hidden_drop_threshold_points / 2.0
    ):
        verdict = "warn"
        reasons.append("latency regression is below the failure threshold but large enough to inspect")
    if verdict == "pass":
        breakdown_warnings = (
            _breakdown_regression_reasons(
                "phase", phase_deltas, breakdown_threshold_pct / 2.0, breakdown_absolute_threshold_us / 2.0
            )
            + _breakdown_regression_reasons(
                "operation", op_deltas, breakdown_threshold_pct / 2.0, breakdown_absolute_threshold_us / 2.0
            )
        )
        if breakdown_warnings:
            verdict = "warn"
            reasons.extend(breakdown_warnings)
    if not compatibility["compatible"]:
        if verdict == "pass":
            verdict = "warn"
        reasons.extend(compatibility["reasons"])
    uncertainty = _uncertainty_summary(baseline, candidate)
    uncertainty_reasons = _uncertainty_reasons(uncertainty)
    if uncertainty_reasons:
        if verdict == "pass":
            verdict = "warn"
        reasons.extend(uncertainty_reasons)
    if not reasons:
        reasons.append("candidate is within configured thresholds")

    comparison: JsonDict = {
        "format": COMPARE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reasons": reasons,
        "thresholds": {
            "p99_threshold_pct": p99_threshold_pct,
            "p95_threshold_pct": p95_threshold_pct,
            "median_threshold_pct": median_threshold_pct,
            "p99_absolute_threshold_us": p99_absolute_threshold_us,
            "p95_absolute_threshold_us": p95_absolute_threshold_us,
            "median_absolute_threshold_us": median_absolute_threshold_us,
            "hidden_drop_threshold_points": hidden_drop_threshold_points,
            "breakdown_threshold_pct": breakdown_threshold_pct,
            "breakdown_absolute_threshold_us": breakdown_absolute_threshold_us,
        },
        "compatibility": compatibility,
        "uncertainty": uncertainty,
        "baseline": {
            "backend": copy.deepcopy(baseline.get("backend", {})),
            "metrics": copy.deepcopy(base_metrics),
        },
        "candidate": {
            "backend": copy.deepcopy(candidate.get("backend", {})),
            "metrics": copy.deepcopy(cand_metrics),
        },
        "delta": {
            "median_pct": _round_optional(median_delta),
            "median_relative_status": _delta_status(base_median, cand_median),
            "p95_pct": _round_optional(p95_delta),
            "p95_relative_status": _delta_status(base_p95, cand_p95),
            "p99_pct": _round_optional(p99_delta),
            "p99_relative_status": _delta_status(base_p99, cand_p99),
            "median_absolute_us": round(cand_median - base_median, 3),
            "p95_absolute_us": round(cand_p95 - base_p95, 3),
            "p99_absolute_us": round(cand_p99 - base_p99, 3),
            "communication_hidden_pct_points": round(hidden_delta, 2),
        },
        "breakdown_delta": {"by_phase": phase_deltas, "by_op": op_deltas},
        "worst_regressions": {
            "phase": phase_deltas[0] if phase_deltas else None,
            "operation": op_deltas[0] if op_deltas else None,
        },
    }
    comparison["evaluations"] = comparison_policy_evaluations(comparison)
    comparison["derived_verdict"] = derive_comparison_verdict(comparison)
    comparison["verdict"] = comparison["derived_verdict"]
    validate_comparison(comparison)
    return comparison


def _breakdown_deltas(base_rows: Any, candidate_rows: Any) -> List[JsonDict]:
    base = {
        str(row.get("name")): row
        for row in base_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    candidate = {
        str(row.get("name")): row
        for row in candidate_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    rows: List[JsonDict] = []
    for name in sorted(set(base) | set(candidate)):
        base_row = base.get(name)
        candidate_row = candidate.get(name)
        row: JsonDict = {
            "name": name,
            "baseline_count": as_int(base_row.get("count")) if base_row else 0,
            "candidate_count": as_int(candidate_row.get("count")) if candidate_row else 0,
            "status": "matched" if base_row and candidate_row else ("added" if candidate_row else "removed"),
        }
        for key in ("median_us", "p95_us", "p99_us"):
            base_value = as_float(base_row.get(key)) if base_row else 0.0
            candidate_value = as_float(candidate_row.get(key)) if candidate_row else 0.0
            row[f"baseline_{key}"] = base_value
            row[f"candidate_{key}"] = candidate_value
            pct_delta = _pct_delta(base_value, candidate_value)
            row[f"{key[:-3]}_pct"] = _round_optional(pct_delta)
            row[f"{key[:-3]}_relative_status"] = _delta_status(base_value, candidate_value)
            row[f"{key[:-3]}_absolute_us"] = round(candidate_value - base_value, 3)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("p99_relative_status") == "new_nonzero_regression",
            as_float(row.get("p99_pct")),
            as_float(row.get("p99_absolute_us")),
            as_float(row.get("median_pct")),
        ),
        reverse=True,
    )
    return rows


def _breakdown_regression_reasons(
    scope: str,
    rows: List[JsonDict],
    threshold_pct: float,
    absolute_threshold_us: float,
) -> List[str]:
    reasons: List[str] = []
    for row in rows:
        name = str(row.get("name"))
        for metric in ("p99", "p95", "median"):
            base = as_float(row.get(f"baseline_{metric}_us"))
            candidate = as_float(row.get(f"candidate_{metric}_us"))
            delta = row.get(f"{metric}_pct")
            delta_pct = as_float(delta) if delta is not None else None
            if _exceeds_relative_threshold(delta_pct, base, candidate, threshold_pct, absolute_threshold_us):
                reasons.append(
                    _regression_reason(f"{scope} {name!r} {metric}", delta_pct, base, candidate, threshold_pct)
                )
                break
    return reasons


def _pct_delta(base: float, candidate: float) -> Optional[float]:
    if base == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - base) / base * 100.0


def _round_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 2)


def _delta_status(base: float, candidate: float) -> str:
    if base == 0.0 and candidate == 0.0:
        return "both_zero"
    if base == 0.0:
        return "new_nonzero_regression" if candidate > 0.0 else "undefined"
    return "finite"


def _exceeds_relative_threshold(
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


def _regression_reason(
    label: str,
    delta_pct: Optional[float],
    base: float,
    candidate: float,
    threshold_pct: float,
) -> str:
    if delta_pct is None:
        return (
            f"{label} regression is new nonzero latency "
            f"({candidate - base:.3f} us absolute delta from a zero baseline)"
        )
    return f"{label} regression {delta_pct:.1f}% exceeds {threshold_pct:.1f}%"


def _compatibility(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    reasons: List[str] = []
    base_canary = baseline.get("canary", {})
    cand_canary = candidate.get("canary", {})
    base_scheduler = base_canary.get("scheduler_execution_sha256")
    cand_scheduler = cand_canary.get("scheduler_execution_sha256")
    base_semantic = base_canary.get("execution_semantic_sha256")
    cand_semantic = cand_canary.get("execution_semantic_sha256")
    if base_scheduler and cand_scheduler:
        if base_scheduler != cand_scheduler:
            reasons.append("reports were produced from different scheduler-execution fingerprints")
    elif base_semantic and cand_semantic:
        if base_semantic != cand_semantic:
            reasons.append("reports were produced from different executable canary fingerprints")
    elif base_canary.get("sha256") != cand_canary.get("sha256"):
        reasons.append("reports were produced from different canary fingerprints")
    base_model = baseline.get("simulation_model", {})
    cand_model = candidate.get("simulation_model", {})
    if base_model.get("version") != cand_model.get("version"):
        reasons.append("reports use different simulation model versions")
    base_protocol = baseline.get("replay_protocol", {})
    cand_protocol = candidate.get("replay_protocol", {})
    if base_protocol.get("sha256") != cand_protocol.get("sha256"):
        reasons.append("reports use different replay protocol fingerprints")
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "baseline_canary_sha256": base_canary.get("sha256"),
        "candidate_canary_sha256": cand_canary.get("sha256"),
        "baseline_execution_semantic_sha256": base_semantic,
        "candidate_execution_semantic_sha256": cand_semantic,
        "baseline_scheduler_execution_sha256": base_scheduler,
        "candidate_scheduler_execution_sha256": cand_scheduler,
        "baseline_simulation_model": base_model.get("version"),
        "candidate_simulation_model": cand_model.get("version"),
        "baseline_replay_protocol_sha256": base_protocol.get("sha256"),
        "candidate_replay_protocol_sha256": cand_protocol.get("sha256"),
    }


def _uncertainty_summary(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    summary: JsonDict = {}
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        uncertainty = report.get("canary_summary", {}).get("capture_uncertainty", {})
        count = 0
        if isinstance(uncertainty, Mapping):
            count = as_int(uncertainty.get("compute_fields_uncertain_events"), 0)
        summary[f"{label}_compute_fields_uncertain_events"] = count
    return summary


def _uncertainty_reasons(uncertainty: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    for label in ("baseline", "candidate"):
        count = as_int(uncertainty.get(f"{label}_compute_fields_uncertain_events"), 0)
        if count > 0:
            reasons.append(f"{label} has {count} events with uncertain rank-local compute fields")
    return reasons


def _non_negative_threshold(value: Any, name: str) -> float:
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed
