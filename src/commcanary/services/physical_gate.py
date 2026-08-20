"""Policy evaluation for qualified physical decision canary observations."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.physical_canary import (
    PHYSICAL_CANARY_STATUSES,
    validate_physical_canary_policy,
    validate_physical_gate_observation,
    validate_physical_gate_result,
)
from ..errors import SchemaError
from ..formats import PHYSICAL_GATE_RESULT_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..statistics import median, percentile


def evaluate_physical_gate(
    *,
    bundle_id: str,
    bundle_status: str,
    certified_baseline_subject_sha256: Optional[str],
    certified_canary_et_sha256: str,
    policy: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Dict[str, Any]:
    """Compare two observations under the exact policy bound by a bundle.

    The comparison is interval based.  A point estimate cannot distinguish "no
    regression" from "a regression this measurement could not resolve", and
    collapsing the second case into the first is a severe-false-negative
    generator -- the failure this gate exists to prevent.  Every metric is
    therefore screened for replicate count and timing stability, then decided
    against a percentile bootstrap interval on the median difference.  A metric
    whose interval straddles its acceptance boundary is ``inconclusive``, and a
    mandatory inconclusive metric makes the whole gate inconclusive rather than
    passing it.
    """

    validate_physical_canary_policy(policy, limits=limits)
    if bundle_status not in PHYSICAL_CANARY_STATUSES:
        raise SchemaError("physical gate bundle_status is unsupported")
    validate_physical_gate_observation(baseline, policy, limits=limits)
    validate_physical_gate_observation(candidate, policy, limits=limits)
    if baseline.get("role") != "baseline" or candidate.get("role") != "candidate":
        raise SchemaError("physical gate observations must be supplied in baseline, candidate order")
    if baseline.get("bundle_id") != bundle_id or candidate.get("bundle_id") != bundle_id:
        raise SchemaError("physical gate observations do not identify the verified canary bundle")
    if (
        baseline.get("executable_sha256") != certified_canary_et_sha256
        or candidate.get("executable_sha256") != certified_canary_et_sha256
    ):
        raise SchemaError("physical gate observations do not identify the bundle's canary executable")
    if (
        certified_baseline_subject_sha256 is not None
        and baseline.get("subject_sha256") != certified_baseline_subject_sha256
    ):
        raise SchemaError("physical gate baseline does not match the bundle's certified baseline subject")
    if baseline.get("subject_sha256") == candidate.get("subject_sha256"):
        raise SchemaError("physical gate baseline and candidate subjects must differ")
    if baseline.get("evidence_sha256") == candidate.get("evidence_sha256"):
        raise SchemaError("physical gate baseline and candidate evidence commitments must differ")

    issues = []
    if bundle_status != "qualified_physical_decision_canary":
        issues.append(f"bundle_not_qualified:{bundle_status}")
    if baseline.get("environment_sha256") != candidate.get("environment_sha256"):
        issues.append("environment_mismatch")

    minimum_samples = int(policy["minimum_samples"])
    max_relative_iqr_pct = float(policy["noise"]["max_relative_iqr_pct"])
    uncertainty = policy["uncertainty"]

    metric_results = []
    mandatory_failure = False
    mandatory_inconclusive = False
    for metric in policy["gate_metrics"]:
        name = str(metric["name"])
        baseline_samples = [float(value) for value in baseline["samples"][name]]
        candidate_samples = [float(value) for value in candidate["samples"][name]]
        baseline_median = median(baseline_samples)
        candidate_median = median(candidate_samples)
        direction = str(metric["direction"])
        threshold_pct = float(metric["regression_threshold_pct"])
        mandatory = bool(metric["mandatory"])

        samples_considered = min(len(baseline_samples), len(candidate_samples))
        baseline_iqr_pct = relative_iqr_pct(baseline_samples)
        candidate_iqr_pct = relative_iqr_pct(candidate_samples)

        regression_pct, _ = _regression(
            baseline_median,
            candidate_median,
            direction=direction,
            threshold_pct=threshold_pct,
        )

        lower: Optional[float] = None
        upper: Optional[float] = None
        resamples = 0
        if samples_considered < minimum_samples:
            status = "inconclusive"
            issues.append(f"insufficient_samples:{name}")
        elif baseline_iqr_pct > max_relative_iqr_pct or candidate_iqr_pct > max_relative_iqr_pct:
            status = "inconclusive"
            issues.append(f"unstable_measurement:{name}")
        else:
            resamples = int(uncertainty["bootstrap_resamples"])
            lower, upper = bootstrap_difference_interval(
                baseline_samples,
                candidate_samples,
                direction=direction,
                confidence=float(uncertainty["confidence"]),
                resamples=resamples,
                seed=int(uncertainty["seed"]),
            )
            status = _interval_status(
                lower,
                upper,
                baseline_median=baseline_median,
                threshold_pct=threshold_pct,
                mandatory=mandatory,
            )
            if status == "fail":
                issues.append(f"mandatory_metric_regression:{name}")
            elif status == "warn":
                issues.append(f"advisory_metric_regression:{name}")
            elif status == "inconclusive":
                issues.append(f"undecidable_metric:{name}")

        if status == "fail":
            mandatory_failure = True
        elif status == "inconclusive" and mandatory:
            mandatory_inconclusive = True

        metric_results.append(
            {
                "name": name,
                "direction": metric["direction"],
                "mandatory": mandatory,
                "baseline_median": baseline_median,
                "candidate_median": candidate_median,
                "regression_pct": regression_pct,
                "regression_threshold_pct": metric["regression_threshold_pct"],
                "samples_considered": samples_considered,
                "baseline_relative_iqr_pct": baseline_iqr_pct,
                "candidate_relative_iqr_pct": candidate_iqr_pct,
                "bootstrap_resamples": resamples,
                "confidence_interval_lower": lower,
                "confidence_interval_upper": upper,
                "status": status,
            }
        )

    if issues and (
        bundle_status != "qualified_physical_decision_canary"
        or baseline.get("environment_sha256") != candidate.get("environment_sha256")
    ):
        outcome = "incomparable"
    elif mandatory_failure:
        outcome = "fail"
    elif mandatory_inconclusive:
        outcome = "inconclusive"
    else:
        outcome = "pass"
    result = {
        "format": PHYSICAL_GATE_RESULT_FORMAT,
        "bundle_id": bundle_id,
        "runner_oci_digest": policy["runner"]["oci_digest"],
        "executable_sha256": certified_canary_et_sha256,
        "baseline_subject_sha256": baseline["subject_sha256"],
        "candidate_subject_sha256": candidate["subject_sha256"],
        "baseline_environment_sha256": baseline["environment_sha256"],
        "candidate_environment_sha256": candidate["environment_sha256"],
        "baseline_evidence_sha256": baseline["evidence_sha256"],
        "candidate_evidence_sha256": candidate["evidence_sha256"],
        "outcome": outcome,
        "issues": sorted(set(issues)),
        "metric_results": metric_results,
    }
    validate_physical_gate_result(result, limits=limits)
    return result


def relative_iqr_pct(samples: Sequence[float]) -> float:
    """Interquartile range as a percentage of the median, zero when undefined."""

    center = median(samples)
    if center <= 0:
        return 0.0
    return (percentile(samples, 75.0) - percentile(samples, 25.0)) / center * 100.0


def bootstrap_difference_interval(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    direction: str,
    confidence: float,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    """Percentile bootstrap interval on the regression-oriented median difference.

    The difference is oriented so that a positive value always means "worse than
    baseline", whichever direction the metric improves in.  The generator is
    seeded from the frozen policy, so the interval is reproducible and an
    independent verifier recomputes the same bounds from the same inputs.
    """

    if direction not in {"lower_is_better", "higher_is_better"}:
        raise SchemaError(f"unsupported physical gate direction {direction!r}")
    generator = random.Random(seed)
    differences: List[float] = []
    for _ in range(resamples):
        baseline_draw = [baseline_samples[generator.randrange(len(baseline_samples))] for _ in baseline_samples]
        candidate_draw = [candidate_samples[generator.randrange(len(candidate_samples))] for _ in candidate_samples]
        baseline_center = median(baseline_draw)
        candidate_center = median(candidate_draw)
        if direction == "lower_is_better":
            differences.append(candidate_center - baseline_center)
        else:
            differences.append(baseline_center - candidate_center)
    alpha_pct = (1.0 - confidence) * 50.0
    return percentile(differences, alpha_pct), percentile(differences, 100.0 - alpha_pct)


def _interval_status(
    lower: float,
    upper: float,
    *,
    baseline_median: float,
    threshold_pct: float,
    mandatory: bool,
) -> str:
    """Decide one metric from its interval against the acceptance boundary."""

    acceptance = baseline_median * threshold_pct / 100.0
    if lower > acceptance:
        return "fail" if mandatory else "warn"
    if upper <= acceptance:
        return "pass"
    return "inconclusive"


def _regression(
    baseline: float,
    candidate: float,
    *,
    direction: str,
    threshold_pct: float,
) -> tuple[Any, bool]:
    if direction == "lower_is_better":
        difference = candidate - baseline
    elif direction == "higher_is_better":
        difference = baseline - candidate
    else:  # Policy validation constrains this branch.
        raise SchemaError(f"unsupported physical gate direction {direction!r}")
    if difference <= 0.0:
        return 0.0, False
    if baseline == 0.0:
        return None, True
    regression_pct = difference / baseline * 100.0
    return regression_pct, regression_pct > threshold_pct


__all__ = [
    "bootstrap_difference_interval",
    "evaluate_physical_gate",
    "relative_iqr_pct",
]
