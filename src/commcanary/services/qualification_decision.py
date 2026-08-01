"""Deterministic four-state interpretation of raw qualification observations."""

from __future__ import annotations

import random
from typing import Any, List, Mapping, Sequence, Tuple

from ..artifacts.qualification_decision import (
    qualification_verdict_sha256,
    validate_qualification_observation,
    validate_qualification_policy,
    validate_qualification_verdict,
)
from ..artifacts.wire import JsonDict, as_float, as_int
from ..errors import SchemaError
from ..formats import QUALIFICATION_VERDICT_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..statistics import median, percentile


def evaluate_qualification_observations(
    policy: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Apply one predeclared policy without inventing missing decision state."""

    validate_qualification_policy(policy, limits=limits)
    validate_qualification_observation(baseline, limits=limits)
    validate_qualification_observation(candidate, limits=limits)
    policy_id = str(policy["policy_id"])
    if baseline["policy_id"] != policy_id or candidate["policy_id"] != policy_id:
        raise SchemaError("qualification observations are not bound to the supplied policy")
    if baseline["role"] != "baseline" or candidate["role"] != "candidate":
        raise SchemaError("qualification evaluation requires baseline and candidate observation roles")

    baseline_samples = _samples(baseline)
    candidate_samples = _samples(candidate)
    statistics = _base_statistics(policy, baseline_samples, candidate_samples)
    verdict: str
    reason_codes: List[str]
    explanation: str

    if (
        baseline["request_id"] != candidate["request_id"]
        or baseline["materialization_id"] != candidate["materialization_id"]
    ):
        verdict = "incomparable"
        reason_codes = ["workload_identity_mismatch"]
        explanation = "The observations do not execute the same request and materialization."
    elif baseline["metric"] != candidate["metric"]:
        verdict = "incomparable"
        reason_codes = ["metric_semantics_mismatch"]
        explanation = "The observations use different metric semantics."
    else:
        environment_reasons = _environment_mismatch_reasons(policy, baseline, candidate)
        if environment_reasons:
            verdict = "incomparable"
            reason_codes = environment_reasons
            explanation = "The observations violate the policy's environment comparability requirements."
        elif _correctness_status(baseline) != "pass":
            verdict = "incomparable"
            reason_codes = ["baseline_correctness_failed"]
            explanation = "The baseline failed correctness checks and cannot anchor a comparison."
        elif _correctness_status(candidate) != "pass":
            verdict = "fail"
            reason_codes = ["candidate_correctness_failed"]
            explanation = "The candidate failed physical correctness checks."
        else:
            completeness_reasons = _completeness_reasons(policy, baseline, candidate)
            if completeness_reasons:
                verdict = "inconclusive"
                reason_codes = completeness_reasons
                explanation = "The observations do not meet the predeclared repetition requirements."
            else:
                noise_reasons = _noise_reasons(policy, baseline, candidate, statistics)
                if noise_reasons:
                    verdict = "inconclusive"
                    reason_codes = noise_reasons
                    explanation = "The observed environment or timing variability exceeds the policy bound."
                else:
                    lower, upper = _bootstrap_interval(policy, baseline_samples, candidate_samples)
                    statistics["confidence_interval_lower_us"] = lower
                    statistics["confidence_interval_upper_us"] = upper
                    statistics["bootstrap_resamples"] = as_int(policy["uncertainty"]["bootstrap_resamples"])
                    threshold = as_float(statistics["acceptance_threshold_us"])
                    if lower > threshold:
                        verdict = "fail"
                        reason_codes = ["regression_confidently_exceeds_threshold"]
                        explanation = "The entire confidence interval exceeds the acceptance boundary."
                    elif upper <= threshold:
                        verdict = "pass"
                        reason_codes = ["regression_within_threshold"]
                        explanation = "The confidence interval remains within the acceptance boundary."
                    else:
                        verdict = "inconclusive"
                        reason_codes = ["confidence_interval_crosses_threshold"]
                        explanation = "The confidence interval crosses the acceptance boundary."

    result: JsonDict = {
        "format": QUALIFICATION_VERDICT_FORMAT,
        "policy_id": policy_id,
        "baseline_observation_id": baseline["observation_id"],
        "candidate_observation_id": candidate["observation_id"],
        "verdict": verdict,
        "reason_codes": sorted(set(reason_codes)),
        "explanation": explanation,
        "statistics": statistics,
        "claims": {
            "policy_application": "recomputed",
            "physical_decision": verdict,
            "producer_authenticity": "unsigned",
        },
    }
    result["verdict_id"] = qualification_verdict_sha256(result)
    validate_qualification_verdict(result, limits=limits)
    return result


def _samples(observation: Mapping[str, Any]) -> List[float]:
    return [as_float(value) for value in observation["measurement"]["samples_us"]]


def _correctness_status(observation: Mapping[str, Any]) -> str:
    return str(observation["measurement"]["correctness"]["status"])


def _base_statistics(
    policy: Mapping[str, Any],
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
) -> JsonDict:
    baseline_median = median(baseline_samples)
    candidate_median = median(candidate_samples)
    difference = candidate_median - baseline_median
    difference_pct = difference / baseline_median * 100.0
    regression = policy["regression"]
    threshold = max(
        as_float(regression["absolute_threshold_us"]),
        baseline_median * as_float(regression["relative_threshold_pct"]) / 100.0,
    )
    confidence = as_float(policy["uncertainty"]["confidence"])
    return {
        "baseline_sample_count": len(baseline_samples),
        "candidate_sample_count": len(candidate_samples),
        "baseline_median_us": baseline_median,
        "candidate_median_us": candidate_median,
        "signed_difference_us": difference,
        "signed_difference_pct": difference_pct,
        "acceptance_threshold_us": threshold,
        "confidence": confidence,
        "confidence_interval_lower_us": None,
        "confidence_interval_upper_us": None,
        "bootstrap_resamples": 0,
        "baseline_relative_iqr_pct": _relative_iqr_pct(baseline_samples),
        "candidate_relative_iqr_pct": _relative_iqr_pct(candidate_samples),
    }


def _environment_mismatch_reasons(
    policy: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> List[str]:
    declared = policy["environment"]
    baseline_environment = baseline["environment"]
    candidate_environment = candidate["environment"]
    reasons = []
    if declared["require_same_gpu_count"] and baseline_environment["gpu_count"] != candidate_environment["gpu_count"]:
        reasons.append("gpu_count_mismatch")
    if (
        declared["require_same_topology_class"]
        and baseline_environment["topology_class"] != candidate_environment["topology_class"]
    ):
        reasons.append("topology_class_mismatch")
    return reasons


def _completeness_reasons(
    policy: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> List[str]:
    required_warmup = as_int(policy["repetitions"]["warmup"])
    required_measured = as_int(policy["repetitions"]["minimum_measured"])
    reasons = []
    if as_int(baseline["measurement"]["warmup_count"]) < required_warmup:
        reasons.append("baseline_warmup_incomplete")
    if as_int(candidate["measurement"]["warmup_count"]) < required_warmup:
        reasons.append("candidate_warmup_incomplete")
    if len(baseline["measurement"]["samples_us"]) < required_measured:
        reasons.append("baseline_measurements_incomplete")
    if len(candidate["measurement"]["samples_us"]) < required_measured:
        reasons.append("candidate_measurements_incomplete")
    return reasons


def _noise_reasons(
    policy: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> List[str]:
    max_clock = as_float(policy["environment"]["max_clock_variation_pct"])
    max_iqr = as_float(policy["noise"]["max_relative_iqr_pct"])
    reasons = []
    if as_float(baseline["environment"]["clock_variation_pct"]) > max_clock:
        reasons.append("baseline_clock_variation_exceeded")
    if as_float(candidate["environment"]["clock_variation_pct"]) > max_clock:
        reasons.append("candidate_clock_variation_exceeded")
    if as_float(statistics["baseline_relative_iqr_pct"]) > max_iqr:
        reasons.append("baseline_timing_variability_exceeded")
    if as_float(statistics["candidate_relative_iqr_pct"]) > max_iqr:
        reasons.append("candidate_timing_variability_exceeded")
    return reasons


def _relative_iqr_pct(samples: Sequence[float]) -> float:
    center = median(samples)
    if center <= 0:
        return 0.0
    return (percentile(samples, 75.0) - percentile(samples, 25.0)) / center * 100.0


def _bootstrap_interval(
    policy: Mapping[str, Any],
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
) -> Tuple[float, float]:
    uncertainty = policy["uncertainty"]
    resamples = as_int(uncertainty["bootstrap_resamples"])
    generator = random.Random(as_int(uncertainty["seed"]))
    differences = []
    for _ in range(resamples):
        baseline_draw = [baseline_samples[generator.randrange(len(baseline_samples))] for _ in baseline_samples]
        candidate_draw = [candidate_samples[generator.randrange(len(candidate_samples))] for _ in candidate_samples]
        differences.append(median(candidate_draw) - median(baseline_draw))
    alpha_pct = (1.0 - as_float(uncertainty["confidence"])) * 50.0
    return percentile(differences, alpha_pct), percentile(differences, 100.0 - alpha_pct)


__all__ = ["evaluate_qualification_observations"]
