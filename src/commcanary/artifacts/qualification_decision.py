"""Explicit physical-qualification policy, observation, and verdict contracts."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, List, Mapping

from ..errors import SchemaError
from ..formats import (
    QUALIFICATION_OBSERVATION_FORMAT,
    QUALIFICATION_POLICY_FORMAT,
    QUALIFICATION_VERDICT_FORMAT,
)
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping
from .json_codec import canonical_json_bytes
from .wire import as_float, as_int, require_format, validate_nonempty_string, validate_sha256

QUALIFICATION_VERDICTS = frozenset({"pass", "fail", "inconclusive", "incomparable"})
QUALIFICATION_REASON_CODES = frozenset(
    {
        "baseline_clock_variation_exceeded",
        "baseline_correctness_failed",
        "baseline_measurements_incomplete",
        "baseline_timing_variability_exceeded",
        "baseline_warmup_incomplete",
        "candidate_clock_variation_exceeded",
        "candidate_correctness_failed",
        "candidate_measurements_incomplete",
        "candidate_timing_variability_exceeded",
        "candidate_warmup_incomplete",
        "confidence_interval_crosses_threshold",
        "gpu_count_mismatch",
        "metric_semantics_mismatch",
        "regression_confidently_exceeds_threshold",
        "regression_within_threshold",
        "topology_class_mismatch",
        "workload_identity_mismatch",
    }
)


def qualification_policy_sha256(policy: Mapping[str, Any]) -> str:
    return _content_identity(policy, "policy_id")


def qualification_observation_sha256(observation: Mapping[str, Any]) -> str:
    return _content_identity(observation, "observation_id")


def qualification_verdict_sha256(verdict: Mapping[str, Any]) -> str:
    return _content_identity(verdict, "verdict_id")


def validate_qualification_policy(
    policy: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(policy, "qualification policy", limits)
    require_format(policy, QUALIFICATION_POLICY_FORMAT, "qualification policy")
    _exact_fields(
        policy,
        {
            "format",
            "policy_id",
            "primary_metric",
            "repetitions",
            "regression",
            "uncertainty",
            "noise",
            "environment",
            "outcome_policy",
        },
        "qualification policy",
    )
    _identity(policy, "policy_id", qualification_policy_sha256, "qualification policy")
    if policy.get("primary_metric") != "program_median_us":
        raise SchemaError("qualification policy primary_metric must be 'program_median_us'")

    repetitions = _mapping(policy.get("repetitions"), "qualification policy repetitions")
    _exact_fields(repetitions, {"warmup", "minimum_measured"}, "qualification policy repetitions")
    _bounded_int(repetitions.get("warmup"), "qualification policy repetitions.warmup", 0, 1_000_000)
    _bounded_int(
        repetitions.get("minimum_measured"),
        "qualification policy repetitions.minimum_measured",
        2,
        1_000_000,
    )

    regression = _mapping(policy.get("regression"), "qualification policy regression")
    _exact_fields(
        regression,
        {"relative_threshold_pct", "absolute_threshold_us", "threshold_combination"},
        "qualification policy regression",
    )
    _finite_nonnegative(regression.get("relative_threshold_pct"), "relative_threshold_pct")
    _finite_nonnegative(regression.get("absolute_threshold_us"), "absolute_threshold_us")
    if regression.get("threshold_combination") != "larger_of_absolute_or_relative":
        raise SchemaError("qualification policy regression.threshold_combination is unsupported")

    uncertainty = _mapping(policy.get("uncertainty"), "qualification policy uncertainty")
    _exact_fields(
        uncertainty,
        {"method", "confidence", "bootstrap_resamples", "seed"},
        "qualification policy uncertainty",
    )
    if uncertainty.get("method") != "percentile_bootstrap_median_difference":
        raise SchemaError("qualification policy uncertainty.method is unsupported")
    confidence = _finite_nonnegative(uncertainty.get("confidence"), "qualification policy confidence")
    if not 0.5 < confidence < 1.0:
        raise SchemaError("qualification policy uncertainty.confidence must be in (0.5, 1.0)")
    _bounded_int(uncertainty.get("bootstrap_resamples"), "bootstrap_resamples", 100, 100_000)
    _bounded_int(uncertainty.get("seed"), "bootstrap seed", 0, (1 << 63) - 1)

    noise = _mapping(policy.get("noise"), "qualification policy noise")
    _exact_fields(noise, {"max_relative_iqr_pct"}, "qualification policy noise")
    _finite_nonnegative(noise.get("max_relative_iqr_pct"), "max_relative_iqr_pct")

    environment = _mapping(policy.get("environment"), "qualification policy environment")
    _exact_fields(
        environment,
        {"max_clock_variation_pct", "require_same_gpu_count", "require_same_topology_class"},
        "qualification policy environment",
    )
    _finite_nonnegative(environment.get("max_clock_variation_pct"), "max_clock_variation_pct")
    _boolean(environment.get("require_same_gpu_count"), "require_same_gpu_count")
    _boolean(environment.get("require_same_topology_class"), "require_same_topology_class")

    outcome = _mapping(policy.get("outcome_policy"), "qualification policy outcome_policy")
    expected_outcomes = {
        "incomplete_measurement": "inconclusive",
        "unstable_measurement": "inconclusive",
        "environment_mismatch": "incomparable",
        "baseline_correctness_failure": "incomparable",
        "candidate_correctness_failure": "fail",
        "confidence_interval_crosses_threshold": "inconclusive",
    }
    if dict(outcome) != expected_outcomes:
        raise SchemaError("qualification policy outcome_policy must explicitly preserve the four-state contract")


def validate_qualification_observation(
    observation: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(observation, "qualification observation", limits)
    require_format(observation, QUALIFICATION_OBSERVATION_FORMAT, "qualification observation")
    _exact_fields(
        observation,
        {
            "format",
            "observation_id",
            "request_id",
            "materialization_id",
            "policy_id",
            "role",
            "metric",
            "environment",
            "execution",
            "measurement",
            "claims",
        },
        "qualification observation",
    )
    _identity(
        observation,
        "observation_id",
        qualification_observation_sha256,
        "qualification observation",
    )
    for field in ("request_id", "materialization_id", "policy_id"):
        validate_sha256(observation.get(field), f"qualification observation {field}")
    if observation.get("role") not in {"baseline", "candidate"}:
        raise SchemaError("qualification observation role must be 'baseline' or 'candidate'")

    metric = _mapping(observation.get("metric"), "qualification observation metric")
    if dict(metric) != {
        "name": "program_median_us",
        "unit": "microseconds",
        "lower_is_better": True,
    }:
        raise SchemaError("qualification observation metric is unsupported")

    environment = _mapping(observation.get("environment"), "qualification observation environment")
    _exact_fields(
        environment,
        {"gpu_count", "topology_class", "software_stack_sha256", "clock_variation_pct"},
        "qualification observation environment",
    )
    _bounded_int(environment.get("gpu_count"), "qualification observation gpu_count", 1, 1024)
    validate_nonempty_string(environment.get("topology_class"), "qualification observation topology_class")
    validate_sha256(
        environment.get("software_stack_sha256"),
        "qualification observation software_stack_sha256",
    )
    _finite_nonnegative(environment.get("clock_variation_pct"), "qualification observation clock_variation_pct")

    execution = _mapping(observation.get("execution"), "qualification observation execution")
    _exact_fields(
        execution,
        {"executor", "executor_version", "hostname", "job_id", "observed_at"},
        "qualification observation execution",
    )
    for field in ("executor", "executor_version", "hostname", "observed_at"):
        validate_nonempty_string(execution.get(field), f"qualification observation execution.{field}")
    job_id = execution.get("job_id")
    if job_id is not None:
        validate_nonempty_string(job_id, "qualification observation execution.job_id")

    measurement = _mapping(observation.get("measurement"), "qualification observation measurement")
    _exact_fields(
        measurement,
        {"warmup_count", "samples_us", "discarded_attempts", "correctness"},
        "qualification observation measurement",
    )
    _bounded_int(measurement.get("warmup_count"), "qualification observation warmup_count", 0, 1_000_000)
    _bounded_int(
        measurement.get("discarded_attempts"),
        "qualification observation discarded_attempts",
        0,
        1_000_000,
    )
    _samples(measurement.get("samples_us"), limits=limits)
    correctness = _mapping(measurement.get("correctness"), "qualification observation correctness")
    _exact_fields(correctness, {"status", "check_count"}, "qualification observation correctness")
    if correctness.get("status") not in {"pass", "fail"}:
        raise SchemaError("qualification observation correctness.status is unsupported")
    _bounded_int(correctness.get("check_count"), "qualification observation correctness.check_count", 1, 1_000_000)

    claims = _mapping(observation.get("claims"), "qualification observation claims")
    expected_claims = {
        "physical_execution": "observed",
        "executor_conformance": "unproven",
        "physical_decision": "not_interpreted",
        "producer_authenticity": "unsigned",
    }
    if dict(claims) != expected_claims:
        raise SchemaError("qualification observation claims exceed the raw-observation assurance boundary")


def validate_qualification_verdict(
    verdict: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(verdict, "qualification verdict", limits)
    require_format(verdict, QUALIFICATION_VERDICT_FORMAT, "qualification verdict")
    _exact_fields(
        verdict,
        {
            "format",
            "verdict_id",
            "policy_id",
            "baseline_observation_id",
            "candidate_observation_id",
            "verdict",
            "reason_codes",
            "explanation",
            "statistics",
            "claims",
        },
        "qualification verdict",
    )
    _identity(verdict, "verdict_id", qualification_verdict_sha256, "qualification verdict")
    for field in ("policy_id", "baseline_observation_id", "candidate_observation_id"):
        validate_sha256(verdict.get(field), f"qualification verdict {field}")
    if verdict.get("verdict") not in QUALIFICATION_VERDICTS:
        raise SchemaError("qualification verdict verdict is unsupported")
    reasons = verdict.get("reason_codes")
    if not isinstance(reasons, list) or not reasons:
        raise SchemaError("qualification verdict reason_codes must be a non-empty array")
    for index, reason in enumerate(reasons):
        validate_nonempty_string(reason, f"qualification verdict reason_codes[{index}]")
        if reason not in QUALIFICATION_REASON_CODES:
            raise SchemaError(f"qualification verdict reason_codes[{index}] is unsupported")
    if reasons != sorted(set(reasons)):
        raise SchemaError("qualification verdict reason_codes must be sorted and unique")
    validate_nonempty_string(verdict.get("explanation"), "qualification verdict explanation")
    statistics = _mapping(verdict.get("statistics"), "qualification verdict statistics")
    _exact_fields(
        statistics,
        {
            "baseline_sample_count",
            "candidate_sample_count",
            "baseline_median_us",
            "candidate_median_us",
            "signed_difference_us",
            "signed_difference_pct",
            "acceptance_threshold_us",
            "confidence",
            "confidence_interval_lower_us",
            "confidence_interval_upper_us",
            "bootstrap_resamples",
            "baseline_relative_iqr_pct",
            "candidate_relative_iqr_pct",
        },
        "qualification verdict statistics",
    )
    for field in ("baseline_sample_count", "candidate_sample_count"):
        _bounded_int(statistics.get(field), f"qualification verdict statistics.{field}", 1, 1_000_000)
    bootstrap_resamples = _bounded_int(
        statistics.get("bootstrap_resamples"),
        "qualification verdict statistics.bootstrap_resamples",
        0,
        100_000,
    )
    for field in (
        "baseline_median_us",
        "candidate_median_us",
        "signed_difference_us",
        "signed_difference_pct",
        "acceptance_threshold_us",
        "confidence",
        "confidence_interval_lower_us",
        "confidence_interval_upper_us",
        "baseline_relative_iqr_pct",
        "candidate_relative_iqr_pct",
    ):
        value = statistics.get(field)
        if value is not None:
            _finite(value, f"qualification verdict statistics.{field}")
    for field in ("baseline_median_us", "candidate_median_us"):
        if statistics.get(field) is None or as_float(statistics[field]) <= 0:
            raise SchemaError(f"qualification verdict statistics.{field} must be positive")
    for field in ("acceptance_threshold_us", "baseline_relative_iqr_pct", "candidate_relative_iqr_pct"):
        if statistics.get(field) is None or as_float(statistics[field]) < 0:
            raise SchemaError(f"qualification verdict statistics.{field} must be non-negative")
    confidence = statistics.get("confidence")
    if confidence is None or not 0.5 < as_float(confidence) < 1.0:
        raise SchemaError("qualification verdict statistics.confidence must be in (0.5, 1.0)")
    lower = statistics.get("confidence_interval_lower_us")
    upper = statistics.get("confidence_interval_upper_us")
    if bootstrap_resamples:
        if lower is None or upper is None or as_float(lower) > as_float(upper):
            raise SchemaError("qualification verdict confidence interval is incomplete or reversed")
    elif lower is not None or upper is not None:
        raise SchemaError("qualification verdict without bootstrap resamples cannot include a confidence interval")
    claims = _mapping(verdict.get("claims"), "qualification verdict claims")
    if dict(claims) != {
        "policy_application": "recomputed",
        "physical_decision": str(verdict["verdict"]),
        "producer_authenticity": "unsigned",
    }:
        raise SchemaError("qualification verdict claims do not match its four-state decision")


def _content_identity(value: Mapping[str, Any], identity_field: str) -> str:
    stable = {key: item for key, item in value.items() if key != identity_field}
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def _identity(value: Mapping[str, Any], field: str, derive: Any, label: str) -> None:
    validate_sha256(value.get(field), f"{label} {field}")
    if derive(value) != value.get(field):
        raise SchemaError(f"{label} {field} does not match canonical content")


def _resource_check(value: Mapping[str, Any], label: str, limits: ResourceLimits) -> None:
    try:
        validate_json_mapping(value, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"{label} violates JSON resource constraints: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    unknown = sorted(set(value) - expected_set)
    if missing:
        raise SchemaError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    parsed = as_int(value)
    if not minimum <= parsed <= maximum:
        raise SchemaError(f"{label} must be in [{minimum}, {maximum}]")
    return parsed


def _finite(value: Any, label: str) -> float:
    parsed = as_float(value)
    if not math.isfinite(parsed):
        raise SchemaError(f"{label} must be finite")
    return parsed


def _finite_nonnegative(value: Any, label: str) -> float:
    parsed = _finite(value, label)
    if parsed < 0:
        raise SchemaError(f"{label} must be non-negative")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{label} must be boolean")
    return value


def _samples(value: Any, *, limits: ResourceLimits) -> List[float]:
    if not isinstance(value, list) or not value:
        raise SchemaError("qualification observation samples_us must be a non-empty array")
    if len(value) > limits.max_execution_observation_samples:
        raise SchemaError(
            "qualification observation samples_us exceeds "
            f"max_execution_observation_samples={limits.max_execution_observation_samples}"
        )
    samples = [
        _finite_nonnegative(item, f"qualification observation samples_us[{index}]") for index, item in enumerate(value)
    ]
    if any(sample <= 0 for sample in samples):
        raise SchemaError("qualification observation samples_us must be positive")
    return samples


__all__ = [
    "QUALIFICATION_VERDICTS",
    "QUALIFICATION_REASON_CODES",
    "qualification_observation_sha256",
    "qualification_policy_sha256",
    "qualification_verdict_sha256",
    "validate_qualification_observation",
    "validate_qualification_policy",
    "validate_qualification_verdict",
]
