"""Replicated actual-application evidence across independent allocations."""

from __future__ import annotations

import copy
import hashlib
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..errors import SchemaError
from ..formats import (
    APPLICATION_EVIDENCE_SET_FORMAT,
    APPLICATION_MEASUREMENT_FORMAT,
    APPLICATION_ORACLE_FORMAT,
)
from ..statistics import median
from .application_measurement import (
    application_regression_decision,
    application_workload_identity,
    validate_application_measurement,
)
from .json_codec import canonical_json_bytes


def build_application_oracle(
    allocations: Sequence[Mapping[str, Any]],
    *,
    minimum_allocations: int = 8,
    minimum_distinct_days: int = 2,
    maximum_allocation_elapsed_seconds: int = 7200,
) -> Dict[str, Any]:
    """Build one aggregate without collapsing its allocation-level evidence."""

    if not allocations:
        raise SchemaError("application oracle requires allocation evidence")
    first_measurement = _mapping(allocations[0].get("measurement"), "application oracle first measurement")
    validate_application_measurement(first_measurement)
    raw: Dict[str, Any] = {
        "format": APPLICATION_ORACLE_FORMAT,
        "runner": copy.deepcopy(first_measurement["runner"]),
        "subject_sha256": first_measurement["subject_sha256"],
        "subject_configuration": copy.deepcopy(first_measurement["subject_configuration"]),
        "perturbation_id": first_measurement["perturbation_id"],
        "application": copy.deepcopy(first_measurement["application"]),
        "workload": copy.deepcopy(first_measurement["workload"]),
        "allocations": [copy.deepcopy(dict(row)) for row in allocations],
        "summary": _aggregate_summary(
            [_mapping(row.get("measurement"), "application oracle measurement") for row in allocations]
        ),
        "environment_sha256": _environment_identity(allocations),
        "allocation_policy": {
            "minimum_allocations": minimum_allocations,
            "minimum_distinct_days": minimum_distinct_days,
            "maximum_allocation_elapsed_seconds": maximum_allocation_elapsed_seconds,
        },
    }
    raw["allocation_assessment"] = _allocation_assessment(raw)
    raw["oracle_id"] = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    validate_application_oracle(raw)
    return raw


def validate_application_oracle(oracle: Mapping[str, Any]) -> None:
    """Recompute allocation bindings, aggregate metrics, and comparability."""

    _closed(
        oracle,
        {
            "format",
            "oracle_id",
            "runner",
            "subject_sha256",
            "subject_configuration",
            "perturbation_id",
            "application",
            "workload",
            "allocations",
            "summary",
            "environment_sha256",
            "allocation_policy",
            "allocation_assessment",
        },
        "application oracle",
    )
    if oracle.get("format") != APPLICATION_ORACLE_FORMAT:
        raise SchemaError("application oracle format is unsupported")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in oracle.items() if key != "oracle_id"})
    ).hexdigest()
    if oracle.get("oracle_id") != expected_id:
        raise SchemaError("application oracle_id does not match canonical content")
    policy = _mapping(oracle.get("allocation_policy"), "application oracle allocation policy")
    _closed(
        policy,
        {
            "minimum_allocations",
            "minimum_distinct_days",
            "maximum_allocation_elapsed_seconds",
        },
        "application oracle allocation policy",
    )
    for name in (
        "minimum_allocations",
        "minimum_distinct_days",
        "maximum_allocation_elapsed_seconds",
    ):
        _positive_integer(policy.get(name), f"application oracle {name}")

    allocations = oracle.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise SchemaError("application oracle allocations must be a non-empty array")
    repetitions: List[int] = []
    planned_positions: List[int] = []
    allocation_ids: List[str] = []
    measurement_ids: List[str] = []
    first_identity: Dict[str, Any] | None = None
    for index, raw_row in enumerate(allocations):
        row = _mapping(raw_row, f"application oracle allocations[{index}]")
        _closed(
            row,
            {
                "repetition",
                "planned_position",
                "chunk_id",
                "allocation_id",
                "node",
                "start_time",
                "end_time",
                "elapsed_seconds",
                "measurement",
            },
            f"application oracle allocations[{index}]",
        )
        repetition = _nonnegative_integer(row.get("repetition"), "application oracle repetition")
        planned_position = _nonnegative_integer(
            row.get("planned_position"),
            "application oracle planned position",
        )
        for field in ("chunk_id", "allocation_id", "node"):
            _nonempty(row.get(field), f"application oracle {field}")
        start = _timestamp(row.get("start_time"), "application oracle start_time")
        end = _timestamp(row.get("end_time"), "application oracle end_time")
        if end < start:
            raise SchemaError("application oracle allocation ends before it starts")
        elapsed = _positive_integer(row.get("elapsed_seconds"), "application oracle elapsed_seconds")
        measurement = _mapping(row.get("measurement"), "application oracle measurement")
        validate_application_measurement(measurement)
        environment = _mapping(measurement.get("environment"), "application oracle measurement environment")
        if environment.get("slurm_job_id") != row.get("allocation_id"):
            raise SchemaError("application oracle allocation ID does not match measurement SLURM job ID")
        if environment.get("slurm_node") != row.get("node"):
            raise SchemaError("application oracle node does not match measurement SLURM node")
        identity = {
            "runner": measurement["runner"],
            "subject_sha256": measurement["subject_sha256"],
            "subject_configuration": measurement["subject_configuration"],
            "perturbation_id": measurement["perturbation_id"],
            "application": measurement["application"],
            "workload": measurement["workload"],
        }
        if first_identity is None:
            first_identity = copy.deepcopy(identity)
        elif identity != first_identity:
            raise SchemaError("application oracle allocations do not measure one exact subject and workload")
        repetitions.append(repetition)
        planned_positions.append(planned_position)
        allocation_ids.append(str(row["allocation_id"]))
        measurement_ids.append(str(measurement["measurement_id"]))
        if elapsed > int(policy["maximum_allocation_elapsed_seconds"]):
            # The assessment below records the issue. Validation still accepts
            # the evidence so the failed comparison remains inspectable.
            pass
    if repetitions != list(range(len(repetitions))):
        raise SchemaError("application oracle repetitions must be contiguous canonical integers")
    if sorted(planned_positions) != list(range(len(planned_positions))):
        raise SchemaError("application oracle must occupy each planned position exactly once")
    if len(allocation_ids) != len(set(allocation_ids)):
        raise SchemaError("application oracle allocation IDs must be unique")
    if len(measurement_ids) != len(set(measurement_ids)):
        raise SchemaError("application oracle measurement IDs must be unique")
    if first_identity is None:
        raise SchemaError("application oracle identity is missing")
    for field, value in first_identity.items():
        if oracle.get(field) != value:
            raise SchemaError(f"application oracle top-level {field} does not match its allocations")
    if oracle.get("summary") != _aggregate_summary(
        [_mapping(row["measurement"], "application oracle measurement") for row in allocations]
    ):
        raise SchemaError("application oracle summary does not recompute")
    if oracle.get("environment_sha256") != _environment_identity(allocations):
        raise SchemaError("application oracle environment_sha256 does not recompute")
    if oracle.get("allocation_assessment") != _allocation_assessment(oracle):
        raise SchemaError("application oracle allocation assessment does not recompute")


def validate_application_evidence(evidence: Mapping[str, Any]) -> None:
    """Validate either one allocation or a replicated application oracle."""

    if evidence.get("format") == APPLICATION_MEASUREMENT_FORMAT:
        validate_application_measurement(evidence)
        return
    if evidence.get("format") == APPLICATION_ORACLE_FORMAT:
        validate_application_oracle(evidence)
        return
    raise SchemaError("application evidence format is unsupported")


def application_evidence_comparable(evidence: Mapping[str, Any]) -> bool:
    validate_application_evidence(evidence)
    if evidence.get("format") == APPLICATION_MEASUREMENT_FORMAT:
        assessment = _mapping(evidence.get("telemetry_assessment"), "application telemetry assessment")
        return assessment.get("comparable") is True
    assessment = _mapping(evidence.get("allocation_assessment"), "application allocation assessment")
    return assessment.get("comparable") is True


def application_evidence_id(evidence: Mapping[str, Any]) -> str:
    validate_application_evidence(evidence)
    return str(
        evidence["measurement_id"]
        if evidence.get("format") == APPLICATION_MEASUREMENT_FORMAT
        else evidence["oracle_id"]
    )


def application_evidence_regression_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    threshold_pct: float,
) -> Dict[str, Any]:
    """Apply the same SLO rule to single or replicated application evidence."""

    validate_application_evidence(baseline)
    validate_application_evidence(candidate)
    if baseline.get("format") == candidate.get("format") == APPLICATION_MEASUREMENT_FORMAT:
        return application_regression_decision(baseline, candidate, threshold_pct=threshold_pct)
    if baseline.get("perturbation_id") == candidate.get("perturbation_id"):
        raise SchemaError("application baseline and candidate perturbation identities must differ")
    if baseline.get("subject_sha256") == candidate.get("subject_sha256"):
        raise SchemaError("application baseline and candidate stack subjects must differ")
    if application_workload_identity(baseline) != application_workload_identity(candidate):
        raise SchemaError("application baseline and candidate must execute the same application workload")
    if not isinstance(threshold_pct, (int, float)) or isinstance(threshold_pct, bool):
        raise SchemaError("application regression threshold must be numeric")
    threshold = float(threshold_pct)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise SchemaError("application regression threshold must be finite and positive")
    baseline_summary = _mapping(baseline.get("summary"), "application baseline summary")
    candidate_summary = _mapping(candidate.get("summary"), "application candidate summary")
    metrics = []
    magnitudes = []
    for name, direction in (
        ("p99_batch_latency_ms", "lower_is_better"),
        ("median_output_token_throughput_per_second", "higher_is_better"),
    ):
        baseline_value = float(baseline_summary[name])
        candidate_value = float(candidate_summary[name])
        difference = (
            candidate_value - baseline_value if direction == "lower_is_better" else baseline_value - candidate_value
        )
        magnitude = max(0.0, difference / baseline_value * 100.0)
        magnitudes.append(magnitude)
        metrics.append(
            {
                "name": name,
                "direction": direction,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "regression_pct": magnitude,
                "failed": magnitude > threshold,
            }
        )
    maximum = max(magnitudes)
    return {
        "decision": "fail" if any(row["failed"] for row in metrics) else "pass",
        "regression_magnitude_pct": maximum,
        "severe": maximum >= threshold and any(row["failed"] for row in metrics),
        "metrics": metrics,
    }


def build_application_evidence_set(
    *,
    training_baseline: Mapping[str, Any],
    training_perturbations: Mapping[str, Mapping[str, Any]],
    holdout_baseline: Mapping[str, Any],
    holdout_perturbations: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the complete application-oracle evidence opened by active synthesis."""

    validate_application_evidence(training_baseline)
    runner_oci_digest = str(_mapping(training_baseline.get("runner"), "application runner")["oci_digest"])
    raw: Dict[str, Any] = {
        "format": APPLICATION_EVIDENCE_SET_FORMAT,
        "runner_oci_digest": runner_oci_digest,
        "splits": {
            "training": _evidence_split(training_baseline, training_perturbations),
            "holdout": _evidence_split(holdout_baseline, holdout_perturbations),
        },
    }
    raw["evidence_set_id"] = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    validate_application_evidence_set(raw)
    return raw


def validate_application_evidence_set(evidence_set: Mapping[str, Any]) -> None:
    """Recompute a closed training/holdout application evidence inventory."""

    _closed(
        evidence_set,
        {"format", "evidence_set_id", "runner_oci_digest", "splits"},
        "application evidence set",
    )
    if evidence_set.get("format") != APPLICATION_EVIDENCE_SET_FORMAT:
        raise SchemaError("application evidence set format is unsupported")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in evidence_set.items() if key != "evidence_set_id"})
    ).hexdigest()
    if evidence_set.get("evidence_set_id") != expected_id:
        raise SchemaError("application evidence_set_id does not match canonical content")
    runner_oci_digest = _nonempty(evidence_set.get("runner_oci_digest"), "application evidence runner digest")
    if not runner_oci_digest.startswith("sha256:") or len(runner_oci_digest) != 71:
        raise SchemaError("application evidence runner digest must be an OCI SHA-256 digest")
    digest = runner_oci_digest.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in digest):
        raise SchemaError("application evidence runner digest must be an OCI SHA-256 digest")

    splits = _mapping(evidence_set.get("splits"), "application evidence splits")
    _closed(splits, {"training", "holdout"}, "application evidence splits")
    baseline_identity: Dict[str, Any] | None = None
    baseline_subject: str | None = None
    perturbation_ids: set[str] = set()
    perturbation_subjects: set[str] = set()
    for split_name in ("training", "holdout"):
        split = _mapping(splits.get(split_name), f"application evidence {split_name} split")
        _closed(split, {"baseline", "perturbations"}, f"application evidence {split_name} split")
        baseline = _mapping(split.get("baseline"), f"application evidence {split_name} baseline")
        validate_application_evidence(baseline)
        if not application_evidence_comparable(baseline):
            raise SchemaError(f"application evidence {split_name} baseline is environmentally incomparable")
        _evidence_runner(baseline, runner_oci_digest)
        identity = application_workload_identity(baseline)
        subject = str(baseline.get("subject_sha256"))
        if baseline_identity is None:
            baseline_identity = identity
            baseline_subject = subject
        elif identity != baseline_identity or subject != baseline_subject:
            raise SchemaError("application evidence split baselines do not bind one workload and subject")

        rows = split.get("perturbations")
        if not isinstance(rows, list) or not rows:
            raise SchemaError(f"application evidence {split_name} perturbations must be a non-empty array")
        ids_in_order: List[str] = []
        for index, raw_row in enumerate(rows):
            row = _mapping(raw_row, f"application evidence {split_name} perturbations[{index}]")
            _closed(row, {"perturbation_id", "evidence"}, "application evidence perturbation row")
            perturbation_id = _nonempty(row.get("perturbation_id"), "application evidence perturbation_id")
            evidence = _mapping(row.get("evidence"), "application perturbation evidence")
            validate_application_evidence(evidence)
            if not application_evidence_comparable(evidence):
                raise SchemaError(f"application evidence perturbation {perturbation_id!r} is incomparable")
            if evidence.get("perturbation_id") != perturbation_id:
                raise SchemaError("application evidence perturbation ID does not match its evidence")
            _evidence_runner(evidence, runner_oci_digest)
            if application_workload_identity(evidence) != baseline_identity:
                raise SchemaError("application evidence perturbations do not execute one exact workload")
            subject_sha256 = str(evidence.get("subject_sha256"))
            if subject_sha256 == baseline_subject or subject_sha256 in perturbation_subjects:
                raise SchemaError("application evidence perturbation subjects must be globally unique")
            if perturbation_id in perturbation_ids:
                raise SchemaError("application evidence perturbation IDs must be globally unique")
            perturbation_ids.add(perturbation_id)
            perturbation_subjects.add(subject_sha256)
            ids_in_order.append(perturbation_id)
        if ids_in_order != sorted(ids_in_order):
            raise SchemaError("application evidence perturbations must use canonical ID order")


def _evidence_split(
    baseline: Mapping[str, Any],
    perturbations: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if not perturbations:
        raise SchemaError("application evidence split requires perturbations")
    return {
        "baseline": copy.deepcopy(dict(baseline)),
        "perturbations": [
            {
                "perturbation_id": perturbation_id,
                "evidence": copy.deepcopy(dict(perturbations[perturbation_id])),
            }
            for perturbation_id in sorted(perturbations)
        ],
    }


def _evidence_runner(evidence: Mapping[str, Any], expected: str) -> None:
    runner = _mapping(evidence.get("runner"), "application evidence runner")
    if runner.get("oci_digest") != expected:
        raise SchemaError("application evidence runner does not match the evidence set")


def _aggregate_summary(measurements: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    latencies: List[float] = []
    output_throughput: List[float] = []
    total_throughput: List[float] = []
    memory: List[int] = []
    for measurement in measurements:
        validate_application_measurement(measurement)
        for raw_sample in measurement["samples"]:
            sample = _mapping(raw_sample, "application oracle sample")
            latencies.append(float(sample["batch_latency_ms"]))
            output_throughput.append(float(sample["output_token_throughput_per_second"]))
            total_throughput.append(float(sample["total_token_throughput_per_second"]))
            memory.append(int(sample["observed_gpu_memory_used_bytes"]))
    return {
        "median_batch_latency_ms": median(latencies),
        "p99_batch_latency_ms": _percentile(latencies, 0.99),
        "median_output_token_throughput_per_second": median(output_throughput),
        "median_total_token_throughput_per_second": median(total_throughput),
        "max_observed_gpu_memory_used_bytes": max(memory),
    }


def _environment_identity(allocations: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for raw_row in allocations:
        row = _mapping(raw_row, "application oracle allocation")
        measurement = _mapping(row.get("measurement"), "application oracle measurement")
        rows.append(
            {
                "repetition": row.get("repetition"),
                "planned_position": row.get("planned_position"),
                "allocation_id": row.get("allocation_id"),
                "node": row.get("node"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "environment_sha256": measurement.get("environment_sha256"),
            }
        )
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _allocation_assessment(oracle: Mapping[str, Any]) -> Dict[str, Any]:
    allocations = oracle.get("allocations")
    policy = _mapping(oracle.get("allocation_policy"), "application oracle allocation policy")
    if not isinstance(allocations, list):
        raise SchemaError("application oracle allocations must be an array")
    issues: List[str] = []
    nodes = set()
    days = set()
    elapsed_values = []
    for raw_row in allocations:
        row = _mapping(raw_row, "application oracle allocation")
        measurement = _mapping(row.get("measurement"), "application oracle measurement")
        assessment = _mapping(measurement.get("telemetry_assessment"), "application telemetry assessment")
        if assessment.get("comparable") is not True:
            issues.append(f"repetition-{row.get('repetition')}:telemetry-incomparable")
        nodes.add(str(row.get("node")))
        days.add(_timestamp(row.get("start_time"), "application oracle start_time").date().isoformat())
        elapsed_values.append(_positive_integer(row.get("elapsed_seconds"), "application oracle elapsed_seconds"))
    if len(allocations) < int(policy["minimum_allocations"]):
        issues.append("insufficient-independent-allocations")
    if len(days) < int(policy["minimum_distinct_days"]):
        issues.append("insufficient-distinct-days")
    if elapsed_values and max(elapsed_values) > int(policy["maximum_allocation_elapsed_seconds"]):
        issues.append("allocation-time-window-exceeded")
    return {
        "comparable": not issues,
        "issues": sorted(issues),
        "allocation_count": len(allocations),
        "distinct_node_count": len(nodes),
        "distinct_day_count": len(days),
        "maximum_allocation_elapsed_seconds": max(elapsed_values) if elapsed_values else 0,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timestamp(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError(f"{label} must be an ISO-8601 timestamp") from exc
    return parsed


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if set(value) != set(expected):
        raise SchemaError(f"{label} fields are not closed")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{label} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SchemaError(f"{label} must be a positive integer")
    return value


__all__ = [
    "application_evidence_comparable",
    "application_evidence_id",
    "application_evidence_regression_decision",
    "build_application_evidence_set",
    "build_application_oracle",
    "validate_application_evidence",
    "validate_application_evidence_set",
    "validate_application_oracle",
]
