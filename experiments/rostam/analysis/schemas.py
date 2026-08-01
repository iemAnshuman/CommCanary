"""Runtime validators for the committed local experiment schemas."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

from ..harness import CELL_RESULT_SCHEMA, ContractError, file_sha256, strict_json_loads

LOCAL_PREPARE_MEASUREMENT_SCHEMA = "commcanary.experiment.local.prepare-measurement.v1"
LOCAL_CONSUME_MEASUREMENT_SCHEMA = "commcanary.experiment.local.consume-measurement.v1"
LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA = "commcanary.experiment.local.fail-once-measurement.v1"
PHYSICAL_MICRO_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.micro-measurement.v1"
PHYSICAL_FULL_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.full-measurement.v1"
PHYSICAL_PARAM_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.param-measurement.v1"
PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.overlap-measurement.v1"
PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.capture-measurement.v1"
PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.qualification-measurement.v1"
PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA = "commcanary.rostam.physical.decision-gate-measurement.v1"
RAW_ARCHIVE_DESCRIPTOR_SCHEMA = "commcanary.rostam.raw-archive-descriptor.v1"
CROSS_COMMIT_COMPATIBILITY_SCHEMA = "commcanary.rostam.cross-commit-compatibility.v1"
DECISION_FIDELITY_POLICY_SCHEMA = "commcanary.rostam.decision-fidelity-policy.v1"
DECISION_FIDELITY_VERDICT_SCHEMA = "commcanary.rostam.decision-fidelity-verdict.v1"

_SCHEMA_DIRECTORY = Path(__file__).resolve().parent.parent / "schemas"
_SCHEMA_FILES = {
    CELL_RESULT_SCHEMA: "cell-result-v1.schema.json",
    LOCAL_PREPARE_MEASUREMENT_SCHEMA: "local-prepare-measurement-v1.schema.json",
    LOCAL_CONSUME_MEASUREMENT_SCHEMA: "local-consume-measurement-v1.schema.json",
    LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA: "local-fail-once-measurement-v1.schema.json",
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA: "physical-micro-measurement-v1.schema.json",
    PHYSICAL_FULL_MEASUREMENT_SCHEMA: "physical-full-measurement-v1.schema.json",
    PHYSICAL_PARAM_MEASUREMENT_SCHEMA: "physical-param-measurement-v1.schema.json",
    PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA: "physical-overlap-measurement-v1.schema.json",
    PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA: "physical-capture-measurement-v1.schema.json",
    PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA: "physical-qualification-measurement-v1.schema.json",
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA: "physical-decision-gate-measurement-v1.schema.json",
    RAW_ARCHIVE_DESCRIPTOR_SCHEMA: "raw-archive-descriptor-v1.schema.json",
    CROSS_COMMIT_COMPATIBILITY_SCHEMA: "cross-commit-compatibility-v1.schema.json",
    DECISION_FIDELITY_POLICY_SCHEMA: "decision-fidelity-policy-v1.schema.json",
    DECISION_FIDELITY_VERDICT_SCHEMA: "decision-fidelity-verdict-v1.schema.json",
}
_PHYSICAL_PRODUCER_CONTRACTS = {
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.micro-producer.v1",
    PHYSICAL_FULL_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.full-producer.v1",
    PHYSICAL_PARAM_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.param-producer.v1",
    PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.overlap-producer.v1",
    PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.capture-producer.v1",
    PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.qualification-producer.v1",
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.decision-gate-producer.v1",
}
_PRODUCER_CONTRACTS = {
    LOCAL_PREPARE_MEASUREMENT_SCHEMA: ("commcanary.experiment.prepare.v1", "success"),
    LOCAL_CONSUME_MEASUREMENT_SCHEMA: ("commcanary.experiment.consume.v1", "success"),
    LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA: ("commcanary.experiment.fail-once.v1", "fail-once"),
}
_MEASUREMENT_FIELDS = {
    "attempt_id",
    "config_value",
    "mode",
    "samples_us",
    "secret_present",
    "value_us",
}
_PHYSICAL_COMMON_FIELDS = {
    "attempt_id",
    "count",
    "global_ranks",
    "iqr_us",
    "operation",
    "runtime",
    "samples_us",
    "value_us",
    "wall_time_s",
    "world_size",
}
_PHYSICAL_SPECIFIC_FIELDS = {
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA: {"dtype", "message_sizes_bytes"},
    PHYSICAL_FULL_MEASUREMENT_SCHEMA: {"dtype", "gemm_m", "gemm_n", "hidden", "layers", "tokens"},
    PHYSICAL_PARAM_MEASUREMENT_SCHEMA: {"replay_mode", "trace_sha256"},
    PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA: {"replay_mode", "trace_sha256"},
    PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA: {"artifacts"},
    PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA: {
        "absolute_relative_median_error_pct",
        "comparison_claims",
        "replay_mode",
        "request_id",
        "materialization_id",
        "program_sha256",
        "correctness_check_count",
        "rank_compute_operations_per_pass",
        "rank_tensor_bytes",
        "relative_median_error_pct",
        "signed_median_error_us",
        "source_capture_diagnostic_id",
        "source_capture_evidence_sha256",
        "source_capture_job_id",
        "source_capture_node",
        "source_capture_stdout_sha256",
        "source_iqr_us",
        "source_samples_us",
        "source_timing_semantics",
        "source_value_us",
    },
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA: {
        "correctness_check_count",
        "decision_claims",
        "execution",
        "materialization",
        "policy",
        "representations",
        "request",
    },
}
_PARAM_REPLAY_MODES = {"timestamp-paced-blocking", "compute-filled-blocking"}
_OVERLAP_REPLAY_MODES = {"explicit-wait-overlap", "fixed-input-explicit-wait-overlap"}
_ARTIFACT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECISION_GATE_REPRESENTATIONS = (
    "source",
    "exact_work",
    "stratified",
    "isolated",
    "no_overlap",
    "no_rank_skew",
)
_DECISION_GATE_REPRESENTATION_CONTRACTS = {
    "source": ("ground_truth", "direct-source-issue-rank-work-wait"),
    "exact_work": ("product_candidate", "verified-materialization-issue-rank-work-wait"),
    "stratified": ("kill_condition_baseline", "first-observed-per-collective-shape.v1"),
    "isolated": ("incumbent_baseline", "full-message-sequence-blocking-all-reduce-no-compute"),
    "no_overlap": ("causal_ablation", "blocking-all-reduce-then-exact-rank-work"),
    "no_rank_skew": ("causal_ablation", "issue-rank-zero-work-on-every-rank-wait"),
}


class MeasurementValidationError(ContractError):
    """Raised when a selected result does not satisfy its declared schema."""


@dataclass(frozen=True)
class PhysicalRuntime:
    hostname: str
    job_id: Optional[str]
    python_version: str
    runtime_nccl_version_code: int
    torch_cuda_version: Optional[str]
    torch_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "job_id": self.job_id,
            "python_version": self.python_version,
            "runtime_nccl_version_code": self.runtime_nccl_version_code,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_version": self.torch_version,
        }


@dataclass(frozen=True)
class PhysicalArtifact:
    artifact_id: str
    path: str
    sha256: str
    size_bytes: int

    def to_reference(self) -> Dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class PhysicalMeasurement:
    operation: str
    world_size: int
    global_ranks: Tuple[int, ...]
    iqr_us: float
    wall_time_s: float
    runtime: PhysicalRuntime
    attributes: Mapping[str, Any]
    artifacts: Tuple[PhysicalArtifact, ...]


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise MeasurementValidationError(f"{field} must be finite and non-negative")
    return number


def _finite_signed_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MeasurementValidationError(f"{field} must be finite")
    return number


def _median(values: Tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _iqr(values: Tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        lower = tuple(ordered[:middle])
        upper = tuple(ordered[middle + 1 :])
    else:
        lower = tuple(ordered[:middle])
        upper = tuple(ordered[middle:])
    if not lower or not upper:
        return 0.0
    return _median(upper) - _median(lower)


def _strict_object(raw: Any, field: str, expected_fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise MeasurementValidationError(f"{field} does not match its closed schema")
    return raw


def _decision_gate_attributes(
    raw: Mapping[str, Any],
    *,
    world_size: int,
    samples: Tuple[float, ...],
) -> Dict[str, Any]:
    request = _strict_object(raw["request"], "measurement.request", {"format", "request_id"})
    if request["format"] != "commcanary.qualification_request.v2":
        raise MeasurementValidationError("decision-gate request format is unsupported")
    materialization = _strict_object(
        raw["materialization"],
        "measurement.materialization",
        {"materialization_id", "program_sha256"},
    )
    policy = _strict_object(raw["policy"], "measurement.policy", {"format", "policy_id"})
    if policy["format"] != "commcanary.qualification_policy.v1":
        raise MeasurementValidationError("decision-gate policy format is unsupported")
    for field, value in (
        ("measurement.request.request_id", request["request_id"]),
        ("measurement.materialization.materialization_id", materialization["materialization_id"]),
        ("measurement.materialization.program_sha256", materialization["program_sha256"]),
        ("measurement.policy.policy_id", policy["policy_id"]),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise MeasurementValidationError(f"{field} is invalid")

    execution = _strict_object(
        raw["execution"],
        "measurement.execution",
        {
            "iterations",
            "order_method",
            "representation_order_by_iteration",
            "source_event_count",
            "stratified_method",
            "stratified_source_event_indices",
            "timing_semantics",
            "warmup",
            "world_size",
        },
    )
    iterations = execution["iterations"]
    warmup = execution["warmup"]
    event_count = execution["source_event_count"]
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations != len(samples)
        or not 1 <= iterations <= 1000
        or isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or not 0 <= warmup <= 100
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count <= 0
        or execution["world_size"] != world_size
        or execution["timing_semantics"] != "maximum-rank-cuda-event-whole-program-duration"
        or execution["order_method"] != "iteration-rotated-latin-cycle.v1"
        or execution["stratified_method"] != "first-observed-per-collective-shape.v1"
    ):
        raise MeasurementValidationError("decision-gate execution contract is inconsistent")
    selected = execution["stratified_source_event_indices"]
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < event_count for index in selected
        )
    ):
        raise MeasurementValidationError("decision-gate stratified source selection is invalid")
    orders = execution["representation_order_by_iteration"]
    if not isinstance(orders, list) or len(orders) != iterations:
        raise MeasurementValidationError("decision-gate representation order inventory is incomplete")
    for iteration, order in enumerate(orders):
        offset = iteration % len(_DECISION_GATE_REPRESENTATIONS)
        expected = list(_DECISION_GATE_REPRESENTATIONS[offset:] + _DECISION_GATE_REPRESENTATIONS[:offset])
        if order != expected:
            raise MeasurementValidationError(f"decision-gate representation order is invalid at iteration {iteration}")

    check_count = raw["correctness_check_count"]
    if isinstance(check_count, bool) or not isinstance(check_count, int) or check_count <= 0:
        raise MeasurementValidationError("decision-gate correctness_check_count must be positive")
    representations = _strict_object(
        raw["representations"],
        "measurement.representations",
        set(_DECISION_GATE_REPRESENTATIONS),
    )
    normalized_representations: Dict[str, Any] = {}
    for representation in _DECISION_GATE_REPRESENTATIONS:
        field = f"measurement.representations.{representation}"
        value = _strict_object(
            representations[representation],
            field,
            {
                "category",
                "executed_event_count",
                "metrics",
                "rank_timings_us",
                "semantics",
                "template_count",
                "timings_us",
            },
        )
        expected_category, expected_semantics = _DECISION_GATE_REPRESENTATION_CONTRACTS[representation]
        expected_event_count = len(selected) if representation == "stratified" else event_count
        expected_template_count = len(selected) if representation in {"stratified", "isolated"} else event_count
        if (
            value["category"] != expected_category
            or value["semantics"] != expected_semantics
            or value["executed_event_count"] != expected_event_count
            or value["template_count"] != expected_template_count
        ):
            raise MeasurementValidationError(f"{field} execution semantics are inconsistent")
        rank_timings_raw = value["rank_timings_us"]
        if not isinstance(rank_timings_raw, list) or len(rank_timings_raw) != world_size:
            raise MeasurementValidationError(f"{field}.rank_timings_us does not cover every rank")
        rank_timings = tuple(
            tuple(_finite_number(item, f"{field}.rank_timings_us[{rank}][{index}]") for index, item in enumerate(row))
            if isinstance(row, list)
            else ()
            for rank, row in enumerate(rank_timings_raw)
        )
        if any(len(row) != iterations for row in rank_timings):
            raise MeasurementValidationError(f"{field}.rank_timings_us has the wrong sample count")
        timings_raw = value["timings_us"]
        if not isinstance(timings_raw, list):
            raise MeasurementValidationError(f"{field}.timings_us must be an array")
        timings = tuple(_finite_number(item, f"{field}.timings_us[{index}]") for index, item in enumerate(timings_raw))
        maxima = tuple(max(rank_timings[rank][index] for rank in range(world_size)) for index in range(iterations))
        if len(timings) != iterations or timings != maxima:
            raise MeasurementValidationError(f"{field}.timings_us disagrees with max-rank timings")
        metrics = _strict_object(
            value["metrics"],
            f"{field}.metrics",
            {"count", "iqr_us", "max_us", "median_us", "min_us"},
        )
        expected_metrics = {
            "count": len(timings),
            "iqr_us": _iqr(timings),
            "max_us": max(timings),
            "median_us": _median(timings),
            "min_us": min(timings),
        }
        if metrics["count"] != expected_metrics["count"]:
            raise MeasurementValidationError(f"{field}.metrics.count is inconsistent")
        for metric_name in ("iqr_us", "max_us", "median_us", "min_us"):
            observed = _finite_number(metrics[metric_name], f"{field}.metrics.{metric_name}")
            if abs(observed - float(expected_metrics[metric_name])) > 0.001:
                raise MeasurementValidationError(f"{field}.metrics.{metric_name} is inconsistent")
        normalized_representations[representation] = dict(value)
    if tuple(normalized_representations["source"]["timings_us"]) != samples:
        raise MeasurementValidationError("decision-gate scalar samples must be the source representation")

    claims = _strict_object(
        raw["decision_claims"],
        "measurement.decision_claims",
        {"physical_decision_fidelity", "physical_execution", "qualification_verdict"},
    )
    expected_claims = {
        "physical_execution": "same_allocation_self_reported",
        "physical_decision_fidelity": "not_analyzed",
        "qualification_verdict": "policy_bound_not_issued",
    }
    if dict(claims) != expected_claims:
        raise MeasurementValidationError("decision-gate claims exceed the pre-analysis boundary")
    return {
        "request": dict(request),
        "materialization": dict(materialization),
        "policy": dict(policy),
        "execution": dict(execution),
        "correctness_check_count": check_count,
        "representations": normalized_representations,
        "decision_claims": expected_claims,
    }


def _physical_measurement(
    schema: str,
    producer_schema: str,
    attempt_id: str,
    raw: Any,
) -> ScalarMeasurement:
    expected_producer = _PHYSICAL_PRODUCER_CONTRACTS[schema]
    if producer_schema != expected_producer:
        raise MeasurementValidationError(f"measurement schema {schema!r} requires producer {expected_producer!r}")
    if not isinstance(raw, Mapping):
        raise MeasurementValidationError("measurement must be an object")
    expected_fields = _PHYSICAL_COMMON_FIELDS | _PHYSICAL_SPECIFIC_FIELDS[schema]
    missing = sorted(expected_fields - set(raw))
    unknown = sorted(set(raw) - expected_fields)
    if missing:
        raise MeasurementValidationError(f"measurement is missing required fields: {', '.join(missing)}")
    if unknown:
        raise MeasurementValidationError(f"measurement has unknown fields: {', '.join(unknown)}")
    if raw["attempt_id"] != attempt_id:
        raise MeasurementValidationError("measurement attempt_id does not match selected attempt")
    if raw["operation"] != "all_reduce":
        raise MeasurementValidationError("physical measurement operation must be all_reduce")
    world_size = raw["world_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int) or not 1 <= world_size <= 1024:
        raise MeasurementValidationError("measurement.world_size must be an integer in [1, 1024]")
    ranks_raw = raw["global_ranks"]
    if not isinstance(ranks_raw, list) or any(
        isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks_raw
    ):
        raise MeasurementValidationError("measurement.global_ranks must contain integers")
    ranks = tuple(cast(int, rank) for rank in ranks_raw)
    if ranks != tuple(range(world_size)):
        raise MeasurementValidationError("physical measurement must declare the dense world process group")
    samples_raw = raw["samples_us"]
    if not isinstance(samples_raw, list) or not 1 <= len(samples_raw) <= 1_000_000:
        raise MeasurementValidationError("measurement.samples_us must contain 1..1000000 values")
    samples = tuple(
        _finite_number(value, f"measurement.samples_us[{index}]") for index, value in enumerate(samples_raw)
    )
    count = raw["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(samples):
        raise MeasurementValidationError("measurement.count must equal the number of samples")
    value_us = _finite_number(raw["value_us"], "measurement.value_us")
    if value_us != _median(samples):
        raise MeasurementValidationError("measurement.value_us must equal the median of measurement.samples_us")
    iqr_us = _finite_number(raw["iqr_us"], "measurement.iqr_us")
    if iqr_us != _iqr(samples):
        raise MeasurementValidationError("measurement.iqr_us must equal the IQR of measurement.samples_us")
    wall_time_s = _finite_number(raw["wall_time_s"], "measurement.wall_time_s")
    runtime = raw["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "hostname",
        "job_id",
        "python_version",
        "runtime_nccl_version_code",
        "torch_cuda_version",
        "torch_version",
    }:
        raise MeasurementValidationError("measurement.runtime does not match the physical runtime schema")
    if not isinstance(runtime["hostname"], str) or not runtime["hostname"]:
        raise MeasurementValidationError("measurement.runtime.hostname must be non-empty")
    for field in ("python_version", "torch_version"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise MeasurementValidationError(f"measurement.runtime.{field} must be non-empty")
    for field in ("job_id", "torch_cuda_version"):
        if runtime[field] is not None and not isinstance(runtime[field], str):
            raise MeasurementValidationError(f"measurement.runtime.{field} must be string or null")
    nccl_code = runtime["runtime_nccl_version_code"]
    if isinstance(nccl_code, bool) or not isinstance(nccl_code, int) or not 1 <= nccl_code <= 99_999:
        raise MeasurementValidationError("measurement.runtime.runtime_nccl_version_code is invalid")
    physical_attributes: Dict[str, Any] = {}
    physical_artifacts: Tuple[PhysicalArtifact, ...] = ()
    if schema == PHYSICAL_MICRO_MEASUREMENT_SCHEMA:
        if raw["dtype"] not in {"bf16", "fp16", "fp32"}:
            raise MeasurementValidationError("physical micro dtype is unsupported")
        sizes = raw["message_sizes_bytes"]
        if (
            not isinstance(sizes, list)
            or not sizes
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in sizes)
        ):
            raise MeasurementValidationError("physical micro message_sizes_bytes is invalid")
        physical_attributes = {"dtype": raw["dtype"], "message_sizes_bytes": list(sizes)}
    elif schema == PHYSICAL_FULL_MEASUREMENT_SCHEMA:
        if raw["dtype"] not in {"bf16", "fp16", "fp32"}:
            raise MeasurementValidationError("physical full dtype is unsupported")
        for field in ("gemm_m", "gemm_n", "hidden", "layers", "tokens"):
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MeasurementValidationError(f"physical full {field} must be a positive integer")
        physical_attributes = {
            "dtype": raw["dtype"],
            "gemm_m": raw["gemm_m"],
            "gemm_n": raw["gemm_n"],
            "hidden": raw["hidden"],
            "layers": raw["layers"],
            "tokens": raw["tokens"],
        }
    elif schema in {PHYSICAL_PARAM_MEASUREMENT_SCHEMA, PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA}:
        trace_sha256 = raw["trace_sha256"]
        if not isinstance(trace_sha256, str) or _SHA256_RE.fullmatch(trace_sha256) is None:
            raise MeasurementValidationError("physical replay trace_sha256 is invalid")
        expected_modes = _PARAM_REPLAY_MODES if schema == PHYSICAL_PARAM_MEASUREMENT_SCHEMA else _OVERLAP_REPLAY_MODES
        if raw["replay_mode"] not in expected_modes:
            raise MeasurementValidationError("physical replay mode is not allowed by its committed schema")
        physical_attributes = {"replay_mode": raw["replay_mode"], "trace_sha256": trace_sha256}
    elif schema == PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA:
        if len(samples) != 1 or count != 1 or iqr_us != 0.0:
            raise MeasurementValidationError("physical capture measurements require exactly one sample and zero IQR")
        artifacts = raw["artifacts"]
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise MeasurementValidationError("physical capture artifacts must be a non-empty object")
        parsed_artifacts: List[PhysicalArtifact] = []
        for artifact_id, reference in sorted(artifacts.items()):
            if not isinstance(artifact_id, str) or _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
                raise MeasurementValidationError("physical capture artifact id is invalid")
            if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256", "size_bytes"}:
                raise MeasurementValidationError("physical capture artifact reference is invalid")
            if not isinstance(reference["path"], str) or not reference["path"]:
                raise MeasurementValidationError("physical capture artifact path is invalid")
            digest = reference["sha256"]
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise MeasurementValidationError("physical capture artifact SHA-256 is invalid")
            size = reference["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise MeasurementValidationError("physical capture artifact size is invalid")
            parsed_artifacts.append(
                PhysicalArtifact(
                    artifact_id=artifact_id,
                    path=reference["path"],
                    sha256=digest,
                    size_bytes=size,
                )
            )
        physical_artifacts = tuple(parsed_artifacts)
    elif schema == PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA:
        if raw["replay_mode"] != "source-bound-exact-rank-work":
            raise MeasurementValidationError("physical qualification replay mode is unsupported")
        identities = {}
        for field in ("request_id", "materialization_id", "program_sha256"):
            value = raw[field]
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise MeasurementValidationError(f"physical qualification {field} is invalid")
            identities[field] = value
        correctness_check_count = raw["correctness_check_count"]
        if (
            isinstance(correctness_check_count, bool)
            or not isinstance(correctness_check_count, int)
            or correctness_check_count <= 0
        ):
            raise MeasurementValidationError("physical qualification correctness_check_count must be positive")
        rank_vectors: Dict[str, List[int]] = {}
        for field in ("rank_compute_operations_per_pass", "rank_tensor_bytes"):
            values = raw[field]
            if (
                not isinstance(values, list)
                or len(values) != world_size
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            ):
                raise MeasurementValidationError(
                    f"physical qualification {field} must cover every rank with non-negative integers"
                )
            rank_vectors[field] = list(values)
        source_samples_raw = raw["source_samples_us"]
        if not isinstance(source_samples_raw, list) or len(source_samples_raw) != len(samples):
            raise MeasurementValidationError(
                "physical qualification source_samples_us must match the replay sample count"
            )
        source_samples = tuple(
            _finite_number(value, f"measurement.source_samples_us[{index}]")
            for index, value in enumerate(source_samples_raw)
        )
        source_value_us = _finite_number(raw["source_value_us"], "measurement.source_value_us")
        if source_value_us <= 0.0 or source_value_us != _median(source_samples):
            raise MeasurementValidationError(
                "physical qualification source_value_us must equal the positive source median"
            )
        source_iqr_us = _finite_number(raw["source_iqr_us"], "measurement.source_iqr_us")
        if source_iqr_us != _iqr(source_samples):
            raise MeasurementValidationError("physical qualification source_iqr_us must equal the source sample IQR")
        source_identities = {}
        for field in ("source_capture_evidence_sha256", "source_capture_stdout_sha256"):
            value = raw[field]
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise MeasurementValidationError(f"physical qualification {field} is invalid")
            source_identities[field] = value
        for field in ("source_capture_diagnostic_id", "source_capture_job_id", "source_capture_node"):
            value = raw[field]
            if not isinstance(value, str) or not value or "\x00" in value:
                raise MeasurementValidationError(f"physical qualification {field} is invalid")
            source_identities[field] = value
        if raw["source_timing_semantics"] != "maximum-rank-unprofiled-whole-program-duration":
            raise MeasurementValidationError("physical qualification source timing semantics are unsupported")
        signed_error = _finite_signed_number(
            raw["signed_median_error_us"],
            "measurement.signed_median_error_us",
        )
        relative_error = _finite_signed_number(
            raw["relative_median_error_pct"],
            "measurement.relative_median_error_pct",
        )
        absolute_error = _finite_number(
            raw["absolute_relative_median_error_pct"],
            "measurement.absolute_relative_median_error_pct",
        )
        expected_signed = value_us - source_value_us
        expected_relative = expected_signed / source_value_us * 100.0
        if (
            signed_error != expected_signed
            or relative_error != expected_relative
            or absolute_error != abs(expected_relative)
        ):
            raise MeasurementValidationError("physical qualification comparison metrics are inconsistent")
        claims = raw["comparison_claims"]
        expected_claims = {
            "single_configuration_timing_comparison": "diagnostic",
            "physical_fidelity": "unproven",
            "multi_configuration_ranking": "not_measured",
            "qualification_verdict": "not_issued",
        }
        if not isinstance(claims, Mapping) or dict(claims) != expected_claims:
            raise MeasurementValidationError("physical qualification comparison claims exceed the diagnostic boundary")
        physical_attributes = {
            "replay_mode": raw["replay_mode"],
            **identities,
            "correctness_check_count": correctness_check_count,
            **rank_vectors,
            "source_samples_us": list(source_samples),
            "source_value_us": source_value_us,
            "source_iqr_us": source_iqr_us,
            "source_timing_semantics": raw["source_timing_semantics"],
            **source_identities,
            "signed_median_error_us": signed_error,
            "relative_median_error_pct": relative_error,
            "absolute_relative_median_error_pct": absolute_error,
            "comparison_claims": expected_claims,
        }
    elif schema == PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA:
        physical_attributes = _decision_gate_attributes(
            raw,
            world_size=world_size,
            samples=samples,
        )
    physical = PhysicalMeasurement(
        operation="all_reduce",
        world_size=world_size,
        global_ranks=ranks,
        iqr_us=iqr_us,
        wall_time_s=wall_time_s,
        runtime=PhysicalRuntime(
            hostname=runtime["hostname"],
            job_id=runtime["job_id"],
            python_version=runtime["python_version"],
            runtime_nccl_version_code=nccl_code,
            torch_cuda_version=runtime["torch_cuda_version"],
            torch_version=runtime["torch_version"],
        ),
        attributes=physical_attributes,
        artifacts=physical_artifacts,
    )
    return ScalarMeasurement(
        schema=schema,
        producer_schema=producer_schema,
        attempt_id=attempt_id,
        mode="physical",
        config_value=None,
        value_us=value_us,
        samples_us=samples,
        iqr_us=iqr_us,
        physical=physical,
    )


@dataclass(frozen=True)
class ScalarMeasurement:
    schema: str
    producer_schema: str
    attempt_id: str
    mode: str
    config_value: Optional[str]
    value_us: float
    samples_us: Tuple[float, ...]
    iqr_us: float
    physical: Optional[PhysicalMeasurement]

    def to_dict(self) -> Dict[str, Any]:
        physical = None
        if self.physical is not None:
            physical = {
                "operation": self.physical.operation,
                "world_size": self.physical.world_size,
                "global_ranks": list(self.physical.global_ranks),
                "iqr_us": self.physical.iqr_us,
                "wall_time_s": self.physical.wall_time_s,
                "runtime": self.physical.runtime.to_dict(),
                "attributes": dict(self.physical.attributes),
                "artifacts": [
                    {"artifact_id": artifact.artifact_id, **artifact.to_reference()}
                    for artifact in self.physical.artifacts
                ],
            }
        return {
            "schema": self.schema,
            "producer_schema": self.producer_schema,
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "config_value": self.config_value,
            "value_us": self.value_us,
            "samples_us": list(self.samples_us),
            "iqr_us": self.iqr_us,
            "physical": physical,
        }


def validate_scalar_measurement(
    schema: str,
    producer_schema: str,
    attempt_id: str,
    raw: Any,
) -> ScalarMeasurement:
    """Validate one producer-specific scalar measurement without coercion."""

    if schema in _PHYSICAL_PRODUCER_CONTRACTS:
        return _physical_measurement(schema, producer_schema, attempt_id, raw)
    if schema not in _PRODUCER_CONTRACTS:
        raise MeasurementValidationError(f"unsupported measurement schema {schema!r}")
    expected_producer, expected_mode = _PRODUCER_CONTRACTS[schema]
    if producer_schema != expected_producer:
        raise MeasurementValidationError(f"measurement schema {schema!r} requires producer {expected_producer!r}")
    if not isinstance(raw, Mapping):
        raise MeasurementValidationError("measurement must be an object")
    actual_fields = set(raw)
    missing = sorted(_MEASUREMENT_FIELDS - actual_fields)
    unknown = sorted(actual_fields - _MEASUREMENT_FIELDS)
    if missing:
        raise MeasurementValidationError(f"measurement is missing required fields: {', '.join(missing)}")
    if unknown:
        raise MeasurementValidationError(f"measurement has unknown fields: {', '.join(unknown)}")
    observed_attempt_id = raw["attempt_id"]
    if observed_attempt_id != attempt_id:
        raise MeasurementValidationError("measurement attempt_id does not match selected attempt")
    mode = raw["mode"]
    if mode != expected_mode:
        raise MeasurementValidationError(f"measurement mode must be {expected_mode!r} for schema {schema!r}")
    config_value_raw = raw["config_value"]
    if config_value_raw is not None and not isinstance(config_value_raw, str):
        raise MeasurementValidationError("measurement.config_value must be string or null")
    secret_present = raw["secret_present"]
    if secret_present is not False:
        raise MeasurementValidationError("measurement proves that a non-allowlisted secret leaked")
    raw_samples = raw["samples_us"]
    if not isinstance(raw_samples, list) or not 1 <= len(raw_samples) <= 10_000:
        raise MeasurementValidationError("measurement.samples_us must contain 1..10000 values")
    samples = tuple(
        _finite_number(value, f"measurement.samples_us[{index}]") for index, value in enumerate(raw_samples)
    )
    value_us = _finite_number(raw["value_us"], "measurement.value_us")
    if value_us != _median(samples):
        raise MeasurementValidationError("measurement.value_us must equal the median of measurement.samples_us")
    return ScalarMeasurement(
        schema=schema,
        producer_schema=producer_schema,
        attempt_id=attempt_id,
        mode=expected_mode,
        config_value=config_value_raw,
        value_us=value_us,
        samples_us=samples,
        iqr_us=_iqr(samples),
        physical=None,
    )


def validate_schema_documents(schema_ids: Optional[Tuple[str, ...]] = None) -> Tuple[Dict[str, str], ...]:
    """Validate and hash the exact committed schema documents used by analysis."""

    expected_names = set(_SCHEMA_FILES.values())
    actual_names = {path.name for path in _SCHEMA_DIRECTORY.glob("*.json")}
    if actual_names != expected_names:
        raise MeasurementValidationError(
            f"schema directory mismatch: expected {sorted(expected_names)!r}, observed {sorted(actual_names)!r}"
        )
    selected = set(_SCHEMA_FILES) if schema_ids is None else set(schema_ids)
    unknown = sorted(selected - set(_SCHEMA_FILES))
    if unknown:
        raise MeasurementValidationError(f"unknown requested schema documents: {unknown!r}")
    rows: List[Dict[str, str]] = []
    for schema_id, filename in sorted(_SCHEMA_FILES.items()):
        if schema_id not in selected:
            continue
        path = _SCHEMA_DIRECTORY / filename
        raw = strict_json_loads(path.read_bytes())
        if not isinstance(raw, Mapping) or raw.get("$id") != schema_id:
            raise MeasurementValidationError(f"schema document {filename!r} has the wrong $id")
        if raw.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise MeasurementValidationError(f"schema document {filename!r} has the wrong dialect")
        if raw.get("additionalProperties") is not False:
            raise MeasurementValidationError(f"schema document {filename!r} must reject unknown fields")
        rows.append(
            {
                "schema": schema_id,
                "path": f"experiments/rostam/schemas/{filename}",
                "sha256": file_sha256(path),
            }
        )
    return tuple(rows)
