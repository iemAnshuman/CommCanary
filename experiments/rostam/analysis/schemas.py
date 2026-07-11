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
RAW_ARCHIVE_DESCRIPTOR_SCHEMA = "commcanary.rostam.raw-archive-descriptor.v1"

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
    RAW_ARCHIVE_DESCRIPTOR_SCHEMA: "raw-archive-descriptor-v1.schema.json",
}
_PHYSICAL_PRODUCER_CONTRACTS = {
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.micro-producer.v1",
    PHYSICAL_FULL_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.full-producer.v1",
    PHYSICAL_PARAM_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.param-producer.v1",
    PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.overlap-producer.v1",
    PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA: "commcanary.rostam.physical.capture-producer.v1",
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
}
_PARAM_REPLAY_MODES = {"timestamp-paced-blocking", "compute-filled-blocking"}
_OVERLAP_REPLAY_MODES = {"explicit-wait-overlap", "fixed-input-explicit-wait-overlap"}
_ARTIFACT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
