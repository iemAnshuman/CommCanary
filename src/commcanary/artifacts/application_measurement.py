"""Validation and regression decisions for actual application measurements."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..errors import SchemaError
from ..formats import APPLICATION_MEASUREMENT_FORMAT
from ..statistics import median
from .json_codec import canonical_json_bytes
from .physical_execution import telemetry_assessment

APPLICATION_PERTURBATION_ENVIRONMENT_NAMES = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_NET",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "CUDA_LAUNCH_BLOCKING",
)


def application_subject_configuration(
    *,
    runner_oci_digest: str,
    application_engine: str,
    application_engine_version: str,
    engine_configuration: Mapping[str, Any],
    perturbation_environment: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct the exact stack projection committed by ``subject_sha256``."""

    _oci_digest(runner_oci_digest, "application subject runner OCI digest")
    if application_engine not in {"vllm", "sglang"}:
        raise SchemaError("application subject engine is unsupported")
    _nonempty(application_engine_version, "application subject engine version")
    if set(perturbation_environment) != set(APPLICATION_PERTURBATION_ENVIRONMENT_NAMES):
        raise SchemaError("application subject perturbation environment fields are not closed")
    if any(value is not None and not isinstance(value, str) for value in perturbation_environment.values()):
        raise SchemaError("application subject perturbation environment values must be strings or null")
    return {
        "runner_oci_digest": runner_oci_digest,
        "application_engine": application_engine,
        "application_engine_version": application_engine_version,
        "engine_configuration": dict(engine_configuration),
        "perturbation_environment": {
            name: perturbation_environment[name] for name in APPLICATION_PERTURBATION_ENVIRONMENT_NAMES
        },
    }


def application_subject_sha256(configuration: Mapping[str, Any]) -> str:
    """Return the canonical subject identity used by application and site runners."""

    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def validate_application_measurement(measurement: Mapping[str, Any]) -> None:
    """Recompute a vLLM/SGLang application oracle record."""

    _closed(
        measurement,
        {
            "format",
            "measurement_id",
            "runner",
            "subject_sha256",
            "subject_configuration",
            "perturbation_id",
            "application",
            "workload",
            "samples",
            "summary",
            "environment",
            "environment_sha256",
            "profile",
            "telemetry",
            "telemetry_assessment",
        },
        "application measurement",
    )
    if measurement.get("format") != APPLICATION_MEASUREMENT_FORMAT:
        raise SchemaError("application measurement format is unsupported")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in measurement.items() if key != "measurement_id"})
    ).hexdigest()
    if measurement.get("measurement_id") != expected_id:
        raise SchemaError("application measurement_id does not match canonical content")

    runner = _mapping(measurement.get("runner"), "application runner")
    _closed(runner, {"oci_digest", "execution_protocol"}, "application runner")
    _oci_digest(runner.get("oci_digest"), "application runner OCI digest")
    if runner.get("execution_protocol") not in {
        "vllm-offline-tensor-parallel.v1",
        "sglang-offline-tensor-parallel.v1",
    }:
        raise SchemaError("application runner execution protocol is unsupported")
    subject_sha256 = _sha256(measurement.get("subject_sha256"), "application subject_sha256")
    _nonempty(measurement.get("perturbation_id"), "application perturbation_id")

    application = _mapping(measurement.get("application"), "application identity")
    _closed(
        application,
        {
            "name",
            "engine",
            "engine_version",
            "ground_truth_kind",
            "model_sha256",
            "model_files",
        },
        "application identity",
    )
    _nonempty(application.get("name"), "application name")
    engine = _nonempty(application.get("engine"), "application engine")
    _nonempty(application.get("engine_version"), "application engine_version")
    if application.get("ground_truth_kind") != "application_ground_truth":
        raise SchemaError("application measurement must be actual application ground truth")
    expected_protocol = {
        "vllm": "vllm-offline-tensor-parallel.v1",
        "sglang": "sglang-offline-tensor-parallel.v1",
    }.get(engine)
    if expected_protocol is None or runner.get("execution_protocol") != expected_protocol:
        raise SchemaError("application engine does not match its runner execution protocol")
    _sha256(application.get("model_sha256"), "application model_sha256")
    model_rows = _file_inventory(application.get("model_files"), "application model files")
    if application.get("model_sha256") != hashlib.sha256(canonical_json_bytes(model_rows)).hexdigest():
        raise SchemaError("application model_sha256 does not match its file inventory")

    workload = _mapping(measurement.get("workload"), "application workload")
    _closed(
        workload,
        {
            "batch_size",
            "input_length",
            "output_length",
            "tensor_parallel_size",
            "warmups",
            "iterations",
            "seed",
            "load_format",
            "dtype",
            "prefix_caching",
            "worker_multiprocessing_method",
        },
        "application workload",
    )
    for field in (
        "batch_size",
        "input_length",
        "output_length",
        "tensor_parallel_size",
        "iterations",
    ):
        _integer(workload.get(field), f"application workload {field}", lower=1)
    _integer(workload.get("warmups"), "application workload warmups", lower=0)
    _integer(workload.get("seed"), "application workload seed", lower=0)
    if workload.get("load_format") != "dummy":
        raise SchemaError("application workload load_format must be dummy for the committed public oracle")
    if workload.get("dtype") != "bfloat16":
        raise SchemaError("application workload dtype is unsupported")
    if workload.get("prefix_caching") is not False:
        raise SchemaError("application workload prefix caching must be disabled")
    if workload.get("worker_multiprocessing_method") != "spawn":
        raise SchemaError("application workload multiprocessing method must be spawn")
    subject_configuration = _mapping(
        measurement.get("subject_configuration"),
        "application subject configuration",
    )
    _closed(
        subject_configuration,
        {
            "runner_oci_digest",
            "application_engine",
            "application_engine_version",
            "engine_configuration",
            "perturbation_environment",
        },
        "application subject configuration",
    )
    if subject_configuration.get("runner_oci_digest") != runner.get("oci_digest"):
        raise SchemaError("application subject runner digest does not match its runner")
    if subject_configuration.get("application_engine") != engine:
        raise SchemaError("application subject engine does not match its application")
    if subject_configuration.get("application_engine_version") != application.get("engine_version"):
        raise SchemaError("application subject engine version does not match its application")
    _validate_engine_configuration(
        subject_configuration.get("engine_configuration"),
        engine=str(application["engine"]),
        batch_size=int(workload["batch_size"]),
        input_length=int(workload["input_length"]),
    )
    perturbation_environment = _mapping(
        subject_configuration.get("perturbation_environment"),
        "application perturbation environment",
    )
    expected_environment_names = set(APPLICATION_PERTURBATION_ENVIRONMENT_NAMES)
    _closed(perturbation_environment, expected_environment_names, "application perturbation environment")
    if any(value is not None and not isinstance(value, str) for value in perturbation_environment.values()):
        raise SchemaError("application perturbation environment values must be strings or null")
    if subject_sha256 != hashlib.sha256(canonical_json_bytes(subject_configuration)).hexdigest():
        raise SchemaError("application subject_sha256 does not match its canonical subject configuration")

    samples = measurement.get("samples")
    iterations = int(workload["iterations"])
    if not isinstance(samples, list) or len(samples) != iterations:
        raise SchemaError("application sample count does not match workload iterations")
    latency: List[float] = []
    output_throughput: List[float] = []
    total_throughput: List[float] = []
    observed_memory: List[int] = []
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"application samples[{index}]")
        _closed(
            sample,
            {
                "batch_latency_ms",
                "output_token_throughput_per_second",
                "total_token_throughput_per_second",
                "observed_gpu_memory_used_bytes",
            },
            f"application samples[{index}]",
        )
        latency.append(_positive(sample.get("batch_latency_ms"), "application batch latency"))
        output_throughput.append(
            _positive(sample.get("output_token_throughput_per_second"), "application output throughput")
        )
        total_throughput.append(
            _positive(sample.get("total_token_throughput_per_second"), "application total throughput")
        )
        observed_memory.append(
            _integer(
                sample.get("observed_gpu_memory_used_bytes"),
                "application observed GPU memory",
                lower=1,
            )
        )

    summary = _mapping(measurement.get("summary"), "application summary")
    _closed(
        summary,
        {
            "median_batch_latency_ms",
            "p99_batch_latency_ms",
            "median_output_token_throughput_per_second",
            "median_total_token_throughput_per_second",
            "max_observed_gpu_memory_used_bytes",
        },
        "application summary",
    )
    expected_summary = {
        "median_batch_latency_ms": median(latency),
        "p99_batch_latency_ms": _percentile(latency, 0.99),
        "median_output_token_throughput_per_second": median(output_throughput),
        "median_total_token_throughput_per_second": median(total_throughput),
        "max_observed_gpu_memory_used_bytes": max(observed_memory),
    }
    for name, expected in expected_summary.items():
        actual = _positive(summary.get(name), f"application summary {name}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise SchemaError(f"application summary {name} does not recompute")

    environment = _mapping(measurement.get("environment"), "application environment")
    _closed(
        environment,
        {
            "hostname",
            "platform",
            "python",
            "torch",
            "cuda",
            "nccl",
            "visible_gpu_count",
            "slurm_job_id",
            "slurm_node",
            "nccl_configuration",
            "runtime_configuration",
        },
        "application environment",
    )
    if environment.get("visible_gpu_count") != workload.get("tensor_parallel_size"):
        raise SchemaError("application visible GPU count does not match tensor parallel size")
    nccl_configuration = _mapping(environment.get("nccl_configuration"), "application NCCL configuration")
    runtime_configuration = _mapping(
        environment.get("runtime_configuration"),
        "application runtime configuration",
    )
    observed_perturbation_environment = {
        **dict(nccl_configuration),
        **dict(runtime_configuration),
    }
    if observed_perturbation_environment != dict(perturbation_environment):
        raise SchemaError("application observed environment does not match its subject configuration")
    if measurement.get("environment_sha256") != hashlib.sha256(canonical_json_bytes(environment)).hexdigest():
        raise SchemaError("application environment_sha256 does not match canonical content")

    profile = _mapping(measurement.get("profile"), "application profile")
    _closed(
        profile,
        {"captured", "excluded_from_timed_samples", "artifact_sha256", "files"},
        "application profile",
    )
    if profile.get("excluded_from_timed_samples") is not True:
        raise SchemaError("application profile must be excluded from timed samples")
    captured = profile.get("captured")
    if not isinstance(captured, bool):
        raise SchemaError("application profile captured must be boolean")
    profile_rows = _file_inventory(profile.get("files"), "application profile files", require_nonempty=captured)
    if captured:
        _sha256(profile.get("artifact_sha256"), "application profile artifact_sha256")
        if profile.get("artifact_sha256") != hashlib.sha256(canonical_json_bytes(profile_rows)).hexdigest():
            raise SchemaError("application profile artifact_sha256 does not match its file inventory")
    elif profile.get("artifact_sha256") is not None or profile_rows:
        raise SchemaError("application without a profile must not claim profile artifacts")

    telemetry = measurement.get("telemetry")
    if not isinstance(telemetry, list):
        raise SchemaError("application telemetry must be an array")
    expected_phases = [
        "before_warmup",
        "before_measured_cycle_1",
        *(f"after_measured_cycle_{index + 1}" for index in range(iterations)),
        "final",
    ]
    if [row.get("phase") if isinstance(row, Mapping) else None for row in telemetry] != expected_phases:
        raise SchemaError("application telemetry phases are incomplete or out of order")
    expected_assessment = telemetry_assessment(telemetry, world_size=int(workload["tensor_parallel_size"]))
    if measurement.get("telemetry_assessment") != expected_assessment:
        raise SchemaError("application telemetry assessment does not recompute")


def application_regression_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    threshold_pct: float,
) -> Dict[str, Any]:
    """Return the deployment decision from actual application SLO metrics."""

    validate_application_measurement(baseline)
    validate_application_measurement(candidate)
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

    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    metric_rows = []
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
        metric_rows.append(
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
        "decision": "fail" if any(row["failed"] for row in metric_rows) else "pass",
        "regression_magnitude_pct": maximum,
        "severe": maximum >= threshold and any(row["failed"] for row in metric_rows),
        "metrics": metric_rows,
    }


def application_workload_identity(measurement: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the invariant application workload, excluding the stack version under test."""

    application = _mapping(measurement.get("application"), "application identity")
    workload = _mapping(measurement.get("workload"), "application workload")
    return {
        "application": {
            key: application[key]
            for key in (
                "name",
                "engine",
                "ground_truth_kind",
                "model_sha256",
                "model_files",
            )
        },
        "workload": dict(workload),
    }


def _validate_engine_configuration(
    value: Any,
    *,
    engine: str,
    batch_size: int,
    input_length: int,
) -> None:
    configuration = _mapping(value, "application engine configuration")
    if engine == "vllm":
        _closed(
            configuration,
            {
                "kind",
                "enforce_eager",
                "disable_custom_all_reduce",
                "gpu_memory_utilization",
                "kv_cache_memory_bytes",
                "max_num_batched_tokens",
                "max_num_seqs",
            },
            "vLLM engine configuration",
        )
        if configuration.get("kind") != "vllm.v1":
            raise SchemaError("vLLM engine configuration kind is unsupported")
        for field in ("enforce_eager", "disable_custom_all_reduce"):
            if not isinstance(configuration.get(field), bool):
                raise SchemaError(f"vLLM engine configuration {field} must be boolean")
        utilization = configuration.get("gpu_memory_utilization")
        if not isinstance(utilization, (int, float)) or isinstance(utilization, bool):
            raise SchemaError("vLLM gpu_memory_utilization must be numeric")
        if not math.isfinite(float(utilization)) or not 0.1 <= float(utilization) <= 0.95:
            raise SchemaError("vLLM gpu_memory_utilization must be in [0.1, 0.95]")
        for field in ("kv_cache_memory_bytes", "max_num_batched_tokens", "max_num_seqs"):
            _integer(configuration.get(field), f"vLLM engine configuration {field}", lower=1)
        if int(configuration["max_num_seqs"]) < batch_size:
            raise SchemaError("vLLM max_num_seqs must cover application batch_size")
        if int(configuration["max_num_batched_tokens"]) < batch_size * input_length:
            raise SchemaError("vLLM max_num_batched_tokens must cover the complete application prefill")
        return
    if engine == "sglang":
        _closed(
            configuration,
            {
                "kind",
                "disable_cuda_graph",
                "disable_custom_all_reduce",
                "disable_overlap_schedule",
                "mem_fraction_static",
                "max_running_requests",
                "max_total_tokens",
                "chunked_prefill_size",
                "max_prefill_tokens",
            },
            "SGLang engine configuration",
        )
        if configuration.get("kind") != "sglang.v1":
            raise SchemaError("SGLang engine configuration kind is unsupported")
        for field in ("disable_cuda_graph", "disable_custom_all_reduce", "disable_overlap_schedule"):
            if not isinstance(configuration.get(field), bool):
                raise SchemaError(f"SGLang engine configuration {field} must be boolean")
        fraction = configuration.get("mem_fraction_static")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise SchemaError("SGLang mem_fraction_static must be numeric")
        if not math.isfinite(float(fraction)) or not 0.1 <= float(fraction) <= 0.95:
            raise SchemaError("SGLang mem_fraction_static must be in [0.1, 0.95]")
        for field in ("max_running_requests", "max_total_tokens", "chunked_prefill_size", "max_prefill_tokens"):
            _integer(configuration.get(field), f"SGLang engine configuration {field}", lower=1)
        if int(configuration["max_running_requests"]) < batch_size:
            raise SchemaError("SGLang max_running_requests must cover application batch_size")
        prefill_tokens = batch_size * input_length
        if int(configuration["max_total_tokens"]) < prefill_tokens:
            raise SchemaError("SGLang max_total_tokens must cover the complete application prefill")
        if int(configuration["max_prefill_tokens"]) < prefill_tokens:
            raise SchemaError("SGLang max_prefill_tokens must cover the complete application prefill")
        return
    raise SchemaError("application engine configuration is unsupported")


def _file_inventory(value: Any, label: str, *, require_nonempty: bool = True) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or (require_nonempty and not value):
        raise SchemaError(f"{label} must be {'a non-empty' if require_nonempty else 'an'} array")
    paths: List[str] = []
    rows: List[Mapping[str, Any]] = []
    for index, raw_row in enumerate(value):
        row = _mapping(raw_row, f"{label}[{index}]")
        _closed(row, {"path", "bytes", "sha256"}, f"{label}[{index}]")
        path = _nonempty(row.get("path"), f"{label}[{index}].path")
        if path.startswith("/") or ".." in path.split("/") or "\\" in path:
            raise SchemaError(f"{label}[{index}].path must be a safe relative POSIX path")
        paths.append(path)
        _integer(row.get("bytes"), f"{label}[{index}].bytes", lower=0)
        _sha256(row.get("sha256"), f"{label}[{index}].sha256")
        rows.append(row)
    if paths != sorted(set(paths)):
        raise SchemaError(f"{label} must use unique canonical path order")
    return tuple(rows)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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


def _sha256(value: Any, label: str) -> str:
    result = _nonempty(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SchemaError(f"{label} must be a lowercase SHA-256")
    return result


def _oci_digest(value: Any, label: str) -> str:
    result = _nonempty(value, label)
    algorithm, separator, digest = result.partition(":")
    if algorithm != "sha256" or separator != ":":
        raise SchemaError(f"{label} must use sha256:<64 lowercase hex>")
    _sha256(digest, label)
    return result


def _integer(value: Any, label: str, *, lower: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        raise SchemaError(f"{label} must be an integer of at least {lower}")
    return value


def _positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SchemaError(f"{label} must be finite and positive")
    return result


__all__ = [
    "APPLICATION_PERTURBATION_ENVIRONMENT_NAMES",
    "application_regression_decision",
    "application_subject_configuration",
    "application_subject_sha256",
    "application_workload_identity",
    "validate_application_measurement",
]
