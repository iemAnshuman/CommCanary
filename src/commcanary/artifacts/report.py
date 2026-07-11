"""Replay report artifact contract validation and reconciliation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..errors import SchemaError
from ..formats import REPORT_FORMAT
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_multiply,
    require_within,
    validate_json_mapping,
)
from ..statistics import percentile_from_sorted, summarize_latencies
from .wire import (
    JsonDict,
    as_float,
    as_int,
    replay_protocol_sha256,
    require_format,
    require_optional_mapping,
    validate_sha256,
)


def validate_report(
    report: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    require_format(report, REPORT_FORMAT, "report")
    _validate_report_json_resources(report, limits=limits)
    require_optional_mapping(report, "workload", "report")
    require_optional_mapping(report, "system", "report")
    require_optional_mapping(report, "canary_summary", "report")
    require_optional_mapping(report, "backend", "report")

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SchemaError("report must contain a metrics object")
    required_metrics = {
        "count": "count",
        "median_us": "latency",
        "p95_us": "latency",
        "p99_us": "latency",
        "max_us": "latency",
        "mean_us": "latency",
        "arrival_skew_median_us": "latency",
        "arrival_skew_p95_us": "latency",
        "arrival_skew_max_us": "latency",
        "avg_rank_wait_median_us": "latency",
        "communication_hidden_pct": "percent",
    }
    for key, kind in required_metrics.items():
        if key not in metrics:
            raise SchemaError(f"report metrics missing {key!r}")
        if kind == "count":
            if as_int(metrics.get(key)) < 0:
                raise SchemaError("report metric count must be non-negative")
        elif kind == "percent":
            value = as_float(metrics.get(key))
            if not 0.0 <= value <= 100.0:
                raise SchemaError(f"report metric {key} must be between 0 and 100")
        elif as_float(metrics.get(key)) < 0.0:
            raise SchemaError(f"report metric {key} must be non-negative")
    _validate_latency_quantiles(metrics, "report metrics")

    canary = report.get("canary")
    if not isinstance(canary, Mapping) or not canary.get("sha256"):
        raise SchemaError("report must contain canary.sha256")
    validate_sha256(canary.get("sha256"), "canary.sha256")
    if "execution_semantic_sha256" in canary:
        validate_sha256(canary.get("execution_semantic_sha256"), "canary.execution_semantic_sha256")
    if "scheduler_execution_sha256" in canary:
        validate_sha256(canary.get("scheduler_execution_sha256"), "canary.scheduler_execution_sha256")
    if "calibration_evaluation_sha256" in canary:
        validate_sha256(canary.get("calibration_evaluation_sha256"), "canary.calibration_evaluation_sha256")
    if "source_events" not in canary:
        raise SchemaError("report canary.source_events is required")
    source_events = as_int(canary.get("source_events"))
    if source_events < 0:
        raise SchemaError("report canary.source_events must be non-negative")

    model = report.get("simulation_model")
    if not isinstance(model, Mapping) or not isinstance(model.get("version"), str) or not model.get("version"):
        raise SchemaError("report must contain simulation_model.version")
    protocol = report.get("replay_protocol")
    if not isinstance(protocol, Mapping) or not protocol.get("sha256"):
        raise SchemaError("report must contain replay_protocol.sha256")
    required_protocol = ("model_name", "model_version", "seed", "iterations", "quantile_method", "bandwidth_unit")
    for key in required_protocol:
        if key not in protocol:
            raise SchemaError(f"report replay_protocol missing {key!r}")
    for key in ("model_name", "model_version", "quantile_method", "bandwidth_unit"):
        if not isinstance(protocol.get(key), str) or not protocol.get(key):
            raise SchemaError(f"report replay_protocol.{key} must be a non-empty string")
    if not isinstance(protocol.get("seed"), int) or isinstance(protocol.get("seed"), bool):
        raise SchemaError("report replay_protocol.seed must be an integer")
    if not isinstance(protocol.get("iterations"), int) or isinstance(protocol.get("iterations"), bool):
        raise SchemaError("report replay_protocol.iterations must be an integer")
    if as_int(protocol.get("iterations")) <= 0:
        raise SchemaError("report replay_protocol.iterations must be positive")
    validate_sha256(protocol.get("sha256"), "replay_protocol.sha256")
    if replay_protocol_sha256(protocol, limits=limits) != protocol.get("sha256"):
        raise SchemaError("replay_protocol.sha256 does not match replay protocol fields")

    backend = report.get("backend", {})
    if not isinstance(backend, Mapping):
        raise SchemaError("report backend must be an object")
    _validate_report_backend(backend)
    for key in ("seed", "iterations", "bandwidth_unit"):
        if key in backend and backend.get(key) != protocol.get(key):
            raise SchemaError(f"report backend.{key} must match replay_protocol.{key}")
    if model.get("version") != protocol.get("model_version"):
        raise SchemaError("simulation_model.version must match replay_protocol.model_version")
    if model.get("name") and model.get("name") != protocol.get("model_name"):
        raise SchemaError("simulation_model.name must match replay_protocol.model_name")

    try:
        expected_count = checked_multiply(
            source_events,
            as_int(protocol.get("iterations")),
            label="report replay events",
        )
        require_within(
            expected_count,
            limits.max_replay_events,
            label="report replay events",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    if as_int(metrics.get("count")) != expected_count:
        raise SchemaError("report metrics.count must match source events times iterations")

    breakdowns: Dict[str, List[Mapping[str, Any]]] = {}
    for breakdown_key in ("by_phase", "by_op"):
        rows = report.get(breakdown_key, [])
        if not isinstance(rows, list):
            raise SchemaError(f"report {breakdown_key} must be a list")
        if as_int(metrics.get("count")) > 0 and not rows:
            raise SchemaError(f"report {breakdown_key} must not be empty when metrics.count is positive")
        names = set()
        validated_rows: List[Mapping[str, Any]] = []
        for row_index, row in enumerate(rows):
            label = f"report {breakdown_key} row {row_index}"
            if not isinstance(row, Mapping):
                raise SchemaError(f"{label} must be an object")
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise SchemaError(f"{label} name must be a non-empty string")
            if name in names:
                raise SchemaError(f"report {breakdown_key} names must be unique")
            names.add(name)
            _validate_breakdown_row(row, label)
            validated_rows.append(row)
        row_total = sum(as_int(row.get("count")) for row in validated_rows)
        if rows and row_total != as_int(metrics.get("count")):
            raise SchemaError(f"report {breakdown_key} counts must sum to metrics.count")
        _reconcile_breakdown_summary(metrics, validated_rows, breakdown_key)
        breakdowns[breakdown_key] = validated_rows

    calibration = report.get("calibration")
    if calibration is not None:
        _validate_calibration(calibration)
        if as_int(calibration.get("count")) != as_int(metrics.get("count")):
            raise SchemaError("report calibration.count must match metrics.count")

    samples = report.get("samples")
    if samples is not None:
        if not isinstance(samples, list):
            raise SchemaError("report samples must be a list")
        if len(samples) != as_int(metrics.get("count")):
            raise SchemaError("report sample count must match metrics.count")
        _reconcile_report_samples(report, samples, breakdowns)


def _validate_report_json_resources(
    report: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> None:
    try:
        validate_json_mapping(report, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"report violates JSON resource constraints: {exc}") from exc


def _validate_breakdown_row(row: Mapping[str, Any], label: str) -> None:
    required = ("count", "median_us", "p95_us", "p99_us", "max_us", "mean_us")
    for key in required:
        if key not in row:
            raise SchemaError(f"{label} missing {key!r}")
    if as_int(row.get("count")) <= 0:
        raise SchemaError(f"{label} count must be positive")
    for key in required[1:]:
        if as_float(row.get(key)) < 0.0:
            raise SchemaError(f"{label} {key} must be non-negative")
    _validate_latency_quantiles(row, label)


def _reconcile_breakdown_summary(overall: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], label: str) -> None:
    if not rows:
        return
    total_count = sum(as_int(row.get("count")) for row in rows)
    weighted_mean = sum(as_float(row.get("mean_us")) * as_int(row.get("count")) for row in rows) / total_count
    if abs(weighted_mean - as_float(overall.get("mean_us"))) > 0.0021:
        raise SchemaError(f"report {label} weighted mean must match metrics.mean_us")
    row_max = max(as_float(row.get("max_us")) for row in rows)
    if abs(row_max - as_float(overall.get("max_us"))) > 0.0011:
        raise SchemaError(f"report {label} maximum must match metrics.max_us")


def _validate_calibration(calibration: Any) -> None:
    if not isinstance(calibration, Mapping):
        raise SchemaError("report calibration must be an object")
    required = (
        "signal",
        "count",
        "mean_absolute_error_us",
        "median_absolute_error_us",
        "p95_absolute_error_us",
        "max_absolute_error_us",
        "mean_bias_us",
        "mean_absolute_percentage_error_pct",
        "percentage_count",
    )
    for key in required:
        if key not in calibration:
            raise SchemaError(f"report calibration missing {key!r}")
    if calibration.get("signal") != "observed_exposed_us":
        raise SchemaError("report calibration.signal is unsupported")
    count = as_int(calibration.get("count"))
    percentage_count = as_int(calibration.get("percentage_count"))
    if count <= 0 or percentage_count < 0 or percentage_count > count:
        raise SchemaError("report calibration counts are invalid")
    for key in (
        "mean_absolute_error_us",
        "median_absolute_error_us",
        "p95_absolute_error_us",
        "max_absolute_error_us",
        "mean_absolute_percentage_error_pct",
    ):
        if as_float(calibration.get(key)) < 0.0:
            raise SchemaError(f"report calibration {key} must be non-negative")
    median_error = as_float(calibration.get("median_absolute_error_us"))
    p95_error = as_float(calibration.get("p95_absolute_error_us"))
    max_error = as_float(calibration.get("max_absolute_error_us"))
    if not median_error <= p95_error <= max_error:
        raise SchemaError("report calibration absolute-error quantiles are unordered")
    as_float(calibration.get("mean_bias_us"))


def _validate_report_backend(backend: Mapping[str, Any]) -> None:
    if "bandwidth_gbps" in backend and as_float(backend.get("bandwidth_gbps")) <= 0.0:
        raise SchemaError("report backend.bandwidth_gbps must be positive")
    if "latency_floor_us" in backend and as_float(backend.get("latency_floor_us")) < 0.0:
        raise SchemaError("report backend.latency_floor_us must be non-negative")
    if "compute_pressure" in backend and as_float(backend.get("compute_pressure")) < 0.0:
        raise SchemaError("report backend.compute_pressure must be non-negative")
    if "overlap_efficiency" in backend:
        overlap_efficiency = as_float(backend.get("overlap_efficiency"))
        if not 0.0 <= overlap_efficiency <= 1.0:
            raise SchemaError("report backend.overlap_efficiency must be between 0 and 1")


def _reconcile_report_samples(
    report: Mapping[str, Any],
    samples: Sequence[Any],
    breakdowns: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    required_sample_keys = {
        "index",
        "iteration",
        "phase",
        "op",
        "group",
        "first_arrival_us",
        "last_arrival_us",
        "collective_start_us",
        "completion_us",
        "total_us",
        "hidden_us",
        "exposed_us",
        "arrival_skew_us",
        "avg_rank_wait_us",
        "compute_overlap_us",
        "collective_us",
        "queue_wait_us",
    }
    exposed: List[float] = []
    skew: List[float] = []
    wait: List[float] = []
    hidden_total = 0.0
    total_sum = 0.0
    phase_values: Dict[str, List[float]] = {}
    op_values: Dict[str, List[float]] = {}
    observed_flags: List[bool] = []
    observed_values: List[float] = []
    seen_indices: List[int] = []
    group_available: Dict[Tuple[int, str], float] = {}
    iterations = as_int(report.get("replay_protocol", {}).get("iterations"), 1)
    source_events = as_int(report.get("canary", {}).get("source_events"))
    overlap_efficiency = as_float(report.get("backend", {}).get("overlap_efficiency"), 0.0)
    if not 0.0 <= overlap_efficiency <= 1.0:
        raise SchemaError("report backend.overlap_efficiency must be between 0 and 1")

    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise SchemaError(f"report sample {index} must be an object")
        missing = sorted(required_sample_keys - set(sample.keys()))
        if missing:
            raise SchemaError(f"report sample {index} missing {missing[0]!r}")
        if not isinstance(sample.get("phase"), str) or not sample.get("phase"):
            raise SchemaError(f"report sample {index} phase must be a non-empty string")
        if not isinstance(sample.get("op"), str) or not sample.get("op"):
            raise SchemaError(f"report sample {index} op must be a non-empty string")
        if not isinstance(sample.get("group"), str) or not sample.get("group"):
            raise SchemaError(f"report sample {index} group must be a non-empty string")
        scheduler_resource = sample.get("scheduler_resource", sample.get("group"))
        if not isinstance(scheduler_resource, str) or not scheduler_resource:
            raise SchemaError(f"report sample {index} scheduler_resource must be a non-empty string")
        sequence_index = as_int(sample.get("index"))
        iteration = as_int(sample.get("iteration"))
        if sequence_index != index:
            raise SchemaError("report sample indices must be contiguous and ordered")
        if iteration < 0 or iteration >= iterations:
            raise SchemaError(f"report sample {index} iteration is outside replay_protocol.iterations")
        if source_events > 0 and iteration != index // source_events:
            raise SchemaError(f"report sample {index} iteration does not match replay partitioning")
        seen_indices.append(sequence_index)
        for key in required_sample_keys - {"index", "iteration", "phase", "op", "group"}:
            if as_float(sample.get(key)) < 0.0:
                raise SchemaError(f"report sample {index} {key} must be non-negative")
        first_arrival_us = as_float(sample.get("first_arrival_us"))
        last_arrival_us = as_float(sample.get("last_arrival_us"))
        arrival_skew_us = as_float(sample.get("arrival_skew_us"))
        avg_rank_wait_us = as_float(sample.get("avg_rank_wait_us"))
        collective_start_us = as_float(sample.get("collective_start_us"))
        queue_wait_us = as_float(sample.get("queue_wait_us"))
        collective_us = as_float(sample.get("collective_us"))
        completion_us = as_float(sample.get("completion_us"))
        total_us = as_float(sample.get("total_us"))
        hidden_us = as_float(sample.get("hidden_us"))
        exposed_us = as_float(sample.get("exposed_us"))
        compute_overlap_us = as_float(sample.get("compute_overlap_us"))
        tolerance = 0.0051
        if abs((last_arrival_us - first_arrival_us) - arrival_skew_us) > tolerance:
            raise SchemaError(f"report sample {index} last_arrival_us must match arrival_skew_us")
        if avg_rank_wait_us > arrival_skew_us + tolerance:
            raise SchemaError(f"report sample {index} avg_rank_wait_us must not exceed arrival_skew_us")
        group_key = (iteration, scheduler_resource)
        expected_start = max(last_arrival_us, group_available.get(group_key, 0.0))
        if abs(collective_start_us - expected_start) > tolerance:
            raise SchemaError(f"report sample {index} collective_start_us is inconsistent")
        if abs(queue_wait_us - (collective_start_us - last_arrival_us)) > tolerance:
            raise SchemaError(f"report sample {index} queue_wait_us is inconsistent")
        if abs(completion_us - (collective_start_us + collective_us)) > tolerance:
            raise SchemaError(f"report sample {index} completion_us is inconsistent")
        if abs(total_us - (arrival_skew_us + queue_wait_us + collective_us)) > tolerance:
            raise SchemaError(f"report sample {index} total_us decomposition is inconsistent")
        if hidden_us > total_us or abs((hidden_us + exposed_us) - total_us) > 0.0021:
            raise SchemaError(f"report sample {index} hidden_us + exposed_us must equal total_us")
        maximum_hideable_us = min(total_us, max(0.0, compute_overlap_us) * overlap_efficiency)
        if abs(hidden_us - maximum_hideable_us) > tolerance:
            raise SchemaError(f"report sample {index} hidden_us does not match deterministic overlap model")
        exposed.append(exposed_us)
        skew.append(arrival_skew_us)
        wait.append(avg_rank_wait_us)
        hidden_total += hidden_us
        total_sum += total_us
        group_available[group_key] = completion_us
        phase_values.setdefault(str(sample.get("phase")), []).append(exposed_us)
        op_values.setdefault(str(sample.get("op")), []).append(exposed_us)
        has_observed = "observed_exposed_us" in sample
        observed_flags.append(has_observed)
        if has_observed:
            observed = as_float(sample.get("observed_exposed_us"))
            if observed < 0.0:
                raise SchemaError(f"report sample {index} observed_exposed_us must be non-negative")
            observed_values.append(observed)

    if seen_indices != list(range(len(samples))):
        raise SchemaError("report sample indices must be unique and contiguous")

    metrics = report.get("metrics", {})
    expected_metrics = summarize_latencies(exposed)
    ordered_skew = sorted(skew)
    ordered_wait = sorted(wait)
    hidden_pct = (hidden_total / total_sum * 100.0) if total_sum else 0.0
    expected_metrics.update(
        {
            "arrival_skew_median_us": round(percentile_from_sorted(ordered_skew, 50.0), 3),
            "arrival_skew_p95_us": round(percentile_from_sorted(ordered_skew, 95.0), 3),
            "arrival_skew_max_us": round(ordered_skew[-1], 3) if ordered_skew else 0.0,
            "avg_rank_wait_median_us": round(percentile_from_sorted(ordered_wait, 50.0), 3),
            "communication_hidden_pct": round(hidden_pct, 2),
        }
    )
    for key, expected in expected_metrics.items():
        tolerance = 0.02 if key == "communication_hidden_pct" else 0.0021
        if abs(as_float(metrics.get(key)) - as_float(expected)) > tolerance:
            raise SchemaError(f"report metrics.{key} does not match included samples")

    for breakdown_key, values_by_name in (("by_phase", phase_values), ("by_op", op_values)):
        expected_rows: Dict[str, JsonDict] = {}
        for name, values in values_by_name.items():
            row = {"name": name}
            row.update(summarize_latencies(values))
            expected_rows[name] = row
        actual_rows = {str(row.get("name")): row for row in breakdowns.get(breakdown_key, [])}
        if set(actual_rows) != set(expected_rows):
            raise SchemaError(f"report {breakdown_key} names do not match included samples")
        for name, expected_row in expected_rows.items():
            actual_row = actual_rows[name]
            for key in ("count", "median_us", "p95_us", "p99_us", "max_us", "mean_us"):
                if key == "count":
                    if as_int(actual_row.get(key)) != as_int(expected_row.get(key)):
                        raise SchemaError(f"report {breakdown_key} row {name!r} count does not match samples")
                elif abs(as_float(actual_row.get(key)) - as_float(expected_row.get(key))) > 0.0021:
                    raise SchemaError(f"report {breakdown_key} row {name!r} {key} does not match samples")

    calibration = report.get("calibration")
    if any(observed_flags) and not all(observed_flags):
        raise SchemaError("report samples must either all include observed_exposed_us or none")
    if all(observed_flags) and observed_flags:
        if calibration is None:
            raise SchemaError("report with observed samples must contain calibration")
        errors = [modeled - observed for modeled, observed in zip(exposed, observed_values)]
        absolute = sorted(abs(value) for value in errors)
        percentage = [
            abs(modeled - observed) / observed * 100.0
            for modeled, observed in zip(exposed, observed_values)
            if observed > 0.0
        ]
        expected_calibration = {
            "count": len(errors),
            "mean_absolute_error_us": round(sum(absolute) / len(absolute), 3),
            "median_absolute_error_us": round(percentile_from_sorted(absolute, 50.0), 3),
            "p95_absolute_error_us": round(percentile_from_sorted(absolute, 95.0), 3),
            "max_absolute_error_us": round(absolute[-1], 3),
            "mean_bias_us": round(sum(errors) / len(errors), 3),
            "mean_absolute_percentage_error_pct": round(sum(percentage) / len(percentage), 3) if percentage else 0.0,
            "percentage_count": len(percentage),
        }
        for key, expected in expected_calibration.items():
            if key in {"count", "percentage_count"}:
                if as_int(calibration.get(key)) != expected:
                    raise SchemaError(f"report calibration.{key} does not match samples")
            elif abs(as_float(calibration.get(key)) - as_float(expected)) > 0.01:
                raise SchemaError(f"report calibration.{key} does not match samples")
    elif calibration is not None:
        raise SchemaError("report calibration requires observed_exposed_us samples")


def _validate_latency_quantiles(metrics: Mapping[str, Any], label: str) -> None:
    keys = ("median_us", "p95_us", "p99_us", "max_us")
    if not all(key in metrics for key in keys):
        return
    median_us, p95_us, p99_us, max_us = (as_float(metrics.get(key)) for key in keys)
    if not median_us <= p95_us <= p99_us <= max_us:
        raise SchemaError(f"{label} must satisfy median_us <= p95_us <= p99_us <= max_us")
    if "mean_us" in metrics and as_float(metrics.get("mean_us")) > max_us:
        raise SchemaError(f"{label} mean_us must be no greater than max_us")
    skew_keys = ("arrival_skew_median_us", "arrival_skew_p95_us", "arrival_skew_max_us")
    if all(key in metrics for key in skew_keys):
        skew_median, skew_p95, skew_max = (as_float(metrics.get(key)) for key in skew_keys)
        if not skew_median <= skew_p95 <= skew_max:
            raise SchemaError(
                f"{label} must satisfy arrival_skew_median_us <= arrival_skew_p95_us <= arrival_skew_max_us"
            )


__all__ = ["validate_report"]
