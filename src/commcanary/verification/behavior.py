"""Behavioral replay and pairwise-ranking verification."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ..artifacts.canary import validate_canary
from ..artifacts.wire import JsonDict, as_float, as_int
from ..behavior_config import (
    BEHAVIORAL_RANKING_METRICS,
    behavior_replay_arguments,
    parse_behavior_configurations,
    preflight_behavior_ranking_work,
    ranking_relation,
)
from ..compilation import compile_trace_core
from ..errors import SchemaError
from ..formats import BEHAVIOR_VERIFICATION_FORMAT, FIDELITY_VERIFICATION_FORMAT
from ..replay.core import replay_canary
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..statistics import percentile, summarize_latencies
from .fidelity import verify_canary_fidelity

_BEHAVIORAL_LATENCY_METRICS = ("median_us", "p95_us", "p99_us", "max_us", "mean_us")


def verify_canary_behavior(
    trace: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    configurations: Optional[Sequence[Mapping[str, Any]]] = None,
    relative_tolerance_pct: float = 10.0,
    absolute_tolerance_us: float = 1.0,
    hidden_tolerance_points: float = 5.0,
    tail_recall_threshold: float = 0.80,
    ranking_tie_tolerance_us: float = 0.001,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Replay source and canary canaries and compare behavioral metrics.

    This verifier separates four questions that are easy to conflate in a
    compressed trace artifact: representation fidelity, source verification,
    simulator-visible behavior, and pairwise configuration ranking.
    """

    validate_canary(canary, limits=limits)
    relative_tolerance_pct = _optional_non_negative(relative_tolerance_pct, "relative_tolerance_pct") or 0.0
    absolute_tolerance_us = _optional_non_negative(absolute_tolerance_us, "absolute_tolerance_us") or 0.0
    hidden_tolerance_points = _optional_non_negative(hidden_tolerance_points, "hidden_tolerance_points") or 0.0
    tail_recall_threshold = _optional_non_negative(tail_recall_threshold, "tail_recall_threshold") or 0.0
    ranking_tie_tolerance_us = _optional_non_negative(ranking_tie_tolerance_us, "ranking_tie_tolerance_us") or 0.0
    if tail_recall_threshold > 1.0:
        raise SchemaError("tail_recall_threshold must be between 0 and 1")
    normalized_configurations = parse_behavior_configurations(
        configurations,
        max_configurations=limits.max_behavior_configurations,
    )
    preflight_behavior_ranking_work(normalized_configurations, candidate_count=1, limits=limits)

    compiler = canary.get("compiler", {})
    source_events = as_int(compiler.get("source_events"))
    fidelity = compiler.get("fidelity", {}) if isinstance(compiler.get("fidelity"), Mapping) else {}
    representation_fidelity_status = str(fidelity.get("mode", "missing_fidelity_metadata"))

    try:
        source_verification = verify_canary_fidelity(trace, canary, limits=limits)
        source_verified_status = str(source_verification.get("status"))
    except SchemaError as exc:
        source_verification = {
            "format": FIDELITY_VERIFICATION_FORMAT,
            "status": "failed",
            "assurance_state": "internally_consistent",
            "checks": [{"name": "source_verification_exception", "status": "fail", "reason": str(exc)}],
        }
        source_verified_status = "failed"

    trace_events = trace.get("events", [])
    trace_event_count = len(trace_events) if isinstance(trace_events, list) else 0
    source_coverage_status = "full_source" if source_events == trace_event_count else "partial_source"
    if source_verified_status == "source_verified" and source_coverage_status != "full_source":
        source_verified_status = "partial_source_verified"
    full_canary = compile_trace_core(
        trace,
        timing_sample_limit=max(2, trace_event_count),
        require_lossless_timing=True,
        allow_empty=trace_event_count == 0,
        limits=limits,
    )
    config_rows: List[JsonDict] = []
    for raw_config in normalized_configurations:
        name = raw_config["name"]
        replay_args = behavior_replay_arguments(raw_config)
        source_report = replay_canary(
            full_canary,
            backend_label=name,
            include_samples=True,
            limits=limits,
            **replay_args,
        )
        canary_report = replay_canary(
            canary,
            backend_label=name,
            include_samples=True,
            limits=limits,
            **replay_args,
        )

        metric_checks = [
            _behavior_count_check(source_report["metrics"], canary_report["metrics"]),
        ]
        metric_checks.extend(
            _behavior_metric_check(
                metric,
                source_report["metrics"],
                canary_report["metrics"],
                relative_tolerance_pct=relative_tolerance_pct,
                absolute_tolerance_us=absolute_tolerance_us,
                hidden_tolerance_points=hidden_tolerance_points,
            )
            for metric in _BEHAVIORAL_LATENCY_METRICS
        )
        metric_checks.append(
            _behavior_metric_check(
                "communication_hidden_pct",
                source_report["metrics"],
                canary_report["metrics"],
                relative_tolerance_pct=relative_tolerance_pct,
                absolute_tolerance_us=absolute_tolerance_us,
                hidden_tolerance_points=hidden_tolerance_points,
            )
        )

        source_queue = _sample_distribution(source_report.get("samples", []), "queue_wait_us")
        canary_queue = _sample_distribution(canary_report.get("samples", []), "queue_wait_us")
        queue_checks = [
            _behavior_metric_check(
                metric,
                source_queue,
                canary_queue,
                relative_tolerance_pct=relative_tolerance_pct,
                absolute_tolerance_us=absolute_tolerance_us,
                hidden_tolerance_points=hidden_tolerance_points,
            )
            for metric in _BEHAVIORAL_LATENCY_METRICS
        ]
        phase_checks = _breakdown_behavior_checks(
            "phase",
            source_report.get("by_phase", []),
            canary_report.get("by_phase", []),
            relative_tolerance_pct=relative_tolerance_pct,
            absolute_tolerance_us=absolute_tolerance_us,
        )
        op_checks = _breakdown_behavior_checks(
            "op",
            source_report.get("by_op", []),
            canary_report.get("by_op", []),
            relative_tolerance_pct=relative_tolerance_pct,
            absolute_tolerance_us=absolute_tolerance_us,
        )
        tail_recall = _tail_recall_summary(
            source_report.get("samples", []),
            canary_report.get("samples", []),
            threshold=tail_recall_threshold,
        )
        checks_pass = (
            all(check["status"] == "pass" for check in metric_checks)
            and all(check["status"] == "pass" for check in queue_checks)
            and all(check["status"] == "pass" for check in phase_checks)
            and all(check["status"] == "pass" for check in op_checks)
            and tail_recall["status"] == "pass"
        )
        source_metrics = {
            metric: source_report["metrics"][metric]
            for metric in (*_BEHAVIORAL_LATENCY_METRICS, "communication_hidden_pct")
        }
        canary_metrics = {
            metric: canary_report["metrics"][metric]
            for metric in (*_BEHAVIORAL_LATENCY_METRICS, "communication_hidden_pct")
        }
        config_rows.append(
            {
                "name": name,
                "status": "pass" if checks_pass else "fail",
                "source_metrics": source_metrics,
                "canary_metrics": canary_metrics,
                "source_queue_wait_metrics": source_queue,
                "canary_queue_wait_metrics": canary_queue,
                "checks": metric_checks,
                "queue_wait_checks": queue_checks,
                "phase_checks": phase_checks,
                "op_checks": op_checks,
                "tail_event_recall": tail_recall,
            }
        )

    ranking = _pairwise_ranking_summary(
        config_rows,
        metrics=BEHAVIORAL_RANKING_METRICS,
        tie_tolerance_us=ranking_tie_tolerance_us,
    )
    behavioral_fidelity_status = "pass" if all(row["status"] == "pass" for row in config_rows) else "fail"
    configuration_ranking_status = ranking["status"]
    uncertainty = _behavior_capture_uncertainty(canary)

    passed = (
        source_coverage_status == "full_source"
        and source_verified_status == "source_verified"
        and behavioral_fidelity_status == "pass"
        and configuration_ranking_status == "pass"
        and uncertainty["status"] == "certain"
    )
    if passed:
        status = "behaviorally_verified"
    elif (
        source_coverage_status == "full_source"
        and source_verified_status == "source_verified"
        and behavioral_fidelity_status == "pass"
        and configuration_ranking_status == "pass"
        and uncertainty["status"] != "certain"
    ):
        status = "behaviorally_unverified"
    else:
        status = "failed"

    return {
        "format": BEHAVIOR_VERIFICATION_FORMAT,
        "status": status,
        "assurance_state": (
            "behaviorally_verified"
            if status == "behaviorally_verified"
            else (
                "source_corresponding"
                if source_verified_status in {"source_verified", "partial_source_verified"}
                else "internally_consistent"
            )
        ),
        "representation_fidelity_status": representation_fidelity_status,
        "source_verified_status": source_verified_status,
        "source_coverage_status": source_coverage_status,
        "behavioral_fidelity_status": behavioral_fidelity_status,
        "configuration_ranking_status": configuration_ranking_status,
        "source_events": source_events,
        "relative_tolerance_pct": relative_tolerance_pct,
        "absolute_tolerance_us": absolute_tolerance_us,
        "hidden_tolerance_points": hidden_tolerance_points,
        "tail_recall_threshold": tail_recall_threshold,
        "ranking_tie_tolerance_us": ranking_tie_tolerance_us,
        "capture_uncertainty": uncertainty,
        "source_verification": source_verification,
        "configurations": config_rows,
        "ranking": ranking,
    }


def _behavior_count_check(source_metrics: Mapping[str, Any], canary_metrics: Mapping[str, Any]) -> JsonDict:
    source_count = as_int(source_metrics.get("count"))
    canary_count = as_int(canary_metrics.get("count"))
    return {
        "metric": "count",
        "status": "pass" if source_count == canary_count else "fail",
        "source": source_count,
        "canary": canary_count,
        "absolute_delta": canary_count - source_count,
        "relative_delta_pct": (
            None if source_count == 0 else round((canary_count - source_count) / source_count * 100.0, 2)
        ),
    }


def _behavior_metric_check(
    metric: str,
    source_metrics: Mapping[str, Any],
    canary_metrics: Mapping[str, Any],
    *,
    relative_tolerance_pct: float,
    absolute_tolerance_us: float,
    hidden_tolerance_points: float,
) -> JsonDict:
    source_value = as_float(source_metrics.get(metric))
    canary_value = as_float(canary_metrics.get(metric))
    absolute_delta = canary_value - source_value
    if metric == "communication_hidden_pct":
        passed = abs(absolute_delta) <= hidden_tolerance_points
        relative_delta = None
    else:
        relative_delta = None if source_value == 0.0 else absolute_delta / source_value * 100.0
        passed = abs(absolute_delta) <= absolute_tolerance_us or (
            relative_delta is not None and abs(relative_delta) <= relative_tolerance_pct
        )
    return {
        "metric": metric,
        "status": "pass" if passed else "fail",
        "source": round(source_value, 3),
        "canary": round(canary_value, 3),
        "absolute_delta": round(absolute_delta, 3),
        "relative_delta_pct": None if relative_delta is None else round(relative_delta, 2),
    }


def _sample_distribution(samples: Any, field: str) -> JsonDict:
    values = [as_float(sample.get(field)) for sample in samples if isinstance(sample, Mapping) and field in sample]
    return summarize_latencies(values)


def _breakdown_behavior_checks(
    scope: str,
    source_rows: Any,
    canary_rows: Any,
    *,
    relative_tolerance_pct: float,
    absolute_tolerance_us: float,
) -> List[JsonDict]:
    source = {
        str(row.get("name")): row
        for row in source_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    canary = {
        str(row.get("name")): row
        for row in canary_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    checks: List[JsonDict] = []
    for name in sorted(set(source) | set(canary)):
        if name not in source or name not in canary:
            checks.append(
                {
                    "scope": scope,
                    "name": name,
                    "metric": "presence",
                    "status": "fail",
                    "source_present": name in source,
                    "canary_present": name in canary,
                }
            )
            continue
        for metric in _BEHAVIORAL_LATENCY_METRICS:
            check = _behavior_metric_check(
                metric,
                source[name],
                canary[name],
                relative_tolerance_pct=relative_tolerance_pct,
                absolute_tolerance_us=absolute_tolerance_us,
                hidden_tolerance_points=0.0,
            )
            check["scope"] = scope
            check["name"] = name
            checks.append(check)
    return checks


def _tail_recall_summary(
    source_samples: Any,
    canary_samples: Any,
    *,
    threshold: float,
) -> JsonDict:
    checks = [
        _tail_recall_check(source_samples, canary_samples, quantile=95.0, threshold=threshold),
        _tail_recall_check(source_samples, canary_samples, quantile=99.0, threshold=threshold),
    ]
    return {
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "checks": checks,
    }


def _tail_recall_check(
    source_samples: Any,
    canary_samples: Any,
    *,
    quantile: float,
    threshold: float,
) -> JsonDict:
    source_values = [
        (as_int(sample.get("index")), as_float(sample.get("exposed_us")))
        for sample in source_samples
        if isinstance(sample, Mapping) and "index" in sample and "exposed_us" in sample
    ]
    canary_values = [
        (as_int(sample.get("index")), as_float(sample.get("exposed_us")))
        for sample in canary_samples
        if isinstance(sample, Mapping) and "index" in sample and "exposed_us" in sample
    ]
    if not source_values or not canary_values:
        return {"quantile": quantile, "status": "pass", "source_tail_count": 0, "recall": 1.0}
    cutoff = percentile((value for _index, value in source_values), quantile)
    source_tail = {index for index, value in source_values if value >= cutoff}
    if not source_tail:
        return {"quantile": quantile, "status": "pass", "source_tail_count": 0, "recall": 1.0}
    top_count = len(source_tail)
    canary_top = {
        index for index, _value in sorted(canary_values, key=lambda item: (item[1], -item[0]), reverse=True)[:top_count]
    }
    overlap = len(source_tail & canary_top)
    recall = overlap / len(source_tail)
    return {
        "quantile": quantile,
        "status": "pass" if recall >= threshold else "fail",
        "source_threshold_us": round(cutoff, 3),
        "source_tail_count": len(source_tail),
        "canary_top_count": len(canary_top),
        "overlap_count": overlap,
        "recall": round(recall, 4),
        "required_recall": threshold,
    }


def _pairwise_ranking_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    tie_tolerance_us: float,
) -> JsonDict:
    pair_checks: List[JsonDict] = []
    for metric in metrics:
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left = rows[left_index]
                right = rows[right_index]
                left_name = str(left.get("name"))
                right_name = str(right.get("name"))
                source_relation = ranking_relation(
                    as_float(left.get("source_metrics", {}).get(metric)),
                    as_float(right.get("source_metrics", {}).get(metric)),
                    tie_tolerance_us,
                )
                canary_relation = ranking_relation(
                    as_float(left.get("canary_metrics", {}).get(metric)),
                    as_float(right.get("canary_metrics", {}).get(metric)),
                    tie_tolerance_us,
                )
                status = "pass" if source_relation == canary_relation else "fail"
                pair_checks.append(
                    {
                        "metric": metric,
                        "left": left_name,
                        "right": right_name,
                        "status": status,
                        "source_relation": source_relation,
                        "canary_relation": canary_relation,
                        "source_left": as_float(left.get("source_metrics", {}).get(metric)),
                        "source_right": as_float(right.get("source_metrics", {}).get(metric)),
                        "canary_left": as_float(left.get("canary_metrics", {}).get(metric)),
                        "canary_right": as_float(right.get("canary_metrics", {}).get(metric)),
                    }
                )
    passed = sum(1 for check in pair_checks if check["status"] == "pass")
    total = len(pair_checks)
    return {
        "metric": "pairwise_latency_metrics",
        "status": "pass" if passed == total else "fail",
        "agreement": round(passed / total, 4) if total else 1.0,
        "passed_pairs": passed,
        "total_pairs": total,
        "tie_tolerance_us": tie_tolerance_us,
        "orders": {
            metric: {
                "source_order": _metric_order(rows, "source_metrics", metric),
                "canary_order": _metric_order(rows, "canary_metrics", metric),
            }
            for metric in metrics
        },
        "pairwise": pair_checks,
    }


def _metric_order(rows: Sequence[Mapping[str, Any]], key: str, metric: str) -> List[str]:
    return [
        str(row.get("name"))
        for row in sorted(rows, key=lambda row: (as_float(row.get(key, {}).get(metric)), str(row.get("name"))))
    ]


def _behavior_capture_uncertainty(canary: Mapping[str, Any]) -> JsonDict:
    compiler = canary.get("compiler", {})
    raw = compiler.get("capture_uncertainty", {}) if isinstance(compiler, Mapping) else {}
    count = 0
    if isinstance(raw, Mapping):
        count = as_int(raw.get("compute_fields_uncertain_events"), 0)
    return {
        "status": "certain" if count == 0 else "rank_local_compute_uncertain",
        "compute_fields_uncertain_events": count,
    }


def _optional_non_negative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed


__all__ = ["verify_canary_behavior"]
