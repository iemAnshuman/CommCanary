from __future__ import annotations

import copy
import hashlib
import json
from bisect import bisect_left
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    CANARY_FORMAT,
    TRACE_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    arrival_skew_us,
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    median,
    normalize_arrival_offsets,
    normalize_ranks,
    percentile,
    summarize_latencies,
    validate_canary,
    validate_trace,
)

DEFAULT_TIMING_SAMPLE_LIMIT = 128
_US_TOLERANCE = 1e-6

_FIDELITY_FIELDS = (
    "max_gap_error_us",
    "max_skew_error_us",
    "max_arrival_offset_error_us",
    "max_compute_before_error_us",
    "max_overlap_error_us",
    "max_pressure_error",
    "max_observed_exposed_error_us",
    "max_prefix_gap_error_us",
)
_BEHAVIORAL_LATENCY_METRICS = ("median_us", "p95_us", "p99_us", "max_us", "mean_us")
_BEHAVIORAL_RANKING_METRICS = ("median_us", "p95_us", "p99_us", "mean_us")
_DEFAULT_BEHAVIORAL_CONFIGS = (
    {
        "name": "baseline",
        "bandwidth_gbps": 55.0,
        "latency_floor_us": 7.5,
        "compute_pressure": 0.55,
        "overlap_efficiency": 0.72,
        "seed": 7,
    },
    {
        "name": "low_latency",
        "bandwidth_gbps": 55.0,
        "latency_floor_us": 3.5,
        "compute_pressure": 0.55,
        "overlap_efficiency": 0.72,
        "seed": 7,
    },
    {
        "name": "high_bandwidth",
        "bandwidth_gbps": 110.0,
        "latency_floor_us": 10.0,
        "compute_pressure": 0.55,
        "overlap_efficiency": 0.72,
        "seed": 7,
    },
    {
        "name": "overlap_friendly",
        "bandwidth_gbps": 55.0,
        "latency_floor_us": 9.0,
        "compute_pressure": 0.55,
        "overlap_efficiency": 0.95,
        "seed": 7,
    },
    {
        "name": "congested",
        "bandwidth_gbps": 28.0,
        "latency_floor_us": 7.5,
        "compute_pressure": 0.95,
        "overlap_efficiency": 0.72,
        "seed": 7,
    },
)


def compile_trace(
    trace: Mapping[str, Any],
    *,
    max_events: Optional[int] = None,
    timing_sample_limit: int = DEFAULT_TIMING_SAMPLE_LIMIT,
    max_gap_error_us: Optional[float] = None,
    max_skew_error_us: Optional[float] = None,
    max_arrival_offset_error_us: Optional[float] = None,
    max_compute_before_error_us: Optional[float] = None,
    max_overlap_error_us: Optional[float] = None,
    max_pressure_error: Optional[float] = None,
    max_observed_exposed_error_us: Optional[float] = None,
    max_prefix_gap_error_us: Optional[float] = None,
    require_lossless_timing: bool = False,
    allow_empty: bool = False,
    enable_sequence_motifs: bool = True,
) -> JsonDict:
    """Compile a communication trace into a compact, fidelity-audited canary.

    Exact run/pattern encodings are preferred. If the timing stream cannot fit
    within ``timing_sample_limit``, ordered bounded intervals are emitted with
    explicit approximation errors. Optional budgets make compilation fail
    closed rather than silently exceeding an acceptable error.
    """

    validate_trace(trace)
    if not isinstance(require_lossless_timing, bool):
        raise SchemaError("require_lossless_timing must be a boolean")
    if not isinstance(allow_empty, bool):
        raise SchemaError("allow_empty must be a boolean")
    if not isinstance(enable_sequence_motifs, bool):
        raise SchemaError("enable_sequence_motifs must be a boolean")

    parsed_max_events: Optional[int]
    if max_events is None:
        parsed_max_events = None
    else:
        parsed_max_events = as_int(max_events)
        if parsed_max_events < 0:
            raise SchemaError("max_events must be non-negative")

    timing_sample_limit = as_int(timing_sample_limit)
    if timing_sample_limit < 2:
        raise SchemaError("timing_sample_limit must be at least 2")

    budgets = {
        "max_gap_error_us": _optional_non_negative(max_gap_error_us, "max_gap_error_us"),
        "max_skew_error_us": _optional_non_negative(max_skew_error_us, "max_skew_error_us"),
        "max_arrival_offset_error_us": _optional_non_negative(
            max_arrival_offset_error_us, "max_arrival_offset_error_us"
        ),
        "max_compute_before_error_us": _optional_non_negative(
            max_compute_before_error_us, "max_compute_before_error_us"
        ),
        "max_overlap_error_us": _optional_non_negative(max_overlap_error_us, "max_overlap_error_us"),
        "max_pressure_error": _optional_non_negative(max_pressure_error, "max_pressure_error"),
        "max_observed_exposed_error_us": _optional_non_negative(
            max_observed_exposed_error_us, "max_observed_exposed_error_us"
        ),
        "max_prefix_gap_error_us": _optional_non_negative(
            max_prefix_gap_error_us, "max_prefix_gap_error_us"
        ),
    }

    ordered_events, ordered_gaps, timing_mode = _ordered_trace_events(list(trace.get("events", [])))
    if parsed_max_events is not None:
        ordered_events = ordered_events[:parsed_max_events]
        ordered_gaps = ordered_gaps[:parsed_max_events]
    if not ordered_events and not allow_empty:
        raise SchemaError("cannot compile an empty trace without allow_empty=True")

    observed_flags = ["observed_exposed_us" in event for event in ordered_events]
    if any(observed_flags) and not all(observed_flags):
        raise SchemaError("observed_exposed_us must be present on every selected trace event or none")
    has_observed_tail = bool(observed_flags and all(observed_flags))

    canary_events: List[JsonDict] = []
    signature_occurrences: Dict[Tuple[Any, ...], int] = {}
    for source_index, (event, gap_us) in enumerate(zip(ordered_events, ordered_gaps)):
        step = _event_to_step(
            event,
            source_index=source_index,
            gap_us=gap_us,
            sample_limit=timing_sample_limit,
        )
        signature = _signature(step)
        step["_execution_occurrence_base"] = signature_occurrences.get(signature, 0)
        signature_occurrences[signature] = step["_execution_occurrence_base"] + 1
        if canary_events and canary_events[-1].get("_signature") == signature:
            _append_sample(canary_events[-1], step)
        else:
            step["_signature"] = signature
            canary_events.append(step)

    flat_finalized = [_finalize_step(step) for step in canary_events]
    finalized = _compress_sequence_motifs(flat_finalized) if enable_sequence_motifs else flat_finalized
    # Release the uncompressed timing streams before serialising the source
    # trace. This avoids holding two large canonical representations at once.
    canary_events.clear()

    source_count = len(ordered_events)
    compiled_count = len(finalized)
    logical_events = list(iter_canary_logical_events(finalized))
    stored_leaf_events = list(iter_canary_stored_leaf_events(finalized))
    expanded_count = sum(as_int(event.get("repeat"), 1) for event in logical_events)
    if expanded_count != source_count:
        raise SchemaError("sequence motif compression changed the logical event count")
    event_ratio = round(source_count / compiled_count, 3) if compiled_count else 0.0
    recursive_records = sum(_recursive_timing_record_count(event.get("timing_samples")) for event in logical_events)
    approximate_records = sum(_approximate_record_count(event.get("timing_samples")) for event in logical_events)
    stored_recursive_records = sum(_recursive_timing_record_count(event.get("timing_samples")) for event in stored_leaf_events)
    stored_approximate_records = sum(_approximate_record_count(event.get("timing_samples")) for event in stored_leaf_events)
    compute_uncertain_events = sum(
        _timing_records_uncertain_weight(event.get("timing_samples"))
        for event in logical_events
    )
    sequence_motif_count = sum(
        1 for event in finalized if isinstance(event, Mapping) and event.get("program") == "sequence_motif"
    )

    source_gap_total = _round_us(sum(ordered_gaps))
    encoded_gap_total = _round_us(
        sum(
            _timing_records_gap_sum(event.get("timing_samples", []))
            for event in logical_events
        )
    )
    total_gap_error = _round_us(abs(source_gap_total - encoded_gap_total))
    if total_gap_error > _US_TOLERANCE:
        raise SchemaError(
            f"timing compression changed total gap duration by {total_gap_error} us"
        )

    fidelity = _summarize_fidelity(
        logical_events,
        source_gap_total=source_gap_total,
        encoded_gap_total=encoded_gap_total,
        total_gap_error=total_gap_error,
    )
    if require_lossless_timing and fidelity["mode"] != "lossless_timing":
        raise SchemaError("lossless timing was requested but bounded approximation was required")
    _enforce_fidelity_budgets(fidelity, budgets)

    measured_trace = dict(trace)
    measured_trace["events"] = ordered_events
    source_bytes = _json_size(measured_trace)
    source_sha = hashlib.sha256(_canonical_json_bytes(measured_trace)).hexdigest()

    canary: JsonDict = {
        "format": CANARY_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_format": trace.get("format", TRACE_FORMAT),
        "workload": dict(trace.get("workload", {})),
        "system": dict(trace.get("system", {})),
        "compiler": {
            "compression": "ordered exact patterns, replay-equivalent sequence motifs, and fidelity-audited bounded intervals",
            "timing_sample_limit": timing_sample_limit,
            "timing_mode": timing_mode,
            "tail_signal": "observed_exposed_us" if has_observed_tail else "structural-proxy",
            "source_events": source_count,
            "canary_events": compiled_count,
            "expanded_canary_events": len(logical_events),
            "sequence_motif_count": sequence_motif_count,
            "compression_ratio": event_ratio,
            "event_compression_ratio": event_ratio,
            "recursive_timing_records": recursive_records,
            "approximate_timing_records": approximate_records,
            "stored_recursive_timing_records": stored_recursive_records,
            "stored_approximate_timing_records": stored_approximate_records,
            "source_bytes": source_bytes,
            "source_trace_sha256": source_sha,
            "source_normalized_sha256": source_sha,
            "fidelity": fidelity,
            "fidelity_budget": {key: value for key, value in budgets.items() if value is not None},
        },
        "events": finalized,
    }
    if compute_uncertain_events:
        canary["compiler"]["capture_uncertainty"] = {
            "compute_fields_uncertain_events": compute_uncertain_events,
            "status": "rank_local_compute_fields_uncertain",
        }
    canary["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(canary)
    canary["compiler"]["scheduler_execution_sha256"] = canary_scheduler_execution_sha256(canary)
    canary["compiler"]["calibration_evaluation_sha256"] = canary_calibration_sha256(canary)
    canary["compiler"]["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)
    _update_size_metrics(canary)
    validate_canary(canary)
    return canary


def verify_canary_fidelity(trace: Mapping[str, Any], canary: Mapping[str, Any]) -> JsonDict:
    """Recompute compiler fidelity from source trace and compare it to a canary."""

    validate_canary(canary)
    compiler = canary.get("compiler", {})
    source_events = as_int(compiler.get("source_events"))
    trace_events = trace.get("events", [])
    trace_event_count = len(trace_events) if isinstance(trace_events, list) else 0
    max_events = source_events if source_events < trace_event_count else None
    expected = compile_trace(
        trace,
        max_events=max_events,
        timing_sample_limit=as_int(compiler.get("timing_sample_limit"), DEFAULT_TIMING_SAMPLE_LIMIT),
        allow_empty=source_events == 0,
    )
    expected_compiler = expected.get("compiler", {})
    source_commitments = _verify_source_commitments(
        trace,
        canary,
        max_events=max_events,
        timing_sample_limit=as_int(compiler.get("timing_sample_limit"), DEFAULT_TIMING_SAMPLE_LIMIT),
    )
    checks = [
        _verification_check(
            "source_trace_sha256",
            expected_compiler.get("source_trace_sha256"),
            compiler.get("source_trace_sha256"),
        ),
        _verification_check(
            "source_normalized_sha256",
            expected_compiler.get("source_normalized_sha256"),
            compiler.get("source_normalized_sha256"),
        ),
        _verification_check(
            "execution_semantic_sha256",
            expected_compiler.get("execution_semantic_sha256"),
            compiler.get("execution_semantic_sha256"),
        ),
        _verification_check(
            "scheduler_execution_sha256",
            expected_compiler.get("scheduler_execution_sha256"),
            compiler.get("scheduler_execution_sha256"),
        ),
        _verification_check(
            "calibration_evaluation_sha256",
            expected_compiler.get("calibration_evaluation_sha256"),
            compiler.get("calibration_evaluation_sha256"),
        ),
        _verification_check("fidelity", expected_compiler.get("fidelity"), compiler.get("fidelity")),
        _verification_check(
            "recursive_timing_records",
            expected_compiler.get("recursive_timing_records"),
            compiler.get("recursive_timing_records"),
        ),
        _verification_check(
            "approximate_timing_records",
            expected_compiler.get("approximate_timing_records"),
            compiler.get("approximate_timing_records"),
        ),
        source_commitments,
    ]
    passed = all(check["status"] == "pass" for check in checks)
    return {
        "format": "commcanary.fidelity_verification.v1",
        "status": "source_verified" if passed else "failed",
        "source_events": source_events,
        "timing_sample_limit": as_int(compiler.get("timing_sample_limit"), DEFAULT_TIMING_SAMPLE_LIMIT),
        "checks": checks,
    }


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
) -> JsonDict:
    """Replay source and canary canaries and compare behavioral metrics.

    This verifier separates four questions that are easy to conflate in a
    compressed trace artifact: representation fidelity, source verification,
    simulator-visible behavior, and pairwise configuration ranking.
    """

    from .replay import replay_canary

    validate_canary(canary)
    relative_tolerance_pct = _optional_non_negative(relative_tolerance_pct, "relative_tolerance_pct") or 0.0
    absolute_tolerance_us = _optional_non_negative(absolute_tolerance_us, "absolute_tolerance_us") or 0.0
    hidden_tolerance_points = _optional_non_negative(hidden_tolerance_points, "hidden_tolerance_points") or 0.0
    tail_recall_threshold = _optional_non_negative(tail_recall_threshold, "tail_recall_threshold") or 0.0
    ranking_tie_tolerance_us = _optional_non_negative(ranking_tie_tolerance_us, "ranking_tie_tolerance_us") or 0.0
    if tail_recall_threshold > 1.0:
        raise SchemaError("tail_recall_threshold must be between 0 and 1")

    compiler = canary.get("compiler", {})
    source_events = as_int(compiler.get("source_events"))
    fidelity = compiler.get("fidelity", {}) if isinstance(compiler.get("fidelity"), Mapping) else {}
    representation_fidelity_status = str(fidelity.get("mode", "missing_fidelity_metadata"))

    try:
        source_verification = verify_canary_fidelity(trace, canary)
        source_verified_status = str(source_verification.get("status"))
    except SchemaError as exc:
        source_verification = {
            "format": "commcanary.fidelity_verification.v1",
            "status": "failed",
            "checks": [{"name": "source_verification_exception", "status": "fail", "reason": str(exc)}],
        }
        source_verified_status = "failed"

    trace_events = trace.get("events", [])
    trace_event_count = len(trace_events) if isinstance(trace_events, list) else 0
    max_events = source_events if source_events < trace_event_count else None
    full_canary = compile_trace(
        trace,
        max_events=max_events,
        timing_sample_limit=max(2, source_events),
        require_lossless_timing=True,
        allow_empty=source_events == 0,
    )
    config_rows: List[JsonDict] = []
    for raw_config in configurations or _DEFAULT_BEHAVIORAL_CONFIGS:
        config = dict(raw_config)
        name = str(config.pop("name", f"config-{len(config_rows)}"))
        replay_args = _behavioral_replay_args(config)
        source_report = replay_canary(full_canary, backend_label=name, include_samples=True, **replay_args)
        canary_report = replay_canary(canary, backend_label=name, include_samples=True, **replay_args)

        metric_checks = [
            _behavior_metric_check(
                metric,
                source_report["metrics"],
                canary_report["metrics"],
                relative_tolerance_pct=relative_tolerance_pct,
                absolute_tolerance_us=absolute_tolerance_us,
                hidden_tolerance_points=hidden_tolerance_points,
            )
            for metric in _BEHAVIORAL_LATENCY_METRICS
        ]
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
        metrics=_BEHAVIORAL_RANKING_METRICS,
        tie_tolerance_us=ranking_tie_tolerance_us,
    )
    behavioral_fidelity_status = "pass" if all(row["status"] == "pass" for row in config_rows) else "fail"
    configuration_ranking_status = ranking["status"]
    uncertainty = _behavior_capture_uncertainty(canary)

    passed = (
        source_verified_status == "source_verified"
        and behavioral_fidelity_status == "pass"
        and configuration_ranking_status == "pass"
        and uncertainty["status"] == "certain"
    )
    if passed:
        status = "behaviorally_verified"
    elif (
        source_verified_status == "source_verified"
        and behavioral_fidelity_status == "pass"
        and configuration_ranking_status == "pass"
        and uncertainty["status"] != "certain"
    ):
        status = "behaviorally_unverified"
    else:
        status = "failed"

    return {
        "format": "commcanary.behavior_verification.v1",
        "status": status,
        "representation_fidelity_status": representation_fidelity_status,
        "source_verified_status": source_verified_status,
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


def _verify_source_commitments(
    trace: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    max_events: Optional[int],
    timing_sample_limit: int,
) -> JsonDict:
    ordered_events, ordered_gaps, _timing_mode = _ordered_trace_events(list(trace.get("events", [])))
    if max_events is not None:
        ordered_events = ordered_events[:max_events]
        ordered_gaps = ordered_gaps[:max_events]
    source_steps = [
        _event_to_step(
            event,
            source_index=index,
            gap_us=gap_us,
            sample_limit=timing_sample_limit,
        )
        for index, (event, gap_us) in enumerate(zip(ordered_events, ordered_gaps))
    ]
    failures: List[JsonDict] = []
    checked_intervals = 0
    pointer = 0
    for event_index, event in enumerate(iter_canary_logical_events(canary.get("events", []))):
        if not isinstance(event, Mapping):
            continue
        repeat = as_int(event.get("repeat"), 1)
        source_slice = source_steps[pointer : pointer + repeat]
        if len(source_slice) != repeat:
            failures.append(
                {
                    "event_index": event_index,
                    "reason": "source slice shorter than canary repeat",
                    "expected": repeat,
                    "actual": len(source_slice),
                }
            )
            break
        event_signature = _signature(event)
        for local_index, source_step in enumerate(source_slice):
            if _signature(source_step) != event_signature:
                failures.append(
                    {
                        "event_index": event_index,
                        "source_local_index": local_index,
                        "reason": "source event signature does not match canary event",
                        "source_signature": list(_signature(source_step)),
                        "canary_signature": list(event_signature),
                    }
                )
                break
        source_samples = [step["timing_samples"][0] for step in source_slice]
        for record in _walk_timing_records(event.get("timing_samples")):
            if record.get("approximation") != "bounded_interval":
                continue
            checked_intervals += 1
            start = as_int(record.get("source_start"))
            end = as_int(record.get("source_end"))
            if end >= len(source_samples):
                failures.append(
                    {
                        "event_index": event_index,
                        "source_start": start,
                        "source_end": end,
                        "reason": "bounded interval exceeds source slice",
                    }
                )
                continue
            expected = _recompute_interval_commitment(source_samples[start : end + 1], record, source_start=start)
            mismatch = _first_commitment_mismatch(expected, record)
            if mismatch is not None:
                failures.append({"event_index": event_index, **mismatch})
        pointer += repeat
    if pointer != len(source_steps):
        failures.append(
            {
                "reason": "canary events do not consume all selected source events",
                "consumed": pointer,
                "source_events": len(source_steps),
            }
        )
    return {
        "name": "source_commitments",
        "status": "pass" if not failures else "fail",
        "checked_bounded_intervals": checked_intervals,
        "failures": failures[:20],
    }


def _recompute_interval_commitment(
    segment: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    *,
    source_start: int,
) -> JsonDict:
    weight = len(segment)
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in segment)
    encoded_gap = as_float(record.get("gap_us"), 0.0)
    offsets = [_round_us(as_float(value)) for value in record.get("arrival_offsets_us", [])]
    representative_skew = arrival_skew_us(offsets)
    max_offset_error = 0.0
    for sample in segment:
        source_offsets = [as_float(value) for value in sample.get("arrival_offsets_us", [])]
        if len(source_offsets) == len(offsets):
            max_offset_error = max(
                max_offset_error,
                max((abs(left - right) for left, right in zip(source_offsets, offsets)), default=0.0),
            )
    prefix_source = 0.0
    prefix_encoded = 0.0
    max_prefix_error = 0.0
    for sample in segment:
        prefix_source += as_float(sample.get("gap_us"), 0.0)
        prefix_encoded += encoded_gap
        max_prefix_error = max(max_prefix_error, abs(prefix_source - prefix_encoded))
    representative_source_index = as_int(record.get("representative_source_index"))
    representative_local_index = representative_source_index - source_start
    representative_gap_error = 0.0
    if 0 <= representative_local_index < len(segment):
        representative_gap_error = abs(as_float(segment[representative_local_index].get("gap_us"), 0.0) - encoded_gap)
    errors: JsonDict = {
        "max_gap_error_us": _round_us(
            max(abs(as_float(sample.get("gap_us"), 0.0) - encoded_gap) for sample in segment)
            if segment
            else 0.0
        ),
        "max_skew_error_us": _round_us(
            max(abs(as_float(sample.get("arrival_skew_us"), 0.0) - representative_skew) for sample in segment)
            if segment
            else 0.0
        ),
        "max_arrival_offset_error_us": _round_us(max_offset_error),
        "max_compute_before_error_us": _round_us(
            max(
                abs(as_float(sample.get("compute_before_us"), 0.0) - as_float(record.get("compute_before_us"), 0.0))
                for sample in segment
            )
            if segment
            else 0.0
        ),
        "max_overlap_error_us": _round_us(
            max(
                abs(as_float(sample.get("compute_overlap_us"), 0.0) - as_float(record.get("compute_overlap_us"), 0.0))
                for sample in segment
            )
            if segment
            else 0.0
        ),
        "max_pressure_error": round(
            max(
                abs(as_float(sample.get("compute_pressure"), 0.5) - as_float(record.get("compute_pressure"), 0.5))
                for sample in segment
            )
            if segment
            else 0.0,
            6,
        ),
        "representative_gap_error_us": _round_us(representative_gap_error),
        "max_prefix_gap_error_us": _round_us(max_prefix_error),
    }
    if "observed_exposed_us" in record:
        errors["max_observed_exposed_error_us"] = _round_us(
            max(
                abs(as_float(sample.get("observed_exposed_us")) - as_float(record.get("observed_exposed_us")))
                for sample in segment
            )
            if segment
            else 0.0
        )
    return {
        "source_count": weight,
        "source_gap_sum_us": _round_us(gap_sum_us),
        "source_segment_sha256": _source_segment_sha256(segment),
        "error_vector": errors,
        **errors,
    }


def _first_commitment_mismatch(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Optional[JsonDict]:
    for key in ("source_count", "source_gap_sum_us", "source_segment_sha256"):
        if actual.get(key) != expected.get(key):
            return {
                "source_start": actual.get("source_start"),
                "source_end": actual.get("source_end"),
                "field": key,
                "expected": expected.get(key),
                "actual": actual.get(key),
            }
    actual_vector = actual.get("error_vector", {})
    expected_vector = expected.get("error_vector", {})
    if not isinstance(actual_vector, Mapping):
        return {
            "source_start": actual.get("source_start"),
            "source_end": actual.get("source_end"),
            "field": "error_vector",
            "expected": expected_vector,
            "actual": actual_vector,
        }
    for key, expected_value in expected_vector.items():
        if key not in actual_vector:
            return {
                "source_start": actual.get("source_start"),
                "source_end": actual.get("source_end"),
                "field": f"error_vector.{key}",
                "expected": expected_value,
                "actual": None,
            }
        if abs(as_float(actual_vector.get(key)) - as_float(expected_value)) > 1e-6:
            return {
                "source_start": actual.get("source_start"),
                "source_end": actual.get("source_end"),
                "field": f"error_vector.{key}",
                "expected": expected_value,
                "actual": actual_vector.get(key),
            }
    return None


def _behavioral_replay_args(config: Mapping[str, Any]) -> JsonDict:
    allowed = {
        "bandwidth_gbps",
        "latency_floor_us",
        "compute_pressure",
        "overlap_efficiency",
        "iterations",
        "seed",
        "max_replay_events",
    }
    return {key: config[key] for key in allowed if key in config}


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
    values = [
        as_float(sample.get(field))
        for sample in samples
        if isinstance(sample, Mapping) and field in sample
    ]
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
        index
        for index, _value in sorted(canary_values, key=lambda item: (item[1], -item[0]), reverse=True)[:top_count]
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
                source_relation = _ranking_relation(
                    as_float(left.get("source_metrics", {}).get(metric)),
                    as_float(right.get("source_metrics", {}).get(metric)),
                    tie_tolerance_us,
                )
                canary_relation = _ranking_relation(
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


def _ranking_relation(left: float, right: float, tolerance: float) -> str:
    if abs(left - right) <= tolerance:
        return "tie"
    return "left_better" if left < right else "right_better"


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


def _verification_check(name: str, expected: Any, actual: Any) -> JsonDict:
    return {
        "name": name,
        "status": "pass" if expected == actual else "fail",
        "expected": expected,
        "actual": actual,
    }


def _ordered_trace_events(events: List[Mapping[str, Any]]) -> Tuple[List[Mapping[str, Any]], List[float], str]:
    if not events:
        return [], [], "empty"
    start_flags = ["start_us" in event for event in events]
    if all(start_flags):
        ordered = sorted(enumerate(events), key=lambda pair: (as_float(pair[1].get("start_us")), pair[0]))
        result = [event for _position, event in ordered]
        gaps: List[float] = []
        previous_start: Optional[float] = None
        for index, event in enumerate(result):
            start_us = as_float(event.get("start_us"))
            derived_gap = 0.0 if previous_start is None else start_us - previous_start
            if derived_gap < -_US_TOLERANCE:
                raise SchemaError("start_us values must be non-decreasing after ordering")
            derived_gap = max(0.0, derived_gap)
            if "gap_us" in event:
                explicit_gap = as_float(event.get("gap_us"))
                if index == 0:
                    gap_us = explicit_gap
                else:
                    if abs(explicit_gap - derived_gap) > 0.001:
                        raise SchemaError("gap_us conflicts with the difference between start_us values")
                    gap_us = derived_gap
            else:
                gap_us = derived_gap
            gaps.append(_round_us(gap_us))
            previous_start = start_us
        return result, gaps, "absolute_start_us"

    if not any(start_flags):
        result = list(events)
        gaps = [
            _round_us(
                as_float(event.get("gap_us"))
                if "gap_us" in event
                else as_float(event.get("compute_before_us"), 0.0)
            )
            for event in result
        ]
        return result, gaps, "relative_gap_us"

    # A partial absolute clock cannot be ordered safely. It is only usable when
    # every event also carries an explicit relative gap, in which case input
    # order is the authoritative sequence.
    if not all("gap_us" in event for event in events):
        raise SchemaError(
            "mixed timestamped and untimestamped events require explicit gap_us on every event"
        )
    return list(events), [_round_us(as_float(event.get("gap_us"))) for event in events], "explicit_relative_mixed"


def _event_to_step(
    event: Mapping[str, Any],
    *,
    source_index: int,
    gap_us: float,
    sample_limit: int,
) -> JsonDict:
    if event.get("arrival_skew_unknown"):
        raise SchemaError("cannot compile uncalibrated cross-rank arrival skew")
    ranks = normalize_ranks(event.get("ranks"))
    offsets = normalize_arrival_offsets(event, ranks)
    skew = arrival_skew_us(offsets)
    overlap_us = as_float(event.get("compute_overlap_us"), 0.0)
    compute_before_us = as_float(event.get("compute_before_us"), 0.0)
    compute_pressure = as_float(event.get("compute_pressure"), 0.5)
    observed_exposed = event.get("observed_exposed_us")
    source_id = event.get("id", source_index)

    timing_sample: JsonDict = {
        "gap_us": _round_us(gap_us),
        "arrival_offsets_us": [_round_us(value) for value in offsets],
        "arrival_skew_us": _round_us(skew),
        "compute_before_us": _round_us(compute_before_us),
        "compute_overlap_us": _round_us(overlap_us),
        "compute_pressure": round(compute_pressure, 6),
        "source_index": 0,
        "weight": 1,
    }
    if observed_exposed is not None:
        timing_sample["observed_exposed_us"] = _round_us(as_float(observed_exposed))
    if event.get("compute_fields_uncertain") is True:
        timing_sample["compute_fields_uncertain"] = True
        timing_sample["uncertain_weight"] = 1

    hasher = hashlib.sha256()
    _update_source_digest(hasher, source_id)
    step: JsonDict = {
        "phase": str(event.get("phase", "unknown")),
        "op": str(event.get("op")),
        "bytes": as_int(event.get("bytes")),
        "ranks": ranks,
        "rank_count": len(ranks),
        "group": str(event.get("group", "default")),
        "repeat": 1,
        "gap_us": _round_us(gap_us),
        "arrival_skew_us": _round_us(skew),
        "arrival_offsets_us": [_round_us(value) for value in offsets],
        "compute_before_us": _round_us(compute_before_us),
        "compute_overlap_us": _round_us(overlap_us),
        "compute_pressure": round(compute_pressure, 6),
        "concurrent_groups": as_int(event.get("concurrent_groups"), 1),
        "timing_samples": [timing_sample],
        "source": {"count": 1, "first_id": source_id, "last_id": source_id},
        "_source_hasher": hasher,
        "_all_timing_samples": [timing_sample],
        "_sample_limit": sample_limit,
    }
    if event.get("custom_op") is True:
        step["custom_op"] = True
    return step


def _append_sample(target: Dict[str, Any], sample: Mapping[str, Any]) -> None:
    current_repeat = as_int(target.get("repeat"), 1)
    source = target["source"]
    sample_source = sample["source"]
    source["count"] = as_int(source.get("count"), 1) + 1
    source["last_id"] = sample_source.get("last_id")
    _update_source_digest(target["_source_hasher"], sample_source.get("last_id"))
    timing_sample = dict(sample["timing_samples"][0])
    timing_sample["source_index"] = current_repeat
    target["_all_timing_samples"].append(timing_sample)
    target["repeat"] = current_repeat + 1


def _finalize_step(step: Dict[str, Any]) -> JsonDict:
    all_samples: List[JsonDict] = step.get("_all_timing_samples", step.get("timing_samples", []))
    timing_samples = _compress_timing_samples(
        all_samples,
        as_int(step.get("_sample_limit"), DEFAULT_TIMING_SAMPLE_LIMIT),
    )
    result = {key: value for key, value in step.items() if not key.startswith("_")}
    result["timing_samples"] = timing_samples
    result["gap_us"] = _round_us(median(as_float(sample.get("gap_us")) for sample in all_samples))
    result["arrival_skew_us"] = _round_us(
        median(as_float(sample.get("arrival_skew_us")) for sample in all_samples)
    )
    result["compute_overlap_us"] = _round_us(
        median(as_float(sample.get("compute_overlap_us")) for sample in all_samples)
    )
    result["compute_before_us"] = _round_us(
        median(as_float(sample.get("compute_before_us")) for sample in all_samples)
    )
    result["compute_pressure"] = round(
        median(as_float(sample.get("compute_pressure"), 0.5) for sample in all_samples), 6
    )
    result["arrival_offsets_us"] = list(all_samples[0].get("arrival_offsets_us", [])) if all_samples else []
    if all_samples and "observed_exposed_us" in all_samples[0]:
        result["observed_exposed_us"] = _round_us(
            median(as_float(sample.get("observed_exposed_us")) for sample in all_samples)
        )
    result["source"]["digest"] = step["_source_hasher"].hexdigest()
    result["execution_occurrence_base"] = as_int(step.get("_execution_occurrence_base"), 0)
    result["source"]["sampled_timing_records"] = _recursive_timing_record_count(timing_samples)
    if any(_timing_record_uncertain_weight(record) for record in _walk_timing_records(timing_samples)):
        result["compute_fields_uncertain"] = True
    else:
        result.pop("compute_fields_uncertain", None)
    return result


def _compress_sequence_motifs(events: List[JsonDict]) -> List[JsonDict]:
    if len(events) < 4:
        return events
    keys = [_sequence_motif_key(event) for event in events]
    output: List[JsonDict] = []
    index = 0
    motif_index = 0
    max_sequence_length = min(16, len(events) // 2)
    while index < len(events):
        best: Optional[Tuple[int, int, int]] = None
        max_here = min(max_sequence_length, (len(events) - index) // 2)
        for sequence_length in range(2, max_here + 1):
            sequence = keys[index : index + sequence_length]
            repeats = 1
            cursor = index + sequence_length
            while cursor + sequence_length <= len(events) and keys[cursor : cursor + sequence_length] == sequence:
                repeats += 1
                cursor += sequence_length
            if repeats < 2:
                continue
            saved_events = sequence_length * repeats - 1
            if best is None or saved_events > best[0] or (saved_events == best[0] and sequence_length > best[1]):
                best = (saved_events, sequence_length, repeats)
        if best is None:
            output.append(events[index])
            index += 1
            continue
        _saved, sequence_length, repeats = best
        output.append(
            _sequence_motif_record(
                events[index : index + sequence_length * repeats],
                sequence_length,
                repeats,
                motif_index,
            )
        )
        motif_index += 1
        index += sequence_length * repeats
    return output


def _sequence_motif_key(event: Mapping[str, Any]) -> str:
    template = _strip_sequence_source_fields(event)
    return hashlib.sha256(_canonical_json_bytes(template)).hexdigest()


def _strip_sequence_source_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        stripped: JsonDict = {}
        for key, child in value.items():
            if key in {"source", "execution_occurrence_base", "execution_occurrence_stride"}:
                continue
            stripped[key] = _strip_sequence_source_fields(child)
        return stripped
    if isinstance(value, list):
        return [_strip_sequence_source_fields(child) for child in value]
    return value


def _sequence_motif_record(events: Sequence[JsonDict], sequence_length: int, repeats: int, motif_index: int) -> JsonDict:
    template = [copy.deepcopy(event) for event in events[:sequence_length]]
    strides: Dict[Tuple[Any, ...], int] = {}
    for child in template:
        sig = _signature(child)
        strides[sig] = strides.get(sig, 0) + 1
    for child in template:
        child["execution_occurrence_stride"] = strides[_signature(child)]
    all_sources = [event.get("source", {}) for event in events if isinstance(event.get("source"), Mapping)]
    first_source = all_sources[0] if all_sources else {}
    last_source = all_sources[-1] if all_sources else {}
    source_count = sum(as_int(source.get("count"), 1) for source in all_sources)
    source_digest_inputs = [source.get("digest", [source.get("first_id"), source.get("last_id")]) for source in all_sources]
    source_digest = hashlib.sha256(_canonical_json_bytes({"sources": source_digest_inputs})).hexdigest()
    return {
        "program": "sequence_motif",
        "motif_id": f"sequence-{motif_index}",
        "program_repeats": repeats,
        "source": {
            "count": source_count,
            "first_id": first_source.get("first_id"),
            "last_id": last_source.get("last_id"),
            "digest": source_digest,
        },
        "events": template,
    }


def _signature(step: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        step.get("phase"),
        step.get("op"),
        step.get("bytes"),
        tuple(step.get("ranks", [])),
        step.get("group"),
        as_int(step.get("concurrent_groups"), 1),
        step.get("custom_op") is True,
    )


def _update_source_digest(hasher: Any, source_id: Any) -> None:
    try:
        encoded = json.dumps(source_id, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"source id is not JSON serializable: {exc}") from exc
    hasher.update(encoded)
    hasher.update(b"\0")


def _compress_timing_samples(samples: List[JsonDict], sample_limit: int) -> List[JsonDict]:
    if len(samples) <= sample_limit:
        return [_timing_record(samples[index], index, index) for index in range(len(samples))]

    pattern_length = _short_repeated_pattern_length(samples, sample_limit)
    if pattern_length:
        return [_pattern_record(samples, pattern_length)]

    prefix_pattern = _prefix_pattern_records(samples, sample_limit)
    if prefix_pattern:
        return prefix_pattern

    runs = _run_length_timing_samples(samples)
    if _recursive_timing_record_count(runs) <= sample_limit:
        return runs

    return _bounded_interval_records(samples, sample_limit)


def _bounded_interval_records(samples: List[JsonDict], sample_limit: int) -> List[JsonDict]:
    anchors = _important_timing_indices(samples, sample_limit)
    records = _records_for_anchors(samples, anchors)
    if len(records) > sample_limit:  # Defensive; the selector should already guarantee this.
        records = _stratified_interval_records(samples, sample_limit)
    return records


def _important_timing_indices(samples: List[JsonDict], sample_limit: int) -> List[int]:
    if not samples:
        return []
    if sample_limit <= 1:
        return [0]

    last_index = len(samples) - 1
    scores: Dict[int, float] = {0: float("inf"), last_index: float("inf")}
    gaps = [as_float(sample.get("gap_us"), 0.0) for sample in samples]
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    baseline_gap = median(positive_gaps) if positive_gaps else 0.0

    for index, gap_us in enumerate(gaps):
        previous_gap = gaps[index - 1] if index else gap_us
        if (gap_us == 0.0) != (previous_gap == 0.0):
            scores[index] = max(scores.get(index, 0.0), 1e9 + abs(gap_us - previous_gap))
        if gap_us > 0.0 and (baseline_gap == 0.0 or gap_us >= baseline_gap * 4.0):
            scores[index] = max(scores.get(index, 0.0), 1e10 + gap_us)

    observed = [
        as_float(sample.get("observed_exposed_us"))
        for sample in samples
        if "observed_exposed_us" in sample
    ]
    observed_p95 = percentile(observed, 95.0) if observed else None
    skews = [as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples]
    overlaps = [as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples]
    pressures = [as_float(sample.get("compute_pressure"), 0.5) for sample in samples]
    skew_p95 = percentile(skews, 95.0) if skews else 0.0
    overlap_p95 = percentile(overlaps, 95.0) if overlaps else 0.0
    pressure_p95 = percentile(pressures, 95.0) if pressures else 0.0

    backlog = 0.0
    previous_backlog = 0.0
    for index, sample in enumerate(samples):
        skew = skews[index]
        overlap = overlaps[index]
        pressure = pressures[index]
        # This is a sensitivity proxy, not a physical backend model: high skew,
        # pressure, and overlap are the windows most likely to change exposed
        # latency or backend ranking under different replay configurations.
        proxy_service_us = 8.0 + skew * 0.15 + pressure * 8.0
        backlog = max(0.0, backlog + proxy_service_us - gaps[index])
        if (previous_backlog == 0.0) != (backlog == 0.0):
            scores[index] = max(scores.get(index, 0.0), 9e11 + abs(backlog - previous_backlog))
        if skew_p95 > 0.0 and skew >= skew_p95:
            scores[index] = max(scores.get(index, 0.0), 8e11 + skew)
        if overlap_p95 > 0.0 and overlap >= overlap_p95:
            scores[index] = max(scores.get(index, 0.0), 7e11 + overlap)
        if pressure_p95 > 0.0 and pressure >= pressure_p95:
            scores[index] = max(scores.get(index, 0.0), 6e11 + pressure)
        if skew > 0.0 and overlap > 0.0:
            scores[index] = max(scores.get(index, 0.0), 8.5e11 + skew + overlap)
        previous_backlog = backlog

    for index in range(1, len(samples)):
        delta = _timing_delta(samples[index - 1], samples[index])
        if delta > 0.0:
            scores[index] = max(scores.get(index, 0.0), delta)
        previous_high = skews[index - 1] >= skew_p95 or overlaps[index - 1] >= overlap_p95
        current_high = skews[index] >= skew_p95 or overlaps[index] >= overlap_p95
        if previous_high != current_high:
            scores[index] = max(scores.get(index, 0.0), 5e11 + delta)
        if observed_p95 is not None:
            value = as_float(samples[index].get("observed_exposed_us"))
            if value >= observed_p95:
                scores[index] = max(scores.get(index, 0.0), 1e12 + value)

    candidates = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    anchors: List[int] = []
    for index, _score in candidates:
        position = bisect_left(anchors, index)
        if position < len(anchors) and anchors[position] == index:
            continue
        candidate = anchors[:position] + [index] + anchors[position:]
        if _record_count_for_anchors(len(samples), candidate) <= sample_limit:
            anchors = candidate
    return anchors or [0]


def _record_count_for_anchors(sample_count: int, anchors: Sequence[int]) -> int:
    count = 0
    previous = 0
    for index in anchors:
        if index < previous:
            continue
        if previous < index:
            count += 1
        count += 1
        previous = index + 1
    if previous < sample_count:
        count += 1
    return count


def _records_for_anchors(samples: List[JsonDict], anchors: Sequence[int]) -> List[JsonDict]:
    records: List[JsonDict] = []
    previous = 0
    for index in sorted(set(anchors)):
        if index < previous:
            continue
        if previous < index:
            records.append(_aggregate_interval_record(samples, previous, index))
        records.append(_timing_record(samples[index], index, index))
        previous = index + 1
    if previous < len(samples):
        records.append(_aggregate_interval_record(samples, previous, len(samples)))
    return records


def _timing_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    gap_delta = abs(as_float(left.get("gap_us"), 0.0) - as_float(right.get("gap_us"), 0.0))
    skew_delta = abs(
        as_float(left.get("arrival_skew_us"), 0.0) - as_float(right.get("arrival_skew_us"), 0.0)
    )
    before_delta = abs(
        as_float(left.get("compute_before_us"), 0.0) - as_float(right.get("compute_before_us"), 0.0)
    )
    overlap_delta = abs(
        as_float(left.get("compute_overlap_us"), 0.0) - as_float(right.get("compute_overlap_us"), 0.0)
    )
    pressure_delta = abs(
        as_float(left.get("compute_pressure"), 0.5) - as_float(right.get("compute_pressure"), 0.5)
    )
    observed_delta = 0.0
    if "observed_exposed_us" in left and "observed_exposed_us" in right:
        observed_delta = abs(
            as_float(left.get("observed_exposed_us")) - as_float(right.get("observed_exposed_us"))
        )
    return gap_delta + skew_delta + before_delta + overlap_delta + pressure_delta + observed_delta


def _aggregate_interval_record(samples: List[JsonDict], start: int, end: int) -> JsonDict:
    segment = samples[start:end]
    weight = len(segment)
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in segment)
    average_gap_us = gap_sum_us / weight
    representative = _joint_medoid_sample(segment, average_gap_us)
    representative_source_index = start + next(
        offset for offset, sample in enumerate(segment) if sample is representative
    )
    offsets = [_round_us(as_float(value)) for value in representative.get("arrival_offsets_us", [])]
    representative_skew = arrival_skew_us(offsets)

    max_offset_error = 0.0
    for sample in segment:
        source_offsets = [as_float(value) for value in sample.get("arrival_offsets_us", [])]
        if len(source_offsets) == len(offsets):
            max_offset_error = max(
                max_offset_error,
                max((abs(left - right) for left, right in zip(source_offsets, offsets)), default=0.0),
            )

    prefix_source = 0.0
    prefix_encoded = 0.0
    max_prefix_error = 0.0
    for sample in segment:
        prefix_source += as_float(sample.get("gap_us"), 0.0)
        prefix_encoded += average_gap_us
        max_prefix_error = max(max_prefix_error, abs(prefix_source - prefix_encoded))

    record: JsonDict = {
        "gap_us": _round_us(average_gap_us),
        "arrival_offsets_us": offsets,
        "arrival_skew_us": _round_us(representative_skew),
        "compute_before_us": _round_us(as_float(representative.get("compute_before_us"), 0.0)),
        "compute_overlap_us": _round_us(as_float(representative.get("compute_overlap_us"), 0.0)),
        "compute_pressure": round(as_float(representative.get("compute_pressure"), 0.5), 6),
        "source_index": start,
        "source_start": start,
        "source_end": end - 1,
        "weight": weight,
        "source_count": weight,
        "representative_source_index": representative_source_index,
        "source_gap_sum_us": _round_us(gap_sum_us),
        "gap_sum_us": _round_us(gap_sum_us),
        "approximation": "bounded_interval",
        "max_gap_error_us": _round_us(
            max(abs(as_float(sample.get("gap_us"), 0.0) - average_gap_us) for sample in segment)
        ),
        "max_skew_error_us": _round_us(
            max(
                abs(as_float(sample.get("arrival_skew_us"), 0.0) - representative_skew)
                for sample in segment
            )
        ),
        "max_arrival_offset_error_us": _round_us(max_offset_error),
        "max_compute_before_error_us": _round_us(
            max(
                abs(
                    as_float(sample.get("compute_before_us"), 0.0)
                    - as_float(representative.get("compute_before_us"), 0.0)
                )
                for sample in segment
            )
        ),
        "max_overlap_error_us": _round_us(
            max(
                abs(
                    as_float(sample.get("compute_overlap_us"), 0.0)
                    - as_float(representative.get("compute_overlap_us"), 0.0)
                )
                for sample in segment
            )
        ),
        "max_pressure_error": round(
            max(
                abs(
                    as_float(sample.get("compute_pressure"), 0.5)
                    - as_float(representative.get("compute_pressure"), 0.5)
                )
                for sample in segment
            ),
            6,
        ),
        "representative_gap_error_us": _round_us(
            abs(as_float(representative.get("gap_us"), 0.0) - average_gap_us)
        ),
        "max_prefix_gap_error_us": _round_us(max_prefix_error),
    }
    uncertain_weight = sum(_source_sample_uncertain_weight(sample) for sample in segment)
    if uncertain_weight:
        record["compute_fields_uncertain"] = True
        record["uncertain_weight"] = uncertain_weight
    if "observed_exposed_us" in representative:
        representative_observed = as_float(representative.get("observed_exposed_us"))
        record["observed_exposed_us"] = _round_us(representative_observed)
        record["max_observed_exposed_error_us"] = _round_us(
            max(
                abs(as_float(sample.get("observed_exposed_us")) - representative_observed)
                for sample in segment
            )
        )
    record["source_segment_sha256"] = _source_segment_sha256(segment)
    record["representative_selection_method"] = (
        "joint_medoid_normalized_l1_gap_skew_offsets_compute_overlap_pressure_observed"
    )
    error_fields = (
        "max_gap_error_us",
        "max_skew_error_us",
        "max_arrival_offset_error_us",
        "max_compute_before_error_us",
        "max_overlap_error_us",
        "max_pressure_error",
        "max_observed_exposed_error_us",
        "representative_gap_error_us",
        "max_prefix_gap_error_us",
    )
    record["error_vector"] = {field: record[field] for field in error_fields if field in record}
    return record


def _joint_medoid_sample(samples: List[JsonDict], average_gap_us: float) -> JsonDict:
    gap_scale = max(1.0, max(as_float(sample.get("gap_us"), 0.0) for sample in samples))
    skew_scale = max(1.0, max(as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples))
    before_scale = max(1.0, max(as_float(sample.get("compute_before_us"), 0.0) for sample in samples))
    overlap_scale = max(1.0, max(as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples))
    pressure_scale = max(1.0, max(as_float(sample.get("compute_pressure"), 0.5) for sample in samples))
    observed_values = [
        as_float(sample.get("observed_exposed_us"))
        for sample in samples
        if "observed_exposed_us" in sample
    ]
    observed_scale = max(1.0, max(observed_values, default=1.0))

    median_skew = median(as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples)
    median_before = median(as_float(sample.get("compute_before_us"), 0.0) for sample in samples)
    median_overlap = median(as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples)
    median_pressure = median(as_float(sample.get("compute_pressure"), 0.5) for sample in samples)
    median_observed = median(observed_values) if observed_values else 0.0

    def distance(sample: Mapping[str, Any]) -> Tuple[float, int]:
        value = (
            abs(as_float(sample.get("arrival_skew_us"), 0.0) - median_skew) / skew_scale
            + abs(as_float(sample.get("gap_us"), 0.0) - average_gap_us) / gap_scale
            + abs(as_float(sample.get("compute_before_us"), 0.0) - median_before) / before_scale
            + abs(as_float(sample.get("compute_overlap_us"), 0.0) - median_overlap) / overlap_scale
            + abs(as_float(sample.get("compute_pressure"), 0.5) - median_pressure) / pressure_scale
        )
        if observed_values:
            value += abs(as_float(sample.get("observed_exposed_us")) - median_observed) / observed_scale
        return value, as_int(sample.get("source_index"), 0)

    return min(samples, key=distance)


def _stratified_interval_records(samples: List[JsonDict], sample_limit: int) -> List[JsonDict]:
    records: List[JsonDict] = []
    total = len(samples)
    for bucket in range(sample_limit):
        start = bucket * total // sample_limit
        end = (bucket + 1) * total // sample_limit
        if end <= start:
            end = start + 1
        records.append(_aggregate_interval_record(samples, start, end))
    return records


def _timing_record(sample: Mapping[str, Any], source_start: int, source_end: int) -> JsonDict:
    weight = source_end - source_start + 1
    gap_us = as_float(sample.get("gap_us"), 0.0)
    record: JsonDict = {
        "gap_us": _round_us(gap_us),
        "arrival_offsets_us": [_round_us(as_float(value)) for value in sample.get("arrival_offsets_us", [])],
        "arrival_skew_us": _round_us(as_float(sample.get("arrival_skew_us"), 0.0)),
        "compute_before_us": _round_us(as_float(sample.get("compute_before_us"), 0.0)),
        "compute_overlap_us": _round_us(as_float(sample.get("compute_overlap_us"), 0.0)),
        "compute_pressure": round(as_float(sample.get("compute_pressure"), 0.5), 6),
        "source_index": source_start,
        "source_start": source_start,
        "source_end": source_end,
        "weight": weight,
        "gap_sum_us": _round_us(gap_us * weight),
    }
    uncertain_weight = _source_sample_uncertain_weight(sample) * weight
    if uncertain_weight:
        record["compute_fields_uncertain"] = True
        record["uncertain_weight"] = uncertain_weight
    if "observed_exposed_us" in sample:
        record["observed_exposed_us"] = _round_us(as_float(sample.get("observed_exposed_us")))
    return record


def _pattern_record(
    samples: List[JsonDict],
    pattern_length: int,
    *,
    start: int = 0,
    end: Optional[int] = None,
) -> JsonDict:
    end = len(samples) if end is None else end
    pattern = [_timing_record(samples[index], index, index) for index in range(start, start + pattern_length)]
    repeats = (end - start) // pattern_length
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in samples[start:end])
    record = dict(pattern[0])
    record.update(
        {
            "gap_us": _round_us(gap_sum_us / (end - start)),
            "source_index": start,
            "source_start": start,
            "source_end": end - 1,
            "weight": end - start,
            "gap_sum_us": _round_us(gap_sum_us),
            "timing_pattern": pattern,
            "pattern_repeats": repeats,
        }
    )
    uncertain_weight = sum(_timing_record_uncertain_weight(child) for child in pattern) * repeats
    if uncertain_weight:
        record["compute_fields_uncertain"] = True
        record["uncertain_weight"] = uncertain_weight
    return record


def _run_length_timing_samples(samples: List[JsonDict]) -> List[JsonDict]:
    runs: List[JsonDict] = []
    start = 0
    while start < len(samples):
        end = start + 1
        while end < len(samples) and _timing_equal(samples[start], samples[end]):
            end += 1
        runs.append(_timing_record(samples[start], start, end - 1))
        start = end
    return runs


def _short_repeated_pattern_length(samples: List[JsonDict], sample_limit: int) -> Optional[int]:
    max_pattern = min(max(1, sample_limit - 1), len(samples) // 2)
    for pattern_length in range(1, max_pattern + 1):
        if len(samples) % pattern_length:
            continue
        if all(_timing_equal(samples[index], samples[index % pattern_length]) for index in range(len(samples))):
            return pattern_length
    return None


def _prefix_pattern_records(samples: List[JsonDict], sample_limit: int) -> Optional[List[JsonDict]]:
    if sample_limit < 4 or len(samples) < 4:
        return None
    for start in range(1, min(4, len(samples) - 1)):
        remaining = len(samples) - start
        max_pattern = min(max(1, sample_limit - start - 1), remaining // 2)
        for pattern_length in range(1, max_pattern + 1):
            if remaining % pattern_length:
                continue
            if all(
                _timing_equal(samples[index], samples[start + ((index - start) % pattern_length)])
                for index in range(start, len(samples))
            ):
                records = [_timing_record(samples[index], index, index) for index in range(start)]
                records.append(_pattern_record(samples, pattern_length, start=start, end=len(samples)))
                if _recursive_timing_record_count(records) <= sample_limit:
                    return records
    return None


def _timing_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    scalar_keys = (
        "gap_us",
        "arrival_skew_us",
        "compute_before_us",
        "compute_overlap_us",
        "compute_pressure",
    )
    if any(_round_us(as_float(left.get(key), 0.0)) != _round_us(as_float(right.get(key), 0.0)) for key in scalar_keys):
        return False
    left_offsets = tuple(_round_us(as_float(value)) for value in left.get("arrival_offsets_us", []))
    right_offsets = tuple(_round_us(as_float(value)) for value in right.get("arrival_offsets_us", []))
    if left_offsets != right_offsets:
        return False
    if bool(left.get("compute_fields_uncertain")) != bool(right.get("compute_fields_uncertain")):
        return False
    left_observed = "observed_exposed_us" in left
    right_observed = "observed_exposed_us" in right
    if left_observed != right_observed:
        return False
    if left_observed:
        return _round_us(as_float(left.get("observed_exposed_us"))) == _round_us(
            as_float(right.get("observed_exposed_us"))
        )
    return True


def _recursive_timing_record_count(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    total = 0
    for record in records:
        if isinstance(record, Mapping):
            total += 1 + _recursive_timing_record_count(record.get("timing_pattern"))
    return total


def _approximate_record_count(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    total = 0
    for record in records:
        if isinstance(record, Mapping):
            total += int(record.get("approximation") == "bounded_interval")
            total += _approximate_record_count(record.get("timing_pattern"))
    return total


def _timing_record_gap_sum(record: Mapping[str, Any]) -> float:
    if "gap_sum_us" in record:
        return as_float(record.get("gap_sum_us"))
    return as_float(record.get("gap_us"), 0.0) * as_int(record.get("weight"), 1)


def _timing_records_gap_sum(records: Any) -> float:
    if not isinstance(records, list):
        return 0.0
    return sum(_timing_record_gap_sum(record) for record in records if isinstance(record, Mapping))


def _timing_records_uncertain_weight(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    return sum(
        _timing_record_logical_uncertain_weight(record)
        for record in records
        if isinstance(record, Mapping)
    )


def _summarize_fidelity(
    events: Sequence[Mapping[str, Any]],
    *,
    source_gap_total: float,
    encoded_gap_total: float,
    total_gap_error: float,
) -> JsonDict:
    maxima = {field: 0.0 for field in _FIDELITY_FIELDS}
    approximate = 0
    for event in events:
        for record in _walk_timing_records(event.get("timing_samples")):
            if record.get("approximation") == "bounded_interval":
                approximate += 1
            for field in _FIDELITY_FIELDS:
                if field in record:
                    maxima[field] = max(maxima[field], as_float(record.get(field)))
    return {
        "mode": "bounded_approximate" if approximate else "lossless_timing",
        "approximate_timing_records": approximate,
        **{key: _round_us(value) for key, value in maxima.items()},
        "source_gap_total_us": _round_us(source_gap_total),
        "encoded_gap_total_us": _round_us(encoded_gap_total),
        "total_gap_error_us": _round_us(total_gap_error),
    }


def _walk_timing_records(records: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, Mapping):
            continue
        yield record
        yield from _walk_timing_records(record.get("timing_pattern"))


def _source_sample_uncertain_weight(sample: Mapping[str, Any]) -> int:
    return 1 if sample.get("compute_fields_uncertain") is True else 0


def _timing_record_uncertain_weight(record: Mapping[str, Any]) -> int:
    if "uncertain_weight" in record:
        return as_int(record.get("uncertain_weight"))
    if record.get("compute_fields_uncertain") is True:
        return as_int(record.get("weight"), 1)
    return 0


def _timing_record_logical_uncertain_weight(record: Mapping[str, Any]) -> int:
    if "uncertain_weight" in record:
        return as_int(record.get("uncertain_weight"))
    pattern = record.get("timing_pattern")
    if isinstance(pattern, list) and pattern:
        repeats = as_int(record.get("pattern_repeats"), 1)
        return sum(
            _timing_record_logical_uncertain_weight(child)
            for child in pattern
            if isinstance(child, Mapping)
        ) * repeats
    return _timing_record_uncertain_weight(record)


def _enforce_fidelity_budgets(fidelity: Mapping[str, Any], budgets: Mapping[str, Optional[float]]) -> None:
    for field, budget in budgets.items():
        if budget is None:
            continue
        actual = as_float(fidelity.get(field), 0.0)
        if actual > budget + _US_TOLERANCE:
            raise SchemaError(f"timing fidelity {field}={actual} us exceeds budget {budget} us")


def _optional_non_negative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed


def _round_us(value: float) -> float:
    return round(as_float(value), 9)


def _source_segment_sha256(samples: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for sample in samples:
        normalized.append(
            {
                "gap_us": _round_us(as_float(sample.get("gap_us"), 0.0)),
                "arrival_offsets_us": [
                    _round_us(as_float(value)) for value in sample.get("arrival_offsets_us", [])
                ],
                "arrival_skew_us": _round_us(as_float(sample.get("arrival_skew_us"), 0.0)),
                "compute_before_us": _round_us(as_float(sample.get("compute_before_us"), 0.0)),
                "compute_overlap_us": _round_us(as_float(sample.get("compute_overlap_us"), 0.0)),
                "compute_pressure": round(as_float(sample.get("compute_pressure"), 0.5), 6),
                **(
                    {"observed_exposed_us": _round_us(as_float(sample.get("observed_exposed_us")))}
                    if "observed_exposed_us" in sample
                    else {}
                ),
                **(
                    {"compute_fields_uncertain": True}
                    if sample.get("compute_fields_uncertain") is True
                    else {}
                ),
            }
        )
    return hashlib.sha256(_canonical_json_bytes({"samples": normalized})).hexdigest()


def _canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"cannot canonicalize JSON: {exc}") from exc


def _json_size(data: Mapping[str, Any]) -> int:
    return len(_canonical_json_bytes(data))


def _update_size_metrics(canary: JsonDict) -> None:
    compiler = canary["compiler"]
    last_size = -1
    for _ in range(12):
        compiler["canary_bytes"] = max(0, last_size)
        source_bytes = as_int(compiler.get("source_bytes"), 0)
        compiler["byte_compression_ratio"] = (
            round(source_bytes / last_size, 3) if source_bytes and last_size > 0 else 0.0
        )
        current_size = _json_size(canary)
        if current_size == last_size:
            break
        last_size = current_size
    compiler["canary_bytes"] = _json_size(canary)
    compiler["byte_compression_ratio"] = (
        round(as_int(compiler.get("source_bytes")) / compiler["canary_bytes"], 3)
        if compiler["canary_bytes"]
        else 0.0
    )
