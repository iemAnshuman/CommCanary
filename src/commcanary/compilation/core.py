"""Pure trace-to-canary compilation core.

This boundary normalizes and compresses a trace. It intentionally knows
nothing about verification or behavior-search orchestration.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from ..artifacts.canary import (
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    validate_canary,
)
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_int
from ..errors import SchemaError
from ..formats import (
    ARTIFACT_PROVENANCE_ALGORITHM,
    CANARY_FORMAT,
    CANARY_INTEGRITY_PROFILE,
    TRACE_FORMAT,
)
from ..operation_identity import CompressionKey, OperationIdentity
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ._constants import DEFAULT_TIMING_SAMPLE_LIMIT, US_TOLERANCE
from .metrics import (
    _approximate_record_count,
    _enforce_fidelity_budgets,
    _json_size,
    _normalize_timing_group_limits,
    _optional_non_negative,
    _recursive_timing_record_count,
    _round_us,
    _summarize_fidelity,
    _timing_records_gap_sum,
    _timing_records_uncertain_weight,
    _update_size_metrics,
)
from .normalization import (
    _append_sample,
    _compress_sequence_motifs,
    _event_to_step,
    _finalize_step,
    _ordered_trace_events,
)


def compile_trace_core(
    trace: Mapping[str, Any],
    *,
    max_events: Optional[int] = None,
    timing_sample_limit: int = DEFAULT_TIMING_SAMPLE_LIMIT,
    timing_sample_limits_by_group: Optional[Mapping[Any, int]] = None,
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
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Compile a communication trace into a compact, fidelity-audited canary.

    Exact run/pattern encodings are preferred. If the timing stream cannot fit
    within ``timing_sample_limit``, ordered bounded intervals are emitted with
    explicit approximation errors. Optional budgets make compilation fail
    closed rather than silently exceeding an acceptable error.
    """

    validate_trace(trace, limits=limits)
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
    timing_group_limits = _normalize_timing_group_limits(timing_sample_limits_by_group, timing_sample_limit)

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
        "max_prefix_gap_error_us": _optional_non_negative(max_prefix_gap_error_us, "max_prefix_gap_error_us"),
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
    signature_occurrences: Dict[CompressionKey, int] = {}
    for source_index, (event, gap_us) in enumerate(zip(ordered_events, ordered_gaps)):
        step = _event_to_step(
            event,
            source_index=source_index,
            gap_us=gap_us,
            sample_limit=timing_sample_limit,
        )
        signature = OperationIdentity.from_mapping(step).compression_key()
        step["_execution_occurrence_base"] = signature_occurrences.get(signature, 0)
        signature_occurrences[signature] = step["_execution_occurrence_base"] + 1
        if canary_events and canary_events[-1].get("_signature") == signature:
            _append_sample(canary_events[-1], step)
        else:
            group_index = len(canary_events)
            step["_sample_limit"] = timing_group_limits.get(group_index, timing_sample_limit)
            step["_signature"] = signature
            canary_events.append(step)

    flat_finalized = [_finalize_step(step) for step in canary_events]
    timing_group_count = len(flat_finalized)
    invalid_group_ids = [group_id for group_id in timing_group_limits if group_id >= timing_group_count]
    if invalid_group_ids:
        raise SchemaError(
            "timing_sample_limits_by_group references unknown timing groups: "
            + ", ".join(str(group_id) for group_id in sorted(invalid_group_ids))
        )
    finalized = _compress_sequence_motifs(flat_finalized) if enable_sequence_motifs else flat_finalized
    # Release the uncompressed timing streams before serialising the source
    # trace. This avoids holding two large canonical representations at once.
    canary_events.clear()

    source_count = len(ordered_events)
    compiled_count = len(finalized)
    logical_events = list(iter_canary_logical_events(finalized, limits=limits))
    stored_leaf_events = list(iter_canary_stored_leaf_events(finalized, limits=limits))
    expanded_count = sum(as_int(event.get("repeat"), 1) for event in logical_events)
    if expanded_count != source_count:
        raise SchemaError("sequence motif compression changed the logical event count")
    event_ratio = round(source_count / compiled_count, 3) if compiled_count else 0.0
    recursive_records = sum(_recursive_timing_record_count(event.get("timing_samples")) for event in logical_events)
    approximate_records = sum(_approximate_record_count(event.get("timing_samples")) for event in logical_events)
    stored_recursive_records = sum(
        _recursive_timing_record_count(event.get("timing_samples")) for event in stored_leaf_events
    )
    stored_approximate_records = sum(
        _approximate_record_count(event.get("timing_samples")) for event in stored_leaf_events
    )
    compute_uncertain_events = sum(
        _timing_records_uncertain_weight(event.get("timing_samples")) for event in logical_events
    )
    sequence_motif_count = sum(
        1 for event in finalized if isinstance(event, Mapping) and event.get("program") == "sequence_motif"
    )

    source_gap_total = _round_us(sum(ordered_gaps))
    encoded_gap_total = _round_us(
        sum(_timing_records_gap_sum(event.get("timing_samples", [])) for event in logical_events)
    )
    total_gap_error = _round_us(abs(source_gap_total - encoded_gap_total))
    if total_gap_error > US_TOLERANCE:
        raise SchemaError(f"timing compression changed total gap duration by {total_gap_error} us")

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
    source_sha = hashlib.sha256(canonical_json_bytes(measured_trace)).hexdigest()

    canary: JsonDict = {
        "format": CANARY_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_format": trace.get("format", TRACE_FORMAT),
        "workload": copy.deepcopy(dict(trace.get("workload", {}))),
        "system": copy.deepcopy(dict(trace.get("system", {}))),
        "compiler": {
            "integrity_profile": CANARY_INTEGRITY_PROFILE,
            "artifact_provenance_algorithm": ARTIFACT_PROVENANCE_ALGORITHM,
            "compression": "ordered exact patterns, replay-equivalent sequence motifs, and fidelity-audited bounded intervals",
            "timing_sample_limit": timing_sample_limit,
            "timing_sample_limit_mode": "per_group" if timing_group_limits else "uniform",
            "timing_group_count": timing_group_count,
            **(
                {
                    "timing_sample_limits_by_group": {
                        str(group_id): limit for group_id, limit in sorted(timing_group_limits.items())
                    }
                }
                if timing_group_limits
                else {}
            ),
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
    canary["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(
        canary,
        limits=limits,
    )
    canary["compiler"]["scheduler_execution_sha256"] = canary_scheduler_execution_sha256(
        canary,
        limits=limits,
    )
    canary["compiler"]["calibration_evaluation_sha256"] = canary_calibration_sha256(
        canary,
        limits=limits,
    )
    canary["compiler"]["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)
    _update_size_metrics(canary)
    validate_canary(canary, limits=limits)
    return canary


__all__ = ["compile_trace_core"]
