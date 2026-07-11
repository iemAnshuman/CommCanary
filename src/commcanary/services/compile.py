"""Verified compile orchestration and the stable compile entry point."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..artifacts.canary import canary_artifact_provenance_sha256, validate_canary
from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict
from ..compilation import DEFAULT_TIMING_SAMPLE_LIMIT, compile_trace_core, update_size_metrics
from ..errors import SchemaError
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..verification.behavior import verify_canary_behavior
from .behavior_search import synthesize_behavioral_canary


def compile_trace(
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
    require_behavior_verification: bool = False,
    behavior_search: bool = False,
    behavior_search_min_sample_limit: int = 2,
    behavior_configurations: Optional[Sequence[Mapping[str, Any]]] = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Compile a trace, optionally composing verification in this service layer."""

    if behavior_search:
        return synthesize_behavioral_canary(
            trace,
            max_events=max_events,
            min_timing_sample_limit=behavior_search_min_sample_limit,
            max_timing_sample_limit=timing_sample_limit,
            max_gap_error_us=max_gap_error_us,
            max_skew_error_us=max_skew_error_us,
            max_arrival_offset_error_us=max_arrival_offset_error_us,
            max_compute_before_error_us=max_compute_before_error_us,
            max_overlap_error_us=max_overlap_error_us,
            max_pressure_error=max_pressure_error,
            max_observed_exposed_error_us=max_observed_exposed_error_us,
            max_prefix_gap_error_us=max_prefix_gap_error_us,
            require_lossless_timing=require_lossless_timing,
            allow_empty=allow_empty,
            enable_sequence_motifs=enable_sequence_motifs,
            behavior_configurations=behavior_configurations,
            limits=limits,
        )

    validate_trace(trace, limits=limits)
    if not isinstance(require_behavior_verification, bool):
        raise SchemaError("require_behavior_verification must be a boolean")

    canary = compile_trace_core(
        trace,
        max_events=max_events,
        timing_sample_limit=timing_sample_limit,
        timing_sample_limits_by_group=timing_sample_limits_by_group,
        max_gap_error_us=max_gap_error_us,
        max_skew_error_us=max_skew_error_us,
        max_arrival_offset_error_us=max_arrival_offset_error_us,
        max_compute_before_error_us=max_compute_before_error_us,
        max_overlap_error_us=max_overlap_error_us,
        max_pressure_error=max_pressure_error,
        max_observed_exposed_error_us=max_observed_exposed_error_us,
        max_prefix_gap_error_us=max_prefix_gap_error_us,
        require_lossless_timing=require_lossless_timing,
        allow_empty=allow_empty,
        enable_sequence_motifs=enable_sequence_motifs,
        limits=limits,
    )
    if require_behavior_verification:
        behavior = verify_canary_behavior(
            trace,
            canary,
            configurations=behavior_configurations,
            limits=limits,
        )
        compiler = canary["compiler"]
        compiler["behavior_verification_status"] = behavior["status"]
        compiler["configuration_ranking_status"] = behavior["configuration_ranking_status"]
        compiler["behavioral_fidelity_status"] = behavior["behavioral_fidelity_status"]
        compiler["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)
        update_size_metrics(canary)
        validate_canary(canary, limits=limits)
        if behavior["status"] != "behaviorally_verified":
            raise SchemaError("compiled canary failed required behavior verification")
    return canary


__all__ = ["compile_trace"]
