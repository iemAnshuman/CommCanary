"""Verification-driven behavior-search orchestration."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from ..artifacts.canary import validate_canary
from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_int
from ..behavior_config import BehaviorConfiguration, parse_behavior_configurations, preflight_behavior_ranking_work
from ..compilation import DEFAULT_TIMING_SAMPLE_LIMIT, compile_trace_core
from ..compilation.metrics import compiler_timing_group_limits, refresh_canary_hashes_and_size
from ..errors import SchemaError
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..verification.behavior import verify_canary_behavior


class BehaviorSearchSizeKey(NamedTuple):
    """Declared total ordering for uniform and per-group behavior search."""

    canary_bytes: int
    stored_timing_records: int
    stored_events: int
    timing_limit_sum: int


def synthesize_behavioral_canary(
    trace: Mapping[str, Any],
    *,
    max_events: Optional[int] = None,
    min_timing_sample_limit: int = 2,
    max_timing_sample_limit: int = DEFAULT_TIMING_SAMPLE_LIMIT,
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
    behavior_configurations: Optional[Sequence[Mapping[str, Any]]] = None,
    relative_tolerance_pct: float = 10.0,
    absolute_tolerance_us: float = 1.0,
    hidden_tolerance_points: float = 5.0,
    tail_recall_threshold: float = 0.80,
    ranking_tie_tolerance_us: float = 0.001,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Search for the smallest behaviorally and ranking-verified canary.

    This is deliberately verification-driven rather than field-budget-driven:
    every candidate in the requested timing-sample range is compiled, replayed
    against the lossless source canary, and rejected unless source fidelity,
    behavioral fidelity, and pairwise configuration ranking all pass. The
    chosen artifact minimises serialized canary bytes, then stored event count,
    then timing sample limit. It is a research-mode compiler path; it trades
    speed for a fail-closed behavioral claim.
    """

    validate_trace(trace, limits=limits)
    if not isinstance(require_lossless_timing, bool):
        raise SchemaError("require_lossless_timing must be a boolean")
    if not isinstance(allow_empty, bool):
        raise SchemaError("allow_empty must be a boolean")
    if not isinstance(enable_sequence_motifs, bool):
        raise SchemaError("enable_sequence_motifs must be a boolean")
    min_limit = as_int(min_timing_sample_limit)
    max_limit = as_int(max_timing_sample_limit)
    if min_limit < 2:
        raise SchemaError("min_timing_sample_limit must be at least 2")
    if max_limit < min_limit:
        raise SchemaError("max_timing_sample_limit must be at least min_timing_sample_limit")
    candidate_count = max_limit - min_limit + 1
    if candidate_count > limits.max_behavior_candidates:
        raise SchemaError(
            f"behavior search would evaluate {candidate_count} candidates, "
            f"above resource policy limit={limits.max_behavior_candidates}"
        )
    if candidate_count > limits.max_retained_ledger_rows:
        raise SchemaError(
            f"behavior search would retain {candidate_count} candidate rows, "
            f"above resource policy limit={limits.max_retained_ledger_rows}"
        )
    normalized_behavior_configurations = parse_behavior_configurations(
        behavior_configurations,
        max_configurations=limits.max_behavior_configurations,
    )
    preflight_behavior_ranking_work(
        normalized_behavior_configurations,
        candidate_count=candidate_count,
        limits=limits,
    )

    rows: List[JsonDict] = []
    verification: Mapping[str, Any]
    best: Optional[Tuple[BehaviorSearchSizeKey, JsonDict, Mapping[str, Any]]] = None
    for limit in range(min_limit, max_limit + 1):
        try:
            candidate = compile_trace_core(
                trace,
                max_events=max_events,
                timing_sample_limit=limit,
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
        except SchemaError as exc:
            rows.append(
                {
                    "timing_sample_limit": limit,
                    "status": "compile_failed",
                    "reason": str(exc),
                }
            )
            continue

        try:
            verification = verify_canary_behavior(
                trace,
                candidate,
                configurations=normalized_behavior_configurations,
                relative_tolerance_pct=relative_tolerance_pct,
                absolute_tolerance_us=absolute_tolerance_us,
                hidden_tolerance_points=hidden_tolerance_points,
                tail_recall_threshold=tail_recall_threshold,
                ranking_tie_tolerance_us=ranking_tie_tolerance_us,
                limits=limits,
            )
            status = str(verification.get("status"))
            row = _behavior_search_row(limit, candidate, verification)
        except SchemaError as exc:
            verification = {
                "status": "failed",
                "source_verified_status": "failed",
                "behavioral_fidelity_status": "failed",
                "configuration_ranking_status": "failed",
            }
            status = "failed"
            row = _behavior_search_row(limit, candidate, verification)
            row["status"] = "verification_failed"
            row["reason"] = str(exc)
        rows.append(row)

        if status != "behaviorally_verified":
            continue
        key = _behavior_search_size_key(candidate)
        if best is None or key < best[0]:
            best = (key, candidate, verification)

    if best is None:
        raise SchemaError("no behaviorally verified canary found in the requested timing sample limit range")

    _key, selected, verification = best
    selected = copy.deepcopy(selected)
    selected, verification, refinement = _refine_behavior_search_timing_groups(
        trace,
        selected,
        verification,
        min_timing_sample_limit=min_limit,
        max_events=max_events,
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
        behavior_configurations=normalized_behavior_configurations,
        relative_tolerance_pct=relative_tolerance_pct,
        absolute_tolerance_us=absolute_tolerance_us,
        hidden_tolerance_points=hidden_tolerance_points,
        tail_recall_threshold=tail_recall_threshold,
        ranking_tie_tolerance_us=ranking_tie_tolerance_us,
        limits=limits,
        candidate_budget=limits.max_behavior_candidates - len(rows),
        ledger_budget=limits.max_retained_ledger_rows - len(rows),
    )
    compiler = selected["compiler"]
    accepted = [row for row in rows if row.get("status") == "behaviorally_verified"]
    compiler["behavior_verification_status"] = verification["status"]
    compiler["configuration_ranking_status"] = verification["configuration_ranking_status"]
    compiler["behavioral_fidelity_status"] = verification["behavioral_fidelity_status"]
    compiler["behavior_search"] = {
        "mode": "exhaustive_timing_sample_limit_search_with_per_group_refinement",
        "objective": "minimize serialized canary bytes subject to source, behavioral, and ranking verification",
        "selection_metric": "canary_bytes_then_stored_timing_records_then_stored_events_then_timing_limits",
        "min_timing_sample_limit": min_limit,
        "max_timing_sample_limit": max_limit,
        "candidate_count": len(rows),
        "accepted_candidates": len(accepted),
        "selected_timing_sample_limit": as_int(compiler.get("timing_sample_limit")),
        "selected_timing_sample_limit_mode": str(compiler.get("timing_sample_limit_mode", "uniform")),
        "selected_timing_sample_limits_by_group": dict(compiler.get("timing_sample_limits_by_group", {})),
        "selected_canary_bytes_without_search_metadata": as_int(compiler.get("canary_bytes")),
        "selected_canary_events": as_int(compiler.get("canary_events")),
        "ranking_status": verification["configuration_ranking_status"],
        "behavioral_status": verification["behavioral_fidelity_status"],
        "source_verified_status": verification["source_verified_status"],
        "per_group_refinement": refinement,
        "candidates": rows,
    }
    refresh_canary_hashes_and_size(selected, limits=limits)
    validate_canary(selected, limits=limits)
    return selected


def _refine_behavior_search_timing_groups(
    trace: Mapping[str, Any],
    selected: JsonDict,
    verification: Mapping[str, Any],
    *,
    min_timing_sample_limit: int,
    max_events: Optional[int],
    max_gap_error_us: Optional[float],
    max_skew_error_us: Optional[float],
    max_arrival_offset_error_us: Optional[float],
    max_compute_before_error_us: Optional[float],
    max_overlap_error_us: Optional[float],
    max_pressure_error: Optional[float],
    max_observed_exposed_error_us: Optional[float],
    max_prefix_gap_error_us: Optional[float],
    require_lossless_timing: bool,
    allow_empty: bool,
    enable_sequence_motifs: bool,
    behavior_configurations: Sequence[BehaviorConfiguration],
    relative_tolerance_pct: float,
    absolute_tolerance_us: float,
    hidden_tolerance_points: float,
    tail_recall_threshold: float,
    ranking_tie_tolerance_us: float,
    limits: ResourceLimits,
    candidate_budget: int,
    ledger_budget: int,
) -> Tuple[JsonDict, Mapping[str, Any], JsonDict]:
    """Greedily lower timing budgets for individual signature groups.

    The global budget search is a coarse approximation: quiet groups can often
    be represented with fewer timing records than tail- or ranking-sensitive
    groups. This refinement accepts a lower per-group budget only when the
    resulting canary remains source-, behavior-, and ranking-verified and does
    not worsen the selected size objective.
    """

    selected_limit = as_int(selected.get("compiler", {}).get("timing_sample_limit"))
    min_limit = as_int(min_timing_sample_limit)
    group_count = as_int(selected.get("compiler", {}).get("timing_group_count"), 0)
    if group_count <= 0 or min_limit >= selected_limit:
        return (
            selected,
            verification,
            {
                "mode": "greedy_per_group_timing_sample_limit_refinement",
                "status": "skipped",
                "reason": "no lower per-group limits are available",
                "group_count": group_count,
                "attempted_candidates": 0,
                "accepted_candidates": 0,
                "selected_limits_by_group": {},
                "candidates": [],
            },
        )

    current = copy.deepcopy(selected)
    current_verification: Mapping[str, Any] = dict(verification)
    current_limits: Dict[int, int] = compiler_timing_group_limits(current.get("compiler", {}), selected_limit)
    planned_candidates = sum(
        max(0, current_limits.get(group_id, selected_limit) - min_limit) for group_id in range(group_count)
    )
    if planned_candidates > candidate_budget:
        raise SchemaError(
            f"behavior refinement would evaluate {planned_candidates} candidates, "
            f"above remaining resource policy budget={candidate_budget}"
        )
    if planned_candidates > ledger_budget:
        raise SchemaError(
            f"behavior refinement would retain {planned_candidates} candidate rows, "
            f"above remaining resource policy budget={ledger_budget}"
        )
    preflight_behavior_ranking_work(
        behavior_configurations,
        candidate_count=planned_candidates,
        limits=limits,
    )
    current_key = _behavior_search_size_key(current)
    rows: List[JsonDict] = []
    accepted_count = 0

    for group_id in range(group_count):
        current_group_limit = current_limits.get(group_id, selected_limit)
        if current_group_limit <= min_limit:
            continue
        group_best: Optional[Tuple[BehaviorSearchSizeKey, JsonDict, Mapping[str, Any], Dict[int, int]]] = None
        for candidate_limit in range(min_limit, current_group_limit):
            proposed_limits = dict(current_limits)
            proposed_limits[group_id] = candidate_limit
            try:
                candidate = compile_trace_core(
                    trace,
                    max_events=max_events,
                    timing_sample_limit=selected_limit,
                    timing_sample_limits_by_group=proposed_limits,
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
                candidate_verification = verify_canary_behavior(
                    trace,
                    candidate,
                    configurations=behavior_configurations,
                    relative_tolerance_pct=relative_tolerance_pct,
                    absolute_tolerance_us=absolute_tolerance_us,
                    hidden_tolerance_points=hidden_tolerance_points,
                    tail_recall_threshold=tail_recall_threshold,
                    ranking_tie_tolerance_us=ranking_tie_tolerance_us,
                    limits=limits,
                )
                row = _behavior_search_refinement_row(group_id, candidate_limit, candidate, candidate_verification)
            except SchemaError as exc:
                candidate_verification = {
                    "status": "failed",
                    "source_verified_status": "failed",
                    "behavioral_fidelity_status": "failed",
                    "configuration_ranking_status": "failed",
                }
                row = {
                    "group_id": group_id,
                    "timing_sample_limit": candidate_limit,
                    "status": "failed",
                    "source_verified_status": "failed",
                    "behavioral_fidelity_status": "failed",
                    "configuration_ranking_status": "failed",
                    "reason": str(exc),
                }
                rows.append(row)
                continue
            rows.append(row)
            if candidate_verification.get("status") != "behaviorally_verified":
                continue
            candidate_key = _behavior_search_size_key(candidate)
            if candidate_key >= current_key:
                continue
            if group_best is None or candidate_key < group_best[0]:
                group_best = (candidate_key, candidate, candidate_verification, proposed_limits)
        if group_best is None:
            continue
        current_key, current, current_verification, current_limits = group_best
        accepted_count += 1

    return (
        current,
        current_verification,
        {
            "mode": "greedy_per_group_timing_sample_limit_refinement",
            "status": "refined" if accepted_count else "no_smaller_verified_candidate",
            "group_count": group_count,
            "attempted_candidates": len(rows),
            "accepted_candidates": accepted_count,
            "selected_limits_by_group": {str(group_id): limit for group_id, limit in sorted(current_limits.items())},
            "selected_size_key": list(current_key),
            "candidates": rows,
        },
    )


def _behavior_search_size_key(candidate: Mapping[str, Any]) -> BehaviorSearchSizeKey:
    compiler = candidate.get("compiler", {})
    group_limits = compiler.get("timing_sample_limits_by_group", {})
    default_limit = as_int(compiler.get("timing_sample_limit"), 0)
    group_count = as_int(compiler.get("timing_group_count"), 0)
    limit_sum = default_limit * group_count
    if isinstance(group_limits, Mapping):
        limit_sum = sum(
            as_int(
                group_limits.get(
                    str(group_id),
                    group_limits.get(group_id, default_limit),
                )
            )
            for group_id in range(group_count)
        )
    return BehaviorSearchSizeKey(
        as_int(compiler.get("canary_bytes"), 0),
        as_int(compiler.get("stored_recursive_timing_records"), 0),
        as_int(compiler.get("canary_events"), 0),
        limit_sum,
    )


def _behavior_search_refinement_row(
    group_id: int,
    timing_sample_limit: int,
    candidate: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> JsonDict:
    row = _behavior_search_row(timing_sample_limit, candidate, verification)
    row["group_id"] = group_id
    row["timing_sample_limit_mode"] = str(candidate.get("compiler", {}).get("timing_sample_limit_mode", "uniform"))
    row["timing_sample_limits_by_group"] = dict(candidate.get("compiler", {}).get("timing_sample_limits_by_group", {}))
    return row


def _behavior_search_row(
    timing_sample_limit: int,
    candidate: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> JsonDict:
    compiler = candidate.get("compiler", {})
    return {
        "timing_sample_limit": timing_sample_limit,
        "status": str(verification.get("status")),
        "source_verified_status": str(verification.get("source_verified_status", "unknown")),
        "behavioral_fidelity_status": str(verification.get("behavioral_fidelity_status", "unknown")),
        "configuration_ranking_status": str(verification.get("configuration_ranking_status", "unknown")),
        "canary_bytes": as_int(compiler.get("canary_bytes"), 0),
        "canary_events": as_int(compiler.get("canary_events"), 0),
        "sequence_motif_count": as_int(compiler.get("sequence_motif_count"), 0),
        "approximate_timing_records": as_int(compiler.get("approximate_timing_records"), 0),
        "recursive_timing_records": as_int(compiler.get("recursive_timing_records"), 0),
        "stored_recursive_timing_records": as_int(compiler.get("stored_recursive_timing_records"), 0),
    }


behavior_search_size_key = _behavior_search_size_key

__all__ = [
    "BehaviorSearchSizeKey",
    "behavior_search_size_key",
    "synthesize_behavioral_canary",
]
