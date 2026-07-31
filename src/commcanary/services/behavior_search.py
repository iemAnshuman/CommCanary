"""Verification-driven behavior-search orchestration."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Mapping, MutableMapping, NamedTuple, Optional, Sequence, Tuple

from ..artifacts.canary import validate_canary
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_int, validate_sha256
from ..behavior_config import BehaviorConfiguration, parse_behavior_configurations, preflight_behavior_ranking_work
from ..compilation import DEFAULT_TIMING_SAMPLE_LIMIT, compile_trace_core
from ..compilation.metrics import compiler_timing_group_limits, refresh_canary_hashes_and_size
from ..errors import SchemaError
from ..formats import BEHAVIOR_SEARCH_EVIDENCE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping
from ..verification.behavior import verify_canary_behavior


class BehaviorSearchSizeKey(NamedTuple):
    """Declared total ordering for uniform and per-group behavior search."""

    canary_bytes: int
    stored_timing_records: int
    stored_events: int
    timing_limit_sum: int


_SEARCH_METHOD = "exhaustive_timing_sample_limit_search_with_greedy_per_group_refinement"
_SEARCH_OBJECTIVE = (
    "smallest verified candidate found in the declared search space by canonical candidate bytes before the "
    "fixed search summary"
)
_SEARCH_SELECTION_METRIC = "candidate_bytes_then_stored_timing_records_then_stored_events_then_timing_limit_sum"


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
    evidence_output: Optional[MutableMapping[str, Any]] = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Search for a compact canary that preserves declared model behavior.

    This is deliberately verification-driven rather than field-budget-driven:
    every candidate in the requested timing-sample range is compiled, replayed
    against the lossless source canary, and rejected unless source fidelity,
    deterministic-model behavior and pairwise configuration ranking all pass. The
    chosen candidate minimises canonical bytes before the fixed search summary,
    then stored timing records, stored events, and timing budgets. The final
    artifact may be larger after its compact summary is added. If
    ``evidence_output`` is supplied, it must be empty and receives the detached
    candidate/refinement ledger whose digest is bound by the canary.
    """

    validate_trace(trace, require_known_overlap=True, limits=limits)
    if not isinstance(require_lossless_timing, bool):
        raise SchemaError("require_lossless_timing must be a boolean")
    if not isinstance(allow_empty, bool):
        raise SchemaError("allow_empty must be a boolean")
    if not isinstance(enable_sequence_motifs, bool):
        raise SchemaError("enable_sequence_motifs must be a boolean")
    if evidence_output is not None and len(evidence_output) != 0:
        raise SchemaError("behavior search evidence_output must be empty")
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
                "model_behavior_preservation_status": "failed",
                "configuration_ranking_status": "failed",
            }
            status = "failed"
            row = _behavior_search_row(limit, candidate, verification)
            row["status"] = "verification_failed"
            row["reason"] = str(exc)
        rows.append(row)

        if status != "model_behavior_preserved":
            continue
        key = _behavior_search_size_key(candidate)
        if best is None or key < best[0]:
            best = (key, candidate, verification)

    if best is None:
        raise SchemaError("no model-behavior-preserving canary found in the requested timing sample limit range")

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
    accepted = [row for row in rows if row.get("status") == "model_behavior_preserved"]
    selected_candidate_bytes = as_int(compiler.get("canary_bytes"))
    selected_candidate = {
        "execution_semantic_sha256": str(compiler.get("execution_semantic_sha256")),
        "timing_sample_limit": as_int(compiler.get("timing_sample_limit")),
        "timing_sample_limit_mode": str(compiler.get("timing_sample_limit_mode", "uniform")),
        "timing_sample_limits_by_group": dict(compiler.get("timing_sample_limits_by_group", {})),
        "candidate_bytes_before_search_summary": selected_candidate_bytes,
        "stored_timing_records": as_int(compiler.get("stored_recursive_timing_records")),
        "stored_events": as_int(compiler.get("canary_events")),
    }
    evidence: JsonDict = {
        "format": BEHAVIOR_SEARCH_EVIDENCE_FORMAT,
        "method": _SEARCH_METHOD,
        "objective": _SEARCH_OBJECTIVE,
        "selection_metric": _SEARCH_SELECTION_METRIC,
        "search_space": {
            "min_timing_sample_limit": min_limit,
            "max_timing_sample_limit": max_limit,
            "uniform_candidate_count": len(rows),
        },
        "selected_candidate": copy.deepcopy(selected_candidate),
        "selected_verification": copy.deepcopy(dict(verification)),
        "uniform_candidates": copy.deepcopy(rows),
        "per_group_refinement": copy.deepcopy(refinement),
    }
    evidence_bytes = canonical_json_bytes(evidence)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    compiler["model_behavior_verification_status"] = verification["status"]
    compiler["configuration_ranking_status"] = verification["configuration_ranking_status"]
    compiler["model_behavior_preservation_status"] = verification["model_behavior_preservation_status"]
    compiler["behavior_search"] = {
        "method": _SEARCH_METHOD,
        "objective": _SEARCH_OBJECTIVE,
        "selection_metric": _SEARCH_SELECTION_METRIC,
        "search_space": {
            "min_timing_sample_limit": min_limit,
            "max_timing_sample_limit": max_limit,
            "uniform_candidate_count": len(rows),
        },
        "accepted_candidates": len(accepted),
        "selected_candidate": selected_candidate,
        "verification_summary": {
            "status": verification["status"],
            "source_verified_status": verification["source_verified_status"],
            "model_behavior_preservation_status": verification["model_behavior_preservation_status"],
            "configuration_ranking_status": verification["configuration_ranking_status"],
        },
        "evidence": {
            "format": BEHAVIOR_SEARCH_EVIDENCE_FORMAT,
            "sha256": evidence_sha256,
            "canonical_bytes": len(evidence_bytes),
        },
    }
    refresh_canary_hashes_and_size(selected, limits=limits)
    validate_canary(selected, limits=limits)
    if evidence_output is not None:
        evidence_output.update(copy.deepcopy(evidence))
    return selected


def validate_behavior_search_evidence(
    evidence: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Verify the detached search ledger against its selected canary."""

    try:
        validate_json_mapping(evidence, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"behavior search evidence violates JSON resource constraints: {exc}") from exc
    if evidence.get("format") != BEHAVIOR_SEARCH_EVIDENCE_FORMAT:
        raise SchemaError("behavior search evidence format is unsupported")
    validate_canary(canary, limits=limits)
    compiler = canary.get("compiler")
    if not isinstance(compiler, Mapping):
        raise SchemaError("behavior search canary must contain compiler metadata")
    summary = compiler.get("behavior_search")
    if not isinstance(summary, Mapping):
        raise SchemaError("canary does not bind behavior search evidence")
    evidence_identity = summary.get("evidence")
    if not isinstance(evidence_identity, Mapping):
        raise SchemaError("canary behavior search summary is missing evidence identity")
    validate_sha256(evidence_identity.get("sha256"), "canary behavior search evidence sha256")
    encoded = canonical_json_bytes(evidence)
    if hashlib.sha256(encoded).hexdigest() != evidence_identity.get("sha256"):
        raise SchemaError("behavior search evidence sha256 does not match the canary")
    if len(encoded) != as_int(evidence_identity.get("canonical_bytes"), -1):
        raise SchemaError("behavior search evidence byte size does not match the canary")
    for key in ("method", "objective", "selection_metric", "search_space"):
        if evidence.get(key) != summary.get(key):
            raise SchemaError(f"behavior search evidence {key} does not match the canary summary")
    selected = evidence.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise SchemaError("behavior search evidence selected_candidate must be an object")
    if selected != summary.get("selected_candidate"):
        raise SchemaError("behavior search evidence selected candidate does not match the canary summary")
    if selected.get("execution_semantic_sha256") != compiler.get("execution_semantic_sha256"):
        raise SchemaError("behavior search evidence selects different executable semantics")
    rows = evidence.get("uniform_candidates")
    search_space = evidence.get("search_space")
    if not isinstance(rows, list) or not isinstance(search_space, Mapping):
        raise SchemaError("behavior search evidence candidate ledger is invalid")
    if len(rows) != as_int(search_space.get("uniform_candidate_count"), -1):
        raise SchemaError("behavior search evidence candidate count is inconsistent")
    verification = evidence.get("selected_verification")
    verification_summary = summary.get("verification_summary")
    if not isinstance(verification, Mapping) or not isinstance(verification_summary, Mapping):
        raise SchemaError("behavior search evidence selected verification is invalid")
    for key in (
        "status",
        "source_verified_status",
        "model_behavior_preservation_status",
        "configuration_ranking_status",
    ):
        if verification.get(key) != verification_summary.get(key):
            raise SchemaError(f"behavior search evidence verification {key} does not match the canary summary")


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
                    "model_behavior_preservation_status": "failed",
                    "configuration_ranking_status": "failed",
                }
                row = {
                    "group_id": group_id,
                    "timing_sample_limit": candidate_limit,
                    "status": "failed",
                    "source_verified_status": "failed",
                    "model_behavior_preservation_status": "failed",
                    "configuration_ranking_status": "failed",
                    "reason": str(exc),
                }
                rows.append(row)
                continue
            rows.append(row)
            if candidate_verification.get("status") != "model_behavior_preserved":
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
        "model_behavior_preservation_status": str(verification.get("model_behavior_preservation_status", "unknown")),
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
    "validate_behavior_search_evidence",
]
