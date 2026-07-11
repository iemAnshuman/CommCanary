"""Decision-preserving ddmin reduction application service."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_float, as_int
from ..behavior_config import behavior_replay_arguments, parse_behavior_configurations, preflight_behavior_ranking_work
from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..replay.core import replay_canary
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ._ranking import RANKING_METRICS, ranking_relation
from .compile import compile_trace


def ddmin_ranking_reduction(
    trace: Mapping[str, Any],
    *,
    configurations: Optional[Sequence[Mapping[str, Any]]] = None,
    ranking_tie_tolerance_us: float = 0.001,
    timing_sample_limit: Optional[int] = None,
    max_oracle_calls: int = 256,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Reduce a trace to a small event subset that preserves config rankings.

    Classic ddmin over the ordered event list. A candidate subset is accepted
    only when compiling and replaying it across the configuration set yields
    exactly the same pairwise ranking relations as the full trace, for every
    ranking metric. Uncompilable subsets are rejected rather than repaired.
    The output trace is a decision-preserving subset; it is intentionally not
    source-verified and cannot receive a strong behavioral claim.
    """

    validate_trace(trace, limits=limits)
    events = list(trace.get("events", []))
    if not events:
        raise SchemaError("cannot reduce an empty trace")
    tie_tolerance = as_float(ranking_tie_tolerance_us)
    if tie_tolerance < 0.0:
        raise SchemaError("ranking_tie_tolerance_us must be non-negative")
    call_budget = as_int(max_oracle_calls)
    if call_budget <= 0:
        raise SchemaError("max_oracle_calls must be positive")
    if call_budget > limits.max_reduction_oracle_calls:
        raise SchemaError(f"max_oracle_calls cannot exceed resource policy limit={limits.max_reduction_oracle_calls}")
    sample_limit = None if timing_sample_limit is None else as_int(timing_sample_limit)
    if sample_limit is not None and sample_limit < 2:
        raise SchemaError("timing_sample_limit must be at least 2")
    configs = parse_behavior_configurations(
        configurations,
        max_configurations=limits.max_behavior_configurations,
    )
    preflight_behavior_ranking_work(configs, candidate_count=call_budget + 1, limits=limits)

    def replay_relations(subset: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], str]:
        candidate = {
            "format": TRACE_FORMAT,
            "workload": dict(trace.get("workload", {})),
            "system": dict(trace.get("system", {})),
            "events": list(subset),
        }
        if sample_limit is None:
            canary = compile_trace(
                candidate,
                timing_sample_limit=max(2, len(subset)),
                require_lossless_timing=True,
                limits=limits,
            )
        else:
            canary = compile_trace(
                candidate,
                timing_sample_limit=sample_limit,
                limits=limits,
            )
        metrics_by_name: Dict[str, Mapping[str, Any]] = {}
        for raw_config in configs:
            name = raw_config["name"]
            replay_args = behavior_replay_arguments(raw_config)
            report = replay_canary(
                canary,
                backend_label=name,
                limits=limits,
                **replay_args,
            )
            metrics_by_name[name] = report["metrics"]
        names = sorted(metrics_by_name)
        relations: Dict[Tuple[str, str, str], str] = {}
        for metric in RANKING_METRICS:
            for left_index in range(len(names)):
                for right_index in range(left_index + 1, len(names)):
                    left = names[left_index]
                    right = names[right_index]
                    relations[(metric, left, right)] = ranking_relation(
                        as_float(metrics_by_name[left].get(metric)),
                        as_float(metrics_by_name[right].get(metric)),
                        tie_tolerance,
                    )
        return relations

    reference_relations = replay_relations(events)

    oracle_calls = 0
    budget_exhausted = False

    def oracle(subset: Sequence[Mapping[str, Any]]) -> bool:
        nonlocal oracle_calls, budget_exhausted
        if budget_exhausted or oracle_calls >= call_budget:
            budget_exhausted = True
            return False
        oracle_calls += 1
        try:
            return replay_relations(subset) == reference_relations
        except SchemaError:
            return False

    current = list(events)
    granularity = 2
    while len(current) >= 2:
        chunk = len(current) / granularity
        subsets = [current[int(index * chunk) : int((index + 1) * chunk)] for index in range(granularity)]
        subsets = [subset for subset in subsets if subset]
        reduced = False
        for subset in subsets:
            if len(subset) < len(current) and oracle(subset):
                current = subset
                granularity = 2
                reduced = True
                break
        if not reduced and granularity > 2:
            for index in range(len(subsets)):
                complement = [
                    event for subset_index, subset in enumerate(subsets) if subset_index != index for event in subset
                ]
                if complement and len(complement) < len(current) and oracle(complement):
                    current = complement
                    granularity = max(granularity - 1, 2)
                    reduced = True
                    break
        if not reduced:
            if granularity >= len(current) or budget_exhausted:
                break
            granularity = min(len(current), granularity * 2)

    workload = copy.deepcopy(dict(trace.get("workload", {})))
    notes = str(workload.get("notes", ""))
    suffix = (
        "CommCanary research reduction: ddmin_ranking. Decision-preserving "
        "event subset; not source-verified against the original trace."
    )
    workload["notes"] = f"{notes} {suffix}".strip()
    workload["reduction_method"] = "ddmin_ranking"
    workload["reduction"] = {
        "method": "ddmin_ranking",
        "original_events": len(events),
        "reduced_events": len(current),
        "oracle_calls": oracle_calls,
        "oracle_call_budget": call_budget,
        "budget_exhausted": budget_exhausted,
        "ranking_tie_tolerance_us": tie_tolerance,
        "ranking_metrics": list(RANKING_METRICS),
        "configurations": [config["name"] for config in configs],
        "timing_sample_limit": sample_limit,
    }
    reduced_trace: JsonDict = {
        "format": TRACE_FORMAT,
        "workload": workload,
        "system": copy.deepcopy(dict(trace.get("system", {}))),
        "events": [copy.deepcopy(event) for event in current],
    }
    validate_trace(reduced_trace, limits=limits)
    return reduced_trace


__all__ = ["ddmin_ranking_reduction"]
