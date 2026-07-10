"""ddmin-style decision-preserving trace reduction.

Research baseline required by RESEARCH_SPEC.md: generic property-preserving
minimization to compare against behavior-search compilation. The oracle
deliberately preserves only the decision — pairwise configuration rankings of
the replayed workload — not full behavioral fidelity: an event subset cannot
preserve metric counts by construction, so this reducer measures what
decision-only event reduction keeps or gives up relative to CommCanary's
stricter fail-closed behavioral verifier.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .compiler import (
    _BEHAVIORAL_RANKING_METRICS,
    _DEFAULT_BEHAVIORAL_CONFIGS,
    _behavioral_replay_args,
    _ranking_relation,
    compile_trace,
)
from .replay import replay_canary
from .schema import (
    TRACE_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    validate_trace,
)


def ddmin_ranking_reduction(
    trace: Mapping[str, Any],
    *,
    configurations: Optional[Sequence[Mapping[str, Any]]] = None,
    ranking_tie_tolerance_us: float = 0.001,
    timing_sample_limit: Optional[int] = None,
    max_oracle_calls: int = 256,
) -> JsonDict:
    """Reduce a trace to a small event subset that preserves config rankings.

    Classic ddmin over the ordered event list. A candidate subset is accepted
    only when compiling and replaying it across the configuration set yields
    exactly the same pairwise ranking relations as the full trace, for every
    ranking metric. Uncompilable subsets are rejected rather than repaired.
    The output trace is a decision-preserving subset; it is intentionally not
    source-verified and cannot receive a strong behavioral claim.
    """

    validate_trace(trace)
    events = list(trace.get("events", []))
    if not events:
        raise SchemaError("cannot reduce an empty trace")
    tie_tolerance = as_float(ranking_tie_tolerance_us)
    if tie_tolerance < 0.0:
        raise SchemaError("ranking_tie_tolerance_us must be non-negative")
    call_budget = as_int(max_oracle_calls)
    if call_budget <= 0:
        raise SchemaError("max_oracle_calls must be positive")
    sample_limit = None if timing_sample_limit is None else as_int(timing_sample_limit)
    if sample_limit is not None and sample_limit < 2:
        raise SchemaError("timing_sample_limit must be at least 2")
    configs = [dict(config) for config in (configurations or _DEFAULT_BEHAVIORAL_CONFIGS)]
    if len(configs) < 2:
        raise SchemaError("ranking reduction requires at least two configurations")
    config_names = [
        str(config.get("name", f"config-{index}")) for index, config in enumerate(configs)
    ]
    if len(set(config_names)) != len(config_names):
        raise SchemaError(
            "ranking reduction requires unique configuration names; duplicates "
            "would silently collapse the pairwise ranking oracle"
        )

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
            )
        else:
            canary = compile_trace(candidate, timing_sample_limit=sample_limit)
        metrics_by_name: Dict[str, Mapping[str, Any]] = {}
        for index, raw_config in enumerate(configs):
            config = dict(raw_config)
            name = str(config.pop("name", f"config-{index}"))
            replay_args = _behavioral_replay_args(config)
            report = replay_canary(canary, backend_label=name, **replay_args)
            metrics_by_name[name] = report["metrics"]
        names = sorted(metrics_by_name)
        relations: Dict[Tuple[str, str, str], str] = {}
        for metric in _BEHAVIORAL_RANKING_METRICS:
            for left_index in range(len(names)):
                for right_index in range(left_index + 1, len(names)):
                    left = names[left_index]
                    right = names[right_index]
                    relations[(metric, left, right)] = _ranking_relation(
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
        subsets = [
            current[int(index * chunk) : int((index + 1) * chunk)]
            for index in range(granularity)
        ]
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
                    event
                    for subset_index, subset in enumerate(subsets)
                    if subset_index != index
                    for event in subset
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

    workload = dict(trace.get("workload", {}))
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
        "ranking_metrics": list(_BEHAVIORAL_RANKING_METRICS),
        "configurations": [
            str(config.get("name", f"config-{index}")) for index, config in enumerate(configs)
        ],
        "timing_sample_limit": sample_limit,
    }
    reduced_trace: JsonDict = {
        "format": TRACE_FORMAT,
        "workload": workload,
        "system": dict(trace.get("system", {})),
        "events": [copy.deepcopy(event) for event in current],
    }
    validate_trace(reduced_trace)
    return reduced_trace
