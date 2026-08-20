"""Counterexample-guided synthesis over measured physical candidate rows."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..artifacts.chakra import ChakraExecutionTrace, encode_chakra_subgraph
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.physical_canary import (
    leakage_assessment,
    validate_chakra_projection,
    validate_physical_canary_policy,
    validate_physical_oracle_corpus,
    with_content_identity,
)
from ..errors import SchemaError
from ..formats import (
    PHYSICAL_FIDELITY_CERTIFICATE_FORMAT,
    PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT,
    PHYSICAL_SYNTHESIS_LEDGER_FORMAT,
)
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits


@dataclass(frozen=True)
class PhysicalSynthesisResult:
    """Pure synthesis result; bundle publication is a separate workflow."""

    status: str
    canary_et: bytes
    selected_region_ids: Tuple[str, ...]
    selected_node_ids: Tuple[int, ...]
    selected_candidate_id: Optional[str]
    ledger: Dict[str, Any]
    certificate: Dict[str, Any]
    leakage: Dict[str, Any]


def synthesize_physical_decision_canary(
    trace: ChakraExecutionTrace,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus: Optional[Mapping[str, Any]] = None,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> PhysicalSynthesisResult:
    """Select a dependency-closed Chakra subgraph without holdout leakage.

    Candidate measurements are supplied as an immutable corpus.  The search
    uses only training decisions for selection.  Corpus schema validation sees
    every row, but held-out decision values do not enter synthesis statistics
    until after the selected region set has been content-committed.
    """

    validate_physical_canary_policy(policy, limits=limits)
    validate_chakra_projection(projection, trace, limits=limits)
    regions = _projection_regions(projection)
    required_tags = set(policy["required_feature_tags"])

    if corpus is None:
        selected_regions = _initial_unmeasured_selection(regions, required_tags)
        canary_et, selected_nodes = _encode_selection(trace, regions, selected_regions)
        selection = _selection_record(
            trace,
            projection,
            policy,
            corpus_id=None,
            candidate_id=None,
            selected_region_ids=selected_regions,
            selected_node_ids=selected_nodes,
            canary_et=canary_et,
        )
        raw_ledger: Dict[str, Any] = {
            "format": PHYSICAL_SYNTHESIS_LEDGER_FORMAT,
            "algorithm": "counterexample-guided-physical-minimization.v1",
            "holdout_access_policy": "schema_validated_but_not_used_until_selection_frozen",
            "status": "blocked_missing_physical_oracle_corpus",
            "selection": selection,
            "training_candidate_screen": [],
            "counterexample_steps": [],
            "holdout_evaluation": None,
        }
        ledger = with_content_identity(raw_ledger, "ledger_id")
        raw_leakage = {
            "format": PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT,
            **leakage_assessment(projection, selected_regions),
        }
        leakage = with_content_identity(raw_leakage, "assessment_id")
        certificate = _certificate(
            status="blocked_missing_physical_oracle_corpus",
            selection=selection,
            evidence_kind="absent",
            application=None,
            training=None,
            holdout=None,
            baselines={},
            leakage=leakage,
        )
        return PhysicalSynthesisResult(
            status="blocked_missing_physical_oracle_corpus",
            canary_et=canary_et,
            selected_region_ids=selected_regions,
            selected_node_ids=selected_nodes,
            selected_candidate_id=None,
            ledger=ledger,
            certificate=certificate,
            leakage=leakage,
        )

    validate_physical_oracle_corpus(corpus, trace, projection, policy, limits=limits)
    perturbations = _perturbation_map(corpus)
    reduced = [evaluation for evaluation in corpus["evaluations"] if evaluation["method"] == "reduced_decision_canary"]
    training_screen = []
    for evaluation in reduced:
        stats = _candidate_statistics(evaluation, perturbations, split="training")
        training_screen.append(
            {
                "candidate_id": evaluation["candidate_id"],
                "selected_region_ids": list(evaluation["selected_region_ids"]),
                "selected_node_ids": sorted(_candidate_node_ids(evaluation, regions)),
                "required_feature_tags_covered": _covers_required_tags(
                    regions,
                    evaluation["selected_region_ids"],
                    required_tags,
                ),
                "statistics": stats,
            }
        )
    training_screen.sort(key=lambda row: str(row["candidate_id"]))
    eligible = [
        evaluation
        for evaluation in reduced
        if _covers_required_tags(regions, evaluation["selected_region_ids"], required_tags)
    ]
    if not eligible:
        raise SchemaError("no measured reduced candidate covers the policy's required feature tags")
    current = min(
        eligible,
        key=lambda evaluation: _training_cost_key(evaluation, perturbations, regions),
    )
    counterexample_steps: List[Dict[str, Any]] = []
    allowed_false_negatives = int(policy["qualification_gates"]["maximum_severe_false_negatives"])

    while True:
        counterexamples = _severe_false_negative_ids(current, perturbations, split="training")
        if len(counterexamples) <= allowed_false_negatives:
            break
        current_regions = set(current["selected_region_ids"])
        current_nodes = _candidate_node_ids(current, regions)
        improving: List[Mapping[str, Any]] = []
        for candidate in eligible:
            candidate_nodes = _candidate_node_ids(candidate, regions)
            if not current_nodes < candidate_nodes:
                continue
            candidate_counterexamples = _severe_false_negative_ids(
                candidate,
                perturbations,
                split="training",
            )
            if len(candidate_counterexamples) < len(counterexamples):
                improving.append(candidate)
        if not improving:
            break
        next_candidate = min(
            improving,
            key=lambda evaluation: (
                len(_severe_false_negative_ids(evaluation, perturbations, split="training")),
                len(_candidate_node_ids(evaluation, regions) - current_nodes),
                *_training_cost_key(evaluation, perturbations, regions),
            ),
        )
        counterexample_steps.append(
            {
                "from_candidate_id": current["candidate_id"],
                "counterexample_perturbation_ids": counterexamples,
                "to_candidate_id": next_candidate["candidate_id"],
                "added_region_ids": sorted(set(next_candidate["selected_region_ids"]).difference(current_regions)),
                "added_node_ids": sorted(_candidate_node_ids(next_candidate, regions).difference(current_nodes)),
            }
        )
        current = next_candidate

    current_stats = _candidate_statistics(current, perturbations, split="training")
    if not _statistics_pass(current_stats, policy):
        current_regions = set(current["selected_region_ids"])
        current_nodes = _candidate_node_ids(current, regions)
        passing_supersets = [
            candidate
            for candidate in eligible
            if current_nodes.issubset(_candidate_node_ids(candidate, regions))
            and _statistics_pass(
                _candidate_statistics(candidate, perturbations, split="training"),
                policy,
            )
        ]
        if passing_supersets:
            next_candidate = min(
                passing_supersets,
                key=lambda evaluation: _training_cost_key(evaluation, perturbations, regions),
            )
            if next_candidate["candidate_id"] != current["candidate_id"]:
                counterexample_steps.append(
                    {
                        "from_candidate_id": current["candidate_id"],
                        "counterexample_perturbation_ids": _policy_failure_codes(current_stats, policy),
                        "to_candidate_id": next_candidate["candidate_id"],
                        "added_region_ids": sorted(
                            set(next_candidate["selected_region_ids"]).difference(current_regions)
                        ),
                        "added_node_ids": sorted(
                            _candidate_node_ids(next_candidate, regions).difference(current_nodes)
                        ),
                    }
                )
                current = next_candidate

    # Pruning sees training rows only and can select only a measured strict
    # subset that independently passes every training gate.
    while True:
        current_regions = set(current["selected_region_ids"])
        current_nodes = _candidate_node_ids(current, regions)
        passing_subsets = [
            candidate
            for candidate in eligible
            if _candidate_node_ids(candidate, regions) < current_nodes
            and _statistics_pass(
                _candidate_statistics(candidate, perturbations, split="training"),
                policy,
            )
        ]
        if not passing_subsets:
            break
        next_candidate = min(
            passing_subsets,
            key=lambda evaluation: _training_cost_key(evaluation, perturbations, regions),
        )
        counterexample_steps.append(
            {
                "from_candidate_id": current["candidate_id"],
                "counterexample_perturbation_ids": [],
                "to_candidate_id": next_candidate["candidate_id"],
                "removed_region_ids": sorted(current_regions.difference(next_candidate["selected_region_ids"])),
                "removed_node_ids": sorted(current_nodes.difference(_candidate_node_ids(next_candidate, regions))),
            }
        )
        current = next_candidate

    final_training = _candidate_statistics(current, perturbations, split="training")
    selected_regions = tuple(current["selected_region_ids"])
    canary_et, selected_nodes = _encode_selection(trace, regions, selected_regions)
    selection = _selection_record(
        trace,
        projection,
        policy,
        corpus_id=str(corpus["corpus_id"]),
        candidate_id=str(current["candidate_id"]),
        selected_region_ids=selected_regions,
        selected_node_ids=selected_nodes,
        canary_et=canary_et,
    )

    # The selection commitment above is frozen before held-out decision values
    # enter any synthesis statistic.  Corpus validation has already checked
    # the structure and values of every row.
    holdout = _candidate_statistics(current, perturbations, split="holdout")
    baseline_stats = _baseline_statistics(corpus, perturbations)
    missing_baselines = sorted(set(policy["required_baselines"]).difference(baseline_stats))
    raw_leakage = {
        "format": PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT,
        **leakage_assessment(projection, selected_regions),
    }
    leakage = with_content_identity(raw_leakage, "assessment_id")
    status = _qualification_status(
        corpus=corpus,
        policy=policy,
        training=final_training,
        holdout=holdout,
        missing_baselines=missing_baselines,
        leakage=leakage,
    )
    raw_ledger = {
        "format": PHYSICAL_SYNTHESIS_LEDGER_FORMAT,
        "algorithm": "counterexample-guided-physical-minimization.v1",
        "holdout_access_policy": "schema_validated_but_not_used_until_selection_frozen",
        "status": status,
        "selection": selection,
        "training_candidate_screen": training_screen,
        "counterexample_steps": counterexample_steps,
        "holdout_evaluation": {
            "candidate_id": current["candidate_id"],
            "statistics": holdout,
            "read_after_selection_sha256": selection["selection_sha256"],
        },
    }
    ledger = with_content_identity(raw_ledger, "ledger_id")
    certificate = _certificate(
        status=status,
        selection=selection,
        evidence_kind=str(corpus["evidence_kind"]),
        application=corpus["application"],
        training=final_training,
        holdout=holdout,
        baselines=baseline_stats,
        leakage=leakage,
    )
    return PhysicalSynthesisResult(
        status=status,
        canary_et=canary_et,
        selected_region_ids=selected_regions,
        selected_node_ids=selected_nodes,
        selected_candidate_id=str(current["candidate_id"]),
        ledger=ledger,
        certificate=certificate,
        leakage=leakage,
    )


def _projection_regions(projection: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(region["region_id"]): region for region in projection["regions"]}


def _candidate_node_ids(
    evaluation: Mapping[str, Any],
    regions: Mapping[str, Mapping[str, Any]],
) -> Set[int]:
    return {
        int(node_id)
        for region_id in evaluation["selected_region_ids"]
        for node_id in regions[str(region_id)]["node_ids"]
    }


def _training_cost_key(
    evaluation: Mapping[str, Any],
    perturbations: Mapping[str, Mapping[str, Any]],
    regions: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, int, int, int, str]:
    statistics = _candidate_statistics(evaluation, perturbations, split="training")
    return (
        float(statistics["physical_runtime_max_seconds"]),
        int(statistics["executed_collectives_max"]),
        int(statistics["executed_flops_max"]),
        len(_candidate_node_ids(evaluation, regions)),
        str(evaluation["candidate_id"]),
    )


def _initial_unmeasured_selection(
    regions: Mapping[str, Mapping[str, Any]],
    required_tags: Set[str],
) -> Tuple[str, ...]:
    uncovered = set(required_tags)
    selected: Set[str] = set()
    while uncovered:
        ranked = []
        for region_id, region in regions.items():
            new_tags = uncovered.intersection(region["feature_tags"])
            if new_tags:
                ranked.append((-len(new_tags), len(region["node_ids"]), region_id))
        if not ranked:
            raise SchemaError(f"Chakra projection cannot cover required feature tags: {sorted(uncovered)}")
        _negative_coverage, _node_count, chosen = min(ranked)
        selected.add(chosen)
        uncovered.difference_update(regions[chosen]["feature_tags"])
    if not selected:
        selected.add(min(regions, key=lambda region_id: (len(regions[region_id]["node_ids"]), region_id)))
    return tuple(sorted(selected))


def _encode_selection(
    trace: ChakraExecutionTrace,
    regions: Mapping[str, Mapping[str, Any]],
    selected_region_ids: Sequence[str],
) -> Tuple[bytes, Tuple[int, ...]]:
    requested = {int(node_id) for region_id in selected_region_ids for node_id in regions[region_id]["node_ids"]}
    encoded, selected_nodes = encode_chakra_subgraph(trace, requested)
    if len(selected_nodes) >= len(trace.nodes):
        raise SchemaError("unsupported_for_physical_canary: reason: selected_program_does_not_execute_fewer_nodes")
    return encoded, selected_nodes


def _selection_record(
    trace: ChakraExecutionTrace,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    corpus_id: Optional[str],
    candidate_id: Optional[str],
    selected_region_ids: Sequence[str],
    selected_node_ids: Sequence[int],
    canary_et: bytes,
) -> Dict[str, Any]:
    selection: Dict[str, Any] = {
        "source_et_sha256": trace.source_sha256,
        "projection_id": projection["projection_id"],
        "policy_id": policy["policy_id"],
        "corpus_id": corpus_id,
        "candidate_id": candidate_id,
        "selected_region_ids": list(selected_region_ids),
        "selected_node_ids": list(selected_node_ids),
        "canary_et_sha256": hashlib.sha256(canary_et).hexdigest(),
        "canary_et_bytes": len(canary_et),
    }
    selection["selection_sha256"] = hashlib.sha256(canonical_json_bytes(selection)).hexdigest()
    return selection


def _perturbation_map(corpus: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["perturbation_id"]): row for row in corpus["perturbations"]}


def _observation_map(evaluation: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["perturbation_id"]): row for row in evaluation["observations"]}


def _candidate_statistics(
    evaluation: Mapping[str, Any],
    perturbations: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
) -> Dict[str, Any]:
    observations = _observation_map(evaluation)
    rows = [row for row in perturbations.values() if row["split"] == split]
    severe_false_negatives = 0
    false_positives = 0
    passing_truth = 0
    agreements = 0
    runtime_reductions: List[float] = []
    candidate_runtimes: List[float] = []
    gpu_seconds: List[float] = []
    collectives: List[int] = []
    flops: List[int] = []
    peak_memory: List[int] = []
    for truth in rows:
        observation = observations[str(truth["perturbation_id"])]
        decision = observation["decision"]
        if decision == truth["application_decision"]:
            agreements += 1
        if truth["severe"] and truth["application_decision"] == "fail" and decision == "pass":
            severe_false_negatives += 1
        if truth["application_decision"] == "pass":
            passing_truth += 1
            if decision == "fail":
                false_positives += 1
        application_runtime = float(truth["application_metrics"]["physical_runtime_seconds"])
        metrics = observation["physical_metrics"]
        candidate_runtime = float(metrics["physical_runtime_seconds"])
        runtime_reductions.append(application_runtime / candidate_runtime)
        candidate_runtimes.append(candidate_runtime)
        gpu_seconds.append(float(metrics["gpu_seconds"]))
        collectives.append(int(metrics["executed_collectives"]))
        flops.append(int(metrics["executed_flops"]))
        peak_memory.append(int(metrics["peak_memory_bytes"]))
    return {
        "split": split,
        "decision_pairs": len(rows),
        "severe_false_negatives": severe_false_negatives,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / passing_truth if passing_truth else 0.0,
        "pairwise_decision_agreement": agreements / len(rows),
        "minimum_runtime_reduction_ratio": min(runtime_reductions),
        "physical_runtime_max_seconds": max(candidate_runtimes),
        "gpu_seconds_max": max(gpu_seconds),
        "executed_collectives_max": max(collectives),
        "executed_flops_max": max(flops),
        "peak_memory_bytes_max": max(peak_memory),
    }


def _severe_false_negative_ids(
    evaluation: Mapping[str, Any],
    perturbations: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
) -> List[str]:
    observations = _observation_map(evaluation)
    return sorted(
        str(row["perturbation_id"])
        for row in perturbations.values()
        if row["split"] == split
        and row["severe"]
        and row["application_decision"] == "fail"
        and observations[str(row["perturbation_id"])]["decision"] == "pass"
    )


def _covers_required_tags(
    regions: Mapping[str, Mapping[str, Any]],
    selected_region_ids: Sequence[str],
    required_tags: Set[str],
) -> bool:
    covered = {str(tag) for region_id in selected_region_ids for tag in regions[str(region_id)]["feature_tags"]}
    return required_tags.issubset(covered)


def _statistics_pass(statistics: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    gates = policy["qualification_gates"]
    return bool(
        statistics["severe_false_negatives"] <= gates["maximum_severe_false_negatives"]
        and statistics["false_positive_rate"] <= gates["maximum_false_positive_rate"]
        and statistics["pairwise_decision_agreement"] >= gates["minimum_pairwise_decision_agreement"]
        and statistics["minimum_runtime_reduction_ratio"] >= gates["minimum_runtime_reduction_ratio"]
        and statistics["physical_runtime_max_seconds"] <= policy["runtime_budget_seconds"]
    )


def _policy_failure_codes(statistics: Mapping[str, Any], policy: Mapping[str, Any]) -> List[str]:
    gates = policy["qualification_gates"]
    failures = []
    if statistics["severe_false_negatives"] > gates["maximum_severe_false_negatives"]:
        failures.append("severe_false_negative_limit")
    if statistics["false_positive_rate"] > gates["maximum_false_positive_rate"]:
        failures.append("false_positive_rate_limit")
    if statistics["pairwise_decision_agreement"] < gates["minimum_pairwise_decision_agreement"]:
        failures.append("pairwise_decision_agreement_limit")
    if statistics["minimum_runtime_reduction_ratio"] < gates["minimum_runtime_reduction_ratio"]:
        failures.append("runtime_reduction_limit")
    if statistics["physical_runtime_max_seconds"] > policy["runtime_budget_seconds"]:
        failures.append("runtime_budget")
    return failures


def _baseline_statistics(
    corpus: Mapping[str, Any],
    perturbations: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for evaluation in corpus["evaluations"]:
        method = str(evaluation["method"])
        if method == "reduced_decision_canary":
            continue
        stats = _candidate_statistics(evaluation, perturbations, split="holdout")
        previous = result.get(method)
        row = {"candidate_id": evaluation["candidate_id"], "statistics": stats}
        if previous is None or (
            stats["physical_runtime_max_seconds"],
            str(evaluation["candidate_id"]),
        ) < (
            previous["statistics"]["physical_runtime_max_seconds"],
            previous["candidate_id"],
        ):
            result[method] = row
    return result


def _qualification_status(
    *,
    corpus: Mapping[str, Any],
    policy: Mapping[str, Any],
    training: Mapping[str, Any],
    holdout: Mapping[str, Any],
    missing_baselines: Sequence[str],
    leakage: Mapping[str, Any],
) -> str:
    if corpus["evidence_kind"] != "measured":
        return "blocked_synthetic_evidence"
    if missing_baselines:
        return "blocked_incomplete_baseline_corpus"
    if not _statistics_pass(training, policy):
        return "blocked_no_decision_preserving_training_candidate"
    if not _statistics_pass(holdout, policy):
        return "failed_held_out_validation"
    if leakage["leakage_score"] > policy["privacy"]["maximum_leakage_score"]:
        return "failed_privacy_policy"
    return "qualified_physical_decision_canary"


def _certificate(
    *,
    status: str,
    selection: Mapping[str, Any],
    evidence_kind: str,
    application: Optional[Mapping[str, Any]],
    training: Optional[Mapping[str, Any]],
    holdout: Optional[Mapping[str, Any]],
    baselines: Mapping[str, Any],
    leakage: Mapping[str, Any],
) -> Dict[str, Any]:
    qualified = status == "qualified_physical_decision_canary"
    raw: Dict[str, Any] = {
        "format": PHYSICAL_FIDELITY_CERTIFICATE_FORMAT,
        "status": status,
        "selection_sha256": selection["selection_sha256"],
        "application": copy.deepcopy(application),
        "evidence_kind": evidence_kind,
        "metrics": {
            "training": copy.deepcopy(training),
            "holdout": copy.deepcopy(holdout),
            "baselines": copy.deepcopy(dict(baselines)),
        },
        "leakage_assessment_id": leakage["assessment_id"],
        "claims": {
            "application_ground_truth": "measured" if evidence_kind == "measured" else "unproven",
            "decision_preservation": "held_out_measured" if qualified else "unproven",
            "physical_runtime_reduction": "held_out_measured" if qualified else "unproven",
            "producer_authenticity": "unsigned",
        },
    }
    return with_content_identity(raw, "certificate_id")


__all__ = ["PhysicalSynthesisResult", "synthesize_physical_decision_canary"]
