"""Predeclared evaluation of the same-allocation physical decision gate."""

from __future__ import annotations

import math
import random
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

from ..harness import (
    ContractError,
    JSONResourceLimits,
    canonical_json_bytes,
    canonical_sha256,
    read_bounded_bytes,
    sha256_hex,
    strict_json_loads,
)
from .pipeline import ANALYSIS_SCHEMA
from .schemas import (
    DECISION_FIDELITY_POLICY_SCHEMA,
    DECISION_FIDELITY_VERDICT_SCHEMA,
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
)

_POLICY_LIMITS = JSONResourceLimits(max_document_bytes=1024 * 1024, max_items=10_000)
_REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")
_EVALUATED_REPRESENTATIONS = ("exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")
_STABILITY_REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated")
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class DecisionFidelityError(ContractError):
    """Raised when evidence or policy cannot support deterministic evaluation."""


def _object(value: Any, field: str, expected: Optional[Set[str]] = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionFidelityError(f"{field} must be an object")
    if expected is not None and set(value) != expected:
        raise DecisionFidelityError(f"{field} does not match its closed schema")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionFidelityError(f"{field} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise DecisionFidelityError(f"{field} is outside its declared range")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0, maximum: int = 100_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise DecisionFidelityError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise DecisionFidelityError(f"{field} must be a lowercase SHA-256")
    return value


def decision_fidelity_policy_sha256(policy: Mapping[str, Any]) -> str:
    projection = dict(policy)
    projection.pop("policy_id", None)
    return canonical_sha256(projection)


def validate_decision_fidelity_policy(raw: Any) -> Dict[str, Any]:
    """Validate the immutable experiment policy without optional dependencies."""

    policy = _object(
        raw,
        "decision fidelity policy",
        {"schema", "policy_id", "scope", "measurement", "comparison", "pass_criteria", "outcomes", "kill_or_reframe"},
    )
    if policy["schema"] != DECISION_FIDELITY_POLICY_SCHEMA:
        raise DecisionFidelityError("decision fidelity policy schema is unsupported")
    policy_id = _sha256(policy["policy_id"], "decision fidelity policy.policy_id")
    if policy_id != decision_fidelity_policy_sha256(policy):
        raise DecisionFidelityError("decision fidelity policy ID does not recompute")
    scope = _object(
        policy["scope"],
        "decision fidelity policy.scope",
        {"workload_id", "supported_domain", "configuration_ids", "representations"},
    )
    configurations = scope["configuration_ids"]
    if (
        scope["workload_id"] != "decision-gate"
        or not isinstance(scope["supported_domain"], str)
        or not scope["supported_domain"]
        or not isinstance(configurations, list)
        or len(configurations) < 2
        or any(not isinstance(item, str) or not item for item in configurations)
    ):
        raise DecisionFidelityError("decision fidelity policy scope is invalid")
    if configurations != sorted(set(configurations)):
        raise DecisionFidelityError("decision fidelity configurations must be unique and sorted")
    representations = _object(scope["representations"], "decision fidelity policy.scope.representations")
    if dict(representations) != {
        "source": "ground_truth",
        "exact_work": "product_candidate",
        "stratified": "kill_condition_baseline",
        "isolated": "incumbent_baseline",
        "no_overlap": "causal_ablation",
        "no_rank_skew": "causal_ablation",
    }:
        raise DecisionFidelityError("decision fidelity representation roles are invalid")
    measurement = _object(
        policy["measurement"],
        "decision fidelity policy.measurement",
        {
            "allocation_policy",
            "order_method",
            "timing_semantics",
            "warmup",
            "measured_repetitions",
            "required_correctness",
            "max_relative_iqr_pct",
        },
    )
    expected_measurement = {
        "allocation_policy": "all-representations-in-one-cell-and-process-group-per-configuration",
        "order_method": "iteration-rotated-latin-cycle.v1",
        "timing_semantics": "maximum-rank-cuda-event-whole-program-duration",
        "required_correctness": "passed",
    }
    if any(measurement.get(field) != expected for field, expected in expected_measurement.items()):
        raise DecisionFidelityError("decision fidelity measurement semantics are unsupported")
    _integer(measurement["warmup"], "decision fidelity policy.measurement.warmup", maximum=100)
    _integer(
        measurement["measured_repetitions"],
        "decision fidelity policy.measurement.measured_repetitions",
        minimum=1,
        maximum=1000,
    )
    _number(
        measurement["max_relative_iqr_pct"],
        "decision fidelity policy.measurement.max_relative_iqr_pct",
        minimum=0.0,
        maximum=1000.0,
    )
    comparison = _object(
        policy["comparison"],
        "decision fidelity policy.comparison",
        {
            "primary_metric",
            "configuration_pair_order",
            "pair_label_method",
            "absolute_tie_threshold_us",
            "relative_tie_threshold_pct",
            "relative_threshold_reference",
            "classification_error_counting",
            "uncertainty",
            "reported_metrics",
        },
    )
    if (
        comparison.get("primary_metric") != "program_median_us"
        or comparison.get("configuration_pair_order") != "lexicographic-unordered-pairs"
        or comparison.get("pair_label_method") != "larger-of-absolute-or-relative-tie-band.v1"
        or comparison.get("relative_threshold_reference") != "smaller_pair_median"
    ):
        raise DecisionFidelityError("decision fidelity comparison method is unsupported")
    _number(comparison.get("absolute_tie_threshold_us"), "absolute tie threshold", minimum=0.0)
    _number(comparison.get("relative_tie_threshold_pct"), "relative tie threshold", minimum=0.0)
    classification = _object(
        comparison.get("classification_error_counting"),
        "decision fidelity policy.comparison.classification_error_counting",
        {"source_direction_candidate_tie", "source_tie_candidate_direction", "opposite_direction"},
    )
    if dict(classification) != {
        "source_direction_candidate_tie": "false_negative",
        "source_tie_candidate_direction": "false_positive",
        "opposite_direction": "one_false_negative_and_one_false_positive",
    }:
        raise DecisionFidelityError("decision fidelity classification semantics are unsupported")
    uncertainty = _object(
        comparison.get("uncertainty"),
        "decision fidelity policy.comparison.uncertainty",
        {"method", "confidence", "resamples", "seed", "boundary_crossing_outcome"},
    )
    if (
        uncertainty.get("method") != "independent-percentile-bootstrap-median-difference"
        or uncertainty.get("boundary_crossing_outcome") != "inconclusive"
    ):
        raise DecisionFidelityError("decision fidelity uncertainty method is unsupported")
    confidence = _number(uncertainty.get("confidence"), "bootstrap confidence")
    if not 0.5 < confidence < 1.0:
        raise DecisionFidelityError("bootstrap confidence must be greater than 0.5 and less than 1")
    _integer(uncertainty.get("resamples"), "bootstrap resamples", minimum=100, maximum=100_000)
    _integer(uncertainty.get("seed"), "bootstrap seed", maximum=2**63 - 1)
    reported_metrics = comparison.get("reported_metrics")
    expected_reported_metrics = [
        "pairwise_ranking_agreement",
        "kendall_tau_b",
        "false_negative_count",
        "false_positive_count",
        "median_absolute_relative_error_pct",
        "p95_absolute_relative_error_pct",
        "execution_time_ratio_to_source",
    ]
    if reported_metrics != expected_reported_metrics:
        raise DecisionFidelityError("decision fidelity reported metric inventory is unsupported")
    criteria = _object(policy["pass_criteria"], "decision fidelity policy.pass_criteria")
    expected_criteria = {
        "exact_work_min_pairwise_agreement",
        "exact_work_min_kendall_tau_b",
        "exact_work_max_false_negative_count",
        "exact_work_max_false_positive_count",
        "exact_work_max_median_absolute_relative_error_pct",
        "exact_work_max_p95_absolute_relative_error_pct",
        "exact_work_min_agreement_advantage_pairs_over_isolated",
        "exact_work_min_agreement_advantage_pairs_over_stratified",
    }
    if set(criteria) != expected_criteria:
        raise DecisionFidelityError("decision fidelity pass criteria are incomplete")
    _number(criteria["exact_work_min_pairwise_agreement"], "minimum pairwise agreement", minimum=0, maximum=1)
    _number(criteria["exact_work_min_kendall_tau_b"], "minimum Kendall tau-b", minimum=-1, maximum=1)
    for field in (
        "exact_work_max_false_negative_count",
        "exact_work_max_false_positive_count",
        "exact_work_min_agreement_advantage_pairs_over_isolated",
        "exact_work_min_agreement_advantage_pairs_over_stratified",
    ):
        _integer(criteria[field], field)
    for field in (
        "exact_work_max_median_absolute_relative_error_pct",
        "exact_work_max_p95_absolute_relative_error_pct",
    ):
        _number(criteria[field], field, minimum=0)
    outcomes = _object(policy["outcomes"], "decision fidelity policy.outcomes")
    if set(outcomes) != {"pass", "fail", "inconclusive", "incomparable"} or any(
        not isinstance(value, str) or not value for value in outcomes.values()
    ):
        raise DecisionFidelityError("decision fidelity outcomes are invalid")
    kill = _object(
        policy["kill_or_reframe"],
        "decision fidelity policy.kill_or_reframe",
        {"condition", "outcome", "cost_claim", "generality_claim", "independent_operator_claim"},
    )
    if (
        kill.get("condition") != "exact_work_pairwise_agreement_not_greater_than_stratified"
        or kill.get("outcome") != "reframe_as_evidence_and_research_framework"
        or kill.get("cost_claim") != "not_evaluated_by_this_reduced-source-campaign"
        or kill.get("generality_claim") != "not_evaluated_beyond_the_declared_supported_domain"
        or kill.get("independent_operator_claim") != "not_evaluated_by_this_campaign"
    ):
        raise DecisionFidelityError("decision fidelity reframe policy is unsupported")
    return cast(Dict[str, Any], dict(policy))


def load_decision_fidelity_policy(path: Path) -> bytes:
    data = read_bounded_bytes(path, max_bytes=_POLICY_LIMITS.max_document_bytes, field="decision fidelity policy")
    validate_decision_fidelity_policy(strict_json_loads(data, limits=_POLICY_LIMITS))
    return data


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _pair_threshold(first: float, second: float, comparison: Mapping[str, Any]) -> float:
    absolute = float(comparison["absolute_tie_threshold_us"])
    relative = min(first, second) * float(comparison["relative_tie_threshold_pct"]) / 100.0
    return max(absolute, relative)


def _pair_label(first: float, second: float, comparison: Mapping[str, Any]) -> str:
    threshold = _pair_threshold(first, second, comparison)
    difference = second - first
    if difference > threshold:
        return "first_faster"
    if difference < -threshold:
        return "second_faster"
    return "tie"


def _bootstrap_pair(
    first: Sequence[float],
    second: Sequence[float],
    *,
    comparison: Mapping[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    uncertainty = cast(Mapping[str, Any], comparison["uncertainty"])
    resamples = int(uncertainty["resamples"])
    differences = []
    for _ in range(resamples):
        first_median = _median([first[rng.randrange(len(first))] for _ in first])
        second_median = _median([second[rng.randrange(len(second))] for _ in second])
        differences.append(second_median - first_median)
    alpha = (1.0 - float(uncertainty["confidence"])) / 2.0
    lower = _percentile(differences, alpha)
    upper = _percentile(differences, 1.0 - alpha)
    first_median = _median(first)
    second_median = _median(second)
    threshold = _pair_threshold(first_median, second_median, comparison)
    observed_label = _pair_label(first_median, second_median, comparison)
    if lower > threshold:
        uncertainty_label = "first_faster"
    elif upper < -threshold:
        uncertainty_label = "second_faster"
    elif lower >= -threshold and upper <= threshold:
        uncertainty_label = "tie"
    else:
        uncertainty_label = "inconclusive"
    return {
        "observed_label": observed_label,
        "uncertainty_label": uncertainty_label,
        "difference_us": second_median - first_median,
        "tie_threshold_us": threshold,
        "confidence_interval_us": [lower, upper],
    }


def _policy_label_tau_b(
    representation: str,
    pair_rows: Sequence[Mapping[str, Any]],
) -> float:
    """Compute tau-b from the same policy-level pair labels used by the gate."""

    order = {"first_faster": 1, "tie": 0, "second_faster": -1}
    concordant = discordant = source_ties = candidate_ties = 0
    for row in pair_rows:
        representations = cast(Mapping[str, Any], row["representations"])
        source_label = cast(Mapping[str, Any], representations["source"])["observed_label"]
        candidate_label = cast(Mapping[str, Any], representations[representation])["observed_label"]
        source_relation = order[str(source_label)]
        candidate_relation = order[str(candidate_label)]
        if source_relation == 0 and candidate_relation == 0:
            continue
        if source_relation == 0:
            source_ties += 1
        elif candidate_relation == 0:
            candidate_ties += 1
        elif source_relation == candidate_relation:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + source_ties) * (concordant + discordant + candidate_ties))
    return 0.0 if denominator == 0.0 else (concordant - discordant) / denominator


def _representation_metrics(
    representation: str,
    *,
    configurations: Sequence[str],
    samples: Mapping[str, Mapping[str, Tuple[float, ...]]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_medians = [_median(samples[configuration]["source"]) for configuration in configurations]
    candidate_medians = [_median(samples[configuration][representation]) for configuration in configurations]
    agreement_count = false_negatives = false_positives = 0
    for row in pair_rows:
        source_label = cast(Mapping[str, Any], row["representations"])["source"]["observed_label"]
        candidate_label = cast(Mapping[str, Any], row["representations"])[representation]["observed_label"]
        if source_label == candidate_label:
            agreement_count += 1
        elif source_label == "tie":
            false_positives += 1
        elif candidate_label == "tie":
            false_negatives += 1
        else:
            false_negatives += 1
            false_positives += 1
    errors = [abs(candidate - source) / source * 100.0 for source, candidate in zip(source_medians, candidate_medians)]
    ratios = [candidate / source for source, candidate in zip(source_medians, candidate_medians)]
    return {
        "pairwise_agreement_count": agreement_count,
        "pairwise_pair_count": len(pair_rows),
        "pairwise_ranking_agreement": agreement_count / len(pair_rows),
        "kendall_tau_b": _policy_label_tau_b(representation, pair_rows),
        "false_negative_count": false_negatives,
        "false_positive_count": false_positives,
        "median_absolute_relative_error_pct": _median(errors),
        "p95_absolute_relative_error_pct": _percentile(errors, 0.95),
        "median_execution_time_ratio_to_source": _median(ratios),
        "configuration_medians_us": {
            configuration: candidate_medians[index] for index, configuration in enumerate(configurations)
        },
    }


def _policy_input_binding(
    aggregate: Mapping[str, Any],
    *,
    policy_sha256: str,
    policy_size_bytes: int,
) -> Mapping[str, Any]:
    provenance = _object(aggregate.get("provenance"), "aggregate.provenance")
    campaigns = provenance.get("campaigns")
    if not isinstance(campaigns, list) or len(campaigns) != 1:
        raise DecisionFidelityError("decision gate evaluation requires exactly one campaign")
    if provenance.get("trusted_join_sha256") != canonical_sha256(campaigns):
        raise DecisionFidelityError("decision gate aggregate trusted join does not recompute")
    campaign = _object(campaigns[0], "aggregate.provenance.campaigns[0]")
    run_id = campaign.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise DecisionFidelityError("decision gate campaign run ID is missing")
    for field in ("manifest_sha256", "selection_sha256", "verdict_sha256"):
        _sha256(campaign.get(field), f"decision gate campaign.{field}")
    inputs = campaign.get("inputs")
    if not isinstance(inputs, list):
        raise DecisionFidelityError("decision gate campaign input inventory is missing")
    matches = [item for item in inputs if isinstance(item, Mapping) and item.get("id") == "decision-fidelity-policy"]
    if (
        len(matches) != 1
        or matches[0].get("sha256") != policy_sha256
        or matches[0].get("size_bytes") != policy_size_bytes
    ):
        raise DecisionFidelityError("decision fidelity policy bytes are not bound by the frozen campaign")
    return campaign


def evaluate_decision_fidelity(
    aggregate: Mapping[str, Any],
    policy_bytes: bytes,
) -> Dict[str, Any]:
    """Evaluate complete trusted evidence under the predeclared policy."""

    aggregate = _object(aggregate, "aggregate")
    if not isinstance(policy_bytes, bytes) or not 0 < len(policy_bytes) <= _POLICY_LIMITS.max_document_bytes:
        raise DecisionFidelityError("decision fidelity policy bytes are outside the supported limit")
    validated_policy = validate_decision_fidelity_policy(strict_json_loads(policy_bytes, limits=_POLICY_LIMITS))
    policy_sha256 = sha256_hex(policy_bytes)
    policy_size_bytes = len(policy_bytes)
    if aggregate.get("schema") != ANALYSIS_SCHEMA:
        raise DecisionFidelityError("decision gate requires a trusted validated aggregate")
    campaign = _policy_input_binding(
        aggregate,
        policy_sha256=policy_sha256,
        policy_size_bytes=policy_size_bytes,
    )
    completeness = _object(aggregate.get("completeness"), "aggregate.completeness")
    configurations = list(validated_policy["scope"]["configuration_ids"])
    selected = aggregate.get("selected_cells")
    if not isinstance(selected, list):
        raise DecisionFidelityError("aggregate selected cell inventory is invalid")
    issues: List[Dict[str, Any]] = []
    if completeness.get("complete") is not True or completeness.get("issue_codes") != []:
        issues.append({"code": "incomplete_evidence", "detail": "persisted completeness is not complete"})
    rows: Dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(selected):
        row = _object(raw_row, f"aggregate.selected_cells[{index}]")
        configuration = row.get("configuration_id")
        if (
            row.get("workload_id") != "decision-gate"
            or row.get("measurement_schema") != PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA
            or row.get("repetition") != 0
            or not isinstance(configuration, str)
            or configuration in rows
        ):
            raise DecisionFidelityError("aggregate selected cell is outside the frozen decision gate")
        rows[configuration] = row
    if set(rows) != set(configurations):
        issues.append({"code": "incomplete_configuration_inventory", "detail": "selected configurations differ"})

    samples: Dict[str, Dict[str, Tuple[float, ...]]] = {}
    nodes = set()
    identities = set()
    if not issues:
        for configuration in configurations:
            row = rows[configuration]
            gate = _object(row.get("decision_gate"), f"selected cell {configuration}.decision_gate")
            execution = _object(gate.get("execution"), f"selected cell {configuration}.execution")
            if (
                execution.get("iterations") != validated_policy["measurement"]["measured_repetitions"]
                or execution.get("warmup") != validated_policy["measurement"]["warmup"]
                or execution.get("order_method") != validated_policy["measurement"]["order_method"]
                or execution.get("timing_semantics") != validated_policy["measurement"]["timing_semantics"]
            ):
                raise DecisionFidelityError("selected decision gate execution disagrees with policy")
            representations = _object(gate.get("representations"), f"selected cell {configuration}.representations")
            samples[configuration] = {}
            for representation in _REPRESENTATIONS:
                value = _object(representations.get(representation), f"representation {representation}")
                timings = value.get("timings_us")
                if not isinstance(timings, list):
                    raise DecisionFidelityError("decision gate timings are invalid")
                parsed = tuple(_number(item, "decision gate timing", minimum=0.0) for item in timings)
                if len(parsed) != validated_policy["measurement"]["measured_repetitions"]:
                    raise DecisionFidelityError("decision gate timing inventory is incomplete")
                samples[configuration][representation] = parsed
            if _median(samples[configuration]["source"]) <= 0.0:
                issues.append(
                    {
                        "code": "nonpositive_source_timing",
                        "detail": f"source timing is not positive for {configuration}",
                    }
                )
            runtime = _object(row.get("decision_gate_runtime"), f"selected cell {configuration}.runtime")
            hostname = runtime.get("hostname")
            if not isinstance(hostname, str) or not hostname:
                raise DecisionFidelityError("decision gate runtime hostname is missing")
            nodes.add(hostname.split(".", 1)[0])
            identities.add(
                (
                    canonical_sha256(gate["request"]),
                    canonical_sha256(gate["materialization"]),
                    canonical_sha256(gate["policy"]),
                )
            )
        if len(nodes) != 1:
            issues.append({"code": "node_mismatch", "detail": "configurations did not execute on one node"})
        if len(identities) != 1:
            issues.append({"code": "artifact_identity_mismatch", "detail": "configurations used different inputs"})

    pair_rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    uncertainty_issues: List[str] = []
    stability_issues: List[str] = []
    if not issues:
        comparison = cast(Mapping[str, Any], validated_policy["comparison"])
        rng = random.Random(int(comparison["uncertainty"]["seed"]))
        for first, second in combinations(configurations, 2):
            representation_rows: Dict[str, Any] = {}
            for representation in _REPRESENTATIONS:
                comparison_row = _bootstrap_pair(
                    samples[first][representation],
                    samples[second][representation],
                    comparison=comparison,
                    rng=rng,
                )
                representation_rows[representation] = comparison_row
                if (
                    representation in {"source", "exact_work", "stratified", "isolated"}
                    and comparison_row["uncertainty_label"] != comparison_row["observed_label"]
                ):
                    uncertainty_issues.append(
                        f"{first}|{second}|{representation}|"
                        f"observed={comparison_row['observed_label']}|"
                        f"uncertainty={comparison_row['uncertainty_label']}"
                    )
            pair_rows.append(
                {
                    "first_configuration_id": first,
                    "second_configuration_id": second,
                    "representations": representation_rows,
                }
            )
        for representation in _EVALUATED_REPRESENTATIONS:
            metrics[representation] = _representation_metrics(
                representation,
                configurations=configurations,
                samples=samples,
                pair_rows=pair_rows,
            )
        maximum_iqr = float(validated_policy["measurement"]["max_relative_iqr_pct"])
        for configuration in configurations:
            for representation in _STABILITY_REPRESENTATIONS:
                values = samples[configuration][representation]
                median = _median(values)
                ordered = sorted(values)
                midpoint = len(ordered) // 2
                lower = ordered[:midpoint]
                upper = ordered[midpoint:] if len(ordered) % 2 == 0 else ordered[midpoint + 1 :]
                iqr = 0.0 if not lower or not upper else _median(upper) - _median(lower)
                relative_iqr = (
                    math.inf if median == 0.0 and iqr > 0.0 else (0.0 if median == 0.0 else iqr / median * 100.0)
                )
                if relative_iqr > maximum_iqr:
                    stability_issues.append(f"{configuration}|{representation}|{relative_iqr}")

    criteria_rows: List[Dict[str, Any]] = []
    reframe = False
    reframe_evaluated = False
    if metrics:
        exact = metrics["exact_work"]
        criteria = validated_policy["pass_criteria"]
        checks = (
            (
                "exact_work_min_pairwise_agreement",
                exact["pairwise_ranking_agreement"],
                ">=",
                criteria["exact_work_min_pairwise_agreement"],
            ),
            ("exact_work_min_kendall_tau_b", exact["kendall_tau_b"], ">=", criteria["exact_work_min_kendall_tau_b"]),
            (
                "exact_work_max_false_negative_count",
                exact["false_negative_count"],
                "<=",
                criteria["exact_work_max_false_negative_count"],
            ),
            (
                "exact_work_max_false_positive_count",
                exact["false_positive_count"],
                "<=",
                criteria["exact_work_max_false_positive_count"],
            ),
            (
                "exact_work_max_median_absolute_relative_error_pct",
                exact["median_absolute_relative_error_pct"],
                "<=",
                criteria["exact_work_max_median_absolute_relative_error_pct"],
            ),
            (
                "exact_work_max_p95_absolute_relative_error_pct",
                exact["p95_absolute_relative_error_pct"],
                "<=",
                criteria["exact_work_max_p95_absolute_relative_error_pct"],
            ),
            (
                "exact_work_min_agreement_advantage_pairs_over_isolated",
                exact["pairwise_agreement_count"] - metrics["isolated"]["pairwise_agreement_count"],
                ">=",
                criteria["exact_work_min_agreement_advantage_pairs_over_isolated"],
            ),
            (
                "exact_work_min_agreement_advantage_pairs_over_stratified",
                exact["pairwise_agreement_count"] - metrics["stratified"]["pairwise_agreement_count"],
                ">=",
                criteria["exact_work_min_agreement_advantage_pairs_over_stratified"],
            ),
        )
        for criterion_id, observed, operator, required in checks:
            passed = observed >= required if operator == ">=" else observed <= required
            criteria_rows.append(
                {
                    "criterion_id": criterion_id,
                    "observed": observed,
                    "operator": operator,
                    "required": required,
                    "passed": passed,
                }
            )
    if issues:
        outcome = "incomparable"
    elif uncertainty_issues or stability_issues:
        outcome = "inconclusive"
    elif criteria_rows and all(row["passed"] for row in criteria_rows):
        outcome = "pass"
    else:
        outcome = "fail"
    if metrics and outcome in {"pass", "fail"}:
        reframe_evaluated = True
        reframe = metrics["exact_work"]["pairwise_agreement_count"] <= metrics["stratified"]["pairwise_agreement_count"]
    if reframe:
        positioning = "evidence_and_research_framework"
    elif outcome == "pass":
        positioning = "narrow_physical_fidelity_supported_research_alpha"
    else:
        positioning = "research_alpha_unvalidated"
    result: Dict[str, Any] = {
        "schema": DECISION_FIDELITY_VERDICT_SCHEMA,
        "outcome": outcome,
        "policy": {
            "schema": validated_policy["schema"],
            "policy_id": validated_policy["policy_id"],
            "sha256": policy_sha256,
            "size_bytes": policy_size_bytes,
        },
        "evidence": {
            "aggregate_sha256": canonical_sha256(aggregate),
            "run_id": campaign.get("run_id"),
            "manifest_sha256": campaign.get("manifest_sha256"),
            "selection_sha256": campaign.get("selection_sha256"),
            "completeness_verdict_sha256": campaign.get("verdict_sha256"),
            "configuration_count": len(rows),
            "configuration_pair_count": len(pair_rows),
            "same_node": next(iter(nodes)) if len(nodes) == 1 else None,
        },
        "issues": issues,
        "uncertainty": {
            "status": "inconclusive" if uncertainty_issues or stability_issues else "decisive",
            "inconclusive_pairs": uncertainty_issues,
            "unstable_cells": stability_issues,
        },
        "pairwise_comparisons": pair_rows,
        "representation_metrics": metrics,
        "criteria": criteria_rows,
        "product_interpretation": {
            "kill_or_reframe_evaluated": reframe_evaluated,
            "kill_or_reframe_triggered": reframe,
            "positioning": positioning,
            "cost_claim": validated_policy["kill_or_reframe"]["cost_claim"],
            "generality_claim": validated_policy["kill_or_reframe"]["generality_claim"],
            "independent_operator_claim": validated_policy["kill_or_reframe"]["independent_operator_claim"],
        },
    }
    result["verdict_id"] = canonical_sha256(result)
    return result


def write_decision_fidelity_verdict(path: Path, verdict: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise DecisionFidelityError(f"decision fidelity verdict output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(verdict)
    with path.open("xb") as handle:
        handle.write(data)
