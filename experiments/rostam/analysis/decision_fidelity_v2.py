"""Hierarchical, simultaneous evaluation for replicated exact-work gates."""

from __future__ import annotations

import math
import random
import statistics
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

from ..harness import JSONResourceLimits, canonical_sha256, sha256_hex, strict_json_loads
from ..lib.executor_artifact import ExecutorArtifact
from .decision_fidelity import (
    REPORTED_METRIC_TO_VERDICT_FIELD,
    DecisionFidelityError,
    _frozen_evaluator_record,
    _integer,
    _median,
    _number,
    _object,
    _pair_label,
    _pair_threshold,
    _percentile,
    _policy_input_binding,
    _representation_metrics,
    _sha256,
    decision_fidelity_policy_sha256,
)
from .pipeline import ANALYSIS_SCHEMA
from .schemas import (
    DECISION_FIDELITY_POLICY_SCHEMA_V2,
    DECISION_FIDELITY_VERDICT_SCHEMA_V2,
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
)

_POLICY_LIMITS = JSONResourceLimits(max_document_bytes=1024 * 1024, max_items=10_000)
_REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")
_EVALUATED_REPRESENTATIONS = ("exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")
_UNCERTAINTY_REPRESENTATIONS = ("source", "exact_work")
_STABILITY_REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated")

BlockSamples = Dict[int, Dict[str, Dict[str, Tuple[float, ...]]]]
MedianVector = Dict[str, Dict[str, float]]


def _environment_identity(raw: Any, field: str) -> str:
    environment = _object(
        raw,
        field,
        {
            "schema",
            "driver_version",
            "nccl_library_sha256",
            "gpus",
            "topology",
            "node_state",
            "binding",
            "observation_sha256",
        },
    )
    if environment["schema"] != "commcanary.rostam.runtime-observation.v2":
        raise DecisionFidelityError(f"{field}.schema is unsupported")
    _sha256(environment["observation_sha256"], f"{field}.observation_sha256")
    _sha256(environment["nccl_library_sha256"], f"{field}.nccl_library_sha256")
    if not isinstance(environment["driver_version"], str) or not environment["driver_version"]:
        raise DecisionFidelityError(f"{field}.driver_version is invalid")
    gpus = environment["gpus"]
    required_gpu_fields = {
        "index",
        "uuid",
        "name",
        "driver_version",
        "pci_bus_id",
        "persistence_mode",
        "performance_state",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
    }
    if not isinstance(gpus, list) or len(gpus) != 4:
        raise DecisionFidelityError(f"{field}.gpus must contain the four-GPU inventory")
    for index, raw_gpu in enumerate(gpus):
        gpu = _object(raw_gpu, f"{field}.gpus[{index}]", required_gpu_fields)
        if gpu["index"] != index or gpu["driver_version"] != environment["driver_version"]:
            raise DecisionFidelityError(f"{field}.gpus[{index}] identity is invalid")
        for text_field in ("uuid", "name", "pci_bus_id", "persistence_mode", "performance_state"):
            if not isinstance(gpu[text_field], str) or not gpu[text_field]:
                raise DecisionFidelityError(f"{field}.gpus[{index}].{text_field} is invalid")
        _integer(gpu["temperature_c"], f"{field}.gpus[{index}].temperature_c", minimum=-50, maximum=200)
        _number(gpu["power_draw_w"], f"{field}.gpus[{index}].power_draw_w", minimum=0.0)
        _number(gpu["power_limit_w"], f"{field}.gpus[{index}].power_limit_w", minimum=0.000001)
        _integer(gpu["sm_clock_mhz"], f"{field}.gpus[{index}].sm_clock_mhz", maximum=100_000)
        _integer(gpu["memory_clock_mhz"], f"{field}.gpus[{index}].memory_clock_mhz", maximum=100_000)
    topology = _object(environment["topology"], f"{field}.topology", {"method", "text"})
    node_state = _object(environment["node_state"], f"{field}.node_state", {"method", "text"})
    binding = _object(
        environment["binding"],
        f"{field}.binding",
        {"environment", "cpu_affinity", "cpu_affinity_method"},
    )
    if (
        topology["method"] != "nvidia-smi topo -m"
        or not isinstance(topology["text"], str)
        or not topology["text"]
        or node_state["method"] != "scontrol show node --oneliner HOSTNAME"
        or not isinstance(node_state["text"], str)
        or not node_state["text"]
        or binding["cpu_affinity_method"] != "sched_getaffinity"
        or not isinstance(binding["cpu_affinity"], list)
        or not binding["cpu_affinity"]
    ):
        raise DecisionFidelityError(f"{field} lacks required topology, node-state, or affinity evidence")
    for cpu in binding["cpu_affinity"]:
        _integer(cpu, f"{field}.binding.cpu_affinity[]", maximum=1_000_000)
    return str(environment["observation_sha256"])


def validate_decision_fidelity_policy_v2(raw: Any) -> Dict[str, Any]:
    """Validate the v2 policy without relying on optional JSON Schema tooling."""

    policy = _object(
        raw,
        "decision fidelity policy v2",
        {"schema", "policy_id", "scope", "measurement", "comparison", "pass_criteria", "outcomes", "claim_boundary"},
    )
    if policy["schema"] != DECISION_FIDELITY_POLICY_SCHEMA_V2:
        raise DecisionFidelityError("decision fidelity policy v2 schema is unsupported")
    policy_id = _sha256(policy["policy_id"], "decision fidelity policy v2.policy_id")
    if policy_id != decision_fidelity_policy_sha256(policy):
        raise DecisionFidelityError("decision fidelity policy v2 ID does not recompute")

    scope = _object(
        policy["scope"],
        "decision fidelity policy v2.scope",
        {"workload_id", "supported_domain", "configuration_ids", "representations"},
    )
    configurations = scope["configuration_ids"]
    if (
        scope["workload_id"] != "decision-gate-exact-replicated"
        or not isinstance(scope["supported_domain"], str)
        or not scope["supported_domain"]
        or not isinstance(configurations, list)
        or len(configurations) < 2
        or configurations != sorted(set(configurations))
    ):
        raise DecisionFidelityError("decision fidelity policy v2 scope is invalid")
    representations = _object(scope["representations"], "decision fidelity policy v2.scope.representations")
    if dict(representations) != {
        "source": "ground_truth",
        "exact_work": "positive_conformance_control",
        "stratified": "sampling_baseline",
        "isolated": "incumbent_baseline",
        "no_overlap": "causal_ablation",
        "no_rank_skew": "causal_ablation",
    }:
        raise DecisionFidelityError("decision fidelity policy v2 representation roles are invalid")

    measurement = _object(
        policy["measurement"],
        "decision fidelity policy v2.measurement",
        {
            "allocation_policy",
            "allocation_blocks",
            "block_pairing",
            "order_method",
            "timing_semantics",
            "warmup",
            "measured_repetitions",
            "required_correctness",
            "required_environment_evidence",
            "max_relative_iqr_pct",
            "require_distinct_job_ids",
            "retry_policy",
        },
    )
    expected_measurement = {
        "allocation_policy": "one-fresh-exclusive-allocation-per-configuration-and-block",
        "block_pairing": "campaign-repetition-index",
        "order_method": "allocation-block-rotated-latin-cycle.v2",
        "timing_semantics": "maximum-rank-cuda-event-whole-program-duration",
        "measured_repetitions": 24,
        "required_correctness": "passed",
        "required_environment_evidence": [
            "cpu_affinity",
            "gpu_clocks",
            "gpu_persistence_mode",
            "gpu_power",
            "gpu_temperature",
            "node_state",
        ],
        "require_distinct_job_ids": True,
        "retry_policy": "infrastructure-failure-only-never-retry-for-noise",
    }
    if any(measurement.get(field) != expected for field, expected in expected_measurement.items()):
        raise DecisionFidelityError("decision fidelity policy v2 measurement semantics are unsupported")
    _integer(measurement["allocation_blocks"], "allocation_blocks", minimum=5, maximum=10)
    _integer(measurement["warmup"], "warmup", maximum=100)
    _number(measurement["max_relative_iqr_pct"], "max_relative_iqr_pct", minimum=0.0, maximum=1000.0)

    comparison = _object(
        policy["comparison"],
        "decision fidelity policy v2.comparison",
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
        comparison.get("primary_metric") != "median-of-allocation-medians-us"
        or comparison.get("configuration_pair_order") != "lexicographic-unordered-pairs"
        or comparison.get("pair_label_method") != "larger-of-absolute-or-relative-tie-band.v1"
        or comparison.get("relative_threshold_reference") != "smaller_pair_median"
    ):
        raise DecisionFidelityError("decision fidelity policy v2 comparison method is unsupported")
    _number(comparison["absolute_tie_threshold_us"], "absolute tie threshold", minimum=0.0)
    _number(comparison["relative_tie_threshold_pct"], "relative tie threshold", minimum=0.0)
    classification = _object(
        comparison["classification_error_counting"],
        "decision fidelity policy v2 classification",
        {"source_direction_candidate_tie", "source_tie_candidate_direction", "opposite_direction"},
    )
    if dict(classification) != {
        "source_direction_candidate_tie": "false_negative",
        "source_tie_candidate_direction": "false_positive",
        "opposite_direction": "one_false_negative_and_one_false_positive",
    }:
        raise DecisionFidelityError("decision fidelity policy v2 classification semantics are unsupported")
    uncertainty = _object(
        comparison["uncertainty"],
        "decision fidelity policy v2 uncertainty",
        {
            "method",
            "confidence",
            "resamples",
            "seed",
            "outer_unit",
            "inner_unit",
            "simultaneous_scope",
            "boundary_crossing_outcome",
        },
    )
    expected_uncertainty = {
        "method": "hierarchical-paired-block-bootstrap-policy-margin-standardized-max.v2",
        "outer_unit": "complete-allocation-block",
        "inner_unit": "measured-iteration-vector-within-cell",
        "simultaneous_scope": "source-and-exact-work-pair-margins-exact-work-metrics-and-criteria",
        "boundary_crossing_outcome": "inconclusive",
    }
    if any(uncertainty.get(field) != expected for field, expected in expected_uncertainty.items()):
        raise DecisionFidelityError("decision fidelity policy v2 uncertainty semantics are unsupported")
    confidence = _number(uncertainty["confidence"], "bootstrap confidence")
    if not 0.5 < confidence < 1.0:
        raise DecisionFidelityError("bootstrap confidence must be greater than 0.5 and less than 1")
    _integer(uncertainty["resamples"], "bootstrap resamples", minimum=100, maximum=100_000)
    _integer(uncertainty["seed"], "bootstrap seed", maximum=2**63 - 1)
    if comparison["reported_metrics"] != list(REPORTED_METRIC_TO_VERDICT_FIELD.values()):
        raise DecisionFidelityError("decision fidelity policy v2 reported metric inventory is unsupported")

    criteria = _object(policy["pass_criteria"], "decision fidelity policy v2.pass_criteria")
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
        raise DecisionFidelityError("decision fidelity policy v2 pass criteria are incomplete")
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

    outcomes = _object(policy["outcomes"], "decision fidelity policy v2.outcomes")
    if set(outcomes) != {"pass", "fail", "inconclusive", "incomparable"} or any(
        not isinstance(value, str) or not value for value in outcomes.values()
    ):
        raise DecisionFidelityError("decision fidelity policy v2 outcomes are invalid")
    boundary = _object(
        policy["claim_boundary"],
        "decision fidelity policy v2.claim_boundary",
        {
            "mode",
            "exact_work_claim",
            "reduced_canary_claim",
            "cost_claim",
            "generality_claim",
            "independent_operator_claim",
        },
    )
    if dict(boundary) != {
        "mode": "exact_qualification_capsule",
        "exact_work_claim": "portable_reconstruction_positive_control",
        "reduced_canary_claim": "not_evaluated",
        "cost_claim": "not_evaluated_by_exact_work_control",
        "generality_claim": "not_evaluated_beyond_the_declared_supported_domain",
        "independent_operator_claim": "not_evaluated_by_this_campaign",
    }:
        raise DecisionFidelityError("decision fidelity policy v2 claim boundary is unsupported")
    return cast(Dict[str, Any], dict(policy))


def _nested_median_vector(
    samples: BlockSamples,
    *,
    blocks: Sequence[int],
    configurations: Sequence[str],
) -> MedianVector:
    return {
        configuration: {
            representation: _median([_median(samples[block][configuration][representation]) for block in blocks])
            for representation in _REPRESENTATIONS
        }
        for configuration in configurations
    }


def _policy_margin(
    first: float,
    second: float,
    *,
    target_label: str,
    comparison: Mapping[str, Any],
) -> float:
    difference = second - first
    threshold = _pair_threshold(first, second, comparison)
    if target_label == "first_faster":
        return difference - threshold
    if target_label == "second_faster":
        return -difference - threshold
    if target_label == "tie":
        return threshold - abs(difference)
    raise DecisionFidelityError(f"unknown pair label {target_label!r}")


def _pair_rows(
    vector: MedianVector,
    *,
    configurations: Sequence[str],
    comparison: Mapping[str, Any],
    targets: Optional[Mapping[Tuple[str, str, str], str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for first, second in combinations(configurations, 2):
        representations: Dict[str, Any] = {}
        for representation in _REPRESENTATIONS:
            first_median = vector[first][representation]
            second_median = vector[second][representation]
            observed_label = _pair_label(first_median, second_median, comparison)
            target = observed_label if targets is None else targets[(first, second, representation)]
            representations[representation] = {
                "observed_label": observed_label,
                "difference_us": second_median - first_median,
                "tie_threshold_us": _pair_threshold(first_median, second_median, comparison),
                "policy_margin_us": _policy_margin(
                    first_median,
                    second_median,
                    target_label=target,
                    comparison=comparison,
                ),
            }
        rows.append(
            {
                "first_configuration_id": first,
                "second_configuration_id": second,
                "representations": representations,
            }
        )
    return rows


def _metrics(
    vector: MedianVector,
    *,
    configurations: Sequence[str],
    pair_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    singleton_samples = {
        configuration: {representation: (vector[configuration][representation],) for representation in _REPRESENTATIONS}
        for configuration in configurations
    }
    return {
        representation: _representation_metrics(
            representation,
            configurations=configurations,
            samples=singleton_samples,
            pair_rows=pair_rows,
        )
        for representation in _EVALUATED_REPRESENTATIONS
    }


def _criterion_values(metrics: Mapping[str, Any], criteria: Mapping[str, Any]) -> Dict[str, Tuple[float, str, float]]:
    exact = cast(Mapping[str, Any], metrics["exact_work"])
    isolated = cast(Mapping[str, Any], metrics["isolated"])
    stratified = cast(Mapping[str, Any], metrics["stratified"])
    return {
        "exact_work_min_pairwise_agreement": (
            float(exact["pairwise_ranking_agreement"]),
            ">=",
            float(criteria["exact_work_min_pairwise_agreement"]),
        ),
        "exact_work_min_kendall_tau_b": (
            float(exact["kendall_tau_b"]),
            ">=",
            float(criteria["exact_work_min_kendall_tau_b"]),
        ),
        "exact_work_max_false_negative_count": (
            float(exact["false_negative_count"]),
            "<=",
            float(criteria["exact_work_max_false_negative_count"]),
        ),
        "exact_work_max_false_positive_count": (
            float(exact["false_positive_count"]),
            "<=",
            float(criteria["exact_work_max_false_positive_count"]),
        ),
        "exact_work_max_median_absolute_relative_error_pct": (
            float(exact["median_absolute_relative_error_pct"]),
            "<=",
            float(criteria["exact_work_max_median_absolute_relative_error_pct"]),
        ),
        "exact_work_max_p95_absolute_relative_error_pct": (
            float(exact["p95_absolute_relative_error_pct"]),
            "<=",
            float(criteria["exact_work_max_p95_absolute_relative_error_pct"]),
        ),
        "exact_work_min_agreement_advantage_pairs_over_isolated": (
            float(exact["pairwise_agreement_count"] - isolated["pairwise_agreement_count"]),
            ">=",
            float(criteria["exact_work_min_agreement_advantage_pairs_over_isolated"]),
        ),
        "exact_work_min_agreement_advantage_pairs_over_stratified": (
            float(exact["pairwise_agreement_count"] - stratified["pairwise_agreement_count"]),
            ">=",
            float(criteria["exact_work_min_agreement_advantage_pairs_over_stratified"]),
        ),
    }


def _criterion_margin(value: float, operator: str, required: float) -> float:
    return value - required if operator == ">=" else required - value


def _statistics(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for row in pair_rows:
        first = str(row["first_configuration_id"])
        second = str(row["second_configuration_id"])
        representations = cast(Mapping[str, Any], row["representations"])
        for representation in _UNCERTAINTY_REPRESENTATIONS:
            result = cast(Mapping[str, Any], representations[representation])
            values[f"pair|{first}|{second}|{representation}"] = float(result["policy_margin_us"])
    exact_metrics = cast(Mapping[str, Any], metrics["exact_work"])
    for metric in REPORTED_METRIC_TO_VERDICT_FIELD.values():
        values[f"metric|exact_work|{metric}"] = float(exact_metrics[metric])
    for criterion_id, (observed, operator, required) in _criterion_values(metrics, criteria).items():
        values[f"criterion|{criterion_id}"] = _criterion_margin(observed, operator, required)
    return values


def _bootstrap_vector(
    samples: BlockSamples,
    *,
    blocks: Sequence[int],
    configurations: Sequence[str],
    repetitions: int,
    rng: random.Random,
) -> MedianVector:
    selected_blocks = [blocks[rng.randrange(len(blocks))] for _ in blocks]
    block_medians: Dict[str, Dict[str, List[float]]] = {
        configuration: {representation: [] for representation in _REPRESENTATIONS} for configuration in configurations
    }
    for block in selected_blocks:
        for configuration in configurations:
            indices = [rng.randrange(repetitions) for _ in range(repetitions)]
            for representation in _REPRESENTATIONS:
                values = samples[block][configuration][representation]
                block_medians[configuration][representation].append(_median([values[index] for index in indices]))
    return {
        configuration: {
            representation: _median(block_medians[configuration][representation]) for representation in _REPRESENTATIONS
        }
        for configuration in configurations
    }


def _simultaneous_intervals(
    observed: Mapping[str, float],
    bootstrap: Sequence[Mapping[str, float]],
    *,
    confidence: float,
    pair_count: int,
) -> Tuple[Dict[str, Tuple[float, float]], float]:
    if not bootstrap:
        raise DecisionFidelityError("simultaneous bootstrap requires at least one resample")
    scales: Dict[str, float] = {}
    constant_keys: Set[str] = set()
    for key in observed:
        values = [row[key] for row in bootstrap]
        if all(value == observed[key] for value in values):
            constant_keys.add(key)
            continue
        scale = statistics.stdev(values) if len(values) > 1 else 0.0
        if not scale > 0.0 or not math.isfinite(scale):
            raise DecisionFidelityError(
                f"bootstrap statistic {key!r} has zero variance but disagrees with its observation"
            )
        scales[key] = scale
    maxima = [
        max(abs((row[key] - observed[key]) / scales[key]) for key in scales)
        for row in bootstrap
    ] if scales else [0.0]
    critical_value = _percentile(maxima, confidence)
    intervals: Dict[str, Tuple[float, float]] = {}
    for key, value in observed.items():
        if key in constant_keys:
            intervals[key] = (value, value)
            continue
        lower = value - critical_value * scales[key]
        upper = value + critical_value * scales[key]
        if key == "metric|exact_work|pairwise_ranking_agreement":
            lower, upper = max(0.0, lower), min(1.0, upper)
        elif key == "metric|exact_work|kendall_tau_b":
            lower, upper = max(-1.0, lower), min(1.0, upper)
        elif key in {
            "metric|exact_work|false_negative_count",
            "metric|exact_work|false_positive_count",
        }:
            lower, upper = max(0.0, lower), min(float(pair_count), upper)
        intervals[key] = (lower, upper)
    return intervals, critical_value


def _relative_iqr(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint:] if len(ordered) % 2 == 0 else ordered[midpoint + 1 :]
    iqr = 0.0 if not lower or not upper else _median(upper) - _median(lower)
    median = _median(values)
    return math.inf if median == 0.0 and iqr > 0.0 else (0.0 if median == 0.0 else iqr / median * 100.0)


def evaluate_decision_fidelity_v2(
    aggregate: Mapping[str, Any],
    policy_bytes: bytes,
    *,
    executor_artifact: Optional[ExecutorArtifact] = None,
) -> Dict[str, Any]:
    """Evaluate complete allocation blocks under the immutable v2 method."""

    aggregate = _object(aggregate, "aggregate")
    if not isinstance(policy_bytes, bytes) or not 0 < len(policy_bytes) <= _POLICY_LIMITS.max_document_bytes:
        raise DecisionFidelityError("decision fidelity policy v2 bytes are outside the supported limit")
    policy = validate_decision_fidelity_policy_v2(strict_json_loads(policy_bytes, limits=_POLICY_LIMITS))
    policy_sha256 = sha256_hex(policy_bytes)
    if aggregate.get("schema") != ANALYSIS_SCHEMA:
        raise DecisionFidelityError("decision fidelity v2 requires a trusted validated aggregate")
    campaign = _policy_input_binding(aggregate, policy_sha256=policy_sha256, policy_size_bytes=len(policy_bytes))
    analyzer_record = _frozen_evaluator_record(
        aggregate,
        campaign,
        executor_artifact=executor_artifact,
        policy_sha256=policy_sha256,
    )
    completeness = _object(aggregate.get("completeness"), "aggregate.completeness")
    configurations = list(policy["scope"]["configuration_ids"])
    block_count = int(policy["measurement"]["allocation_blocks"])
    blocks = list(range(block_count))
    workload_id = str(policy["scope"]["workload_id"])
    selected = aggregate.get("selected_cells")
    if not isinstance(selected, list):
        raise DecisionFidelityError("aggregate selected cell inventory is invalid")
    issues: List[Dict[str, Any]] = []
    if completeness.get("complete") is not True or completeness.get("issue_codes") != []:
        issues.append({"code": "incomplete_evidence", "detail": "persisted completeness is not complete"})

    rows: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for index, raw_row in enumerate(selected):
        row = _object(raw_row, f"aggregate.selected_cells[{index}]")
        configuration = row.get("configuration_id")
        repetition = row.get("repetition")
        if (
            row.get("workload_id") != workload_id
            or row.get("measurement_schema") != PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2
            or not isinstance(configuration, str)
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or not 0 <= repetition < block_count
            or (repetition, configuration) in rows
        ):
            raise DecisionFidelityError("aggregate selected cell is outside the replicated decision gate")
        rows[(repetition, configuration)] = row
    expected_cells = {(block, configuration) for block in blocks for configuration in configurations}
    if set(rows) != expected_cells:
        issues.append(
            {
                "code": "incomplete_allocation_block_inventory",
                "detail": "selected cells do not cover every configuration in every allocation block",
            }
        )

    samples: BlockSamples = {}
    identities: Set[Tuple[str, str, str]] = set()
    environment_observations: Set[str] = set()
    jobs: List[str] = []
    nodes: Set[str] = set()
    repetitions = int(policy["measurement"]["measured_repetitions"])
    if not issues:
        for block in blocks:
            samples[block] = {}
            for configuration in configurations:
                row = rows[(block, configuration)]
                gate = _object(row.get("decision_gate"), f"block {block} configuration {configuration}.decision_gate")
                execution = _object(gate.get("execution"), f"block {block} configuration {configuration}.execution")
                if (
                    execution.get("allocation_block") != block
                    or execution.get("iterations") != repetitions
                    or execution.get("warmup") != policy["measurement"]["warmup"]
                    or execution.get("order_method") != policy["measurement"]["order_method"]
                    or execution.get("timing_semantics") != policy["measurement"]["timing_semantics"]
                ):
                    raise DecisionFidelityError("replicated decision-gate execution disagrees with policy")
                representations = _object(gate.get("representations"), "replicated decision-gate representations")
                samples[block][configuration] = {}
                for representation in _REPRESENTATIONS:
                    value = _object(representations.get(representation), f"representation {representation}")
                    timings = value.get("timings_us")
                    if not isinstance(timings, list):
                        raise DecisionFidelityError("replicated decision-gate timings are invalid")
                    parsed = tuple(_number(item, "replicated decision-gate timing", minimum=0.0) for item in timings)
                    if len(parsed) != repetitions:
                        raise DecisionFidelityError("replicated decision-gate timing inventory is incomplete")
                    samples[block][configuration][representation] = parsed
                if _median(samples[block][configuration]["source"]) <= 0.0:
                    issues.append(
                        {
                            "code": "nonpositive_source_timing",
                            "detail": f"source timing is not positive for block {block}, {configuration}",
                        }
                    )
                runtime = _object(row.get("decision_gate_runtime"), "replicated decision-gate runtime")
                job_id = runtime.get("job_id")
                hostname = runtime.get("hostname")
                if not isinstance(job_id, str) or not job_id or not isinstance(hostname, str) or not hostname:
                    raise DecisionFidelityError("replicated decision-gate job and hostname are required")
                jobs.append(job_id)
                nodes.add(hostname.split(".", 1)[0])
                environment_observations.add(
                    _environment_identity(
                        row.get("decision_gate_environment"),
                        f"block {block} configuration {configuration}.decision_gate_environment",
                    )
                )
                identities.add(
                    (
                        canonical_sha256(gate["request"]),
                        canonical_sha256(gate["materialization"]),
                        canonical_sha256(gate["policy"]),
                    )
                )
        if len(set(jobs)) != len(jobs):
            issues.append(
                {
                    "code": "allocation_job_reuse",
                    "detail": "every configuration/block cell must have a distinct scheduler job ID",
                }
            )
        if len(identities) != 1:
            issues.append({"code": "artifact_identity_mismatch", "detail": "cells used different immutable inputs"})
        if len(environment_observations) != len(rows):
            issues.append(
                {
                    "code": "environment_observation_reuse",
                    "detail": "every configuration/block cell must bind a distinct runtime observation",
                }
            )

    stability_issues: List[str] = []
    observed_pair_rows: List[Dict[str, Any]] = []
    observed_metrics: Dict[str, Any] = {}
    criteria_rows: List[Dict[str, Any]] = []
    metric_intervals: Dict[str, Any] = {}
    inconclusive_pairs: List[str] = []
    critical_value: Optional[float] = None
    if not issues:
        maximum_iqr = float(policy["measurement"]["max_relative_iqr_pct"])
        for block in blocks:
            for configuration in configurations:
                for representation in _STABILITY_REPRESENTATIONS:
                    relative_iqr = _relative_iqr(samples[block][configuration][representation])
                    if relative_iqr > maximum_iqr:
                        stability_issues.append(f"{block}|{configuration}|{representation}|{relative_iqr}")

        comparison = cast(Mapping[str, Any], policy["comparison"])
        observed_vector = _nested_median_vector(samples, blocks=blocks, configurations=configurations)
        observed_pair_rows = _pair_rows(
            observed_vector,
            configurations=configurations,
            comparison=comparison,
        )
        observed_targets = {
            (
                str(row["first_configuration_id"]),
                str(row["second_configuration_id"]),
                representation,
            ): str(cast(Mapping[str, Any], row["representations"])[representation]["observed_label"])
            for row in observed_pair_rows
            for representation in _REPRESENTATIONS
        }
        observed_metrics = _metrics(
            observed_vector,
            configurations=configurations,
            pair_rows=observed_pair_rows,
        )
        observed_statistics = _statistics(
            pair_rows=observed_pair_rows,
            metrics=observed_metrics,
            criteria=policy["pass_criteria"],
        )
        uncertainty = cast(Mapping[str, Any], comparison["uncertainty"])
        rng = random.Random(int(uncertainty["seed"]))
        bootstrap_statistics: List[Dict[str, float]] = []
        for _ in range(int(uncertainty["resamples"])):
            vector = _bootstrap_vector(
                samples,
                blocks=blocks,
                configurations=configurations,
                repetitions=repetitions,
                rng=rng,
            )
            pairs = _pair_rows(
                vector,
                configurations=configurations,
                comparison=comparison,
                targets=observed_targets,
            )
            metrics = _metrics(vector, configurations=configurations, pair_rows=pairs)
            bootstrap_statistics.append(_statistics(pair_rows=pairs, metrics=metrics, criteria=policy["pass_criteria"]))
        intervals, critical_value = _simultaneous_intervals(
            observed_statistics,
            bootstrap_statistics,
            confidence=float(uncertainty["confidence"]),
            pair_count=len(observed_pair_rows),
        )

        for row in observed_pair_rows:
            first = str(row["first_configuration_id"])
            second = str(row["second_configuration_id"])
            representation_rows = cast(Dict[str, Any], row["representations"])
            for representation, pair_result in representation_rows.items():
                if representation not in _UNCERTAINTY_REPRESENTATIONS:
                    pair_result["uncertainty_label"] = "not_evaluated"
                    pair_result["simultaneous_margin_interval_us"] = []
                    continue
                key = f"pair|{first}|{second}|{representation}"
                interval = intervals[key]
                pair_result["simultaneous_margin_interval_us"] = list(interval)
                if interval[0] > 0.0:
                    pair_result["uncertainty_label"] = pair_result["observed_label"]
                else:
                    pair_result["uncertainty_label"] = "inconclusive"
                    inconclusive_pairs.append(
                        f"{first}|{second}|{representation}|observed={pair_result['observed_label']}"
                    )

        metric_intervals["exact_work"] = {
            metric: list(intervals[f"metric|exact_work|{metric}"])
            for metric in REPORTED_METRIC_TO_VERDICT_FIELD.values()
        }
        for criterion_id, (observed, operator, required) in _criterion_values(
            observed_metrics,
            policy["pass_criteria"],
        ).items():
            margin_interval = intervals[f"criterion|{criterion_id}"]
            criteria_rows.append(
                {
                    "criterion_id": criterion_id,
                    "observed": observed,
                    "operator": operator,
                    "required": required,
                    "observed_margin": _criterion_margin(observed, operator, required),
                    "simultaneous_margin_interval": list(margin_interval),
                    "status": (
                        "pass"
                        if margin_interval[0] >= 0.0
                        else ("fail" if margin_interval[1] < 0.0 else "inconclusive")
                    ),
                }
            )

    criteria_inconclusive = any(row["status"] == "inconclusive" for row in criteria_rows)
    if issues:
        outcome = "incomparable"
    elif stability_issues or inconclusive_pairs or criteria_inconclusive:
        outcome = "inconclusive"
    elif any(row["status"] == "fail" for row in criteria_rows):
        outcome = "fail"
    else:
        outcome = "pass"
    positioning = (
        "exact_capsule_positive_control_supported"
        if outcome == "pass"
        else "exact_capsule_positive_control_unvalidated"
    )
    result: Dict[str, Any] = {
        "schema": DECISION_FIDELITY_VERDICT_SCHEMA_V2,
        "outcome": outcome,
        "policy": {
            "schema": policy["schema"],
            "policy_id": policy["policy_id"],
            "sha256": policy_sha256,
            "size_bytes": len(policy_bytes),
        },
        "evidence": {
            "aggregate_sha256": canonical_sha256(aggregate),
            "run_id": campaign.get("run_id"),
            "manifest_sha256": campaign.get("manifest_sha256"),
            "selection_sha256": campaign.get("selection_sha256"),
            "completeness_verdict_sha256": campaign.get("verdict_sha256"),
            "allocation_block_count": block_count,
            "configuration_count": len(configurations),
            "configuration_pair_count": len(observed_pair_rows),
            "distinct_job_count": len(set(jobs)),
            "environment_observation_count": len(environment_observations),
            "nodes": sorted(nodes),
        },
        "issues": issues,
        "uncertainty": {
            "status": (
                "not_evaluated"
                if issues
                else ("inconclusive" if stability_issues or inconclusive_pairs or criteria_inconclusive else "decisive")
            ),
            "method": policy["comparison"]["uncertainty"]["method"],
            "confidence": policy["comparison"]["uncertainty"]["confidence"],
            "resamples": policy["comparison"]["uncertainty"]["resamples"],
            "standardized_max_critical_value": critical_value,
            "inconclusive_pairs": inconclusive_pairs,
            "unstable_cells": stability_issues,
            "metric_intervals": metric_intervals,
        },
        "pairwise_comparisons": observed_pair_rows,
        "representation_metrics": observed_metrics,
        "criteria": criteria_rows,
        "product_interpretation": {
            **policy["claim_boundary"],
            "positioning": positioning,
        },
    }
    if analyzer_record is not None:
        result["analyzer"] = analyzer_record
    result["verdict_id"] = canonical_sha256(result)
    return result


__all__ = ["evaluate_decision_fidelity_v2", "validate_decision_fidelity_policy_v2"]
