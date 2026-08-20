"""Contracts for dependency-preserving physical decision canaries.

These artifacts deliberately separate three kinds of evidence:

* a byte-identified Chakra execution trace;
* an owner-supplied semantic projection over that trace; and
* measured application/candidate decisions across declared perturbations.

The projection can describe a narrow supported workload without pretending
that CommCanary can infer model semantics from opaque protobuf attributes.  A
corpus marked ``synthetic`` is useful for testing the synthesizer but can never
qualify a physical canary.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..errors import SchemaError
from ..formats import (
    CHAKRA_PROJECTION_FORMAT,
    PHYSICAL_CANARY_POLICY_FORMAT,
    PHYSICAL_GATE_OBSERVATION_FORMAT,
    PHYSICAL_GATE_RESULT_FORMAT,
    PHYSICAL_ORACLE_CORPUS_FORMAT,
)
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping
from .chakra import (
    ChakraExecutionTrace,
    chakra_dependency_closure,
    chakra_int64_attribute,
    encode_chakra_subgraph,
)
from .dtypes import dtype_size_bytes
from .json_codec import canonical_json_bytes

SUPPORTED_PHYSICAL_DOMAIN = "single-node-tensor-parallel-dense-decoder-all-reduce-gemm-overlap.v1"
SUPPORTED_PROJECTION_OPERATIONS = frozenset({"gemm", "all_reduce"})
SUPPORTED_CANDIDATE_METHODS = frozenset(
    {
        "reduced_decision_canary",
        "independent_exact_replay",
        "random_sampling",
        "stratified_sampling",
        "ddmin",
        "communication_only_microbenchmark",
    }
)
DISCLOSURE_CATEGORIES = frozenset(
    {
        "model_family",
        "hidden_dimension",
        "parallelism_degree",
        "batch_token_geometry",
        "topology",
    }
)
PHYSICAL_METRIC_FIELDS = frozenset(
    {
        "physical_runtime_seconds",
        "gpu_seconds",
        "gpu_count",
        "executed_collectives",
        "executed_flops",
        "peak_memory_bytes",
    }
)
PHYSICAL_CANARY_STATUSES = frozenset(
    {
        "blocked_missing_physical_oracle_corpus",
        "blocked_synthetic_evidence",
        "blocked_incomplete_baseline_corpus",
        "blocked_no_decision_preserving_training_candidate",
        "failed_held_out_validation",
        "failed_privacy_policy",
        "qualified_physical_decision_canary",
    }
)


def physical_policy_sha256(policy: Mapping[str, Any]) -> str:
    return _content_identity(policy, "policy_id")


def chakra_projection_sha256(projection: Mapping[str, Any]) -> str:
    return _content_identity(projection, "projection_id")


def physical_oracle_corpus_sha256(corpus: Mapping[str, Any]) -> str:
    return _content_identity(corpus, "corpus_id")


def with_content_identity(document: Mapping[str, Any], identity_field: str) -> Dict[str, Any]:
    """Copy a JSON object and set its self-excluding canonical SHA-256."""

    result = dict(document)
    result.pop(identity_field, None)
    result[identity_field] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def validate_physical_canary_policy(
    policy: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(policy, "physical canary policy", limits)
    _exact_fields(
        policy,
        {
            "format",
            "policy_id",
            "supported_domain",
            "runtime_budget_seconds",
            "severe_regression_threshold_pct",
            "required_feature_tags",
            "required_baselines",
            "runner",
            "qualification_gates",
            "gate_metrics",
            "minimum_samples",
            "uncertainty",
            "noise",
            "privacy",
        },
        "physical canary policy",
    )
    if policy.get("format") != PHYSICAL_CANARY_POLICY_FORMAT:
        raise SchemaError(f"physical canary policy format must be {PHYSICAL_CANARY_POLICY_FORMAT!r}")
    _validate_identity(policy, "policy_id", physical_policy_sha256, "physical canary policy")
    if policy.get("supported_domain") != SUPPORTED_PHYSICAL_DOMAIN:
        raise SchemaError("unsupported_for_physical_canary: reason: supported_domain_not_qualified")
    _positive_number(policy.get("runtime_budget_seconds"), "physical canary runtime_budget_seconds")
    _positive_number(
        policy.get("severe_regression_threshold_pct"),
        "physical canary severe_regression_threshold_pct",
    )
    _sorted_unique_strings(policy.get("required_feature_tags"), "physical canary required_feature_tags")
    required_baselines = _sorted_unique_strings(
        policy.get("required_baselines"),
        "physical canary required_baselines",
    )
    unknown_baselines = sorted(set(required_baselines).difference(SUPPORTED_CANDIDATE_METHODS))
    if unknown_baselines or "reduced_decision_canary" in required_baselines:
        raise SchemaError("physical canary required_baselines contains an unsupported baseline method")

    runner = _mapping(policy.get("runner"), "physical canary runner")
    _exact_fields(
        runner,
        {"oci_digest", "execution_protocol", "world_size"},
        "physical canary runner",
    )
    _oci_digest(runner.get("oci_digest"), "physical canary runner.oci_digest")
    if runner.get("execution_protocol") != "chakra-et-collective-graph.v1":
        raise SchemaError("physical canary runner.execution_protocol is unsupported")
    _bounded_int(
        runner.get("world_size"),
        "physical canary runner.world_size",
        2,
        65_536,
    )

    gates = _mapping(policy.get("qualification_gates"), "physical canary qualification_gates")
    _exact_fields(
        gates,
        {
            "minimum_runtime_reduction_ratio",
            "maximum_severe_false_negatives",
            "maximum_false_positive_rate",
            "minimum_pairwise_decision_agreement",
        },
        "physical canary qualification_gates",
    )
    minimum_reduction = _positive_number(
        gates.get("minimum_runtime_reduction_ratio"),
        "physical canary minimum_runtime_reduction_ratio",
    )
    if minimum_reduction < 1.0:
        raise SchemaError("physical canary minimum_runtime_reduction_ratio must be at least 1")
    _bounded_int(
        gates.get("maximum_severe_false_negatives"),
        "physical canary maximum_severe_false_negatives",
        0,
        limits.max_physical_perturbations,
    )
    _unit_interval(
        gates.get("maximum_false_positive_rate"),
        "physical canary maximum_false_positive_rate",
    )
    _unit_interval(
        gates.get("minimum_pairwise_decision_agreement"),
        "physical canary minimum_pairwise_decision_agreement",
    )

    metrics = policy.get("gate_metrics")
    if not isinstance(metrics, list) or not metrics:
        raise SchemaError("physical canary gate_metrics must be a non-empty array")
    metric_names: Set[str] = set()
    for index, raw_metric in enumerate(metrics):
        metric = _mapping(raw_metric, f"physical canary gate_metrics[{index}]")
        _exact_fields(
            metric,
            {"name", "direction", "regression_threshold_pct", "mandatory"},
            f"physical canary gate_metrics[{index}]",
        )
        name = _nonempty_string(metric.get("name"), f"physical canary gate_metrics[{index}].name")
        if name in metric_names:
            raise SchemaError(f"physical canary gate_metrics repeats name {name!r}")
        metric_names.add(name)
        if metric.get("direction") not in {"lower_is_better", "higher_is_better"}:
            raise SchemaError(f"physical canary gate_metrics[{index}].direction is unsupported")
        _nonnegative_number(
            metric.get("regression_threshold_pct"),
            f"physical canary gate_metrics[{index}].regression_threshold_pct",
        )
        _boolean(metric.get("mandatory"), f"physical canary gate_metrics[{index}].mandatory")
    if not any(metric.get("mandatory") is True for metric in metrics):
        raise SchemaError("physical canary gate_metrics must contain at least one mandatory metric")

    # A gate that cannot express uncertainty reports an unresolvable difference
    # as a pass, so the replicate floor, the stability bound, and the interval
    # method are predeclared in the policy and bound to its identity.
    _bounded_int(policy.get("minimum_samples"), "physical canary minimum_samples", 2, 1_000_000)

    uncertainty = _mapping(policy.get("uncertainty"), "physical canary uncertainty")
    _exact_fields(
        uncertainty,
        {"method", "confidence", "bootstrap_resamples", "seed"},
        "physical canary uncertainty",
    )
    if uncertainty.get("method") != "percentile_bootstrap_median_difference":
        raise SchemaError("physical canary uncertainty.method is unsupported")
    confidence = _nonnegative_number(uncertainty.get("confidence"), "physical canary uncertainty.confidence")
    if not 0.5 <= confidence < 1.0:
        raise SchemaError("physical canary uncertainty.confidence must be at least 0.5 and below 1")
    _bounded_int(
        uncertainty.get("bootstrap_resamples"),
        "physical canary uncertainty.bootstrap_resamples",
        100,
        100_000,
    )
    _bounded_int(uncertainty.get("seed"), "physical canary uncertainty.seed", 0, (1 << 63) - 1)

    noise = _mapping(policy.get("noise"), "physical canary noise")
    _exact_fields(noise, {"max_relative_iqr_pct"}, "physical canary noise")
    _nonnegative_number(noise.get("max_relative_iqr_pct"), "physical canary noise.max_relative_iqr_pct")

    privacy = _mapping(policy.get("privacy"), "physical canary privacy")
    _exact_fields(
        privacy,
        {"maximum_leakage_score", "private_exchange_requires_reviewed_opaque_attributes"},
        "physical canary privacy",
    )
    _unit_interval(privacy.get("maximum_leakage_score"), "physical canary maximum_leakage_score")
    if privacy.get("private_exchange_requires_reviewed_opaque_attributes") is not True:
        raise SchemaError("physical canary private_exchange_requires_reviewed_opaque_attributes must be true")


def validate_chakra_projection(
    projection: Mapping[str, Any],
    trace: ChakraExecutionTrace,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(projection, "Chakra projection", limits)
    _exact_fields(
        projection,
        {
            "format",
            "projection_id",
            "source_et",
            "capture_provenance",
            "supported_domain",
            "nodes",
            "regions",
            "privacy_review",
        },
        "Chakra projection",
    )
    if projection.get("format") != CHAKRA_PROJECTION_FORMAT:
        raise SchemaError(f"Chakra projection format must be {CHAKRA_PROJECTION_FORMAT!r}")
    _validate_identity(projection, "projection_id", chakra_projection_sha256, "Chakra projection")
    source = _mapping(projection.get("source_et"), "Chakra projection source_et")
    _exact_fields(
        source,
        {"sha256", "bytes", "metadata_version"},
        "Chakra projection source_et",
    )
    _sha256(source.get("sha256"), "Chakra projection source_et.sha256")
    if source.get("sha256") != trace.source_sha256 or source.get("bytes") != trace.source_bytes:
        raise SchemaError("Chakra projection source_et does not identify the exact supplied ET bytes")
    if source.get("metadata_version") != trace.metadata_version:
        raise SchemaError("Chakra projection source_et metadata_version mismatch")
    provenance = _mapping(projection.get("capture_provenance"), "Chakra projection capture_provenance")
    _exact_fields(
        provenance,
        {"adapter", "source_format", "source_sha256"},
        "Chakra projection capture_provenance",
    )
    _nonempty_string(provenance.get("adapter"), "Chakra projection capture_provenance.adapter")
    _nonempty_string(provenance.get("source_format"), "Chakra projection capture_provenance.source_format")
    _sha256(provenance.get("source_sha256"), "Chakra projection capture_provenance.source_sha256")
    if projection.get("supported_domain") != SUPPORTED_PHYSICAL_DOMAIN:
        raise SchemaError("unsupported_for_physical_canary: reason: supported_domain_not_qualified")

    raw_nodes = projection.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SchemaError("Chakra projection nodes must be a non-empty array")
    if len(raw_nodes) > limits.max_chakra_messages - 1:
        raise SchemaError("Chakra projection node count exceeds the Chakra message policy")
    projected_node_ids: List[int] = []
    node_features: Dict[int, Tuple[str, ...]] = {}
    node_disclosures: Dict[int, Tuple[str, ...]] = {}
    trace_by_id = {node.node_id: node for node in trace.nodes}
    for index, raw_node in enumerate(raw_nodes):
        node = _mapping(raw_node, f"Chakra projection nodes[{index}]")
        _exact_fields(
            node,
            {
                "node_id",
                "chakra_node_type",
                "operation",
                "executed_flops",
                "tensor_bytes",
                "phase",
                "feature_tags",
                "disclosures",
                "execution",
            },
            f"Chakra projection nodes[{index}]",
        )
        node_id = _bounded_int(
            node.get("node_id"),
            f"Chakra projection nodes[{index}].node_id",
            1,
            (1 << 64) - 1,
        )
        projected_node_ids.append(node_id)
        source_node = trace_by_id.get(node_id)
        if source_node is None:
            raise SchemaError(f"Chakra projection references absent Node.id={node_id}")
        if node.get("chakra_node_type") != source_node.node_type:
            raise SchemaError(f"Chakra projection Node.id={node_id} type mismatch")
        operation = node.get("operation")
        if operation not in SUPPORTED_PROJECTION_OPERATIONS:
            reason = str(operation).replace(" ", "_")
            raise SchemaError(f"unsupported_for_physical_canary: reason: {reason}_not_qualified")
        expected_type = 4 if operation == "gemm" else 7
        if source_node.node_type != expected_type:
            raise SchemaError(
                f"Chakra projection Node.id={node_id} operation {operation!r} conflicts with Chakra node type"
            )
        executed_flops = _bounded_int(
            node.get("executed_flops"),
            f"Chakra projection nodes[{index}].executed_flops",
            0,
            (1 << 63) - 1,
        )
        tensor_bytes = _bounded_int(
            node.get("tensor_bytes"),
            f"Chakra projection nodes[{index}].tensor_bytes",
            0,
            (1 << 63) - 1,
        )
        if operation == "gemm":
            if executed_flops == 0 or tensor_bytes == 0:
                raise SchemaError(
                    f"Chakra projection GEMM Node.id={node_id} must declare positive work and tensor bytes"
                )
            _validate_gemm_execution(
                node.get("execution"),
                node_id=node_id,
                executed_flops=executed_flops,
                tensor_bytes=tensor_bytes,
            )
        else:
            if executed_flops != 0 or tensor_bytes == 0:
                raise SchemaError(
                    f"Chakra projection all-reduce Node.id={node_id} must declare zero FLOPs and positive bytes"
                )
            comm_type = chakra_int64_attribute(source_node, "comm_type")
            comm_size = chakra_int64_attribute(source_node, "comm_size")
            if comm_type != 0:
                raise SchemaError(f"unsupported_for_physical_canary: reason: collective_type_{comm_type}_not_qualified")
            if comm_size != tensor_bytes:
                raise SchemaError(f"Chakra projection Node.id={node_id} tensor_bytes does not match Chakra comm_size")
            _validate_all_reduce_execution(
                node.get("execution"),
                node_id=node_id,
                tensor_bytes=tensor_bytes,
            )
        _nonempty_string(node.get("phase"), f"Chakra projection nodes[{index}].phase")
        node_features[node_id] = _sorted_unique_strings(
            node.get("feature_tags"),
            f"Chakra projection nodes[{index}].feature_tags",
        )
        node_disclosures[node_id] = _disclosures(
            node.get("disclosures"),
            f"Chakra projection nodes[{index}].disclosures",
        )
    if projected_node_ids != list(trace.node_ids):
        raise SchemaError("Chakra projection nodes must cover every source Node exactly in source order")

    raw_regions = projection.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise SchemaError("Chakra projection regions must be a non-empty array")
    if len(raw_regions) > limits.max_physical_regions:
        raise SchemaError(f"Chakra projection regions exceed max_physical_regions={limits.max_physical_regions}")
    region_ids: Set[str] = set()
    source_order = {node_id: index for index, node_id in enumerate(trace.node_ids)}
    for index, raw_region in enumerate(raw_regions):
        region = _mapping(raw_region, f"Chakra projection regions[{index}]")
        _exact_fields(
            region,
            {"region_id", "node_ids", "feature_tags", "disclosures"},
            f"Chakra projection regions[{index}]",
        )
        region_id = _nonempty_string(
            region.get("region_id"),
            f"Chakra projection regions[{index}].region_id",
        )
        if region_id in region_ids:
            raise SchemaError(f"Chakra projection repeats region_id={region_id!r}")
        region_ids.add(region_id)
        node_ids = _unique_positive_ints(
            region.get("node_ids"),
            f"Chakra projection regions[{index}].node_ids",
        )
        if tuple(sorted(node_ids, key=source_order.__getitem__)) != node_ids:
            raise SchemaError(f"Chakra projection region {region_id!r} node_ids must use source order")
        closure = chakra_dependency_closure(trace, node_ids)
        if closure != node_ids:
            missing = [node_id for node_id in closure if node_id not in set(node_ids)]
            raise SchemaError(
                f"Chakra projection region {region_id!r} is not dependency closed; missing Node IDs {missing[:10]}"
            )
        expected_features = tuple(sorted({tag for node_id in node_ids for tag in node_features[node_id]}))
        features = _sorted_unique_strings(
            region.get("feature_tags"),
            f"Chakra projection regions[{index}].feature_tags",
        )
        if features != expected_features:
            raise SchemaError(f"Chakra projection region {region_id!r} feature_tags do not match its nodes")
        expected_disclosures = tuple(sorted({item for node_id in node_ids for item in node_disclosures[node_id]}))
        disclosures = _disclosures(
            region.get("disclosures"),
            f"Chakra projection regions[{index}].disclosures",
        )
        if disclosures != expected_disclosures:
            raise SchemaError(f"Chakra projection region {region_id!r} disclosures do not match its nodes")

    privacy = _mapping(projection.get("privacy_review"), "Chakra projection privacy_review")
    _exact_fields(
        privacy,
        {"opaque_chakra_attributes_reviewed", "reviewed_disclosures"},
        "Chakra projection privacy_review",
    )
    _boolean(
        privacy.get("opaque_chakra_attributes_reviewed"),
        "Chakra projection opaque_chakra_attributes_reviewed",
    )
    reviewed = _disclosures(
        privacy.get("reviewed_disclosures"),
        "Chakra projection reviewed_disclosures",
    )
    all_disclosures = tuple(sorted({item for values in node_disclosures.values() for item in values}))
    if reviewed != all_disclosures:
        raise SchemaError("Chakra projection reviewed_disclosures do not match projected node disclosures")


def validate_physical_oracle_corpus(
    corpus: Mapping[str, Any],
    trace: ChakraExecutionTrace,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(corpus, "physical oracle corpus", limits)
    _exact_fields(
        corpus,
        {
            "format",
            "corpus_id",
            "source_et_sha256",
            "projection_id",
            "policy_id",
            "runner_oci_digest",
            "baseline_subject_sha256",
            "evidence_kind",
            "application",
            "perturbations",
            "evaluations",
        },
        "physical oracle corpus",
    )
    if corpus.get("format") != PHYSICAL_ORACLE_CORPUS_FORMAT:
        raise SchemaError(f"physical oracle corpus format must be {PHYSICAL_ORACLE_CORPUS_FORMAT!r}")
    _validate_identity(corpus, "corpus_id", physical_oracle_corpus_sha256, "physical oracle corpus")
    _sha256(corpus.get("source_et_sha256"), "physical oracle corpus source_et_sha256")
    source = _mapping(projection.get("source_et"), "Chakra projection source_et")
    if corpus.get("source_et_sha256") != source.get("sha256"):
        raise SchemaError("physical oracle corpus source_et_sha256 does not match the projection")
    if corpus.get("projection_id") != projection.get("projection_id"):
        raise SchemaError("physical oracle corpus projection_id mismatch")
    if corpus.get("policy_id") != policy.get("policy_id"):
        raise SchemaError("physical oracle corpus policy_id mismatch")
    _oci_digest(corpus.get("runner_oci_digest"), "physical oracle corpus runner_oci_digest")
    if corpus.get("runner_oci_digest") != policy["runner"]["oci_digest"]:
        raise SchemaError("physical oracle corpus runner_oci_digest does not match the policy")
    _sha256(corpus.get("baseline_subject_sha256"), "physical oracle corpus baseline_subject_sha256")
    if corpus.get("evidence_kind") not in {"measured", "synthetic"}:
        raise SchemaError("physical oracle corpus evidence_kind must be 'measured' or 'synthetic'")

    application = _mapping(corpus.get("application"), "physical oracle corpus application")
    _exact_fields(
        application,
        {"ground_truth_kind", "name", "engine", "artifact_sha256"},
        "physical oracle corpus application",
    )
    if application.get("ground_truth_kind") != "application_ground_truth":
        raise SchemaError("physical oracle corpus ground truth must come from the actual application")
    _nonempty_string(application.get("name"), "physical oracle corpus application.name")
    _nonempty_string(application.get("engine"), "physical oracle corpus application.engine")
    _sha256(application.get("artifact_sha256"), "physical oracle corpus application.artifact_sha256")

    perturbations = corpus.get("perturbations")
    if not isinstance(perturbations, list) or len(perturbations) < 2:
        raise SchemaError("physical oracle corpus requires at least two perturbations")
    if len(perturbations) > limits.max_physical_perturbations:
        raise SchemaError(
            "physical oracle corpus perturbations exceed "
            f"max_physical_perturbations={limits.max_physical_perturbations}"
        )
    perturbation_ids: Set[str] = set()
    splits: Set[str] = set()
    severe_threshold = float(policy["severe_regression_threshold_pct"])
    for index, raw_perturbation in enumerate(perturbations):
        perturbation = _mapping(raw_perturbation, f"physical oracle corpus perturbations[{index}]")
        _exact_fields(
            perturbation,
            {
                "perturbation_id",
                "split",
                "application_decision",
                "regression_magnitude_pct",
                "severe",
                "application_metrics",
                "application_evidence_sha256",
                "environment_sha256",
                "candidate_subject_sha256",
            },
            f"physical oracle corpus perturbations[{index}]",
        )
        perturbation_id = _nonempty_string(
            perturbation.get("perturbation_id"),
            f"physical oracle corpus perturbations[{index}].perturbation_id",
        )
        if perturbation_id in perturbation_ids:
            raise SchemaError(f"physical oracle corpus repeats perturbation_id={perturbation_id!r}")
        perturbation_ids.add(perturbation_id)
        split = perturbation.get("split")
        if split not in {"training", "holdout"}:
            raise SchemaError(f"physical oracle corpus perturbation {perturbation_id!r} split is unsupported")
        splits.add(str(split))
        if perturbation.get("application_decision") not in {"pass", "fail"}:
            raise SchemaError(
                f"physical oracle corpus perturbation {perturbation_id!r} application_decision is unsupported"
            )
        magnitude = _nonnegative_number(
            perturbation.get("regression_magnitude_pct"),
            f"physical oracle corpus perturbation {perturbation_id!r} regression_magnitude_pct",
        )
        severe = _boolean(
            perturbation.get("severe"),
            f"physical oracle corpus perturbation {perturbation_id!r} severe",
        )
        expected_severe = perturbation.get("application_decision") == "fail" and magnitude >= severe_threshold
        if severe is not expected_severe:
            raise SchemaError(
                f"physical oracle corpus perturbation {perturbation_id!r} severe flag disagrees with policy"
            )
        _sha256(
            perturbation.get("application_evidence_sha256"),
            f"physical oracle corpus perturbation {perturbation_id!r} application_evidence_sha256",
        )
        _sha256(
            perturbation.get("environment_sha256"),
            f"physical oracle corpus perturbation {perturbation_id!r} environment_sha256",
        )
        _sha256(
            perturbation.get("candidate_subject_sha256"),
            f"physical oracle corpus perturbation {perturbation_id!r} candidate_subject_sha256",
        )
        if perturbation.get("candidate_subject_sha256") == corpus.get("baseline_subject_sha256"):
            raise SchemaError(
                f"physical oracle corpus perturbation {perturbation_id!r} does not change the stack subject"
            )
        _physical_metrics(
            perturbation.get("application_metrics"),
            f"physical oracle corpus perturbation {perturbation_id!r} application_metrics",
            world_size=int(policy["runner"]["world_size"]),
        )
    if splits != {"training", "holdout"}:
        raise SchemaError("physical oracle corpus requires both training and holdout perturbations")

    regions = projection.get("regions")
    if not isinstance(regions, list):
        raise SchemaError("Chakra projection regions must be an array")
    region_ids = {str(region["region_id"]) for region in regions}
    regions_by_id = {str(region["region_id"]): region for region in regions}
    projection_nodes = projection.get("nodes")
    if not isinstance(projection_nodes, list):
        raise SchemaError("Chakra projection nodes must be an array")
    nodes_by_id = {int(node["node_id"]): node for node in projection_nodes}
    evaluations = corpus.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise SchemaError("physical oracle corpus evaluations must be a non-empty array")
    if len(evaluations) > limits.max_physical_candidate_evaluations:
        raise SchemaError(
            "physical oracle corpus evaluations exceed "
            f"max_physical_candidate_evaluations={limits.max_physical_candidate_evaluations}"
        )
    candidate_ids: Set[str] = set()
    candidate_node_sets: Set[Tuple[int, ...]] = set()
    methods: Set[str] = set()
    for index, raw_evaluation in enumerate(evaluations):
        evaluation = _mapping(raw_evaluation, f"physical oracle corpus evaluations[{index}]")
        _exact_fields(
            evaluation,
            {"candidate_id", "method", "selected_region_ids", "executable_sha256", "observations"},
            f"physical oracle corpus evaluations[{index}]",
        )
        candidate_id = _nonempty_string(
            evaluation.get("candidate_id"),
            f"physical oracle corpus evaluations[{index}].candidate_id",
        )
        if candidate_id in candidate_ids:
            raise SchemaError(f"physical oracle corpus repeats candidate_id={candidate_id!r}")
        candidate_ids.add(candidate_id)
        method = evaluation.get("method")
        if method not in SUPPORTED_CANDIDATE_METHODS:
            raise SchemaError(f"physical oracle corpus candidate {candidate_id!r} method is unsupported")
        methods.add(str(method))
        _sha256(
            evaluation.get("executable_sha256"),
            f"physical oracle corpus candidate {candidate_id!r} executable_sha256",
        )
        selected = _sorted_unique_strings(
            evaluation.get("selected_region_ids"),
            f"physical oracle corpus candidate {candidate_id!r} selected_region_ids",
            require_nonempty=True,
        )
        unknown = sorted(set(selected).difference(region_ids))
        if unknown:
            raise SchemaError(
                f"physical oracle corpus candidate {candidate_id!r} references unknown regions {unknown[:10]}"
            )
        if method == "communication_only_microbenchmark":
            selected_node_ids = {int(regions_by_id[region_id]["node_ids"][0]) for region_id in selected}
            if any(nodes_by_id[node_id]["operation"] != "all_reduce" for node_id in selected_node_ids):
                raise SchemaError(
                    f"physical oracle corpus candidate {candidate_id!r} communication-only selection is not collective"
                )
        else:
            selected_node_ids = {
                int(node_id) for region_id in selected for node_id in regions_by_id[region_id]["node_ids"]
            }
        if method == "reduced_decision_canary":
            physical_node_set = tuple(sorted(selected_node_ids))
            if len(physical_node_set) >= len(trace.nodes):
                raise SchemaError(
                    f"physical oracle corpus candidate {candidate_id!r} does not execute fewer "
                    "Chakra nodes than its source"
                )
            if physical_node_set in candidate_node_sets:
                raise SchemaError("physical oracle corpus repeats a reduced candidate executable node set")
            candidate_node_sets.add(physical_node_set)
            for region_id in selected:
                other_node_ids = {
                    int(node_id)
                    for other_region_id in selected
                    if other_region_id != region_id
                    for node_id in regions_by_id[other_region_id]["node_ids"]
                }
                if set(regions_by_id[region_id]["node_ids"]).issubset(other_node_ids):
                    raise SchemaError(
                        f"physical oracle corpus candidate {candidate_id!r} contains redundant region {region_id!r}"
                    )
        candidate_et, closure = encode_chakra_subgraph(trace, selected_node_ids)
        if method == "communication_only_microbenchmark" and set(closure) != selected_node_ids:
            raise SchemaError(
                f"physical oracle corpus candidate {candidate_id!r} communication-only graph has non-collective dependencies"
            )
        if evaluation.get("executable_sha256") != hashlib.sha256(candidate_et).hexdigest():
            raise SchemaError(
                f"physical oracle corpus candidate {candidate_id!r} executable_sha256 does not match its Chakra ET"
            )
        expected_collectives = sum(nodes_by_id[node_id]["operation"] == "all_reduce" for node_id in selected_node_ids)
        expected_flops = sum(int(nodes_by_id[node_id]["executed_flops"]) for node_id in selected_node_ids)
        _evaluation_observations(
            evaluation.get("observations"),
            candidate_id=candidate_id,
            perturbation_ids=perturbation_ids,
            expected_collectives=expected_collectives,
            expected_flops=expected_flops,
            world_size=int(policy["runner"]["world_size"]),
        )
    if "reduced_decision_canary" not in methods:
        raise SchemaError("physical oracle corpus contains no reduced decision canary candidates")


def leakage_assessment(
    projection: Mapping[str, Any],
    selected_region_ids: Sequence[str],
) -> Dict[str, Any]:
    """Return a deterministic declared-disclosure assessment for one selection."""

    regions = projection.get("regions")
    if not isinstance(regions, list):
        raise SchemaError("Chakra projection regions must be an array")
    by_id = {str(region["region_id"]): region for region in regions}
    selected = set(selected_region_ids)
    unknown = sorted(selected.difference(by_id))
    if unknown:
        raise SchemaError(f"leakage assessment references unknown regions {unknown[:10]}")
    declared_disclosures = sorted(
        {str(disclosure) for region_id in selected for disclosure in by_id[region_id].get("disclosures", [])}
    )
    selected_node_ids = {int(node_id) for region_id in selected for node_id in by_id[region_id].get("node_ids", [])}
    probes = _leakage_inference_probes(projection, selected_node_ids)
    inferred_disclosures = {
        category
        for category, probe in probes.items()
        if probe["status"] in {"exact_value_exposed", "candidate_values_exposed"}
    }
    disclosures = sorted(set(declared_disclosures).union(inferred_disclosures))
    privacy = _mapping(projection.get("privacy_review"), "Chakra projection privacy_review")
    return {
        "method": "deterministic-leakage-probes.v1",
        "categories": sorted(DISCLOSURE_CATEGORIES),
        "declared_disclosed_categories": declared_disclosures,
        "inferred_disclosed_categories": sorted(inferred_disclosures),
        "disclosed_categories": disclosures,
        "leakage_score": len(disclosures) / len(DISCLOSURE_CATEGORIES),
        "inference_probes": probes,
        "opaque_chakra_attributes_reviewed": privacy.get("opaque_chakra_attributes_reviewed") is True,
        "limitations": [
            "Deterministic probes expose inferable candidates; they are not information-theoretic privacy bounds.",
            "Opaque Chakra attributes require a separate owner review.",
        ],
    }


def _leakage_inference_probes(
    projection: Mapping[str, Any],
    selected_node_ids: Set[int],
) -> Dict[str, Dict[str, Any]]:
    raw_nodes = projection.get("nodes")
    if not isinstance(raw_nodes, list):
        raise SchemaError("Chakra projection nodes must be an array")
    nodes = [node for node in raw_nodes if int(node["node_id"]) in selected_node_ids]
    rank_counts: Set[int] = set()
    message_sizes: Set[int] = set()
    hidden_candidates: Set[int] = set()
    token_candidates: Set[int] = set()
    for node in nodes:
        execution = _mapping(node.get("execution"), "leakage probe execution")
        if node.get("operation") == "all_reduce":
            ranks = execution.get("ranks")
            if isinstance(ranks, list):
                rank_counts.add(len(ranks))
            message_sizes.add(int(node["tensor_bytes"]))
        elif node.get("operation") == "gemm":
            recipes = execution.get("rank_recipes")
            if isinstance(recipes, list):
                for recipe in recipes:
                    if not isinstance(recipe, Mapping):
                        continue
                    token_candidates.add(int(recipe["m"]))
                    hidden_candidates.update((int(recipe["n"]), int(recipe["k"])))
    feature_projection = {
        "rank_counts": sorted(rank_counts),
        "collective_message_bytes": sorted(message_sizes),
        "gemm_hidden_axis_candidates": sorted(hidden_candidates),
        "gemm_batch_token_axis_candidates": sorted(token_candidates),
    }
    feature_sha256 = hashlib.sha256(canonical_json_bytes(feature_projection)).hexdigest()

    def probe(
        status: str,
        confidence: float,
        candidates: Sequence[Any],
        evidence: str,
    ) -> Dict[str, Any]:
        values = list(candidates)
        return {
            "status": status,
            "confidence": confidence,
            "candidate_count": len(values),
            "candidates": values[:32],
            "evidence": evidence,
            "feature_projection_sha256": feature_sha256,
        }

    return {
        "model_family": probe(
            "not_inferred",
            0.0,
            [],
            "No model-family classifier is qualified from this narrow projection.",
        ),
        "hidden_dimension": probe(
            "candidate_values_exposed" if hidden_candidates else "not_inferred",
            0.75 if hidden_candidates else 0.0,
            sorted(hidden_candidates),
            "Exact GEMM n/k axes can reveal or constrain hidden dimensions.",
        ),
        "parallelism_degree": probe(
            "exact_value_exposed"
            if len(rank_counts) == 1
            else "candidate_values_exposed"
            if rank_counts
            else "not_inferred",
            1.0 if len(rank_counts) == 1 else 0.5 if rank_counts else 0.0,
            sorted(rank_counts),
            "Collective rank lists expose tensor-parallel group cardinality.",
        ),
        "batch_token_geometry": probe(
            "candidate_values_exposed" if token_candidates else "not_inferred",
            0.6 if token_candidates else 0.0,
            sorted(token_candidates),
            "Exact GEMM m axes constrain batch and token geometry.",
        ),
        "topology": probe(
            "not_inferred",
            0.0,
            [],
            "Logical ranks alone do not establish physical interconnect topology.",
        ),
    }


def validate_physical_gate_observation(
    observation: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(observation, "physical gate observation", limits)
    _exact_fields(
        observation,
        {
            "format",
            "bundle_id",
            "role",
            "subject_sha256",
            "environment_sha256",
            "runner_oci_digest",
            "executable_sha256",
            "evidence_sha256",
            "samples",
        },
        "physical gate observation",
    )
    if observation.get("format") != PHYSICAL_GATE_OBSERVATION_FORMAT:
        raise SchemaError(f"physical gate observation format must be {PHYSICAL_GATE_OBSERVATION_FORMAT!r}")
    _sha256(observation.get("bundle_id"), "physical gate observation bundle_id")
    if observation.get("role") not in {"baseline", "candidate"}:
        raise SchemaError("physical gate observation role must be 'baseline' or 'candidate'")
    _sha256(observation.get("subject_sha256"), "physical gate observation subject_sha256")
    _sha256(observation.get("environment_sha256"), "physical gate observation environment_sha256")
    _oci_digest(
        observation.get("runner_oci_digest"),
        "physical gate observation runner_oci_digest",
    )
    if observation.get("runner_oci_digest") != policy["runner"]["oci_digest"]:
        raise SchemaError("physical gate observation runner_oci_digest does not match the policy")
    _sha256(observation.get("executable_sha256"), "physical gate observation executable_sha256")
    _sha256(observation.get("evidence_sha256"), "physical gate observation evidence_sha256")
    samples = observation.get("samples")
    if not isinstance(samples, Mapping):
        raise SchemaError("physical gate observation samples must be an object")
    expected = {str(metric["name"]) for metric in policy["gate_metrics"]}
    if set(samples) != expected:
        raise SchemaError("physical gate observation samples must cover exactly the policy metrics")
    total = 0
    for name, values in samples.items():
        if not isinstance(values, list) or not values:
            raise SchemaError(f"physical gate observation metric {name!r} must have non-empty samples")
        total += len(values)
        if total > limits.max_execution_observation_samples:
            raise SchemaError(
                "physical gate observation samples exceed "
                f"max_execution_observation_samples={limits.max_execution_observation_samples}"
            )
        for index, value in enumerate(values):
            _nonnegative_number(value, f"physical gate observation samples.{name}[{index}]")


def validate_physical_gate_result(
    result: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    _resource_check(result, "physical gate result", limits)
    _exact_fields(
        result,
        {
            "format",
            "bundle_id",
            "runner_oci_digest",
            "executable_sha256",
            "baseline_subject_sha256",
            "candidate_subject_sha256",
            "baseline_environment_sha256",
            "candidate_environment_sha256",
            "baseline_evidence_sha256",
            "candidate_evidence_sha256",
            "outcome",
            "issues",
            "metric_results",
        },
        "physical gate result",
    )
    if result.get("format") != PHYSICAL_GATE_RESULT_FORMAT:
        raise SchemaError(f"physical gate result format must be {PHYSICAL_GATE_RESULT_FORMAT!r}")
    _sha256(result.get("bundle_id"), "physical gate result bundle_id")
    _oci_digest(result.get("runner_oci_digest"), "physical gate result runner_oci_digest")
    _sha256(result.get("executable_sha256"), "physical gate result executable_sha256")
    _sha256(result.get("baseline_subject_sha256"), "physical gate result baseline_subject_sha256")
    _sha256(result.get("candidate_subject_sha256"), "physical gate result candidate_subject_sha256")
    _sha256(
        result.get("baseline_environment_sha256"),
        "physical gate result baseline_environment_sha256",
    )
    _sha256(
        result.get("candidate_environment_sha256"),
        "physical gate result candidate_environment_sha256",
    )
    _sha256(result.get("baseline_evidence_sha256"), "physical gate result baseline_evidence_sha256")
    _sha256(result.get("candidate_evidence_sha256"), "physical gate result candidate_evidence_sha256")
    if result.get("baseline_subject_sha256") == result.get("candidate_subject_sha256"):
        raise SchemaError("physical gate result baseline and candidate subjects must differ")
    if result.get("baseline_evidence_sha256") == result.get("candidate_evidence_sha256"):
        raise SchemaError("physical gate result baseline and candidate evidence must differ")
    if result.get("outcome") not in {"pass", "fail", "inconclusive", "incomparable"}:
        raise SchemaError("physical gate result outcome is unsupported")
    _sorted_unique_strings(result.get("issues"), "physical gate result issues")
    rows = result.get("metric_results")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("physical gate result metric_results must be a non-empty array")
    if len(rows) > limits.max_execution_observation_samples:
        raise SchemaError(
            "physical gate result metrics exceed "
            f"max_execution_observation_samples={limits.max_execution_observation_samples}"
        )

    metric_names: Set[str] = set()
    expected_metric_issues: Set[str] = set()
    screened_metrics: Set[str] = set()
    has_mandatory_failure = False
    has_mandatory_inconclusive = False
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"physical gate result metric_results[{index}]")
        _exact_fields(
            row,
            {
                "name",
                "direction",
                "mandatory",
                "baseline_median",
                "candidate_median",
                "regression_pct",
                "regression_threshold_pct",
                "samples_considered",
                "baseline_relative_iqr_pct",
                "candidate_relative_iqr_pct",
                "bootstrap_resamples",
                "confidence_interval_lower",
                "confidence_interval_upper",
                "status",
            },
            f"physical gate result metric_results[{index}]",
        )
        name = _nonempty_string(row.get("name"), f"physical gate result metric_results[{index}].name")
        if name in metric_names:
            raise SchemaError(f"physical gate result repeats metric {name!r}")
        metric_names.add(name)
        direction = row.get("direction")
        if direction not in {"lower_is_better", "higher_is_better"}:
            raise SchemaError(f"physical gate result metric_results[{index}].direction is unsupported")
        mandatory = _boolean(
            row.get("mandatory"),
            f"physical gate result metric_results[{index}].mandatory",
        )
        baseline = _nonnegative_number(
            row.get("baseline_median"),
            f"physical gate result metric_results[{index}].baseline_median",
        )
        candidate = _nonnegative_number(
            row.get("candidate_median"),
            f"physical gate result metric_results[{index}].candidate_median",
        )
        threshold = _nonnegative_number(
            row.get("regression_threshold_pct"),
            f"physical gate result metric_results[{index}].regression_threshold_pct",
        )
        expected_pct, _ = _gate_regression(
            baseline,
            candidate,
            direction=str(direction),
            threshold_pct=threshold,
        )
        raw_pct = row.get("regression_pct")
        if raw_pct is None:
            actual_pct = None
        else:
            actual_pct = _nonnegative_number(
                raw_pct,
                f"physical gate result metric_results[{index}].regression_pct",
            )
        if expected_pct is None:
            if actual_pct is not None:
                raise SchemaError(f"physical gate result metric {name!r} regression_pct is inconsistent")
        elif actual_pct is None or not math.isclose(actual_pct, expected_pct, rel_tol=1e-12, abs_tol=1e-12):
            raise SchemaError(f"physical gate result metric {name!r} regression_pct is inconsistent")

        _bounded_int(
            row.get("samples_considered"),
            f"physical gate result metric_results[{index}].samples_considered",
            1,
            limits.max_execution_observation_samples,
        )
        _nonnegative_number(
            row.get("baseline_relative_iqr_pct"),
            f"physical gate result metric_results[{index}].baseline_relative_iqr_pct",
        )
        _nonnegative_number(
            row.get("candidate_relative_iqr_pct"),
            f"physical gate result metric_results[{index}].candidate_relative_iqr_pct",
        )
        resamples = _bounded_int(
            row.get("bootstrap_resamples"),
            f"physical gate result metric_results[{index}].bootstrap_resamples",
            0,
            100_000,
        )
        lower_raw = row.get("confidence_interval_lower")
        upper_raw = row.get("confidence_interval_upper")
        status = row.get("status")
        if status not in {"pass", "fail", "warn", "inconclusive"}:
            raise SchemaError(f"physical gate result metric {name!r} status is unsupported")

        # A screened-out metric carries no interval; a decided one must carry a
        # well-ordered interval that reproduces its own status under the
        # acceptance boundary.  The point estimate never decides anything.
        if lower_raw is None or upper_raw is None:
            if lower_raw is not None or upper_raw is not None:
                raise SchemaError(f"physical gate result metric {name!r} has a half-open confidence interval")
            if resamples:
                raise SchemaError(f"physical gate result metric {name!r} reports resamples without an interval")
            if status != "inconclusive":
                raise SchemaError(f"physical gate result metric {name!r} decided without a confidence interval")
            screened_metrics.add(name)
        else:
            if not resamples:
                raise SchemaError(f"physical gate result metric {name!r} reports an interval without resamples")
            lower = _finite_number(
                lower_raw,
                f"physical gate result metric_results[{index}].confidence_interval_lower",
            )
            upper = _finite_number(
                upper_raw,
                f"physical gate result metric_results[{index}].confidence_interval_upper",
            )
            if lower > upper:
                raise SchemaError(f"physical gate result metric {name!r} confidence interval is inverted")
            acceptance = baseline * threshold / 100.0
            if lower > acceptance:
                expected_status = "fail" if mandatory else "warn"
            elif upper <= acceptance:
                expected_status = "pass"
            else:
                expected_status = "inconclusive"
            if status != expected_status:
                raise SchemaError(f"physical gate result metric {name!r} status is inconsistent")
            if expected_status == "fail":
                expected_metric_issues.add(f"mandatory_metric_regression:{name}")
            elif expected_status == "warn":
                expected_metric_issues.add(f"advisory_metric_regression:{name}")
            elif expected_status == "inconclusive":
                expected_metric_issues.add(f"undecidable_metric:{name}")

        if status == "fail":
            has_mandatory_failure = True
        elif status == "inconclusive" and mandatory:
            has_mandatory_inconclusive = True

    issues = set(result["issues"])
    comparability_issues: Set[str] = set()
    screened_reported: Set[str] = set()
    for issue in issues:
        if issue == "environment_mismatch":
            comparability_issues.add(issue)
            continue
        if issue.startswith("bundle_not_qualified:"):
            status = issue.partition(":")[2]
            if status not in PHYSICAL_CANARY_STATUSES or status == "qualified_physical_decision_canary":
                raise SchemaError("physical gate result contains an invalid bundle status issue")
            comparability_issues.add(issue)
            continue
        prefix, _, subject = issue.partition(":")
        if prefix in {"insufficient_samples", "unstable_measurement"}:
            if subject not in screened_metrics:
                raise SchemaError(f"physical gate result screens metric {subject!r} that was decided")
            screened_reported.add(subject)
            continue
        if issue not in expected_metric_issues:
            raise SchemaError(f"physical gate result contains unsupported issue {issue!r}")
    if expected_metric_issues.difference(issues):
        raise SchemaError("physical gate result omits a metric regression issue")
    if screened_metrics.difference(screened_reported):
        raise SchemaError("physical gate result omits the reason a metric was screened out")

    outcome = result.get("outcome")
    if comparability_issues:
        if outcome != "incomparable":
            raise SchemaError("physical gate result comparability issue requires incomparable outcome")
    elif outcome == "incomparable":
        raise SchemaError("physical gate result incomparable outcome lacks a comparability issue")
    elif has_mandatory_failure and outcome != "fail":
        raise SchemaError("physical gate result mandatory failure requires fail outcome")
    elif not has_mandatory_failure and has_mandatory_inconclusive and outcome != "inconclusive":
        raise SchemaError("physical gate result undecided mandatory metric requires inconclusive outcome")
    elif not has_mandatory_failure and not has_mandatory_inconclusive and outcome != "pass":
        raise SchemaError("physical gate result without a mandatory failure or undecided metric must pass")


def _gate_regression(
    baseline: float,
    candidate: float,
    *,
    direction: str,
    threshold_pct: float,
) -> Tuple[Optional[float], bool]:
    difference = candidate - baseline if direction == "lower_is_better" else baseline - candidate
    if difference <= 0.0:
        return 0.0, False
    if baseline == 0.0:
        return None, True
    regression_pct = difference / baseline * 100.0
    return regression_pct, regression_pct > threshold_pct


def _evaluation_observations(
    raw_observations: Any,
    *,
    candidate_id: str,
    perturbation_ids: Set[str],
    expected_collectives: Optional[int],
    expected_flops: Optional[int],
    world_size: int,
) -> None:
    if not isinstance(raw_observations, list):
        raise SchemaError(f"physical oracle corpus candidate {candidate_id!r} observations must be an array")
    observed: Set[str] = set()
    for index, raw_observation in enumerate(raw_observations):
        observation = _mapping(
            raw_observation,
            f"physical oracle corpus candidate {candidate_id!r} observations[{index}]",
        )
        _exact_fields(
            observation,
            {"perturbation_id", "decision", "physical_metrics", "evidence_sha256"},
            f"physical oracle corpus candidate {candidate_id!r} observations[{index}]",
        )
        perturbation_id = _nonempty_string(
            observation.get("perturbation_id"),
            f"physical oracle corpus candidate {candidate_id!r} observations[{index}].perturbation_id",
        )
        if perturbation_id not in perturbation_ids:
            raise SchemaError(
                f"physical oracle corpus candidate {candidate_id!r} references unknown perturbation {perturbation_id!r}"
            )
        if perturbation_id in observed:
            raise SchemaError(
                f"physical oracle corpus candidate {candidate_id!r} repeats perturbation {perturbation_id!r}"
            )
        observed.add(perturbation_id)
        if observation.get("decision") not in {"pass", "fail"}:
            raise SchemaError(f"physical oracle corpus candidate {candidate_id!r} decision is unsupported")
        _sha256(
            observation.get("evidence_sha256"),
            f"physical oracle corpus candidate {candidate_id!r} observations[{index}].evidence_sha256",
        )
        raw_metrics = observation.get("physical_metrics")
        _physical_metrics(
            raw_metrics,
            f"physical oracle corpus candidate {candidate_id!r} observations[{index}].physical_metrics",
            world_size=world_size,
        )
        metrics = _mapping(raw_metrics, "physical oracle corpus candidate physical_metrics")
        if expected_collectives is not None and (
            metrics.get("executed_collectives") != expected_collectives
            or metrics.get("executed_flops") != expected_flops
        ):
            raise SchemaError(
                f"physical oracle corpus candidate {candidate_id!r} executed work does not match its regions"
            )
    if observed != perturbation_ids:
        missing = sorted(perturbation_ids.difference(observed))
        raise SchemaError(f"physical oracle corpus candidate {candidate_id!r} lacks perturbations {missing[:10]}")


def _physical_metrics(raw: Any, label: str, *, world_size: int) -> None:
    metrics = _mapping(raw, label)
    _exact_fields(metrics, PHYSICAL_METRIC_FIELDS, label)
    runtime = _positive_number(metrics.get("physical_runtime_seconds"), f"{label}.physical_runtime_seconds")
    gpu_seconds = _positive_number(metrics.get("gpu_seconds"), f"{label}.gpu_seconds")
    gpu_count = _bounded_int(metrics.get("gpu_count"), f"{label}.gpu_count", 1, 65_536)
    if gpu_count != world_size:
        raise SchemaError(f"{label}.gpu_count does not match the policy runner world_size")
    if not math.isclose(gpu_seconds, runtime * gpu_count, rel_tol=1e-12, abs_tol=1e-12):
        raise SchemaError(f"{label}.gpu_seconds does not match runtime multiplied by gpu_count")
    for field in ("executed_collectives", "executed_flops", "peak_memory_bytes"):
        _bounded_int(metrics.get(field), f"{label}.{field}", 0, (1 << 63) - 1)


def _validate_gemm_execution(
    raw: Any,
    *,
    node_id: int,
    executed_flops: int,
    tensor_bytes: int,
) -> None:
    execution = _mapping(raw, f"Chakra projection GEMM Node.id={node_id} execution")
    _exact_fields(
        execution,
        {"dtype", "rank_recipes"},
        f"Chakra projection GEMM Node.id={node_id} execution",
    )
    dtype = _nonempty_string(execution.get("dtype"), f"Chakra projection GEMM Node.id={node_id} dtype")
    try:
        element_bytes = dtype_size_bytes(dtype)
    except SchemaError as exc:
        raise SchemaError(f"Chakra projection GEMM Node.id={node_id} dtype is unsupported") from exc
    raw_recipes = execution.get("rank_recipes")
    if not isinstance(raw_recipes, list) or not raw_recipes:
        raise SchemaError(f"Chakra projection GEMM Node.id={node_id} rank_recipes must be a non-empty array")
    ranks: List[int] = []
    expected_flops = 0
    expected_tensor_bytes = 0
    for index, raw_recipe in enumerate(raw_recipes):
        recipe = _mapping(raw_recipe, f"Chakra projection GEMM Node.id={node_id} rank_recipes[{index}]")
        _exact_fields(
            recipe,
            {"rank", "m", "n", "k"},
            f"Chakra projection GEMM Node.id={node_id} rank_recipes[{index}]",
        )
        rank = _bounded_int(
            recipe.get("rank"),
            f"Chakra projection GEMM Node.id={node_id} rank_recipes[{index}].rank",
            0,
            65_535,
        )
        ranks.append(rank)
        m, n, k = (
            _bounded_int(
                recipe.get(name),
                f"Chakra projection GEMM Node.id={node_id} rank_recipes[{index}].{name}",
                1,
                1 << 31,
            )
            for name in ("m", "n", "k")
        )
        expected_flops += 2 * m * n * k
        expected_tensor_bytes += element_bytes * (m * k + k * n + m * n)
    if ranks != sorted(set(ranks)):
        raise SchemaError(f"Chakra projection GEMM Node.id={node_id} rank_recipes must use unique rank order")
    if expected_flops != executed_flops or expected_tensor_bytes != tensor_bytes:
        raise SchemaError(f"Chakra projection GEMM Node.id={node_id} declared work does not match rank_recipes")


def _validate_all_reduce_execution(raw: Any, *, node_id: int, tensor_bytes: int) -> None:
    execution = _mapping(raw, f"Chakra projection all-reduce Node.id={node_id} execution")
    _exact_fields(
        execution,
        {"dtype", "ranks", "reduction"},
        f"Chakra projection all-reduce Node.id={node_id} execution",
    )
    dtype = _nonempty_string(execution.get("dtype"), f"Chakra projection all-reduce Node.id={node_id} dtype")
    try:
        element_bytes = dtype_size_bytes(dtype)
    except SchemaError as exc:
        raise SchemaError(f"Chakra projection all-reduce Node.id={node_id} dtype is unsupported") from exc
    if tensor_bytes % element_bytes != 0:
        raise SchemaError(f"Chakra projection all-reduce Node.id={node_id} tensor bytes are not dtype aligned")
    raw_ranks = execution.get("ranks")
    if not isinstance(raw_ranks, list) or len(raw_ranks) < 2:
        raise SchemaError(f"Chakra projection all-reduce Node.id={node_id} ranks must contain at least two ranks")
    ranks = [
        _bounded_int(
            raw_rank,
            f"Chakra projection all-reduce Node.id={node_id} ranks[{index}]",
            0,
            65_535,
        )
        for index, raw_rank in enumerate(raw_ranks)
    ]
    if ranks != sorted(set(ranks)):
        raise SchemaError(f"Chakra projection all-reduce Node.id={node_id} ranks must use unique rank order")
    if execution.get("reduction") != "sum":
        raise SchemaError(f"Chakra projection all-reduce Node.id={node_id} reduction must be 'sum'")


def _content_identity(document: Mapping[str, Any], identity_field: str) -> str:
    payload = {key: value for key, value in document.items() if key != identity_field}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_identity(
    document: Mapping[str, Any],
    field: str,
    calculator: Any,
    label: str,
) -> None:
    _sha256(document.get(field), f"{label} {field}")
    if document.get(field) != calculator(document):
        raise SchemaError(f"{label} {field} does not match canonical content")


def _resource_check(document: Mapping[str, Any], label: str, limits: ResourceLimits) -> None:
    try:
        validate_json_mapping(document, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"{label} violates resource policy: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        missing = sorted(expected_set.difference(actual))
        extra = sorted(actual.difference(expected_set))
        raise SchemaError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaError(f"{label} must be a lowercase SHA-256")
    return text


def _oci_digest(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if not text.startswith("sha256:"):
        raise SchemaError(f"{label} must use a sha256 OCI digest")
    _sha256(text.removeprefix("sha256:"), label)
    return text


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{label} must be a boolean")
    return value


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise SchemaError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SchemaError(f"{label} must be finite")
    return parsed


def _nonnegative_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SchemaError(f"{label} must be a finite non-negative number")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _nonnegative_number(value, label)
    if result <= 0.0:
        raise SchemaError(f"{label} must be positive")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _nonnegative_number(value, label)
    if result > 1.0:
        raise SchemaError(f"{label} must be in [0, 1]")
    return result


def _sorted_unique_strings(value: Any, label: str, *, require_nonempty: bool = False) -> Tuple[str, ...]:
    if not isinstance(value, list) or (require_nonempty and not value):
        qualifier = "a non-empty" if require_nonempty else "an"
        raise SchemaError(f"{label} must be {qualifier} array")
    result = tuple(_nonempty_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if result != tuple(sorted(set(result))):
        raise SchemaError(f"{label} must be sorted and unique")
    return result


def _disclosures(value: Any, label: str) -> Tuple[str, ...]:
    result = _sorted_unique_strings(value, label)
    unknown = sorted(set(result).difference(DISCLOSURE_CATEGORIES))
    if unknown:
        raise SchemaError(f"{label} contains unsupported categories {unknown}")
    return result


def _unique_positive_ints(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SchemaError(f"{label} must be a non-empty array")
    result = tuple(_bounded_int(item, f"{label}[{index}]", 1, (1 << 64) - 1) for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise SchemaError(f"{label} must contain unique Node IDs")
    return result


__all__ = [
    "DISCLOSURE_CATEGORIES",
    "PHYSICAL_CANARY_STATUSES",
    "PHYSICAL_METRIC_FIELDS",
    "SUPPORTED_CANDIDATE_METHODS",
    "SUPPORTED_PHYSICAL_DOMAIN",
    "SUPPORTED_PROJECTION_OPERATIONS",
    "chakra_projection_sha256",
    "leakage_assessment",
    "physical_oracle_corpus_sha256",
    "physical_policy_sha256",
    "validate_chakra_projection",
    "validate_physical_canary_policy",
    "validate_physical_gate_observation",
    "validate_physical_gate_result",
    "validate_physical_oracle_corpus",
    "with_content_identity",
]
