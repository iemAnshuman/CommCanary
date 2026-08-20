"""Active counterexample-guided synthesis using measured physical executions."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple, Union

from ..artifacts.application_measurement import application_workload_identity
from ..artifacts.application_oracle import (
    application_evidence_comparable,
    application_evidence_id,
    application_evidence_regression_decision,
    build_application_evidence_set,
    validate_application_evidence,
)
from ..artifacts.chakra import ChakraExecutionTrace, encode_chakra_subgraph
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.physical_canary import (
    validate_chakra_projection,
    validate_physical_canary_policy,
    validate_physical_oracle_corpus,
    with_content_identity,
)
from ..artifacts.physical_execution import (
    build_physical_execution_evidence_set,
    validate_physical_execution_measurement,
)
from ..errors import SchemaError
from ..formats import ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT, PHYSICAL_ORACLE_CORPUS_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits

GRAPH_BASELINE_METHODS = frozenset(
    {
        "independent_exact_replay",
        "random_sampling",
        "stratified_sampling",
        "ddmin",
        "communication_only_microbenchmark",
    }
)


@dataclass(frozen=True)
class PhysicalCandidateRequest:
    """One immutable measurement request handed to a site executor."""

    candidate_id: str
    method: str
    selected_region_ids: Tuple[str, ...]
    selected_node_ids: Tuple[int, ...]
    executable_et: bytes
    executable_sha256: str
    role: str
    split: str
    perturbation_id: str
    subject_sha256: str


PhysicalMeasurementExecutor = Callable[[PhysicalCandidateRequest], Mapping[str, Any]]


@dataclass(frozen=True)
class HoldoutApplicationBatch:
    """Paired holdout baseline plus predeclared held-out perturbations."""

    baseline: Mapping[str, Any]
    perturbations: Mapping[str, Mapping[str, Any]]


HoldoutApplicationLoader = Callable[
    [],
    Union[Mapping[str, Mapping[str, Any]], HoldoutApplicationBatch],
]
SelectionFreezer = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ActivePhysicalSynthesisResult:
    status: str
    selected_region_ids: Tuple[str, ...]
    selected_node_ids: Tuple[int, ...]
    canary_et: bytes
    corpus: Dict[str, Any]
    ledger: Dict[str, Any]
    application_evidence: Dict[str, Any]
    physical_evidence: Dict[str, Any]


def synthesize_active_physical_canary(
    trace: ChakraExecutionTrace,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    baseline_application: Mapping[str, Any],
    training_applications: Mapping[str, Mapping[str, Any]],
    holdout_perturbation_ids: Sequence[str],
    holdout_application_loader: HoldoutApplicationLoader,
    executor: PhysicalMeasurementExecutor,
    selection_freezer: SelectionFreezer | None = None,
    max_candidate_evaluations: int = 256,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> ActivePhysicalSynthesisResult:
    """Search measured training counterexamples, freeze, then open holdout.

    The holdout loader is not invoked until the exact selection SHA-256 has
    been constructed. The site executor may submit SLURM jobs, call a local
    four-rank runner, or serve deterministic tests; every returned physical
    measurement is semantically revalidated here.
    """

    validate_physical_canary_policy(policy, limits=limits)
    validate_chakra_projection(projection, trace, limits=limits)
    validate_application_evidence(baseline_application)
    if not isinstance(max_candidate_evaluations, int) or max_candidate_evaluations < 1:
        raise SchemaError("active physical synthesis max_candidate_evaluations must be positive")
    if not training_applications:
        raise SchemaError("active physical synthesis requires training application perturbations")
    holdout_ids = _unique_strings(holdout_perturbation_ids, "active physical holdout perturbation IDs")
    if not holdout_ids:
        raise SchemaError("active physical synthesis requires held-out perturbations")
    if set(holdout_ids).intersection(training_applications):
        raise SchemaError("active physical training and holdout perturbation IDs overlap")

    runner_digest = str(policy["runner"]["oci_digest"])
    _validate_application_set(
        baseline_application,
        training_applications,
        runner_digest=runner_digest,
        expected_ids=set(training_applications),
    )
    regions = _regions(projection)
    region_order = tuple(regions)
    if _overlapping_regions(regions):
        raise SchemaError("active physical synthesis requires non-overlapping executable regions")
    required_tags = set(str(value) for value in policy["required_feature_tags"])
    initial = _greedy_feature_selection(regions, required_tags)
    full_node_ids = set(trace.node_ids)
    if _selected_nodes(regions, initial) == full_node_ids:
        raise SchemaError("active physical synthesis initial candidate does not execute fewer source nodes")

    threshold = float(policy["severe_regression_threshold_pct"])
    training_truth = _application_truth_rows(
        baseline_application,
        training_applications,
        split="training",
        threshold_pct=threshold,
        projection=projection,
        world_size=int(policy["runner"]["world_size"]),
    )
    execution_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    evaluation_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    request_ledger: List[Dict[str, Any]] = []
    candidate_steps: List[Dict[str, Any]] = []

    def measure(request: PhysicalCandidateRequest) -> Mapping[str, Any]:
        key = (request.candidate_id, request.subject_sha256)
        cached = execution_cache.get(key)
        if cached is not None:
            return cached
        measurement = executor(request)
        validate_physical_execution_measurement(measurement)
        _bind_measurement_to_request(
            measurement,
            request,
            trace=trace,
            projection=projection,
            runner_digest=runner_digest,
        )
        if measurement["telemetry_assessment"]["comparable"] is not True:
            raise SchemaError(
                "active physical execution is environmentally incomparable: "
                f"{measurement['telemetry_assessment']['issues']}"
            )
        execution_cache[key] = copy.deepcopy(dict(measurement))
        request_ledger.append(
            {
                "candidate_id": request.candidate_id,
                "method": request.method,
                "selected_region_ids": list(request.selected_region_ids),
                "executable_sha256": request.executable_sha256,
                "split": request.split,
                "perturbation_id": request.perturbation_id,
                "subject_sha256": request.subject_sha256,
                "measurement_id": measurement["measurement_id"],
            }
        )
        return execution_cache[key]

    def training_evaluation(selection: Tuple[str, ...]) -> Dict[str, Any]:
        canonical = _canonical_selection(selection, region_order)
        cached = evaluation_cache.get(canonical)
        if cached is not None:
            return cached
        if len(evaluation_cache) >= max_candidate_evaluations:
            raise SchemaError("active physical synthesis exhausted max_candidate_evaluations")
        if _selected_nodes(regions, canonical) == full_node_ids:
            raise SchemaError("active physical reduced candidate cannot execute the complete source graph")
        result = _execute_candidate(
            trace,
            regions,
            canonical,
            method="reduced_decision_canary",
            baseline_application=baseline_application,
            perturbation_applications=training_applications,
            split="training",
            threshold_pct=threshold,
            measure=measure,
        )
        result["statistics"] = _decision_statistics(result["observations"], training_truth, policy)
        evaluation_cache[canonical] = result
        return result

    current_selection = _canonical_selection(initial, region_order)
    current = training_evaluation(current_selection)
    while not _statistics_pass(current["statistics"], policy):
        current_key = _statistics_key(current["statistics"])
        additions: List[Tuple[Tuple[Any, ...], Tuple[str, ...], Dict[str, Any]]] = []
        for region_id in region_order:
            if region_id in current_selection:
                continue
            proposed = _canonical_selection((*current_selection, region_id), region_order)
            if _selected_nodes(regions, proposed) == full_node_ids:
                continue
            evaluated = training_evaluation(proposed)
            additions.append((_statistics_key(evaluated["statistics"]), proposed, evaluated))
        if not additions:
            break
        best_key, best_selection, best = min(additions, key=lambda row: (row[0], _candidate_cost(row[2]), row[1]))
        if best_key >= current_key:
            break
        candidate_steps.append(
            {
                "action": "counterexample_addition",
                "from_candidate_id": current["candidate_id"],
                "to_candidate_id": best["candidate_id"],
                "counterexample_perturbation_ids": _severe_false_negative_ids(current["observations"], training_truth),
                "added_region_ids": [region_id for region_id in best_selection if region_id not in current_selection],
            }
        )
        current_selection = best_selection
        current = best

    if _statistics_pass(current["statistics"], policy):
        while True:
            removable: List[Tuple[Tuple[Any, ...], Tuple[str, ...], Dict[str, Any]]] = []
            for region_id in current_selection:
                proposed = tuple(value for value in current_selection if value != region_id)
                if not proposed or not _covers_tags(regions, proposed, required_tags):
                    continue
                evaluated = training_evaluation(proposed)
                if _statistics_pass(evaluated["statistics"], policy):
                    removable.append((_candidate_cost(evaluated), proposed, evaluated))
            if not removable:
                break
            _cost, next_selection, next_candidate = min(removable, key=lambda row: (row[0], row[1]))
            candidate_steps.append(
                {
                    "action": "ddmin_prune",
                    "from_candidate_id": current["candidate_id"],
                    "to_candidate_id": next_candidate["candidate_id"],
                    "removed_region_ids": [
                        region_id for region_id in current_selection if region_id not in next_selection
                    ],
                }
            )
            current_selection = next_selection
            current = next_candidate

    canary_et, selected_node_ids = _encode_selection(trace, regions, current_selection)
    selection = {
        "source_et_sha256": trace.source_sha256,
        "projection_id": projection["projection_id"],
        "policy_id": policy["policy_id"],
        "selected_region_ids": list(current_selection),
        "selected_node_ids": list(selected_node_ids),
        "canary_et_sha256": hashlib.sha256(canary_et).hexdigest(),
    }
    selection_sha256 = hashlib.sha256(canonical_json_bytes(selection)).hexdigest()
    selection["selection_sha256"] = selection_sha256

    # This is the first call site that can materialize holdout decisions.
    training_measurement_request_count_before_freeze = len(request_ledger)
    if selection_freezer is not None:
        selection_freezer(copy.deepcopy(selection))
    loaded_holdout = holdout_application_loader()
    if isinstance(loaded_holdout, HoldoutApplicationBatch):
        holdout_baseline = loaded_holdout.baseline
        holdout_applications = loaded_holdout.perturbations
    else:
        # Compatibility for callers with one allocation-era evidence. New
        # physical studies should return a paired holdout baseline.
        holdout_baseline = baseline_application
        holdout_applications = loaded_holdout
    validate_application_evidence(holdout_baseline)
    _validate_application_set(
        holdout_baseline,
        holdout_applications,
        runner_digest=runner_digest,
        expected_ids=set(holdout_ids),
    )
    application_evidence = build_application_evidence_set(
        training_baseline=baseline_application,
        training_perturbations=training_applications,
        holdout_baseline=holdout_baseline,
        holdout_perturbations=holdout_applications,
    )
    holdout_truth = _application_truth_rows(
        holdout_baseline,
        holdout_applications,
        split="holdout",
        threshold_pct=threshold,
        projection=projection,
        world_size=int(policy["runner"]["world_size"]),
    )
    final_all = _execute_candidate(
        trace,
        regions,
        current_selection,
        method="reduced_decision_canary",
        baseline_application=baseline_application,
        perturbation_applications={**training_applications, **holdout_applications},
        split="all",
        threshold_pct=threshold,
        measure=measure,
    )
    final_training_stats = _decision_statistics(final_all["observations"], training_truth, policy)
    final_holdout_stats = _decision_statistics(final_all["observations"], holdout_truth, policy)

    baseline_evaluations = _execute_required_baselines(
        trace=trace,
        regions=regions,
        projection=projection,
        policy=policy,
        selected=current_selection,
        baseline_application=baseline_application,
        applications={**training_applications, **holdout_applications},
        threshold_pct=threshold,
        measure=measure,
    )
    perturbations = [*training_truth.values(), *holdout_truth.values()]
    raw_corpus: Dict[str, Any] = {
        "format": PHYSICAL_ORACLE_CORPUS_FORMAT,
        "source_et_sha256": trace.source_sha256,
        "projection_id": projection["projection_id"],
        "policy_id": policy["policy_id"],
        "runner_oci_digest": runner_digest,
        "baseline_subject_sha256": baseline_application["subject_sha256"],
        "evidence_kind": "measured",
        "application": {
            "ground_truth_kind": "application_ground_truth",
            "name": baseline_application["application"]["name"],
            "engine": baseline_application["application"]["engine"],
            "artifact_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "application": baseline_application["application"],
                        "workload": baseline_application["workload"],
                    }
                )
            ).hexdigest(),
        },
        "perturbations": perturbations,
        "evaluations": [
            {
                "candidate_id": final_all["candidate_id"],
                "method": "reduced_decision_canary",
                "selected_region_ids": list(current_selection),
                "executable_sha256": final_all["executable_sha256"],
                "observations": final_all["observations"],
            },
            *baseline_evaluations,
        ],
    }
    corpus = with_content_identity(raw_corpus, "corpus_id")
    validate_physical_oracle_corpus(corpus, trace, projection, policy, limits=limits)
    status = (
        "qualified_active_candidate"
        if _statistics_pass(final_training_stats, policy) and _statistics_pass(final_holdout_stats, policy)
        else "failed_active_candidate"
    )
    measurements_by_id = {str(measurement["measurement_id"]): measurement for measurement in execution_cache.values()}
    physical_evidence = build_physical_execution_evidence_set(
        [
            {
                "candidate_id": request["candidate_id"],
                "method": request["method"],
                "measurement": measurements_by_id[str(request["measurement_id"])],
            }
            for request in request_ledger
        ]
    )
    raw_ledger: Dict[str, Any] = {
        "format": ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT,
        "algorithm": "active-counterexample-guided-physical-minimization.v1",
        "status": status,
        "source_et_sha256": trace.source_sha256,
        "projection_id": projection["projection_id"],
        "policy_id": policy["policy_id"],
        "predeclared_holdout_perturbation_ids": list(holdout_ids),
        "selection": selection,
        "holdout_access": {
            "first_access_after_selection_sha256": selection_sha256,
            "loaded_perturbation_ids": list(holdout_ids),
            "baseline_application_evidence_sha256": application_evidence_id(holdout_baseline),
        },
        "candidate_steps": candidate_steps,
        "training_candidate_count": len(evaluation_cache),
        "training_measurement_request_count_before_freeze": training_measurement_request_count_before_freeze,
        "measurement_requests": request_ledger,
        "final_statistics": {"training": final_training_stats, "holdout": final_holdout_stats},
        "corpus_id": corpus["corpus_id"],
        "application_evidence_set_id": application_evidence["evidence_set_id"],
        "physical_evidence_set_id": physical_evidence["evidence_set_id"],
    }
    ledger = with_content_identity(raw_ledger, "ledger_id")
    validate_active_physical_study_ledger(ledger)
    return ActivePhysicalSynthesisResult(
        status=status,
        selected_region_ids=current_selection,
        selected_node_ids=selected_node_ids,
        canary_et=canary_et,
        corpus=corpus,
        ledger=ledger,
        application_evidence=application_evidence,
        physical_evidence=physical_evidence,
    )


def validate_active_physical_study_ledger(ledger: Mapping[str, Any]) -> None:
    """Validate selection-before-holdout ordering and every ledger identity."""

    expected_fields = {
        "format",
        "ledger_id",
        "algorithm",
        "status",
        "source_et_sha256",
        "projection_id",
        "policy_id",
        "predeclared_holdout_perturbation_ids",
        "selection",
        "holdout_access",
        "candidate_steps",
        "training_candidate_count",
        "training_measurement_request_count_before_freeze",
        "measurement_requests",
        "final_statistics",
        "corpus_id",
        "application_evidence_set_id",
        "physical_evidence_set_id",
    }
    if set(ledger) != expected_fields:
        raise SchemaError("active physical study ledger fields are not closed")
    if ledger.get("format") != ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT:
        raise SchemaError("active physical study ledger format is unsupported")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in ledger.items() if key != "ledger_id"})
    ).hexdigest()
    if ledger.get("ledger_id") != expected_id:
        raise SchemaError("active physical study ledger_id does not match canonical content")
    if ledger.get("algorithm") != "active-counterexample-guided-physical-minimization.v1":
        raise SchemaError("active physical study algorithm is unsupported")
    if ledger.get("status") not in {"qualified_active_candidate", "failed_active_candidate"}:
        raise SchemaError("active physical study status is unsupported")
    for field in (
        "source_et_sha256",
        "projection_id",
        "policy_id",
        "corpus_id",
        "application_evidence_set_id",
        "physical_evidence_set_id",
    ):
        _sha256_value(ledger.get(field), f"active physical study {field}")
    holdout_ids = _unique_strings(
        ledger.get("predeclared_holdout_perturbation_ids"),
        "active physical study holdout IDs",
    )
    selection = _require_mapping(ledger.get("selection"), "active physical study selection")
    if set(selection) != {
        "source_et_sha256",
        "projection_id",
        "policy_id",
        "selected_region_ids",
        "selected_node_ids",
        "canary_et_sha256",
        "selection_sha256",
    }:
        raise SchemaError("active physical study selection fields are not closed")
    expected_selection_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in selection.items() if key != "selection_sha256"})
    ).hexdigest()
    if selection.get("selection_sha256") != expected_selection_id:
        raise SchemaError("active physical study selection_sha256 does not match canonical content")
    for field in ("source_et_sha256", "projection_id", "policy_id"):
        if selection.get(field) != ledger.get(field):
            raise SchemaError(f"active physical study selection {field} mismatch")
    _sha256_value(selection.get("canary_et_sha256"), "active physical study canary ET")
    _unique_strings(selection.get("selected_region_ids"), "active physical study selected regions")
    selected_nodes = selection.get("selected_node_ids")
    if (
        not isinstance(selected_nodes, list)
        or not selected_nodes
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in selected_nodes)
        or len(selected_nodes) != len(set(selected_nodes))
    ):
        raise SchemaError("active physical study selected nodes must be unique positive integers")

    holdout_access = _require_mapping(ledger.get("holdout_access"), "active physical study holdout access")
    if set(holdout_access) != {
        "first_access_after_selection_sha256",
        "loaded_perturbation_ids",
        "baseline_application_evidence_sha256",
    }:
        raise SchemaError("active physical study holdout access fields are not closed")
    if holdout_access.get("first_access_after_selection_sha256") != selection.get("selection_sha256"):
        raise SchemaError("active physical holdout was not bound after the frozen selection")
    loaded = _unique_strings(holdout_access.get("loaded_perturbation_ids"), "active physical loaded holdout IDs")
    if set(loaded) != set(holdout_ids):
        raise SchemaError("active physical loaded holdout inventory does not match its predeclaration")
    _sha256_value(
        holdout_access.get("baseline_application_evidence_sha256"),
        "active physical holdout baseline evidence",
    )

    training_count = ledger.get("training_candidate_count")
    if not isinstance(training_count, int) or isinstance(training_count, bool) or training_count < 1:
        raise SchemaError("active physical training candidate count must be positive")
    freeze_count = ledger.get("training_measurement_request_count_before_freeze")
    requests = ledger.get("measurement_requests")
    if (
        not isinstance(freeze_count, int)
        or isinstance(freeze_count, bool)
        or freeze_count < 1
        or not isinstance(requests, list)
        or freeze_count > len(requests)
    ):
        raise SchemaError("active physical training request freeze boundary is invalid")
    for index, raw_request in enumerate(requests):
        request = _require_mapping(raw_request, f"active physical measurement_requests[{index}]")
        if set(request) != {
            "candidate_id",
            "method",
            "selected_region_ids",
            "executable_sha256",
            "split",
            "perturbation_id",
            "subject_sha256",
            "measurement_id",
        }:
            raise SchemaError("active physical measurement request fields are not closed")
        for field in ("candidate_id", "executable_sha256", "subject_sha256", "measurement_id"):
            _sha256_value(request.get(field), f"active physical request {field}")
        perturbation_id = request.get("perturbation_id")
        if not isinstance(perturbation_id, str) or not perturbation_id:
            raise SchemaError("active physical request perturbation_id must be non-empty")
        if index < freeze_count and perturbation_id in set(holdout_ids):
            raise SchemaError("active physical holdout measurement was requested before selection freeze")
    if not any(request.get("perturbation_id") in set(holdout_ids) for request in requests[freeze_count:]):
        raise SchemaError("active physical ledger contains no post-freeze holdout measurement")
    if not isinstance(ledger.get("candidate_steps"), list):
        raise SchemaError("active physical candidate_steps must be an array")
    if not isinstance(ledger.get("final_statistics"), Mapping):
        raise SchemaError("active physical final_statistics must be an object")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_application_set(
    baseline: Mapping[str, Any],
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    runner_digest: str,
    expected_ids: Set[str],
) -> None:
    if not application_evidence_comparable(baseline):
        raise SchemaError("application baseline is environmentally incomparable")
    if set(measurements) != expected_ids:
        raise SchemaError("application perturbation inventory does not match its predeclared IDs")
    seen_subjects = {str(baseline.get("subject_sha256"))}
    baseline_workload_identity = application_workload_identity(baseline)
    for perturbation_id, measurement in measurements.items():
        validate_application_evidence(measurement)
        if not application_evidence_comparable(measurement):
            raise SchemaError(f"application perturbation {perturbation_id!r} is environmentally incomparable")
        if measurement.get("perturbation_id") != perturbation_id:
            raise SchemaError("application perturbation key does not match its measurement")
        if measurement["runner"]["oci_digest"] != runner_digest:
            raise SchemaError("application measurement runner digest does not match physical policy")
        subject_sha256 = str(measurement.get("subject_sha256"))
        if subject_sha256 in seen_subjects:
            raise SchemaError("application perturbations must use unique stack subjects")
        seen_subjects.add(subject_sha256)
        if application_workload_identity(measurement) != baseline_workload_identity:
            raise SchemaError("application perturbations must execute one exact workload")
    if baseline["runner"]["oci_digest"] != runner_digest:
        raise SchemaError("application baseline runner digest does not match physical policy")


def _application_truth_rows(
    baseline: Mapping[str, Any],
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    threshold_pct: float,
    projection: Mapping[str, Any],
    world_size: int,
) -> Dict[str, Dict[str, Any]]:
    nodes = projection["nodes"]
    collectives = sum(node["operation"] == "all_reduce" for node in nodes)
    flops = sum(int(node["executed_flops"]) for node in nodes)
    result: Dict[str, Dict[str, Any]] = {}
    for perturbation_id, measurement in measurements.items():
        decision = application_evidence_regression_decision(baseline, measurement, threshold_pct=threshold_pct)
        runtime = float(measurement["summary"]["median_batch_latency_ms"]) / 1000.0
        result[perturbation_id] = {
            "perturbation_id": perturbation_id,
            "split": split,
            "application_decision": decision["decision"],
            "regression_magnitude_pct": decision["regression_magnitude_pct"],
            "severe": decision["decision"] == "fail" and decision["regression_magnitude_pct"] >= threshold_pct,
            "application_metrics": {
                "physical_runtime_seconds": runtime,
                "gpu_seconds": runtime * world_size,
                "gpu_count": world_size,
                "executed_collectives": collectives,
                "executed_flops": flops,
                "peak_memory_bytes": int(measurement["summary"]["max_observed_gpu_memory_used_bytes"]),
            },
            "application_evidence_sha256": application_evidence_id(measurement),
            "environment_sha256": measurement["environment_sha256"],
            "candidate_subject_sha256": measurement["subject_sha256"],
        }
    return result


def _execute_candidate(
    trace: ChakraExecutionTrace,
    regions: Mapping[str, Mapping[str, Any]],
    selected: Tuple[str, ...],
    *,
    method: str,
    baseline_application: Mapping[str, Any],
    perturbation_applications: Mapping[str, Mapping[str, Any]],
    split: str,
    threshold_pct: float,
    measure: Callable[[PhysicalCandidateRequest], Mapping[str, Any]],
) -> Dict[str, Any]:
    communication_only = method == "communication_only_microbenchmark"
    executable, selected_nodes = (
        _encode_communication_only_selection(trace, regions, selected)
        if communication_only
        else _encode_selection(trace, regions, selected)
    )
    executable_sha256 = hashlib.sha256(executable).hexdigest()
    candidate_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "method": method,
                "selected_region_ids": list(selected),
                "executable_sha256": executable_sha256,
            }
        )
    ).hexdigest()
    role = "trace_derived_reference" if set(selected_nodes) == set(trace.node_ids) else "reduced_decision_canary"
    baseline_request = PhysicalCandidateRequest(
        candidate_id=candidate_id,
        method=method,
        selected_region_ids=selected,
        selected_node_ids=selected_nodes,
        executable_et=executable,
        executable_sha256=executable_sha256,
        role=role,
        split=split,
        perturbation_id=str(baseline_application["perturbation_id"]),
        subject_sha256=str(baseline_application["subject_sha256"]),
    )
    baseline_measurement = measure(baseline_request)
    observations = []
    for perturbation_id, application in perturbation_applications.items():
        request = PhysicalCandidateRequest(
            candidate_id=candidate_id,
            method=method,
            selected_region_ids=selected,
            selected_node_ids=selected_nodes,
            executable_et=executable,
            executable_sha256=executable_sha256,
            role=role,
            split=split,
            perturbation_id=perturbation_id,
            subject_sha256=str(application["subject_sha256"]),
        )
        candidate_measurement = measure(request)
        baseline_runtime = float(baseline_measurement["physical_metrics"]["physical_runtime_seconds"])
        candidate_runtime = float(candidate_measurement["physical_metrics"]["physical_runtime_seconds"])
        regression_pct = max(0.0, (candidate_runtime - baseline_runtime) / baseline_runtime * 100.0)
        observations.append(
            {
                "perturbation_id": perturbation_id,
                "decision": "fail" if regression_pct > threshold_pct else "pass",
                "physical_metrics": copy.deepcopy(candidate_measurement["physical_metrics"]),
                "evidence_sha256": candidate_measurement["measurement_id"],
            }
        )
    observations.sort(key=lambda row: str(row["perturbation_id"]))
    return {
        "candidate_id": candidate_id,
        "method": method,
        "selected_region_ids": list(selected),
        "selected_node_ids": list(selected_nodes),
        "executable_sha256": executable_sha256,
        "observations": observations,
    }


def _bind_measurement_to_request(
    measurement: Mapping[str, Any],
    request: PhysicalCandidateRequest,
    *,
    trace: ChakraExecutionTrace,
    projection: Mapping[str, Any],
    runner_digest: str,
) -> None:
    expected = {
        "role": request.role,
        "subject_sha256": request.subject_sha256,
        "perturbation_id": request.perturbation_id,
        "source_et_sha256": trace.source_sha256,
        "executable_sha256": request.executable_sha256,
        "projection_id": projection["projection_id"],
        "selected_region_ids": list(request.selected_region_ids),
        "selected_node_ids": list(request.selected_node_ids),
    }
    for field, value in expected.items():
        if measurement.get(field) != value:
            raise SchemaError(f"physical measurement does not match request field {field}")
    if measurement["runner"]["oci_digest"] != runner_digest:
        raise SchemaError("physical measurement runner digest does not match policy")


def _decision_statistics(
    observations: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    by_id = {str(row["perturbation_id"]): row for row in observations}
    rows = [truth[key] for key in sorted(truth)]
    severe_false_negatives = 0
    false_positives = 0
    passing = 0
    agreements = 0
    reductions: List[float] = []
    runtimes: List[float] = []
    for truth_row in rows:
        observation = by_id[str(truth_row["perturbation_id"])]
        if observation["decision"] == truth_row["application_decision"]:
            agreements += 1
        if truth_row["severe"] and truth_row["application_decision"] == "fail" and observation["decision"] == "pass":
            severe_false_negatives += 1
        if truth_row["application_decision"] == "pass":
            passing += 1
            if observation["decision"] == "fail":
                false_positives += 1
        runtime = float(observation["physical_metrics"]["physical_runtime_seconds"])
        runtimes.append(runtime)
        reductions.append(float(truth_row["application_metrics"]["physical_runtime_seconds"]) / runtime)
    return {
        "decision_pairs": len(rows),
        "severe_false_negatives": severe_false_negatives,
        "false_positives": false_positives,
        "false_positive_rate": _false_positive_rate(false_positives, passing),
        "passing_perturbations": passing,
        "pairwise_decision_agreement": agreements / len(rows),
        "minimum_runtime_reduction_ratio": min(reductions),
        "physical_runtime_max_seconds": max(runtimes),
    }


def _false_positive_rate(false_positives: int, passing: int) -> float:
    """False-positive rate, vacuously zero when nothing could be a false positive.

    With no passing perturbation the rate is undefined rather than perfect, so
    the denominator is published beside it as ``passing_perturbations`` and the
    qualification gate refuses to treat a vacuous zero as evidence.  Reporting
    ``0.0`` alone let a corpus of nothing but regressions satisfy the
    maximum-false-positive gate without measuring anything.
    """

    return false_positives / passing if passing else 0.0


def _statistics_pass(statistics: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    gates = policy["qualification_gates"]
    return bool(
        # A vacuous false-positive rate is not a measured one.
        (statistics["passing_perturbations"] > 0 or gates["maximum_false_positive_rate"] >= 1.0)
        and statistics["severe_false_negatives"] <= gates["maximum_severe_false_negatives"]
        and statistics["false_positive_rate"] <= gates["maximum_false_positive_rate"]
        and statistics["pairwise_decision_agreement"] >= gates["minimum_pairwise_decision_agreement"]
        and statistics["minimum_runtime_reduction_ratio"] >= gates["minimum_runtime_reduction_ratio"]
        and statistics["physical_runtime_max_seconds"] <= policy["runtime_budget_seconds"]
    )


def _statistics_key(statistics: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(statistics["severe_false_negatives"]),
        float(statistics["false_positive_rate"]),
        -float(statistics["pairwise_decision_agreement"]),
        -float(statistics["minimum_runtime_reduction_ratio"]),
        float(statistics["physical_runtime_max_seconds"]),
    )


def _candidate_cost(candidate: Mapping[str, Any]) -> Tuple[float, int, str]:
    return (
        float(candidate["statistics"]["physical_runtime_max_seconds"]),
        len(candidate["selected_node_ids"]),
        str(candidate["candidate_id"]),
    )


def _severe_false_negative_ids(
    observations: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    by_id = {str(row["perturbation_id"]): row for row in observations}
    return sorted(
        perturbation_id
        for perturbation_id, row in truth.items()
        if row["severe"] and row["application_decision"] == "fail" and by_id[perturbation_id]["decision"] == "pass"
    )


def _execute_required_baselines(
    *,
    trace: ChakraExecutionTrace,
    regions: Mapping[str, Mapping[str, Any]],
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    selected: Tuple[str, ...],
    baseline_application: Mapping[str, Any],
    applications: Mapping[str, Mapping[str, Any]],
    threshold_pct: float,
    measure: Callable[[PhysicalCandidateRequest], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    required = tuple(str(value) for value in policy["required_baselines"])
    unsupported = sorted(set(required).difference(GRAPH_BASELINE_METHODS))
    if unsupported:
        raise SchemaError(f"active physical baseline executor is not implemented for {unsupported}")
    order = tuple(regions)
    selections: Dict[str, Tuple[str, ...]] = {}
    for method in required:
        if method == "independent_exact_replay":
            selections[method] = order
        elif method == "random_sampling":
            generator = random.Random(str(policy["policy_id"]))
            selections[method] = _canonical_selection(
                generator.sample(list(order), min(len(selected), len(order) - 1)),
                order,
            )
        elif method == "stratified_sampling":
            selections[method] = _greedy_feature_selection(
                regions,
                set(str(value) for value in policy["required_feature_tags"]),
            )
        elif method == "ddmin":
            selections[method] = selected
        elif method == "communication_only_microbenchmark":
            selections[method] = order
    rows = []
    for method in required:
        evaluation = _execute_candidate(
            trace,
            regions,
            selections[method],
            method=method,
            baseline_application=baseline_application,
            perturbation_applications=applications,
            split="baseline",
            threshold_pct=threshold_pct,
            measure=measure,
        )
        rows.append(
            {
                "candidate_id": evaluation["candidate_id"],
                "method": method,
                "selected_region_ids": evaluation["selected_region_ids"],
                "executable_sha256": evaluation["executable_sha256"],
                "observations": evaluation["observations"],
            }
        )
    return rows


def _regions(projection: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["region_id"]): row for row in projection["regions"]}


def _overlapping_regions(regions: Mapping[str, Mapping[str, Any]]) -> bool:
    seen: Set[int] = set()
    for region in regions.values():
        node_ids = {int(value) for value in region["node_ids"]}
        if seen.intersection(node_ids):
            return True
        seen.update(node_ids)
    return False


def _greedy_feature_selection(
    regions: Mapping[str, Mapping[str, Any]],
    required_tags: Set[str],
) -> Tuple[str, ...]:
    uncovered = set(required_tags)
    selected: Set[str] = set()
    while uncovered:
        choices = []
        for region_id, region in regions.items():
            coverage = uncovered.intersection(str(value) for value in region["feature_tags"])
            if coverage:
                choices.append((-len(coverage), len(region["node_ids"]), region_id))
        if not choices:
            raise SchemaError(f"physical regions do not cover required feature tags {sorted(uncovered)}")
        _coverage, _nodes, region_id = min(choices)
        selected.add(region_id)
        uncovered.difference_update(str(value) for value in regions[region_id]["feature_tags"])
    if not selected:
        selected.add(min(regions, key=lambda key: (len(regions[key]["node_ids"]), key)))
    return _canonical_selection(tuple(selected), tuple(regions))


def _covers_tags(
    regions: Mapping[str, Mapping[str, Any]],
    selected: Sequence[str],
    required_tags: Set[str],
) -> bool:
    covered = {str(tag) for region_id in selected for tag in regions[region_id]["feature_tags"]}
    return required_tags.issubset(covered)


def _canonical_selection(values: Sequence[str], order: Sequence[str]) -> Tuple[str, ...]:
    selected = set(values)
    if not selected or selected.difference(order):
        raise SchemaError("active physical selection is empty or references unknown regions")
    return tuple(region_id for region_id in order if region_id in selected)


def _selected_nodes(regions: Mapping[str, Mapping[str, Any]], selected: Sequence[str]) -> Set[int]:
    return {int(value) for region_id in selected for value in regions[region_id]["node_ids"]}


def _encode_selection(
    trace: ChakraExecutionTrace,
    regions: Mapping[str, Mapping[str, Any]],
    selected: Sequence[str],
) -> Tuple[bytes, Tuple[int, ...]]:
    return encode_chakra_subgraph(trace, _selected_nodes(regions, selected))


def _encode_communication_only_selection(
    trace: ChakraExecutionTrace,
    regions: Mapping[str, Mapping[str, Any]],
    selected: Sequence[str],
) -> Tuple[bytes, Tuple[int, ...]]:
    collective_nodes = []
    for region_id in selected:
        raw_node_ids = regions[region_id]["node_ids"]
        if not isinstance(raw_node_ids, list) or not raw_node_ids:
            raise SchemaError("communication-only baseline region has no nodes")
        collective_nodes.append(int(raw_node_ids[0]))
    encoded, closure = encode_chakra_subgraph(trace, collective_nodes)
    if set(closure) != set(collective_nodes):
        raise SchemaError("communication-only baseline has a non-collective dependency")
    return encoded, closure


def _unique_strings(values: Any, label: str) -> Tuple[str, ...]:
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise SchemaError(f"{label} must be non-empty strings")
    if len(values) != len(set(values)):
        raise SchemaError(f"{label} must be unique")
    return tuple(values)


__all__ = [
    "ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT",
    "ActivePhysicalSynthesisResult",
    "HoldoutApplicationBatch",
    "PhysicalCandidateRequest",
    "SelectionFreezer",
    "synthesize_active_physical_canary",
    "validate_active_physical_study_ledger",
]
