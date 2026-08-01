"""Fail-closed qualification-readiness diagnostics for imported traces."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

from ..artifacts import canonical_json_bytes, validate_trace
from ..artifacts.dtypes import PARAM_COMPUTE_DTYPES, PARAM_DTYPES
from ..artifacts.param import PARAM_COLLECTIVE_OP_NAMES, param_materialization_requirements
from ..artifacts.qualification import QUALIFICATION_COMPUTE_RECIPE_METHOD
from ..artifacts.qualification_program import (
    qualification_compute_recipe_audit,
    qualification_compute_tensor_bytes,
)
from ..artifacts.wire import JsonDict, as_int, normalize_ranks
from ..errors import CommCanaryError, SchemaError
from ..formats import DOCTOR_REPORT_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..verification.canary import verify_canary_fidelity
from .compile import compile_trace

_SUPPORTED_EXACT_WORK_OPS = frozenset(PARAM_COLLECTIVE_OP_NAMES)
_REDUCTION_OPS = frozenset({"all_reduce", "reduce_scatter"})


def qualification_readiness_report(
    trace: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Diagnose whether one imported trace can enter the qualification workflow."""

    validate_trace(trace, limits=limits)
    raw_events = trace.get("events")
    events = list(raw_events) if isinstance(raw_events, list) else []
    if not events:
        raise SchemaError("doctor requires at least one imported collective event")

    checks: List[JsonDict] = []
    checks.append(_coverage_check("known_overlap", events, _known_overlap, "unknown_compute_overlap"))
    checks.append(
        _coverage_check(
            "exact_message_shapes",
            events,
            _exact_message_shape,
            "unknown_message_shape",
        )
    )
    checks.append(
        _coverage_check(
            "communication_dtypes",
            events,
            _supported_communication_dtype,
            "unsupported_or_missing_communication_dtype",
        )
    )
    checks.append(
        _coverage_check(
            "collective_semantics",
            events,
            _known_collective_semantics,
            "unknown_collective_semantics",
        )
    )
    checks.append(
        _coverage_check(
            "rank_local_compute_recipes",
            events,
            _complete_compute_recipe,
            "missing_or_unsupported_rank_local_compute_recipe",
        )
    )
    checks.append(_process_group_check(events))
    checks.append(_rank_domain_check(events))
    checks.append(_source_inventory_check(trace, events))

    canary: Optional[JsonDict] = None
    fidelity: Optional[JsonDict] = None
    try:
        canary = compile_trace(
            trace,
            require_lossless_timing=True,
            limits=limits,
        )
        fidelity = verify_canary_fidelity(trace, canary, limits=limits)
        compiler_passed = fidelity.get("status") == "source_verified"
        compiler_detail = None if compiler_passed else "lossless canary did not verify against its source"
    except CommCanaryError as exc:
        compiler_passed = False
        compiler_detail = str(exc)
    checks.append(
        _binary_check(
            "lossless_source_compilation",
            compiler_passed,
            "source_trace_not_losslessly_compilable",
            detail=compiler_detail,
        )
    )

    materialization_detail: Optional[str] = None
    materialization_passed = False
    audit: Optional[JsonDict] = None
    if canary is not None and compiler_passed:
        try:
            param_materialization_requirements(
                canary,
                require_event_dtype=True,
                require_reduction_op=True,
                limits=limits,
            )
            audit = qualification_compute_recipe_audit(trace, limits=limits)
            materialization_passed = True
        except CommCanaryError as exc:
            materialization_detail = str(exc)
    else:
        materialization_detail = "lossless source compilation did not pass"
    checks.append(
        _binary_check(
            "exact_work_materialization",
            materialization_passed,
            "qualification_materialization_refused",
            detail=materialization_detail,
        )
    )

    required = [check for check in checks if check["required"]]
    passed_required = sum(check["status"] == "pass" for check in required)
    qualification_ready = passed_required == len(required)
    simulatable = compiler_passed
    tier = "qualification_ready" if qualification_ready else "simulatable" if simulatable else "observational"

    report: JsonDict = {
        "format": DOCTOR_REPORT_FORMAT,
        "status": "qualification_ready" if qualification_ready else "not_qualification_ready",
        "supported_product_tier": tier,
        "summary": {
            "event_count": len(events),
            "required_checks": len(required),
            "passed_required_checks": passed_required,
            "failed_required_checks": len(required) - passed_required,
            "warning_checks": sum(check["status"] == "warn" for check in checks),
            "qualification_readiness_pct": round(100.0 * passed_required / len(required), 1),
        },
        "checks": checks,
        "estimates": _estimates(trace, canary=canary, fidelity=fidelity, audit=audit),
        "privacy": _privacy_report(events),
        "next_actions": _next_actions(checks),
        "claims": {
            "source_correspondence": "verified" if compiler_passed else "not_verified",
            "physical_execution": "not_observed",
            "physical_conformance": "unproven",
            "physical_decision_fidelity": "not_measured",
            "producer_authenticity": "unsigned",
        },
    }
    validate_doctor_report(report)
    return report


def import_failure_readiness_report(error: CommCanaryError) -> JsonDict:
    """Return a bounded diagnostic when source profiles cannot be imported."""

    reason_code = _classify_import_failure(str(error))
    report: JsonDict = {
        "format": DOCTOR_REPORT_FORMAT,
        "status": "not_importable",
        "supported_product_tier": "not_importable",
        "summary": {
            "event_count": 0,
            "required_checks": 1,
            "passed_required_checks": 0,
            "failed_required_checks": 1,
            "warning_checks": 0,
            "qualification_readiness_pct": 0.0,
        },
        "checks": [
            {
                "check_id": "profile_import",
                "status": "fail",
                "required": True,
                "reason_code": reason_code,
                "message": "Profiler evidence could not be imported.",
                "coverage": {"passed": 0, "total": 1, "pct": 0.0},
                "locations": [],
                "detail": str(error),
            }
        ],
        "estimates": {
            "qualification_bundle": {
                "status": "not_estimable",
                "reason_code": "profile_import_failed",
            },
            "execution_memory": {
                "status": "not_estimable",
                "reason_code": "profile_import_failed",
            },
            "physical_runtime": {
                "status": "not_estimable",
                "reason_code": "target_measurements_required",
            },
        },
        "privacy": {
            "disclosure_level": "not_assessed",
            "exposed": [],
            "not_exposed_by_bundle": [],
            "note": "No bundle was constructed because profile import failed.",
        },
        "next_actions": [_next_action(reason_code)],
        "claims": {
            "source_correspondence": "not_verified",
            "physical_execution": "not_observed",
            "physical_conformance": "unproven",
            "physical_decision_fidelity": "not_measured",
            "producer_authenticity": "unsigned",
        },
    }
    validate_doctor_report(report)
    return report


def validate_doctor_report(report: Mapping[str, Any]) -> None:
    """Validate the closed machine-readable doctor report contract."""

    if report.get("format") != DOCTOR_REPORT_FORMAT:
        raise SchemaError(f"doctor report format must be {DOCTOR_REPORT_FORMAT!r}")
    expected = {
        "format",
        "status",
        "supported_product_tier",
        "summary",
        "checks",
        "estimates",
        "privacy",
        "next_actions",
        "claims",
    }
    if set(report) != expected:
        raise SchemaError("doctor report fields do not match the supported contract")
    if report.get("status") not in {"qualification_ready", "not_qualification_ready", "not_importable"}:
        raise SchemaError("doctor report status is unsupported")
    if report.get("supported_product_tier") not in {
        "qualification_ready",
        "simulatable",
        "observational",
        "not_importable",
    }:
        raise SchemaError("doctor report supported_product_tier is unsupported")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise SchemaError("doctor report checks must be a non-empty array")
    seen = set()
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            raise SchemaError(f"doctor report check {index} must be an object")
        required_fields = {"check_id", "status", "required", "reason_code", "message", "coverage", "locations"}
        if not required_fields.issubset(check) or set(check) - (required_fields | {"detail"}):
            raise SchemaError(f"doctor report check {index} fields do not match the supported contract")
        check_id = check.get("check_id")
        if not isinstance(check_id, str) or not check_id or check_id in seen:
            raise SchemaError("doctor report check_id values must be non-empty and unique")
        seen.add(check_id)
        if check.get("status") not in {"pass", "fail", "warn"}:
            raise SchemaError(f"doctor report check {index} status is unsupported")
        if not isinstance(check.get("required"), bool):
            raise SchemaError(f"doctor report check {index} required must be boolean")
    next_actions = report.get("next_actions")
    if not isinstance(next_actions, list) or any(not isinstance(item, str) or not item for item in next_actions):
        raise SchemaError("doctor report next_actions must contain non-empty strings")


def _coverage_check(
    check_id: str,
    events: Sequence[Mapping[str, Any]],
    predicate: Any,
    reason_code: str,
) -> JsonDict:
    failed = [(index, event) for index, event in enumerate(events) if not predicate(event)]
    passed = len(events) - len(failed)
    return {
        "check_id": check_id,
        "status": "pass" if not failed else "fail",
        "required": True,
        "reason_code": "ready" if not failed else reason_code,
        "message": (f"{passed}/{len(events)} event(s) satisfy {check_id.replace('_', ' ')}."),
        "coverage": _coverage(passed, len(events)),
        "locations": [_event_location(index, event) for index, event in failed],
    }


def _binary_check(
    check_id: str,
    passed: bool,
    failure_reason: str,
    *,
    detail: Optional[str] = None,
) -> JsonDict:
    result: JsonDict = {
        "check_id": check_id,
        "status": "pass" if passed else "fail",
        "required": True,
        "reason_code": "ready" if passed else failure_reason,
        "message": f"{check_id.replace('_', ' ')} {'passed' if passed else 'failed'}.",
        "coverage": _coverage(1 if passed else 0, 1),
        "locations": [],
    }
    if detail:
        result["detail"] = detail
    return result


def _process_group_check(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    memberships: dict[str, tuple[int, ...]] = {}
    failed: List[tuple[int, Mapping[str, Any]]] = []
    for index, event in enumerate(events):
        group = str(event.get("group", "default"))
        ranks = tuple(normalize_ranks(event.get("ranks")))
        if group in memberships and memberships[group] != ranks:
            failed.append((index, event))
        else:
            memberships[group] = ranks
    return {
        "check_id": "process_group_membership",
        "status": "pass" if not failed else "fail",
        "required": True,
        "reason_code": "ready" if not failed else "inconsistent_process_group_membership",
        "message": f"{len(events) - len(failed)}/{len(events)} event(s) preserve stable process-group membership.",
        "coverage": _coverage(len(events) - len(failed), len(events)),
        "locations": [_event_location(index, event) for index, event in failed],
    }


def _rank_domain_check(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    ranks = sorted({rank for event in events for rank in normalize_ranks(event.get("ranks"))})
    passed = bool(ranks) and ranks == list(range(max(ranks) + 1))
    result = _binary_check("dense_rank_domain", passed, "sparse_or_nonzero_rank_domain")
    if not passed:
        result["detail"] = f"observed ranks: {ranks}"
    return result


def _source_inventory_check(trace: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> JsonDict:
    system = trace.get("system")
    profiles = system.get("kineto_source_profiles") if isinstance(system, Mapping) else None
    if not isinstance(profiles, list):
        return {
            "check_id": "source_profile_inventory",
            "status": "warn",
            "required": False,
            "reason_code": "source_profile_inventory_unavailable",
            "message": "Source-profile byte identities are not recorded in this in-memory trace.",
            "coverage": _coverage(0, 1),
            "locations": [],
        }
    expected = sorted({rank for event in events for rank in normalize_ranks(event.get("ranks"))})
    observed = sorted(
        as_int(profile["rank"]) for profile in profiles if isinstance(profile, Mapping) and "rank" in profile
    )
    passed = observed == expected
    result = _binary_check(
        "source_profile_inventory",
        passed,
        "incomplete_rank_profile_inventory",
    )
    if not passed:
        result["detail"] = f"expected ranks {expected}; observed source profiles {observed}"
    return result


def _known_overlap(event: Mapping[str, Any]) -> bool:
    return "compute_overlap_us" in event and event.get("compute_overlap_unknown") is not True


def _exact_message_shape(event: Mapping[str, Any]) -> bool:
    metadata = event.get("metadata")
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get("kineto_message_shape_status") == "derived"
        and metadata.get("kineto_message_shape_method") == "record-param-comms-in-out-nelems.v1"
    )


def _supported_communication_dtype(event: Mapping[str, Any]) -> bool:
    return event.get("dtype") in PARAM_DTYPES


def _known_collective_semantics(event: Mapping[str, Any]) -> bool:
    operation = event.get("op")
    if operation not in _SUPPORTED_EXACT_WORK_OPS:
        return False
    if operation in _REDUCTION_OPS and not isinstance(event.get("reduction_op"), str):
        return False
    if operation == "broadcast" and not isinstance(event.get("root_rank"), int):
        return False
    return True


def _complete_compute_recipe(event: Mapping[str, Any]) -> bool:
    metadata = event.get("metadata")
    ranks = normalize_ranks(event.get("ranks"))
    recipes = event.get("compute_recipe_by_rank")
    if not (
        isinstance(metadata, Mapping)
        and metadata.get("kineto_compute_recipe_status") == "derived"
        and metadata.get("kineto_compute_recipe_method") == QUALIFICATION_COMPUTE_RECIPE_METHOD
        and isinstance(recipes, Mapping)
        and set(recipes) == {str(rank) for rank in ranks}
    ):
        return False
    return all(
        isinstance(recipes[str(rank)], list)
        and all(
            isinstance(operation, Mapping) and operation.get("dtype") in PARAM_COMPUTE_DTYPES
            for operation in recipes[str(rank)]
        )
        for rank in ranks
    )


def _estimates(
    trace: Mapping[str, Any],
    *,
    canary: Optional[Mapping[str, Any]],
    fidelity: Optional[Mapping[str, Any]],
    audit: Optional[Mapping[str, Any]],
) -> JsonDict:
    bundle: JsonDict
    if canary is None or fidelity is None:
        bundle = {"status": "not_estimable", "reason_code": "lossless_compilation_required"}
    else:
        component_bytes = {
            "source_trace": len(canonical_json_bytes(trace)),
            "canary": len(canonical_json_bytes(canary)),
            "source_proof": len(canonical_json_bytes(fidelity)),
        }
        bundle = {
            "status": "lower_bound",
            "canonical_payload_bytes": sum(component_bytes.values()),
            "component_canonical_bytes": component_bytes,
            "excludes": ["request_manifest", "pretty_print_whitespace", "filesystem_metadata", "signatures"],
        }

    rank_bytes: dict[int, int] = {}
    events = trace.get("events")
    if audit is not None and isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping) or not isinstance(event.get("compute_recipe_by_rank"), Mapping):
                continue
            for rank_text, operations in event["compute_recipe_by_rank"].items():
                if not isinstance(operations, list):
                    continue
                rank = int(rank_text)
                for operation in operations:
                    if isinstance(operation, Mapping):
                        rank_bytes[rank] = rank_bytes.get(rank, 0) + sum(qualification_compute_tensor_bytes(operation))
    memory: JsonDict
    if rank_bytes:
        memory = {
            "status": "upper_bound_for_compute_recipe_tensors",
            "rank_bytes": {str(rank): rank_bytes[rank] for rank in sorted(rank_bytes)},
            "max_rank_bytes": max(rank_bytes.values()),
            "excludes": ["communication_buffers", "framework_allocator_overhead", "runtime_workspace"],
        }
    else:
        memory = {"status": "not_estimable", "reason_code": "exact_compute_recipe_required"}
    return {
        "qualification_bundle": bundle,
        "execution_memory": memory,
        "physical_runtime": {
            "status": "not_estimable",
            "reason_code": "target_measurements_required",
            "note": "Source kernel durations are evidence, not a target-runtime prediction.",
        },
    }


def _privacy_report(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    gemm_shapes = set()
    compute_dtypes = set()
    for event in events:
        recipes = event.get("compute_recipe_by_rank")
        if not isinstance(recipes, Mapping):
            continue
        for operations in recipes.values():
            if not isinstance(operations, list):
                continue
            for operation in operations:
                if isinstance(operation, Mapping):
                    gemm_shapes.add((operation.get("m"), operation.get("n"), operation.get("k")))
                    compute_dtypes.add(str(operation.get("dtype")))
    return {
        "disclosure_level": "high",
        "exposed": [
            "world_size_and_process_groups",
            "ordered_collective_sequence",
            "exact_message_sizes",
            "relative_event_timing_and_overlap",
            "rank_local_gemm_shapes",
            "communication_and_compute_dtypes",
        ],
        "not_exposed_by_bundle": [
            "tensor_values",
            "model_weights",
            "prompts",
            "source_code",
            "dataset_records",
        ],
        "distinct_gemm_shapes": len(gemm_shapes),
        "compute_dtypes": sorted(compute_dtypes),
        "note": "Structural metadata can reveal architecture family or parallelism strategy.",
    }


def _coverage(passed: int, total: int) -> JsonDict:
    return {"passed": passed, "total": total, "pct": round(100.0 * passed / total, 1) if total else 0.0}


def _event_location(index: int, event: Mapping[str, Any]) -> JsonDict:
    location: JsonDict = {"event_index": index, "ranks": normalize_ranks(event.get("ranks"))}
    if isinstance(event.get("id"), str):
        location["event_id"] = event["id"]
    return location


def _classify_import_failure(message: str) -> str:
    lowered = message.lower()
    classifiers: Iterable[tuple[str, str]] = (
        ("clock", "clock_alignment_unproven"),
        ("process group", "process_group_evidence_invalid"),
        ("rank", "rank_contribution_evidence_invalid"),
        ("input", "profile_input_invalid"),
        ("json", "profile_json_invalid"),
        ("collective", "collective_evidence_unavailable"),
    )
    for needle, reason_code in classifiers:
        if needle in lowered:
            return reason_code
    return "profile_import_failed"


def _next_actions(checks: Sequence[Mapping[str, Any]]) -> List[str]:
    actions: List[str] = []
    for check in checks:
        if check.get("status") == "pass":
            continue
        action = _next_action(str(check.get("reason_code")))
        if action not in actions:
            actions.append(action)
    if not actions:
        actions.append("Build and review a qualification request before target execution.")
    return actions


def _next_action(reason_code: str) -> str:
    return {
        "clock_alignment_unproven": "Recapture with a shared clock or provide one explicit offset for every rank.",
        "unknown_compute_overlap": "Capture linked communication and compute kernels so overlap is observable.",
        "unknown_message_shape": "Enable record_param_comms input/output element and split-size metadata.",
        "unknown_collective_semantics": "Capture reduction operators and broadcast roots, and remove unsupported collectives.",
        "unsupported_or_missing_communication_dtype": "Capture a canonical communication dtype supported by the executor.",
        "missing_or_unsupported_rank_local_compute_recipe": (
            "Capture same-thread asynchronous issue, contiguous supported GEMMs, and an explicit linked wait on every rank."
        ),
        "inconsistent_process_group_membership": "Use one stable rank membership for each process-group identity.",
        "sparse_or_nonzero_rank_domain": "Capture a complete dense global rank domain starting at rank zero.",
        "incomplete_rank_profile_inventory": "Provide exactly one source profile for every participating global rank.",
        "source_profile_inventory_unavailable": "Retain source-profile byte identities when building the request.",
        "source_trace_not_losslessly_compilable": "Resolve the reported trace/compiler refusal before building a request.",
        "qualification_materialization_refused": "Resolve the exact-work materialization refusal before target execution.",
        "profile_json_invalid": "Export a complete bounded PyTorch Kineto JSON profile and retry.",
        "profile_input_invalid": "Check the profile paths, sizes, and JSON structure.",
        "process_group_evidence_invalid": "Capture consistent process-group names, ranks, and sizes on every rank.",
        "rank_contribution_evidence_invalid": "Provide one complete, uniquely owned profile contribution per rank.",
        "collective_evidence_unavailable": "Enable record_param_comms and capture at least one supported collective.",
        "profile_import_failed": "Inspect the import detail, correct the profiler evidence, and retry doctor.",
    }.get(reason_code, "Inspect the failed readiness check and recapture the missing source evidence.")


__all__ = [
    "import_failure_readiness_report",
    "qualification_readiness_report",
    "validate_doctor_report",
]
