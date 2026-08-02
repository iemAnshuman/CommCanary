"""Portable hardware-qualification request manifest contract."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from ..errors import SchemaError
from ..formats import (
    CANARY_FORMAT,
    FIDELITY_VERIFICATION_FORMAT,
    QUALIFICATION_POLICY_FORMAT,
    QUALIFICATION_REQUEST_FORMAT,
    QUALIFICATION_REQUEST_V1_FORMAT,
    TRACE_FORMAT,
)
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    validate_json_mapping,
)
from .dtypes import require_param_dtype
from .json_codec import canonical_json_bytes
from .wire import (
    SUPPORTED_REDUCTION_OPS,
    as_int,
    require_format,
    validate_nonempty_string,
    validate_sha256,
)

QUALIFICATION_REQUEST_FILENAME = "qualification-request.json"
QUALIFICATION_PROGRAM_ENCODING = "commcanary.source-bound-compute-recipe.v2"
QUALIFICATION_EXECUTOR_CONTRACT = "async-issue-rank-gemm-recipe-explicit-wait.v2"
QUALIFICATION_COMPUTE_RECIPE_METHOD = "explicit-wait-linked-contiguous-gemm.v1"
QUALIFICATION_COMPUTE_RECIPE_PROJECTION = "commcanary.compute-recipe-projection.v1"
QUALIFICATION_EXECUTION_ADAPTER = "conforming-adapter-required"
QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY = "not_claimed"
QUALIFICATION_ARTIFACT_PATHS_V1 = {
    "source_trace": "source.trace.json",
    "canary": "canary.json",
    "fidelity_verification": "fidelity.json",
}
QUALIFICATION_ARTIFACT_FORMATS_V1 = {
    "source_trace": TRACE_FORMAT,
    "canary": CANARY_FORMAT,
    "fidelity_verification": FIDELITY_VERIFICATION_FORMAT,
}
QUALIFICATION_ARTIFACT_PATHS = {
    **QUALIFICATION_ARTIFACT_PATHS_V1,
    "qualification_policy": "qualification-policy.json",
}
QUALIFICATION_ARTIFACT_FORMATS = {
    **QUALIFICATION_ARTIFACT_FORMATS_V1,
    "qualification_policy": QUALIFICATION_POLICY_FORMAT,
}

_TOP_LEVEL_FIELDS = {
    "format",
    "request_id",
    "purpose",
    "producer",
    "claims",
    "artifacts",
    "bindings",
    "target_execution",
}
_V2_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS | {"decision_policy"}
_BINDING_FIELDS = {
    "source_trace_sha256",
    "source_normalized_sha256",
    "execution_semantic_sha256",
    "scheduler_execution_sha256",
    "calibration_evaluation_sha256",
    "artifact_provenance_sha256",
}


def qualification_request_sha256(request: Mapping[str, Any]) -> str:
    """Identify a request by canonical content excluding its own identifier."""

    stable = {key: value for key, value in request.items() if key != "request_id"}
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def validate_qualification_request(
    request: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Validate one portable owner-to-lab request without reading its files."""

    try:
        validate_json_mapping(request, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"qualification request violates JSON resource constraints: {exc}") from exc
    request_format = request.get("format")
    if request_format not in {QUALIFICATION_REQUEST_V1_FORMAT, QUALIFICATION_REQUEST_FORMAT}:
        require_format(request, QUALIFICATION_REQUEST_FORMAT, "qualification request")
    is_current = request_format == QUALIFICATION_REQUEST_FORMAT
    _require_exact_fields(
        request,
        _V2_TOP_LEVEL_FIELDS if is_current else _TOP_LEVEL_FIELDS,
        "qualification request",
    )
    validate_sha256(request.get("request_id"), "qualification request request_id")
    if qualification_request_sha256(request) != request.get("request_id"):
        raise SchemaError("qualification request request_id does not match canonical request content")
    if request.get("purpose") != "hardware-vendor-qualification":
        raise SchemaError("qualification request purpose is unsupported")

    producer = _mapping(request.get("producer"), "qualification request producer")
    _require_exact_fields(producer, {"name", "version"}, "qualification request producer")
    if producer.get("name") != "commcanary":
        raise SchemaError("qualification request producer.name must be 'commcanary'")
    validate_nonempty_string(producer.get("version"), "qualification request producer.version")

    claims = _mapping(request.get("claims"), "qualification request claims")
    expected_claims = {
        "source_correspondence": "source_verified",
        "physical_measurement": "not_included",
        "physical_fidelity": "unproven",
        "qualification_verdict": "policy_bound_not_issued" if is_current else "not_issued",
    }
    if dict(claims) != expected_claims:
        raise SchemaError("qualification request claims must preserve the request-only assurance boundary")

    artifacts = _mapping(request.get("artifacts"), "qualification request artifacts")
    artifact_paths = QUALIFICATION_ARTIFACT_PATHS if is_current else QUALIFICATION_ARTIFACT_PATHS_V1
    artifact_formats = QUALIFICATION_ARTIFACT_FORMATS if is_current else QUALIFICATION_ARTIFACT_FORMATS_V1
    _require_exact_fields(
        artifacts,
        set(artifact_paths),
        "qualification request artifacts",
    )
    for artifact_id in sorted(artifact_paths):
        reference = _mapping(
            artifacts.get(artifact_id),
            f"qualification request artifacts.{artifact_id}",
        )
        _require_exact_fields(
            reference,
            {"path", "format", "sha256", "size_bytes"},
            f"qualification request artifacts.{artifact_id}",
        )
        if reference.get("path") != artifact_paths[artifact_id]:
            raise SchemaError(f"qualification request artifacts.{artifact_id}.path is not canonical")
        if reference.get("format") != artifact_formats[artifact_id]:
            raise SchemaError(f"qualification request artifacts.{artifact_id}.format is unsupported")
        validate_sha256(
            reference.get("sha256"),
            f"qualification request artifacts.{artifact_id}.sha256",
        )
        if as_int(reference.get("size_bytes")) <= 0:
            raise SchemaError(f"qualification request artifacts.{artifact_id}.size_bytes must be positive")

    bindings = _mapping(request.get("bindings"), "qualification request bindings")
    _require_exact_fields(bindings, _BINDING_FIELDS, "qualification request bindings")
    for field in sorted(_BINDING_FIELDS):
        validate_sha256(bindings.get(field), f"qualification request bindings.{field}")

    if is_current:
        decision_policy = _mapping(request.get("decision_policy"), "qualification request decision_policy")
        _require_exact_fields(
            decision_policy,
            {"policy_id", "policy_format", "application", "outcomes"},
            "qualification request decision_policy",
        )
        validate_sha256(decision_policy.get("policy_id"), "qualification request decision_policy.policy_id")
        if decision_policy.get("policy_format") != QUALIFICATION_POLICY_FORMAT:
            raise SchemaError("qualification request decision_policy.policy_format is unsupported")
        if decision_policy.get("application") != "required_before_execution":
            raise SchemaError("qualification request decision_policy.application is unsupported")
        if decision_policy.get("outcomes") != ["fail", "incomparable", "inconclusive", "pass"]:
            raise SchemaError("qualification request decision_policy.outcomes must name the canonical four states")

    target = _mapping(request.get("target_execution"), "qualification request target_execution")
    base_target_fields = {
            "materialization",
            "program_encoding",
            "executor_contract",
            "execution_adapter",
            "upstream_param_compatibility",
            "communication_dtype_source",
            "communication_dtypes",
            "communication_reduction_source",
            "communication_reduction_ops",
            "communication_message_shape_source",
            "all_to_all_split_policy",
            "rank_arrival_timing",
            "compute_work_source",
            "compute_recipe_method",
            "compute_recipe_projection_sha256",
            "compute_recipe_event_count",
            "compute_recipe_operation_count",
            "target_compute_calibration",
            "source_overlap_observation",
            "overlap_structure",
            "inflight_communication_policy",
            "timestamp_pacing",
            "privacy_disclosure",
            "physical_observation",
    }
    inventory_fields = {
        "communication_inventory_source",
        "communication_operations",
        "communication_message_shapes",
    }
    allowed_target_fields = {
        frozenset(base_target_fields),
        frozenset(base_target_fields | inventory_fields),
    }
    if frozenset(target) not in allowed_target_fields:
        _require_exact_fields(
            target,
            base_target_fields | inventory_fields,
            "qualification request target_execution",
        )
    expected_fixed_target = {
        "materialization": "deterministic_from_verified_request",
        "program_encoding": QUALIFICATION_PROGRAM_ENCODING,
        "executor_contract": QUALIFICATION_EXECUTOR_CONTRACT,
        "execution_adapter": QUALIFICATION_EXECUTION_ADAPTER,
        "upstream_param_compatibility": QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
        "communication_dtype_source": "source-bound-per-event",
        "communication_reduction_source": "source-bound-per-event",
        "communication_message_shape_source": "source-validated-per-event",
        "all_to_all_split_policy": "equal-split-only",
        "rank_arrival_timing": "emerges-from-source-bound-rank-local-work",
        "compute_work_source": "source-bound-per-rank-exact-recipe",
        "compute_recipe_method": "explicit-wait-linked-contiguous-gemm.v1",
        "target_compute_calibration": "not_used",
        "source_overlap_observation": "bound-not-duration-paced",
        "overlap_structure": "async-issue-exact-rank-work-explicit-wait",
        "inflight_communication_policy": "single-collective",
        "timestamp_pacing": "disabled",
        "privacy_disclosure": "gemm-shapes-and-dtypes-revealed",
        "physical_observation": "required_before_qualification_verdict",
    }
    for field, expected in expected_fixed_target.items():
        if target.get(field) != expected:
            raise SchemaError(f"qualification request target_execution.{field} is unsupported")
    raw_dtypes = target.get("communication_dtypes")
    if not isinstance(raw_dtypes, list) or not raw_dtypes:
        raise SchemaError("qualification request target_execution.communication_dtypes must be a non-empty list")
    communication_dtypes = [
        require_param_dtype(value, label="qualification request communication dtype") for value in raw_dtypes
    ]
    if communication_dtypes != sorted(set(communication_dtypes)):
        raise SchemaError("qualification request target_execution.communication_dtypes must be sorted and unique")
    raw_reduction_ops = target.get("communication_reduction_ops")
    if not isinstance(raw_reduction_ops, list):
        raise SchemaError("qualification request target_execution.communication_reduction_ops must be a list")
    if any(not isinstance(value, str) or value not in SUPPORTED_REDUCTION_OPS for value in raw_reduction_ops):
        raise SchemaError(
            "qualification request target_execution.communication_reduction_ops contains an unsupported operator"
        )
    if raw_reduction_ops != sorted(set(raw_reduction_ops)):
        raise SchemaError(
            "qualification request target_execution.communication_reduction_ops must be sorted and unique"
        )
    if inventory_fields <= set(target):
        if target.get("communication_inventory_source") != "full-generated-program":
            raise SchemaError(
                "qualification request target_execution.communication_inventory_source is unsupported"
            )
        raw_operations = target.get("communication_operations")
        supported_operations = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast"}
        if (
            not isinstance(raw_operations, list)
            or not raw_operations
            or any(not isinstance(value, str) or value not in supported_operations for value in raw_operations)
            or raw_operations != sorted(set(raw_operations))
        ):
            raise SchemaError(
                "qualification request target_execution.communication_operations must be a sorted unique list"
            )
        raw_shapes = target.get("communication_message_shapes")
        if not isinstance(raw_shapes, list) or not raw_shapes:
            raise SchemaError(
                "qualification request target_execution.communication_message_shapes must be a non-empty list"
            )
        normalized_shapes = []
        for index, raw_shape in enumerate(raw_shapes):
            shape = _mapping(
                raw_shape,
                f"qualification request target_execution.communication_message_shapes[{index}]",
            )
            _require_exact_fields(
                shape,
                {"operation", "dtype", "world_size", "in_msg_size", "out_msg_size"},
                f"qualification request target_execution.communication_message_shapes[{index}]",
            )
            operation = shape.get("operation")
            if not isinstance(operation, str) or operation not in supported_operations:
                raise SchemaError("qualification request communication message shape operation is unsupported")
            dtype = require_param_dtype(
                shape.get("dtype"),
                label="qualification request communication message shape dtype",
            )
            world_size = as_int(shape.get("world_size"))
            in_msg_size = as_int(shape.get("in_msg_size"))
            out_msg_size = as_int(shape.get("out_msg_size"))
            if world_size <= 0 or in_msg_size <= 0 or out_msg_size <= 0:
                raise SchemaError("qualification request communication message shapes must be positive")
            normalized_shapes.append((operation, dtype, world_size, in_msg_size, out_msg_size))
        if normalized_shapes != sorted(set(normalized_shapes)):
            raise SchemaError(
                "qualification request target_execution.communication_message_shapes must be sorted and unique"
            )
    validate_sha256(
        target.get("compute_recipe_projection_sha256"),
        "qualification request target_execution.compute_recipe_projection_sha256",
    )
    recipe_event_count = as_int(target.get("compute_recipe_event_count"))
    if recipe_event_count <= 0:
        raise SchemaError("qualification request target_execution.compute_recipe_event_count must be positive")
    if recipe_event_count > limits.max_stored_events:
        raise SchemaError(
            "qualification request target_execution.compute_recipe_event_count exceeds "
            f"max_stored_events={limits.max_stored_events}"
        )
    recipe_operation_count = as_int(target.get("compute_recipe_operation_count"))
    if recipe_operation_count < 0:
        raise SchemaError("qualification request target_execution.compute_recipe_operation_count must be non-negative")
    if recipe_operation_count > limits.max_param_compute_operations:
        raise SchemaError(
            "qualification request target_execution.compute_recipe_operation_count exceeds "
            f"max_param_compute_operations={limits.max_param_compute_operations}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise SchemaError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{label} contains unsupported fields: {', '.join(unknown)}")


__all__ = [
    "QUALIFICATION_ARTIFACT_FORMATS",
    "QUALIFICATION_ARTIFACT_PATHS",
    "QUALIFICATION_ARTIFACT_FORMATS_V1",
    "QUALIFICATION_ARTIFACT_PATHS_V1",
    "QUALIFICATION_COMPUTE_RECIPE_METHOD",
    "QUALIFICATION_COMPUTE_RECIPE_PROJECTION",
    "QUALIFICATION_EXECUTION_ADAPTER",
    "QUALIFICATION_EXECUTOR_CONTRACT",
    "QUALIFICATION_PROGRAM_ENCODING",
    "QUALIFICATION_REQUEST_FILENAME",
    "QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY",
    "qualification_request_sha256",
    "validate_qualification_request",
]
