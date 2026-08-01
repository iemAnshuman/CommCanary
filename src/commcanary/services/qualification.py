"""Prepare and independently verify portable qualification-request bundles."""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Tuple

from ..artifacts import (
    SENSITIVE_JSON_POLICY,
    atomic_write_json,
    load_json,
    validate_canary,
    validate_qualification_policy,
    validate_trace,
)
from ..artifacts.dtypes import dtype_size_bytes
from ..artifacts.param import param_materialization_requirements
from ..artifacts.qualification import (
    QUALIFICATION_ARTIFACT_FORMATS,
    QUALIFICATION_ARTIFACT_FORMATS_V1,
    QUALIFICATION_ARTIFACT_PATHS,
    QUALIFICATION_ARTIFACT_PATHS_V1,
    QUALIFICATION_EXECUTION_ADAPTER,
    QUALIFICATION_EXECUTOR_CONTRACT,
    QUALIFICATION_PROGRAM_ENCODING,
    QUALIFICATION_REQUEST_FILENAME,
    QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
    qualification_request_sha256,
    validate_qualification_request,
)
from ..artifacts.qualification_program import qualification_compute_recipe_audit
from ..artifacts.wire import JsonDict, as_int, normalize_ranks
from ..errors import CommCanaryIOError, SchemaError
from ..formats import QUALIFICATION_REQUEST_FORMAT, QUALIFICATION_REQUEST_V1_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..verification.canary import verify_canary_fidelity
from ..version import package_version

_IMMUTABLE_QUALIFICATION_JSON_POLICY = replace(
    SENSITIVE_JSON_POLICY,
    artifact_label="qualification bundle artifact",
    create_parents=False,
    overwrite=False,
)


def prepare_qualification_request(
    output_directory: str,
    trace: Mapping[str, Any],
    canary: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Write a new complete request directory, installing its manifest last."""

    validate_trace(trace, require_known_overlap=True, limits=limits)
    validate_canary(canary, limits=limits)
    validate_qualification_policy(policy, limits=limits)
    fidelity = verify_canary_fidelity(trace, canary, limits=limits)
    if fidelity.get("status") != "source_verified":
        raise SchemaError("qualification request requires source-verified canary fidelity")
    materialization = param_materialization_requirements(
        canary,
        require_event_dtype=True,
        require_reduction_op=True,
        limits=limits,
    )
    _validate_source_message_shapes(trace)
    compute_audit = qualification_compute_recipe_audit(trace, limits=limits)

    output = Path(output_directory)
    _create_new_bundle_directory(output)
    documents = {
        "source_trace": copy.deepcopy(dict(trace)),
        "canary": copy.deepcopy(dict(canary)),
        "fidelity_verification": fidelity,
        "qualification_policy": copy.deepcopy(dict(policy)),
    }
    references: JsonDict = {}
    for artifact_id in ("source_trace", "canary", "fidelity_verification", "qualification_policy"):
        path = output / QUALIFICATION_ARTIFACT_PATHS[artifact_id]
        atomic_write_json(
            path,
            documents[artifact_id],
            indent=2,
            policy=_IMMUTABLE_QUALIFICATION_JSON_POLICY,
        )
        sha256, size_bytes = _bounded_file_identity(path, limits=limits)
        references[artifact_id] = {
            "path": QUALIFICATION_ARTIFACT_PATHS[artifact_id],
            "format": QUALIFICATION_ARTIFACT_FORMATS[artifact_id],
            "sha256": sha256,
            "size_bytes": size_bytes,
        }

    compiler = canary.get("compiler")
    if not isinstance(compiler, Mapping):
        raise SchemaError("qualification canary compiler metadata must be an object")
    bindings = {
        field: compiler.get(field)
        for field in (
            "source_trace_sha256",
            "source_normalized_sha256",
            "execution_semantic_sha256",
            "scheduler_execution_sha256",
            "calibration_evaluation_sha256",
            "artifact_provenance_sha256",
        )
    }
    request: JsonDict = {
        "format": QUALIFICATION_REQUEST_FORMAT,
        "purpose": "hardware-vendor-qualification",
        "producer": {
            "name": "commcanary",
            "version": package_version(),
        },
        "claims": {
            "source_correspondence": "source_verified",
            "physical_measurement": "not_included",
            "physical_fidelity": "unproven",
            "qualification_verdict": "policy_bound_not_issued",
        },
        "artifacts": references,
        "bindings": bindings,
        "decision_policy": {
            "policy_id": policy["policy_id"],
            "policy_format": policy["format"],
            "application": "required_before_execution",
            "outcomes": ["fail", "incomparable", "inconclusive", "pass"],
        },
        "target_execution": {
            "materialization": "deterministic_from_verified_request",
            "program_encoding": QUALIFICATION_PROGRAM_ENCODING,
            "executor_contract": QUALIFICATION_EXECUTOR_CONTRACT,
            "execution_adapter": QUALIFICATION_EXECUTION_ADAPTER,
            "upstream_param_compatibility": QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
            "communication_dtype_source": "source-bound-per-event",
            "communication_dtypes": materialization["communication_dtypes"],
            "communication_reduction_source": "source-bound-per-event",
            "communication_reduction_ops": materialization["communication_reduction_ops"],
            "communication_message_shape_source": "source-validated-per-event",
            "all_to_all_split_policy": "equal-split-only",
            "rank_arrival_timing": "emerges-from-source-bound-rank-local-work",
            "compute_work_source": "source-bound-per-rank-exact-recipe",
            "compute_recipe_method": compute_audit["method"],
            "compute_recipe_projection_sha256": compute_audit["projection_sha256"],
            "compute_recipe_event_count": compute_audit["event_count"],
            "compute_recipe_operation_count": compute_audit["operation_count"],
            "target_compute_calibration": "not_used",
            "source_overlap_observation": "bound-not-duration-paced",
            "overlap_structure": "async-issue-exact-rank-work-explicit-wait",
            "inflight_communication_policy": "single-collective",
            "timestamp_pacing": "disabled",
            "privacy_disclosure": "gemm-shapes-and-dtypes-revealed",
            "physical_observation": "required_before_qualification_verdict",
        },
    }
    request["request_id"] = qualification_request_sha256(request)
    validate_qualification_request(request, limits=limits)
    atomic_write_json(
        output / QUALIFICATION_REQUEST_FILENAME,
        request,
        indent=2,
        policy=_IMMUTABLE_QUALIFICATION_JSON_POLICY,
    )
    return copy.deepcopy(request)


def verify_qualification_request(
    bundle_directory: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Verify exact inventory, bytes, semantic bindings, and source fidelity."""

    directory = Path(bundle_directory)
    _validate_bundle_directory(directory)
    manifest_path = directory / QUALIFICATION_REQUEST_FILENAME
    _require_regular_file(manifest_path)
    request = load_json(str(manifest_path), limits=limits)
    validate_qualification_request(request, limits=limits)
    is_current = request["format"] == QUALIFICATION_REQUEST_FORMAT
    if request["format"] not in {QUALIFICATION_REQUEST_FORMAT, QUALIFICATION_REQUEST_V1_FORMAT}:
        raise SchemaError("qualification bundle request format is unsupported")
    artifact_paths = QUALIFICATION_ARTIFACT_PATHS if is_current else QUALIFICATION_ARTIFACT_PATHS_V1
    artifact_formats = QUALIFICATION_ARTIFACT_FORMATS if is_current else QUALIFICATION_ARTIFACT_FORMATS_V1
    expected_names = {QUALIFICATION_REQUEST_FILENAME, *artifact_paths.values()}
    observed_names = {entry.name for entry in directory.iterdir()}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise SchemaError(f"qualification bundle inventory mismatch: missing={missing!r}, unexpected={unexpected!r}")
    for name in sorted(expected_names):
        _require_regular_file(directory / name)

    artifacts = request["artifacts"]
    loaded: dict[str, JsonDict] = {}
    for artifact_id in artifact_paths:
        reference = artifacts[artifact_id]
        path = directory / artifact_paths[artifact_id]
        observed_sha256, observed_size = _bounded_file_identity(path, limits=limits)
        if observed_sha256 != reference["sha256"] or observed_size != reference["size_bytes"]:
            raise SchemaError(f"qualification bundle artifact {artifact_id!r} bytes do not match manifest")
        loaded[artifact_id] = load_json(str(path), limits=limits)
        if reference["format"] != artifact_formats[artifact_id]:
            raise SchemaError(f"qualification bundle artifact {artifact_id!r} format is unsupported")

    if is_current:
        policy = loaded["qualification_policy"]
        validate_qualification_policy(policy, limits=limits)
        if request["decision_policy"]["policy_id"] != policy["policy_id"]:
            raise SchemaError("qualification bundle decision policy ID does not match the bound policy")

    trace = loaded["source_trace"]
    canary = loaded["canary"]
    validate_trace(trace, require_known_overlap=True, limits=limits)
    validate_canary(canary, limits=limits)
    expected_fidelity = verify_canary_fidelity(trace, canary, limits=limits)
    if loaded["fidelity_verification"] != expected_fidelity:
        raise SchemaError("qualification bundle fidelity verification does not recompute exactly")
    if expected_fidelity.get("status") != "source_verified":
        raise SchemaError("qualification bundle canary is not source verified")

    compiler = canary.get("compiler")
    if not isinstance(compiler, Mapping):
        raise SchemaError("qualification bundle canary compiler metadata must be an object")
    if any(request["bindings"].get(field) != compiler.get(field) for field in request["bindings"]):
        raise SchemaError("qualification bundle manifest bindings do not match canary commitments")
    materialization = param_materialization_requirements(
        canary,
        require_event_dtype=True,
        require_reduction_op=True,
        limits=limits,
    )
    _validate_source_message_shapes(trace)
    if request["target_execution"]["communication_dtypes"] != materialization["communication_dtypes"]:
        raise SchemaError("qualification bundle target communication dtypes do not match canary events")
    if request["target_execution"]["communication_reduction_ops"] != materialization["communication_reduction_ops"]:
        raise SchemaError("qualification bundle target communication reduction operators do not match canary events")
    compute_audit = qualification_compute_recipe_audit(trace, limits=limits)
    target = request["target_execution"]
    if (
        target["compute_recipe_method"] != compute_audit["method"]
        or target["compute_recipe_projection_sha256"] != compute_audit["projection_sha256"]
        or target["compute_recipe_event_count"] != compute_audit["event_count"]
        or target["compute_recipe_operation_count"] != compute_audit["operation_count"]
    ):
        raise SchemaError("qualification bundle target compute-recipe commitments do not match the source trace")
    return copy.deepcopy(request)


def _validate_source_message_shapes(trace: Mapping[str, Any]) -> None:
    """Independently close Kineto input/output and split-size semantics."""

    workload = trace.get("workload")
    system = trace.get("system")
    workload = workload if isinstance(workload, Mapping) else {}
    system = system if isinstance(system, Mapping) else {}
    if workload.get("import_source") != "pytorch-kineto" and system.get("source_format") != "pytorch-kineto":
        return
    skipped_empty = _strict_nonnegative_int(
        workload.get("skipped_empty_events"),
        "qualification Kineto skipped_empty_events",
    )
    if skipped_empty:
        raise SchemaError(
            "qualification refuses a Kineto source with skipped zero-sized or missing-size communication events"
        )
    events = trace.get("events")
    if not isinstance(events, list):
        raise SchemaError("qualification source trace events must be a list")
    supported_ops = {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast",
        "point_to_point",
        "send",
        "recv",
    }
    equal_shape_ops = {
        "all_reduce",
        "broadcast",
        "point_to_point",
        "send",
        "recv",
    }
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise SchemaError(f"qualification source trace event {index} must be an object")
        op = str(event.get("op"))
        if op not in supported_ops:
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SchemaError(f"qualification Kineto event {index} lacks source-bound message-shape metadata")
        if metadata.get("kineto_message_shape_status") != "derived":
            raise SchemaError(
                f"qualification Kineto event {index} message shape is not exactly materializable: "
                f"{metadata.get('kineto_message_shape_status')!r}"
            )
        if metadata.get("kineto_message_shape_method") != "record-param-comms-in-out-nelems.v1":
            raise SchemaError(f"qualification Kineto event {index} has an unsupported message-shape derivation")
        in_nelems = _strict_positive_int(
            metadata.get("kineto_in_msg_nelems"),
            f"qualification Kineto event {index} input elements",
        )
        out_nelems = _strict_positive_int(
            metadata.get("kineto_out_msg_nelems"),
            f"qualification Kineto event {index} output elements",
        )
        input_splits = _strict_split_sizes(
            metadata.get("kineto_in_split_sizes"),
            f"qualification Kineto event {index} input split sizes",
        )
        output_splits = _strict_split_sizes(
            metadata.get("kineto_out_split_sizes"),
            f"qualification Kineto event {index} output split sizes",
        )
        if input_splits or output_splits:
            raise SchemaError(
                f"qualification Kineto event {index} uses explicit split sizes; "
                "the reference executor supports only source-verified equal-split operations"
            )
        rank_count = len(normalize_ranks(event.get("ranks")))
        if op in equal_shape_ops:
            shape_matches = in_nelems == out_nelems
        elif op == "all_gather":
            shape_matches = out_nelems == in_nelems * rank_count
        elif op == "reduce_scatter":
            shape_matches = in_nelems == out_nelems * rank_count
        else:
            shape_matches = in_nelems == out_nelems and in_nelems % rank_count == 0
        if not shape_matches:
            raise SchemaError(
                f"qualification Kineto event {index} input/output element counts "
                f"do not match {op} semantics for {rank_count} ranks"
            )
        dtype = event.get("dtype")
        if not isinstance(dtype, str):
            raise SchemaError(f"qualification Kineto event {index} lacks a source-bound dtype")
        expected_bytes = max(in_nelems, out_nelems) * dtype_size_bytes(dtype)
        if as_int(event.get("bytes")) != expected_bytes:
            raise SchemaError(
                f"qualification Kineto event {index} bytes do not match its exact input/output element counts"
            )


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")
    return value


def _strict_positive_int(value: Any, label: str) -> int:
    result = _strict_nonnegative_int(value, label)
    if result == 0:
        raise SchemaError(f"{label} must be positive")
    return result


def _strict_split_sizes(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{label} must be an exact list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise SchemaError(f"{label} must contain non-negative integers")
    return tuple(value)


def _create_new_bundle_directory(output: Path) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot create qualification bundle directory {output}: {exc}",
            path=str(output),
            operation="create bundle directory",
        ) from exc


def _validate_bundle_directory(directory: Path) -> None:
    try:
        mode = os.lstat(str(directory)).st_mode
    except FileNotFoundError as exc:
        raise SchemaError(f"{directory} does not exist") from exc
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot inspect qualification bundle directory {directory}: {exc}",
            path=str(directory),
            operation="inspect bundle directory",
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SchemaError("qualification bundle path must be a real directory, not a symlink")


def _require_regular_file(path: Path) -> None:
    try:
        mode = os.lstat(str(path)).st_mode
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot inspect qualification bundle artifact {path}: {exc}",
            path=str(path),
            operation="inspect bundle artifact",
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SchemaError(f"qualification bundle artifact must be a regular file: {path.name}")


def _bounded_file_identity(
    path: Path,
    *,
    limits: ResourceLimits,
) -> Tuple[str, int]:
    _require_regular_file(path)
    try:
        size_bytes = path.stat().st_size
        if size_bytes > limits.max_input_bytes:
            raise SchemaError(
                f"qualification bundle artifact {path.name} exceeds max_input_bytes={limits.max_input_bytes}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except SchemaError:
        raise
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot hash qualification bundle artifact {path}: {exc}",
            path=str(path),
            operation="hash bundle artifact",
        ) from exc
    return digest.hexdigest(), size_bytes


__all__ = ["prepare_qualification_request", "verify_qualification_request"]
