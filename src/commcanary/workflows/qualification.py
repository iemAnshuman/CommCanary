"""Request-bound target materialization across qualification and PARAM adapters."""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Tuple

from ..artifacts import (
    SENSITIVE_JSON_POLICY,
    atomic_write_json,
    formatted_json_bytes,
    load_json,
)
from ..artifacts.qualification import (
    QUALIFICATION_ARTIFACT_PATHS,
    QUALIFICATION_EXECUTION_ADAPTER,
    QUALIFICATION_EXECUTOR_CONTRACT,
    QUALIFICATION_PROGRAM_ENCODING,
    QUALIFICATION_REQUEST_FILENAME,
    QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
)
from ..artifacts.qualification_materialization import (
    QUALIFICATION_MATERIALIZATION_FILENAME,
    QUALIFICATION_REPLAY_PROGRAM_FILENAME,
    qualification_materialization_sha256,
    validate_qualification_materialization,
)
from ..artifacts.qualification_program import trace_to_qualification_program
from ..artifacts.wire import JsonDict
from ..errors import CommCanaryIOError, SchemaError
from ..formats import QUALIFICATION_MATERIALIZATION_FORMAT, QUALIFICATION_REQUEST_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..services.qualification import verify_qualification_request

_IMMUTABLE_MATERIALIZATION_JSON_POLICY = replace(
    SENSITIVE_JSON_POLICY,
    artifact_label="qualification materialization artifact",
    create_parents=False,
    overwrite=False,
)


def materialize_qualification(
    request_directory: str,
    output_directory: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Materialize exact source-bound work without target timing calibration."""

    request, trace, _canary, request_identity = _verified_request_inputs(
        Path(request_directory),
        limits=limits,
    )
    target = request["target_execution"]
    entries, compute_work_audit = trace_to_qualification_program(trace, limits=limits)
    _require_target_compute_commitment(target, compute_work_audit)
    program_bytes = formatted_json_bytes(entries, indent=1)

    output = Path(output_directory)
    _create_new_directory(output, label="qualification materialization")
    program_path = output / QUALIFICATION_REPLAY_PROGRAM_FILENAME
    atomic_write_json(
        program_path,
        entries,
        indent=1,
        policy=_IMMUTABLE_MATERIALIZATION_JSON_POLICY,
    )
    program_sha256, program_size = _bounded_file_identity(program_path, limits=limits)
    if program_bytes != program_path.read_bytes():
        raise SchemaError("qualification materialization program bytes are not deterministic")

    materialization: JsonDict = {
        "format": QUALIFICATION_MATERIALIZATION_FORMAT,
        "request": {
            "format": QUALIFICATION_REQUEST_FORMAT,
            "request_id": request["request_id"],
            "manifest_sha256": request_identity[0],
            "manifest_size_bytes": request_identity[1],
        },
        "compute_work": compute_work_audit,
        "program": {
            "path": QUALIFICATION_REPLAY_PROGRAM_FILENAME,
            "encoding": QUALIFICATION_PROGRAM_ENCODING,
            "sha256": program_sha256,
            "size_bytes": program_size,
            "entry_count": len(entries),
            "compute_operation_count": compute_work_audit["operation_count"],
        },
        "executor": {
            "contract": QUALIFICATION_EXECUTOR_CONTRACT,
            "adapter": QUALIFICATION_EXECUTION_ADAPTER,
            "upstream_param_compatibility": QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
            "timestamp_pacing": "disabled",
        },
        "claims": {
            "materialization": "request_bound",
            "compute_work_provenance": "source_trace_verified",
            "physical_execution": "not_included",
            "physical_measurement": "not_included",
            "qualification_verdict": "not_issued",
        },
    }
    materialization["materialization_id"] = qualification_materialization_sha256(materialization)
    validate_qualification_materialization(materialization, limits=limits)
    atomic_write_json(
        output / QUALIFICATION_MATERIALIZATION_FILENAME,
        materialization,
        indent=2,
        policy=_IMMUTABLE_MATERIALIZATION_JSON_POLICY,
    )
    return copy.deepcopy(materialization)


def verify_qualification_materialization(
    request_directory: str,
    materialization_directory: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Recompute an exact-work program from its verified request."""

    request, trace, _canary, request_identity = _verified_request_inputs(
        Path(request_directory),
        limits=limits,
    )
    directory = Path(materialization_directory)
    _validate_directory(directory, label="qualification materialization")
    expected_names = {
        QUALIFICATION_MATERIALIZATION_FILENAME,
        QUALIFICATION_REPLAY_PROGRAM_FILENAME,
    }
    observed_names = {entry.name for entry in directory.iterdir()}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise SchemaError(
            f"qualification materialization inventory mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )
    for name in sorted(expected_names):
        _require_regular_file(directory / name)

    manifest = load_json(
        str(directory / QUALIFICATION_MATERIALIZATION_FILENAME),
        limits=limits,
    )
    validate_qualification_materialization(manifest, limits=limits)
    if (
        manifest["request"]["request_id"] != request["request_id"]
        or manifest["request"]["manifest_sha256"] != request_identity[0]
        or manifest["request"]["manifest_size_bytes"] != request_identity[1]
    ):
        raise SchemaError("qualification materialization does not bind the verified request manifest")

    target = request["target_execution"]
    entries, expected_compute_work = trace_to_qualification_program(trace, limits=limits)
    _require_target_compute_commitment(target, expected_compute_work)
    if manifest["compute_work"] != expected_compute_work:
        raise SchemaError("qualification materialization compute-work audit does not recompute")
    expected_bytes = formatted_json_bytes(entries, indent=1)
    program_path = directory / QUALIFICATION_REPLAY_PROGRAM_FILENAME
    program_sha256, program_size = _bounded_file_identity(program_path, limits=limits)
    reference = manifest["program"]
    if program_sha256 != reference["sha256"] or program_size != reference["size_bytes"]:
        raise SchemaError("qualification materialization program bytes do not match its manifest")
    if len(entries) != reference["entry_count"]:
        raise SchemaError("qualification materialization program entry count does not recompute")
    if expected_compute_work["operation_count"] != reference["compute_operation_count"]:
        raise SchemaError("qualification materialization compute operation count does not recompute")
    if program_path.read_bytes() != expected_bytes:
        raise SchemaError("qualification materialization program does not recompute exactly from the request")
    return copy.deepcopy(manifest)


def _verified_request_inputs(
    directory: Path,
    *,
    limits: ResourceLimits,
) -> Tuple[JsonDict, JsonDict, JsonDict, Tuple[str, int]]:
    request = verify_qualification_request(str(directory), limits=limits)
    manifest_path = directory / QUALIFICATION_REQUEST_FILENAME
    manifest_identity = _bounded_file_identity(manifest_path, limits=limits)
    canary_path = directory / QUALIFICATION_ARTIFACT_PATHS["canary"]
    canary_identity = _bounded_file_identity(canary_path, limits=limits)
    reference = request["artifacts"]["canary"]
    if canary_identity != (reference["sha256"], reference["size_bytes"]):
        raise SchemaError("qualification request canary changed after verification")
    canary = load_json(str(canary_path), limits=limits)
    trace_path = directory / QUALIFICATION_ARTIFACT_PATHS["source_trace"]
    trace_identity = _bounded_file_identity(trace_path, limits=limits)
    trace_reference = request["artifacts"]["source_trace"]
    if trace_identity != (trace_reference["sha256"], trace_reference["size_bytes"]):
        raise SchemaError("qualification request source trace changed after verification")
    trace = load_json(str(trace_path), limits=limits)
    return request, trace, canary, manifest_identity


def _create_new_directory(output: Path, *, label: str) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot create {label} directory {output}: {exc}",
            path=str(output),
            operation=f"create {label} directory",
        ) from exc


def _validate_directory(directory: Path, *, label: str) -> None:
    try:
        mode = os.lstat(str(directory)).st_mode
    except FileNotFoundError as exc:
        raise SchemaError(f"{directory} does not exist") from exc
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot inspect {label} directory {directory}: {exc}",
            path=str(directory),
            operation=f"inspect {label} directory",
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SchemaError(f"{label} path must be a real directory, not a symlink")


def _require_regular_file(path: Path) -> None:
    try:
        mode = os.lstat(str(path)).st_mode
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot inspect qualification materialization artifact {path}: {exc}",
            path=str(path),
            operation="inspect qualification materialization artifact",
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SchemaError(f"qualification materialization artifact must be a regular file: {path.name}")


def _bounded_file_identity(
    path: Path,
    *,
    limits: ResourceLimits,
) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limits.max_input_bytes:
                    raise SchemaError(
                        f"qualification materialization artifact {path.name} exceeds "
                        f"max_input_bytes={limits.max_input_bytes}"
                    )
                digest.update(chunk)
    except SchemaError:
        raise
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot hash qualification materialization artifact {path}: {exc}",
            path=str(path),
            operation="hash qualification materialization artifact",
        ) from exc
    return digest.hexdigest(), size


def _require_target_compute_commitment(
    target: JsonDict,
    audit: JsonDict,
) -> None:
    if (
        target["compute_recipe_method"] != audit["method"]
        or target["compute_recipe_projection_sha256"] != audit["projection_sha256"]
        or target["compute_recipe_event_count"] != audit["event_count"]
        or target["compute_recipe_operation_count"] != audit["operation_count"]
    ):
        raise SchemaError("qualification request compute-recipe commitment does not match its source trace")


__all__ = ["materialize_qualification", "verify_qualification_materialization"]
