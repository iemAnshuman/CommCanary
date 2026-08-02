"""Request-bound target materialization across qualification and PARAM adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple

from ..artifacts import (
    SENSITIVE_JSON_POLICY,
    atomic_write_json,
    formatted_json_bytes,
)
from ..artifacts.qualification import (
    QUALIFICATION_EXECUTION_ADAPTER,
    QUALIFICATION_EXECUTOR_CONTRACT,
    QUALIFICATION_PROGRAM_ENCODING,
    QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
)
from ..artifacts.qualification_materialization import (
    QUALIFICATION_MATERIALIZATION_FILENAME,
    QUALIFICATION_REPLAY_PROGRAM_FILENAME,
    qualification_materialization_sha256,
    validate_qualification_materialization,
)
from ..artifacts.qualification_program import (
    qualification_program_communication_inventory,
    trace_to_qualification_program,
)
from ..artifacts.wire import JsonDict
from ..errors import CommCanaryIOError, SchemaError
from ..formats import QUALIFICATION_MATERIALIZATION_FORMAT
from ..qualification_io import VerifiedDirectory
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..services.qualification import load_verified_qualification_inputs

_IMMUTABLE_MATERIALIZATION_JSON_POLICY = replace(
    SENSITIVE_JSON_POLICY,
    artifact_label="qualification materialization artifact",
    create_parents=False,
    overwrite=False,
)


@dataclass(frozen=True)
class VerifiedQualificationMaterialization:
    """Materialization manifest and program parsed from one verified snapshot."""

    manifest: JsonDict
    program_entries: Tuple[JsonDict, ...]


def materialize_qualification(
    request_directory: str,
    output_directory: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Materialize exact source-bound work without target timing calibration."""

    verified_request = load_verified_qualification_inputs(request_directory, limits=limits)
    request = verified_request.request
    trace = verified_request.trace
    request_identity = (
        verified_request.manifest_sha256,
        verified_request.manifest_size_bytes,
    )
    target = request["target_execution"]
    entries, compute_work_audit = trace_to_qualification_program(trace, limits=limits)
    _require_target_compute_commitment(target, compute_work_audit)
    _require_target_communication_inventory(target, entries)
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
    with VerifiedDirectory(output, label="qualification materialization") as opened:
        program_snapshot = opened.read_bytes(QUALIFICATION_REPLAY_PROGRAM_FILENAME, limits=limits)
    program_sha256 = program_snapshot.sha256
    program_size = program_snapshot.size_bytes
    if program_bytes != program_snapshot.raw:
        raise SchemaError("qualification materialization program bytes are not deterministic")

    materialization: JsonDict = {
        "format": QUALIFICATION_MATERIALIZATION_FORMAT,
        "request": {
            "format": request["format"],
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

    verified = load_verified_qualification_materialization(
        request_directory,
        materialization_directory,
        limits=limits,
    )
    return copy.deepcopy(verified.manifest)


def load_verified_qualification_materialization(
    request_directory: str,
    materialization_directory: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> VerifiedQualificationMaterialization:
    """Return a manifest and program parsed from the exact bytes verified together."""

    verified_request = load_verified_qualification_inputs(request_directory, limits=limits)
    request = verified_request.request
    trace = verified_request.trace
    request_identity = (
        verified_request.manifest_sha256,
        verified_request.manifest_size_bytes,
    )
    directory = Path(materialization_directory)
    expected_names = {
        QUALIFICATION_MATERIALIZATION_FILENAME,
        QUALIFICATION_REPLAY_PROGRAM_FILENAME,
    }
    with VerifiedDirectory(directory, label="qualification materialization") as opened:
        observed_names = opened.names()
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            unexpected = sorted(observed_names - expected_names)
            raise SchemaError(
                f"qualification materialization inventory mismatch: missing={missing!r}, unexpected={unexpected!r}"
            )
        manifest_snapshot = opened.read_json(
            QUALIFICATION_MATERIALIZATION_FILENAME,
            limits=limits,
            require_object=True,
        )
        program_snapshot = opened.read_json(
            QUALIFICATION_REPLAY_PROGRAM_FILENAME,
            limits=limits,
            require_object=False,
        )
        if opened.names() != observed_names:
            raise SchemaError("qualification materialization inventory changed while it was verified")

    manifest = manifest_snapshot.value
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
    _require_target_communication_inventory(target, entries)
    if manifest["compute_work"] != expected_compute_work:
        raise SchemaError("qualification materialization compute-work audit does not recompute")
    expected_bytes = formatted_json_bytes(entries, indent=1)
    reference = manifest["program"]
    if program_snapshot.sha256 != reference["sha256"] or program_snapshot.size_bytes != reference["size_bytes"]:
        raise SchemaError("qualification materialization program bytes do not match its manifest")
    if len(entries) != reference["entry_count"]:
        raise SchemaError("qualification materialization program entry count does not recompute")
    if expected_compute_work["operation_count"] != reference["compute_operation_count"]:
        raise SchemaError("qualification materialization compute operation count does not recompute")
    if program_snapshot.raw != expected_bytes or program_snapshot.value != entries:
        raise SchemaError("qualification materialization program does not recompute exactly from the request")
    return VerifiedQualificationMaterialization(
        manifest=copy.deepcopy(manifest),
        program_entries=tuple(copy.deepcopy(entry) for entry in entries),
    )


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


def _require_target_communication_inventory(
    target: JsonDict,
    entries: list[JsonDict],
) -> None:
    inventory = qualification_program_communication_inventory(entries)
    if target["communication_dtypes"] != inventory["communication_dtypes"]:
        raise SchemaError("qualification request communication dtypes do not match its generated program")
    if target["communication_reduction_ops"] != inventory["communication_reduction_ops"]:
        raise SchemaError("qualification request communication reduction operators do not match its generated program")
    if "communication_inventory_source" in target and (
        target["communication_inventory_source"] != "full-generated-program"
        or target["communication_operations"] != inventory["communication_operations"]
        or target["communication_message_shapes"] != inventory["communication_message_shapes"]
    ):
        raise SchemaError("qualification request communication inventory does not match its generated program")


__all__ = [
    "VerifiedQualificationMaterialization",
    "load_verified_qualification_materialization",
    "materialize_qualification",
    "verify_qualification_materialization",
]
