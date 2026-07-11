"""Completeness-gated deterministic analysis and publication pipeline."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..harness import (
    CELL_RESULT_SCHEMA,
    ArtifactReference,
    ContractError,
    FrozenRun,
    IncompleteCampaignError,
    RunManifest,
    SelectionSnapshot,
    canonical_json_bytes,
    canonical_sha256,
    evaluate_completeness,
    load_attempt_record,
    load_cell_attempts,
    load_cell_result,
    load_completeness_verdict,
    load_frozen_run,
    load_selection_snapshot,
    read_bounded_bytes,
    sha256_hex,
    verify_artifact_reference,
)
from ..harness.completeness import CompletenessVerdict
from .archive import verify_archive_descriptor
from .claims import build_trusted_claims
from .schemas import (
    LOCAL_CONSUME_MEASUREMENT_SCHEMA,
    LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA,
    LOCAL_PREPARE_MEASUREMENT_SCHEMA,
    PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA,
    PHYSICAL_FULL_MEASUREMENT_SCHEMA,
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
    PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
    PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
    RAW_ARCHIVE_DESCRIPTOR_SCHEMA,
    validate_scalar_measurement,
    validate_schema_documents,
)

PathLike = Union[str, "Path"]

ANALYSIS_SCHEMA = "commcanary.experiment.validated-aggregate.v2"
AGGREGATE_JSON_FILENAME = "aggregate.json"
AGGREGATE_CSV_FILENAME = "aggregate.csv"
PAPER_FRAGMENT_FILENAME = "paper-fragment.md"
PUBLICATION_FILENAMES = (
    AGGREGATE_JSON_FILENAME,
    AGGREGATE_CSV_FILENAME,
    PAPER_FRAGMENT_FILENAME,
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_CSV_FIELDS = (
    "record_kind",
    "completeness",
    "allow_incomplete",
    "issue_codes",
    "run_id",
    "manifest_sha256",
    "selection_id",
    "selection_sha256",
    "verdict_sha256",
    "cell_id",
    "cell_identity_sha256",
    "attempt_id",
    "attempt_record_sha256",
    "configuration_id",
    "workload_id",
    "repetition",
    "value_us",
    "sample_count",
    "measurement_schema",
    "producer_schema",
    "measurement_artifact_sha256",
    "repository_commit",
    "repository_dirty",
    "repository_patch_sha256",
    "source_archive_sha256",
    "environment_sha256",
    "execution_identity_sha256",
    "input_hashes",
    "regeneration_command",
)


class AnalysisValidationError(ContractError):
    """Raised when validated evidence cannot support deterministic analysis."""


class PersistedVerdictStaleError(AnalysisValidationError):
    """Raised when the persisted completeness verdict no longer matches evidence."""


class PublicationMismatchError(AnalysisValidationError):
    """Raised when generated publication bytes differ from the golden directory."""


@dataclass(frozen=True)
class GeneratedPublication:
    output_directory: Path
    aggregate: Mapping[str, Any]
    output_sha256: Mapping[str, str]
    matched_golden: bool


@dataclass(frozen=True)
class CampaignEvidence:
    run_directory: PathLike
    selection_id: str
    verdict_sha256: str


@dataclass(frozen=True)
class _LoadedEvidence:
    manifest: RunManifest
    frozen: FrozenRun
    snapshot: SelectionSnapshot
    verdict: CompletenessVerdict


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise AnalysisValidationError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _regeneration_command(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 8192 or "\x00" in value:
        raise AnalysisValidationError(
            "regeneration_command must be a non-empty NUL-free string of at most 8192 characters"
        )
    return value


def _median(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisValidationError("cannot aggregate an empty value sequence")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        lower = ordered[:middle]
        upper = ordered[middle + 1 :]
    else:
        lower = ordered[:middle]
        upper = ordered[middle:]
    return _median(upper) - _median(lower)


def _load_fresh_evidence(
    run_directory: PathLike,
    selection_id: str,
    verdict_sha256: str,
    *,
    allow_incomplete: bool,
) -> Tuple[RunManifest, FrozenRun, SelectionSnapshot, CompletenessVerdict]:
    manifest, frozen = load_frozen_run(run_directory)
    snapshot, stored_selection = load_selection_snapshot(frozen.directory, selection_id)
    persisted, stored_verdict = load_completeness_verdict(
        frozen.directory,
        selection_id,
        verdict_sha256,
    )
    if persisted.selection_sha256 != stored_selection.selection_sha256:
        raise PersistedVerdictStaleError("persisted verdict does not bind the frozen selection")
    if persisted.allow_incomplete != allow_incomplete:
        if not persisted.complete and not allow_incomplete:
            raise IncompleteCampaignError(persisted)
        raise PersistedVerdictStaleError("allow_incomplete does not match the policy recorded in the persisted verdict")
    recomputed = evaluate_completeness(
        frozen.directory,
        snapshot,
        allow_incomplete=allow_incomplete,
    )
    if recomputed.to_json_bytes() != persisted.to_json_bytes():
        raise PersistedVerdictStaleError("persisted completeness verdict is stale for the current attempt inventory")
    if stored_verdict.verdict_sha256 != persisted.sha256:
        raise PersistedVerdictStaleError("persisted verdict content address is inconsistent")
    return manifest, frozen, snapshot, persisted


def _attempt_accounting(
    manifest: RunManifest,
    frozen: FrozenRun,
    snapshot: SelectionSnapshot,
) -> Dict[str, Any]:
    selected_pairs = {(entry.cell_id, entry.attempt_id) for entry in snapshot.entries}
    status_counts = {
        "success": 0,
        "failed": 0,
        "parse-failed": 0,
        "cancelled": 0,
        "excluded": 0,
    }
    selected_status_counts = dict(status_counts)
    total_attempts = 0
    retries = 0
    selected_records = 0
    invalid_attempt_cells = 0
    for cell in manifest.cells:
        try:
            attempts = load_cell_attempts(frozen.directory, cell.id)
        except (ContractError, OSError):
            invalid_attempt_cells += 1
            continue
        total_attempts += len(attempts)
        retries += max(0, len(attempts) - 1)
        for attempt in attempts:
            status_counts[attempt.status] += 1
            if (attempt.cell_id, attempt.attempt_id) in selected_pairs:
                selected_records += 1
                selected_status_counts[attempt.status] += 1
    return {
        "terminal_attempts": total_attempts,
        "attempted_cells": sum(1 for cell in manifest.cells if _cell_has_attempts(frozen, cell.id)),
        "retries": retries,
        "unselected_terminal_attempts": max(0, total_attempts - selected_records),
        "by_status": status_counts,
        "selected_by_status": selected_status_counts,
        "invalid_attempt_cells": invalid_attempt_cells,
    }


def _cell_has_attempts(frozen: FrozenRun, cell_id: str) -> bool:
    try:
        return bool(load_cell_attempts(frozen.directory, cell_id))
    except (ContractError, OSError):
        return False


def _required_provenance(metadata: Any, field: str) -> Tuple[str, str, Optional[str]]:
    if not isinstance(metadata, Mapping):
        raise AnalysisValidationError(f"{field} must be an object")
    environment_sha256 = _sha256(metadata.get("environment_sha256"), f"{field}.environment_sha256")
    execution_identity_sha256 = _sha256(
        metadata.get("execution_identity_sha256"),
        f"{field}.execution_identity_sha256",
    )
    plan_raw = metadata.get("execution_plan_sha256")
    plan_sha256 = None if plan_raw is None else _sha256(plan_raw, f"{field}.execution_plan_sha256")
    return environment_sha256, execution_identity_sha256, plan_sha256


def _parameters(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisValidationError(f"{field} must be an object")
    return value


def _command_option(parameters: Mapping[str, Any], option: str) -> str:
    command = parameters.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise AnalysisValidationError(f"physical workload command is missing while binding {option}")
    positions = [index for index, item in enumerate(command) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise AnalysisValidationError(f"physical workload command must bind {option} exactly once")
    return str(command[positions[0] + 1])


def _optional_command_option(parameters: Mapping[str, Any], option: str) -> Optional[str]:
    command = parameters.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise AnalysisValidationError(f"physical workload command is missing while binding {option}")
    positions = [index for index, item in enumerate(command) if item == option]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise AnalysisValidationError(f"physical workload command must bind {option} at most once")
    return str(command[positions[0] + 1])


def _positive_option(parameters: Mapping[str, Any], option: str, *, default: Optional[int] = None) -> int:
    raw = _command_option(parameters, option) if default is None else _optional_command_option(parameters, option)
    if raw is None:
        assert default is not None
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise AnalysisValidationError(f"physical workload option {option} must be an integer") from exc
    if value <= 0:
        raise AnalysisValidationError(f"physical workload option {option} must be positive")
    return value


def _message_sizes(value: str) -> List[int]:
    result: List[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            raise AnalysisValidationError("manifest message-size list contains an empty item")
        suffix = item[-1].upper()
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(suffix, 1)
        number = item[:-1] if suffix in {"K", "M", "G"} else item
        try:
            parsed = float(number)
        except ValueError as exc:
            raise AnalysisValidationError("manifest message-size list is invalid") from exc
        size = int(parsed * scale)
        if parsed <= 0 or size <= 0:
            raise AnalysisValidationError("manifest message sizes must be positive")
        result.append(size)
    return result


def _selected_entry_map(snapshot: SelectionSnapshot) -> Dict[str, Any]:
    grouped: Dict[str, List[Any]] = {}
    for entry in snapshot.entries:
        grouped.setdefault(entry.cell_id, []).append(entry)
    return {cell_id: entries[0] for cell_id, entries in grouped.items() if len(entries) == 1}


def _validate_dependency_evidence(
    metadata: Mapping[str, Any],
    cell: Any,
    selected_entries: Mapping[str, Any],
) -> None:
    raw = metadata.get("dependency_attempts", [])
    if not isinstance(raw, list):
        raise AnalysisValidationError(f"attempt metadata for {cell.id!r} has invalid dependency_attempts")
    actual: Dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise AnalysisValidationError(f"dependency_attempts[{index}] must be an object")
        allowed = {"cell_id", "cell_identity_sha256", "attempt_id", "attempt_record_sha256"}
        if set(item) not in ({"cell_id", "attempt_id", "attempt_record_sha256"}, allowed):
            raise AnalysisValidationError(f"dependency_attempts[{index}] has invalid fields")
        dependency_cell_id = item.get("cell_id")
        if not isinstance(dependency_cell_id, str) or dependency_cell_id in actual:
            raise AnalysisValidationError("dependency attempt ownership is invalid or duplicated")
        actual[dependency_cell_id] = item
    if set(actual) != set(cell.dependencies):
        raise AnalysisValidationError(f"attempt dependency evidence does not match manifest cell {cell.id!r}")
    for dependency_cell_id in cell.dependencies:
        selected = selected_entries.get(dependency_cell_id)
        if selected is None:
            raise AnalysisValidationError(f"dependency {dependency_cell_id!r} has no unique selected attempt")
        item = actual[dependency_cell_id]
        if (
            item.get("attempt_id") != selected.attempt_id
            or item.get("attempt_record_sha256") != selected.attempt_record_sha256
            or ("cell_identity_sha256" in item and item.get("cell_identity_sha256") != selected.cell_identity_sha256)
        ):
            raise AnalysisValidationError(
                f"cell {cell.id!r} was executed against a dependency attempt outside the trusted selection"
            )


def _selected_capture_artifacts(
    manifest: RunManifest,
    frozen: FrozenRun,
    dependency_cell_id: str,
    selected_entry: Any,
) -> Dict[str, Any]:
    cells = {cell.id: cell for cell in manifest.cells}
    workloads = {workload.id: workload for workload in manifest.campaign.workloads}
    dependency_cell = cells[dependency_cell_id]
    dependency_workload = workloads[dependency_cell.workload_id]
    record, stored = load_attempt_record(frozen.directory, dependency_cell_id, selected_entry.attempt_id)
    if stored.record_sha256 != selected_entry.attempt_record_sha256 or record.measurement is None:
        raise AnalysisValidationError("selected dependency attempt content address is stale")
    result = load_cell_result(
        verify_artifact_reference(frozen.directory, record.measurement).path,
        cell_id=dependency_cell.id,
        cell_identity_sha256=dependency_cell.identity_sha256,
        producer_schema=dependency_workload.producer_schema,
        measurement_schema=dependency_workload.measurement_schema,
        max_bytes=max(1, record.measurement.size_bytes),
    )
    scalar = validate_scalar_measurement(
        dependency_workload.measurement_schema,
        dependency_workload.producer_schema,
        record.attempt_id,
        result.measurement.to_value(),
    )
    if scalar.physical is None or dependency_workload.measurement_schema != PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA:
        raise AnalysisValidationError("replay dependency is not a physical capture measurement")
    artifacts = {artifact.artifact_id: artifact for artifact in scalar.physical.artifacts}
    for artifact in artifacts.values():
        verify_artifact_reference(frozen.directory, ArtifactReference.from_dict(artifact.to_reference()))
    return artifacts


def _trace_binding_sha256(
    manifest: RunManifest,
    frozen: FrozenRun,
    cell: Any,
    workload: Any,
    selected_entries: Mapping[str, Any],
) -> str:
    parameters = _parameters(workload.parameters.to_value(), "workload.parameters")
    token = _command_option(parameters, "--trace-path")
    if token.startswith("{dependency:") and token.endswith("}"):
        parts = token[1:-1].split(":")
        if len(parts) != 3:
            raise AnalysisValidationError("manifest replay dependency placeholder is malformed")
        dependency_workload_id, artifact_id = parts[1], parts[2]
        matching = [
            dependency_cell_id
            for dependency_cell_id in cell.dependencies
            if next(item for item in manifest.cells if item.id == dependency_cell_id).workload_id
            == dependency_workload_id
        ]
        if len(matching) != 1:
            raise AnalysisValidationError("manifest replay trace does not resolve to one dependency cell")
        selected = selected_entries.get(matching[0])
        if selected is None:
            raise AnalysisValidationError("manifest replay dependency has no unique selected attempt")
        artifacts = _selected_capture_artifacts(manifest, frozen, matching[0], selected)
        if artifact_id not in artifacts:
            raise AnalysisValidationError(f"selected capture does not own replay artifact {artifact_id!r}")
        return str(artifacts[artifact_id].sha256)
    if token.startswith("{input:") and token.endswith("}"):
        input_id = token[1:-1].split(":", 1)[1]
        inputs = {artifact.id: artifact for artifact in manifest.campaign.inputs}
        if input_id not in inputs:
            raise AnalysisValidationError(f"manifest replay input {input_id!r} is not hash-bound")
        return inputs[input_id].sha256
    raise AnalysisValidationError("physical replay trace path is not bound to a dependency artifact or manifest input")


def _physical_binding(
    manifest: RunManifest,
    frozen: FrozenRun,
    cell: Any,
    configuration: Any,
    workload: Any,
    record: Any,
    scalar: Any,
    selected_entries: Mapping[str, Any],
) -> Dict[str, Any]:
    physical = scalar.physical
    if physical is None:
        return {}
    parameters = _parameters(workload.parameters.to_value(), "workload.parameters")
    if physical.operation != parameters.get("operation"):
        raise AnalysisValidationError(f"physical measurement operation is stale for cell {cell.id!r}")
    if physical.world_size != parameters.get("world_size") or list(physical.global_ranks) != parameters.get(
        "global_ranks"
    ):
        raise AnalysisValidationError(f"physical process-group layout is stale for cell {cell.id!r}")
    expected_runtime = _parameters(configuration.expected_runtime.to_value(), "configuration.expected_runtime")
    observed_runtime = physical.runtime.to_dict()
    for field in ("python_version", "torch_version", "runtime_nccl_version_code"):
        if field not in expected_runtime or observed_runtime[field] != expected_runtime[field]:
            raise AnalysisValidationError(f"physical runtime {field} is stale for cell {cell.id!r}")
    if physical.runtime.hostname != record.observed.hostname or physical.runtime.job_id != record.observed.job_id:
        raise AnalysisValidationError(f"physical runtime host/job ownership is stale for cell {cell.id!r}")
    metadata = _parameters(record.observed.metadata.to_value(), "attempt.observed.metadata")
    _validate_dependency_evidence(metadata, cell, selected_entries)
    input_hashes = {artifact.id: artifact.sha256 for artifact in manifest.campaign.inputs}
    if metadata.get("input_hashes") != input_hashes:
        raise AnalysisValidationError(f"physical input hashes are stale for cell {cell.id!r}")
    runtime_observation = metadata.get("runtime_observation")
    if (
        not isinstance(runtime_observation, Mapping)
        or runtime_observation.get("schema") != "commcanary.rostam.runtime-observation.v1"
        or runtime_observation.get("runtime") != observed_runtime
    ):
        raise AnalysisValidationError(f"physical runtime observation is stale for cell {cell.id!r}")
    attributes = physical.attributes
    if workload.measurement_schema == PHYSICAL_MICRO_MEASUREMENT_SCHEMA:
        if attributes.get("dtype") != _command_option(parameters, "--dtype") or attributes.get(
            "message_sizes_bytes"
        ) != _message_sizes(_command_option(parameters, "--msg-sizes")):
            raise AnalysisValidationError(f"physical micro workload shape is stale for cell {cell.id!r}")
    elif workload.measurement_schema == PHYSICAL_FULL_MEASUREMENT_SCHEMA:
        hidden = _positive_option(parameters, "--hidden")
        expected_shape = {
            "dtype": _command_option(parameters, "--dtype"),
            "layers": _positive_option(parameters, "--layers"),
            "tokens": _positive_option(parameters, "--tokens"),
            "hidden": hidden,
            "gemm_m": _positive_option(parameters, "--gemm-m"),
            "gemm_n": _positive_option(parameters, "--gemm-n", default=hidden),
        }
        if dict(attributes) != expected_shape:
            raise AnalysisValidationError(f"physical full workload shape is stale for cell {cell.id!r}")
    elif workload.measurement_schema in {
        PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
        PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
    }:
        if attributes.get("replay_mode") != parameters.get("replay_mode"):
            raise AnalysisValidationError(f"physical replay mode is stale for cell {cell.id!r}")
        if attributes.get("trace_sha256") != _trace_binding_sha256(manifest, frozen, cell, workload, selected_entries):
            raise AnalysisValidationError(f"physical replay trace hash is stale for cell {cell.id!r}")
    elif workload.measurement_schema == PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA:
        outputs = parameters.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != {
            artifact.artifact_id for artifact in physical.artifacts
        }:
            raise AnalysisValidationError(f"physical capture outputs are stale for cell {cell.id!r}")
        artifact_by_id = {artifact.artifact_id: artifact for artifact in physical.artifacts}
        for artifact_id, raw_path in outputs.items():
            if not isinstance(raw_path, str) or not raw_path.startswith("{workspace}/"):
                raise AnalysisValidationError(f"physical capture output path is stale for cell {cell.id!r}")
            relative = PurePosixPath(raw_path.removeprefix("{workspace}/"))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise AnalysisValidationError(f"physical capture output path is unsafe for cell {cell.id!r}")
            expected_path = PurePosixPath("workspaces", cell.id, record.attempt_id, *relative.parts).as_posix()
            if artifact_by_id[str(artifact_id)].path != expected_path:
                raise AnalysisValidationError(f"physical capture artifact path is stale for cell {cell.id!r}")
        capture_references = {item.path: item.to_reference() for item in physical.artifacts}
        attempt_references = {item.path: item.to_dict() for item in record.partial_outputs}
        if capture_references != attempt_references:
            raise AnalysisValidationError(f"physical capture artifacts are not bound by attempt {record.attempt_id!r}")
        for artifact in physical.artifacts:
            verify_artifact_reference(frozen.directory, ArtifactReference.from_dict(artifact.to_reference()))
    return {
        "wall_time_s": physical.wall_time_s,
        "measurement_iqr_us": physical.iqr_us,
        "artifacts": [
            {"artifact_id": item.artifact_id, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in physical.artifacts
        ],
        "binding_sha256": canonical_sha256(
            {
                "configuration": configuration.to_dict(),
                "workload": workload.to_dict(),
                "runtime_observation_sha256": canonical_sha256(runtime_observation),
                "attributes": dict(attributes),
                "artifacts": [item.to_reference() for item in physical.artifacts],
            }
        ),
    }


def _selected_rows(
    manifest: RunManifest,
    frozen: FrozenRun,
    snapshot: SelectionSnapshot,
    verdict: CompletenessVerdict,
) -> Tuple[Dict[str, Any], ...]:
    groups: Dict[str, List[Any]] = {}
    for entry in snapshot.entries:
        groups.setdefault(entry.cell_id, []).append(entry)
    invalid_cells = {issue.cell_id for issue in verdict.issues if issue.cell_id is not None}
    global_invalid = any(issue.cell_id is None for issue in verdict.issues)
    configurations = {configuration.id: configuration for configuration in manifest.campaign.configurations}
    workloads = {workload.id: workload for workload in manifest.campaign.workloads}
    selected_entries = _selected_entry_map(snapshot)
    rows: List[Dict[str, Any]] = []
    for cell in manifest.cells:
        entries = groups.get(cell.id, [])
        if global_invalid or cell.id in invalid_cells or len(entries) != 1:
            continue
        entry = entries[0]
        try:
            record, stored_attempt = load_attempt_record(
                frozen.directory,
                cell.id,
                entry.attempt_id,
            )
        except (ContractError, OSError):
            continue
        if record.status != "success" or record.measurement is None:
            continue
        verified = verify_artifact_reference(frozen.directory, record.measurement)
        workload = workloads[cell.workload_id]
        result = load_cell_result(
            verified.path,
            cell_id=cell.id,
            cell_identity_sha256=cell.identity_sha256,
            producer_schema=workload.producer_schema,
            measurement_schema=workload.measurement_schema,
            max_bytes=max(1, record.measurement.size_bytes),
        )
        scalar = validate_scalar_measurement(
            workload.measurement_schema,
            workload.producer_schema,
            record.attempt_id,
            result.measurement.to_value(),
        )
        configuration = configurations[cell.configuration_id]
        configuration_environment = configuration.environment.to_value()
        expected_config_value = configuration_environment.get("LOCAL_CONFIG")
        if expected_config_value is not None and scalar.config_value != expected_config_value:
            raise AnalysisValidationError(f"selected result for cell {cell.id!r} reports a stale configuration value")
        environment_sha256, execution_identity_sha256, execution_plan_sha256 = _required_provenance(
            record.observed.metadata.to_value(),
            f"attempt {record.attempt_id} metadata",
        )
        binding = _physical_binding(
            manifest,
            frozen,
            cell,
            configuration,
            workload,
            record,
            scalar,
            selected_entries,
        )
        rows.append(
            {
                "source_run_id": manifest.run_id,
                "source_manifest_sha256": frozen.manifest_sha256,
                "cell_id": cell.id,
                "cell_identity_sha256": cell.identity_sha256,
                "attempt_id": record.attempt_id,
                "attempt_record_sha256": stored_attempt.record_sha256,
                "configuration_id": cell.configuration_id,
                "configuration_sha256": canonical_sha256(configuration.to_dict()),
                "workload_id": cell.workload_id,
                "workload_sha256": canonical_sha256(workload.to_dict()),
                "repetition": cell.repetition,
                "producer_schema": workload.producer_schema,
                "measurement_schema": workload.measurement_schema,
                "measurement_artifact_sha256": record.measurement.sha256,
                "measurement_artifact_size_bytes": record.measurement.size_bytes,
                "value_us": scalar.value_us,
                "measurement_iqr_us": scalar.iqr_us,
                "samples_us": list(scalar.samples_us),
                "sample_count": len(scalar.samples_us),
                "environment_sha256": environment_sha256,
                "execution_identity_sha256": execution_identity_sha256,
                "execution_plan_sha256": execution_plan_sha256,
                **binding,
            }
        )
    return tuple(sorted(rows, key=lambda row: row["cell_id"]))


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    semantic_cells = [(str(row["workload_id"]), str(row["configuration_id"]), int(row["repetition"])) for row in rows]
    if len(set(semantic_cells)) != len(semantic_cells):
        raise AnalysisValidationError("trusted campaign join contains duplicate semantic cells")
    for row in rows:
        grouped.setdefault(
            (str(row["workload_id"]), str(row["configuration_id"])),
            [],
        ).append(row)
    aggregates: List[Dict[str, Any]] = []
    for (workload_id, configuration_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: str(row["cell_id"]))
        values = [float(row["value_us"]) for row in ordered]
        within_cell_iqrs = [float(row["measurement_iqr_us"]) for row in ordered]
        measurement_schemas = {str(row["measurement_schema"]) for row in ordered}
        producer_schemas = {str(row["producer_schema"]) for row in ordered}
        if len(measurement_schemas) != 1 or len(producer_schemas) != 1:
            raise AnalysisValidationError("aggregate group mixes incompatible producer schemas")
        aggregates.append(
            {
                "workload_id": workload_id,
                "configuration_id": configuration_id,
                "producer_schema": next(iter(producer_schemas)),
                "measurement_schema": next(iter(measurement_schemas)),
                "selected_repetitions": len(ordered),
                "median_us": _median(values),
                "iqr_us": _iqr(values) if len(values) > 1 else _median(within_cell_iqrs),
                "cell_ids": [str(row["cell_id"]) for row in ordered],
                "attempt_ids": [str(row["attempt_id"]) for row in ordered],
                "attempt_record_sha256s": [str(row["attempt_record_sha256"]) for row in ordered],
            }
        )
    return tuple(aggregates)


def _completeness_payload(evidences: Sequence[_LoadedEvidence]) -> Dict[str, Any]:
    complete = all(evidence.verdict.complete for evidence in evidences)
    issues = [
        {"run_id": evidence.manifest.run_id, **issue.to_dict()}
        for evidence in evidences
        for issue in evidence.verdict.issues
    ]
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "allow_incomplete": any(evidence.verdict.allow_incomplete for evidence in evidences),
        "expected_cells": sum(evidence.verdict.expected_cells for evidence in evidences),
        "attempted_cells": sum(evidence.verdict.attempted_cells for evidence in evidences),
        "selected_cells": sum(evidence.verdict.selected_cells for evidence in evidences),
        "successful_selected_cells": sum(evidence.verdict.successful_selected_cells for evidence in evidences),
        "issue_codes": sorted({issue.code for evidence in evidences for issue in evidence.verdict.issues}),
        "issues": issues,
    }


def _validate_trusted_join(evidences: Sequence[_LoadedEvidence]) -> None:
    if not evidences:
        raise AnalysisValidationError("trusted analysis requires at least one campaign")
    manifest_hashes = [evidence.frozen.manifest_sha256 for evidence in evidences]
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise AnalysisValidationError("trusted campaign join repeats a frozen manifest")
    repository = evidences[0].manifest.campaign.repository.to_dict()
    expected_site = evidences[0].manifest.campaign.expected_site.to_dict()
    policy = evidences[0].manifest.campaign.policy.to_value()
    configurations: Dict[str, str] = {}
    workloads: Dict[str, str] = {}
    inputs: Dict[str, Tuple[str, int]] = {}
    campaign_identities: set[Tuple[str, str]] = set()
    for evidence in evidences:
        campaign = evidence.manifest.campaign
        identity = (evidence.manifest.run_id, campaign.campaign_id)
        if identity in campaign_identities:
            raise AnalysisValidationError("trusted campaign join repeats a run/campaign identity")
        campaign_identities.add(identity)
        if campaign.repository.to_dict() != repository:
            raise AnalysisValidationError("trusted campaign join mixes repository identities")
        if campaign.expected_site.to_dict() != expected_site:
            raise AnalysisValidationError("trusted campaign join mixes expected site contracts")
        if campaign.policy.to_value() != policy:
            raise AnalysisValidationError("trusted campaign join mixes analysis policies")
        for configuration in campaign.configurations:
            digest = canonical_sha256(configuration.to_dict())
            previous = configurations.setdefault(configuration.id, digest)
            if previous != digest:
                raise AnalysisValidationError(f"trusted campaign join disagrees on configuration {configuration.id!r}")
        for workload in campaign.workloads:
            digest = canonical_sha256(workload.to_dict())
            previous_workload = workloads.setdefault(workload.id, digest)
            if previous_workload != digest:
                raise AnalysisValidationError(f"trusted campaign join disagrees on workload {workload.id!r}")
        for artifact in campaign.inputs:
            input_identity = (artifact.sha256, artifact.size_bytes)
            previous_input = inputs.setdefault(artifact.id, input_identity)
            if previous_input != input_identity:
                raise AnalysisValidationError(f"trusted campaign join disagrees on input {artifact.id!r}")


def _combined_attempt_accounting(evidences: Sequence[_LoadedEvidence]) -> Dict[str, Any]:
    campaign_rows = [
        {
            "run_id": evidence.manifest.run_id,
            **_attempt_accounting(evidence.manifest, evidence.frozen, evidence.snapshot),
        }
        for evidence in evidences
    ]
    statuses = ("success", "failed", "parse-failed", "cancelled", "excluded")
    return {
        "campaigns": campaign_rows,
        "terminal_attempts": sum(int(row["terminal_attempts"]) for row in campaign_rows),
        "attempted_cells": sum(int(row["attempted_cells"]) for row in campaign_rows),
        "retries": sum(int(row["retries"]) for row in campaign_rows),
        "unselected_terminal_attempts": sum(int(row["unselected_terminal_attempts"]) for row in campaign_rows),
        "invalid_attempt_cells": sum(int(row["invalid_attempt_cells"]) for row in campaign_rows),
        "by_status": {status: sum(int(row["by_status"][status]) for row in campaign_rows) for status in statuses},
        "selected_by_status": {
            status: sum(int(row["selected_by_status"][status]) for row in campaign_rows) for status in statuses
        },
    }


def _evidence_provenance(evidence: _LoadedEvidence) -> Dict[str, Any]:
    return {
        "run_id": evidence.manifest.run_id,
        "campaign_id": evidence.manifest.campaign.campaign_id,
        "campaign_sha256": evidence.manifest.campaign_sha256,
        "manifest_sha256": evidence.frozen.manifest_sha256,
        "selection_id": evidence.snapshot.selection_id,
        "selection_sha256": evidence.snapshot.sha256,
        "verdict_sha256": evidence.verdict.sha256,
        "repository": evidence.manifest.campaign.repository.to_dict(),
        "inputs": [artifact.to_dict() for artifact in evidence.manifest.campaign.inputs],
    }


def _build_aggregate(
    evidences: Sequence[_LoadedEvidence],
    *,
    regeneration_command: str,
    raw_archive: Mapping[str, Any],
    baseline_config: Optional[str],
    candidate_config: Optional[str],
    relative_threshold_pct: float,
    absolute_threshold_us: float,
) -> Dict[str, Any]:
    _validate_trusted_join(evidences)
    declared_measurement_schemas = {
        workload.measurement_schema for evidence in evidences for workload in evidence.manifest.campaign.workloads
    }
    schema_ids = {CELL_RESULT_SCHEMA}
    if declared_measurement_schemas & {
        LOCAL_PREPARE_MEASUREMENT_SCHEMA,
        LOCAL_CONSUME_MEASUREMENT_SCHEMA,
        LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA,
    }:
        schema_ids.update(
            {
                LOCAL_PREPARE_MEASUREMENT_SCHEMA,
                LOCAL_CONSUME_MEASUREMENT_SCHEMA,
                LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA,
            }
        )
    if raw_archive.get("verified") is True:
        schema_ids.add(RAW_ARCHIVE_DESCRIPTOR_SCHEMA)
    if declared_measurement_schemas & {
        PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
        PHYSICAL_FULL_MEASUREMENT_SCHEMA,
        PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
        PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
        PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA,
    }:
        schema_ids.update(
            {
                PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
                PHYSICAL_FULL_MEASUREMENT_SCHEMA,
                PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
                PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
                PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA,
            }
        )
    schema_documents = validate_schema_documents(tuple(sorted(schema_ids)))
    rows = tuple(
        row
        for evidence in evidences
        for row in _selected_rows(
            evidence.manifest,
            evidence.frozen,
            evidence.snapshot,
            evidence.verdict,
        )
    )
    aggregates = _aggregate_rows(rows)
    completeness = _completeness_payload(evidences)
    campaigns = [_evidence_provenance(evidence) for evidence in evidences]
    return {
        "schema": ANALYSIS_SCHEMA,
        "completeness": completeness,
        "provenance": {
            "campaigns": campaigns,
            "trusted_join_sha256": canonical_sha256(campaigns),
            "schema_documents": list(schema_documents),
            "raw_archive": dict(raw_archive),
            "regeneration_command": regeneration_command,
        },
        "failure_accounting": _combined_attempt_accounting(evidences),
        "selected_cell_count": len(rows),
        "selected_cells": list(sorted(rows, key=lambda row: (row["source_run_id"], row["cell_id"]))),
        "aggregates": list(aggregates),
        "claims": build_trusted_claims(
            aggregates,
            rows,
            complete=bool(completeness["complete"]),
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            relative_threshold_pct=relative_threshold_pct,
            absolute_threshold_us=absolute_threshold_us,
        ),
    }


def _csv_bytes(aggregate: Mapping[str, Any]) -> bytes:
    provenance = aggregate["provenance"]
    completeness = aggregate["completeness"]
    issue_codes = ";".join(completeness["issue_codes"])
    global_fields = {
        "completeness": str(completeness["status"]).upper(),
        "allow_incomplete": str(completeness["allow_incomplete"]).lower(),
        "issue_codes": issue_codes,
        "regeneration_command": provenance["regeneration_command"],
    }
    campaign_by_manifest = {campaign["manifest_sha256"]: campaign for campaign in provenance["campaigns"]}
    rows: List[Dict[str, Any]] = []
    for campaign in provenance["campaigns"]:
        repository = campaign["repository"]
        rows.append(
            {
                **global_fields,
                "record_kind": "campaign",
                "run_id": campaign["run_id"],
                "manifest_sha256": campaign["manifest_sha256"],
                "selection_id": campaign["selection_id"],
                "selection_sha256": campaign["selection_sha256"],
                "verdict_sha256": campaign["verdict_sha256"],
                "repository_commit": repository["commit"],
                "repository_dirty": str(repository["dirty"]).lower(),
                "repository_patch_sha256": repository["patch_sha256"] or "",
                "source_archive_sha256": repository["source_archive_sha256"] or "",
                "input_hashes": ";".join(f"{item['id']}={item['sha256']}" for item in campaign["inputs"]),
            }
        )
    for selected in aggregate["selected_cells"]:
        campaign = campaign_by_manifest[selected["source_manifest_sha256"]]
        repository = campaign["repository"]
        rows.append(
            {
                **global_fields,
                "record_kind": "measurement",
                "run_id": campaign["run_id"],
                "manifest_sha256": campaign["manifest_sha256"],
                "selection_id": campaign["selection_id"],
                "selection_sha256": campaign["selection_sha256"],
                "verdict_sha256": campaign["verdict_sha256"],
                "cell_id": selected["cell_id"],
                "cell_identity_sha256": selected["cell_identity_sha256"],
                "attempt_id": selected["attempt_id"],
                "attempt_record_sha256": selected["attempt_record_sha256"],
                "configuration_id": selected["configuration_id"],
                "workload_id": selected["workload_id"],
                "repetition": selected["repetition"],
                "value_us": f"{selected['value_us']:.6f}",
                "sample_count": selected["sample_count"],
                "measurement_schema": selected["measurement_schema"],
                "producer_schema": selected["producer_schema"],
                "measurement_artifact_sha256": selected["measurement_artifact_sha256"],
                "environment_sha256": selected["environment_sha256"],
                "execution_identity_sha256": selected["execution_identity_sha256"],
                "repository_commit": repository["commit"],
                "repository_dirty": str(repository["dirty"]).lower(),
                "repository_patch_sha256": repository["patch_sha256"] or "",
                "source_archive_sha256": repository["source_archive_sha256"] or "",
                "input_hashes": ";".join(f"{item['id']}={item['sha256']}" for item in campaign["inputs"]),
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in _CSV_FIELDS})
    return output.getvalue().encode("utf-8")


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "\\`")


def _markdown_bytes(aggregate: Mapping[str, Any]) -> bytes:
    provenance = aggregate["provenance"]
    completeness = aggregate["completeness"]
    lines = ["<!-- generated: do not edit -->", "# Validated Experiment Fragment", ""]
    if completeness["complete"]:
        lines.append(
            f"> **COMPLETENESS: COMPLETE** — {completeness['successful_selected_cells']}/"
            f"{completeness['expected_cells']} expected cells have selected successful attempts."
        )
    else:
        codes = ", ".join(completeness["issue_codes"]) or "unspecified"
        lines.append(
            "> **WARNING — INCOMPLETE EVIDENCE.** Generated only because "
            f"`--allow-incomplete` was explicit. Issues: **{_markdown_escape(codes)}**."
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Trusted join SHA-256: `{provenance['trusted_join_sha256']}`",
            f"- Campaigns: {len(provenance['campaigns'])}",
        ]
    )
    for campaign in provenance["campaigns"]:
        lines.extend(
            [
                f"  - Run `{_markdown_escape(campaign['run_id'])}` / campaign "
                f"`{_markdown_escape(campaign['campaign_id'])}`",
                f"    - Manifest: `{campaign['manifest_sha256']}`",
                f"    - Selection: `{_markdown_escape(campaign['selection_id'])}` (`{campaign['selection_sha256']}`)",
                f"    - Completeness verdict: `{campaign['verdict_sha256']}`",
                f"    - Repository commit: `{campaign['repository']['commit']}`",
            ]
        )
    archive = provenance["raw_archive"]
    if archive["verified"]:
        descriptor = archive["descriptor"]
        lines.append(
            f"- Verified raw archive: `{_markdown_escape(descriptor['uri'])}` / "
            f"`{descriptor['sha256']}` ({descriptor['size_bytes']} bytes)"
        )
    lines.extend(
        [
            "",
            "## Validated aggregates",
            "",
            "| workload | configuration | selected reps | median us | IQR us | cell IDs |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in aggregate["aggregates"]:
        lines.append(
            f"| {_markdown_escape(row['workload_id'])} | "
            f"{_markdown_escape(row['configuration_id'])} | {row['selected_repetitions']} | "
            f"{row['median_us']:.6f} | {row['iqr_us']:.6f} | "
            f"{_markdown_escape(', '.join(row['cell_ids']))} |"
        )
    lines.extend(
        [
            "",
            "## Selected-cell trace",
            "",
            "| cell ID | attempt | attempt record SHA-256 | environment SHA-256 | measurement SHA-256 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in aggregate["selected_cells"]:
        lines.append(
            f"| {_markdown_escape(row['cell_id'])} | {_markdown_escape(row['attempt_id'])} | "
            f"`{row['attempt_record_sha256']}` | `{row['environment_sha256']}` | "
            f"`{row['measurement_artifact_sha256']}` |"
        )
    accounting = aggregate["failure_accounting"]
    lines.extend(
        [
            "",
            "## Failure and retry accounting",
            "",
            f"- Terminal attempts: {accounting['terminal_attempts']}",
            f"- Retries preserved: {accounting['retries']}",
            f"- Unselected terminal attempts: {accounting['unselected_terminal_attempts']}",
            f"- Status counts: `{canonical_json_bytes(accounting['by_status']).decode('utf-8')}`",
            "",
            "## Claims",
            "",
        ]
    )
    claims = aggregate["claims"]
    if claims["status"] == "supported-by-complete-selected-evidence":
        lines.extend(
            [
                "The rankings, pairwise relations, Kendall agreement, regression verdicts, and costs below are "
                "computed exclusively from the complete manifest-bound rows above.",
                "",
                "### Rankings",
                "",
                "| workload | rank | configuration | median us | IQR us |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for workload_name, rankings in sorted(claims["rankings"].items()):
            for ranking in rankings:
                lines.append(
                    f"| {_markdown_escape(workload_name)} | {ranking['rank']} | "
                    f"{_markdown_escape(ranking['config'])} | {ranking['median_us']:.6f} | "
                    f"{ranking['iqr_us']:.6f} |"
                )
        lines.extend(
            [
                "",
                "### Pairwise agreement with W-full",
                "",
                "| workload | pairs | agreement | Kendall tau |",
                "|---|---:|---:|---:|",
            ]
        )
        for agreement in claims["agreements"].values():
            lines.append(
                f"| {_markdown_escape(agreement['workload'])} | {agreement['pairs']} | "
                f"{agreement['agreement_pct']:.6f}% | {agreement['kendall_tau']:.6f} |"
            )
        regression = claims["regression_2x2"]
        lines.extend(
            [
                "",
                "### Regression 2x2",
                "",
                f"Baseline: `{_markdown_escape(regression['baseline_config'])}`; candidate: "
                f"`{_markdown_escape(regression['candidate_config'])}`.",
                "",
                "| workload | baseline us | candidate us | delta pct | regression | vs W-full |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        confusion = regression["confusion_vs_full"]
        for workload_name, row in sorted(regression["workloads"].items()):
            cell = "reference" if workload_name == "W-full" else confusion.get(workload_name, {}).get("cell", "")
            lines.append(
                f"| {_markdown_escape(workload_name)} | {row['baseline_median_us']:.6f} | "
                f"{row['candidate_median_us']:.6f} | {row['delta_pct']:.6f}% | "
                f"{str(row['regression']).lower()} | {_markdown_escape(cell)} |"
            )
        lines.extend(
            [
                "",
                "### Cost",
                "",
                "| workload | runs | median wall s | median artifact sizes (bytes) |",
                "|---|---:|---:|---|",
            ]
        )
        for workload_name, row in sorted(claims["costs"].items()):
            artifact_sizes = canonical_json_bytes(row["artifact_size_bytes_median"]).decode("utf-8")
            lines.append(
                f"| {_markdown_escape(workload_name)} | {row['runs']} | {row['wall_time_s_median']:.6f} | "
                f"`{_markdown_escape(artifact_sizes)}` |"
            )
    elif claims["status"] == "withheld-incomplete":
        lines.append(
            "**No performance or ranking claim is supported by this incomplete output.** "
            "Rows are retained only for debugging and failure accounting."
        )
    else:
        lines.append(
            "No Rostam ranking claim is applicable because this complete evidence set does not contain W-full."
        )
    lines.extend(
        [
            "",
            "## Exact regeneration command",
            "",
            "```sh",
            provenance["regeneration_command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _publication_bytes(aggregate: Mapping[str, Any]) -> Dict[str, bytes]:
    return {
        AGGREGATE_JSON_FILENAME: canonical_json_bytes(aggregate) + b"\n",
        AGGREGATE_CSV_FILENAME: _csv_bytes(aggregate),
        PAPER_FRAGMENT_FILENAME: _markdown_bytes(aggregate),
    }


def _write_atomic(directory: Path, filename: str, data: bytes) -> Path:
    destination = directory / filename
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.tmp-", dir=str(directory))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_publication(output_directory: Path, files: Mapping[str, bytes]) -> None:
    if output_directory.is_symlink():
        raise AnalysisValidationError("output directory may not be a symlink")
    output_directory.mkdir(parents=True, exist_ok=True)
    if not output_directory.is_dir():
        raise AnalysisValidationError("output path is not a directory")
    for entry in output_directory.iterdir():
        if entry.name not in PUBLICATION_FILENAMES:
            raise AnalysisValidationError(f"output directory contains unexpected entry {entry.name!r}")
        if entry.is_symlink() or not entry.is_file():
            raise AnalysisValidationError(f"output publication path is not a real regular file: {entry.name!r}")
    for filename in PUBLICATION_FILENAMES:
        _write_atomic(output_directory, filename, files[filename])


def compare_publication_to_golden(
    files: Mapping[str, bytes],
    golden_directory: PathLike,
) -> None:
    """Byte-compare all publication outputs and reject stale or extra files."""

    golden = Path(golden_directory).expanduser()
    if golden.is_symlink() or not golden.is_dir():
        raise PublicationMismatchError("golden output path must be a real directory")
    expected_names = set(PUBLICATION_FILENAMES)
    golden_entries = list(golden.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in golden_entries):
        raise PublicationMismatchError("golden output directory contains a non-regular entry")
    observed_names = {path.name for path in golden_entries}
    if observed_names != expected_names:
        raise PublicationMismatchError(
            f"golden output file set mismatch: expected {sorted(expected_names)!r}, observed {sorted(observed_names)!r}"
        )
    mismatches = []
    for filename in PUBLICATION_FILENAMES:
        try:
            observed = read_bounded_bytes(
                golden / filename,
                max_bytes=max(1, len(files[filename])),
                field=f"golden publication {filename}",
            )
        except (ContractError, OSError):
            mismatches.append(filename)
            continue
        if observed != files[filename]:
            mismatches.append(filename)
    if mismatches:
        raise PublicationMismatchError("generated publication differs from golden files: " + ", ".join(mismatches))


def verify_regenerate_campaigns(
    evidence_sources: Sequence[CampaignEvidence],
    output_directory: PathLike,
    *,
    regeneration_command: str,
    allow_incomplete: bool = False,
    archive_descriptor: Optional[PathLike] = None,
    raw_archive: Optional[PathLike] = None,
    golden_directory: Optional[PathLike] = None,
    baseline_config: Optional[str] = None,
    candidate_config: Optional[str] = None,
    relative_threshold_pct: float = 8.0,
    absolute_threshold_us: float = 1.0,
) -> GeneratedPublication:
    """Validate one or more complete campaigns before deriving publication claims."""

    if not isinstance(allow_incomplete, bool):
        raise AnalysisValidationError("allow_incomplete must be boolean")
    if not evidence_sources:
        raise AnalysisValidationError("trusted analysis requires at least one campaign evidence source")
    for value, field in (
        (relative_threshold_pct, "relative_threshold_pct"),
        (absolute_threshold_us, "absolute_threshold_us"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < float("inf"):
            raise AnalysisValidationError(f"{field} must be finite and non-negative")
    command = _regeneration_command(regeneration_command)
    loaded = tuple(
        _LoadedEvidence(
            *_load_fresh_evidence(
                source.run_directory,
                source.selection_id,
                source.verdict_sha256,
                allow_incomplete=allow_incomplete,
            )
        )
        for source in evidence_sources
    )
    archive_bindings = [
        {
            "run_id": evidence.manifest.run_id,
            "campaign_id": evidence.manifest.campaign.campaign_id,
            "repository_commit": evidence.manifest.campaign.repository.commit,
            "manifest_sha256": evidence.frozen.manifest_sha256,
            "selection_id": evidence.snapshot.selection_id,
            "selection_sha256": evidence.snapshot.sha256,
            "verdict_sha256": evidence.verdict.sha256,
        }
        for evidence in loaded
    ]
    archive = verify_archive_descriptor(archive_bindings, archive_descriptor, raw_archive)
    aggregate = _build_aggregate(
        loaded,
        regeneration_command=command,
        raw_archive=archive,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        relative_threshold_pct=float(relative_threshold_pct),
        absolute_threshold_us=float(absolute_threshold_us),
    )
    for evidence in loaded:
        final_verdict = evaluate_completeness(
            evidence.frozen.directory,
            evidence.snapshot,
            allow_incomplete=allow_incomplete,
        )
        if final_verdict.to_json_bytes() != evidence.verdict.to_json_bytes():
            raise PersistedVerdictStaleError("attempt evidence changed while the publication was being generated")
    files = _publication_bytes(aggregate)
    output = Path(output_directory).expanduser()
    if golden_directory is not None and output.resolve() == Path(golden_directory).expanduser().resolve():
        raise AnalysisValidationError("output_directory and golden_directory must be different")
    _write_publication(output, files)
    matched = False
    if golden_directory is not None:
        compare_publication_to_golden(files, golden_directory)
        matched = True
    return GeneratedPublication(
        output_directory=output.resolve(),
        aggregate=aggregate,
        output_sha256={filename: sha256_hex(files[filename]) for filename in files},
        matched_golden=matched,
    )


def verify_regenerate_compare(
    run_directory: PathLike,
    selection_id: str,
    verdict_sha256: str,
    output_directory: PathLike,
    *,
    regeneration_command: str,
    allow_incomplete: bool = False,
    joined_evidence: Sequence[CampaignEvidence] = (),
    archive_descriptor: Optional[PathLike] = None,
    raw_archive: Optional[PathLike] = None,
    golden_directory: Optional[PathLike] = None,
    baseline_config: Optional[str] = None,
    candidate_config: Optional[str] = None,
    relative_threshold_pct: float = 8.0,
    absolute_threshold_us: float = 1.0,
) -> GeneratedPublication:
    """Compatibility wrapper for one primary campaign plus explicit trusted joins."""

    return verify_regenerate_campaigns(
        (CampaignEvidence(run_directory, selection_id, verdict_sha256), *joined_evidence),
        output_directory,
        regeneration_command=regeneration_command,
        allow_incomplete=allow_incomplete,
        archive_descriptor=archive_descriptor,
        raw_archive=raw_archive,
        golden_directory=golden_directory,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        relative_threshold_pct=relative_threshold_pct,
        absolute_threshold_us=absolute_threshold_us,
    )
