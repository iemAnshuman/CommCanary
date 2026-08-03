"""Completeness-gated deterministic analysis and publication pipeline."""

from __future__ import annotations

import csv
import io
import math
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
    JSONResourceLimits,
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
from ..lib.executor_artifact import (
    EXECUTOR_ANALYZE_ENTRY_POINT,
    EXECUTOR_ARTIFACT_INPUT_ID,
    EXECUTOR_POLICY_FORMAT,
    ExecutorArtifact,
)
from .archive import verify_archive_descriptor
from .claims import build_trusted_claims
from .compatibility import (
    CROSS_COMMIT_COMPATIBILITY_SCHEMA,
    CrossCommitCompatibility,
    CrossCommitCompatibilityError,
    analysis_implementation_record,
    load_cross_commit_compatibility,
)
from .schemas import (
    LOCAL_CONSUME_MEASUREMENT_SCHEMA,
    LOCAL_FAIL_ONCE_MEASUREMENT_SCHEMA,
    LOCAL_PREPARE_MEASUREMENT_SCHEMA,
    PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA,
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
    PHYSICAL_FULL_MEASUREMENT_SCHEMA,
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
    PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
    PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
    PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA,
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
_PER_CAMPAIGN_POLICY_FIELDS = frozenset({"catalog_profile", "input_paths"})
_RUNTIME_OBSERVATION_SCHEMA_V1 = "commcanary.rostam.runtime-observation.v1"
_RUNTIME_OBSERVATION_SCHEMA_V2 = "commcanary.rostam.runtime-observation.v2"
_BINDING_ENVIRONMENT_FIELDS = {
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_PROCID",
    "SLURM_STEP_GPUS",
}
# The publication serializer emits an aggregate this module just built from
# already-validated evidence, so its budget bounds our own output rather than
# untrusted input.  A measured 280-cell core/shared/overlap join produced
# 1,013,696 items in 8,101,263 bytes, just past the 1,000,000-item default.
_PUBLICATION_JSON_LIMITS = JSONResourceLimits(max_items=4_000_000)
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
class PreparedCrossCommitCompatibility:
    output_path: Path
    contract_sha256: str
    status: str
    ground_truth_publication_sha256: Mapping[str, str]


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
        verify_artifact_reference(frozen.directory, record.measurement).raw,
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


def _replicated_environment_binding(
    runtime_observation: Mapping[str, Any],
    *,
    world_size: int,
    cell_id: str,
) -> Dict[str, Any]:
    required_observation_fields = {
        "schema",
        "runtime",
        "driver_version",
        "gpu_count",
        "gpus",
        "topology",
        "node_state",
        "binding",
        "probe_policy",
    }
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
    driver_version = runtime_observation.get("driver_version")
    gpus = runtime_observation.get("gpus")
    if (
        runtime_observation.get("schema") != _RUNTIME_OBSERVATION_SCHEMA_V2
        or set(runtime_observation) != required_observation_fields
        or not isinstance(driver_version, str)
        or not driver_version
        or not isinstance(gpus, list)
        or len(gpus) != world_size
        or runtime_observation.get("gpu_count") != len(gpus)
        or any(not isinstance(gpu, Mapping) or set(gpu) != required_gpu_fields for gpu in gpus)
    ):
        raise AnalysisValidationError(
            f"replicated decision-gate environment evidence is incomplete for cell {cell_id!r}"
        )
    gpu_indices: List[int] = []
    for gpu in gpus:
        index = gpu["index"]
        text_fields = (
            "uuid",
            "name",
            "driver_version",
            "pci_bus_id",
            "persistence_mode",
            "performance_state",
        )
        temperature = gpu["temperature_c"]
        powers = (gpu["power_draw_w"], gpu["power_limit_w"])
        clocks = (gpu["sm_clock_mhz"], gpu["memory_clock_mhz"])
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or any(not isinstance(gpu[field], str) or not gpu[field] for field in text_fields)
            or gpu["driver_version"] != driver_version
            or isinstance(temperature, bool)
            or not isinstance(temperature, int)
            or not -50 <= temperature <= 200
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in powers
            )
            or float(gpu["power_limit_w"]) <= 0.0
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in clocks)
        ):
            raise AnalysisValidationError(f"replicated decision-gate GPU evidence is invalid for cell {cell_id!r}")
        gpu_indices.append(index)
    if sorted(gpu_indices) != list(range(world_size)):
        raise AnalysisValidationError(f"replicated decision-gate GPU inventory is stale for cell {cell_id!r}")

    topology = runtime_observation.get("topology")
    node_state = runtime_observation.get("node_state")
    binding = runtime_observation.get("binding")
    probe_policy = runtime_observation.get("probe_policy")
    if (
        not isinstance(topology, Mapping)
        or set(topology) != {"method", "text"}
        or topology.get("method") != "nvidia-smi topo -m"
        or not isinstance(topology.get("text"), str)
        or not topology["text"]
        or not isinstance(node_state, Mapping)
        or set(node_state) != {"method", "text"}
        or node_state.get("method") != "scontrol show node --oneliner HOSTNAME"
        or not isinstance(node_state.get("text"), str)
        or not node_state["text"]
        or not isinstance(binding, Mapping)
        or set(binding) != {"environment", "cpu_affinity", "cpu_affinity_method"}
        or binding.get("cpu_affinity_method") != "sched_getaffinity"
        or not isinstance(probe_policy, Mapping)
        or set(probe_policy) != {"timeout_seconds", "max_output_bytes_per_stream"}
        or isinstance(probe_policy.get("timeout_seconds"), bool)
        or not isinstance(probe_policy.get("timeout_seconds"), int)
        or probe_policy["timeout_seconds"] <= 0
        or isinstance(probe_policy.get("max_output_bytes_per_stream"), bool)
        or not isinstance(probe_policy.get("max_output_bytes_per_stream"), int)
        or probe_policy["max_output_bytes_per_stream"] <= 0
    ):
        raise AnalysisValidationError(
            f"replicated decision-gate host environment evidence is incomplete for cell {cell_id!r}"
        )
    environment = binding.get("environment")
    affinity = binding.get("cpu_affinity")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _BINDING_ENVIRONMENT_FIELDS
        or any(value is not None and (not isinstance(value, str) or not value) for value in environment.values())
        or not isinstance(affinity, list)
        or not affinity
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in affinity)
        or affinity != sorted(set(affinity))
    ):
        raise AnalysisValidationError(
            f"replicated decision-gate process binding evidence is incomplete for cell {cell_id!r}"
        )
    return {
        "schema": runtime_observation["schema"],
        "driver_version": driver_version,
        "gpus": list(gpus),
        "topology": dict(topology),
        "node_state": dict(node_state),
        "binding": dict(binding),
        "observation_sha256": canonical_sha256(runtime_observation),
    }


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
        or runtime_observation.get("schema") not in {_RUNTIME_OBSERVATION_SCHEMA_V1, _RUNTIME_OBSERVATION_SCHEMA_V2}
        or runtime_observation.get("runtime") != observed_runtime
    ):
        raise AnalysisValidationError(f"physical runtime observation is stale for cell {cell.id!r}")
    replicated_environment: Optional[Dict[str, Any]] = None
    if workload.measurement_schema == PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2:
        replicated_environment = _replicated_environment_binding(
            runtime_observation,
            world_size=physical.world_size,
            cell_id=cell.id,
        )
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
    elif workload.measurement_schema == PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA:
        input_sha256 = {item.id: item.sha256 for item in manifest.campaign.inputs}
        expected_attributes = {
            "replay_mode": parameters.get("replay_mode"),
            "request_id": parameters.get("expected_request_id"),
            "materialization_id": parameters.get("expected_materialization_id"),
            "program_sha256": parameters.get("expected_program_sha256"),
            "source_capture_diagnostic_id": parameters.get("expected_source_capture_diagnostic_id"),
            "source_capture_job_id": parameters.get("expected_source_job_id"),
            "source_capture_node": parameters.get("expected_source_node"),
            "source_capture_evidence_sha256": input_sha256.get("source-capture-evidence"),
            "source_capture_stdout_sha256": input_sha256.get("source-capture-stdout"),
        }
        for field, expected in expected_attributes.items():
            if attributes.get(field) != expected:
                raise AnalysisValidationError(f"physical qualification {field} is stale for cell {cell.id!r}")
        if not isinstance(attributes.get("correctness_check_count"), int):
            raise AnalysisValidationError(f"physical qualification correctness evidence is stale for cell {cell.id!r}")
        if attributes.get("source_capture_node") != physical.runtime.hostname.split(".", 1)[0]:
            raise AnalysisValidationError(f"physical qualification comparison is not same-node for cell {cell.id!r}")
        if attributes.get("comparison_claims") != {
            "single_configuration_timing_comparison": "diagnostic",
            "physical_fidelity": "unproven",
            "multi_configuration_ranking": "not_measured",
            "qualification_verdict": "not_issued",
        }:
            raise AnalysisValidationError(f"physical qualification claims are stale for cell {cell.id!r}")
    elif workload.measurement_schema in {
        PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
        PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
    }:
        expected_attributes = {
            "request_id": parameters.get("expected_request_id"),
            "materialization_id": parameters.get("expected_materialization_id"),
            "program_sha256": parameters.get("expected_program_sha256"),
            "policy_id": parameters.get("expected_policy_id"),
        }
        observed_attributes = {
            "request_id": attributes.get("request", {}).get("request_id"),
            "materialization_id": attributes.get("materialization", {}).get("materialization_id"),
            "program_sha256": attributes.get("materialization", {}).get("program_sha256"),
            "policy_id": attributes.get("policy", {}).get("policy_id"),
        }
        if observed_attributes != expected_attributes:
            raise AnalysisValidationError(f"physical decision-gate identities are stale for cell {cell.id!r}")
        execution = attributes.get("execution")
        expected_execution = {
            "iterations": parameters.get("iterations"),
            "warmup": parameters.get("warmup"),
            "source_event_count": parameters.get("expected_source_event_count"),
            "stratified_source_event_indices": parameters.get("expected_stratified_source_event_indices"),
            "world_size": parameters.get("world_size"),
        }
        if workload.measurement_schema == PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2:
            expected_execution["allocation_block"] = cell.repetition
        if not isinstance(execution, Mapping) or any(
            execution.get(field) != expected for field, expected in expected_execution.items()
        ):
            raise AnalysisValidationError(f"physical decision-gate execution is stale for cell {cell.id!r}")
        if attributes.get("decision_claims") != {
            "physical_execution": "same_allocation_self_reported",
            "physical_decision_fidelity": "not_analyzed",
            "qualification_verdict": "policy_bound_not_issued",
        }:
            raise AnalysisValidationError(f"physical decision-gate claims are stale for cell {cell.id!r}")
    binding = {
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
    if workload.measurement_schema in {
        PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
        PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
    }:
        binding["decision_gate"] = dict(attributes)
        binding["decision_gate_runtime"] = physical.runtime.to_dict()
    if replicated_environment is not None:
        binding["decision_gate_environment"] = replicated_environment
    return binding


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
            verified.raw,
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


def _joinable_policy(
    campaign: Any,
    *,
    additional_ignored_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return the policy fields whose agreement a trusted join actually requires.

    ``catalog_profile`` names the campaign's own recipe and ``input_paths``
    records where its inputs happened to live on the producing host.  Both
    differ by construction between campaigns of different profiles, so
    comparing them would make every cross-profile join impossible while
    proving nothing: input identity is enforced separately, and by digest.
    """

    policy = campaign.policy.to_value()
    if not isinstance(policy, Mapping):
        raise AnalysisValidationError("campaign policy must be an object")
    ignored = _PER_CAMPAIGN_POLICY_FIELDS | frozenset(additional_ignored_fields)
    return {key: value for key, value in policy.items() if key not in ignored}


def _cross_commit_policy_differences(evidences: Sequence[_LoadedEvidence]) -> Tuple[str, ...]:
    policies = [_joinable_policy(evidence.manifest.campaign) for evidence in evidences]
    fields = sorted({key for policy in policies for key in policy})
    missing = object()
    differences = []
    for field in fields:
        values = [policy.get(field, missing) for policy in policies]
        first = values[0]
        if any(value != first for value in values[1:]):
            differences.append(field)
    return tuple(differences)


def _cross_commit_input_differences(evidences: Sequence[_LoadedEvidence]) -> Tuple[str, ...]:
    identities: Dict[str, set[Tuple[str, int]]] = {}
    occurrences: Dict[str, int] = {}
    for evidence in evidences:
        for artifact in evidence.manifest.campaign.inputs:
            identities.setdefault(artifact.id, set()).add((artifact.sha256, artifact.size_bytes))
            occurrences[artifact.id] = occurrences.get(artifact.id, 0) + 1
    return tuple(
        sorted(input_id for input_id, observed in identities.items() if occurrences[input_id] > 1 and len(observed) > 1)
    )


def _validate_trusted_join(
    evidences: Sequence[_LoadedEvidence],
    *,
    cross_commit_compatibility: Optional[CrossCommitCompatibility] = None,
) -> None:
    if not evidences:
        raise AnalysisValidationError("trusted analysis requires at least one campaign")
    manifest_hashes = [evidence.frozen.manifest_sha256 for evidence in evidences]
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise AnalysisValidationError("trusted campaign join repeats a frozen manifest")
    repositories = {canonical_sha256(evidence.manifest.campaign.repository.to_dict()) for evidence in evidences}
    if len(repositories) > 1 and cross_commit_compatibility is None:
        raise AnalysisValidationError("trusted campaign join mixes repository identities")
    if len(repositories) == 1 and cross_commit_compatibility is not None:
        raise AnalysisValidationError("cross-commit compatibility contract is unnecessary for one repository identity")
    if cross_commit_compatibility is not None:
        observed_policy_differences = _cross_commit_policy_differences(evidences)
        if observed_policy_differences != cross_commit_compatibility.allowed_policy_fields:
            raise AnalysisValidationError(
                "cross-commit contract policy differences do not exactly match the joined manifests"
            )
        observed_input_differences = _cross_commit_input_differences(evidences)
        if observed_input_differences != cross_commit_compatibility.allowed_input_ids:
            raise AnalysisValidationError(
                "cross-commit contract input differences do not exactly match the joined manifests"
            )
    ignored_policy_fields = (
        cross_commit_compatibility.allowed_policy_fields if cross_commit_compatibility is not None else ()
    )
    expected_site = evidences[0].manifest.campaign.expected_site.to_dict()
    policy = _joinable_policy(
        evidences[0].manifest.campaign,
        additional_ignored_fields=ignored_policy_fields,
    )
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
        if campaign.expected_site.to_dict() != expected_site:
            raise AnalysisValidationError("trusted campaign join mixes expected site contracts")
        if _joinable_policy(campaign, additional_ignored_fields=ignored_policy_fields) != policy:
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
            allowed_input_difference = (
                cross_commit_compatibility is not None and artifact.id in cross_commit_compatibility.allowed_input_ids
            )
            if previous_input != input_identity and not allowed_input_difference:
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


def _frozen_analyzer_record(
    evidences: Sequence[_LoadedEvidence],
    executor_artifact: Optional[ExecutorArtifact],
) -> Optional[Dict[str, Any]]:
    declarations = []
    for evidence in evidences:
        policy = evidence.manifest.campaign.policy.to_value()
        executor = policy.get("executor") if isinstance(policy, Mapping) else None
        if executor is None:
            declarations.append(None)
            continue
        if not isinstance(executor, Mapping):
            raise AnalysisValidationError("campaign executor policy must be an object")
        declarations.append(executor)
    if all(declaration is None for declaration in declarations):
        if executor_artifact is not None:
            raise AnalysisValidationError("legacy campaigns may not acquire an unbound frozen analyzer identity")
        return None
    if any(declaration is None for declaration in declarations):
        raise AnalysisValidationError("cannot mix executor-bound and legacy campaigns in one analysis")
    if executor_artifact is None:
        raise AnalysisValidationError("executor-bound campaigns must be analyzed by their frozen executor artifact")
    for evidence, declaration in zip(evidences, declarations):
        assert declaration is not None
        inputs = {artifact.id: artifact for artifact in evidence.manifest.campaign.inputs}
        bound = inputs.get(EXECUTOR_ARTIFACT_INPUT_ID)
        expected = {
            "format": EXECUTOR_POLICY_FORMAT,
            "artifact_input_id": EXECUTOR_ARTIFACT_INPUT_ID,
            "inventory_sha256": executor_artifact.inventory_sha256,
            "source_inventory_sha256": executor_artifact.source_inventory_sha256,
            "schema_inventory_sha256": executor_artifact.schema_inventory_sha256,
            "source_file_count": len(executor_artifact.source_files),
            "schema_file_count": len(executor_artifact.schema_files),
        }
        if (
            bound is None
            or bound.sha256 != executor_artifact.sha256
            or bound.size_bytes != executor_artifact.size_bytes
            or any(declaration.get(key) != value for key, value in expected.items())
        ):
            raise AnalysisValidationError(
                f"campaign {evidence.manifest.run_id!r} does not bind the running frozen analyzer"
            )
    return executor_artifact.analyzer_record(EXECUTOR_ANALYZE_ENTRY_POINT)


def _compatibility_evidence_binding(evidence: _LoadedEvidence) -> Dict[str, Any]:
    return {
        "manifest_sha256": evidence.frozen.manifest_sha256,
        "run_id": evidence.manifest.run_id,
        "campaign_id": evidence.manifest.campaign.campaign_id,
        "selection_sha256": evidence.snapshot.sha256,
        "verdict_sha256": evidence.verdict.sha256,
        "repository": evidence.manifest.campaign.repository.to_dict(),
    }


def _archive_bindings(evidences: Sequence[_LoadedEvidence]) -> List[Dict[str, str]]:
    return [
        {
            "run_id": evidence.manifest.run_id,
            "campaign_id": evidence.manifest.campaign.campaign_id,
            "repository_commit": evidence.manifest.campaign.repository.commit,
            "manifest_sha256": evidence.frozen.manifest_sha256,
            "selection_id": evidence.snapshot.selection_id,
            "selection_sha256": evidence.snapshot.sha256,
            "verdict_sha256": evidence.verdict.sha256,
        }
        for evidence in evidences
    ]


def _build_aggregate(
    evidences: Sequence[_LoadedEvidence],
    *,
    regeneration_command: str,
    raw_archive: Mapping[str, Any],
    baseline_config: Optional[str],
    candidate_config: Optional[str],
    relative_threshold_pct: float,
    absolute_threshold_us: float,
    cross_commit_compatibility: Optional[CrossCommitCompatibility] = None,
    analyzer_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _validate_trusted_join(
        evidences,
        cross_commit_compatibility=cross_commit_compatibility,
    )
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
    if cross_commit_compatibility is not None:
        schema_ids.add(CROSS_COMMIT_COMPATIBILITY_SCHEMA)
    if declared_measurement_schemas & {
        PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
        PHYSICAL_FULL_MEASUREMENT_SCHEMA,
        PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
        PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
        PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA,
        PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
        PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
        PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA,
    }:
        schema_ids.update(
            {
                PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
                PHYSICAL_FULL_MEASUREMENT_SCHEMA,
                PHYSICAL_PARAM_MEASUREMENT_SCHEMA,
                PHYSICAL_OVERLAP_MEASUREMENT_SCHEMA,
                PHYSICAL_CAPTURE_MEASUREMENT_SCHEMA,
                PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
                PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
                PHYSICAL_QUALIFICATION_MEASUREMENT_SCHEMA,
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
    provenance: Dict[str, Any] = {
        "campaigns": campaigns,
        "trusted_join_sha256": canonical_sha256(campaigns),
        "schema_documents": list(schema_documents),
        "raw_archive": dict(raw_archive),
        "regeneration_command": regeneration_command,
    }
    if analyzer_record is not None:
        provenance["analysis_implementation"] = dict(analyzer_record)
    if cross_commit_compatibility is not None:
        provenance["cross_commit_compatibility"] = cross_commit_compatibility.provenance_summary()
    return {
        "schema": ANALYSIS_SCHEMA,
        "completeness": completeness,
        "provenance": provenance,
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
        AGGREGATE_JSON_FILENAME: canonical_json_bytes(aggregate, limits=_PUBLICATION_JSON_LIMITS) + b"\n",
        AGGREGATE_CSV_FILENAME: _csv_bytes(aggregate),
        PAPER_FRAGMENT_FILENAME: _markdown_bytes(aggregate),
    }


def _verify_cross_commit_ground_truth(
    contract: CrossCommitCompatibility,
    evidences: Sequence[_LoadedEvidence],
    *,
    golden_directory: PathLike,
    archive_descriptor: Optional[PathLike],
    raw_archive: Optional[PathLike],
    analyzer_record: Optional[Mapping[str, Any]],
) -> None:
    expected_bindings = sorted(
        (_compatibility_evidence_binding(evidence) for evidence in evidences),
        key=lambda item: str(item["manifest_sha256"]),
    )
    if expected_bindings != list(contract.campaign_bindings):
        raise CrossCommitCompatibilityError(
            "cross-commit contract campaign bindings do not exactly match the requested evidence join"
        )
    ground_manifest_set = set(contract.ground_truth_manifest_sha256s)
    ground_evidences = tuple(
        evidence for evidence in evidences if evidence.frozen.manifest_sha256 in ground_manifest_set
    )
    if {evidence.frozen.manifest_sha256 for evidence in ground_evidences} != ground_manifest_set:
        raise CrossCommitCompatibilityError("cross-commit contract ground-truth evidence is incomplete")
    ground_archive = verify_archive_descriptor(
        _archive_bindings(ground_evidences),
        archive_descriptor,
        raw_archive,
    )
    if bool(ground_archive.get("verified")) != contract.raw_archive_verified:
        raise CrossCommitCompatibilityError(
            "cross-commit ground-truth archive verification state does not match the reviewed contract"
        )
    ground_aggregate = _build_aggregate(
        ground_evidences,
        regeneration_command=contract.regeneration_command,
        raw_archive=ground_archive,
        baseline_config=contract.baseline_config,
        candidate_config=contract.candidate_config,
        relative_threshold_pct=contract.relative_threshold_pct,
        absolute_threshold_us=contract.absolute_threshold_us,
        analyzer_record=analyzer_record,
    )
    ground_files = _publication_bytes(ground_aggregate)
    observed_sha256 = {filename: sha256_hex(data) for filename, data in ground_files.items()}
    if observed_sha256 != dict(contract.publication_sha256):
        raise CrossCommitCompatibilityError(
            "current analyzer does not reproduce the contract's ground-truth publication hashes"
        )
    compare_publication_to_golden(ground_files, golden_directory)


def prepare_cross_commit_compatibility(
    ground_truth_sources: Sequence[CampaignEvidence],
    extension_sources: Sequence[CampaignEvidence],
    output_path: PathLike,
    *,
    regeneration_command: str,
    golden_directory: PathLike,
    archive_descriptor: Optional[PathLike] = None,
    raw_archive: Optional[PathLike] = None,
    baseline_config: Optional[str] = None,
    candidate_config: Optional[str] = None,
    relative_threshold_pct: float = 8.0,
    absolute_threshold_us: float = 1.0,
    reviewed: bool = False,
    executor_artifact: Optional[ExecutorArtifact] = None,
) -> PreparedCrossCommitCompatibility:
    """Prepare an exact, reviewable bridge after reproducing old golden bytes.

    The default output is a non-executable ``candidate``. Repeating the same
    command with ``reviewed=True`` is the explicit acknowledgement that the
    exact manifest and difference inventory has been reviewed.
    """

    if not isinstance(reviewed, bool):
        raise AnalysisValidationError("reviewed must be boolean")
    if not ground_truth_sources:
        raise AnalysisValidationError("cross-commit preparation requires ground-truth evidence")
    if not extension_sources:
        raise AnalysisValidationError("cross-commit preparation requires extension evidence")
    command = _regeneration_command(regeneration_command)
    for value, field in (
        (relative_threshold_pct, "relative_threshold_pct"),
        (absolute_threshold_us, "absolute_threshold_us"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < float("inf"):
            raise AnalysisValidationError(f"{field} must be finite and non-negative")

    ground = tuple(
        _LoadedEvidence(
            *_load_fresh_evidence(
                source.run_directory,
                source.selection_id,
                source.verdict_sha256,
                allow_incomplete=False,
            )
        )
        for source in ground_truth_sources
    )
    extensions = tuple(
        _LoadedEvidence(
            *_load_fresh_evidence(
                source.run_directory,
                source.selection_id,
                source.verdict_sha256,
                allow_incomplete=False,
            )
        )
        for source in extension_sources
    )
    combined = (*ground, *extensions)
    analyzer_record = _frozen_analyzer_record(combined, executor_artifact)
    ground_archive = verify_archive_descriptor(
        _archive_bindings(ground),
        archive_descriptor,
        raw_archive,
    )
    ground_aggregate = _build_aggregate(
        ground,
        regeneration_command=command,
        raw_archive=ground_archive,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        relative_threshold_pct=float(relative_threshold_pct),
        absolute_threshold_us=float(absolute_threshold_us),
        analyzer_record=analyzer_record,
    )
    ground_files = _publication_bytes(ground_aggregate)
    compare_publication_to_golden(ground_files, golden_directory)
    publication_sha256 = {filename: sha256_hex(data) for filename, data in sorted(ground_files.items())}
    campaign_bindings = sorted(
        (_compatibility_evidence_binding(evidence) for evidence in combined),
        key=lambda item: str(item["manifest_sha256"]),
    )
    contract: Dict[str, Any] = {
        "schema": CROSS_COMMIT_COMPATIBILITY_SCHEMA,
        "status": "reviewed",
        "analysis_implementation": analysis_implementation_record(),
        "campaigns": campaign_bindings,
        "ground_truth": {
            "manifest_sha256s": sorted(evidence.frozen.manifest_sha256 for evidence in ground),
            "regeneration_command": command,
            "baseline_config": baseline_config,
            "candidate_config": candidate_config,
            "relative_threshold_pct": float(relative_threshold_pct),
            "absolute_threshold_us": float(absolute_threshold_us),
            "publication_sha256": publication_sha256,
            "raw_archive_verified": bool(ground_archive.get("verified")),
        },
        "allowed_differences": {
            "analysis_policy_fields": list(_cross_commit_policy_differences(combined)),
            "input_ids": list(_cross_commit_input_differences(combined)),
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    reviewed_contract = CrossCommitCompatibility.from_mapping(contract)
    _validate_trusted_join(
        combined,
        cross_commit_compatibility=reviewed_contract,
    )
    if not reviewed:
        contract["status"] = "candidate"
        contract["contract_sha256"] = canonical_sha256(
            {key: value for key, value in contract.items() if key != "contract_sha256"}
        )

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise AnalysisValidationError("cross-commit contract output parent must be a real directory")
    if destination.exists() or destination.is_symlink():
        raise AnalysisValidationError("cross-commit contract output already exists")
    encoded = canonical_json_bytes(contract)
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AnalysisValidationError(f"cannot write cross-commit compatibility contract: {exc}") from exc
    return PreparedCrossCommitCompatibility(
        output_path=destination.resolve(),
        contract_sha256=str(contract["contract_sha256"]),
        status=str(contract["status"]),
        ground_truth_publication_sha256=publication_sha256,
    )


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
    cross_commit_contract: Optional[PathLike] = None,
    compatibility_golden_directory: Optional[PathLike] = None,
    compatibility_archive_descriptor: Optional[PathLike] = None,
    compatibility_raw_archive: Optional[PathLike] = None,
    executor_artifact: Optional[ExecutorArtifact] = None,
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
    analyzer_record = _frozen_analyzer_record(loaded, executor_artifact)
    compatibility: Optional[CrossCommitCompatibility] = None
    compatibility_arguments = (
        compatibility_golden_directory,
        compatibility_archive_descriptor,
        compatibility_raw_archive,
    )
    if cross_commit_contract is None:
        if any(value is not None for value in compatibility_arguments):
            raise AnalysisValidationError("compatibility ground-truth inputs require cross_commit_contract")
    else:
        if compatibility_golden_directory is None:
            raise AnalysisValidationError("cross_commit_contract requires compatibility_golden_directory")
        compatibility = load_cross_commit_compatibility(cross_commit_contract)
        _verify_cross_commit_ground_truth(
            compatibility,
            loaded,
            golden_directory=compatibility_golden_directory,
            archive_descriptor=compatibility_archive_descriptor,
            raw_archive=compatibility_raw_archive,
            analyzer_record=analyzer_record,
        )
    archive = verify_archive_descriptor(_archive_bindings(loaded), archive_descriptor, raw_archive)
    aggregate = _build_aggregate(
        loaded,
        regeneration_command=command,
        raw_archive=archive,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        relative_threshold_pct=float(relative_threshold_pct),
        absolute_threshold_us=float(absolute_threshold_us),
        cross_commit_compatibility=compatibility,
        analyzer_record=analyzer_record,
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
    cross_commit_contract: Optional[PathLike] = None,
    compatibility_golden_directory: Optional[PathLike] = None,
    compatibility_archive_descriptor: Optional[PathLike] = None,
    compatibility_raw_archive: Optional[PathLike] = None,
    executor_artifact: Optional[ExecutorArtifact] = None,
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
        cross_commit_contract=cross_commit_contract,
        compatibility_golden_directory=compatibility_golden_directory,
        compatibility_archive_descriptor=compatibility_archive_descriptor,
        compatibility_raw_archive=compatibility_raw_archive,
        executor_artifact=executor_artifact,
    )
