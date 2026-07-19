"""Precompute, freeze, and explicitly submit a Rostam SLURM ownership plan."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..harness import (
    CHECKSUM_MAX_BYTES,
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    derive_attempt_id,
    file_sha256,
    load_cell_attempts,
    load_frozen_run,
    read_bounded_bytes,
    read_bounded_text,
    strict_json_loads,
)
from .physical_results import validate_physical_layout

PathLike = Union[str, "Path"]
SUBMISSION_PLAN_SCHEMA = "commcanary.rostam.submission-plan.v1"
SUBMISSION_LEDGER_SCHEMA = "commcanary.rostam.submission-ledger-entry.v1"
PLAN_DIRNAME = "submission-plans"
PLAN_FILENAME = "plan.json"
PLAN_SHA256_FILENAME = "plan.sha256"

_WRAPPERS = {
    "micro": "run_micro.sbatch",
    "full": "run_full.sbatch",
    "canary": "run_canary.sbatch",
    "shared-capture": "capture_shared_trace.sbatch",
    "shared": "run_shared.sbatch",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^[0-9]+$")


class SubmissionPlanError(ContractError):
    """Raised before any scheduler process is started."""


def _load_bounded_json(path: Path, field: str) -> Any:
    try:
        raw = read_bounded_bytes(
            path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field=field,
        )
        return strict_json_loads(raw)
    except CanonicalJSONError as exc:
        raise SubmissionPlanError(f"cannot load {field}: {exc}") from exc


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SubmissionPlanError(f"{field} must be an object")
    return value


def _fields(
    value: Mapping[str, Any],
    field: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise SubmissionPlanError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SubmissionPlanError(f"{field} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SubmissionPlanError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_path(path: Path, field: str, *, regular_file: bool = False) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SubmissionPlanError(f"{field} may not be a symlink")
    resolved = expanded.resolve()
    if regular_file and not resolved.is_file():
        raise SubmissionPlanError(f"{field} must be a regular file: {resolved}")
    return resolved


def _slurm_time(seconds: int) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 86_400:
        raise SubmissionPlanError("workload timeout_seconds must be an integer in [1, 86400]")
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class PlannedCell:
    sequence: int
    cell_id: str
    cell_identity_sha256: str
    configuration_id: str
    workload_id: str
    repetition: int
    action: str
    reason: str
    attempt_id: Optional[str]
    reuse_attempt_id: Optional[str]
    dependency_attempts: Tuple[Tuple[str, str], ...]
    scheduler_dependency_cells: Tuple[str, ...]
    wrapper_path: str
    sbatch_argv: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "cell_id": self.cell_id,
            "cell_identity_sha256": self.cell_identity_sha256,
            "configuration_id": self.configuration_id,
            "workload_id": self.workload_id,
            "repetition": self.repetition,
            "action": self.action,
            "reason": self.reason,
            "attempt_id": self.attempt_id,
            "reuse_attempt_id": self.reuse_attempt_id,
            "dependency_attempts": dict(self.dependency_attempts),
            "scheduler_dependency_cells": list(self.scheduler_dependency_cells),
            "wrapper_path": self.wrapper_path,
            "sbatch_argv": list(self.sbatch_argv),
        }


@dataclass(frozen=True)
class SubmissionPlan:
    schema: str
    manifest_sha256: str
    run_directory: str
    experiment_directory: str
    flags: Tuple[Tuple[str, bool], ...]
    input_hashes: Tuple[Tuple[str, str], ...]
    cells: Tuple[PlannedCell, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_sha256": self.manifest_sha256,
            "run_directory": self.run_directory,
            "experiment_directory": self.experiment_directory,
            "flags": dict(self.flags),
            "input_hashes": dict(self.input_hashes),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def plan_id(self) -> str:
        return f"p-{self.sha256[:24]}"


def _verify_bound_inputs(manifest: Any, experiment_directory: Optional[Path] = None) -> Tuple[Tuple[str, str], ...]:
    policy = _object(manifest.campaign.policy.to_value(), "campaign.policy")
    paths_raw = _object(policy.get("input_paths"), "campaign.policy.input_paths")
    artifacts = {artifact.id: artifact for artifact in manifest.campaign.inputs}
    if set(paths_raw) != set(artifacts):
        raise SubmissionPlanError("manifest input_paths ownership does not match campaign inputs")
    hashes: List[Tuple[str, str]] = []
    for input_id, artifact in sorted(artifacts.items()):
        raw_path = paths_raw[input_id]
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise SubmissionPlanError(f"input path for {input_id!r} is invalid")
        path = _safe_path(Path(raw_path), f"input {input_id!r}", regular_file=True)
        if path.stat().st_size != artifact.size_bytes or file_sha256(path) != artifact.sha256:
            raise SubmissionPlanError(f"manifest input {input_id!r} is missing or stale")
        hashes.append((input_id, artifact.sha256))
    for contract_id, expected_schema in (
        ("environment-lock", "commcanary.rostam.environment-contract.v1"),
        ("param-patch-contract", "commcanary.rostam.param-patch-contract.v1"),
    ):
        if contract_id not in paths_raw:
            raise SubmissionPlanError(f"manifest lacks required input {contract_id!r}")
        contract = _object(_load_bounded_json(Path(paths_raw[contract_id]), contract_id), contract_id)
        if contract.get("schema") != expected_schema or contract.get("status") != "reviewed":
            raise SubmissionPlanError(f"{contract_id} is not a reviewed {expected_schema} document")
    script_hashes = _object(policy.get("script_hashes"), "campaign.policy.script_hashes")
    if not script_hashes:
        raise SubmissionPlanError("manifest does not bind any physical execution scripts")
    if experiment_directory is None:
        catalog_path = Path(paths_raw["rostam-catalog"]).resolve()
        experiment_directory = catalog_path.parent
    for relative, expected_raw in sorted(script_hashes.items()):
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SubmissionPlanError(f"manifest execution script path is unsafe: {relative!r}")
        expected = _sha256(expected_raw, f"campaign.policy.script_hashes.{relative}")
        path = (experiment_directory / relative).resolve()
        try:
            path.relative_to(experiment_directory.resolve())
        except ValueError as exc:
            raise SubmissionPlanError(f"manifest execution script escapes experiment directory: {relative}") from exc
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise SubmissionPlanError(f"manifest execution script is missing or stale: {relative}")
        hashes.append((f"script:{relative}", expected))
    # Keep the in-memory inventory in the same canonical key order used when
    # plans are serialized and loaded.  A campaign input may sort after the
    # ``script:`` namespace (for example ``shared-param-trace``), so appending
    # script bindings after input bindings is not itself a canonical order.
    return tuple(sorted(hashes))


def _topological_workloads(workloads: Sequence[Any]) -> Tuple[str, ...]:
    by_id = {workload.id: workload for workload in workloads}
    result: List[str] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(workload_id: str) -> None:
        if workload_id in visited:
            return
        if workload_id in visiting:
            raise SubmissionPlanError("manifest workload graph contains a cycle")
        visiting.add(workload_id)
        for dependency in by_id[workload_id].depends_on:
            if dependency not in by_id:
                raise SubmissionPlanError(f"unknown workload dependency {dependency!r}")
            visit(dependency)
        visiting.remove(workload_id)
        visited.add(workload_id)
        result.append(workload_id)

    for workload_id in sorted(by_id):
        visit(workload_id)
    return tuple(result)


def _ordered_cells(manifest: Any) -> Tuple[Any, ...]:
    cells = {(cell.configuration_id, cell.workload_id, cell.repetition): cell for cell in manifest.cells}
    if len(cells) != len(manifest.cells):
        raise SubmissionPlanError("manifest contains duplicate cell ownership")
    configurations = [item.id for item in manifest.campaign.configurations]
    workloads = _topological_workloads(manifest.campaign.workloads)
    result = []
    for repetition in range(manifest.campaign.repetitions):
        for workload_id in workloads:
            for configuration_id in configurations:
                key = (configuration_id, workload_id, repetition)
                if key not in cells:
                    raise SubmissionPlanError(f"manifest matrix is missing cell ownership {key!r}")
                result.append(cells[key])
    if len({cell.id for cell in result}) != len(result):
        raise SubmissionPlanError("manifest contains duplicate stable cell IDs")
    return tuple(result)


def _attempt_action(
    attempts: Sequence[Any],
    *,
    resume: bool,
    only_missing: bool,
    retry_failed: bool,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    retryable = {"failed", "parse-failed", "cancelled"}
    if not attempts:
        if retry_failed and not resume:
            return "skip", "retry-failed skips a cell with no failed attempt", None, None
        return "run", "cell has no prior terminal attempt", derive_attempt_id(1), None
    latest = attempts[-1]
    successful = [attempt for attempt in attempts if attempt.status == "success"]
    if only_missing:
        reuse = successful[-1].attempt_id if successful else latest.attempt_id
        return "skip", "only-missing preserves every terminal attempt", None, reuse
    if resume:
        if successful:
            return "skip", "resume found a successful manifest-bound attempt", None, successful[-1].attempt_id
        if retry_failed and latest.status in retryable:
            return "run", "resume explicitly retries the latest failure", derive_attempt_id(len(attempts) + 1), None
        return "skip", "resume preserves failed evidence without --retry-failed", None, latest.attempt_id
    if retry_failed:
        if latest.status in retryable:
            return "run", "retry-failed retries the latest failure", derive_attempt_id(len(attempts) + 1), None
        return "skip", "retry-failed skips a latest non-failure", None, latest.attempt_id
    raise SubmissionPlanError("existing attempts require --resume, --only-missing, or --retry-failed")


_VENV_WHEEL_MARKER = "commcanary-wheel.sha256"


def _verify_configuration_venvs(manifest: Any, experiment_dir: Path) -> None:
    """Refuse to plan a submittable run against venvs setup.sh has not certified."""

    artifacts = {artifact.id: artifact for artifact in manifest.campaign.inputs}
    wheel = artifacts.get("commcanary-wheel")
    repository_root = experiment_dir.parent.parent
    for configuration in manifest.campaign.configurations:
        venv = _configuration_venv(repository_root, configuration)
        if not (venv / "bin" / "python").exists():
            raise SubmissionPlanError(
                f"configuration {configuration.id!r} venv has no interpreter; run setup.sh before planning"
            )
        if wheel is None:
            continue
        marker = venv / _VENV_WHEEL_MARKER
        if marker.is_symlink() or not marker.is_file():
            raise SubmissionPlanError(
                f"configuration {configuration.id!r} venv does not record its installed "
                "CommCanary wheel; archive it and rerun setup.sh"
            )
        recorded = read_bounded_text(marker, max_bytes=256, field="venv wheel marker").strip()
        if recorded != wheel.sha256:
            raise SubmissionPlanError(
                f"configuration {configuration.id!r} venv holds CommCanary wheel {recorded}, "
                f"but the manifest binds {wheel.sha256}; rerun setup.sh with the reviewed wheel"
            )


def _configuration_venv(repository_root: Path, configuration: Any) -> Path:
    parameters = _object(configuration.parameters.to_value(), "configuration.parameters")
    if set(parameters) != {"venv"} or not isinstance(parameters["venv"], str):
        raise SubmissionPlanError("physical configuration must declare exactly one relative venv path")
    raw = Path(parameters["venv"])
    if raw.is_absolute() or ".." in raw.parts:
        raise SubmissionPlanError("physical configuration venv must be repository-relative")
    path = (repository_root / raw).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise SubmissionPlanError("physical configuration venv escapes the repository") from exc
    return path


def _build_sbatch_argv(
    *,
    manifest: Any,
    manifest_sha256: str,
    run_directory: Path,
    experiment_directory: Path,
    cell: Any,
    configuration: Any,
    workload: Any,
    wrapper: str,
    attempt_id: str,
    dependency_attempts: Sequence[Tuple[str, str]],
) -> Tuple[str, ...]:
    site = manifest.campaign.expected_site
    parameters = _object(workload.parameters.to_value(), "workload.parameters")
    timeout_raw = parameters.get("timeout_seconds")
    if isinstance(timeout_raw, bool) or not isinstance(timeout_raw, int) or not 1 <= timeout_raw <= 86_400:
        raise SubmissionPlanError("workload timeout_seconds must be an integer in [1, 86400]")
    timeout = timeout_raw
    repository_root = experiment_directory.parent.parent
    venv_python = _configuration_venv(repository_root, configuration) / "bin" / "python"
    wrapper_path = experiment_directory / _WRAPPERS[wrapper]
    output_path = run_directory / "scheduler" / "%x-%j.out"
    if "," in str(experiment_directory):
        raise SubmissionPlanError("experiment directory may not contain a comma because SLURM --export uses commas")
    argv: List[str] = [
        "sbatch",
        "--parsable",
        f"--partition={site.partition}",
        f"--nodes={site.nodes}",
        f"--time={_slurm_time(timeout)}",
        f"--job-name=cc-{workload.id[:32]}",
        f"--output={output_path}",
        f"--export=NONE,COMMCANARY_EXPERIMENT_DIR={experiment_directory}",
    ]
    if site.exclusive:
        argv.append("--exclusive")
    if site.node_constraints:
        argv.append(f"--nodelist={','.join(site.node_constraints)}")
    if site.account is not None:
        argv.append(f"--account={site.account}")
    argv.extend(
        [
            str(wrapper_path),
            str(venv_python),
            "--run-directory",
            str(run_directory),
            "--cell-id",
            cell.id,
            "--attempt-id",
            attempt_id,
            "--manifest-sha256",
            manifest_sha256,
        ]
    )
    for dependency_cell, dependency_attempt in dependency_attempts:
        argv.extend(["--dependency-attempt", f"{dependency_cell}={dependency_attempt}"])
    return tuple(argv)


def build_submission_plan(
    run_directory: PathLike,
    experiment_directory: PathLike,
    *,
    resume: bool = False,
    only_missing: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    max_cells: Optional[int] = None,
) -> SubmissionPlan:
    """Validate the full matrix and decide every owner before any ``sbatch``."""

    for name, value in (
        ("resume", resume),
        ("only_missing", only_missing),
        ("retry_failed", retry_failed),
        ("dry_run", dry_run),
    ):
        if not isinstance(value, bool):
            raise SubmissionPlanError(f"{name} must be boolean")
    if max_cells is not None and (isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1):
        raise SubmissionPlanError("max_cells must be a positive integer")
    if only_missing and (resume or retry_failed):
        raise SubmissionPlanError("--only-missing cannot be combined with --resume or --retry-failed")
    manifest, frozen = load_frozen_run(run_directory)
    if manifest.campaign.expected_site.site_id != "rostam" or manifest.campaign.expected_site.scheduler != "slurm":
        raise SubmissionPlanError("submission plan requires the frozen rostam/slurm site contract")
    experiment_dir = _safe_path(Path(experiment_directory), "experiment_directory")
    if not experiment_dir.is_dir():
        raise SubmissionPlanError("experiment_directory must exist")
    input_hashes = _verify_bound_inputs(manifest, experiment_dir)
    if not dry_run:
        _verify_configuration_venvs(manifest, experiment_dir)
    configurations = {item.id: item for item in manifest.campaign.configurations}
    workloads = {item.id: item for item in manifest.campaign.workloads}
    ordered = _ordered_cells(manifest)
    decisions: Dict[str, Dict[str, Any]] = {}
    planned: List[PlannedCell] = []
    scheduled = 0
    for sequence, cell in enumerate(ordered):
        configuration = configurations[cell.configuration_id]
        workload = workloads[cell.workload_id]
        parameters = _object(workload.parameters.to_value(), "workload.parameters")
        validate_physical_layout(parameters)
        readiness = parameters.get("readiness", "ready")
        if readiness != "ready":
            raise SubmissionPlanError(
                f"workload {workload.id!r} is not submit-ready: {readiness}; collect and manifest-bind the target value first"
            )
        wrapper = parameters.get("wrapper")
        if wrapper not in _WRAPPERS:
            raise SubmissionPlanError(f"workload {workload.id!r} has unsupported wrapper {wrapper!r}")
        wrapper_path = experiment_dir / _WRAPPERS[wrapper]
        if wrapper_path.is_symlink() or not wrapper_path.is_file():
            raise SubmissionPlanError(f"wrapper is missing or unsafe: {wrapper_path}")
        attempts = load_cell_attempts(frozen.directory, cell.id)
        action, reason, attempt_id, reuse_attempt_id = _attempt_action(
            attempts,
            resume=resume,
            only_missing=only_missing,
            retry_failed=retry_failed,
        )
        if action == "run" and max_cells is not None and scheduled >= max_cells:
            # Low-footprint chunking for a shared cluster: the deferred tail is
            # replanned by the next --resume invocation once this chunk drains.
            action = "skip"
            attempt_id = None
            reason = "max-cells defers this cell to a later submission"
        dependency_attempts: List[Tuple[str, str]] = []
        scheduler_dependencies: List[str] = []
        for dependency_cell_id in cell.dependencies:
            if dependency_cell_id not in decisions:
                raise SubmissionPlanError("cell ordering failed to place a dependency before its consumer")
            dependency = decisions[dependency_cell_id]
            binding = dependency.get("attempt_id") or dependency.get("reuse_attempt_id")
            if binding is None or dependency.get("terminal_status") not in {None, "success"}:
                action = "blocked"
                attempt_id = None
                reason = f"dependency {dependency_cell_id} has no successful or planned attempt"
                break
            dependency_attempts.append((dependency_cell_id, binding))
            if dependency["action"] == "run":
                scheduler_dependencies.append(dependency_cell_id)
        if attempts and action != "run" and reuse_attempt_id is not None:
            terminal_status = next(attempt.status for attempt in attempts if attempt.attempt_id == reuse_attempt_id)
        else:
            terminal_status = None
        if action == "run" and attempt_id is not None:
            sbatch_argv = _build_sbatch_argv(
                manifest=manifest,
                manifest_sha256=frozen.manifest_sha256,
                run_directory=frozen.directory,
                experiment_directory=experiment_dir,
                cell=cell,
                configuration=configuration,
                workload=workload,
                wrapper=wrapper,
                attempt_id=attempt_id,
                dependency_attempts=dependency_attempts,
            )
        else:
            sbatch_argv = ()
        if action == "run":
            scheduled += 1
        decision = {
            "action": action,
            "attempt_id": attempt_id,
            "reuse_attempt_id": reuse_attempt_id,
            "terminal_status": terminal_status,
        }
        decisions[cell.id] = decision
        planned.append(
            PlannedCell(
                sequence=sequence,
                cell_id=cell.id,
                cell_identity_sha256=cell.identity_sha256,
                configuration_id=cell.configuration_id,
                workload_id=cell.workload_id,
                repetition=cell.repetition,
                action=action,
                reason=reason,
                attempt_id=attempt_id,
                reuse_attempt_id=reuse_attempt_id,
                dependency_attempts=tuple(sorted(dependency_attempts)),
                scheduler_dependency_cells=tuple(sorted(scheduler_dependencies)),
                wrapper_path=str(wrapper_path),
                sbatch_argv=sbatch_argv,
            )
        )
    return SubmissionPlan(
        schema=SUBMISSION_PLAN_SCHEMA,
        manifest_sha256=frozen.manifest_sha256,
        run_directory=str(frozen.directory),
        experiment_directory=str(experiment_dir),
        flags=tuple(
            sorted(
                {
                    "resume": resume,
                    "only_missing": only_missing,
                    "retry_failed": retry_failed,
                    "dry_run": dry_run,
                }.items()
            )
        ),
        input_hashes=input_hashes,
        cells=tuple(planned),
    )


def freeze_submission_plan(plan: SubmissionPlan) -> Path:
    plan_root = Path(plan.run_directory) / PLAN_DIRNAME
    plan_root.mkdir(exist_ok=True)
    destination = plan_root / plan.plan_id
    if destination.exists() or destination.is_symlink():
        existing = destination / PLAN_FILENAME
        expected = canonical_json_bytes(plan.to_dict())
        if existing.is_file() and not existing.is_symlink():
            try:
                observed = read_bounded_bytes(
                    existing,
                    max_bytes=len(expected),
                    field="existing submission plan",
                )
            except CanonicalJSONError:
                observed = None
            if observed == expected:
                return existing
        raise SubmissionPlanError(f"submission plan collision: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{plan.plan_id}.tmp-", dir=str(plan_root)))
    try:
        plan_path = temporary / PLAN_FILENAME
        checksum_path = temporary / PLAN_SHA256_FILENAME
        plan_path.write_bytes(canonical_json_bytes(plan.to_dict()))
        checksum_path.write_text(f"{plan.sha256}  {PLAN_FILENAME}\n", encoding="ascii")
        os.chmod(plan_path, 0o444)
        os.chmod(checksum_path, 0o444)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return destination / PLAN_FILENAME


def load_submission_plan(path: PathLike) -> SubmissionPlan:
    plan_path = _safe_path(Path(path), "submission plan", regular_file=True)
    try:
        raw_bytes = read_bounded_bytes(
            plan_path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field="submission plan",
        )
        raw = _object(strict_json_loads(raw_bytes), "submission plan")
    except CanonicalJSONError as exc:
        raise SubmissionPlanError(f"cannot load submission plan: {exc}") from exc
    _fields(
        raw,
        "submission plan",
        required=(
            "schema",
            "manifest_sha256",
            "run_directory",
            "experiment_directory",
            "flags",
            "input_hashes",
            "cells",
        ),
    )
    if raw["schema"] != SUBMISSION_PLAN_SCHEMA:
        raise SubmissionPlanError(f"unsupported submission plan schema {raw['schema']!r}")
    flags_raw = _object(raw["flags"], "submission plan.flags")
    _fields(flags_raw, "submission plan.flags", required=("resume", "only_missing", "retry_failed", "dry_run"))
    if any(not isinstance(value, bool) for value in flags_raw.values()):
        raise SubmissionPlanError("submission plan flags must be boolean")
    input_hashes_raw = _object(raw["input_hashes"], "submission plan.input_hashes")
    input_hashes = tuple(
        sorted((str(key), _sha256(value, f"input_hashes.{key}")) for key, value in input_hashes_raw.items())
    )
    cells_raw = raw["cells"]
    if not isinstance(cells_raw, list):
        raise SubmissionPlanError("submission plan.cells must be an array")
    cells: List[PlannedCell] = []
    for index, cell_raw in enumerate(cells_raw):
        data = _object(cell_raw, f"submission plan.cells[{index}]")
        _fields(
            data,
            f"submission plan.cells[{index}]",
            required=(
                "sequence",
                "cell_id",
                "cell_identity_sha256",
                "configuration_id",
                "workload_id",
                "repetition",
                "action",
                "reason",
                "attempt_id",
                "reuse_attempt_id",
                "dependency_attempts",
                "scheduler_dependency_cells",
                "wrapper_path",
                "sbatch_argv",
            ),
        )
        dependencies_raw = _object(data["dependency_attempts"], "dependency_attempts")
        scheduler_raw = data["scheduler_dependency_cells"]
        argv_raw = data["sbatch_argv"]
        if (
            not isinstance(scheduler_raw, list)
            or not isinstance(argv_raw, list)
            or any(not isinstance(item, str) or not item or "\x00" in item for item in argv_raw)
        ):
            raise SubmissionPlanError("submission plan cell scheduler dependencies or argv are invalid")
        cells.append(
            PlannedCell(
                sequence=data["sequence"],
                cell_id=data["cell_id"],
                cell_identity_sha256=_sha256(data["cell_identity_sha256"], "cell_identity_sha256"),
                configuration_id=data["configuration_id"],
                workload_id=data["workload_id"],
                repetition=data["repetition"],
                action=data["action"],
                reason=data["reason"],
                attempt_id=data["attempt_id"],
                reuse_attempt_id=data["reuse_attempt_id"],
                dependency_attempts=tuple(sorted((str(key), str(value)) for key, value in dependencies_raw.items())),
                scheduler_dependency_cells=tuple(scheduler_raw),
                wrapper_path=data["wrapper_path"],
                sbatch_argv=tuple(argv_raw),
            )
        )
    plan = SubmissionPlan(
        schema=SUBMISSION_PLAN_SCHEMA,
        manifest_sha256=_sha256(raw["manifest_sha256"], "manifest_sha256"),
        run_directory=raw["run_directory"],
        experiment_directory=raw["experiment_directory"],
        flags=tuple(sorted((str(key), bool(value)) for key, value in flags_raw.items())),
        input_hashes=input_hashes,
        cells=tuple(cells),
    )
    if raw_bytes != canonical_json_bytes(plan.to_dict()):
        raise SubmissionPlanError("submission plan is not in canonical byte form")
    checksum_path = plan_path.parent / PLAN_SHA256_FILENAME
    expected_checksum = f"{plan.sha256}  {PLAN_FILENAME}\n"
    try:
        checksum = read_bounded_text(
            checksum_path,
            max_bytes=CHECKSUM_MAX_BYTES,
            field="submission plan checksum",
            encoding="ascii",
        )
    except CanonicalJSONError as exc:
        raise SubmissionPlanError(f"cannot load submission plan checksum: {exc}") from exc
    if checksum != expected_checksum or plan_path.parent.name != plan.plan_id:
        raise SubmissionPlanError("submission plan checksum or directory identity is stale")
    return plan


def _ledger_path(plan: SubmissionPlan, cell: PlannedCell) -> Path:
    return Path(plan.run_directory) / "submissions" / plan.plan_id / f"{cell.sequence:06d}-{cell.cell_id}.json"


def _write_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
    except FileExistsError as exc:
        raise SubmissionPlanError(f"submission ledger entry already exists: {path}") from exc


def submit_frozen_plan(plan: SubmissionPlan, *, execute: bool) -> Tuple[Dict[str, Any], ...]:
    """Submit a reviewed plan; this is the only function that starts ``sbatch``."""

    if not execute:
        raise SubmissionPlanError("submission requires the explicit --execute acknowledgement")
    if dict(plan.flags).get("dry_run"):
        raise SubmissionPlanError("a --dry-run plan cannot be submitted; freeze a submit-ready plan")
    manifest, frozen = load_frozen_run(plan.run_directory)
    if frozen.manifest_sha256 != plan.manifest_sha256:
        raise SubmissionPlanError("submission plan is stale for the frozen manifest")
    current_inputs = _verify_bound_inputs(manifest, Path(plan.experiment_directory))
    if current_inputs != plan.input_hashes:
        raise SubmissionPlanError("submission plan input inventory is stale")
    (Path(plan.run_directory) / "scheduler").mkdir(exist_ok=True)
    jobs: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    for cell in plan.cells:
        if cell.action != "run":
            continue
        ledger_path = _ledger_path(plan, cell)
        if ledger_path.exists():
            existing = _object(_load_bounded_json(ledger_path, "submission ledger"), "submission ledger")
            if existing.get("status") != "submitted" or existing.get("cell_id") != cell.cell_id:
                raise SubmissionPlanError(f"existing submission ledger is not reusable: {ledger_path}")
            jobs[cell.cell_id] = str(existing["job_id"])
            rows.append(dict(existing))
            continue
        dependency_job_ids = []
        for dependency_cell in cell.scheduler_dependency_cells:
            if dependency_cell not in jobs:
                raise SubmissionPlanError(f"scheduler dependency {dependency_cell!r} lacks a submitted job ID")
            dependency_job_ids.append(jobs[dependency_cell])
        argv = list(cell.sbatch_argv)
        if dependency_job_ids:
            try:
                wrapper_index = argv.index(cell.wrapper_path)
            except ValueError as exc:
                raise SubmissionPlanError("planned sbatch argv no longer contains its wrapper") from exc
            argv.insert(wrapper_index, f"--dependency=afterok:{':'.join(dependency_job_ids)}")
        started_at = _timestamp()
        completed = subprocess.run(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        finished_at = _timestamp()
        raw_job_id = completed.stdout.strip().split(";", 1)[0]
        submitted = completed.returncode == 0 and _JOB_ID_RE.fullmatch(raw_job_id) is not None
        row = {
            "schema": SUBMISSION_LEDGER_SCHEMA,
            "plan_sha256": plan.sha256,
            "manifest_sha256": plan.manifest_sha256,
            "cell_id": cell.cell_id,
            "attempt_id": cell.attempt_id,
            "status": "submitted" if submitted else "submission-failed",
            "job_id": raw_job_id if submitted else None,
            "argv": argv,
            "started_at": started_at,
            "finished_at": finished_at,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        _write_ledger(ledger_path, row)
        rows.append(row)
        if not submitted:
            raise SubmissionPlanError(f"sbatch failed for {cell.cell_id}: {completed.stderr.strip()}")
        jobs[cell.cell_id] = raw_job_id
    return tuple(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--run-directory", type=Path, required=True)
    plan.add_argument("--experiment-directory", type=Path, required=True)
    plan.add_argument("--resume", action="store_true")
    plan.add_argument("--only-missing", action="store_true")
    plan.add_argument("--retry-failed", action="store_true")
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="schedule at most this many cells and defer the rest to a later submission",
    )
    submit = subparsers.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_submission_plan(
                args.run_directory,
                args.experiment_directory,
                resume=args.resume,
                only_missing=args.only_missing,
                retry_failed=args.retry_failed,
                dry_run=args.dry_run,
                max_cells=args.max_cells,
            )
            path = freeze_submission_plan(plan)
            result = {
                "plan_id": plan.plan_id,
                "plan_path": str(path),
                "plan_sha256": plan.sha256,
                "run": sum(cell.action == "run" for cell in plan.cells),
                "skip": sum(cell.action == "skip" for cell in plan.cells),
                "blocked": sum(cell.action == "blocked" for cell in plan.cells),
                "dry_run": dict(plan.flags)["dry_run"],
                "commands": [list(cell.sbatch_argv) for cell in plan.cells if cell.action == "run"],
            }
        else:
            plan = load_submission_plan(args.plan)
            rows = submit_frozen_plan(plan, execute=args.execute)
            result = {"plan_id": plan.plan_id, "submitted": len(rows), "ledger": list(rows)}
    except (SubmissionPlanError, ContractError, OSError, UnicodeError) as exc:
        raise SystemExit(f"submission plan error: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
