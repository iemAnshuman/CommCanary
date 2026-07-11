"""Strictly local, bounded execution for one frozen manifest cell.

This controls elapsed time, captured stdout/stderr, accepted result size, and
environment inheritance.  It is an evidence-preserving runner, not an OS
sandbox: it does not restrict child system calls or impose a total workspace
filesystem quota.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from .attempts import (
    ATTEMPT_SCHEMA,
    ArtifactReference,
    AttemptRecord,
    AttemptStoreError,
    FrozenAttempt,
    derive_attempt_id,
    load_attempt_record,
    load_cell_attempts,
    utc_timestamp,
    verify_attempt_artifacts,
    write_attempt_record,
)
from .canonical import (
    CanonicalJSONError,
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    contained_path,
    file_sha256,
    read_bounded_bytes,
    safe_slug,
    sha256_hex,
    strict_json_loads,
)
from .manifest import FrozenRun, load_frozen_run
from .model import CELL_ID_MAX_LENGTH, FrozenJSON, RunManifest

PathLike = Union[str, "Path"]

CELL_RESULT_SCHEMA = "commcanary.experiment.cell-result.v1"
WORKSPACES_DIRNAME = "workspaces"
EXECUTION_PLAN_FILENAME = "execution_plan.json"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"
RESULT_FILENAME = "result.json"

INHERITED_ENV_ALLOWLIST = frozenset({"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"})
RESERVED_ENV_KEYS = frozenset(
    {
        "COMMCANARY_ATTEMPT_ID",
        "COMMCANARY_CELL_ID",
        "COMMCANARY_CELL_IDENTITY_SHA256",
        "COMMCANARY_MEASUREMENT_SCHEMA",
        "COMMCANARY_PRODUCER_SCHEMA",
        "COMMCANARY_RESULT_PATH",
        "COMMCANARY_RESULT_SCHEMA",
        "COMMCANARY_RUN_ID",
    }
)

_MAX_ARGV_ITEMS = 1024
_MAX_ARGV_BYTES = 1024 * 1024
_MAX_ENV_ITEMS = 128
_MAX_ENV_BYTES = 64 * 1024
_MAX_ENV_VALUE_BYTES = 16 * 1024
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_CAPTURE_BYTES = 1024 * 1024 * 1024
_STREAM_CHUNK_BYTES = 64 * 1024


class RunnerValidationError(ContractError):
    """Raised when a local execution request violates the runner contract."""


class DependencyValidationError(RunnerValidationError):
    """Raised when explicit dependency ownership is absent or invalid."""


class ExistingAttemptError(RunnerValidationError):
    """Raised when default planning would duplicate an existing attempt."""


class StaleExecutionIdentityError(RunnerValidationError):
    """Raised when prior evidence does not match the requested execution identity."""


class WorkspaceCollisionError(RunnerValidationError):
    """Raised when a one-shot attempt workspace already exists."""


class ResultValidationError(RunnerValidationError):
    """Raised when a cell result violates its schema or expected ownership."""


class CellRunInterrupted(RunnerValidationError):
    """Raised after a keyboard interruption has been preserved as a terminal record."""

    def __init__(self, outcome: "CellRunOutcome") -> None:
        super().__init__("cell run was interrupted after its cancelled attempt was recorded")
        self.outcome = outcome


def _expect_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerValidationError(f"{field} must be an object")
    return value


def _strict_fields(value: Mapping[str, Any], *, field: str, required: Iterable[str]) -> None:
    required_set = set(required)
    actual = set(value)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - required_set)
    if missing:
        raise RunnerValidationError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise RunnerValidationError(f"{field} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunnerValidationError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _bounded_integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RunnerValidationError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return cast(int, value)


def _argv(value: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not value or len(value) > _MAX_ARGV_ITEMS:
        raise RunnerValidationError(f"argv must contain 1..{_MAX_ARGV_ITEMS} arguments")
    result: List[str] = []
    total = 0
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item or "\x00" in item:
            raise RunnerValidationError(f"argv[{index}] must be a non-empty NUL-free string")
        total += len(item.encode("utf-8"))
        if total > _MAX_ARGV_BYTES:
            raise RunnerValidationError(f"argv exceeds the {_MAX_ARGV_BYTES}-byte budget")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class CellResult:
    """Validated producer output for one manifest cell."""

    schema: str
    cell_id: str
    cell_identity_sha256: str
    producer_schema: str
    measurement_schema: str
    measurement: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "CellResult":
        data = _expect_object(raw, "result")
        _strict_fields(
            data,
            field="result",
            required=(
                "schema",
                "cell_id",
                "cell_identity_sha256",
                "producer_schema",
                "measurement_schema",
                "measurement",
            ),
        )
        if data["schema"] != CELL_RESULT_SCHEMA:
            raise ResultValidationError(f"unsupported cell result schema {data['schema']!r}")
        measurement = _expect_object(data["measurement"], "result.measurement")
        result = cls(
            schema=CELL_RESULT_SCHEMA,
            cell_id=safe_slug(data["cell_id"], field="result.cell_id", max_length=CELL_ID_MAX_LENGTH),
            cell_identity_sha256=_sha256(
                data["cell_identity_sha256"],
                "result.cell_identity_sha256",
            ),
            producer_schema=_result_schema_id(data["producer_schema"], "result.producer_schema"),
            measurement_schema=_result_schema_id(
                data["measurement_schema"],
                "result.measurement_schema",
            ),
            measurement=FrozenJSON.from_value(measurement, "result.measurement"),
        )
        result.validate()
        return result

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "CellResult":
        return cls.from_dict(strict_json_loads(data))

    def validate(self) -> None:
        if self.schema != CELL_RESULT_SCHEMA:
            raise ResultValidationError(f"unsupported cell result schema {self.schema!r}")
        safe_slug(self.cell_id, field="result.cell_id", max_length=CELL_ID_MAX_LENGTH)
        _sha256(self.cell_identity_sha256, "result.cell_identity_sha256")
        _result_schema_id(self.producer_schema, "result.producer_schema")
        _result_schema_id(self.measurement_schema, "result.measurement_schema")
        self.measurement.validate("result.measurement")
        _expect_object(self.measurement.to_value(), "result.measurement")

    def validate_expected(
        self,
        *,
        cell_id: str,
        cell_identity_sha256: str,
        producer_schema: str,
        measurement_schema: str,
    ) -> None:
        self.validate()
        expected = (
            cell_id,
            cell_identity_sha256,
            producer_schema,
            measurement_schema,
        )
        observed = (
            self.cell_id,
            self.cell_identity_sha256,
            self.producer_schema,
            self.measurement_schema,
        )
        if observed != expected:
            raise ResultValidationError("cell result ownership or schema does not match its execution plan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "cell_id": self.cell_id,
            "cell_identity_sha256": self.cell_identity_sha256,
            "producer_schema": self.producer_schema,
            "measurement_schema": self.measurement_schema,
            "measurement": self.measurement.to_value(),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_json_bytes())


def _result_schema_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ResultValidationError(f"{field} must be a non-empty string of at most 160 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    boundary = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if value[0] not in boundary or value[-1] not in boundary or any(character not in allowed for character in value):
        raise ResultValidationError(f"{field} is not a valid schema identifier")
    return value


def write_cell_result(path: PathLike, result: CellResult) -> Path:
    """Atomically create one canonical result file without replacing content."""

    result.validate()
    destination = Path(path).expanduser()
    parent = destination.parent.resolve()
    if not parent.is_dir() or parent.is_symlink():
        raise ResultValidationError("result parent must be an existing real directory")
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        raise ResultValidationError(f"result already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=str(parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(result.to_json_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ResultValidationError(f"result already exists: {destination}") from exc
        _fsync_directory(parent)
        return destination
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_cell_result(
    path: PathLike,
    *,
    cell_id: str,
    cell_identity_sha256: str,
    producer_schema: str,
    measurement_schema: str,
    max_bytes: int,
) -> CellResult:
    """Load one bounded canonical result and validate its exact ownership."""

    maximum = _bounded_integer(max_bytes, "max_bytes", minimum=1, maximum=_MAX_CAPTURE_BYTES)
    result_path = Path(path)
    if result_path.is_symlink() or not result_path.is_file():
        raise ResultValidationError("result path must be a real regular file")
    try:
        raw = read_bounded_bytes(result_path, max_bytes=maximum, field="cell result")
    except CanonicalJSONError as exc:
        raise ResultValidationError(str(exc)) from exc
    try:
        result = CellResult.from_json_bytes(raw)
    except CanonicalJSONError as exc:
        raise ResultValidationError(f"invalid cell result JSON: {exc}") from exc
    if raw != result.to_json_bytes():
        raise ResultValidationError("result is valid JSON but not in canonical byte form")
    result.validate_expected(
        cell_id=cell_id,
        cell_identity_sha256=cell_identity_sha256,
        producer_schema=producer_schema,
        measurement_schema=measurement_schema,
    )
    return result


@dataclass(frozen=True)
class DependencyBinding:
    cell_id: str
    cell_identity_sha256: str
    attempt_id: str
    attempt_record_sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "cell_id": self.cell_id,
            "cell_identity_sha256": self.cell_identity_sha256,
            "attempt_id": self.attempt_id,
            "attempt_record_sha256": self.attempt_record_sha256,
        }


@dataclass(frozen=True)
class CellRunPlan:
    """Reusable decision and immutable inputs for one local cell execution."""

    run_directory: Path
    manifest_sha256: str
    cell_id: str
    cell_identity_sha256: str
    producer_schema: str
    measurement_schema: str
    argv: Tuple[str, ...]
    environment: Tuple[Tuple[str, str], ...]
    environment_sha256: str
    dependencies: Tuple[DependencyBinding, ...]
    execution_identity_sha256: str
    action: str
    reason: str
    dry_run: bool
    attempt_number: Optional[int]
    attempt_id: Optional[str]
    reuse_attempt_id: Optional[str]
    timeout_seconds: int
    max_output_bytes: int
    max_result_bytes: int

    def environment_dict(self) -> Dict[str, str]:
        return dict(self.environment)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "cell_id": self.cell_id,
            "cell_identity_sha256": self.cell_identity_sha256,
            "producer_schema": self.producer_schema,
            "measurement_schema": self.measurement_schema,
            "argv": list(self.argv),
            "environment": dict(self.environment),
            "environment_sha256": self.environment_sha256,
            "dependencies": [binding.to_dict() for binding in self.dependencies],
            "execution_identity_sha256": self.execution_identity_sha256,
            "action": self.action,
            "reason": self.reason,
            "dry_run": self.dry_run,
            "attempt_number": self.attempt_number,
            "attempt_id": self.attempt_id,
            "reuse_attempt_id": self.reuse_attempt_id,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_result_bytes": self.max_result_bytes,
        }


@dataclass(frozen=True)
class CellRunOutcome:
    plan: CellRunPlan
    workspace: Optional[Path]
    record: Optional[AttemptRecord]
    frozen_attempt: Optional[FrozenAttempt]
    result: Optional[CellResult]


def _configuration_and_workload(manifest: RunManifest, cell_id: str) -> Tuple[Any, Any, Any]:
    cells = {cell.id: cell for cell in manifest.cells}
    if cell_id not in cells:
        raise RunnerValidationError(f"unknown manifest cell {cell_id!r}")
    cell = cells[cell_id]
    configuration = next(
        configuration for configuration in manifest.campaign.configurations if configuration.id == cell.configuration_id
    )
    workload = next(workload for workload in manifest.campaign.workloads if workload.id == cell.workload_id)
    return cell, configuration, workload


def _bounded_environment(
    configuration_environment: Mapping[str, Any],
    inherited_environment: Mapping[str, str],
) -> Tuple[Tuple[str, str], ...]:
    environment: Dict[str, str] = {}
    for key in sorted(INHERITED_ENV_ALLOWLIST):
        if key in inherited_environment:
            value = inherited_environment[key]
            if not isinstance(value, str):
                raise RunnerValidationError(f"inherited environment {key} must be a string")
            environment[key] = value
    for key, raw_value in configuration_environment.items():
        if key in RESERVED_ENV_KEYS:
            raise RunnerValidationError(f"configuration environment may not set reserved key {key}")
        if not isinstance(raw_value, str):
            raise RunnerValidationError(f"configuration environment {key} must be a string")
        environment[key] = raw_value
    if len(environment) > _MAX_ENV_ITEMS:
        raise RunnerValidationError(f"execution environment exceeds {_MAX_ENV_ITEMS} entries")
    total = 0
    for key, value in environment.items():
        if not key or "\x00" in key or "=" in key or "\x00" in value:
            raise RunnerValidationError(f"execution environment contains invalid entry {key!r}")
        value_bytes = len(value.encode("utf-8"))
        if value_bytes > _MAX_ENV_VALUE_BYTES:
            raise RunnerValidationError(f"execution environment value {key!r} exceeds {_MAX_ENV_VALUE_BYTES} bytes")
        total += len(key.encode("utf-8")) + value_bytes + 2
    if total > _MAX_ENV_BYTES:
        raise RunnerValidationError(f"execution environment exceeds the {_MAX_ENV_BYTES}-byte budget")
    return tuple(sorted(environment.items()))


def _dependency_bindings(
    frozen: FrozenRun,
    manifest: RunManifest,
    cell_id: str,
    dependency_attempts: Mapping[str, str],
) -> Tuple[DependencyBinding, ...]:
    cell, _configuration, _workload = _configuration_and_workload(manifest, cell_id)
    expected = set(cell.dependencies)
    actual = set(dependency_attempts)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: List[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unexpected:
            details.append(f"unexpected={unexpected!r}")
        raise DependencyValidationError("explicit dependency attempts do not match the cell: " + "; ".join(details))
    manifest_cells = {manifest_cell.id: manifest_cell for manifest_cell in manifest.cells}
    bindings: List[DependencyBinding] = []
    for dependency_cell_id in sorted(expected):
        attempt_id = dependency_attempts[dependency_cell_id]
        record, stored = load_attempt_record(
            frozen.directory,
            dependency_cell_id,
            attempt_id,
        )
        if record.status != "success":
            raise DependencyValidationError(f"dependency attempt {dependency_cell_id}/{attempt_id} is not successful")
        verify_attempt_artifacts(frozen.directory, record)
        bindings.append(
            DependencyBinding(
                cell_id=dependency_cell_id,
                cell_identity_sha256=manifest_cells[dependency_cell_id].identity_sha256,
                attempt_id=attempt_id,
                attempt_record_sha256=stored.record_sha256,
            )
        )
    return tuple(bindings)


def _execution_identity_payload(
    *,
    frozen: FrozenRun,
    manifest: RunManifest,
    cell_id: str,
    argv: Tuple[str, ...],
    environment_sha256: str,
    dependencies: Tuple[DependencyBinding, ...],
    timeout_seconds: int,
    max_output_bytes: int,
    max_result_bytes: int,
) -> Dict[str, Any]:
    cell, configuration, workload = _configuration_and_workload(manifest, cell_id)
    return {
        "manifest_sha256": frozen.manifest_sha256,
        "repository": manifest.campaign.repository.to_dict(),
        "inputs": [artifact.to_dict() for artifact in manifest.campaign.inputs],
        "cell_id": cell.id,
        "cell_identity_sha256": cell.identity_sha256,
        "configuration": configuration.to_dict(),
        "workload": workload.to_dict(),
        "argv": list(argv),
        "environment_sha256": environment_sha256,
        "dependencies": [binding.to_dict() for binding in dependencies],
        "execution_policy": {
            "timeout_seconds": timeout_seconds,
            "max_output_bytes_per_stream": max_output_bytes,
            "max_result_bytes": max_result_bytes,
        },
    }


def _stored_execution_identity(record: AttemptRecord) -> Optional[str]:
    metadata = _expect_object(record.observed.metadata.to_value(), "attempt.observed.metadata")
    value = metadata.get("execution_identity_sha256")
    return value if isinstance(value, str) else None


def _validate_previous_identities(
    attempts: Sequence[AttemptRecord],
    execution_identity_sha256: str,
    argv: Tuple[str, ...],
) -> None:
    for attempt in attempts:
        if _stored_execution_identity(attempt) != execution_identity_sha256:
            raise StaleExecutionIdentityError(
                f"attempt {attempt.attempt_id} does not match the requested manifest/code/config/input identity"
            )
        if attempt.command != argv:
            raise StaleExecutionIdentityError(f"attempt {attempt.attempt_id} command does not match the requested argv")


def plan_cell(
    run_directory: PathLike,
    cell_id: str,
    argv: Sequence[str],
    *,
    dependency_attempts: Optional[Mapping[str, str]] = None,
    inherited_environment: Optional[Mapping[str, str]] = None,
    resume: bool = False,
    only_missing: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    timeout_seconds: int = 300,
    max_output_bytes: int = 4 * 1024 * 1024,
    max_result_bytes: int = 4 * 1024 * 1024,
) -> CellRunPlan:
    """Plan one cell without creating a workspace or launching a process."""

    for flag_name, flag_value in (
        ("resume", resume),
        ("only_missing", only_missing),
        ("retry_failed", retry_failed),
        ("dry_run", dry_run),
    ):
        if not isinstance(flag_value, bool):
            raise RunnerValidationError(f"{flag_name} must be boolean")
    if only_missing and (resume or retry_failed):
        raise RunnerValidationError("only_missing cannot be combined with resume or retry_failed")
    timeout = _bounded_integer(
        timeout_seconds,
        "timeout_seconds",
        minimum=1,
        maximum=_MAX_TIMEOUT_SECONDS,
    )
    output_budget = _bounded_integer(
        max_output_bytes,
        "max_output_bytes",
        minimum=1,
        maximum=_MAX_CAPTURE_BYTES,
    )
    result_budget = _bounded_integer(
        max_result_bytes,
        "max_result_bytes",
        minimum=1,
        maximum=_MAX_CAPTURE_BYTES,
    )
    parsed_argv = _argv(argv)
    manifest, frozen = load_frozen_run(run_directory)
    if manifest.campaign.expected_site.site_id != "local" or manifest.campaign.expected_site.scheduler != "local":
        raise RunnerValidationError(
            "the strictly local runner requires expected_site site_id and scheduler to be 'local'"
        )
    safe_slug(cell_id, field="cell_id", max_length=CELL_ID_MAX_LENGTH)
    cell, configuration, workload = _configuration_and_workload(manifest, cell_id)
    raw_configuration_environment = _expect_object(
        configuration.environment.to_value(),
        "configuration.environment",
    )
    parent_environment = os.environ if inherited_environment is None else inherited_environment
    environment = _bounded_environment(raw_configuration_environment, parent_environment)
    environment_sha256 = canonical_sha256(dict(environment))
    dependencies = _dependency_bindings(
        frozen,
        manifest,
        cell_id,
        {} if dependency_attempts is None else dependency_attempts,
    )
    identity = canonical_sha256(
        _execution_identity_payload(
            frozen=frozen,
            manifest=manifest,
            cell_id=cell_id,
            argv=parsed_argv,
            environment_sha256=environment_sha256,
            dependencies=dependencies,
            timeout_seconds=timeout,
            max_output_bytes=output_budget,
            max_result_bytes=result_budget,
        )
    )
    attempts = load_cell_attempts(frozen.directory, cell_id)
    _validate_previous_identities(attempts, identity, parsed_argv)

    action = "run"
    reason = "cell has no prior attempts"
    reuse_attempt_id: Optional[str] = None
    if attempts:
        latest = attempts[-1]
        if only_missing:
            action = "skip"
            reason = "only-missing skips cells with any terminal attempt"
            reuse_attempt_id = latest.attempt_id
        elif resume:
            successful = [attempt for attempt in attempts if attempt.status == "success"]
            if successful:
                action = "skip"
                reason = "resume found a successful matching attempt"
                reuse_attempt_id = successful[-1].attempt_id
            elif retry_failed and latest.status in {"failed", "parse-failed", "cancelled"}:
                reason = "resume is explicitly retrying a failed terminal attempt"
            else:
                action = "skip"
                reason = "resume preserves non-success evidence unless a retryable failure is explicit"
                reuse_attempt_id = latest.attempt_id
        elif retry_failed:
            if latest.status not in {"failed", "parse-failed", "cancelled"}:
                action = "skip"
                reason = "retry-failed skips a latest attempt that is not retryable"
                reuse_attempt_id = latest.attempt_id
            else:
                reason = "retry-failed is retrying the latest failed terminal attempt"
        else:
            raise ExistingAttemptError(f"cell {cell_id!r} already has attempts; use an explicit planning mode")
    elif retry_failed and not resume:
        action = "skip"
        reason = "retry-failed skips a cell with no failed attempt"

    attempt_number = len(attempts) + 1 if action == "run" else None
    attempt_id = derive_attempt_id(attempt_number) if attempt_number is not None else None
    return CellRunPlan(
        run_directory=frozen.directory,
        manifest_sha256=frozen.manifest_sha256,
        cell_id=cell.id,
        cell_identity_sha256=cell.identity_sha256,
        producer_schema=workload.producer_schema,
        measurement_schema=workload.measurement_schema,
        argv=parsed_argv,
        environment=environment,
        environment_sha256=environment_sha256,
        dependencies=dependencies,
        execution_identity_sha256=identity,
        action=action,
        reason=reason,
        dry_run=dry_run,
        attempt_number=attempt_number,
        attempt_id=attempt_id,
        reuse_attempt_id=reuse_attempt_id,
        timeout_seconds=timeout,
        max_output_bytes=output_budget,
        max_result_bytes=result_budget,
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, data: bytes, *, mode: int = 0o444) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _workspace_directory(frozen: FrozenRun, plan: CellRunPlan) -> Path:
    if plan.attempt_id is None:
        raise RunnerValidationError("a runnable plan must have an attempt_id")
    root = contained_path(frozen.directory, WORKSPACES_DIRNAME)
    if root.is_symlink():
        raise WorkspaceCollisionError("workspaces root may not be a symlink")
    root.mkdir(exist_ok=True)
    root = contained_path(frozen.directory, WORKSPACES_DIRNAME)
    cell_directory = contained_path(root, plan.cell_id)
    if cell_directory.is_symlink():
        raise WorkspaceCollisionError("cell workspace root may not be a symlink")
    cell_directory.mkdir(exist_ok=True)
    cell_directory = contained_path(root, plan.cell_id)
    workspace = contained_path(cell_directory, plan.attempt_id)
    try:
        workspace.mkdir()
    except FileExistsError as exc:
        raise WorkspaceCollisionError(f"attempt workspace already exists: {workspace}") from exc
    _fsync_directory(cell_directory)
    return workspace


def _validate_plan_before_execution(plan: CellRunPlan) -> Tuple[RunManifest, FrozenRun]:
    if _argv(plan.argv) != plan.argv:
        raise RunnerValidationError("execution plan argv is not canonical")
    _bounded_integer(
        plan.timeout_seconds,
        "timeout_seconds",
        minimum=1,
        maximum=_MAX_TIMEOUT_SECONDS,
    )
    _bounded_integer(
        plan.max_output_bytes,
        "max_output_bytes",
        minimum=1,
        maximum=_MAX_CAPTURE_BYTES,
    )
    _bounded_integer(
        plan.max_result_bytes,
        "max_result_bytes",
        minimum=1,
        maximum=_MAX_CAPTURE_BYTES,
    )
    if plan.action not in {"run", "skip"} or not isinstance(plan.dry_run, bool):
        raise RunnerValidationError("execution plan action or dry_run flag is invalid")
    if not plan.reason or len(plan.reason) > 4096 or "\x00" in plan.reason:
        raise RunnerValidationError("execution plan reason is invalid")
    if tuple(sorted(plan.environment)) != plan.environment or len(dict(plan.environment)) != len(plan.environment):
        raise RunnerValidationError("execution plan environment must be sorted and unique")
    _bounded_environment(dict(plan.environment), {})
    manifest, frozen = load_frozen_run(plan.run_directory)
    if manifest.campaign.expected_site.site_id != "local" or manifest.campaign.expected_site.scheduler != "local":
        raise RunnerValidationError("execution plan targets a non-local site")
    if plan.manifest_sha256 != frozen.manifest_sha256:
        raise StaleExecutionIdentityError("execution plan is stale for the frozen manifest")
    cell, configuration, workload = _configuration_and_workload(manifest, plan.cell_id)
    if cell.identity_sha256 != plan.cell_identity_sha256:
        raise StaleExecutionIdentityError("execution plan has a stale cell identity")
    if workload.producer_schema != plan.producer_schema or workload.measurement_schema != plan.measurement_schema:
        raise StaleExecutionIdentityError("execution plan has stale producer/result schemas")
    plan_environment = plan.environment_dict()
    configuration_environment = _expect_object(
        configuration.environment.to_value(),
        "configuration.environment",
    )
    allowed_environment_keys = set(configuration_environment) | set(INHERITED_ENV_ALLOWLIST)
    if set(plan_environment) - allowed_environment_keys:
        raise StaleExecutionIdentityError("execution plan contains an environment key outside the allowlist")
    for key, value in configuration_environment.items():
        if plan_environment.get(key) != value:
            raise StaleExecutionIdentityError(f"execution plan does not preserve configuration environment key {key}")
    dependency_ids = [binding.cell_id for binding in plan.dependencies]
    if tuple(sorted(dependency_ids)) != tuple(dependency_ids) or len(set(dependency_ids)) != len(dependency_ids):
        raise DependencyValidationError("execution plan dependencies must be sorted and unique")
    if set(dependency_ids) != set(cell.dependencies):
        raise DependencyValidationError("execution plan dependencies do not match the manifest cell")
    if canonical_sha256(plan.environment_dict()) != plan.environment_sha256:
        raise StaleExecutionIdentityError("execution plan environment hash is stale")
    for binding in plan.dependencies:
        record, stored = load_attempt_record(
            frozen.directory,
            binding.cell_id,
            binding.attempt_id,
        )
        if record.status != "success" or stored.record_sha256 != binding.attempt_record_sha256:
            raise StaleExecutionIdentityError("execution plan dependency evidence is stale")
        if record.cell_identity_sha256 != binding.cell_identity_sha256:
            raise StaleExecutionIdentityError("execution plan dependency cell identity is stale")
        verify_attempt_artifacts(frozen.directory, record)
    expected_identity = canonical_sha256(
        _execution_identity_payload(
            frozen=frozen,
            manifest=manifest,
            cell_id=plan.cell_id,
            argv=plan.argv,
            environment_sha256=plan.environment_sha256,
            dependencies=plan.dependencies,
            timeout_seconds=plan.timeout_seconds,
            max_output_bytes=plan.max_output_bytes,
            max_result_bytes=plan.max_result_bytes,
        )
    )
    if expected_identity != plan.execution_identity_sha256:
        raise StaleExecutionIdentityError("execution plan identity does not match its committed inputs")
    attempts = load_cell_attempts(frozen.directory, plan.cell_id)
    _validate_previous_identities(attempts, plan.execution_identity_sha256, plan.argv)
    if plan.action == "run":
        if plan.attempt_number != len(attempts) + 1:
            raise AttemptStoreError(f"execution plan attempt number is stale; expected {len(attempts) + 1}")
        if plan.attempt_id != derive_attempt_id(plan.attempt_number):
            raise RunnerValidationError("execution plan attempt id is inconsistent")
    return manifest, frozen


def _capture_stream(
    stream: IO[bytes],
    destination: Path,
    limit: int,
    exceeded: threading.Event,
    failures: List[str],
) -> None:
    try:
        written = 0
        with destination.open("xb") as handle:
            while True:
                chunk = stream.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                remaining = max(0, limit - written)
                if remaining:
                    accepted = chunk[:remaining]
                    handle.write(accepted)
                    written += len(accepted)
                if len(chunk) > remaining:
                    exceeded.set()
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o444)
    except Exception as exc:  # thread boundary; converted into terminal failure evidence
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=0.5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _poll_sleep(seconds: float) -> None:
    """Small indirection so interruption handling can be tested deterministically."""

    time.sleep(seconds)


def _artifact_reference(run_directory: Path, path: Path) -> ArtifactReference:
    if path.is_symlink() or not path.is_file():
        raise RunnerValidationError(f"cannot reference non-regular artifact {path}")
    try:
        relative = path.resolve().relative_to(run_directory.resolve()).as_posix()
    except ValueError as exc:
        raise RunnerValidationError(f"artifact escapes frozen run: {path}") from exc
    return ArtifactReference(
        path=relative,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _optional_partial_result(run_directory: Path, path: Path) -> Tuple[ArtifactReference, ...]:
    if path.is_symlink() or not path.is_file():
        return ()
    return (_artifact_reference(run_directory, path),)


def _runtime_environment(plan: CellRunPlan, result_path: Path) -> Dict[str, str]:
    if plan.attempt_id is None:
        raise RunnerValidationError("runnable plan is missing attempt_id")
    environment = plan.environment_dict()
    environment.update(
        {
            "COMMCANARY_ATTEMPT_ID": plan.attempt_id,
            "COMMCANARY_CELL_ID": plan.cell_id,
            "COMMCANARY_CELL_IDENTITY_SHA256": plan.cell_identity_sha256,
            "COMMCANARY_MEASUREMENT_SCHEMA": plan.measurement_schema,
            "COMMCANARY_PRODUCER_SCHEMA": plan.producer_schema,
            "COMMCANARY_RESULT_PATH": str(result_path),
            "COMMCANARY_RESULT_SCHEMA": CELL_RESULT_SCHEMA,
            "COMMCANARY_RUN_ID": plan.run_directory.name,
        }
    )
    return environment


def _attempt_metadata(plan: CellRunPlan, *, output_limit_exceeded: bool) -> Dict[str, Any]:
    return {
        "execution_identity_sha256": plan.execution_identity_sha256,
        "execution_plan_sha256": canonical_sha256(plan.to_dict()),
        "environment_sha256": plan.environment_sha256,
        "dependency_attempts": [binding.to_dict() for binding in plan.dependencies],
        "timeout_seconds": plan.timeout_seconds,
        "max_output_bytes_per_stream": plan.max_output_bytes,
        "max_result_bytes": plan.max_result_bytes,
        "output_limit_exceeded": output_limit_exceeded,
        "result_schema": CELL_RESULT_SCHEMA,
    }


def run_cell(plan: CellRunPlan) -> CellRunOutcome:
    """Execute one planned local cell exactly once, without a shell."""

    _manifest, frozen = _validate_plan_before_execution(plan)
    if plan.action == "skip" or plan.dry_run:
        return CellRunOutcome(
            plan=plan,
            workspace=None,
            record=None,
            frozen_attempt=None,
            result=None,
        )
    if plan.action != "run" or plan.attempt_id is None or plan.attempt_number is None:
        raise RunnerValidationError("execution plan is neither a valid run nor skip action")

    workspace = _workspace_directory(frozen, plan)
    plan_path = workspace / EXECUTION_PLAN_FILENAME
    _write_exclusive(plan_path, canonical_json_bytes(plan.to_dict()))
    stdout_path = workspace / STDOUT_FILENAME
    stderr_path = workspace / STDERR_FILENAME
    result_path = workspace / RESULT_FILENAME
    started_at = utc_timestamp()
    started_monotonic = time.monotonic()
    output_exceeded = threading.Event()
    capture_failures: List[str] = []
    interrupted = False
    termination_reason: Optional[str] = None
    spawn_error: Optional[str] = None
    return_code: Optional[int] = None

    process: Optional[subprocess.Popen[bytes]] = None
    threads: List[threading.Thread] = []
    try:
        try:
            process = subprocess.Popen(
                plan.argv,
                cwd=str(workspace),
                env=_runtime_environment(plan, result_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            spawn_error = f"cannot start process: {type(exc).__name__}: {exc}"
            _write_exclusive(stdout_path, b"")
            _write_exclusive(stderr_path, spawn_error.encode("utf-8")[: plan.max_output_bytes])
        except KeyboardInterrupt:
            interrupted = True
            termination_reason = "execution interrupted by KeyboardInterrupt while starting process"
            _write_exclusive(stdout_path, b"")
            _write_exclusive(stderr_path, b"")
        if process is not None:
            if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
                raise RunnerValidationError("subprocess pipes were not created")
            threads = [
                threading.Thread(
                    target=_capture_stream,
                    args=(
                        process.stdout,
                        stdout_path,
                        plan.max_output_bytes,
                        output_exceeded,
                        capture_failures,
                    ),
                    daemon=True,
                ),
                threading.Thread(
                    target=_capture_stream,
                    args=(
                        process.stderr,
                        stderr_path,
                        plan.max_output_bytes,
                        output_exceeded,
                        capture_failures,
                    ),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            try:
                while process.poll() is None:
                    if output_exceeded.is_set():
                        termination_reason = (
                            f"stdout or stderr exceeded the {plan.max_output_bytes}-byte per-stream budget"
                        )
                        break
                    if time.monotonic() - started_monotonic >= plan.timeout_seconds:
                        termination_reason = f"execution exceeded the {plan.timeout_seconds}-second timeout"
                        break
                    _poll_sleep(0.01)
            except KeyboardInterrupt:
                interrupted = True
                termination_reason = "execution interrupted by KeyboardInterrupt"
            if termination_reason is not None:
                _terminate_process_group(process)
            return_code = process.wait()
            for thread in threads:
                thread.join(timeout=1.0)
            if any(thread.is_alive() for thread in threads):
                _terminate_process_group(process)
                for thread in threads:
                    thread.join(timeout=1.0)
            if any(thread.is_alive() for thread in threads):
                capture_failures.append("capture thread did not terminate")
            if output_exceeded.is_set() and termination_reason is None:
                termination_reason = f"stdout or stderr exceeded the {plan.max_output_bytes}-byte per-stream budget"
    finally:
        finished_at = utc_timestamp()

    result: Optional[CellResult] = None
    measurement: Optional[ArtifactReference] = None
    partial_outputs: Tuple[ArtifactReference, ...] = ()
    if spawn_error is not None:
        status = "failed"
        reason: Optional[str] = spawn_error
    elif interrupted or termination_reason is not None:
        status = "cancelled"
        reason = termination_reason or "execution cancelled"
        partial_outputs = _optional_partial_result(frozen.directory, result_path)
    elif capture_failures:
        status = "failed"
        reason = "output capture failed: " + "; ".join(capture_failures)
        partial_outputs = _optional_partial_result(frozen.directory, result_path)
    elif return_code != 0:
        status = "failed"
        reason = f"process exited with code {return_code}"
        partial_outputs = _optional_partial_result(frozen.directory, result_path)
    else:
        try:
            result = load_cell_result(
                result_path,
                cell_id=plan.cell_id,
                cell_identity_sha256=plan.cell_identity_sha256,
                producer_schema=plan.producer_schema,
                measurement_schema=plan.measurement_schema,
                max_bytes=plan.max_result_bytes,
            )
        except (ContractError, OSError) as exc:
            status = "parse-failed"
            reason = f"cannot validate result: {exc}"
            partial_outputs = _optional_partial_result(frozen.directory, result_path)
        else:
            status = "success"
            reason = None
            measurement = _artifact_reference(frozen.directory, result_path)

    if reason is not None:
        reason = reason.replace("\x00", "\\0")[:4096]

    stdout_reference = _artifact_reference(frozen.directory, stdout_path)
    stderr_reference = _artifact_reference(frozen.directory, stderr_path)
    record_exit_code = return_code
    if status in {"failed", "cancelled"} and record_exit_code == 0:
        record_exit_code = None
    hostname = socket.gethostname() or "localhost"
    record = AttemptRecord.from_dict(
        {
            "schema": ATTEMPT_SCHEMA,
            "run_id": frozen.directory.name,
            "manifest_sha256": plan.manifest_sha256,
            "cell_id": plan.cell_id,
            "cell_identity_sha256": plan.cell_identity_sha256,
            "attempt_id": plan.attempt_id,
            "attempt_number": plan.attempt_number,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": list(plan.argv),
            "observed": {
                "executor": "local",
                "site_id": "local",
                "hostname": hostname,
                "scheduler": None,
                "job_id": None,
                "nodes": [hostname],
                "account": None,
                "partition": None,
                "metadata": _attempt_metadata(
                    plan,
                    output_limit_exceeded=output_exceeded.is_set(),
                ),
            },
            "exit_code": record_exit_code,
            "reason": reason,
            "stdout": stdout_reference.to_dict(),
            "stderr": stderr_reference.to_dict(),
            "measurement": None if measurement is None else measurement.to_dict(),
            "partial_outputs": [reference.to_dict() for reference in partial_outputs],
        }
    )
    frozen_attempt = write_attempt_record(frozen.directory, record)
    verify_attempt_artifacts(frozen.directory, record)
    outcome = CellRunOutcome(
        plan=plan,
        workspace=workspace,
        record=record,
        frozen_attempt=frozen_attempt,
        result=result,
    )
    if interrupted:
        raise CellRunInterrupted(outcome)
    return outcome
