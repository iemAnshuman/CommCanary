"""Immutable terminal attempt records for manifest cells.

An attempt is evidence, including when it fails.  This module keeps terminal
records append-only, binds every record to one frozen manifest cell, and makes
the record plus checksum visible as one atomically-renamed directory.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from .canonical import (
    CHECKSUM_MAX_BYTES,
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    ContractError,
    canonical_json_bytes,
    contained_path,
    file_sha256,
    read_bounded_bytes,
    read_bounded_text,
    safe_slug,
    sha256_hex,
)
from .manifest import FrozenRun, load_frozen_run
from .model import CELL_ID_MAX_LENGTH, FrozenJSON, RunManifest

PathLike = Union[str, "Path"]

ATTEMPT_SCHEMA = "commcanary.experiment.cell-attempt.v1"
ATTEMPTS_DIRNAME = "attempts"
ATTEMPT_RECORD_FILENAME = "attempt.json"
ATTEMPT_SHA256_FILENAME = "attempt.sha256"
TERMINAL_STATUSES = frozenset({"success", "failed", "parse-failed", "cancelled", "excluded"})

_ATTEMPT_ID_RE = re.compile(r"^a-([0-9]{6})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  attempt\.json\n$")
_MAX_ATTEMPT_NUMBER = 999_999
_MAX_COMMAND_ARGUMENTS = 1024
_MAX_COMMAND_BYTES = 1024 * 1024


class AttemptValidationError(ContractError):
    """Raised when an attempt record violates its schema or manifest binding."""


class AttemptStoreError(AttemptValidationError):
    """Raised when immutable attempt storage is missing, corrupt, or colliding."""


class ArtifactVerificationError(AttemptValidationError):
    """Raised when a referenced artifact is missing, unsafe, or stale."""

    def __init__(self, code: str, reference: "ArtifactReference", detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.reference = reference


def _expect_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptValidationError(f"{field} must be an object")
    return value


def _strict_fields(
    value: Mapping[str, Any],
    *,
    field: str,
    required: Iterable[str],
) -> None:
    required_set = set(required)
    actual = set(value)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - required_set)
    if missing:
        raise AttemptValidationError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise AttemptValidationError(f"{field} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AttemptValidationError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _optional_token(value: Any, field: str, *, max_length: int = 256) -> Optional[str]:
    if value is None:
        return None
    return _external_token(value, field, max_length=max_length)


def _external_token(value: Any, field: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise AttemptValidationError(f"{field} must be a non-empty string of at most {max_length} characters")
    if value != value.strip() or _CONTROL_RE.search(value) or "/" in value or "\\" in value:
        raise AttemptValidationError(
            f"{field} may not contain surrounding whitespace, control characters, or path separators"
        )
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise AttemptValidationError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return cast(int, value)


def _timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise AttemptValidationError(f"{field} must be a canonical UTC timestamp with microseconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise AttemptValidationError(f"{field} is not a valid UTC timestamp") from exc
    return value


def utc_timestamp(value: Optional[datetime] = None) -> str:
    """Return a canonical microsecond-resolution UTC timestamp."""

    instant = datetime.now(timezone.utc) if value is None else value
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise AttemptValidationError("timestamp datetime must be timezone-aware")
    instant = instant.astimezone(timezone.utc)
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def derive_attempt_id(attempt_number: int) -> str:
    """Derive the stable identifier for a one-based cell attempt number."""

    number = _integer(
        attempt_number,
        "attempt.attempt_number",
        minimum=1,
        maximum=_MAX_ATTEMPT_NUMBER,
    )
    return f"a-{number:06d}"


def _relative_artifact_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise AttemptValidationError(f"{field} must be a non-empty relative path of at most 512 characters")
    if _CONTROL_RE.search(value) or "\\" in value:
        raise AttemptValidationError(f"{field} contains control characters or backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise AttemptValidationError(f"{field} must be a normalized relative POSIX path without traversal")
    return value


@dataclass(frozen=True)
class ArtifactReference:
    """Content-addressed reference to an artifact below the frozen run."""

    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_dict(cls, raw: Any, field: str = "artifact") -> "ArtifactReference":
        data = _expect_object(raw, field)
        _strict_fields(data, field=field, required=("path", "sha256", "size_bytes"))
        return cls(
            path=_relative_artifact_path(data["path"], f"{field}.path"),
            sha256=_sha256(data["sha256"], f"{field}.sha256"),
            size_bytes=_integer(
                data["size_bytes"],
                f"{field}.size_bytes",
                minimum=0,
                maximum=2**63 - 1,
            ),
        )

    def validate(self, field: str = "artifact") -> None:
        _relative_artifact_path(self.path, f"{field}.path")
        _sha256(self.sha256, f"{field}.sha256")
        _integer(self.size_bytes, f"{field}.size_bytes", minimum=0, maximum=2**63 - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class VerifiedArtifact:
    """A reference whose contained regular file matches its size and digest."""

    reference: ArtifactReference
    path: Path


def verify_artifact_reference(run_directory: PathLike, reference: ArtifactReference) -> VerifiedArtifact:
    """Verify one content-addressed artifact below a run directory."""

    reference.validate()
    root = Path(run_directory).expanduser().resolve()
    parts = PurePosixPath(reference.path).parts
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ArtifactVerificationError(
                "artifact-invalid",
                reference,
                f"artifact path contains a symlink: {reference.path}",
            )
    path = contained_path(root, *parts)
    if not path.exists():
        raise ArtifactVerificationError(
            "artifact-missing",
            reference,
            f"artifact does not exist: {reference.path}",
        )
    if not path.is_file():
        raise ArtifactVerificationError(
            "artifact-invalid",
            reference,
            f"artifact is not a regular file: {reference.path}",
        )
    observed_size = path.stat().st_size
    if observed_size != reference.size_bytes:
        raise ArtifactVerificationError(
            "artifact-stale",
            reference,
            f"artifact size mismatch for {reference.path}: expected {reference.size_bytes}, observed {observed_size}",
        )
    observed_sha256 = file_sha256(path)
    if observed_sha256 != reference.sha256:
        raise ArtifactVerificationError(
            "artifact-stale",
            reference,
            f"artifact SHA-256 mismatch for {reference.path}: expected {reference.sha256}, observed {observed_sha256}",
        )
    return VerifiedArtifact(reference=reference, path=path)


@dataclass(frozen=True)
class ObservedExecution:
    """Observed execution metadata, deliberately kept out of the manifest."""

    executor: str
    site_id: str
    hostname: str
    scheduler: Optional[str]
    job_id: Optional[str]
    nodes: Tuple[str, ...]
    account: Optional[str]
    partition: Optional[str]
    metadata: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "ObservedExecution":
        data = _expect_object(raw, "attempt.observed")
        _strict_fields(
            data,
            field="attempt.observed",
            required=(
                "executor",
                "site_id",
                "hostname",
                "scheduler",
                "job_id",
                "nodes",
                "account",
                "partition",
                "metadata",
            ),
        )
        raw_nodes = data["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise AttemptValidationError("attempt.observed.nodes must be a non-empty array")
        nodes = tuple(sorted(_external_token(item, "attempt.observed.nodes[]") for item in raw_nodes))
        if len(set(nodes)) != len(nodes):
            raise AttemptValidationError("attempt.observed.nodes contains duplicates")
        metadata = _expect_object(data["metadata"], "attempt.observed.metadata")
        result = cls(
            executor=safe_slug(data["executor"], field="attempt.observed.executor"),
            site_id=safe_slug(data["site_id"], field="attempt.observed.site_id"),
            hostname=_external_token(data["hostname"], "attempt.observed.hostname"),
            scheduler=_optional_token(data["scheduler"], "attempt.observed.scheduler"),
            job_id=_optional_token(data["job_id"], "attempt.observed.job_id"),
            nodes=nodes,
            account=_optional_token(data["account"], "attempt.observed.account"),
            partition=_optional_token(data["partition"], "attempt.observed.partition"),
            metadata=FrozenJSON.from_value(metadata, "attempt.observed.metadata"),
        )
        result.validate()
        return result

    def validate(self) -> None:
        safe_slug(self.executor, field="attempt.observed.executor")
        safe_slug(self.site_id, field="attempt.observed.site_id")
        _external_token(self.hostname, "attempt.observed.hostname")
        if self.scheduler is not None:
            _external_token(self.scheduler, "attempt.observed.scheduler")
        if self.job_id is not None:
            _external_token(self.job_id, "attempt.observed.job_id")
        if self.job_id is not None and self.scheduler is None:
            raise AttemptValidationError("attempt.observed.job_id requires a scheduler")
        if not self.nodes or tuple(sorted(self.nodes)) != self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise AttemptValidationError("attempt.observed.nodes must be non-empty, sorted, and unique")
        for node in self.nodes:
            _external_token(node, "attempt.observed.nodes[]")
        if self.account is not None:
            _external_token(self.account, "attempt.observed.account")
        if self.partition is not None:
            _external_token(self.partition, "attempt.observed.partition")
        self.metadata.validate("attempt.observed.metadata")
        _expect_object(self.metadata.to_value(), "attempt.observed.metadata")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor": self.executor,
            "site_id": self.site_id,
            "hostname": self.hostname,
            "scheduler": self.scheduler,
            "job_id": self.job_id,
            "nodes": list(self.nodes),
            "account": self.account,
            "partition": self.partition,
            "metadata": self.metadata.to_value(),
        }


def _optional_artifact(raw: Any, field: str) -> Optional[ArtifactReference]:
    return None if raw is None else ArtifactReference.from_dict(raw, field)


def _validate_command(raw: Any) -> Tuple[str, ...]:
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_COMMAND_ARGUMENTS:
        raise AttemptValidationError(f"attempt.command must contain 1..{_MAX_COMMAND_ARGUMENTS} string arguments")
    command: List[str] = []
    total_bytes = 0
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item or "\x00" in item:
            raise AttemptValidationError(f"attempt.command[{index}] must be a non-empty NUL-free string")
        total_bytes += len(item.encode("utf-8"))
        if total_bytes > _MAX_COMMAND_BYTES:
            raise AttemptValidationError(f"attempt.command exceeds the {_MAX_COMMAND_BYTES}-byte storage budget")
        command.append(item)
    return tuple(command)


@dataclass(frozen=True)
class AttemptRecord:
    """One immutable terminal outcome for a manifest cell."""

    schema: str
    run_id: str
    manifest_sha256: str
    cell_id: str
    cell_identity_sha256: str
    attempt_id: str
    attempt_number: int
    status: str
    started_at: str
    finished_at: str
    command: Tuple[str, ...]
    observed: ObservedExecution
    exit_code: Optional[int]
    reason: Optional[str]
    stdout: Optional[ArtifactReference]
    stderr: Optional[ArtifactReference]
    measurement: Optional[ArtifactReference]
    partial_outputs: Tuple[ArtifactReference, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "AttemptRecord":
        data = _expect_object(raw, "attempt")
        _strict_fields(
            data,
            field="attempt",
            required=(
                "schema",
                "run_id",
                "manifest_sha256",
                "cell_id",
                "cell_identity_sha256",
                "attempt_id",
                "attempt_number",
                "status",
                "started_at",
                "finished_at",
                "command",
                "observed",
                "exit_code",
                "reason",
                "stdout",
                "stderr",
                "measurement",
                "partial_outputs",
            ),
        )
        if data["schema"] != ATTEMPT_SCHEMA:
            raise AttemptValidationError(f"unsupported attempt schema {data['schema']!r}")
        status = data["status"]
        if not isinstance(status, str) or status not in TERMINAL_STATUSES:
            raise AttemptValidationError("attempt.status must be success, failed, parse-failed, cancelled, or excluded")
        raw_exit_code = data["exit_code"]
        exit_code = (
            None
            if raw_exit_code is None
            else _integer(raw_exit_code, "attempt.exit_code", minimum=-(2**31), maximum=2**31 - 1)
        )
        reason = data["reason"]
        if reason is not None:
            if not isinstance(reason, str) or not reason or len(reason) > 4096 or "\x00" in reason:
                raise AttemptValidationError(
                    "attempt.reason must be null or a non-empty NUL-free string of at most 4096 characters"
                )
        raw_partial = data["partial_outputs"]
        if not isinstance(raw_partial, list):
            raise AttemptValidationError("attempt.partial_outputs must be an array")
        partial_outputs = tuple(
            sorted(
                (
                    ArtifactReference.from_dict(item, f"attempt.partial_outputs[{index}]")
                    for index, item in enumerate(raw_partial)
                ),
                key=lambda item: item.path,
            )
        )
        result = cls(
            schema=ATTEMPT_SCHEMA,
            run_id=safe_slug(data["run_id"], field="attempt.run_id"),
            manifest_sha256=_sha256(data["manifest_sha256"], "attempt.manifest_sha256"),
            cell_id=safe_slug(data["cell_id"], field="attempt.cell_id", max_length=CELL_ID_MAX_LENGTH),
            cell_identity_sha256=_sha256(
                data["cell_identity_sha256"],
                "attempt.cell_identity_sha256",
            ),
            attempt_id=safe_slug(data["attempt_id"], field="attempt.attempt_id"),
            attempt_number=_integer(
                data["attempt_number"],
                "attempt.attempt_number",
                minimum=1,
                maximum=_MAX_ATTEMPT_NUMBER,
            ),
            status=status,
            started_at=_timestamp(data["started_at"], "attempt.started_at"),
            finished_at=_timestamp(data["finished_at"], "attempt.finished_at"),
            command=_validate_command(data["command"]),
            observed=ObservedExecution.from_dict(data["observed"]),
            exit_code=exit_code,
            reason=reason,
            stdout=_optional_artifact(data["stdout"], "attempt.stdout"),
            stderr=_optional_artifact(data["stderr"], "attempt.stderr"),
            measurement=_optional_artifact(data["measurement"], "attempt.measurement"),
            partial_outputs=partial_outputs,
        )
        result.validate()
        return result

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "AttemptRecord":
        from .canonical import strict_json_loads

        return cls.from_dict(strict_json_loads(data))

    def validate(self) -> None:
        if self.schema != ATTEMPT_SCHEMA:
            raise AttemptValidationError(f"unsupported attempt schema {self.schema!r}")
        safe_slug(self.run_id, field="attempt.run_id")
        _sha256(self.manifest_sha256, "attempt.manifest_sha256")
        safe_slug(self.cell_id, field="attempt.cell_id", max_length=CELL_ID_MAX_LENGTH)
        _sha256(self.cell_identity_sha256, "attempt.cell_identity_sha256")
        if self.attempt_id != derive_attempt_id(self.attempt_number):
            raise AttemptValidationError("attempt.attempt_id does not match attempt.attempt_number")
        if self.status not in TERMINAL_STATUSES:
            raise AttemptValidationError("attempt.status is not terminal")
        started = _timestamp(self.started_at, "attempt.started_at")
        finished = _timestamp(self.finished_at, "attempt.finished_at")
        if finished < started:
            raise AttemptValidationError("attempt.finished_at precedes attempt.started_at")
        _validate_command(list(self.command))
        self.observed.validate()
        if self.exit_code is not None:
            _integer(self.exit_code, "attempt.exit_code", minimum=-(2**31), maximum=2**31 - 1)
        if self.reason is not None and (not self.reason or len(self.reason) > 4096 or "\x00" in self.reason):
            raise AttemptValidationError("attempt.reason is invalid")
        for field, artifact in (
            ("attempt.stdout", self.stdout),
            ("attempt.stderr", self.stderr),
            ("attempt.measurement", self.measurement),
        ):
            if artifact is not None:
                artifact.validate(field)
        if tuple(sorted(self.partial_outputs, key=lambda item: item.path)) != self.partial_outputs:
            raise AttemptValidationError("attempt.partial_outputs must be sorted by path")
        partial_paths = [artifact.path for artifact in self.partial_outputs]
        if len(set(partial_paths)) != len(partial_paths):
            raise AttemptValidationError("attempt.partial_outputs contains duplicate paths")
        for index, artifact in enumerate(self.partial_outputs):
            artifact.validate(f"attempt.partial_outputs[{index}]")
        self._validate_status_contract()

    def _validate_status_contract(self) -> None:
        if self.status == "success":
            if self.exit_code != 0 or self.reason is not None or self.measurement is None:
                raise AttemptValidationError("a successful attempt requires exit_code 0, no reason, and a measurement")
            return
        if self.reason is None:
            raise AttemptValidationError(f"a {self.status} attempt requires a reason")
        if self.measurement is not None:
            raise AttemptValidationError(f"a {self.status} attempt cannot declare a valid measurement")
        if self.status in {"failed", "cancelled"} and self.exit_code == 0:
            raise AttemptValidationError(f"a {self.status} attempt cannot have exit_code 0")
        if self.status == "excluded" and self.exit_code is not None:
            raise AttemptValidationError("an excluded attempt cannot have an exit code")

    def validate_against_manifest(self, manifest: RunManifest, manifest_sha256: str) -> None:
        """Prove that this record belongs to exactly one cell in *manifest*."""

        self.validate()
        if self.run_id != manifest.run_id:
            raise AttemptValidationError("attempt.run_id does not match the frozen manifest")
        if self.manifest_sha256 != manifest_sha256:
            raise AttemptValidationError("attempt.manifest_sha256 does not match the frozen manifest")
        cells = {cell.id: cell for cell in manifest.cells}
        if self.cell_id not in cells:
            raise AttemptValidationError(f"attempt refers to unknown cell {self.cell_id!r}")
        if self.cell_identity_sha256 != cells[self.cell_id].identity_sha256:
            raise AttemptValidationError("attempt.cell_identity_sha256 does not match its manifest cell")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "cell_id": self.cell_id,
            "cell_identity_sha256": self.cell_identity_sha256,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command": list(self.command),
            "observed": self.observed.to_dict(),
            "exit_code": self.exit_code,
            "reason": self.reason,
            "stdout": None if self.stdout is None else self.stdout.to_dict(),
            "stderr": None if self.stderr is None else self.stderr.to_dict(),
            "measurement": None if self.measurement is None else self.measurement.to_dict(),
            "partial_outputs": [artifact.to_dict() for artifact in self.partial_outputs],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def artifact_references(self) -> Tuple[ArtifactReference, ...]:
        """Return every declared artifact in deterministic role order."""

        references: List[ArtifactReference] = []
        for reference in (self.stdout, self.stderr, self.measurement):
            if reference is not None:
                references.append(reference)
        references.extend(self.partial_outputs)
        return tuple(references)

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_json_bytes())


@dataclass(frozen=True)
class FrozenAttempt:
    directory: Path
    record_path: Path
    checksum_path: Path
    record_sha256: str


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


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


def _known_cell(manifest: RunManifest, cell_id: str) -> None:
    safe_slug(cell_id, field="cell_id", max_length=CELL_ID_MAX_LENGTH)
    if cell_id not in {cell.id for cell in manifest.cells}:
        raise AttemptStoreError(f"unknown manifest cell {cell_id!r}")


def _attempts_root(frozen: FrozenRun, *, create: bool) -> Path:
    root = contained_path(frozen.directory, ATTEMPTS_DIRNAME)
    if root.is_symlink():
        raise AttemptStoreError("attempts directory may not be a symlink")
    if create:
        root.mkdir(exist_ok=True)
        root = contained_path(frozen.directory, ATTEMPTS_DIRNAME)
    return root


def _cell_attempt_directory(frozen: FrozenRun, cell_id: str, *, create: bool) -> Path:
    root = _attempts_root(frozen, create=create)
    cell_directory = contained_path(root, cell_id)
    if cell_directory.is_symlink():
        raise AttemptStoreError("cell attempt directory may not be a symlink")
    if create:
        cell_directory.mkdir(exist_ok=True)
        cell_directory = contained_path(root, cell_id)
    return cell_directory


def _load_attempt_directory(
    manifest: RunManifest,
    frozen: FrozenRun,
    directory: Path,
) -> Tuple[AttemptRecord, FrozenAttempt]:
    if directory.is_symlink() or not directory.is_dir():
        raise AttemptStoreError(f"attempt path is not a real directory: {directory}")
    match = _ATTEMPT_ID_RE.fullmatch(directory.name)
    if match is None:
        raise AttemptStoreError(f"invalid attempt directory name {directory.name!r}")
    record_path = directory / ATTEMPT_RECORD_FILENAME
    checksum_path = directory / ATTEMPT_SHA256_FILENAME
    expected_entries = {ATTEMPT_RECORD_FILENAME, ATTEMPT_SHA256_FILENAME}
    actual_entries = {path.name for path in directory.iterdir()}
    if actual_entries != expected_entries:
        raise AttemptStoreError(f"attempt directory {directory.name!r} has unexpected or missing entries")
    if record_path.is_symlink() or checksum_path.is_symlink():
        raise AttemptStoreError("attempt record and checksum may not be symlinks")
    try:
        checksum_text = read_bounded_text(
            checksum_path,
            max_bytes=CHECKSUM_MAX_BYTES,
            field="attempt checksum",
            encoding="ascii",
        )
    except CanonicalJSONError as exc:
        raise AttemptStoreError(f"cannot read attempt checksum: {exc}") from exc
    checksum_match = _CHECKSUM_RE.fullmatch(checksum_text)
    if checksum_match is None:
        raise AttemptStoreError("attempt checksum file has invalid syntax")
    expected_sha256 = checksum_match.group(1)
    try:
        raw = read_bounded_bytes(
            record_path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field="attempt record",
        )
    except CanonicalJSONError as exc:
        raise AttemptStoreError(str(exc)) from exc
    if sha256_hex(raw) != expected_sha256:
        raise AttemptStoreError("attempt record SHA-256 does not match its checksum")
    try:
        record = AttemptRecord.from_json_bytes(raw)
    except CanonicalJSONError as exc:
        raise AttemptStoreError(f"invalid attempt record JSON: {exc}") from exc
    if raw != record.to_json_bytes():
        raise AttemptStoreError("attempt record is valid JSON but not in canonical byte form")
    if record.attempt_id != directory.name:
        raise AttemptStoreError("attempt directory name does not match record attempt_id")
    if record.cell_id != directory.parent.name:
        raise AttemptStoreError("cell attempt directory name does not match record cell_id")
    record.validate_against_manifest(manifest, frozen.manifest_sha256)
    return record, FrozenAttempt(
        directory=directory,
        record_path=record_path,
        checksum_path=checksum_path,
        record_sha256=expected_sha256,
    )


def _load_cell_attempts(
    manifest: RunManifest,
    frozen: FrozenRun,
    cell_id: str,
    *,
    ignored_names: Sequence[str] = (),
) -> Tuple[AttemptRecord, ...]:
    _known_cell(manifest, cell_id)
    attempts_root = _attempts_root(frozen, create=False)
    if not attempts_root.exists():
        return ()
    cell_directory = _cell_attempt_directory(frozen, cell_id, create=False)
    if not cell_directory.exists():
        return ()
    ignored = set(ignored_names)
    directories: List[Path] = []
    for entry in cell_directory.iterdir():
        if entry.name in ignored:
            continue
        if _ATTEMPT_ID_RE.fullmatch(entry.name) is None or entry.is_symlink() or not entry.is_dir():
            raise AttemptStoreError(f"unexpected entry in cell attempt directory: {entry.name!r}")
        directories.append(entry)
    records = tuple(
        record
        for record, _frozen_attempt in (
            _load_attempt_directory(manifest, frozen, directory)
            for directory in sorted(directories, key=lambda path: path.name)
        )
    )
    expected_numbers = tuple(range(1, len(records) + 1))
    actual_numbers = tuple(record.attempt_number for record in records)
    if actual_numbers != expected_numbers:
        raise AttemptStoreError(f"attempt sequence for cell {cell_id!r} is not contiguous from one")
    return records


def write_attempt_record(run_directory: PathLike, record: AttemptRecord) -> FrozenAttempt:
    """Append one terminal record without overwriting any earlier attempt."""

    manifest, frozen = load_frozen_run(run_directory)
    record.validate_against_manifest(manifest, frozen.manifest_sha256)
    cell_directory = _cell_attempt_directory(frozen, record.cell_id, create=True)
    lock_name = f".{record.attempt_id}.lock"
    lock_path = contained_path(cell_directory, lock_name)
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise AttemptStoreError(f"attempt slot {record.attempt_id!r} is already being written") from exc

    destination = contained_path(cell_directory, record.attempt_id)
    temporary: Optional[Path] = None
    try:
        existing = _load_cell_attempts(
            manifest,
            frozen,
            record.cell_id,
            ignored_names=(lock_name,),
        )
        expected_number = len(existing) + 1
        if record.attempt_number != expected_number:
            raise AttemptStoreError(
                f"attempt number {record.attempt_number} cannot be appended; expected {expected_number}"
            )
        if destination.exists() or destination.is_symlink():
            raise AttemptStoreError(f"attempt directory already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{record.attempt_id}.tmp-", dir=str(cell_directory)))
        record_path = temporary / ATTEMPT_RECORD_FILENAME
        checksum_path = temporary / ATTEMPT_SHA256_FILENAME
        record_bytes = record.to_json_bytes()
        record_sha256 = record.sha256
        _write_exclusive(record_path, record_bytes)
        _write_exclusive(
            checksum_path,
            f"{record_sha256}  {ATTEMPT_RECORD_FILENAME}\n".encode("ascii"),
        )
        os.chmod(record_path, 0o444)
        os.chmod(checksum_path, 0o444)
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise AttemptStoreError(f"attempt directory already exists: {destination}") from exc
            raise
        temporary = None
        _fsync_directory(cell_directory)
        return FrozenAttempt(
            directory=destination,
            record_path=destination / ATTEMPT_RECORD_FILENAME,
            checksum_path=destination / ATTEMPT_SHA256_FILENAME,
            record_sha256=record_sha256,
        )
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


def load_attempt_record(
    run_directory: PathLike,
    cell_id: str,
    attempt_id: str,
) -> Tuple[AttemptRecord, FrozenAttempt]:
    """Load one attempt by its manifest cell and stable attempt ID."""

    manifest, frozen = load_frozen_run(run_directory)
    _known_cell(manifest, cell_id)
    match = _ATTEMPT_ID_RE.fullmatch(attempt_id)
    if match is None:
        raise AttemptStoreError(f"invalid attempt id {attempt_id!r}")
    cell_directory = _cell_attempt_directory(frozen, cell_id, create=False)
    directory = contained_path(cell_directory, attempt_id)
    if not directory.exists() and not directory.is_symlink():
        raise AttemptStoreError(f"attempt does not exist: {attempt_id}")
    return _load_attempt_directory(manifest, frozen, directory)


def load_cell_attempts(run_directory: PathLike, cell_id: str) -> Tuple[AttemptRecord, ...]:
    """Load and validate every immutable attempt for one manifest cell."""

    manifest, frozen = load_frozen_run(run_directory)
    return _load_cell_attempts(manifest, frozen, cell_id)


def verify_attempt_artifacts(
    run_directory: PathLike,
    record: AttemptRecord,
) -> Tuple[VerifiedArtifact, ...]:
    """Verify the manifest binding and all artifacts declared by an attempt."""

    manifest, frozen = load_frozen_run(run_directory)
    record.validate_against_manifest(manifest, frozen.manifest_sha256)
    return tuple(verify_artifact_reference(frozen.directory, reference) for reference in record.artifact_references())


def select_terminal_attempt(
    attempts: Sequence[AttemptRecord],
    attempt_id: str,
) -> AttemptRecord:
    """Resolve one explicit selection without an implicit latest/success rule."""

    if _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise AttemptValidationError(f"invalid selected attempt id {attempt_id!r}")
    matches = [attempt for attempt in attempts if attempt.attempt_id == attempt_id]
    if len(matches) != 1:
        raise AttemptValidationError(f"selected attempt {attempt_id!r} must match exactly one terminal record")
    matches[0].validate()
    return matches[0]
