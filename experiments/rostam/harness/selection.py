"""Immutable, explicit attempt-selection snapshots.

A selection is never inferred from recency or success.  Callers name the exact
attempt for each cell, freeze that mapping under a stable selection ID, and
then pass the snapshot to completeness validation.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .attempts import AttemptValidationError, load_attempt_record
from .canonical import (
    CHECKSUM_MAX_BYTES,
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    canonical_json_bytes,
    contained_path,
    read_bounded_bytes,
    read_bounded_text,
    safe_slug,
    sha256_hex,
    strict_json_loads,
)
from .manifest import FrozenRun, load_frozen_run
from .model import CELL_ID_MAX_LENGTH

PathLike = Union[str, "Path"]

SELECTION_SCHEMA = "commcanary.experiment.attempt-selection.v1"
SELECTIONS_DIRNAME = "selections"
SELECTION_FILENAME = "selection.json"
SELECTION_SHA256_FILENAME = "selection.sha256"

_ATTEMPT_ID_RE = re.compile(r"^a-[0-9]{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  selection\.json\n$")


class SelectionValidationError(AttemptValidationError):
    """Raised when a selection snapshot violates its structural contract."""


class SelectionStoreError(SelectionValidationError):
    """Raised when immutable selection storage is corrupt or colliding."""


def _expect_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionValidationError(f"{field} must be an object")
    return value


def _strict_fields(value: Mapping[str, Any], *, field: str, required: Iterable[str]) -> None:
    required_set = set(required)
    actual = set(value)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - required_set)
    if missing:
        raise SelectionValidationError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SelectionValidationError(f"{field} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SelectionValidationError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _attempt_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ATTEMPT_ID_RE.fullmatch(value):
        raise SelectionValidationError(f"{field} must have the form a-NNNNNN")
    return value


@dataclass(frozen=True)
class SelectionEntry:
    """One explicit selected record, cryptographically bound to its cell."""

    cell_id: str
    cell_identity_sha256: str
    attempt_id: str
    attempt_record_sha256: str

    @classmethod
    def from_dict(cls, raw: Any) -> "SelectionEntry":
        data = _expect_object(raw, "selection.entry")
        _strict_fields(
            data,
            field="selection.entry",
            required=(
                "cell_id",
                "cell_identity_sha256",
                "attempt_id",
                "attempt_record_sha256",
            ),
        )
        return cls(
            cell_id=safe_slug(
                data["cell_id"],
                field="selection.entry.cell_id",
                max_length=CELL_ID_MAX_LENGTH,
            ),
            cell_identity_sha256=_sha256(
                data["cell_identity_sha256"],
                "selection.entry.cell_identity_sha256",
            ),
            attempt_id=_attempt_id(data["attempt_id"], "selection.entry.attempt_id"),
            attempt_record_sha256=_sha256(
                data["attempt_record_sha256"],
                "selection.entry.attempt_record_sha256",
            ),
        )

    def validate(self) -> None:
        safe_slug(self.cell_id, field="selection.entry.cell_id", max_length=CELL_ID_MAX_LENGTH)
        _sha256(self.cell_identity_sha256, "selection.entry.cell_identity_sha256")
        _attempt_id(self.attempt_id, "selection.entry.attempt_id")
        _sha256(self.attempt_record_sha256, "selection.entry.attempt_record_sha256")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_identity_sha256": self.cell_identity_sha256,
            "attempt_id": self.attempt_id,
            "attempt_record_sha256": self.attempt_record_sha256,
        }


def _entry_key(entry: SelectionEntry) -> Tuple[str, str, str, str]:
    return (
        entry.cell_id,
        entry.attempt_id,
        entry.cell_identity_sha256,
        entry.attempt_record_sha256,
    )


@dataclass(frozen=True)
class SelectionSnapshot:
    """A named immutable view of which terminal attempt each cell uses."""

    schema: str
    selection_id: str
    run_id: str
    manifest_sha256: str
    entries: Tuple[SelectionEntry, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "SelectionSnapshot":
        data = _expect_object(raw, "selection")
        _strict_fields(
            data,
            field="selection",
            required=("schema", "selection_id", "run_id", "manifest_sha256", "entries"),
        )
        if data["schema"] != SELECTION_SCHEMA:
            raise SelectionValidationError(f"unsupported selection schema {data['schema']!r}")
        raw_entries = data["entries"]
        if not isinstance(raw_entries, list):
            raise SelectionValidationError("selection.entries must be an array")
        result = cls(
            schema=SELECTION_SCHEMA,
            selection_id=safe_slug(data["selection_id"], field="selection.selection_id"),
            run_id=safe_slug(data["run_id"], field="selection.run_id"),
            manifest_sha256=_sha256(data["manifest_sha256"], "selection.manifest_sha256"),
            entries=tuple(SelectionEntry.from_dict(item) for item in raw_entries),
        )
        result.validate()
        return result

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "SelectionSnapshot":
        return cls.from_dict(strict_json_loads(data))

    def validate(self) -> None:
        if self.schema != SELECTION_SCHEMA:
            raise SelectionValidationError(f"unsupported selection schema {self.schema!r}")
        safe_slug(self.selection_id, field="selection.selection_id")
        safe_slug(self.run_id, field="selection.run_id")
        _sha256(self.manifest_sha256, "selection.manifest_sha256")
        for entry in self.entries:
            entry.validate()
        if tuple(sorted(self.entries, key=_entry_key)) != self.entries:
            raise SelectionValidationError("selection.entries must be sorted canonically")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "selection_id": self.selection_id,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_json_bytes())


@dataclass(frozen=True)
class FrozenSelection:
    directory: Path
    selection_path: Path
    checksum_path: Path
    selection_sha256: str


def build_selection_snapshot(
    run_directory: PathLike,
    selection_id: str,
    selected_attempts: Mapping[str, str],
) -> SelectionSnapshot:
    """Build a structurally valid snapshot from explicit cell-to-attempt IDs."""

    manifest, frozen = load_frozen_run(run_directory)
    safe_slug(selection_id, field="selection.selection_id")
    entries: List[SelectionEntry] = []
    for cell_id, attempt_id in selected_attempts.items():
        record, stored = load_attempt_record(frozen.directory, cell_id, attempt_id)
        cell = next(cell for cell in manifest.cells if cell.id == cell_id)
        entries.append(
            SelectionEntry(
                cell_id=cell_id,
                cell_identity_sha256=cell.identity_sha256,
                attempt_id=attempt_id,
                attempt_record_sha256=stored.record_sha256,
            )
        )
        if record.cell_identity_sha256 != cell.identity_sha256:  # defensive after loader validation
            raise SelectionValidationError("selected attempt has a stale cell identity")
    snapshot = SelectionSnapshot(
        schema=SELECTION_SCHEMA,
        selection_id=selection_id,
        run_id=manifest.run_id,
        manifest_sha256=frozen.manifest_sha256,
        entries=tuple(sorted(entries, key=_entry_key)),
    )
    snapshot.validate()
    return snapshot


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


def _selection_root(frozen: FrozenRun, *, create: bool) -> Path:
    root = contained_path(frozen.directory, SELECTIONS_DIRNAME)
    if root.is_symlink():
        raise SelectionStoreError("selections directory may not be a symlink")
    if create:
        root.mkdir(exist_ok=True)
        root = contained_path(frozen.directory, SELECTIONS_DIRNAME)
    return root


def freeze_selection_snapshot(run_directory: PathLike, snapshot: SelectionSnapshot) -> FrozenSelection:
    """Atomically freeze a named selection without replacing prior views."""

    manifest, frozen = load_frozen_run(run_directory)
    snapshot.validate()
    if snapshot.run_id != manifest.run_id:
        raise SelectionValidationError("selection.run_id does not match the frozen manifest")
    if snapshot.manifest_sha256 != frozen.manifest_sha256:
        raise SelectionValidationError("selection.manifest_sha256 does not match the frozen manifest")
    root = _selection_root(frozen, create=True)
    destination = contained_path(root, snapshot.selection_id)
    lock_path = contained_path(root, f".{snapshot.selection_id}.lock")
    if destination.exists() or destination.is_symlink():
        raise SelectionStoreError(f"selection directory already exists: {destination}")
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise SelectionStoreError(f"selection {snapshot.selection_id!r} is already being frozen") from exc

    temporary: Optional[Path] = None
    try:
        if destination.exists() or destination.is_symlink():
            raise SelectionStoreError(f"selection directory already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot.selection_id}.tmp-", dir=str(root)))
        selection_path = temporary / SELECTION_FILENAME
        checksum_path = temporary / SELECTION_SHA256_FILENAME
        selection_bytes = snapshot.to_json_bytes()
        selection_sha256 = snapshot.sha256
        _write_exclusive(selection_path, selection_bytes)
        _write_exclusive(
            checksum_path,
            f"{selection_sha256}  {SELECTION_FILENAME}\n".encode("ascii"),
        )
        os.chmod(selection_path, 0o444)
        os.chmod(checksum_path, 0o444)
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise SelectionStoreError(f"selection directory already exists: {destination}") from exc
            raise
        temporary = None
        _fsync_directory(root)
        return FrozenSelection(
            directory=destination,
            selection_path=destination / SELECTION_FILENAME,
            checksum_path=destination / SELECTION_SHA256_FILENAME,
            selection_sha256=selection_sha256,
        )
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


def load_selection_snapshot(
    run_directory: PathLike,
    selection_id: str,
) -> Tuple[SelectionSnapshot, FrozenSelection]:
    """Load a frozen selection and verify its checksum and canonical bytes."""

    _manifest, frozen = load_frozen_run(run_directory)
    safe_slug(selection_id, field="selection.selection_id")
    root = _selection_root(frozen, create=False)
    directory = contained_path(root, selection_id)
    if directory.is_symlink() or not directory.is_dir():
        raise SelectionStoreError(f"selection does not exist as a real directory: {selection_id}")
    selection_path = directory / SELECTION_FILENAME
    checksum_path = directory / SELECTION_SHA256_FILENAME
    allowed_entries = {SELECTION_FILENAME, SELECTION_SHA256_FILENAME, "verdicts"}
    actual_entries = {path.name for path in directory.iterdir()}
    if not {SELECTION_FILENAME, SELECTION_SHA256_FILENAME}.issubset(actual_entries):
        raise SelectionStoreError("selection directory is missing its record or checksum")
    if actual_entries - allowed_entries:
        raise SelectionStoreError("selection directory contains unexpected entries")
    if selection_path.is_symlink() or checksum_path.is_symlink():
        raise SelectionStoreError("selection record and checksum may not be symlinks")
    try:
        checksum_text = read_bounded_text(
            checksum_path,
            max_bytes=CHECKSUM_MAX_BYTES,
            field="selection checksum",
            encoding="ascii",
        )
    except CanonicalJSONError as exc:
        raise SelectionStoreError(f"cannot read selection checksum: {exc}") from exc
    match = _CHECKSUM_RE.fullmatch(checksum_text)
    if match is None:
        raise SelectionStoreError("selection checksum file has invalid syntax")
    expected_sha256 = match.group(1)
    try:
        raw = read_bounded_bytes(
            selection_path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field="selection record",
        )
    except CanonicalJSONError as exc:
        raise SelectionStoreError(str(exc)) from exc
    if sha256_hex(raw) != expected_sha256:
        raise SelectionStoreError("selection SHA-256 does not match its checksum")
    try:
        snapshot = SelectionSnapshot.from_json_bytes(raw)
    except CanonicalJSONError as exc:
        raise SelectionStoreError(f"invalid selection JSON: {exc}") from exc
    if raw != snapshot.to_json_bytes():
        raise SelectionStoreError("selection is valid JSON but not in canonical byte form")
    if snapshot.selection_id != directory.name:
        raise SelectionStoreError("selection directory name does not match selection_id")
    return snapshot, FrozenSelection(
        directory=directory,
        selection_path=selection_path,
        checksum_path=checksum_path,
        selection_sha256=expected_sha256,
    )
