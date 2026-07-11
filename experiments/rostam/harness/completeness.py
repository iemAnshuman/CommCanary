"""Fail-closed completeness verdicts for frozen experiment selections."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from .attempts import (
    ATTEMPTS_DIRNAME,
    ArtifactVerificationError,
    AttemptRecord,
    load_cell_attempts,
    verify_artifact_reference,
)
from .canonical import (
    CHECKSUM_MAX_BYTES,
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    contained_path,
    file_sha256,
    read_bounded_bytes,
    read_bounded_text,
    safe_slug,
    sha256_hex,
)
from .manifest import FrozenRun, load_frozen_run
from .selection import (
    FrozenSelection,
    SelectionEntry,
    SelectionSnapshot,
    SelectionValidationError,
    load_selection_snapshot,
)

PathLike = Union[str, "Path"]

COMPLETENESS_SCHEMA = "commcanary.experiment.completeness-verdict.v1"
VERDICTS_DIRNAME = "verdicts"
VERDICT_FILENAME = "verdict.json"
VERDICT_SHA256_FILENAME = "verdict.sha256"

ISSUE_CODES = frozenset(
    {
        "artifact-invalid",
        "artifact-missing",
        "artifact-stale",
        "duplicate-selection",
        "invalid-attempt-store",
        "missing-attempt",
        "selected-attempt-failed",
        "selected-attempt-missing",
        "stale-selection",
        "unexpected-attempt-cell",
        "unexpected-selection",
        "unselected-cell",
    }
)


class CompletenessValidationError(SelectionValidationError):
    """Raised when a completeness verdict violates its structural contract."""


class CompletenessStoreError(CompletenessValidationError):
    """Raised when persisted completeness evidence is corrupt or colliding."""


class IncompleteCampaignError(CompletenessValidationError):
    """Raised by default when a campaign does not have a complete selection."""

    def __init__(self, verdict: "CompletenessVerdict") -> None:
        codes = ", ".join(sorted({issue.code for issue in verdict.issues}))
        super().__init__(f"campaign is incomplete: {codes}")
        self.verdict = verdict


def _expect_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompletenessValidationError(f"{field} must be an object")
    return value


def _strict_fields(value: Mapping[str, Any], *, field: str, required: Iterable[str]) -> None:
    required_set = set(required)
    actual = set(value)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - required_set)
    if missing:
        raise CompletenessValidationError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise CompletenessValidationError(f"{field} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompletenessValidationError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _optional_string(value: Any, field: str, *, max_length: int) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > max_length or "\x00" in value:
        raise CompletenessValidationError(
            f"{field} must be null or a non-empty NUL-free string of at most {max_length} characters"
        )
    return value


def _integer(value: Any, field: str, *, maximum: int = 1_000_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise CompletenessValidationError(f"{field} must be an integer in [0, {maximum}]")
    return cast(int, value)


@dataclass(frozen=True)
class CompletenessIssue:
    """One machine-readable reason that a selection is not complete."""

    code: str
    cell_id: Optional[str]
    attempt_id: Optional[str]
    artifact_path: Optional[str]
    detail: str

    @classmethod
    def from_dict(cls, raw: Any) -> "CompletenessIssue":
        data = _expect_object(raw, "verdict.issue")
        _strict_fields(
            data,
            field="verdict.issue",
            required=("code", "cell_id", "attempt_id", "artifact_path", "detail"),
        )
        result = cls(
            code=data["code"],
            cell_id=_optional_string(data["cell_id"], "verdict.issue.cell_id", max_length=255),
            attempt_id=_optional_string(
                data["attempt_id"],
                "verdict.issue.attempt_id",
                max_length=64,
            ),
            artifact_path=_optional_string(
                data["artifact_path"],
                "verdict.issue.artifact_path",
                max_length=512,
            ),
            detail=_optional_string(data["detail"], "verdict.issue.detail", max_length=4096) or "",
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.code not in ISSUE_CODES:
            raise CompletenessValidationError(f"unknown completeness issue code {self.code!r}")
        _optional_string(self.cell_id, "verdict.issue.cell_id", max_length=255)
        _optional_string(self.attempt_id, "verdict.issue.attempt_id", max_length=64)
        _optional_string(self.artifact_path, "verdict.issue.artifact_path", max_length=512)
        if not self.detail or len(self.detail) > 4096 or "\x00" in self.detail:
            raise CompletenessValidationError("verdict.issue.detail is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "artifact_path": self.artifact_path,
            "detail": self.detail,
        }


def _issue_key(issue: CompletenessIssue) -> Tuple[str, str, str, str, str]:
    return (
        issue.code,
        issue.cell_id or "",
        issue.attempt_id or "",
        issue.artifact_path or "",
        issue.detail,
    )


@dataclass(frozen=True)
class CompletenessVerdict:
    """Canonical verdict binding one selection to one observed attempt tree."""

    schema: str
    run_id: str
    manifest_sha256: str
    selection_id: str
    selection_sha256: str
    attempt_inventory_sha256: str
    allow_incomplete: bool
    complete: bool
    expected_cells: int
    attempted_cells: int
    selected_cells: int
    successful_selected_cells: int
    issues: Tuple[CompletenessIssue, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "CompletenessVerdict":
        data = _expect_object(raw, "verdict")
        _strict_fields(
            data,
            field="verdict",
            required=(
                "schema",
                "run_id",
                "manifest_sha256",
                "selection_id",
                "selection_sha256",
                "attempt_inventory_sha256",
                "allow_incomplete",
                "complete",
                "expected_cells",
                "attempted_cells",
                "selected_cells",
                "successful_selected_cells",
                "issues",
            ),
        )
        if data["schema"] != COMPLETENESS_SCHEMA:
            raise CompletenessValidationError(f"unsupported completeness schema {data['schema']!r}")
        raw_issues = data["issues"]
        if not isinstance(raw_issues, list):
            raise CompletenessValidationError("verdict.issues must be an array")
        allow_incomplete = data["allow_incomplete"]
        complete = data["complete"]
        if not isinstance(allow_incomplete, bool) or not isinstance(complete, bool):
            raise CompletenessValidationError("verdict completion flags must be boolean")
        result = cls(
            schema=COMPLETENESS_SCHEMA,
            run_id=safe_slug(data["run_id"], field="verdict.run_id"),
            manifest_sha256=_sha256(data["manifest_sha256"], "verdict.manifest_sha256"),
            selection_id=safe_slug(data["selection_id"], field="verdict.selection_id"),
            selection_sha256=_sha256(data["selection_sha256"], "verdict.selection_sha256"),
            attempt_inventory_sha256=_sha256(
                data["attempt_inventory_sha256"],
                "verdict.attempt_inventory_sha256",
            ),
            allow_incomplete=allow_incomplete,
            complete=complete,
            expected_cells=_integer(data["expected_cells"], "verdict.expected_cells"),
            attempted_cells=_integer(data["attempted_cells"], "verdict.attempted_cells"),
            selected_cells=_integer(data["selected_cells"], "verdict.selected_cells"),
            successful_selected_cells=_integer(
                data["successful_selected_cells"],
                "verdict.successful_selected_cells",
            ),
            issues=tuple(CompletenessIssue.from_dict(item) for item in raw_issues),
        )
        result.validate()
        return result

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "CompletenessVerdict":
        from .canonical import strict_json_loads

        return cls.from_dict(strict_json_loads(data))

    def validate(self) -> None:
        if self.schema != COMPLETENESS_SCHEMA:
            raise CompletenessValidationError(f"unsupported completeness schema {self.schema!r}")
        safe_slug(self.run_id, field="verdict.run_id")
        safe_slug(self.selection_id, field="verdict.selection_id")
        _sha256(self.manifest_sha256, "verdict.manifest_sha256")
        _sha256(self.selection_sha256, "verdict.selection_sha256")
        _sha256(self.attempt_inventory_sha256, "verdict.attempt_inventory_sha256")
        if not isinstance(self.allow_incomplete, bool) or not isinstance(self.complete, bool):
            raise CompletenessValidationError("verdict completion flags must be boolean")
        for field, value in (
            ("expected_cells", self.expected_cells),
            ("attempted_cells", self.attempted_cells),
            ("selected_cells", self.selected_cells),
            ("successful_selected_cells", self.successful_selected_cells),
        ):
            _integer(value, f"verdict.{field}")
        if self.attempted_cells > self.expected_cells:
            raise CompletenessValidationError("verdict.attempted_cells exceeds expected_cells")
        if self.selected_cells > self.expected_cells:
            raise CompletenessValidationError("verdict.selected_cells exceeds expected_cells")
        if self.successful_selected_cells > self.selected_cells:
            raise CompletenessValidationError("verdict.successful_selected_cells exceeds selected_cells")
        for issue in self.issues:
            issue.validate()
        if tuple(sorted(self.issues, key=_issue_key)) != self.issues:
            raise CompletenessValidationError("verdict.issues must be sorted canonically")
        if len(set(self.issues)) != len(self.issues):
            raise CompletenessValidationError("verdict.issues contains duplicates")
        if self.complete != (not self.issues):
            raise CompletenessValidationError("verdict.complete must be true exactly when issues are empty")
        if self.complete and (
            self.attempted_cells != self.expected_cells
            or self.selected_cells != self.expected_cells
            or self.successful_selected_cells != self.expected_cells
        ):
            raise CompletenessValidationError("a complete verdict must account for every expected cell")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "selection_id": self.selection_id,
            "selection_sha256": self.selection_sha256,
            "attempt_inventory_sha256": self.attempt_inventory_sha256,
            "allow_incomplete": self.allow_incomplete,
            "complete": self.complete,
            "expected_cells": self.expected_cells,
            "attempted_cells": self.attempted_cells,
            "selected_cells": self.selected_cells,
            "successful_selected_cells": self.successful_selected_cells,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_json_bytes())


def _issue(
    code: str,
    detail: str,
    *,
    cell_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
) -> CompletenessIssue:
    issue = CompletenessIssue(
        code=code,
        cell_id=cell_id,
        attempt_id=attempt_id,
        artifact_path=artifact_path,
        detail=detail,
    )
    issue.validate()
    return issue


def _path_descriptor(path: Path, relative: str) -> Dict[str, Any]:
    if path.is_symlink():
        return {"path": relative, "kind": "symlink"}
    if path.is_file():
        try:
            return {
                "path": relative,
                "kind": "file",
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        except OSError:
            return {"path": relative, "kind": "unreadable-file"}
    if path.is_dir():
        return {"path": relative, "kind": "directory"}
    return {"path": relative, "kind": "other"}


def _attempt_inventory_sha256(frozen: FrozenRun) -> str:
    root = frozen.directory / ATTEMPTS_DIRNAME
    descriptors: List[Dict[str, Any]] = []
    if not root.exists() and not root.is_symlink():
        return canonical_sha256({"entries": descriptors})
    descriptors.append(_path_descriptor(root, ATTEMPTS_DIRNAME))
    if root.is_symlink() or not root.is_dir():
        return canonical_sha256({"entries": descriptors})
    try:
        cell_entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        descriptors.append({"path": ATTEMPTS_DIRNAME, "kind": "unreadable-directory"})
        return canonical_sha256({"entries": descriptors})
    for cell_entry in cell_entries:
        cell_relative = f"{ATTEMPTS_DIRNAME}/{cell_entry.name}"
        descriptors.append(_path_descriptor(cell_entry, cell_relative))
        if cell_entry.is_symlink() or not cell_entry.is_dir():
            continue
        try:
            attempt_entries = sorted(cell_entry.iterdir(), key=lambda path: path.name)
        except OSError:
            descriptors.append({"path": cell_relative, "kind": "unreadable-directory"})
            continue
        for attempt_entry in attempt_entries:
            attempt_relative = f"{cell_relative}/{attempt_entry.name}"
            descriptors.append(_path_descriptor(attempt_entry, attempt_relative))
            if attempt_entry.is_symlink() or not attempt_entry.is_dir():
                continue
            try:
                record_entries = sorted(attempt_entry.iterdir(), key=lambda path: path.name)
            except OSError:
                descriptors.append({"path": attempt_relative, "kind": "unreadable-directory"})
                continue
            for record_entry in record_entries:
                descriptors.append(
                    _path_descriptor(
                        record_entry,
                        f"{attempt_relative}/{record_entry.name}",
                    )
                )
    return canonical_sha256({"entries": descriptors})


def _selection_groups(entries: Sequence[SelectionEntry]) -> Dict[str, List[SelectionEntry]]:
    groups: Dict[str, List[SelectionEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.cell_id, []).append(entry)
    return groups


def _record_artifact_issues(
    frozen: FrozenRun,
    record: AttemptRecord,
) -> List[CompletenessIssue]:
    issues: List[CompletenessIssue] = []
    for reference in record.artifact_references():
        try:
            verify_artifact_reference(frozen.directory, reference)
        except ArtifactVerificationError as exc:
            issues.append(
                _issue(
                    exc.code,
                    str(exc),
                    cell_id=record.cell_id,
                    attempt_id=record.attempt_id,
                    artifact_path=reference.path,
                )
            )
        except (ContractError, OSError) as exc:
            issues.append(
                _issue(
                    "artifact-invalid",
                    f"cannot verify artifact {reference.path}: {exc}",
                    cell_id=record.cell_id,
                    attempt_id=record.attempt_id,
                    artifact_path=reference.path,
                )
            )
    return issues


def evaluate_completeness(
    run_directory: PathLike,
    snapshot: SelectionSnapshot,
    *,
    allow_incomplete: bool = False,
) -> CompletenessVerdict:
    """Evaluate a snapshot, raising unless every expected cell is trustworthy."""

    if not isinstance(allow_incomplete, bool):
        raise CompletenessValidationError("allow_incomplete must be boolean")
    snapshot.validate()
    manifest, frozen = load_frozen_run(run_directory)
    issues: List[CompletenessIssue] = []
    expected_cells = {cell.id: cell for cell in manifest.cells}
    groups = _selection_groups(snapshot.entries)

    if snapshot.run_id != manifest.run_id:
        issues.append(_issue("stale-selection", "selection run_id does not match the manifest"))
    if snapshot.manifest_sha256 != frozen.manifest_sha256:
        issues.append(
            _issue(
                "stale-selection",
                "selection manifest_sha256 does not match the frozen manifest",
            )
        )

    for cell_id, entries in sorted(groups.items()):
        if cell_id not in expected_cells:
            for entry in entries:
                issues.append(
                    _issue(
                        "unexpected-selection",
                        f"selection refers to unknown cell {cell_id!r}",
                        cell_id=cell_id,
                        attempt_id=entry.attempt_id,
                    )
                )
        if len(entries) > 1:
            issues.append(
                _issue(
                    "duplicate-selection",
                    f"cell {cell_id!r} has {len(entries)} selected attempts",
                    cell_id=cell_id,
                )
            )

    raw_attempts_root = frozen.directory / ATTEMPTS_DIRNAME
    if raw_attempts_root.is_symlink() or (raw_attempts_root.exists() and not raw_attempts_root.is_dir()):
        issues.append(
            _issue(
                "invalid-attempt-store",
                "attempts root is not a real directory",
            )
        )
    elif raw_attempts_root.is_dir():
        try:
            for path_entry in sorted(raw_attempts_root.iterdir(), key=lambda path: path.name):
                if path_entry.name not in expected_cells:
                    issues.append(
                        _issue(
                            "unexpected-attempt-cell",
                            f"attempt store contains unknown cell entry {path_entry.name!r}",
                            cell_id=path_entry.name,
                        )
                    )
        except OSError as exc:
            issues.append(
                _issue(
                    "invalid-attempt-store",
                    f"cannot enumerate attempt store: {exc}",
                )
            )

    attempted_cells = 0
    selected_cells = 0
    successful_selected_cells = 0
    for cell_id, cell in sorted(expected_cells.items()):
        try:
            attempts = load_cell_attempts(frozen.directory, cell_id)
        except (ContractError, OSError) as exc:
            issues.append(
                _issue(
                    "invalid-attempt-store",
                    f"cannot validate attempts for cell {cell_id!r}: {exc}",
                    cell_id=cell_id,
                )
            )
            attempts = ()
        if attempts:
            attempted_cells += 1
        else:
            issues.append(
                _issue(
                    "missing-attempt",
                    f"cell {cell_id!r} has no terminal attempts",
                    cell_id=cell_id,
                )
            )
        for attempt in attempts:
            issues.extend(_record_artifact_issues(frozen, attempt))

        entries = groups.get(cell_id, [])
        if not entries:
            issues.append(
                _issue(
                    "unselected-cell",
                    f"cell {cell_id!r} has no explicitly selected attempt",
                    cell_id=cell_id,
                )
            )
            continue
        if len(entries) != 1:
            continue
        selected_cells += 1
        entry = entries[0]
        selection_fresh = True
        if entry.cell_identity_sha256 != cell.identity_sha256:
            selection_fresh = False
            issues.append(
                _issue(
                    "stale-selection",
                    f"selection has a stale identity hash for cell {cell_id!r}",
                    cell_id=cell_id,
                    attempt_id=entry.attempt_id,
                )
            )
        matches = [attempt for attempt in attempts if attempt.attempt_id == entry.attempt_id]
        if len(matches) != 1:
            issues.append(
                _issue(
                    "selected-attempt-missing",
                    f"selected attempt {entry.attempt_id!r} does not exist exactly once",
                    cell_id=cell_id,
                    attempt_id=entry.attempt_id,
                )
            )
            continue
        selected = matches[0]
        if selected.sha256 != entry.attempt_record_sha256:
            selection_fresh = False
            issues.append(
                _issue(
                    "stale-selection",
                    f"selected attempt {entry.attempt_id!r} no longer matches its record hash",
                    cell_id=cell_id,
                    attempt_id=entry.attempt_id,
                )
            )
        if selected.status != "success":
            issues.append(
                _issue(
                    "selected-attempt-failed",
                    f"selected attempt {entry.attempt_id!r} has terminal status {selected.status!r}",
                    cell_id=cell_id,
                    attempt_id=entry.attempt_id,
                )
            )
        elif selection_fresh:
            successful_selected_cells += 1

    normalized_issues = tuple(sorted(set(issues), key=_issue_key))
    verdict = CompletenessVerdict(
        schema=COMPLETENESS_SCHEMA,
        run_id=manifest.run_id,
        manifest_sha256=frozen.manifest_sha256,
        selection_id=snapshot.selection_id,
        selection_sha256=snapshot.sha256,
        attempt_inventory_sha256=_attempt_inventory_sha256(frozen),
        allow_incomplete=allow_incomplete,
        complete=not normalized_issues,
        expected_cells=len(expected_cells),
        attempted_cells=attempted_cells,
        selected_cells=selected_cells,
        successful_selected_cells=successful_selected_cells,
        issues=normalized_issues,
    )
    verdict.validate()
    if not verdict.complete and not allow_incomplete:
        raise IncompleteCampaignError(verdict)
    return verdict


@dataclass(frozen=True)
class FrozenCompletenessVerdict:
    directory: Path
    verdict_path: Path
    checksum_path: Path
    verdict_sha256: str


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


def _verdict_root(selection: FrozenSelection, *, create: bool) -> Path:
    root = contained_path(selection.directory, VERDICTS_DIRNAME)
    if root.is_symlink():
        raise CompletenessStoreError("verdicts directory may not be a symlink")
    if create:
        root.mkdir(exist_ok=True)
        root = contained_path(selection.directory, VERDICTS_DIRNAME)
    return root


def freeze_completeness_verdict(
    run_directory: PathLike,
    verdict: CompletenessVerdict,
) -> FrozenCompletenessVerdict:
    """Persist a complete verdict or an explicitly allowed incomplete verdict."""

    verdict.validate()
    if not verdict.complete and not verdict.allow_incomplete:
        raise CompletenessValidationError("an incomplete verdict may be persisted only with allow_incomplete=true")
    manifest, frozen = load_frozen_run(run_directory)
    snapshot, stored_selection = load_selection_snapshot(
        frozen.directory,
        verdict.selection_id,
    )
    if verdict.run_id != manifest.run_id or verdict.manifest_sha256 != frozen.manifest_sha256:
        raise CompletenessValidationError("verdict does not match the frozen manifest")
    if verdict.selection_sha256 != stored_selection.selection_sha256:
        raise CompletenessValidationError("verdict does not match the frozen selection")
    if snapshot.sha256 != verdict.selection_sha256:
        raise CompletenessValidationError("stored selection has inconsistent canonical bytes")

    root = _verdict_root(stored_selection, create=True)
    destination = contained_path(root, verdict.sha256)
    lock_path = contained_path(root, f".{verdict.sha256}.lock")
    if destination.exists() or destination.is_symlink():
        loaded, stored = load_completeness_verdict(
            frozen.directory,
            verdict.selection_id,
            verdict.sha256,
        )
        if loaded != verdict:
            raise CompletenessStoreError("verdict hash collision with different content")
        return stored
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise CompletenessStoreError("another process is freezing this completeness verdict") from exc

    temporary: Optional[Path] = None
    try:
        if destination.exists() or destination.is_symlink():
            raise CompletenessStoreError(f"verdict directory already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{verdict.sha256}.tmp-", dir=str(root)))
        verdict_path = temporary / VERDICT_FILENAME
        checksum_path = temporary / VERDICT_SHA256_FILENAME
        verdict_bytes = verdict.to_json_bytes()
        _write_exclusive(verdict_path, verdict_bytes)
        _write_exclusive(
            checksum_path,
            f"{verdict.sha256}  {VERDICT_FILENAME}\n".encode("ascii"),
        )
        os.chmod(verdict_path, 0o444)
        os.chmod(checksum_path, 0o444)
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise CompletenessStoreError(f"verdict directory already exists: {destination}") from exc
            raise
        temporary = None
        _fsync_directory(root)
        return FrozenCompletenessVerdict(
            directory=destination,
            verdict_path=destination / VERDICT_FILENAME,
            checksum_path=destination / VERDICT_SHA256_FILENAME,
            verdict_sha256=verdict.sha256,
        )
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


def load_completeness_verdict(
    run_directory: PathLike,
    selection_id: str,
    verdict_sha256: str,
) -> Tuple[CompletenessVerdict, FrozenCompletenessVerdict]:
    """Load one immutable verdict by its full content hash."""

    expected_sha256 = _sha256(verdict_sha256, "verdict_sha256")
    manifest, frozen = load_frozen_run(run_directory)
    snapshot, stored_selection = load_selection_snapshot(frozen.directory, selection_id)
    root = _verdict_root(stored_selection, create=False)
    directory = contained_path(root, expected_sha256)
    if directory.is_symlink() or not directory.is_dir():
        raise CompletenessStoreError(f"completeness verdict does not exist: {expected_sha256}")
    verdict_path = directory / VERDICT_FILENAME
    checksum_path = directory / VERDICT_SHA256_FILENAME
    actual_entries = {path.name for path in directory.iterdir()}
    if actual_entries != {VERDICT_FILENAME, VERDICT_SHA256_FILENAME}:
        raise CompletenessStoreError("verdict directory has unexpected or missing entries")
    if verdict_path.is_symlink() or checksum_path.is_symlink():
        raise CompletenessStoreError("verdict record and checksum may not be symlinks")
    expected_checksum = f"{expected_sha256}  {VERDICT_FILENAME}\n"
    try:
        observed_checksum = read_bounded_text(
            checksum_path,
            max_bytes=CHECKSUM_MAX_BYTES,
            field="verdict checksum",
            encoding="ascii",
        )
    except CanonicalJSONError as exc:
        raise CompletenessStoreError(f"cannot read verdict checksum: {exc}") from exc
    if observed_checksum != expected_checksum:
        raise CompletenessStoreError("verdict checksum file has invalid content")
    try:
        raw = read_bounded_bytes(
            verdict_path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field="completeness verdict",
        )
    except CanonicalJSONError as exc:
        raise CompletenessStoreError(str(exc)) from exc
    if sha256_hex(raw) != expected_sha256:
        raise CompletenessStoreError("verdict SHA-256 does not match its content address")
    try:
        verdict = CompletenessVerdict.from_json_bytes(raw)
    except CanonicalJSONError as exc:
        raise CompletenessStoreError(f"invalid completeness verdict JSON: {exc}") from exc
    if raw != verdict.to_json_bytes():
        raise CompletenessStoreError("verdict is valid JSON but not in canonical byte form")
    if verdict.run_id != manifest.run_id or verdict.manifest_sha256 != frozen.manifest_sha256:
        raise CompletenessStoreError("verdict is stale for the frozen manifest")
    if verdict.selection_id != snapshot.selection_id:
        raise CompletenessStoreError("verdict selection_id does not match its directory")
    if verdict.selection_sha256 != stored_selection.selection_sha256:
        raise CompletenessStoreError("verdict is stale for its frozen selection")
    return verdict, FrozenCompletenessVerdict(
        directory=directory,
        verdict_path=verdict_path,
        checksum_path=checksum_path,
        verdict_sha256=expected_sha256,
    )


def evaluate_and_persist_completeness(
    run_directory: PathLike,
    selection_id: str,
    *,
    allow_incomplete: bool = False,
) -> Tuple[CompletenessVerdict, FrozenCompletenessVerdict]:
    """Load an explicit selection, evaluate it, and persist its verdict."""

    snapshot, _stored = load_selection_snapshot(run_directory, selection_id)
    verdict = evaluate_completeness(
        run_directory,
        snapshot,
        allow_incomplete=allow_incomplete,
    )
    return verdict, freeze_completeness_verdict(run_directory, verdict)
