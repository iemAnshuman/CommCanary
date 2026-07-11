"""Build, freeze, and load immutable run manifests."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .canonical import (
    CHECKSUM_MAX_BYTES,
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    contained_path,
    read_bounded_bytes,
    read_bounded_text,
    safe_slug,
    sha256_hex,
)
from .model import (
    MANIFEST_SCHEMA,
    CampaignSpec,
    CellSpec,
    ManifestValidationError,
    RunManifest,
    derive_cell_identity,
)

PathLike = Union[str, "Path"]
MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SHA256_FILENAME = "run_manifest.sha256"
_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  run_manifest\.json\n$")


class ManifestFreezeError(ManifestValidationError):
    """Raised when an immutable run directory cannot be created safely."""


@dataclass(frozen=True)
class FrozenRun:
    directory: Path
    manifest_path: Path
    checksum_path: Path
    manifest_sha256: str


def build_run_manifest(campaign: CampaignSpec) -> RunManifest:
    """Deterministically expand a campaign's declarative Cartesian matrix."""

    campaign.validate()
    campaign_sha256 = campaign.sha256
    workload_by_id = {workload.id: workload for workload in campaign.workloads}
    cells: List[CellSpec] = []
    for configuration in campaign.configurations:
        for workload in campaign.workloads:
            for repetition in range(campaign.repetitions):
                cell_id, identity_sha256 = derive_cell_identity(
                    run_id=campaign.run_id,
                    campaign_sha256=campaign_sha256,
                    configuration_id=configuration.id,
                    workload_id=workload.id,
                    repetition=repetition,
                )
                dependencies = []
                for dependency_id in workload_by_id[workload.id].depends_on:
                    dependency_cell_id, _ = derive_cell_identity(
                        run_id=campaign.run_id,
                        campaign_sha256=campaign_sha256,
                        configuration_id=configuration.id,
                        workload_id=dependency_id,
                        repetition=repetition,
                    )
                    dependencies.append(dependency_cell_id)
                cells.append(
                    CellSpec(
                        id=cell_id,
                        identity_sha256=identity_sha256,
                        configuration_id=configuration.id,
                        workload_id=workload.id,
                        repetition=repetition,
                        dependencies=tuple(sorted(dependencies)),
                    )
                )
    manifest = RunManifest(
        schema=MANIFEST_SCHEMA,
        campaign_sha256=campaign_sha256,
        campaign=campaign,
        cells=tuple(sorted(cells, key=lambda cell: cell.id)),
    )
    manifest.validate()
    return manifest


def load_run_manifest(path: PathLike, *, expected_sha256: Optional[str] = None) -> RunManifest:
    """Load and validate a run manifest, optionally checking its exact bytes."""

    manifest_path = Path(path)
    try:
        raw = read_bounded_bytes(
            manifest_path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field="run manifest",
        )
    except CanonicalJSONError as exc:
        raise ManifestValidationError(str(exc)) from exc
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ManifestValidationError("expected manifest SHA-256 is malformed")
        actual = sha256_hex(raw)
        if actual != expected_sha256:
            raise ManifestValidationError(f"manifest SHA-256 mismatch: expected {expected_sha256}, observed {actual}")
    try:
        manifest = RunManifest.from_json_bytes(raw)
    except CanonicalJSONError as exc:
        raise ManifestValidationError(f"invalid run manifest JSON: {exc}") from exc
    if raw != manifest.to_json_bytes():
        raise ManifestValidationError("run manifest is valid JSON but not in canonical byte form")
    return manifest


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


def freeze_run_manifest(manifest: RunManifest, results_root: PathLike) -> FrozenRun:
    """Atomically freeze *manifest* below ``results/<run_id>``.

    Existing run directories are never reused or replaced. The manifest and
    checksum are read-only; later lifecycle code may add sibling cell data.
    """

    manifest.validate()
    safe_slug(manifest.run_id, field="manifest.run_id")
    root = Path(results_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    destination = contained_path(root, manifest.run_id)
    lock_path = contained_path(root, f".freeze-{manifest.run_id}.lock")
    if destination.exists() or destination.is_symlink():
        raise ManifestFreezeError(f"run directory already exists: {destination}")
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise ManifestFreezeError(f"another process is freezing run {manifest.run_id!r}") from exc

    temporary: Optional[Path] = None
    try:
        if destination.exists() or destination.is_symlink():
            raise ManifestFreezeError(f"run directory already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{manifest.run_id}.tmp-", dir=str(root)))
        manifest_path = temporary / MANIFEST_FILENAME
        checksum_path = temporary / MANIFEST_SHA256_FILENAME
        manifest_bytes = manifest.to_json_bytes()
        manifest_sha256 = manifest.sha256
        _write_exclusive(manifest_path, manifest_bytes)
        _write_exclusive(
            checksum_path,
            f"{manifest_sha256}  {MANIFEST_FILENAME}\n".encode("ascii"),
        )
        os.chmod(manifest_path, 0o444)
        os.chmod(checksum_path, 0o444)
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise ManifestFreezeError(f"run directory already exists: {destination}") from exc
            raise
        temporary = None
        _fsync_directory(root)
        return FrozenRun(
            directory=destination,
            manifest_path=destination / MANIFEST_FILENAME,
            checksum_path=destination / MANIFEST_SHA256_FILENAME,
            manifest_sha256=manifest_sha256,
        )
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


def freeze_campaign(campaign: CampaignSpec, results_root: PathLike) -> FrozenRun:
    """Build and freeze a campaign manifest in one operation."""

    return freeze_run_manifest(build_run_manifest(campaign), results_root)


def load_frozen_run(run_directory: PathLike) -> Tuple[RunManifest, FrozenRun]:
    """Load a frozen run and verify its detached checksum and directory name."""

    directory = Path(run_directory).expanduser().resolve()
    manifest_path = directory / MANIFEST_FILENAME
    checksum_path = directory / MANIFEST_SHA256_FILENAME
    try:
        checksum_text = read_bounded_text(
            checksum_path,
            max_bytes=CHECKSUM_MAX_BYTES,
            field="manifest checksum",
            encoding="ascii",
        )
    except CanonicalJSONError as exc:
        raise ManifestValidationError(f"cannot read manifest checksum: {exc}") from exc
    match = _CHECKSUM_RE.fullmatch(checksum_text)
    if match is None:
        raise ManifestValidationError("manifest checksum file has invalid syntax")
    expected = match.group(1)
    manifest = load_run_manifest(manifest_path, expected_sha256=expected)
    if directory.name != manifest.run_id:
        raise ManifestValidationError(
            f"run directory name {directory.name!r} does not match manifest run_id {manifest.run_id!r}"
        )
    return manifest, FrozenRun(
        directory=directory,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        manifest_sha256=expected,
    )
