"""Generate deterministic release checksums, inventory, and SPDX metadata.

The generator deliberately operates on the already-built wheel and sdist.  It
does not rebuild, extract, or modify either distribution.  All metadata is
rendered twice before an output directory is atomically installed so input
drift and accidental nondeterminism fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import IO, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

INVENTORY_NAME = "release-inventory.json"
CHECKSUMS_NAME = "SHA256SUMS"
SPDX_SUFFIX = ".spdx.json"
INVENTORY_SCHEMA = "commcanary.release-inventory.v1"
GENERATOR_NAME = "commcanary-release-metadata/1"
_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")
_CHUNK_SIZE = 1024 * 1024
_OPTIONAL_EXTRA_MARKER_RE = re.compile(
    r"^\(?\s*extra\s*==\s*(['\"])[A-Za-z0-9_.-]+\1\s*\)?$",
    flags=re.IGNORECASE,
)


class ReleaseMetadataError(RuntimeError):
    """Release distributions or metadata output violated a safety contract."""


@dataclass(frozen=True)
class DistributionSet:
    """The exact wheel and sdist for which metadata will be generated."""

    wheel: Path
    sdist: Path


@dataclass(frozen=True)
class ReleaseMetadataFiles:
    """Paths installed by :func:`write_release_metadata`."""

    checksums: Path
    inventory: Path
    sbom: Path


@dataclass(frozen=True)
class _ArchiveMember:
    path: str
    kind: str
    size: int
    sha256: Optional[str]

    def as_json(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "type": self.kind,
        }


@dataclass(frozen=True)
class _InspectedArtifact:
    filename: str
    format: str
    sha1: str
    sha256: str
    size: int
    members: Tuple[_ArchiveMember, ...]
    metadata: Message


def _canonical_project_name(project: str) -> str:
    canonical = re.sub(r"[-_.]+", "-", project).lower()
    if not canonical or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", canonical):
        raise ReleaseMetadataError(f"unsafe project name {project!r}")
    return canonical


def _validate_version(version: str) -> None:
    if not version or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
        raise ReleaseMetadataError(f"unsafe version {version!r}")


def _source_epoch(value: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ReleaseMetadataError("source date epoch must be a canonical non-negative integer")
    epoch = int(value)
    try:
        datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ReleaseMetadataError(f"invalid source date epoch {value!r}") from exc
    return epoch


def _digest_path(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReleaseMetadataError(f"cannot hash distribution {path}: {exc}") from exc
    return digest.hexdigest()


def _digest_stream(handle: IO[bytes]) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
        size += len(chunk)
        digest.update(chunk)
    return digest.hexdigest(), size


def _distribution_candidates(directory: Path) -> List[Path]:
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise ReleaseMetadataError(f"cannot inspect distribution directory {directory}: {exc}") from exc
    return sorted(
        (path for path in entries if path.name.endswith(_DISTRIBUTION_SUFFIXES)),
        key=lambda path: path.name,
    )


def _validate_distribution_set(
    artifacts: DistributionSet,
    *,
    project: str,
    version: str,
) -> Tuple[Path, Path]:
    canonical_project = _canonical_project_name(project)
    _validate_version(version)
    wheel = artifacts.wheel.absolute()
    sdist = artifacts.sdist.absolute()
    for label, path in (("wheel", wheel), ("sdist", sdist)):
        if path.is_symlink():
            raise ReleaseMetadataError(f"{label} must not be a symlink: {path}")
        if not path.is_file():
            raise ReleaseMetadataError(f"missing {label} distribution: {path}")
    if wheel.parent != sdist.parent:
        raise ReleaseMetadataError("wheel and sdist must be in the same distribution directory")
    if wheel == sdist or os.path.samefile(wheel, sdist):
        raise ReleaseMetadataError("wheel and sdist paths collide")

    expected_wheel = f"{canonical_project.replace('-', '_')}-{version}-py3-none-any.whl"
    expected_sdist = f"{canonical_project}-{version}.tar.gz"
    if wheel.name != expected_wheel:
        raise ReleaseMetadataError(f"stale or unexpected wheel filename {wheel.name!r}; expected {expected_wheel!r}")
    if sdist.name != expected_sdist:
        raise ReleaseMetadataError(f"stale or unexpected sdist filename {sdist.name!r}; expected {expected_sdist!r}")

    candidates = _distribution_candidates(wheel.parent)
    names: Dict[str, str] = {}
    for candidate in candidates:
        key = unicodedata.normalize("NFC", candidate.name).casefold()
        previous = names.get(key)
        if previous is not None:
            raise ReleaseMetadataError(f"colliding distribution filenames: {previous!r} and {candidate.name!r}")
        names[key] = candidate.name
    expected_paths = {wheel, sdist}
    actual_paths = {path.absolute() for path in candidates}
    if actual_paths != expected_paths:
        missing = sorted(path.name for path in expected_paths - actual_paths)
        extra = sorted(path.name for path in actual_paths - expected_paths)
        raise ReleaseMetadataError(f"distribution set mismatch: missing={missing!r}, extra={extra!r}")
    return wheel, sdist


def _safe_member_path(raw_name: str) -> Tuple[str, str]:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name or raw_name.startswith("/"):
        raise ReleaseMetadataError(f"unsafe archive member path {raw_name!r}")
    if any(ord(character) < 32 for character in raw_name):
        raise ReleaseMetadataError(f"unsafe archive member path {raw_name!r}")
    name = raw_name[:-1] if raw_name.endswith("/") else raw_name
    parts = name.split("/")
    if not name or any(part in {"", ".", ".."} for part in parts):
        raise ReleaseMetadataError(f"unsafe archive member path {raw_name!r}")
    if re.match(r"^[A-Za-z]:", parts[0]):
        raise ReleaseMetadataError(f"unsafe archive member path {raw_name!r}")
    key = unicodedata.normalize("NFC", name).casefold()
    return name, key


def _validate_member_collisions(entries: Iterable[Tuple[str, str]]) -> None:
    paths: Dict[str, Tuple[str, str]] = {}
    for raw_name, kind in entries:
        _, key = _safe_member_path(raw_name)
        previous = paths.get(key)
        if previous is not None:
            raise ReleaseMetadataError(f"duplicate or colliding archive members: {previous[0]!r} and {raw_name!r}")
        parts = key.split("/")
        for index in range(1, len(parts)):
            parent_key = "/".join(parts[:index])
            parent = paths.get(parent_key)
            if parent is not None and parent[1] == "file":
                raise ReleaseMetadataError(f"archive file/directory collision: {parent[0]!r} contains {raw_name!r}")
        if kind == "file":
            prefix = key + "/"
            descendant = next((value for path_key, value in paths.items() if path_key.startswith(prefix)), None)
            if descendant is not None:
                raise ReleaseMetadataError(f"archive file/directory collision: {raw_name!r} contains {descendant[0]!r}")
        paths[key] = (raw_name, kind)


def _zip_kind(info: zipfile.ZipInfo) -> str:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if info.is_dir():
        if info.file_size != 0 or file_type not in (0, stat.S_IFDIR):
            raise ReleaseMetadataError(f"invalid wheel directory member {info.filename!r}")
        return "directory"
    if file_type not in (0, stat.S_IFREG):
        raise ReleaseMetadataError(f"wheel contains a symlink or special member: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise ReleaseMetadataError(f"wheel contains an encrypted member: {info.filename!r}")
    return "file"


def _inspect_wheel(path: Path, *, epoch: int) -> _InspectedArtifact:
    expected_timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc).timetuple()[:6]
    members: List[_ArchiveMember] = []
    metadata_payloads: List[bytes] = []
    collision_entries: List[Tuple[str, str]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                kind = _zip_kind(info)
                _safe_member_path(info.filename)
                collision_entries.append((info.filename, "file" if kind == "file" else "directory"))
                if info.date_time != expected_timestamp:
                    raise ReleaseMetadataError(f"wheel member {info.filename!r} has stale timestamp {info.date_time!r}")
                if kind == "directory":
                    members.append(_ArchiveMember(path=info.filename, kind=kind, size=0, sha256=None))
                    continue
                with archive.open(info, "r") as handle:
                    digest, actual_size = _digest_stream(handle)
                if actual_size != info.file_size:
                    raise ReleaseMetadataError(f"wheel member {info.filename!r} changed size while reading")
                members.append(_ArchiveMember(path=info.filename, kind=kind, size=actual_size, sha256=digest))
                if info.filename.endswith(".dist-info/METADATA"):
                    metadata_payloads.append(archive.read(info))
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ReleaseMetadataError):
            raise
        raise ReleaseMetadataError(f"cannot inspect wheel {path}: {exc}") from exc
    _validate_member_collisions(collision_entries)
    if len(metadata_payloads) != 1:
        raise ReleaseMetadataError(f"wheel has {len(metadata_payloads)} METADATA members")
    return _InspectedArtifact(
        filename=path.name,
        format="wheel",
        sha1=_digest_path(path, "sha1"),
        sha256=_digest_path(path, "sha256"),
        size=path.stat().st_size,
        members=tuple(sorted(members, key=lambda member: member.path)),
        metadata=BytesParser(policy=email_policy).parsebytes(metadata_payloads[0]),
    )


def _inspect_sdist(path: Path, *, epoch: int) -> _InspectedArtifact:
    members: List[_ArchiveMember] = []
    metadata_payloads: List[bytes] = []
    collision_entries: List[Tuple[str, str]] = []
    try:
        with tarfile.open(path, "r:gz") as archive:
            for info in archive.getmembers():
                if info.isfile():
                    kind = "file"
                elif info.isdir():
                    kind = "directory"
                else:
                    raise ReleaseMetadataError(f"sdist contains a symlink or special member: {info.name!r}")
                _safe_member_path(info.name)
                collision_entries.append((info.name, kind))
                if info.mtime != epoch:
                    raise ReleaseMetadataError(f"sdist member {info.name!r} has stale timestamp {info.mtime!r}")
                if kind == "directory":
                    if info.size != 0:
                        raise ReleaseMetadataError(f"invalid sdist directory member {info.name!r}")
                    members.append(_ArchiveMember(path=info.name, kind=kind, size=0, sha256=None))
                    continue
                extracted = archive.extractfile(info)
                if extracted is None:
                    raise ReleaseMetadataError(f"cannot read sdist member {info.name!r}")
                with extracted:
                    digest, actual_size = _digest_stream(extracted)
                if actual_size != info.size:
                    raise ReleaseMetadataError(f"sdist member {info.name!r} changed size while reading")
                members.append(_ArchiveMember(path=info.name, kind=kind, size=actual_size, sha256=digest))
                if info.name.count("/") == 1 and info.name.endswith("/PKG-INFO"):
                    second = archive.extractfile(info)
                    if second is None:
                        raise ReleaseMetadataError(f"cannot reread sdist metadata {info.name!r}")
                    with second:
                        metadata_payloads.append(second.read())
    except (OSError, tarfile.TarError) as exc:
        if isinstance(exc, ReleaseMetadataError):
            raise
        raise ReleaseMetadataError(f"cannot inspect sdist {path}: {exc}") from exc
    _validate_member_collisions(collision_entries)
    if len(metadata_payloads) != 1:
        raise ReleaseMetadataError(f"sdist has {len(metadata_payloads)} top-level PKG-INFO members")
    return _InspectedArtifact(
        filename=path.name,
        format="sdist",
        sha1=_digest_path(path, "sha1"),
        sha256=_digest_path(path, "sha256"),
        size=path.stat().st_size,
        members=tuple(sorted(members, key=lambda member: member.path)),
        metadata=BytesParser(policy=email_policy).parsebytes(metadata_payloads[0]),
    )


def _metadata_requirements(metadata: Message) -> Tuple[List[str], List[str]]:
    runtime: List[str] = []
    optional: List[str] = []
    for requirement in metadata.get_all("Requires-Dist", []):
        if ";" in requirement and _OPTIONAL_EXTRA_MARKER_RE.fullmatch(requirement.split(";", 1)[1].strip()):
            optional.append(requirement)
        else:
            runtime.append(requirement)
    return sorted(runtime), sorted(optional)


def _validate_package_metadata(
    wheel: _InspectedArtifact,
    sdist: _InspectedArtifact,
    *,
    project: str,
    version: str,
    requires_python: str,
    license_id: str,
) -> List[str]:
    canonical_project = _canonical_project_name(project)
    metadata_rows: List[Tuple[str, Message]] = [("wheel", wheel.metadata), ("sdist", sdist.metadata)]
    optional_sets: List[List[str]] = []
    for label, metadata in metadata_rows:
        actual_name = metadata.get("Name")
        if actual_name is None or _canonical_project_name(actual_name) != canonical_project:
            raise ReleaseMetadataError(f"{label} metadata project name {actual_name!r} is stale")
        if metadata.get("Version") != version:
            raise ReleaseMetadataError(f"{label} metadata version {metadata.get('Version')!r} is stale")
        if metadata.get("Requires-Python") != requires_python:
            raise ReleaseMetadataError(
                f"{label} Requires-Python {metadata.get('Requires-Python')!r} does not match {requires_python!r}"
            )
        if metadata.get("License") != license_id:
            raise ReleaseMetadataError(
                f"{label} license metadata {metadata.get('License')!r} does not match {license_id!r}"
            )
        runtime, optional = _metadata_requirements(metadata)
        if runtime:
            raise ReleaseMetadataError(f"{label} declares runtime dependencies: {runtime!r}")
        optional_sets.append(optional)
    if optional_sets[0] != optional_sets[1]:
        raise ReleaseMetadataError("wheel and sdist dependency metadata disagree")
    return optional_sets[0]


def _validate_archive_layout(
    wheel: _InspectedArtifact,
    sdist: _InspectedArtifact,
    *,
    project: str,
    version: str,
) -> None:
    canonical_project = _canonical_project_name(project)
    wheel_component = canonical_project.replace("-", "_")
    expected_metadata = f"{wheel_component}-{version}.dist-info/METADATA"
    wheel_paths = {member.path.rstrip("/") for member in wheel.members}
    if expected_metadata not in wheel_paths:
        raise ReleaseMetadataError(f"wheel metadata path is stale; expected {expected_metadata!r}")

    expected_root = f"{canonical_project}-{version}"
    sdist_paths = [member.path.rstrip("/") for member in sdist.members]
    misplaced = sorted(
        path for path in sdist_paths if path != expected_root and not path.startswith(expected_root + "/")
    )
    if misplaced:
        raise ReleaseMetadataError(f"sdist contains members outside {expected_root!r}: {misplaced!r}")
    if f"{expected_root}/PKG-INFO" not in sdist_paths:
        raise ReleaseMetadataError(f"sdist metadata path is stale; expected {expected_root + '/PKG-INFO'!r}")


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _artifact_inventory(artifact: _InspectedArtifact) -> Dict[str, object]:
    return {
        "filename": artifact.filename,
        "format": artifact.format,
        "members": [member.as_json() for member in artifact.members],
        "sha256": artifact.sha256,
        "size": artifact.size,
    }


def _package_verification_code(artifacts: Iterable[_InspectedArtifact]) -> str:
    concatenated = "".join(sorted(artifact.sha1 for artifact in artifacts)).encode("ascii")
    return hashlib.sha1(concatenated).hexdigest()  # noqa: S324 - SPDX 2.3 mandates SHA-1 here.


def _spdx_id_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-.")
    if not result:
        raise ReleaseMetadataError(f"cannot construct SPDX identifier from {value!r}")
    return result


def _render_release_metadata(
    artifacts: DistributionSet,
    *,
    project: str,
    version: str,
    source_date_epoch: str,
    requires_python: str,
    license_id: str,
    repository_url: str,
) -> Dict[str, bytes]:
    wheel_path, sdist_path = _validate_distribution_set(artifacts, project=project, version=version)
    epoch = _source_epoch(source_date_epoch)
    wheel = _inspect_wheel(wheel_path, epoch=epoch)
    sdist = _inspect_sdist(sdist_path, epoch=epoch)
    inspected = tuple(sorted((wheel, sdist), key=lambda artifact: artifact.filename))
    _validate_archive_layout(wheel, sdist, project=project, version=version)
    optional_dependencies = _validate_package_metadata(
        wheel,
        sdist,
        project=project,
        version=version,
        requires_python=requires_python,
        license_id=license_id,
    )

    inventory: Dict[str, object] = {
        "artifacts": [_artifact_inventory(artifact) for artifact in inspected],
        "package_metadata": {
            "optional_requirements": optional_dependencies,
            "requires_python": requires_python,
            "runtime_requirements": [],
        },
        "project": _canonical_project_name(project),
        "schema": INVENTORY_SCHEMA,
        "source_date_epoch": epoch,
        "version": version,
    }
    identity = "\n".join(f"{artifact.filename}:{artifact.sha256}" for artifact in inspected)
    namespace_uuid = uuid.uuid5(uuid.NAMESPACE_URL, identity)
    package_id = f"SPDXRef-Package-{_spdx_id_component(project)}"
    file_ids = {artifact.filename: f"SPDXRef-File-{_spdx_id_component(artifact.format)}" for artifact in inspected}
    created = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sbom: Dict[str, object] = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {"created": created, "creators": [f"Tool: {GENERATOR_NAME}"]},
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://spdx.org/spdxdocs/{project}-{version}-{namespace_uuid}",
        "files": [
            {
                "SPDXID": file_ids[artifact.filename],
                "checksums": [{"algorithm": "SHA256", "checksumValue": artifact.sha256}],
                "copyrightText": "NOASSERTION",
                "fileName": f"./{artifact.filename}",
                "licenseConcluded": "NOASSERTION",
            }
            for artifact in inspected
        ],
        "name": f"{project}-{version}-release",
        "packages": [
            {
                "SPDXID": package_id,
                "comment": "The project declares no required runtime dependencies.",
                "copyrightText": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceLocator": f"pkg:pypi/{_canonical_project_name(project)}@{version}",
                        "referenceType": "purl",
                    }
                ],
                "filesAnalyzed": True,
                "homepage": repository_url,
                "licenseConcluded": license_id,
                "licenseDeclared": license_id,
                "name": project,
                "packageVerificationCode": {"packageVerificationCodeValue": _package_verification_code(inspected)},
                "primaryPackagePurpose": "LIBRARY",
                "supplier": "NOASSERTION",
                "versionInfo": version,
            }
        ],
        "relationships": [
            {
                "relatedSpdxElement": package_id,
                "relationshipType": "DESCRIBES",
                "spdxElementId": "SPDXRef-DOCUMENT",
            },
            *(
                {
                    "relatedSpdxElement": file_ids[artifact.filename],
                    "relationshipType": "CONTAINS",
                    "spdxElementId": package_id,
                }
                for artifact in inspected
            ),
        ],
        "spdxVersion": "SPDX-2.3",
    }
    checksums = "".join(f"{artifact.sha256}  {artifact.filename}\n" for artifact in inspected).encode("ascii")
    return {
        CHECKSUMS_NAME: checksums,
        INVENTORY_NAME: _canonical_json(inventory),
        f"{_canonical_project_name(project)}-{version}{SPDX_SUFFIX}": _canonical_json(sbom),
    }


def _validate_output_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise ReleaseMetadataError(f"metadata output destination must not be a symlink: {destination}")
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ReleaseMetadataError(f"metadata output destination is not a directory: {destination}")
    try:
        first = next(destination.iterdir(), None)
    except OSError as exc:
        raise ReleaseMetadataError(f"cannot inspect metadata output destination {destination}: {exc}") from exc
    if first is not None:
        raise ReleaseMetadataError(f"metadata output destination is not empty: {destination}")


def _write_staging_directory(directory: Path, rendered: Mapping[str, bytes]) -> None:
    for name, payload in sorted(rendered.items()):
        if Path(name).name != name:
            raise ReleaseMetadataError(f"unsafe metadata filename {name!r}")
        target = directory / name
        try:
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ReleaseMetadataError(f"cannot stage release metadata {target}: {exc}") from exc
        if target.read_bytes() != payload:
            raise ReleaseMetadataError(f"staged release metadata changed unexpectedly: {target}")


def _validate_expected_sha256(rendered: Mapping[str, bytes], expected: Mapping[str, str]) -> None:
    normalized: Dict[str, str] = {}
    for filename, digest in expected.items():
        if Path(filename).name != filename or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ReleaseMetadataError(f"invalid expected SHA-256 entry for {filename!r}")
        normalized[filename] = digest
    expected_payload = "".join(f"{digest}  {filename}\n" for filename, digest in sorted(normalized.items())).encode(
        "ascii"
    )
    if rendered[CHECKSUMS_NAME] != expected_payload:
        raise ReleaseMetadataError("distribution SHA-256 values changed after artifact testing")


def _install_staging_directory(staging: Path, destination: Path) -> None:
    lock = destination.parent / f".{destination.name}.commcanary-release-metadata.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ReleaseMetadataError(f"metadata output is already being created: {destination}") from exc
    except OSError as exc:
        raise ReleaseMetadataError(f"cannot reserve metadata output {destination}: {exc}") from exc
    try:
        os.close(descriptor)
        _validate_output_destination(destination)
        if destination.exists():
            destination.rmdir()
        os.rename(staging, destination)
    except OSError as exc:
        raise ReleaseMetadataError(f"cannot atomically install metadata output {destination}: {exc}") from exc
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def write_release_metadata(
    artifacts: DistributionSet,
    destination: Path,
    *,
    project: str,
    version: str,
    source_date_epoch: str,
    requires_python: str = ">=3.9",
    license_id: str = "MIT",
    repository_url: str = "https://github.com/iemAnshuman/commcanary",
    expected_sha256: Optional[Mapping[str, str]] = None,
) -> ReleaseMetadataFiles:
    """Validate distributions and atomically install deterministic metadata."""

    destination = destination.absolute()
    if not destination.parent.is_dir():
        raise ReleaseMetadataError(f"metadata output parent does not exist: {destination.parent}")
    _validate_output_destination(destination)
    first = _render_release_metadata(
        artifacts,
        project=project,
        version=version,
        source_date_epoch=source_date_epoch,
        requires_python=requires_python,
        license_id=license_id,
        repository_url=repository_url,
    )
    second = _render_release_metadata(
        artifacts,
        project=project,
        version=version,
        source_date_epoch=source_date_epoch,
        requires_python=requires_python,
        license_id=license_id,
        repository_url=repository_url,
    )
    if first != second:
        raise ReleaseMetadataError("release metadata or distribution inputs are nondeterministic")
    if expected_sha256 is not None:
        _validate_expected_sha256(first, expected_sha256)

    raw_staging = tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent))
    staging = Path(raw_staging)
    try:
        _write_staging_directory(staging, first)
        _install_staging_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    sbom_name = f"{_canonical_project_name(project)}-{version}{SPDX_SUFFIX}"
    return ReleaseMetadataFiles(
        checksums=destination / CHECKSUMS_NAME,
        inventory=destination / INVENTORY_NAME,
        sbom=destination / sbom_name,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--project", default="commcanary")
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-date-epoch", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    candidates = _distribution_candidates(args.distribution_dir)
    wheels = [path for path in candidates if path.name.endswith(".whl")]
    sdists = [path for path in candidates if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseMetadataError(
            f"expected one wheel and one sdist, found wheels={len(wheels)}, sdists={len(sdists)}"
        )
    files = write_release_metadata(
        DistributionSet(wheel=wheels[0], sdist=sdists[0]),
        args.output_dir,
        project=args.project,
        version=args.version,
        source_date_epoch=args.source_date_epoch,
    )
    print(files.checksums)
    print(files.inventory)
    print(files.sbom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
