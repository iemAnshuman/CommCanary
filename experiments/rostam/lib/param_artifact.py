"""Build and privately stage the complete reviewed PARAM implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence, Tuple

PARAM_RUNTIME_ARTIFACT_INPUT_ID = "param-runtime-artifact"
PARAM_RUNTIME_ARTIFACT_SCHEMA = "commcanary.rostam.param-runtime-artifact.v1"
PARAM_RUNTIME_INVENTORY_NAME = "param-runtime.json"
# Production paths stream in 1 MiB chunks and keep only the canonical
# inventory resident. These ceilings bind the hostile-input working set to a
# documented 64 MiB budget rather than permitting simultaneous multi-GiB
# archive/member copies.
_WORKING_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_INVENTORY_BYTES = 16 * 1024 * 1024
_MAX_FILES = 50_000
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ParamArtifactError(RuntimeError):
    """Raised when a PARAM tree or artifact is incomplete or unsafe."""


@dataclass(frozen=True)
class ParamArtifact:
    path: Path
    sha256: str
    size_bytes: int
    inventory_sha256: str
    file_count: int


@dataclass(frozen=True)
class StagedParamRuntime:
    root: Path
    import_paths: Tuple[Path, ...]


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ParamArtifactError(f"PARAM artifact path is unsafe: {value!r}")
    return path


def _read_regular_bytes(path: Path, *, maximum: int, field: str, allow_empty: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ParamArtifactError(f"cannot open {field}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum or (before.st_size == 0 and not allow_empty):
            raise ParamArtifactError(f"{field} must be a bounded real regular file")
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise ParamArtifactError(f"{field} exceeds its byte limit")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or size != before.st_size:
            raise ParamArtifactError(f"{field} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _zip_entry_bytes(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    maximum: int,
    field: str,
) -> bytes:
    if (
        info.flag_bits & 0x1
        or info.compress_type != zipfile.ZIP_STORED
        or info.file_size > maximum
        or info.compress_size != info.file_size
    ):
        raise ParamArtifactError(f"{field} has an unsupported ZIP encoding or size")
    with archive.open(info, "r") as handle:
        raw = handle.read(maximum + 1)
        if len(raw) > maximum or handle.read(1):
            raise ParamArtifactError(f"{field} exceeds its byte limit")
    if len(raw) != info.file_size:
        raise ParamArtifactError(f"{field} size disagrees with its ZIP header")
    return raw


def _source_files(param_directory: Path) -> Tuple[Path, ...]:
    if param_directory.is_symlink():
        raise ParamArtifactError("PARAM source root must be a real directory")
    root = param_directory.resolve()
    if not root.is_dir():
        raise ParamArtifactError("PARAM source root must be a real directory")
    files = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        names[:] = sorted(name for name in names if name != ".git")
        for name in names:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ParamArtifactError(f"PARAM source directory is unsafe: {child}")
        for name in sorted(filenames):
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise ParamArtifactError(f"PARAM source file is unsafe: {child}")
            files.append(child)
            if len(files) > _MAX_FILES:
                raise ParamArtifactError(f"PARAM source exceeds the {_MAX_FILES}-file limit")
    replay = root / "train" / "comms" / "pt" / "commsTraceReplay.py"
    if replay not in files:
        raise ParamArtifactError("PARAM source lacks train/comms/pt/commsTraceReplay.py")
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _zip_info(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | (0o555 if executable else 0o444)) << 16
    return info


def _remove_staging_tree(path: Path) -> None:
    if not path.exists():
        return
    for directory, names, filenames in os.walk(path):
        os.chmod(directory, 0o700)
        for name in names:
            child = Path(directory) / name
            if not child.is_symlink():
                os.chmod(child, 0o700)
        for name in filenames:
            os.chmod(Path(directory) / name, 0o600)
    shutil.rmtree(path, ignore_errors=True)


def _stream_source_member(
    archive: zipfile.ZipFile,
    path: Path,
    *,
    relative: str,
    remaining_expanded: int,
) -> Dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        maximum = min(_MAX_MEMBER_BYTES, remaining_expanded)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ParamArtifactError(f"PARAM source file {relative!r} exceeds its deterministic limit")
        digest = hashlib.sha256()
        observed = 0
        info = _zip_info(relative, executable=bool(before.st_mode & 0o111))
        info.file_size = before.st_size
        with archive.open(info, "w", force_zip64=True) as output:
            while True:
                chunk = os.read(descriptor, _STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > before.st_size:
                    raise ParamArtifactError(f"PARAM source file {relative!r} grew while it was archived")
                digest.update(chunk)
                output.write(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or observed != before.st_size:
            raise ParamArtifactError(f"PARAM source file {relative!r} changed while it was archived")
        return {
            "path": relative,
            "sha256": digest.hexdigest(),
            "size_bytes": observed,
            "executable": bool(before.st_mode & 0o111),
        }
    finally:
        os.close(descriptor)


def _render_param_to_path(param_directory: Path, output_path: Path) -> Dict[str, Any]:
    root = param_directory.resolve()
    rows = []
    expanded = 0
    with output_path.open("xb") as output:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in _source_files(root):
                relative = path.relative_to(root).as_posix()
                row = _stream_source_member(
                    archive,
                    path,
                    relative=relative,
                    remaining_expanded=_MAX_EXPANDED_BYTES - expanded,
                )
                expanded += int(row["size_bytes"])
                rows.append(row)
            inventory: Dict[str, Any] = {
                "schema": PARAM_RUNTIME_ARTIFACT_SCHEMA,
                "files": rows,
                "expanded_size_bytes": expanded,
            }
            inventory_raw = _canonical_bytes(inventory)
            if len(inventory_raw) > _MAX_INVENTORY_BYTES:
                raise ParamArtifactError("PARAM inventory exceeds the working-memory budget")
            archive.writestr(_zip_info(PARAM_RUNTIME_INVENTORY_NAME), inventory_raw)
        output.flush()
        os.fsync(output.fileno())
    size = output_path.stat().st_size
    if not 0 < size <= _MAX_ARTIFACT_BYTES:
        raise ParamArtifactError("PARAM artifact size is outside the supported limit")
    return inventory


def _hash_file(path: Path, *, maximum: int, field: str) -> Tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ParamArtifactError(f"{field} is not a bounded regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _STREAM_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise ParamArtifactError(f"{field} exceeds its byte limit")
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or size != before.st_size:
            raise ParamArtifactError(f"{field} changed while it was hashed")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def render_param_artifact(param_directory: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Compatibility renderer restricted to the documented in-memory budget."""

    with tempfile.TemporaryDirectory(prefix="commcanary-param-render-") as raw_directory:
        output = Path(raw_directory) / "param-runtime.zip"
        inventory = _render_param_to_path(param_directory, output)
        rendered = _read_regular_bytes(
            output,
            maximum=_WORKING_MEMORY_BUDGET_BYTES,
            field="rendered PARAM compatibility artifact",
        )
    return rendered, inventory


def prepare_param_artifact(param_directory: Path, artifact_directory: Path) -> ParamArtifact:
    expanded_root = artifact_directory.expanduser()
    if expanded_root.is_symlink():
        raise ParamArtifactError("PARAM artifact directory must be a real directory")
    destination_root = expanded_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ParamArtifactError("PARAM artifact directory must be a real directory")
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".param-runtime-", suffix=".tmp", dir=destination_root
    )
    os.close(temporary_descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        inventory = _render_param_to_path(param_directory, temporary)
        digest, size = _hash_file(temporary, maximum=_MAX_ARTIFACT_BYTES, field="rendered PARAM artifact")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    destination = destination_root / f"param-runtime-{digest}.zip"
    try:
        os.link(temporary, destination, follow_symlinks=False)
        os.chmod(destination, 0o444)
    except FileExistsError:
        observed_digest, observed_size = _hash_file(
            destination,
            maximum=_MAX_ARTIFACT_BYTES,
            field="existing PARAM artifact",
        )
        if observed_digest != digest or observed_size != size:
            raise ParamArtifactError(f"PARAM artifact collision: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    return ParamArtifact(
        path=destination,
        sha256=digest,
        size_bytes=size,
        inventory_sha256=hashlib.sha256(_canonical_bytes(inventory)).hexdigest(),
        file_count=len(inventory["files"]),
    )


def stage_param_artifact(artifact: Path, destination: Path) -> StagedParamRuntime:
    """Verify and extract exact PARAM bytes into a private import root."""

    if destination.exists() or destination.is_symlink():
        raise ParamArtifactError("PARAM staging destination already exists")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ParamArtifactError("PARAM staging parent must be a real directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=parent))
    root = temporary / "param"
    root.mkdir(mode=0o700)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(artifact, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_ARTIFACT_BYTES:
            raise ParamArtifactError("PARAM artifact must be a bounded real regular file")
        with os.fdopen(os.dup(descriptor), "rb") as artifact_handle, zipfile.ZipFile(artifact_handle) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(infos) > _MAX_FILES + 1 or len(names) != len(set(names)):
                raise ParamArtifactError("PARAM artifact contains duplicate entries")
            expanded_headers = 0
            for info in infos:
                _safe_relative(info.filename)
                maximum = _MAX_INVENTORY_BYTES if info.filename == PARAM_RUNTIME_INVENTORY_NAME else _MAX_MEMBER_BYTES
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or not 0 <= info.file_size <= maximum
                ):
                    raise ParamArtifactError("PARAM artifact member encoding or size is unsupported")
                if info.filename != PARAM_RUNTIME_INVENTORY_NAME:
                    expanded_headers += info.file_size
                    if expanded_headers > _MAX_EXPANDED_BYTES:
                        raise ParamArtifactError("PARAM artifact expanded headers exceed their limit")
            info_by_name = {info.filename: info for info in infos}
            inventory_info = info_by_name.get(PARAM_RUNTIME_INVENTORY_NAME)
            if inventory_info is None:
                raise ParamArtifactError("PARAM artifact lacks its inventory")
            inventory_raw = _zip_entry_bytes(
                archive,
                inventory_info,
                maximum=_MAX_INVENTORY_BYTES,
                field="PARAM artifact inventory",
            )
            inventory = json.loads(inventory_raw)
            if (
                not isinstance(inventory, dict)
                or set(inventory) != {"schema", "files", "expanded_size_bytes"}
                or inventory.get("schema") != PARAM_RUNTIME_ARTIFACT_SCHEMA
                or inventory_raw != _canonical_bytes(inventory)
            ):
                raise ParamArtifactError("PARAM artifact inventory is unsupported or noncanonical")
            rows = inventory["files"]
            expanded = inventory["expanded_size_bytes"]
            if (
                not isinstance(rows, list)
                or not rows
                or len(rows) > _MAX_FILES
                or isinstance(expanded, bool)
                or not isinstance(expanded, int)
                or not 0 <= expanded <= _MAX_EXPANDED_BYTES
            ):
                raise ParamArtifactError("PARAM artifact inventory exceeds its resource limits")
            observed_names = []
            observed_size = 0
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes", "executable"}:
                    raise ParamArtifactError("PARAM artifact file entry is malformed")
                relative = _safe_relative(row["path"])
                digest = row["sha256"]
                size = row["size_bytes"]
                executable = row["executable"]
                if (
                    not isinstance(digest, str)
                    or _SHA256_RE.fullmatch(digest) is None
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or not 0 <= size <= _MAX_MEMBER_BYTES
                    or not isinstance(executable, bool)
                    or relative.as_posix() in observed_names
                ):
                    raise ParamArtifactError("PARAM artifact file identity is malformed")
                member_info = info_by_name.get(relative.as_posix())
                if member_info is None or member_info.file_size != size:
                    raise ParamArtifactError("PARAM artifact lacks an inventoried file")
                observed_names.append(relative.as_posix())
                observed_size += size
            if observed_names != sorted(observed_names) or observed_size != expanded:
                raise ParamArtifactError("PARAM artifact inventory ordering or size does not recompute")
            expected_names = set(observed_names) | {PARAM_RUNTIME_INVENTORY_NAME}
            if set(names) != expected_names or len(names) != len(expected_names):
                raise ParamArtifactError("PARAM artifact contains unbound entries")
            if observed_size != expanded_headers:
                raise ParamArtifactError("PARAM artifact expanded header size does not recompute")

            for row in rows:
                relative = _safe_relative(str(row["path"]))
                size = int(row["size_bytes"])
                digest = str(row["sha256"])
                executable = bool(row["executable"])
                info = info_by_name[relative.as_posix()]
                target = root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                member_digest = hashlib.sha256()
                member_size = 0
                with archive.open(info, "r") as source, target.open("xb") as handle:
                    while True:
                        chunk = source.read(_STREAM_CHUNK_BYTES)
                        if not chunk:
                            break
                        member_size += len(chunk)
                        if member_size > size:
                            raise ParamArtifactError("PARAM artifact member exceeds its inventoried size")
                        member_digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if member_size != size or member_digest.hexdigest() != digest:
                    raise ParamArtifactError("PARAM artifact file bytes do not match the inventory")
                os.chmod(target, 0o500 if executable else 0o400)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after:
            raise ParamArtifactError("PARAM artifact changed while it was staged")
    except ParamArtifactError:
        _remove_staging_tree(temporary)
        raise
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        _remove_staging_tree(temporary)
        raise ParamArtifactError(f"cannot stage PARAM artifact: {exc}") from exc
    except BaseException:
        _remove_staging_tree(temporary)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        alias = temporary / "param_bench"
        os.symlink("param", alias, target_is_directory=True)
        commit = temporary / ".stage-complete"
        with commit.open("xb") as handle:
            handle.write(b"commcanary-param-stage-v1\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(commit, 0o400)
        for directory, names, _files in os.walk(root, topdown=False):
            for name in names:
                os.chmod(Path(directory) / name, 0o500)
            os.chmod(directory, 0o500)
        if destination.exists() or destination.is_symlink():
            raise ParamArtifactError("PARAM staging destination appeared during verification")
        os.rename(temporary, destination)
        os.chmod(destination, 0o500)
    except BaseException:
        _remove_staging_tree(temporary)
        raise
    staged_root = destination / "param"
    return StagedParamRuntime(root=staged_root, import_paths=(destination, staged_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param-directory", required=True, type=Path)
    parser.add_argument("--artifact-directory", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = prepare_param_artifact(args.param_directory, args.artifact_directory)
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "inventory_sha256": artifact.inventory_sha256,
                "file_count": artifact.file_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARAM_RUNTIME_ARTIFACT_INPUT_ID",
    "PARAM_RUNTIME_ARTIFACT_SCHEMA",
    "ParamArtifact",
    "ParamArtifactError",
    "StagedParamRuntime",
    "prepare_param_artifact",
    "render_param_artifact",
    "stage_param_artifact",
]
