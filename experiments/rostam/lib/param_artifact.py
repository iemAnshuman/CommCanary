"""Build and privately stage the complete reviewed PARAM implementation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence, Tuple

PARAM_RUNTIME_ARTIFACT_INPUT_ID = "param-runtime-artifact"
PARAM_RUNTIME_ARTIFACT_SCHEMA = "commcanary.rostam.param-runtime-artifact.v1"
PARAM_RUNTIME_INVENTORY_NAME = "param-runtime.json"
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_INVENTORY_BYTES = 64 * 1024 * 1024
_MAX_FILES = 100_000
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


def render_param_artifact(param_directory: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Render deterministic bytes for every non-Git file in the patched tree."""

    root = param_directory.resolve()
    rows = []
    payloads: Dict[str, bytes] = {}
    expanded = 0
    for path in _source_files(root):
        relative = path.relative_to(root).as_posix()
        before = os.stat(path, follow_symlinks=False)
        raw = _read_regular_bytes(
            path,
            maximum=_MAX_EXPANDED_BYTES - expanded,
            field=f"PARAM source file {relative!r}",
            allow_empty=True,
        )
        expanded += len(raw)
        if expanded > _MAX_EXPANDED_BYTES:
            raise ParamArtifactError("PARAM source exceeds the expanded-byte limit")
        executable = bool(before.st_mode & 0o111)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "executable": executable,
            }
        )
    inventory: Dict[str, Any] = {
        "schema": PARAM_RUNTIME_ARTIFACT_SCHEMA,
        "files": rows,
        "expanded_size_bytes": expanded,
    }
    inventory_raw = _canonical_bytes(inventory)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for row in rows:
            relative = str(row["path"])
            archive.writestr(_zip_info(relative, executable=bool(row["executable"])), payloads[relative])
        archive.writestr(_zip_info(PARAM_RUNTIME_INVENTORY_NAME), inventory_raw)
    rendered = buffer.getvalue()
    if not rendered or len(rendered) > _MAX_ARTIFACT_BYTES:
        raise ParamArtifactError("PARAM artifact size is outside the supported limit")
    return rendered, inventory


def prepare_param_artifact(param_directory: Path, artifact_directory: Path) -> ParamArtifact:
    rendered, inventory = render_param_artifact(param_directory)
    digest = hashlib.sha256(rendered).hexdigest()
    expanded_root = artifact_directory.expanduser()
    if expanded_root.is_symlink():
        raise ParamArtifactError("PARAM artifact directory must be a real directory")
    destination_root = expanded_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ParamArtifactError("PARAM artifact directory must be a real directory")
    destination = destination_root / f"param-runtime-{digest}.zip"
    try:
        with destination.open("xb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o444)
    except FileExistsError:
        if _read_regular_bytes(destination, maximum=_MAX_ARTIFACT_BYTES, field="existing PARAM artifact") != rendered:
            raise ParamArtifactError(f"PARAM artifact collision: {destination}")
    return ParamArtifact(
        path=destination,
        sha256=digest,
        size_bytes=len(rendered),
        inventory_sha256=hashlib.sha256(_canonical_bytes(inventory)).hexdigest(),
        file_count=len(inventory["files"]),
    )


def stage_param_artifact(artifact: Path, destination: Path) -> StagedParamRuntime:
    """Verify and extract exact PARAM bytes into a private import root."""

    raw = _read_regular_bytes(artifact, maximum=_MAX_ARTIFACT_BYTES, field="PARAM artifact")
    if destination.exists() or destination.is_symlink():
        raise ParamArtifactError("PARAM staging destination already exists")
    destination.mkdir(parents=True, mode=0o700)
    root = destination / "param"
    root.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ParamArtifactError("PARAM artifact contains duplicate entries")
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
                    or size < 0
                    or not isinstance(executable, bool)
                    or relative.as_posix() in observed_names
                ):
                    raise ParamArtifactError("PARAM artifact file identity is malformed")
                info = info_by_name.get(relative.as_posix())
                if info is None or info.is_dir() or info.filename.endswith("/"):
                    raise ParamArtifactError("PARAM artifact lacks an inventoried file")
                payload = _zip_entry_bytes(
                    archive,
                    info,
                    maximum=min(size, _MAX_EXPANDED_BYTES - observed_size),
                    field=f"PARAM artifact file {relative.as_posix()!r}",
                )
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                    raise ParamArtifactError("PARAM artifact file bytes do not match the inventory")
                observed_names.append(relative.as_posix())
                observed_size += size
                target = root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with target.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(target, 0o500 if executable else 0o400)
            if observed_names != sorted(observed_names) or observed_size != expanded:
                raise ParamArtifactError("PARAM artifact inventory ordering or size does not recompute")
            expected_names = set(observed_names) | {PARAM_RUNTIME_INVENTORY_NAME}
            if set(names) != expected_names or len(names) != len(expected_names):
                raise ParamArtifactError("PARAM artifact contains unbound entries")
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise ParamArtifactError(f"cannot stage PARAM artifact: {exc}") from exc
    alias = destination / "param_bench"
    os.symlink("param", alias, target_is_directory=True)
    for directory, names, _files in os.walk(root, topdown=False):
        for name in names:
            os.chmod(Path(directory) / name, 0o500)
        os.chmod(directory, 0o500)
    os.chmod(destination, 0o500)
    return StagedParamRuntime(root=root, import_paths=(destination, root))


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
