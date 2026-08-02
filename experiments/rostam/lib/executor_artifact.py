"""Build the content-addressed Python executor used by new Rostam runs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

EXECUTOR_ARTIFACT_SCHEMA = "commcanary.rostam.executor-artifact.v1"
EXECUTOR_ARTIFACT_INPUT_ID = "rostam-executor-artifact"
EXECUTOR_BOOTSTRAP_INPUT_ID = "rostam-executor-bootstrap"
EXECUTOR_POLICY_FORMAT = "python-zipapp.v1"
EXECUTOR_INVENTORY_NAME = "rostam-executor.json"
EXECUTOR_MAIN = ("from experiments.rostam.lib.cell_entrypoint import main\nraise SystemExit(main())\n").encode("utf-8")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_EXECUTOR_BYTES = 16 * 1024 * 1024


class ExecutorArtifactError(RuntimeError):
    """Raised when executor sources or staged artifact bytes are unsafe."""


@dataclass(frozen=True)
class ExecutorArtifact:
    path: Path
    sha256: str
    size_bytes: int
    inventory_sha256: str
    source_files: Tuple[str, ...]


def executor_source_files(experiment_directory: Path) -> Tuple[Path, ...]:
    """Discover importable Rostam package sources without a handwritten list."""

    root = experiment_directory.resolve()
    if root.name != "rostam" or root.parent.name != "experiments":
        raise ExecutorArtifactError("executor source root must be the experiments/rostam package")
    package_init = root.parent / "__init__.py"
    if package_init.is_symlink() or not package_init.is_file():
        raise ExecutorArtifactError("experiments package initializer is missing or unsafe")
    discovered = [package_init]
    pending = [root]
    while pending:
        directory = pending.pop()
        if directory.is_symlink() or not directory.is_dir():
            raise ExecutorArtifactError(f"executor package directory is missing or unsafe: {directory}")
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ExecutorArtifactError(f"executor source path may not be a symlink: {child}")
            if child.is_file() and child.suffix == ".py":
                discovered.append(child)
            elif child.is_dir() and (child / "__init__.py").is_file():
                pending.append(child)
    return tuple(sorted(discovered, key=lambda item: item.relative_to(root.parent.parent).as_posix()))


def _source_payloads(experiment_directory: Path) -> Dict[str, bytes]:
    repository_root = experiment_directory.resolve().parent.parent
    payloads: Dict[str, bytes] = {}
    for path in executor_source_files(experiment_directory):
        relative = path.relative_to(repository_root).as_posix()
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ExecutorArtifactError(f"executor source must be a real regular file: {relative}")
        raw = path.read_bytes()
        after = os.lstat(path)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(raw) != before.st_size:
            raise ExecutorArtifactError(f"executor source changed while it was read: {relative}")
        payloads[relative] = raw
    return payloads


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    return info


def render_executor_artifact(experiment_directory: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Render deterministic zipapp bytes and their embedded source inventory."""

    payloads = _source_payloads(experiment_directory)
    source_inventory = [
        {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for name, raw in sorted(payloads.items())
    ]
    inventory: Dict[str, Any] = {
        "schema": EXECUTOR_ARTIFACT_SCHEMA,
        "entrypoint": "experiments.rostam.lib.cell_entrypoint:main",
        "source_files": source_inventory,
    }
    inventory_bytes = (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        archive.writestr(_zip_info("__main__.py"), EXECUTOR_MAIN)
        for name, raw in sorted(payloads.items()):
            archive.writestr(_zip_info(name), raw)
        archive.writestr(_zip_info(EXECUTOR_INVENTORY_NAME), inventory_bytes)
    rendered = buffer.getvalue()
    if not rendered or len(rendered) > _MAX_EXECUTOR_BYTES:
        raise ExecutorArtifactError("rendered Rostam executor size is outside the supported limit")
    return rendered, inventory


def prepare_executor_artifact(experiment_directory: Path, artifact_directory: Path) -> ExecutorArtifact:
    """Install or reuse the exact content-addressed executor zipapp."""

    rendered, inventory = render_executor_artifact(experiment_directory)
    digest = hashlib.sha256(rendered).hexdigest()
    inventory_bytes = (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    destination_root = artifact_directory.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ExecutorArtifactError("executor artifact directory must be a real directory")
    destination = destination_root / f"rostam-executor-{digest}.pyz"
    try:
        with destination.open("xb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o444)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != rendered:
            raise ExecutorArtifactError(f"executor artifact collision: {destination}")
    return ExecutorArtifact(
        path=destination,
        sha256=digest,
        size_bytes=len(rendered),
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        source_files=tuple(str(item["path"]) for item in inventory["source_files"]),
    )


def validate_executor_artifact(experiment_directory: Path, artifact: Path) -> ExecutorArtifact:
    """Require an artifact to equal a fresh rendering of all package sources."""

    if artifact.is_symlink() or not artifact.is_file():
        raise ExecutorArtifactError("Rostam executor artifact must be a real regular file")
    rendered, inventory = render_executor_artifact(experiment_directory)
    observed = artifact.read_bytes()
    if observed != rendered:
        raise ExecutorArtifactError("Rostam executor artifact does not match the current package sources")
    digest = hashlib.sha256(rendered).hexdigest()
    inventory_bytes = (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return ExecutorArtifact(
        path=artifact.resolve(),
        sha256=digest,
        size_bytes=len(rendered),
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        source_files=tuple(str(item["path"]) for item in inventory["source_files"]),
    )


__all__ = [
    "EXECUTOR_ARTIFACT_INPUT_ID",
    "EXECUTOR_ARTIFACT_SCHEMA",
    "EXECUTOR_BOOTSTRAP_INPUT_ID",
    "EXECUTOR_POLICY_FORMAT",
    "ExecutorArtifact",
    "ExecutorArtifactError",
    "executor_source_files",
    "prepare_executor_artifact",
    "render_executor_artifact",
    "validate_executor_artifact",
]
