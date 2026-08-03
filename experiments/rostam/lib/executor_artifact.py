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
from typing import Any, Dict, Optional, Tuple

EXECUTOR_ARTIFACT_SCHEMA = "commcanary.rostam.executor-artifact.v2"
EXECUTOR_ARTIFACT_INPUT_ID = "rostam-executor-artifact"
EXECUTOR_BOOTSTRAP_INPUT_ID = "rostam-executor-bootstrap"
EXECUTOR_POLICY_FORMAT = "python-zipapp.v1"
EXECUTOR_INVENTORY_NAME = "rostam-executor.json"
EXECUTOR_ANALYSIS_VERSION = "commcanary.rostam.frozen-analysis.v1"
EXECUTOR_ANALYZE_ENTRY_POINT = "experiments.rostam.analyze:main"
EXECUTOR_EVALUATE_ENTRY_POINT = "experiments.rostam.evaluate_decision_gate:main"
EXECUTOR_CELL_ENTRY_POINT = "experiments.rostam.lib.cell_entrypoint:main"
EXECUTOR_RUN_PYTHON_ENTRY_POINT = "experiments.rostam.lib.executor_cli:run_python"
EXECUTOR_MAIN = ("from experiments.rostam.lib.executor_cli import main\nraise SystemExit(main())\n").encode("utf-8")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_EXECUTOR_BYTES = 16 * 1024 * 1024


class ExecutorArtifactError(RuntimeError):
    """Raised when executor sources or staged artifact bytes are unsafe."""


def _read_regular_bytes(path: Path, *, maximum: int, field: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExecutorArtifactError(f"cannot open {field}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ExecutorArtifactError(f"{field} must be a bounded real regular file")
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise ExecutorArtifactError(f"{field} exceeds its byte limit")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or size != before.st_size:
            raise ExecutorArtifactError(f"{field} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ExecutorArtifact:
    path: Path
    sha256: str
    size_bytes: int
    inventory_sha256: str
    source_inventory_sha256: str
    schema_inventory_sha256: str
    source_files: Tuple[str, ...]
    schema_files: Tuple[str, ...]

    def analyzer_record(self, entry_point: str, *, policy_sha256: Optional[str] = None) -> Dict[str, Any]:
        if entry_point not in {EXECUTOR_ANALYZE_ENTRY_POINT, EXECUTOR_EVALUATE_ENTRY_POINT}:
            raise ExecutorArtifactError(f"unsupported frozen analyzer entry point: {entry_point}")
        result: Dict[str, Any] = {
            "schema": EXECUTOR_ANALYSIS_VERSION,
            "artifact_sha256": self.sha256,
            "artifact_size_bytes": self.size_bytes,
            "inventory_sha256": self.inventory_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "schema_inventory_sha256": self.schema_inventory_sha256,
            "entry_point": entry_point,
            "version": EXECUTOR_ANALYSIS_VERSION,
        }
        if policy_sha256 is not None:
            result["policy_sha256"] = policy_sha256
        return result


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


def executor_schema_files(experiment_directory: Path) -> Tuple[Path, ...]:
    """Discover the closed JSON-schema resource inventory used by analysis."""

    root = experiment_directory.resolve()
    schema_directory = root / "schemas"
    if schema_directory.is_symlink() or not schema_directory.is_dir():
        raise ExecutorArtifactError("executor schema directory is missing or unsafe")
    files = []
    for child in sorted(schema_directory.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise ExecutorArtifactError(f"executor schema path may not be a symlink: {child}")
        if child.is_file() and child.suffix == ".json":
            files.append(child)
        elif child.is_dir():
            raise ExecutorArtifactError(f"executor schema directory contains an unexpected directory: {child}")
    if not files:
        raise ExecutorArtifactError("executor schema inventory is empty")
    return tuple(files)


def _payloads(experiment_directory: Path, paths: Tuple[Path, ...], *, field: str) -> Dict[str, bytes]:
    repository_root = experiment_directory.resolve().parent.parent
    payloads: Dict[str, bytes] = {}
    for path in paths:
        relative = path.relative_to(repository_root).as_posix()
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ExecutorArtifactError(f"executor {field} must be a real regular file: {relative}")
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
            raise ExecutorArtifactError(f"executor {field} changed while it was read: {relative}")
        payloads[relative] = raw
    return payloads


def _inventory_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _inventory_sha256(value: Any) -> str:
    return hashlib.sha256(_inventory_bytes(value)).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    return info


def render_executor_artifact(experiment_directory: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Render deterministic zipapp bytes and their embedded source inventory."""

    source_payloads = _payloads(
        experiment_directory,
        executor_source_files(experiment_directory),
        field="source",
    )
    schema_payloads = _payloads(
        experiment_directory,
        executor_schema_files(experiment_directory),
        field="schema",
    )
    source_inventory = [
        {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for name, raw in sorted(source_payloads.items())
    ]
    schema_inventory = [
        {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for name, raw in sorted(schema_payloads.items())
    ]
    inventory: Dict[str, Any] = {
        "schema": EXECUTOR_ARTIFACT_SCHEMA,
        "entrypoints": {
            "analyze": EXECUTOR_ANALYZE_ENTRY_POINT,
            "evaluate-decision-gate": EXECUTOR_EVALUATE_ENTRY_POINT,
            "execute-cell": EXECUTOR_CELL_ENTRY_POINT,
            "run-python": EXECUTOR_RUN_PYTHON_ENTRY_POINT,
        },
        "analysis_version": EXECUTOR_ANALYSIS_VERSION,
        "source_inventory_sha256": _inventory_sha256(source_inventory),
        "schema_inventory_sha256": _inventory_sha256(schema_inventory),
        "source_files": source_inventory,
        "schema_files": schema_inventory,
    }
    inventory_bytes = _inventory_bytes(inventory)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        archive.writestr(_zip_info("__main__.py"), EXECUTOR_MAIN)
        for name, raw in sorted({**source_payloads, **schema_payloads}.items()):
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
    inventory_bytes = _inventory_bytes(inventory)
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
        if _read_regular_bytes(destination, maximum=_MAX_EXECUTOR_BYTES, field="existing executor artifact") != rendered:
            raise ExecutorArtifactError(f"executor artifact collision: {destination}")
    return ExecutorArtifact(
        path=destination,
        sha256=digest,
        size_bytes=len(rendered),
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        source_inventory_sha256=str(inventory["source_inventory_sha256"]),
        schema_inventory_sha256=str(inventory["schema_inventory_sha256"]),
        source_files=tuple(str(item["path"]) for item in inventory["source_files"]),
        schema_files=tuple(str(item["path"]) for item in inventory["schema_files"]),
    )


def validate_executor_artifact(experiment_directory: Path, artifact: Path) -> ExecutorArtifact:
    """Require an artifact to equal a fresh rendering of all package sources."""

    rendered, inventory = render_executor_artifact(experiment_directory)
    observed = _read_regular_bytes(artifact, maximum=_MAX_EXECUTOR_BYTES, field="Rostam executor artifact")
    if observed != rendered:
        raise ExecutorArtifactError("Rostam executor artifact does not match the current package sources")
    digest = hashlib.sha256(rendered).hexdigest()
    inventory_bytes = _inventory_bytes(inventory)
    return ExecutorArtifact(
        path=artifact.resolve(),
        sha256=digest,
        size_bytes=len(rendered),
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        source_inventory_sha256=str(inventory["source_inventory_sha256"]),
        schema_inventory_sha256=str(inventory["schema_inventory_sha256"]),
        source_files=tuple(str(item["path"]) for item in inventory["source_files"]),
        schema_files=tuple(str(item["path"]) for item in inventory["schema_files"]),
    )


def load_executor_artifact(artifact: Path) -> ExecutorArtifact:
    """Load and self-verify one executor artifact for frozen analysis."""

    raw = _read_regular_bytes(artifact, maximum=_MAX_EXECUTOR_BYTES, field="Rostam executor artifact")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            inventory_raw = archive.read(EXECUTOR_INVENTORY_NAME)
            inventory = json.loads(inventory_raw)
            if not isinstance(inventory, dict) or inventory.get("schema") != EXECUTOR_ARTIFACT_SCHEMA:
                raise ExecutorArtifactError("Rostam executor inventory schema is unsupported")
            expected_fields = {
                "schema",
                "entrypoints",
                "analysis_version",
                "source_inventory_sha256",
                "schema_inventory_sha256",
                "source_files",
                "schema_files",
            }
            if set(inventory) != expected_fields:
                raise ExecutorArtifactError("Rostam executor inventory fields are not closed")
            if inventory["entrypoints"] != {
                "analyze": EXECUTOR_ANALYZE_ENTRY_POINT,
                "evaluate-decision-gate": EXECUTOR_EVALUATE_ENTRY_POINT,
                "execute-cell": EXECUTOR_CELL_ENTRY_POINT,
                "run-python": EXECUTOR_RUN_PYTHON_ENTRY_POINT,
            } or inventory["analysis_version"] != EXECUTOR_ANALYSIS_VERSION:
                raise ExecutorArtifactError("Rostam executor entry-point inventory is unsupported")
            for collection, digest_field in (
                ("source_files", "source_inventory_sha256"),
                ("schema_files", "schema_inventory_sha256"),
            ):
                rows = inventory[collection]
                if not isinstance(rows, list) or not rows or _inventory_sha256(rows) != inventory[digest_field]:
                    raise ExecutorArtifactError(f"Rostam executor {collection} inventory does not recompute")
                names = []
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
                        raise ExecutorArtifactError(f"Rostam executor {collection} entry is malformed")
                    name = row["path"]
                    if not isinstance(name, str) or not name or name.startswith("/") or ".." in Path(name).parts:
                        raise ExecutorArtifactError(f"Rostam executor {collection} path is unsafe")
                    payload = archive.read(name)
                    if (
                        name in names
                        or hashlib.sha256(payload).hexdigest() != row["sha256"]
                        or len(payload) != row["size_bytes"]
                    ):
                        raise ExecutorArtifactError(f"Rostam executor {collection} bytes do not match inventory")
                    names.append(name)
                if names != sorted(names):
                    raise ExecutorArtifactError(f"Rostam executor {collection} inventory is not sorted")
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ExecutorArtifactError):
            raise
        raise ExecutorArtifactError(f"cannot load Rostam executor artifact: {exc}") from exc
    canonical_inventory = _inventory_bytes(inventory)
    if inventory_raw != canonical_inventory:
        raise ExecutorArtifactError("Rostam executor inventory is not canonical")
    return ExecutorArtifact(
        path=artifact.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        inventory_sha256=hashlib.sha256(canonical_inventory).hexdigest(),
        source_inventory_sha256=str(inventory["source_inventory_sha256"]),
        schema_inventory_sha256=str(inventory["schema_inventory_sha256"]),
        source_files=tuple(str(item["path"]) for item in inventory["source_files"]),
        schema_files=tuple(str(item["path"]) for item in inventory["schema_files"]),
    )


__all__ = [
    "EXECUTOR_ARTIFACT_INPUT_ID",
    "EXECUTOR_ARTIFACT_SCHEMA",
    "EXECUTOR_ANALYSIS_VERSION",
    "EXECUTOR_ANALYZE_ENTRY_POINT",
    "EXECUTOR_BOOTSTRAP_INPUT_ID",
    "EXECUTOR_POLICY_FORMAT",
    "EXECUTOR_RUN_PYTHON_ENTRY_POINT",
    "EXECUTOR_EVALUATE_ENTRY_POINT",
    "ExecutorArtifact",
    "ExecutorArtifactError",
    "executor_source_files",
    "executor_schema_files",
    "load_executor_artifact",
    "prepare_executor_artifact",
    "render_executor_artifact",
    "validate_executor_artifact",
]
