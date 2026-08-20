#!/usr/bin/env python3
"""Verify and privately stage the frozen Rostam executor before importing it."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_EXECUTOR_INPUT_ID = "rostam-executor-artifact"
_EXECUTOR_FORMAT = "python-zipapp.v1"
_MANIFEST_NAME = "run_manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_EXECUTOR_BYTES = 16 * 1024 * 1024
_MAX_EXECUTOR_MEMBERS = 256
_MAX_EXECUTOR_MEMBER_BYTES = 2 * 1024 * 1024
_MAX_EXECUTOR_EXPANDED_BYTES = 16 * 1024 * 1024
_EXECUTOR_INVENTORY_NAME = "rostam-executor.json"
_EXECUTOR_ARTIFACT_SCHEMA = "commcanary.rostam.executor-artifact.v2"
_EXECUTOR_ANALYSIS_VERSION = "commcanary.rostam.frozen-analysis.v1"
_EXECUTOR_MAIN = b"from experiments.rostam.lib.executor_cli import main\nraise SystemExit(main())\n"
_EXECUTOR_ENTRYPOINTS = {
    "analyze": "experiments.rostam.analyze:main",
    "evaluate-decision-gate": "experiments.rostam.evaluate_decision_gate:main",
    "execute-cell": "experiments.rostam.lib.cell_entrypoint:main",
    "run-python": "experiments.rostam.lib.executor_cli:run_python",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExecutorBootstrapError(RuntimeError):
    """Raised before any project module is imported from an untrusted path."""


@dataclass
class StagedExecutor:
    path: Path
    sha256: str
    _temporary_directory: Any

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> "StagedExecutor":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _strict_object_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutorBootstrapError(f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutorBootstrapError(f"{field} must be an object")
    return value


def _read_regular(path: Path, *, maximum: int, field: str) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ExecutorBootstrapError(f"{field} must not be a symbolic link")
    resolved = expanded.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ExecutorBootstrapError(f"{field} size or file type is unsupported")
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise ExecutorBootstrapError(f"{field} exceeds its byte limit")
        after = os.fstat(descriptor)
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
        ) or size != before.st_size:
            raise ExecutorBootstrapError(f"{field} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _inventory_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _safe_member_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ExecutorBootstrapError("executor archive member name is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExecutorBootstrapError(f"executor archive member path is unsafe: {value!r}")
    return value


def _validate_executor_archive(raw: bytes) -> None:
    """Validate every executable/resource member before Python sees the zipapp."""

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(infos) > _MAX_EXECUTOR_MEMBERS or len(names) != len(set(names)):
                raise ExecutorBootstrapError("executor archive member inventory is empty, oversized, or duplicated")
            expanded = 0
            for info in infos:
                _safe_member_name(info.filename)
                if info.is_dir() or info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
                    raise ExecutorBootstrapError("executor members must be unencrypted ZIP_STORED files")
                if not 0 <= info.file_size <= _MAX_EXECUTOR_MEMBER_BYTES:
                    raise ExecutorBootstrapError("executor member exceeds its expanded-size limit")
                expanded += info.file_size
                if expanded > _MAX_EXECUTOR_EXPANDED_BYTES:
                    raise ExecutorBootstrapError("executor expanded bytes exceed their limit")
            inventory_raw = archive.read(_EXECUTOR_INVENTORY_NAME)
            inventory = _object(
                json.loads(inventory_raw, object_pairs_hook=_strict_object_pairs),
                "executor inventory",
            )
            expected_fields = {
                "schema",
                "entrypoints",
                "analysis_version",
                "source_inventory_sha256",
                "schema_inventory_sha256",
                "source_files",
                "schema_files",
            }
            if (
                set(inventory) != expected_fields
                or inventory.get("schema") != _EXECUTOR_ARTIFACT_SCHEMA
                or inventory.get("entrypoints") != _EXECUTOR_ENTRYPOINTS
                or inventory.get("analysis_version") != _EXECUTOR_ANALYSIS_VERSION
                or inventory_raw != _inventory_bytes(inventory)
            ):
                raise ExecutorBootstrapError("executor inventory format is unsupported or noncanonical")
            inventories: Dict[str, Sequence[Mapping[str, Any]]] = {}
            for collection, digest_field in (
                ("source_files", "source_inventory_sha256"),
                ("schema_files", "schema_inventory_sha256"),
            ):
                rows = inventory[collection]
                digest = inventory[digest_field]
                if (
                    not isinstance(rows, list)
                    or not rows
                    or not isinstance(digest, str)
                    or _SHA256_RE.fullmatch(digest) is None
                    or hashlib.sha256(_inventory_bytes(rows)).hexdigest() != digest
                ):
                    raise ExecutorBootstrapError(f"executor {collection} inventory does not recompute")
                observed_names = []
                for row_index, raw_row in enumerate(rows):
                    row = _object(raw_row, f"executor {collection}[{row_index}]")
                    if set(row) != {"path", "sha256", "size_bytes"}:
                        raise ExecutorBootstrapError(f"executor {collection} row is not closed")
                    name = _safe_member_name(row["path"])
                    row_digest = row["sha256"]
                    size = row["size_bytes"]
                    if (
                        not isinstance(row_digest, str)
                        or _SHA256_RE.fullmatch(row_digest) is None
                        or isinstance(size, bool)
                        or not isinstance(size, int)
                        or not 0 <= size <= _MAX_EXECUTOR_MEMBER_BYTES
                    ):
                        raise ExecutorBootstrapError(f"executor {collection} row is malformed")
                    payload = archive.read(name)
                    if (
                        name in observed_names
                        or len(payload) != size
                        or hashlib.sha256(payload).hexdigest() != row_digest
                    ):
                        raise ExecutorBootstrapError(f"executor {collection} bytes do not match inventory")
                    observed_names.append(name)
                if observed_names != sorted(observed_names):
                    raise ExecutorBootstrapError(f"executor {collection} inventory is not sorted")
                inventories[collection] = rows
            source_names = {str(row["path"]) for row in inventories["source_files"]}
            schema_names = {str(row["path"]) for row in inventories["schema_files"]}
            if set(names) != source_names | schema_names | {_EXECUTOR_INVENTORY_NAME}:
                raise ExecutorBootstrapError("executor archive contains unlisted or missing members")
            if archive.read("__main__.py") != _EXECUTOR_MAIN:
                raise ExecutorBootstrapError("executor __main__.py is unsupported")
            if "__main__.py" not in source_names or any(
                name.endswith(".py") and name not in source_names for name in names
            ):
                raise ExecutorBootstrapError("executor Python members are not completely inventoried")
    except ExecutorBootstrapError:
        raise
    except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ExecutorBootstrapError(f"executor archive is invalid: {exc}") from exc


def _executor_binding(run_directory: Path, manifest_sha256: str) -> Tuple[Path, str, int]:
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ExecutorBootstrapError("expected manifest SHA-256 is malformed")
    raw = _read_regular(
        run_directory / _MANIFEST_NAME,
        maximum=_MAX_MANIFEST_BYTES,
        field="run manifest",
    )
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ExecutorBootstrapError("run manifest SHA-256 does not match the submitted plan")
    try:
        manifest = _object(json.loads(raw, object_pairs_hook=_strict_object_pairs), "run manifest")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutorBootstrapError(f"run manifest is not strict JSON: {exc}") from exc
    campaign = _object(manifest.get("campaign"), "run manifest.campaign")
    policy = _object(campaign.get("policy"), "run manifest.campaign.policy")
    executor = _object(policy.get("executor"), "run manifest.campaign.policy.executor")
    if executor.get("format") != _EXECUTOR_FORMAT or executor.get("artifact_input_id") != _EXECUTOR_INPUT_ID:
        raise ExecutorBootstrapError("run manifest does not bind the supported executor artifact")
    inputs = campaign.get("inputs")
    if not isinstance(inputs, list):
        raise ExecutorBootstrapError("run manifest.campaign.inputs must be an array")
    matches = [item for item in inputs if isinstance(item, Mapping) and item.get("id") == _EXECUTOR_INPUT_ID]
    if len(matches) != 1:
        raise ExecutorBootstrapError("run manifest must bind exactly one executor artifact input")
    reference = _object(matches[0], "executor artifact input")
    digest = reference.get("sha256")
    size = reference.get("size_bytes")
    paths = _object(policy.get("input_paths"), "run manifest.campaign.policy.input_paths")
    raw_path = paths.get(_EXECUTOR_INPUT_ID)
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= _MAX_EXECUTOR_BYTES
        or not isinstance(raw_path, str)
        or not raw_path
    ):
        raise ExecutorBootstrapError("executor artifact binding is malformed")
    return Path(raw_path), digest, size


def stage_executor_artifact(run_directory: Path, manifest_sha256: str) -> StagedExecutor:
    """Stage exact manifest-bound bytes without importing any project module."""

    source, expected_sha256, expected_size = _executor_binding(run_directory, manifest_sha256)
    raw = _read_regular(source, maximum=_MAX_EXECUTOR_BYTES, field="executor artifact")
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ExecutorBootstrapError("executor artifact bytes do not match the frozen campaign")
    _validate_executor_archive(raw)
    temporary = tempfile.TemporaryDirectory(prefix="commcanary-rostam-executor-")
    root = Path(temporary.name)
    os.chmod(root, 0o700)
    staged = root / f"rostam-executor-{expected_sha256}.pyz"
    try:
        with staged.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o400)
        staged_raw = _read_regular(staged, maximum=_MAX_EXECUTOR_BYTES, field="staged executor artifact")
        if staged_raw != raw:
            raise ExecutorBootstrapError("staged executor artifact changed before execution")
        return StagedExecutor(path=staged, sha256=expected_sha256, _temporary_directory=temporary)
    except (ExecutorBootstrapError, OSError):
        temporary.cleanup()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--executor-command",
        choices=("analyze", "evaluate-decision-gate", "execute-cell"),
        default="execute-cell",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, executor_args = _parser().parse_known_args(argv)
    if executor_args[:1] == ["--"]:
        executor_args = executor_args[1:]
    staged = stage_executor_artifact(args.run_directory, args.manifest_sha256)
    try:
        environment = dict(os.environ)
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["COMMCANARY_EXECUTOR_PATH"] = str(staged.path)
        environment["COMMCANARY_EXECUTOR_SHA256"] = staged.sha256
        command = [
            sys.executable,
            "-I",
            "-S",
            str(staged.path),
            args.executor_command,
            "--run-directory",
            str(args.run_directory),
            "--manifest-sha256",
            args.manifest_sha256,
            *executor_args,
        ]
        return int(subprocess.run(command, check=False, env=environment).returncode)
    finally:
        staged.close()


if __name__ == "__main__":  # pragma: no cover - exercised by SLURM wrappers
    raise SystemExit(main())
