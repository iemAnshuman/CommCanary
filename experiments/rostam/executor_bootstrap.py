#!/usr/bin/env python3
"""Verify and privately stage the frozen Rostam executor before importing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_EXECUTOR_INPUT_ID = "rostam-executor-artifact"
_EXECUTOR_FORMAT = "python-zipapp.v1"
_MANIFEST_NAME = "run_manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_EXECUTOR_BYTES = 16 * 1024 * 1024
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
            raise ExecutorBootstrapError(f"run manifest contains duplicate key {key!r}")
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, executor_args = _parser().parse_known_args(argv)
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
            str(staged.path),
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
