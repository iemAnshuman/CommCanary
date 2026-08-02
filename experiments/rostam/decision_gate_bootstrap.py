"""Load the exact manifest-bound CommCanary wheel before the physical gate.

The reviewed Rostam virtual environments retain their original wheel marker.
This bootstrap preserves that guard while making the product code under test an
independent, exact campaign input.  It intentionally imports no CommCanary code
until after the wheel has been validated and placed first on ``sys.path``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_WHEEL_BYTES = 128 * 1024 * 1024


class DecisionGateBootstrapError(RuntimeError):
    """Raised when the separately bound product wheel cannot be trusted."""


@dataclass
class StagedWheel:
    """A private wheel copy whose lifetime is owned by this capability."""

    path: Path
    sha256: str
    _temporary_directory: Any

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> "StagedWheel":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while staging bound wheel")
        view = view[written:]


def _descriptor_digest(descriptor: int, *, maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > maximum:
            raise DecisionGateBootstrapError("bound wheel exceeds the supported size limit")
        digest.update(chunk)
    return digest.hexdigest(), size


def validate_bound_wheel(path: Path, expected_sha256: str) -> StagedWheel:
    """Copy the exact verified wheel bytes into a private immutable path."""

    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise DecisionGateBootstrapError("expected wheel SHA-256 is invalid")
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise DecisionGateBootstrapError("bound wheel must not be a symbolic link")
    resolved = expanded.resolve(strict=True)
    if resolved.suffix != ".whl":
        raise DecisionGateBootstrapError("bound product input must be a .whl file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    temporary = tempfile.TemporaryDirectory(prefix="commcanary-bound-wheel-")
    staging_root = Path(temporary.name)
    os.chmod(staging_root, 0o700)
    staged = staging_root / resolved.name
    descriptor = -1
    staged_descriptor = -1
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        temporary.cleanup()
        raise DecisionGateBootstrapError(f"cannot open bound wheel safely: {exc}") from exc
    try:
        before_name = os.stat(resolved, follow_symlinks=False)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or (before_name.st_dev, before_name.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise DecisionGateBootstrapError("bound wheel must be a regular file")
        if not 0 < metadata.st_size <= _MAX_WHEEL_BYTES:
            raise DecisionGateBootstrapError("bound wheel size is outside the supported limit")
        staged_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            staged_flags |= os.O_CLOEXEC
        staged_descriptor = os.open(staged, staged_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_WHEEL_BYTES + 1 - copied))
            if not chunk:
                break
            copied += len(chunk)
            if copied > _MAX_WHEEL_BYTES:
                raise DecisionGateBootstrapError("bound wheel exceeds the supported size limit")
            digest.update(chunk)
            _write_all(staged_descriptor, chunk)
        after = os.fstat(descriptor)
        identity_before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or copied != metadata.st_size:
            raise DecisionGateBootstrapError("bound wheel changed while it was staged")
        if digest.hexdigest() != expected_sha256:
            raise DecisionGateBootstrapError("bound wheel SHA-256 does not match the frozen campaign")
        os.fsync(staged_descriptor)
        os.close(staged_descriptor)
        staged_descriptor = -1
        os.chmod(staged, 0o400)
        staged_check = os.open(staged, flags)
        try:
            staged_digest, staged_size = _descriptor_digest(staged_check, maximum=_MAX_WHEEL_BYTES)
        finally:
            os.close(staged_check)
        if staged_size != metadata.st_size or staged_digest != expected_sha256:
            raise DecisionGateBootstrapError("private staged wheel does not match the verified source bytes")
        return StagedWheel(path=staged, sha256=expected_sha256, _temporary_directory=temporary)
    except (DecisionGateBootstrapError, OSError):
        temporary.cleanup()
        raise
    finally:
        if staged_descriptor >= 0:
            os.close(staged_descriptor)
        if descriptor >= 0:
            os.close(descriptor)


def import_bound_commcanary(wheel: StagedWheel) -> ModuleType:
    """Import CommCanary exclusively from ``wheel`` in a fresh child process."""

    if any(name == "commcanary" or name.startswith("commcanary.") for name in sys.modules):
        raise DecisionGateBootstrapError("CommCanary was imported before the bound wheel was activated")
    sys.path.insert(0, str(wheel.path))
    module = importlib.import_module("commcanary")
    origin = getattr(module, "__file__", None)
    expected_prefix = f"{wheel.path}{os.sep}"
    if not isinstance(origin, str) or not origin.startswith(expected_prefix):
        raise DecisionGateBootstrapError("CommCanary did not import from the bound product wheel")
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, runner_args = _parser().parse_known_args(argv)
    wheel = validate_bound_wheel(args.wheel, args.wheel_sha256)
    try:
        import_bound_commcanary(wheel)
        runner = importlib.import_module("experiments.rostam.decision_gate_physical")
        return int(runner.main(runner_args))
    finally:
        wheel.close()


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun on Rostam
    raise SystemExit(main())
