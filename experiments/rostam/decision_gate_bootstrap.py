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
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_WHEEL_BYTES = 128 * 1024 * 1024


class DecisionGateBootstrapError(RuntimeError):
    """Raised when the separately bound product wheel cannot be trusted."""


def validate_bound_wheel(path: Path, expected_sha256: str) -> Path:
    """Validate one regular wheel by bounded bytes and return its absolute path."""

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
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise DecisionGateBootstrapError(f"cannot open bound wheel safely: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DecisionGateBootstrapError("bound wheel must be a regular file")
        if not 0 < metadata.st_size <= _MAX_WHEEL_BYTES:
            raise DecisionGateBootstrapError("bound wheel size is outside the supported limit")
        digest = hashlib.sha256()
        remaining = _MAX_WHEEL_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining == 0 and os.read(descriptor, 1):
            raise DecisionGateBootstrapError("bound wheel exceeds the supported size limit")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise DecisionGateBootstrapError("bound wheel SHA-256 does not match the frozen campaign")
    return resolved


def import_bound_commcanary(wheel: Path) -> ModuleType:
    """Import CommCanary exclusively from ``wheel`` in a fresh child process."""

    if any(name == "commcanary" or name.startswith("commcanary.") for name in sys.modules):
        raise DecisionGateBootstrapError("CommCanary was imported before the bound wheel was activated")
    sys.path.insert(0, str(wheel))
    module = importlib.import_module("commcanary")
    origin = getattr(module, "__file__", None)
    expected_prefix = f"{wheel}{os.sep}"
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
    import_bound_commcanary(wheel)
    runner = importlib.import_module("experiments.rostam.decision_gate_physical")
    return int(runner.main(runner_args))


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun on Rostam
    raise SystemExit(main())
