from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from experiments.rostam.decision_gate_bootstrap import DecisionGateBootstrapError, validate_bound_wheel


def _wheel(tmp_path: Path) -> tuple[Path, str]:
    wheel = tmp_path / "commcanary-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("commcanary/__init__.py", 'ORIGIN_MARKER = "bound-wheel"\n')
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_validate_bound_wheel_requires_exact_bytes(tmp_path: Path) -> None:
    wheel, digest = _wheel(tmp_path)

    assert validate_bound_wheel(wheel, digest) == wheel.resolve()

    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(DecisionGateBootstrapError, match="does not match"):
        validate_bound_wheel(wheel, digest)


def test_bound_wheel_wins_over_checkout_in_fresh_process(tmp_path: Path) -> None:
    wheel, digest = _wheel(tmp_path)
    script = """
import json
from pathlib import Path
from experiments.rostam.decision_gate_bootstrap import import_bound_commcanary, validate_bound_wheel
wheel = validate_bound_wheel(Path(__import__('sys').argv[1]), __import__('sys').argv[2])
module = import_bound_commcanary(wheel)
print(json.dumps({'marker': module.ORIGIN_MARKER, 'origin': module.__file__}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(wheel), digest],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["marker"] == "bound-wheel"
    assert result["origin"].startswith(f"{wheel.resolve()}/")


def test_bound_wheel_refuses_symlink_target(tmp_path: Path) -> None:
    wheel, digest = _wheel(tmp_path)
    link = tmp_path / "linked.whl"
    link.symlink_to(wheel)

    with pytest.raises(DecisionGateBootstrapError, match="symbolic link"):
        validate_bound_wheel(link, digest)
