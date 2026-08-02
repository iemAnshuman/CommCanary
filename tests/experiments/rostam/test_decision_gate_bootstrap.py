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
        archive.writestr("commcanary/lazy.py", 'LAZY_MARKER = "bound-wheel-lazy"\n')
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def _replacement_wheel(path: Path) -> Path:
    replacement = path.with_name("replacement.whl")
    with zipfile.ZipFile(replacement, "w") as archive:
        archive.writestr("commcanary/__init__.py", 'ORIGIN_MARKER = "replacement-wheel"\n')
        archive.writestr("commcanary/lazy.py", 'LAZY_MARKER = "replacement-wheel-lazy"\n')
    return replacement


def test_validate_bound_wheel_requires_exact_bytes(tmp_path: Path) -> None:
    wheel, digest = _wheel(tmp_path)

    staged = validate_bound_wheel(wheel, digest)
    try:
        assert staged.path != wheel.resolve()
        assert staged.path.read_bytes() == wheel.read_bytes()
        assert staged.path.stat().st_mode & 0o222 == 0
        assert staged.path.parent.stat().st_mode & 0o077 == 0
    finally:
        staged.close()

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
try:
    module = import_bound_commcanary(wheel)
    print(json.dumps({'marker': module.ORIGIN_MARKER, 'origin': module.__file__, 'staged': str(wheel.path)}))
finally:
    wheel.close()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(wheel), digest],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["marker"] == "bound-wheel"
    assert result["origin"].startswith(f"{result['staged']}/")
    assert result["staged"] != str(wheel.resolve())


def test_original_wheel_replacement_before_import_cannot_change_loaded_bytes(tmp_path: Path) -> None:
    wheel, digest = _wheel(tmp_path)
    replacement = _replacement_wheel(wheel)
    script = """
import json
import os
from pathlib import Path
from experiments.rostam.decision_gate_bootstrap import import_bound_commcanary, validate_bound_wheel
original = Path(__import__('sys').argv[1])
replacement = Path(__import__('sys').argv[2])
wheel = validate_bound_wheel(original, __import__('sys').argv[3])
os.replace(replacement, original)
try:
    module = import_bound_commcanary(wheel)
    print(json.dumps({'marker': module.ORIGIN_MARKER, 'origin': module.__file__}))
finally:
    wheel.close()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(wheel), str(replacement), digest],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["marker"] == "bound-wheel"
    assert str(wheel.resolve()) not in result["origin"]


def test_original_wheel_replacement_after_top_level_import_cannot_change_lazy_submodule(
    tmp_path: Path,
) -> None:
    wheel, digest = _wheel(tmp_path)
    replacement = _replacement_wheel(wheel)
    script = """
import importlib
import json
import os
from pathlib import Path
from experiments.rostam.decision_gate_bootstrap import import_bound_commcanary, validate_bound_wheel
original = Path(__import__('sys').argv[1])
replacement = Path(__import__('sys').argv[2])
wheel = validate_bound_wheel(original, __import__('sys').argv[3])
try:
    module = import_bound_commcanary(wheel)
    os.replace(replacement, original)
    lazy = importlib.import_module('commcanary.lazy')
    print(json.dumps({'marker': module.ORIGIN_MARKER, 'lazy': lazy.LAZY_MARKER}))
finally:
    wheel.close()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(wheel), str(replacement), digest],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "marker": "bound-wheel",
        "lazy": "bound-wheel-lazy",
    }


def test_bound_wheel_refuses_symlink_target(tmp_path: Path) -> None:
    wheel, digest = _wheel(tmp_path)
    link = tmp_path / "linked.whl"
    link.symlink_to(wheel)

    with pytest.raises(DecisionGateBootstrapError, match="symbolic link"):
        validate_bound_wheel(link, digest)
