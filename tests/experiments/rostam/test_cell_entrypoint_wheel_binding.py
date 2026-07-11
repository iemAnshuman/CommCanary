"""The cell entrypoint refuses a venv whose installed wheel is not the bound one.

A manifest binds the reviewed CommCanary wheel file, but cells execute whatever
setup.sh last installed into the venv. The marker written at install time is
the only link between the two; a venv without it, or with a different digest,
must fail closed instead of producing evidence under an unreviewed wheel.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest  # type: ignore[import-not-found]

from experiments.rostam.lib import cell_entrypoint

_BOUND_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def _manifest(wheel_sha256: Optional[str] = _BOUND_DIGEST) -> Any:
    inputs = [SimpleNamespace(id="rostam-catalog", sha256="c" * 64)]
    if wheel_sha256 is not None:
        inputs.append(SimpleNamespace(id="commcanary-wheel", sha256=wheel_sha256))
    return SimpleNamespace(campaign=SimpleNamespace(inputs=tuple(inputs)))


def _write_marker(venv: Path, content: str) -> None:
    (venv / "commcanary-wheel.sha256").write_text(content, encoding="ascii")


def test_marker_matching_the_bound_wheel_is_accepted(tmp_path: Path) -> None:
    _write_marker(tmp_path, f"{_BOUND_DIGEST}\n")
    cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())


def test_venv_without_a_marker_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="does not record"):
        cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())


def test_stale_marker_names_both_digests(tmp_path: Path) -> None:
    _write_marker(tmp_path, f"{_OTHER_DIGEST}\n")
    with pytest.raises(cell_entrypoint.CellEntrypointError) as captured:
        cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())
    message = str(captured.value)
    assert _OTHER_DIGEST in message
    assert _BOUND_DIGEST in message
    assert "rerun setup.sh" in message


def test_symlinked_marker_is_refused_even_when_its_target_matches(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.sha256"
    real.write_text(f"{_BOUND_DIGEST}\n", encoding="ascii")
    (tmp_path / "commcanary-wheel.sha256").symlink_to(real)
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="does not record"):
        cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())


def test_marker_that_is_not_a_digest_is_refused(tmp_path: Path) -> None:
    _write_marker(tmp_path, "commcanary-0.3.0-py3-none-any.whl\n")
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="not a sha256 digest"):
        cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())


def test_uppercase_digest_is_refused_rather_than_normalized(tmp_path: Path) -> None:
    _write_marker(tmp_path, f"{_BOUND_DIGEST.upper()}\n")
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="not a sha256 digest"):
        cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())


def test_oversized_marker_is_rejected_by_the_bounded_read(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 4096)
    with pytest.raises(cell_entrypoint.ContractError):
        cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest())


def test_manifest_without_a_wheel_input_needs_no_marker(tmp_path: Path) -> None:
    cell_entrypoint._verify_venv_wheel_binding(tmp_path, _manifest(wheel_sha256=None))
