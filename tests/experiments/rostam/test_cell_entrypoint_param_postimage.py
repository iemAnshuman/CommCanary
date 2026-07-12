"""Cells refuse to execute when PARAM is not at the reviewed postimage.

setup.sh applies the reviewed patch, but nothing prevented a cell from running
against a preimage checkout if setup was interrupted after venv creation. The
entrypoint now verifies the patched target against the manifest-bound patch
contract before any command runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import pytest  # type: ignore[import-not-found]

from experiments.rostam.lib import cell_entrypoint

_PATCHED = b"if getattr(collectiveArgs, 'use_triton', False):\n"
_POSTIMAGE = hashlib.sha256(_PATCHED).hexdigest()


def _contract(tmp_path: Path, *, postimage: str = _POSTIMAGE, target: str = "train/comms/pt/backend.py") -> Path:
    path = tmp_path / "param-patch-contract.json"
    path.write_text(
        json.dumps(
            {
                "schema": "commcanary.rostam.param-patch-contract.v1",
                "status": "reviewed",
                "target": {"path": target, "postimage_sha256": postimage},
            }
        ),
        encoding="utf-8",
    )
    return path


def _experiment(tmp_path: Path, *, content: bytes = _PATCHED) -> Path:
    experiment = tmp_path / "rostam"
    target = experiment / "third_party" / "param" / "train" / "comms" / "pt" / "backend.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    return experiment


def _inputs(contract: Path) -> Dict[str, Path]:
    return {"param-patch-contract": contract, "commcanary-wheel": contract.parent / "unused.whl"}


def test_postimage_match_is_accepted(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    cell_entrypoint._verify_param_postimage(experiment, _inputs(_contract(tmp_path)))


def test_unpatched_target_is_refused(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path, content=b"if collectiveArgs.use_triton:\n")
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="patch is not applied"):
        cell_entrypoint._verify_param_postimage(experiment, _inputs(_contract(tmp_path)))


def test_missing_target_is_refused(tmp_path: Path) -> None:
    experiment = tmp_path / "rostam"
    experiment.mkdir()
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="missing its patched target"):
        cell_entrypoint._verify_param_postimage(experiment, _inputs(_contract(tmp_path)))


def test_traversal_target_binding_is_refused(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    contract = _contract(tmp_path, target="../outside.py")
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="malformed or unsafe"):
        cell_entrypoint._verify_param_postimage(experiment, _inputs(contract))


def test_malformed_postimage_is_refused(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    contract = _contract(tmp_path, postimage="not-a-digest")
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="malformed or unsafe"):
        cell_entrypoint._verify_param_postimage(experiment, _inputs(contract))


def test_manifest_without_patch_contract_needs_no_checkout(tmp_path: Path) -> None:
    experiment = tmp_path / "rostam"
    experiment.mkdir()
    cell_entrypoint._verify_param_postimage(experiment, {})
