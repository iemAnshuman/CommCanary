from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from experiments.rostam.executor_bootstrap import ExecutorBootstrapError, stage_executor_artifact
from experiments.rostam.lib.executor_artifact import (
    EXECUTOR_ARTIFACT_INPUT_ID,
    EXECUTOR_ARTIFACT_SCHEMA,
    EXECUTOR_INVENTORY_NAME,
    EXECUTOR_POLICY_FORMAT,
    executor_source_files,
    prepare_executor_artifact,
    render_executor_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "rostam"


def _copy_executor_sources(tmp_path: Path) -> Path:
    copied_root = tmp_path / "copy"
    copied_experiment = copied_root / "experiments" / "rostam"
    for source in executor_source_files(EXPERIMENT_DIRECTORY):
        destination = copied_root / source.relative_to(REPOSITORY_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return copied_experiment


def _run_directory(tmp_path: Path, artifact: Path, digest: str, size: int) -> tuple[Path, str]:
    run_directory = tmp_path / "run"
    run_directory.mkdir(exist_ok=True)
    manifest = {
        "campaign": {
            "inputs": [
                {
                    "id": EXECUTOR_ARTIFACT_INPUT_ID,
                    "sha256": digest,
                    "size_bytes": size,
                }
            ],
            "policy": {
                "executor": {
                    "format": EXECUTOR_POLICY_FORMAT,
                    "artifact_input_id": EXECUTOR_ARTIFACT_INPUT_ID,
                },
                "input_paths": {EXECUTOR_ARTIFACT_INPUT_ID: str(artifact)},
            },
        }
    }
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (run_directory / "run_manifest.json").write_bytes(raw)
    return run_directory, hashlib.sha256(raw).hexdigest()


def test_executor_artifact_is_deterministic_and_inventories_every_package_source(tmp_path: Path) -> None:
    first = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")
    second = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")

    assert first == second
    assert first.sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()
    expected = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in executor_source_files(EXPERIMENT_DIRECTORY)}
    assert set(first.source_files) == expected
    with zipfile.ZipFile(first.path) as archive:
        inventory = json.loads(archive.read(EXECUTOR_INVENTORY_NAME))
        assert inventory["schema"] == EXECUTOR_ARTIFACT_SCHEMA
        assert {item["path"] for item in inventory["source_files"]} == expected
        assert archive.read("__main__.py").startswith(b"from experiments.rostam.lib.cell_entrypoint")


def test_bootstrap_stages_valid_executor_and_isolated_python_imports_it(tmp_path: Path) -> None:
    artifact = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")
    run_directory, manifest_sha256 = _run_directory(
        tmp_path,
        artifact.path,
        artifact.sha256,
        artifact.size_bytes,
    )

    staged = stage_executor_artifact(run_directory, manifest_sha256)
    try:
        assert staged.path != artifact.path
        assert staged.path.read_bytes() == artifact.path.read_bytes()
        completed = subprocess.run(
            [sys.executable, "-I", str(staged.path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert "Execute exactly one manifest-owned physical cell" in completed.stdout
    finally:
        staged.close()


def test_every_python_source_mutation_is_rejected_before_executor_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_experiment = _copy_executor_sources(tmp_path)
    baseline, _inventory = render_executor_artifact(copied_experiment)
    candidate = tmp_path / "candidate.pyz"
    candidate.write_bytes(baseline)
    baseline_sha256 = hashlib.sha256(baseline).hexdigest()
    run_directory, manifest_sha256 = _run_directory(
        tmp_path,
        candidate,
        baseline_sha256,
        len(baseline),
    )
    imported = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("mutated executor must not start")

    monkeypatch.setattr("experiments.rostam.executor_bootstrap.subprocess.run", forbidden)
    copied_root = copied_experiment.parent.parent
    sources = executor_source_files(copied_experiment)
    assert sources
    for source in sources:
        original = source.read_bytes()
        source.write_bytes(original + b'\nraise RuntimeError("mutated executor source imported")\n')
        try:
            mutated, _ = render_executor_artifact(copied_experiment)
            candidate.write_bytes(mutated)
            with pytest.raises(ExecutorBootstrapError, match="do not match"):
                stage_executor_artifact(run_directory, manifest_sha256)
            assert hashlib.sha256(mutated).hexdigest() != baseline_sha256
        finally:
            source.write_bytes(original)
            candidate.write_bytes(baseline)
    assert not imported
    assert {path.relative_to(copied_root).as_posix() for path in sources} == {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in executor_source_files(EXPERIMENT_DIRECTORY)
    }
