from __future__ import annotations

import hashlib
import json
import os
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
    ExecutorArtifactError,
    executor_schema_files,
    executor_source_files,
    load_executor_artifact,
    prepare_executor_artifact,
    render_executor_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "rostam"


def _copy_executor_sources(tmp_path: Path) -> Path:
    copied_root = tmp_path / "copy"
    copied_experiment = copied_root / "experiments" / "rostam"
    for source in (*executor_source_files(EXPERIMENT_DIRECTORY), *executor_schema_files(EXPERIMENT_DIRECTORY)):
        destination = copied_root / source.relative_to(REPOSITORY_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return copied_experiment


def _run_directory(tmp_path: Path, artifact: Path, digest: str, size: int) -> tuple[Path, str]:
    run_directory = tmp_path / "run"
    run_directory.mkdir(parents=True, exist_ok=True)
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


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _archive_members(path: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    return members, json.loads(members[EXECUTOR_INVENTORY_NAME])


def _write_archive(
    path: Path,
    members: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
    duplicate: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        if duplicate is not None:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(duplicate, members[duplicate])


def _bind_and_reject_archive(tmp_path: Path, candidate: Path, match: str) -> None:
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    run_directory, manifest_sha256 = _run_directory(tmp_path, candidate, digest, candidate.stat().st_size)
    with pytest.raises(ExecutorBootstrapError, match=match):
        stage_executor_artifact(run_directory, manifest_sha256)
    with pytest.raises(ExecutorArtifactError, match=match):
        load_executor_artifact(candidate)


def test_executor_artifact_is_deterministic_and_inventories_every_package_source(tmp_path: Path) -> None:
    first = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")
    second = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")

    assert first == second
    assert first.sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()
    expected = {
        "__main__.py",
        *(path.relative_to(REPOSITORY_ROOT).as_posix() for path in executor_source_files(EXPERIMENT_DIRECTORY)),
    }
    expected_schemas = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in executor_schema_files(EXPERIMENT_DIRECTORY)
    }
    assert set(first.source_files) == expected
    assert set(first.schema_files) == expected_schemas
    assert load_executor_artifact(first.path) == first
    with zipfile.ZipFile(first.path) as archive:
        inventory = json.loads(archive.read(EXECUTOR_INVENTORY_NAME))
        assert inventory["schema"] == EXECUTOR_ARTIFACT_SCHEMA
        assert {item["path"] for item in inventory["source_files"]} == expected
        assert {item["path"] for item in inventory["schema_files"]} == expected_schemas
        assert archive.read("__main__.py").startswith(b"from experiments.rostam.lib.executor_cli")


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
        environment = dict(os.environ)
        environment["COMMCANARY_EXECUTOR_PATH"] = str(staged.path)
        environment["COMMCANARY_EXECUTOR_SHA256"] = staged.sha256
        analysis_help = subprocess.run(
            [sys.executable, "-I", "-S", str(staged.path), "analyze", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert analysis_help.returncode == 0
        assert "Validate a frozen manifest" in analysis_help.stdout
        evaluator_help = subprocess.run(
            [sys.executable, "-I", "-S", str(staged.path), "evaluate-decision-gate", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert evaluator_help.returncode == 0
        assert "Evaluate one trusted physical decision-gate aggregate" in evaluator_help.stdout
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


def test_executor_rejects_honestly_inventoried_modified_main(tmp_path: Path) -> None:
    artifact = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")
    members, inventory = _archive_members(artifact.path)
    members["__main__.py"] = b'raise SystemExit("modified entry point")\n'
    source_rows = inventory["source_files"]
    assert isinstance(source_rows, list)
    main_row = next(row for row in source_rows if row["path"] == "__main__.py")
    main_row.update(
        {
            "sha256": hashlib.sha256(members["__main__.py"]).hexdigest(),
            "size_bytes": len(members["__main__.py"]),
        }
    )
    inventory["source_inventory_sha256"] = hashlib.sha256(_canonical_json(source_rows)).hexdigest()
    members[EXECUTOR_INVENTORY_NAME] = _canonical_json(inventory)
    candidate = tmp_path / "modified-main.pyz"
    _write_archive(candidate, members)

    _bind_and_reject_archive(tmp_path / "modified-main", candidate, "__main__")


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("unlisted-python", "unlisted|member set|members"),
        ("duplicate-name", "duplicated"),
        ("compressed", "ZIP_STORED"),
        ("extra-schema", "unlisted|member set|members"),
    ),
)
def test_executor_rejects_nonclosed_zip_member_policies(tmp_path: Path, case: str, match: str) -> None:
    artifact = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")
    members, _inventory = _archive_members(artifact.path)
    candidate = tmp_path / f"{case}.pyz"
    if case == "unlisted-python":
        members["experiments/rostam/unlisted.py"] = b"RAISED = True\n"
        _write_archive(candidate, members)
    elif case == "duplicate-name":
        _write_archive(candidate, members, duplicate="__main__.py")
    elif case == "compressed":
        _write_archive(candidate, members, compression=zipfile.ZIP_DEFLATED)
    else:
        members["experiments/rostam/schemas/unlisted.schema.json"] = b"{}\n"
        _write_archive(candidate, members)

    _bind_and_reject_archive(tmp_path / case, candidate, match)


def test_executor_rejects_encrypted_member_flag_before_reading_payload(tmp_path: Path) -> None:
    artifact = prepare_executor_artifact(EXPERIMENT_DIRECTORY, tmp_path / "artifacts")
    raw = bytearray(artifact.path.read_bytes())
    central = raw.find(b"PK\x01\x02")
    assert central >= 0
    flags = int.from_bytes(raw[central + 8 : central + 10], "little") | 0x1
    raw[central + 8 : central + 10] = flags.to_bytes(2, "little")
    candidate = tmp_path / "encrypted-flag.pyz"
    candidate.write_bytes(raw)

    _bind_and_reject_archive(tmp_path / "encrypted", candidate, "unencrypted")
