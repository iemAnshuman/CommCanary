from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest  # type: ignore[import-not-found]

from experiments.rostam.lib.param_artifact import (
    PARAM_RUNTIME_INVENTORY_NAME,
    ParamArtifactError,
    prepare_param_artifact,
    stage_param_artifact,
)


def _param_tree(tmp_path: Path) -> Path:
    root = tmp_path / "param"
    replay = root / "train" / "comms" / "pt" / "commsTraceReplay.py"
    replay.parent.mkdir(parents=True)
    replay.write_text("from param_bench import helper\nprint(helper.VALUE)\n", encoding="utf-8")
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "empty.txt").write_bytes(b"")
    git = root / ".git"
    git.mkdir()
    (git / "mutable-state").write_text("ignored\n", encoding="utf-8")
    return root


def _restore_write_permissions(root: Path) -> None:
    if not root.exists():
        return
    for directory, names, filenames in os.walk(root):
        os.chmod(directory, 0o700)
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, 0o700)
        for name in filenames:
            os.chmod(Path(directory) / name, 0o600)


def test_complete_param_artifact_is_deterministic_and_privately_staged(tmp_path: Path) -> None:
    source = _param_tree(tmp_path)
    first = prepare_param_artifact(source, tmp_path / "artifacts")
    second = prepare_param_artifact(source, tmp_path / "artifacts")

    assert first == second
    assert hashlib.sha256(first.path.read_bytes()).hexdigest() == first.sha256
    private = tmp_path / "private"
    try:
        staged = stage_param_artifact(first.path, private)
        assert (staged.root / "helper.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert not (staged.root / ".git").exists()
        assert staged.import_paths == (private, private / "param")
        assert (staged.root / "helper.py").stat().st_mode & 0o222 == 0
        assert staged.root.stat().st_mode & 0o222 == 0
    finally:
        _restore_write_permissions(private)


def test_any_param_runtime_mutation_changes_the_bound_artifact(tmp_path: Path) -> None:
    source = _param_tree(tmp_path)
    before = prepare_param_artifact(source, tmp_path / "before")
    (source / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = prepare_param_artifact(source, tmp_path / "after")

    assert before.sha256 != after.sha256
    assert before.inventory_sha256 != after.inventory_sha256


def test_param_artifact_rejects_duplicate_and_unbound_entries(tmp_path: Path) -> None:
    source = _param_tree(tmp_path)
    artifact = prepare_param_artifact(source, tmp_path / "artifacts")
    with zipfile.ZipFile(artifact.path) as archive:
        inventory = json.loads(archive.read(PARAM_RUNTIME_INVENTORY_NAME))
        payloads = [(info, archive.read(info)) for info in archive.infolist()]

    forged = tmp_path / "forged.zip"
    with zipfile.ZipFile(forged, "w", compression=zipfile.ZIP_STORED) as archive:
        for info, payload in payloads:
            archive.writestr(info, payload)
        archive.writestr("unbound.py", b"raise SystemExit(1)\n")
    with pytest.raises(ParamArtifactError, match="unbound entries"):
        stage_param_artifact(forged, tmp_path / "unbound-stage")

    duplicate = tmp_path / "duplicate.zip"
    buffer = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for info, payload in payloads:
                archive.writestr(info, payload)
            replay_path = str(inventory["files"][-1]["path"])
            archive.writestr(replay_path, b"replacement\n")
    duplicate.write_bytes(buffer.getvalue())
    with pytest.raises(ParamArtifactError, match="duplicate entries"):
        stage_param_artifact(duplicate, tmp_path / "duplicate-stage")


def test_param_source_symlinks_are_refused(tmp_path: Path) -> None:
    source = _param_tree(tmp_path)
    target = source / "helper.py"
    target.unlink()
    os.symlink("empty.txt", target)

    with pytest.raises(ParamArtifactError, match="unsafe"):
        prepare_param_artifact(source, tmp_path / "artifacts")


def test_staged_param_files_preserve_only_reviewed_execute_permission(tmp_path: Path) -> None:
    source = _param_tree(tmp_path)
    replay = source / "train" / "comms" / "pt" / "commsTraceReplay.py"
    replay.chmod(replay.stat().st_mode | stat.S_IXUSR)
    artifact = prepare_param_artifact(source, tmp_path / "artifacts")
    private = tmp_path / "private"
    try:
        staged = stage_param_artifact(artifact.path, private)
        assert (staged.root / "train" / "comms" / "pt" / "commsTraceReplay.py").stat().st_mode & 0o100
    finally:
        _restore_write_permissions(private)
