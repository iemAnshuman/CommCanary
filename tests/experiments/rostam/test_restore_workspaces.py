from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Optional

import pytest  # type: ignore[import-not-found]

from experiments.rostam import restore_workspaces as tool

RUN_ID = "demo-20260101-r1"
CELL = "c-demo-r000000-0000000000000000"


def _campaign(tmp_path: Path, files: Dict[str, bytes], extra: Optional[Dict[str, bytes]] = None) -> Dict[str, Path]:
    """A run directory whose selected attempt references ``files``, plus its
    raw archive and descriptor, shaped like the committed Rostam campaigns."""

    run = tmp_path / "results" / RUN_ID
    attempt = run / "attempts" / CELL / "a-000001"
    attempt.mkdir(parents=True)
    references = [
        {"path": f"workspaces/{CELL}/a-000001/{name}", "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    ]
    (attempt / "attempt.json").write_text(json.dumps({"artifacts": references}), encoding="utf-8")
    selection = run / "selections" / "primary"
    selection.mkdir(parents=True)
    (selection / "selection.json").write_text(
        json.dumps({"entries": [{"cell_id": CELL, "attempt_id": "a-000001"}]}), encoding="utf-8"
    )
    archive = tmp_path / "archive.raw.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in {**files, **(extra or {})}.items():
            info = tarfile.TarInfo(f"{RUN_ID}/workspaces/{CELL}/a-000001/{name}")
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    descriptor = tmp_path / "archive.raw-archive-descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": tool.DESCRIPTOR_SCHEMA,
                "campaigns": [{"run_id": RUN_ID}],
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "size_bytes": archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    return {"run": run, "archive": archive, "descriptor": descriptor}


def test_restores_exactly_the_referenced_files(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, {"result.json": b'{"ok": true}', "stdout.log": b"line\n"}, {"unused.log": b"x" * 10})
    restored = tool.restore(paths["run"], paths["archive"], paths["descriptor"])
    workspace = paths["run"] / "workspaces" / CELL / "a-000001"
    assert len(restored) == 2
    assert (workspace / "result.json").read_bytes() == b'{"ok": true}'
    assert not (workspace / "unused.log").exists()
    # A second run finds the files already in place and restores nothing.
    assert tool.restore(paths["run"], paths["archive"], paths["descriptor"]) == []


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, {"result.json": b"{}"})
    assert len(tool.restore(paths["run"], paths["archive"], paths["descriptor"], dry_run=True)) == 1
    assert not (paths["run"] / "workspaces").exists()


def test_an_archive_that_does_not_match_its_descriptor_is_refused(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, {"result.json": b"{}"})
    descriptor = json.loads(paths["descriptor"].read_text(encoding="utf-8"))
    descriptor["sha256"] = "0" * 64
    paths["descriptor"].write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(tool.RestoreError, match="SHA-256"):
        tool.restore(paths["run"], paths["archive"], paths["descriptor"])


def test_an_lfs_pointer_in_place_of_the_archive_says_to_pull(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, {"result.json": b"{}"})
    paths["archive"].write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(tool.RestoreError, match="git lfs pull"):
        tool.restore(paths["run"], paths["archive"], paths["descriptor"])


def test_a_file_whose_bytes_differ_from_its_attempt_record_is_refused(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, {"result.json": b"{}"})
    attempt = paths["run"] / "attempts" / CELL / "a-000001" / "attempt.json"
    record = json.loads(attempt.read_text(encoding="utf-8"))
    record["artifacts"][0]["sha256"] = "f" * 64
    attempt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(tool.RestoreError, match="attempt recorded"):
        tool.restore(paths["run"], paths["archive"], paths["descriptor"])
    assert not (paths["run"] / "workspaces" / CELL / "a-000001" / "result.json").exists()


def test_a_referenced_file_missing_from_the_archive_is_refused(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, {"result.json": b"{}"})
    attempt = paths["run"] / "attempts" / CELL / "a-000001" / "attempt.json"
    record = json.loads(attempt.read_text(encoding="utf-8"))
    record["artifacts"].append({"path": f"workspaces/{CELL}/a-000001/absent.log", "sha256": "0" * 64})
    attempt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(tool.RestoreError, match="lacks 1 referenced"):
        tool.restore(paths["run"], paths["archive"], paths["descriptor"])


def test_members_outside_the_workspace_tree_are_never_extracted(tmp_path: Path) -> None:
    escape = f"{RUN_ID}/workspaces/../../escaped.txt"
    assert tool._safe_relative(escape, RUN_ID) is None
    assert tool._safe_relative(f"{RUN_ID}/attempts/x.json", RUN_ID) is None
    assert tool._safe_relative(f"other-run/workspaces/{CELL}/a.log", RUN_ID) is None
    assert tool._safe_relative(f"{RUN_ID}/workspaces/{CELL}/a.log", RUN_ID) == f"workspaces/{CELL}/a.log"


def test_too_little_free_disk_is_refused_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoring a Rostam campaign takes several GiB. Running out of disk half
    way leaves a machine without room to work, so the check comes first."""

    paths = _campaign(tmp_path, {"result.json": b"{}"})

    class Usage:
        free = 1024

    monkeypatch.setattr(tool.shutil, "disk_usage", lambda _path: Usage())
    with pytest.raises(tool.RestoreError, match="GiB is free"):
        tool.restore(paths["run"], paths["archive"], paths["descriptor"])
    assert not (paths["run"] / "workspaces").exists()


def test_command_line_reports_failure_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _campaign(tmp_path, {"result.json": b"{}"})
    paths["archive"].write_bytes(b"not an archive")
    code = tool.main(
        [
            "--run-directory",
            str(paths["run"]),
            "--archive",
            str(paths["archive"]),
            "--descriptor",
            str(paths["descriptor"]),
        ]
    )
    assert code == 1
    assert "restore failed" in capsys.readouterr().err
