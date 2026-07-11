from __future__ import annotations

import ast
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import commcanary.artifacts.io as artifact_io
import commcanary.schema as legacy_schema
import commcanary.statistics as shared_statistics
from commcanary.artifacts import (
    SENSITIVE_JSON_POLICY,
    SHAREABLE_HTML_POLICY,
    AtomicWritePolicy,
    SymlinkPolicy,
    TempPlacement,
    atomic_write_bytes,
    canonical_json_bytes,
    formatted_json_bytes,
    load_schema_bytes,
)
from commcanary.errors import CommCanaryError, CommCanaryIOError, SchemaError
from commcanary.formats import format_capabilities
from commcanary.html_report import render_report_html, write_report_html
from commcanary.interop import write_param_comms_trace
from commcanary.schema import write_json

ROOT = Path(__file__).resolve().parents[1]


def _temporary_files(directory: Path, target_name: str) -> list[Path]:
    return list(directory.glob(f".{target_name}.*.tmp"))


def test_legacy_foundation_imports_are_identity_compatible() -> None:
    assert legacy_schema.CommCanaryError is CommCanaryError
    assert legacy_schema.SchemaError is SchemaError
    assert legacy_schema.canonical_json_bytes is canonical_json_bytes
    assert legacy_schema.median is shared_statistics.median
    assert legacy_schema.percentile is shared_statistics.percentile
    assert legacy_schema.percentile_from_sorted is shared_statistics.percentile_from_sorted
    assert legacy_schema.summarize_latencies is shared_statistics.summarize_latencies
    assert issubclass(CommCanaryIOError, SchemaError)


def test_statistics_preserve_legacy_interpolation_and_summary() -> None:
    assert shared_statistics.median([]) == 0.0
    assert shared_statistics.median([9.0]) == 9.0
    assert shared_statistics.percentile([30.0, 10.0, 20.0], 95.0) == pytest.approx(29.0)
    assert shared_statistics.percentile_from_sorted([10.0, 20.0, 30.0], 50.0) == 20.0
    assert shared_statistics.summarize_latencies([1.1114, 3.3336, 2.2225]) == {
        "count": 3,
        "median_us": 2.223,
        "p95_us": 3.222,
        "p99_us": 3.311,
        "max_us": 3.334,
        "mean_us": 2.223,
    }
    with pytest.raises(SchemaError, match="finite numeric"):
        shared_statistics.percentile([float("nan")], 50.0)


def test_capability_schema_loader_returns_repository_bytes_offline() -> None:
    for capability in format_capabilities():
        expected = (ROOT / capability.schema).read_bytes()
        assert load_schema_bytes(capability) == expected


def test_existing_writer_formats_are_byte_exact_and_modes_are_explicit(tmp_path: Path) -> None:
    value = {"z": "é", "a": [1, {"b": True}]}
    json_path = tmp_path / "nested" / "artifact.json"
    write_json(str(json_path), value)
    assert json_path.read_bytes() == (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

    entries = [{"comms": "all_reduce", "in_msg_size": 4, "out_msg_size": 4}]
    param_path = tmp_path / "param" / "trace.json"
    write_param_comms_trace(str(param_path), entries)
    assert param_path.read_bytes() == (json.dumps(entries, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()

    report = json.loads((ROOT / "tests/fixtures/contracts/report.valid.json").read_text(encoding="utf-8"))
    html_path = tmp_path / "html" / "report.html"
    write_report_html(str(html_path), report)
    assert html_path.read_text(encoding="utf-8") == render_report_html(report)

    if os.name != "nt":
        assert stat.S_IMODE(json_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(param_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(html_path.stat().st_mode) == 0o644


def test_json_encoding_failure_precedes_parent_creation(tmp_path: Path) -> None:
    target = tmp_path / "not-created" / "artifact.json"
    with pytest.raises(SchemaError, match="cannot encode JSON"):
        write_json(str(target), {"invalid": float("nan")})
    assert not target.parent.exists()


def test_no_overwrite_is_race_safe_and_preserves_existing_bytes(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")
    policy = replace(SENSITIVE_JSON_POLICY, overwrite=False)

    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"replacement", policy=policy)

    assert isinstance(raised.value.__cause__, FileExistsError)
    assert raised.value.operation == "inspect target"
    assert target.read_bytes() == b"original"
    assert _temporary_files(tmp_path, target.name) == []


def test_final_symlink_is_rejected_without_touching_referent(tmp_path: Path) -> None:
    referent = tmp_path / "referent.txt"
    referent.write_bytes(b"referent")
    target = tmp_path / "artifact.txt"
    try:
        target.symlink_to(referent.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(CommCanaryIOError, match="final path is a symlink"):
        atomic_write_bytes(target, b"new", policy=SENSITIVE_JSON_POLICY)

    assert target.is_symlink()
    assert referent.read_bytes() == b"referent"
    assert _temporary_files(tmp_path, target.name) == []


def test_replace_link_policy_replaces_directory_entry_not_referent(tmp_path: Path) -> None:
    referent = tmp_path / "referent.txt"
    referent.write_bytes(b"referent")
    target = tmp_path / "artifact.txt"
    try:
        target.symlink_to(referent.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    policy = replace(SENSITIVE_JSON_POLICY, symlink=SymlinkPolicy.REPLACE_LINK)

    atomic_write_bytes(target, b"new", policy=policy)

    assert not target.is_symlink()
    assert target.read_bytes() == b"new"
    assert referent.read_bytes() == b"referent"


def test_missing_parent_is_typed_when_creation_is_disabled(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "artifact.bin"
    policy = replace(SENSITIVE_JSON_POLICY, create_parents=False)
    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"new", policy=policy)
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert raised.value.operation == "prepare parent"


def test_temporary_open_failure_is_wrapped_with_original_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"

    def fail_open(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("injected open failure")

    monkeypatch.setattr(artifact_io.tempfile, "mkstemp", fail_open)
    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"new", policy=SENSITIVE_JSON_POLICY)

    assert isinstance(raised.value.__cause__, PermissionError)
    assert raised.value.operation == "create temporary file"
    assert not target.exists()
    assert _temporary_files(tmp_path, target.name) == []


class _FaultingStream:
    def __init__(self, stream: Any, failure: BaseException) -> None:
        self._stream = stream
        self._failure = failure

    def __enter__(self) -> "_FaultingStream":
        self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._stream.__exit__(*args)

    def write(self, content: bytes) -> int:
        del content
        raise self._failure

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


def _inject_write_failure(monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> None:
    real_fdopen = artifact_io.os.fdopen

    def faulting_fdopen(fd: int, mode: str) -> _FaultingStream:
        return _FaultingStream(real_fdopen(fd, mode), failure)

    monkeypatch.setattr(artifact_io.os, "fdopen", faulting_fdopen)


def test_write_failure_is_wrapped_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")
    _inject_write_failure(monkeypatch, OSError("injected write failure"))

    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"replacement", policy=SENSITIVE_JSON_POLICY)

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.operation == "write temporary file"
    assert target.read_bytes() == b"original"
    assert _temporary_files(tmp_path, target.name) == []


def test_non_os_base_exception_is_not_swallowed_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedAbort(BaseException):
        pass

    target = tmp_path / "artifact.bin"
    failure = InjectedAbort("stop")
    _inject_write_failure(monkeypatch, failure)

    with pytest.raises(InjectedAbort) as raised:
        atomic_write_bytes(target, b"new", policy=SENSITIVE_JSON_POLICY)

    assert raised.value is failure
    assert not target.exists()
    assert _temporary_files(tmp_path, target.name) == []


def test_file_fsync_failure_is_wrapped_and_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")

    def fail_fsync(fd: int) -> None:
        del fd
        raise OSError("injected fsync failure")

    monkeypatch.setattr(artifact_io.os, "fsync", fail_fsync)
    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"replacement", policy=SENSITIVE_JSON_POLICY)

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.operation == "fsync temporary file"
    assert target.read_bytes() == b"original"
    assert _temporary_files(tmp_path, target.name) == []


def test_replace_failure_is_wrapped_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")

    def fail_replace(source: str, destination: str) -> None:
        del source, destination
        raise OSError("injected replace failure")

    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)
    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"replacement", policy=SENSITIVE_JSON_POLICY)

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.operation == "install target"
    assert target.read_bytes() == b"original"
    assert _temporary_files(tmp_path, target.name) == []


@pytest.mark.skipif(os.name == "nt", reason="Windows has no portable directory fsync")
def test_parent_fsync_failure_reports_committed_target_without_temp_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"
    real_fsync = artifact_io.os.fsync
    calls = 0

    def fail_second_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(artifact_io.os, "fsync", fail_second_fsync)
    with pytest.raises(CommCanaryIOError) as raised:
        atomic_write_bytes(target, b"committed", policy=SENSITIVE_JSON_POLICY)

    assert raised.value.operation == "fsync parent directory"
    assert target.read_bytes() == b"committed"
    assert _temporary_files(tmp_path, target.name) == []


def test_foundation_dependency_dag_has_no_upward_engine_imports() -> None:
    expected = {
        "errors.py": set(),
        "formats.py": set(),
        "resources.py": set(),
        "statistics.py": {"errors"},
        "artifacts/json_codec.py": {"errors"},
        "artifacts/io.py": {"errors", "json_codec"},
        "artifacts/schemas.py": {"errors", "formats"},
    }
    package = ROOT / "src/commcanary"
    for relative, allowed in expected.items():
        tree = ast.parse((package / relative).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported == allowed, relative


def test_policy_objects_make_every_atomic_dimension_explicit() -> None:
    assert SENSITIVE_JSON_POLICY.create_parents is True
    assert SENSITIVE_JSON_POLICY.overwrite is True
    assert SENSITIVE_JSON_POLICY.mode == 0o600
    assert SENSITIVE_JSON_POLICY.flush is True
    assert SENSITIVE_JSON_POLICY.fsync_file is True
    assert SENSITIVE_JSON_POLICY.fsync_parent is True
    assert SENSITIVE_JSON_POLICY.symlink is SymlinkPolicy.REJECT
    assert SHAREABLE_HTML_POLICY.mode == 0o644


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"artifact_label": ""}, "artifact_label"),
        ({"temp_placement": "elsewhere"}, "target-parent"),
        ({"mode": True}, "POSIX permission"),
        ({"flush": False, "fsync_file": True}, "requires flush"),
    ],
)
def test_atomic_policy_rejects_incoherent_dimensions(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(SENSITIVE_JSON_POLICY, **changes)


@pytest.mark.parametrize("indent", [-1, True, "2"])
def test_formatted_json_rejects_invalid_indentation(indent: Any) -> None:
    with pytest.raises(ValueError, match="indent"):
        formatted_json_bytes({}, indent=indent)


def test_atomic_bytes_rejects_text_before_filesystem_work(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "artifact.bin"
    with pytest.raises(TypeError, match="content must be bytes"):
        atomic_write_bytes(target, "text", policy=SENSITIVE_JSON_POLICY)  # type: ignore[arg-type]
    assert not target.parent.exists()


def test_minimal_create_if_absent_policy_exercises_disabled_durability(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    policy = AtomicWritePolicy(
        artifact_label="ephemeral test artifact",
        create_parents=False,
        overwrite=False,
        temp_placement=TempPlacement.TARGET_PARENT,
        mode=0o640,
        flush=False,
        fsync_file=False,
        fsync_parent=False,
        symlink=SymlinkPolicy.REJECT,
    )

    atomic_write_bytes(target, b"created", policy=policy)

    assert target.read_bytes() == b"created"
    assert _temporary_files(tmp_path, target.name) == []


class _ShortWritingStream:
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __enter__(self) -> "_ShortWritingStream":
        self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._stream.__exit__(*args)

    def write(self, content: bytes) -> int:
        del content
        return 0

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return int(self._stream.fileno())


def test_short_write_is_typed_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"
    real_fdopen = artifact_io.os.fdopen

    def short_fdopen(fd: int, mode: str) -> _ShortWritingStream:
        return _ShortWritingStream(real_fdopen(fd, mode))

    monkeypatch.setattr(artifact_io.os, "fdopen", short_fdopen)
    with pytest.raises(CommCanaryIOError, match="short write"):
        atomic_write_bytes(target, b"not-written", policy=SENSITIVE_JSON_POLICY)
    assert not target.exists()
    assert _temporary_files(tmp_path, target.name) == []


def test_fdopen_failure_closes_descriptor_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"

    def fail_fdopen(fd: int, mode: str) -> Any:
        del fd, mode
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(artifact_io.os, "fdopen", fail_fdopen)
    with pytest.raises(CommCanaryIOError, match="fdopen failure"):
        atomic_write_bytes(target, b"new", policy=SENSITIVE_JSON_POLICY)
    assert not target.exists()
    assert _temporary_files(tmp_path, target.name) == []


def test_writer_operates_when_fchmod_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"
    monkeypatch.delattr(artifact_io.os, "fchmod", raising=False)
    atomic_write_bytes(target, b"new", policy=SENSITIVE_JSON_POLICY)
    assert target.read_bytes() == b"new"


def test_directory_fsync_is_explicitly_skipped_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_io.os, "name", "nt")
    artifact_io._fsync_parent_directory(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="directory descriptors are POSIX-only")
def test_directory_fsync_works_without_o_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(artifact_io.os, "O_DIRECTORY", raising=False)
    artifact_io._fsync_parent_directory(tmp_path)


def test_cleanup_helpers_ignore_missing_and_secondary_os_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_io._close_quietly(None)
    artifact_io._unlink_quietly(None)
    artifact_io._unlink_quietly(Path("definitely-missing-commcanary-temp"))

    def fail_close(fd: int) -> None:
        del fd
        raise OSError("secondary close failure")

    class UnlinkFailure:
        def unlink(self) -> None:
            raise OSError("secondary unlink failure")

    monkeypatch.setattr(artifact_io.os, "close", fail_close)
    artifact_io._close_quietly(123)
    artifact_io._unlink_quietly(UnlinkFailure())  # type: ignore[arg-type]
