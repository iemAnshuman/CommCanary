"""Process-identity, environment, and lifecycle behavior of the capture adapter.

Covers ``TraceRecorder.from_env``, close/save idempotency, the fork-detection
reset path (simulated in-process via a patched ``os.getpid`` rather than a
real ``os.fork``, so coverage attributes to this process), the auto-recorder
singleton in ``get_recorder``, and the small private helpers that back them.
"""

from __future__ import annotations

import os
import types
import weakref
from pathlib import Path

import pytest

import commcanary.adapters.capture as capture_module
from commcanary.capture import TraceRecorder
from commcanary.schema import SchemaError

_TRACE_ENV_VARS = (
    "COMMCANARY_TRACE_DIR",
    "COMMCANARY_TRACE_OUT",
    "COMMCANARY_WORKLOAD_NAME",
    "COMMCANARY_CAPTURE_SESSION_ID",
    "COMMCANARY_TRACE_SHARDED",
)


def _clear_trace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _TRACE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _isolate_auto_recorder_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot auto-recorder globals so a test may freely mutate them.

    ``monkeypatch.setattr`` records the value present *at call time* as the
    restore point, even when later code changes the attribute directly via a
    ``global`` statement rather than through ``setattr``. Re-setting each
    global to its own current value here is enough to guarantee teardown
    restores it, without assuming what the test will do to it.
    """

    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", capture_module._AUTO_RECORDER)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_SIGNATURE", capture_module._AUTO_RECORDER_SIGNATURE)
    monkeypatch.setattr(
        capture_module, "_AUTO_RECORDER_ATEXIT_REGISTERED", capture_module._AUTO_RECORDER_ATEXIT_REGISTERED
    )
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_LOCK", capture_module._AUTO_RECORDER_LOCK)
    monkeypatch.setattr(capture_module, "_DIRECT_OUTPUT_CLAIM_LOCK", capture_module._DIRECT_OUTPUT_CLAIM_LOCK)
    monkeypatch.setattr(capture_module, "_DIRECT_OUTPUT_CLAIMS", capture_module._DIRECT_OUTPUT_CLAIMS)


def _record_one(recorder) -> None:
    recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], rank_arrival_us={"0": 0.0})


def test_from_env_defaults_to_commcanary_trace_json_when_unset(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    recorder = TraceRecorder.from_env()
    try:
        assert recorder.output_path == "commcanary.trace.json"
    finally:
        recorder.close()


def test_from_env_prefers_configured_trace_dir(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    monkeypatch.setenv("COMMCANARY_TRACE_DIR", str(tmp_path))
    recorder = TraceRecorder.from_env()
    try:
        assert Path(recorder.output_path).parent == tmp_path.resolve()
    finally:
        recorder.close()


def test_close_after_close_is_a_silent_noop(tmp_path) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    recorder.close()
    assert recorder._closed is True
    recorder.close()
    assert recorder._closed is True


def test_save_does_not_rewrite_when_generation_already_persisted(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    original_write_json = capture_module.write_json

    def counting_write_json(path, data):
        calls.append(path)
        original_write_json(path, data)

    monkeypatch.setattr(capture_module, "write_json", counting_write_json)

    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    _record_one(recorder)
    recorder.save()
    recorder.save()

    assert len(calls) == 1


def test_direct_output_recorder_resets_and_detaches_claim_on_process_id_change(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    _record_one(recorder)
    original_output_path = recorder.output_path
    assert recorder._claim_finalizer is not None

    fake_pid = os.getpid() + 424_242
    monkeypatch.setattr(os, "getpid", lambda: fake_pid)
    _record_one(recorder)

    assert recorder._pid == fake_pid
    assert len(recorder.events) == 1
    assert recorder.output_path != original_output_path
    assert recorder._claim_finalizer is None


def test_sharded_output_recorder_reset_has_no_claim_to_detach(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    monkeypatch.setenv("COMMCANARY_TRACE_DIR", str(tmp_path))
    recorder = TraceRecorder(str(tmp_path / "unused.json"))
    assert recorder._claim_finalizer is None
    _record_one(recorder)
    original_output_path = recorder.output_path

    fake_pid = os.getpid() + 424_242
    monkeypatch.setattr(os, "getpid", lambda: fake_pid)
    _record_one(recorder)

    assert recorder._pid == fake_pid
    assert recorder._claim_finalizer is None
    assert len(recorder.events) == 1
    assert recorder.output_path != original_output_path


def test_get_recorder_returns_cached_instance_for_unchanged_signature(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    _isolate_auto_recorder_globals(monkeypatch)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_SIGNATURE", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_ATEXIT_REGISTERED", True)
    monkeypatch.setenv("COMMCANARY_TRACE_DIR", str(tmp_path))

    first = capture_module.get_recorder()
    second = capture_module.get_recorder()
    try:
        assert first is second
    finally:
        first.close()


def test_get_recorder_registers_atexit_hook_once_when_disabled(monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    _isolate_auto_recorder_globals(monkeypatch)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_SIGNATURE", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_ATEXIT_REGISTERED", False)
    registered: list = []
    monkeypatch.setattr(capture_module, "atexit", types.SimpleNamespace(register=registered.append))

    recorder = capture_module.get_recorder()

    assert isinstance(recorder, capture_module.NullRecorder)
    assert capture_module._AUTO_RECORDER_ATEXIT_REGISTERED is True
    assert registered == [capture_module._close_auto_recorder_at_exit]


def test_get_recorder_registers_atexit_hook_once_when_enabled(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    _isolate_auto_recorder_globals(monkeypatch)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_SIGNATURE", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_ATEXIT_REGISTERED", False)
    monkeypatch.setenv("COMMCANARY_TRACE_DIR", str(tmp_path))
    registered: list = []
    monkeypatch.setattr(capture_module, "atexit", types.SimpleNamespace(register=registered.append))

    recorder = capture_module.get_recorder()
    try:
        assert isinstance(recorder, TraceRecorder)
        assert capture_module._AUTO_RECORDER_ATEXIT_REGISTERED is True
        assert registered == [capture_module._close_auto_recorder_at_exit]
    finally:
        recorder.close()


def test_close_auto_recorder_at_exit_ignores_non_trace_recorder(monkeypatch) -> None:
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", None)
    capture_module._close_auto_recorder_at_exit()

    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", capture_module.NullRecorder())
    capture_module._close_auto_recorder_at_exit()


def test_close_auto_recorder_at_exit_saves_and_closes_live_recorder(tmp_path, monkeypatch) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    _record_one(recorder)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", recorder)

    capture_module._close_auto_recorder_at_exit()

    assert recorder._closed is True
    assert Path(recorder.output_path).is_file()


def test_module_record_collective_accepts_byte_count_and_forwards_it(monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    _isolate_auto_recorder_globals(monkeypatch)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_SIGNATURE", None)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER_ATEXIT_REGISTERED", True)

    # With no trace destination configured, get_recorder() resolves to a
    # NullRecorder; the call must still complete without error.
    capture_module.record_collective(op="all_reduce", ranks=[0], byte_count=16)

    assert isinstance(capture_module._AUTO_RECORDER, capture_module.NullRecorder)


def test_rank_filename_component_passes_through_the_unknown_sentinel() -> None:
    assert capture_module._rank_filename_component("unknown") == "unknown"


def test_recorder_rejects_non_mapping_workload(tmp_path) -> None:
    with pytest.raises(SchemaError, match="workload must be an object"):
        TraceRecorder(str(tmp_path / "trace.json"), workload=["not", "a", "mapping"])


def test_report_suppressed_save_error_prefers_add_note_when_present() -> None:
    workload_error = RuntimeError("workload failed")
    notes: list[str] = []
    workload_error.add_note = notes.append
    save_error = SchemaError("simulated save failure")

    capture_module._report_suppressed_save_error(workload_error, save_error)

    assert len(notes) == 1
    assert "simulated save failure" in notes[0]
    assert workload_error.commcanary_save_error is save_error


def test_reset_auto_recorder_after_fork_resets_auto_and_other_recorders(tmp_path, monkeypatch) -> None:
    _clear_trace_env(monkeypatch)
    _isolate_auto_recorder_globals(monkeypatch)

    auto_recorder = TraceRecorder(str(tmp_path / "auto.json"))
    other_recorder = TraceRecorder(str(tmp_path / "other.json"))
    _record_one(auto_recorder)
    _record_one(other_recorder)
    monkeypatch.setattr(capture_module, "_RECORDERS", weakref.WeakSet({auto_recorder, other_recorder}))
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", auto_recorder)

    capture_module._reset_auto_recorder_after_fork()

    assert auto_recorder.events == []
    assert other_recorder.events == []


def test_reset_auto_recorder_after_fork_is_a_noop_with_nothing_registered(monkeypatch) -> None:
    _isolate_auto_recorder_globals(monkeypatch)
    monkeypatch.setattr(capture_module, "_AUTO_RECORDER", None)
    monkeypatch.setattr(capture_module, "_RECORDERS", weakref.WeakSet())

    capture_module._reset_auto_recorder_after_fork()

    assert capture_module._AUTO_RECORDER_SIGNATURE == capture_module._auto_recorder_environment_signature()
