from __future__ import annotations

import ctypes
import importlib
import io
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest  # type: ignore[import-not-found]

from experiments.rostam.lib import cell_entrypoint


class _CompletedProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", return_code: int = 0) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.return_code = return_code
        self.pid = 424242

    def poll(self) -> int:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.return_code

    def terminate(self) -> None:
        self.return_code = -15

    def kill(self) -> None:
        self.return_code = -9


def _popen_with_output(stdout: bytes, stderr: bytes = b"", return_code: int = 0) -> Any:
    def spawn(_command: Sequence[str], **_kwargs: Any) -> _CompletedProcess:
        return _CompletedProcess(stdout, stderr, return_code)

    return spawn


def test_pipeline_catches_and_truncates_a_final_stdout_burst(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _popen_with_output(b"x" * 4096),
    )
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    return_code, _elapsed, reason, exceeded = cell_entrypoint._run_pipeline(
        (("mock-producer",),),
        workspace=tmp_path,
        environment={},
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=5,
        max_output_bytes=512,
    )

    assert return_code == cell_entrypoint._OUTPUT_LIMIT_EXIT_CODE
    assert exceeded is True
    assert reason == "stdout or stderr exceeded 512 bytes"
    assert stdout_path.stat().st_size == 512
    assert stderr_path.stat().st_size <= 512


def test_pipeline_keeps_successful_output_bounded_across_steps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls = iter((b"first\n", b"second\n"))

    def spawn(_command: Sequence[str], **_kwargs: Any) -> _CompletedProcess:
        return _CompletedProcess(next(calls))

    monkeypatch.setattr(subprocess, "Popen", spawn)
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    return_code, _elapsed, reason, exceeded = cell_entrypoint._run_pipeline(
        (("first",), ("second",)),
        workspace=tmp_path,
        environment={},
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=5,
        max_output_bytes=1024,
    )

    assert (return_code, reason, exceeded) == (0, None, False)
    assert stdout_path.read_bytes() == b"first\nsecond\n"
    assert stderr_path.read_bytes().count(b"[commcanary physical step") == 2


def test_pipeline_final_check_truncates_direct_log_path_writes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    def spawn(_command: Sequence[str], **_kwargs: Any) -> _CompletedProcess:
        stdout_path.write_bytes(b"direct" * 1024)
        return _CompletedProcess(b"")

    monkeypatch.setattr(subprocess, "Popen", spawn)
    return_code, _elapsed, reason, exceeded = cell_entrypoint._run_pipeline(
        (("direct-writer",),),
        workspace=tmp_path,
        environment={},
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=5,
        max_output_bytes=256,
    )

    assert return_code == cell_entrypoint._OUTPUT_LIMIT_EXIT_CODE
    assert exceeded is True
    assert reason == "stdout or stderr exceeded 256 bytes"
    assert stdout_path.stat().st_size == 256


def test_runtime_probe_enforces_its_memory_cap_with_mocked_process(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _popen_with_output(b"x" * 32),
    )
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="exceeded 8 bytes per stream"):
        cell_entrypoint._run_bounded_probe(("mock-probe",), max_output_bytes=8)


def test_runtime_probe_normalizes_nonzero_exit_without_exposing_stderr(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _popen_with_output(b"", b"private scheduler detail", return_code=9),
    )
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="mock-probe.*exited 9") as captured:
        cell_entrypoint._run_bounded_probe(("mock-probe",))
    assert "private scheduler detail" not in str(captured.value)


def test_runtime_fingerprint_records_driver_gpu_topology_binding_and_clocks(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    torch = SimpleNamespace(
        __version__="2.4.1+cu121",
        version=SimpleNamespace(cuda="12.1"),
    )

    class Library:
        @staticmethod
        def ncclGetVersion(pointer: Any) -> int:
            pointer._obj.value = 22005
            return 0

    inventory = (
        "0, GPU-a, NVIDIA A100-PCIE-40GB, 550.54.15, 00000000:01:00.0, Enabled, 1410, 1215\n"
        "1, GPU-b, NVIDIA A100-PCIE-40GB, 550.54.15, 00000000:02:00.0, Enabled, 1395, 1215\n"
    )
    observed_commands: list[tuple[str, ...]] = []

    def probe(command: Sequence[str], **_kwargs: Any) -> str:
        normalized = tuple(command)
        observed_commands.append(normalized)
        return "GPU0 GPU1\nGPU0 X PIX\nGPU1 PIX X\n" if normalized[1:3] == ("topo", "-m") else inventory

    monkeypatch.setattr(importlib, "import_module", lambda _name: torch)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: Library())
    monkeypatch.setattr(cell_entrypoint, "_run_bounded_probe", probe)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: {0, 2, 4},
        raising=False,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("SLURM_LOCALID", "0")

    runtime, evidence = cell_entrypoint._runtime_fingerprint(
        tmp_path / "libnccl.so.2",
        {"hostname": "toranj0", "job_id": "12345"},
    )

    assert runtime["torch_version"] == "2.4.1"
    assert runtime["torch_cuda_version"] == "12.1"
    assert runtime["runtime_nccl_version_code"] == 22005
    assert evidence["driver_version"] == "550.54.15"
    assert evidence["gpu_count"] == 2
    assert evidence["gpus"][0]["uuid"] == "GPU-a"
    assert evidence["gpus"][0]["sm_clock_mhz"] == 1410
    assert evidence["gpus"][1]["memory_clock_mhz"] == 1215
    assert evidence["gpus"][0]["persistence_mode"] == "Enabled"
    assert evidence["topology"]["text"].startswith("GPU0 GPU1")
    assert evidence["binding"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert evidence["binding"]["cpu_affinity"] == [0, 2, 4]
    assert len(observed_commands) == 2


def test_runtime_fingerprint_rejects_malformed_probe_rows_without_running_a_probe(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    torch = SimpleNamespace(__version__="2.4.1", version=SimpleNamespace(cuda="12.1"))

    class Library:
        @staticmethod
        def ncclGetVersion(pointer: Any) -> int:
            pointer._obj.value = 22005
            return 0

    monkeypatch.setattr(importlib, "import_module", lambda _name: torch)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: Library())
    monkeypatch.setattr(cell_entrypoint, "_run_bounded_probe", lambda _command: "0, too-few-fields\n")

    with pytest.raises(cell_entrypoint.CellEntrypointError, match="has 2 fields"):
        cell_entrypoint._runtime_fingerprint(
            tmp_path / "libnccl.so.2",
            {"hostname": "toranj0", "job_id": "12345"},
        )


def test_runtime_fingerprint_normalizes_nccl_load_failure(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    torch = SimpleNamespace(__version__="2.4.1", version=SimpleNamespace(cuda="12.1"))
    monkeypatch.setattr(importlib, "import_module", lambda _name: torch)

    def fail_load(_path: str) -> Any:
        raise OSError("private filesystem detail")

    monkeypatch.setattr(ctypes, "CDLL", fail_load)
    with pytest.raises(cell_entrypoint.CellEntrypointError, match="cannot load.*NCCL") as captured:
        cell_entrypoint._runtime_fingerprint(
            tmp_path / "private" / "libnccl.so.2",
            {"hostname": "toranj0", "job_id": "12345"},
        )
    assert "private filesystem detail" not in str(captured.value)
