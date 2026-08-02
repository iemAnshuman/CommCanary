from __future__ import annotations

import ctypes
import hashlib
import importlib
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest  # type: ignore[import-not-found]

from experiments.rostam.lib import cell_entrypoint
from experiments.rostam.lib.executor_artifact import prepare_executor_artifact


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


def test_runtime_environment_exposes_only_staged_experiment_modules(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    experiment_directory = Path(cell_entrypoint.__file__).resolve().parents[1]
    executor = prepare_executor_artifact(experiment_directory, tmp_path / "executors")
    monkeypatch.setenv("PYTHONPATH", "/unreviewed/inherited/path")
    configuration = SimpleNamespace(environment=SimpleNamespace(to_value=lambda: {}))

    environment = cell_entrypoint._runtime_environment(
        configuration,
        experiment_directory,
        tmp_path / "nccl" / "libnccl.so.2",
        executor.path,
    )

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(executor.path),
        str(experiment_directory / "third_party"),
        str(experiment_directory / "third_party" / "param"),
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "spec = importlib.util.find_spec('experiments.rostam.qualification_physical'); "
                "raise SystemExit(spec is None)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def test_bound_input_staging_survives_same_size_source_replacement(tmp_path: Path) -> None:
    source = tmp_path / "bound.whl"
    original = b"wheel-a"
    replacement = b"wheel-b"
    source.write_bytes(original)
    staging = tmp_path / "private-inputs"
    staging.mkdir(mode=0o700)

    staged = cell_entrypoint._stage_bound_input(
        source,
        input_id="decision-gate-wheel",
        expected_size=len(original),
        expected_sha256=hashlib.sha256(original).hexdigest(),
        staging_directory=staging,
    )
    newer = tmp_path / "newer.whl"
    newer.write_bytes(replacement)
    os.replace(newer, source)

    assert staged.suffix == ".whl"
    assert staged.read_bytes() == original
    assert staged.stat().st_mode & 0o222 == 0


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
        "0, GPU-a, NVIDIA A100-PCIE-40GB, 550.54.15, 00000000:01:00.0, Enabled, P0, 55, 120.5, 250.0, 1410, 1215\n"
        "1, GPU-b, NVIDIA A100-PCIE-40GB, 550.54.15, 00000000:02:00.0, Enabled, P2, 57, 118.0, 250.0, 1395, 1215\n"
    )
    observed_commands: list[tuple[str, ...]] = []

    def probe(command: Sequence[str], **_kwargs: Any) -> str:
        normalized = tuple(command)
        observed_commands.append(normalized)
        if normalized[0] == "scontrol":
            return "NodeName=toranj0 State=ALLOCATED CPULoad=1.25\n"
        if normalized[1:3] == ("topo", "-m"):
            return "GPU0 GPU1\nGPU0 X PIX\nGPU1 PIX X\n"
        return inventory

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
    assert evidence["schema"] == "commcanary.rostam.runtime-observation.v2"
    assert evidence["driver_version"] == "550.54.15"
    assert evidence["gpu_count"] == 2
    assert evidence["gpus"][0]["uuid"] == "GPU-a"
    assert evidence["gpus"][0]["sm_clock_mhz"] == 1410
    assert evidence["gpus"][1]["memory_clock_mhz"] == 1215
    assert evidence["gpus"][0]["persistence_mode"] == "Enabled"
    assert evidence["gpus"][0]["performance_state"] == "P0"
    assert evidence["gpus"][0]["temperature_c"] == 55
    assert evidence["gpus"][0]["power_draw_w"] == 120.5
    assert evidence["gpus"][0]["power_limit_w"] == 250.0
    assert evidence["topology"]["text"].startswith("GPU0 GPU1")
    assert evidence["node_state"]["text"].startswith("NodeName=toranj0 State=ALLOCATED")
    assert evidence["binding"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert evidence["binding"]["cpu_affinity"] == [0, 2, 4]
    assert len(observed_commands) == 3


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


def test_repetition_placeholder_resolves_to_the_manifest_cell_index(tmp_path: Path) -> None:
    resolved = cell_entrypoint._resolve_argument(
        "{repetition}",
        repetition=7,
        workspace=tmp_path / "workspace",
        experiment_directory=tmp_path / "experiment",
        venv_directory=tmp_path / "venv",
        dependency_paths={},
        input_paths={},
    )

    assert resolved == "7"
