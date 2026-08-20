from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Mapping

import pytest

from commcanary.artifacts.application_measurement import (
    APPLICATION_PERTURBATION_ENVIRONMENT_NAMES,
    application_subject_configuration,
    application_subject_sha256,
    validate_application_measurement,
)
from commcanary.artifacts.json_codec import canonical_json_bytes
from commcanary.product import application_driver, sglang_application_driver

RUNNER_DIGEST = f"sha256:{hashlib.sha256(b'runner').hexdigest()}"


def _telemetry_factory(world_size: int) -> Any:
    timestamp = 0

    def capture(phase: str) -> Dict[str, Any]:
        nonlocal timestamp
        timestamp += 1
        xid_lines: list[str] = []
        return {
            "phase": phase,
            "monotonic_ns": timestamp,
            "gpu_observation": {
                "status": "complete",
                "command": ["nvidia-smi"],
                "returncode": 0,
                "gpus": [
                    {
                        "index": rank,
                        "uuid": f"GPU-{rank}",
                        "pstate": "P0",
                        "sm_clock_mhz": 1200.0,
                        "memory_clock_mhz": 1215.0,
                        "temperature_c": 40.0,
                        "power_w": 100.0,
                        "clock_event_reasons_active": "0x0",
                        "corrected_volatile_ecc": 0,
                        "uncorrected_volatile_ecc": 0,
                    }
                    for rank in range(world_size)
                ],
                "stderr": "",
            },
            "xid_observation": {
                "status": "complete",
                "command": ["journalctl"],
                "lines": xid_lines,
                "sha256": hashlib.sha256(canonical_json_bytes(xid_lines)).hexdigest(),
                "stderr": "",
            },
            "slurm_node_observation": {
                "status": "complete",
                "command": ["scontrol"],
                "node": "toranj0",
                "state": "ALLOCATED",
                "stderr": "",
            },
        }

    return capture


def _fake_torch(world_size: int) -> ModuleType:
    module = ModuleType("torch")
    module.__version__ = "2.11.0"  # type: ignore[attr-defined]
    module.version = SimpleNamespace(cuda="13.0")  # type: ignore[attr-defined]
    module.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        nccl=SimpleNamespace(version=lambda: (2, 28, 9)),
        device_count=lambda: world_size,
    )
    return module


def _perturbation_environment() -> Dict[str, None]:
    return {name: None for name in APPLICATION_PERTURBATION_ENVIRONMENT_NAMES}


def _subject(
    *,
    engine: str,
    version: str,
    engine_configuration: Mapping[str, Any],
) -> str:
    return application_subject_sha256(
        application_subject_configuration(
            runner_oci_digest=RUNNER_DIGEST,
            application_engine=engine,
            application_engine_version=version,
            engine_configuration=engine_configuration,
            perturbation_environment=_perturbation_environment(),
        )
    )


def _prepare_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*APPLICATION_PERTURBATION_ENVIRONMENT_NAMES, "COMMCANARY_RUNNER_OCI_DIGEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "toranj0")
    monkeypatch.setenv("COMMCANARY_RUNNER_OCI_DIGEST", RUNNER_DIGEST)


def test_vllm_application_driver_emits_valid_profiled_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_environment(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "result" / "application.json"
    profile = tmp_path / "profile"
    engine_configuration = {
        "kind": "vllm.v1",
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "gpu_memory_utilization": 0.5,
        "kv_cache_memory_bytes": 1024,
        "max_num_batched_tokens": 8,
        "max_num_seqs": 2,
    }
    subject = _subject(engine="vllm", version="9.9.9", engine_configuration=engine_configuration)

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def generate(self, prompts: list[dict[str, Any]], sampling_params: Any, *, use_tqdm: bool) -> list[Any]:
            assert use_tqdm is False
            return [
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(range(sampling_params.max_tokens)))])
                for _ in prompts
            ]

        def start_profile(self, *, profile_prefix: str) -> None:
            assert profile_prefix == "commcanary"
            Path(self.kwargs["profiler_config"]["torch_profiler_dir"], "trace.json").write_text(
                "{}\n", encoding="utf-8"
            )

        def stop_profile(self) -> None:
            return None

    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "9.9.9"  # type: ignore[attr-defined]
    fake_vllm.LLM = LLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(2))
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(application_driver, "capture_cycle_telemetry", _telemetry_factory(2))
    monkeypatch.setattr(application_driver, "_observed_gpu_memory_used_bytes", lambda count: 4096)

    result = application_driver.main(
        [
            "--model",
            str(model),
            "--output",
            str(output),
            "--profile-dir",
            str(profile),
            "--runner-oci-digest",
            RUNNER_DIGEST,
            "--subject-sha256",
            subject,
            "--perturbation-id",
            "baseline",
            "--tensor-parallel-size",
            "2",
            "--batch-size",
            "2",
            "--input-length",
            "4",
            "--output-length",
            "3",
            "--warmups",
            "1",
            "--iterations",
            "2",
            "--gpu-memory-utilization",
            "0.5",
            "--kv-cache-memory-bytes",
            "1024",
            "--max-num-batched-tokens",
            "8",
            "--max-num-seqs",
            "2",
            "--skip-gpu-health-check",
        ]
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert evidence["application"]["engine"] == "vllm"
    assert evidence["profile"]["captured"] is True
    assert evidence["telemetry_assessment"]["comparable"] is True
    validate_application_measurement(evidence)


def test_sglang_application_driver_emits_valid_profiled_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_environment(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "result" / "application.json"
    profile = tmp_path / "profile"
    engine_configuration = {
        "kind": "sglang.v1",
        "disable_cuda_graph": True,
        "disable_custom_all_reduce": True,
        "disable_overlap_schedule": True,
        "mem_fraction_static": 0.5,
        "max_running_requests": 2,
        "max_total_tokens": 8,
        "chunked_prefill_size": 8,
        "max_prefill_tokens": 8,
    }
    subject = _subject(engine="sglang", version="8.8.8", engine_configuration=engine_configuration)
    engines: list[Any] = []

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.shutdown_called = False
            engines.append(self)

        def generate(self, **kwargs: Any) -> list[Dict[str, Any]]:
            output_length = int(kwargs["sampling_params"]["max_new_tokens"])
            return [
                {
                    "output_ids": list(range(output_length)),
                    "meta_info": {
                        "prompt_tokens": len(tokens),
                        "completion_tokens": output_length,
                        "cached_tokens": 0,
                        "finish_reason": {"type": "length", "length": output_length},
                    },
                }
                for tokens in kwargs["input_ids"]
            ]

        def start_profile(self, *, output_dir: str) -> None:
            Path(output_dir, "trace.json").write_text("{}\n", encoding="utf-8")

        def stop_profile(self) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_called = True

    fake_sglang = ModuleType("sglang")
    fake_sglang.Engine = Engine  # type: ignore[attr-defined]
    fake_version = ModuleType("sglang.version")
    fake_version.__version__ = "8.8.8"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(2))
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.version", fake_version)
    monkeypatch.setattr(sglang_application_driver.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_application_driver, "capture_cycle_telemetry", _telemetry_factory(2))
    monkeypatch.setattr(sglang_application_driver, "_observed_gpu_memory_used_bytes", lambda count: 8192)

    result = sglang_application_driver.main(
        [
            "--model",
            str(model),
            "--output",
            str(output),
            "--profile-dir",
            str(profile),
            "--runner-oci-digest",
            RUNNER_DIGEST,
            "--subject-sha256",
            subject,
            "--perturbation-id",
            "baseline",
            "--tensor-parallel-size",
            "2",
            "--batch-size",
            "2",
            "--input-length",
            "4",
            "--output-length",
            "3",
            "--warmups",
            "1",
            "--iterations",
            "2",
            "--mem-fraction-static",
            "0.5",
            "--max-running-requests",
            "2",
            "--max-total-tokens",
            "8",
            "--chunked-prefill-size",
            "8",
            "--max-prefill-tokens",
            "8",
            "--disable-overlap-schedule",
            "--skip-gpu-health-check",
        ]
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert engines[0].shutdown_called is True
    assert evidence["application"]["engine"] == "sglang"
    assert evidence["profile"]["captured"] is True
    assert evidence["telemetry_assessment"]["comparable"] is True
    validate_application_measurement(evidence)


def test_vllm_memory_observation_and_generation_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="10\n20\n", stderr=""),
    )
    assert application_driver._observed_gpu_memory_used_bytes(2) == 20 * 1024 * 1024

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="no gpu"),
    )
    with pytest.raises(RuntimeError, match="memory observation failed"):
        application_driver._observed_gpu_memory_used_bytes(2)

    class WrongLLM:
        def generate(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    with pytest.raises(RuntimeError, match="returned 0 outputs"):
        application_driver._run_generation(
            WrongLLM(),
            SimpleNamespace(max_tokens=1),
            batch_size=1,
            input_length=1,
            output_length=1,
            seed=1,
            iteration=0,
        )


def test_vllm_driver_helpers_reject_invalid_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        application_driver._require_positive(0, "batch-size")
    with pytest.raises(ValueError, match="no regular files"):
        application_driver._model_commitment(tmp_path)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        application_driver._sha256("A" * 64, "digest")
    with pytest.raises(ValueError, match="must use sha256"):
        application_driver._oci_digest("not-a-digest")
    with pytest.raises(ValueError, match="empty sequence"):
        application_driver._percentile([], 0.99)
    assert application_driver._percentile([3.0], 0.99) == 3.0

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="10\n", stderr=""),
    )
    with pytest.raises(RuntimeError, match="returned 1 GPUs"):
        application_driver._observed_gpu_memory_used_bytes(2)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="bad\nbad\n", stderr=""),
    )
    with pytest.raises(RuntimeError, match="invalid memory.used"):
        application_driver._observed_gpu_memory_used_bytes(2)

    class WrongTokensLLM:
        def generate(self, *args: Any, **kwargs: Any) -> list[Any]:
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[])])]

    with pytest.raises(RuntimeError, match="returned 0 tokens"):
        application_driver._run_generation(
            WrongTokensLLM(),
            SimpleNamespace(max_tokens=1),
            batch_size=1,
            input_length=1,
            output_length=1,
            seed=1,
            iteration=0,
        )


def test_vllm_generation_rejects_invalid_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    class FastLLM:
        def generate(self, *args: Any, **kwargs: Any) -> list[Any]:
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1])])]

    monkeypatch.setattr(application_driver.time, "perf_counter_ns", lambda: 0)
    with pytest.raises(RuntimeError, match="invalid elapsed time"):
        application_driver._run_generation(
            FastLLM(),
            SimpleNamespace(max_tokens=1),
            batch_size=1,
            input_length=1,
            output_length=1,
            seed=1,
            iteration=0,
        )


def test_gpu_health_check_refuses_a_node_that_needs_a_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state that made toranj1 fail every job for sixteen days.

    A wedged GPU keeps answering NVML, so `nvidia-smi` succeeds and only CUDA
    context creation fails. Slurm does not drain the node either, so work keeps
    landing on it. The driver must refuse rather than add to the pile.
    """

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, P0\n1, [GPU requires reset]\n2, P0\n3, [GPU requires reset]\n",
            stderr="",
        )

    monkeypatch.setattr(application_driver.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="need an administrator reset"):
        application_driver.require_healthy_gpus(4)


def test_gpu_health_check_accepts_healthy_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="0, P0\n1, P0\n2, P0\n3, P0\n", stderr="")

    monkeypatch.setattr(application_driver.subprocess, "run", fake_run)
    assert len(application_driver.require_healthy_gpus(4)) == 4


def test_gpu_health_check_refuses_an_unexpected_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="0, P0\n1, P0\n", stderr="")

    monkeypatch.setattr(application_driver.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="reported 2 GPUs; expected 4"):
        application_driver.require_healthy_gpus(4)


def test_watchdog_terminates_a_run_that_outlives_its_budget() -> None:
    """The path that only runs when the engine is already hung.

    A driver that cannot be made to stop is how GPUs end up held by a dead run,
    so the timer must actually fire and actually signal this process.
    """

    signalled: List[int] = []
    original = signal.signal(signal.SIGTERM, lambda *_: signalled.append(1))
    try:
        timer = application_driver._start_watchdog(0.05, "unit test")
        deadline = time.monotonic() + 5.0
        while not signalled and time.monotonic() < deadline:
            time.sleep(0.01)
        timer.cancel()
    finally:
        signal.signal(signal.SIGTERM, original)
    assert signalled, "the watchdog did not signal the process"


def test_watchdog_does_not_fire_when_cancelled_in_time() -> None:
    signalled: List[int] = []
    original = signal.signal(signal.SIGTERM, lambda *_: signalled.append(1))
    try:
        application_driver._start_watchdog(30.0, "unit test").cancel()
        time.sleep(0.1)
    finally:
        signal.signal(signal.SIGTERM, original)
    assert not signalled


def test_engine_teardown_survives_a_failing_shutdown_hook() -> None:
    """Teardown must not mask the error that caused it to run."""

    calls: List[str] = []

    class _Executor:
        def shutdown(self) -> None:
            calls.append("executor")
            raise RuntimeError("executor refused to stop")

    class _Engine:
        model_executor = _Executor()

        def shutdown(self) -> None:
            calls.append("engine")

    class _Distributed:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def destroy_process_group() -> None:
            calls.append("process_group")

    class _Cuda:
        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

    torch = SimpleNamespace(distributed=_Distributed(), cuda=_Cuda())
    application_driver._shutdown_engine(SimpleNamespace(llm_engine=_Engine()), torch)

    # Every stage is attempted even though the first one raised.
    assert calls == ["executor", "engine", "process_group", "empty_cache"]
