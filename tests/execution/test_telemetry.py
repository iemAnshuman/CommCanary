from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

from commcanary.execution.telemetry import capture_cycle_telemetry


def test_capture_cycle_telemetry_parses_complete_observations(monkeypatch: Any) -> None:
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-0, P0, 1200, 1215, 41, 101.5, 0x0, 0, 0\n1, GPU-1, P0, 1190, 1215, 42, 102.5, 0x0, 1, 0\n"
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout="ordinary kernel line\nNVRM: Xid (PCI:0000): 31\n",
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout="NodeName=toranj0 State=ALLOCATED ThreadsPerCore=1\n",
                stderr="",
            ),
        ]
    )
    monkeypatch.setenv("SLURMD_NODENAME", "toranj0")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    snapshot = capture_cycle_telemetry("before_warmup")

    assert snapshot["phase"] == "before_warmup"
    assert snapshot["gpu_observation"]["status"] == "complete"
    assert snapshot["gpu_observation"]["gpus"][1]["uuid"] == "GPU-1"
    assert snapshot["gpu_observation"]["gpus"][1]["corrected_volatile_ecc"] == 1
    assert snapshot["xid_observation"]["status"] == "complete"
    assert snapshot["xid_observation"]["lines"] == ["NVRM: Xid (PCI:0000): 31"]
    assert snapshot["slurm_node_observation"]["state"] == "ALLOCATED"


def test_capture_cycle_telemetry_records_unavailable_tools(monkeypatch: Any) -> None:
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(args[0], 10)
        raise OSError("journal unavailable")

    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    monkeypatch.setattr(subprocess, "run", fail)

    snapshot = capture_cycle_telemetry("final")

    assert snapshot["gpu_observation"]["status"] == "unavailable"
    assert "TimeoutExpired" in snapshot["gpu_observation"]["stderr"]
    assert snapshot["xid_observation"]["status"] == "unavailable"
    assert "journal unavailable" in snapshot["xid_observation"]["stderr"]
    assert snapshot["slurm_node_observation"]["status"] == "unavailable"
    assert calls == 2


def test_capture_cycle_telemetry_fails_closed_on_malformed_rows(monkeypatch: Any) -> None:
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="0,too,few\n", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="journal denied"),
            SimpleNamespace(returncode=0, stdout="NodeName=toranj0 State=\n", stderr=""),
        ]
    )
    monkeypatch.setenv("SLURMD_NODENAME", "toranj0")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    snapshot = capture_cycle_telemetry("cycle")

    assert snapshot["gpu_observation"]["gpus"] == []
    assert "telemetry parse error" in snapshot["gpu_observation"]["stderr"]
    assert snapshot["xid_observation"]["status"] == "unavailable"
    assert snapshot["slurm_node_observation"]["state"] is None
