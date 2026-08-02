from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_two_process_gloo_conforms_to_injective_correctness_oracle() -> None:
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("the installed PyTorch build has no Gloo backend")
    repository_root = Path(__file__).resolve().parents[2]
    worker = Path(__file__).with_name("_gloo_correctness_worker.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gloo correctness oracle passed" in completed.stdout
