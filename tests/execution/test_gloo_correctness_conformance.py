from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_four_process_gloo_conforms_to_injective_correctness_oracle() -> None:
    required = os.environ.get("COMMCANARY_REQUIRE_GLOO") == "1"
    try:
        import torch
    except ImportError:
        if required:
            pytest.fail("the required PyTorch verification dependency is not installed")
        pytest.skip("PyTorch is not installed")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        if required:
            pytest.fail("the required PyTorch build has no Gloo backend")
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
            "--nproc_per_node=4",
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
