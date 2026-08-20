from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import pytest

from experiments.rostam.product_canary import slurm_site


def _completed(
    argv: Sequence[str],
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(list(argv), returncode, stdout=stdout, stderr=stderr)


def _manifest(tmp_path: Path) -> dict[str, object]:
    sif = tmp_path / "runner.sif"
    sif.write_bytes(b"sif-bytes")
    return {
        "manifest_id": "1" * 64,
        "runner": {
            "sif_path": str(sif),
            "sif_identity": {
                "sha256": hashlib.sha256(b"sif-bytes").hexdigest(),
                "bytes": 9,
            },
            "oci_digest": f"sha256:{'2' * 64}",
        },
    }


def test_submit_job_spools_frozen_wrapper_through_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(slurm_site, "verify_frozen_product_study", lambda _path: manifest)
    calls: list[tuple[list[str], Optional[bytes]]] = []

    def runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
        calls.append((list(argv), stdin))
        return _completed(argv, stdout=b"12345;rostam\n")

    submitted = slurm_site.submit_job(
        tmp_path / "study",
        phase="application",
        logical_id="repetition-0",
        repetition=0,
        planned_position=0,
        chunk_id="application-0",
        environment={"NCCL_ALGO": "Tree", "NCCL_PROTO": None},
        job_arguments=("job", "application-repetition", "--repetition", "0"),
        timeout_seconds=7200,
        execute_acknowledged=True,
        command_runner=runner,
    )

    assert submitted.job_id == "12345"
    assert len(calls) == 1
    argv, spooled = calls[0]
    assert argv[:2] == ["sbatch", "--parsable"]
    assert "--partition=cuda-A100" in argv
    assert "--exclusive" in argv
    assert spooled == (submitted.attempt_directory / "wrapper.sbatch").read_bytes()
    assert str(submitted.attempt_directory / "wrapper.sbatch") not in argv
    wrapper = spooled.decode("utf-8")
    assert "sha256sum -c -" in wrapper
    assert "apptainer exec --cleanenv --nv" in wrapper
    assert "NCCL_ALGO=Tree" in wrapper
    request = json.loads((submitted.attempt_directory / "request.json").read_text(encoding="utf-8"))
    assert request["planned_position"] == 0
    assert request["chunk_id"] == "application-0"
    assert request["wrapper_sha256"] == hashlib.sha256(spooled).hexdigest()


def test_submit_job_requires_explicit_execute(tmp_path: Path) -> None:
    with pytest.raises(slurm_site.SlurmSiteError, match="--execute"):
        slurm_site.submit_job(
            tmp_path,
            phase="physical",
            logical_id="candidate",
            repetition=0,
            planned_position=0,
            chunk_id="physical-0",
            environment={},
            job_arguments=("job", "physical-batch"),
            timeout_seconds=60,
            execute_acknowledged=False,
        )


def test_submit_failure_is_preserved_as_an_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(slurm_site, "verify_frozen_product_study", lambda _path: manifest)

    def runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
        assert stdin is not None
        return _completed(argv, returncode=1, stderr=b"partition unavailable\n")

    with pytest.raises(slurm_site.SlurmSiteError, match="sbatch failed"):
        slurm_site.submit_job(
            tmp_path / "study",
            phase="physical",
            logical_id="candidate",
            repetition=0,
            planned_position=0,
            chunk_id="physical-0",
            environment={},
            job_arguments=("job", "physical-batch"),
            timeout_seconds=60,
            execute_acknowledged=True,
            command_runner=runner,
        )

    attempt = tmp_path / "study" / "state" / "attempts" / "physical" / "candidate" / "a-000001"
    assert (attempt / "request.json").is_file()
    assert (attempt / "wrapper.sbatch").is_file()
    assert json.loads((attempt / "submission-failure.json").read_text(encoding="utf-8"))["status"] == (
        "submission_failed"
    )


def test_query_job_uses_literal_sacct_id() -> None:
    calls: list[tuple[list[str], Optional[bytes]]] = []

    def runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
        calls.append((list(argv), stdin))
        return _completed(argv, stdout=b"8182|COMPLETED|0:0|toranj0|2026-08-05T10:00:00|2026-08-05T10:01:02|62|\n")

    observation = slurm_site.query_job("8182", command_runner=runner)

    assert observation.successful is True
    assert observation.node == "toranj0"
    assert observation.elapsed_seconds == 62
    assert calls[0][0][calls[0][0].index("-j") + 1] == "8182"
    assert calls[0][1] is None


def test_wait_for_terminal_appends_scheduler_observations(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    submitted = slurm_site.SubmittedAttempt("a-000001", attempt, "9192", "3" * 64, "4" * 64)
    outputs = iter(
        (
            b"9192|RUNNING|0:0|toranj1|2026-08-05T10:00:00|Unknown|2|\n",
            b"9192|COMPLETED|0:0|toranj1|2026-08-05T10:00:00|2026-08-05T10:00:04|4|\n",
        )
    )

    def runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
        return _completed(argv, stdout=next(outputs))

    observation = slurm_site.wait_for_terminal(
        submitted,
        poll_interval_seconds=1,
        maximum_wait_seconds=5,
        command_runner=runner,
        sleeper=lambda _seconds: None,
    )

    assert observation.successful is True
    assert sorted(path.name for path in (attempt / "scheduler").iterdir()) == [
        "s-000001.json",
        "s-000002.json",
    ]
    assert json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))["status"] == "success"
