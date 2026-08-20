"""Append-only SLURM boundary for the frozen Rostam product study."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

_contract = importlib.import_module(f"{__package__}.study_contract" if __package__ else "study_contract")
verify_frozen_product_study = _contract.verify_frozen_product_study

ATTEMPT_FORMAT = "commcanary.rostam.product_study_attempt.v1"
SCHEDULER_FORMAT = "commcanary.rostam.product_study_scheduler_observation.v1"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_JOB_ID = re.compile(r"^[0-9]+$")
_ATTEMPT_ID = re.compile(r"^a-([0-9]{6})$")
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)
_MAX_OUTPUT_BYTES = 256 * 1024
_BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0,'/commcanary/study/scripts');"
    "from study import main;"
    "raise SystemExit(main(sys.argv[1:]))"
)


class SlurmSiteError(RuntimeError):
    """Raised when a product-study scheduler boundary cannot be verified."""


@dataclass(frozen=True)
class SubmittedAttempt:
    attempt_id: str
    attempt_directory: Path
    job_id: str
    request_sha256: str
    wrapper_sha256: str


@dataclass(frozen=True)
class SchedulerObservation:
    job_id: str
    state: str
    exit_code: str
    node: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    elapsed_seconds: Optional[int]

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    @property
    def successful(self) -> bool:
        return self.state == "COMPLETED" and self.exit_code == "0:0"


CommandRunner = Callable[[Sequence[str], Optional[bytes]], subprocess.CompletedProcess[bytes]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SlurmSiteError(f"{label} must be a safe lowercase identifier")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value) + b"\n"
    _write_new(path, raw)
    return _sha256(raw)


def _default_runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        input=stdin,
        check=False,
        capture_output=True,
        timeout=60,
    )


def _bounded_output(raw: bytes, label: str) -> str:
    if len(raw) > _MAX_OUTPUT_BYTES:
        raise SlurmSiteError(f"{label} exceeded {_MAX_OUTPUT_BYTES} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SlurmSiteError(f"{label} was not UTF-8") from exc


def _next_attempt_id(owner_directory: Path) -> str:
    if owner_directory.is_symlink():
        raise SlurmSiteError("attempt owner directory may not be a symlink")
    if not owner_directory.exists():
        return "a-000001"
    if not owner_directory.is_dir():
        raise SlurmSiteError("attempt owner path is not a directory")
    numbers = []
    for child in sorted(owner_directory.iterdir(), key=lambda path: path.name):
        if child.is_symlink() or not child.is_dir():
            raise SlurmSiteError("attempt owner directory contains an unsafe entry")
        match = _ATTEMPT_ID.fullmatch(child.name)
        if match is None:
            raise SlurmSiteError("attempt owner directory contains an unknown entry")
        numbers.append(int(match.group(1)))
    if numbers != list(range(1, len(numbers) + 1)):
        raise SlurmSiteError("attempt IDs are not contiguous")
    return f"a-{len(numbers) + 1:06d}"


def _slurm_time(timeout_seconds: int) -> str:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 86_400:
        raise SlurmSiteError("SLURM timeout must be an integer in [1, 86400]")
    hours, remainder = divmod(timeout_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_wrapper(
    *,
    study_directory: Path,
    attempt_directory: Path,
    sif_path: Path,
    sif_sha256: str,
    environment: Mapping[str, Optional[str]],
    job_arguments: Sequence[str],
) -> bytes:
    """Render the exact wrapper bytes that will be spooled through stdin."""

    if not job_arguments or any(not isinstance(value, str) or "\x00" in value for value in job_arguments):
        raise SlurmSiteError("product study job arguments must be non-empty strings")
    if len(sif_sha256) != 64 or any(character not in "0123456789abcdef" for character in sif_sha256):
        raise SlurmSiteError("product study SIF SHA-256 is malformed")
    for name, value in environment.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", name):
            raise SlurmSiteError(f"product study environment name is unsafe: {name!r}")
        if value is not None and (
            not isinstance(value, str) or not value or any(ord(character) < 32 for character in value)
        ):
            raise SlurmSiteError(f"product study environment value is unsafe: {name!r}")

    study = shlex.quote(str(study_directory.resolve()))
    attempt = shlex.quote(str(attempt_directory.resolve()))
    digest_line = shlex.quote(f"{sif_sha256}  {sif_path.resolve()}")
    environment_argv = [f"{name}={value}" for name, value in sorted(environment.items()) if value is not None]
    command = [
        "apptainer",
        "exec",
        "--cleanenv",
        "--nv",
        "--bind",
        f"{study_directory.resolve()}:/commcanary/study:ro",
        "--bind",
        f"{attempt_directory.resolve()}:/commcanary/output:rw",
        str(sif_path.resolve()),
        "env",
        *environment_argv,
        "python3",
        "-I",
        "-c",
        _BOOTSTRAP,
        *job_arguments,
    ]
    command_text = " ".join(shlex.quote(value) for value in command)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "umask 077",
        f"test -d {study}",
        f"test -d {attempt}",
        f"printf '%s\\n' {digest_line} | sha256sum -c -",
        f"nvidia-smi --query-gpu=uuid,index,name --format=csv,noheader > {attempt}/gpu-pre.csv",
        f"post_gpu() {{ nvidia-smi --query-gpu=uuid,index,name --format=csv,noheader > {attempt}/gpu-post.csv || true; }}",
        "trap post_gpu EXIT",
        command_text,
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def submit_job(
    study_directory: Path,
    *,
    phase: str,
    logical_id: str,
    repetition: int,
    planned_position: int,
    chunk_id: str,
    environment: Mapping[str, Optional[str]],
    job_arguments: Sequence[str],
    timeout_seconds: int,
    execute_acknowledged: bool,
    begin_delay_days: int = 0,
    command_runner: CommandRunner = _default_runner,
) -> SubmittedAttempt:
    """Verify, materialize one attempt, and invoke ``sbatch`` exactly once."""

    if execute_acknowledged is not True:
        raise SlurmSiteError("scheduler submission requires the explicit --execute acknowledgement")
    phase = _safe_id(phase, "product study phase")
    logical_id = _safe_id(logical_id, "product study logical job id")
    chunk_id = _safe_id(chunk_id, "product study chunk id")
    if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0:
        raise SlurmSiteError("product study repetition must be a non-negative integer")
    if isinstance(planned_position, bool) or not isinstance(planned_position, int) or planned_position < 0:
        raise SlurmSiteError("product study planned position must be a non-negative integer")
    if isinstance(begin_delay_days, bool) or not isinstance(begin_delay_days, int) or not 0 <= begin_delay_days <= 30:
        raise SlurmSiteError("product study begin delay must be an integer in [0, 30] days")
    slurm_timeout = _slurm_time(timeout_seconds)

    root = study_directory.resolve()
    manifest = verify_frozen_product_study(root)
    runner = manifest["runner"]
    sif_path = Path(str(runner["sif_path"]))
    sif_sha256 = str(runner["sif_identity"]["sha256"])
    owner = root / "state" / "attempts" / phase / logical_id
    attempt_id = _next_attempt_id(owner)
    attempt_directory = owner / attempt_id

    bound_environment = dict(environment)
    bound_environment["COMMCANARY_RUNNER_OCI_DIGEST"] = str(runner["oci_digest"])
    wrapper = render_wrapper(
        study_directory=root,
        attempt_directory=attempt_directory,
        sif_path=sif_path,
        sif_sha256=sif_sha256,
        environment=bound_environment,
        job_arguments=job_arguments,
    )
    attempt_directory.mkdir(parents=True, mode=0o700)
    wrapper_sha256 = _sha256(wrapper)
    request: Dict[str, Any] = {
        "format": ATTEMPT_FORMAT,
        "manifest_id": manifest["manifest_id"],
        "phase": phase,
        "logical_id": logical_id,
        "attempt_id": attempt_id,
        "repetition": repetition,
        "planned_position": planned_position,
        "chunk_id": chunk_id,
        "created_at": _timestamp(),
        "environment": bound_environment,
        "job_arguments": list(job_arguments),
        "timeout_seconds": timeout_seconds,
        "begin_delay_days": begin_delay_days,
        "wrapper_sha256": wrapper_sha256,
    }
    request_sha256 = _write_new_json(attempt_directory / "request.json", request)
    _write_new(attempt_directory / "wrapper.sbatch", wrapper, mode=0o500)

    job_name = f"cc-{phase[:24]}-{logical_id[:48]}-{attempt_id}"
    argv = [
        "sbatch",
        "--parsable",
        "--partition=cuda-A100",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:4",
        "--exclusive",
        f"--time={slurm_timeout}",
        f"--job-name={job_name}",
        f"--output={attempt_directory / 'slurm-%j.out'}",
        f"--error={attempt_directory / 'slurm-%j.err'}",
    ]
    if begin_delay_days:
        unit = "day" if begin_delay_days == 1 else "days"
        argv.append(f"--begin=now+{begin_delay_days}{unit}")
    completed = command_runner(argv, wrapper)
    stdout = _bounded_output(completed.stdout, "sbatch stdout")
    stderr = _bounded_output(completed.stderr, "sbatch stderr")
    if completed.returncode != 0:
        _write_new_json(
            attempt_directory / "submission-failure.json",
            {
                "format": ATTEMPT_FORMAT,
                "status": "submission_failed",
                "request_sha256": request_sha256,
                "wrapper_sha256": wrapper_sha256,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "observed_at": _timestamp(),
            },
        )
        raise SlurmSiteError(f"sbatch failed for {logical_id!r} with exit {completed.returncode}")
    parsed = stdout.strip().split(";", 1)[0]
    if _JOB_ID.fullmatch(parsed) is None:
        _write_new_json(
            attempt_directory / "submission-failure.json",
            {
                "format": ATTEMPT_FORMAT,
                "status": "invalid_scheduler_response",
                "request_sha256": request_sha256,
                "wrapper_sha256": wrapper_sha256,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "observed_at": _timestamp(),
            },
        )
        raise SlurmSiteError("sbatch did not return one literal numeric job ID")
    _write_new_json(
        attempt_directory / "submission.json",
        {
            "format": ATTEMPT_FORMAT,
            "status": "submitted",
            "job_id": parsed,
            "request_sha256": request_sha256,
            "wrapper_sha256": wrapper_sha256,
            "submitted_at": _timestamp(),
            "scheduler_stderr": stderr,
        },
    )
    return SubmittedAttempt(
        attempt_id=attempt_id,
        attempt_directory=attempt_directory,
        job_id=parsed,
        request_sha256=request_sha256,
        wrapper_sha256=wrapper_sha256,
    )


def query_job(job_id: str, *, command_runner: CommandRunner = _default_runner) -> SchedulerObservation:
    """Read one literal job from accounting without mutating scheduler state."""

    if _JOB_ID.fullmatch(job_id) is None:
        raise SlurmSiteError("SLURM job ID must be a literal numeric identifier")
    argv = [
        "sacct",
        "-X",
        "-j",
        job_id,
        "--noheader",
        "--parsable2",
        "--starttime=1970-01-01",
        "--format=JobIDRaw,State,ExitCode,NodeList,Start,End,ElapsedRaw",
    ]
    completed = command_runner(argv, None)
    stdout = _bounded_output(completed.stdout, "sacct stdout")
    stderr = _bounded_output(completed.stderr, "sacct stderr")
    if completed.returncode != 0:
        raise SlurmSiteError(f"sacct failed for job {job_id}: {stderr.strip()}")
    rows = [line for line in stdout.splitlines() if line.strip()]
    parsed_rows = [line.split("|") for line in rows]
    exact = [row for row in parsed_rows if len(row) >= 7 and row[0] == job_id]
    if len(exact) != 1:
        raise SlurmSiteError(f"sacct did not return one allocation row for job {job_id}")
    row = exact[0]
    state = row[1].split()[0].rstrip("+")
    elapsed = None
    if row[6] not in {"", "Unknown"}:
        try:
            elapsed = int(row[6])
        except ValueError as exc:
            raise SlurmSiteError("sacct ElapsedRaw was not an integer") from exc
    return SchedulerObservation(
        job_id=job_id,
        state=state,
        exit_code=row[2],
        node=None if row[3] in {"", "None assigned", "Unknown"} else row[3],
        start_time=None if row[4] in {"", "Unknown"} else row[4],
        end_time=None if row[5] in {"", "Unknown"} else row[5],
        elapsed_seconds=elapsed,
    )


def record_scheduler_observation(attempt_directory: Path, observation: SchedulerObservation) -> Path:
    """Append one immutable scheduler observation to an attempt."""

    scheduler_directory = attempt_directory / "scheduler"
    if scheduler_directory.is_symlink():
        raise SlurmSiteError("scheduler observation directory may not be a symlink")
    existing = sorted(scheduler_directory.iterdir(), key=lambda path: path.name) if scheduler_directory.is_dir() else []
    if any(path.is_symlink() or not path.is_file() for path in existing):
        raise SlurmSiteError("scheduler observation directory contains an unsafe entry")
    expected_names = [f"s-{index:06d}.json" for index in range(1, len(existing) + 1)]
    if [path.name for path in existing] != expected_names:
        raise SlurmSiteError("scheduler observation sequence is not contiguous")
    path = scheduler_directory / f"s-{len(existing) + 1:06d}.json"
    _write_new_json(
        path,
        {
            "format": SCHEDULER_FORMAT,
            "observed_at": _timestamp(),
            "job_id": observation.job_id,
            "state": observation.state,
            "exit_code": observation.exit_code,
            "node": observation.node,
            "start_time": observation.start_time,
            "end_time": observation.end_time,
            "elapsed_seconds": observation.elapsed_seconds,
            "terminal": observation.terminal,
            "successful": observation.successful,
        },
    )
    return path


def wait_for_terminal(
    submitted: SubmittedAttempt,
    *,
    poll_interval_seconds: int,
    maximum_wait_seconds: int,
    command_runner: CommandRunner = _default_runner,
    sleeper: Callable[[float], None] = time.sleep,
) -> SchedulerObservation:
    """Poll accounting and persist every state until one terminal row appears."""

    if not 1 <= poll_interval_seconds <= 60:
        raise SlurmSiteError("SLURM poll interval must be in [1, 60] seconds")
    if maximum_wait_seconds < poll_interval_seconds:
        raise SlurmSiteError("SLURM maximum wait must cover at least one poll interval")
    terminal_path = submitted.attempt_directory / "terminal.json"
    if terminal_path.is_file() and not terminal_path.is_symlink():
        with terminal_path.open("rb") as handle:
            raw = handle.read(_MAX_OUTPUT_BYTES + 1)
        if len(raw) > _MAX_OUTPUT_BYTES:
            raise SlurmSiteError("existing SLURM terminal record is too large")
        try:
            terminal = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SlurmSiteError("existing SLURM terminal record is invalid") from exc
        if not isinstance(terminal, Mapping) or terminal.get("job_id") != submitted.job_id:
            raise SlurmSiteError("existing SLURM terminal record names the wrong job")
        observation = SchedulerObservation(
            job_id=submitted.job_id,
            state=str(terminal.get("state")),
            exit_code=str(terminal.get("exit_code")),
            node=None if terminal.get("node") is None else str(terminal.get("node")),
            start_time=None if terminal.get("start_time") is None else str(terminal.get("start_time")),
            end_time=None if terminal.get("end_time") is None else str(terminal.get("end_time")),
            elapsed_seconds=(None if terminal.get("elapsed_seconds") is None else int(terminal["elapsed_seconds"])),
        )
        if not observation.terminal:
            raise SlurmSiteError("existing SLURM terminal record contains a nonterminal state")
        return observation
    started = time.monotonic()
    while True:
        try:
            observation = query_job(submitted.job_id, command_runner=command_runner)
        except SlurmSiteError as exc:
            if "did not return one allocation row" not in str(exc):
                raise
            if time.monotonic() - started >= maximum_wait_seconds:
                raise SlurmSiteError(
                    f"timed out waiting for SLURM job {submitted.job_id}; job was not cancelled"
                ) from exc
            sleeper(float(poll_interval_seconds))
            continue
        record_scheduler_observation(submitted.attempt_directory, observation)
        if observation.terminal:
            _write_new_json(
                terminal_path,
                {
                    "format": ATTEMPT_FORMAT,
                    "status": "success" if observation.successful else "failed",
                    "job_id": observation.job_id,
                    "state": observation.state,
                    "exit_code": observation.exit_code,
                    "node": observation.node,
                    "start_time": observation.start_time,
                    "end_time": observation.end_time,
                    "elapsed_seconds": observation.elapsed_seconds,
                    "observed_at": _timestamp(),
                },
            )
            return observation
        if time.monotonic() - started >= maximum_wait_seconds:
            raise SlurmSiteError(f"timed out waiting for SLURM job {submitted.job_id}; job was not cancelled")
        sleeper(float(poll_interval_seconds))


__all__ = [
    "ATTEMPT_FORMAT",
    "SCHEDULER_FORMAT",
    "SchedulerObservation",
    "SlurmSiteError",
    "SubmittedAttempt",
    "query_job",
    "record_scheduler_observation",
    "render_wrapper",
    "submit_job",
    "wait_for_terminal",
]
