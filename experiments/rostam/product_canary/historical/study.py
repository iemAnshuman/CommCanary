"""Execute and analyze the frozen vLLM issue 2971 study on Rostam."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__:
    _contract = importlib.import_module(f"{__package__}.contract")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _contract = importlib.import_module("contract")

HistoricalStudyError = _contract.HistoricalStudyError
freeze_historical_study = _contract.freeze_historical_study
verify_frozen_historical_study = _contract.verify_frozen_historical_study

MEASUREMENT_FORMAT = "commcanary.historical_vllm_measurement.v1"
BATCH_FORMAT = "commcanary.historical_vllm_repetition.v1"
ATTEMPT_FORMAT = "commcanary.historical_vllm_attempt.v1"
SCHEDULER_FORMAT = "commcanary.historical_vllm_scheduler_observation.v1"
RESULT_FORMAT = "commcanary.historical_vllm_result.v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze the historical runtime, model, design, and scripts")
    freeze.add_argument("--study-id", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--image-lock", required=True)
    freeze.add_argument("--model-lock", required=True)
    freeze.add_argument("--study-lock", required=True)
    freeze.add_argument("--image-descriptor", required=True)
    freeze.add_argument("--model-descriptor", required=True)
    verify = subparsers.add_parser("verify", help="rehash a frozen historical study")
    verify.add_argument("--study", required=True)
    run = subparsers.add_parser("run", help="submit or resume the replicated Rostam study")
    run.add_argument("--study", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--no-wait", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    status = subparsers.add_parser("status", help="inspect immutable historical attempts")
    status.add_argument("--study", required=True)
    job = subparsers.add_parser("job", help=argparse.SUPPRESS)
    job.add_argument("--study", required=True)
    job.add_argument("--output", required=True)
    job.add_argument("--repetition", type=int, required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalStudyError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if set(value) != set(expected):
        raise HistoricalStudyError(f"{label} fields are not closed")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HistoricalStudyError(f"{label} must be a lowercase SHA-256")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalStudyError(f"{label} is missing or unsafe: {path}")
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise HistoricalStudyError(f"{label} exceeds {MAX_JSON_BYTES} bytes")

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalStudyError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalStudyError(f"{label} is not strict UTF-8 JSON") from exc
    return _object(value, label)


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HistoricalStudyError(f"historical artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    return _write_new(path, _canonical_bytes(value) + b"\n")


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != len(raw)
            or _file_sha256(path) != hashlib.sha256(raw).hexdigest()
        ):
            raise HistoricalStudyError(f"historical publication differs from recomputation: {path}")
        return
    _write_new(path, raw)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _positive_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HistoricalStudyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise HistoricalStudyError(f"{label} must be finite and positive")
    return result


def _condition_maps(manifest: Mapping[str, Any]) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    conditions = manifest.get("conditions")
    images = manifest.get("runtime_images")
    if not isinstance(conditions, list) or not isinstance(images, list):
        raise HistoricalStudyError("historical condition or image inventory is missing")
    return (
        {str(row["id"]): _object(row, "historical condition") for row in conditions},
        {str(row["id"]): _object(row, "historical image") for row in images},
    )


def validate_measurement(
    measurement: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    configuration_id: str,
) -> None:
    """Recompute all portable measurement fields and runtime bindings."""

    _closed(
        measurement,
        {
            "format",
            "measurement_id",
            "scope",
            "configuration_id",
            "image_manifest_digest",
            "sif_sha256",
            "vllm_version",
            "enforce_eager",
            "workload",
            "engine_initialization_seconds",
            "samples",
            "summary",
            "environment",
            "environment_sha256",
            "telemetry",
        },
        "historical vLLM measurement",
    )
    if measurement.get("format") != MEASUREMENT_FORMAT:
        raise HistoricalStudyError("historical vLLM measurement format is unsupported")
    expected_id = hashlib.sha256(
        _canonical_bytes({key: value for key, value in measurement.items() if key != "measurement_id"})
    ).hexdigest()
    if measurement.get("measurement_id") != expected_id:
        raise HistoricalStudyError("historical vLLM measurement identity does not recompute")
    conditions, images = _condition_maps(manifest)
    if configuration_id not in conditions or measurement.get("configuration_id") != configuration_id:
        raise HistoricalStudyError("historical measurement names the wrong configuration")
    condition = conditions[configuration_id]
    image = images[str(condition["image_id"])]
    expected_bindings = {
        "image_manifest_digest": image["manifest_digest"],
        "sif_sha256": image["sif_identity"]["sha256"],
        "vllm_version": image["versions"]["vllm"],
        "enforce_eager": condition["enforce_eager"],
        "workload": manifest["workload"],
        "scope": {
            "study": manifest["scope"],
            "application": manifest["claim_boundary"]["application"],
            "hardware": manifest["claim_boundary"]["hardware"],
            "request_stream": manifest["claim_boundary"]["execution"],
        },
    }
    for field, value in expected_bindings.items():
        if measurement.get(field) != value:
            raise HistoricalStudyError(f"historical measurement does not match frozen {field}")
    _positive_number(measurement.get("engine_initialization_seconds"), "engine initialization seconds")
    samples = measurement.get("samples")
    iterations = int(manifest["workload"]["iterations"])
    if not isinstance(samples, list) or len(samples) != iterations:
        raise HistoricalStudyError("historical measurement sample count differs from the frozen workload")
    latency: List[float] = []
    output_throughput: List[float] = []
    total_throughput: List[float] = []
    for index, raw_sample in enumerate(samples):
        sample = _object(raw_sample, f"historical sample {index}")
        _closed(
            sample,
            {
                "iteration",
                "batch_latency_ms",
                "output_token_throughput_per_second",
                "total_token_throughput_per_second",
                "output_commitment_sha256",
            },
            f"historical sample {index}",
        )
        if sample.get("iteration") != index:
            raise HistoricalStudyError("historical measurement iterations are not canonical")
        latency.append(_positive_number(sample.get("batch_latency_ms"), "batch latency"))
        output_throughput.append(
            _positive_number(sample.get("output_token_throughput_per_second"), "output throughput")
        )
        total_throughput.append(_positive_number(sample.get("total_token_throughput_per_second"), "total throughput"))
        _sha256(sample.get("output_commitment_sha256"), "historical output commitment")
    expected_summary = {
        "median_batch_latency_ms": statistics.median(latency),
        "p99_batch_latency_ms": _percentile(latency, 0.99),
        "median_output_token_throughput_per_second": statistics.median(output_throughput),
        "median_total_token_throughput_per_second": statistics.median(total_throughput),
    }
    if measurement.get("summary") != expected_summary:
        raise HistoricalStudyError("historical measurement summary does not recompute")
    environment = _object(measurement.get("environment"), "historical measurement environment")
    if measurement.get("environment_sha256") != hashlib.sha256(_canonical_bytes(environment)).hexdigest():
        raise HistoricalStudyError("historical measurement environment identity does not recompute")
    if environment.get("visible_gpu_count") != 4:
        raise HistoricalStudyError("historical measurement did not observe four GPUs")
    names = environment.get("device_names")
    if not isinstance(names, list) or len(names) != 4 or any("A100" not in str(name) for name in names):
        raise HistoricalStudyError("historical measurement did not execute on four A100 GPUs")
    if not environment.get("slurm_job_id") or not environment.get("slurm_node"):
        raise HistoricalStudyError("historical measurement lacks its SLURM allocation identity")
    telemetry = measurement.get("telemetry")
    expected_phases = [
        "before_engine_initialization",
        "before_warmup",
        "before_measured_cycle_1",
        *(f"after_measured_cycle_{index + 1}" for index in range(iterations)),
        "final",
    ]
    if (
        not isinstance(telemetry, list)
        or [row.get("phase") for row in telemetry if isinstance(row, Mapping)] != expected_phases
    ):
        raise HistoricalStudyError("historical measurement telemetry phases are incomplete")
    first_uuids: Optional[List[str]] = None
    for raw_snapshot in telemetry:
        snapshot = _object(raw_snapshot, "historical telemetry snapshot")
        _closed(snapshot, {"phase", "monotonic_ns", "command", "gpus"}, "historical telemetry snapshot")
        gpus = snapshot.get("gpus")
        if not isinstance(gpus, list) or len(gpus) != 4:
            raise HistoricalStudyError("historical telemetry snapshot does not contain four GPUs")
        uuids = [str(_object(row, "historical telemetry GPU").get("uuid")) for row in gpus]
        if first_uuids is None:
            first_uuids = uuids
        elif uuids != first_uuids:
            raise HistoricalStudyError("historical telemetry GPU UUID order changed during execution")


def _measurement_issues(measurement: Mapping[str, Any]) -> List[str]:
    issues: List[str] = []
    telemetry = measurement["telemetry"]
    measured = [
        row
        for row in telemetry
        if row["phase"] == "before_measured_cycle_1" or str(row["phase"]).startswith("after_measured_cycle_")
    ]
    first = {str(row["uuid"]): row for row in telemetry[0]["gpus"]}
    final = {str(row["uuid"]): row for row in telemetry[-1]["gpus"]}
    for snapshot in measured:
        for gpu in snapshot["gpus"]:
            uuid = str(gpu["uuid"])
            if gpu["pstate"] not in {"P0", "P2"}:
                issues.append(f"{uuid}:measured-pstate-{gpu['pstate']}")
            try:
                throttle = int(str(gpu["clock_event_reasons_active"]), 0)
            except ValueError:
                issues.append(f"{uuid}:unparsed-clock-event-reasons")
            else:
                if throttle & ~1:
                    issues.append(f"{uuid}:non-idle-clock-event-reason-{throttle:#x}")
            if float(gpu["sm_clock_mhz"]) <= 0.0 or float(gpu["memory_clock_mhz"]) <= 0.0:
                issues.append(f"{uuid}:nonpositive-clock")
    for uuid in first:
        if int(final[uuid]["corrected_volatile_ecc"]) > int(first[uuid]["corrected_volatile_ecc"]):
            issues.append(f"{uuid}:corrected-ecc-increase")
        if int(final[uuid]["uncorrected_volatile_ecc"]) > int(first[uuid]["uncorrected_volatile_ecc"]):
            issues.append(f"{uuid}:uncorrected-ecc-increase")
    return sorted(set(issues))


def _job_argv(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    attempt: Path,
    configuration_id: str,
) -> List[str]:
    conditions, images = _condition_maps(manifest)
    condition = conditions[configuration_id]
    image = images[str(condition["image_id"])]
    model_path = Path(str(manifest["model"]["model_path"]))
    sif_path = Path(str(image["sif_path"]))
    output_inside = f"/commcanary/output/{configuration_id}.json"
    argv = ["apptainer", "exec", "--cleanenv", "--nv"]
    for name in ("SLURM_JOB_ID", "SLURMD_NODENAME", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value:
            argv.extend(("--env", f"{name}={value}"))
    argv.extend(
        (
            "--bind",
            f"{model_path.resolve()}:/commcanary/model:ro",
            "--bind",
            f"{(root / 'scripts').resolve()}:/commcanary/scripts:ro",
            "--bind",
            f"{(attempt / 'measurements').resolve()}:/commcanary/output:rw",
            str(sif_path.resolve()),
            "python3",
            "-I",
            "/commcanary/scripts/driver.py",
            "--model",
            "/commcanary/model",
            "--output",
            output_inside,
            "--configuration-id",
            configuration_id,
            "--image-manifest-digest",
            str(image["manifest_digest"]),
            "--sif-sha256",
            str(image["sif_identity"]["sha256"]),
            "--expected-vllm-version",
            str(image["versions"]["vllm"]),
        )
    )
    if condition["enforce_eager"]:
        argv.append("--enforce-eager")
    for name, value in manifest["workload"].items():
        argv.extend((f"--{str(name).replace('_', '-')}", str(value)))
    return argv


def _run_job(root: Path, output: Path, repetition: int) -> int:
    manifest = verify_frozen_historical_study(root)
    repetitions = int(manifest["execution"]["repetitions"])
    if not 0 <= repetition < repetitions:
        raise HistoricalStudyError("historical repetition is outside the frozen schedule")
    if output.is_symlink() or not output.is_dir():
        raise HistoricalStudyError("historical job output must be the pre-created attempt directory")
    if (output / "batch-result.json").exists():
        raise HistoricalStudyError("historical attempt already contains a terminal batch result")
    measurements = output / "measurements"
    logs = output / "logs"
    measurements.mkdir(mode=0o700)
    logs.mkdir(mode=0o700)
    row = manifest["schedule"]["configuration_order_by_repetition"][repetition]
    started_ns = time.monotonic_ns()
    rows = []
    child_timeout = int(manifest["execution"]["allocation_timeout_seconds"]) // len(row)
    for position, configuration_id in enumerate(row):
        child_started = time.monotonic_ns()
        result_path = measurements / f"{configuration_id}.json"
        with (
            (logs / f"{position:02d}-{configuration_id}.stdout").open("xb") as stdout,
            (logs / f"{position:02d}-{configuration_id}.stderr").open("xb") as stderr,
        ):
            try:
                completed = subprocess.run(
                    _job_argv(
                        manifest,
                        root=root,
                        attempt=output,
                        configuration_id=str(configuration_id),
                    ),
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=child_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise HistoricalStudyError(
                    f"historical configuration {configuration_id} exceeded {child_timeout} seconds"
                ) from exc
        if completed.returncode != 0:
            raise HistoricalStudyError(
                f"historical configuration {configuration_id} failed with exit {completed.returncode}"
            )
        measurement = _read_json(result_path, f"historical measurement {configuration_id}")
        validate_measurement(measurement, manifest=manifest, configuration_id=str(configuration_id))
        rows.append(
            {
                "configuration_id": configuration_id,
                "planned_position": position,
                "elapsed_from_repetition_start_seconds": (child_started - started_ns) / 1_000_000_000.0,
                "measurement_id": measurement["measurement_id"],
                "path": result_path.relative_to(output).as_posix(),
                "sha256": _file_sha256(result_path),
            }
        )
    batch: Dict[str, Any] = {
        "format": BATCH_FORMAT,
        "manifest_id": manifest["manifest_id"],
        "repetition": repetition,
        "configuration_order": list(row),
        "rows": rows,
        "completed_at": _timestamp(),
    }
    batch["batch_id"] = hashlib.sha256(_canonical_bytes(batch)).hexdigest()
    _write_json(output / "batch-result.json", batch)
    return 0


def _default_runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(argv), input=stdin, check=False, capture_output=True, timeout=60)


def _bounded_output(raw: bytes, label: str) -> str:
    if len(raw) > MAX_OUTPUT_BYTES:
        raise HistoricalStudyError(f"{label} exceeded {MAX_OUTPUT_BYTES} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HistoricalStudyError(f"{label} was not UTF-8") from exc


def _next_attempt_id(owner: Path) -> str:
    if owner.is_symlink():
        raise HistoricalStudyError("historical attempt owner may not be a symlink")
    if not owner.exists():
        return "a-000001"
    if not owner.is_dir():
        raise HistoricalStudyError("historical attempt owner is not a directory")
    numbers = []
    for child in sorted(owner.iterdir(), key=lambda path: path.name):
        if child.is_symlink() or not child.is_dir():
            raise HistoricalStudyError("historical attempt owner contains an unsafe entry")
        match = _ATTEMPT_ID.fullmatch(child.name)
        if match is None:
            raise HistoricalStudyError("historical attempt owner contains an unknown entry")
        numbers.append(int(match.group(1)))
    if numbers != list(range(1, len(numbers) + 1)):
        raise HistoricalStudyError("historical attempt identifiers are not contiguous")
    return f"a-{len(numbers) + 1:06d}"


def render_wrapper(*, root: Path, attempt: Path, repetition: int) -> bytes:
    command = [
        "python3",
        "-I",
        str((root / "scripts" / "study.py").resolve()),
        "job",
        "--study",
        str(root.resolve()),
        "--output",
        str(attempt.resolve()),
        "--repetition",
        str(repetition),
    ]
    attempt_text = shlex.quote(str(attempt.resolve()))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "umask 077",
        f"test -d {shlex.quote(str(root.resolve()))}",
        f"test -d {attempt_text}",
        f"nvidia-smi --query-gpu=uuid,index,name --format=csv,noheader > {attempt_text}/gpu-pre.csv",
        f"post_gpu() {{ nvidia-smi --query-gpu=uuid,index,name --format=csv,noheader > {attempt_text}/gpu-post.csv || true; }}",
        "trap post_gpu EXIT",
        " ".join(shlex.quote(value) for value in command),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _slurm_time(seconds: int) -> str:
    if isinstance(seconds, bool) or not 1 <= seconds <= 86_400:
        raise HistoricalStudyError("historical SLURM timeout is invalid")
    hours, remainder = divmod(seconds, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{final_seconds:02d}"


def submit_repetition(
    root: Path,
    repetition: int,
    *,
    execute_acknowledged: bool,
    command_runner: CommandRunner = _default_runner,
) -> SubmittedAttempt:
    if execute_acknowledged is not True:
        raise HistoricalStudyError("historical scheduler submission requires explicit --execute")
    manifest = verify_frozen_historical_study(root)
    repetitions = int(manifest["execution"]["repetitions"])
    if not 0 <= repetition < repetitions:
        raise HistoricalStudyError("historical repetition is outside the frozen schedule")
    owner = root / "state" / "attempts" / f"repetition-{repetition:02d}"
    attempt_id = _next_attempt_id(owner)
    attempt = owner / attempt_id
    attempt.mkdir(parents=True, mode=0o700)
    wrapper = render_wrapper(root=root, attempt=attempt, repetition=repetition)
    wrapper_sha256 = hashlib.sha256(wrapper).hexdigest()
    delay = int(manifest["execution"]["repetition_day_offset_by_repetition"][repetition])
    request: Dict[str, Any] = {
        "format": ATTEMPT_FORMAT,
        "manifest_id": manifest["manifest_id"],
        "logical_id": f"repetition-{repetition:02d}",
        "attempt_id": attempt_id,
        "repetition": repetition,
        "planned_position": repetition,
        "chunk_id": f"historical-day-{delay}",
        "created_at": _timestamp(),
        "timeout_seconds": int(manifest["execution"]["allocation_timeout_seconds"]),
        "begin_delay_days": delay,
        "wrapper_sha256": wrapper_sha256,
    }
    request_sha256 = _write_json(attempt / "request.json", request)
    _write_new(attempt / "wrapper.sbatch", wrapper, mode=0o500)
    argv = [
        "sbatch",
        "--parsable",
        "--partition=cuda-A100",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:4",
        "--exclusive",
        f"--time={_slurm_time(int(manifest['execution']['allocation_timeout_seconds']))}",
        f"--job-name=cc-historical-{repetition:02d}-{attempt_id}",
        f"--output={attempt / 'slurm-%j.out'}",
        f"--error={attempt / 'slurm-%j.err'}",
    ]
    if delay:
        unit = "day" if delay == 1 else "days"
        argv.append(f"--begin=now+{delay}{unit}")
    completed = command_runner(argv, wrapper)
    stdout = _bounded_output(completed.stdout, "historical sbatch stdout")
    stderr = _bounded_output(completed.stderr, "historical sbatch stderr")
    if completed.returncode != 0:
        _write_json(
            attempt / "submission-failure.json",
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
        raise HistoricalStudyError(f"historical sbatch failed for repetition {repetition}")
    job_id = stdout.strip().split(";", 1)[0]
    if _JOB_ID.fullmatch(job_id) is None:
        _write_json(
            attempt / "submission-failure.json",
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
        raise HistoricalStudyError("historical sbatch did not return one numeric job ID")
    _write_json(
        attempt / "submission.json",
        {
            "format": ATTEMPT_FORMAT,
            "status": "submitted",
            "job_id": job_id,
            "request_sha256": request_sha256,
            "wrapper_sha256": wrapper_sha256,
            "submitted_at": _timestamp(),
            "scheduler_stderr": stderr,
        },
    )
    return SubmittedAttempt(attempt_id, attempt, job_id, request_sha256, wrapper_sha256)


def query_job(job_id: str, *, command_runner: CommandRunner = _default_runner) -> SchedulerObservation:
    if _JOB_ID.fullmatch(job_id) is None:
        raise HistoricalStudyError("historical SLURM job ID must be literal numeric text")
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
    stdout = _bounded_output(completed.stdout, "historical sacct stdout")
    stderr = _bounded_output(completed.stderr, "historical sacct stderr")
    if completed.returncode != 0:
        raise HistoricalStudyError(f"historical sacct failed with exit {completed.returncode}: {stderr}")
    rows = [line for line in stdout.splitlines() if line.strip()]
    exact = [line for line in rows if line.split("|", 1)[0] == job_id]
    if len(exact) != 1:
        raise HistoricalStudyError("historical sacct did not return one allocation row")
    fields = exact[0].split("|")
    if len(fields) < 7:
        raise HistoricalStudyError("historical sacct row is incomplete")
    state = fields[1].split()[0].rstrip("+")
    node = None if fields[3] in {"", "Unknown", "None assigned"} else fields[3]
    start = None if fields[4] in {"", "Unknown"} else fields[4]
    end = None if fields[5] in {"", "Unknown"} else fields[5]
    elapsed = None if fields[6] in {"", "Unknown"} else int(fields[6])
    return SchedulerObservation(job_id, state, fields[2], node, start, end, elapsed)


def _scheduler_record(attempt: Path, observation: SchedulerObservation) -> None:
    directory = attempt / "scheduler"
    if directory.is_symlink():
        raise HistoricalStudyError("historical scheduler directory may not be a symlink")
    existing = sorted(directory.iterdir(), key=lambda path: path.name) if directory.is_dir() else []
    expected = [f"s-{index:06d}.json" for index in range(1, len(existing) + 1)]
    if [path.name for path in existing] != expected or any(
        path.is_symlink() or not path.is_file() for path in existing
    ):
        raise HistoricalStudyError("historical scheduler observations are not a contiguous safe sequence")
    _write_json(
        directory / f"s-{len(existing) + 1:06d}.json",
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


def wait_for_terminal(
    submitted: SubmittedAttempt,
    *,
    poll_interval_seconds: int,
    maximum_wait_seconds: int = 86_400,
    command_runner: CommandRunner = _default_runner,
    sleeper: Callable[[float], None] = time.sleep,
) -> SchedulerObservation:
    terminal_path = submitted.attempt_directory / "terminal.json"
    if terminal_path.is_file() and not terminal_path.is_symlink():
        terminal = _read_json(terminal_path, "historical terminal record")
        observation = SchedulerObservation(
            submitted.job_id,
            str(terminal.get("state")),
            str(terminal.get("exit_code")),
            None if terminal.get("node") is None else str(terminal["node"]),
            None if terminal.get("start_time") is None else str(terminal["start_time"]),
            None if terminal.get("end_time") is None else str(terminal["end_time"]),
            None if terminal.get("elapsed_seconds") is None else int(terminal["elapsed_seconds"]),
        )
        if not observation.terminal:
            raise HistoricalStudyError("historical terminal record contains a nonterminal state")
        return observation
    started = time.monotonic()
    while True:
        try:
            observation = query_job(submitted.job_id, command_runner=command_runner)
        except HistoricalStudyError as exc:
            if "did not return one allocation row" not in str(exc):
                raise
            if time.monotonic() - started >= maximum_wait_seconds:
                raise HistoricalStudyError(
                    f"timed out waiting for historical job {submitted.job_id}; job was not cancelled"
                ) from exc
            sleeper(float(poll_interval_seconds))
            continue
        _scheduler_record(submitted.attempt_directory, observation)
        if observation.terminal:
            _write_json(
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
            raise HistoricalStudyError(
                f"timed out waiting for historical job {submitted.job_id}; job was not cancelled"
            )
        sleeper(float(poll_interval_seconds))


def _attempts(root: Path, repetition: int) -> List[Path]:
    owner = root / "state" / "attempts" / f"repetition-{repetition:02d}"
    if not owner.exists():
        return []
    if owner.is_symlink() or not owner.is_dir():
        raise HistoricalStudyError("historical attempt owner is unsafe")
    rows = sorted(owner.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_dir() or _ATTEMPT_ID.fullmatch(path.name) is None for path in rows):
        raise HistoricalStudyError("historical attempt inventory contains an unsafe entry")
    if [path.name for path in rows] != [f"a-{index:06d}" for index in range(1, len(rows) + 1)]:
        raise HistoricalStudyError("historical attempt inventory is not contiguous")
    return rows


def _submitted(path: Path) -> Optional[SubmittedAttempt]:
    submission_path = path / "submission.json"
    request_path = path / "request.json"
    wrapper_path = path / "wrapper.sbatch"
    if not submission_path.is_file():
        return None
    submission = _read_json(submission_path, "historical submission")
    _read_json(request_path, "historical request")
    job_id = str(submission.get("job_id"))
    if _JOB_ID.fullmatch(job_id) is None:
        raise HistoricalStudyError("historical submission job ID is malformed")
    return SubmittedAttempt(
        path.name,
        path,
        job_id,
        _file_sha256(request_path),
        _file_sha256(wrapper_path),
    )


def _successful_attempt(root: Path, repetition: int) -> Optional[Path]:
    successes = []
    for attempt in _attempts(root, repetition):
        terminal_path = attempt / "terminal.json"
        if not terminal_path.is_file():
            continue
        terminal = _read_json(terminal_path, "historical terminal record")
        if terminal.get("status") == "success":
            if not (attempt / "batch-result.json").is_file():
                raise HistoricalStudyError("successful historical attempt lacks its batch result")
            successes.append(attempt)
    if len(successes) > 1:
        raise HistoricalStudyError("historical repetition has multiple successful attempts")
    return successes[0] if successes else None


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalStudyError("historical scheduler timestamp is not ISO-8601") from exc


def _comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, float]:
    left_summary = left["summary"]
    right_summary = right["summary"]
    latency = (
        (float(right_summary["p99_batch_latency_ms"]) - float(left_summary["p99_batch_latency_ms"]))
        / float(left_summary["p99_batch_latency_ms"])
        * 100.0
    )
    throughput = (
        (
            float(left_summary["median_output_token_throughput_per_second"])
            - float(right_summary["median_output_token_throughput_per_second"])
        )
        / float(left_summary["median_output_token_throughput_per_second"])
        * 100.0
    )
    return {
        "p99_latency_regression_pct": latency,
        "output_throughput_regression_pct": throughput,
        "maximum_regression_pct": max(0.0, latency, throughput),
    }


def _aggregate(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    repetitions = int(manifest["execution"]["repetitions"])
    by_repetition: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    allocation_rows = []
    issues: List[str] = []
    days = set()
    for repetition in range(repetitions):
        attempt = _successful_attempt(root, repetition)
        if attempt is None:
            raise HistoricalStudyError("cannot aggregate an incomplete historical study")
        terminal = _read_json(attempt / "terminal.json", "historical terminal")
        node = terminal.get("node")
        start = terminal.get("start_time")
        end = terminal.get("end_time")
        elapsed = terminal.get("elapsed_seconds")
        if not isinstance(node, str) or not node or not isinstance(start, str) or not isinstance(end, str):
            issues.append(f"repetition-{repetition}:incomplete-scheduler-identity")
        else:
            start_time = _parse_time(start)
            end_time = _parse_time(end)
            if end_time < start_time:
                issues.append(f"repetition-{repetition}:scheduler-time-reversal")
            days.add(start_time.date().isoformat())
        if not isinstance(elapsed, int) or elapsed < 1:
            issues.append(f"repetition-{repetition}:invalid-elapsed-time")
        elif elapsed > int(manifest["execution"]["maximum_repetition_span_seconds"]):
            issues.append(f"repetition-{repetition}:time-window-exceeded")
        batch = _read_json(attempt / "batch-result.json", "historical repetition result")
        expected_batch_id = hashlib.sha256(
            _canonical_bytes({key: value for key, value in batch.items() if key != "batch_id"})
        ).hexdigest()
        if batch.get("batch_id") != expected_batch_id or batch.get("manifest_id") != manifest["manifest_id"]:
            raise HistoricalStudyError("historical repetition result identity does not recompute")
        expected_order = manifest["schedule"]["configuration_order_by_repetition"][repetition]
        if batch.get("repetition") != repetition or batch.get("configuration_order") != expected_order:
            raise HistoricalStudyError("historical repetition result differs from the frozen schedule")
        raw_rows = batch.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(expected_order):
            raise HistoricalStudyError("historical repetition result has an invalid row count")
        measurements: Dict[str, Mapping[str, Any]] = {}
        gpu_order: Optional[List[str]] = None
        for position, raw_row in enumerate(raw_rows):
            row = _object(raw_row, "historical repetition row")
            configuration_id = str(row.get("configuration_id"))
            if configuration_id != expected_order[position] or row.get("planned_position") != position:
                raise HistoricalStudyError("historical repetition row differs from its planned position")
            path = attempt / str(row.get("path"))
            if _file_sha256(path) != row.get("sha256"):
                raise HistoricalStudyError("historical measurement bytes changed after completion")
            measurement = _read_json(path, f"historical measurement {configuration_id}")
            validate_measurement(measurement, manifest=manifest, configuration_id=configuration_id)
            if measurement.get("measurement_id") != row.get("measurement_id"):
                raise HistoricalStudyError("historical measurement ID differs from its repetition row")
            environment = measurement["environment"]
            if environment["slurm_job_id"] != terminal.get("job_id") or environment["slurm_node"] != node:
                issues.append(f"repetition-{repetition}:{configuration_id}:scheduler-binding-mismatch")
            current_order = [str(gpu["uuid"]) for gpu in measurement["telemetry"][0]["gpus"]]
            if gpu_order is None:
                gpu_order = current_order
            elif current_order != gpu_order:
                issues.append(f"repetition-{repetition}:{configuration_id}:gpu-order-mismatch")
            issues.extend(
                f"repetition-{repetition}:{configuration_id}:{issue}" for issue in _measurement_issues(measurement)
            )
            measurements[configuration_id] = measurement
        by_repetition[repetition] = measurements
        allocation_rows.append(
            {
                "repetition": repetition,
                "attempt_id": attempt.name,
                "job_id": terminal.get("job_id"),
                "node": node,
                "start_time": start,
                "end_time": end,
                "elapsed_seconds": elapsed,
                "batch_id": batch["batch_id"],
                "configuration_order": list(expected_order),
                "measurement_ids": {
                    configuration_id: measurement["measurement_id"]
                    for configuration_id, measurement in sorted(measurements.items())
                },
            }
        )
    if len(days) < int(manifest["execution"]["minimum_distinct_days"]):
        issues.append("insufficient-distinct-allocation-days")
    reported_left, reported_right = manifest["decision"]["reported_version_comparison"]
    eager_left, eager_right = manifest["decision"]["eager_mitigation_comparison"]
    version_rows = [
        {"repetition": repetition, **_comparison(rows[reported_left], rows[reported_right])}
        for repetition, rows in sorted(by_repetition.items())
    ]
    eager_regression_rows = [
        _comparison(rows[eager_left], rows[eager_right]) for _repetition, rows in sorted(by_repetition.items())
    ]
    eager_rows = [
        {
            "repetition": repetition,
            "p99_latency_improvement_pct": -row["p99_latency_regression_pct"],
            "output_throughput_improvement_pct": -row["output_throughput_regression_pct"],
            "maximum_improvement_pct": max(
                0.0,
                -row["p99_latency_regression_pct"],
                -row["output_throughput_regression_pct"],
            ),
        }
        for (repetition, _rows), row in zip(sorted(by_repetition.items()), eager_regression_rows)
    ]
    version_median = statistics.median(row["maximum_regression_pct"] for row in version_rows)
    eager_median = statistics.median(row["maximum_improvement_pct"] for row in eager_rows)
    threshold = float(manifest["decision"]["regression_severity_threshold_pct"])
    comparable = not issues
    if not comparable:
        conclusion = "incomparable"
    elif version_median >= threshold and eager_median >= threshold:
        conclusion = "reported-version-and-eager-mitigation-pattern-observed"
    elif version_median >= threshold:
        conclusion = "version-regression-observed-without-eager-mitigation"
    elif eager_median >= threshold:
        conclusion = "eager-mitigation-observed-without-reported-version-regression"
    else:
        conclusion = "reported-pattern-not-observed"
    raw_result: Dict[str, Any] = {
        "format": RESULT_FORMAT,
        "manifest_id": manifest["manifest_id"],
        "scope": manifest["scope"],
        "claim_boundary": dict(manifest["claim_boundary"]),
        "status": "complete_comparable" if comparable else "complete_incomparable",
        "issues": sorted(set(issues)),
        "allocation_count": len(allocation_rows),
        "distinct_allocation_days": sorted(days),
        "allocations": allocation_rows,
        "reported_version_comparison": {
            "conditions": [reported_left, reported_right],
            "rows": version_rows,
            "median_maximum_regression_pct": version_median,
            "severity_threshold_pct": threshold,
            "threshold_crossed": version_median >= threshold,
        },
        "eager_mitigation_comparison": {
            "conditions": [eager_left, eager_right],
            "rows": eager_rows,
            "median_maximum_improvement_pct": eager_median,
            "severity_threshold_pct": threshold,
            "threshold_crossed": eager_median >= threshold,
        },
        "conclusion": conclusion,
        "causal_claim": "not_issued",
    }
    raw_result["result_id"] = hashlib.sha256(_canonical_bytes(raw_result)).hexdigest()
    return raw_result


def _run_study(root: Path, *, execute: bool, no_wait: bool, retry_failed: bool) -> int:
    manifest = verify_frozen_historical_study(root)
    pending: List[SubmittedAttempt] = []
    for repetition in range(int(manifest["execution"]["repetitions"])):
        if _successful_attempt(root, repetition) is not None:
            continue
        attempts = _attempts(root, repetition)
        latest = attempts[-1] if attempts else None
        submitted = _submitted(latest) if latest is not None else None
        if submitted is not None and not (submitted.attempt_directory / "terminal.json").exists():
            pending.append(submitted)
            continue
        if latest is not None and not retry_failed:
            raise HistoricalStudyError(
                f"historical repetition {repetition} failed; preserve it and pass --retry-failed for a new attempt"
            )
        if not execute:
            raise HistoricalStudyError("historical evidence is missing and scheduler execution was not acknowledged")
        pending.append(submit_repetition(root, repetition, execute_acknowledged=True))
    if no_wait:
        print(json.dumps({"manifest_id": manifest["manifest_id"], "pending_job_ids": [row.job_id for row in pending]}))
        return 0
    for submitted in pending:
        observation = wait_for_terminal(
            submitted,
            poll_interval_seconds=int(manifest["execution"]["poll_interval_seconds"]),
        )
        if not observation.successful:
            raise HistoricalStudyError(f"historical SLURM job {submitted.job_id} failed with state {observation.state}")
    result = _aggregate(root, manifest)
    _write_or_verify_json(root / "publication" / "result.json", result)
    print(
        json.dumps(
            {
                "manifest_id": manifest["manifest_id"],
                "result_id": result["result_id"],
                "conclusion": result["conclusion"],
            },
            sort_keys=True,
        )
    )
    return 0


def _status(root: Path) -> int:
    manifest = verify_frozen_historical_study(root)
    rows = []
    for repetition in range(int(manifest["execution"]["repetitions"])):
        attempts = _attempts(root, repetition)
        success = _successful_attempt(root, repetition)
        rows.append(
            {
                "repetition": repetition,
                "attempt_count": len(attempts),
                "selected_success": None if success is None else success.name,
            }
        )
    result_path = root / "publication" / "result.json"
    print(
        json.dumps(
            {
                "manifest_id": manifest["manifest_id"],
                "repetitions": rows,
                "result_id": None
                if not result_path.is_file()
                else _read_json(result_path, "historical result").get("result_id"),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        manifest = freeze_historical_study(
            study_id=args.study_id,
            output_directory=Path(args.output),
            image_lock_path=Path(args.image_lock),
            model_lock_path=Path(args.model_lock),
            study_lock_path=Path(args.study_lock),
            image_descriptor_path=Path(args.image_descriptor),
            model_descriptor_path=Path(args.model_descriptor),
        )
        print(json.dumps({"manifest_id": manifest["manifest_id"]}, sort_keys=True))
        return 0
    root = Path(args.study).resolve(strict=True)
    if args.command == "verify":
        manifest = verify_frozen_historical_study(root)
        print(json.dumps({"manifest_id": manifest["manifest_id"]}, sort_keys=True))
        return 0
    if args.command == "status":
        return _status(root)
    if args.command == "job":
        return _run_job(root, Path(args.output).resolve(strict=True), int(args.repetition))
    if args.command == "run":
        return _run_study(
            root,
            execute=bool(args.execute),
            no_wait=bool(args.no_wait),
            retry_failed=bool(args.retry_failed),
        )
    raise HistoricalStudyError("historical study command is unsupported")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SchedulerObservation",
    "SubmittedAttempt",
    "query_job",
    "render_wrapper",
    "submit_repetition",
    "validate_measurement",
    "wait_for_terminal",
]
