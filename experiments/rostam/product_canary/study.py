"""Freeze, execute, resume, and verify the Rostam product-canary study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from commcanary.artifacts.application_measurement import (
    APPLICATION_PERTURBATION_ENVIRONMENT_NAMES,
    validate_application_measurement,
)
from commcanary.artifacts.application_oracle import (
    build_application_oracle,
    validate_application_oracle,
)
from commcanary.artifacts.chakra import decode_chakra_execution_trace, encode_chakra_subgraph
from commcanary.artifacts.json_codec import canonical_json_bytes
from commcanary.artifacts.physical_canary import (
    validate_chakra_projection,
    validate_physical_canary_policy,
)
from commcanary.artifacts.physical_execution import validate_physical_execution_measurement
from commcanary.resources import decode_bounded_json_bytes
from commcanary.services.active_physical_synthesis import (
    HoldoutApplicationBatch,
    PhysicalCandidateRequest,
    synthesize_active_physical_canary,
)
from commcanary.workflows.physical_canary import (
    build_physical_canary_bundle,
    verify_physical_canary_bundle,
)

_package = __package__
_contract = importlib.import_module(f"{_package}.study_contract" if _package else "study_contract")
_site = importlib.import_module(f"{_package}.slurm_site" if _package else "slurm_site")
ProductStudyError = _contract.ProductStudyError
freeze_product_study = _contract.freeze_product_study
subject_map = _contract.subject_map
verify_frozen_product_study = _contract.verify_frozen_product_study
SubmittedAttempt = _site.SubmittedAttempt
submit_job = _site.submit_job
wait_for_terminal = _site.wait_for_terminal

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_TRACE_BYTES = 64 * 1024 * 1024
CANDIDATE_FORMAT = "commcanary.rostam.product_study_candidate.v1"
BATCH_FORMAT = "commcanary.rostam.product_study_batch_result.v1"
SELECTION_FORMAT = "commcanary.rostam.product_study_attempt_selection.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze all study and runtime inputs")
    freeze.add_argument("--study-id", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--source-et", required=True)
    freeze.add_argument("--projection", required=True)
    freeze.add_argument("--policy", required=True)
    freeze.add_argument("--perturbations", required=True)
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--runner-descriptor", required=True)
    freeze.add_argument("--runner-sif", required=True)
    freeze.add_argument("--wheel", required=True)

    verify = subparsers.add_parser("verify", help="recompute a frozen study")
    verify.add_argument("--study", required=True)

    run = subparsers.add_parser("run", help="run or resume the frozen study")
    run.add_argument("--study", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--no-wait", action="store_true")
    run.add_argument("--retry-failed", action="store_true")

    status = subparsers.add_parser("status", help="report immutable local attempt state")
    status.add_argument("--study", required=True)

    job = subparsers.add_parser("job", help=argparse.SUPPRESS)
    job_subparsers = job.add_subparsers(dest="job_command", required=True)
    application = job_subparsers.add_parser("application-repetition")
    application.add_argument("--study", required=True)
    application.add_argument("--output", required=True)
    application.add_argument("--split", choices=("training", "holdout"), required=True)
    application.add_argument("--repetition", type=int, required=True)
    physical = job_subparsers.add_parser("physical-batch")
    physical.add_argument("--study", required=True)
    physical.add_argument("--output", required=True)
    physical.add_argument("--candidate-id", required=True)
    physical.add_argument("--scope", choices=("training", "holdout", "all"), required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductStudyError(f"{label} is missing or unsafe: {path}")
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ProductStudyError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = decode_bounded_json_bytes(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProductStudyError(f"{label} is not bounded strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ProductStudyError(f"{label} must be an object")
    return value


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    _write_or_verify_bytes(path, raw)


def _write_or_verify_bytes(path: Path, raw: bytes) -> None:
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != len(raw)
            or _file_sha256(path) != hashlib.sha256(raw).hexdigest()
        ):
            raise ProductStudyError(f"immutable study bytes disagree with recomputation: {path}")
        return
    _write_new(path, raw)


def _bounded_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ProductStudyError(f"{label} is missing or unsafe: {path}")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ProductStudyError(f"{label} exceeds {maximum} bytes")
    return raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan(manifest: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    plan = _read_json(root / "inputs" / "perturbations.json", "frozen perturbation plan")
    engine = str(manifest["runner"]["application_engine"])
    validated = _contract.validate_perturbation_plan(plan, engine=engine)
    if not isinstance(validated, Mapping):
        raise ProductStudyError("validated perturbation plan is not an object")
    return validated


def _perturbations(plan: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    training = plan.get("training")
    holdout = plan.get("holdout")
    if not isinstance(training, list) or not isinstance(holdout, list):
        raise ProductStudyError("validated perturbation arrays disappeared")
    rows = [plan["baseline"], *training, *holdout]
    return {str(row["id"]): row for row in rows}


def _child_environment(perturbation: Mapping[str, Any], runner_digest: str) -> Dict[str, str]:
    environment = dict(os.environ)
    for name in APPLICATION_PERTURBATION_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    raw_values = perturbation.get("environment")
    if not isinstance(raw_values, Mapping):
        raise ProductStudyError("perturbation environment is missing")
    for name in APPLICATION_PERTURBATION_ENVIRONMENT_NAMES:
        value = raw_values[name]
        if value is not None:
            environment[name] = str(value)
    environment["COMMCANARY_RUNNER_OCI_DIGEST"] = runner_digest
    return environment


def _application_argv(
    manifest: Mapping[str, Any],
    subject: Mapping[str, Any],
    *,
    model_path: Path,
    output_path: Path,
    profile_path: Optional[Path],
) -> List[str]:
    application = manifest["application"]
    workload = application["workload_arguments"]
    configuration = subject["subject_configuration"]["engine_configuration"]
    argv = [
        sys.executable,
        "-I",
        "-m",
        str(application["driver_module"]),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--runner-oci-digest",
        str(manifest["runner"]["oci_digest"]),
        "--subject-sha256",
        str(subject["subject_sha256"]),
        "--perturbation-id",
        str(subject["perturbation_id"]),
    ]
    if profile_path is not None:
        argv.extend(("--profile-dir", str(profile_path)))
    for key in (
        "tensor_parallel_size",
        "batch_size",
        "input_length",
        "output_length",
        "warmups",
        "iterations",
        "seed",
    ):
        argv.extend((f"--{key.replace('_', '-')}", str(workload[key])))
    engine = manifest["runner"]["application_engine"]
    if engine == "vllm":
        for key in (
            "gpu_memory_utilization",
            "kv_cache_memory_bytes",
            "max_num_batched_tokens",
            "max_num_seqs",
        ):
            argv.extend((f"--{key.replace('_', '-')}", str(configuration[key])))
        for key in ("disable_custom_all_reduce", "enforce_eager"):
            if configuration[key]:
                argv.append(f"--{key.replace('_', '-')}")
    elif engine == "sglang":
        for key in (
            "mem_fraction_static",
            "max_running_requests",
            "max_total_tokens",
            "chunked_prefill_size",
            "max_prefill_tokens",
        ):
            argv.extend((f"--{key.replace('_', '-')}", str(configuration[key])))
        for key in ("disable_cuda_graph", "disable_custom_all_reduce", "disable_overlap_schedule"):
            if configuration[key]:
                argv.append(f"--{key.replace('_', '-')}")
    else:
        raise ProductStudyError("application engine is unsupported")
    return argv


def _run_child(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> None:
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            completed = subprocess.run(
                list(argv),
                check=False,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProductStudyError(f"child process exceeded {timeout_seconds} seconds") from exc
    if completed.returncode != 0:
        raise ProductStudyError(f"child process failed with exit {completed.returncode}: {argv[0]}")


def _job_application_repetition(root: Path, output: Path, *, split: str, repetition: int) -> int:
    manifest = verify_frozen_product_study(root)
    repetitions = int(manifest["execution"]["application_repetitions"])
    if not 0 <= repetition < repetitions:
        raise ProductStudyError("application repetition is outside the frozen schedule")
    schedule_name = f"{split}_configuration_order_by_repetition"
    schedule = manifest["schedule"][schedule_name]
    row = schedule[repetition]
    plan = _plan(manifest, root)
    perturbation_by_id = _perturbations(plan)
    subjects = subject_map(manifest)
    if output.is_symlink() or not output.is_dir():
        raise ProductStudyError("application job output must be the pre-created attempt directory")
    if (output / "batch-result.json").exists():
        raise ProductStudyError("application attempt already contains a terminal batch result")
    measurement_directory = output / "measurements"
    measurement_directory.mkdir()
    log_directory = output / "logs"
    log_directory.mkdir()
    profile_parent = output / "profiles"
    model_path = root / "inputs" / "model"
    started_ns = time.monotonic_ns()
    rows = []
    for position, perturbation_id in enumerate(row):
        subject = subjects[perturbation_id]
        result_path = measurement_directory / f"{perturbation_id}.json"
        profile_path = None
        if split == "training" and repetition == 0 and perturbation_id == "baseline":
            profile_parent.mkdir(exist_ok=True)
            profile_path = profile_parent / "baseline"
        child_started = time.monotonic_ns()
        _run_child(
            _application_argv(
                manifest,
                subject,
                model_path=model_path,
                output_path=result_path,
                profile_path=profile_path,
            ),
            environment=_child_environment(
                perturbation_by_id[perturbation_id],
                str(manifest["runner"]["oci_digest"]),
            ),
            stdout_path=log_directory / f"{position:02d}-{perturbation_id}.stdout",
            stderr_path=log_directory / f"{position:02d}-{perturbation_id}.stderr",
            timeout_seconds=int(manifest["execution"]["application_timeout_seconds"]),
        )
        measurement = _read_json(result_path, f"application measurement {perturbation_id}")
        validate_application_measurement(measurement)
        if measurement["subject_sha256"] != subject["subject_sha256"]:
            raise ProductStudyError("application measurement subject does not match the frozen schedule")
        rows.append(
            {
                "perturbation_id": perturbation_id,
                "planned_position": position,
                "elapsed_from_repetition_start_seconds": (child_started - started_ns) / 1_000_000_000.0,
                "measurement_id": measurement["measurement_id"],
                "path": result_path.relative_to(output).as_posix(),
                "sha256": _file_sha256(result_path),
            }
        )
    batch = {
        "format": BATCH_FORMAT,
        "kind": "application_repetition",
        "manifest_id": manifest["manifest_id"],
        "split": split,
        "repetition": repetition,
        "configuration_order": list(row),
        "rows": rows,
        "completed_at": _timestamp(),
    }
    batch["batch_id"] = hashlib.sha256(canonical_json_bytes(batch)).hexdigest()
    _write_new(output / "batch-result.json", _canonical_bytes(batch) + b"\n")
    return 0


def _candidate_record(root: Path, candidate_id: str) -> Tuple[Mapping[str, Any], Path]:
    directory = root / "state" / "candidates" / candidate_id
    record = _read_json(directory / "candidate.json", "physical candidate record")
    expected_fields = {
        "format",
        "candidate_id",
        "method",
        "selected_region_ids",
        "selected_node_ids",
        "executable_sha256",
        "role",
    }
    if set(record) != expected_fields or record.get("format") != CANDIDATE_FORMAT:
        raise ProductStudyError("physical candidate record fields are not closed")
    executable = directory / "executable.et"
    if record.get("candidate_id") != candidate_id or _file_sha256(executable) != record.get("executable_sha256"):
        raise ProductStudyError("physical candidate bytes do not match their record")
    expected_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "method": record["method"],
                "selected_region_ids": record["selected_region_ids"],
                "executable_sha256": record["executable_sha256"],
            }
        )
    ).hexdigest()
    if expected_id != candidate_id:
        raise ProductStudyError("physical candidate_id does not recompute")
    return record, executable


def _physical_argv(
    manifest: Mapping[str, Any],
    subject: Mapping[str, Any],
    candidate: Mapping[str, Any],
    executable: Path,
    *,
    root: Path,
    output_path: Path,
) -> List[str]:
    execution = manifest["execution"]
    argv = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "commcanary.execution.physical_runner",
        "--source-et",
        str(root / "inputs" / "source.et"),
        "--executable-et",
        str(executable),
        "--projection",
        str(root / "inputs" / "projection.json"),
        "--output",
        str(output_path),
        "--runner-oci-digest",
        str(manifest["runner"]["oci_digest"]),
        "--subject-sha256",
        str(subject["subject_sha256"]),
        "--perturbation-id",
        str(subject["perturbation_id"]),
        "--role",
        str(candidate["role"]),
        "--warmups",
        str(execution["candidate_warmups"]),
        "--iterations",
        str(execution["candidate_iterations"]),
    ]
    for region_id in candidate["selected_region_ids"]:
        argv.extend(("--selected-region", str(region_id)))
    physical_arguments = subject["physical_runner_arguments"]
    if physical_arguments["disable_overlap"]:
        argv.append("--disable-overlap")
    argv.extend(("--rank-skew-us", str(physical_arguments["rank_skew_us"])))
    if candidate["method"] == "communication_only_microbenchmark":
        argv.append("--communication-only")
    return argv


def _job_physical_batch(root: Path, output: Path, *, candidate_id: str, scope: str) -> int:
    manifest = verify_frozen_product_study(root)
    candidate, executable = _candidate_record(root, candidate_id)
    plan = _plan(manifest, root)
    perturbation_by_id = _perturbations(plan)
    if scope == "training":
        identifiers = [plan["baseline"]["id"], *(row["id"] for row in plan["training"])]
    elif scope == "holdout":
        identifiers = [plan["baseline"]["id"], *(row["id"] for row in plan["holdout"])]
    else:
        identifiers = [
            plan["baseline"]["id"],
            *(row["id"] for row in plan["training"]),
            *(row["id"] for row in plan["holdout"]),
        ]
    subjects = subject_map(manifest)
    if output.is_symlink() or not output.is_dir():
        raise ProductStudyError("physical job output must be the pre-created attempt directory")
    if (output / "batch-result.json").exists():
        raise ProductStudyError("physical attempt already contains a terminal batch result")
    measurement_directory = output / "measurements"
    measurement_directory.mkdir()
    log_directory = output / "logs"
    log_directory.mkdir()
    rows = []
    for position, perturbation_id in enumerate(identifiers):
        subject = subjects[perturbation_id]
        result_path = measurement_directory / f"{perturbation_id}.json"
        _run_child(
            _physical_argv(
                manifest,
                subject,
                candidate,
                executable,
                root=root,
                output_path=result_path,
            ),
            environment=_child_environment(
                perturbation_by_id[perturbation_id],
                str(manifest["runner"]["oci_digest"]),
            ),
            stdout_path=log_directory / f"{position:02d}-{perturbation_id}.stdout",
            stderr_path=log_directory / f"{position:02d}-{perturbation_id}.stderr",
            timeout_seconds=int(manifest["execution"]["physical_timeout_seconds"]),
        )
        measurement = _read_json(result_path, f"physical measurement {perturbation_id}")
        validate_physical_execution_measurement(measurement)
        rows.append(
            {
                "perturbation_id": perturbation_id,
                "measurement_id": measurement["measurement_id"],
                "path": result_path.relative_to(output).as_posix(),
                "sha256": _file_sha256(result_path),
            }
        )
    batch = {
        "format": BATCH_FORMAT,
        "kind": "physical_batch",
        "manifest_id": manifest["manifest_id"],
        "candidate_id": candidate_id,
        "scope": scope,
        "configuration_order": identifiers,
        "rows": rows,
        "completed_at": _timestamp(),
    }
    batch["batch_id"] = hashlib.sha256(canonical_json_bytes(batch)).hexdigest()
    _write_new(output / "batch-result.json", _canonical_bytes(batch) + b"\n")
    return 0


def _attempt_directories(root: Path, phase: str, logical_id: str) -> List[Path]:
    owner = root / "state" / "attempts" / phase / logical_id
    if not owner.exists():
        return []
    if owner.is_symlink() or not owner.is_dir():
        raise ProductStudyError("product study attempt owner is unsafe")
    attempts = sorted(
        (path for path in owner.iterdir() if path.is_dir() and not path.is_symlink()), key=lambda p: p.name
    )
    if [path.name for path in attempts] != [f"a-{index:06d}" for index in range(1, len(attempts) + 1)]:
        raise ProductStudyError("product study attempt inventory is not contiguous")
    return attempts


def _submitted_attempt(path: Path) -> Optional[Any]:
    submission_path = path / "submission.json"
    if not submission_path.is_file():
        return None
    submission = _read_json(submission_path, "SLURM submission")
    return SubmittedAttempt(
        attempt_id=path.name,
        attempt_directory=path,
        job_id=str(submission["job_id"]),
        request_sha256=str(submission["request_sha256"]),
        wrapper_sha256=str(submission["wrapper_sha256"]),
    )


def _latest_successful_attempt(root: Path, phase: str, logical_id: str) -> Optional[Path]:
    successful = []
    for path in _attempt_directories(root, phase, logical_id):
        terminal_path = path / "terminal.json"
        if terminal_path.is_file() and _read_json(terminal_path, "SLURM terminal").get("status") == "success":
            successful.append(path)
    return successful[-1] if successful else None


def _ensure_batch_jobs(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    phase: str,
    split: str,
    execute: bool,
    no_wait: bool,
    retry_failed: bool,
) -> bool:
    repetitions = int(manifest["execution"]["application_repetitions"])
    pending: List[Any] = []
    for repetition in range(repetitions):
        logical_id = f"{split}-repetition-{repetition}"
        if _latest_successful_attempt(root, phase, logical_id) is not None:
            continue
        attempts = _attempt_directories(root, phase, logical_id)
        latest = attempts[-1] if attempts else None
        submitted = _submitted_attempt(latest) if latest is not None else None
        if latest is not None and submitted is not None and not (latest / "terminal.json").exists():
            pending.append(submitted)
            continue
        if latest is not None and not retry_failed:
            raise ProductStudyError(f"{logical_id} failed; pass --retry-failed to append a new attempt")
        if not execute:
            return False
        day_offset = int(manifest["execution"]["repetition_day_offset_by_repetition"][repetition])
        pending.append(
            submit_job(
                root,
                phase=phase,
                logical_id=logical_id,
                repetition=repetition,
                planned_position=0,
                chunk_id=f"{split}-{repetition}",
                environment={},
                job_arguments=(
                    "job",
                    "application-repetition",
                    "--study",
                    "/commcanary/study",
                    "--output",
                    "/commcanary/output",
                    "--split",
                    split,
                    "--repetition",
                    str(repetition),
                ),
                timeout_seconds=int(manifest["execution"]["application_timeout_seconds"]),
                execute_acknowledged=True,
                begin_delay_days=day_offset,
            )
        )
    if no_wait:
        return not pending
    for submitted in pending:
        observation = wait_for_terminal(
            submitted,
            poll_interval_seconds=int(manifest["execution"]["poll_interval_seconds"]),
            maximum_wait_seconds=3 * 86_400,
        )
        if not observation.successful:
            raise ProductStudyError(f"SLURM job {submitted.job_id} failed with state {observation.state}")
    return True


def _application_oracles(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    phase: str,
    split: str,
) -> Dict[str, Mapping[str, Any]]:
    schedule = manifest["schedule"][f"{split}_configuration_order_by_repetition"]
    allocation_rows: Dict[str, List[Dict[str, Any]]] = {}
    selected_attempts = []
    for repetition, configuration_order in enumerate(schedule):
        logical_id = f"{split}-repetition-{repetition}"
        attempt = _latest_successful_attempt(root, phase, logical_id)
        if attempt is None:
            raise ProductStudyError(f"application evidence is incomplete: {logical_id}")
        terminal = _read_json(attempt / "terminal.json", "application terminal")
        request = _read_json(attempt / "request.json", "application request")
        batch = _read_json(attempt / "batch-result.json", "application batch result")
        if batch.get("configuration_order") != configuration_order:
            raise ProductStudyError("application execution order differs from the frozen schedule")
        rows = batch.get("rows")
        if not isinstance(rows, list) or len(rows) != len(configuration_order):
            raise ProductStudyError("application batch result is incomplete")
        by_id = {str(row["perturbation_id"]): row for row in rows}
        for position, perturbation_id in enumerate(configuration_order):
            raw_row = by_id[perturbation_id]
            measurement_path = attempt / str(raw_row["path"])
            if _file_sha256(measurement_path) != raw_row["sha256"]:
                raise ProductStudyError("application measurement changed after its batch completed")
            measurement = _read_json(measurement_path, "selected application measurement")
            validate_application_measurement(measurement)
            if measurement["measurement_id"] != raw_row["measurement_id"]:
                raise ProductStudyError("application measurement identity differs from its batch row")
            allocation_rows.setdefault(perturbation_id, []).append(
                {
                    "repetition": repetition,
                    "planned_position": position,
                    "chunk_id": request["chunk_id"],
                    "allocation_id": terminal["job_id"],
                    "node": terminal["node"],
                    "start_time": terminal["start_time"],
                    "end_time": terminal["end_time"],
                    "elapsed_seconds": terminal["elapsed_seconds"],
                    "measurement": copy.deepcopy(dict(measurement)),
                }
            )
        selected_attempts.append(
            {
                "repetition": repetition,
                "logical_id": logical_id,
                "attempt_id": attempt.name,
                "job_id": terminal["job_id"],
                "batch_id": batch["batch_id"],
            }
        )
    result: Dict[str, Mapping[str, Any]] = {}
    for perturbation_id, rows in sorted(allocation_rows.items()):
        oracle = build_application_oracle(
            rows,
            minimum_allocations=int(manifest["execution"]["application_repetitions"]),
            minimum_distinct_days=int(manifest["execution"]["minimum_distinct_days"]),
            maximum_allocation_elapsed_seconds=int(manifest["site"]["maximum_repetition_span_seconds"]),
        )
        validate_application_oracle(oracle)
        _write_or_verify_json(root / "state" / "application" / split / f"{perturbation_id}.oracle.json", oracle)
        result[perturbation_id] = oracle
    selection = {
        "format": SELECTION_FORMAT,
        "kind": "application_attempts",
        "manifest_id": manifest["manifest_id"],
        "split": split,
        "attempts": selected_attempts,
        "oracle_ids": {key: value["oracle_id"] for key, value in sorted(result.items())},
    }
    selection["selection_id"] = hashlib.sha256(canonical_json_bytes(selection)).hexdigest()
    _write_or_verify_json(root / "state" / "selections" / f"application-{split}.json", selection)
    return result


class _SitePhysicalExecutor:
    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        *,
        execute: bool,
        retry_failed: bool,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.execute = execute
        self.retry_failed = retry_failed
        self.plan = _plan(manifest, root)
        self.training_ids = {"baseline", *(str(row["id"]) for row in self.plan["training"])}
        self.holdout_ids = {str(row["id"]) for row in self.plan["holdout"]}
        self.cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}

    def __call__(self, request: PhysicalCandidateRequest) -> Mapping[str, Any]:
        self._freeze_candidate(request)
        key = (request.candidate_id, request.perturbation_id)
        if key in self.cache:
            return self.cache[key]
        if request.split == "training":
            scope = "training"
        elif request.perturbation_id in self.holdout_ids:
            scope = "holdout"
        else:
            scope = "all"
        self._load_or_run_batch(request, scope=scope)
        if key not in self.cache:
            raise ProductStudyError("physical batch did not produce the requested perturbation")
        return self.cache[key]

    def _freeze_candidate(self, request: PhysicalCandidateRequest) -> None:
        directory = self.root / "state" / "candidates" / request.candidate_id
        record = {
            "format": CANDIDATE_FORMAT,
            "candidate_id": request.candidate_id,
            "method": request.method,
            "selected_region_ids": list(request.selected_region_ids),
            "selected_node_ids": list(request.selected_node_ids),
            "executable_sha256": request.executable_sha256,
            "role": request.role,
        }
        _write_or_verify_bytes(directory / "executable.et", request.executable_et)
        _write_or_verify_json(directory / "candidate.json", record)

    def _load_or_run_batch(self, request: PhysicalCandidateRequest, *, scope: str) -> None:
        logical_id = f"{request.candidate_id[:24]}-{scope}"
        attempt = _latest_successful_attempt(self.root, "physical", logical_id)
        if attempt is None:
            attempts = _attempt_directories(self.root, "physical", logical_id)
            latest = attempts[-1] if attempts else None
            submitted = _submitted_attempt(latest) if latest is not None else None
            if latest is not None and submitted is not None and not (latest / "terminal.json").exists():
                pass
            elif latest is not None and not self.retry_failed:
                raise ProductStudyError(f"physical batch {logical_id} failed; pass --retry-failed")
            else:
                if not self.execute:
                    raise ProductStudyError("physical evidence is missing and scheduler execution was not acknowledged")
                submitted = submit_job(
                    self.root,
                    phase="physical",
                    logical_id=logical_id,
                    repetition=0,
                    planned_position=0,
                    chunk_id=f"physical-{request.candidate_id[:16]}",
                    environment={},
                    job_arguments=(
                        "job",
                        "physical-batch",
                        "--study",
                        "/commcanary/study",
                        "--output",
                        "/commcanary/output",
                        "--candidate-id",
                        request.candidate_id,
                        "--scope",
                        scope,
                    ),
                    timeout_seconds=int(self.manifest["execution"]["physical_timeout_seconds"]),
                    execute_acknowledged=True,
                )
            observation = wait_for_terminal(
                submitted,
                poll_interval_seconds=int(self.manifest["execution"]["poll_interval_seconds"]),
                maximum_wait_seconds=86_400,
            )
            if not observation.successful:
                raise ProductStudyError(f"physical SLURM job {submitted.job_id} failed with {observation.state}")
            attempt = submitted.attempt_directory
        self._load_batch(request.candidate_id, attempt)

    def _load_batch(self, candidate_id: str, attempt: Path) -> None:
        batch = _read_json(attempt / "batch-result.json", "physical batch result")
        if batch.get("candidate_id") != candidate_id:
            raise ProductStudyError("physical batch result names the wrong candidate")
        rows = batch.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ProductStudyError("physical batch result contains no measurements")
        for row in rows:
            perturbation_id = str(row["perturbation_id"])
            measurement_path = attempt / str(row["path"])
            if _file_sha256(measurement_path) != row["sha256"]:
                raise ProductStudyError("physical measurement changed after batch completion")
            measurement = _read_json(measurement_path, "physical batch measurement")
            validate_physical_execution_measurement(measurement)
            if measurement["measurement_id"] != row["measurement_id"]:
                raise ProductStudyError("physical measurement identity differs from its batch row")
            self.cache[(candidate_id, perturbation_id)] = measurement


def _run_study(root: Path, *, execute: bool, no_wait: bool, retry_failed: bool) -> int:
    manifest = verify_frozen_product_study(root)
    training_complete = _ensure_batch_jobs(
        root,
        manifest,
        phase="application-training",
        split="training",
        execute=execute,
        no_wait=no_wait,
        retry_failed=retry_failed,
    )
    if not training_complete:
        print(
            json.dumps({"status": "awaiting_training_application_allocations", "manifest_id": manifest["manifest_id"]})
        )
        return 0
    training = _application_oracles(
        root,
        manifest,
        phase="application-training",
        split="training",
    )
    baseline = training.pop("baseline")
    trace = decode_chakra_execution_trace(
        _bounded_bytes(
            root / "inputs" / "source.et",
            maximum=MAX_TRACE_BYTES,
            label="frozen source Chakra ET",
        )
    )
    projection = _read_json(root / "inputs" / "projection.json", "frozen projection")
    policy = _read_json(root / "inputs" / "policy.json", "frozen policy")
    validate_chakra_projection(projection, trace)
    validate_physical_canary_policy(policy)
    plan = _plan(manifest, root)
    holdout_ids = [str(row["id"]) for row in plan["holdout"]]
    site_executor = _SitePhysicalExecutor(root, manifest, execute=execute, retry_failed=retry_failed)

    def freeze_selection(selection: Mapping[str, Any]) -> None:
        canary_et, selected_nodes = encode_chakra_subgraph(trace, selection["selected_node_ids"])
        if list(selected_nodes) != selection["selected_node_ids"]:
            raise ProductStudyError("active selection dependency closure changed before freeze")
        if hashlib.sha256(canary_et).hexdigest() != selection["canary_et_sha256"]:
            raise ProductStudyError("active selection canary bytes changed before freeze")
        _write_or_verify_bytes(root / "state" / "selection" / "canary.et", canary_et)
        _write_or_verify_json(root / "state" / "selection" / "selection.json", selection)

    def load_holdout() -> HoldoutApplicationBatch:
        selection_path = root / "state" / "selection" / "selection.json"
        if selection_path.is_symlink() or not selection_path.is_file():
            raise ProductStudyError("holdout access requires a persisted active selection")
        complete = _ensure_batch_jobs(
            root,
            manifest,
            phase="application-holdout",
            split="holdout",
            execute=execute,
            no_wait=False,
            retry_failed=retry_failed,
        )
        if not complete:
            raise ProductStudyError("holdout application evidence is not complete")
        holdout = _application_oracles(
            root,
            manifest,
            phase="application-holdout",
            split="holdout",
        )
        paired_baseline = holdout.pop("baseline")
        return HoldoutApplicationBatch(baseline=paired_baseline, perturbations=holdout)

    result = synthesize_active_physical_canary(
        trace,
        projection,
        policy,
        baseline_application=baseline,
        training_applications=training,
        holdout_perturbation_ids=holdout_ids,
        holdout_application_loader=load_holdout,
        executor=site_executor,
        selection_freezer=freeze_selection,
        max_candidate_evaluations=int(manifest["execution"]["max_candidate_evaluations"]),
    )
    _write_or_verify_bytes(root / "state" / "result" / "canary.et", result.canary_et)
    _write_or_verify_json(root / "state" / "result" / "oracle-corpus.json", result.corpus)
    _write_or_verify_json(root / "state" / "result" / "active-study-ledger.json", result.ledger)
    _write_or_verify_json(
        root / "state" / "result" / "application-evidence.json",
        result.application_evidence,
    )
    _write_or_verify_json(
        root / "state" / "result" / "physical-evidence.json",
        result.physical_evidence,
    )
    bundle_path = root / "state" / "result" / "bundle"
    if bundle_path.exists():
        bundle = verify_physical_canary_bundle(str(bundle_path))
        if (
            bundle.get("corpus_id") != result.corpus["corpus_id"]
            or bundle.get("active_ledger_id") != result.ledger["ledger_id"]
            or bundle.get("application_evidence_set_id") != result.application_evidence["evidence_set_id"]
            or bundle.get("physical_evidence_set_id") != result.physical_evidence["evidence_set_id"]
        ):
            raise ProductStudyError("existing product bundle does not match the active study")
    else:
        bundle = build_physical_canary_bundle(
            str(root / "inputs" / "source.et"),
            projection,
            policy,
            str(bundle_path),
            corpus=result.corpus,
            active_ledger=result.ledger,
            application_evidence=result.application_evidence,
            physical_evidence=result.physical_evidence,
            mode="internal",
        )
    summary = {
        "format": "commcanary.rostam.product_study_result.v1",
        "manifest_id": manifest["manifest_id"],
        "status": result.status,
        "corpus_id": result.corpus["corpus_id"],
        "ledger_id": result.ledger["ledger_id"],
        "application_evidence_set_id": result.application_evidence["evidence_set_id"],
        "physical_evidence_set_id": result.physical_evidence["evidence_set_id"],
        "selection_sha256": result.ledger["selection"]["selection_sha256"],
        "canary_et_sha256": hashlib.sha256(result.canary_et).hexdigest(),
        "bundle_id": bundle["bundle_id"],
    }
    summary["result_id"] = hashlib.sha256(canonical_json_bytes(summary)).hexdigest()
    _write_or_verify_json(root / "state" / "result" / "result.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if result.status == "qualified_active_candidate" else 2


def _status(root: Path) -> Dict[str, Any]:
    manifest = verify_frozen_product_study(root)
    attempts_root = root / "state" / "attempts"
    counts = {"submitted": 0, "success": 0, "failed": 0, "unsubmitted": 0}
    phases: Dict[str, Dict[str, int]] = {}
    if attempts_root.is_dir():
        for phase_path in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
            phase_counts = {"submitted": 0, "success": 0, "failed": 0}
            for owner in sorted(path for path in phase_path.iterdir() if path.is_dir()):
                for attempt in _attempt_directories(root, phase_path.name, owner.name):
                    terminal = attempt / "terminal.json"
                    if terminal.is_file():
                        status = str(_read_json(terminal, "SLURM terminal")["status"])
                        phase_counts[status] += 1
                        counts[status] += 1
                    elif (attempt / "submission.json").is_file():
                        phase_counts["submitted"] += 1
                        counts["submitted"] += 1
                    else:
                        counts["unsubmitted"] += 1
            phases[phase_path.name] = phase_counts
    result_path = root / "state" / "result" / "result.json"
    return {
        "manifest_id": manifest["manifest_id"],
        "attempt_counts": counts,
        "phases": phases,
        "result": None if not result_path.is_file() else dict(_read_json(result_path, "product study result")),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        manifest = freeze_product_study(
            study_id=args.study_id,
            output_directory=Path(args.output),
            source_et=Path(args.source_et),
            projection_path=Path(args.projection),
            policy_path=Path(args.policy),
            perturbation_path=Path(args.perturbations),
            model_directory=Path(args.model),
            runner_descriptor_path=Path(args.runner_descriptor),
            runner_sif_path=Path(args.runner_sif),
            commcanary_wheel_path=Path(args.wheel),
        )
        print(json.dumps({"manifest_id": manifest["manifest_id"], "output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "verify":
        manifest = verify_frozen_product_study(Path(args.study))
        print(json.dumps({"manifest_id": manifest["manifest_id"], "status": "verified"}, sort_keys=True))
        return 0
    if args.command == "run":
        return _run_study(
            Path(args.study).resolve(),
            execute=bool(args.execute),
            no_wait=bool(args.no_wait),
            retry_failed=bool(args.retry_failed),
        )
    if args.command == "status":
        print(json.dumps(_status(Path(args.study).resolve()), sort_keys=True))
        return 0
    if args.command == "job":
        root = Path(args.study).resolve()
        output = Path(args.output).resolve()
        if args.job_command == "application-repetition":
            return _job_application_repetition(
                root,
                output,
                split=str(args.split),
                repetition=int(args.repetition),
            )
        if args.job_command == "physical-batch":
            return _job_physical_batch(
                root,
                output,
                candidate_id=str(args.candidate_id),
                scope=str(args.scope),
            )
    raise ProductStudyError("unsupported product study command")


if __name__ == "__main__":
    raise SystemExit(main())
