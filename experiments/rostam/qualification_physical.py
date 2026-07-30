#!/usr/bin/env python3
"""Run one manifest-bound exact-work qualification materialization on Rostam.

The campaign binds each portable request/materialization file independently.
Every torchrun rank copies those immutable bytes into its own new canonical
directory, revalidates the complete chain through the installed CommCanary
wheel, and only then initializes torch.distributed. Rank 0 emits the reference
executor's single JSON object for the strict physical-result adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

from commcanary.artifacts import (
    QUALIFICATION_ARTIFACT_PATHS,
    QUALIFICATION_MATERIALIZATION_FILENAME,
    QUALIFICATION_REPLAY_PROGRAM_FILENAME,
    QUALIFICATION_REQUEST_FILENAME,
)
from commcanary.execution import (
    distributed_execution_environment,
    execute_qualification_materialization,
)
from experiments.rostam.harness import (
    ContractError,
    file_sha256,
    read_bounded_bytes,
    strict_json_loads,
)

MAX_BOUND_INPUT_BYTES = 64 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
QUALIFICATION_COMPARISON_STDOUT_SCHEMA = "commcanary.rostam.qualification-comparison.stdout.v1"
SOURCE_CAPTURE_EVIDENCE_SCHEMA = "commcanary.rostam.overlap-capture-evidence.v2"
SOURCE_CAPTURE_STDOUT_SCHEMA = "commcanary.rostam.workload-overlap-capture.stdout.v2"
SOURCE_TIMING_SEMANTICS = "maximum-rank-unprofiled-whole-program-duration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--replay-program", type=Path, required=True)
    parser.add_argument("--source-capture-evidence", type=Path, required=True)
    parser.add_argument("--source-capture-stdout", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=300)
    return parser


def _copy_bound_input(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"qualification input must be a real regular file: {source}")
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"qualification staging destination already exists: {destination}")
    copied = 0
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            while chunk := source_handle.read(COPY_CHUNK_BYTES):
                copied += len(chunk)
                if copied > MAX_BOUND_INPUT_BYTES:
                    raise SystemExit(f"qualification input exceeds {MAX_BOUND_INPUT_BYTES} bytes: {source.name}")
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    except OSError as exc:
        raise SystemExit(f"cannot stage qualification input {source}: {exc}") from exc
    if copied <= 0:
        raise SystemExit(f"qualification input is empty: {source}")
    os.chmod(destination, 0o444)


def stage_qualification_inputs(
    sources: Mapping[str, Path],
    *,
    rank: int,
    workspace: Path,
) -> Tuple[Path, Path]:
    expected = {
        "request_manifest",
        "source_trace",
        "canary",
        "fidelity",
        "materialization_manifest",
        "replay_program",
    }
    if set(sources) != expected:
        raise SystemExit("qualification staging input ownership is incomplete")
    root = workspace / f"qualification-input-rank-{rank:05d}"
    request_directory = root / "request"
    materialization_directory = root / "materialization"
    try:
        root.mkdir(mode=0o700)
        request_directory.mkdir(mode=0o700)
        materialization_directory.mkdir(mode=0o700)
    except OSError as exc:
        raise SystemExit(f"cannot create rank-local qualification staging directories: {exc}") from exc
    request_names = {
        "request_manifest": QUALIFICATION_REQUEST_FILENAME,
        "source_trace": QUALIFICATION_ARTIFACT_PATHS["source_trace"],
        "canary": QUALIFICATION_ARTIFACT_PATHS["canary"],
        "fidelity": QUALIFICATION_ARTIFACT_PATHS["fidelity_verification"],
    }
    materialization_names = {
        "materialization_manifest": QUALIFICATION_MATERIALIZATION_FILENAME,
        "replay_program": QUALIFICATION_REPLAY_PROGRAM_FILENAME,
    }
    for source_id, filename in request_names.items():
        _copy_bound_input(sources[source_id], request_directory / filename)
    for source_id, filename in materialization_names.items():
        _copy_bound_input(sources[source_id], materialization_directory / filename)
    return request_directory, materialization_directory


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"{field} must be an object")
    return value


def _strict(value: Mapping[str, Any], field: str, expected: Tuple[str, ...]) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing:
        raise SystemExit(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SystemExit(f"{field} has unknown fields: {', '.join(unknown)}")


def _samples(value: Any, field: str) -> List[float]:
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{field} must contain timing samples")
    samples: List[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SystemExit(f"{field}[{index}] must be a finite non-negative number")
        sample = float(item)
        if not 0.0 <= sample < float("inf"):
            raise SystemExit(f"{field}[{index}] must be a finite non-negative number")
        samples.append(sample)
    return samples


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{field} must be a finite non-negative number")
    result = float(value)
    if not 0.0 <= result < float("inf"):
        raise SystemExit(f"{field} must be a finite non-negative number")
    return result


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _iqr(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        lower = ordered[:middle]
        upper = ordered[middle + 1 :]
    else:
        lower = ordered[:middle]
        upper = ordered[middle:]
    return _median(upper) - _median(lower)


def _bounded_json(path: Path, field: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{field} must be a real regular file: {path}")
    try:
        raw = read_bounded_bytes(path, max_bytes=MAX_BOUND_INPUT_BYTES, field=field)
        return _object(strict_json_loads(raw), field)
    except (ContractError, OSError, UnicodeError) as exc:
        raise SystemExit(f"cannot load {field}: {exc}") from exc


def load_source_capture_observation(
    evidence_path: Path,
    stdout_path: Path,
    *,
    world_size: int,
    iterations: int,
) -> Dict[str, Any]:
    """Verify and retain the exact same-node source timing observation."""

    evidence = _bounded_json(evidence_path, "source capture evidence")
    _strict(
        evidence,
        "source capture evidence",
        (
            "schema",
            "diagnostic_id",
            "attempt_id",
            "scheduler",
            "input_bindings",
            "capture",
            "import",
            "artifacts",
            "claims",
        ),
    )
    if evidence["schema"] != SOURCE_CAPTURE_EVIDENCE_SCHEMA:
        raise SystemExit(f"source capture evidence requires schema {SOURCE_CAPTURE_EVIDENCE_SCHEMA!r}")
    scheduler = _object(evidence["scheduler"], "source capture evidence.scheduler")
    _strict(scheduler, "source capture evidence.scheduler", ("job_id", "node", "partition"))
    if not all(isinstance(scheduler[field], str) and scheduler[field] for field in scheduler):
        raise SystemExit("source capture scheduler fields must be non-empty strings")

    artifacts = _object(evidence["artifacts"], "source capture evidence.artifacts")
    stdout_reference = _object(
        artifacts.get("capture.stdout.json"),
        "source capture evidence.artifacts[capture.stdout.json]",
    )
    _strict(
        stdout_reference,
        "source capture evidence.artifacts[capture.stdout.json]",
        ("sha256", "size_bytes"),
    )
    if stdout_path.is_symlink() or not stdout_path.is_file():
        raise SystemExit(f"source capture stdout must be a real regular file: {stdout_path}")
    stdout_sha256 = file_sha256(stdout_path)
    if stdout_reference["sha256"] != stdout_sha256 or stdout_reference["size_bytes"] != stdout_path.stat().st_size:
        raise SystemExit("source capture stdout bytes disagree with the preserved evidence")

    source = _bounded_json(stdout_path, "source capture stdout")
    _strict(
        source,
        "source capture stdout",
        (
            "schema",
            "rank",
            "world_size",
            "tokens",
            "layers",
            "hidden",
            "gemm_m_rank0",
            "gemm_n",
            "dtype",
            "msg_sizes_bytes",
            "inject_skew",
            "distributed_timeout_seconds",
            "execution_semantics",
            "warmup_programs",
            "measurement_iterations",
            "profile_warmup_programs",
            "profiled_programs",
            "rank_timings_us",
            "timings_us",
            "timing_semantics",
            "metrics",
            "profiles",
        ),
    )
    if source["schema"] != SOURCE_CAPTURE_STDOUT_SCHEMA:
        raise SystemExit(f"source capture stdout requires schema {SOURCE_CAPTURE_STDOUT_SCHEMA!r}")
    if source["rank"] != 0 or source["world_size"] != world_size:
        raise SystemExit("source capture stdout does not belong to rank 0 of the declared world")
    if source["measurement_iterations"] != iterations:
        raise SystemExit("source capture iteration count disagrees with the qualification workload")
    if source["timing_semantics"] != SOURCE_TIMING_SEMANTICS:
        raise SystemExit("source capture timing semantics are unsupported")
    if source["execution_semantics"] != "async-all-reduce-then-gemm-then-explicit-wait":
        raise SystemExit("source capture execution semantics are unsupported")

    rank_timings_raw = source["rank_timings_us"]
    if not isinstance(rank_timings_raw, list) or len(rank_timings_raw) != world_size:
        raise SystemExit("source capture rank timings do not cover every rank")
    rank_timings = [
        _samples(values, f"source capture rank_timings_us[{rank}]") for rank, values in enumerate(rank_timings_raw)
    ]
    if any(len(values) != iterations for values in rank_timings):
        raise SystemExit("source capture rank timing count disagrees with measurement_iterations")
    timings = _samples(source["timings_us"], "source capture timings_us")
    if len(timings) != iterations:
        raise SystemExit("source capture timing count disagrees with measurement_iterations")
    recomputed = [max(rank_timings[rank][iteration] for rank in range(world_size)) for iteration in range(iterations)]
    if timings != recomputed:
        raise SystemExit("source capture maximum-rank timings disagree with rank timings")

    metrics = _object(source["metrics"], "source capture stdout.metrics")
    _strict(metrics, "source capture stdout.metrics", ("count", "median_us", "iqr_us"))
    if (
        metrics["count"] != len(timings)
        or abs(_finite(metrics["median_us"], "source capture median_us") - _median(timings)) > 0.001
        or abs(_finite(metrics["iqr_us"], "source capture iqr_us") - _iqr(timings)) > 0.001
    ):
        raise SystemExit("source capture metrics disagree with timing samples")
    capture = _object(evidence["capture"], "source capture evidence.capture")
    evidence_metrics = _object(capture.get("metrics"), "source capture evidence.capture.metrics")
    timing_range = _object(
        capture.get("timing_range_us"),
        "source capture evidence.capture.timing_range_us",
    )
    if (
        capture.get("timing_semantics") != SOURCE_TIMING_SEMANTICS
        or capture.get("execution_semantics") != source["execution_semantics"]
        or capture.get("measurement_iterations") != len(timings)
        or dict(evidence_metrics) != dict(metrics)
        or timing_range.get("min") != min(timings)
        or timing_range.get("max") != max(timings)
    ):
        raise SystemExit("source capture evidence summary disagrees with source timing bytes")

    return {
        "evidence_sha256": file_sha256(evidence_path),
        "stdout_sha256": stdout_sha256,
        "diagnostic_id": evidence["diagnostic_id"],
        "scheduler": dict(scheduler),
        "execution_semantics": source["execution_semantics"],
        "timing_semantics": source["timing_semantics"],
        "rank_timings_us": rank_timings,
        "timings_us": timings,
        "metrics": {
            "count": len(timings),
            "median_us": _median(timings),
            "iqr_us": _iqr(timings),
            "min_us": min(timings),
            "max_us": max(timings),
        },
    }


def run(args: argparse.Namespace) -> int:
    rank, world_size, local_rank = distributed_execution_environment(os.environ)
    sources: Dict[str, Path] = {
        "request_manifest": args.request_manifest,
        "source_trace": args.source_trace,
        "canary": args.canary,
        "fidelity": args.fidelity,
        "materialization_manifest": args.materialization_manifest,
        "replay_program": args.replay_program,
    }
    request_directory, materialization_directory = stage_qualification_inputs(
        sources,
        rank=rank,
        workspace=Path.cwd(),
    )
    source_capture = load_source_capture_observation(
        args.source_capture_evidence,
        args.source_capture_stdout,
        world_size=world_size,
        iterations=args.iterations,
    )
    result = execute_qualification_materialization(
        str(request_directory),
        str(materialization_directory),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=args.device,
        backend=args.backend,
        iterations=args.iterations,
        warmup=args.warmup,
        distributed_timeout_seconds=args.distributed_timeout_seconds,
    )
    if result is not None:
        payload = {
            "schema": QUALIFICATION_COMPARISON_STDOUT_SCHEMA,
            "source_capture": source_capture,
            "execution": cast(Mapping[str, Any], result),
            "claims": {
                "single_configuration_timing_comparison": "diagnostic",
                "physical_fidelity": "unproven",
                "multi_configuration_ranking": "not_measured",
                "qualification_verdict": "not_issued",
            },
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
