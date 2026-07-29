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
from typing import Dict, Mapping, Tuple

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

MAX_BOUND_INPUT_BYTES = 64 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--replay-program", type=Path, required=True)
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
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
