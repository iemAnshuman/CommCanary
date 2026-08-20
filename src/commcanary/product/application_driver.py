"""Deterministic real-application oracle for the first CommCanary product wedge.

This driver uses vLLM's actual tensor-parallel decoder implementation with
dummy-initialized model weights. Dummy weights avoid distributing a model
checkpoint; they do not replace model execution. Profiling is deliberately a
separate untimed pass so profiler overhead cannot enter the oracle samples.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from commcanary.artifacts.application_measurement import (
    application_subject_configuration,
    application_subject_sha256,
    validate_application_measurement,
)
from commcanary.artifacts.physical_execution import telemetry_assessment
from commcanary.execution.telemetry import capture_cycle_telemetry

FORMAT = "commcanary.application_measurement.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner-oci-digest", required=True)
    parser.add_argument("--subject-sha256", required=True)
    parser.add_argument("--perturbation-id", required=True)
    parser.add_argument("--profile-dir")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-length", type=int, default=256)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=536_870_912)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    # Both of the following default to the *safe* setting and must be opted
    # out of explicitly. CUDA graphs are outside the domain
    # docs/product-status.md declares qualified, and vLLM's custom all-reduce
    # exchanges peer-to-peer IPC handles between GPUs -- on PCIe parts with no
    # NVLink that is the riskiest multi-GPU path available and it is not what
    # this driver measures. Job 180257 left all four A100s on toranj1 in
    # "GPU requires reset" on 2026-08-04; the flags below were the wrong way
    # round at the time.
    parser.add_argument(
        "--allow-cuda-graphs",
        action="store_true",
        help="permit CUDA graph capture; outside the qualified domain",
    )
    parser.add_argument(
        "--allow-custom-all-reduce",
        action="store_true",
        help="permit vLLM's peer-to-peer custom all-reduce; unqualified on PCIe parts",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=1800.0,
        help="abort if the run exceeds this wall clock, so a hung engine cannot hold GPUs",
    )
    parser.add_argument(
        "--skip-gpu-health-check",
        action="store_true",
        help="do not refuse to start on GPUs that already report a pending reset",
    )
    return parser


def _require_positive(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_commitment(model_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(model_dir).as_posix()
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    if not rows:
        raise ValueError("model directory contains no regular files")
    return {"files": rows, "sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest()}


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _oci_digest(value: str) -> str:
    algorithm, separator, digest = value.partition(":")
    if algorithm != "sha256" or separator != ":":
        raise ValueError("runner-oci-digest must use sha256:<64 lowercase hex>")
    _sha256(digest, "runner-oci-digest")
    return value


def _prompt_tokens(*, batch_size: int, input_length: int, seed: int, iteration: int) -> List[List[int]]:
    # Avoid special tokens and make each request/iteration distinct so engine
    # caches cannot turn later measurements into a different workload.
    modulus = 31_997
    return [
        [
            3 + ((seed + iteration * 104_729 + request * 8_191 + position * 131) % modulus)
            for position in range(input_length)
        ]
        for request in range(batch_size)
    ]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


#: `nvidia-smi` reports a wedged GPU by replacing the performance state with a
#: bracketed message rather than by failing, so the text is matched directly.
_UNHEALTHY_GPU_MARKERS = ("requires reset", "unknown error", "inforom", "fell off the bus")


def require_healthy_gpus(expected_gpu_count: int) -> List[str]:
    """Refuse to start on GPUs that are already wedged.

    A GPU left in "requires reset" keeps answering NVML queries, so `nvidia-smi`
    looks fine and only CUDA context creation fails. Slurm does not drain the
    node either, so work keeps landing on it and failing in seconds. Checking
    here converts that into one clear refusal instead of a mystery, and stops
    this driver from piling more load onto hardware that is already broken.
    """

    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pstate", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi health query failed: {completed.stderr[:1024]}")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != expected_gpu_count:
        raise RuntimeError(f"nvidia-smi reported {len(rows)} GPUs; expected {expected_gpu_count}")
    unhealthy = [row for row in rows if any(marker in row.lower() for marker in _UNHEALTHY_GPU_MARKERS)]
    if unhealthy:
        raise RuntimeError(
            "refusing to start: "
            + str(len(unhealthy))
            + " of "
            + str(len(rows))
            + " GPUs are not healthy and need an administrator reset -> "
            + "; ".join(unhealthy)
        )
    return rows


def _start_watchdog(limit_seconds: float, label: str) -> "threading.Timer":
    """Terminate this process if the run outlives its declared budget.

    SIGTERM first so the teardown path still runs, then a hard exit if the
    engine is wedged badly enough to ignore it.
    """

    def _expire() -> None:
        print(
            f"watchdog: {label} exceeded {limit_seconds:.0f}s; terminating so the GPUs are released",
            file=sys.stderr,
            flush=True,
        )
        os.kill(os.getpid(), signal.SIGTERM)
        grace = threading.Timer(30.0, lambda: os._exit(75))
        grace.daemon = True
        grace.start()

    timer = threading.Timer(limit_seconds, _expire)
    timer.daemon = True
    timer.start()
    return timer


def _shutdown_engine(llm: Any, torch: Any) -> None:
    """Release the engine, its workers, and the process group, best effort.

    A tensor-parallel engine owns worker processes and a NCCL communicator.
    Leaving their teardown to interpreter exit after a mid-run failure is how
    GPUs are left dirty, so every step here is attempted and none is allowed to
    mask an earlier error.
    """

    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    for target, method in ((executor, "shutdown"), (engine, "shutdown")):
        hook = getattr(target, method, None)
        if callable(hook):
            try:
                hook()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                print(f"engine shutdown via {method} failed: {exc!r}", file=sys.stderr, flush=True)
    try:
        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()
    except Exception as exc:  # noqa: BLE001 - teardown must not raise
        print(f"process-group teardown failed: {exc!r}", file=sys.stderr, flush=True)
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - teardown must not raise
        print(f"CUDA cache release failed: {exc!r}", file=sys.stderr, flush=True)


def _observed_gpu_memory_used_bytes(expected_gpu_count: int) -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi memory observation failed: {completed.stderr[:1024]}")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != expected_gpu_count:
        raise RuntimeError(f"nvidia-smi returned {len(rows)} GPUs; expected {expected_gpu_count}")
    try:
        return max(int(row) for row in rows) * 1024 * 1024
    except ValueError as exc:
        raise RuntimeError("nvidia-smi returned an invalid memory.used value") from exc


def _run_generation(
    llm: Any,
    sampling_params: Any,
    *,
    batch_size: int,
    input_length: int,
    output_length: int,
    seed: int,
    iteration: int,
) -> Dict[str, float]:
    prompts = [
        {"prompt_token_ids": tokens}
        for tokens in _prompt_tokens(
            batch_size=batch_size,
            input_length=input_length,
            seed=seed,
            iteration=iteration,
        )
    ]
    started = time.perf_counter_ns()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("vLLM generation produced an invalid elapsed time")
    if len(outputs) != batch_size:
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for batch_size={batch_size}")
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    expected_output_tokens = batch_size * output_length
    if output_tokens != expected_output_tokens:
        raise RuntimeError(
            f"vLLM returned {output_tokens} tokens; expected {expected_output_tokens} with ignore_eos enabled"
        )
    return {
        "batch_latency_ms": elapsed_seconds * 1000.0,
        "output_token_throughput_per_second": output_tokens / elapsed_seconds,
        "total_token_throughput_per_second": (batch_size * input_length + output_tokens) / elapsed_seconds,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "tensor_parallel_size",
        "batch_size",
        "input_length",
        "output_length",
        "iterations",
        "kv_cache_memory_bytes",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        _require_positive(int(getattr(args, name)), name.replace("_", "-"))
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("gpu-memory-utilization must be in [0.1, 0.95]")
    if args.max_num_seqs < args.batch_size:
        raise ValueError("max-num-seqs must be at least batch-size")
    _oci_digest(args.runner_oci_digest)
    _sha256(args.subject_sha256, "subject-sha256")
    if not args.perturbation_id:
        raise ValueError("perturbation-id must be non-empty")
    declared_digest = os.environ.get("COMMCANARY_RUNNER_OCI_DIGEST")
    if declared_digest is not None and declared_digest != args.runner_oci_digest:
        raise ValueError("runner OCI digest disagrees with the container environment")

    model_dir = Path(args.model).resolve(strict=True)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite application evidence: {output_path}")
    profile_dir = Path(args.profile_dir).resolve() if args.profile_dir else None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=False)

    # Import after argument and output guards. Spawn is explicit so importing
    # libraries in this coordinator can never leave a forked CUDA runtime in
    # the tensor-parallel workers.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch  # type: ignore[import-not-found]
    import vllm  # type: ignore[import-not-found]
    from vllm import LLM, SamplingParams

    # Refuse a wedged node before allocating anything on it.
    if not args.skip_gpu_health_check:
        require_healthy_gpus(args.tensor_parallel_size)

    enforce_eager = not args.allow_cuda_graphs
    disable_custom_all_reduce = not args.allow_custom_all_reduce
    if args.allow_cuda_graphs:
        print(
            "warning: CUDA graph capture enabled; this is outside the domain docs/product-status.md declares qualified",
            file=sys.stderr,
            flush=True,
        )
    if args.allow_custom_all_reduce:
        print(
            "warning: vLLM custom all-reduce enabled; peer-to-peer IPC is unqualified on PCIe parts",
            file=sys.stderr,
            flush=True,
        )
    watchdog = _start_watchdog(args.max_runtime_seconds, f"perturbation {args.perturbation_id}")

    profiler_config = None
    if profile_dir is not None:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "max_iterations": 1,
            "ignore_frontend": True,
        }
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        load_format="dummy",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
        enable_prefix_caching=False,
        max_model_len=max(512, args.input_length + args.output_length),
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        profiler_config=profiler_config,
    )
    try:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.output_length, ignore_eos=True, seed=args.seed
        )

        telemetry = [capture_cycle_telemetry("before_warmup")]
        for iteration in range(args.warmups):
            _run_generation(
                llm,
                sampling_params,
                batch_size=args.batch_size,
                input_length=args.input_length,
                output_length=args.output_length,
                seed=args.seed,
                iteration=-(args.warmups - iteration),
            )

        telemetry.append(capture_cycle_telemetry("before_measured_cycle_1"))
        samples = []
        for iteration in range(args.iterations):
            sample = _run_generation(
                llm,
                sampling_params,
                batch_size=args.batch_size,
                input_length=args.input_length,
                output_length=args.output_length,
                seed=args.seed,
                iteration=iteration,
            )
            sample["observed_gpu_memory_used_bytes"] = _observed_gpu_memory_used_bytes(args.tensor_parallel_size)
            samples.append(sample)
            telemetry.append(capture_cycle_telemetry(f"after_measured_cycle_{iteration + 1}"))

        if profile_dir is not None:
            llm.start_profile(profile_prefix="commcanary")
            _run_generation(
                llm,
                sampling_params,
                batch_size=args.batch_size,
                input_length=args.input_length,
                output_length=args.output_length,
                seed=args.seed,
                iteration=args.iterations + 1,
            )
            llm.stop_profile()
        telemetry.append(capture_cycle_telemetry("final"))

        latency = [sample["batch_latency_ms"] for sample in samples]
        output_throughput = [sample["output_token_throughput_per_second"] for sample in samples]
        total_throughput = [sample["total_token_throughput_per_second"] for sample in samples]
        observed_memory = [int(sample["observed_gpu_memory_used_bytes"]) for sample in samples]
        model_identity = _model_commitment(model_dir)
        profile_identity = _model_commitment(profile_dir) if profile_dir is not None else None
        environment = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "nccl": list(torch.cuda.nccl.version()),
            "visible_gpu_count": torch.cuda.device_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "nccl_configuration": {
                name: os.environ.get(name)
                for name in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_NET", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
            },
            "runtime_configuration": {"CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING")},
        }
        subject_configuration = application_subject_configuration(
            runner_oci_digest=args.runner_oci_digest,
            application_engine="vllm",
            application_engine_version=str(vllm.__version__),
            engine_configuration={
                "kind": "vllm.v1",
                "enforce_eager": bool(enforce_eager),
                "disable_custom_all_reduce": bool(disable_custom_all_reduce),
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
            },
            perturbation_environment={
                **environment["nccl_configuration"],
                **environment["runtime_configuration"],
            },
        )
        if application_subject_sha256(subject_configuration) != args.subject_sha256:
            raise ValueError("subject-sha256 disagrees with the observed application stack configuration")
        evidence: Dict[str, Any] = {
            "format": FORMAT,
            "runner": {
                "oci_digest": args.runner_oci_digest,
                "execution_protocol": "vllm-offline-tensor-parallel.v1",
            },
            "subject_sha256": args.subject_sha256,
            "subject_configuration": subject_configuration,
            "perturbation_id": args.perturbation_id,
            "application": {
                "name": "vllm-dummy-llama-tp-dense-decoder",
                "engine": "vllm",
                "engine_version": str(vllm.__version__),
                "ground_truth_kind": "application_ground_truth",
                "model_sha256": model_identity["sha256"],
                "model_files": model_identity["files"],
            },
            "workload": {
                "batch_size": args.batch_size,
                "input_length": args.input_length,
                "output_length": args.output_length,
                "tensor_parallel_size": args.tensor_parallel_size,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "seed": args.seed,
                "load_format": "dummy",
                "dtype": "bfloat16",
                "prefix_caching": False,
                "worker_multiprocessing_method": "spawn",
            },
            "samples": samples,
            "summary": {
                "median_batch_latency_ms": statistics.median(latency),
                "p99_batch_latency_ms": _percentile(latency, 0.99),
                "median_output_token_throughput_per_second": statistics.median(output_throughput),
                "median_total_token_throughput_per_second": statistics.median(total_throughput),
                "max_observed_gpu_memory_used_bytes": max(observed_memory),
            },
            "environment": environment,
            "environment_sha256": hashlib.sha256(_canonical_bytes(environment)).hexdigest(),
            "profile": {
                "captured": profile_dir is not None,
                "excluded_from_timed_samples": True,
                "artifact_sha256": None if profile_identity is None else profile_identity["sha256"],
                "files": [] if profile_identity is None else profile_identity["files"],
            },
            "telemetry": telemetry,
            "telemetry_assessment": telemetry_assessment(telemetry, world_size=args.tensor_parallel_size),
        }
        raw_without_identity = _canonical_bytes(evidence)
        evidence["measurement_id"] = hashlib.sha256(raw_without_identity).hexdigest()
        validate_application_measurement(evidence)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as handle:
            handle.write(_canonical_bytes(evidence) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"measurement_id": evidence["measurement_id"], "output": str(output_path)}, sort_keys=True))
        return 0
    finally:
        # The engine owns worker processes and a NCCL communicator. Releasing
        # them here rather than at interpreter exit is what keeps a failed run
        # from leaving the GPUs dirty for the next job.
        watchdog.cancel()
        _shutdown_engine(llm, torch)


if __name__ == "__main__":
    raise SystemExit(main())
