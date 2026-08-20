"""Independent SGLang application oracle for CommCanary's second engine.

The driver executes SGLang's real tensor-parallel dense-decoder engine with
dummy-initialized weights.  It shares only deterministic input and evidence
helpers with the vLLM oracle; SGLang owns process launch, scheduling, kernels,
collectives, buffers, and output handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from commcanary.artifacts.application_measurement import (
    application_subject_configuration,
    application_subject_sha256,
    validate_application_measurement,
)
from commcanary.artifacts.physical_execution import telemetry_assessment
from commcanary.execution.telemetry import capture_cycle_telemetry
from commcanary.product.application_driver import _start_watchdog, require_healthy_gpus

from .application_driver import (
    FORMAT,
    _canonical_bytes,
    _model_commitment,
    _observed_gpu_memory_used_bytes,
    _oci_digest,
    _percentile,
    _prompt_tokens,
    _require_positive,
    _sha256,
)


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
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    # Mirrors the vLLM driver: the unqualified paths default off and must be
    # opted into. CUDA graphs and peer-to-peer custom all-reduce are both
    # outside the domain docs/product-status.md declares qualified.
    parser.add_argument(
        "--allow-cuda-graphs",
        action="store_true",
        help="permit CUDA graph capture; outside the qualified domain",
    )
    parser.add_argument(
        "--allow-custom-all-reduce",
        action="store_true",
        help="permit peer-to-peer custom all-reduce; unqualified on PCIe parts",
    )
    parser.add_argument("--disable-overlap-schedule", action="store_true")
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


def _normalized_outputs(raw_outputs: Any, *, batch_size: int, input_length: int, output_length: int) -> int:
    if isinstance(raw_outputs, Mapping):
        outputs: Sequence[Any] = [raw_outputs]
    elif isinstance(raw_outputs, list):
        outputs = raw_outputs
    else:
        raise RuntimeError("SGLang returned neither one output object nor an output array")
    if len(outputs) != batch_size:
        raise RuntimeError(f"SGLang returned {len(outputs)} outputs for batch_size={batch_size}")
    total = 0
    for index, raw_output in enumerate(outputs):
        if not isinstance(raw_output, Mapping):
            raise RuntimeError(f"SGLang output {index} is not an object")
        meta = raw_output.get("meta_info")
        if not isinstance(meta, Mapping):
            raise RuntimeError(f"SGLang output {index} lacks meta_info")
        completion_tokens = meta.get("completion_tokens")
        prompt_tokens = meta.get("prompt_tokens")
        if completion_tokens != output_length:
            raise RuntimeError(
                f"SGLang output {index} reports {completion_tokens!r} completion tokens; expected {output_length}"
            )
        if prompt_tokens != input_length:
            raise RuntimeError(
                f"SGLang output {index} reports {prompt_tokens!r} prompt tokens; expected {input_length}"
            )
        output_ids = raw_output.get("output_ids")
        if not isinstance(output_ids, list) or len(output_ids) != output_length:
            raise RuntimeError(f"SGLang output {index} does not contain exactly {output_length} output token IDs")
        finish_reason = meta.get("finish_reason")
        if not isinstance(finish_reason, Mapping) or finish_reason.get("type") != "length":
            raise RuntimeError(f"SGLang output {index} did not terminate at the declared length")
        if meta.get("cached_tokens", 0) != 0:
            raise RuntimeError(f"SGLang output {index} used prefix-cached tokens despite a disabled radix cache")
        total += completion_tokens
    return total


def _run_generation(
    engine: Any,
    *,
    batch_size: int,
    input_length: int,
    output_length: int,
    seed: int,
    iteration: int,
) -> Dict[str, float]:
    input_ids = _prompt_tokens(
        batch_size=batch_size,
        input_length=input_length,
        seed=seed,
        iteration=iteration,
    )
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": output_length,
        "ignore_eos": True,
        "sampling_seed": (seed + iteration * 104_729) % (1 << 30),
    }
    started = time.perf_counter_ns()
    raw_outputs = engine.generate(
        input_ids=input_ids,
        sampling_params=sampling_params,
        stream=False,
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("SGLang generation produced an invalid elapsed time")
    output_tokens = _normalized_outputs(
        raw_outputs,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
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
        "max_running_requests",
        "max_total_tokens",
        "chunked_prefill_size",
        "max_prefill_tokens",
    ):
        _require_positive(int(getattr(args, name)), name.replace("_", "-"))
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.1 <= args.mem_fraction_static <= 0.95:
        raise ValueError("mem-fraction-static must be in [0.1, 0.95]")
    if args.max_running_requests < args.batch_size:
        raise ValueError("max-running-requests must be at least batch-size")
    prefill_tokens = args.batch_size * args.input_length
    if args.max_total_tokens < prefill_tokens:
        raise ValueError("max-total-tokens must cover the complete batch prefill")
    if args.max_prefill_tokens < prefill_tokens:
        raise ValueError("max-prefill-tokens must cover the complete batch prefill")
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
        os.environ["SGLANG_TORCH_PROFILER_DIR"] = str(profile_dir)

    multiprocessing.set_start_method("spawn", force=True)
    import sglang as sgl  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from sglang.version import __version__ as sglang_version  # type: ignore[import-not-found]

    if not args.skip_gpu_health_check:
        require_healthy_gpus(args.tensor_parallel_size)

    disable_cuda_graph = not args.allow_cuda_graphs
    disable_custom_all_reduce = not args.allow_custom_all_reduce
    if args.allow_cuda_graphs:
        print(
            "warning: CUDA graph capture enabled; this is outside the domain docs/product-status.md declares qualified",
            file=sys.stderr,
            flush=True,
        )
    if args.allow_custom_all_reduce:
        print(
            "warning: custom all-reduce enabled; peer-to-peer IPC is unqualified on PCIe parts",
            file=sys.stderr,
            flush=True,
        )
    watchdog = _start_watchdog(args.max_runtime_seconds, f"perturbation {args.perturbation_id}")

    engine = sgl.Engine(
        model_path=str(model_dir),
        tp_size=args.tensor_parallel_size,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        load_format="dummy",
        random_seed=args.seed,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        chunked_prefill_size=args.chunked_prefill_size,
        max_prefill_tokens=args.max_prefill_tokens,
        context_length=max(512, args.input_length + args.output_length),
        disable_radix_cache=True,
        disable_cuda_graph=disable_cuda_graph,
        disable_custom_all_reduce=disable_custom_all_reduce,
        disable_overlap_schedule=args.disable_overlap_schedule,
    )
    telemetry = [capture_cycle_telemetry("before_warmup")]
    samples: List[Dict[str, Any]] = []
    try:
        for iteration in range(args.warmups):
            _run_generation(
                engine,
                batch_size=args.batch_size,
                input_length=args.input_length,
                output_length=args.output_length,
                seed=args.seed,
                iteration=-(args.warmups - iteration),
            )

        telemetry.append(capture_cycle_telemetry("before_measured_cycle_1"))
        for iteration in range(args.iterations):
            sample = _run_generation(
                engine,
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
            engine.start_profile(output_dir=str(profile_dir))
            _run_generation(
                engine,
                batch_size=args.batch_size,
                input_length=args.input_length,
                output_length=args.output_length,
                seed=args.seed,
                iteration=args.iterations + 1,
            )
            engine.stop_profile()
        telemetry.append(capture_cycle_telemetry("final"))
    finally:
        watchdog.cancel()
        engine.shutdown()

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
        application_engine="sglang",
        application_engine_version=str(sglang_version),
        engine_configuration={
            "kind": "sglang.v1",
            "disable_cuda_graph": bool(disable_cuda_graph),
            "disable_custom_all_reduce": bool(disable_custom_all_reduce),
            "disable_overlap_schedule": bool(args.disable_overlap_schedule),
            "mem_fraction_static": args.mem_fraction_static,
            "max_running_requests": args.max_running_requests,
            "max_total_tokens": args.max_total_tokens,
            "chunked_prefill_size": args.chunked_prefill_size,
            "max_prefill_tokens": args.max_prefill_tokens,
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
            "execution_protocol": "sglang-offline-tensor-parallel.v1",
        },
        "subject_sha256": args.subject_sha256,
        "subject_configuration": subject_configuration,
        "perturbation_id": args.perturbation_id,
        "application": {
            "name": "sglang-dummy-llama-tp-dense-decoder",
            "engine": "sglang",
            "engine_version": str(sglang_version),
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
    evidence["measurement_id"] = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
    validate_application_measurement(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(_canonical_bytes(evidence) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"measurement_id": evidence["measurement_id"], "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
