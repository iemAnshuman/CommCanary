"""Version-compatible vLLM workload for the scope-limited issue 2971 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

FORMAT = "commcanary.historical_vllm_measurement.v1"
_IMAGE_DIGEST_PREFIX = "sha256:"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--configuration-id", required=True)
    parser.add_argument("--image-manifest-digest", required=True)
    parser.add_argument("--sif-sha256", required=True)
    parser.add_argument("--expected-vllm-version", required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--swap-space-gib", type=int, required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_text(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _image_digest(value: str) -> str:
    if not value.startswith(_IMAGE_DIGEST_PREFIX):
        raise ValueError("image-manifest-digest must use SHA-256")
    _sha256_text(value[len(_IMAGE_DIGEST_PREFIX) :], "image-manifest-digest")
    return value


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _prompt_tokens(
    *,
    batch_size: int,
    input_length: int,
    seed: int,
    iteration: int,
) -> List[List[int]]:
    modulus = 31_997
    return [
        [
            3 + ((seed + iteration * 104_729 + request * 8_191 + position * 131) % modulus)
            for position in range(input_length)
        ]
        for request in range(batch_size)
    ]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _gpu_snapshot(phase: str) -> Dict[str, Any]:
    fields = (
        "index,uuid,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
        "clocks_throttle_reasons.active,ecc.errors.corrected.volatile.total,"
        "ecc.errors.uncorrected.volatile.total"
    )
    argv = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=15)
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi telemetry failed: {completed.stderr[:1024]}")
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 11:
            raise RuntimeError("nvidia-smi telemetry returned an unexpected field count")
        rows.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "pstate": values[3],
                "sm_clock_mhz": float(values[4]),
                "memory_clock_mhz": float(values[5]),
                "temperature_c": float(values[6]),
                "power_w": float(values[7]),
                "clock_event_reasons_active": values[8],
                "corrected_volatile_ecc": int(values[9]),
                "uncorrected_volatile_ecc": int(values[10]),
            }
        )
    if len(rows) != 4 or [row["index"] for row in rows] != [0, 1, 2, 3]:
        raise RuntimeError("historical study requires exactly four visible GPUs in canonical order")
    if any("A100" not in str(row["name"]) for row in rows):
        raise RuntimeError("historical study is qualified only for four A100 GPUs")
    return {
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "command": argv,
        "gpus": rows,
    }


def _run_generation(
    llm: Any,
    sampling_params: Any,
    *,
    batch_size: int,
    input_length: int,
    output_length: int,
    seed: int,
    iteration: int,
) -> Dict[str, Any]:
    prompt_token_ids = _prompt_tokens(
        batch_size=batch_size,
        input_length=input_length,
        seed=seed,
        iteration=iteration,
    )
    started = time.perf_counter_ns()
    outputs = llm.generate(
        prompts=None,
        sampling_params=sampling_params,
        prompt_token_ids=prompt_token_ids,
        use_tqdm=False,
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
    if len(outputs) != batch_size:
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for batch_size={batch_size}")
    generated: List[List[int]] = []
    for output in outputs:
        choices = getattr(output, "outputs", None)
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("vLLM did not return exactly one completion per request")
        token_ids = list(getattr(choices[0], "token_ids", []))
        if len(token_ids) != output_length or any(not isinstance(value, int) for value in token_ids):
            raise RuntimeError("vLLM completion length differs from the fixed workload")
        generated.append(token_ids)
    output_tokens = batch_size * output_length
    return {
        "batch_latency_ms": elapsed_seconds * 1000.0,
        "output_token_throughput_per_second": output_tokens / elapsed_seconds,
        "total_token_throughput_per_second": (batch_size * input_length + output_tokens) / elapsed_seconds,
        "output_commitment_sha256": hashlib.sha256(_canonical_bytes(generated)).hexdigest(),
    }


def _nccl_version(raw: Any) -> List[int]:
    if isinstance(raw, tuple):
        values = list(raw)
    elif isinstance(raw, int):
        major = raw // 10_000
        minor = (raw % 10_000) // 100
        patch = raw % 100
        values = [major, minor, patch]
    else:
        raise RuntimeError("torch returned an unsupported NCCL version value")
    if not values or any(not isinstance(value, int) for value in values):
        raise RuntimeError("torch returned an invalid NCCL version")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "tensor_parallel_size",
        "batch_size",
        "input_length",
        "output_length",
        "max_model_len",
        "iterations",
        "swap_space_gib",
    ):
        _positive(int(getattr(args, name)), name.replace("_", "-"))
    if args.tensor_parallel_size != 4:
        raise ValueError("historical study requires tensor-parallel-size=4")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("gpu-memory-utilization must be in [0.1, 0.95]")
    _image_digest(args.image_manifest_digest)
    _sha256_text(args.sif_sha256, "sif-sha256")
    if not args.configuration_id or not args.expected_vllm_version:
        raise ValueError("configuration and expected vLLM version must be non-empty")
    model = Path(args.model).resolve(strict=True)
    if model.is_symlink() or not model.is_dir():
        raise ValueError("model must be a regular directory")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical measurement: {output}")

    import torch  # type: ignore[import-not-found]
    import vllm  # type: ignore[import-not-found]
    from vllm import LLM, SamplingParams

    if str(vllm.__version__) != args.expected_vllm_version:
        raise RuntimeError(f"runtime reports vLLM {vllm.__version__!r}; expected {args.expected_vllm_version!r}")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("historical study requires exactly four visible CUDA devices")
    telemetry = [_gpu_snapshot("before_engine_initialization")]
    engine_started = time.perf_counter_ns()
    llm = LLM(
        model=str(model),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space_gib,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_context_len_to_capture=args.max_model_len,
    )
    engine_initialization_seconds = (time.perf_counter_ns() - engine_started) / 1_000_000_000.0
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_length,
        ignore_eos=True,
    )
    telemetry.append(_gpu_snapshot("before_warmup"))
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
    telemetry.append(_gpu_snapshot("before_measured_cycle_1"))
    samples: List[Dict[str, Any]] = []
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
        sample["iteration"] = iteration
        samples.append(sample)
        telemetry.append(_gpu_snapshot(f"after_measured_cycle_{iteration + 1}"))
    telemetry.append(_gpu_snapshot("final"))
    latency = [float(row["batch_latency_ms"]) for row in samples]
    output_throughput = [float(row["output_token_throughput_per_second"]) for row in samples]
    total_throughput = [float(row["total_token_throughput_per_second"]) for row in samples]
    device_names = [str(torch.cuda.get_device_name(index)) for index in range(4)]
    environment: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "nccl": _nccl_version(torch.cuda.nccl.version()),
        "visible_gpu_count": torch.cuda.device_count(),
        "device_names": device_names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "nccl_configuration": {
            name: os.environ.get(name)
            for name in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_NET", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
        },
    }
    raw: Dict[str, Any] = {
        "format": FORMAT,
        "scope": {
            "study": "issue-2971-version-sensitivity-on-rostam-a100-tp4",
            "application": "actual-openhermes-2.5-mistral-7b-at-issue-date-revision",
            "hardware": "four-a100-gpus-not-the-reported-single-rtx-4090",
            "request_stream": "fixed-offline-batches-not-the-unreported-original-server-request-stream",
        },
        "configuration_id": args.configuration_id,
        "image_manifest_digest": args.image_manifest_digest,
        "sif_sha256": args.sif_sha256,
        "vllm_version": str(vllm.__version__),
        "enforce_eager": bool(args.enforce_eager),
        "workload": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "batch_size": args.batch_size,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "max_model_len": args.max_model_len,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "seed": args.seed,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "swap_space_gib": args.swap_space_gib,
        },
        "engine_initialization_seconds": engine_initialization_seconds,
        "samples": samples,
        "summary": {
            "median_batch_latency_ms": statistics.median(latency),
            "p99_batch_latency_ms": _percentile(latency, 0.99),
            "median_output_token_throughput_per_second": statistics.median(output_throughput),
            "median_total_token_throughput_per_second": statistics.median(total_throughput),
        },
        "environment": environment,
        "environment_sha256": hashlib.sha256(_canonical_bytes(environment)).hexdigest(),
        "telemetry": telemetry,
    }
    raw["measurement_id"] = hashlib.sha256(_canonical_bytes(raw)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_canonical_bytes(raw) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(output, 0o444)
    print(json.dumps({"measurement_id": raw["measurement_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
