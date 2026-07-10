#!/usr/bin/env bash
#SBATCH -p cuda-A100
# Pinned node: the cuda-A100 partition is heterogeneous (nasrin0 has only
# 2 GPUs and a different driver); toranj0 has 4x A100-PCIE-40GB. One node
# for every cell also removes cross-node variance from the rankings.
#SBATCH -w toranj0
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J rostam-canary
#SBATCH -o experiments/rostam/results/slurm-%x-%j.out

set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
REP="${REP:-${2:-0}}"
if [[ -z "$CONFIG_NAME" ]]; then
  echo "usage: CONFIG_NAME=<name> REP=<n> sbatch run_canary.sbatch" >&2
  echo "   or: sbatch run_canary.sbatch <CONFIG_NAME> <REP>" >&2
  exit 2
fi

# sbatch spools this script to /var/lib/slurm/...; inside a job BASH_SOURCE
# points at the spool copy, not the repo. Resolve via SLURM_SUBMIT_DIR
# (submit from the repo root or from experiments/rostam); fall back to
# BASH_SOURCE only for direct `bash run_*.sbatch` execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/experiments/rostam/configs.json" ]]; then
  EXP_DIR="${SLURM_SUBMIT_DIR}/experiments/rostam"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/configs.json" ]]; then
  EXP_DIR="${SLURM_SUBMIT_DIR}"
else
  EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "$EXP_DIR/configs.json" ]]; then
  echo "cannot locate experiments/rostam/configs.json; submit this job from the repo root" >&2
  exit 2
fi
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
CONFIGS_JSON="$EXP_DIR/configs.json"
RESULTS_DIR="$EXP_DIR/results"
mkdir -p "$RESULTS_DIR"

eval "$(python3 - "$CONFIGS_JSON" "$CONFIG_NAME" <<'PY'
import json
import shlex
import sys

path, name = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    configs = json.load(handle)
for config in configs:
    if config["name"] == name:
        print("CONFIG_VENV=" + shlex.quote(config["venv"]))
        print("CONFIG_JSON=" + shlex.quote(json.dumps(config, sort_keys=True)))
        for key, value in sorted(config.get("env", {}).items()):
            print("export " + key + "=" + shlex.quote(str(value)))
        break
else:
    raise SystemExit(f"unknown config {name!r}")
PY
)"

# The venvs are built on the module python (see setup.sh); load it here too
# so the venv interpreter's runtime environment matches on compute nodes.
module load python/3.12.3 2>/dev/null || true
source "$EXP_DIR/venvs/$CONFIG_VENV/bin/activate"
# third_party gives PARAM its 'param_bench' package name (via symlink) and
# third_party/param exposes its sibling 'et_replay' package.
export PYTHONPATH="$REPO_ROOT/src:$EXP_DIR/third_party:$EXP_DIR/third_party/param:${PYTHONPATH:-}"
NCCL_LIB_DIR="$(python - <<'PY'
import glob
import os
import site

roots = site.getsitepackages() + [site.getusersitepackages()]
for root in roots:
    for path in glob.glob(os.path.join(root, "nvidia", "nccl", "lib")):
        if os.path.isdir(path):
            print(path)
            raise SystemExit(0)
PY
)"
if [[ -n "$NCCL_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

OUT_PREFIX="$RESULTS_DIR/canary-${CONFIG_NAME}-rep${REP}-job${SLURM_JOB_ID:-local}"
ARTIFACT_DIR="$RESULTS_DIR/artifacts-${CONFIG_NAME}-rep${REP}-job${SLURM_JOB_ID:-local}"
mkdir -p "$ARTIFACT_DIR"

FULL_STDOUT="$OUT_PREFIX.full.stdout.txt"
FULL_STDERR="$OUT_PREFIX.full.stderr.txt"
PROFILE_JSON="$ARTIFACT_DIR/profile.json"
TRACE_JSON="$ARTIFACT_DIR/trace.json"
CANARY_JSON="$ARTIFACT_DIR/canary.json"
PARAM_TRACE_JSON="$ARTIFACT_DIR/param_trace.json"

LAYERS="${LAYERS:-32}"
TOKENS="${TOKENS:-256}"
HIDDEN="${HIDDEN:-8192}"
NPROC="${NPROC:-4}"
DTYPE="${DTYPE:-bf16}"
MSG_SIZES="${MSG_SIZES:-64K,128K,256K}"
WARMUP="${WARMUP:-1}"
INJECT_SKEW="${INJECT_SKEW:-0.0}"
# GEMM_M=256 calibrated from the Jul 2026 smoke run: GEMM_M=16 gave ~21us
# of compute vs ~133us of all_reduce per layer (comm-dominated loops
# resemble the microbenchmark, hiding the workload-structure signal).
# 256 lands compute near ~300us/layer, a decode-realistic 2x+ of comm.
GEMM_M="${GEMM_M:-256}"
GEMM_N_ARGS=()
if [[ -n "${GEMM_N:-}" ]]; then
  GEMM_N_ARGS=(--gemm-n "$GEMM_N")
fi

torchrun --standalone --nproc_per_node="${NPROC:-4}" "$EXP_DIR/workload_tp8.py" \
  --layers "$LAYERS" \
  --tokens "$TOKENS" \
  --hidden "$HIDDEN" \
  --gemm-m "$GEMM_M" \
  "${GEMM_N_ARGS[@]}" \
  --dtype "$DTYPE" \
  --msg-sizes "$MSG_SIZES" \
  --warmup "$WARMUP" \
  --inject-skew "$INJECT_SKEW" \
  --profile "$PROFILE_JSON" \
  >"$FULL_STDOUT" 2>"$FULL_STDERR"

commcanary import-kineto "$PROFILE_JSON" -o "$TRACE_JSON" --workload-name "rostam-tp8"
# --require-behavior-verification, not --behavior-search: this workload
# cycles message sizes so every event is its own signature group (repeat=1),
# which makes all behavior-search candidates byte-identical while costing
# ~30 minutes per cell (observed, job 159697). The fail-closed behavioral
# verification gate is kept; only the pointless search is dropped.
commcanary compile "$TRACE_JSON" -o "$CANARY_JSON" \
  --require-behavior-verification \
  --timing-sample-limit 64
commcanary export-param "$CANARY_JSON" -o "$PARAM_TRACE_JSON" --dtype bfloat16 --skip-unsupported

# Expected facebookresearch/param legacy invocation, confirmed by setup.sh when
# the checkout is available:
#   torchrun ... commsTraceReplay.py --trace-type basic --trace-path <json> --use-timestamp
# If a future PARAM checkout renames these flags, set PARAM_EXTRA_ARGS or edit
# this one replay wrapper after inspecting third_party/param/train/comms/pt.
PARAM_REPLAY="$EXP_DIR/third_party/param/train/comms/pt/commsTraceReplay.py"
# --use-one-trace is required: we export ONE trace file, and without it every
# PARAM rank tries to read its own per-rank file and the replay dies at start.
PARAM_EXTRA_ARGS="${PARAM_EXTRA_ARGS:---use-timestamp --use-one-trace}"
if [[ ! -f "$PARAM_REPLAY" ]]; then
  echo "missing PARAM replay entry point: $PARAM_REPLAY; run experiments/rostam/setup.sh" >&2
  exit 1
fi

run_param_replay() {
  local workload="$1"
  local param_trace="$2"
  local result_path="$3"
  local stdout_path="$4"
  local stderr_path="$5"
  # optional 6th arg overrides the replay flags (compute-fill traces must run
  # WITHOUT --use-timestamp: pacing comes from the gemm entries themselves)
  local extra_args="${6:-$PARAM_EXTRA_ARGS}"
  python - "$workload" "$param_trace" "$result_path" "$stdout_path" "$stderr_path" "$CONFIG_JSON" "$CONFIG_NAME" "$REP" "$PARAM_REPLAY" "$extra_args" "$NPROC" <<'PY'
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

(
    workload,
    param_trace,
    result_path,
    stdout_path,
    stderr_path,
    config_json,
    config_name,
    rep,
    replay,
    extra_args,
    nproc,
) = sys.argv[1:]
config = json.loads(config_json)
command = [
    "torchrun",
    "--standalone",
    f"--nproc_per_node={nproc}",
    replay,
    "--trace-type",
    "basic",
    "--trace-path",
    param_trace,
] + shlex.split(extra_args)
start = time.perf_counter()
with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
    completed = subprocess.run(command, stdout=out, stderr=err, check=False)
wall_time_s = time.perf_counter() - start

latencies = []
# PARAM prints one line per replayed collective (format verified against a
# real local replay): "Replayed all_reduce in block [marker]... 21.21 us".
# These per-op samples ARE the latency distribution we want.
pattern = re.compile(r"Replayed\s+\S+\s+in block \[[^\]]*\]\.\.\.\s*([0-9]+(?:\.[0-9]+)?)\s*us")
for path in (stdout_path, stderr_path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # Warm-up pass and gemm compute-fill lines are not comm latency.
            if "[Warm-up]" in line or "compute-fill" in line:
                continue
            for value in pattern.findall(line):
                latencies.append(float(value))
# Fail closed: no per-op latency lines means the replay did not produce
# measurable collectives (crash, format change, empty trace). Fabricating a
# wall-time sample would silently inject one giant bogus latency into the
# ranking, so we record an explicit parse-failure marker instead and leave
# the distribution empty for analyze.py to drop.
parse_failed = not latencies

def median(values):
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)

def iqr(values):
    values = sorted(values)
    if len(values) < 2:
        return 0.0
    mid = len(values) // 2
    lower = values[:mid]
    upper = values[mid + (len(values) % 2):]
    return median(upper) - median(lower) if lower and upper else 0.0

def torch_fingerprint():
    try:
        import torch
        nccl_version = None
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            nccl_version = torch.cuda.nccl.version()
        info = {
            "torch_version": torch.__version__,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            # torch 2.2 reports its compile-time NCCL header here, NOT the
            # loaded library; the runtime fields below are authoritative.
            "torch_nccl_version": nccl_version,
        }
        try:
            import ctypes
            mapped = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libnccl.so" in line})
            info["loaded_libnccl_paths"] = mapped
            if mapped:
                lib = ctypes.CDLL(mapped[0])
                value = ctypes.c_int()
                lib.ncclGetVersion(ctypes.byref(value))
                # 21903 = 2.19.3, 22005 = 2.20.5
                info["runtime_nccl_version_code"] = value.value
        except Exception as exc:
            info["runtime_nccl_error"] = str(exc)
        return info
    except Exception as exc:
        return {"torch_error": str(exc)}

def nvidia_smi_clocks():
    try:
        completed_smi = subprocess.run(
            ["nvidia-smi", "-q", "-d", "CLOCK"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {"returncode": completed_smi.returncode, "stdout": completed_smi.stdout, "stderr": completed_smi.stderr}
    except OSError as exc:
        return {"error": str(exc)}

artifacts = {}
for label, path in (
    ("param_trace_bytes", param_trace),
    ("stdout_bytes", stdout_path),
    ("stderr_bytes", stderr_path),
):
    try:
        artifacts[label] = os.path.getsize(path)
    except OSError:
        artifacts[label] = 0
artifact_dir = os.path.dirname(param_trace)
param_name = os.path.basename(param_trace)
if param_name == "param_trace.json":
    source_trace = os.path.join(artifact_dir, "trace.json")
    source_canary = os.path.join(artifact_dir, "canary.json")
elif param_name.endswith("_param_trace.json"):
    stem = param_name[: -len("_param_trace.json")]
    source_trace = os.path.join(artifact_dir, stem + "_trace.json")
    source_canary = os.path.join(artifact_dir, stem + "_canary.json")
else:
    source_trace = os.path.join(artifact_dir, "trace.json")
    source_canary = os.path.join(artifact_dir, "canary.json")
for label, path in (
    ("raw_trace_bytes", source_trace),
    ("canary_bytes", source_canary),
    ("profile_bytes", os.path.join(artifact_dir, "profile.json")),
):
    try:
        artifacts[label] = os.path.getsize(path)
    except OSError:
        pass

result = {
    "schema": "commcanary.rostam.result.v1",
    "workload": workload,
    "config": config,
    "config_name": config_name,
    "rep": int(rep),
    "status": completed.returncode,
    "command": command,
    "timings_us": [round(value, 3) for value in latencies],
    "parse_failed": parse_failed,
    "metrics": {
        "median_us": round(median(latencies), 3) if latencies else None,
        "iqr_us": round(iqr(latencies), 3) if latencies else None,
        "count": len(latencies),
        "wall_time_s": round(wall_time_s, 6),
    },
    "artifacts": artifacts,
    "logs": {"stdout_path": stdout_path, "stderr_path": stderr_path},
    "fingerprint": {
        **torch_fingerprint(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_clocks": nvidia_smi_clocks(),
    },
}
tmp = result_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, result_path)
raise SystemExit(completed.returncode)
PY
}

# A failed replay must not abort the remaining replays: every workload is
# attempted, each records its own status, and the job exits nonzero at
# the end if anything failed.
REPLAY_FAILURES=0
run_param_replay "W-canary" "$PARAM_TRACE_JSON" "$OUT_PREFIX.canary.json" "$OUT_PREFIX.canary.stdout.txt" "$OUT_PREFIX.canary.stderr.txt" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))

# Compute-filled variant: inter-collective gaps become calibrated gemm
# entries so the replay reproduces compute/communication interference
# instead of idle sleeps. Replayed WITHOUT --use-timestamp by design:
# pacing must come from the compute, not wall-clock sleeps.
# The per-gemm duration is MEASURED on this node (bf16, matching the entry
# dtype the exporter emits) instead of assumed -- CUDA-event timed median
# of 50 kernels after warmup.
COMPUTE_FILL_GEMM_DIM="${COMPUTE_FILL_GEMM_DIM:-1024}"
if [[ -z "${COMPUTE_FILL_US_PER_GEMM:-}" ]]; then
  COMPUTE_FILL_US_PER_GEMM="$(python - "$COMPUTE_FILL_GEMM_DIM" <<'PY'
import statistics
import sys

import torch

dim = int(sys.argv[1])
a = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
b = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    torch.mm(a, b)
torch.cuda.synchronize()
samples = []
for _ in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    samples.append(start.elapsed_time(end) * 1000.0)
print(round(statistics.median(samples), 2))
PY
)"
fi
echo "compute-fill calibration: ${COMPUTE_FILL_US_PER_GEMM} us per gemm (dim ${COMPUTE_FILL_GEMM_DIM}, bf16)"
PARAM_TRACE_COMPUTE_JSON="$ARTIFACT_DIR/param_trace_compute.json"
commcanary export-param "$CANARY_JSON" -o "$PARAM_TRACE_COMPUTE_JSON" --dtype bfloat16 --skip-unsupported \
  --compute-fill-us-per-gemm "$COMPUTE_FILL_US_PER_GEMM" --compute-fill-gemm-dim "$COMPUTE_FILL_GEMM_DIM"
run_param_replay "W-canary-compute" "$PARAM_TRACE_COMPUTE_JSON" "$OUT_PREFIX.canary-compute.json" "$OUT_PREFIX.canary-compute.stdout.txt" "$OUT_PREFIX.canary-compute.stderr.txt" "--use-one-trace" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))

for method in stratified random; do
  baseline_trace="$ARTIFACT_DIR/baseline_${method}_trace.json"
  baseline_canary="$ARTIFACT_DIR/baseline_${method}_canary.json"
  baseline_param="$ARTIFACT_DIR/baseline_${method}_param_trace.json"
  commcanary baseline "$TRACE_JSON" -o "$baseline_trace" --method "$method" --seed "$REP"
  commcanary compile "$baseline_trace" -o "$baseline_canary" --timing-sample-limit 64
  commcanary export-param "$baseline_canary" -o "$baseline_param" --dtype bfloat16 --skip-unsupported
  run_param_replay "W-baseline-$method" "$baseline_param" "$OUT_PREFIX.baseline-${method}.json" "$OUT_PREFIX.baseline-${method}.stdout.txt" "$OUT_PREFIX.baseline-${method}.stderr.txt" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))
done

# Overlap-structured variants: same compute fill, but collectives are issued
# async with explicit waits placed after the next gap's gemms, reconstructing
# compute/communication CONCURRENCY. PARAM's replayer is hardwired blocking,
# so these run under our overlap-aware reference replayer (overlap_replay.py)
# consuming the same exported trace format. Motivated by sweep 2: serialized
# compute-fill changed clock state, not contention state (all configs ~5-10%
# faster, tree-ll still top-3 vs dead-last in W-full).
PARAM_TRACE_OVERLAP_JSON="$ARTIFACT_DIR/param_trace_overlap.json"
commcanary export-param "$CANARY_JSON" -o "$PARAM_TRACE_OVERLAP_JSON" --dtype bfloat16 --skip-unsupported \
  --compute-fill-us-per-gemm "$COMPUTE_FILL_US_PER_GEMM" --compute-fill-gemm-dim "$COMPUTE_FILL_GEMM_DIM" \
  --overlap-structure

run_overlap_replay() {
  local workload="$1"
  local trace_json="$2"
  local out_prefix="$3"
  local stdout_path="$out_prefix.stdout.txt"
  local stderr_path="$out_prefix.stderr.txt"
  local result_path="$out_prefix.json"
  local start_ns end_ns status
  start_ns="$(python -c 'import time; print(time.time_ns())')"
  set +e
  torchrun --standalone --nproc_per_node="${NPROC:-4}" "$EXP_DIR/overlap_replay.py" \
    --trace-path "$trace_json" --device cuda --backend nccl --iters 1 --warmup 1 \
    >"$stdout_path" 2>"$stderr_path"
  status="$?"
  set -e
  end_ns="$(python -c 'import time; print(time.time_ns())')"
  python - "$workload" "$trace_json" "$result_path" "$stdout_path" "$stderr_path" "$CONFIG_JSON" "$CONFIG_NAME" "$REP" "$status" "$start_ns" "$end_ns" <<'PY'
import json
import os
import socket
import subprocess
import sys

(
    workload,
    trace_json,
    result_path,
    stdout_path,
    stderr_path,
    config_json,
    config_name,
    rep,
    status,
    start_ns,
    end_ns,
) = sys.argv[1:]
config = json.loads(config_json)
payload = None
with open(stdout_path, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                pass
if payload is None:
    payload = {"timings_us": [], "metrics": {"median_us": 0.0, "iqr_us": 0.0, "count": 0}}

def torch_fingerprint():
    try:
        import ctypes

        import torch
        nccl_version = None
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            nccl_version = torch.cuda.nccl.version()
        info = {
            "torch_version": torch.__version__,
            "torch_nccl_version": nccl_version,
        }
        try:
            mapped = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libnccl.so" in line})
            info["loaded_libnccl_paths"] = mapped
            if mapped:
                lib = ctypes.CDLL(mapped[0])
                value = ctypes.c_int()
                lib.ncclGetVersion(ctypes.byref(value))
                info["runtime_nccl_version_code"] = value.value
        except Exception as exc:
            info["runtime_nccl_error"] = str(exc)
        return info
    except Exception as exc:
        return {"torch_error": str(exc)}

wall_time_s = (int(end_ns) - int(start_ns)) / 1_000_000_000.0
metrics = dict(payload.get("metrics", {}))
metrics["wall_time_s"] = round(wall_time_s, 6)
artifacts = {}
try:
    artifacts["param_trace_bytes"] = os.path.getsize(trace_json)
except OSError:
    pass
result = {
    "schema": "commcanary.rostam.result.v1",
    "workload": workload,
    "config": config,
    "config_name": config_name,
    "rep": int(rep),
    "status": int(status),
    "command": "torchrun overlap_replay.py --trace-path " + os.path.basename(trace_json),
    "timings_us": payload.get("timings_us", []),
    "metrics": metrics,
    "workload_stdout": payload,
    "artifacts": artifacts,
    "logs": {"stdout_path": stdout_path, "stderr_path": stderr_path},
    "fingerprint": {
        **torch_fingerprint(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    },
}
tmp = result_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, result_path)
raise SystemExit(int(status))
PY
}

run_overlap_replay "W-canary-overlap" "$PARAM_TRACE_OVERLAP_JSON" "$OUT_PREFIX.canary-overlap" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))

# Control: compute fill applied to the timing-destroying stratified baseline.
# If compute fill rescues THIS too, the gemms are doing all the work and
# trace fidelity contributes nothing; if it only rescues the faithful canary,
# fidelity and interference are BOTH necessary.
STRAT_COMPUTE_PARAM="$ARTIFACT_DIR/baseline_stratified_compute_param_trace.json"
commcanary export-param "$ARTIFACT_DIR/baseline_stratified_canary.json" -o "$STRAT_COMPUTE_PARAM" --dtype bfloat16 --skip-unsupported \
  --compute-fill-us-per-gemm "$COMPUTE_FILL_US_PER_GEMM" --compute-fill-gemm-dim "$COMPUTE_FILL_GEMM_DIM"
run_param_replay "W-baseline-stratified-compute" "$STRAT_COMPUTE_PARAM" "$OUT_PREFIX.baseline-stratified-compute.json" "$OUT_PREFIX.baseline-stratified-compute.stdout.txt" "$OUT_PREFIX.baseline-stratified-compute.stderr.txt" "--use-one-trace" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))

STRAT_OVERLAP_PARAM="$ARTIFACT_DIR/baseline_stratified_overlap_param_trace.json"
commcanary export-param "$ARTIFACT_DIR/baseline_stratified_canary.json" -o "$STRAT_OVERLAP_PARAM" --dtype bfloat16 --skip-unsupported \
  --compute-fill-us-per-gemm "$COMPUTE_FILL_US_PER_GEMM" --compute-fill-gemm-dim "$COMPUTE_FILL_GEMM_DIM" \
  --overlap-structure
run_overlap_replay "W-baseline-stratified-overlap" "$STRAT_OVERLAP_PARAM" "$OUT_PREFIX.baseline-stratified-overlap" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))

DDMIN_TRACE="$ARTIFACT_DIR/ddmin_trace.json"
DDMIN_CANARY="$ARTIFACT_DIR/ddmin_canary.json"
DDMIN_PARAM="$ARTIFACT_DIR/ddmin_param_trace.json"
commcanary reduce "$TRACE_JSON" -o "$DDMIN_TRACE" --timing-sample-limit 64
commcanary compile "$DDMIN_TRACE" -o "$DDMIN_CANARY" --timing-sample-limit 64
commcanary export-param "$DDMIN_CANARY" -o "$DDMIN_PARAM" --dtype bfloat16 --skip-unsupported
run_param_replay "W-baseline-ddmin" "$DDMIN_PARAM" "$OUT_PREFIX.baseline-ddmin.json" "$OUT_PREFIX.baseline-ddmin.stdout.txt" "$OUT_PREFIX.baseline-ddmin.stderr.txt" || REPLAY_FAILURES=$((REPLAY_FAILURES + 1))

if [[ "$REPLAY_FAILURES" -gt 0 ]]; then
  echo "$REPLAY_FAILURES PARAM replay(s) failed; see the *.stderr.txt files" >&2
  exit 1
fi
