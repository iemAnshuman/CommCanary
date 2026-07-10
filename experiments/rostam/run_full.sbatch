#!/usr/bin/env bash
#SBATCH -p cuda-A100
# Pinned node: the cuda-A100 partition is heterogeneous (nasrin0 has only
# 2 GPUs and a different driver); toranj0 has 4x A100-PCIE-40GB. One node
# for every cell also removes cross-node variance from the rankings.
#SBATCH -w toranj0
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -J rostam-full
#SBATCH -o experiments/rostam/results/slurm-%x-%j.out

set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
REP="${REP:-${2:-0}}"
if [[ -z "$CONFIG_NAME" ]]; then
  echo "usage: CONFIG_NAME=<name> REP=<n> sbatch run_full.sbatch" >&2
  echo "   or: sbatch run_full.sbatch <CONFIG_NAME> <REP>" >&2
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

OUT_PREFIX="$RESULTS_DIR/full-${CONFIG_NAME}-rep${REP}-job${SLURM_JOB_ID:-local}"
STDOUT_PATH="$OUT_PREFIX.stdout.txt"
STDERR_PATH="$OUT_PREFIX.stderr.txt"
RESULT_PATH="$OUT_PREFIX.json"

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

START_NS="$(python - <<'PY'
import time
print(time.time_ns())
PY
)"
set +e
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
  >"$STDOUT_PATH" 2>"$STDERR_PATH"
STATUS="$?"
set -e
END_NS="$(python - <<'PY'
import time
print(time.time_ns())
PY
)"

COMMAND_DISPLAY="torchrun --standalone --nproc_per_node=${NPROC} workload_tp8.py"
python - "$RESULT_PATH" "$CONFIG_JSON" "$CONFIG_NAME" "$REP" "$STATUS" "$START_NS" "$END_NS" "$STDOUT_PATH" "$STDERR_PATH" "$COMMAND_DISPLAY" <<'PY'
import json
import os
import socket
import subprocess
import sys

result_path, config_json, config_name, rep, status, start_ns, end_ns, stdout_path, stderr_path, command_display = sys.argv[1:]
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
        completed = subprocess.run(
            ["nvidia-smi", "-q", "-d", "CLOCK"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except OSError as exc:
        return {"error": str(exc)}

wall_time_s = (int(end_ns) - int(start_ns)) / 1_000_000_000.0
metrics = dict(payload.get("metrics", {}))
metrics["wall_time_s"] = round(wall_time_s, 6)
result = {
    "schema": "commcanary.rostam.result.v1",
    "workload": "W-full",
    "config": config,
    "config_name": config_name,
    "rep": int(rep),
    "status": int(status),
    "command": command_display,
    "timings_us": payload.get("timings_us", []),
    "metrics": metrics,
    "workload_stdout": payload,
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
PY

exit "$STATUS"
