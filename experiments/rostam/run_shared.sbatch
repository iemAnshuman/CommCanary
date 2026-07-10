#!/usr/bin/env bash
#SBATCH -p cuda-A100
#SBATCH -w toranj0
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J rostam-shared
#SBATCH -o experiments/rostam/results/slurm-%x-%j.out

# Replay the ONE shared canary (captured by capture_shared_trace.sbatch under a
# reference config) under THIS config's NCCL settings. Same trace + same gaps
# for every config; only the NCCL execution varies. This is the operational
# "record once, evaluate candidates" test the per-cell canary sweeps do not
# provide. Workload = W-canary-shared-overlap.
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
REP="${REP:-${2:-0}}"
if [[ -z "$CONFIG_NAME" ]]; then
  echo "usage: CONFIG_NAME=<name> REP=<n> sbatch run_shared.sbatch" >&2
  exit 2
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/experiments/rostam/configs.json" ]]; then
  EXP_DIR="${SLURM_SUBMIT_DIR}/experiments/rostam"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/configs.json" ]]; then
  EXP_DIR="${SLURM_SUBMIT_DIR}"
else
  EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "$EXP_DIR/configs.json" ]]; then
  echo "cannot locate experiments/rostam/configs.json; submit from the repo root" >&2
  exit 2
fi
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
RESULTS_DIR="$EXP_DIR/results"
SHARED_DIR="$RESULTS_DIR/shared"
SHARED_TRACE="$SHARED_DIR/param_trace_overlap.json"
if [[ ! -f "$SHARED_TRACE" ]]; then
  echo "missing shared trace $SHARED_TRACE; run capture_shared_trace.sbatch first" >&2
  exit 1
fi

module load python/3.12.3 2>/dev/null || true

eval "$(python3 - "$EXP_DIR/configs.json" "$CONFIG_NAME" <<'PY'
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

source "$EXP_DIR/venvs/$CONFIG_VENV/bin/activate"
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

NPROC="${NPROC:-4}"
OUT_PREFIX="$RESULTS_DIR/shared-${CONFIG_NAME}-rep${REP}-job${SLURM_JOB_ID:-local}"
STDOUT_PATH="$OUT_PREFIX.stdout.txt"
STDERR_PATH="$OUT_PREFIX.stderr.txt"
RESULT_PATH="$OUT_PREFIX.json"

START_NS="$(python -c 'import time; print(time.time_ns())')"
set +e
torchrun --standalone --nproc_per_node="${NPROC}" "$EXP_DIR/overlap_replay.py" \
  --trace-path "$SHARED_TRACE" --device cuda --backend nccl --iters 1 --warmup 1 \
  >"$STDOUT_PATH" 2>"$STDERR_PATH"
STATUS="$?"
set -e
END_NS="$(python -c 'import time; print(time.time_ns())')"

python - "$RESULT_PATH" "$CONFIG_JSON" "$CONFIG_NAME" "$REP" "$STATUS" "$START_NS" "$END_NS" "$STDOUT_PATH" "$STDERR_PATH" "$SHARED_TRACE" <<'PY'
import json
import os
import socket
import sys

(
    result_path,
    config_json,
    config_name,
    rep,
    status,
    start_ns,
    end_ns,
    stdout_path,
    stderr_path,
    shared_trace,
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
# Fail closed: overlap_replay prints a rank-0 JSON with real per-op latencies;
# its absence is a measurement failure, not a data point.
parse_failed = payload is None
if payload is None:
    payload = {"timings_us": [], "metrics": {"median_us": None, "iqr_us": None, "count": 0}}

def torch_fingerprint():
    try:
        import ctypes

        import torch
        nccl_version = None
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            nccl_version = torch.cuda.nccl.version()
        info = {"torch_version": torch.__version__, "torch_nccl_version": nccl_version}
        try:
            mapped = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libnccl.so" in l})
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
result = {
    "schema": "commcanary.rostam.result.v1",
    "workload": "W-canary-shared-overlap",
    "config": config,
    "config_name": config_name,
    "rep": int(rep),
    "status": int(status),
    "parse_failed": parse_failed,
    "command": "torchrun overlap_replay.py --trace-path " + os.path.basename(shared_trace),
    "timings_us": payload.get("timings_us", []),
    "metrics": metrics,
    "workload_stdout": payload,
    "shared_trace": shared_trace,
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
