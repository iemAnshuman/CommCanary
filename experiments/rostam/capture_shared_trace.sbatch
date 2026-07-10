#!/usr/bin/env bash
#SBATCH -p cuda-A100
#SBATCH -w toranj0
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J rostam-capture-shared
#SBATCH -o experiments/rostam/results/slurm-%x-%j.out

# Capture ONE canary under a reference config, once. run_shared.sbatch then
# replays THIS fixed artifact across all candidate configs -- the operational
# "record once, evaluate candidates" scenario, and the isolation the per-cell
# canary sweeps lack (there each config profiled its own W-full run).
set -euo pipefail

REFERENCE_CONFIG="${REFERENCE_CONFIG:-${1:-nccl-2.20.5-default}}"

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
SHARED_DIR="$EXP_DIR/results/shared"
mkdir -p "$SHARED_DIR"

module load python/3.12.3 2>/dev/null || true

eval "$(python3 - "$EXP_DIR/configs.json" "$REFERENCE_CONFIG" <<'PY'
import json
import shlex
import sys

path, name = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    configs = json.load(handle)
for config in configs:
    if config["name"] == name:
        print("CONFIG_VENV=" + shlex.quote(config["venv"]))
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
LAYERS="${LAYERS:-32}"
TOKENS="${TOKENS:-256}"
HIDDEN="${HIDDEN:-8192}"
DTYPE="${DTYPE:-bf16}"
MSG_SIZES="${MSG_SIZES:-64K,128K,256K}"
WARMUP="${WARMUP:-1}"
GEMM_M="${GEMM_M:-256}"
COMPUTE_FILL_GEMM_DIM="${COMPUTE_FILL_GEMM_DIM:-1024}"

PROFILE_JSON="$SHARED_DIR/profile.json"
TRACE_JSON="$SHARED_DIR/trace.json"
CANARY_JSON="$SHARED_DIR/canary.json"
META_JSON="$SHARED_DIR/shared_meta.json"

# calibrate the compute-fill quantum on this node (bf16), same as run_canary
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
echo "shared capture: reference=$REFERENCE_CONFIG, compute-fill=${COMPUTE_FILL_US_PER_GEMM}us/gemm"

torchrun --standalone --nproc_per_node="${NPROC}" "$EXP_DIR/workload_tp8.py" \
  --layers "$LAYERS" --tokens "$TOKENS" --hidden "$HIDDEN" --gemm-m "$GEMM_M" \
  --dtype "$DTYPE" --msg-sizes "$MSG_SIZES" --warmup "$WARMUP" \
  --profile "$PROFILE_JSON" >"$SHARED_DIR/capture.stdout.txt" 2>"$SHARED_DIR/capture.stderr.txt"

commcanary import-kineto "$PROFILE_JSON" -o "$TRACE_JSON" --workload-name "rostam-tp8-shared"
commcanary compile "$TRACE_JSON" -o "$CANARY_JSON" --require-behavior-verification --timing-sample-limit 64

# the one artifact all candidate configs will share, in overlap grammar
commcanary export-param "$CANARY_JSON" -o "$SHARED_DIR/param_trace_overlap.json" --dtype bfloat16 --skip-unsupported \
  --compute-fill-us-per-gemm "$COMPUTE_FILL_US_PER_GEMM" --compute-fill-gemm-dim "$COMPUTE_FILL_GEMM_DIM" \
  --overlap-structure

python - "$META_JSON" "$REFERENCE_CONFIG" "$COMPUTE_FILL_US_PER_GEMM" "$COMPUTE_FILL_GEMM_DIM" <<'PY'
import json
import sys

meta_path, reference, fill_us, gemm_dim = sys.argv[1:]
with open(meta_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "reference_config": reference,
            "compute_fill_us_per_gemm": float(fill_us),
            "compute_fill_gemm_dim": int(gemm_dim),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
echo "shared overlap trace ready: $SHARED_DIR/param_trace_overlap.json"
