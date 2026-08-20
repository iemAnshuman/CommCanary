#!/usr/bin/env bash
# Four-rank NCCL conformance check for the physical runner's collective buffer
# pool and skew mechanism. This is a runner smoke test: it issues no
# qualification verdict, freezes no campaign, and writes nothing under any
# evidence directory.
#
#   sbatch --export=ALL,SHIP_DIR=...,VENV_PYTHON=... smoke_buffer_pool.sbatch
#
#SBATCH --job-name=cc-smoke-bufpool
#SBATCH --partition=cuda-A100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
set -euo pipefail

: "${SHIP_DIR:?set SHIP_DIR to the CommCanary checkout under test}"
: "${VENV_PYTHON:?set VENV_PYTHON to an interpreter with torch and CUDA}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "missing or non-executable interpreter: $VENV_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$SHIP_DIR/src/commcanary/execution/physical_runner.py" ]]; then
  echo "SHIP_DIR does not look like a CommCanary checkout: $SHIP_DIR" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-$SHIP_DIR/out/smoke-bufpool}"
mkdir -p "$OUTPUT_DIR"

echo "node        $(hostname)"
echo "job         ${SLURM_JOB_ID:-none}"
echo "ship dir    $SHIP_DIR"
echo "interpreter $VENV_PYTHON"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$SHIP_DIR"
export PYTHONPATH="$SHIP_DIR/src"
export OMP_NUM_THREADS=1
export SMOKE_OUTPUT="$OUTPUT_DIR/smoke-${SLURM_JOB_ID:-local}.json"
# One process per GPU. The site exposes no GRES, so ranks bind by local rank.
# The buffer-pool defect is per-rank collective concurrency, not a world-size
# property, so any world of two or more exercises it fully.
# A wedged GPU still answers NVML, so nvidia-smi succeeds and only CUDA context
# creation fails -- and Slurm does not drain the node. Refuse here rather than
# fail obscurely thirty seconds later, as job 186026 did on toranj1.
if UNHEALTHY="$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader 2>/dev/null \
    | grep -iE 'requires reset|unknown error|fell off the bus')"; then
  echo "refusing to start: GPUs need an administrator reset ->" >&2
  echo "$UNHEALTHY" >&2
  exit 2
fi

GPUS="$("$VENV_PYTHON" -c 'import torch;print(torch.cuda.device_count())')"
RANKS="${NPROC:-$GPUS}"
echo "visible gpus $GPUS, launching $RANKS ranks"
if [[ "$RANKS" -lt 2 ]]; then
  echo "this check requires at least two visible GPUs, found $GPUS" >&2
  exit 2
fi

exec "$VENV_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$RANKS" \
  experiments/rostam/smoke_buffer_pool_worker.py
