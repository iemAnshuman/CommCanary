#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
# PARAM (pinned commit) uses PEP 604 annotations (str | None) and needs
# Python >= 3.10; Rostam's default python3 is 3.9. Point PYTHON_BIN at a
# 3.10+ interpreter (e.g. PYTHON_BIN=python3.11 bash setup.sh).
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "error: $PYTHON_BIN is $("$PYTHON_BIN" -V 2>&1); PARAM needs Python >= 3.10." >&2
  echo "On Rostam: module load python/3.12.3, then PYTHON_BIN=python3 bash experiments/rostam/setup.sh" >&2
  exit 1
fi
echo "using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"
VENVS_DIR="$EXP_DIR/venvs"
THIRD_PARTY_DIR="$EXP_DIR/third_party"
PARAM_DIR="$THIRD_PARTY_DIR/param"
# Full 40-char SHA required: GitHub's fetch-by-SHA accepts only
# unabbreviated hashes ("couldn't find remote ref" otherwise).
# a437fce ("backout D70007712", 2025-03-26): the last commit where the
# legacy comms replay path is internally consistent AND torch<=2.4
# compatible. Later commits break one or the other: f2f54d3 (Jun 13) has
# comms_utils expecting args.use_device_time that commsTraceReplay's parser
# never defines; Jun 17+ imports torch.distributed._symmetric_memory
# (torch>=2.5). This commit was verified by a COMPLETE local replay of a
# CommCanary-exported trace (gloo CPU, 1 and 4 ranks).
PARAM_COMMIT="a437fcebd3add1aee66fba880f28cec9fd744589"

mkdir -p "$VENVS_DIR" "$THIRD_PARTY_DIR" "$EXP_DIR/results"

create_venv() {
  local nccl_version="$1"
  local name="nccl-$nccl_version"
  local venv="$VENVS_DIR/$name"
  local python="$venv/bin/python"

  if [[ ! -x "$python" ]]; then
    echo "creating venv: $venv"
    "$PYTHON_BIN" -m venv "$venv"
  else
    echo "reusing venv: $venv"
  fi

  "$python" -m pip install --upgrade pip setuptools wheel
  # torch 2.4.1 is pinned deliberately, on Rostam-verified evidence:
  # - it is the FIRST torch whose exported Kineto traces carry the named
  #   collective args (Collective name, In/Out msg nelems, Process Group
  #   Ranks, ...) over NCCL: probe job 159694 showed 2.3.1 emits none and
  #   2.4.1 emits all; 2.2.2 emitted none (jobs 159692/159693).
  # - its native NCCL pin is 2.20.5, and NCCL 2.19.3 link-loads under it
  #   (verified via /proc/self/maps + ncclGetVersion on the login node), so
  #   both experiment NCCLs run under ONE identical torch binary.
  # - torch 2.8 references NCCL 2.27 symbols (ncclMemFree,
  #   ncclGroupSimulateEnd) and cannot load either pinned version.
  "$python" -m pip install "torch==2.4.1"
  # torch 2.2 is incompatible with NumPy 2.x and warns without any numpy;
  # pin a compatible one so the profiler path has no soft failures.
  "$python" -m pip install "numpy<2" pydot
  "$python" -m pip install "nvidia-nccl-cu12==$nccl_version" --force-reinstall
  "$python" -m pip install -e "$REPO_ROOT"
  echo "$nccl_version" > "$venv/.commcanary-nccl-version"
}

clone_if_missing() {
  local url="$1"
  local target="$2"
  if [[ -d "$target/.git" ]]; then
    echo "reusing clone: $target"
    return 0
  fi
  if [[ -e "$target" ]]; then
    echo "error: $target exists but is not a git clone" >&2
    return 1
  fi
  echo "cloning $url -> $target"
  git clone --depth 1 "$url" "$target"
}

check_param_invocation() {
  local replay="$PARAM_DIR/train/comms/pt/commsTraceReplay.py"
  if [[ ! -f "$replay" ]]; then
    echo "warning: PARAM replay entry point not found at $replay" >&2
    return 0
  fi
  local missing=0
  for flag in "--trace-type" "--trace-path"; do
    if ! grep -R -q -- "$flag" "$replay" "$PARAM_DIR/train/comms/pt" 2>/dev/null; then
      echo "warning: PARAM replay flag $flag was not found; inspect $replay before running run_canary.sbatch" >&2
      missing=1
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    echo "confirmed PARAM legacy replay flags near $replay: --trace-type basic --trace-path <json>"
  fi
  if grep -R -q -- "--use-timestamp" "$replay" "$PARAM_DIR/train/comms/pt" 2>/dev/null; then
    echo "confirmed PARAM timestamp pacing flag: --use-timestamp"
  else
    echo "warning: PARAM --use-timestamp flag was not found; adjust PARAM_EXTRA_ARGS if this checkout differs" >&2
  fi
}

create_venv "2.19.3"
create_venv "2.20.5"

if [[ ! -d "$PARAM_DIR/.git" ]]; then
  if [[ -e "$PARAM_DIR" ]]; then
    echo "error: $PARAM_DIR exists but is not a git clone" >&2
    exit 1
  fi
  echo "cloning param -> $PARAM_DIR"
  git clone "https://github.com/facebookresearch/param.git" "$PARAM_DIR"
fi
if ! git -C "$PARAM_DIR" cat-file -e "$PARAM_COMMIT^{commit}" 2>/dev/null; then
  git -C "$PARAM_DIR" fetch --depth 1 origin "$PARAM_COMMIT"
fi
git -C "$PARAM_DIR" checkout -q "$PARAM_COMMIT"
echo "param pinned at $(git -C "$PARAM_DIR" log --format='%h %ad %s' --date=short -n 1)"
# PARAM internal-drift shim (verified necessary at this pin): the gemm kernel
# reads collectiveArgs.use_triton, which only commsComputeBench ever sets --
# commsTraceReplay does not, so compute entries crash without this default.
if grep -q 'if collectiveArgs.use_triton:' "$PARAM_DIR/train/comms/pt/pytorch_dist_backend.py"; then
  sed -i 's/if collectiveArgs.use_triton:/if getattr(collectiveArgs, "use_triton", False):/' \
    "$PARAM_DIR/train/comms/pt/pytorch_dist_backend.py"
  echo "applied use_triton compat shim to PARAM checkout"
fi
# PARAM imports itself as the package 'param_bench' and its parser needs the
# sibling 'et_replay' package; the symlink plus PYTHONPATH entries for
# third_party and third_party/param (set in run_canary.sbatch) satisfy both.
ln -sfn "param" "$THIRD_PARTY_DIR/param_bench"
check_param_invocation

echo "W-micro is pure torch, so no CUDA toolkit or nvcc build step is required."

echo "setup complete"
