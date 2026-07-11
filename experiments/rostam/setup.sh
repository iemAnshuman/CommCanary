#!/usr/bin/env bash
# Reproduce the reviewed Rostam user-space environments. This script is
# intentionally fail-closed: unresolved locks, missing hashes, a dirty/wrong
# PARAM checkout, or a stale CommCanary wheel abort before mkdir/venv/install.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PARAM_DIR="${PARAM_DIR:-$EXP_DIR/third_party/param}"
VENVS_DIR="$EXP_DIR/venvs"
PATCH_PATH="$EXP_DIR/patches/param-use-triton-default.patch"

: "${COMMCANARY_WHEEL:?set COMMCANARY_WHEEL to the reviewed wheel path}"
: "${COMMCANARY_WHEEL_SHA256:?set COMMCANARY_WHEEL_SHA256 to its reviewed digest}"

cd "$REPO_ROOT"

# Static contract hashes and every target-observed field are checked before
# this script creates or modifies anything.
"$PYTHON_BIN" -m experiments.rostam.lib.environment_contract \
  --experiment-dir "$EXP_DIR" verify-ready \
  --wheel "$COMMCANARY_WHEEL" \
  --wheel-sha256 "$COMMCANARY_WHEEL_SHA256"
"$PYTHON_BIN" -m experiments.rostam.lib.environment_contract \
  --experiment-dir "$EXP_DIR" verify-param-preimage \
  --param-dir "$PARAM_DIR"

if [[ -e "$VENVS_DIR/nccl-2.19.3" || -e "$VENVS_DIR/nccl-2.20.5" ]]; then
  echo "refusing to reuse an existing Rostam venv; archive it and rerun setup" >&2
  exit 1
fi

mkdir -p "$VENVS_DIR"

create_reviewed_venv() {
  local environment_id="$1"
  local lock="$EXP_DIR/constraints/locks/${environment_id}.lock.txt"
  local venv="$VENVS_DIR/$environment_id"
  local python="$venv/bin/python"
  local freeze_file
  local wheel_digest

  "$PYTHON_BIN" -m venv "$venv"
  # The complete lock enumerates every transitive wheel. --no-deps is required
  # for the reviewed torch-2.4.1/NCCL-2.19.3 substitution; --require-hashes
  # prevents an index or cache from silently changing any artifact.
  "$python" -m pip install --no-deps --require-hashes -r "$lock"
  "$python" -m pip install --no-deps "$COMMCANARY_WHEEL"

  # Record which wheel this venv actually holds so the cell entrypoint can
  # refuse a stale venv whose install predates the manifest-bound wheel. The
  # digest is recomputed at install time and must still match the reviewed one.
  wheel_digest="$(sha256sum "$COMMCANARY_WHEEL" | awk '{print $1}')"
  if [[ "$wheel_digest" != "$COMMCANARY_WHEEL_SHA256" ]]; then
    echo "CommCanary wheel changed after verify-ready; refusing to record a stale binding" >&2
    exit 1
  fi
  printf '%s\n' "$wheel_digest" >"$venv/commcanary-wheel.sha256"

  freeze_file="$(mktemp "${TMPDIR:-/tmp}/commcanary-${environment_id}-freeze.XXXXXX")"
  trap 'rm -f "$freeze_file"' RETURN
  "$python" -m pip freeze --all >"$freeze_file"
  "$PYTHON_BIN" -m experiments.rostam.lib.environment_contract \
    --experiment-dir "$EXP_DIR" verify-freeze \
    --environment-id "$environment_id" \
    --freeze "$freeze_file"
  rm -f "$freeze_file"
  trap - RETURN
}

create_reviewed_venv "nccl-2.19.3"
create_reviewed_venv "nccl-2.20.5"

# The preimage and commit were already verified. git applies the committed
# patch only after a dry check; a postimage hash proves the exact mutation.
git -C "$PARAM_DIR" apply --check "$PATCH_PATH"
git -C "$PARAM_DIR" apply "$PATCH_PATH"
"$PYTHON_BIN" -m experiments.rostam.lib.environment_contract \
  --experiment-dir "$EXP_DIR" verify-param-postimage \
  --param-dir "$PARAM_DIR"

PARAM_LINK="$EXP_DIR/third_party/param_bench"
if [[ -L "$PARAM_LINK" ]]; then
  if [[ "$(readlink "$PARAM_LINK")" != "param" ]]; then
    echo "existing PARAM compatibility link has an unexpected target" >&2
    exit 1
  fi
elif [[ -e "$PARAM_LINK" ]]; then
  echo "PARAM compatibility path exists and is not a symlink" >&2
  exit 1
else
  ln -s "param" "$PARAM_LINK"
fi

echo "reviewed Rostam environments and PARAM patch installed"
