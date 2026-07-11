#!/usr/bin/env bash
# Common body for spooled SLURM wrappers. Site resources are supplied by the
# manifest-bound sbatch argv; this layer only fixes wrapper identity and starts
# the reviewed venv interpreter without eval, heredocs, or a nested shell.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: common.sh <wrapper-kind> <venv-python> [cell arguments...]" >&2
  exit 2
fi

WRAPPER_KIND="$1"
PYTHON_EXECUTABLE="$2"
shift 2

: "${COMMCANARY_EXPERIMENT_DIR:?planner must export COMMCANARY_EXPERIMENT_DIR}"

if [[ ! -d "$COMMCANARY_EXPERIMENT_DIR/lib" || -L "$COMMCANARY_EXPERIMENT_DIR" ]]; then
  echo "unsafe or missing experiment directory: $COMMCANARY_EXPERIMENT_DIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "unsafe or missing reviewed venv interpreter: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

REPO_ROOT="$(cd "$COMMCANARY_EXPERIMENT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
exec "$PYTHON_EXECUTABLE" -m experiments.rostam.lib.cell_entrypoint \
  --site-wrapper "$WRAPPER_KIND" "$@"
