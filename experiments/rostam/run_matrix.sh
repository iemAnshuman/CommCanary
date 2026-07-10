#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_JSON="$EXP_DIR/configs.json"
# Submit from the repo root so SLURM_SUBMIT_DIR-based resolution inside the
# spooled sbatch scripts and the relative #SBATCH -o log path both work.
cd "$EXP_DIR/../.."
REPS="${REPS:-${1:-5}}"
WORKLOADS="${WORKLOADS:-micro full canary}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG_NAMES=()
while IFS= read -r config_name; do
  CONFIG_NAMES+=("$config_name")
done < <(python3 - "$CONFIGS_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for config in json.load(handle):
        print(config["name"])
PY
)

submit_job() {
  local workload="$1"
  local config="$2"
  local rep="$3"
  local dependency="$4"
  local script="$EXP_DIR/run_${workload}.sbatch"
  if [[ ! -f "$script" ]]; then
    echo "unknown workload script: $script" >&2
    exit 2
  fi
  local cmd=(sbatch --parsable)
  if [[ -n "$dependency" ]]; then
    # afterany, not afterok: a single failed cell must not leave the rest of
    # the sweep pending forever; failures are recorded in each cell's status.
    cmd+=(--dependency="afterany:$dependency")
  fi
  cmd+=("$script" "$config" "$rep")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    JOB_ID="dryrun-${workload}-${config}-rep${rep}"
  else
    JOB_ID="$("${cmd[@]}")"
    JOB_ID="${JOB_ID%%;*}"
  fi
}

previous_job=""
for ((rep = 0; rep < REPS; rep++)); do
  for config in "${CONFIG_NAMES[@]}"; do
    for workload in $WORKLOADS; do
      JOB_ID=""
      submit_job "$workload" "$config" "$rep" "$previous_job"
      previous_job="$JOB_ID"
      echo "submitted $workload config=$config rep=$rep job=$previous_job"
    done
  done
done
