#!/usr/bin/env bash
# Single-source-trace sweep: capture ONE canary under a reference config, then
# replay that fixed artifact across all configs x reps. Chains a one-time
# capture job, then rep-major interleaved shared-replay cells: each cell
# requires the capture to succeed (afterok) and serializes on the previous
# cell (afterany, so one failed cell does not wedge the rest).
# Submit from the repo root. DRY_RUN=1 prints the plan without submitting.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_DIR/../.."
CONFIGS_JSON="$EXP_DIR/configs.json"
REPS="${REPS:-${1:-5}}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-nccl-2.20.5-default}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG_NAMES=()
while IFS= read -r name; do
  CONFIG_NAMES+=("$name")
done < <(python3 -c "import json,sys; [print(c['name']) for c in json.load(open(sys.argv[1]))]" "$CONFIGS_JSON")

if [[ "$DRY_RUN" == "1" ]]; then
  echo "REFERENCE_CONFIG=$REFERENCE_CONFIG sbatch --parsable experiments/rostam/capture_shared_trace.sbatch $REFERENCE_CONFIG"
  CAP_JOB="dryrun-capture"
else
  CAP_JOB="$(REFERENCE_CONFIG="$REFERENCE_CONFIG" sbatch --parsable experiments/rostam/capture_shared_trace.sbatch "$REFERENCE_CONFIG")"
  CAP_JOB="${CAP_JOB%%;*}"
fi
echo "capture job=$CAP_JOB reference=$REFERENCE_CONFIG"

prev="$CAP_JOB"
for ((rep = 0; rep < REPS; rep++)); do
  for config in "${CONFIG_NAMES[@]}"; do
    cmd=(sbatch --parsable --dependency="afterok:$CAP_JOB,afterany:$prev"
         experiments/rostam/run_shared.sbatch "$config" "$rep")
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; printf '\n'
      prev="dryrun-shared-$config-rep$rep"
    else
      prev="$("${cmd[@]}")"; prev="${prev%%;*}"
    fi
    echo "submitted shared config=$config rep=$rep job=$prev"
  done
done
