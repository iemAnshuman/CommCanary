#!/usr/bin/env bash
# Plan is the default. Scheduler mutation requires the separate, explicit form:
#   run_matrix.sh submit --plan ... --execute
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "plan" || "${1:-}" == "submit" ]]; then
  exec python3 -m experiments.rostam.lib.submission "$@"
fi
exec python3 -m experiments.rostam.lib.submission plan \
  --experiment-directory "$EXP_DIR" "$@"
