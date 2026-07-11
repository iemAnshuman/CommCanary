#!/usr/bin/env bash
# Shared capture and replay are separate frozen campaigns. The replay manifest
# must bind the selected capture's exact param-trace path and SHA-256 as the
# `shared-param-trace` input; no shared results directory is searched.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "plan" || "${1:-}" == "submit" ]]; then
  exec python3 -m experiments.rostam.lib.submission "$@"
fi
exec python3 -m experiments.rostam.lib.submission plan \
  --experiment-directory "$EXP_DIR" "$@"
