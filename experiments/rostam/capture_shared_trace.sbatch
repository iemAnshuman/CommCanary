#!/usr/bin/env bash
set -euo pipefail
: "${COMMCANARY_EXPERIMENT_DIR:?submitted plan did not export the experiment directory}"
exec "$COMMCANARY_EXPERIMENT_DIR/lib/common.sh" shared-capture "$@"
