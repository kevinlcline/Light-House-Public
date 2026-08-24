#!/usr/bin/env bash
# Cron-friendly wrapper around deploy/pull_main.sh (logging + timestamps).
# Does not restart Light-House — Kevin restarts from the UI when ready.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LIGHT_HOUSE_PULL_LOG_DIR:-${ROOT}/data/logs}"
LOG_FILE="${LOG_DIR}/pull_main.log"

mkdir -p "${LOG_DIR}"

{
  echo "=== $(date -Is) cron_pull_main start ==="
  ec=0
  "${ROOT}/deploy/pull_main.sh" || ec=$?
  echo "=== cron_pull_main exit=${ec} ==="
} >>"${LOG_FILE}" 2>&1
