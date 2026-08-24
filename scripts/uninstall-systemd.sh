#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="light-house"
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
  sudo systemctl stop "${SERVICE_NAME}"
fi

if systemctl is-enabled --quiet "${SERVICE_NAME}" 2>/dev/null; then
  sudo systemctl disable "${SERVICE_NAME}"
fi

if [[ -f "${UNIT}" ]]; then
  sudo rm -f "${UNIT}"
  sudo systemctl daemon-reload
  echo "Removed ${UNIT}"
else
  echo "No unit file at ${UNIT}"
fi
