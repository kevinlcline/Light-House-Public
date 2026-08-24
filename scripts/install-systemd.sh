#!/usr/bin/env bash
# Install Light-House as a system service (boot + survives desktop logout).
set -euo pipefail

LIGHT_HOUSE_HOME="${LIGHT_HOUSE_HOME:-/home/kevin/Light-House}"
SERVICE_NAME="light-house"
UNIT_DEST="/etc/systemd/system/${SERVICE_NAME}.service"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "${LIGHT_HOUSE_HOME}" ]]; then
  echo "error: LIGHT_HOUSE_HOME not found: ${LIGHT_HOUSE_HOME}" >&2
  echo "Clone the repo to ~/Light-House or set LIGHT_HOUSE_HOME=/path/to/Light-House" >&2
  exit 1
fi

if [[ ! -x "${LIGHT_HOUSE_HOME}/.venv/bin/python3" ]]; then
  echo "error: venv missing at ${LIGHT_HOUSE_HOME}/.venv" >&2
  echo "Run: cd ${LIGHT_HOUSE_HOME} && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -f "${LIGHT_HOUSE_HOME}/.env" ]]; then
  echo "error: .env missing at ${LIGHT_HOUSE_HOME}/.env" >&2
  exit 1
fi

TMP="$(mktemp)"
sed "s|/home/kevin/Light-House|${LIGHT_HOUSE_HOME}|g" \
  "${REPO_ROOT}/deploy/light-house.service" > "${TMP}"

echo "Installing ${UNIT_DEST} (home=${LIGHT_HOUSE_HOME})"
sudo cp "${TMP}" "${UNIT_DEST}"
rm -f "${TMP}"

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager || true

echo ""
echo "Light-House installed. Logs: journalctl -u ${SERVICE_NAME} -f"
