#!/usr/bin/env bash
# Install or remove the hourly git-pull cron job for this machine.
#
# Usage:
#   bash deploy/install-pull-cron.sh          # install (default)
#   bash deploy/install-pull-cron.sh remove   # uninstall

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPPER="${ROOT}/deploy/cron_pull_main.sh"
MARKER="Light-House hourly pull_main"
SCHEDULE="${LIGHT_HOUSE_PULL_CRON_SCHEDULE:-0 * * * *}"

if [[ ! -x "${WRAPPER}" ]]; then
  echo "error: missing executable ${WRAPPER}" >&2
  exit 1
fi

CRON_LINE="${SCHEDULE} flock -n ${ROOT}/data/logs/pull_main.lock ${WRAPPER} # ${MARKER}"

_remove() {
  if ! crontab -l >/dev/null 2>&1; then
    echo "No crontab for $(whoami); nothing to remove."
    return 0
  fi
  if ! crontab -l | grep -Fq "${MARKER}"; then
    echo "Cron job not installed (${MARKER})."
    return 0
  fi
  crontab -l | grep -Fv "${MARKER}" | crontab -
  echo "Removed hourly pull cron job."
}

_install() {
  mkdir -p "${ROOT}/data/logs"
  existing=""
  if crontab -l >/dev/null 2>&1; then
    existing="$(crontab -l)"
  fi
  if printf '%s\n' "${existing}" | grep -Fq "${MARKER}"; then
    echo "Cron job already installed:"
    printf '%s\n' "${existing}" | grep -F "${MARKER}"
    exit 0
  fi
  {
    printf '%s\n' "${existing}" | sed '/^$/d'
    echo "${CRON_LINE}"
  } | crontab -
  echo "Installed hourly pull cron job for $(whoami):"
  echo "  ${CRON_LINE}"
  echo
  echo "Log: ${ROOT}/data/logs/pull_main.log"
  echo "Restart Light-House from the UI after a pull updates main."
}

case "${1:-install}" in
  install) _install ;;
  remove | uninstall) _remove ;;
  *)
    echo "Usage: $0 [install|remove]" >&2
    exit 1
    ;;
esac
