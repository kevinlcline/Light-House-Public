#!/usr/bin/env bash
# Pull merged main from GitHub into this machine's Light-House checkout.
# Does not restart the server — Kevin restarts from the UI when ready.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not a git repository: ${ROOT}" >&2
  exit 1
fi

branch="${LIGHT_HOUSE_PULL_BRANCH:-main}"
remote="${LIGHT_HOUSE_PULL_REMOTE:-origin}"

before="$(git rev-parse --short HEAD)"
echo "pull_main: repo=${ROOT}"
echo "pull_main: branch=${branch} remote=${remote}"
echo "pull_main: before=${before}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes; commit or stash before pulling" >&2
  git status --short >&2 || true
  exit 2
fi

git fetch "${remote}" "${branch}"
git pull --ff-only "${remote}" "${branch}"

after="$(git rev-parse --short HEAD)"
echo "pull_main: after=${after}"

if [[ "${before}" == "${after}" ]]; then
  echo "pull_main: already up to date"
else
  echo "pull_main: updated ${before} -> ${after}"
  echo "pull_main: restart Light-House from the UI when you want new code live"
fi
