#!/usr/bin/env bash
# Keep qwen2.5:14b resident in RAM for Light-House (memory curator / Ollama fallback).
# Run once with sudo: sudo ./deploy/setup-ollama-qwen25.sh

set -euo pipefail

OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="${OVERRIDE_DIR}/override.conf"

mkdir -p "${OVERRIDE_DIR}"

cat > "${OVERRIDE_FILE}" <<'EOF'
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="OLLAMA_KEEP_ALIVE=-1"
ExecStartPost=/bin/sh -c 'sleep 3 && curl -sf http://127.0.0.1:11434/api/generate -d "{\"model\":\"qwen2.5:14b\",\"prompt\":\"\",\"keep_alive\":-1}" >/dev/null || true'
EOF

systemctl daemon-reload
systemctl restart ollama.service
sleep 5
ollama ps

echo "Ollama configured: qwen2.5:14b preloaded with keep_alive=-1"
