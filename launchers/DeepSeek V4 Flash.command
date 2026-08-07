#!/bin/bash
# DeepSeek V4 Flash — Matt's own terminal agent on the local ds4 engine.
# Double-click to launch.
#
# THE 1M-CONTEXT BEAST — 284B Mixture-of-Experts via Antirez's ds4.c engine.
# Fully off-cloud. Disk-backed KV cache. 2-bit asymmetric quantization.
# The ds4-server speaks the Anthropic wire protocol on :8000; the shared
# agent engine talks to it directly (AGENT_BACKEND=http) — the server owns
# the model and its memory.

DS4_DIR="$HOME/Desktop/PROJECTS/Local AI Setup/ds4"
PORT=8000

# Boot ds4-server if it isn't already up.
if ! lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  Loading DeepSeek V4 Flash on ds4.c (this takes ~30s on first start)..."
    "$HOME/.local/bin/ds4-server-up" || {
        echo "Failed to start ds4-server. Make sure ds4flash.gguf is in $DS4_DIR." >&2
        echo "Press any key to close..."
        read -n 1
        exit 1
    }
fi

cd "$HOME/Desktop/PROJECTS/ineedhemp website" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="DeepSeek V4 Flash"
export AGENT_MODEL="deepseek-v4-flash"
export AGENT_BACKEND="http"
export AGENT_BASE_URL="http://127.0.0.1:${PORT}"
export AGENT_AUTH_TOKEN="dsv4-local"

exec python3 "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
