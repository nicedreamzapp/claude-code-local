#!/bin/bash
# DeepSeek V4 Flash — Claude Code on the local ds4 inference engine.
# Double-click to launch.
#
# THE 1M-CONTEXT BEAST — 284B Mixture-of-Experts via Antirez's ds4.c engine.
# Fully off-cloud. Disk-backed KV cache. 2-bit asymmetric quantization.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh" 2>/dev/null || true

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
DS4_DIR="$HOME/Desktop/PROJECTS/Local AI Setup/ds4"
DS4_BIN="$DS4_DIR/ds4-server"
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

clear
echo ""
echo "  → Claude Code with LOCAL AI (DeepSeek V4 Flash · 284B MoE · 1M context)"
echo "  → ds4.c native engine on Metal — no cloud, no API key, no rate limit"
echo "  → Antirez build · disk-backed KV cache · 2-bit asymmetric quant"
echo ""

# Same launch shape as the other local launchers — point claude at the local
# Anthropic-compatible endpoint, force API-key auth, keep personal config.
export CLAUDE_SESSION_LABEL='DeepSeek V4 Flash'
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$CLAUDE_BIN" \
  --permission-mode auto \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
