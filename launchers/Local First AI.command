#!/bin/bash
# Local First AI — Claude Code on YOUR hardware, tuned for speed.
# Double-click to launch.
#
# DeepSeek V4 Flash (284B MoE, 1M ctx) on this Mac, with a SLIMMED prompt:
# no MCP tool schemas, no auto-memory — that bloat made the first turn take
# ~45K tokens (~2.5 min of prefill). Bare mode cuts it to a fraction.
# No cloud, no API key, no rate limit, no quota roulette.
#
# Want the QUICK local brain instead? Use "Gemma 4 Code.command" (~15 tok/s,
# loads in seconds). This one is the deep thinker.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh" 2>/dev/null || true

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
PORT=8000

# One deep brain at a time: if the Gemma MLX server (:4000) is loaded, evict it
# first — DeepSeek + Gemma together overflow 128GB and Metal OOMs (2026-06-11).
if lsof -i :4000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  Stopping Gemma MLX server to free GPU memory (relaunch it anytime — they swap)..."
  kill $(lsof -ti :4000 -sTCP:LISTEN) 2>/dev/null
  sleep 2
fi

# Boot ds4-server if it isn't already up (idempotent; ~30s cold start, mmap).
if ! lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  Loading DeepSeek V4 Flash (one-time ~30s)..."
    "$HOME/.local/bin/ds4-server-up" || {
        echo "Failed to start ds4-server — check /tmp/ds4-server.log" >&2
        echo "Press any key to close..."; read -n 1; exit 1
    }
fi

clear
echo ""
echo "  → Claude Code — 100% LOCAL (DeepSeek V4 Flash · 284B MoE · 1M context)"
echo "  → Slim prompt mode: fast first answer, no MCP bloat, no cloud, no caps"
echo "  → Quick option: Gemma 4 Code.command (small + snappy, also local)"
echo ""

export CLAUDE_SESSION_LABEL='Local First AI'
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
export API_TIMEOUT_MS=600000   # big prompts prefill slowly on first turn; don't let the client bail

# --bare = skip hooks, plugins, auto-memory (the 45K-token prompt killer).
# CLAUDE.md still appended so Matt's house rules ride along.
exec "$CLAUDE_BIN" --bare \
  --permission-mode auto \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md"
