#!/bin/bash
# Free AI — Claude Code on the FREE multi-provider cloud stack with auto-failover.
# Chain: Gemini -> OpenRouter -> Cerebras -> Groq -> LOCAL DeepSeek V4 Flash.
# Double-click to launch.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh" 2>/dev/null || true

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
KEYS="$HOME/.config/free-api/keys.env"
PORT=4001

# Boot the gateway if it isn't already up.
if ! lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  Starting Free AI gateway (LiteLLM on :${PORT})..."
  if ! "$HOME/.local/bin/free-api-up"; then
    echo ""
    echo "  No free API keys yet — opening the keys file."
    echo "  Paste at least one key, save, then re-launch. All free, no card:"
    echo "    Groq        https://console.groq.com/keys"
    echo "    Cerebras    https://cloud.cerebras.ai   (API Keys)"
    echo "    Gemini      https://aistudio.google.com/apikey"
    echo "    OpenRouter  https://openrouter.ai/keys"
    open -e "$KEYS" 2>/dev/null
    echo ""
    echo "  Press any key to close..."
    read -n 1
    exit 1
  fi
fi

# Load the gateway master key for Claude Code's auth token.
set -a; [ -f "$KEYS" ] && . "$KEYS"; set +a

clear
echo ""
echo "  → Claude Code on the FREE CLOUD STACK (automatic failover)"
echo "  → CLOUD: Gemma 4 31B → GLM-4.7 → Llama 4 Scout → Nemotron 550B   then LOCAL: DeepSeek V4 on this Mac"
echo "  → Free tiers · big models · never fully down"
echo ""

export CLAUDE_SESSION_LABEL='Free AI'
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"
export ANTHROPIC_AUTH_TOKEN="${LITELLM_MASTER_KEY:-sk-freeai-local}"
export ANTHROPIC_MODEL="free-stack"
export ANTHROPIC_DEFAULT_SONNET_MODEL="free-stack"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="free-stack"
export ANTHROPIC_DEFAULT_OPUS_MODEL="free-stack"
export CLAUDE_CODE_SUBAGENT_MODEL="free-stack"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000
export API_TIMEOUT_MS=300000   # free pools can take minutes to fail over; default client timeout gave "The operation timed out"

# The free backends (Gemini/Groq/Cerebras/OpenRouter) don't emit Anthropic-format
# "thinking blocks", but ~/.claude/settings.json sets effortLevel=high, which makes
# Claude Code request one — that is the "Content block is not a thinking block" error.
# Force thinking off for this stack (real Opus sessions keep high effort).
export MAX_THINKING_TOKENS=0

exec "$CLAUDE_BIN" \
  --permission-mode auto \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
