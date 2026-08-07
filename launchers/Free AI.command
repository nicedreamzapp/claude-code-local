#!/bin/bash
# Free AI — Matt's own terminal agent on the FREE multi-provider cloud stack.
# Chain: Gemini -> OpenRouter -> Cerebras -> Groq -> LOCAL DeepSeek V4 Flash.
# Double-click to launch.
#
# The LiteLLM gateway on :4001 speaks the Anthropic wire protocol and does
# the failover; the shared agent engine talks to it directly.

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

# Load the gateway master key for the agent's auth token.
set -a; [ -f "$KEYS" ] && . "$KEYS"; set +a

cd "$HOME/Desktop/PROJECTS/ineedhemp website" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="Free AI Cloud Stack"
export AGENT_MODEL="free-stack"
export AGENT_BACKEND="http"
export AGENT_BASE_URL="http://127.0.0.1:${PORT}"
export AGENT_AUTH_TOKEN="${LITELLM_MASTER_KEY:-sk-freeai-local}"

exec python3 "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
