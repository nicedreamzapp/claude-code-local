#!/bin/bash
# ONE AI — the only launcher you need.
# Double-click. It picks the best local model for every prompt automatically:
#   quick stuff -> Gemma 4   ·   code/agentic -> Qwen 3 Coder   ·   huge context / hard reasoning -> DeepSeek V4 Flash
# The router on :4010 speaks the Anthropic wire protocol; the shared agent
# engine talks to it directly. Force a lane with /fast, /code, or /deep at
# the start of a message.

ROUTER="$HOME/Desktop/PROJECTS/Local AI Setup/smart-router/router.py"
PORT=4010
LOG="/tmp/one-ai-router.log"

# 1) bring the router up (instant; it manages the backends on demand)
if ! lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  Starting ONE AI router on :${PORT}..."
  nohup python3 "$ROUTER" >"$LOG" 2>&1 &
  disown
  for i in $(seq 1 20); do
    curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 0.25
  done
fi

# 2) start the warm pool (Qwen :4000 + Gemma :4002 stay loaded together)
bash "$HOME/Desktop/PROJECTS/Local AI Setup/smart-router/warm_pool.sh" start 2>/dev/null || true

# 3) load business credentials (HQ_TOKEN / WooCommerce / etc.) into the
#    session env so the model can actually DO the workflows, and start in the
#    ineedhemp dir so file tools land where the work is.
IH="$HOME/Desktop/PROJECTS/ineedhemp website"
set -a; [ -f "$IH/.env" ] && source "$IH/.env" 2>/dev/null; set +a
cd "$IH" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="One AI"
export AGENT_MODEL="one-ai"
export AGENT_BACKEND="http"
export AGENT_BASE_URL="http://127.0.0.1:${PORT}"
export AGENT_AUTH_TOKEN="sk-local"

exec python3 "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
