#!/bin/bash
# ONE AI — the only launcher you need.
# Double-click. It picks the best local model for every prompt automatically:
#   quick stuff -> Gemma 4   ·   code/agentic -> Qwen 3 Coder   ·   huge context / hard reasoning -> DeepSeek V4 Flash
# You can force one with /fast, /code, or /deep at the start of a message.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh" 2>/dev/null || true
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
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

# 2) start the warm pool (Qwen :4000 + Gemma :4001 stay loaded together, ~46GB)
#    so switching between code/email and quick chat is instant — no swapping.
bash "$HOME/Desktop/PROJECTS/Local AI Setup/smart-router/warm_pool.sh" start 2>/dev/null || true

clear
echo ""
echo "  → ONE AI — picks the best local model for every prompt, automatically"
echo "  → quick → Gemma · code/agentic → Qwen · huge context / hard → DeepSeek V4 (284B)"
echo "  → force one with /fast  /code  /deep   ·   decision is instant; the 284B costs a load"
echo ""

# 3) load business credentials (HQ_TOKEN / HQ_URL / WooCommerce / etc.) into the
#    session env so the local model can actually DO the workflows — email triage,
#    invoices, reddit, publishing. Without these the model has no key to reach the
#    inbox/store and flails. Start in the ineedhemp dir so its project CLAUDE.md
#    and helpers resolve, same as the working "Divine Tribe HQ" launcher.
IH="$HOME/Desktop/PROJECTS/ineedhemp website"
set -a; [ -f "$IH/.env" ] && source "$IH/.env" 2>/dev/null; set +a
cd "$IH" 2>/dev/null || cd "$HOME"

export CLAUDE_SESSION_LABEL='ONE AI'
ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}" \
ANTHROPIC_API_KEY=sk-local \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
