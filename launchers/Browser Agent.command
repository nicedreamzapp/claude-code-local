#!/bin/bash
# Brave Browser Agent — Direct MLX + Chrome DevTools Protocol (pure local)
# Double-click to launch
#
# Override the model with: MLX_MODEL=mlx-community/<model-id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/local-common.sh"

AGENT="$HOME/.local/browser-agent/agent.py"
MODEL_NAME="${MLX_MODEL_LABEL:-Gemma 4 31B}"
MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx" \
  "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")"

# Browser mode: force-restart so MLX_BROWSER_MODE=1 is picked up even if a
# non-browser-mode server is already running.
export MLX_BROWSER_MODE=1
force_restart_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading $MODEL_NAME in browser mode..."

# Start Brave with remote debugging
if ! lsof -i :9222 >/dev/null 2>&1; then
  # If Brave is running WITHOUT debug port, restart it with the flag
  if pgrep -f "Brave Browser" >/dev/null 2>&1; then
    echo "  Restarting Brave with remote debugging enabled..."
    osascript -e 'quit app "Brave Browser"'
    sleep 2
  else
    echo "  Launching Brave with remote debugging..."
  fi
  open -a "Brave Browser" --args --remote-debugging-port=9222
  # Wait until debug port is actually ready (up to 15s)
  echo -n "  Waiting for Brave debug port"
  for i in $(seq 1 15); do
    if lsof -i :9222 >/dev/null 2>&1; then echo " ready!"; break; fi
    echo -n "."
    sleep 1
  done
  if ! lsof -i :9222 >/dev/null 2>&1; then
    echo ""
    echo "  ERROR: Brave debug port (9222) didn't open. Try closing Brave and running again."
    exit 1
  fi
fi

# Always ensure at least one page-type tab is registered with CDP, regardless
# of whether Brave was already running. On macOS, Brave can keep the debug
# port open even when every window is closed — lsof sees the port, but /json
# returns zero page targets, and the agent bails with "No browser pages".
echo -n "  Checking for Brave page tab"
for i in $(seq 1 10); do
  if curl -s http://127.0.0.1:9222/json 2>/dev/null | grep -q '"type": "page"'; then
    echo " ready!"
    break
  fi
  echo -n "."
  sleep 1
done
if ! curl -s http://127.0.0.1:9222/json 2>/dev/null | grep -q '"type": "page"'; then
  echo "  No page tab found — creating new tab..."
  curl -s -X PUT 'http://127.0.0.1:9222/json/new' >/dev/null 2>&1
  sleep 1
fi

clear
echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Brave Browser Agent                          ║"
echo "  ║  $MODEL_NAME · MLX · Brave DevTools           ║"
echo "  ║  iframes + Shadow DOM · 100% local            ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

exec "$MLX_PYTHON" "$AGENT"
