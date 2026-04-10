#!/bin/bash
set -euo pipefail
# Brave Browser Agent — Direct MLX + Chrome DevTools Protocol (no Claude Code)
# Double-click to launch
#
# Override the model with: MLX_MODEL=mlx-community/<model-id>

MLX_PYTHON="$HOME/.local/mlx-server/bin/python3"
MLX_SERVER="$HOME/.local/mlx-native-server/server.py"
AGENT="$HOME/.local/browser-agent/agent.py"
MODEL_NAME="${MLX_MODEL_LABEL:-Gemma 4 31B}"
MLX_MODEL_DEFAULT="${MLX_MODEL:-divinetribe/gemma-4-31b-it-abliterated-4bit-mlx}"
MLX_KV_BITS_DEFAULT="${MLX_KV_BITS:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"
PROFILE_NAME="${LAUNCHER_PROFILE:-browser}"
load_launcher_profile "$SCRIPT_DIR" "$PROFILE_NAME"
MODEL_NAME="${MLX_MODEL_LABEL:-${LAUNCHER_MODEL_NAME_DEFAULT:-$MODEL_NAME}}"
MLX_MODEL_DEFAULT="${MLX_MODEL:-${LAUNCHER_MLX_MODEL_DEFAULT:-$MLX_MODEL_DEFAULT}}"
MLX_KV_BITS_DEFAULT="${MLX_KV_BITS:-${LAUNCHER_MLX_KV_BITS_DEFAULT:-$MLX_KV_BITS_DEFAULT}}"

cleanup() {
  if [ "${STARTED_MLX_SERVER:-0}" -eq 1 ] && [ -n "${MLX_SERVER_PID:-}" ]; then
    if kill -0 "$MLX_SERVER_PID" >/dev/null 2>&1; then
      kill "$MLX_SERVER_PID" >/dev/null 2>&1 || true
      wait "$MLX_SERVER_PID" 2>/dev/null || true
    fi
  fi
}

trap cleanup EXIT INT TERM

if [ ! -f "$MLX_PYTHON" ]; then
  echo "  ERROR: MLX Python not found at $MLX_PYTHON"
  exit 1
fi

if [ ! -f "$MLX_SERVER" ]; then
  echo "  ERROR: MLX server not found at $MLX_SERVER"
  exit 1
fi

if [ ! -f "$AGENT" ]; then
  echo "  ERROR: Browser agent not found at $AGENT"
  exit 1
fi

if lsof -i :4000 >/dev/null 2>&1; then
  echo "  Restarting MLX server in browser mode..."
  pkill -f "mlx-native-server/server.py" 2>/dev/null || true
  for i in $(seq 1 20); do
    if ! lsof -i :4000 >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! lsof -i :4000 >/dev/null 2>&1; then
  echo "  Loading $MODEL_NAME..."
  MLX_BROWSER_MODE=1 MLX_KV_BITS="$MLX_KV_BITS_DEFAULT" MLX_MODEL="$MLX_MODEL_DEFAULT" "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  MLX_SERVER_PID=$!
  STARTED_MLX_SERVER=1
  while ! curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; do sleep 2; done
fi

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
  # Wait for at least one page/tab to register with CDP
  echo -n "  Waiting for Brave tab"
  for i in $(seq 1 10); do
    if curl -s http://127.0.0.1:9222/json 2>/dev/null | grep -q '"webSocketDebuggerUrl"'; then
      echo " ready!"
      break
    fi
    echo -n "."
    sleep 1
  done
  # If still no tabs, force-create one (Brave requires PUT)
  if ! curl -s http://127.0.0.1:9222/json 2>/dev/null | grep -q '"webSocketDebuggerUrl"'; then
    echo "  Creating new tab..."
    curl -s -X PUT 'http://127.0.0.1:9222/json/new' >/dev/null 2>&1
    sleep 1
  fi
fi

clear
echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Brave Browser Agent                          ║"
echo "  ║  $MODEL_NAME · MLX · Brave DevTools           ║"
echo "  ║  iframes + Shadow DOM · 100% local            ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

"$MLX_PYTHON" "$AGENT"
exit $?
