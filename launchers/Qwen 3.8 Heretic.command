#!/bin/bash
# Qwen 3.8 Heretic — YOUR terminal, running Qwen3.8-27B Heretic (full bf16).
# Double-click to launch.
#
# Your own agent engine (../agent/agent.py): framed input box, live context %%,
# your typed text in soft blue, /resume, slash commands. No cloud, no billing.
#
# THE MAX-PRECISION ONE: full bfloat16 (~50 GB), refusals removed, Qwen's
# newest 27B. It is big enough that it needs Song Forge stopped while it runs,
# so this launcher pauses Song Forge on start and puts it back when you quit.

set -u
PLIST="$HOME/Library/LaunchAgents/com.nicedreamz.songforge-m5-stack.plist"
FORGE_STOPPED=0

restore_forge() {
  [ "$FORGE_STOPPED" = "1" ] || return 0
  echo ""
  echo "  putting Song Forge back..."
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load "$PLIST" 2>/dev/null
  launchctl start com.nicedreamz.songforge-m5-stack 2>/dev/null
  for _ in $(seq 1 30); do
    if curl -s -m 3 http://127.0.0.1:8767/api/status 2>/dev/null | grep -q '"ace_up": *true'; then
      echo "  Song Forge is back up."
      return 0
    fi
    sleep 2
  done
  echo "  (Song Forge is restarting in the background.)"
}
trap restore_forge EXIT INT TERM

# If Song Forge is up, it holds the memory this 50 GB model needs.
if curl -s -m 3 http://127.0.0.1:8767/api/status 2>/dev/null | grep -q '"ace_up"'; then
  clear
  echo ""
  echo "  ■ Song Forge stopped — to resume Song Forge, just close this terminal."
  echo ""
  launchctl bootout "gui/$(id -u)/com.nicedreamz.songforge-m5-stack" 2>/dev/null
  for port in 8001 8767 9420; do
    p=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null | head -1)
    [ -n "$p" ] && kill "$p" 2>/dev/null
  done
  FORGE_STOPPED=1
  sleep 3

  # Backstop for closing the WINDOW (which skips the trap above). This guardian
  # is nohup'd so the window-close SIGHUP can't kill it; it watches THIS
  # launcher's pid and, the moment the launcher dies by any means, brings Song
  # Forge back — but only if it's actually down, so it won't double-start when
  # the clean-exit trap already handled it.
  LAUNCHER_PID=$$
  UID_NUM=$(id -u)
  nohup bash -c '
    while kill -0 '"$LAUNCHER_PID"' 2>/dev/null; do sleep 2; done
    sleep 2
    if ! curl -s -m 3 http://127.0.0.1:8767/api/status >/dev/null 2>&1; then
      launchctl bootstrap "gui/'"$UID_NUM"'" "'"$PLIST"'" 2>/dev/null \
        || launchctl load "'"$PLIST"'" 2>/dev/null
      launchctl start com.nicedreamz.songforge-m5-stack 2>/dev/null
    fi
  ' >/dev/null 2>&1 &
  disown
fi

cd "$HOME/Desktop/PROJECTS/ineedhemp website" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="Qwen 3.8 Heretic"
export AGENT_MODEL="donedynamics/Qwen3.8-27B-heretic-MLX-bf16"
export AGENT_BACKEND="mlx"
export AGENT_DIALECT="native"
# forge_guard caps any non-Song-Forge reservation at ~50GB, so ask for 48 —
# 56 could NEVER be granted and the launcher hung waiting for it. The model's
# real ~50GB use slightly overruns 48, which forge_guard charges but never
# kills; with Song Forge stopped the box has ~100GB free, so no swap.
export AGENT_LEASE_NAME="agent-qwen38"
export AGENT_LEASE_GB="48"

# Firm knowledge (store / invoice / email / shipping), refreshed each launch.
FIRM_KNOWLEDGE="$HOME/Desktop/PROJECTS/Local AI Setup/agent/firm-knowledge.md"
cp "$HOME/Desktop/PROJECTS/ineedhemp website/CLAUDE.md" "$FIRM_KNOWLEDGE" 2>/dev/null
export AGENT_PROMPT_FILE="$FIRM_KNOWLEDGE"
export AGENT_TEMP="0"

echo "  loading the model (~50 GB, give it a minute)..."
"${AGENT_PYTHON:-$HOME/.local/mlx-server/bin/python3}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
# (no exec — so the trap runs and Song Forge comes back when you quit)
