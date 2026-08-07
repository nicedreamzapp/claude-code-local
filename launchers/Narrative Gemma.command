#!/bin/bash
# Narrative Gemma — Matt's own terminal agent with auto-narration.
# Double-click to launch.
#
# Gemma 4 31B Abliterated (4-bit MLX) loaded directly into the shared agent
# engine; every final answer is spoken aloud through ~/.local/bin/speak
# (AGENT_SPEAK=1 — narration is done in code now, not by prompt rules).
# Hands-free dictation binds to this Terminal window if available.

clear
echo ""
echo "  → NARRATIVE GEMMA — local AI with auto-narration"
echo "  → Gemma 4 31B Abliterated · 4-bit · ~15 tok/s"
echo "  → Every response spoken aloud via ~/.local/bin/speak"
echo ""

# ── Bind hands-free dictation to THIS Terminal window ─────────────────
DICT_DIR="$HOME/NarrateClaude/dictation"
DICT="$DICT_DIR/bin/dictation"
STATE_DIR="$DICT_DIR/state"
if [ -x "$DICT" ]; then
  "$DICT" stop >/dev/null 2>&1 || true
  MY_TTY="$(tty 2>/dev/null || true)"
  WIN_ID=""
  if [ -n "$MY_TTY" ]; then
    WIN_ID=$(/usr/bin/osascript <<OSA 2>/dev/null
tell application "Terminal"
    set foundId to ""
    repeat with w in windows
        repeat with t in tabs of w
            try
                if tty of t is "$MY_TTY" then
                    set foundId to (id of w) as text
                    exit repeat
                end if
            end try
        end repeat
        if foundId is not "" then exit repeat
    end repeat
    return foundId
end tell
OSA
)
  fi
  if [ -z "$WIN_ID" ]; then
    WIN_ID=$(/usr/bin/osascript -e 'tell application "Terminal" to id of front window' 2>/dev/null)
  fi
  if [ -n "$WIN_ID" ]; then
    mkdir -p "$STATE_DIR"
    cat > "$STATE_DIR/target.json" <<JSON
{ "app": "Terminal", "window_id": $WIN_ID }
JSON
    NARRATE_DICTATION_LAUNCHER="Narrative Gemma.command" \
      "$DICT" start >/dev/null 2>&1 || \
      echo "  ⚠ dictation listener failed to start — see $STATE_DIR/dictation.log.stderr"
  else
    echo "  ⚠ couldn't find this Terminal window — voice mode disabled this session"
  fi
fi

cd "$HOME/Desktop/PROJECTS/Local AI Setup/NarrativeGemma" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="Narrative Gemma 4"
export AGENT_MODEL="${MLX_MODEL:-divinetribe/gemma-4-31b-it-abliterated-4bit-mlx}"
export AGENT_BACKEND="mlx"
export AGENT_DIALECT="prompted"
export AGENT_LEASE_NAME="agent-gemma4"
export AGENT_LEASE_GB="28"
export AGENT_SPEAK=1
# Eval-proven addendum (12/12 vs 11/12 baseline): Gemma writes files via
# shell echo if allowed, and sh echo eats backslashes. See agent/prompts/.
export AGENT_PROMPT_FILE="$HOME/Desktop/PROJECTS/Local AI Setup/agent/prompts/gemma4.md"

exec "${AGENT_PYTHON:-$HOME/.local/mlx-server/bin/python3}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
