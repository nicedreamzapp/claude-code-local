#!/bin/bash
# Gemma 4 Chat — plain terminal chat with Gemma 4 31B Abliterated.
#
# Same window as the other local launchers — framed input box with the live
# context %% in the top rule, spinner, slash commands, readline history (the
# UI is imported from agent.py, not reimplemented). What's stripped is the
# agent machinery only: no tools, no file access, no step loop. Just talk.
#
# 2026-09-16: this used to point at Song Forge's lyric server on :9420, which
# since 8/17 holds SuperGemma 12B, so every chat quietly loaded a second model
# into Song Forge's server. Now it runs its own server on :9421 with the full
# bf16 Gemma 4 31B (vision intact, ~62GB), pauses Song Forge while it is open,
# and stops that server when you close the window.

source "$HOME/Desktop/PROJECTS/Local AI Setup/launchers/lib/local-common.sh"
pause_songforge_while_running

# 2026-09-17: the model now loads inside this window through mlx_vlm (chat.py's
# CHAT_BACKEND=mlx). mlx_lm.server on :9421 served the same weights text-only,
# so a picture dropped on the window reached Gemma as a bare path it couldn't
# open. In-process it gets the picture itself, and keeps the agent's cache reuse.
PY="$HOME/.local/mlx-server/bin/python3"
export CHAT_BACKEND=mlx
export CHAT_MODEL="$GEMMA4_VL_MODEL"
export AGENT_LEASE_NAME="gemma4-chat"
export AGENT_LEASE_GB="48"
echo "  loading Gemma 4 31B (~62 GB, give it a minute)..."
exec "${AGENT_PYTHON:-$PY}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/chat.py" "$@"
