#!/bin/bash
# Gemma 4 Chat — plain terminal chat with Gemma 4 31B Abliterated.
#
# Same window as the other local launchers — framed input box with the live
# context %% in the top rule, spinner, slash commands, readline history (the
# UI is imported from agent.py, not reimplemented). What's stripped is the
# agent machinery only: no tools, no file access, no step loop. Just talk.
#
# Talks to the mlx_lm.server already running on :9420 (the same warm Gemma
# that writes Song Forge lyrics), so this costs ZERO extra RAM — no second
# 18 GB copy of the model, no forge_guard lease, instant startup.

exec "${AGENT_PYTHON:-$HOME/.local/mlx-server/bin/python3}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/chat.py" "$@"
