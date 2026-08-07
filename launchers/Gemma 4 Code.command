#!/bin/bash
# Gemma 4 — Matt's own terminal agent on Gemma 4 31B Abliterated (4-bit MLX).
# Double-click to launch.
#
# THE QUICK ONE — ~15 tok/s, ~18 GB RAM, abliterated, instruction-tuned.
# Runs the shared agent engine with the model loaded directly into the
# process. Gemma's own tool-call template is a custom pseudo-JSON that
# re-introduces escaping bugs, so the engine teaches it our XML dialect
# in the system prompt instead (AGENT_DIALECT=prompted).

cd "$HOME/Desktop/PROJECTS/ineedhemp website" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="Gemma 4 31B"
export AGENT_MODEL="${MLX_MODEL:-divinetribe/gemma-4-31b-it-abliterated-4bit-mlx}"
export AGENT_BACKEND="mlx"
export AGENT_DIALECT="prompted"
export AGENT_LEASE_NAME="agent-gemma4"
export AGENT_LEASE_GB="28"
# Eval-proven addendum (12/12 vs 11/12 baseline): Gemma writes files via
# shell echo if allowed, and sh echo eats backslashes. See agent/prompts/.
export AGENT_PROMPT_FILE="$HOME/Desktop/PROJECTS/Local AI Setup/agent/prompts/gemma4.md"

exec "${AGENT_PYTHON:-$HOME/.local/mlx-server/bin/python3}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
