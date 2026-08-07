#!/bin/bash
# Qwen 3 Coder — Matt's own terminal coding agent, 100% local.
# Double-click to launch (Desktop/Launchers/Qwen 3 Coder.app opens this).
#
# Runs the shared agent engine (../agent/agent.py) with Qwen3-Coder 30B-A3B
# (8-bit MLX) loaded directly into the process, speaking the model's native
# XML tool-call format. No server, no HTTP hop, no cloud.

cd "$HOME/Desktop/PROJECTS/ineedhemp website" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="Qwen3 Coder 30B"
export AGENT_MODEL="lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit"
export AGENT_BACKEND="mlx"
export AGENT_DIALECT="native"
export AGENT_LEASE_NAME="agent-qwen3"
export AGENT_LEASE_GB="36"

exec "${AGENT_PYTHON:-$HOME/.local/mlx-server/bin/python3}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
