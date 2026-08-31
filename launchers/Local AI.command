#!/bin/bash
# Local AI — YOUR terminal for local-AI work (models, inference, voice,
# talking-head, mesh nodes). Double-click to launch.
#
# Runs on your own agent engine (../agent/agent.py) with your best local
# brain, Qwen3.8-27B Heretic (full bf16). No cloud, no billing — this used
# to open a cloud session; now it's fully yours and fully local.
#
# Focused on the local-AI stack, NOT the business — no firm knowledge loaded.

cd "$HOME/Desktop/PROJECTS/Local AI Setup" 2>/dev/null || cd "$HOME"

export AGENT_TITLE="Local AI"
export AGENT_MODEL="donedynamics/Qwen3.8-27B-heretic-MLX-bf16"
export AGENT_BACKEND="mlx"
export AGENT_DIALECT="native"
export AGENT_LEASE_NAME="agent-localai"
export AGENT_LEASE_GB="56"
export AGENT_TEMP="0"

exec "${AGENT_PYTHON:-$HOME/.local/mlx-server/bin/python3}" \
  "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
