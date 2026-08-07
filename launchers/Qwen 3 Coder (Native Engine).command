#!/bin/bash
# Qwen 3 Coder — the NATIVE ENGINE (no Claude Code binary, no server).
# Double-click to launch.
#
# Qwen3-Coder's chat template defines tool calls as XML with RAW TEXT
# parameter values — no JSON escaping at all, which is the whole reason this
# engine exists. AGENT_DIALECT=native lets the template do the talking.
# The classic Claude-Code-on-a-proxy launcher is unchanged next door.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MLX_PYTHON="${MLX_PYTHON:-$HOME/.local/mlx-server/bin/python3}"

cd "$HOME" || exit 1

export AGENT_TITLE="Qwen 3 Coder 30B"
export AGENT_MODEL="${MLX_MODEL:-lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit}"
export AGENT_BACKEND="mlx"
export AGENT_DIALECT="native"

exec "$MLX_PYTHON" "$SCRIPT_DIR/../agent/agent.py"
