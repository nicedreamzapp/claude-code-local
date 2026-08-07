#!/bin/bash
# Gemma 4 Code — the NATIVE ENGINE (no Claude Code binary, no server).
# Double-click to launch.
#
# Loads the model straight into this process via MLX and keeps the KV cache
# across turns, so replies start in well under a second even deep into a long
# session. Same tools (Bash/Read/Write/Edit/Glob/Grep), tiny fixed prompt.
# The classic Claude-Code-on-a-proxy launcher is unchanged next door.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MLX_PYTHON="${MLX_PYTHON:-$HOME/.local/mlx-server/bin/python3}"

cd "$HOME" || exit 1

export AGENT_TITLE="Gemma 4 31B"
export AGENT_MODEL="${MLX_MODEL:-divinetribe/gemma-4-31b-it-abliterated-4bit-mlx}"
export AGENT_BACKEND="mlx"
# Gemma's own tool-call template is a custom pseudo-JSON that re-introduces
# escaping bugs, so the engine teaches it our XML dialect in the prompt.
export AGENT_DIALECT="prompted"

exec "$MLX_PYTHON" "$SCRIPT_DIR/../agent/agent.py"
