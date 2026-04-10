#!/bin/bash
set -euo pipefail
# Claude Code — Local AI (FAST mode)
# Speed-first launcher:
# - --bare (skip skills/plugins/hooks discovery)
# - constrained built-in tool set
# - Gemma 4 31B default model

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
MLX_SERVER="${MLX_SERVER:-$HOME/.local/mlx-native-server/server.py}"
MLX_PYTHON="${MLX_PYTHON:-$HOME/.local/mlx-server/bin/python3}"
MODEL_NAME="${MLX_MODEL_LABEL:-Gemma 4 31B (Fast)}"
MLX_MODEL_DEFAULT="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"
REQUESTED_MODEL="${MLX_MODEL:-$MLX_MODEL_DEFAULT}"
MCP_CONFIG="$(build_user_mcp_config)"

cleanup() {
  cleanup_mlx_server
  rm -f "$MCP_CONFIG"
}

trap cleanup EXIT INT TERM

require_file "$CLAUDE_BIN" "Claude Code CLI"
require_file "$MLX_SERVER" "MLX server"
require_file "$MLX_PYTHON" "MLX Python"

ensure_mlx_server "$REQUESTED_MODEL" "  Loading $MODEL_NAME on MLX..."

clear
echo ""
echo "  -> Claude Code LOCAL FAST mode"
echo "  -> $MODEL_NAME"
echo "  -> bare mode + minimal tools for lower latency"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
"$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --tools "Bash,Read,Edit,Write,Glob,Grep,LS,MultiEdit" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$MCP_CONFIG"
exit $?
