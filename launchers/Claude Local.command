#!/bin/bash
set -euo pipefail
# Claude Code — Local AI (runs on your Mac, no cloud)
# Double-click to launch
# MLX Native Server — direct Anthropic API, no proxy needed
#
# Override the model with: MLX_MODEL=mlx-community/<model-id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"
PROFILE_NAME="${LAUNCHER_PROFILE:-standard}"
load_launcher_profile "$SCRIPT_DIR" "$PROFILE_NAME"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
MLX_SERVER="$HOME/.local/mlx-native-server/server.py"
MLX_PYTHON="$HOME/.local/mlx-server/bin/python3"
MLX_KV_BITS="${MLX_KV_BITS:-${LAUNCHER_MLX_KV_BITS_DEFAULT:-0}}"
MODEL_NAME="${MLX_MODEL_LABEL:-${LAUNCHER_MODEL_NAME_DEFAULT:-Qwen 3.5 122B}}"
MLX_MODEL_DEFAULT="${LAUNCHER_MLX_MODEL_DEFAULT:-mlx-community/Qwen3.5-122B-A10B-4bit}"
REQUESTED_MODEL="${MLX_MODEL:-$MLX_MODEL_DEFAULT}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-${LAUNCHER_CLAUDE_PERMISSION_MODE_DEFAULT:-auto}}"
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
echo "  → Claude Code with LOCAL AI ($MODEL_NAME)"
echo "  → MLX Native: zero proxy, zero cloud, zero API fees"
echo "  → Running 100% on your Apple Silicon GPU"
echo ""

# Preserve normal Claude Code features like skills and hooks.
# MCP servers are extracted from the current Claude settings into a
# schema-valid temp file for this session.
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
"$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode "$CLAUDE_PERMISSION_MODE" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$MCP_CONFIG"
exit $?
