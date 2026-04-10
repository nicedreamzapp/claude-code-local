#!/bin/bash
set -euo pipefail
# Llama 70B — Claude Code on Llama 3.3 70B Abliterated (8-bit MLX)
# Double-click to launch
#
# THE WISE ONE — ~7 tok/s, ~70 GB RAM, full 8-bit precision, abliterated.
# Slower but the most capable reasoning in the lineup. Needs 96+ GB unified memory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"
PROFILE_NAME="${LAUNCHER_PROFILE:-llama}"
load_launcher_profile "$SCRIPT_DIR" "$PROFILE_NAME"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
MLX_SERVER="$HOME/.local/mlx-native-server/server.py"
MLX_PYTHON="$HOME/.local/mlx-server/bin/python3"
MLX_KV_BITS="${MLX_KV_BITS:-${LAUNCHER_MLX_KV_BITS_DEFAULT:-0}}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-${LAUNCHER_CLAUDE_PERMISSION_MODE_DEFAULT:-auto}}"
MODEL_NAME="${MLX_MODEL_LABEL:-${LAUNCHER_MODEL_NAME_DEFAULT:-Llama 3.3 70B Abliterated}}"
MCP_CONFIG="$(build_user_mcp_config)"

cleanup() {
  cleanup_mlx_server
  rm -f "$MCP_CONFIG"
}

trap cleanup EXIT INT TERM

require_file "$CLAUDE_BIN" "Claude Code CLI"
require_file "$MLX_SERVER" "MLX server"
require_file "$MLX_PYTHON" "MLX Python"

# Default points at our own abliterated MLX upload:
#   https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx
# Override with MLX_MODEL=<your-path-or-hf-id>
MLX_MODEL_DEFAULT="${LAUNCHER_MLX_MODEL_DEFAULT:-divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx}"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" "  Loading $MODEL_NAME on MLX (~7 tok/s, 8-bit full precision)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI ($MODEL_NAME)"
echo "  → MLX Native: ~7 tok/s, 8-bit full precision, abliterated"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
CLAUDE_SESSION_LABEL="Llama 70B · Local" \
"$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode "$CLAUDE_PERMISSION_MODE" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$MCP_CONFIG"
exit $?
