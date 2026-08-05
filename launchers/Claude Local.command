#!/bin/bash
# Claude Code — Local AI (runs on your Mac, no cloud)
# Double-click to launch
# MLX Native Server — direct Anthropic API, no proxy needed
#
# Override the model with: MLX_MODEL=mlx-community/<model-id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || echo $HOME/.local/bin/claude)}"
require_claude_bin
MODEL_NAME="${MLX_MODEL_LABEL:-Gemma 4 31B}"

# Default model matches server.py's default so this launcher behaves like
# "the easy button." Override with MLX_MODEL=... before double-clicking, or
# use the model-specific launchers for Llama/Qwen.
MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx" \
  "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading $MODEL_NAME on MLX..."

clear
echo ""
echo "  → Claude Code with LOCAL AI ($MODEL_NAME)"
echo "  → MLX Native: zero proxy, zero cloud, zero API fees"
echo "  → Running 100% on your Apple Silicon GPU"
echo ""

# --bare forces API-key auth (blocks OAuth/Claude Max).
# CLAUDE.md and MCP config are added back explicitly so personal config still loads.
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
