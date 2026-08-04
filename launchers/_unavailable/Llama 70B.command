#!/bin/bash
# Llama 70B — Claude Code on Llama 3.3 70B Abliterated (8-bit MLX)
# Double-click to launch
#
# THE WISE ONE — ~7 tok/s, ~70 GB RAM, full 8-bit precision, abliterated.
# Slower but the most capable reasoning in the lineup. Needs 96+ GB unified memory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"

# Default points at our own abliterated MLX upload:
#   https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx
# Override with MLX_MODEL=<your-path-or-hf-id>. Prefers a local flat-folder
# cache if already downloaded, so we load directly instead of re-pulling.
MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/Llama-3.3-70B-Instruct-abliterated-8bit-mlx" \
  "divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx")"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Llama 3.3 70B Abliterated on MLX (~7 tok/s, 8-bit full precision)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI (Llama 3.3 70B Abliterated)"
echo "  → MLX Native: ~7 tok/s, 8-bit full precision, abliterated"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
CLAUDE_SESSION_LABEL="Llama 70B · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --settings "$SCRIPT_DIR/lib/local-settings.json" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
