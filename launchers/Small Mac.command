#!/bin/bash
# Small Mac — Claude Code on an 8 GB or 16 GB Apple Silicon Mac
# Double-click to launch
#
# Picks the tier from this Mac's memory (scripts/mac-tier.sh):
#   16 GB (up to 31 GB): Qwen 3.5 4B, bf16, 16K context
#   8 GB:                Qwen 3.5 2B, 8-bit, 16K context
# Both read images (a Read on a .png, a pasted screenshot), write code and call
# tools. Force a tier with CCL_TIER=8gb or CCL_TIER=16gb. Measured numbers and
# the limits of each tier: docs/SMALL-MACS.md

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || echo $HOME/.local/bin/claude)}"
require_claude_bin

source "$REPO_DIR/scripts/mac-tier.sh"
if [ "$CCL_TIER" = "none" ]; then
  echo "  This Mac has 32 GB or more, so it does not need the small-Mac tier."
  echo "  Use Claude Local.command, or force a tier: CCL_TIER=16gb \"$0\""
  exit 1
fi

if ! "$MLX_PYTHON" -c "import mlx_vlm" 2>/dev/null; then
  echo "  The small-Mac tier needs mlx-vlm (it is what lets the model see images):"
  echo "    $MLX_PYTHON -m pip install mlx-vlm"
  exit 1
fi

if [ ! -f "$MLX_MODEL/config.json" ] && [ "$MLX_MODEL" = "$CCL_MODEL_DIR" ]; then
  echo "  The model is not downloaded yet. One time, $CCL_MODEL_LABEL:"
  echo "    bash \"$REPO_DIR/scripts/download-and-import.sh\" small"
  exit 1
fi

ensure_mlx_server "$MLX_MODEL" \
  "  Loading $CCL_MODEL_LABEL on MLX ($CCL_TIER tier)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI ($CCL_MODEL_LABEL)"
echo "  → $CCL_TIER tier: ${MLX_MAX_CONTEXT}-token context, compacts when it fills"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

# --bare forces API-key auth, same as Claude Local.command. The tier's
# CLAUDE_CODE_MAX_OUTPUT_TOKENS and friends are exported by mac-tier.sh and
# reach Claude Code from this environment. The 16K window is enforced by the
# server (MLX_MAX_CONTEXT), not CLAUDE_CODE_AUTO_COMPACT_WINDOW, which Claude
# Code ignores under 100000.
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
CLAUDE_SESSION_LABEL="Small Mac · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare
