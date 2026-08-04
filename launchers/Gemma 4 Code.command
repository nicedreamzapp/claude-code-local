#!/bin/bash
# Gemma 4 Code — Claude Code on Gemma 4 31B Abliterated (4-bit MLX)
# Double-click to launch
#
# THE QUICK ONE — ~15 tok/s, ~18 GB RAM, abliterated, instruction-tuned.
# Best balance of speed and quality for daily coding.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"

# Override with MLX_MODEL=<your-path-or-hf-id>. Prefers a local flat-folder
# cache if you already downloaded the model via scripts/download-and-import.sh,
# so mlx-lm loads directly from disk instead of re-pulling from HF.
MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx" \
  "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Gemma 4 31B Abliterated on MLX (~15 tok/s, 4-bit)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI (Gemma 4 31B Abliterated)"
echo "  → MLX Native: 4-bit IT, abliterated, instruction tuned for coding"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

# Launch from the HQ project dir — same as Divine Tribe HQ.app — so CLAUDE.md
# auto-discovery picks up the full business stack (this session is NOT --bare,
# so auto-discovery + auto-memory both work):
#   .../ineedhemp website/CLAUDE.md ..... store + invoice + email rules
#   ~/CLAUDE.md ......................... $HOME / HQ session briefing (parent chain)
#   ~/.claude/CLAUDE.md ................. global private rules
#   ~/.claude/.../memory/MEMORY.md ...... auto-memory
HQ_DIR="$HOME/Desktop/PROJECTS/ineedhemp website"
cd "$HQ_DIR" 2>/dev/null || cd "$HOME"

ANTHROPIC_BASE_URL=http://localhost:4000 \
CLAUDE_SESSION_LABEL="Gemma 4 · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode bypassPermissions \
  --settings "$SCRIPT_DIR/lib/local-settings.json" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --add-dir "$HQ_DIR" \
  --mcp-config "$HOME/.claude.json"
