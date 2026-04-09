#!/bin/bash
# Gemma 4 Code — Claude Code on Gemma 4 31B Abliterated (4-bit MLX)
# Double-click to launch
#
# THE QUICK ONE — ~15 tok/s, ~18 GB RAM, abliterated, instruction-tuned.
# Best balance of speed and quality for daily coding.

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
MLX_SERVER="$HOME/.local/mlx-native-server/server.py"
MLX_PYTHON="$HOME/.local/mlx-server/bin/python3"

# Override with MLX_MODEL=<your-path-or-hf-id>
MLX_MODEL_DEFAULT="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"

# Start MLX server if not running (or if a different model is loaded)
if ! lsof -i :4000 >/dev/null 2>&1; then
  MLX_MODEL="${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  echo "  Loading Gemma 4 31B Abliterated on MLX (~15 tok/s, 4-bit)..."
  while ! curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; do
    sleep 2
  done
fi

clear
echo ""
echo "  → Claude Code with LOCAL AI (Gemma 4 31B Abliterated)"
echo "  → MLX Native: 4-bit IT, abliterated, instruction tuned for coding"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
CLAUDE_SESSION_LABEL="Gemma 4 · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
