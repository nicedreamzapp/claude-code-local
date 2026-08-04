#!/bin/bash
# Llama 70B Fast — Claude Code on Llama 3.3 70B Abliterated (4-bit MLX)
# Double-click to launch
#
# THE FAST WISE ONE — measured 13.4 tok/s decode (1.9× the 8-bit launcher),
# TTFT down from 6.6s to 1.0s (6× faster first token), peak ~38 GB RAM.
# Same abliterated source weights, requantized to 4-bit. Small but real
# quality tradeoff on multi-step reasoning vs the 8-bit launcher.
#
# This launcher is a sibling of "Llama 70B.command" — it does NOT replace it.
# The original 8-bit launcher stays available for when full precision matters.
# Numbers from benchmarks/results/4bit-baseline--{code,prose}.json on M5 Max.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"

# 4-bit abliterated MLX upload by frscrcc (same source weights as our 8-bit,
# requantized). Override with MLX_MODEL=<your-path-or-hf-id>.
MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/Llama-3.3-70B-Instruct-abliterated-4bit-mlx" \
  "frscrcc/Llama-3.3-70B-Instruct-abliterated-mlx-4Bit")"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Llama 3.3 70B Abliterated on MLX (4-bit, 13.4 tok/s, ~38 GB)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI (Llama 3.3 70B Abliterated · 4-bit FAST)"
echo "  → MLX Native: 4-bit, abliterated — 1.9× faster decode, 6× faster TTFT"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
CLAUDE_SESSION_LABEL="Llama 70B Fast · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --settings "$SCRIPT_DIR/lib/local-settings.json" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
