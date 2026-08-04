#!/bin/bash
# Qwen Coder — Claude Code on Qwen3-Coder 30B-A3B (8-bit MLX)
# Double-click to launch
#
# THE AGENTIC ONE — 30B MoE, 3B active, RL-tuned for tool calls and code edits.
# Best fit for Claude-Code-style multi-step loops where Gemma can't keep up
# and the bigger dense models (Llama 70B, Qwen 122B) got confused.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
MLX_MODEL_LABEL="Qwen3-Coder 30B-A3B (8-bit)"

# Prefer the local LM Studio cache (downloaded 2026-04-26 to that location);
# fall back to the HF repo id so first-run on a fresh machine still works.
MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit" \
  "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit")"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading $MLX_MODEL_LABEL on MLX (~70 tok/s, agentic-tuned)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI ($MLX_MODEL_LABEL)"
echo "  → MLX Native: 30B MoE, 3B active, tool-calling RL'd"
echo "  → Built for the Claude-Code agentic loop — local, sovereign, fast"
echo ""

# Launch shape matches Gemma 4 Code.command EXACTLY (minus the model) so this
# session handles the business like Divine Tribe HQ.app. Local auth is via the
# apiKeyHelper in local-settings.json (echo sk-local) — NOT --bare, which would
# strip hooks, plugins, and auto-memory and make this LESS capable than HQ.
# cd into the HQ project dir so CLAUDE.md auto-discovery + auto-memory load the
# full stack: ineedhemp website/CLAUDE.md, ~/CLAUDE.md, ~/.claude/CLAUDE.md, MEMORY.md.
HQ_DIR="$HOME/Desktop/PROJECTS/ineedhemp website"
cd "$HQ_DIR" 2>/dev/null || cd "$HOME"

export CLAUDE_SESSION_LABEL='Qwen 3 Coder'
ANTHROPIC_BASE_URL=http://localhost:4000 \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode bypassPermissions \
  --settings "$SCRIPT_DIR/lib/local-settings.json" \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --add-dir "$HQ_DIR" \
  --mcp-config "$HOME/.claude.json"
