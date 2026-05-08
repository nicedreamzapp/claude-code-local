#!/bin/bash
# Claude Chat - Claude Code on Qwen 2.5 Coder 14B in CHAT-ONLY mode (no tools)
# Double-click to launch
#
# Designed for Mac base/Pro with 16-32 GB unified memory, where agentic
# tool-use sessions can be unreliable due to RAM pressure and the Claude Code
# 2.1 extended-thinking flow that splits each turn into two model calls.
#
# Why Qwen 2.5 Coder 14B 4-bit MLX:
#   - 7.8 GB of weights → fits in 16 GB without swapping
#   - 14B is large enough to follow Claude Code's system prompt structure
#   - Strong code/reasoning quality, good PT-BR support
#   - Native MLX 4-bit quant for Apple Silicon (no GGUF translation layer)
#
# This launcher:
#   - disables all tools (--tools "")
#   - forces --effort low so Claude Code does NOT request extended thinking
#     (small/quantized models exhaust their budget thinking and emit empty
#      replies on the second call → "(No output)")
#   - applies the macOS keychain auth workaround (ANTHROPIC_AUTH_TOKEN +
#     hasCompletedOnboarding=true) so the local API key is actually used
#     instead of the model-selection login prompt
#   - disables all non-essential traffic to api.anthropic.com (telemetry,
#     statsig, marketplace auto-install, autoupdater, background tasks).
#     Without CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 the CLI still
#     reaches out to api.anthropic.com on startup even with
#     ANTHROPIC_BASE_URL pointing at localhost — which means "running
#     offline" is not actually offline by default.
#
# Use this for: code Q&A, snippet generation, debugging help, conversations.
# For tool-driven sessions, see "Claude Agentico.command".

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || echo $HOME/.local/bin/claude)}"

MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/Qwen2.5-Coder-14B-Instruct-4bit" \
  "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")"

# 4-bit KV cache from token 0 keeps the chat session under 16 GB even after
# many turns of accumulated context.
export MLX_KV_BITS="${MLX_KV_BITS:-4}"
export MLX_KV_QUANT_START="${MLX_KV_QUANT_START:-0}"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Qwen 2.5 Coder 14B 4-bit on MLX (KV=4-bit, ~10-15 tok/s in 16 GB)..."

clear
echo ""
echo "  Claude Code LOCAL - Modo CHAT (sem ferramentas)"
echo "  Modelo: Qwen 2.5 Coder 14B (4-bit MLX, ~7.8 GB)"
echo "  100% on-device - sem cloud, sem custo de API"
echo ""
echo "  Use para: perguntas de codigo, conceitos, debug, conversa."
echo "  Para criar/editar arquivos use o launcher Agentico."
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
ANTHROPIC_AUTH_TOKEN=sk-local \
DISABLE_LOGIN_COMMAND=1 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
DISABLE_AUTOUPDATER=1 \
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1 \
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 \
CLAUDE_SESSION_LABEL="Local Chat" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --tools "" \
  --effort low \
  --settings "$SCRIPT_DIR/lib/local-settings.json"
