#!/bin/bash
# Claude Agentico - Claude Code on Qwen 2.5 Coder 14B in AGENTIC mode (with tools)
# Double-click to launch
#
# Designed for Mac base/Pro with 16-32 GB unified memory:
#   - forces --effort low to disable Claude Code 2.1 extended thinking
#     (small models can't handle the two-call thinking → answer flow)
#   - applies the macOS keychain auth workaround so ANTHROPIC_API_KEY is
#     actually honored in interactive mode
#   - disables non-essential traffic so the CLI does not reach out to
#     api.anthropic.com on startup (telemetry, statsig, marketplace,
#     autoupdater). Without this, "100% offline" is not actually offline.
#
# Why Qwen 2.5 Coder 14B 4-bit MLX:
#   - 7.8 GB weights → fits in 16 GB unified memory without swapping
#   - Reliable tool-call emission (validated against Bash/Read/Glob)
#   - Native MLX 4-bit quant for Apple Silicon (no GGUF translation)
#
# Why MLX_KV_BITS=4 / MLX_KV_QUANT_START=0:
#   - Claude Code 2.1 sends a ~5860-token system prompt on every request,
#     which makes the KV cache the main RAM consumer (not the weights).
#   - Quantizing the cache to 4-bit from token 0 keeps prefill below the
#     16 GB ceiling and avoids kIOGPUCommandBufferCallbackErrorOutOfMemory
#     crashes mid-conversation.
#
# Tool-call reliability on a 16 GB Mac is lower than on Max/Ultra hardware.
# Expect ~10-15 tok/s and occasional garbled tool calls; the server already
# retries via recover_garbled_tool_json. For mission-critical agentic work,
# fall back to the cloud Claude (claude without ANTHROPIC_BASE_URL).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || echo $HOME/.local/bin/claude)}"

MLX_MODEL_DEFAULT="$(resolve_mlx_model \
  "$HOME/.cache/huggingface/hub/Qwen2.5-Coder-14B-Instruct-4bit" \
  "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")"

# 4-bit KV cache from token 0: required to stay under 16 GB with Claude
# Code 2.1's ~5860-token system prompt. Override with stronger machines.
export MLX_KV_BITS="${MLX_KV_BITS:-4}"
export MLX_KV_QUANT_START="${MLX_KV_QUANT_START:-0}"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Qwen 2.5 Coder 14B 4-bit on MLX (KV=4-bit, ~10-15 tok/s in 16 GB)..."

clear
echo ""
echo "  Claude Code LOCAL - Modo AGENTICO (com ferramentas)"
echo "  Modelo: Qwen 2.5 Coder 14B (4-bit MLX, ~7.8 GB)"
echo "  100% on-device - sem cloud, sem custo de API"
echo ""
echo "  Em 16 GB de RAM espere ~10-15 tok/s e ocasionais"
echo "  falhas em sequencias longas de tool-calls."
echo "  Para conversa simples, use o launcher Chat."
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
ANTHROPIC_AUTH_TOKEN=sk-local \
DISABLE_LOGIN_COMMAND=1 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
DISABLE_AUTOUPDATER=1 \
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1 \
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 \
CLAUDE_SESSION_LABEL="Local Agentic" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --effort low \
  --permission-mode auto \
  --settings "$SCRIPT_DIR/lib/local-settings.json"
