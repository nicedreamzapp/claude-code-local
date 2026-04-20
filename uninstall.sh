#!/bin/bash
# Claude Code Local — Uninstall
# Removes everything setup.sh created. Does NOT uninstall Homebrew or Python.
#
# Usage: bash uninstall.sh [--keep-models]
#   --keep-models   Skip deleting downloaded HuggingFace models (can be 18-75 GB)

set -e

KEEP_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --keep-models) KEEP_MODELS=1 ;;
    -h|--help)
      echo "Usage: bash uninstall.sh [--keep-models]"
      echo "  --keep-models   Keep downloaded HuggingFace model weights"
      exit 0
      ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Uninstall                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

removed=0

# ── Desktop launcher ──────────────────────────────────────────
LAUNCHER="$HOME/Desktop/Claude Local.command"
if [ -f "$LAUNCHER" ]; then
  rm "$LAUNCHER"
  echo "✓ Removed desktop launcher: $LAUNCHER"
  removed=$((removed + 1))
else
  echo "· Desktop launcher not found (already removed)"
fi

# ── MLX server symlink dir ────────────────────────────────────
SERVER_DIR="$HOME/.local/mlx-native-server"
if [ -d "$SERVER_DIR" ]; then
  rm -rf "$SERVER_DIR"
  echo "✓ Removed MLX server dir: $SERVER_DIR"
  removed=$((removed + 1))
else
  echo "· MLX server dir not found (already removed)"
fi

# ── MLX virtualenv ────────────────────────────────────────────
MLX_VENV="$HOME/.local/mlx-server"
if [ -d "$MLX_VENV" ]; then
  rm -rf "$MLX_VENV"
  echo "✓ Removed MLX virtualenv: $MLX_VENV"
  removed=$((removed + 1))
else
  echo "· MLX virtualenv not found (already removed)"
fi

# ── Phone/iMessage scripts ────────────────────────────────────
CLAUDE_DIR="$HOME/.claude"
PHONE_SCRIPTS=(
  "imessage-send.sh"
  "imessage-send-image.sh"
  "imessage-send-video.sh"
  "imessage-toggle.sh"
  "imessage-receive.sh"
  "screen-to-phone-config.sh"
)

phone_removed=0
for s in "${PHONE_SCRIPTS[@]}"; do
  if [ -f "$CLAUDE_DIR/$s" ]; then
    rm "$CLAUDE_DIR/$s"
    phone_removed=$((phone_removed + 1))
  fi
done

if [ "$phone_removed" -gt 0 ]; then
  echo "✓ Removed $phone_removed phone/iMessage scripts from $CLAUDE_DIR/"
  removed=$((removed + phone_removed))
else
  echo "· No phone scripts found (already removed)"
fi

# ── Kill running MLX server ───────────────────────────────────
if lsof -i :4000 >/dev/null 2>&1; then
  MLX_PID=$(lsof -ti :4000 2>/dev/null || true)
  if [ -n "$MLX_PID" ]; then
    kill "$MLX_PID" 2>/dev/null || true
    echo "✓ Stopped MLX server (PID $MLX_PID)"
  fi
else
  echo "· No MLX server running on :4000"
fi

# ── MLX server log ────────────────────────────────────────────
if [ -f /tmp/mlx-server.log ]; then
  rm /tmp/mlx-server.log
  echo "✓ Removed /tmp/mlx-server.log"
fi

# ── Downloaded models (optional) ──────────────────────────────
HF_CACHE="$HOME/.cache/huggingface/hub"
if [ "$KEEP_MODELS" -eq 1 ]; then
  echo ""
  echo "· Keeping downloaded models in $HF_CACHE (--keep-models)"
elif [ -d "$HF_CACHE" ]; then
  # Only list the models setup.sh might have downloaded
  MODEL_DIRS=(
    "models--mlx-community--Qwen3.5-122B-A10B-4bit"
    "models--divinetribe--gemma-4-31b-it-abliterated-4bit-mlx"
    "models--mlx-community--Qwen3.5-4B-4bit"
  )

  models_removed=0
  for d in "${MODEL_DIRS[@]}"; do
    if [ -d "$HF_CACHE/$d" ]; then
      SIZE=$(du -sh "$HF_CACHE/$d" 2>/dev/null | cut -f1)
      rm -rf "$HF_CACHE/$d"
      echo "✓ Removed model $d ($SIZE)"
      models_removed=$((models_removed + 1))
    fi
  done

  if [ "$models_removed" -eq 0 ]; then
    echo "· No Claude Code Local models found in HuggingFace cache"
  fi
else
  echo "· No HuggingFace cache found"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Uninstall complete.                          ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  NOT removed (shared system tools):              ║"
echo "║  · Homebrew                                      ║"
echo "║  · Python 3.12                                   ║"
echo "║  · Claude Code CLI                               ║"
echo "║                                                  ║"
echo "║  To reinstall: bash setup.sh                     ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
