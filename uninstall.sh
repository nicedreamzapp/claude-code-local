#!/bin/bash
# Claude Code Local — Uninstall
# Removes everything setup.sh created. Does NOT uninstall Homebrew or Python.
#
# Usage: bash uninstall.sh [--keep-models] [--with voice,browser,... | --all] [--purge]
#   --keep-models   Skip deleting downloaded HuggingFace models (can be 18-75 GB)
#   --with LIST     Also remove these add-ons (voice browser phone remote moonstone failover)
#   --all           Also remove every add-on
#   --purge         With --with/--all: add-ons delete their data and config too

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEEP_MODELS=0
WITH=""
PURGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --keep-models) KEEP_MODELS=1 ;;
    --with) WITH="${WITH:+$WITH,}${2:-}"; shift ;;
    --with=*) WITH="${WITH:+$WITH,}${1#--with=}" ;;
    --all) WITH="all" ;;
    --purge) PURGE="--purge" ;;
    -h|--help)
      sed -n '5,9p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
  shift
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

# ── "keep going" command and its shell rc line ─────────────────
# setup.sh writes it to ~/.zshrc; portable/setup-linux.sh to ~/.bashrc (and ~/.zshrc if present).
KG="$HOME/.local/bin/keepgoing"
if [ -L "$KG" ] && readlink "$KG" | grep -q "/bin/keepgoing$"; then
  rm -f "$KG"
  echo "✓ Removed $KG"
fi
for RC in "$HOME/.zshrc" "$HOME/.bashrc"; do
  if [ -f "$RC" ] && grep -q "scripts/keepgoing-shell.sh" "$RC"; then
    tmp="$(mktemp)"
    grep -v -e "scripts/keepgoing-shell.sh" -e '^# Claude Code Local: type "keep going"' "$RC" >"$tmp" || true
    cat "$tmp" >"$RC" && rm -f "$tmp"
    echo "✓ Removed the keep going line from ~/$(basename "$RC")"
  fi
done

# The iMessage phone scripts in ~/.claude/ belong to the "phone" add-on
# (claude-screen-to-phone); remove them with --with phone.

# ── Stop the running MLX server ───────────────────────────────
# Only a server started from our server.py; anything else on the port is left alone.
CCL_PORT="${CCL_PORT:-4000}"
MLX_PID=$(lsof -ti "tcp:$CCL_PORT" -sTCP:LISTEN 2>/dev/null || true)
stopped=0
for pid in $MLX_PID; do
  if ps -o command= -p "$pid" 2>/dev/null | grep -q "mlx-native-server/server.py\|proxy/server.py"; then
    kill "$pid" 2>/dev/null || true
    echo "✓ Stopped MLX server (PID $pid)"
    stopped=1
  fi
done
[ "$stopped" -eq 1 ] || echo "· No Claude Code Local server running on :$CCL_PORT"

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
    "models--lmstudio-community--Qwen3.8-27B-MLX-8bit"
    "models--divinetribe--gemma-4-31b-it-abliterated-4bit-mlx"
    "models--divinetribe--gemma-4-12B-it-abliterated-4bit-mlx-text"
    "models--divinetribe--Hermes-4-14B-abliterated-4bit-mlx"
    "models--mlx-community--gemma-4-e4b-it-4bit"
    # small-Mac tier (scripts/mac-tier.sh), flat folders
    "Qwen3.5-4B-bf16"
    "Qwen3.5-2B-MLX-8bit"
    # older setup.sh picks
    "models--mlx-community--Qwen3.5-122B-A10B-4bit"
    "models--mlx-community--Qwen3.5-4B-4bit"
  )

  models_removed=0
  for d in "${MODEL_DIRS[@]}"; do
    if [ -d "$HF_CACHE/$d" ]; then
      SIZE=$(du -sh "$HF_CACHE/$d" 2>/dev/null | cut -f1)
      rm -rf "${HF_CACHE:?}/$d"
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

# ── Add-ons (each repo's own uninstaller) ─────────────────────
if [ -n "$WITH" ]; then
  bash "$SCRIPT_DIR/scripts/addons.sh" uninstall "$WITH" ${PURGE:+"$PURGE"} || true
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
echo "║  Add-ons stay unless you pass --with or --all.   ║"
echo "║  To reinstall: bash setup.sh                     ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
