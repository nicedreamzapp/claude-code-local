#!/bin/bash
# Claude Code Local — one-line installer
#
#   curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
#
# Clones the repo to ~/claude-code-local (or updates it if it's already there)
# and hands off to the right setup for your machine:
#   macOS (Apple Silicon) -> setup.sh         MLX, on-device
#   Linux                 -> portable/setup-linux.sh   llama.cpp + a vision model
# Windows users run install.ps1 instead (see the README).
#
# Add-ons (macOS): after the model is set up you get a menu. Skip it with:
#   curl -fsSL .../install.sh | bash -s -- --with voice,browser,phone,remote,moonstone,failover
#   curl -fsSL .../install.sh | bash -s -- --with all
#   curl -fsSL .../install.sh | bash -s -- --no-addons

set -euo pipefail

SETUP_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --with) SETUP_ARGS+=("--with" "${2:-}"); shift ;;
    --with=*|--all|--no-addons) SETUP_ARGS+=("$1") ;;
    -h|--help) sed -n '2,17p' "$0" 2>/dev/null || true; exit 0 ;;
    *) echo "✗ Unknown option: $1"; exit 1 ;;
  esac
  shift
done

REPO_URL="${CCL_REPO:-https://github.com/nicedreamzapp/claude-code-local}"
INSTALL_DIR="${CCL_DIR:-$HOME/claude-code-local}"
OS="$(uname -s)"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local v1.0.0 — Installer         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Per-OS preflight ──────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
  if [ "$(uname -m)" != "arm64" ]; then
    echo "✗ macOS build needs Apple Silicon (M1 or newer). Detected: $(uname -m)"
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "✗ git not found. Install the Xcode command line tools first:"
    echo "    xcode-select --install"
    exit 1
  fi
  MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}')
  echo "  Mac:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
  echo "  Memory: ${MEM_GB} GB"
  echo ""
  if [ "$MEM_GB" -lt 16 ]; then
    echo "  ⚠️  Under 16 GB. setup.sh will pick Qwen 3.5 2B (8-bit, reads images, 16K context)."
    echo "      Single reads, edits and commands work; keep tasks small (docs/SMALL-MACS.md)."
    echo ""
  fi
elif [ "$OS" = "Linux" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "✗ git not found. Install it with your package manager, e.g.:"
    echo "    sudo apt-get install -y git   (Debian/Ubuntu)"
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "✗ python3 not found. Install it first, e.g.:"
    echo "    sudo apt-get install -y python3   (Debian/Ubuntu)"
    exit 1
  fi
  echo "  Linux:  $(uname -m)"
  echo ""
else
  echo "✗ Unsupported OS: $OS. macOS and Linux use this script; Windows uses install.ps1."
  exit 1
fi

# ── Clone or update ───────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
  echo "→ Found an existing install at $INSTALL_DIR, updating..."
  git -C "$INSTALL_DIR" pull --ff-only
elif [ -e "$INSTALL_DIR" ]; then
  echo "✗ $INSTALL_DIR already exists and is not a git checkout."
  echo "  Move it aside, or set a different location:"
  echo "    CCL_DIR=~/somewhere-else curl -fsSL .../install.sh | bash"
  exit 1
else
  echo "→ Cloning into $INSTALL_DIR..."
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

echo ""
cd "$INSTALL_DIR"
if [ "$OS" = "Darwin" ]; then
  exec bash setup.sh ${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"}
else
  if [ ${#SETUP_ARGS[@]} -gt 0 ]; then
    echo "  (add-ons are macOS only for now; ignoring ${SETUP_ARGS[*]})"
  fi
  exec bash portable/setup-linux.sh
fi
