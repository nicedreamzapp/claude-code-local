#!/bin/bash
# Claude Code Local — one-line installer
#
#   curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
#
# Clones the repo to ~/claude-code-local (or updates it if it's already there)
# and hands off to setup.sh, which picks a model for your Mac's RAM and builds
# the desktop launcher. Apple Silicon only.

set -euo pipefail

REPO_URL="https://github.com/nicedreamzapp/claude-code-local"
INSTALL_DIR="${CCL_DIR:-$HOME/claude-code-local}"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Installer                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Refuse to run anywhere it can't work ──────────────────────
if [ "$(uname -s)" != "Darwin" ]; then
  echo "✗ This only runs on macOS (Apple Silicon)."
  exit 1
fi

if [ "$(uname -m)" != "arm64" ]; then
  echo "✗ This needs Apple Silicon (M1 or newer). Detected: $(uname -m)"
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
  echo "  ⚠️  Under 16 GB. setup.sh will pick a small 4B model — it works,"
  echo "      but tool calling will be noticeably less reliable."
  echo ""
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
exec bash setup.sh
