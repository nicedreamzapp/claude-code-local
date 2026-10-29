#!/data/data/com.termux/files/usr/bin/bash
# Claude Code Local — Android one-liner (run inside Termux):
#
#   curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install-android.sh | bash
#
# Everything runs on the phone: llama.cpp serves a vision + tool-calling model,
# the claude-code-local proxy speaks the Anthropic API to it, and Claude Code
# talks to the proxy. No cloud. Needs an arm64 phone; 8 GB RAM runs a 2B model
# at full BF16 precision, 12 GB+ gets a 4B.

set -euo pipefail

REPO_URL="${CCL_REPO:-https://github.com/nicedreamzapp/claude-code-local}"
BRANCH="${CCL_BRANCH:-main}"
INSTALL_DIR="${CCL_DIR:-$HOME/claude-code-local}"

if [ -z "${PREFIX:-}" ] || [ "${PREFIX#/data/data/com.termux}" = "$PREFIX" ]; then
  echo "✗ Run this inside Termux on Android (https://github.com/termux/termux-app)."
  exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
  echo "✗ Needs an arm64 phone. Detected: $(uname -m)"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
echo "→ Updating Termux packages (one time)..."
yes | pkg upgrade -y -o Dpkg::Options::=--force-confnew >/dev/null
yes | pkg install -y git python patchelf ripgrep >/dev/null

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "→ Updating $INSTALL_DIR..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  echo "→ Cloning into $INSTALL_DIR..."
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

exec bash "$INSTALL_DIR/portable/setup-android.sh"
