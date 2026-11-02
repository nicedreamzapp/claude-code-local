#!/bin/bash
# Claude Code Local — Linux setup (llama.cpp backend, vision model).
# Called by install.sh after the repo is cloned. Detects your hardware, downloads
# a pinned llama.cpp build and the best vision model that fits, and writes a
# launcher. No Homebrew, no MLX — just Python 3 and llama.cpp.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Linux setup              ║"
echo "║     llama.cpp + a vision model, 100% local       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

PY="$(command -v python3)"
echo "Detecting hardware and picking a model..."
"$PY" -I "$SCRIPT_DIR/ccl.py" detect || true
echo ""

echo "Downloading llama.cpp + model (one time, can be large)..."
"$PY" -I "$SCRIPT_DIR/ccl.py" install

# ── Launcher ──────────────────────────────────────────────────
LAUNCHER="$HOME/.local/bin/claude-local"
mkdir -p "$HOME/.local/bin"
cat > "$LAUNCHER" <<LAUNCH
#!/bin/bash
# Claude Code — local AI (llama.cpp). Starts the servers and launches Claude Code.
exec "$PY" -I "$SCRIPT_DIR/ccl.py" run "\$@"
LAUNCH
chmod +x "$LAUNCHER"

# ── "keep going": continue a Claude conversation on a free or local model ──
# Same command as on a Mac: bin/keepgoing hands off to ccl.py keepgoing on Linux.
ln -sf "$REPO_DIR/bin/keepgoing" "$HOME/.local/bin/keepgoing"
KEEP_LINE="source \"$REPO_DIR/scripts/keepgoing-shell.sh\""
if [ "${CCL_NO_SHELL_HOOK:-0}" = "1" ]; then
  echo "CCL_NO_SHELL_HOOK=1: not adding the keep going line to your shell rc"
else
  for RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
    # .bashrc always (it is the default shell); .zshrc only if you use zsh
    [ "$RC" = "$HOME/.zshrc" ] && [ ! -f "$RC" ] && continue
    if ! grep -qF "$KEEP_LINE" "$RC" 2>/dev/null; then
      printf '\n# Claude Code Local: type "keep going" when Claude runs out of credits\n%s\n' "$KEEP_LINE" >> "$RC"
    fi
  done
fi
echo "keep going installed → ~/.local/bin/keepgoing (open a new terminal to use it)"

# A desktop entry too, if this is a graphical session.
if [ -d "$HOME/.local/share/applications" ] || mkdir -p "$HOME/.local/share/applications" 2>/dev/null; then
  cat > "$HOME/.local/share/applications/claude-code-local.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Claude Code Local
Comment=Claude Code running 100% on-device via llama.cpp
Exec=$LAUNCHER
Terminal=true
Categories=Development;
DESK
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Setup complete!                              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Start it with:   claude-local                   ║"
echo "║  (or:  python3 $SCRIPT_DIR/ccl.py run)"
echo "║                                                  ║"
echo "║  Hit your Claude limit? Type exit, then:         ║"
echo "║                   keep going                     ║"
echo "║                                                  ║"
echo "║  If 'claude-local' isn't found, add this to      ║"
echo "║  your shell rc:  export PATH=\$HOME/.local/bin:\$PATH ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
if ! command -v claude >/dev/null 2>&1; then
  echo "Note: Claude Code CLI not found. Install it with:"
  echo "  curl -fsSL https://claude.ai/install.sh | bash"
  echo ""
fi
