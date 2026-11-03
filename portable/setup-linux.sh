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

# llama-server needs GNU OpenMP (libgomp.so.1); stock Ubuntu 24.04 doesn't ship it.
. "$SCRIPT_DIR/linux-deps.sh"
ccl_ensure_libgomp || echo "  (continuing; llama-server won't start until libgomp is installed)"
echo ""

echo "Downloading llama.cpp + model (one time, can be large)..."
"$PY" -I "$SCRIPT_DIR/ccl.py" install

# Check the binary we just installed can actually load: every library it needs is present.
CCL_CFG="${CCL_HOME:-$HOME/.cache/claude-code-local}/config.json"  # same folder as ccl.home()
LLAMA_BIN="$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["server_bin"])' \
  "$CCL_CFG" 2>/dev/null || true)"
if [ -n "$LLAMA_BIN" ] && [ -x "$LLAMA_BIN" ]; then
  MISSING="$(ccl_missing_libs "$LLAMA_BIN")"
  if [ -n "$MISSING" ]; then
    echo ""
    echo "WARNING: llama-server can't start, these libraries are missing: $MISSING"
    echo "  Install them (Debian/Ubuntu: sudo apt-get install -y libgomp1), then run: claude-local"
  fi
fi

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
# Claude Code itself: installed if missing, with the same official installer as on a Mac.
. "$REPO_DIR/scripts/install-claude-code.sh"
ccl_ensure_claude >/dev/null || true
echo ""
