#!/data/data/com.termux/files/usr/bin/bash
# Claude Code Local — Android setup (Termux, llama.cpp, a vision model, on the phone).
# Called by install-android.sh after the repo is cloned. Installs the few Termux
# packages it needs, then ccl.py picks the model that fits this phone's RAM at the
# highest precision, downloads the pinned llama.cpp android-arm64 build, the
# model, and Claude Code's linux-arm64-musl build with its musl loader.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${PREFIX:-}" ] || [ "${PREFIX#/data/data/com.termux}" = "$PREFIX" ]; then
  echo "✗ This script runs inside Termux on Android."
  exit 1
fi

echo ""
echo "Claude Code Local — Android setup (llama.cpp + a vision model, 100% on the phone)"
echo ""

export DEBIAN_FRONTEND=noninteractive
NEED=""
command -v python3  >/dev/null 2>&1 || NEED="$NEED python"
command -v patchelf >/dev/null 2>&1 || NEED="$NEED patchelf"
command -v rg       >/dev/null 2>&1 || NEED="$NEED ripgrep"
if [ -n "$NEED" ]; then
  echo "→ pkg install$NEED"
  yes | pkg install -y $NEED
fi

PY="$(command -v python3)"
"$PY" -I "$SCRIPT_DIR/ccl.py" detect || true
echo ""
echo "Downloading llama.cpp, the model and Claude Code (one time)..."
"$PY" -I "$SCRIPT_DIR/ccl.py" install

LAUNCHER="$PREFIX/bin/claude-local"
cat > "$LAUNCHER" <<LAUNCH
#!/data/data/com.termux/files/usr/bin/bash
# Claude Code — local AI on this phone. Starts llama-server + the proxy, then Claude Code.
exec "$PY" -I "$SCRIPT_DIR/ccl.py" run "\$@"
LAUNCH
chmod +x "$LAUNCHER"

echo ""
echo "Setup complete. Start it with:  claude-local"
echo "Servers only (for another app on the phone):  claude-local --no-claude"
echo ""
