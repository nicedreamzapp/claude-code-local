#!/bin/bash
# Claude Code Local — One-command setup
# Apple Silicon only. Installs MLX, downloads a model from the lineup,
# and creates a desktop launcher that runs Claude Code 100% on-device.
#
# Usage: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Setup                    ║"
echo "║     Pick your fighter. Run AI on your Mac.       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── System detection ──────────────────────────────────────────
MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}')
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')

echo "Detected: $CHIP"
echo "Memory:   ${MEM_GB} GB"
echo ""

if [[ $(uname -m) != "arm64" ]]; then
  echo "ERROR: This requires Apple Silicon (M1 or later)."
  exit 1
fi

# ── Homebrew ──────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# ── Python 3.12 + MLX ─────────────────────────────────────────
if ! command -v python3.12 &>/dev/null; then
  echo "Installing Python 3.12 (required for MLX)..."
  brew install python@3.12
fi

MLX_VENV="$HOME/.local/mlx-server"
if [ ! -d "$MLX_VENV" ]; then
  echo "Creating MLX virtualenv at $MLX_VENV..."
  python3.12 -m venv "$MLX_VENV"
fi

echo "Installing mlx-lm into virtualenv..."
"$MLX_VENV/bin/pip" install --quiet --upgrade pip
"$MLX_VENV/bin/pip" install --quiet --upgrade mlx-lm

# ── Pick a model from the lineup ──────────────────────────────
echo ""
echo "Selecting a model from the lineup for your ${MEM_GB} GB Mac..."
if [ "$MEM_GB" -ge 96 ]; then
  MODEL_ID="mlx-community/Qwen3.5-122B-A10B-4bit"
  MODEL_LABEL="Qwen 3.5 122B (THE BEAST — 65 tok/s)"
  MODEL_TIER="🔵 max"
elif [ "$MEM_GB" -ge 64 ]; then
  MODEL_ID="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"
  MODEL_LABEL="Gemma 4 31B Abliterated (THE QUICK ONE — ~15 tok/s)"
  MODEL_TIER="🟢 fast"
elif [ "$MEM_GB" -ge 32 ]; then
  MODEL_ID="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"
  MODEL_LABEL="Gemma 4 31B Abliterated (tight fit, may swap)"
  MODEL_TIER="🟡 squeeze"
else
  MODEL_ID="mlx-community/Qwen3.5-4B-4bit"
  MODEL_LABEL="Qwen 3.5 4B (lightweight, browser-agent friendly)"
  MODEL_TIER="🟠 small"
fi

echo "Selected: $MODEL_TIER  $MODEL_LABEL"
echo "Model ID: $MODEL_ID"
echo ""

# ── Download model ────────────────────────────────────────────
echo "Downloading $MODEL_ID (one time, can be 18-75 GB)..."
"$MLX_VENV/bin/python3" - <<PY
from mlx_lm.utils import load
load("$MODEL_ID")
print("Done.")
PY

# ── Install MLX server ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$HOME/.local/mlx-native-server"
mkdir -p "$SERVER_DIR"
cp "$SCRIPT_DIR/proxy/server.py" "$SERVER_DIR/server.py"
echo "MLX server installed → $SERVER_DIR/server.py"

# ── Create desktop launcher ───────────────────────────────────
CLAUDE_BIN=$(which claude 2>/dev/null || echo "$HOME/.local/bin/claude")
if [ ! -f "$CLAUDE_BIN" ]; then
  echo ""
  echo "WARNING: Claude Code not found. Install it with:"
  echo "  npm install -g @anthropic-ai/claude-code"
  echo ""
  CLAUDE_BIN="\$HOME/.local/bin/claude"
fi

LAUNCHER_LIB="$SCRIPT_DIR/launchers/lib/claude-local-common.sh"
LAUNCHER="$HOME/Desktop/Claude Local.command"
cat > "$LAUNCHER" <<LAUNCH
#!/bin/bash
set -euo pipefail
# Claude Code — Local AI ($MODEL_LABEL)
CLAUDE_BIN="$CLAUDE_BIN"
MLX_PYTHON="$MLX_VENV/bin/python3"
MLX_SERVER="$SERVER_DIR/server.py"
MLX_MODEL_DEFAULT="$MODEL_ID"
LAUNCHER_LIB="$LAUNCHER_LIB"

source "\$LAUNCHER_LIB"

MCP_CONFIG="\$(build_user_mcp_config)"

cleanup() {
  cleanup_mlx_server
  rm -f "\$MCP_CONFIG"
}

trap cleanup EXIT INT TERM

require_file "\$CLAUDE_BIN" "Claude Code CLI"
require_file "\$MLX_PYTHON" "MLX Python"
require_file "\$MLX_SERVER" "MLX server"

ensure_mlx_server "\${MLX_MODEL:-\$MLX_MODEL_DEFAULT}" "  Loading $MODEL_LABEL on MLX..."

clear
echo ""
echo "  → Claude Code with LOCAL AI"
echo "  → $MODEL_LABEL"
echo "  → 100% on-device, no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \\
ANTHROPIC_API_KEY=sk-local \\
"\$CLAUDE_BIN" --model claude-sonnet-4-6 \\
  --permission-mode auto \\
  --append-system-prompt-file "\$HOME/.claude/CLAUDE.md" \\
  --mcp-config "\$MCP_CONFIG"
exit \$?
LAUNCH

chmod +x "$LAUNCHER"

# ── Optional: iMessage / Screen-to-Phone tools ────────────────
echo ""
echo "Checking for optional iMessage phone-control tools..."
CLAUDE_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_DIR"

PHONE_SCRIPTS=(
  "scripts/imessage-send.sh"
  "scripts/imessage-send-image.sh"
  "scripts/imessage-send-video.sh"
  "scripts/imessage-toggle.sh"
  "scripts/imessage-receive.sh"
)

MISSING=0
for s in "${PHONE_SCRIPTS[@]}"; do
  if [ -f "$SCRIPT_DIR/$s" ]; then
    cp "$SCRIPT_DIR/$s" "$CLAUDE_DIR/$(basename $s)"
    chmod +x "$CLAUDE_DIR/$(basename $s)"
  else
    MISSING=1
  fi
done

if [ "$MISSING" -eq 0 ]; then
  echo "  ✅ Phone tools installed to ~/.claude/"
  if [ -f "$SCRIPT_DIR/config.example.sh" ] && [ ! -f "$SCRIPT_DIR/config.sh" ]; then
    cp "$SCRIPT_DIR/config.example.sh" "$SCRIPT_DIR/config.sh"
    echo "  → Edit config.sh with your iPhone number + Apple ID"
  fi
  cp "$SCRIPT_DIR/config.sh" "$CLAUDE_DIR/screen-to-phone-config.sh" 2>/dev/null || true
else
  echo "  ℹ️  Phone scripts not bundled — clone them separately if you want phone control:"
  echo "     https://github.com/nicedreamzapp/claude-screen-to-phone"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Setup complete!                              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  Model:    $MODEL_ID"
echo "║  Server:   $SERVER_DIR/server.py"
echo "║  Launcher: ~/Desktop/Claude Local.command"
echo "║                                                  ║"
echo "║  Double-click 'Claude Local' on your Desktop     ║"
echo "║  to start coding with local AI.                  ║"
echo "║                                                  ║"
echo "║  Want a different fighter? See launchers/ for    ║"
echo "║  Gemma 4 Code, Llama 70B, Browser Agent,         ║"
echo "║  and Narrative Gemma launchers.                  ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
