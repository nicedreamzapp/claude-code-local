#!/bin/bash
# Claude Code Local — One-command setup
# Works on any Apple Silicon Mac
# Usage: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Setup                   ║"
echo "║     Run AI coding agents on your Mac            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Detect memory
MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}')
echo "Detected: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "Memory: ${MEM_GB} GB"
echo ""

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
  echo "ERROR: This requires Apple Silicon (M1 or later)."
  exit 1
fi

# Install Homebrew if missing
if ! command -v brew &>/dev/null; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install Ollama
if ! command -v ollama &>/dev/null; then
  echo "Installing Ollama..."
  brew install ollama
fi

# Start Ollama
if ! pgrep -x "ollama" >/dev/null 2>&1; then
  echo "Starting Ollama..."
  ollama serve >/dev/null 2>&1 &
  sleep 3
fi

# Choose model based on RAM
echo "Selecting model for your ${MEM_GB} GB Mac..."
if [ "$MEM_GB" -ge 96 ]; then
  MODEL="qwen3.5:122b"
  MODEL_DESC="122B (81 GB) — full power"
elif [ "$MEM_GB" -ge 32 ]; then
  MODEL="qwen3.5:35b"
  MODEL_DESC="35B (20 GB) — great coding"
else
  MODEL="qwen3.5:4b"
  MODEL_DESC="4B (3.4 GB) — lightweight"
fi

echo "Selected: $MODEL — $MODEL_DESC"
echo ""

# Pull models
echo "Downloading $MODEL (this may take a while)..."
ollama pull "$MODEL"

# Always pull the small model for browser agent
if [ "$MODEL" != "qwen3.5:4b" ]; then
  echo "Downloading qwen3.5:4b for browser agent..."
  ollama pull qwen3.5:4b
fi

# Set up proxy
PROXY_DIR="$HOME/.local/claude-local-proxy"
mkdir -p "$PROXY_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/proxy/proxy.py" "$PROXY_DIR/proxy.py"

# Update MODEL_MAP if not using 122b
if [ "$MODEL" != "qwen3.5:122b" ]; then
  sed -i '' "s/qwen3.5:122b/$MODEL/g" "$PROXY_DIR/proxy.py"
fi

# Create desktop launcher
CLAUDE_BIN=$(which claude 2>/dev/null || echo "$HOME/.local/bin/claude")
if [ ! -f "$CLAUDE_BIN" ]; then
  echo ""
  echo "WARNING: Claude Code not found. Install it with:"
  echo "  npm install -g @anthropic-ai/claude-code"
  echo ""
  CLAUDE_BIN="\$HOME/.local/bin/claude"
fi

cat > "$HOME/Desktop/Claude Local.command" << LAUNCHER
#!/bin/bash
# Claude Code — Local AI
CLAUDE_BIN="$CLAUDE_BIN"
PROXY="$PROXY_DIR/proxy.py"

if ! pgrep -x "ollama" >/dev/null 2>&1; then
  ollama serve >/dev/null 2>&1 &
  sleep 3
fi

if ! lsof -i :4000 >/dev/null 2>&1; then
  python3 "\$PROXY" >/dev/null 2>&1 &
  sleep 2
fi

clear
echo ""
echo "  → Claude Code with LOCAL AI ($MODEL)"
echo "  → Running on your Mac — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \\
ANTHROPIC_API_KEY=sk-local \\
exec "\$CLAUDE_BIN" --model claude-sonnet-4-6
LAUNCHER

chmod +x "$HOME/Desktop/Claude Local.command"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Setup complete!                              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  Model: $MODEL"
echo "║  Proxy: $PROXY_DIR/proxy.py"
echo "║  Launcher: ~/Desktop/Claude Local.command"
echo "║                                                  ║"
echo "║  Double-click 'Claude Local' on your Desktop     ║"
echo "║  to start coding with local AI.                  ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
