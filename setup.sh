#!/bin/bash
# Claude Code Local — One-command setup
# Apple Silicon only. Installs MLX, downloads a model from the lineup,
# and creates a desktop launcher that runs Claude Code 100% on-device.
#
# Usage: bash setup.sh [--with voice,browser,phone,remote,moonstone,failover | --with all] [--no-addons]
#
# After the model is set up it offers the add-ons (scripts/addons.sh). --with installs
# them without asking; with no terminal and no --with, none are installed.
# CCL_SKIP_MODEL=1 skips the mlx-lm install and the model download (for testing).

set -e

WITH="${CCL_WITH:-}"
NO_ADDONS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --with) WITH="${WITH:+$WITH,}${2:-}"; shift ;;
    --with=*) WITH="${WITH:+$WITH,}${1#--with=}" ;;
    --all) WITH="all" ;;
    --no-addons) NO_ADDONS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
  shift
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Setup                    ║"
echo "║     Pick your fighter. Run AI on your Mac.       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── System detection ──────────────────────────────────────────
MEM_GB=$(/usr/sbin/sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}')
CHIP=$(/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')

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

if [ "${CCL_SKIP_MODEL:-0}" = "1" ]; then
  echo "CCL_SKIP_MODEL=1: skipping the mlx-lm install"
else
  echo "Installing mlx-lm into virtualenv..."
  "$MLX_VENV/bin/pip" install --quiet --upgrade pip
  "$MLX_VENV/bin/pip" install --quiet --upgrade mlx-lm
fi

# ── Pick a model from the lineup ──────────────────────────────
# Under 32 GB the small-Mac tier decides (scripts/mac-tier.sh, measured in
# docs/SMALL-MACS.md): a bf16 model that reads images, writes code and calls
# tools, with the context capped to what the Mac can hold.
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CCL_MEM_GB="${CCL_MEM_GB:-$MEM_GB}"
source "$REPO_DIR/scripts/mac-tier.sh"

echo ""
echo "Selecting a model from the lineup for your ${MEM_GB} GB Mac..."
if [ "$CCL_TIER" != "none" ]; then
  MODEL_ID="$CCL_MODEL_ID"
  MODEL_LABEL="$CCL_MODEL_LABEL, ${CCL_CONTEXT}-token context"
  MODEL_TIER="🟡 small Mac ($CCL_TIER)"
elif [ "$MEM_GB" -ge 96 ]; then
  # Passed a Claude Code smoke test on an M5 Max, 2026-09-19 (wrote hello.py, ran it, 38 s).
  MODEL_ID="lmstudio-community/Qwen3.8-27B-MLX-8bit"
  MODEL_LABEL="Qwen 3.8 27B, 8-bit (~29 GB)"
  MODEL_TIER="🟣 max"
elif [ "$MEM_GB" -ge 64 ]; then
  MODEL_ID="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"
  MODEL_LABEL="Gemma 4 31B Abliterated, 4-bit (~18 GB)"
  MODEL_TIER="🟢 fast"
else
  MODEL_ID="divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text"
  MODEL_LABEL="Gemma 4 12B Abliterated, 4-bit (~11 GB)"
  MODEL_TIER="🟢 fast"
fi
# Hermes 4 14B (4-bit, text only) was the 16 GB pick before the tier and still
# works there: CCL_TIER=none MLX_MODEL=divinetribe/Hermes-4-14B-abliterated-4bit-mlx

echo "Selected: $MODEL_TIER  $MODEL_LABEL"
echo "Model ID: $MODEL_ID"
echo ""

# ── Download model ────────────────────────────────────────────
if [ "${CCL_SKIP_MODEL:-0}" = "1" ]; then
  echo "CCL_SKIP_MODEL=1: skipping the download of $MODEL_ID"
elif [ "$CCL_TIER" != "none" ]; then
  # The tier models are vision models: mlx-vlm loads them whole, and they go to
  # a flat folder (the layout resolve_mlx_model and the launchers look for).
  echo "Installing mlx-vlm (lets the model see images)..."
  "$MLX_VENV/bin/pip" install --quiet --upgrade mlx-vlm
  echo "Downloading $MODEL_ID to $CCL_MODEL_DIR (one time)..."
  "$MLX_VENV/bin/python3" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_ID", local_dir="$CCL_MODEL_DIR")
print("Done.")
PY
else
  echo "Downloading $MODEL_ID (one time, 5-30 GB)..."
  "$MLX_VENV/bin/python3" - <<PY
from mlx_lm.utils import load
load("$MODEL_ID")
print("Done.")
PY
fi

# ── Install MLX server ────────────────────────────────────────
# We install it as a symlink into this repo, not a copy. That way if you edit
# proxy/server.py in the repo (to fix a bug, add a feature, pull a PR), the
# change takes effect on the running server after the next restart — with no
# risk of the running copy silently drifting from the version in git.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$HOME/.local/mlx-native-server"
mkdir -p "$SERVER_DIR"
ln -sf "$SCRIPT_DIR/proxy/server.py" "$SERVER_DIR/server.py"
echo "MLX server installed (symlink) → $SERVER_DIR/server.py -> $SCRIPT_DIR/proxy/server.py"

# ── Create desktop launcher ───────────────────────────────────
# Claude Code itself: installed here if missing, with the same official installer as on Linux.
. "$SCRIPT_DIR/scripts/install-claude-code.sh"
if ! CLAUDE_BIN="$(ccl_ensure_claude)"; then
  echo ""
  CLAUDE_BIN="\$HOME/.local/bin/claude"
fi

mkdir -p "$HOME/Desktop"
LAUNCHER="$HOME/Desktop/Claude Local.command"
if [ "$CCL_TIER" != "none" ]; then
  # Small Macs: the repo's own launcher, which re-reads the tier settings
  # (model, context ceiling, compaction) every time it starts.
  cat > "$LAUNCHER" <<LAUNCH
#!/bin/bash
# Claude Code — Local AI, small-Mac tier ($MODEL_LABEL)
CCL_TIER="\${CCL_TIER:-$CCL_TIER}" CLAUDE_BIN="$CLAUDE_BIN" MLX_PYTHON="$MLX_VENV/bin/python3" MLX_SERVER="$SERVER_DIR/server.py" \\
  exec "$SCRIPT_DIR/launchers/Small Mac.command"
LAUNCH
else
cat > "$LAUNCHER" <<LAUNCH
#!/bin/bash
# Claude Code — Local AI ($MODEL_LABEL)
CLAUDE_BIN="$CLAUDE_BIN"
MLX_PYTHON="$MLX_VENV/bin/python3"
MLX_SERVER="$SERVER_DIR/server.py"

if ! lsof -i :4000 >/dev/null 2>&1; then
  MLX_MODEL="$MODEL_ID" "\$MLX_PYTHON" "\$MLX_SERVER" >"/tmp/mlx-server-\$(id -un).log" 2>&1 &
  echo "  Loading $MODEL_LABEL on MLX..."
  while ! curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; do
    sleep 2
  done
fi

clear
echo ""
echo "  → Claude Code with LOCAL AI"
echo "  → $MODEL_LABEL"
echo "  → 100% on-device, no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \\
ANTHROPIC_API_KEY=sk-local \\
exec "\$CLAUDE_BIN" --model claude-sonnet-4-6
LAUNCH
fi

chmod +x "$LAUNCHER"

# ── "keep going": continue a Claude conversation on a free or local model ──
mkdir -p "$HOME/.local/bin"
ln -sf "$SCRIPT_DIR/bin/keepgoing" "$HOME/.local/bin/keepgoing"
ZSHRC="$HOME/.zshrc"
KEEP_LINE="source \"$SCRIPT_DIR/scripts/keepgoing-shell.sh\""
if ! grep -qF "$KEEP_LINE" "$ZSHRC" 2>/dev/null; then
  printf '\n# Claude Code Local: type "keep going" when Claude runs out of credits\n%s\n' "$KEEP_LINE" >> "$ZSHRC"
fi
echo "keep going installed → ~/.local/bin/keepgoing (open a new terminal to use it)"
# Optional: run keep going by itself when Claude hits a usage limit. setup.sh never edits
# ~/.claude/settings.json, so it only prints the snippet (docs/AUTO-HANDOFF.md explains it).
echo ""
echo "Want keep going to start by itself when you hit a Claude usage limit? Add this to"
echo "~/.claude/settings.json (merge it into an existing \"hooks\" block if you have one):"
cat <<HOOK
  "hooks": {
    "StopFailure": [
      { "matcher": "rate_limit",
        "hooks": [ { "type": "command", "command": "python3 $SCRIPT_DIR/scripts/limit-handoff.py" } ] }
    ]
  }
HOOK

# ── Add-ons: voice, browser, phone, remote screen, Moonstone, failover ──
# Each one is installed by its own repo's one-line installer (scripts/addons.sh).
# The iMessage phone scripts now come from the "phone" add-on (claude-screen-to-phone),
# which puts them in ~/.claude/ exactly where this script used to.
if [ "$NO_ADDONS" = "1" ]; then
  echo ""
  echo "Skipping add-ons (--no-addons). Add them later: bash scripts/addons.sh menu"
elif [ -n "$WITH" ]; then
  bash "$SCRIPT_DIR/scripts/addons.sh" install "$WITH" || echo "  (setup itself finished; see the add-on messages above)"
else
  bash "$SCRIPT_DIR/scripts/addons.sh" menu || echo "  (setup itself finished; see the add-on messages above)"
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
echo "║  Want a different model? See launchers/ for      ║"
echo "║  Gemma 4 Code, Qwen 3.8 Code, Browser Agent,     ║"
echo "║  and Narrative Gemma launchers.                  ║"
echo "║                                                  ║"
echo "║  Add-ons any time: bash scripts/addons.sh menu   ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
