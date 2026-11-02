#!/bin/bash
# Shared helpers for the claude-code-local launchers.
#
# The model-aware restart logic here was diagnosed by 0xshugo in PR #5
# (nicedreamzapp/claude-code-local#5). That PR as a whole had compatibility
# issues we couldn't merge, but this specific observation — that launchers
# only checked `lsof -i :4000` and would happily connect to the wrong model
# if one was already running — was the correct diagnosis, and this file
# fixes it.

MLX_SERVER="${MLX_SERVER:-$HOME/.local/mlx-native-server/server.py}"
MLX_PYTHON="${MLX_PYTHON:-$HOME/.local/mlx-server/bin/python3}"
# The server log, one per user: a shared /tmp/mlx-server.log mixed users' logs, and once another
# user owned it, this user's launcher couldn't write it at all.
MLX_LOG="${MLX_LOG:-/tmp/mlx-server-$(id -un).log}"
# How long to wait for /health: 180 checks, 2 seconds apart.
MLX_HEALTH_ATTEMPTS=180

# Read the running server's /health and extract the "model" field. Prints the
# model path/id on stdout, or nothing if the server isn't up.
_get_running_mlx_model() {
  curl -sf http://127.0.0.1:4000/health 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("model", ""))
except Exception:
    pass
' 2>/dev/null
}

# Compare desired vs running model. The running server reports either an HF
# id (e.g. "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx") or a resolved
# local path (e.g. "/Users/me/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx"),
# depending on how it was started. We compare the *basename* (last path
# component) case-insensitively to match either form.
_mlx_model_matches() {
  local desired="$1"
  local running="$2"
  [ -z "$running" ] && return 1
  local desired_base="${desired##*/}"
  local running_base="${running##*/}"
  local dl rl
  dl="$(printf '%s' "$desired_base" | tr '[:upper:]' '[:lower:]')"
  rl="$(printf '%s' "$running_base" | tr '[:upper:]' '[:lower:]')"
  [ "$dl" = "$rl" ]
}

_wait_for_mlx_health() {
  # 180 attempts × 2s = 6 minutes. Enough for a cold load of a large 8-bit model
  # on a warm file cache; not enough for a first-time download from HF — use
  # resolve_mlx_model to point at a local path and avoid downloads entirely.
  local attempts="${1:-$MLX_HEALTH_ATTEMPTS}"
  local i
  for i in $(seq 1 "$attempts"); do
    if curl -s http://localhost:4000/health 2>/dev/null | grep -q '"status": "ok"'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# Resolve a model reference to something mlx-lm will load without triggering
# a HuggingFace download. Prefers the local flat-folder path if it exists
# (i.e. the layout created by scripts/download-and-import.sh — a simple
# directory with config.json + safetensors files, NOT the standard
# "models--org--name/snapshots/<commit>" hub layout). Falls back to the HF
# id for users who haven't downloaded the model yet, in which case mlx-lm
# will pull it on first run.
#
# Usage:
#   MLX_MODEL_DEFAULT="$(resolve_mlx_model \
#     "$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx" \
#     "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")"
resolve_mlx_model() {
  local local_path="$1"
  local hf_id="$2"
  if [ -d "$local_path" ] && [ -f "$local_path/config.json" ]; then
    printf '%s\n' "$local_path"
  else
    printf '%s\n' "$hf_id"
  fi
}

# The launchers' server listens on 4000. Pids of the claude-code-local server(s) listening there:
# the listener on that one port whose command line is our server.py. A server on another port
# (another user's, a demo, a benchmark) is never touched, and neither is a non-server process.
_mlx_server_pids() {
  local port="${1:-4000}" pid listeners
  listeners="$(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null)"
  # Some Linux boxes (CI runners, containers) give lsof nothing; ss reads the same sockets.
  if [ -z "$listeners" ] && command -v ss >/dev/null 2>&1; then
    listeners="$(ss -Hltnp "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"
  fi
  # Last resort on Linux: match the listening socket's inode in /proc to the process holding it.
  if [ -z "$listeners" ] && [ -r /proc/net/tcp ]; then
    local hex inodes
    hex="$(printf '%04X' "$port")"
    inodes="$(awk -v h=":$hex" '$2 ~ h"$" && $4 == "0A" {print $10}' /proc/net/tcp /proc/net/tcp6 2>/dev/null)"
    for i in $inodes; do
      listeners="$listeners $(ls -l /proc/[0-9]*/fd 2>/dev/null | awk -v s="socket:[$i]" '/^\/proc\//{d=$1} index($0,s){split(d,a,"/"); print a[3]}' | sort -u)"
    done
  fi
  for pid in $listeners; do
    if ps -o command= -p "$pid" 2>/dev/null | grep -qE "mlx-native-server/server\.py|proxy/server\.py"; then
      printf '%s\n' "$pid"
    fi
  done
}

# Stop only the server on port 4000 (the one these launchers start), never every server.py on the
# machine: a `pkill -f mlx-native-server/server.py` here used to take down servers on other ports.
_stop_mlx_server() {
  local pids
  pids="$(_mlx_server_pids 4000)"
  [ -n "$pids" ] && kill $pids 2>/dev/null
  local i
  for i in $(seq 1 15); do
    if ! lsof -i :4000 >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Start the MLX server with the given model, or confirm an already-running
# server is loaded with that model. If the wrong model is running, stop it
# and restart with the desired one.
#
#   ensure_mlx_server DESIRED_MODEL LOADING_MESSAGE
#
# Any extra env vars the caller has already exported (MLX_BROWSER_MODE,
# MLX_APPEND_SYSTEM_PROMPT_FILE, etc.) will be inherited by the spawned
# server process.
ensure_mlx_server() {
  local desired="$1"
  local msg="$2"

  if lsof -i :4000 >/dev/null 2>&1; then
    local running
    running="$(_get_running_mlx_model)"
    if _mlx_model_matches "$desired" "$running"; then
      return 0
    fi
    echo "  Different model is loaded (${running:-unknown}) — restarting MLX server..."
    _stop_mlx_server || echo "  Warning: existing MLX server didn't exit cleanly, continuing anyway"
  fi

  echo "$msg"
  MLX_MODEL="$desired" \
  MLX_KV_BITS="${MLX_KV_BITS:-}" \
  MLX_KV_QUANT_START="${MLX_KV_QUANT_START:-}" \
  "$MLX_PYTHON" "$MLX_SERVER" >"$MLX_LOG" 2>&1 &
  if ! _wait_for_mlx_health; then
    echo "  ERROR: MLX server failed to respond on port 4000 within $((MLX_HEALTH_ATTEMPTS * 2))s"
    echo "  Check $MLX_LOG for details"
    exit 1
  fi
}

# Force a fresh MLX server start regardless of what's already running. Used
# by launchers like Narrative Gemma that need the server to pick up new env
# vars (MLX_APPEND_SYSTEM_PROMPT_FILE) which can only be applied at startup.
force_restart_mlx_server() {
  local desired="$1"
  local msg="$2"

  if lsof -i :4000 >/dev/null 2>&1; then
    echo "  Stopping existing MLX server so new env vars take effect..."
    _stop_mlx_server || echo "  Warning: existing MLX server didn't exit cleanly, continuing anyway"
  fi

  echo "$msg"
  MLX_MODEL="$desired" \
  MLX_KV_BITS="${MLX_KV_BITS:-}" \
  MLX_KV_QUANT_START="${MLX_KV_QUANT_START:-}" \
  "$MLX_PYTHON" "$MLX_SERVER" >"$MLX_LOG" 2>&1 &
  if ! _wait_for_mlx_health; then
    echo "  ERROR: MLX server failed to respond on port 4000 within $((MLX_HEALTH_ATTEMPTS * 2))s"
    echo "  Check $MLX_LOG for details"
    exit 1
  fi
}

# Resolve the Claude Code CLI. Launchers used to hardcode $HOME/.local/bin/claude,
# which fails outright for anyone who installed it via npm, Homebrew, or the
# official installer into a different prefix (issue #40). Check PATH first, then
# the usual install locations, and fail with something actionable instead of
# "No such file or directory".
require_claude_bin() {
  if [ -n "$CLAUDE_BIN" ] && [ -x "$CLAUDE_BIN" ]; then
    return 0
  fi
  local c
  for c in "$(command -v claude 2>/dev/null)" \
           "$HOME/.local/bin/claude" \
           "$HOME/.claude/local/claude" \
           "/opt/homebrew/bin/claude" \
           "/usr/local/bin/claude"; do
    if [ -n "$c" ] && [ -x "$c" ]; then
      CLAUDE_BIN="$c"
      return 0
    fi
  done
  echo ""
  echo "  ERROR: Claude Code CLI not found."
  echo ""
  echo "  Looked on your PATH and in:"
  echo "    \$HOME/.local/bin/claude"
  echo "    \$HOME/.claude/local/claude"
  echo "    /opt/homebrew/bin/claude"
  echo "    /usr/local/bin/claude"
  echo ""
  echo "  Install it with:"
  echo "    curl -fsSL https://claude.ai/install.sh | bash"
  echo ""
  echo "  Already installed somewhere else? Point this launcher at it:"
  echo "    CLAUDE_BIN=/path/to/claude \"\$0\""
  echo ""
  exit 1
}
