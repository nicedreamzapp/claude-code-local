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
  # 180 attempts × 2s = 6 minutes. Enough for a cold load of Llama 70B 8-bit
  # on a warm file cache; not enough for a first-time download from HF — use
  # resolve_mlx_model to point at a local path and avoid downloads entirely.
  local attempts="${1:-180}"
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

_stop_mlx_server() {
  pkill -f "mlx-native-server/server.py" 2>/dev/null || true
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
  "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  if ! _wait_for_mlx_health; then
    echo "  ERROR: MLX server failed to respond on port 4000 within 120s"
    echo "  Check /tmp/mlx-server.log for details"
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
  "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  if ! _wait_for_mlx_health; then
    echo "  ERROR: MLX server failed to respond on port 4000 within 120s"
    echo "  Check /tmp/mlx-server.log for details"
    exit 1
  fi
}
