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

# Auto-load HQ credentials (HQ_URL, HQ_TOKEN, WooCommerce keys, etc.) so any
# launcher that sources this lib gives its in-session Bash calls — email
# briefing, reply-customer, invoices — working creds without a manual
# `source .env`. Absolute path, so it works regardless of the launcher's cwd.
_HQ_ENV="$HOME/Desktop/PROJECTS/ineedhemp website/.env"
[ -f "$_HQ_ENV" ] && set -a && . "$_HQ_ENV" && set +a

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

# Reserve room with forge_guard (:8790), start the server, and keep the seat
# alive for as long as the server lives.
#
# 2026-08-03: without this the server died the moment a real Claude Code
# request arrived. forge_guard owns memory policy on this box and SIGTERMs any
# big process that never asked for room; a ~18GB model server whose CPU reads
# ~0 during a GPU prefill is the textbook "idle hog" its evict-idle sweep
# exists to kill. Log line was `evict-idle unregistered pid=... <- SIGTERM
# (available 14.0GB)` five seconds after the first 10k-token prompt, and
# Claude Code then spun on ConnectionRefused with nothing left to reconnect
# to. Asking politely also means the launcher WAITS for room instead of
# racing a customer song into swap.
MEM_CLIENT="${MEM_CLIENT:-$HOME/SongForgeM5/mem_client.py}"
# 28, not 24. Measured 2026-08-04 with the fixed prefill chunk, weights 16.7GB:
# 18.4GB peak at an 8k prompt, 21.9GB at 38k, 24.0GB at 58k — call it +1GB per
# 15k tokens of context. 24 was under the truth even for a middling session, and
# an under-declared lease is exactly what the guard punishes: it logs an
# overrun, charges the real number, and if the box is tight by then the process
# gets SIGSTOPed and SIGTERMed mid-request. Ask honestly and it never has to.
MLX_LEASE_GB="${MLX_LEASE_GB:-28}"
MLX_LEASE_TIMEOUT="${MLX_LEASE_TIMEOUT:-600}"

_start_mlx_server() {
  local desired="$1"
  local msg="$2"
  local lease_id=""

  if [ -f "$MEM_CLIENT" ]; then
    echo "  Reserving ${MLX_LEASE_GB}GB with forge_guard..."
    lease_id="$(/usr/bin/python3 "$MEM_CLIENT" wait localclaude "$MLX_LEASE_GB" \
      --timeout "$MLX_LEASE_TIMEOUT" 2>/dev/null)"
    if [ -z "$lease_id" ]; then
      echo "  ⚠ forge_guard refused ${MLX_LEASE_GB}GB — starting anyway, but this"
      echo "    server may be evicted while Song Forge is busy. Check:"
      echo "    /usr/bin/python3 $MEM_CLIENT state"
    fi
  fi

  echo "$msg"
  MLX_MODEL="$desired" \
  MLX_KV_BITS="${MLX_KV_BITS:-}" \
  MLX_KV_QUANT_START="${MLX_KV_QUANT_START:-}" \
  "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  local server_pid=$!

  if [ -n "$lease_id" ]; then
    /usr/bin/python3 "$(dirname "${BASH_SOURCE[0]}")/hold_mem_lease.py" \
      "$lease_id" "$server_pid" >/dev/null 2>&1 &
  fi

  if ! _wait_for_mlx_health; then
    echo "  ERROR: MLX server failed to respond on port 4000 within 120s"
    echo "  Check /tmp/mlx-server.log for details"
    [ -n "$lease_id" ] && /usr/bin/python3 "$MEM_CLIENT" release "$lease_id" >/dev/null 2>&1
    exit 1
  fi
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

  _start_mlx_server "$desired" "$msg"
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

  _start_mlx_server "$desired" "$msg"
}
