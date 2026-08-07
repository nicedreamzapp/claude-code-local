#!/bin/bash
# Warm pool — keep the small models loaded simultaneously on separate ports so
# the router switches between them with ZERO load/unload.
#   Qwen 3 Coder  :4000   (default / code / agentic)
#   Gemma 4 31B   :4002   (quick / trivial)
# Gemma moved 4001 -> 4002 on 2026-08-07: the Free AI LiteLLM gateway owns
# :4001, so the pool's Gemma silently never started AND `stop` (also used by
# the router's DeepSeek-exclusive path) was kill -9ing the gateway.
# Total ~48 GB — fits in 128 GB with headroom. The 80 GB giants (GLM, DeepSeek)
# can't coexist with this pool, so they stay on-demand (router unloads the pool).
set -uo pipefail
L="$HOME/Desktop/PROJECTS/Local AI Setup/launchers/lib/claude-local-common.sh"
source "$L"
MLX_PY="$HOME/.local/mlx-server/bin/python3"
MLX_SRV="$HOME/.local/mlx-native-server/server.py"
HOLD="$HOME/Desktop/PROJECTS/Local AI Setup/launchers/lib/hold_mem_lease.py"

start_one() {  # port lease_name gb local_path repo
  local port="$1" lease_name="$2" gb="$3" local_path="$4" repo="$5"
  if lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1; then echo "  :$port already up"; return; fi
  local M; M="$(resolve_mlx_model "$local_path" "$repo")"
  # Ask forge_guard for the room BEFORE loading (2026-08-07): these servers
  # used to start unleased, so the guard read them as idle "unregistered"
  # hogs and could evict the pool under pressure. hold_mem_lease heartbeats
  # and releases the seat when the server dies.
  # honest=False + exact sizes on purpose: both pool leases must fit the 50GB
  # non-forge budget TOGETHER (30+18). The shared learned-peak hint would
  # inflate the asks past it and the guard would refuse the second server.
  local lease_id=""
  if [ -f "$MEM_CLIENT" ]; then
    lease_id="$(/usr/bin/python3 - "$lease_name" "$gb" <<'PY' 2>/dev/null
import contextlib, io, os, sys
sys.path.insert(0, os.path.expanduser("~/SongForgeM5"))
import mem_client
# mem_client.acquire prints "[mem] ... waited Ns" to stdout — swallow it, or
# it pollutes the captured lease id and the holder heartbeats a junk string.
with contextlib.redirect_stdout(io.StringIO()):
    lid = mem_client.acquire(sys.argv[1], float(sys.argv[2]), timeout=600, honest=False)
print(lid or "")
PY
)"
    [ -z "$lease_id" ] && echo "  ⚠ forge_guard refused ${gb}GB for :$port — starting anyway"
  fi
  echo "  starting $(basename "$M") on :$port"
  # CODE_MODE left at default (1): the FULL Claude Code prompt drowns a 30B
  # (tested 2026-06-14 — 6min timeouts). Lean mode keeps One AI usable for the
  # quick/code work it's actually good at.
  MLX_PORT="$port" MLX_MODEL="$M" nohup "$MLX_PY" "$MLX_SRV" >"/tmp/mlx-$port.log" 2>&1 &
  local pid=$!
  disown
  if [ -n "$lease_id" ]; then
    nohup /usr/bin/python3 "$HOLD" "$lease_id" "$pid" >/dev/null 2>&1 &
    disown
  fi
}

case "${1:-start}" in
  start)
    # NOTE: Qwen3-VL (vision) is NOT in the pool — it needs mlx-vlm, not this
    # mlx_lm text server (verified: "missing arg tie_word_embeddings"). Vision is
    # a separate setup. Warm pair = the two text models used daily.
    # DEFAULT coder = Qwen3-Coder-30B-A3B 8-bit (benchmarked best daily driver 2026-06-16).
    start_one 4000 localclaude-qwen 30 "$HOME/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit" "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit"
    start_one 4002 localclaude-gemma 18 "$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx" "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"
    echo "  warm pool starting (Qwen :4000 · Gemma :4002)"
    ;;
  stop)
    for p in 4000 4002; do lsof -ti ":$p" 2>/dev/null | xargs -r kill -9 2>/dev/null; done
    echo "  warm pool stopped"
    ;;
  status)
    for p in 4000 4002; do
      m=$(curl -s --max-time 2 "http://127.0.0.1:$p/health" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('model','?').split('/')[-1])" 2>/dev/null)
      echo "  :$p -> ${m:-down}"
    done
    ;;
esac
