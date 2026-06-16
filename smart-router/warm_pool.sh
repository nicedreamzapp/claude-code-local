#!/bin/bash
# Warm pool — keep the small models loaded simultaneously on separate ports so
# the router switches between them with ZERO load/unload.
#   Qwen 3 Coder  :4000   (default / code / agentic)
#   Gemma 4 31B   :4001   (quick / trivial)
#   Qwen3-VL 32B  :4002   (vision)
# Total ~64 GB — fits in 128 GB with headroom. The 80 GB giants (GLM, DeepSeek)
# can't coexist with this pool, so they stay on-demand (router unloads the pool).
set -uo pipefail
L="$HOME/Desktop/PROJECTS/Local AI Setup/launchers/lib/claude-local-common.sh"
source "$L"
MLX_PY="$HOME/.local/mlx-server/bin/python3"
MLX_SRV="$HOME/.local/mlx-native-server/server.py"

start_one() {  # port local_path repo
  local port="$1" local_path="$2" repo="$3"
  if lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1; then echo "  :$port already up"; return; fi
  local M; M="$(resolve_mlx_model "$local_path" "$repo")"
  echo "  starting $(basename "$M") on :$port"
  # CODE_MODE left at default (1): the FULL Claude Code prompt drowns a 30B
  # (tested 2026-06-14 — 6min timeouts). Lean mode keeps One AI usable for the
  # quick/code work it's actually good at.
  MLX_PORT="$port" MLX_MODEL="$M" nohup "$MLX_PY" "$MLX_SRV" >"/tmp/mlx-$port.log" 2>&1 &
  disown
}

case "${1:-start}" in
  start)
    # NOTE: Qwen3-VL (vision) is NOT in the pool — it needs mlx-vlm, not this
    # mlx_lm text server (verified: "missing arg tie_word_embeddings"). Vision is
    # a separate setup. Warm pair = the two text models used daily.
    # DEFAULT coder = Qwen3-Coder-30B-A3B 8-bit (benchmarked best daily driver 2026-06-16).
    # New 80B reasoning model is MLX_MODELS["qwen-new"], loaded on-demand for hard tasks.
    start_one 4000 "$HOME/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit" "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit"
    start_one 4001 "$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx" "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"
    echo "  warm pool starting (Qwen :4000 · Gemma :4001)"
    ;;
  stop)
    for p in 4000 4001; do lsof -ti ":$p" 2>/dev/null | xargs -r kill -9 2>/dev/null; done
    echo "  warm pool stopped"
    ;;
  status)
    for p in 4000 4001; do
      m=$(curl -s --max-time 2 "http://127.0.0.1:$p/health" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('model','?').split('/')[-1])" 2>/dev/null)
      echo "  :$p -> ${m:-down}"
    done
    ;;
esac
