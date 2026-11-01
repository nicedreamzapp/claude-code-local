#!/bin/bash
# Full run, ABBA order so neither side always goes first. One server at a time.
# Usage: bash bench/vs-ollama/run_all.sh [reps]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPS="${1:-3}"
stop_ollama() { pkill -f "/opt/homebrew/bin/ollama serve"; while lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done; }
run() { python3 "$HERE/run_bench.py" --setup "$1" --model "$2" --reps "$REPS"; }

stop_ollama; run mlx qwen35-2b
run mlx-raw qwen35-2b
run ollama qwen35-2b
run ollama qwen35-4b
stop_ollama; run mlx qwen35-4b
run mlx-raw qwen35-4b
if [ "${WITH_GEMMA:-0}" = "1" ]; then
  stop_ollama; run mlx gemma4-31b
  run ollama gemma4-31b
fi
stop_ollama
python3 "$HERE/summarize.py"
