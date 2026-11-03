#!/bin/bash
# Phase 2, run after run_all.sh:
#  - Ollama at 16K context (the MLX small-Mac tier's context) as a control,
#    so memory and speed can be compared at equal context
#  - the Gemma 4 31B 8-bit pair, 600 s per run on both sides
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPS="${1:-3}"
stop_ollama() { lsof -nP -t -iTCP:11434 -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null; while lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done; }

stop_ollama
python3 "$HERE/run_bench.py" --setup ollama-16k --model qwen35-2b --reps "$REPS"
python3 "$HERE/run_bench.py" --setup ollama-16k --model qwen35-4b --reps "$REPS"
stop_ollama
if [ "${WITH_GEMMA:-0}" = "1" ]; then
  python3 "$HERE/run_bench.py" --setup mlx --model gemma4-31b --reps "$REPS" --timeout 600
  python3 "$HERE/run_bench.py" --setup ollama --model gemma4-31b --reps "$REPS" --timeout 600
  stop_ollama
fi
python3 "$HERE/summarize.py"
