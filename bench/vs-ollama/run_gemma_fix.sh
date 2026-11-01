#!/bin/bash
# Re-run the claude-code-local side of the Gemma 4 E4B round on the fixed
# server (structured tool_calls), plus a Qwen 3.5 2B regression pass.
# Ollama's Gemma results from run_gemma_e4b.sh are reused.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
pkill -f "/opt/homebrew/bin/ollama serve"; while lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done
python3 "$HERE/run_bench.py" --setup mlx-fix --model gemma4-e4b --reps "${1:-3}"
python3 "$HERE/run_bench.py" --setup mlx-fix --model qwen35-2b --reps "${2:-1}"
python3 "$HERE/summarize.py"
