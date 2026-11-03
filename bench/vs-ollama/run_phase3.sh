#!/bin/bash
# Phase 3: Ollama with claude-code-local's sampling settings (temperature 0.2),
# Ollama's documented 64K context. Separates "server" from "sampling defaults".
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPS="${1:-3}"
stop_ollama() { lsof -nP -t -iTCP:11434 -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null; while lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done; }
stop_ollama
cd /tmp && OLLAMA_CONTEXT_LENGTH=65536 nohup /opt/homebrew/bin/ollama serve >/tmp/ccl-bench/ollama-t02-serve.log 2>&1 </dev/null &
until curl -sf http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done
bash "$HERE/make_t02_tags.sh"
python3 "$HERE/run_bench.py" --setup ollama-t02 --model qwen35-2b --reps "$REPS"
python3 "$HERE/run_bench.py" --setup ollama-t02 --model qwen35-4b --reps "$REPS"
stop_ollama
python3 "$HERE/summarize.py"
