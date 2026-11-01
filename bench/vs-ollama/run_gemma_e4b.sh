#!/bin/bash
# Gemma 4 E4B round: Ollama first this time (the Qwen round started with MLX),
# then claude-code-local, then the Ollama temperature-0.2 control.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPS="${1:-3}"
stop_ollama() { pkill -f "/opt/homebrew/bin/ollama serve"; while lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done; }
stop_ollama
python3 "$HERE/run_bench.py" --setup ollama --model gemma4-e4b --reps "$REPS"
stop_ollama
python3 "$HERE/run_bench.py" --setup mlx --model gemma4-e4b --reps "$REPS"
cd /tmp && OLLAMA_CONTEXT_LENGTH=65536 nohup /opt/homebrew/bin/ollama serve >/tmp/ccl-bench/ollama-gemma-t02-serve.log 2>&1 </dev/null &
until curl -sf http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done
f=$(mktemp)
printf 'FROM gemma4:e4b-it-q8_0\nPARAMETER temperature 0.2\nPARAMETER top_k 0\nPARAMETER top_p 1.0\n' > "$f"
/opt/homebrew/bin/ollama create bench-gemma4:e4b-it-q8_0-t02 -f "$f"; rm -f "$f"
python3 "$HERE/run_bench.py" --setup ollama-t02 --model gemma4-e4b --reps "$REPS"
stop_ollama
python3 "$HERE/summarize.py"
