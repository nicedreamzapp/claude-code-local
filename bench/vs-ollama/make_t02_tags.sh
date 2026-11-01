#!/bin/bash
# Sampling control for Ollama: the same weights with claude-code-local's
# sampling (temperature 0.2; the MLX server applies no top-k, top-p or
# presence penalty, so those are switched off here).
set -e
for base in qwen3.5:2b-q8_0 qwen3.5:4b-bf16; do
  f=$(mktemp)
  printf 'FROM %s\nPARAMETER temperature 0.2\nPARAMETER top_k 0\nPARAMETER top_p 1.0\nPARAMETER presence_penalty 0\n' "$base" > "$f"
  /opt/homebrew/bin/ollama create "bench-$base-t02" -f "$f"
  rm -f "$f"
done
