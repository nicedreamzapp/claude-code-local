#!/bin/bash
# Download a model from the lineup directly into the MLX virtualenv.
#
# Usage:
#   bash scripts/download-and-import.sh                  # default Gemma 4 31B
#   bash scripts/download-and-import.sh qwen             # Qwen 3.5 122B
#   bash scripts/download-and-import.sh llama            # Llama 3.3 70B
#   MLX_MODEL=<hf-id> bash scripts/download-and-import.sh

set -e

MLX_PYTHON="${MLX_PYTHON:-$HOME/.local/mlx-server/bin/python3}"

if [ ! -x "$MLX_PYTHON" ]; then
  echo "ERROR: MLX virtualenv not found at $MLX_PYTHON"
  echo "Run setup.sh first."
  exit 1
fi

case "${1:-}" in
  qwen|qwen122|122b)
    MODEL="${MLX_MODEL:-mlx-community/Qwen3.5-122B-A10B-4bit}"
    LABEL="Qwen 3.5 122B (THE BEAST — 65 tok/s, ~75 GB RAM)"
    ;;
  llama|llama70|70b)
    # Our own abliterated MLX upload:
    #   https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx
    MODEL="${MLX_MODEL:-divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx}"
    LABEL="Llama 3.3 70B Abliterated (THE WISE ONE — ~7 tok/s, ~75 GB disk, divinetribe/8-bit MLX)"
    ;;
  gemma|gemma31|31b|"")
    # Our own abliterated MLX upload:
    #   https://huggingface.co/divinetribe/gemma-4-31b-it-abliterated-4bit-mlx
    MODEL="${MLX_MODEL:-divinetribe/gemma-4-31b-it-abliterated-4bit-mlx}"
    LABEL="Gemma 4 31B Abliterated (THE QUICK ONE — ~15 tok/s, ~18 GB RAM, divinetribe/4-bit MLX)"
    ;;
  *)
    MODEL="$1"
    LABEL="$1"
    ;;
esac

echo "=== Downloading $LABEL ==="
echo "    HuggingFace ID: $MODEL"
echo ""

"$MLX_PYTHON" - <<PY
from mlx_lm.utils import load
print("Downloading + loading $MODEL ...")
load("$MODEL")
print("Done.")
PY

echo ""
echo "=== DONE! Start the server with:"
echo "    MLX_MODEL=$MODEL bash scripts/start-mlx-server.sh"
