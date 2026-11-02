#!/bin/bash
# Small-Mac tier: picks the model and the memory settings for an 8 GB or 16 GB
# Apple Silicon Mac. docs/SMALL-MACS.md has the measured numbers behind it.
#
#   bash scripts/mac-tier.sh            # print the tier this Mac gets
#   bash scripts/mac-tier.sh --env      # print it as export lines
#   source scripts/mac-tier.sh          # set the variables in this shell
#
# The tier comes from `sysctl hw.memsize`. Override it with CCL_TIER=8gb or
# CCL_TIER=16gb (or CCL_MEM_GB=<n> to pretend the Mac has n GB). Macs with
# 32 GB or more get CCL_TIER=none and keep the normal lineup in setup.sh.
#
# Both tier models were picked under the same rules: they must handle vision,
# code and tool calls (vision tower in config.json plus a tool-call chat
# template), bf16 weights where they fit and 8-bit only where bf16 does not
# (never 4-bit), and fit the memory a Mac has left after macOS and your apps,
# KV cache included.

ccl_tier_for_mem() {
  # Pure: GB in, tier name out. Kept separate so the tests can call it.
  local gb="$1"
  if [ -z "$gb" ] || ! [ "$gb" -eq "$gb" ] 2>/dev/null; then
    echo "none"; return
  fi
  if [ "$gb" -lt 16 ]; then echo "8gb"
  elif [ "$gb" -lt 32 ]; then echo "16gb"
  else echo "none"
  fi
}

ccl_detect_tier() {
  case "${CCL_TIER:-}" in
    8|8gb|8GB) CCL_TIER=8gb ;;
    16|16gb|16GB) CCL_TIER=16gb ;;
    none|off) CCL_TIER=none ;;
    "")
      local gb="${CCL_MEM_GB:-}"
      if [ -z "$gb" ]; then
        gb=$(/usr/sbin/sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}')
      fi
      CCL_TIER="$(ccl_tier_for_mem "$gb")"
      ;;
    *) echo "mac-tier: unknown CCL_TIER '$CCL_TIER' (use 8gb, 16gb or none)" >&2; CCL_TIER=none ;;
  esac

  case "$CCL_TIER" in
    16gb)
      # Qwen 3.5 4B, bf16: 9.1 GB of weights (vision tower included).
      CCL_MODEL_ID="mlx-community/Qwen3.5-4B-bf16"
      CCL_MODEL_DIR="$HOME/.cache/huggingface/hub/Qwen3.5-4B-bf16"
      CCL_MODEL_LABEL="Qwen 3.5 4B, bf16, vision + tools (~9 GB)"
      CCL_CONTEXT=16384
      CCL_MAX_OUTPUT=8192
      CCL_CACHE_LIMIT_GB=1
      ;;
    8gb)
      # Qwen 3.5 2B, 8-bit: 2.7 GB of weights (vision tower included). The
      # bf16 build was tried first: 4.6 GB just loaded and 5.6 to 6.0 GB at
      # peak, past the ~5 GB an 8 GB Mac has free, so this tier is 8-bit.
      CCL_MODEL_ID="mlx-community/Qwen3.5-2B-MLX-8bit"
      CCL_MODEL_DIR="$HOME/.cache/huggingface/hub/Qwen3.5-2B-MLX-8bit"
      CCL_MODEL_LABEL="Qwen 3.5 2B, 8-bit, vision + tools (~2.7 GB)"
      CCL_CONTEXT=16384
      CCL_MAX_OUTPUT=4096
      CCL_CACHE_LIMIT_GB=0.5
      ;;
    *)
      CCL_TIER=none
      CCL_MODEL_ID=""; CCL_MODEL_DIR=""; CCL_MODEL_LABEL=""
      CCL_CONTEXT=""; CCL_MAX_OUTPUT=""; CCL_CACHE_LIMIT_GB=""
      return 0
      ;;
  esac

  # What the server and Claude Code read. MLX_MAX_CONTEXT is the server's hard
  # ceiling: a longer prompt gets the API's "prompt is too long" 400, which makes
  # Claude Code compact, and each answer is trimmed so prompt + answer fits.
  export MLX_MODEL="${MLX_MODEL:-$CCL_MODEL_DIR}"
  export MLX_VISION="${MLX_VISION:-1}"
  export MLX_MAX_CONTEXT="${MLX_MAX_CONTEXT:-$CCL_CONTEXT}"
  export MLX_CACHE_LIMIT_GB="${MLX_CACHE_LIMIT_GB:-$CCL_CACHE_LIMIT_GB}"
  # Let these small models think before they answer. With thinking off the 4B
  # wrote a FizzBuzz with Fizz and Buzz swapped in 4 of 5 runs; with it on it
  # was right 11 of 11, for about twice the time per task (docs/SMALL-MACS.md).
  export MLX_SUPPRESS_THINKING="${MLX_SUPPRESS_THINKING:-0}"
  # Screenshots are shrunk to this longest side before the vision tower.
  export MLX_MAX_IMAGE_SIDE="${MLX_MAX_IMAGE_SIDE:-1024}"
  # Claude Code ignores an auto-compact window under 100000 (2.1.306 treats it as
  # 100k), so a 16K value here would claim a window the model doesn't have. Only
  # set it when the tier's context is at least that big; below it the server's
  # ceiling above does the job.
  if [ -z "${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-}" ] && [ -n "$CCL_CONTEXT" ] \
     && [ "$CCL_CONTEXT" -ge "$CCL_MIN_COMPACT_WINDOW" ]; then
    export CLAUDE_CODE_AUTO_COMPACT_WINDOW="$CCL_CONTEXT"
  fi
  export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-$CCL_MAX_OUTPUT}"
  # Keep any single tool result to a slice of the context: a whole-file Read or
  # a long log used to land in one turn and overflow it before Claude Code
  # could compact. Bigger files get read in pieces (Read's offset/limit).
  export CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS:-4000}"
  export BASH_MAX_OUTPUT_LENGTH="${BASH_MAX_OUTPUT_LENGTH:-12000}"
  export CCL_TIER CCL_MODEL_ID CCL_MODEL_DIR CCL_MODEL_LABEL CCL_CONTEXT CCL_MAX_OUTPUT CCL_CACHE_LIMIT_GB
}

# Smallest CLAUDE_CODE_AUTO_COMPACT_WINDOW Claude Code honors.
CCL_MIN_COMPACT_WINDOW=100000

ccl_detect_tier

# Run directly (not sourced): print the pick.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  if [ "${1:-}" = "--env" ]; then
    [ "$CCL_TIER" = "none" ] && { echo "CCL_TIER=none"; exit 0; }
    for v in CCL_TIER CCL_MODEL_ID MLX_MODEL MLX_VISION MLX_MAX_CONTEXT MLX_CACHE_LIMIT_GB MLX_SUPPRESS_THINKING MLX_MAX_IMAGE_SIDE \
             CLAUDE_CODE_AUTO_COMPACT_WINDOW CLAUDE_CODE_MAX_OUTPUT_TOKENS \
             CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS BASH_MAX_OUTPUT_LENGTH; do
      [ "$v" = CLAUDE_CODE_AUTO_COMPACT_WINDOW ] && [ -z "${!v}" ] && continue
      printf 'export %s=%q\n' "$v" "${!v}"
    done
  elif [ "$CCL_TIER" = "none" ]; then
    echo "This Mac is 32 GB or more: no small-Mac tier, setup.sh's normal lineup applies."
  else
    echo "Tier:     $CCL_TIER"
    echo "Model:    $CCL_MODEL_LABEL"
    echo "          $CCL_MODEL_ID"
    echo "Context:  $CCL_CONTEXT tokens (server ceiling; Claude Code compacts when it is hit)"
    echo "Download: bash scripts/download-and-import.sh small"
    echo "Launch:   double-click launchers/Small Mac.command"
  fi
fi
