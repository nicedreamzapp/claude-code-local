#!/bin/bash
# Claude Code — Local AI (runs on your Mac, no cloud)
# Double-click to launch
# MLX Native Server — direct Anthropic API, no proxy needed

CLAUDE_BIN="/Users/dtribe/.local/bin/claude"
MLX_SERVER="/Users/dtribe/.local/mlx-native-server/server.py"
MLX_PYTHON="/Users/dtribe/.local/mlx-server/bin/python3"

# Start MLX server if not running
if ! lsof -i :4000 >/dev/null 2>&1; then
  "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  echo "  Loading Qwen 3.5 122B on MLX (48 tok/s)..."
  while ! curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; do
    sleep 2
  done
fi

clear
echo ""
echo "  → Claude Code with LOCAL AI (Qwen 3.5 122B)"
echo "  → MLX Native: 48 tok/s, 4-bit KV cache"
echo "  → Running on your M5 Max — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 --permission-mode auto
