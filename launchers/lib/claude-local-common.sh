#!/bin/bash

STARTED_MLX_SERVER=0
MLX_SERVER_PID=""

require_file() {
  local path="$1"
  local label="$2"

  if [ ! -f "$path" ]; then
    echo "  ERROR: $label not found at $path"
    exit 1
  fi
}

build_user_mcp_config() {
  local output_path
  output_path="$(mktemp -t claude-local-mcp).json"

  python3 - "$output_path" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
settings_paths = [
    pathlib.Path.home() / ".claude" / "settings.json",
    pathlib.Path.home() / ".claude" / "settings.local.json",
]

merged = {}
for path in settings_paths:
    if not path.exists():
        continue

    try:
        data = json.loads(path.read_text())
    except Exception:
        continue

    servers = data.get("mcpServers")
    if isinstance(servers, dict):
        merged.update(servers)

output_path.write_text(json.dumps({"mcpServers": merged}, indent=2))
PY

  echo "$output_path"
}

get_running_mlx_model() {
  curl -sf http://127.0.0.1:4000/health 2>/dev/null | python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(1)

model = payload.get("model", "")
if model:
    print(model)
'
}

get_running_mlx_kv_bits() {
  curl -sf http://127.0.0.1:4000/health 2>/dev/null | python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(1)

kv_bits = payload.get("kv_bits")
if kv_bits is not None:
    print(kv_bits)
'
}

wait_for_mlx_server() {
  local attempts="${1:-60}"
  local delay="${2:-2}"
  local i

  for i in $(seq 1 "$attempts"); do
    if curl -sf http://127.0.0.1:4000/health >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done

  echo "  ERROR: Timed out waiting for MLX server on port 4000"
  return 1
}

wait_for_mlx_server_shutdown() {
  local attempts="${1:-20}"
  local delay="${2:-1}"
  local i

  for i in $(seq 1 "$attempts"); do
    if ! lsof -i :4000 >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done

  echo "  WARNING: MLX server is still listening on port 4000"
  return 1
}

ensure_mlx_server() {
  local desired_model="$1"
  local loading_message="$2"
  shift 2
  local -a env_args=("$@")
  local running_model=""
  local running_kv_bits=""
  local desired_kv_bits="${MLX_KV_BITS:-0}"

  if lsof -i :4000 >/dev/null 2>&1; then
    running_model="$(get_running_mlx_model || true)"
    running_kv_bits="$(get_running_mlx_kv_bits || true)"
    if [ -n "$desired_model" ] && [ "$running_model" != "$desired_model" ]; then
      echo "  Restarting MLX server for requested model..."
      pkill -f "mlx-native-server/server.py" 2>/dev/null || true
      wait_for_mlx_server_shutdown || true
    elif [ -z "$running_kv_bits" ] || [ "$running_kv_bits" != "$desired_kv_bits" ]; then
      echo "  Restarting MLX server to apply KV cache setting (MLX_KV_BITS=$desired_kv_bits)..."
      pkill -f "mlx-native-server/server.py" 2>/dev/null || true
      wait_for_mlx_server_shutdown || true
    fi
  fi

  if ! lsof -i :4000 >/dev/null 2>&1; then
    echo "$loading_message"
    if [ ${#env_args[@]} -gt 0 ]; then
      env "${env_args[@]}" MLX_KV_BITS="$desired_kv_bits" MLX_MODEL="$desired_model" "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
    else
      MLX_KV_BITS="$desired_kv_bits" MLX_MODEL="$desired_model" "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
    fi
    MLX_SERVER_PID=$!
    STARTED_MLX_SERVER=1
    wait_for_mlx_server
  fi
}

cleanup_mlx_server() {
  if [ "${STARTED_MLX_SERVER:-0}" -eq 1 ] && [ -n "${MLX_SERVER_PID:-}" ]; then
    if kill -0 "$MLX_SERVER_PID" >/dev/null 2>&1; then
      kill "$MLX_SERVER_PID" >/dev/null 2>&1 || true
      wait "$MLX_SERVER_PID" 2>/dev/null || true
    fi
  fi
}
