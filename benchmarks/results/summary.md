# MLX inference sweep — Llama 70B on M5 Max

_Generated: 2026-04-26 11:00:06_


## Workload: `code`

| label | decode tok/s | prefill tok/s | TTFT (s) | peak GB | gen tok | finish |
|---|---:|---:|---:|---:|---:|---|
| 4bit+spec1B-k4 | 24.0 | 175 | 0.87 | 38.4 | 256 | max_tokens |
| 4bit+spec3B-k4 | 21.6 | 166 | 0.91 | 39.5 | 256 | max_tokens |
| 4bit+spec1B-k4+kv8 | 22.6 | 167 | 0.91 | 38.4 | 256 | max_tokens |

## Workload: `prose`

| label | decode tok/s | prefill tok/s | TTFT (s) | peak GB | gen tok | finish |
|---|---:|---:|---:|---:|---:|---|
| 4bit+spec1B-k4 | 13.8 | 129 | 0.71 | 38.4 | 256 | max_tokens |
| 4bit+spec3B-k4 | 14.4 | 132 | 0.70 | 39.5 | 256 | max_tokens |
| 4bit+spec1B-k4+kv8 | 12.4 | 137 | 0.67 | 38.4 | 256 | max_tokens |
