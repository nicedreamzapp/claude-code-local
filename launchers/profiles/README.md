# Launcher Profiles

Each launcher can load a profile file via `LAUNCHER_PROFILE=<name>`.

Profile files are in this directory as `<name>.env` and can override defaults
without editing launcher scripts.

## Built-in Profiles

- `standard` - Qwen default for general Claude Code use
- `fast` - Gemma + bare mode + minimal tools for lower latency
- `gemma` - Gemma-tuned standard launcher
- `llama` - Llama-tuned standard launcher
- `browser` - Browser-agent launcher defaults
- `narrative` - Narrative launcher defaults

## Example

```bash
LAUNCHER_PROFILE=fast /Users/shugo/claude-code-local/launchers/Claude\ Local\ Fast.command
```

Supported override keys:

- `LAUNCHER_MODEL_NAME_DEFAULT`
- `LAUNCHER_MLX_MODEL_DEFAULT`
- `LAUNCHER_MLX_KV_BITS_DEFAULT`
- `LAUNCHER_CLAUDE_PERMISSION_MODE_DEFAULT`
- `LAUNCHER_CLAUDE_BARE_DEFAULT` (fast launcher)
- `LAUNCHER_CLAUDE_TOOLS_DEFAULT` (fast launcher)
