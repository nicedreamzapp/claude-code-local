# Tool-Call Reliability (v2 — March 2026)

Local models don't format tool calls perfectly. They *want* to call a tool but mix XML and JSON syntax. Claude Code sees no valid tool call, re-prompts, and the model does it again. The result: **infinite loops where the AI says "let me do that" but never actually does anything.**

We fixed this. Here's what was happening and what we did about it.

## The Problem

The model was generating garbled tool calls like this:
```
<tool_call>
<function=Bash><parameter=command>rm -rf /tmp/old</parameter></function>
</tool_call>
```

Instead of the correct JSON format Claude Code expects:
```json
<tool_call>
{"name": "Bash", "arguments": {"command": "rm -rf /tmp/old"}}
</tool_call>
```

The JSON parser choked, Claude Code saw no tool call, re-prompted the model, and the model garbled it the exact same way again — creating an infinite loop.

## The Fix (4 changes to `server.py`)

| Change | What | Why |
|--------|------|-----|
| **KV Cache** | 4-bit → 8-bit, quantization starts at token 1024 | Model retains conversation context instead of "forgetting" earlier messages |
| **Temperature** | 0.7 → 0.2 | Less randomness = more consistent tool formatting |
| **Garbled Recovery** | New `recover_garbled_tool_json()` function | Catches XML-in-JSON hybrids, `<function=X><parameter=Y>` inside `<tool_call>` tags, and infers tool names from parameter keys |
| **Retry Logic** | Up to 2 retries when tool intent is detected but parsing fails | Re-prompts with explicit formatting instructions before giving up |

## Test Results

We built an automated test suite (`scripts/test_mlx_server.py`) that sends real Anthropic API requests to the server simulating multi-step tasks — the exact kind that were failing before.

```
Test Suite: 14 tests per run
─────────────────────────────
  ✅ Simple Bash commands
  ✅ Directory creation (mkdir -p)
  ✅ File reading (Read tool)
  ✅ Complex Bash with pipes
  ✅ File editing (Edit tool with find/replace)
  ✅ Multi-tool sequences (Glob → Read)
  ✅ 5 rapid-fire sequential commands
  ✅ Multi-step calendar scenario (create → delete → verify)
```

**Results: 98/98 tests passed across 7 consecutive runs. Zero failures.**

The multi-step calendar scenario — create 12 month folders, delete all but September, verify — was the exact task that triggered infinite loops before the fix. Now it passes every time.

```bash
# Run the test suite yourself:
python3 scripts/test_mlx_server.py
```
