#!/usr/bin/env python3
"""Unit check for parse_tool_calls's Format 3.6 fenced-JSON branch.

Standalone (no MLX runtime, no server process) — stubs the mlx.* imports
so proxy/server.py can be imported for its pure parsing logic alone.
Covers the array-shaped fence regression (see issue #54): Hermes 4 14B
emits `[{"name": ..., "arguments": ...}]` instead of a bare object, and
the old startswith("{") guard skipped the whole fence.
"""
import sys
import types
from pathlib import Path

for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_lm.utils",
             "mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.models",
             "mlx_lm.models.cache"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["mlx_lm.utils"].load = lambda *a, **k: None
sys.modules["mlx_lm.generate"].stream_generate = lambda *a, **k: None
sys.modules["mlx_lm.sample_utils"].make_sampler = lambda *a, **k: None
sys.modules["mlx_lm.models.cache"].make_prompt_cache = lambda *a, **k: None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy"))
from server import parse_tool_calls  # noqa: E402

def check(label, text, expected):
    calls, _ = parse_tool_calls(text)
    got = [(c["name"], c["arguments"]) for c in calls]
    assert got == expected, f"{label}: expected {expected}, got {got}"
    print(f"ok - {label}")

check(
    "single object fence (existing behavior)",
    '```json\n{"name": "Bash", "arguments": {"command": "ls"}}\n```',
    [("Bash", {"command": "ls"})],
)

check(
    "Gemma 4 boolean argument next to string arguments",
    '<|tool_call>call:Edit{file_path:<|"|>/a<|"|>,old_string:<|"|>x<|"|>,'
    'new_string:<|"|>y<|"|>,replace_all:true}<tool_call|>',
    [("Edit", {"file_path": "/a", "old_string": "x", "new_string": "y",
               "replace_all": True})],
)

check(
    "Gemma 4 number arguments next to a string argument",
    '<|tool_call>call:Read{file_path:<|"|>/a<|"|>,offset:10,limit:50}<tool_call|>',
    [("Read", {"file_path": "/a", "offset": 10, "limit": 50})],
)

check(
    "Gemma 4 colon inside a string value is not read as a key",
    '<|tool_call>call:Bash{command:<|"|>echo a:b, c:d<|"|>,timeout:5000}<tool_call|>',
    [("Bash", {"command": "echo a:b, c:d", "timeout": 5000})],
)

check(
    "back-to-back objects, no array (existing behavior)",
    '```json\n{"name": "Bash", "arguments": {"command": "ls"}}\n'
    '{"name": "Read", "arguments": {"path": "/tmp/x"}}\n```',
    [("Bash", {"command": "ls"}), ("Read", {"path": "/tmp/x"})],
)

check(
    "array inside <tool_call> tags (raised AttributeError, request 500'd)",
    '<tool_call>[{"name": "Bash", "arguments": {"command": "ls"}}, '
    '{"name": "Read", "arguments": {"file_path": "/a"}}]</tool_call>',
    [("Bash", {"command": "ls"}), ("Read", {"file_path": "/a"})],
)

check(
    "array inside <|tool_call|> tags",
    '<|tool_call|>[{"name": "Bash", "arguments": {"command": "ls"}}]<|/tool_call|>',
    [("Bash", {"command": "ls"})],
)

check(
    "non-object JSON inside <tool_call> tags is skipped, not a crash",
    '<tool_call>"Bash"</tool_call>',
    [],
)

check(
    "array of objects (Hermes 4 14B shape, was silently dropped)",
    '```json\n[{"name": "Bash", "arguments": {"command": "ls"}}, '
    '{"name": "Read", "arguments": {"path": "/tmp/x"}}]\n```',
    [("Bash", {"command": "ls"}), ("Read", {"path": "/tmp/x"})],
)

check(
    "array using alternate field names (tool/params)",
    '```json\n[{"tool": "Grep", "parameters": {"pattern": "foo"}}]\n```',
    [("Grep", {"pattern": "foo"})],
)

check(
    "two <tool_call> calls to the same tool, different arguments",
    '<tool_call>{"name": "Read", "arguments": {"file_path": "/a"}}</tool_call>\n'
    '<tool_call>{"name": "Read", "arguments": {"file_path": "/b"}}</tool_call>',
    [("Read", {"file_path": "/a"}), ("Read", {"file_path": "/b"})],
)

check(
    "two Gemma 4 calls to the same tool, different arguments",
    '<|tool_call>call:Bash{command:<|"|>ls<|"|>}<tool_call|>'
    '<|tool_call>call:Bash{command:<|"|>pwd<|"|>}<tool_call|>',
    [("Bash", {"command": "ls"}), ("Bash", {"command": "pwd"})],
)

check(
    "two Llama 3.3 raw JSON calls to the same tool, different arguments",
    '{"type": "function", "name": "Grep", "parameters": {"pattern": "foo"}}\n'
    '{"type": "function", "name": "Grep", "parameters": {"pattern": "bar"}}',
    [("Grep", {"pattern": "foo"}), ("Grep", {"pattern": "bar"})],
)

check(
    "two <function=> calls to the same tool, different arguments",
    '<function=Read><parameter=file_path>/a</parameter></function>\n'
    '<function=Read><parameter=file_path>/b</parameter></function>',
    [("Read", {"file_path": "/a"}), ("Read", {"file_path": "/b"})],
)

check(
    "the same call emitted twice is still deduped",
    '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>\n'
    '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>',
    [("Bash", {"command": "ls"})],
)

check(
    "arguments sent as a JSON string (OpenAI style) are decoded",
    '<tool_call>{"name": "Bash", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>',
    [("Bash", {"command": "ls"})],
)

check(
    "Gemma 4 hyphenated keys (Grep -i, -n)",
    '<|tool_call>call:Grep{pattern:<|"|>foo<|"|>,-i:true,-n:true}<tool_call|>',
    [("Grep", {"pattern": "foo", "-i": True, "-n": True})],
)

check(
    "Gemma 4 nested array of objects (TodoWrite todos)",
    '<|tool_call>call:TodoWrite{todos:[{content:<|"|>a, b<|"|>,status:<|"|>pending<|"|>},'
    '{content:<|"|>c<|"|>,status:<|"|>done<|"|>}]}<tool_call|>',
    [("TodoWrite", {"todos": [{"content": "a, b", "status": "pending"},
                              {"content": "c", "status": "done"}]})],
)

check(
    "Gemma 4 call with only bare values keeps numbers as numbers",
    '<|tool_call>call:Bash{command:<|"|>sleep 1<|"|>}<tool_call|>'
    '<|tool_call>call:BashOutput{bash_id:7}<tool_call|>',
    [("Bash", {"command": "sleep 1"}), ("BashOutput", {"bash_id": 7})],
)

check(
    "Gemma 4 unquoted string value still falls back to a string",
    '<|tool_call>call:Bash{command:ls -la}<tool_call|>',
    [("Bash", {"command": "ls -la"})],
)

print("all checks passed")
