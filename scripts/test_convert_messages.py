#!/usr/bin/env python3
"""Unit check for convert_messages's tool_result handling.

Standalone (no MLX runtime, no server process), same mlx.* stubs as
test_parse_tool_calls.py. Claude Code's Read tool returns an image as an
image block inside tool_result; its base64 data must not be pasted into
the prompt as text.
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
from server import convert_messages  # noqa: E402

DATA = "iVBORw0KGgo" + "A" * 200_000

body = {"messages": [{"role": "user", "content": [{
    "type": "tool_result",
    "tool_use_id": "toolu_1",
    "content": [
        {"type": "text", "text": "Read 1 image"},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png", "data": DATA}},
    ],
}]}]}

msgs = convert_messages(body)
assert len(msgs) == 1 and msgs[0]["role"] == "tool", msgs
content = msgs[0]["content"]
assert "Read 1 image" in content, content[:200]
assert DATA[:40] not in content, f"base64 image data leaked into prompt ({len(content)} chars)"
assert len(content) < 1000, f"tool result is {len(content)} chars"
print("ok - image block in tool_result is not pasted as base64 text")
print("all checks passed")
