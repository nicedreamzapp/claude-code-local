#!/usr/bin/env python3
"""generate_response and the live stream, driven by a fake model.

Standalone (no MLX runtime, no model). The mlx.* modules are stubbed and
stream_generate is replaced by a fake that "generates" a scripted string,
so the request/response handling around the model can be checked here.
"""
import io
import json
import os
import sys
import types
from pathlib import Path

os.environ["MLX_MODEL"] = "stub-model"
os.environ.pop("MLX_APPEND_SYSTEM_PROMPT_FILE", None)
for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_lm.utils",
             "mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.models",
             "mlx_lm.models.cache"):
    sys.modules[name] = types.ModuleType(name)
for fn in ("get_cache_memory", "get_peak_memory", "get_active_memory"):
    setattr(sys.modules["mlx.core"], fn, lambda: 0)
sys.modules["mlx.core"].clear_cache = lambda: None
sys.modules["mlx.core"].reset_peak_memory = lambda: None
sys.modules["mlx_lm.utils"].load = lambda *a, **k: None
sys.modules["mlx_lm.generate"].stream_generate = lambda *a, **k: None
sys.modules["mlx_lm.sample_utils"].make_sampler = lambda *a, **k: None
sys.modules["mlx_lm.models.cache"].make_prompt_cache = lambda *a, **k: None
sys.modules["mlx_lm.models.cache"].RotatingKVCache = type("RotatingKVCache", (), {})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy"))
import server  # noqa: E402


class Tok:
    def apply_chat_template(self, messages, **kw):
        return [ord(c) % 256 for c in json.dumps(messages)]


class Cache:
    offset = 0

    def trim(self, n):
        self.offset -= n


class Resp:
    def __init__(self, text, n, finish):
        self.text, self.generation_tokens, self.finish_reason = text, n, finish


OUTPUTS = []


def fake_stream_generate(model, tokenizer, prompt, max_tokens, **kw):
    text = OUTPUTS.pop(0)
    kw["prompt_cache"][0].offset += len(prompt)
    pieces = [text[i:i + 3] for i in range(0, len(text), 3)]
    for i, piece in enumerate(pieces):
        kw["prompt_cache"][0].offset += 1
        yield Resp(piece, i + 1, "stop" if i == len(pieces) - 1 else None)


server.tokenizer = Tok()
server.model = object()
server.make_prompt_cache = lambda model: [Cache()]
server.make_sampler = lambda **k: None
server.stream_generate = fake_stream_generate


def run(body, output):
    server._prompt_cache = None
    server._cached_token_prefix = None
    OUTPUTS[:] = [output]
    return server.generate_response(body)


class FakeHandler:
    def __init__(self):
        self.wfile = io.BytesIO()

    def send_response(self, code):
        pass

    def send_header(self, k, v):
        pass

    def end_headers(self):
        pass


def run_live(body, output):
    server._prompt_cache = None
    server._cached_token_prefix = None
    OUTPUTS[:] = [output]
    h = FakeHandler()
    server.send_anthropic_stream_live(h, body)
    events = [json.loads(l[5:]) for l in h.wfile.getvalue().decode().splitlines()
              if l.startswith("data:")]
    text = "".join(e["delta"]["text"] for e in events
                   if e["type"] == "content_block_delta")
    delta = [e for e in events if e["type"] == "message_delta"][0]["delta"]
    return text, delta


def check(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"FAIL - {label}: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"ok - {label}")


MSG = [{"role": "user", "content": "hi"}]


def no_tools_no_tool_use():
    out = '{"name": "Ada", "arguments": {"age": 36}}'
    r = run({"messages": MSG, "max_tokens": 50}, out)
    assert r["stop_reason"] == "end_turn", r
    assert [b["type"] for b in r["content"]] == ["text"], r["content"]
    assert r["content"][0]["text"] == out, r["content"]


def no_tools_live_stop_reason():
    out = '{"name": "Ada", "arguments": {"age": 36}}'
    text, delta = run_live({"messages": MSG, "max_tokens": 50, "stream": True}, out)
    assert text == out, repr(text)
    assert delta["stop_reason"] == "end_turn", delta


def stop_sequence():
    r = run({"messages": MSG, "max_tokens": 50, "stop_sequences": ["END"]},
            "one two END three")
    assert r["content"] == [{"type": "text", "text": "one two"}], r["content"]
    assert r["stop_reason"] == "stop_sequence", r
    assert r["stop_sequence"] == "END", r


def stop_sequence_stops_generation():
    out = "a fairly long first part of the answer STOP" + " and more" * 50
    r = run({"messages": MSG, "max_tokens": 500, "stop_sequences": ["STOP"]}, out)
    assert r["content"][0]["text"] == "a fairly long first part of the answer", r["content"]
    assert r["stop_sequence"] == "STOP", r
    assert r["usage"]["output_tokens"] < 30, r["usage"]


def stop_sequence_live():
    text, delta = run_live({"messages": MSG, "max_tokens": 50, "stream": True,
                            "stop_sequences": ["END"]}, "one two END three")
    assert text == "one two", repr(text)
    assert delta == {"stop_reason": "stop_sequence", "stop_sequence": "END"}, delta


def tool_result_is_error():
    msgs = server.convert_messages({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
         "content": "No such file"}]}]})
    assert msgs[0]["role"] == "tool", msgs
    assert "error" in msgs[0]["content"].lower(), msgs[0]["content"]


check("request without tools never turns JSON text into tool_use", no_tools_no_tool_use)
check("live stream without tools never ends with stop_reason tool_use", no_tools_live_stop_reason)
check("stop_sequences ends the text and sets stop_reason", stop_sequence)
check("stop_sequences stops generating once matched", stop_sequence_stops_generation)
check("stop_sequences on the live stream", stop_sequence_live)
check("tool_result is_error reaches the model", tool_result_is_error)
print("all checks passed")
