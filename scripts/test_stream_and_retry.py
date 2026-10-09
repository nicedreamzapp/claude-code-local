#!/usr/bin/env python3
"""Live stream filtering and the tool-call retry path, driven by a fake model.

Standalone (no MLX runtime, no model). The mlx.* modules are stubbed and
stream_generate is replaced by a fake that "generates" scripted strings.
The fake prompt cache records every token it holds, the way mlx_lm's cache
is "updated in place", so a check can compare what the model would see
against the prompt the server built.
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
CLEARS = []
for fn in ("get_cache_memory", "get_peak_memory", "get_active_memory"):
    setattr(sys.modules["mlx.core"], fn, lambda: 0)
sys.modules["mlx.core"].clear_cache = lambda: CLEARS.append(1)
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
    def __init__(self):
        self.toks = []

    @property
    def offset(self):
        return len(self.toks)

    def trim(self, n):
        del self.toks[len(self.toks) - n:]


class Resp:
    def __init__(self, text, n, finish):
        self.text, self.generation_tokens, self.finish_reason = text, n, finish


OUTPUTS = []
SEEN = []  # what the model saw on each call: cached tokens + new prompt
CACHES = []  # the prompt cache object passed on each call


def fake_stream_generate(model, tokenizer, prompt, max_tokens, **kw):
    cache = kw["prompt_cache"][0]
    CACHES.append(kw["prompt_cache"])
    SEEN.append(cache.toks + list(prompt))
    cache.toks += list(prompt)
    text = OUTPUTS.pop(0)
    pieces = [text[i:i + 3] for i in range(0, len(text), 3)]
    for i, piece in enumerate(pieces):
        cache.toks.append(-1)
        yield Resp(piece, i + 1, "stop" if i == len(pieces) - 1 else None)


server.tokenizer = Tok()
server.model = object()
server.make_prompt_cache = lambda model: [Cache()]
server.make_sampler = lambda **k: None
server.stream_generate = fake_stream_generate


class FakeHandler:
    def __init__(self):
        self.wfile = io.BytesIO()

    def send_response(self, code):
        pass

    def send_header(self, k, v):
        pass

    def end_headers(self):
        pass


def fresh(outputs):
    server._prompt_cache = None
    server._cached_token_prefix = None
    OUTPUTS[:] = outputs
    SEEN.clear()
    CACHES.clear()
    CLEARS.clear()


def check(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"FAIL - {label}: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"ok - {label}")


MSG = [{"role": "user", "content": "hi"}]
TOOLS = [{"name": "Bash", "input_schema": {"type": "object",
                                           "properties": {"command": {"type": "string"}}}}]
BAD = 'Running it now. "name": "Bash" with command ls'
GOOD = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'


def live_text(output):
    fresh([output])
    h = FakeHandler()
    server.send_anthropic_stream_live(h, {"messages": MSG, "max_tokens": 50, "stream": True})
    events = [json.loads(l[5:]) for l in h.wfile.getvalue().decode().splitlines()
              if l.startswith("data:")]
    return "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")


def live_think_block():
    text = live_text("<think>the user said hi, keep it short</think>\n\nHello there.")
    assert text == "Hello there.", repr(text)


def live_unclosed_think_block():
    # A thinking model that hits max_tokens mid-thought never writes </think>.
    out = "<think>\nOkay, 17*23. 17*20 is 340, plus 17*3 is 51, so"
    text = live_text(out)
    assert text == out.strip(), repr(text)


def live_think_block_with_nothing_after():
    out = "<think>just thinking</think>"
    text = live_text(out)
    assert text == out, repr(text)


def retry_sees_only_retry_prompt():
    fresh([BAD, GOOD])
    r = server.generate_response({"messages": MSG, "max_tokens": 50, "tools": TOOLS})
    assert r["stop_reason"] == "tool_use", r
    assert len(SEEN) == 2, len(SEEN)
    assert -1 not in SEEN[1], "retry ran on top of the first attempt's cached prompt and output"


def retry_leaves_shared_cache_alone():
    fresh([BAD, GOOD])
    server.generate_response({"messages": MSG, "max_tokens": 50, "tools": TOOLS})
    first_cache, retry_cache = CACHES
    assert retry_cache is not first_cache, "retry should use its own cache"
    assert server._prompt_cache is first_cache, "shared cache was replaced by the retry"
    assert list(server._cached_token_prefix) == SEEN[0], "shared prefix no longer matches the first attempt"
    assert -1 not in first_cache[0].toks[:len(SEEN[0])], "shared cache prefix was changed"


def retry_releases_memory():
    fresh([BAD, GOOD])
    server.generate_response({"messages": MSG, "max_tokens": 50, "tools": TOOLS})
    assert len(CLEARS) == 2, f"clear_cache ran {len(CLEARS)} times for 2 generations"


def request_after_retry_reuses_cache_correctly():
    fresh([BAD, GOOD, "Done."])
    body = {"messages": MSG, "max_tokens": 50, "tools": TOOLS}
    server.generate_response(dict(body))
    msgs = MSG + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "thanks"}]
    server.generate_response(dict(body, messages=msgs))
    third = SEEN[2]
    full = server.tokenize_messages(server.convert_messages(
        server.optimize_for_code(dict(body, messages=msgs))),
        tools=server.convert_tools_for_llm(server.optimize_for_code(dict(body))["tools"]))
    assert third == list(full), "next request's cached + new tokens don't match its prompt"


check("live stream drops <think> blocks like the buffered path does", live_think_block)
check("live stream sends the text when a <think> block never closes", live_unclosed_think_block)
check("live stream sends the text when nothing follows the <think> block", live_think_block_with_nothing_after)
check("tool-call retry doesn't run on top of the stale prompt cache", retry_sees_only_retry_prompt)
check("tool-call retry leaves the shared cache as the first attempt left it", retry_leaves_shared_cache_alone)
check("tool-call retry releases MLX buffers like the first pass", retry_releases_memory)
check("request after a retry still reuses the cache correctly", request_after_retry_reuses_cache_correctly)
print("all checks passed")
