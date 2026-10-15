"""Shared setup for the pytest suite.

Nothing here needs Apple Silicon, a model, or the network. proxy/server.py
imports mlx and mlx_lm at module level, so those are replaced with stub
modules before it is imported; the tests then swap in a fake tokenizer and a
fake stream_generate wherever generation is involved. Same approach as the
standalone scripts/test_*.py checks.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Pin the env the server reads at import time, so a developer's shell
# (MLX_MODEL=gemma..., MLX_BROWSER_MODE=1, ...) can't change what is tested.
os.environ["MLX_MODEL"] = "stub-model"
for var in ("MLX_APPEND_SYSTEM_PROMPT_FILE", "MLX_BROWSER_MODE", "MLX_CODE_MODE",
            "MLX_DEBUG_REQUEST", "MLX_DEBUG_RESPONSE", "MLX_LIVE_STREAM",
            "MLX_TOOL_RETRIES", "MLX_SUPPRESS_THINKING"):
    os.environ.pop(var, None)

for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_lm.utils",
             "mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.models",
             "mlx_lm.models.cache"):
    sys.modules[name] = types.ModuleType(name)
_mx = sys.modules["mlx.core"]
for fn in ("get_cache_memory", "get_peak_memory", "get_active_memory"):
    setattr(_mx, fn, lambda: 0)
_mx.clear_cache = lambda: None
_mx.reset_peak_memory = lambda: None
_mx.set_cache_limit = lambda n: None
sys.modules["mlx_lm.utils"].load = lambda *a, **k: None
sys.modules["mlx_lm.generate"].stream_generate = lambda *a, **k: None
sys.modules["mlx_lm.sample_utils"].make_sampler = lambda *a, **k: None
sys.modules["mlx_lm.models.cache"].make_prompt_cache = lambda *a, **k: None
sys.modules["mlx_lm.models.cache"].RotatingKVCache = type("RotatingKVCache", (), {})

for sub in ("proxy", "agent", "bin"):
    sys.path.insert(0, str(ROOT / sub))

import server  # noqa: E402


def load_module(name, path):
    """Import a file that isn't on a package path (smart-router has a dash)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Fake model pieces ───────────────────────────────────────────────────────

class FakeTokenizer:
    """apply_chat_template that is deterministic and prefix-stable, like a real one."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return [ord(c) % 256 for c in json.dumps(messages)]

    def encode(self, text, **kw):
        return [ord(c) % 256 for c in text]


class FakeCache:
    offset = 0

    def trim(self, n):
        self.offset -= n


class FakeResp:
    def __init__(self, text, n, finish):
        self.text, self.generation_tokens, self.finish_reason = text, n, finish


class FakeModel:
    """Replaces stream_generate: each call pops the next scripted output and
    yields it in 3-char pieces, the way the real generator yields tokens."""

    def __init__(self):
        self.outputs = []
        self.prompts = []

    def stream_generate(self, model, tokenizer, prompt, max_tokens, **kw):
        self.prompts.append(list(prompt))
        text = self.outputs.pop(0)
        cache = kw["prompt_cache"]
        cache[0].offset += len(prompt)
        pieces = [text[i:i + 3] for i in range(0, len(text), 3)] or [""]
        for i, piece in enumerate(pieces):
            cache[0].offset += 1
            yield FakeResp(piece, i + 1, "stop" if i == len(pieces) - 1 else None)


@pytest.fixture
def fake_model(monkeypatch):
    fm = FakeModel()
    monkeypatch.setattr(server, "tokenizer", FakeTokenizer())
    monkeypatch.setattr(server, "model", object())
    monkeypatch.setattr(server, "make_prompt_cache", lambda model: [FakeCache()])
    monkeypatch.setattr(server, "make_sampler", lambda **k: None)
    monkeypatch.setattr(server, "stream_generate", fm.stream_generate)
    monkeypatch.setattr(server, "_prompt_cache", None)
    monkeypatch.setattr(server, "_cached_token_prefix", None)
    monkeypatch.setattr(server, "_first_request", False)
    return fm
