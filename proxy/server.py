#!/usr/bin/env python3
"""
MLX Native Anthropic Server — Claude Code on Apple Silicon.
Single-file server: MLX inference + Anthropic Messages API + KV cache quantization.
No proxy. No translation layer. Direct.
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache, KVCache

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.5-122B-A10B-4bit")
PORT = int(os.environ.get("MLX_PORT", "4000"))
KV_BITS = int(os.environ.get("MLX_KV_BITS", "4"))
PREFILL_SIZE = int(os.environ.get("MLX_PREFILL_SIZE", "4096"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "8192"))

# ─── Globals ─────────────────────────────────────────────────────────────────

model = None
tokenizer = None
generate_lock = threading.Lock()

# Prompt cache: stores (token_hash, cache_state) for reuse
_prompt_cache = {}
_prompt_cache_lock = threading.Lock()


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model():
    global model, tokenizer
    log(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    elapsed = time.time() - t0
    log(f"Model loaded in {elapsed:.1f}s")
    log(f"KV cache quantization: {KV_BITS}-bit" if KV_BITS else "KV cache: full precision")


# ─── Think Tag Stripping ────────────────────────────────────────────────────

def strip_think_tags(text):
    """Remove <think>...</think> blocks from Qwen's reasoning output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


def clean_response(text):
    """Strip think tags and clean reasoning artifacts."""
    text = strip_think_tags(text)

    # Remove reasoning preamble if present
    if text.lstrip().startswith("Thinking"):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            s = line.strip()
            if any(s.startswith(p) for p in ['```', 'def ', 'class ', 'function ', 'import ', '#', '//']):
                return '\n'.join(lines[i:])

    return text


# ─── Anthropic Message Conversion ───────────────────────────────────────────

def convert_messages(body):
    """Convert Anthropic Messages format to flat chat messages for tokenizer."""
    messages = []

    # System prompt
    if body.get("system"):
        sys_text = body["system"]
        if isinstance(sys_text, list):
            sys_text = "\n".join(b.get("text", "") for b in sys_text if b.get("type") == "text")
        messages.append({"role": "system", "content": sys_text})

    # Conversation messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tc = "\n".join(b.get("text", str(b)) for b in tc)
                    parts.append(str(tc))
                elif block.get("type") == "tool_use":
                    parts.append(f"[Tool call: {block.get('name', '')}({json.dumps(block.get('input', {}))})]")
            content = "\n".join(p for p in parts if p)
        messages.append({"role": role, "content": content})

    return messages


def tokenize_messages(messages):
    """Apply chat template and tokenize."""
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    except Exception:
        # Fallback: concatenate as plain text
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text += "\nassistant: "
        token_ids = tokenizer.encode(text)

    return token_ids


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_response(body):
    """Run MLX inference and return Anthropic-formatted response."""
    messages = convert_messages(body)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.7)

    # Tokenize
    token_ids = tokenize_messages(messages)
    prompt_tokens = len(token_ids)
    log(f"  Prompt: {prompt_tokens} tokens")

    # Build generation kwargs
    gen_kwargs = {
        "prefill_step_size": PREFILL_SIZE,
    }
    if KV_BITS:
        gen_kwargs["kv_bits"] = KV_BITS
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = 0

    if temperature > 0:
        gen_kwargs["sampler"] = make_sampler(temp=temperature)
    else:
        gen_kwargs["sampler"] = make_sampler(temp=0.0)

    # Generate
    full_text = ""
    gen_tokens = 0
    finish_reason = "end_turn"
    t0 = time.time()

    with generate_lock:
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=token_ids,
            max_tokens=max_tokens,
            **gen_kwargs,
        ):
            full_text += response.text
            gen_tokens = response.generation_tokens
            if response.finish_reason == "length":
                finish_reason = "max_tokens"
            elif response.finish_reason == "stop":
                finish_reason = "end_turn"

    elapsed = time.time() - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generated: {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    # Clean output
    text = clean_response(full_text)

    # Build content blocks
    content_blocks = []
    if text.strip():
        content_blocks.append({"type": "text", "text": text.strip()})
    else:
        content_blocks.append({"type": "text", "text": "(No output)"})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "claude-sonnet-4-6"),
        "content": content_blocks,
        "stop_reason": finish_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": gen_tokens,
        }
    }


# ─── HTTP Handler ────────────────────────────────────────────────────────────

def send_json(handler, status, data):
    resp = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", len(resp))
    handler.end_headers()
    handler.wfile.write(resp)


def get_path(full_path):
    return urlparse(full_path).path


class AnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        log(f"HEAD {self.path}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        path = get_path(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(content_length) if content_length else b'{}'
        body = json.loads(raw)
        log(f"POST {self.path} model={body.get('model','-')} max_tokens={body.get('max_tokens','-')}")

        if path in ("/v1/messages", "/messages"):
            try:
                result = generate_response(body)
                preview = result["content"][0].get("text", "")[:80]
                log(f"  ← OK ({result['usage']['output_tokens']} tok) {preview}...")
                send_json(self, 200, result)
            except Exception as e:
                log(f"  ← ERROR: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                send_json(self, 500, {"error": {"type": "server_error", "message": str(e)}})
        else:
            log(f"  Unknown POST: {path}")
            send_json(self, 200, {})

    def do_GET(self):
        path = get_path(self.path)
        log(f"GET {self.path}")

        if path in ("/v1/models", "/models"):
            send_json(self, 200, {
                "object": "list",
                "data": [
                    {"id": "claude-opus-4-6", "object": "model", "created": int(time.time()), "owned_by": "local"},
                    {"id": "claude-sonnet-4-6", "object": "model", "created": int(time.time()), "owned_by": "local"},
                    {"id": "claude-haiku-4-5-20251001", "object": "model", "created": int(time.time()), "owned_by": "local"},
                ]
            })
        elif path == "/health":
            send_json(self, 200, {"status": "ok", "model": MODEL_PATH})
        else:
            send_json(self, 200, {})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  MLX Native Anthropic Server                    ║")
    print("║  Claude Code → MLX → Apple Silicon (direct)     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    load_model()

    print()
    print(f"Serving Anthropic Messages API on http://localhost:{PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"KV cache: {KV_BITS}-bit quantization" if KV_BITS else "KV cache: full precision")
    print()
    print("Claude Code config:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PORT}")
    print(f"  ANTHROPIC_API_KEY=sk-local")
    print()

    server = HTTPServer(("127.0.0.1", PORT), AnthropicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
