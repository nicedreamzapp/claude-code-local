#!/usr/bin/env python3
"""Anthropic Messages API in front of a llama.cpp server (Linux / Windows).

This is the portable sibling of proxy/server.py. The Mac build runs MLX in
process; off a Mac we have no MLX, so this process does the inference over HTTP
against a local `llama-server` (llama.cpp) instead — but it reuses proxy/server.py's
hardened Anthropic<->tool-call translation verbatim rather than re-implementing it.

Why reuse ours instead of llama-server's native tool support:
  - llama-server never speaks the Anthropic /v1/messages shape Claude Code needs
    (tool_use / tool_result blocks, the SSE event order, count_tokens), so a
    translator is required no matter what.
  - Our parser already handles the three <tool_call> spellings these models emit,
    recovers garbled XML/JSON tool calls, retries on tool-intent-without-a-call,
    drops the empty leading text block Claude Code 2.1 chokes on, and filters
    hallucinated input keys. That is strictly more than llama-server's per-template
    tool parsing gives us, and it is tuned on exactly the Gemma/Qwen models here.
So we drive llama-server as a plain text generator (no `tools` field; tools are
injected as text the same way the MLX path does) and let it do only what it does
better than us: run the GGUF and read images through its mmproj vision tower.

Config (env):
    LLAMA_SERVER_URL   where llama-server listens         (default http://127.0.0.1:8080)
    CCL_PORT           port THIS server binds              (default 4000, same as Mac)
    CCL_MODEL_NAME     label returned by /health           (default "local-gguf")
    CCL_MAX_TOKENS     default max_tokens                  (default 8192)
    CCL_REQUEST_TIMEOUT seconds to wait on llama-server    (default 600 — CPU is slow)
    CCL_TOOL_RETRIES   garbled-tool-call retries           (default 2)
Set CCL_BROWSER_MODE=1 / CCL_CODE_MODE=0 to match the Mac MLX_* switches.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Windows' legacy cp1252 console cannot encode the banner/log characters we
# print, which would crash the proxy on startup. Force UTF-8 on our streams.
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Reuse the hardened translation layer from the Mac server. Its MLX import is
# guarded, so importing it here (no MLX installed) is fine and gives us the real,
# battle-tested helpers with zero duplication.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proxy"))
import server as S  # noqa: E402

LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
PORT = int(os.environ.get("CCL_PORT") or 4000)
MODEL_NAME = os.environ.get("CCL_MODEL_NAME", "local-gguf")
DEFAULT_MAX_TOKENS = int(os.environ.get("CCL_MAX_TOKENS") or 8192)
REQUEST_TIMEOUT = float(os.environ.get("CCL_REQUEST_TIMEOUT") or 600)
MAX_TOOL_RETRIES = int(os.environ.get("CCL_TOOL_RETRIES") or 2)
BROWSER_MODE = os.environ.get("CCL_BROWSER_MODE", "0") == "1"
CODE_MODE_ENABLED = os.environ.get("CCL_CODE_MODE", "1") != "0"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ─── Anthropic messages → OpenAI chat messages (image-aware) ──────────────────

def _image_part(source):
    """Anthropic image source block → OpenAI image_url part, or None."""
    if not isinstance(source, dict):
        return None
    stype = source.get("type")
    if stype == "base64":
        media = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if stype == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def convert_messages_openai(body):
    """Anthropic Messages → OpenAI chat messages, preserving images.

    Mirrors proxy/server.py:convert_messages (same tool_use/tool_result handling)
    but keeps image blocks as OpenAI image_url parts so llama-server's mmproj can
    actually see them — the one capability the Mac text path never had.
    """
    messages = []

    if body.get("system"):
        sys_text = body["system"]
        if isinstance(sys_text, list):
            sys_text = "\n".join(b.get("text", "") for b in sys_text if b.get("type") == "text")
        messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        text_parts, image_parts, tool_use_parts, tool_result_parts = [], [], [], []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image":
                p = _image_part(block.get("source", {}))
                if p:
                    image_parts.append(p)
            elif btype == "tool_use":
                tool_use_parts.append(block)
            elif btype == "tool_result":
                tool_result_parts.append(block)

        # Assistant turn that called tools → text with <tool_call> blocks.
        if role == "assistant" and tool_use_parts:
            content_str = "\n".join(p for p in text_parts if p)
            for tu in tool_use_parts:
                call_json = json.dumps(
                    {"name": tu.get("name", ""), "arguments": tu.get("input", {})},
                    ensure_ascii=False)
                content_str += f"\n<tool_call>\n{call_json}\n</tool_call>"
            messages.append({"role": "assistant", "content": content_str.strip()})
            continue

        # Anything that carries a tool_result (and/or images) is a user turn.
        user_text_chunks = [p for p in text_parts if p]
        for tr in tool_result_parts:
            rc = tr.get("content", "")
            if isinstance(rc, list):
                sub_text, sub_imgs = [], []
                for b in rc:
                    if isinstance(b, dict) and b.get("type") == "image":
                        p = _image_part(b.get("source", {}))
                        if p:
                            sub_imgs.append(p)
                    elif isinstance(b, dict):
                        sub_text.append(b.get("text", ""))
                    else:
                        sub_text.append(str(b))
                image_parts.extend(sub_imgs)
                rc = "\n".join(t for t in sub_text if t)
            elif not isinstance(rc, str):
                rc = str(rc)
            if rc.strip():
                user_text_chunks.append(f"Tool result:\n{rc}" if tool_result_parts else rc)

        joined_text = "\n".join(user_text_chunks).strip()

        if image_parts:
            parts = []
            if joined_text:
                parts.append({"type": "text", "text": joined_text})
            parts.extend(image_parts)
            messages.append({"role": "user", "content": parts})
        elif joined_text:
            messages.append({"role": role if not tool_result_parts else "user",
                             "content": joined_text})

    return messages


def inject_tools_as_text(messages, llm_tools):
    """Append format_tools_as_text() to the system message (create one if absent).
    We never pass `tools` to llama-server; this is how the model learns them,
    exactly like the MLX path's system-prompt fallback."""
    if not llm_tools:
        return messages
    tool_text = S.format_tools_as_text(llm_tools)
    out = [dict(m) for m in messages]
    if out and out[0]["role"] == "system" and isinstance(out[0]["content"], str):
        out[0]["content"] = out[0]["content"] + "\n\n" + tool_text
    else:
        out.insert(0, {"role": "system", "content": tool_text})
    return out


# ─── llama.cpp HTTP ───────────────────────────────────────────────────────────

def _post_json(path, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(LLAMA_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def llama_chat(messages, max_tokens, temperature):
    """One non-streaming completion from llama-server. Returns (text, usage)."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "cache_prompt": True,
    }
    out = _post_json("/v1/chat/completions", payload, REQUEST_TIMEOUT)
    choice = (out.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = out.get("usage") or {}
    return text, usage


def count_tokens(body):
    """Best-effort token count via llama-server /tokenize, char estimate on failure."""
    try:
        flat = []
        if body.get("system"):
            s = body["system"]
            flat.append(s if isinstance(s, str)
                        else " ".join(b.get("text", "") for b in s if isinstance(b, dict)))
        for m in body.get("messages", []):
            c = m.get("content", "")
            if isinstance(c, str):
                flat.append(c)
            elif isinstance(c, list):
                flat.extend(b.get("text", "") for b in c
                            if isinstance(b, dict) and b.get("type") == "text")
        out = _post_json("/tokenize", {"content": "\n".join(flat)}, 30)
        toks = out.get("tokens")
        if isinstance(toks, list):
            return len(toks)
    except Exception as e:
        log(f"  count_tokens: /tokenize failed ({e}), char estimate")
    return len(json.dumps(body)) // 4


# ─── Generation (reuses the Mac server's parser + response builder) ───────────

def generate_response(body):
    body = S.hoist_system_messages(body)  # see server.hoist_system_messages
    if BROWSER_MODE:
        body = S.optimize_for_browser(body)
    elif CODE_MODE_ENABLED and S.looks_like_code_session(body):
        body = S.optimize_for_code(body)

    anthropic_tools = body.get("tools", [])
    llm_tools = S.convert_tools_for_llm(anthropic_tools) if anthropic_tools else None
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.2)

    base_messages = convert_messages_openai(body)
    messages = inject_tools_as_text(base_messages, llm_tools)

    if llm_tools:
        names = [t["function"]["name"] for t in llm_tools[:5]]
        log(f"  Tools: {len(llm_tools)} ({', '.join(names)}{'...' if len(llm_tools) > 5 else ''})")

    t0 = time.time()
    full_text, usage = llama_chat(messages, max_tokens, temperature)
    prompt_tokens = usage.get("prompt_tokens") or count_tokens(body)
    gen_tokens = usage.get("completion_tokens") or 0
    elapsed = time.time() - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generated: {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    text = S.clean_response(full_text)
    tool_calls, remaining_text = S.parse_tool_calls(text, llm_tools)

    # Same tool-intent retry as the MLX path, over HTTP instead of stream_generate.
    if (llm_tools and not tool_calls
            and any(p in remaining_text.lower() for p in S.TOOL_INTENT_PHRASES)):
        log(f"  Tool intent detected but parser missed it: {remaining_text[:200]!r}")
        for retry in range(MAX_TOOL_RETRIES):
            log(f"  Retry {retry+1}/{MAX_TOOL_RETRIES}: re-prompting for a clean tool call")
            retry_messages = messages + [
                {"role": "assistant", "content": full_text},
                {"role": "user", "content": (
                    "Your previous response tried to call a tool but the format was wrong. "
                    "Please call the tool now using EXACTLY this format:\n"
                    '<tool_call>\n{"name": "TOOL_NAME", "arguments": {"param": "value"}}\n</tool_call>\n'
                    'Do NOT use <parameter=...> tags inside tool_call. Use pure JSON with "arguments" key.'
                )},
            ]
            retry_text, retry_usage = llama_chat(retry_messages, max_tokens, temperature)
            gen_tokens += retry_usage.get("completion_tokens") or 0
            retry_calls, retry_remaining = S.parse_tool_calls(S.clean_response(retry_text), llm_tools)
            if retry_calls:
                tool_calls = retry_calls
                log(f"  Retry succeeded: {', '.join(tc['name'] for tc in retry_calls)}")
                break
            log(f"  Retry {retry+1} failed, still no valid tool call")

    return S.assemble_anthropic_message(
        remaining_text, tool_calls, body, anthropic_tools, prompt_tokens, gen_tokens)


# ─── HTTP handler (Anthropic shape, same routes as the Mac server) ────────────

def get_path(full_path):
    return urlparse(full_path).path


class AnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        path = get_path(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw)
        log(f"POST {self.path} model={body.get('model','-')} "
            f"max_tokens={body.get('max_tokens','-')} tools={len(body.get('tools', []))}")

        if path in ("/v1/messages", "/messages"):
            try:
                result = generate_response(body)
                first = result["content"][0]
                if first["type"] == "text":
                    log(f"  ← OK ({result['usage']['output_tokens']} tok) {first.get('text','')[:80]}...")
                else:
                    log(f"  ← OK ({result['usage']['output_tokens']} tok) [tool_use: {first['name']}]")
                if body.get("stream"):
                    S.send_anthropic_stream(self, result)
                else:
                    S.send_json(self, 200, result)
            except urllib.error.URLError as e:
                log(f"  ← ERROR reaching llama-server: {e}")
                S.send_json(self, 502, {"error": {"type": "upstream_error",
                            "message": f"llama-server at {LLAMA_URL} unreachable: {e}"}})
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                S.send_json(self, 500, {"error": {"type": "server_error", "message": str(e)}})
        elif path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            S.send_json(self, 200, {"input_tokens": count_tokens(body)})
        else:
            S.send_json(self, 200, {})

    def do_GET(self):
        path = get_path(self.path)
        if path in ("/v1/models", "/models"):
            now = int(time.time())
            S.send_json(self, 200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": now, "owned_by": "local"}
                for m in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001")
            ]})
        elif path == "/health":
            # Report our own readiness AND whether llama-server is answering.
            up = False
            try:
                with urllib.request.urlopen(LLAMA_URL + "/health", timeout=5) as r:
                    up = r.status == 200
            except Exception:
                up = False
            S.send_json(self, 200, {"status": "ok", "model": MODEL_NAME,
                                    "backend": "llama.cpp", "llama_server": LLAMA_URL,
                                    "llama_server_up": up})
        else:
            S.send_json(self, 200, {})


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Claude Code Local — portable (llama.cpp) server ║")
    print("║  Claude Code → Anthropic API → llama-server      ║")
    print("║  Tool use: our hardened parser · Vision: mmproj  ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"Backend llama-server: {LLAMA_URL}")
    print(f"Serving Anthropic Messages API on http://localhost:{PORT}")
    print("Claude Code config:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PORT}")
    print("  ANTHROPIC_API_KEY=sk-local")
    print()
    httpd = HTTPServer(("127.0.0.1", PORT), AnthropicHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()
