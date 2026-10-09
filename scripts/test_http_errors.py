#!/usr/bin/env python3
"""Error responses from the HTTP handler, checked against a real socket.

Standalone (no MLX runtime, no model), same mlx.* stubs as
test_parse_tool_calls.py. generate_response is replaced so a request can
be made to fail on purpose. Anthropic's error body is
{"type": "error", "error": {"type": "api_error", "message": ...}}, with
invalid_request_error for a bad request body.
"""
import http.client
import json
import sys
import threading
import types
from http.server import HTTPServer
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
import server  # noqa: E402


def boom(body, on_start=None, on_text=None):
    raise RuntimeError("boom")


server.generate_response = boom
httpd = HTTPServer(("127.0.0.1", 0), server.AnthropicHandler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
PORT = httpd.server_address[1]


def post(raw):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    conn.request("POST", "/v1/messages", body=raw,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, resp.read().decode()


def check(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"FAIL - {label}: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"ok - {label}")


def generation_error():
    status, text = post(json.dumps({"model": "m", "max_tokens": 10, "tools": [{"name": "Bash"}],
                                    "messages": [{"role": "user", "content": "hi"}]}))
    assert status == 500, status
    body = json.loads(text)
    assert body.get("type") == "error", body
    assert body["error"]["type"] == "api_error", body
    assert body["error"]["message"] == "boom", body


def stream_error_event():
    status, text = post(json.dumps({"model": "m", "max_tokens": 10, "stream": True,
                                    "messages": [{"role": "user", "content": "hi"}]}))
    data = [json.loads(l[5:]) for l in text.splitlines() if l.startswith("data:")]
    errors = [d for d in data if d.get("type") == "error"]
    assert errors, text
    assert errors[0]["error"]["type"] == "api_error", errors[0]


def malformed_body():
    status, text = post("{not json")
    assert status == 400, status
    body = json.loads(text)
    assert body.get("type") == "error", body
    assert body["error"]["type"] == "invalid_request_error", body


def non_object_body():
    status, text = post("[]")
    assert status == 400, status
    assert json.loads(text)["error"]["type"] == "invalid_request_error", text


check("generation failure returns an Anthropic error body", generation_error)
check("live stream error event uses an Anthropic error type", stream_error_event)
check("malformed JSON body gets a 400, not a dropped connection", malformed_body)
check("JSON body that isn't an object gets a 400", non_object_body)
print("all checks passed")
