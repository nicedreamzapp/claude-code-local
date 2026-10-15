"""The Anthropic Messages API as served by proxy/server.py, over a real socket.

A real HTTPServer runs AnthropicHandler on a free localhost port. Generation is
either the scripted fake model from conftest or a stub generate_response.
"""
import http.client
import json
import threading
from http.server import HTTPServer

import pytest

import server

BASH = {"name": "Bash", "input_schema": {"type": "object", "properties": {
    "command": {"type": "string"}}, "required": ["command"]}}


@pytest.fixture(scope="module")
def port():
    httpd = HTTPServer(("127.0.0.1", 0), server.AnthropicHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, payload=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    headers = {"Content-Type": "application/json"} if data is not None else {}
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    out = resp.status, resp.getheader("Content-Type"), resp.read().decode()
    conn.close()
    return out


def sse_events(text):
    """[(event, data_dict), ...] from a text/event-stream body."""
    events = []
    for chunk in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def msg(**kw):
    b = {"model": "claude-sonnet-4-6", "max_tokens": 64,
         "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


# ─── GET / HEAD / unknown routes ─────────────────────────────────────────────

def test_models_list(port):
    status, ctype, body = request(port, "GET", "/v1/models")
    assert status == 200 and ctype == "application/json"
    ids = [m["id"] for m in json.loads(body)["data"]]
    assert "claude-sonnet-4-6" in ids


def test_health(port):
    status, _, body = request(port, "GET", "/health?verbose=1")
    data = json.loads(body)
    assert data["status"] == "ok"
    assert data["model"] == server.MODEL_PATH
    assert data["prefill_size"] == server.PREFILL_SIZE
    assert {"rss_gb", "mx_active_gb", "mx_cache_gb", "mx_peak_gb"} <= data.keys()


def test_head_is_ok(port):
    status, _, _ = request(port, "HEAD", "/")
    assert status == 200


def test_unknown_get_and_post_answer_empty_object(port):
    assert request(port, "GET", "/nope")[::2] == (200, "{}")
    assert request(port, "POST", "/nope", {"a": 1})[::2] == (200, "{}")


# ─── Error paths ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]", b'"str"'])
def test_bad_body_is_invalid_request_error(port, raw):
    status, _, body = request(port, "POST", "/v1/messages", raw=raw)
    assert status == 400
    err = json.loads(body)
    assert err["type"] == "error"
    assert err["error"]["type"] == "invalid_request_error"


def test_generation_failure_is_api_error(port, monkeypatch):
    def boom(body, on_start=None, on_text=None):
        raise RuntimeError("model fell over")
    monkeypatch.setattr(server, "generate_response", boom)
    status, _, body = request(port, "POST", "/v1/messages", msg())
    assert status == 500
    assert json.loads(body) == {"type": "error",
                                "error": {"type": "api_error", "message": "model fell over"}}


def test_live_stream_failure_sends_error_event(port, monkeypatch):
    def boom(body, on_start=None, on_text=None):
        raise RuntimeError("oom")
    monkeypatch.setattr(server, "generate_response", boom)
    status, ctype, body = request(port, "POST", "/v1/messages", msg(stream=True))
    assert status == 200 and ctype == "text/event-stream"
    assert sse_events(body)[-1] == ("error", {"type": "error",
                                              "error": {"type": "api_error", "message": "oom"}})


# ─── count_tokens ────────────────────────────────────────────────────────────

def test_count_tokens_uses_the_real_tokenizer_path(port, fake_model):
    b = msg(tools=[BASH])
    status, _, body = request(port, "POST", "/v1/messages/count_tokens", b)
    assert status == 200
    assert json.loads(body) == {"input_tokens": server.count_prompt_tokens(b)}


def test_count_tokens_falls_back_to_char_estimate(port, monkeypatch):
    def broken(body):
        raise RuntimeError("no tokenizer yet")
    monkeypatch.setattr(server, "count_prompt_tokens", broken)
    b = msg()
    status, _, body = request(port, "POST", "/messages/count_tokens", b)
    assert json.loads(body) == {"input_tokens": len(json.dumps(b)) // 4}


# ─── /v1/messages end to end with the fake model ─────────────────────────────

def test_non_streaming_text(port, fake_model):
    fake_model.outputs = ["Hello!"]
    status, ctype, body = request(port, "POST", "/v1/messages", msg())
    data = json.loads(body)
    assert status == 200 and ctype == "application/json"
    assert data["content"] == [{"type": "text", "text": "Hello!"}]


def test_streaming_tool_call_follows_anthropic_event_order(port, fake_model):
    fake_model.outputs = ['Listing.\n<tool_call>{"name": "Bash", "arguments": '
                          '{"command": "ls"}}</tool_call>']
    status, ctype, body = request(port, "POST", "/v1/messages", msg(stream=True, tools=[BASH]))
    assert status == 200 and ctype == "text/event-stream"
    events = sse_events(body)
    assert [e for e, _ in events] == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "message_delta", "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "Listing."}
    tool_start = events[4][1]["content_block"]
    assert tool_start["type"] == "tool_use" and tool_start["name"] == "Bash"
    assert [d["index"] for e, d in events[1:7]] == [0, 0, 0, 1, 1, 1]
    assert json.loads(events[5][1]["delta"]["partial_json"]) == {"command": "ls"}
    assert events[7][1]["delta"]["stop_reason"] == "tool_use"


def test_live_stream_without_tools(port, fake_model):
    fake_model.outputs = ["  <think>hidden</think>Visible answer.<|im_end|>leak"]
    status, _, body = request(port, "POST", "/v1/messages", msg(stream=True))
    events = sse_events(body)
    names = [e for e, _ in events]
    assert names[0] == "message_start" and names[1] == "content_block_start"
    assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    text = "".join(d["delta"]["text"] for e, d in events if e == "content_block_delta")
    assert text == "Visible answer."


@pytest.mark.parametrize("marker", ["<|im_end|>", "<turn|>", "<|eot_id|>"])
def test_live_stream_stop_marker_at_the_very_end_is_not_sent(port, fake_model, marker):
    # The common case: the model writes its end-of-turn token as text and stops.
    fake_model.outputs = ["Answer." + marker]
    status, _, body = request(port, "POST", "/v1/messages", msg(stream=True))
    text = "".join(d["delta"]["text"] for e, d in sse_events(body) if e == "content_block_delta")
    assert text == "Answer."


def test_live_stream_can_be_disabled(port, fake_model, monkeypatch):
    monkeypatch.setenv("MLX_LIVE_STREAM", "0")
    fake_model.outputs = ["Buffered."]
    status, _, body = request(port, "POST", "/v1/messages", msg(stream=True))
    events = sse_events(body)
    assert events[2] == ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "Buffered."}})
