"""Tool results must reach the model whatever the chat template does with them.

Stock Gemma 4 templates skip role=tool messages unless the assistant turn
before them carries structured tool_calls. Our default is text-form calls
(<tool_call>{json}</tool_call> in the assistant content), so on those
templates every tool result vanished from the prompt. template_messages()
now switches to structured tool_calls only for templates that drop the
text form, and leaves every other template exactly as before.

The fake tokenizers below reproduce each template's behaviour in plain
Python, so no model, jinja or network is needed.
"""
import pytest

import server

BODY = {
    "system": "SYS",
    "messages": [
        {"role": "user", "content": "fix broken.py"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Reading it."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/w/broken.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "RESULT_A"},
            {"type": "text", "text": "<system-reminder>note</system-reminder>"},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {"command": "python3 broken.py"}},
            {"type": "tool_use", "id": "toolu_3", "name": "Read", "input": {"file_path": "/w/x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "RESULT_B", "is_error": True},
            {"type": "tool_result", "tool_use_id": "toolu_3", "content": "RESULT_C"},
        ]},
    ],
}


class GemmaLike:
    """Like the stock Gemma 4 template: a tool message is rendered only by the
    assistant turn with tool_calls that precedes it; otherwise it is skipped."""
    chat_template = "{# gemma-like #} message['tool_calls'] ... role != 'tool'"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        out = []
        for i, m in enumerate(messages):
            if m["role"] == "tool":
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                part = "<model>" + "".join(f"call:{c['function']['name']}" for c in m["tool_calls"])
                ids = {c["id"]: c["function"]["name"] for c in m["tool_calls"]}
                for f in messages[i + 1:]:
                    if f["role"] != "tool":
                        break
                    part += f"<resp {ids.get(f.get('tool_call_id'), 'unknown')}>{f['content']}</resp>"
                out.append(part + m["content"])
            else:
                out.append(f"<{m['role']}>{m['content']}")
        return "".join(out) + ("<model>" if add_generation_prompt else "")


class QwenLike:
    """Like Qwen 3.5: supports tool_calls, but also renders every role=tool
    message on its own, so the text form already works."""
    chat_template = "{# qwen-like #} message.tool_calls ... <tool_response>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        out = []
        for m in messages:
            if m["role"] == "tool":
                out.append(f"<tool_response>{m['content']}</tool_response>")
            else:
                calls = "".join(f"<call {c['function']['name']}>" for c in m.get("tool_calls") or [])
                out.append(f"<{m['role']}>{m['content']}{calls}")
        return "".join(out)


class NoToolCallsTemplate(QwenLike):
    chat_template = "{# a template that never mentions structured calls #}"

    def apply_chat_template(self, messages, **kw):
        assert not any("tool_calls" in m for m in messages), "structured form sent to a template without it"
        return super().apply_chat_template(messages, **kw)


@pytest.fixture(autouse=True)
def fresh_cache():
    server._structured_cache.clear()
    yield
    server._structured_cache.clear()


def test_gemma_like_template_gets_structured_calls_and_sees_every_result():
    tok = GemmaLike()
    msgs = server.convert_messages(BODY, tool_ids=True)
    before = tok.apply_chat_template([server._plain(m) for m in msgs])
    assert "RESULT_A" not in before and "RESULT_C" not in before  # the bug
    assert server.template_needs_structured_tool_calls(tok) is True
    out = server.template_messages(msgs, tok)
    after = tok.apply_chat_template(out)
    for marker in ("RESULT_A", "RESULT_B", "RESULT_C"):
        assert marker in after
    # results are attributed to the right call, error prefix kept
    assert "<resp Read>RESULT_A" in after
    assert "<resp Bash>Error: RESULT_B" in after
    assert "<resp Read>RESULT_C" in after
    # the tool_result turn's own text comes after its results, so the
    # template's forward scan from the tool_calls turn is not interrupted
    assert after.index("RESULT_A") < after.index("<system-reminder>")
    # no private keys and no duplicated text-form call in the structured turn
    for m in out:
        assert not any(k.startswith("_ccl_") for k in m)
        if m.get("tool_calls"):
            assert "<tool_call>" not in m["content"]


def test_tool_call_ids_match_their_results():
    msgs = server.convert_messages(BODY, tool_ids=True)
    out = server._structured(msgs)
    calls = {c["id"]: c["function"]["name"] for m in out for c in m.get("tool_calls") or []}
    tools = [m for m in out if m["role"] == "tool"]
    assert calls == {"toolu_1": "Read", "toolu_2": "Bash", "toolu_3": "Read"}
    assert [(t["tool_call_id"], t["name"]) for t in tools] == [
        ("toolu_1", "Read"), ("toolu_2", "Bash"), ("toolu_3", "Read")]
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"file_path": "/w/broken.py"}


def test_qwen_like_template_keeps_the_text_form_unchanged():
    tok = QwenLike()
    msgs = server.convert_messages(BODY, tool_ids=True)
    assert server.template_needs_structured_tool_calls(tok) is False
    out = server.template_messages(msgs, tok)
    assert out == [server._plain(m) for m in msgs] == server.convert_messages(BODY)
    rendered = tok.apply_chat_template(out)
    assert "<tool_call>" in rendered and "RESULT_A" in rendered
    assert not any("tool_calls" in m for m in out)


def test_template_without_tool_calls_is_never_probed_or_switched():
    tok = NoToolCallsTemplate()
    msgs = server.convert_messages(BODY, tool_ids=True)
    assert server.template_needs_structured_tool_calls(tok) is False
    tok.apply_chat_template(server.template_messages(msgs, tok))


def test_plain_conversation_passes_through_untouched():
    body = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    msgs = server.convert_messages(body, tool_ids=True)
    assert server.template_messages(msgs, GemmaLike()) == msgs == server.convert_messages(body)


def test_orphan_tool_result_after_compaction_becomes_user_text():
    msgs = server.convert_messages(BODY, tool_ids=True)
    compacted = [m for m in msgs if m.get("role") != "assistant"]  # every call dropped
    out = server._structured(compacted)
    assert not any(m["role"] == "tool" for m in out)
    assert any(m["role"] == "user" and m["content"] == "RESULT_A" for m in out)


def test_probe_failure_keeps_the_default():
    class Broken(GemmaLike):
        def apply_chat_template(self, *a, **k):
            raise ValueError("boom")
    assert server.template_needs_structured_tool_calls(Broken()) is False


@pytest.mark.parametrize("text,expected", [
    ("The user wants X.\nI will read it.<channel|>Done.", "Done."),
    ("<|channel>thought\nhidden<channel|>Answer", "Answer"),
    ("plain answer", "plain answer"),
])
def test_lone_gemma_channel_close_drops_the_reasoning(text, expected):
    assert server.strip_think_tags(text) == expected


def test_default_convert_messages_output_is_unchanged():
    plain = server.convert_messages(BODY)
    assert not any(k.startswith("_ccl_") for m in plain for k in m)
    assert [m["role"] for m in plain] == ["system", "user", "assistant", "user", "tool", "assistant", "tool", "tool"]
