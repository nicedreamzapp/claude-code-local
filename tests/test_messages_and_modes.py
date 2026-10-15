"""Anthropic -> chat-template conversion, tool schema handling, code/browser mode,
and tokenizer fallbacks in proxy/server.py."""
import copy
import json

import server
from server import (
    convert_messages,
    convert_tools_for_llm,
    format_tools_as_text,
    looks_like_claude_code_browser_session,
    looks_like_code_session,
    optimize_for_browser,
    optimize_for_code,
    slim_tool,
    tokenize_messages,
)

BASH = {
    "name": "Bash",
    "description": "Run a command. " * 50,
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "the command"},
            "timeout": {"type": "number", "description": "ms"},
        },
        "required": ["command"],
    },
}
WEBFETCH = {"name": "WebFetch", "description": "fetch", "input_schema": {"type": "object"}}


# ─── convert_messages ────────────────────────────────────────────────────────

def test_system_string_and_text_messages():
    msgs = convert_messages({"system": "be brief", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]})
    assert msgs == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_system_as_block_list_keeps_only_text_blocks():
    msgs = convert_messages({"system": [
        {"type": "text", "text": "one"},
        {"type": "image", "source": {}},
        {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}},
    ], "messages": []})
    assert msgs == [{"role": "system", "content": "one\ntwo"}]


def test_assistant_tool_use_becomes_tool_call_json():
    msgs = convert_messages({"messages": [{"role": "assistant", "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls é"}},
    ]}]})
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content.startswith("Checking.\n<tool_call>\n")
    call = json.loads(content.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
    assert call == {"name": "Bash", "arguments": {"command": "ls é"}}
    assert "ls é" in content  # ensure_ascii=False: no \u escapes for the model to copy


def test_tool_results_become_tool_messages_after_user_text():
    msgs = convert_messages({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "out1"},
        {"type": "tool_result", "tool_use_id": "t2",
         "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"type": "tool_result", "tool_use_id": "t3", "content": "nope", "is_error": True},
        {"type": "tool_result", "tool_use_id": "t4", "content": 42},
        {"type": "text", "text": "keep going"},
    ]}]})
    assert msgs == [
        {"role": "user", "content": "keep going"},
        {"role": "tool", "content": "out1"},
        {"role": "tool", "content": "a\nb"},
        {"role": "tool", "content": "Error: nope"},
        {"role": "tool", "content": "42"},
    ]


def test_document_block_in_tool_result_is_replaced_by_marker():
    msgs = convert_messages({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "document", "source": {"data": "JVBERi0" * 1000}}]},
    ]}]})
    assert msgs == [{"role": "tool", "content": "[document omitted]"}]


def test_empty_text_only_message_is_dropped():
    msgs = convert_messages({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "  "}]},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": ""}]},
    ]})
    assert msgs == [{"role": "user", "content": "a"}]


# ─── Tool definitions ────────────────────────────────────────────────────────

def test_convert_tools_for_llm():
    assert convert_tools_for_llm([]) is None
    assert convert_tools_for_llm(None) is None
    out = convert_tools_for_llm([BASH, {"name": "NoSchema"}])
    assert out[0] == {"type": "function", "function": {
        "name": "Bash", "description": BASH["description"], "parameters": BASH["input_schema"]}}
    assert out[1]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_format_tools_as_text_lists_params_and_required():
    text = format_tools_as_text(convert_tools_for_llm([BASH]))
    assert "<tool_call>" in text
    assert "## Bash" in text
    assert "  - command: string (required) — the command" in text
    assert "  - timeout: number — ms" in text


def test_slim_tool_keeps_names_types_required_only():
    assert slim_tool(BASH) == {
        "name": "Bash", "description": "Bash",
        "input_schema": {"type": "object",
                         "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
                         "required": ["command"]},
    }


def test_slim_tool_untyped_property_defaults_to_string():
    tool = {"name": "Read", "input_schema": {"properties": {"file_path": {}}}}
    assert slim_tool(tool)["input_schema"] == {"type": "object",
                                               "properties": {"file_path": {"type": "string"}}}


# ─── Code mode / browser mode ────────────────────────────────────────────────

def test_code_session_detection():
    assert looks_like_code_session({"tools": [BASH]})
    assert not looks_like_code_session({"tools": [WEBFETCH]})
    assert not looks_like_code_session({})


def test_optimize_for_code_replaces_prompt_and_filters_tools():
    body = {"system": "x" * 40000, "tools": [BASH, WEBFETCH]}
    out = optimize_for_code(copy.deepcopy(body))
    assert out["system"] == server.CODE_SYSTEM_PROMPT
    assert [t["name"] for t in out["tools"]] == ["Bash"]
    assert out["tools"][0]["description"] == "Bash"


def test_browser_session_detection_and_passthrough():
    click = {"name": "mcp__chrome-devtools__click", "input_schema": {}}
    shot = {"name": "mcp__chrome-devtools__take_screenshot", "input_schema": {}}
    assert looks_like_claude_code_browser_session({"tools": [click]})
    # A direct client with its own prompt and no chrome-devtools tools is left alone.
    direct = {"system": "mine", "tools": []}
    assert optimize_for_browser(copy.deepcopy(direct)) == direct
    out = optimize_for_browser({"system": "big", "tools": [click, shot, BASH]})
    assert out["system"] == server.BROWSER_SYSTEM_PROMPT
    assert [t["name"] for t in out["tools"]] == ["mcp__chrome-devtools__click"]


# ─── tokenize_messages fallbacks ─────────────────────────────────────────────

class NoToolsTemplate:
    """A chat template that rejects the tools= kwarg, like older tokenizers."""

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, **kw):
        if "tools" in kw:
            raise TypeError("unexpected keyword 'tools'")
        self.seen.append(messages)
        return [1, 2, 3]


class BrokenTemplate:
    def apply_chat_template(self, messages, **kw):
        raise ValueError("no template")

    def encode(self, text):
        self.text = text
        return list(range(len(text)))


def test_tokenize_injects_tools_into_existing_system_prompt(monkeypatch):
    tok = NoToolsTemplate()
    monkeypatch.setattr(server, "tokenizer", tok)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert tokenize_messages(msgs, tools=convert_tools_for_llm([BASH])) == [1, 2, 3]
    sent = tok.seen[0]
    assert sent[0]["content"].startswith("sys\n\n# Available Tools")
    assert msgs[0]["content"] == "sys"  # caller's list is not mutated


def test_tokenize_adds_system_prompt_when_missing(monkeypatch):
    tok = NoToolsTemplate()
    monkeypatch.setattr(server, "tokenizer", tok)
    tokenize_messages([{"role": "user", "content": "hi"}], tools=convert_tools_for_llm([BASH]))
    assert tok.seen[0][0]["role"] == "system"
    assert "## Bash" in tok.seen[0][0]["content"]


def test_tokenize_plain_text_fallback(monkeypatch):
    tok = BrokenTemplate()
    monkeypatch.setattr(server, "tokenizer", tok)
    ids = tokenize_messages([{"role": "user", "content": "hi"}])
    assert tok.text == "user: hi\nassistant: "
    assert len(ids) == len(tok.text)
