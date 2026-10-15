"""generate_response in proxy/server.py, driven by a scripted fake model
(see conftest.FakeModel). Covers the response shape Claude Code depends on."""
import server


def body(**kw):
    b = {"model": "claude-sonnet-4-6", "max_tokens": 100,
         "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


BASH = {"name": "Bash", "input_schema": {"type": "object", "properties": {
    "command": {"type": "string"}}, "required": ["command"]}}


def test_plain_text_response_shape(fake_model):
    fake_model.outputs = ["Hello there."]
    r = server.generate_response(body())
    assert r["type"] == "message" and r["role"] == "assistant"
    assert r["id"].startswith("msg_")
    assert r["model"] == "claude-sonnet-4-6"
    assert r["content"] == [{"type": "text", "text": "Hello there."}]
    assert r["stop_reason"] == "end_turn"
    assert r["stop_sequence"] is None
    assert r["usage"]["input_tokens"] > 0
    assert r["usage"]["output_tokens"] == 4  # 12 chars in 3-char pieces


def test_empty_output_is_reported_not_blank(fake_model):
    fake_model.outputs = [""]
    r = server.generate_response(body())
    assert r["content"] == [{"type": "text", "text": "(No output)"}]


def test_tool_call_becomes_tool_use_block_with_no_empty_text(fake_model):
    fake_model.outputs = ['<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>']
    r = server.generate_response(body(tools=[BASH]))
    assert r["stop_reason"] == "tool_use"
    assert len(r["content"]) == 1
    block = r["content"][0]
    assert block["type"] == "tool_use" and block["name"] == "Bash"
    assert block["id"].startswith("toolu_")
    assert block["input"] == {"command": "ls"}


def test_tool_input_keys_outside_the_schema_are_dropped(fake_model):
    fake_model.outputs = ['<tool_call>{"name": "Bash", "arguments": '
                          '{"command": "ls", "description": "list"}}</tool_call>']
    r = server.generate_response(body(tools=[BASH]))
    assert r["content"][0]["input"] == {"command": "ls"}


def test_code_mode_swaps_system_prompt_before_templating(fake_model):
    fake_model.outputs = ["ok"]
    server.generate_response(body(system="HUGE HARNESS PROMPT", tools=[BASH]))
    messages, kw = server.tokenizer.calls[0]
    assert messages[0] == {"role": "system", "content": server.CODE_SYSTEM_PROMPT}
    assert kw["tools"][0]["function"]["name"] == "Bash"


def test_append_system_prompt_file(fake_model, monkeypatch, tmp_path):
    extra = tmp_path / "extra.md"
    extra.write_text("Narrate everything.\n")
    monkeypatch.setenv("MLX_APPEND_SYSTEM_PROMPT_FILE", str(extra))
    fake_model.outputs = ["ok"]
    server.generate_response(body(system="base"))
    messages, _ = server.tokenizer.calls[0]
    assert messages[0]["content"] == "base\n\n---\n\nNarrate everything."


def test_append_system_prompt_file_with_block_list_system(fake_model, monkeypatch, tmp_path):
    extra = tmp_path / "extra.md"
    extra.write_text("More rules.")
    monkeypatch.setenv("MLX_APPEND_SYSTEM_PROMPT_FILE", str(extra))
    fake_model.outputs = ["ok"]
    server.generate_response(body(system=[{"type": "text", "text": "base"}]))
    messages, _ = server.tokenizer.calls[0]
    assert messages[0]["content"] == "base\n\n\n---\n\nMore rules."


def test_max_tokens_finish_reason(fake_model, monkeypatch):
    def gen(model, tokenizer, prompt, max_tokens, **kw):
        yield type("R", (), {"text": "abc", "generation_tokens": 1, "finish_reason": "length"})()
    monkeypatch.setattr(server, "stream_generate", gen)
    r = server.generate_response(body())
    assert r["stop_reason"] == "max_tokens"


def test_second_request_reuses_the_cached_prefix(fake_model):
    fake_model.outputs = ["one", "two"]
    first = body()
    server.generate_response(first)
    second = body(messages=first["messages"] + [
        {"role": "assistant", "content": "one"}, {"role": "user", "content": "again"}])
    server.generate_response(second)
    full_first, delta_second = fake_model.prompts
    full_second = server.tokenize_messages(server.convert_messages(second))
    shared = next(i for i, (a, b) in enumerate(zip(full_first, full_second)) if a != b)
    assert shared > 0
    assert delta_second == full_second[shared:]  # only the new tail is prefilled


def test_count_prompt_tokens_matches_generation_prompt(fake_model):
    b = body(tools=[BASH])
    n = server.count_prompt_tokens(b)
    fake_model.outputs = ["ok"]
    # count_prompt_tokens does not apply code mode, so compare with it off.
    server.CODE_MODE_ENABLED, saved = False, server.CODE_MODE_ENABLED
    try:
        r = server.generate_response(b)
    finally:
        server.CODE_MODE_ENABLED = saved
    assert r["usage"]["input_tokens"] == n
