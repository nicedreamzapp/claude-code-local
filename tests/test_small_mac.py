"""Small-Mac tier: vision messages, the context ceiling, tool-argument fixes,
the code-mode prompt, and scripts/mac-tier.sh. No model, no MLX, no mlx-vlm."""
import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

import server
from conftest import FakeResp, FakeTokenizer

ROOT = Path(__file__).resolve().parent.parent
PNG = "iVBORw0KGgo" + "A" * 5000


def image_block(data=PNG):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def read_result(*blocks):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                         "content": list(blocks)}]}


# ─── wants_vision ────────────────────────────────────────────────────────────

VISION_CFG = {"model_type": "qwen3_5", "vision_config": {"depth": 24}}


@pytest.mark.parametrize("setting,cfg,vlm,expected", [
    ("auto", VISION_CFG, True, True),
    ("auto", {"model_type": "qwen3"}, True, False),   # text-only model
    ("auto", VISION_CFG, False, False),               # mlx-vlm not installed
    ("0", VISION_CFG, True, False),
    ("off", VISION_CFG, True, False),
    ("1", {}, True, True),                            # forced on
    ("1", VISION_CFG, False, False),
])
def test_wants_vision(setting, cfg, vlm, expected):
    assert server.wants_vision(setting, cfg, vlm) is expected


def test_model_config_reads_a_local_folder(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(VISION_CFG))
    assert server.model_config(str(tmp_path)) == VISION_CFG
    assert server.model_config(str(tmp_path / "missing")) == {}


# ─── convert_messages with images ────────────────────────────────────────────

def test_text_path_still_gets_a_marker_not_base64():
    msgs = server.convert_messages({"messages": [read_result(
        {"type": "text", "text": "Read 1 image"}, image_block())]})
    assert msgs == [{"role": "tool", "content": "Read 1 image\n[image omitted]"}]


def test_vision_keeps_the_image_in_the_newest_message():
    msgs = server.convert_messages({"messages": [
        {"role": "user", "content": "look at badge.png"}, read_result(image_block())]},
        vision=True)
    tool = msgs[-1]
    assert tool["role"] == "tool"
    assert tool["content"] == [{"type": "image", "image": "data:image/png;base64," + PNG}]
    assert server.collect_images(msgs) == ["data:image/png;base64," + PNG]


def test_vision_drops_images_from_older_messages():
    body = {"messages": [
        {"role": "user", "content": "look"},
        read_result(image_block("OLD")),
        {"role": "assistant", "content": "a red circle"},
        read_result(image_block("NEW")),
    ]}
    msgs = server.convert_messages(body, vision=True)
    assert msgs[1]["content"] == "[image omitted]"
    assert server.collect_images(msgs) == ["data:image/png;base64,NEW"]


def test_vision_user_pasted_image_comes_before_its_text():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"}, image_block()]}]}
    msgs = server.convert_messages(body, vision=True)
    assert [i["type"] for i in msgs[0]["content"]] == ["image", "text"]
    # Without vision the pasted image is a marker next to the text.
    plain = server.convert_messages(copy.deepcopy(body))
    assert plain[0]["content"] == "[image omitted]\nwhat is this?"


def test_vision_error_result_keeps_its_prefix():
    msg = read_result(image_block())
    msg["content"][0]["is_error"] = True
    msgs = server.convert_messages({"messages": [msg]}, vision=True)
    assert msgs[0]["content"][0] == {"type": "text", "text": "Error: "}


def test_image_source_url_and_unknown():
    assert server.image_source({"source": {"type": "url", "url": "https://x/y.png"}}) == "https://x/y.png"
    assert server.image_source({"source": {"type": "file", "file_id": "f"}}) is None


# ─── context ceiling ─────────────────────────────────────────────────────────

def test_check_context():
    server.check_context(100, limit=0)
    server.check_context(100, limit=100)
    with pytest.raises(server.PromptTooLong, match="prompt is too long: 101 tokens > 100"):
        server.check_context(101, limit=100)


def test_generate_response_refuses_a_prompt_over_the_ceiling(fake_model, monkeypatch):
    monkeypatch.setattr(server, "MAX_CONTEXT", 10)
    with pytest.raises(server.PromptTooLong):
        server.generate_response({"messages": [{"role": "user", "content": "x" * 500}]})
    assert fake_model.prompts == []  # nothing was generated


# ─── tool arguments ──────────────────────────────────────────────────────────

def test_coerce_args_uses_the_schema():
    props = {"allow_large": {"type": "boolean"}, "limit": {"type": "integer"},
             "ratio": {"type": "number"}, "paths": {"type": "array"},
             "file_path": {"type": "string"}}
    out = server.coerce_args({"allow_large": "True", "limit": "20", "ratio": "0.5",
                              "paths": '["a", "b"]', "file_path": "123"}, props)
    assert out == {"allow_large": True, "limit": 20, "ratio": 0.5,
                   "paths": ["a", "b"], "file_path": "123"}


def test_coerce_args_leaves_bad_values_alone():
    props = {"limit": {"type": "integer"}, "flag": {"type": "boolean"}}
    assert server.coerce_args({"limit": "lots", "flag": "maybe"}, props) == {
        "limit": "lots", "flag": "maybe"}


def test_fix_bash_timeout_reads_small_numbers_as_seconds():
    assert server.fix_bash_timeout("Bash", {"command": "ls", "timeout": 5})["timeout"] == 5000
    assert server.fix_bash_timeout("Bash", {"command": "ls", "timeout": 5000})["timeout"] == 5000
    assert server.fix_bash_timeout("Bash", {"command": "ls", "timeout": 900})["timeout"] == 600000
    assert server.fix_bash_timeout("Read", {"timeout": 5}) == {"timeout": 5}


def test_assembled_tool_use_has_typed_input():
    read = {"name": "Read", "input_schema": {"type": "object", "properties": {
        "file_path": {"type": "string"}, "allow_large": {"type": "boolean"}}}}
    msg = server.assemble_anthropic_message(
        "", [{"name": "Read", "arguments": {"file_path": "a.png", "allow_large": "True"}}],
        {"tools": [read]}, [read], 10, 5)
    assert msg["content"][0]["input"] == {"file_path": "a.png", "allow_large": True}


# ─── code-mode prompt ────────────────────────────────────────────────────────

def test_working_directory_is_found_in_the_harness_prompt():
    system = "You are Claude Code.\n<env>\n - Primary working directory: /Users/me/proj\n</env>"
    assert server.working_directory(system) == "/Users/me/proj"
    assert server.working_directory([{"type": "text", "text": "Working directory: /tmp/x"}]) == "/tmp/x"
    assert server.working_directory("nothing here") is None


def test_code_prompt_lists_only_offered_tools_and_the_cwd():
    bash = {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}
    read = {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}
    body = {"system": "Primary working directory: /tmp/proj", "tools": [bash, read]}
    out = server.optimize_for_code(body)
    assert "- Bash:" in out["system"] and "- Read:" in out["system"]
    assert "- Write:" not in out["system"] and "- Edit:" not in out["system"]
    assert "Working directory: /tmp/proj" in out["system"]


def test_code_prompt_tells_a_vision_model_read_opens_images(monkeypatch):
    monkeypatch.setattr(server, "VISION", True)
    assert "Read also opens images" in server.code_system_prompt({"Read"})
    monkeypatch.setattr(server, "VISION", False)
    assert "Read also opens images" not in server.code_system_prompt({"Read"})


# ─── the vision generation path ──────────────────────────────────────────────

class TextTemplateTokenizer(FakeTokenizer):
    """apply_chat_template(tokenize=False) returns text, like a real HF tokenizer."""

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return json.dumps(messages)


def test_vision_generation_sends_prompt_text_and_images(monkeypatch):
    seen = {}

    def fake_vlm(model, processor, prompt, image=None, **kw):
        seen.update(prompt=prompt, image=image, kw=kw)
        yield FakeResp("a red circle", 4, "stop")

    tok = TextTemplateTokenizer()
    monkeypatch.setattr(server, "VISION", True)
    monkeypatch.setattr(server, "tokenizer", tok)
    monkeypatch.setattr(server, "processor", object())
    monkeypatch.setattr(server, "model", object())
    monkeypatch.setattr(server, "make_sampler", lambda **k: None, raising=False)
    monkeypatch.setattr(server, "vlm_stream_generate", fake_vlm, raising=False)
    monkeypatch.setattr(server, "_prompt_cache", None)
    monkeypatch.setattr(server, "_first_request", False)

    r = server.generate_response({"messages": [
        {"role": "user", "content": "what is in badge.png?"}, read_result(image_block())]})
    assert r["content"][0]["text"] == "a red circle"
    assert seen["image"] == ["data:image/png;base64," + PNG]
    assert seen["kw"]["max_tokens"] == server.DEFAULT_MAX_TOKENS
    # Thinking is switched off through the template on the vision path.
    assert tok.calls[-1][1].get("enable_thinking") is False
    assert tok.calls[-1][1].get("tokenize") is False


# ─── scripts/mac-tier.sh ─────────────────────────────────────────────────────

def tier_env(**env):
    full = {k: v for k, v in os.environ.items()
            if not k.startswith(("CCL_", "MLX_", "CLAUDE_CODE_"))}
    full.update(env)
    out = subprocess.run(["bash", str(ROOT / "scripts" / "mac-tier.sh"), "--env"],
                         capture_output=True, text=True, env=full, check=True).stdout
    vals = {}
    for line in out.splitlines():
        k, _, v = line.removeprefix("export ").partition("=")
        vals[k] = v
    return vals


@pytest.mark.parametrize("gb,tier,model,ctx", [
    ("8", "8gb", "mlx-community/Qwen3.5-2B-MLX-8bit", "16384"),
    ("15", "8gb", "mlx-community/Qwen3.5-2B-MLX-8bit", "16384"),
    ("16", "16gb", "mlx-community/Qwen3.5-4B-bf16", "16384"),
    ("24", "16gb", "mlx-community/Qwen3.5-4B-bf16", "16384"),
])
def test_tier_by_memory(gb, tier, model, ctx):
    v = tier_env(CCL_MEM_GB=gb)
    assert v["CCL_TIER"] == tier
    assert v["CCL_MODEL_ID"] == model
    assert v["MLX_MAX_CONTEXT"] == ctx
    # Claude Code treats any window under 100000 as 100000, so the 16K tiers don't set one;
    # the server's MLX_MAX_CONTEXT ceiling enforces it instead.
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in v
    assert v["MLX_VISION"] == "1"
    assert v["MLX_MODEL"].endswith("/" + model.split("/")[1])


def test_big_macs_get_no_tier():
    assert tier_env(CCL_MEM_GB="64") == {"CCL_TIER": "none"}


def test_tier_override_and_user_settings_win():
    v = tier_env(CCL_MEM_GB="64", CCL_TIER="16gb", MLX_MAX_CONTEXT="12000")
    assert v["CCL_TIER"] == "16gb"
    assert v["MLX_MAX_CONTEXT"] == "12000"


def test_over_long_prompt_is_a_400_invalid_request_over_http(monkeypatch):
    import http.client
    import threading
    from http.server import HTTPServer

    def too_long(body, **kw):
        raise server.PromptTooLong("prompt is too long: 9000 tokens > 8192 maximum")

    monkeypatch.setattr(server, "generate_response", too_long)
    httpd = HTTPServer(("127.0.0.1", 0), server.AnthropicHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.request("POST", "/v1/messages", body=json.dumps(
            {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Bash"}]}),
            headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert resp.status == 400
        assert data["error"]["type"] == "invalid_request_error"
        assert "prompt is too long" in data["error"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()


# ─── dropped leading slash ───────────────────────────────────────────────────

def test_repair_paths_restores_a_dropped_slash_inside_the_cwd():
    cwd = "/private/tmp/proj"
    assert server.repair_paths({"file_path": "private/tmp/proj/notes.txt"}, cwd) == {
        "file_path": "/private/tmp/proj/notes.txt"}
    # Real relative paths, other folders and absolute paths are left alone.
    for p in ("notes.txt", "src/a.py", "./x", "Users/other/x", "/private/tmp/proj/a"):
        assert server.repair_paths({"file_path": p}, cwd) == {"file_path": p}
    assert server.repair_paths({"file_path": "private/tmp/proj/a"}, None) == {
        "file_path": "private/tmp/proj/a"}


def test_assembled_tool_use_gets_the_slash_back():
    read = {"name": "Read", "input_schema": {"type": "object", "properties": {
        "file_path": {"type": "string"}}}}
    body = {"system": server.code_system_prompt({"Read"}, "/private/tmp/proj"), "tools": [read]}
    msg = server.assemble_anthropic_message(
        "", [{"name": "Read", "arguments": {"file_path": "private/tmp/proj/notes.txt"}}],
        body, [read], 10, 5)
    assert msg["content"][0]["input"] == {"file_path": "/private/tmp/proj/notes.txt"}


# ─── compaction requests ─────────────────────────────────────────────────────

COMPACT = ("Your task is to create a detailed summary of the conversation so far, "
           "paying close attention to the user's explicit requests.\n"
           "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.")


def test_looks_like_compaction():
    assert server.looks_like_compaction({"messages": [{"role": "user", "content": COMPACT}]})
    assert server.looks_like_compaction({"messages": [
        {"role": "user", "content": [{"type": "text", "text": COMPACT}]}]})
    assert not server.looks_like_compaction({"messages": [{"role": "user", "content": "summarize x"}]})
    assert not server.looks_like_compaction({"messages": []})


def test_fit_compaction_drops_oldest_turns_but_keeps_task_and_instruction():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}] + [
        {"role": "tool", "content": "x" * 100} for _ in range(10)] + [
        {"role": "user", "content": "summarize"}]
    out, n = server.fit_compaction(msgs, lambda m: sum(len(x["content"]) for x in m), 450)
    assert n <= 450
    assert out[:2] == msgs[:2] and out[-1] == msgs[-1]
    assert len(out) == 7  # 4 of the 10 tool results fit


def test_compaction_request_is_answered_in_text_without_tools(fake_model, monkeypatch):
    fake_model.outputs = ['<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>']
    bash = {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}
    r = server.generate_response({"system": "HARNESS", "tools": [bash], "messages": [
        {"role": "user", "content": "fix calc.py"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": COMPACT}]})
    messages, kw = server.tokenizer.calls[0]
    assert messages[0] == {"role": "system", "content": server.COMPACT_SYSTEM_PROMPT}
    assert "tools" not in kw
    # No tools were offered, so nothing is parsed into a tool_use.
    assert [b["type"] for b in r["content"]] == ["text"]


def test_compaction_over_the_ceiling_is_trimmed_not_refused(fake_model, monkeypatch):
    monkeypatch.setattr(server, "MAX_CONTEXT", 3000)
    fake_model.outputs = ["summary"]
    msgs = [{"role": "user", "content": "task"}]
    for i in range(20):
        msgs += [{"role": "assistant", "content": "a" * 200}, {"role": "user", "content": "b" * 200}]
    msgs.append({"role": "user", "content": COMPACT})
    r = server.generate_response({"messages": msgs})
    assert r["content"][0]["text"] == "summary"
    assert len(fake_model.prompts[0]) <= 3000


def test_shrink_image(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    import base64
    import io
    buf = io.BytesIO()
    PIL.new("RGB", (1600, 900), "red").save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    small = server.shrink_image(uri, 800)
    assert small.size == (800, 450)
    assert server.shrink_image(uri, 0) == uri
    assert server.shrink_image(uri, 2000) == uri
    assert server.shrink_image("https://x/y.png", 800) == "https://x/y.png"


def test_fit_compaction_cuts_one_giant_message():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 10000},
            {"role": "user", "content": "summarize"}]
    out, n = server.fit_compaction(msgs, lambda m: sum(len(x["content"]) for x in m), 3000)
    assert n <= 3000
    assert out[-1]["content"] == "summarize"
    assert "middle cut to fit" in out[1]["content"]


def test_fit_compaction_keeps_the_instruction_wherever_it_is():
    # convert_messages puts a message's text before its tool results, so the
    # instruction can come before the huge tool output.
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "calling Read"},
            {"role": "user", "content": COMPACT}, {"role": "tool", "content": "y" * 20000}]
    out, n = server.fit_compaction(msgs, lambda m: sum(len(x["content"]) for x in m), 4000)
    assert n <= 4000
    assert any(m["content"] == COMPACT for m in out)
