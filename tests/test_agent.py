"""agent/agent.py (the Native Engine): XML tool-call parsing, the six tools,
and the HTTP backend talking to this repo's own proxy over a real socket."""
import json
import shutil
import threading
from http.server import HTTPServer

import pytest

import agent
import server

# ─── XML dialect ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value,typ,expected", [
    (" 42 ", "integer", 42),
    ("4x", "integer", "4x"),
    ("2.5", "number", 2.5),
    ("n/a", "number", "n/a"),
    ("True", "boolean", True),
    ("yes", "boolean", True),
    ("false", "boolean", False),
    ('["a", 1]', "array", ["a", 1]),
    ("[oops", "array", "[oops"),
    ('{"k": "v"}', "object", {"k": "v"}),
    ("  keep  spaces ", "string", "  keep  spaces "),
])
def test_coerce(value, typ, expected):
    assert agent.coerce(value, typ) == expected


def test_parse_tool_calls_raw_text_values_and_types():
    text = (
        "I'll fix it.\n<tool_call>\n<function=Edit>\n<parameter=path>\n/tmp/a.py\n</parameter>\n"
        '<parameter=old_string>\nprint("hi")\n</parameter>\n'
        "<parameter=new_string>\nprint('bye')\n</parameter>\n"
        "<parameter=replace_all>\ntrue\n</parameter>\n</function>\n</tool_call>"
    )
    assert agent.parse_tool_calls(text) == [("Edit", {
        "path": "/tmp/a.py", "old_string": 'print("hi")',
        "new_string": "print('bye')", "replace_all": True})]
    assert agent.strip_calls(text) == "I'll fix it."


def test_parse_tool_calls_without_the_wrapper():
    text = "<function=Read>\n<parameter=path>\n/x\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>"
    assert agent.parse_tool_calls(text) == [("Read", {"path": "/x", "limit": 5})]


def test_parse_unknown_tool_keeps_strings():
    assert agent.parse_tool_calls("<function=Nope><parameter=n>\n3\n</parameter></function>") == [
        ("Nope", {"n": "3"})]


def test_preview():
    assert agent.preview("Bash", {"command": "ls"}) == "ls"
    assert agent.preview("Edit", {"path": "/a"}) == "/a"
    assert agent.preview("Glob", {"pattern": "*.py"}) == "*.py"
    assert agent.preview("Grep", {"pattern": "foo"}) == "foo"
    assert agent.preview("Other", {"a": 1}) == '{"a": 1}'


def test_tools_doc_marks_optional_params():
    doc = agent.tools_doc()
    assert "- Read(path: string, offset?: integer, limit?: integer)" in doc
    assert "<function=ToolName>" in doc


def test_clip():
    assert agent._clip("short", 10) == "short"
    clipped = agent._clip("a" * 50 + "b" * 50, 20)
    assert clipped.startswith("a" * 10) and clipped.endswith("b" * 10)
    assert "[80 chars elided]" in clipped


# ─── Tools ───────────────────────────────────────────────────────────────────

def test_write_read_edit_roundtrip(tmp_path):
    f = tmp_path / "sub" / "f.txt"
    assert "3 lines" in agent.tool_write(str(f), "one\ntwo\nthree")
    assert agent.tool_read(str(f)) == "1  one\n2  two\n3  three"
    assert agent.tool_read(str(f), offset=2, limit=1) == "2  two"
    assert agent.tool_read(str(f), offset=10) == "[empty or past end of file]"
    assert "1 replacement" in agent.tool_edit(str(f), "two", "2")
    assert f.read_text() == "one\n2\nthree"


def test_edit_refuses_missing_or_ambiguous_match(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x x")
    assert agent.tool_edit(str(f), "y", "z").startswith("ERROR: old_string not found")
    assert "appears 2 times" in agent.tool_edit(str(f), "x", "z")
    assert "2 replacement" in agent.tool_edit(str(f), "x", "z", replace_all=True)
    assert f.read_text() == "z z"


def test_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    assert agent.tool_glob("*.py", str(tmp_path)) == str(tmp_path / "a.py")
    assert agent.tool_glob("*.rs", str(tmp_path)) == "[no matches]"


def test_bash_reports_exit_code_and_closes_stdin():
    assert agent.tool_bash("echo hi") == "hi"
    out = agent.tool_bash("echo oops >&2; exit 3")
    assert out.startswith("oops") and out.endswith("[exit 3]")
    assert agent.tool_bash("cat") == "[no output]"  # stdin is /dev/null, so cat returns at once


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep (rg) is not installed")
def test_grep(tmp_path):
    (tmp_path / "a.py").write_text("needle = 1\n")
    assert agent.tool_grep("needle", str(tmp_path)).endswith("a.py:1:needle = 1")
    assert agent.tool_grep("haystack", str(tmp_path)) == "[no matches]"


def test_grep_without_ripgrep(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("rg")
    monkeypatch.setattr(agent.subprocess, "run", missing)
    assert agent.tool_grep("x") == "ERROR: ripgrep (rg) is not installed."


def test_run_tool_error_paths(tmp_path):
    assert agent.run_tool("Nope", {}).startswith("ERROR: no tool named Nope")
    assert agent.run_tool("Read", {"file": "x"}).startswith("ERROR: bad arguments for Read")
    assert agent.run_tool("Read", {"path": str(tmp_path / "missing")}).startswith("ERROR: ")


def test_run_tool_timeout(monkeypatch):
    monkeypatch.setattr(agent, "BASH_TIMEOUT", 1)
    assert agent.run_tool("Bash", {"command": "sleep 5"}) == "ERROR: Bash timed out after 1s."


# ─── HTTPEngine against proxy/server.py ──────────────────────────────────────

@pytest.fixture
def proxy_url(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), server.AnthropicHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setattr(agent, "BASE_URL", url)
    yield url
    httpd.shutdown()
    httpd.server_close()


def test_http_engine_tool_round_trip(proxy_url, fake_model):
    fake_model.outputs = [
        'Looking.\n<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>',
        "There are two files.",
    ]
    eng = agent.HTTPEngine("stub-model")
    eng.add_user("what's here?")
    streamed = []
    prose, calls = eng.step(streamed.append)
    assert prose == "Looking."
    assert "".join(streamed) == "Looking."
    [(call_id, name, args)] = calls
    assert call_id.startswith("toolu_") and name == "Bash" and args == {"command": "ls"}

    eng.add_tool_results([(call_id, name, "a.py\nb.py")])
    prose, calls = eng.step(lambda _: None)
    assert prose == "There are two files." and calls == []

    # The proxy saw the tool result as a tool message on the second turn.
    second_prompt_messages = server.tokenizer.calls[-1][0]
    assert {"role": "tool", "content": "a.py\nb.py"} in second_prompt_messages
    assert [m["role"] for m in eng.messages] == ["user", "assistant", "user", "assistant"]


def test_http_engine_ctx_estimate():
    eng = agent.HTTPEngine("stub-model")
    used, limit = eng.ctx_used()
    assert used == len(eng.system) // 4
    eng.add_user("x" * 400)
    assert eng.ctx_used()[0] > used
    eng.reset()
    assert eng.messages == []
    assert {t["name"] for t in eng.tools} == set(agent.DISPATCH)
    assert all("input_schema" in t for t in eng.tools)
    json.dumps(eng.tools)
