"""parse_tool_calls in proxy/server.py: every model output format it reads.

scripts/test_parse_tool_calls.py keeps its regression cases (and still runs
as part of this suite); this file covers each format branch end to end.
"""
import pytest

from server import (
    _decode_string_args,
    _gemma4_args,
    parse_tool_calls,
    recover_garbled_tool_json,
)

Q = '<|"|>'  # Gemma 4's string delimiter


def calls(text):
    found, _ = parse_tool_calls(text)
    return [(c["name"], c["arguments"]) for c in found]


# ─── Format 0: Gemma 4 native ────────────────────────────────────────────────

@pytest.mark.parametrize("opener", ["<|tool_call>call:", "<|tool_call>", "<tool_call:"])
def test_gemma4_all_three_openers(opener):
    text = f"{opener}Bash{{command:{Q}ls -la{Q}}}<tool_call|>"
    assert calls(text) == [("Bash", {"command": "ls -la"})]


def test_gemma4_nested_array_and_dash_key():
    text = (f"<|tool_call>call:Grep{{pattern:{Q}foo{Q},-i:true,"
            f"paths:[{Q}a{Q},{Q}b{Q}]}}<tool_call|>")
    assert calls(text) == [("Grep", {"pattern": "foo", "-i": True, "paths": ["a", "b"]})]


def test_gemma4_string_with_quotes_and_newlines_survives():
    body = 'print("hi")\nx = {"a": 1}'
    text = f"<|tool_call>call:Write{{file_path:{Q}/t.py{Q},content:{Q}{body}{Q}}}<tool_call|>"
    assert calls(text) == [("Write", {"file_path": "/t.py", "content": body})]


def test_gemma4_two_calls_and_surrounding_prose():
    text = (f"Let me look.\n<|tool_call>call:Read{{file_path:{Q}/a{Q}}}<tool_call|>"
            f"<|tool_call>call:Read{{file_path:{Q}/b{Q}}}<tool_call|>")
    found, remaining = parse_tool_calls(text)
    assert [c["arguments"]["file_path"] for c in found] == ["/a", "/b"]
    assert remaining == "Let me look."


def test_gemma4_stray_closing_angle_is_tolerated():
    text = f"<tool_call:Bash{{command:{Q}pwd{Q}}}><tool_call|>"
    assert calls(text) == [("Bash", {"command": "pwd"})]


def test_gemma4_dotted_tool_name():
    text = f"<|tool_call>call:ns.tool{{q:{Q}x{Q}}}<tool_call|>"
    assert calls(text) == [("ns.tool", {"q": "x"})]


def test_gemma4_dashed_mcp_tool_name():
    # MCP tool names carry the server name, which often has a dash.
    text = f"<|tool_call>call:mcp__chrome-devtools__click{{uid:{Q}12{Q}}}<tool_call|>"
    assert calls(text) == [("mcp__chrome-devtools__click", {"uid": "12"})]


def test_gemma4_call_with_no_arguments():
    # chrome-devtools list_pages takes no arguments; the call still has to go out.
    text = "<|tool_call>call:list_pages{}<tool_call|>"
    assert calls(text) == [("list_pages", {})]


def test_gemma4_unparseable_body_is_dropped_not_raised():
    found, _ = parse_tool_calls("<|tool_call>no braces here<tool_call|>")
    assert found == []


# ─── _gemma4_args ────────────────────────────────────────────────────────────

def test_gemma4_args_quotes_bare_keys():
    assert _gemma4_args(f"a:{Q}x{Q},b:2,c:null") == {"a": "x", "b": 2, "c": None}


def test_gemma4_args_nested_object():
    assert _gemma4_args(f"opts:{{deep:{Q}v{Q}}}") == {"opts": {"deep": "v"}}


def test_gemma4_args_invalid_returns_none():
    assert _gemma4_args("a:unquoted words") is None


def test_gemma4_args_empty_is_empty_dict():
    assert _gemma4_args("") == {}


# ─── Format 0.5: Llama 3.3 raw JSON ──────────────────────────────────────────

def test_llama_raw_json():
    text = '{"type": "function", "name": "Bash", "parameters": {"command": "ls"}}'
    assert calls(text) == [("Bash", {"command": "ls"})]


def test_llama_parameters_as_json_string():
    text = '{"type": "function", "name": "Bash", "parameters": "{\\"command\\": \\"ls\\"}"}'
    assert calls(text) == [("Bash", {"command": "ls"})]


def test_llama_non_function_type_is_ignored():
    found, remaining = parse_tool_calls('{"type": "text", "name": "x"} hello')
    assert found == []
    assert "hello" in remaining


# ─── Format 1: <tool_call>JSON</tool_call> ───────────────────────────────────

def test_tool_call_tag_json_and_remaining_text():
    text = 'Running it.\n<tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call>'
    found, remaining = parse_tool_calls(text)
    assert [(c["name"], c["arguments"]) for c in found] == [("Bash", {"command": "ls"})]
    assert remaining == "Running it."


def test_tool_call_tag_array_of_calls():
    text = ('<tool_call>[{"name": "Read", "arguments": {"file_path": "/a"}},'
            '{"name": "Read", "arguments": {"file_path": "/b"}}]</tool_call>')
    assert calls(text) == [("Read", {"file_path": "/a"}), ("Read", {"file_path": "/b"})]


def test_tool_call_tag_scalar_json_is_skipped():
    assert calls('<tool_call>"just a string"</tool_call>') == []


def test_tool_call_tag_empty_is_skipped():
    assert calls("<tool_call>   </tool_call>") == []


def test_function_format_inside_tool_call_tag():
    text = ("<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n"
            "</function>\n</tool_call>")
    assert calls(text) == [("Bash", {"command": "ls -la"})]


def test_garbled_json_inside_tool_call_tag_is_recovered():
    text = '<tool_call>{"name": "Bash", "parameter=command>cd /tmp && ls</parameter></tool_call>'
    assert calls(text) == [("Bash", {"command": "cd /tmp && ls"})]


def test_unrecoverable_tool_call_tag_yields_nothing():
    assert calls("<tool_call>{not json, no name}</tool_call>") == []


# ─── Format 2: <function=...> ────────────────────────────────────────────────

def test_function_parameter_format():
    text = ("<function=Edit><parameter=file_path>/a.py</parameter>"
            "<parameter=old_string>x</parameter><parameter=new_string>y</parameter></function>")
    assert calls(text) == [("Edit", {"file_path": "/a.py", "old_string": "x", "new_string": "y"})]


# ─── Format 3 / 3.5: <|tool_call|> and <tools> ───────────────────────────────

def test_pipe_tool_call_tags():
    text = '<|tool_call|>{"name": "Glob", "arguments": {"pattern": "*.py"}}<|/tool_call|>'
    assert calls(text) == [("Glob", {"pattern": "*.py"})]


def test_qwen25_tools_wrapper_with_parameters_key():
    text = '<tools>{"name": "Bash", "parameters": {"command": "pwd"}}</tools>'
    assert calls(text) == [("Bash", {"command": "pwd"})]


def test_qwen25_tools_wrapper_array():
    text = '<tools>[{"name": "Bash", "arguments": {"command": "a"}}, {"nope": 1}]</tools>'
    assert calls(text) == [("Bash", {"command": "a"})]


# ─── Format 3.6 / 3.7: fenced and bare JSON ──────────────────────────────────

def test_markdown_fence_alternate_keys():
    text = '```tool_call\n{"tool": "Read", "input": {"file_path": "/x"}}\n```'
    assert calls(text) == [("Read", {"file_path": "/x"})]


def test_markdown_fence_that_is_not_a_tool_call_is_left_alone():
    text = 'Example config:\n```json\n{"port": 4000}\n```'
    found, remaining = parse_tool_calls(text)
    assert found == []
    assert '"port": 4000' in remaining


def test_bare_json_back_to_back():
    text = ('{"name": "Bash", "arguments": {"command": "a"}}\n'
            '{"name": "Bash", "arguments": {"command": "b"}}')
    found, remaining = parse_tool_calls(text)
    assert [c["arguments"]["command"] for c in found] == ["a", "b"]
    assert remaining == ""


def test_plain_prose_is_not_a_tool_call():
    found, remaining = parse_tool_calls("The answer is 42.")
    assert found == []
    assert remaining == "The answer is 42."


# ─── Format 4: tagless garble ────────────────────────────────────────────────

def test_tagless_with_name():
    text = 'Sure. "name": "Bash" <parameter=command>ls</parameter>'
    found, remaining = parse_tool_calls(text)
    assert [(c["name"], c["arguments"]) for c in found] == [("Bash", {"command": "ls"})]
    assert remaining == "Sure."


@pytest.mark.parametrize("param,tool", [
    ("command", "Bash"), ("file_path", "Read"), ("pattern", "Glob"),
])
def test_tagless_name_inferred_from_parameter(param, tool):
    assert calls(f"<parameter={param}>v</parameter>") == [(tool, {param: "v"})]


# ─── Dedup and string arguments ──────────────────────────────────────────────

def test_identical_calls_are_deduplicated_but_parallel_calls_kept():
    one = '<tool_call>{"name": "Read", "arguments": {"file_path": "/a"}}</tool_call>'
    two = '<tool_call>{"name": "Read", "arguments": {"file_path": "/b"}}</tool_call>'
    assert calls(one + one + two) == [("Read", {"file_path": "/a"}), ("Read", {"file_path": "/b"})]


def test_openai_style_string_arguments_are_decoded():
    text = '<tool_call>{"name": "Bash", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>'
    assert calls(text) == [("Bash", {"command": "ls"})]


@pytest.mark.parametrize("value,expected", [
    ('{"a": 1}', {"a": 1}),
    ("not json", "not json"),
    ("[1, 2]", "[1, 2]"),
    ({"a": 1}, {"a": 1}),
    (None, None),
])
def test_decode_string_args(value, expected):
    assert _decode_string_args(value) == expected


# ─── recover_garbled_tool_json ───────────────────────────────────────────────

def test_recover_pattern_a_parameter_equals():
    got = recover_garbled_tool_json('{"name": "Edit", "parameter=file_path>/a.py</parameter>')
    assert got == {"name": "Edit", "arguments": {"file_path": "/a.py"}}


def test_recover_pattern_b_parameter_underscore():
    got = recover_garbled_tool_json('{"name": "Bash", "<parameter_command>["rm -rf build"]')
    assert got == {"name": "Bash", "arguments": {"command": "rm -rf build"}}


def test_recover_pattern_c_broken_arguments_object():
    got = recover_garbled_tool_json('{"name": "Read", "arguments": {"file_path": "/x", "limit": "5"')
    assert got == {"name": "Read", "arguments": {"file_path": "/x", "limit": "5"}}


def test_recover_pattern_d_single_argument_tool():
    got = recover_garbled_tool_json('{"name": "Bash", ls -la /tmp"}')
    assert got == {"name": "Bash", "arguments": {"command": "ls -la /tmp"}}


def test_recover_without_a_name_gives_up():
    assert recover_garbled_tool_json('{"arguments": {"x": "y"}}') is None


def test_recover_unknown_tool_with_no_arguments_gives_up():
    assert recover_garbled_tool_json('{"name": "Mystery"}') is None
