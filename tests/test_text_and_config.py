"""Output cleanup (think blocks, stop markers) and env config in proxy/server.py."""
import pytest

import server
from server import ThinkingFilter, clean_response, env_int, strip_think_tags

# ─── env_int ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [(None, 7), ("", 7), ("   ", 7), ("12", 12)])
def test_env_int(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("CCL_TEST_INT", raising=False)
    else:
        monkeypatch.setenv("CCL_TEST_INT", raw)
    assert env_int("CCL_TEST_INT", 7) == expected


def test_env_int_garbage_raises(monkeypatch):
    monkeypatch.setenv("CCL_TEST_INT", "lots")
    with pytest.raises(ValueError):
        env_int("CCL_TEST_INT", 7)


def test_import_time_defaults():
    # conftest clears the MLX_* env, so these are the shipped defaults.
    assert server.PORT == 4000
    assert server.PREFILL_SIZE == 512
    assert server.KV_BITS == 0
    assert server.MAX_TOOL_RETRIES == 2
    assert server.SUPPRESS_THINKING is True
    assert server.BROWSER_MODE is False
    assert server.CODE_MODE_ENABLED is True


# ─── strip_think_tags ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("<think>plan</think>Answer", "Answer"),
    ("stray</think>Answer", "strayAnswer"),
    ("<|channel>thought\nhmm<channel|>Answer", "Answer"),
    ("Answer<tool_call>  </tool_call>", "Answer"),
    ("<think>only thinking</think>", "<think>only thinking</think>"),
])
def test_strip_think_tags(text, expected):
    assert strip_think_tags(text) == expected


# ─── ThinkingFilter ──────────────────────────────────────────────────────────

GEMMA = "Hi <|channel>thought\nsecret plan<channel|>there"


@pytest.mark.parametrize("cut", range(1, len(GEMMA)))
def test_thinking_filter_any_chunk_boundary(cut):
    tf = ThinkingFilter()
    out = tf.feed(GEMMA[:cut]) + tf.feed(GEMMA[cut:]) + tf.flush()
    assert out == "Hi there"


def test_thinking_filter_one_char_at_a_time():
    tf = ThinkingFilter()
    out = "".join(tf.feed(ch) for ch in GEMMA * 2) + tf.flush()
    assert out == "Hi thereHi there"


def test_thinking_filter_unclosed_block_flushes_nothing():
    tf = ThinkingFilter()
    out = tf.feed("ok <|channel>thought\nnever ends") + tf.flush()
    assert out == "ok "


def test_thinking_filter_custom_markers():
    tf = ThinkingFilter()
    tf.THINK_START, tf.THINK_END = "<think>", "</think>"
    assert tf.feed("a<think>b</think>c") + tf.flush() == "ac"


# ─── clean_response ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("marker", ["<turn|>", "<|turn>", "<|im_end|>", "<|endoftext|>",
                                    "<|im_start|>", "<|end_of_text|>", "<|eot_id|>"])
def test_clean_response_truncates_at_stop_marker(marker):
    assert clean_response(f"Done.{marker}junk after") == "Done."


def test_clean_response_cuts_at_the_earliest_marker():
    # Llama 3 running past <|eot_id|> into the next header, then <|end_of_text|>.
    text = "Done.<|eot_id|><|start_header_id|>user<|end_header_id|>more<|end_of_text|>"
    assert clean_response(text) == "Done."


def test_clean_response_strips_python_tag():
    assert clean_response('<|python_tag|>{"a": 1}') == '{"a": 1}'


def test_clean_response_drops_thinking_preamble_before_code():
    text = "Thinking about it...\nmore musing\n```python\nprint(1)\n```"
    assert clean_response(text) == "```python\nprint(1)\n```"


def test_clean_response_keeps_tool_call_tags():
    text = '<tool_call>{"name": "Bash", "arguments": {}}</tool_call>'
    assert clean_response(text) == text
