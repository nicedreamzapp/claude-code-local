#!/usr/bin/env python3
"""
MLX Native Anthropic Server — Claude Code on Apple Silicon.

Single-file server: MLX inference + Anthropic Messages API + tool use support.
Converts Anthropic tool format <-> the model's native function calling format
(Gemma 4's `<|tool_call>call:Name{...}<tool_call|>`, Llama 3.3's raw-JSON
`{"type":"function",...}`, and the common HuggingFace `<tool_call>` JSON form
used by Qwen and others).

Pick a model from the lineup with the MLX_MODEL env var:
    MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx            (THE QUICK ONE — default, our own MLX upload)
    MLX_MODEL=divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx    (THE WISE ONE — our own MLX upload)
    MLX_MODEL=mlx-community/Qwen3.5-122B-A10B-4bit                       (THE BEAST)

NOTE FOR CONTRIBUTORS: this file is the source of truth. `setup.sh` installs it
at `~/.local/mlx-native-server/server.py` via a symlink, so edits here take
effect on the running server after a restart — no re-copying needed.
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MLX_MODEL", "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")
PORT = int(os.environ.get("MLX_PORT", "4000"))
KV_BITS = int(os.environ.get("MLX_KV_BITS", "0"))  # Gemma 4 RotatingKVCache doesn't support quantization
PREFILL_SIZE = int(os.environ.get("MLX_PREFILL_SIZE", "8192"))
# Pre-fill an empty thinking block to skip Gemma 4 reasoning chains entirely.
# Set MLX_SUPPRESS_THINKING=0 to disable (e.g. when you want reasoning output).
SUPPRESS_THINKING = os.environ.get("MLX_SUPPRESS_THINKING", "1") == "1"
DEFAULT_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "8192"))
KV_QUANT_START = int(os.environ.get("MLX_KV_QUANT_START", "256"))
MAX_TOOL_RETRIES = int(os.environ.get("MLX_TOOL_RETRIES", "2"))
# Browser mode: strip Claude Code bloat, keep only MCP tools
BROWSER_MODE = os.environ.get("MLX_BROWSER_MODE", "0") == "1"
# Code mode: auto-detect Claude Code coding sessions and replace the huge harness
# system prompt with a Llama-tuned one. Set MLX_CODE_MODE=0 to disable.
CODE_MODE_ENABLED = os.environ.get("MLX_CODE_MODE", "1") != "0"

# ─── Globals ─────────────────────────────────────────────────────────────────

model = None
tokenizer = None
generate_lock = threading.Lock()
# Prompt cache: reuse KV state across requests to avoid re-prefilling system+tools
_prompt_cache = None
_cached_token_prefix = None  # token IDs we've already prefilled


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─── Model Loading ───────────────────────────────────────────────────────────

GEMMA4_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% set ns = namespace(system='') %}"
    "{% for message in messages %}{% if message['role'] == 'system' %}{% set ns.system = message['content'] %}{% endif %}{% endfor %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|turn>user\n{% if ns.system and loop.first %}{{ ns.system }}\n\n{% endif %}{{ message['content'] }}<turn|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|turn>model\n{{ message['content'] }}<turn|>"
    "{% elif message['role'] == 'tool' %}"
    "<|turn>tool_response\n{{ message['content'] }}<turn|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)

def load_model():
    global model, tokenizer, KV_BITS
    log(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    # Fallback chat template if model doesn't provide one (Llama 3.3 has its own)
    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = GEMMA4_CHAT_TEMPLATE
        log("Injected Gemma 4 chat template")
    elapsed = time.time() - t0
    log(f"Model loaded in {elapsed:.1f}s")

    # Safety net: Gemma uses sliding-window attention → RotatingKVCache, which
    # mlx-lm can't quantize yet ("RotatingKVCache Quantization NYI"). The
    # default for MLX_KV_BITS is already 0, but if a user explicitly sets it to
    # 8 and happens to be running Gemma, auto-disable it so inference doesn't
    # 500 on the first call. (Credit: asdmoment, PR #7.)
    if KV_BITS and "gemma" in MODEL_PATH.lower():
        log("Gemma detected: disabling KV cache quantization (RotatingKVCache NYI)")
        KV_BITS = 0

    log(f"KV cache quantization: {KV_BITS}-bit" if KV_BITS else "KV cache: full precision")


# ─── Think Tag Stripping ────────────────────────────────────────────────────

def strip_think_tags(text):
    """Remove thinking blocks from model reasoning output."""
    # Standard <think>...</think>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'</think>', '', cleaned).strip()
    # Gemma 4 thinking: <|channel>thought\n...<channel|>
    cleaned = re.sub(r'<\|channel>thought\n.*?<channel\|>', '', cleaned, flags=re.DOTALL).strip()
    # Empty tool_call blocks
    cleaned = re.sub(r'<tool_call>\s*</tool_call>', '', cleaned).strip()
    return cleaned if cleaned else text


class ThinkingFilter:
    """Real-time filter that removes Gemma 4 thinking blocks from the mlx_lm token stream.

    Applied token-by-token inside the stream_generate loop so thinking content is
    discarded as it arrives rather than accumulated and regex-stripped after the fact.
    Works independently of HTTP SSE — the SSE layer (send_anthropic_stream) operates
    on the already-cleaned result returned by generate_response.
    """
    THINK_START = "<|channel>thought\n"
    THINK_END = "<channel|>"

    def __init__(self):
        self.in_thinking = False
        self.buf = ""

    def feed(self, chunk):
        self.buf += chunk
        output = ""
        while True:
            if self.in_thinking:
                idx = self.buf.find(self.THINK_END)
                if idx == -1:
                    safe = max(0, len(self.buf) - len(self.THINK_END) + 1)
                    self.buf = self.buf[safe:]
                    break
                self.in_thinking = False
                self.buf = self.buf[idx + len(self.THINK_END):]
            else:
                idx = self.buf.find(self.THINK_START)
                if idx == -1:
                    safe = max(0, len(self.buf) - len(self.THINK_START) + 1)
                    output += self.buf[:safe]
                    self.buf = self.buf[safe:]
                    break
                output += self.buf[:idx]
                self.in_thinking = True
                self.buf = self.buf[idx + len(self.THINK_START):]
        return output

    def flush(self):
        return "" if self.in_thinking else self.buf


def clean_response(text):
    """Strip think tags, stop tokens, and reasoning artifacts (but preserve tool_call tags)."""
    text = strip_think_tags(text)
    # Llama 3.x: strip function-call prefix token
    text = text.replace('<|python_tag|>', '').strip()
    # Gemma 4 + Qwen + ChatML: truncate at end-of-turn or start of a new turn
    for stop_marker in ['<turn|>', '<|turn>', '<|im_end|>', '<|endoftext|>', '<|im_start|>', '<|end_of_text|>', '<|eot_id|>']:
        if stop_marker in text:
            text = text[:text.index(stop_marker)].strip()
            break

    # Remove reasoning preamble if present
    if text.lstrip().startswith("Thinking"):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            s = line.strip()
            if any(s.startswith(p) for p in ['```', 'def ', 'class ', 'function ', 'import ', '#', '//', '<tool_call>']):
                return '\n'.join(lines[i:])

    return text


# ─── Tool Conversion ────────────────────────────────────────────────────────

def convert_tools_for_llm(anthropic_tools):
    """Convert Anthropic tool definitions to HuggingFace/Llama format."""
    if not anthropic_tools:
        return None
    llm_tools = []
    for tool in anthropic_tools:
        llm_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            }
        })
    return llm_tools


def format_tools_as_text(tools):
    """Format tools as text for system prompt (fallback if chat template doesn't support tools param)."""
    lines = ["# Available Tools\n"]
    lines.append("CRITICAL: You MUST call tools using EXACTLY this JSON format inside <tool_call> tags:")
    lines.append("")
    lines.append('<tool_call>')
    lines.append('{"name": "Bash", "arguments": {"command": "ls -la"}}')
    lines.append('</tool_call>')
    lines.append("")
    lines.append("RULES:")
    lines.append('- The content inside <tool_call> MUST be valid JSON with "name" and "arguments" keys')
    lines.append('- Do NOT use <parameter=...> tags inside <tool_call> — use the "arguments" JSON object')
    lines.append('- Do NOT mix XML and JSON — use ONLY pure JSON inside the tags')
    lines.append("- You may call multiple tools by using multiple <tool_call> blocks")
    lines.append("- Output any reasoning text BEFORE the tool calls, not inside them")
    lines.append("")
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"## {name}")
        if desc:
            lines.append(f"{desc}")
        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            for pname, pdef in props.items():
                req = " (required)" if pname in required else ""
                ptype = pdef.get("type", "any")
                pdesc = pdef.get("description", "")
                lines.append(f"  - {pname}: {ptype}{req} — {pdesc}")
        lines.append("")
    return "\n".join(lines)


def recover_garbled_tool_json(content, original_text=""):
    """Attempt to recover tool name and arguments from garbled JSON inside <tool_call> tags.

    Models sometimes produce hybrid XML/JSON like:
      {"name": "Bash", "parameter=command>cd ~/Desktop && rm -rf ...
      {"name": "Bash", "<parameter_commands>["rm -rf ...
      {"name": "Edit", "parameter=file_path>/some/path</parameter...
    """
    # Extract tool name
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if not name_match:
        return None
    tool_name = name_match.group(1)

    arguments = {}

    # Pattern A: "parameter=key>value" (most common garble)
    # Matches: "parameter=command>cd ~/Desktop..." or parameter=command>value</parameter>
    param_a = re.finditer(r'["\s,]?parameter=(\w+)>\s*(.*?)(?:</parameter>|$)', content, re.DOTALL)
    for m in param_a:
        key = m.group(1)
        val = m.group(2).strip().rstrip('"}\n')
        if val:
            arguments[key] = val

    # Pattern B: "<parameter_key>value" or "<parameter_key>["value"]"
    if not arguments:
        param_b = re.finditer(r'<parameter[_=](\w+)>\s*(.*?)(?:</parameter|<|$)', content, re.DOTALL)
        for m in param_b:
            key = m.group(1)
            val = m.group(2).strip().strip('[]"')
            if val:
                arguments[key] = val

    # Pattern C: "arguments" key exists but is malformed — try to extract the value after it
    if not arguments:
        args_match = re.search(r'"arguments"\s*:\s*\{(.*)', content, re.DOTALL)
        if args_match:
            raw = args_match.group(1)
            # Try to find key-value pairs
            kv_matches = re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
            for m in kv_matches:
                arguments[m.group(1)] = m.group(2)

    # Pattern D: single-argument tools — if we have a tool name and leftover text, use it
    # Common for Bash (command), Read (file_path), etc.
    if not arguments:
        single_arg_tools = {
            "Bash": "command", "Read": "file_path", "Write": "file_path",
            "Glob": "pattern", "Grep": "pattern",
        }
        if tool_name in single_arg_tools:
            # Everything after the tool name declaration is likely the argument value
            after_name = content[name_match.end():]
            # Strip JSON noise
            val = re.sub(r'^[\s,":{}]+', '', after_name)
            val = re.sub(r'[\s"}]+$', '', val)
            # Remove parameter= prefix if present
            val = re.sub(r'^parameter=\w+>\s*', '', val)
            val = re.sub(r'^<parameter[_=]\w+>\s*', '', val)
            if val and len(val) > 2:
                arguments[single_arg_tools[tool_name]] = val

    if arguments:
        log(f"  Recovered garbled tool call: {tool_name} with {list(arguments.keys())}")
        return {"name": tool_name, "arguments": arguments}

    return None


def parse_tool_calls(text):
    """Parse tool calls from generated text. Handles multiple formats including
    Gemma 4 native format. Returns (list of tool calls, remaining text).
    """
    tool_calls = []

    # Format 0: Gemma 4 native — <|tool_call>call:Name{key:<|"|>val<|"|>}<tool_call|>
    # Parse BEFORE replacing escape tokens — use <|"|> as reliable value delimiters
    gemma4_pattern = r'<\|tool_call>(.*?)<tool_call\|>'
    gemma4_matches = list(re.finditer(gemma4_pattern, text, re.DOTALL))
    if gemma4_matches:
        remaining = text
        for match in gemma4_matches:
            remaining = remaining.replace(match.group(0), "", 1)
            content = match.group(1).strip()
            name_m = re.match(r'call:([\w.]+)\{(.*)\}', content, re.DOTALL)
            if not name_m:
                log(f"  Gemma4 no name match: {content[:80]}")
                continue
            name = name_m.group(1)
            args_str = name_m.group(2)
            arguments = {}
            # Primary: extract key:<|"|>value<|"|> pairs (handles embedded quotes)
            for km in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', args_str, re.DOTALL):
                arguments[km.group(1)] = km.group(2)
            # Fallback: unquoted values (numbers, simple strings)
            if not arguments:
                for km in re.finditer(r'(\w+):([^,}]+)', args_str):
                    val = km.group(2).strip().strip('"\'')
                    arguments[km.group(1)] = val
            if arguments:
                tool_calls.append({"name": name, "arguments": arguments})
                log(f"  Gemma4 tool call: {name}({list(arguments.keys())})")
            else:
                log(f"  Gemma4 no args parsed: {content[:80]}")
        if tool_calls:
            return tool_calls, remaining.strip()

    remaining = text

    # Format 0.5: Llama 3.3 raw JSON — {"type":"function","name":"...","parameters":{...}}
    # Use json.JSONDecoder.raw_decode for robust nested JSON parsing
    _decoder = json.JSONDecoder()
    _pos = 0
    while _pos < len(text):
        idx = text.find('{"type"', _pos)
        if idx == -1:
            break
        try:
            obj, end_pos = _decoder.raw_decode(text, idx)
            if obj.get("type") == "function" and "name" in obj:
                name = obj["name"]
                arguments = obj.get("parameters", {})
                tool_calls.append({"name": name, "arguments": arguments})
                remaining = remaining.replace(text[idx:end_pos], "", 1)
                log(f"  Llama tool call: {name}({list(arguments.keys())})")
            _pos = end_pos
        except json.JSONDecodeError:
            _pos = idx + 1
    if tool_calls:
        return tool_calls, remaining.strip()

    # Format 1: <tool_call>{"name": "x", "arguments": {...}}</tool_call>
    pattern1 = r'<tool_call>\s*(.*?)\s*</tool_call>'
    for match in re.finditer(pattern1, text, re.DOTALL):
        content = match.group(1).strip()
        remaining = remaining.replace(match.group(0), "", 1)
        if not content:
            continue
        try:
            call_data = json.loads(content)
            tool_calls.append({
                "name": call_data.get("name", ""),
                "arguments": call_data.get("arguments", {}),
            })
        except json.JSONDecodeError:
            # The model often puts Format 2 (<function=X><parameter=Y>...</parameter></function>)
            # inside <tool_call> tags. Handle that first.
            func_in_tag = re.search(r'<function=([\w.-]+)>(.*)', content, re.DOTALL)
            if func_in_tag:
                fname = func_in_tag.group(1)
                params_text = func_in_tag.group(2)
                arguments = {}
                for pmatch in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*(?:</parameter>|$)', params_text, re.DOTALL):
                    arguments[pmatch.group(1)] = pmatch.group(2).strip()
                if arguments:
                    tool_calls.append({"name": fname, "arguments": arguments})
                    log(f"  Recovered function-in-tag: {fname}")
                else:
                    log(f"  Warning: function-in-tag but no params: {content[:100]}")
            else:
                # Try general garbled recovery
                recovered = recover_garbled_tool_json(content, text)
                if recovered:
                    tool_calls.append(recovered)
                else:
                    log(f"  Warning: unrecoverable tool_call JSON: {content[:100]}")

    # Format 2: <function=name><parameter=key>value</parameter>...</function>
    if not tool_calls:
        pattern2 = r'<function=([\w.-]+)>(.*?)</function>'
        for match in re.finditer(pattern2, text, re.DOTALL):
            func_name = match.group(1)
            params_text = match.group(2)
            arguments = {}
            for pmatch in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_text, re.DOTALL):
                arguments[pmatch.group(1)] = pmatch.group(2)
            tool_calls.append({"name": func_name, "arguments": arguments})
            remaining = remaining.replace(match.group(0), "", 1)

    # Format 3: <|tool_call|>...<|/tool_call|> (some model versions)
    if not tool_calls:
        pattern3 = r'<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>'
        for match in re.finditer(pattern3, text, re.DOTALL):
            remaining = remaining.replace(match.group(0), "", 1)
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append({
                    "name": call_data.get("name", ""),
                    "arguments": call_data.get("arguments", {}),
                })
            except json.JSONDecodeError:
                recovered = recover_garbled_tool_json(match.group(1))
                if recovered:
                    tool_calls.append(recovered)

    # Format 3.5: Qwen 2.5 — <tools>{"name": "x", "arguments": {...}}</tools>
    # Some Qwen 2.5 fine-tunes emit a <tools> wrapper instead of <tool_call>.
    if not tool_calls:
        pattern_qwen25 = r'<tools>\s*(.*?)\s*</tools>'
        for match in re.finditer(pattern_qwen25, text, re.DOTALL):
            content = match.group(1).strip()
            remaining = remaining.replace(match.group(0), "", 1)
            if not content:
                continue
            try:
                call_data = json.loads(content)
                if isinstance(call_data, list):
                    for cd in call_data:
                        if isinstance(cd, dict) and "name" in cd:
                            tool_calls.append({
                                "name": cd.get("name", ""),
                                "arguments": cd.get("arguments", cd.get("parameters", {})),
                            })
                elif isinstance(call_data, dict) and "name" in call_data:
                    tool_calls.append({
                        "name": call_data.get("name", ""),
                        "arguments": call_data.get("arguments", call_data.get("parameters", {})),
                    })
            except json.JSONDecodeError:
                recovered = recover_garbled_tool_json(content, text)
                if recovered:
                    tool_calls.append(recovered)
                else:
                    log(f"  Warning: unrecoverable <tools> JSON: {content[:100]}")

    # Format 3.6: JSON inside markdown code block — ```json\n{"name":..,"arguments":..}\n```
    # Some Qwen 2.5 fine-tunes emit tool calls as a fenced code block when the
    # system prompt / chat template doesn't explicitly tell them to use the
    # native <tool_call> wrapper. Two sub-cases supported:
    #   (a) single JSON object inside the fence
    #   (b) multiple JSON objects back-to-back inside the same fence (no
    #       array, no commas) — the model expressing several sequential
    #       tool calls in one go. Use raw_decode in a loop to extract them.
    if not tool_calls:
        pattern_md = r'```(?:json|tool[_ ]?call)?\s*\n?(.*?)\s*\n?```'
        decoder_md = json.JSONDecoder()
        for match in re.finditer(pattern_md, text, re.DOTALL):
            content = match.group(1).strip()
            if not content.lstrip().startswith("{"):
                continue
            extracted_in_block = []
            pos = 0
            while pos < len(content):
                # Skip whitespace and stray commas between objects
                while pos < len(content) and content[pos] in " \t\n\r,":
                    pos += 1
                if pos >= len(content):
                    break
                if content[pos] != "{":
                    break
                try:
                    obj, end_pos = decoder_md.raw_decode(content, pos)
                except json.JSONDecodeError:
                    break
                pos = end_pos
                if not isinstance(obj, dict):
                    continue
                name = obj.get("name") or obj.get("tool")
                args = (
                    obj.get("arguments")
                    or obj.get("parameters")
                    or obj.get("args")
                    or obj.get("input")
                )
                if name and isinstance(args, dict):
                    extracted_in_block.append({"name": name, "arguments": args})
            if extracted_in_block:
                tool_calls.extend(extracted_in_block)
                remaining = remaining.replace(match.group(0), "", 1)
                log(
                    f"  Markdown-fenced tool calls: "
                    f"{', '.join(t['name'] for t in extracted_in_block)}"
                )

    # Format 4: Garbled — no tags at all, but parameter= patterns in raw text
    if not tool_calls:
        # Look for any tool name followed by parameter patterns
        tool_names_pattern = r'(?:mcp__[\w.-]+|Bash|Read|Write|Edit|Glob|Grep)'
        name_match = re.search(rf'"?name"?\s*[:=]\s*"?({tool_names_pattern})"?', text)
        param_matches = list(re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', text, re.DOTALL))

        if name_match and param_matches:
            arguments = {}
            for pm in param_matches:
                arguments[pm.group(1)] = pm.group(2)
            tool_calls.append({"name": name_match.group(1), "arguments": arguments})
            remaining = text[:name_match.start()].strip()
            log(f"  Recovered tagless tool call: {name_match.group(1)}")
        elif param_matches:
            # We have parameters but no name — try to infer from param keys
            arguments = {}
            for pm in param_matches:
                arguments[pm.group(1)] = pm.group(2)
            if "command" in arguments:
                tool_calls.append({"name": "Bash", "arguments": arguments})
                log(f"  Inferred Bash tool call from 'command' parameter")
            elif "file_path" in arguments:
                tool_calls.append({"name": "Read", "arguments": arguments})
                log(f"  Inferred Read tool call from 'file_path' parameter")
            elif "pattern" in arguments:
                tool_calls.append({"name": "Glob", "arguments": arguments})
                log(f"  Inferred Glob tool call from 'pattern' parameter")
            if tool_calls:
                remaining = text[:param_matches[0].start()].strip()

    # Deduplicate tool calls (model sometimes emits same call in multiple formats)
    seen = set()
    deduped = []
    for tc in tool_calls:
        key = tc["name"]
        if key not in seen:
            seen.add(key)
            deduped.append(tc)
        else:
            log(f"  Deduped: {key}")
    tool_calls = deduped

    # Clean remaining text: strip any leftover <function=...> or <tool_call> fragments
    remaining = re.sub(r'<function=[\w.-]+>.*?</function>', '', remaining, flags=re.DOTALL)
    remaining = re.sub(r'</?tool_call>', '', remaining)
    remaining = remaining.strip()

    return tool_calls, remaining


# ─── Anthropic Message Conversion ───────────────────────────────────────────

def convert_messages(body):
    """Convert Anthropic Messages format to MLX chat messages.

    Handles:
    - Text messages (passthrough)
    - Assistant messages with tool_use blocks → <tool_call> format
    - User messages with tool_result blocks → role="tool" messages
    """
    messages = []

    # System prompt
    if body.get("system"):
        sys_text = body["system"]
        if isinstance(sys_text, list):
            sys_text = "\n".join(b.get("text", "") for b in sys_text if b.get("type") == "text")
        messages.append({"role": "system", "content": sys_text})

    # Conversation messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Simple string content
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        # List of content blocks
        if isinstance(content, list):
            text_parts = []
            tool_use_parts = []
            tool_result_parts = []

            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_use_parts.append(block)
                elif btype == "tool_result":
                    tool_result_parts.append(block)

            # Assistant message with tool_use blocks → convert to LLM format
            if role == "assistant" and tool_use_parts:
                content_str = ""
                if text_parts:
                    content_str = "\n".join(p for p in text_parts if p)
                for tu in tool_use_parts:
                    call_json = json.dumps({
                        "name": tu.get("name", ""),
                        "arguments": tu.get("input", {})
                    }, ensure_ascii=False)
                    content_str += f"\n<tool_call>\n{call_json}\n</tool_call>"
                messages.append({"role": "assistant", "content": content_str.strip()})

            # User message with tool_result blocks → split into tool messages
            elif tool_result_parts:
                # Add any text from the user first
                if text_parts:
                    text = "\n".join(p for p in text_parts if p)
                    if text.strip():
                        messages.append({"role": "user", "content": text})

                # Each tool_result becomes a "tool" role message
                for tr in tool_result_parts:
                    result_content = tr.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            b.get("text", str(b)) for b in result_content
                        )
                    elif not isinstance(result_content, str):
                        result_content = str(result_content)
                    # Include tool name context if we can find it
                    messages.append({"role": "tool", "content": result_content})

            # Regular message with just text
            else:
                text = "\n".join(p for p in text_parts if p)
                if text.strip():
                    messages.append({"role": role, "content": text})

    return messages


def tokenize_messages(messages, tools=None):
    """Apply chat template and tokenize, with optional tool definitions."""
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if tools:
        kwargs["tools"] = tools

    try:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
        if tools:
            log(f"  Tools: {len(tools)} tools passed via chat template")
        return token_ids
    except (TypeError, Exception) as e:
        # If tools param failed, try injecting into system prompt instead
        if tools:
            log(f"  Chat template tools param failed ({e}), injecting into system prompt")
            tool_text = format_tools_as_text(tools)
            msg_copy = [m.copy() for m in messages]
            if msg_copy and msg_copy[0]["role"] == "system":
                msg_copy[0]["content"] = msg_copy[0]["content"] + "\n\n" + tool_text
            else:
                msg_copy.insert(0, {"role": "system", "content": tool_text})

            try:
                return tokenizer.apply_chat_template(
                    msg_copy, add_generation_prompt=True, tokenize=True
                )
            except Exception:
                pass

        # Final fallback: plain text
        log("  Warning: using plain text fallback for tokenization")
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text += "\nassistant: "
        return tokenizer.encode(text)


# ─── Browser Mode Optimization ───────────────────────────────────────────────

BROWSER_SYSTEM_PROMPT = """You are a fast browser agent. You control a web browser via tools.

CORE RULES:
- ONLY use take_snapshot to see the page. NEVER use take_screenshot.
- take_snapshot returns a text DOM tree with uid attributes. Use these uids to click/fill.
- Chain actions quickly. Don't explain, just act.
- Navigate directly to URLs. Don't go to homepages first.

COMMENTING ON ARTICLES (Yahoo, news sites, etc.):
Comment boxes on most news sites are inside iframes that take_snapshot CANNOT see.
You MUST use evaluate_script to interact with comment boxes. Here is the exact process:

Step 1: Click the "Comments" button using its uid from the snapshot to open comments.
Step 2: Use evaluate_script to find and click the comment input inside the iframe:
  function: "() => { const frames = document.querySelectorAll('iframe'); for (const f of frames) { try { const doc = f.contentDocument || f.contentWindow.document; const el = doc.querySelector('[contenteditable=true], textarea, [role=textbox], .ow-comment-textarea, [data-spot-im-class=spcv_editor]'); if (el) { el.click(); el.focus(); return 'Found comment input in iframe'; } } catch(e) {} } return 'No comment input found'; }"

Step 3: Use evaluate_script to type your comment text into the focused element:
  function: "() => { const frames = document.querySelectorAll('iframe'); for (const f of frames) { try { const doc = f.contentDocument || f.contentWindow.document; const el = doc.querySelector('[contenteditable=true], textarea, [role=textbox], .ow-comment-textarea, [data-spot-im-class=spcv_editor]'); if (el) { el.focus(); el.innerText = 'YOUR COMMENT TEXT HERE'; el.dispatchEvent(new Event('input', {bubbles: true})); return 'Comment typed'; } } catch(e) {} } return 'Failed to type'; }"

Step 4: Do NOT click any Send/Post button. Leave the comment as a draft for the user to review.

IMPORTANT: Replace 'YOUR COMMENT TEXT HERE' with an actual thoughtful comment about the article.
The comment should be 2-3 sentences, relevant to the article content you read in the snapshot."""

# Only these tools are needed for browser control
BROWSER_TOOLS_ALLOW = {
    "mcp__chrome-devtools__navigate_page",
    "mcp__chrome-devtools__take_snapshot",
    "mcp__chrome-devtools__click",
    "mcp__chrome-devtools__fill",
    "mcp__chrome-devtools__type_text",
    "mcp__chrome-devtools__press_key",
    "mcp__chrome-devtools__evaluate_script",
    "mcp__chrome-devtools__select_page",
    "mcp__chrome-devtools__list_pages",
}

def looks_like_claude_code_browser_session(body):
    """A real Claude Code MCP browser session registers chrome-devtools tools.
    Direct clients (like ~/.local/browser-agent) bring their own system prompt
    and zero tools — we must NOT clobber those, or the model will call tools
    that don't exist on the client side."""
    tools = body.get("tools", [])
    return any(t.get("name", "") in BROWSER_TOOLS_ALLOW for t in tools)


def optimize_for_browser(body):
    """Strip Claude Code bloat: replace system prompt, keep only essential MCP tools.

    Only fires for actual Claude Code MCP browser sessions. Direct clients that
    bring their own system prompt + tool contract are passed through untouched.
    """
    if not looks_like_claude_code_browser_session(body):
        log("  Browser mode: passthrough (direct client, not Claude Code MCP)")
        return body

    # Replace massive system prompt with compact browser prompt
    body["system"] = BROWSER_SYSTEM_PROMPT

    # Filter tools to only essential chrome-devtools tools (no screenshot!)
    tools = body.get("tools", [])
    browser_tools = [t for t in tools if t.get("name", "") in BROWSER_TOOLS_ALLOW]
    if browser_tools:
        body["tools"] = browser_tools
        log(f"  Browser mode: {len(tools)} tools → {len(browser_tools)}")

    return body


# ─── Code Mode Optimization ──────────────────────────────────────────────────
#
# Claude Code's harness sends a ~10K-token system prompt and 30+ tools, all
# tuned for Claude. Open models (Llama, Qwen, etc.) get confused inside that
# wall of context and emit stock refusals like "I am not able to execute this
# task as it exceeds the limitations of the functions I have been given."
#
# This mode auto-detects Claude Code coding sessions (any of Bash/Read/Edit/
# Write/Grep/Glob in the tool list) and replaces the harness with a slim
# Llama-friendly prompt + filtered tool list. Browser mode takes priority.

CODE_SYSTEM_PROMPT = """You are a local coding assistant running on the user's Mac via MLX. You help with software engineering tasks: reading code, editing files, running shell commands, and searching codebases.

You have these tools available:
- Bash: run a shell command
- Read: read a file from disk (use absolute paths)
- Edit: replace exact text in an existing file
- Write: create a new file
- Grep: search file contents (ripgrep)
- Glob: find files by name pattern

RULES:
- Be concise. Skip preamble. Do the work, then give a brief result.
- Greetings, small talk, or questions about yourself: respond in plain text with NO tool calls.
- For real tasks: read files before editing them, use absolute paths, batch independent tool calls in parallel.
- NEVER say "I am not able to execute this task" or "this exceeds my limitations" — you have full tool access on this machine. If a request is genuinely unclear, ask one short clarifying question instead of refusing.
- When you call a tool, use the <tool_call> JSON format exactly as instructed. Do not wrap it in markdown."""

# Built-in Claude Code tools that signal a coding session and are worth keeping.
CODE_TOOLS_ALLOW = {
    "Bash",
    "Read",
    "Edit",
    "Write",
    "Grep",
    "Glob",
}

def looks_like_code_session(body):
    """Heuristic: if any of Claude Code's core file/shell tools are present,
    treat this as a coding session and apply the slim prompt."""
    tools = body.get("tools", [])
    tool_names = {t.get("name", "") for t in tools}
    return bool(tool_names & CODE_TOOLS_ALLOW)


def slim_tool(tool):
    """Strip verbose Claude Code descriptions from a tool, keeping only name + param names.

    The chat template serializes full descriptions as JSON, ballooning prompts to 5000+
    tokens for just 4 tools. Slimming cuts that to ~150 tokens while preserving tool-call
    functionality (the model already knows what Bash/Read/Edit/Write do from the system prompt).
    """
    schema = tool.get("input_schema", {})
    props = schema.get("properties", {})
    slim_props = {k: {"type": v.get("type", "string")} for k, v in props.items()}
    slim_schema = {"type": "object", "properties": slim_props}
    if schema.get("required"):
        slim_schema["required"] = schema["required"]
    return {"name": tool["name"], "description": tool["name"], "input_schema": slim_schema}


def optimize_for_code(body):
    """Strip Claude Code bloat: replace the 10K-token harness prompt with a
    compact coding prompt and slim tool definitions to ~150 tokens total."""
    body["system"] = CODE_SYSTEM_PROMPT

    tools = body.get("tools", [])
    code_tools = [slim_tool(t) for t in tools if t.get("name", "") in CODE_TOOLS_ALLOW]
    if code_tools:
        stripped = len(tools) - len(code_tools)
        body["tools"] = code_tools
        log(f"  Code mode: {len(tools)} tools → {len(code_tools)} (stripped {stripped}, descriptions slimmed)")

    return body


# ─── Generation ──────────────────────────────────────────────────────────────

_first_request = True

def generate_response(body):
    """Run MLX inference and return Anthropic-formatted response."""
    global _first_request

    # In browser mode, strip Claude Code bloat before inference.
    # Otherwise, auto-detect Claude Code coding sessions and apply code mode.
    if BROWSER_MODE:
        body = optimize_for_browser(body)
    elif CODE_MODE_ENABLED and looks_like_code_session(body):
        body = optimize_for_code(body)

    # Opt-in: append a project-specific system prompt to whatever the
    # current mode produced. Used by Narrative Gemma to inject narration
    # rules without rewriting the whole code/browser prompt.
    extra_path = os.environ.get("MLX_APPEND_SYSTEM_PROMPT_FILE")
    if extra_path and os.path.exists(extra_path):
        try:
            with open(extra_path) as ef:
                extra = ef.read().strip()
            if extra:
                current = body.get("system", "")
                if isinstance(current, list):
                    body["system"] = current + [
                        {"type": "text", "text": "\n\n---\n\n" + extra}
                    ]
                else:
                    body["system"] = (current or "") + "\n\n---\n\n" + extra
                log(f"  Appended {len(extra)} chars from MLX_APPEND_SYSTEM_PROMPT_FILE")
        except Exception as _e:
            log(f"  Failed to append extra system prompt: {_e}")

    if _first_request:
        _first_request = False
        # Dump tool names and system prompt length for debugging
        tools = body.get("tools", [])
        tool_names = [t.get("name", "?") for t in tools]
        sys_prompt = body.get("system", "")
        if isinstance(sys_prompt, list):
            sys_len = sum(len(b.get("text", "")) for b in sys_prompt)
        else:
            sys_len = len(sys_prompt)
        log(f"  [FIRST REQUEST] tools={len(tools)} names={tool_names}")
        log(f"  [FIRST REQUEST] system_prompt_len={sys_len}")
        # Dump first 500 chars of system prompt to see if MCP tools are described there
        sys_text = sys_prompt if isinstance(sys_prompt, str) else str(sys_prompt)[:500]
        log(f"  [FIRST REQUEST] system_start={sys_text[:300]}")

    # Convert tools from Anthropic → MLX format
    anthropic_tools = body.get("tools", [])
    llm_tools = convert_tools_for_llm(anthropic_tools) if anthropic_tools else None

    messages = convert_messages(body)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.2)

    if llm_tools:
        log(f"  Tools: {len(llm_tools)} ({', '.join(t['function']['name'] for t in llm_tools[:5])}{'...' if len(llm_tools) > 5 else ''})")

    # Tokenize (with tools if present)
    token_ids = tokenize_messages(messages, tools=llm_tools)

    # Pre-fill an empty thinking block so the model skips its reasoning chain entirely.
    # Gemma 4 generates 300-500 thinking tokens per request by default — this cuts them out.
    if SUPPRESS_THINKING and "gemma" in MODEL_PATH.lower():
        skip = tokenizer.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
        token_ids = list(token_ids) + list(skip)
        log(f"  Thinking suppressed (+{len(skip)} prefill tokens)")

    prompt_tokens = len(token_ids)
    log(f"  Prompt: {prompt_tokens} tokens")

    # ─── Prompt cache: reuse KV for shared prefix tokens ───
    global _prompt_cache, _cached_token_prefix

    # Check if cache type supports safe trim+reuse (standard KVCache only,
    # RotatingKVCache from Gemma 4 has a circular buffer that breaks on trim+extend)
    from mlx_lm.models.cache import RotatingKVCache
    cache_is_safe = _prompt_cache is not None and not isinstance(_prompt_cache[0], RotatingKVCache)

    # Find how many leading tokens match the previous request's prompt
    cache_hit_len = 0
    if cache_is_safe and _cached_token_prefix is not None:
        max_check = min(len(token_ids), len(_cached_token_prefix))
        for i in range(max_check):
            if token_ids[i] == _cached_token_prefix[i]:
                cache_hit_len = i + 1
            else:
                break

    # Always leave at least 1 token to prefill — mlx_lm.stream_generate raises
    # ValueError if the prompt is empty (happens when new prompt == cached prefix)
    if cache_hit_len >= len(token_ids):
        cache_hit_len = len(token_ids) - 1

    if cache_hit_len > 0:
        # Trim cache back to the shared prefix, then only prefill the delta
        #cache_offset = _prompt_cache[0].offset  # total tokens in cache (prompt + gen)
        # Try .step instead of .offset
        cache_offset = _prompt_cache[0].step if hasattr(_prompt_cache[0], 'step') else 0
        trim_amount = cache_offset - cache_hit_len
        if trim_amount > 0:
            for c in _prompt_cache:
                c.trim(trim_amount)
        delta_tokens = token_ids[cache_hit_len:]
        new_tokens = len(delta_tokens)
        log(f"  Cache hit: {cache_hit_len} reused, {new_tokens} new tokens to prefill (saved {cache_hit_len} tokens)")
        # Feed only the new tokens, with the existing cache
        prompt_for_gen = delta_tokens
    else:
        if _prompt_cache is not None and isinstance(_prompt_cache[0], RotatingKVCache):
            log(f"  RotatingKVCache: fresh cache each request (no trim support)")
        else:
            log(f"  Cache miss: full prefill of {prompt_tokens} tokens")
        _prompt_cache = None
        prompt_for_gen = token_ids

    # Build generation kwargs — always pass a prompt_cache so we can reuse it
    if _prompt_cache is None:
        _prompt_cache = make_prompt_cache(model)
        log(f"  Created new prompt cache ({len(_prompt_cache)} layers)")
    gen_kwargs = {
        "prefill_step_size": PREFILL_SIZE,
        "prompt_cache": _prompt_cache,
    }
    if KV_BITS:
        gen_kwargs["kv_bits"] = KV_BITS
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = KV_QUANT_START

    if temperature > 0:
        gen_kwargs["sampler"] = make_sampler(temp=temperature)
    else:
        gen_kwargs["sampler"] = make_sampler(temp=0.0)

    # Generate — ThinkingFilter removes Gemma 4 thinking blocks in real-time
    # so clean_response never sees them (more robust than regex post-hoc).
    tf = ThinkingFilter()
    full_text = ""
    gen_tokens = 0
    finish_reason = "end_turn"
    t0 = time.time()

    with generate_lock:
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_for_gen,
            max_tokens=max_tokens,
            **gen_kwargs,
        ):
            full_text += tf.feed(response.text)
            gen_tokens = response.generation_tokens
            if response.finish_reason == "length":
                finish_reason = "max_tokens"
            elif response.finish_reason == "stop":
                finish_reason = "end_turn"

    full_text += tf.flush()

    # Cache is updated in-place by MLX — save the token prefix for next request's diff
    _cached_token_prefix = token_ids

    elapsed = time.time() - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generated: {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    # Clean output (preserves <tool_call> tags)
    text = clean_response(full_text)

    # Parse tool calls from model output
    tool_calls, remaining_text = parse_tool_calls(text)

    # ─── Retry logic: if model expressed intent to use a tool but we got no valid calls ───
    tool_intent_phrases = [
        "here's the command", "bash(", "read(", "edit(", "write(",
        "<tool_call>", "<function=", "<tools>", '"name":', '"arguments":',
        '"parameters":', "```json", "```tool",
    ]
    if not tool_calls and any(p in remaining_text.lower() for p in tool_intent_phrases):
        log(f"  Tool intent detected but parser missed it. Raw text (400 chars): {remaining_text[:400]!r}")
        for retry in range(MAX_TOOL_RETRIES):
            log(f"  Retry {retry+1}/{MAX_TOOL_RETRIES}: tool intent detected but no valid tool call, re-prompting")
            retry_messages = messages + [
                {"role": "assistant", "content": full_text},
                {"role": "user", "content": (
                    "Your previous response tried to call a tool but the format was wrong. "
                    "Please call the tool now using EXACTLY this format:\n"
                    '<tool_call>\n{"name": "TOOL_NAME", "arguments": {"param": "value"}}\n</tool_call>\n'
                    "Do NOT use <parameter=...> tags inside tool_call. Use pure JSON with \"arguments\" key."
                )}
            ]
            retry_tokens = tokenize_messages(retry_messages, tools=llm_tools)
            log(f"  Retry prompt: {len(retry_tokens)} tokens")

            retry_text = ""
            retry_gen = 0
            retry_tf = ThinkingFilter()
            with generate_lock:
                for response in stream_generate(
                    model=model, tokenizer=tokenizer, prompt=retry_tokens,
                    max_tokens=max_tokens, **gen_kwargs,
                ):
                    retry_text += retry_tf.feed(response.text)
                    retry_gen = response.generation_tokens
            retry_text += retry_tf.flush()

            retry_text = clean_response(retry_text)
            retry_calls, retry_remaining = parse_tool_calls(retry_text)
            gen_tokens += retry_gen

            if retry_calls:
                tool_calls = retry_calls
                # Preserve original reasoning text, not retry text
                log(f"  Retry succeeded: {', '.join(tc['name'] for tc in retry_calls)}")
                break
            else:
                log(f"  Retry {retry+1} failed, still no valid tool call")

    # Build content blocks
    content_blocks = []

    if remaining_text.strip():
        content_blocks.append({"type": "text", "text": remaining_text.strip()})

    if tool_calls:
        # NOTE: previously this branch prepended an empty text block
        # ({"type":"text","text":""}) "because Anthropic requires at least one
        # block". That's incorrect — the Anthropic API accepts a content list
        # made up of only tool_use blocks. The empty text block actively breaks
        # Claude Code 2.1: it reads the first (empty) text block, decides the
        # response is "(No output)", and silently discards the tool_use blocks
        # that follow. Drop the empty text block entirely.

        # Build a {tool_name: allowed_input_keys} map from the request schema
        # so we can drop any extra keys the model hallucinated. Claude Code
        # 2.1 silently rejects a tool_use response whose input has fields not
        # declared in the tool's input_schema (e.g. Qwen 2.5 likes to add a
        # "description" field next to "command" for Bash).
        # Note: anthropic_tools here is the (possibly filtered) tool list
        # the request actually sent — we must rebuild from `body["tools"]`
        # to honor any code-mode trimming applied earlier in this function.
        current_tools = body.get("tools", []) or anthropic_tools or []
        tool_schemas = {}
        for atool in current_tools:
            schema = atool.get("input_schema") or {}
            props = schema.get("properties") or {}
            tool_schemas[atool.get("name", "")] = set(props.keys())
        if os.environ.get("MLX_DEBUG_RESPONSE"):
            log(f"  TOOL SCHEMAS: {tool_schemas}")

        for tc in tool_calls:
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            raw_input = tc["arguments"] if isinstance(tc.get("arguments"), dict) else {}
            allowed = tool_schemas.get(tc["name"])
            if allowed is not None:
                filtered = {k: v for k, v in raw_input.items() if k in allowed}
                dropped = [k for k in raw_input if k not in allowed]
                if dropped:
                    log(f"  Filtered extra input keys for {tc['name']}: {dropped}")
                tool_input = filtered
            else:
                tool_input = raw_input
            content_blocks.append({
                "type": "tool_use",
                "id": tool_id,
                "name": tc["name"],
                "input": tool_input,
            })
        finish_reason = "tool_use"
        log(f"  Tool calls: {', '.join(tc['name'] for tc in tool_calls)}")

    if not content_blocks:
        content_blocks.append({"type": "text", "text": "(No output)"})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "claude-sonnet-4-6"),
        "content": content_blocks,
        "stop_reason": finish_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": gen_tokens,
        }
    }


# ─── HTTP Handler ────────────────────────────────────────────────────────────

def send_json(handler, status, data):
    resp = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", len(resp))
    handler.end_headers()
    handler.wfile.write(resp)


def send_anthropic_stream(handler, result):
    """Replay a fully-generated Anthropic Messages response as a stream of
    Server-Sent Events. Claude Code 2.1 sends `stream: true` for any request
    that involves tools and silently discards a non-streaming response, so
    we must emit the SSE event sequence even though we already have the
    complete answer.

    Event order (Anthropic Messages streaming spec):
      message_start
      for each content block i:
        content_block_start
        content_block_delta (text_delta for text, input_json_delta for tool_use)
        content_block_stop
      message_delta (with stop_reason)
      message_stop
    """
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    # We replay a fully-generated message in one shot, so signal close so
    # the client (Claude Code 2.1) doesn't keep waiting for more events.
    handler.send_header("Connection", "close")
    handler.end_headers()

    def emit(event_type, payload):
        handler.wfile.write(f"event: {event_type}\n".encode())
        handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        handler.wfile.flush()

    msg_start = {
        "type": "message_start",
        "message": {
            "id": result["id"],
            "type": "message",
            "role": "assistant",
            "model": result["model"],
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": result["usage"]["input_tokens"],
                "output_tokens": 0,
            },
        },
    }
    emit("message_start", msg_start)

    for idx, block in enumerate(result["content"]):
        btype = block.get("type")
        if btype == "text":
            emit("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })
            text = block.get("text", "")
            if text:
                emit("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": text},
                })
            emit("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })
        elif btype == "tool_use":
            emit("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                },
            })
            input_json = json.dumps(block.get("input", {}))
            emit("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            })
            emit("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })

    emit("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": result.get("stop_reason"),
            "stop_sequence": result.get("stop_sequence"),
        },
        "usage": {"output_tokens": result["usage"]["output_tokens"]},
    })
    emit("message_stop", {"type": "message_stop"})


def get_path(full_path):
    return urlparse(full_path).path


class AnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        log(f"HEAD {self.path}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        path = get_path(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(content_length) if content_length else b'{}'
        body = json.loads(raw)
        tools_count = len(body.get("tools", []))
        log(f"POST {self.path} model={body.get('model','-')} max_tokens={body.get('max_tokens','-')} tools={tools_count}")
        if os.environ.get("MLX_DEBUG_REQUEST"):
            try:
                msgs = body.get("messages", [])
                roles_summary = []
                for i, m in enumerate(msgs[-6:]):  # last 6 messages
                    role = m.get("role", "?")
                    content = m.get("content")
                    if isinstance(content, str):
                        roles_summary.append(f"[{role}] str({len(content)})")
                    elif isinstance(content, list):
                        types = [b.get("type", "?") for b in content if isinstance(b, dict)]
                        roles_summary.append(f"[{role}] {types}")
                log(f"  DEBUG msg trail (last 6/{len(msgs)}): {' | '.join(roles_summary)}")
            except Exception as e:
                log(f"  DEBUG dump failed: {e}")

        if path in ("/v1/messages", "/messages"):
            try:
                result = generate_response(body)
                # Log preview of first content block
                first = result["content"][0]
                if first["type"] == "text":
                    preview = first.get("text", "")[:80]
                    log(f"  ← OK ({result['usage']['output_tokens']} tok) {preview}...")
                elif first["type"] == "tool_use":
                    log(f"  ← OK ({result['usage']['output_tokens']} tok) [tool_use: {first['name']}]")
                if os.environ.get("MLX_DEBUG_RESPONSE"):
                    try:
                        log(f"  RESPONSE JSON: {json.dumps(result, ensure_ascii=False)[:1500]}")
                    except Exception as e:
                        log(f"  RESPONSE dump failed: {e}")

                if body.get("stream"):
                    send_anthropic_stream(self, result)
                else:
                    send_json(self, 200, result)
            except Exception as e:
                log(f"  ← ERROR: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                send_json(self, 500, {"error": {"type": "server_error", "message": str(e)}})
        else:
            log(f"  Unknown POST: {path}")
            send_json(self, 200, {})

    def do_GET(self):
        path = get_path(self.path)
        log(f"GET {self.path}")

        if path in ("/v1/models", "/models"):
            send_json(self, 200, {
                "object": "list",
                "data": [
                    {"id": "claude-opus-4-6", "object": "model", "created": int(time.time()), "owned_by": "local"},
                    {"id": "claude-sonnet-4-6", "object": "model", "created": int(time.time()), "owned_by": "local"},
                    {"id": "claude-haiku-4-5-20251001", "object": "model", "created": int(time.time()), "owned_by": "local"},
                ]
            })
        elif path == "/health":
            send_json(self, 200, {"status": "ok", "model": MODEL_PATH})
        else:
            send_json(self, 200, {})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  MLX Native Anthropic Server                    ║")
    print("║  Claude Code → MLX → Apple Silicon (direct)     ║")
    print("║  Tool use: enabled (Anthropic ↔ Llama native)   ║")
    print("║  Prompt caching: enabled (KV reuse)             ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    load_model()

    print()
    print(f"Serving Anthropic Messages API on http://localhost:{PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"KV cache: {KV_BITS}-bit quantization (start at token {KV_QUANT_START})" if KV_BITS else "KV cache: full precision")
    print(f"Prompt cache: enabled (KV reuse across requests)")
    print(f"Tool retry: up to {MAX_TOOL_RETRIES} retries on garbled tool calls")
    print()
    print("Claude Code config:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PORT}")
    print(f"  ANTHROPIC_API_KEY=sk-local")
    print()

    server = HTTPServer(("127.0.0.1", PORT), AnthropicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
