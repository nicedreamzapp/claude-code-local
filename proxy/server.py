#!/usr/bin/env python3
"""
MLX Native Anthropic Server — Claude Code on Apple Silicon.

Single-file server: MLX inference + Anthropic Messages API + KV cache quantization.
Supports tool use: converts Anthropic tool format <-> the model's native function
calling format (Qwen, Gemma, Llama all use the HuggingFace `<tool_call>` JSON convention).

Pick a model from the lineup with the MLX_MODEL env var:
    MLX_MODEL=mlx-community/Qwen3.5-122B-A10B-4bit                       (THE BEAST)
    MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx            (THE QUICK ONE — default, our own MLX upload)
    MLX_MODEL=divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx    (THE WISE ONE — our own MLX upload)
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

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MLX_MODEL", "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")
PORT = int(os.environ.get("MLX_PORT", "4000"))
KV_BITS = int(os.environ.get("MLX_KV_BITS", "8"))
PREFILL_SIZE = int(os.environ.get("MLX_PREFILL_SIZE", "4096"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "8192"))
KV_QUANT_START = int(os.environ.get("MLX_KV_QUANT_START", "1024"))
MAX_TOOL_RETRIES = int(os.environ.get("MLX_TOOL_RETRIES", "2"))
# Browser mode: strip Claude Code bloat, keep only MCP tools
BROWSER_MODE = os.environ.get("MLX_BROWSER_MODE", "0") == "1"

# ─── Globals ─────────────────────────────────────────────────────────────────

model = None
tokenizer = None
generate_lock = threading.Lock()


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model():
    global model, tokenizer, KV_BITS
    log(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    elapsed = time.time() - t0
    log(f"Model loaded in {elapsed:.1f}s")

    # Gemma uses sliding-window attention → RotatingKVCache, which mlx-lm
    # can't quantize yet ("RotatingKVCache Quantization NYI"). Auto-disable
    # KV quantization for Gemma so inference doesn't 500 on the first call.
    if KV_BITS and "gemma" in MODEL_PATH.lower():
        log("Gemma detected: disabling KV cache quantization (RotatingKVCache NYI)")
        KV_BITS = 0

    log(f"KV cache quantization: {KV_BITS}-bit" if KV_BITS else "KV cache: full precision")


# ─── Think Tag Stripping ────────────────────────────────────────────────────

def strip_think_tags(text):
    """Remove <think>...</think> blocks and stray closing tags from Qwen's reasoning output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Remove stray </think> without matching open tag
    cleaned = re.sub(r'</think>', '', cleaned).strip()
    # Remove empty <tool_call></tool_call> blocks (Qwen sometimes emits these alongside <function> calls)
    cleaned = re.sub(r'<tool_call>\s*</tool_call>', '', cleaned).strip()
    return cleaned if cleaned else text


def clean_response(text):
    """Strip think tags and clean reasoning artifacts (but preserve tool_call tags)."""
    text = strip_think_tags(text)

    # Remove reasoning preamble if present
    if text.lstrip().startswith("Thinking"):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            s = line.strip()
            if any(s.startswith(p) for p in ['```', 'def ', 'class ', 'function ', 'import ', '#', '//', '<tool_call>']):
                return '\n'.join(lines[i:])

    return text


# ─── Tool Conversion ────────────────────────────────────────────────────────

def convert_tools_for_qwen(anthropic_tools):
    """Convert Anthropic tool definitions to HuggingFace/Qwen format."""
    if not anthropic_tools:
        return None
    qwen_tools = []
    for tool in anthropic_tools:
        qwen_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            }
        }
        qwen_tools.append(qwen_tool)
    return qwen_tools


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

    Qwen frequently produces hybrid XML/JSON like:
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
    garbled output from smaller models. Returns (list of tool calls, remaining text).
    """
    tool_calls = []
    remaining = text

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

    # Format 3: <|tool_call|>...<|/tool_call|> (some Qwen versions)
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
    """Convert Anthropic Messages format to Qwen chat messages.

    Handles:
    - Text messages (passthrough)
    - Assistant messages with tool_use blocks → Qwen <tool_call> format
    - User messages with tool_result blocks → role="tool" messages for Qwen
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

            # Assistant message with tool_use blocks → convert to Qwen format
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

def optimize_for_browser(body):
    """Strip Claude Code bloat: replace system prompt, keep only essential MCP tools."""
    # Replace massive system prompt with compact browser prompt
    body["system"] = BROWSER_SYSTEM_PROMPT

    # Filter tools to only essential chrome-devtools tools (no screenshot!)
    tools = body.get("tools", [])
    browser_tools = [t for t in tools if t.get("name", "") in BROWSER_TOOLS_ALLOW]
    if browser_tools:
        body["tools"] = browser_tools
        log(f"  Browser mode: {len(tools)} tools → {len(browser_tools)}")

    return body


# ─── Generation ──────────────────────────────────────────────────────────────

_first_request = True

def generate_response(body):
    """Run MLX inference and return Anthropic-formatted response."""
    global _first_request

    # In browser mode, strip Claude Code bloat before inference
    if BROWSER_MODE:
        body = optimize_for_browser(body)

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

    # Convert tools from Anthropic → Qwen format
    anthropic_tools = body.get("tools", [])
    qwen_tools = convert_tools_for_qwen(anthropic_tools) if anthropic_tools else None

    messages = convert_messages(body)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.2)

    if qwen_tools:
        log(f"  Tools: {len(qwen_tools)} ({', '.join(t['function']['name'] for t in qwen_tools[:5])}{'...' if len(qwen_tools) > 5 else ''})")

    # Tokenize (with tools if present)
    token_ids = tokenize_messages(messages, tools=qwen_tools)
    prompt_tokens = len(token_ids)
    log(f"  Prompt: {prompt_tokens} tokens")

    # Build generation kwargs
    gen_kwargs = {
        "prefill_step_size": PREFILL_SIZE,
    }
    if KV_BITS:
        gen_kwargs["kv_bits"] = KV_BITS
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = KV_QUANT_START

    if temperature > 0:
        gen_kwargs["sampler"] = make_sampler(temp=temperature)
    else:
        gen_kwargs["sampler"] = make_sampler(temp=0.0)

    # Generate
    full_text = ""
    gen_tokens = 0
    finish_reason = "end_turn"
    t0 = time.time()

    with generate_lock:
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=token_ids,
            max_tokens=max_tokens,
            **gen_kwargs,
        ):
            full_text += response.text
            gen_tokens = response.generation_tokens
            if response.finish_reason == "length":
                finish_reason = "max_tokens"
            elif response.finish_reason == "stop":
                finish_reason = "end_turn"

    elapsed = time.time() - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generated: {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    # Clean output (preserves <tool_call> tags)
    text = clean_response(full_text)

    # Parse tool calls from Qwen's output
    tool_calls, remaining_text = parse_tool_calls(text)

    # ─── Retry logic: if model expressed intent to use a tool but we got no valid calls ───
    tool_intent_phrases = [
        "let me", "i'll ", "i will", "let's ", "running", "executing",
        "here's the command", "bash(", "read(", "edit(", "write(",
    ]
    if not tool_calls and any(p in remaining_text.lower() for p in tool_intent_phrases):
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
            retry_tokens = tokenize_messages(retry_messages, tools=qwen_tools)
            log(f"  Retry prompt: {len(retry_tokens)} tokens")

            retry_text = ""
            retry_gen = 0
            with generate_lock:
                for response in stream_generate(
                    model=model, tokenizer=tokenizer, prompt=retry_tokens,
                    max_tokens=max_tokens, **gen_kwargs,
                ):
                    retry_text += response.text
                    retry_gen = response.generation_tokens

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
        # If we have tool calls but no text, add empty text block (Anthropic requires at least one)
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        for tc in tool_calls:
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            content_blocks.append({
                "type": "tool_use",
                "id": tool_id,
                "name": tc["name"],
                "input": tc["arguments"],
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
    print("║  Tool use: enabled (Anthropic ↔ HF tool format) ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    load_model()

    print()
    print(f"Serving Anthropic Messages API on http://localhost:{PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"KV cache: {KV_BITS}-bit quantization (start at token {KV_QUANT_START})" if KV_BITS else "KV cache: full precision")
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
