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
import subprocess
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Release version. Keep in step with the VERSION file at the repo root
# (tests/test_version.py checks they match).
__version__ = "1.0.0"

# MLX is the Mac inference backend. Guard the import so this module can also be
# imported on Linux/Windows (which have no MLX) purely to reuse the pure
# Anthropic<->tool-call translation helpers below — portable/portable_server.py
# does exactly that and talks to llama.cpp instead. On a Mac nothing changes:
# the import succeeds and MLX_AVAILABLE is True.
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# mlx-vlm is optional. With it installed, a model that ships a vision tower
# (Qwen 3.5, Gemma 4 E2B/E4B, ...) is loaded whole, and images Claude Code
# sends (a Read on a .png, a pasted screenshot) reach the model instead of
# being replaced with a marker. Without it, nothing changes.
try:
    from mlx_vlm import load as vlm_load
    from mlx_vlm import stream_generate as vlm_stream_generate
    VLM_AVAILABLE = True
except Exception:
    VLM_AVAILABLE = False

# ─── Configuration ───────────────────────────────────────────────────────────

def env_int(name, default):
    """int() an env var, treating unset *and empty* as "use the default".

    os.environ.get(name, default) only falls back when the key is absent, not
    when it is present-but-empty. The launchers pass values through
    ensure_mlx_server() as MLX_KV_BITS="${MLX_KV_BITS:-}", which exports an
    empty string when the caller hasn't set one — so int("") raised ValueError
    and the server died before it ever bound the port.
    """
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


MODEL_PATH = os.environ.get("MLX_MODEL", "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")
PORT = env_int("MLX_PORT", 4000)
KV_BITS = env_int("MLX_KV_BITS", 0)  # Gemma 4 RotatingKVCache doesn't support quantization
# This used to default to 8192, which quietly TRIPLES peak memory on a long
# prompt and is the reason a big Claude Code request could take the machine into
# swap (or straight into the OOM killer on a 32GB Mac) while the model itself
# fits fine.
#
# mlx_lm's prefill loop runs the WHOLE model on each chunk, lm_head included, so
# a single 8192-token chunk allocates an (8192 x vocab) bf16 logits tensor it
# immediately throws away — 4.3GB at Gemma's 262144 vocab. On top of that the
# full attention layers score the entire chunk against every key so far. None of
# it is reused, all of it is live at once.
#
# Measured on gemma-4-31b-it-abliterated-4bit-mlx (16.7GB of weights resident),
# same prompt each time, via mx.get_peak_memory():
#
#     prompt    prefill 8192      prefill 1024      prefill 512
#     21.4k     34.2GB / 49.3s    20.9GB / 37.7s    -      / 39.9s
#     38.5k     -      / -        23.5GB / 83.1s    21.9GB / 71.0s
#
# 17.5GB of transients on a 21k prompt, against 4.2GB for the same work at 512.
# Small chunks are not slower either — the GPU saturates far below 8192 tokens —
# so there was never anything being traded away for that memory.
PREFILL_SIZE = env_int("MLX_PREFILL_SIZE", 512)
# Ceiling on MLX's buffer recycle pool, which counts toward this process's
# memory. Measured at 0.0GB on the model above either way, so treat it as a
# backstop for larger ones (Llama 70B 8-bit) rather than a fix on its own.
# MLX_CACHE_LIMIT_GB=0 disables the cap.
MEM_CACHE_LIMIT_GB = float(os.environ.get("MLX_CACHE_LIMIT_GB") or 3)
# Pre-fill an empty thinking block to skip Gemma 4 reasoning chains entirely.
# Set MLX_SUPPRESS_THINKING=0 to disable (e.g. when you want reasoning output).
SUPPRESS_THINKING = os.environ.get("MLX_SUPPRESS_THINKING", "1") == "1"
DEFAULT_MAX_TOKENS = env_int("MLX_MAX_TOKENS", 8192)
KV_QUANT_START = env_int("MLX_KV_QUANT_START", 256)
MAX_TOOL_RETRIES = env_int("MLX_TOOL_RETRIES", 2)
# Browser mode: strip Claude Code bloat, keep only MCP tools
BROWSER_MODE = os.environ.get("MLX_BROWSER_MODE", "0") == "1"
# Code mode: auto-detect Claude Code coding sessions and replace the huge harness
# system prompt with a Llama-tuned one. Set MLX_CODE_MODE=0 to disable.
CODE_MODE_ENABLED = os.environ.get("MLX_CODE_MODE", "1") != "0"
# Vision: "auto" (default) turns it on when mlx-vlm is installed AND the model's
# config.json has a vision_config. "1" requires it, "0" keeps the text-only path.
VISION_SETTING = (os.environ.get("MLX_VISION") or "auto").strip().lower()
# Hard ceiling on prompt + answer tokens. A prompt over it (or one that leaves
# no room to answer) gets the same 400 the real API sends for an over-long
# prompt, so Claude Code compacts instead of the server allocating KV cache the
# Mac does not have, and max_tokens is trimmed to the room left. 0 = no
# ceiling. This is the only thing that enforces a small window: Claude Code
# 2.1.306 treats any CLAUDE_CODE_AUTO_COMPACT_WINDOW under 100000 as 100000, so
# scripts/mac-tier.sh no longer sets it for the 16K tiers.
MAX_CONTEXT = env_int("MLX_MAX_CONTEXT", 0)
# Longest side, in pixels, an image is shrunk to before the vision tower sees
# it. A 1568x980 screenshot raised the server's peak by about 0.4 GB on
# Qwen 3.5 2B bf16; fewer pixels is fewer patches to encode. 0 = send as is.
MAX_IMAGE_SIDE = env_int("MLX_MAX_IMAGE_SIDE", 0)

# ─── Globals ─────────────────────────────────────────────────────────────────

model = None
tokenizer = None
processor = None      # mlx-vlm processor; set only when VISION is on
VISION = False        # True once a vision model is loaded through mlx-vlm
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

GB = 1024 ** 3


def mem_snapshot():
    """What this process is actually holding, in GB.

    Watch mx_peak, not rss. `ps rss` cannot see Metal buffers — it reported a
    flat 16.7GB straight through a prefill that MLX itself measured at 34GB, so
    a server can be moments from swapping while every RSS reading looks calm.
    Activity Monitor's "Memory" column (phys_footprint) does see them.
    """
    try:
        rss = psutil_rss_gb()
    except Exception:
        rss = 0.0
    return {
        "rss_gb": round(rss, 1),
        "mx_active_gb": round(mx.get_active_memory() / GB, 1),
        "mx_cache_gb": round(mx.get_cache_memory() / GB, 1),
        "mx_peak_gb": round(mx.get_peak_memory() / GB, 1),
    }


def psutil_rss_gb():
    """RSS without a psutil dependency — the mlx venv doesn't ship one."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 / 1024 if out else 0.0


def release_transients(tag=""):
    """Hand MLX's recycle pool back after every generation, and log the peak.

    The log line is half the point. When a memory death happens there is no
    traceback and usually no crash report, because the process is killed from
    outside — so a per-request peak, written down while the server was still
    alive, is the only thing that distinguishes "that prompt was too big" from
    any other silent exit. On a 31B 4-bit the pool itself measures 0.0GB, so the
    clear_cache is insurance for models that do accumulate one.
    """
    before = mx.get_cache_memory() / GB
    peak = mx.get_peak_memory() / GB
    mx.clear_cache()
    log(f"  Memory{tag}: peak {peak:.1f}GB this request, "
        f"released {before:.1f}GB of buffer cache, rss now {psutil_rss_gb():.1f}GB")
    mx.reset_peak_memory()


def model_config(path):
    """The model's config.json as a dict, from a local folder or the HF cache.
    Returns {} when it cannot be read (no network, private repo, ...)."""
    try:
        if os.path.isdir(path):
            cfg = os.path.join(path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            cfg = hf_hub_download(path, "config.json")
        with open(cfg) as f:
            return json.load(f)
    except Exception:
        return {}


def wants_vision(setting, config, vlm_available):
    """Decide whether to load through mlx-vlm. Pure, so it is unit-tested.

    "0"/"off" never; "1"/"on" whenever mlx-vlm is there (load fails loudly if
    the model has no vision tower); "auto" only when both mlx-vlm is installed
    and config.json declares a vision tower.
    """
    if setting in ("0", "off", "false", "no"):
        return False
    if not vlm_available:
        return False
    if setting in ("1", "on", "true", "yes"):
        return True
    return bool(config.get("vision_config"))


def load_model():
    global model, tokenizer, processor, VISION, KV_BITS
    if MEM_CACHE_LIMIT_GB > 0:
        mx.set_cache_limit(int(MEM_CACHE_LIMIT_GB * GB))
        log(f"MLX buffer cache capped at {MEM_CACHE_LIMIT_GB:g}GB")
    log(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    if wants_vision(VISION_SETTING, model_config(MODEL_PATH), VLM_AVAILABLE):
        # One copy of the weights serves both text and image turns.
        model, processor = vlm_load(MODEL_PATH)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        VISION = True
        mx.eval(model.parameters())
        log(f"Vision on (mlx-vlm): images in the newest message reach the model")
        log(f"Model loaded in {time.time() - t0:.1f}s")
        return
    if VISION_SETTING in ("1", "on", "true", "yes") and not VLM_AVAILABLE:
        log("MLX_VISION=1 but mlx-vlm is not installed (pip install mlx-vlm); "
            "loading text-only")
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    # Fallback chat template if model doesn't provide one (Llama 3.3 has its own)
    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = GEMMA4_CHAT_TEMPLATE
        log("Injected Gemma 4 chat template")

    # Some quant repos ship a generation_config.json whose eos_token_id
    # disagrees with tokenizer_config.json's eos_token. mlx-lm seeds its stop
    # set (tokenizer.eos_token_ids) from generation_config, so when the chat
    # template actually terminates turns with the *other* token, generation
    # never stops: the real stop token is decoded as ordinary text and every
    # request runs to max_tokens.
    #
    # Seen on divinetribe/Hermes-4-14B-abliterated-4bit-mlx, whose
    # generation_config says 151643 (<|endoftext|>) while its template emits
    # 151645 (<|im_end|>) — 503s and 4096 tokens for what should be a 20-token
    # tool call. Unioning the two is safe: a token that is EOS in either config
    # is never a legitimate mid-sequence token.
    _tok_eos = getattr(tokenizer, "eos_token_id", None)
    if _tok_eos is not None:
        _stop_set = set(getattr(tokenizer, "eos_token_ids", None) or set())
        if _tok_eos not in _stop_set:
            tokenizer.eos_token_ids = _stop_set | {_tok_eos}
            log(f"Added eos_token_id {_tok_eos} to stop set (was {_stop_set or set()})")

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
    # A lone </think>: the chat template opened the block in the prompt
    # (Qwen 3.5 ends its generation prompt with "<think>\n"), so the output
    # starts mid-thought. Everything before the close is reasoning; it used to
    # go out as the answer, followed by the real answer a second time.
    if '</think>' in cleaned:
        cleaned = cleaned.rsplit('</think>', 1)[1].strip()
    # Gemma 4 thinking: <|channel>thought\n...<channel|>
    cleaned = re.sub(r'<\|channel>thought\n.*?<channel\|>', '', cleaned, flags=re.DOTALL).strip()
    # A lone <channel|>, the Gemma 4 version of the lone </think> above: the
    # thought channel was opened (and closed empty) in the prompt to suppress
    # thinking, and Gemma 4 E4B reasons anyway, then closes the channel again.
    # Everything before the close is reasoning, not the answer.
    if '<channel|>' in cleaned:
        cleaned = cleaned.rsplit('<channel|>', 1)[1].strip()
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
    # Gemma 4 + Qwen + ChatML: truncate at end-of-turn or start of a new turn,
    # whichever marker comes first (the live stream cuts the same way).
    cuts = [text.index(m) for m in ['<turn|>', '<|turn>', '<|im_end|>', '<|endoftext|>',
                                    '<|im_start|>', '<|end_of_text|>', '<|eot_id|>'] if m in text]
    if cuts:
        text = text[:min(cuts)].strip()

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


def _decode_string_args(args):
    """Some models send arguments as a JSON-encoded string, OpenAI style:
    "arguments": "{\"command\": \"ls\"}". Decode it so the call doesn't go
    out with empty input."""
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
        except ValueError:
            return args
        if isinstance(decoded, dict):
            return decoded
    return args


def _gemma4_args(args_str):
    """Parse Gemma 4's argument list (the part inside call:Name{...}) as JSON.

    Gemma 4 writes strings as <|"|>text<|"|>, everything else as bare JSON
    (true, 50, [...], {...}), and object keys unquoted, sometimes with a dash
    (Grep's -i). Swap the strings out, quote the keys, swap the strings back as
    JSON strings, and parse. Returns None when that is not valid JSON, so the
    caller can fall back to the older regex matching.
    """
    strings = []

    def stash(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    s = re.sub(r'<\|"\|>(.*?)<\|"\|>', stash, args_str, flags=re.DOTALL)
    s = re.sub(r'(^|[{,])\s*([\w-]+)\s*:', r'\1"\2":', s)
    s = re.sub(r'\x00(\d+)\x00', lambda m: json.dumps(strings[int(m.group(1))]), s)
    try:
        parsed = json.loads("{" + s + "}")
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_schemas(tools):
    """{name: JSON schema} from Anthropic tools ({name, input_schema}) or the
    HuggingFace/OpenAI shape convert_tools_for_llm produces."""
    schemas = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if name:
            schema = fn.get("input_schema") or fn.get("parameters") or {}
            schemas[name] = schema if isinstance(schema, dict) else {}
    return schemas


def _fits_schema(args, schema):
    """True when args is a dict that has every required key and no key the
    schema does not declare. Used to accept a recovered tool call only when it
    is unambiguous."""
    if not isinstance(args, dict) or not args:
        return False
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    if any(k not in args for k in required):
        return False
    return not props or all(k in props for k in args)


def _recover_gemma4_missing_opener(text, tools):
    """Gemma 4 tool calls that lost their opening marker.

    On long calls (a Write whose content is a whole HTML page) Gemma 4 31B
    8-bit sometimes drops the <|tool_call>call:Write{content:<|"|> head and
    starts straight into the value, so the output ends like
        ...</html>\\n<|"|>,file_path:<|"|>budget.html<|"|>}<tool_call|>
    Format 0 never sees it (no opener), the file is never written, and the
    raw page is shown to the user as text. 4 of 4 tries on 10/29.

    The closer <tool_call|> and the <|"|> delimiters are Gemma special tokens
    that never appear in prose, so they anchor the recovery. A call is only
    accepted when it names (or schema-matches exactly one) declared tool and
    its arguments fit that tool's schema. Three shapes, most explicit first:
      a) the name survived:            call:Write{...}<tool_call|>  /  Write{...}
      b) the name is gone too:         {file_path:<|"|>..<|"|>,content:..}<tool_call|>
      c) the first key is gone too:    <page>..<|"|>,file_path:<|"|>..<|"|>}<tool_call|>
    Returns (tool_calls, remaining_text) or ([], text).
    """
    schemas = _tool_schemas(tools)
    if not schemas or "<tool_call|>" not in text:
        return [], text
    Q = '<|"|>'
    calls, kept, start = [], [], 0
    for closer in re.finditer(r'<tool_call\|>', text):
        segment = text[start:closer.start()]
        start = closer.end()
        body_end = segment.rstrip().rstrip(">").rstrip()
        if not body_end.endswith("}"):
            kept.append(segment)
            continue
        body_end = body_end[:-1]
        found = None

        # a) a declared tool name followed by its argument list.
        names = "|".join(re.escape(n) for n in sorted(schemas, key=len, reverse=True))
        for m in re.finditer(rf'(?:<\|tool_call>|<tool_call:|call:)?(?<![\w.-])({names})\s*\{{', body_end):
            args = _gemma4_args(body_end[m.end():])
            if _fits_schema(args, schemas[m.group(1)]):
                found = (m.start(), m.group(1), args)
                break

        # b) a bare argument list; the schema has to pick exactly one tool.
        if not found:
            brace = body_end.find("{")
            # A name right before the brace is a tool the client never
            # declared (else (a) had it); don't re-route it to another tool.
            named = re.search(r'[\w.-]\s*$', body_end[:brace]) if brace >= 0 else None
            if brace >= 0 and not named and Q in body_end[brace:]:
                args = _gemma4_args(body_end[brace + 1:])
                fits = [n for n, s in schemas.items() if _fits_schema(args, s)]
                if len(fits) == 1:
                    found = (brace, fits[0], args)

        # c) the value of the first argument starts the segment; the one
        #    required string key the rest of the call lacks is its name.
        if not found and Q in body_end:
            first_q = body_end.find(Q)
            tail = body_end[first_q:]
            if re.match(r'<\|"\|>\s*,\s*[\w-]+\s*:', tail):
                rest = _gemma4_args(tail[len(Q):].lstrip().lstrip(","))
                if isinstance(rest, dict):
                    options = []
                    for n, s in schemas.items():
                        props = s.get("properties") or {}
                        missing = [k for k in s.get("required") or [] if k not in rest]
                        if (len(missing) == 1
                                and (props.get(missing[0]) or {}).get("type", "string") == "string"):
                            args = dict(rest, **{missing[0]: body_end[:first_q].strip("\n")})
                            if _fits_schema(args, s):
                                options.append((n, args))
                    if len(options) == 1:
                        found = (0, options[0][0], options[0][1])

        if found:
            pos, name, args = found
            kept.append(segment[:pos])
            calls.append({"name": name, "arguments": args})
            log(f"  Gemma4 tool call with no opening marker recovered: {name}({list(args.keys())})")
        else:
            kept.append(segment)
            kept.append(closer.group(0))
    if not calls:
        return [], text
    kept.append(text[start:])
    return calls, "".join(kept).strip()


def _element_params(fname, params_text, tools):
    """<function=Write> followed by <file_path>..</file_path><content>..</content>.

    Qwen 3.5 0.8B (llama.cpp, phone test 10/30) writes each argument as an
    element named after the parameter instead of <parameter=file_path>, so the
    call went out with no arguments and Claude Code stopped after one turn.
    Only element names that are parameters of the declared tool count, and the
    result has to fit that tool's schema, so HTML or XML inside a value (a
    <div> in content) is never read as an argument. Returns {} when it is not
    that shape.
    """
    schema = _tool_schemas(tools).get(fname)
    if not schema:
        return {}
    props = schema.get("properties") or {}
    arguments = {}
    pos = 0
    while True:
        m = re.compile(r'<(' + "|".join(map(re.escape, props)) + r')>').search(params_text, pos) if props else None
        if not m:
            break
        close = params_text.find(f"</{m.group(1)}>", m.end())
        if close < 0:
            break
        arguments.setdefault(m.group(1), params_text[m.end():close].strip("\n"))
        pos = close + len(m.group(1)) + 3
    return arguments if _fits_schema(arguments, schema) else {}


def parse_tool_calls(text, tools=None):
    """Parse tool calls from generated text. Handles multiple formats including
    Gemma 4 native format. Returns (list of tool calls, remaining text).

    tools: the tools the request declared (Anthropic or HF shape). Only used to
    recover calls whose opening marker the model dropped; without it that
    recovery is off, so a guess can never name a tool the client didn't offer.
    """
    tool_calls = []

    # Format 0: Gemma 4 native — <|tool_call>call:Name{key:<|"|>val<|"|>}<tool_call|>
    # Parse BEFORE replacing escape tokens — use <|"|> as reliable value delimiters
    #
    # Three spellings of the opener turn up in practice. The documented one is
    # <|tool_call>call:Name{...}. Some models drop the call: prefix. And
    # abliterated Gemma 4 31B collapses the whole marker to <tool_call:Name{...},
    # which is the SAME grammar — same <|"|> value delimiters, same <tool_call|>
    # closer, only the opening marker differs — so it is one alternation here
    # rather than a new format further down.
    #
    # That third spelling cost 6 of 14 on scripts/test_mlx_server.py with
    # divinetribe/gemma-4-31b-it-abliterated-4bit-mlx. Every call was well
    # formed, every call was invisible, and the retry path can't rescue it
    # because it asks for <tool_call> tags the model has already decided against.
    # Some also emit a stray > after the closing brace; the body regex tolerates
    # it because re.match doesn't have to consume the whole string.
    gemma4_pattern = r'(?:<\|tool_call>|<tool_call:)(.*?)<tool_call\|>'
    gemma4_matches = list(re.finditer(gemma4_pattern, text, re.DOTALL))
    if gemma4_matches:
        remaining = text
        for match in gemma4_matches:
            remaining = remaining.replace(match.group(0), "", 1)
            content = match.group(1).strip()
            # [\w.-]: MCP tool names carry the server name, which often has a
            # dash (mcp__chrome-devtools__click), same as Formats 1 and 2 allow.
            name_m = re.match(r'(?:call:)?([\w.-]+)\s*\{(.*)\}', content, re.DOTALL)
            if not name_m:
                log(f"  Gemma4 no name match: {content[:80]}")
                continue
            name = name_m.group(1)
            args_str = name_m.group(2)
            # Primary: read the whole argument list as Gemma's JSON dialect, which
            # covers nested arrays/objects and keys like -i that the regexes miss.
            arguments = _gemma4_args(args_str)
            # call:list_pages{} parses to {}: a real call with no arguments,
            # not a failed parse, so it has to go out as one.
            no_args_call = arguments == {}
            if arguments is None:
                arguments = {}
                # Fallback: extract key:<|"|>value<|"|> pairs (handles embedded quotes)
                for km in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', args_str, re.DOTALL):
                    arguments[km.group(1)] = km.group(2)
                # Gemma 4 writes non-string values bare (replace_all:true, limit:50)
                # in the same call as <|"|> strings, and they were dropped. Pick them
                # up from what is left once the strings are removed, so a colon
                # inside a string value can't be read as a key.
                if arguments:
                    bare = re.sub(r'\w+:<\|"\|>.*?<\|"\|>', '', args_str, flags=re.DOTALL)
                    for km in re.finditer(r'(\w+):([^,}]+)', bare):
                        try:
                            arguments.setdefault(km.group(1), json.loads(km.group(2).strip()))
                        except ValueError:
                            pass
                # Fallback: unquoted values (numbers, simple strings)
                if not arguments:
                    for km in re.finditer(r'(\w+):([^,}]+)', args_str):
                        val = km.group(2).strip().strip('"\'')
                        arguments[km.group(1)] = val
            if arguments or no_args_call:
                tool_calls.append({"name": name, "arguments": arguments})
                log(f"  Gemma4 tool call: {name}({list(arguments.keys())})")
            else:
                log(f"  Gemma4 no args parsed: {content[:80]}")
        if tool_calls:
            return tool_calls, remaining.strip()

    # Format 0b: Gemma 4 with the opening marker missing (long Write calls).
    # Needs the declared tools; see _recover_gemma4_missing_opener.
    if tools:
        recovered, rest = _recover_gemma4_missing_opener(text, tools)
        if recovered:
            return recovered, rest

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
                arguments = _decode_string_args(obj.get("parameters", {}))
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
            # Some models wrap several calls in one tag as a JSON array; a
            # list (or any non-object) here used to raise AttributeError.
            for cd in call_data if isinstance(call_data, list) else [call_data]:
                if isinstance(cd, dict):
                    tool_calls.append({
                        "name": cd.get("name", ""),
                        "arguments": cd.get("arguments", {}),
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
                if not arguments:
                    arguments = _element_params(fname, params_text, tools)
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
            if not arguments:
                arguments = _element_params(func_name, params_text, tools)
            tool_calls.append({"name": func_name, "arguments": arguments})
            remaining = remaining.replace(match.group(0), "", 1)

    # Format 3: <|tool_call|>...<|/tool_call|> (some model versions)
    if not tool_calls:
        pattern3 = r'<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>'
        for match in re.finditer(pattern3, text, re.DOTALL):
            remaining = remaining.replace(match.group(0), "", 1)
            try:
                call_data = json.loads(match.group(1))
                for cd in call_data if isinstance(call_data, list) else [call_data]:
                    if isinstance(cd, dict):
                        tool_calls.append({
                            "name": cd.get("name", ""),
                            "arguments": cd.get("arguments", {}),
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
    # native <tool_call> wrapper. Three sub-cases supported:
    #   (a) single JSON object inside the fence
    #   (b) multiple JSON objects back-to-back inside the same fence (no
    #       array, no commas) — the model expressing several sequential
    #       tool calls in one go. Use raw_decode in a loop to extract them.
    #   (c) a single JSON array of call objects — Hermes 4 14B emits this
    #       shape instead of (b); without this branch the whole fence was
    #       skipped (array doesn't start with "{") and fell through to the
    #       garbled-tool-call retry path even though the JSON was valid.
    def _as_call(obj):
        if not isinstance(obj, dict):
            return None
        name = obj.get("name") or obj.get("tool")
        args = (
            obj.get("arguments")
            or obj.get("parameters")
            or obj.get("args")
            or obj.get("input")
        )
        return {"name": name, "arguments": args} if name and isinstance(args, dict) else None

    if not tool_calls:
        pattern_md = r'```(?:json|tool[_ ]?call)?\s*\n?(.*?)\s*\n?```'
        decoder_md = json.JSONDecoder()
        for match in re.finditer(pattern_md, text, re.DOTALL):
            content = match.group(1).strip()
            stripped = content.lstrip()
            if not stripped.startswith("{") and not stripped.startswith("["):
                continue
            extracted_in_block = []
            if stripped.startswith("["):
                try:
                    items = json.loads(content)
                except json.JSONDecodeError:
                    items = []
                for obj in items if isinstance(items, list) else []:
                    call = _as_call(obj)
                    if call:
                        extracted_in_block.append(call)
            else:
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
                    call = _as_call(obj)
                    if call:
                        extracted_in_block.append(call)
            if extracted_in_block:
                tool_calls.extend(extracted_in_block)
                remaining = remaining.replace(match.group(0), "", 1)
                log(
                    f"  Markdown-fenced tool calls: "
                    f"{', '.join(t['name'] for t in extracted_in_block)}"
                )

    # Format 3.7: bare top-level JSON with no wrapper at all —
    #   {"name": "Bash", "arguments": {"command": "ls /tmp"}}
    # Qwen 2.5 Coder emits exactly this: no <tool_call> tags, no <tools>, no
    # markdown fence, just the object. Formats 3.5/3.6 already cover the
    # wrapped and fenced variants of the same shape, so this is the last one.
    # Without it Qwen 2.5 Coder scores 0/12 on scripts/test_mlx_server.py —
    # every call is well-formed and every call is invisible to the parser.
    # Re-prompting doesn't rescue it either: the retry explicitly asks for
    # <tool_call> tags and the model declines, and recover_garbled_tool_json
    # never fires because the JSON was never garbled.
    #
    # Same back-to-back handling as 3.6 (raw_decode in a loop) for models that
    # emit several sequential calls without an array. Runs only after every
    # tagged format has failed, so an explicit wrapper always takes priority.
    if not tool_calls:
        stripped = text.strip()
        if stripped.startswith("{"):
            decoder_bare = json.JSONDecoder()
            extracted_bare = []
            pos = 0
            while pos < len(stripped):
                while pos < len(stripped) and stripped[pos] in " \t\n\r,":
                    pos += 1
                if pos >= len(stripped) or stripped[pos] != "{":
                    break
                try:
                    obj, end_pos = decoder_bare.raw_decode(stripped, pos)
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
                    extracted_bare.append({"name": name, "arguments": args})
            if extracted_bare:
                tool_calls.extend(extracted_bare)
                # The whole message was the tool call; nothing survives as prose.
                remaining = stripped[pos:].strip()
                log(
                    f"  Bare JSON tool calls: "
                    f"{', '.join(t['name'] for t in extracted_bare)}"
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

    # Deduplicate tool calls (model sometimes emits same call in multiple formats).
    # Key on name AND arguments: keying on name alone dropped every parallel
    # call after the first one to the same tool (Read /a, Read /b -> Read /a).
    seen = set()
    deduped = []
    for tc in tool_calls:
        tc["arguments"] = _decode_string_args(tc["arguments"])
        key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            deduped.append(tc)
        else:
            log(f"  Deduped: {tc['name']}")
    tool_calls = deduped

    # Clean remaining text: strip any leftover <function=...> or <tool_call> fragments
    remaining = re.sub(r'<function=[\w.-]+>.*?</function>', '', remaining, flags=re.DOTALL)
    remaining = re.sub(r'</?tool_call>', '', remaining)
    remaining = remaining.strip()

    return tool_calls, remaining


# ─── Anthropic Message Conversion ───────────────────────────────────────────

def image_source(block):
    """An Anthropic image block as something mlx-vlm's load_image takes
    (a data URI or a URL), or None if the block has no usable source."""
    src = block.get("source") or {}
    if src.get("type") == "base64" and src.get("data"):
        return f"data:{src.get('media_type', 'image/png')};base64,{src['data']}"
    if src.get("type") == "url" and src.get("url"):
        return src["url"]
    return None


def _render_blocks(blocks, keep_images):
    """tool_result / user content blocks → a string, or (vision) a content list.

    Images only survive when keep_images is True: the newest message of the
    request. Older images become a marker, so a long session does not re-encode
    every screenshot it ever saw on every turn (and the model already answered
    about them in its own words).
    """
    items = []
    for b in blocks:
        if not isinstance(b, dict):
            items.append({"type": "text", "text": str(b)})
            continue
        bt = b.get("type")
        if bt == "image":
            uri = image_source(b) if keep_images else None
            if uri:
                items.append({"type": "image", "image": uri})
            else:
                items.append({"type": "text", "text": "[image omitted]"})
        elif bt == "document":
            items.append({"type": "text", "text": "[document omitted]"})
        else:
            items.append({"type": "text", "text": b.get("text", str(b))})
    if not any(i["type"] == "image" for i in items):
        return "\n".join(i["text"] for i in items)
    return items


def shrink_image(source, max_side):
    """A data URI shrunk so its longest side is at most max_side, as a PIL
    image (mlx-vlm's load_image takes one). URLs and images already small
    enough are returned unchanged."""
    if not max_side or not isinstance(source, str) or not source.startswith("data:"):
        return source
    try:
        import base64
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(source.split(",", 1)[1])))
        if max(img.size) <= max_side:
            return source
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        return img
    except Exception as e:
        log(f"  Could not shrink image ({e}); sending it as is")
        return source


def collect_images(messages):
    """Image sources in prompt order, from messages built by
    convert_messages(..., vision=True)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            out.extend(i["image"] for i in c if i.get("type") == "image")
    return out


# Private keys convert_messages() leaves on its messages. template_messages()
# removes them (or turns them into structured tool_calls) before any chat
# template sees the list.
STRUCT_TEXT = "_ccl_text"
STRUCT_CALLS = "_ccl_tool_calls"
STRUCT_ID = "_ccl_tool_call_id"
STRUCT_NAME = "_ccl_tool_name"
STRUCT_TURN_TEXT = "_ccl_tool_turn_text"
_STRUCT_KEYS = (STRUCT_TEXT, STRUCT_CALLS, STRUCT_ID, STRUCT_NAME, STRUCT_TURN_TEXT)
_PROBE_MARK = "CCL_TOOL_RESULT_PROBE_7f3a"
_structured_cache = {}


def _plain(m):
    return {k: v for k, v in m.items() if k not in _STRUCT_KEYS}


def _structured(messages):
    """Assistant tool calls as OpenAI-style tool_calls, tool results as
    role=tool messages carrying tool_call_id + name, placed right after the
    assistant turn that made the calls (a tool_result turn's own text goes after
    its results). A tool message with no assistant tool_calls before it (its
    call was compacted away) becomes a user message so it is not dropped."""
    out = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get(STRUCT_TURN_TEXT):
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            if j > i + 1:
                for t in messages[i + 1:j]:
                    out.append(t)
                out.append(m)
                i = j
                continue
        out.append(m)
        i += 1

    final = []
    open_calls = False
    for m in out:
        role = m.get("role")
        if role == "assistant" and m.get(STRUCT_CALLS):
            final.append({"role": "assistant", "content": m.get(STRUCT_TEXT, ""),
                          "tool_calls": m[STRUCT_CALLS]})
            open_calls = True
        elif role == "tool":
            if open_calls:
                t = {"role": "tool", "content": m.get("content", "")}
                if m.get(STRUCT_ID):
                    t["tool_call_id"] = m[STRUCT_ID]
                if m.get(STRUCT_NAME):
                    t["name"] = m[STRUCT_NAME]
                final.append(t)
            else:
                final.append({"role": "user", "content": m.get("content", "")})
        else:
            final.append(_plain(m))
            open_calls = False
    return final


def template_needs_structured_tool_calls(tok):
    """True when this tokenizer's chat template drops a tool result that
    follows a text-form tool call (our default) but renders it after a
    structured tool_calls turn. Stock Gemma 4 templates do this: they skip
    role=tool messages unless the assistant turn before them has tool_calls.
    Decided once per template by rendering a probe conversation both ways."""
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str) or "tool_calls" not in template:
        return False
    key = hash(template)
    if key in _structured_cache:
        return _structured_cache[key]
    probe = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "Read", "arguments": {"file_path": "a"}}\n</tool_call>',
         STRUCT_TEXT: "", STRUCT_CALLS: [{"id": "call_0", "type": "function",
                                          "function": {"name": "Read", "arguments": {"file_path": "a"}}}]},
        {"role": "tool", "content": _PROBE_MARK, STRUCT_ID: "call_0", STRUCT_NAME: "Read"},
    ]
    result = False
    try:
        flat = tok.apply_chat_template([_plain(m) for m in probe], add_generation_prompt=True, tokenize=False)
        if _PROBE_MARK not in flat:
            structured = tok.apply_chat_template(_structured(probe), add_generation_prompt=True, tokenize=False)
            result = _PROBE_MARK in structured
    except Exception as e:  # a template that cannot render the probe keeps the default
        log(f"  Tool-call template probe failed ({e}); keeping text-form tool calls")
    _structured_cache[key] = result
    if result:
        log("  Chat template drops text-form tool results: sending structured tool_calls")
    return result


def template_messages(messages, tok=None):
    """The message list a chat template should see. Text-form tool calls (the
    long-standing default) unless the template would drop the results."""
    tok = tok if tok is not None else tokenizer
    if any(m.get(STRUCT_CALLS) for m in messages) and template_needs_structured_tool_calls(tok):
        return _structured(messages)
    return [_plain(m) for m in messages]


def convert_messages(body, vision=False, tool_ids=False):
    """Convert Anthropic Messages format to MLX chat messages.

    Handles:
    - Text messages (passthrough)
    - Assistant messages with tool_use blocks → <tool_call> format
    - User messages with tool_result blocks → role="tool" messages
    - vision=True: images in the newest message become {"type": "image"}
      content items (see collect_images); everywhere else, and with
      vision=False, an image becomes a short "[image omitted]" marker
    """
    messages = []
    tool_names = {}  # tool_use id -> tool name, for the structured tool messages
    body_messages = body.get("messages", [])

    # System prompt
    if body.get("system"):
        sys_text = body["system"]
        if isinstance(sys_text, list):
            sys_text = "\n".join(b.get("text", "") for b in sys_text if b.get("type") == "text")
        messages.append({"role": "system", "content": sys_text})

    # Conversation messages
    for msg_index, msg in enumerate(body_messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        keep_images = vision and msg_index == len(body_messages) - 1

        # Simple string content
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        # List of content blocks
        if isinstance(content, list):
            text_parts = []
            tool_use_parts = []
            tool_result_parts = []

            image_parts = []

            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "image" and role == "user":
                    image_parts.append(block)
                elif btype == "tool_use":
                    tool_use_parts.append(block)
                elif btype == "tool_result":
                    tool_result_parts.append(block)

            # Assistant message with tool_use blocks → convert to LLM format
            if role == "assistant" and tool_use_parts:
                content_str = ""
                if text_parts:
                    content_str = "\n".join(p for p in text_parts if p)
                text_only = content_str.strip()
                for tu in tool_use_parts:
                    call_json = json.dumps({
                        "name": tu.get("name", ""),
                        "arguments": tu.get("input", {})
                    }, ensure_ascii=False)
                    content_str += f"\n<tool_call>\n{call_json}\n</tool_call>"
                    if tu.get("id"):
                        tool_names[tu["id"]] = tu.get("name", "")
                # The same calls in structured form, for chat templates that only
                # render tool results after an assistant turn with tool_calls
                # (stock Gemma 4). template_messages() picks one form or the other.
                messages.append({"role": "assistant", "content": content_str.strip(),
                                 STRUCT_TEXT: text_only,
                                 STRUCT_CALLS: [{
                                     "id": tu.get("id") or f"call_{i}",
                                     "type": "function",
                                     "function": {"name": tu.get("name", ""),
                                                  "arguments": tu.get("input", {}) or {}},
                                 } for i, tu in enumerate(tool_use_parts)]})

            # User message with tool_result blocks → split into tool messages
            elif tool_result_parts:
                # Add any text from the user first
                if text_parts:
                    text = "\n".join(p for p in text_parts if p)
                    if text.strip():
                        messages.append({"role": "user", "content": text, STRUCT_TURN_TEXT: True})

                # Each tool_result becomes a "tool" role message
                for tr in tool_result_parts:
                    result_content = tr.get("content", "")
                    if isinstance(result_content, list):
                        # An image (e.g. Read on a .png) used to fall through to
                        # str(b) and land in the prompt as base64 text, which can
                        # be hundreds of thousands of tokens. Without vision it
                        # is a marker; with vision it is a real image item.
                        result_content = _render_blocks(result_content, keep_images)
                    elif not isinstance(result_content, str):
                        result_content = str(result_content)
                    if tr.get("is_error"):
                        if isinstance(result_content, list):
                            result_content = [{"type": "text", "text": "Error: "}] + result_content
                        else:
                            result_content = "Error: " + result_content
                    # Include tool name context if we can find it
                    tid = tr.get("tool_use_id", "")
                    messages.append({"role": "tool", "content": result_content,
                                     STRUCT_ID: tid, STRUCT_NAME: tool_names.get(tid, "")})

            # Regular message: text, plus any images the user pasted
            else:
                if image_parts:
                    rendered = _render_blocks(image_parts + [
                        {"type": "text", "text": p} for p in text_parts if p], keep_images)
                    if rendered:
                        messages.append({"role": role, "content": rendered})
                    continue
                text = "\n".join(p for p in text_parts if p)
                if text.strip():
                    messages.append({"role": role, "content": text})

    # tool_ids=True keeps the private keys template_messages() needs to build
    # structured tool_calls; without it the output is the plain message list.
    return messages if tool_ids else [_plain(m) for m in messages]


def template_extras():
    """Extra chat-template variables. Vision models here are Qwen 3.5-style
    thinkers; with MLX_SUPPRESS_THINKING (the default) their template is told
    enable_thinking=False so a small Mac does not spend its tokens on a
    reasoning chain before every tool call. Templates that do not use the
    variable ignore it."""
    return {"enable_thinking": False} if (VISION and SUPPRESS_THINKING) else {}


def render_prompt(messages, tools=None):
    """The chat template as text (vision path: mlx-vlm tokenizes it itself so it
    can expand each image placeholder into that image's patch tokens)."""
    extras = template_extras()
    messages = template_messages(messages)
    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **extras)
    except Exception as e:
        if not tools:
            raise
        log(f"  Chat template tools param failed ({e}), injecting into system prompt")
        msg_copy = [m.copy() for m in messages]
        tool_text = format_tools_as_text(tools)
        if msg_copy and msg_copy[0]["role"] == "system":
            msg_copy[0]["content"] = msg_copy[0]["content"] + "\n\n" + tool_text
        else:
            msg_copy.insert(0, {"role": "system", "content": tool_text})
        return tokenizer.apply_chat_template(
            msg_copy, add_generation_prompt=True, tokenize=False, **extras)


def tokenize_messages(messages, tools=None):
    """Apply chat template and tokenize, with optional tool definitions."""
    if VISION:
        # Text tokens only; each image counts as its placeholder here. Used for
        # logs, count_tokens and the MLX_MAX_CONTEXT check.
        return tokenizer.encode(render_prompt(messages, tools), add_special_tokens=False)
    messages = template_messages(messages)
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
- When you call a tool, use the <tool_call> JSON format exactly as instructed. Do not wrap it in markdown.
- A message containing a <command-name> tag is a slash command invocation: call the Skill tool with skill set to the tag's contents (without the leading slash) and args set to any <command-args> content. Do not guess at what the skill does — call it."""

# Built-in Claude Code tools that signal a coding session and are worth keeping.
CODE_TOOLS_ALLOW = {
    "Bash",
    "Read",
    "Edit",
    "Write",
    "Grep",
    "Glob",
    "Skill",
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


_CWD_RE = re.compile(r"^\s*(?:-\s*)?(?:Primary working directory|Working directory|cwd)\s*:\s*(\S.*?)\s*$",
                     re.IGNORECASE | re.MULTILINE)


def working_directory(system):
    """The working directory Claude Code states in its own system prompt, or None."""
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    m = _CWD_RE.search(system or "")
    return m.group(1) if m else None


# Claude Code's /compact and auto-compact request: the whole conversation plus
# this instruction as the last user message (text from Claude Code 2.1.x).
COMPACT_MARKERS = (
    "Your task is to create a detailed summary of the conversation so far",
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
)
COMPACT_SYSTEM_PROMPT = (
    "You summarize a coding session so it can continue in a fresh context. "
    "Answer in plain text only. Never call a tool, never write a tool call.")


def looks_like_compaction(body):
    msgs = body.get("messages") or []
    if not msgs:
        return False
    content = msgs[-1].get("content")
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text")
    return isinstance(content, str) and any(m in content for m in COMPACT_MARKERS)


def optimize_for_compaction(body):
    """A compaction request gets a short system prompt and no tools.

    Through code mode it kept the tools and lost Claude Code's instruction
    framing, so a small model answered the "summarize" request with a Read
    call, and Claude Code failed the compaction ("summarization produced
    empty response") with the context already full."""
    body = dict(body)
    body["system"] = COMPACT_SYSTEM_PROMPT
    body["tools"] = []
    return body


def fit_compaction(messages, count, limit):
    """Make a compaction request fit under limit. Drop the oldest turns first,
    then cut the biggest remaining text in the middle. The system prompt, the
    first user message, the last message and the summarize instruction itself
    are always kept.
    A summary of most of the session beats a 400 that leaves it stuck full."""
    def protected(i, m):
        c = m.get("content")
        return (m.get("role") == "system" or i == first or i == len(msgs) - 1
                or (isinstance(c, str) and any(k in c for k in COMPACT_MARKERS)))

    msgs = list(messages)
    first = next((i for i, m in enumerate(msgs) if m.get("role") != "system"), 0)
    n = count(msgs)
    while n > limit:
        drop = next((i for i, m in enumerate(msgs) if not protected(i, m)), None)
        if drop is None:
            break
        del msgs[drop]
        n = count(msgs)
    # Still over: a protected message is too big on its own (Claude Code can
    # send a whole file read in the same message as the instruction, or as the
    # first message after an earlier compaction). Cut the biggest text in the
    # middle, never the instruction.
    while n > limit:
        sizes = [(len(m["content"]), i) for i, m in enumerate(msgs)
                 if isinstance(m.get("content"), str) and m.get("role") != "system"
                 and not any(k in m["content"] for k in COMPACT_MARKERS)
                 and not (i == len(msgs) - 1 and m.get("role") == "user")]
        if not sizes or max(sizes)[0] < 200:
            break
        size, i = max(sizes)
        keep = size // 4
        c = msgs[i]["content"]
        msgs[i] = dict(msgs[i], content=c[:keep] + "\n[... middle cut to fit the context ...]\n" + c[-keep:])
        n = count(msgs)
    return msgs, n


def code_system_prompt(tool_names=None, cwd=None):
    """CODE_SYSTEM_PROMPT, listing only the tools this request really offers
    (claude --bare sends Bash, Edit and Read; a small model told Write exists
    calls it and burns a turn on "No such tool"), plus the working directory."""
    prompt = CODE_SYSTEM_PROMPT
    if tool_names:
        lines = []
        for line in prompt.split("\n"):
            m = re.match(r"- (\w+): ", line)
            if m and m.group(1) in CODE_TOOLS_ALLOW and m.group(1) not in tool_names:
                continue
            lines.append(line)
        prompt = "\n".join(lines)
    if VISION and (not tool_names or "Read" in tool_names):
        # Without this a small model treats an image as binary and pokes at it
        # with `file` and PIL pixel dumps instead of just looking at it.
        prompt = prompt.replace(
            "- Read: read a file from disk (use absolute paths)",
            "- Read: read a file from disk (use absolute paths). Read also opens images "
            "(.png, .jpg, .gif, .webp) and you see the picture itself: to look at an "
            "image, Read it")
    if cwd:
        prompt += (f"\n\nWorking directory: {cwd}\nRelative paths are relative to it; "
                   f"absolute paths for files here start with {cwd.rstrip('/')}/")
    return prompt


def optimize_for_code(body):
    """Strip Claude Code bloat: replace the 10K-token harness prompt with a
    compact coding prompt and slim tool definitions to ~150 tokens total.

    The one line kept from the original is the working directory. Without it
    the prompt says "use absolute paths" and gives none, and a small model
    invents one (/Users/username/fizz.py) and writes the file somewhere else.
    """
    cwd = working_directory(body.get("system"))
    tools = body.get("tools", [])
    code_tools = [slim_tool(t) for t in tools if t.get("name", "") in CODE_TOOLS_ALLOW]
    body["system"] = code_system_prompt({t["name"] for t in code_tools}, cwd)
    if code_tools:
        stripped = len(tools) - len(code_tools)
        body["tools"] = code_tools
        log(f"  Code mode: working directory {cwd or 'not stated'}")
        log(f"  Code mode: {len(tools)} tools → {len(code_tools)} (stripped {stripped}, descriptions slimmed)")

    return body


# ─── Generation ──────────────────────────────────────────────────────────────

_first_request = True

# A model that meant to call a tool but produced something parse_tool_calls
# couldn't read usually leaves one of these fingerprints in the text. Shared by
# both backends (MLX here, llama.cpp in portable_server.py) so the retry trigger
# is identical everywhere.
TOOL_INTENT_PHRASES = [
    "here's the command", "bash(", "read(", "edit(", "write(",
    "<tool_call>", "<function=", "<tools>", '"name":', '"arguments":',
    '"parameters":', "```json", "```tool",
    # Gemma 4's closing marker: a call whose opener was lost and could not
    # be recovered still gets the re-prompt instead of going out as text.
    "<tool_call|>",
]


def coerce_args(args, props):
    """Fix argument types against the tool's schema.

    The XML tool-call format (<parameter=x>value</parameter>, Qwen 3.5 and
    Qwen Coder) carries every value as text, so a boolean arrives as "True"
    and Claude Code rejects the whole call ("expected boolean, provided as
    string"). Only plain string values whose schema type is not string are
    touched; anything that does not convert cleanly is left as it was.
    """
    out = {}
    for k, v in args.items():
        t = (props.get(k) or {}).get("type") if isinstance(props.get(k), dict) else None
        if isinstance(v, str) and t and t != "string":
            sv = v.strip()
            try:
                if t == "boolean" and sv.lower() in ("true", "false"):
                    v = sv.lower() == "true"
                elif t == "integer":
                    v = int(sv)
                elif t == "number":
                    v = float(sv) if any(c in sv for c in ".eE") else int(sv)
                elif t in ("array", "object"):
                    decoded = json.loads(sv)
                    if isinstance(decoded, list if t == "array" else dict):
                        v = decoded
            except ValueError:
                pass
        out[k] = v
    return out


PATH_KEYS = ("file_path", "path", "notebook_path")


def repair_paths(args, cwd):
    """Put back the leading "/" a small model drops from an absolute path.

    Qwen 3.5 2B writes "private/tmp/proj/notes.txt" for
    "/private/tmp/proj/notes.txt". Claude Code reads that as relative to the
    working directory, the file is not there, and the model retries the same
    call until it runs out of turns. Only a path that becomes one inside the
    working directory once the slash is restored is touched.
    """
    if not cwd:
        return args
    root = cwd.rstrip("/") + "/"
    out = dict(args)
    for k in PATH_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v and not v.startswith(("/", "~", ".")) and ("/" + v).startswith(root):
            out[k] = "/" + v
            log(f"  {k} {v!r} was missing its leading slash -> {out[k]!r}")
    return out


def fix_bash_timeout(name, args):
    """Bash's timeout is in milliseconds. Code mode slims away the description
    that says so, and models write seconds ("timeout": 5), which Claude Code
    runs as a 5 ms timeout: every command dies at once ("timed out after 0s")
    and a small model retries the same call until max turns. Nobody means a
    sub-second shell timeout, so a value under 1000 is read as seconds."""
    t = args.get("timeout")
    if name == "Bash" and isinstance(t, (int, float)) and not isinstance(t, bool) and 0 < t < 1000:
        fixed = min(int(t * 1000), 600000)
        log(f"  Bash timeout {t} read as seconds -> {fixed} ms")
        return dict(args, timeout=fixed)
    return args


# Tool names small models reach for that Claude Code calls something else.
# Applied only when the target is declared and the alias itself is not, so a
# client that really offers a "Run" tool still gets its own tool.
TOOL_NAME_ALIASES = {
    "run": "Bash", "execute": "Bash", "shell": "Bash", "terminal": "Bash",
    "runcommand": "Bash", "exec": "Bash",
    "createfile": "Write", "writefile": "Write", "save": "Write",
    "readfile": "Read", "open": "Read", "cat": "Read",
}
# Argument names that come with those aliases, moved to the real key only
# when the real key is missing.
TOOL_ARG_ALIASES = {
    "Bash": {"command": ("code", "script", "cmd")},
    "Write": {"file_path": ("path", "filename", "file"), "content": ("text", "contents", "data")},
    "Read": {"file_path": ("path", "filename", "file")},
}


def resolve_tool_name(name, declared):
    """The declared tool a call's name means, or the name unchanged.

    Tool names are case-sensitive in Claude Code, and a 0.8B model on the
    phone test (10/30) called "bash" 30+ times and "Run"/"Execute" for Bash,
    so every call was rejected and the file it wrote never ran. Exact match
    first, then one case-insensitive match, then TOOL_NAME_ALIASES. A name
    that matches no declared tool comes back as is."""
    if not isinstance(name, str) or not declared or name in declared:
        return name
    folded = [d for d in declared if d.lower() == name.lower()]
    if len(folded) == 1:
        log(f"  Tool name case fixed: {name} -> {folded[0]}")
        return folded[0]
    target = TOOL_NAME_ALIASES.get(re.sub(r'[\s_-]', '', name.lower()))
    if target in declared and not folded:
        log(f"  Tool name alias applied: {name} -> {target}")
        return target
    return name


def alias_tool_args(name, args):
    """Move code/script/cmd to command for Bash, path to file_path, etc."""
    if not isinstance(args, dict):
        return args
    args = dict(args)
    for real, alts in TOOL_ARG_ALIASES.get(name, {}).items():
        if real in args:
            continue
        for alt in alts:
            if alt in args:
                args[real] = args.pop(alt)
                log(f"  Tool argument alias applied: {name}.{alt} -> {real}")
                break
    return args


def assemble_anthropic_message(remaining_text, tool_calls, body, anthropic_tools,
                               prompt_tokens, gen_tokens, finish_reason="end_turn",
                               stop_sequence=None):
    """Turn cleaned text + parsed tool calls into an Anthropic Messages response.

    Backend-agnostic: the MLX path and the llama.cpp portable path both hand their
    parsed result here so the Claude-Code-2.1 quirk handling lives in one place —
    no empty leading text block (it makes Claude Code discard the tool_use that
    follows), and input keys the model hallucinated are dropped against the real
    tool schema (Claude Code silently rejects a tool_use with undeclared fields).
    """
    content_blocks = []

    if remaining_text.strip():
        content_blocks.append({"type": "text", "text": remaining_text.strip()})

    if tool_calls:
        current_tools = body.get("tools", []) or anthropic_tools or []
        tool_schemas = {}
        for atool in current_tools:
            schema = atool.get("input_schema") or {}
            props = schema.get("properties") or {}
            tool_schemas[atool.get("name", "")] = props
        if os.environ.get("MLX_DEBUG_RESPONSE"):
            log(f"  TOOL SCHEMAS: {tool_schemas}")

        for tc in tool_calls:
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            resolved = resolve_tool_name(tc.get("name"), tool_schemas)
            if resolved != tc.get("name"):
                tc["name"] = resolved
                tc["arguments"] = alias_tool_args(resolved, tc.get("arguments"))
            raw_input = tc["arguments"] if isinstance(tc.get("arguments"), dict) else {}
            allowed = tool_schemas.get(tc["name"])
            if allowed is not None:
                filtered = coerce_args(
                    {k: v for k, v in raw_input.items() if k in allowed}, allowed)
                filtered = fix_bash_timeout(tc["name"], filtered)
                filtered = repair_paths(filtered, working_directory(body.get("system")))
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
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": gen_tokens,
        }
    }

def count_prompt_tokens(body):
    """Token count for an Anthropic request body, same path inference takes.

    Deliberately runs the real convert + tokenize rather than estimating, so
    the number Claude Code budgets against is the number we will actually
    prefill. Cheap: tokenizing is milliseconds, no model involved.
    """
    anthropic_tools = body.get("tools", [])
    llm_tools = convert_tools_for_llm(anthropic_tools) if anthropic_tools else None
    return len(tokenize_messages(convert_messages(body, vision=VISION, tool_ids=True), tools=llm_tools))


class PromptTooLong(Exception):
    """Prompt is over MLX_MAX_CONTEXT. Answered with the API's own 400 so
    Claude Code treats it like any over-long prompt and compacts."""


def check_context(prompt_tokens, limit=None):
    limit = MAX_CONTEXT if limit is None else limit
    if limit and prompt_tokens > limit:
        raise PromptTooLong(f"prompt is too long: {prompt_tokens} tokens > {limit} maximum")


# Under MLX_MAX_CONTEXT a prompt must leave at least this much room to answer
# (or the whole max_tokens, if that is smaller); otherwise it is "too long".
MIN_ANSWER_ROOM = 1024


def fit_max_tokens(prompt_tokens, max_tokens, limit=None):
    """max_tokens trimmed so prompt + answer stays inside MLX_MAX_CONTEXT. A prompt
    that leaves less than MIN_ANSWER_ROOM gets PromptTooLong, so Claude Code
    compacts instead of getting a cut-off answer."""
    limit = MAX_CONTEXT if limit is None else limit
    check_context(prompt_tokens, limit)
    if not limit:
        return max_tokens
    room = limit - prompt_tokens
    if room < min(max_tokens, MIN_ANSWER_ROOM):
        raise PromptTooLong(f"prompt is too long: {prompt_tokens} tokens leaves {room} of "
                            f"{limit} for the answer")
    return min(max_tokens, room)


def generate_response(body, on_start=None, on_text=None):
    """Run MLX inference and return Anthropic-formatted response.

    on_start(prompt_tokens) / on_text(chunk) are optional live-streaming
    hooks (issue #39): on_start fires once right before generation begins,
    on_text fires with each thinking-filtered text chunk as it is generated.
    """
    global _first_request

    # In browser mode, strip Claude Code bloat before inference.
    # Otherwise, auto-detect Claude Code coding sessions and apply code mode.
    compaction = looks_like_compaction(body)
    if compaction:
        body = optimize_for_compaction(body)
        log("  Compaction request: text only, no tools")
    elif BROWSER_MODE:
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

    messages = convert_messages(body, vision=VISION, tool_ids=True)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.2)
    log(f"  Sampling: temperature {temperature}"
        + ("" if "temperature" in body else " (server default; request set none)"))

    if llm_tools:
        log(f"  Tools: {len(llm_tools)} ({', '.join(t['function']['name'] for t in llm_tools[:5])}{'...' if len(llm_tools) > 5 else ''})")

    # Tokenize (with tools if present)
    token_ids = tokenize_messages(messages, tools=llm_tools)
    # A compaction request is trimmed to fit, leaving room for the summary.
    compact_budget = MAX_CONTEXT - min(max_tokens, MIN_ANSWER_ROOM) if MAX_CONTEXT else 0
    if compaction and MAX_CONTEXT and len(token_ids) > compact_budget:
        before = len(messages)
        messages, _ = fit_compaction(
            messages, lambda m: len(tokenize_messages(m, tools=llm_tools)), compact_budget)
        token_ids = tokenize_messages(messages, tools=llm_tools)
        log(f"  Compaction over MLX_MAX_CONTEXT: dropped {before - len(messages)} oldest "
            f"messages to fit ({len(token_ids)} tokens)")

    # Pre-fill an empty thinking block so the model skips its reasoning chain entirely.
    # Gemma 4 generates 300-500 thinking tokens per request by default — this cuts them out.
    thinking_prefill = ""
    if SUPPRESS_THINKING and "gemma" in MODEL_PATH.lower():
        thinking_prefill = "<|channel>thought\n<channel|>"
        skip = tokenizer.encode(thinking_prefill, add_special_tokens=False)
        token_ids = list(token_ids) + list(skip)
        log(f"  Thinking suppressed (+{len(skip)} prefill tokens)")

    prompt_tokens = len(token_ids)
    log(f"  Prompt: {prompt_tokens} tokens")
    fitted = fit_max_tokens(prompt_tokens, max_tokens)
    if fitted != max_tokens:
        log(f"  max_tokens {max_tokens} -> {fitted} to stay inside MLX_MAX_CONTEXT")
        max_tokens = fitted

    def run_generation(msgs, prompt_ids, kwargs):
        """One generation pass. Vision: mlx-vlm, whole prompt, images from the
        newest message. Text: mlx-lm with the shared prompt cache."""
        if VISION:
            images = [shrink_image(i, MAX_IMAGE_SIDE) for i in collect_images(msgs)]
            if images:
                log(f"  Vision: {len(images)} image(s) in this turn"
                    + (f", longest side capped at {MAX_IMAGE_SIDE}px" if MAX_IMAGE_SIDE else ""))
            return vlm_stream_generate(
                model, processor, render_prompt(msgs, llm_tools) + thinking_prefill,
                image=images or None, max_tokens=max_tokens,
                temperature=temperature, prefill_step_size=PREFILL_SIZE)
        return stream_generate(model=model, tokenizer=tokenizer, prompt=prompt_ids,
                               max_tokens=max_tokens, **kwargs)

    # ─── Prompt cache: reuse KV for shared prefix tokens ───
    global _prompt_cache, _cached_token_prefix

    # Check if cache type supports safe trim+reuse (standard KVCache only,
    # RotatingKVCache from Gemma 4 has a circular buffer that breaks on trim+extend)
    from mlx_lm.models.cache import RotatingKVCache
    cache_is_safe = (not VISION and _prompt_cache is not None
                     and not isinstance(_prompt_cache[0], RotatingKVCache))

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

    # .offset is the live token count; .step is a fixed 256-token allocation
    # increment. Reading .step made trim_amount negative for any prefix longer
    # than 256 tokens, so the trim never ran and the cache kept the previous
    # session's KV state (issue #46). Caches without .offset can't be trimmed
    # safely at all, so fall through to a full prefill instead of guessing.
    cache_offset = getattr(_prompt_cache[0], "offset", None) if _prompt_cache else None
    if cache_hit_len > 0 and cache_offset is None:
        log("  Cache has no offset (untrimmable type) — full prefill")
        cache_hit_len = 0

    if cache_hit_len > 0:
        # Trim cache back to the shared prefix, then only prefill the delta
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
        if VISION:
            pass  # mlx-vlm prefills the whole prompt; see the VISION note below
        elif _prompt_cache is not None and isinstance(_prompt_cache[0], RotatingKVCache):
            log(f"  RotatingKVCache: fresh cache each request (no trim support)")
        else:
            log(f"  Cache miss: full prefill of {prompt_tokens} tokens")
        _prompt_cache = None
        prompt_for_gen = token_ids

    # Build generation kwargs — always pass a prompt_cache so we can reuse it
    if VISION:
        # mlx-vlm builds its own cache per request. The models this path is for
        # (Qwen 3.5) mix linear-attention layers into the stack, and that state
        # cannot be trimmed back to a shared prefix anyway.
        _prompt_cache = None
    elif _prompt_cache is None:
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
    last_response = None
    finish_reason = "end_turn"
    stop_sequences = [x for x in body.get("stop_sequences") or [] if isinstance(x, str) and x]
    stop_sequence = None
    t0 = time.time()

    def find_stop(text, start=0):
        """Earliest (index, sequence) of a stop sequence at or after start."""
        hits = [(text.find(x, start), x) for x in stop_sequences if text.find(x, start) >= 0]
        return min(hits) if hits else None

    if on_start:
        on_start(prompt_tokens)

    with generate_lock:
        try:
            for response in run_generation(messages, prompt_for_gen, gen_kwargs):
                chunk = tf.feed(response.text)
                full_text += chunk
                if on_text and chunk:
                    on_text(chunk)
                gen_tokens = response.generation_tokens
                last_response = response
                if response.finish_reason == "length":
                    finish_reason = "max_tokens"
                elif response.finish_reason == "stop":
                    finish_reason = "end_turn"
                # Anthropic stop_sequences: cut the text at the earliest match
                # and stop generating. Only the tail can hold a new match.
                hit = stop_sequences and chunk and find_stop(
                    full_text, max(0, len(full_text) - len(chunk) - max(map(len, stop_sequences))))
                if hit:
                    full_text, stop_sequence = full_text[:hit[0]], hit[1]
                    finish_reason = "stop_sequence"
                    break
        finally:
            # Give the prefill transients back before the next request (or the
            # error handler) runs. In a finally so a failed generation cannot
            # leave the pool inflated and get us evicted for memory we are no
            # longer using.
            release_transients()

    _tail = "" if stop_sequence else tf.flush()
    full_text += _tail
    if on_text and _tail:
        on_text(_tail)
    hit = stop_sequences and not stop_sequence and find_stop(full_text)
    if hit:
        full_text, stop_sequence = full_text[:hit[0]], hit[1]
        finish_reason = "stop_sequence"

    # Cache is updated in-place by MLX — save the token prefix for next request's diff
    _cached_token_prefix = token_ids

    # An empty completion straight after a cache hit is almost always corrupt
    # KV state, not a real answer. Say so, and drop the cache so the next
    # request self-heals with a clean prefill instead of failing silently.
    if cache_hit_len > 0 and not full_text.strip():
        log(f"  WARNING: empty completion after a {cache_hit_len}-token cache hit "
            f"— discarding prompt cache (see issue #46)")
        _prompt_cache = None
        _cached_token_prefix = None

    elapsed = time.time() - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generated: {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
    # Both mlx-lm and mlx-vlm report prefill and decode speed separately; the
    # line above blends them, which hides a slow prefill behind a fast decode.
    p_tps = getattr(last_response, "prompt_tps", None)
    g_tps = getattr(last_response, "generation_tps", None)
    if isinstance(p_tps, (int, float)) and isinstance(g_tps, (int, float)):
        log(f"  Speed: prefill {p_tps:.0f} tok/s, decode {g_tps:.1f} tok/s")

    # Clean output (preserves <tool_call> tags)
    text = clean_response(full_text)

    # Parse tool calls from model output. A request that declared no tools
    # gets the text as written: otherwise a JSON answer could come back as a
    # tool_use for a tool the client never offered.
    if llm_tools:
        tool_calls, remaining_text = parse_tool_calls(text, llm_tools)
    else:
        tool_calls, remaining_text = [], text

    # ─── Retry logic: if model expressed intent to use a tool but we got no valid calls ───
    # llm_tools guard: a request with no tools can never need a tool-call
    # retry — without it, chat answers containing '```json' etc. triggered
    # a pointless (and confusing) re-generation.
    if llm_tools and not tool_calls and any(p in remaining_text.lower() for p in TOOL_INTENT_PHRASES):
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
            # mlx_lm appends to a prompt cache in place, and the shared one
            # already holds the first attempt's prompt and output, so the full
            # retry prompt would land after them. Run the retry on a throwaway
            # cache instead. The shared cache and its token prefix are left
            # exactly as the first attempt left them, so the next request
            # reuses the same KV state it always did.
            retry_kwargs = (gen_kwargs if VISION
                            else dict(gen_kwargs, prompt_cache=make_prompt_cache(model)))
            with generate_lock:
                try:
                    for response in run_generation(retry_messages, retry_tokens, retry_kwargs):
                        retry_text += retry_tf.feed(response.text)
                        retry_gen = response.generation_tokens
                finally:
                    release_transients(" (retry)")
            retry_text += retry_tf.flush()

            retry_text = clean_response(retry_text)
            retry_calls, retry_remaining = parse_tool_calls(retry_text, llm_tools)
            gen_tokens += retry_gen

            if retry_calls:
                tool_calls = retry_calls
                # Preserve original reasoning text, not retry text
                log(f"  Retry succeeded: {', '.join(tc['name'] for tc in retry_calls)}")
                break
            else:
                log(f"  Retry {retry+1} failed, still no valid tool call")

    # Build the Anthropic response (shared with the llama.cpp portable backend).
    return assemble_anthropic_message(
        remaining_text, tool_calls, body, anthropic_tools,
        prompt_tokens, gen_tokens, finish_reason, stop_sequence,
    )


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


def send_anthropic_stream_live(handler, body):
    """Stream tokens to the client AS THEY ARE GENERATED (issue #39).

    Used only for streaming requests with no tools. The tool path must
    buffer the complete response to parse (and possibly retry) tool calls,
    but plain chat has no reason to wait — before this, time-to-first-token
    equaled full-response latency because send_anthropic_stream replayed a
    finished message.

    A holdback buffer keeps the last few chars unsent so any leaked stop
    marker (the ones clean_response truncates at) never reaches the client,
    even when it straddles a chunk boundary.
    """
    STOP_MARKERS = ['<turn|>', '<|turn>', '<|im_end|>', '<|endoftext|>',
                    '<|im_start|>', '<|end_of_text|>', '<|eot_id|>']
    STOP_MARKERS += [x for x in body.get("stop_sequences") or [] if isinstance(x, str) and x]
    HOLDBACK = max(24, max(map(len, STOP_MARKERS)))  # >= longest stop marker

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = body.get("model", "claude-sonnet-4-6")
    t0 = time.time()
    state = {"first": None, "lead": True, "buf": "", "done": False, "raw": ""}
    # The buffered path drops <think>...</think> in clean_response; this path
    # never runs that, so filter the same blocks out as they stream. "raw"
    # keeps the unfiltered text in case the filter ends up holding everything.
    think = ThinkingFilter()
    think.THINK_START, think.THINK_END = "<think>", "</think>"

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()

    def emit(event_type, payload):
        handler.wfile.write(f"event: {event_type}\n".encode())
        handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        handler.wfile.flush()

    def emit_delta(text):
        if state["lead"]:
            text = text.lstrip()
            if not text:
                return
            state["lead"] = False
        if state["first"] is None:
            state["first"] = time.time() - t0
        emit("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        })

    def on_start(prompt_tokens):
        emit("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            },
        })
        emit("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

    def on_text(chunk):
        if state["done"]:
            return
        state["raw"] += chunk
        state["buf"] += think.feed(chunk)
        cut = -1
        for m in STOP_MARKERS:
            i = state["buf"].find(m)
            if i >= 0 and (cut < 0 or i < cut):
                cut = i
        if cut >= 0:
            out = state["buf"][:cut].rstrip()
            state["buf"] = ""
            state["done"] = True
            if out:
                emit_delta(out)
            return
        if len(state["buf"]) > HOLDBACK:
            out = state["buf"][:-HOLDBACK]
            state["buf"] = state["buf"][-HOLDBACK:]
            emit_delta(out)

    try:
        result = generate_response(body, on_start=on_start, on_text=on_text)
    except (BrokenPipeError, ConnectionResetError):
        log("  live stream: client disconnected mid-generation")
        return
    except PromptTooLong as e:
        log(f"  ← 400 {e}")
        try:
            emit("error", {"type": "error",
                           "error": {"type": "invalid_request_error", "message": str(e)}})
        except Exception:
            pass
        return
    except Exception as e:
        log(f"  live stream ERROR: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        try:
            emit("error", {"type": "error",
                           "error": {"type": "api_error", "message": str(e)}})
        except Exception:
            pass
        return

    try:
        if not state["done"]:
            # The think filter holds back its last few chars until flush, so a
            # stop marker at the very end of the output only shows up here.
            state["buf"] += think.flush()
            cuts = [state["buf"].find(m) for m in STOP_MARKERS if m in state["buf"]]
            if cuts:
                state["buf"] = state["buf"][:min(cuts)]
        if state["first"] is None and not state["buf"].strip():
            # Nothing got past the filter: a <think> block that never closed
            # (the model hit max_tokens mid-thought) or one with nothing after
            # it. Send the unfiltered text, as strip_think_tags does for the
            # buffered path, instead of an empty reply.
            raw = state["raw"]
            cuts = [raw.find(m) for m in STOP_MARKERS if m in raw]
            state["buf"], state["done"] = raw[:min(cuts)] if cuts else raw, False
        if not state["done"] and state["buf"]:
            tail = state["buf"].rstrip()
            state["buf"] = ""
            if tail:
                emit_delta(tail)
        emit("content_block_stop", {"type": "content_block_stop", "index": 0})
        emit("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": result.get("stop_reason"),
                "stop_sequence": result.get("stop_sequence"),
            },
            "usage": {"output_tokens": result["usage"]["output_tokens"]},
        })
        emit("message_stop", {"type": "message_stop"})
    except (BrokenPipeError, ConnectionResetError):
        log("  live stream: client disconnected at finalize")
        return

    ttft = state["first"]
    total = time.time() - t0
    if ttft is not None:
        log(f"  ← OK live-streamed ({result['usage']['output_tokens']} tok) "
            f"first token {ttft:.2f}s, done {total:.2f}s")
    else:
        log(f"  ← OK live-streamed (empty response, {total:.2f}s)")


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
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            # A bad body used to raise right here, before any response was
            # sent, so the client only saw the connection drop.
            log(f"POST {self.path} invalid JSON body")
            send_json(self, 400, {"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "request body must be a JSON object"}})
            return
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
                # Live token streaming for tool-less requests (issue #39).
                # Tool requests keep the buffer-parse-retry path below.
                # Opt out with MLX_LIVE_STREAM=0.
                if (body.get("stream") and not body.get("tools")
                        and os.environ.get("MLX_LIVE_STREAM", "1").lower() not in ("0", "false")):
                    send_anthropic_stream_live(self, body)
                    return

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
            except PromptTooLong as e:
                log(f"  ← 400 {e}")
                send_json(self, 400, {"type": "error", "error": {
                    "type": "invalid_request_error", "message": str(e)}})
            except Exception as e:
                log(f"  ← ERROR: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                send_json(self, 500, {"type": "error",
                                      "error": {"type": "api_error", "message": str(e)}})
        elif path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            # Claude Code asks for this before every turn and used to fall into
            # the "Unknown POST" branch, which answers {} — no input_tokens at
            # all. Answer it properly and its context meter has something real
            # to work from. Runs the same convert + tokenize inference does, so
            # the number it budgets against is the number we will prefill.
            try:
                n = count_prompt_tokens(body)
            except Exception as e:
                log(f"  count_tokens failed ({e}) — falling back to a char estimate")
                n = len(json.dumps(body)) // 4
            log(f"  count_tokens: {n}")
            send_json(self, 200, {"input_tokens": n})

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
            # The memory block is here so `curl :4000/health` answers "how close
            # to the edge is this box?" without attaching to the process.
            # mx_peak_gb is the honest one; see mem_snapshot on why not rss.
            payload = {"status": "ok", "version": __version__, "model": MODEL_PATH,
                       "prefill_size": PREFILL_SIZE, "vision": VISION,
                       "max_context": MAX_CONTEXT}
            payload.update(mem_snapshot())
            send_json(self, 200, payload)
        else:
            send_json(self, 200, {})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  MLX Native Anthropic Server                    ║")
    print("║  Claude Code → MLX → Apple Silicon (direct)     ║")
    print("║  Tool use: enabled (Anthropic ↔ Llama native)   ║")
    print("║  Prompt caching: enabled (KV reuse)             ║")
    print(f"║  claude-code-local v{__version__:<28}║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    load_model()

    print()
    print(f"Serving Anthropic Messages API on http://localhost:{PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"KV cache: {KV_BITS}-bit quantization (start at token {KV_QUANT_START})" if KV_BITS else "KV cache: full precision")
    print(f"Prefill chunk: {PREFILL_SIZE} tokens"
          + (f" · MLX buffer cache capped at {MEM_CACHE_LIMIT_GB:g}GB" if MEM_CACHE_LIMIT_GB > 0 else ""))
    print(f"Memory now: {mem_snapshot()}")
    print("Vision: on (mlx-vlm), images reach the model" if VISION
          else "Prompt cache: enabled (KV reuse across requests)")
    if MAX_CONTEXT:
        print(f"Context ceiling: {MAX_CONTEXT} tokens (MLX_MAX_CONTEXT)")
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
