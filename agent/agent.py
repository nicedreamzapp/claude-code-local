#!/usr/bin/env python3
"""The native engine — a self-contained terminal coding agent, one file.

A launcher (or you, directly) runs this file with env vars picking the model,
the backend, and the display name. No Claude Code binary involved: the system
prompt is ~550 tokens instead of tens of thousands, and it never mutates, so
the KV cache survives from turn to turn.

Backends:
  AGENT_BACKEND=mlx   — load the model INTO this process via MLX. No server,
                        no HTTP hop, nothing on the network. KV cache reuse
                        across turns keeps replies sub-second.
  AGENT_BACKEND=http  — talk to any local Anthropic-protocol /v1/messages
                        server (e.g. this repo's proxy on :4000). The server
                        owns the model and its memory.

MLX tool-call dialects:
  AGENT_DIALECT=native    — the model's own chat template defines tool calls
                            (Qwen3-Coder: XML with RAW TEXT parameter values —
                            no JSON escaping, which is the whole reason this
                            agent exists).
  AGENT_DIALECT=prompted  — model has no usable native format (Gemma 4's is a
                            custom pseudo-JSON that re-introduces escaping);
                            we teach the same XML dialect in the system prompt
                            and reuse the same parser.

Env (all optional unless a launcher needs it):
  AGENT_MODEL  AGENT_TITLE  AGENT_BACKEND  AGENT_DIALECT
  AGENT_BASE_URL  AGENT_AUTH_TOKEN                  (http backend)
  AGENT_LEASE_NAME  AGENT_LEASE_GB                  (mlx backend; 0 = no seat)
  AGENT_SPEAK=1        speak every final answer through ~/.local/bin/speak
  AGENT_PROMPT_FILE    extra system-prompt text, read once at startup
  AGENT_MAX_TOKENS  AGENT_TEMP  AGENT_MAX_STEPS  AGENT_BASH_TIMEOUT
"""

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import readline  # arrow keys, in-line editing, history for input()
except ImportError:
    readline = None

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_DEFAULT = os.environ.get(
    "AGENT_MODEL", "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit"
)
BACKEND = os.environ.get("AGENT_BACKEND", "mlx")
DIALECT = os.environ.get("AGENT_DIALECT", "native")
BASE_URL = os.environ.get("AGENT_BASE_URL", "").rstrip("/")
AUTH_TOKEN = os.environ.get("AGENT_AUTH_TOKEN", "sk-local")
LEASE_NAME = os.environ.get("AGENT_LEASE_NAME", "agent-local")
LEASE_GB = float(os.environ.get("AGENT_LEASE_GB", "36" if BACKEND == "mlx" else "0"))
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))
TEMP = float(os.environ.get("AGENT_TEMP", "0.3"))
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "40"))
BASH_TIMEOUT = int(os.environ.get("AGENT_BASH_TIMEOUT", "120"))
SPEAK = os.environ.get("AGENT_SPEAK", "") in ("1", "true", "yes")
PROMPT_FILE = os.environ.get("AGENT_PROMPT_FILE", "")
OUT_LIMIT = 30000  # chars of tool output fed back to the model

TITLE = os.environ.get("AGENT_TITLE") or MODEL_DEFAULT.rsplit("/", 1)[-1]

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
OFF = "\033[0m"


def c(text, color):
    return f"{color}{text}{OFF}" if sys.stdout.isatty() else str(text)


# ─── Tools ───────────────────────────────────────────────────────────────────
#
# Descriptions stay terse on purpose. They are re-serialized into the prompt on
# every single turn, so a paragraph of prose per tool is a paragraph re-prefilled
# for the life of the session. The model already knows what `grep` is.

TOOLS = [
    {
        "name": "Bash",
        "description": (
            "Run a shell command and return its combined stdout/stderr. "
            "Non-interactive (stdin is closed) — to hand the user an "
            "interactive program, use `open`, never run it here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file. Returns the contents with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path."},
                "offset": {"type": "integer", "description": "First line (1-indexed)."},
                "limit": {"type": "integer", "description": "How many lines to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating or overwriting it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Full file contents."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace an exact substring in a file. old_string must appear "
            "exactly once unless replace_all is true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to edit."},
                "old_string": {"type": "string", "description": "Exact text to find."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every match."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "Glob",
        "description": "List files matching a glob pattern, newest first.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "e.g. src/**/*.py"},
                "path": {"type": "string", "description": "Directory to search in."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with ripgrep. Returns matching lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex to search for."},
                "path": {"type": "string", "description": "File or directory."},
                "glob": {"type": "string", "description": "Filter files, e.g. *.py"},
            },
            "required": ["pattern"],
        },
    },
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def _clip(text, limit=OUT_LIMIT):
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - limit
    return f"{text[:half]}\n\n... [{dropped} chars elided] ...\n\n{text[-half:]}"


def tool_bash(command):
    # stdin closed on purpose: an interactive program (a game, a REPL, vim)
    # would otherwise inherit the terminal and silently fight the user for
    # keystrokes while its output is captured — invisible chaos for up to
    # BASH_TIMEOUT. Closed stdin makes it EOF/crash out in milliseconds.
    r = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        timeout=BASH_TIMEOUT, stdin=subprocess.DEVNULL,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        out += f"\n[exit {r.returncode}]"
    return _clip(out.strip()) or "[no output]"


def tool_read(path, offset=None, limit=None):
    p = Path(path).expanduser()
    lines = p.read_text(errors="replace").splitlines()
    start = (offset - 1) if offset else 0
    end = start + limit if limit else len(lines)
    chunk = lines[start:end]
    if not chunk:
        return "[empty or past end of file]"
    width = len(str(start + len(chunk)))
    return _clip("\n".join(f"{start+i+1:>{width}}  {ln}" for i, ln in enumerate(chunk)))


def tool_write(path, content):
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {p} ({content.count(chr(10))+1} lines)."


def tool_edit(path, old_string, new_string, replace_all=False):
    p = Path(path).expanduser()
    text = p.read_text()
    n = text.count(old_string)
    if n == 0:
        return f"ERROR: old_string not found in {p}. Read the file and match it exactly."
    if n > 1 and not replace_all:
        return f"ERROR: old_string appears {n} times in {p}. Add more context or set replace_all."
    p.write_text(
        text.replace(old_string, new_string)
        if replace_all
        else text.replace(old_string, new_string, 1)
    )
    return f"Edited {p} ({n if replace_all else 1} replacement(s))."


def tool_glob(pattern, path=None):
    root = Path(path).expanduser() if path else Path.cwd()
    hits = [f for f in root.glob(pattern) if f.is_file()]
    hits.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if not hits:
        return "[no matches]"
    return _clip("\n".join(str(f) for f in hits[:300]))


def tool_grep(pattern, path=None, glob=None):
    cmd = ["rg", "-n", "--no-heading", "--color=never", "-S"]
    if glob:
        cmd += ["--glob", glob]
    cmd += [pattern, path or "."]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return "ERROR: ripgrep (rg) is not installed."
    if r.returncode == 1:
        return "[no matches]"
    return _clip(r.stdout.strip() or r.stderr.strip() or "[no matches]")


DISPATCH = {
    "Bash": tool_bash,
    "Read": tool_read,
    "Write": tool_write,
    "Edit": tool_edit,
    "Glob": tool_glob,
    "Grep": tool_grep,
}


def run_tool(name, args):
    fn = DISPATCH.get(name)
    if not fn:
        return f"ERROR: no tool named {name}. Available: {', '.join(DISPATCH)}"
    try:
        return fn(**args)
    except subprocess.TimeoutExpired:
        return f"ERROR: {name} timed out after {BASH_TIMEOUT}s."
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except FileNotFoundError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ─── XML tool-call dialect (shared by both MLX modes) ───────────────────────
#
# Values are raw text between the tags, so we deliberately do NOT json.loads
# them — we only coerce to the declared JSON-schema type, and fall back to the
# raw string when coercion fails rather than throwing the call away.

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Parse on <function=…> alone, NOT the <tool_call> wrapper: the model drops the
# opening <tool_call> often enough that anchoring on it loses real calls
# (observed first launch-test, 2026-08-07 — the call printed as raw XML and
# nothing ran). The wrapper is decoration; the function block is the call.
FUNC_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def coerce(value, declared_type):
    v = value.strip()
    if declared_type == "integer":
        try:
            return int(v)
        except ValueError:
            return v
    if declared_type == "number":
        try:
            return float(v)
        except ValueError:
            return v
    if declared_type == "boolean":
        return v.lower() in ("true", "1", "yes")
    if declared_type in ("array", "object"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return value  # string: preserve exactly, including leading whitespace


def parse_tool_calls(text):
    """Return [(name, args_dict), ...] found in a model turn."""
    calls = []
    for name, body in FUNC_RE.findall(text):
        name = name.strip()
        schema = TOOLS_BY_NAME.get(name, {}).get("parameters", {}).get("properties", {})
        args = {}
        for pname, pval in PARAM_RE.findall(body):
            pname = pname.strip()
            args[pname] = coerce(pval, schema.get(pname, {}).get("type", "string"))
        calls.append((name, args))
    return calls


def strip_calls(text):
    text = FUNC_RE.sub("", text)
    return text.replace("<tool_call>", "").replace("</tool_call>", "").strip()


def preview(name, args):
    """One-line rendering of a call, so the terminal shows intent not XML."""
    if name == "Bash":
        return args.get("command", "")
    if name in ("Read", "Write", "Edit"):
        return args.get("path", "")
    if name == "Glob":
        return args.get("pattern", "")
    if name == "Grep":
        pat = args.get("pattern", "")
        where = args.get("path", "")
        return f"{pat}  {DIM}{where}{OFF}" if where else pat
    return json.dumps(args)[:120]


def tools_doc():
    """XML tool docs for the prompted dialect — models without a usable native
    format get the same dialect described in the system prompt instead."""
    lines = [
        "You have these tools. To use one, emit EXACTLY this XML (values are "
        "RAW TEXT between the tags — no JSON, no quoting, no escaping):",
        "",
        "<tool_call>",
        "<function=ToolName>",
        "<parameter=arg_name>",
        "value",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "",
        "One call at a time. Stop after </function> and wait for the result "
        "(it arrives in a <tool_result> block). Available tools:",
        "",
    ]
    for t in TOOLS:
        props = t["parameters"]["properties"]
        req = set(t["parameters"].get("required", []))
        args = ", ".join(
            f"{n}{'' if n in req else '?'}: {p.get('type','string')}"
            for n, p in props.items()
        )
        lines.append(f"- {t['name']}({args}) — {t['description']}")
    return "\n".join(lines)


# ─── Optional memory-broker seat (mlx backend only) ──────────────────────────
#
# If you run a local memory broker (a daemon that arbitrates RAM between big
# model processes), point AGENT_MEM_BROKER_DIR at a directory containing its
# mem_client.py and the engine will ask for room before loading ~30GB of
# weights, and release it on every exit path. Without the env var this whole
# section is inert — the engine just loads the model.

MEM_DIR = Path(os.environ.get("AGENT_MEM_BROKER_DIR", "/nonexistent")).expanduser()
_mem = None  # the imported mem_client module, when the guard's client exists
_lease_id = None
_hb_stop = None


def _load_mem_client():
    global _mem
    if _mem is None and (MEM_DIR / "mem_client.py").exists():
        sys.path.insert(0, str(MEM_DIR))
        import mem_client

        _mem = mem_client
    return _mem


def _reap_stale_seats():
    """Drop seats under our lease name billed to a process that no longer
    exists. A seat leaks whenever the shell dies between grant and hand-off —
    the guard keeps billing it for the full 30-minute TTL and the next launch
    is refused for room it already owns. `used_gb: 0` with no live process is
    the tell."""
    st = _mem.state() or {}
    for lease in st.get("leases", []):
        if lease.get("name") == LEASE_NAME and lease.get("used_gb", 0) <= 0.5:
            _mem.release(lease["id"])
            print(c(f"  reclaimed orphaned {lease['gb']:.0f}GB seat {lease['id']}", DIM))


def acquire_seat():
    global _lease_id, _hb_stop
    if LEASE_GB <= 0 or not _load_mem_client():
        return
    _reap_stale_seats()
    print(c(f"  reserving {LEASE_GB:.0f}GB…", DIM), end="", flush=True)
    _lease_id, status, _ = _mem.acquire_ex(LEASE_NAME, LEASE_GB, timeout=600)
    if _lease_id:
        print(c(" ok", DIM))
    elif status == "refused":
        # The room genuinely isn't there. Loading 30GB of weights anyway ends
        # in swap death — stop and say why instead.
        print(c(" refused", RED))
        print(f"\n  Not enough free memory for this model right now.")
        print(f"  Quit other model servers or heavy apps, then relaunch.\n")
        if sys.stdin.isatty():
            try:
                input("  press Return to close… ")
            except (EOFError, KeyboardInterrupt):
                pass
        sys.exit(1)
    else:
        print(c(" no guard — proceeding unguarded", DIM))
    if _lease_id:
        # The guard's TTL is 30 minutes. A session left open longer than that
        # loses its seat, its `asked` protection with it, and the process goes
        # back to being an evictable 30GB "unregistered" hog — so heartbeat for
        # the life of the process, exactly like mem_client.reserve() does.
        import threading

        _hb_stop = threading.Event()

        def beat():
            while not _hb_stop.wait(600):
                _mem.heartbeat(_lease_id, 1800)

        threading.Thread(target=beat, daemon=True).start()


def release_seat(*_):
    global _lease_id
    if _hb_stop:
        _hb_stop.set()
    if _lease_id:
        _mem.release(_lease_id)
        _lease_id = None


# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM = """You are {title}, a coding agent working in a terminal on macOS (Apple Silicon).
You are the model {model}, running {where}. If asked what model you are, say so.

Working directory: {cwd}

Use the tools to inspect and change real files. Prefer Read before Edit, and
match old_string exactly as it appears on disk. Chain tools until the task is
genuinely done, then give a short plain answer — no preamble, no summary of
what you are about to do.

Be concise. The user is an expert; skip explanations they did not ask for."""


def build_system(model_path):
    where = (
        "fully locally in-process via MLX on this machine — no cloud, no "
        "external services" if BACKEND == "mlx"
        else f"behind a local gateway at {BASE_URL}"
    )
    text = SYSTEM.format(title=TITLE, model=model_path, where=where, cwd=os.getcwd())
    if BACKEND == "mlx" and DIALECT == "prompted":
        text += "\n\n" + tools_doc()
    if PROMPT_FILE and Path(PROMPT_FILE).expanduser().exists():
        text += "\n\n" + Path(PROMPT_FILE).expanduser().read_text()
    return text


# ─── MLX engine (in-process) ────────────────────────────────────────────────


class MLXEngine:
    """Holds the model, the conversation, and the KV cache across turns.

    The cache is the reason this is fast. mlx_lm keeps KV state for a token
    prefix; as long as the prefix does not change we only prefill the tokens we
    actually added. Re-rendering the chat template each turn produces a prompt
    that is byte-identical up to the new message, so the common prefix is
    everything but the delta — we trim the cache back to the divergence point
    and prefill only what is new. A prompt that mutates its own head (injected
    timestamps, reminders, shuffled context) would defeat this entirely, which
    is why the system prompt is fixed at startup and never touched again.
    """

    def __init__(self, model_path):
        from mlx_lm.utils import load
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        t0 = time.time()
        print(c(f"  loading {model_path}…", DIM), end="", flush=True)
        self.model, self.tokenizer = load(model_path)
        print(c(f" {time.time()-t0:.1f}s", DIM))

        self._make_cache = make_prompt_cache
        self.cache = self._new_cache()
        self.cache_tokens = []
        self.sampler = make_sampler(temp=TEMP)
        self.model_path = model_path
        self.messages = [{"role": "system", "content": build_system(model_path)}]
        self.ctx_limit = self._detect_ctx_limit()

    def _detect_ctx_limit(self):
        """The model's context window, in tokens. AGENT_CTX_LIMIT overrides;
        otherwise probe the loaded config, else assume 128k."""
        env = os.environ.get("AGENT_CTX_LIMIT")
        if env:
            return int(env)
        for holder in (getattr(self.model, "args", None),
                       getattr(self.model, "config", None)):
            for key in ("max_position_embeddings", "max_context_length",
                        "context_length"):
                v = getattr(holder, key, None)
                if isinstance(v, int) and 1024 <= v <= 10_000_000:
                    return v
        v = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(v, int) and 1024 <= v <= 10_000_000:
            return v
        return 131072

    def ctx_used(self):
        """(tokens in context now, context limit). cache_tokens IS the live
        transcript token-for-token — prompt plus everything generated — so no
        re-tokenization is needed."""
        return len(self.cache_tokens), self.ctx_limit

    def _new_cache(self):
        """Fresh KV cache — with rotating caches swapped out for plain ones.

        Gemma-family models hand their sliding-window layers a RotatingKVCache
        capped at the window (1024 for Gemma 4). A rotating cache reports
        untrimmable the moment the transcript passes the window, so
        _prefill_delta silently rebuilds from zero and EVERY turn of a long
        session pays full prefill — measured 6.5-7.2s time-to-first-token at a
        4.5k-token transcript. Plain KVCache on every layer keeps the cache
        trimmable forever (0.37s TTFT, same transcript). Output-identical:
        the sliding-window attention MASK enforces the window, the cache type
        only decides what is stored (verified greedy A/B, 2026-08-07). Cost:
        sliding layers' KV now grows with the transcript instead of capping at
        the window. AGENT_ROLLING_KV=1 restores stock behavior.
        """
        cache = self._make_cache(self.model)
        if os.environ.get("AGENT_ROLLING_KV") == "1":
            return cache
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        if any(isinstance(c, RotatingKVCache) for c in cache):
            return [KVCache() for _ in cache]
        return cache

    def reset(self):
        self.messages = self.messages[:1]
        self.cache = self._new_cache()
        self.cache_tokens = []

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def _render(self):
        if DIALECT == "native":
            return self.tokenizer.apply_chat_template(
                self.messages, tools=TOOLS, add_generation_prompt=True, tokenize=True
            )
        return self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True
        )

    def _prefill_delta(self, tokens):
        """Trim the cache to the shared prefix, return only the new tokens."""
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        n = 0
        for a, b in zip(self.cache_tokens, tokens):
            if a != b:
                break
            n += 1
        # Never reuse the entire prompt — generation needs at least one token in.
        n = min(n, len(tokens) - 1)
        extra = len(self.cache_tokens) - n
        if extra > 0:
            if can_trim_prompt_cache(self.cache) and trim_prompt_cache(self.cache, extra) == extra:
                pass
            else:  # cache can't rewind; rebuild from scratch
                self.cache = self._new_cache()
                n = 0
        self.cache_tokens = list(tokens)
        return tokens[n:], n

    def _generate(self, on_text):
        from mlx_lm.generate import stream_generate

        tokens = self._render()
        delta, reused = self._prefill_delta(tokens)
        if reused:
            status_println(c(f"  [{reused} cached + {len(delta)} new]", DIM))

        out = []
        for resp in stream_generate(
            self.model, self.tokenizer, prompt=delta,
            max_tokens=MAX_TOKENS, sampler=self.sampler, prompt_cache=self.cache,
        ):
            out.append(resp.text)
            on_text(resp.text)
            self.cache_tokens.append(resp.token)
            # Stop the moment a call closes — everything after it is ignored
            # anyway, so generating it is waste. Key on </function>, not
            # </tool_call>: models often skip the wrapper, and waiting for a
            # tag that never comes runs to max_tokens.
            if "</function>" in "".join(out[-6:]):
                break
        return "".join(out)

    def step(self, on_text):
        """One assistant turn. Returns (prose, [(call_id, name, args), ...])."""
        text = self._generate(on_text)
        calls = parse_tool_calls(text)
        prose = strip_calls(text)
        if DIALECT == "native":
            if calls:
                self.messages.append({
                    "role": "assistant",
                    "content": prose,
                    "tool_calls": [{"name": n, "arguments": a} for _, (n, a) in
                                   zip(range(len(calls)), calls)],
                })
            else:
                self.messages.append({"role": "assistant", "content": text})
        else:
            # Prompted dialect: plain roles only. Keep the raw text (calls
            # included) so the transcript the template renders matches what
            # the model actually said.
            self.messages.append({"role": "assistant", "content": text})
        return prose or text, [(None, n, a) for n, a in calls]

    def add_tool_results(self, results):
        for _id, name, result in results:
            if DIALECT == "native":
                self.messages.append({"role": "tool", "content": result})
            else:
                self.messages.append({
                    "role": "user",
                    "content": f"<tool_result tool={name}>\n{result}\n</tool_result>",
                })


# ─── HTTP engine (Anthropic-protocol local servers) ─────────────────────────


class HTTPEngine:
    """Talks to any local Anthropic-format /v1/messages server — e.g. this
    repo's proxy (:4000). The protocol already fronted Claude Code, so it is
    proven. The server owns the model and its memory — no broker seat here."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.system = build_system(model_path)
        self.messages = []
        self.ctx_limit = int(os.environ.get("AGENT_CTX_LIMIT", "131072"))
        self.tools = [
            {"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]}
            for t in TOOLS
        ]

    def ctx_used(self):
        """Estimate only — the server owns the tokenizer, so ~4 chars/token."""
        chars = len(self.system) + sum(len(json.dumps(m)) for m in self.messages)
        return chars // 4, self.ctx_limit

    def reset(self):
        self.messages = []

    def add_user(self, text):
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def _request(self, stream):
        import urllib.request

        body = {
            "model": self.model_path,
            "max_tokens": MAX_TOKENS,
            "system": self.system,
            "messages": self.messages,
            "tools": self.tools,
            "stream": stream,
        }
        req = urllib.request.Request(
            f"{BASE_URL}/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": AUTH_TOKEN,
                "authorization": f"Bearer {AUTH_TOKEN}",
            },
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=600)

    def _step_stream(self, on_text):
        blocks = []
        with self._request(stream=True) as resp:
            cur = None
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload in ("", "[DONE]"):
                    continue
                ev = json.loads(payload)
                t = ev.get("type")
                if t == "content_block_start":
                    cur = dict(ev.get("content_block") or {})
                    if cur.get("type") == "tool_use":
                        cur["_json"] = ""
                    blocks.append(cur)
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta" and blocks:
                        blocks[-1]["text"] = blocks[-1].get("text", "") + d.get("text", "")
                        on_text(d.get("text", ""))
                    elif d.get("type") == "input_json_delta" and blocks:
                        blocks[-1]["_json"] = blocks[-1].get("_json", "") + d.get("partial_json", "")
                elif t == "error":
                    raise RuntimeError(str(ev.get("error")))
        for b in blocks:
            if b.get("type") == "tool_use" and "_json" in b:
                raw = b.pop("_json")
                if raw.strip():
                    try:
                        b["input"] = json.loads(raw)
                    except json.JSONDecodeError:
                        b["input"] = b.get("input") or {}
        return blocks

    def _step_once(self, on_text):
        with self._request(stream=False) as resp:
            data = json.loads(resp.read().decode())
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "text":
                on_text(b.get("text", ""))
        return blocks

    def step(self, on_text):
        try:
            blocks = self._step_stream(on_text)
        except Exception:
            blocks = self._step_once(on_text)
        # Normalize: keep only fields the API accepts when we echo them back.
        clean = []
        for b in blocks:
            if b.get("type") == "text" and b.get("text"):
                clean.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "tool_use":
                clean.append({"type": "tool_use", "id": b.get("id") or f"call_{len(clean)}",
                              "name": b.get("name", ""), "input": b.get("input") or {}})
        if clean:
            self.messages.append({"role": "assistant", "content": clean})
        prose = "\n".join(b["text"] for b in clean if b["type"] == "text").strip()
        calls = [(b["id"], b["name"], b["input"]) for b in clean if b["type"] == "tool_use"]
        return prose, calls

    def add_tool_results(self, results):
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": _id or "call_0",
                 "content": [{"type": "text", "text": result}]}
                for _id, _name, result in results
            ],
        })


# ─── Spinner — the "is it doing anything?" indicator ────────────────────────
#
# The silent stretches are prefill (seconds on a long transcript) and tool-call
# composition (hidden from the screen on purpose). A background thread animates
# one dim line and erases itself the instant real output wants the terminal.


class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        while not self._stop.wait(0.1):
            with self._lock:
                if self._stop.is_set():
                    break
                frame = self.FRAMES[i % len(self.FRAMES)]
                sys.stdout.write(
                    f"\r{DIM}{frame} {self.label}… {time.time()-self._t0:.0f}s{OFF}\033[K"
                )
                sys.stdout.flush()
            i += 1

    def println(self, text):
        """Print a full line from under the spinner without visual tearing."""
        with self._lock:
            sys.stdout.write("\r\033[K" + text + "\n")
            sys.stdout.flush()

    def stop(self):
        self._stop.set()
        with self._lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        self._thread.join(timeout=0.5)


_spinner = None


def start_spinner(label):
    global _spinner
    if _spinner is None and sys.stdout.isatty():
        _spinner = Spinner(label)


def stop_spinner():
    global _spinner
    if _spinner:
        _spinner.stop()
        _spinner = None


def status_println(text):
    """Line printing that stays clean whether or not a spinner is running."""
    if _spinner:
        _spinner.println(text)
    else:
        print(text)


# ─── Agent loop ──────────────────────────────────────────────────────────────


def agent_turn(engine, user_text, quiet=False):
    engine.add_user(user_text)
    final = ""

    CALL_MARKS = ("<tool_call>", "<function=")

    for _ in range(MAX_STEPS):
        # Streaming chunks are a few characters each, so a call tag arrives
        # split across chunks ("<fun", "ction="). A per-chunk substring check
        # misses it and the raw XML leaks to the screen — buffer instead, and
        # hold back any tail that could still be the start of a tag.
        printed = {"any": False, "in_call": False, "acc": "", "emitted": 0}

        def on_text(chunk):
            if printed["in_call"] or quiet:
                return
            printed["acc"] += chunk
            acc = printed["acc"]
            cut = min((i for m in CALL_MARKS if (i := acc.find(m)) != -1), default=-1)
            if cut != -1:
                printed["in_call"] = True
                safe = cut
            else:
                last = acc.rfind("<")
                held = acc[last:] if last != -1 else ""
                safe = last if held and any(m.startswith(held) for m in CALL_MARKS) else len(acc)
            if safe > printed["emitted"]:
                stop_spinner()
                sys.stdout.write(acc[printed["emitted"]:safe])
                sys.stdout.flush()
                printed["emitted"] = safe
                printed["any"] = True

        if not quiet:
            start_spinner("thinking")
        try:
            prose, calls = engine.step(on_text)
        finally:
            stop_spinner()
        if printed["any"] and not prose.endswith("\n"):
            print()

        if not calls:
            return prose

        results = []
        for call_id, name, args in calls:
            print(c(f"  ⚒ {name}", CYAN) + c(f"  {preview(name, args)}", DIM))
            t0 = time.time()
            if not quiet:
                start_spinner("running")
            try:
                result = run_tool(name, args)
            finally:
                stop_spinner()
            took = time.time() - t0
            head = result.splitlines()[0][:100] if result else ""
            bad = result.startswith("ERROR")
            print(
                c(f"    {head}", RED if bad else GREEN)
                + c(f"  ({took:.1f}s, {len(result)} chars)", DIM)
            )
            results.append((call_id, name, result))
        engine.add_tool_results(results)
        final = prose

    return final or "[hit step limit]"


def speak(text):
    """Narrate a final answer through a ~/.local/bin/speak CLI if one exists."""
    binp = Path.home() / ".local" / "bin" / "speak"
    if text and binp.exists():
        subprocess.Popen([str(binp), text], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


BANNER = f"""{BOLD}  {TITLE}{OFF}
  {DIM}/exit  /reset  /context  /tools  /cwd  /model{OFF}
"""

HISTFILE = Path.home() / ".local_agent_history"


def _ctx_parts(engine):
    """(pct, used, limit) of the model's context window, or None."""
    try:
        used, limit = engine.ctx_used()
    except Exception:
        return None
    if not limit:
        return None
    return min(999, round(used * 100 / limit)), used, limit


def _fmt_k(n):
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def framed_input(engine):
    """Read one line inside a visible frame:

      ─── context 8% · 10.9k/131.1k ──────────────
      › the cursor lives here
      ─────────────────────────────────────────────

    Both rules are drawn first, then the cursor is moved back up between
    them, so the typing area is bounded on screen the whole time. The
    context %% in the top rule is the model's live window usage.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input("› ")
    cols = shutil.get_terminal_size((80, 24)).columns
    ctx = _ctx_parts(engine)
    if ctx:
        pct, used, limit = ctx
        label = f" context {pct}% · {_fmt_k(used)}/{_fmt_k(limit)} "
        lcolor = RED if pct >= 90 else YELLOW if pct >= 70 else DIM
    else:
        label, lcolor = "", DIM
    pad = max(0, cols - 3 - len(label))
    sys.stdout.write(c("───", DIM) + c(label, lcolor) + c("─" * pad, DIM) + "\n")
    sys.stdout.write("\n")
    sys.stdout.write(c("─" * cols, DIM) + "\n")
    sys.stdout.write("\033[2A")  # cursor back up to the blank line between rules
    sys.stdout.flush()
    # \001/\002 mark the ANSI codes invisible so readline counts the prompt
    # width correctly (without them, arrow keys misplace the cursor).
    prompt = "\001\033[1m\002› \001\033[0m\002" if readline else c("› ", BOLD)
    try:
        line = input(prompt)
        sys.stdout.write("\n")  # step past the bottom rule
        sys.stdout.flush()
        return line
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")  # land clear of the frame before whoever catches
        sys.stdout.flush()
        raise


def repl(engine):
    if readline:
        try:
            readline.read_history_file(HISTFILE)
        except OSError:
            pass
        readline.set_history_length(1000)
        atexit.register(lambda: readline.write_history_file(HISTFILE))
    print(BANNER)
    while True:
        try:
            line = framed_input(engine).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/reset":
            engine.reset()
            print(c("  conversation cleared", DIM))
            continue
        if line == "/context":
            ctx = _ctx_parts(engine)
            if ctx:
                pct, used, limit = ctx
                print(c(f"  {used:,} / {limit:,} tokens ({pct}%)", DIM))
            else:
                print(c("  context usage unknown", DIM))
            continue
        if line == "/tools":
            print(c("  " + ", ".join(DISPATCH), DIM))
            continue
        if line == "/cwd":
            print(c("  " + os.getcwd(), DIM))
            continue
        if line == "/model":
            print(c("  " + engine.model_path, DIM))
            continue
        t0 = time.time()
        try:
            answer = agent_turn(engine, line)
            if SPEAK:
                speak(answer)
        except KeyboardInterrupt:
            print(c("\n  interrupted", YELLOW))
            continue
        except Exception as e:
            print(c(f"  error: {type(e).__name__}: {e}", RED))
            continue
        print(c(f"  {time.time()-t0:.1f}s", DIM))


def main():
    ap = argparse.ArgumentParser(prog="agent", add_help=True)
    ap.add_argument("-p", "--prompt", help="one-shot prompt, print result and exit")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("-C", "--dir", help="working directory")
    args = ap.parse_args()

    if args.dir:
        os.chdir(os.path.expanduser(args.dir))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: (release_seat(), sys.exit(0)))
    atexit.register(release_seat)

    if BACKEND == "http":
        engine = HTTPEngine(args.model)
    else:
        acquire_seat()
        engine = MLXEngine(args.model)

    if args.prompt:
        print(agent_turn(engine, args.prompt, quiet=True))
    else:
        repl(engine)
    # Skip interpreter/Metal teardown of a ~30GB model — measured 20s+ between
    # the seat being released and the process actually dying, which reads as a
    # hung window after /exit. The seat is released; nothing else needs to run.
    # (After a session with real generations the kernel still needs seconds to
    # reclaim the wired pages, so say goodbye before the window sits on it.)
    release_seat()
    if BACKEND == "mlx":
        print(c("  memory released — window may take a few seconds to close", DIM))
    os._exit(0)


if __name__ == "__main__":
    main()
