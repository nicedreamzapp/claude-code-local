#!/usr/bin/env python3
"""Matt's own terminal coding agent — one engine, every local launcher.

Each launcher (Qwen 3 Coder, Gemma 4, DeepSeek V4 Flash, Free AI, One AI)
execs this file with env vars picking the model, the backend, and the name.
The agent is called BY ITS MODEL'S NAME everywhere — no codenames.

Backends:
  AGENT_BACKEND=mlx   — load the model INTO this process via MLX. No server,
                        no HTTP hop, nothing on the network. KV cache reuse
                        across turns keeps replies sub-second.
  AGENT_BACKEND=http  — talk to a local Anthropic-protocol server (ds4.c on
                        :8000, the Free AI gateway on :4001, the One AI router
                        on :4010). The server owns the model and its memory.

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

import os as _os
# Keep model-load chatter (HF download bars, transformers deprecation notes)
# out of the chat window.
_os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
_os.environ["TRANSFORMERS_VERBOSITY"] = "error"
_os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*use_fast.*")
import argparse
import atexit
import codecs
import json
import os
import re
import select
import shutil
import signal
import subprocess
import tempfile
import sys
import threading
import time
# Windows (the shop PC, 2026-10-16) has no termios/tty/select-on-stdin. Moonstone drives the
# engine over a pipe there, which never needs the raw-mode line editor; plain input() does it.
WIN = os.name == "nt"
try:
    import termios
    import tty
except ImportError:
    termios = tty = None
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attachments  # noqa: E402
import chat_store  # noqa: E402  — same folder, shared with chat.py

try:
    import readline  # history storage only — line editing is _LineEditor's job
except ImportError:
    readline = None

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_DEFAULT = os.environ.get(
    "AGENT_MODEL", "donedynamics/Qwen3.8-27B-heretic-VL-MLX-bf16"
)
BACKEND = os.environ.get("AGENT_BACKEND", "mlx")
DIALECT = os.environ.get("AGENT_DIALECT", "native")
BASE_URL = os.environ.get("AGENT_BASE_URL", "").rstrip("/")
if "x.ai" in BASE_URL:   # Matt 10/21 11:32 PM: "get rid of grok completely"; paid xAI is never called again
    sys.exit("  Grok/xAI was removed for good on 10/21; pick a free or local model.")
AUTH_TOKEN = os.environ.get("AGENT_AUTH_TOKEN", "sk-local")
LEASE_NAME = os.environ.get("AGENT_LEASE_NAME", "agent-local")
LEASE_GB = float(os.environ.get("AGENT_LEASE_GB", "36" if BACKEND == "mlx" else "0"))
# reasoning models (Liquid LFM2.5) spend thousands of tokens thinking before the call; at 4096 the
# 10/26 video job hit the cap mid-thought, wrote nothing, and then claimed work it never did
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "16384"))   # 10/30: 16k for every model (4096 cut reasoning)
# 10/30 ladder: how many skipped repeat calls end a turn (was a hard 3, which ended a looping turn in ~10 calls)
BLOCK_LIMIT = int(os.environ.get("AGENT_BLOCK_LIMIT", "3"))
# 10/30 swap mode: a request may wait at the swap tap while a render holds the memory; give reads that long
HTTP_TIMEOUT = int(os.environ.get("AGENT_HTTP_TIMEOUT", "75"))
TEMP = float(os.environ.get("AGENT_TEMP", "0.3"))
# 40 was too few: on 2026-10-18 HQ Free (Body Play deploy) and Grok (HQ Free on the mini/PC)
# both ran 40 tool rounds of real work and stopped mid-task with nothing on screen. The loop
# breaker below already stops runaway repeats, so the cap only has to catch true runaways.
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "150"))
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
USER_TEXT = "\033[38;5;110m"   # soft blue — what Matt types, vs plain answers
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
            "Run a shell command (PowerShell on the Windows PC, bash on the Macs) and return "
            "its combined stdout/stderr. "
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

# The shared HQ toolbox (browser and more) rides along with the file tools.
# Never fatal: if it fails to import, the agent still runs with the six above.
try:
    import hq_tools  # noqa: E402
    TOOLS += hq_tools.HQ_TOOLS
except Exception as _e:  # pragma: no cover
    hq_tools = None
    print(f"  (HQ toolbox not loaded: {_e})", file=sys.stderr)

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def _clip(text, limit=OUT_LIMIT):
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - limit
    return f"{text[:half]}\n\n... [{dropped} chars elided] ...\n\n{text[-half:]}"


# ─── live token count for Moonstone (2026-10-16) ─────────────────────────────
#
# Moonstone showed "0 tokens" for every local and cloud chat: it only saw the
# visible text, and while a model thinks or calls tools nothing is visible.
# The engines now count what they really generate and, when running under the
# dock (stdout is a pipe), send it as an invisible terminal code the dock reads:
# ESC ] 7717 ; tok=<n> BEL. A real terminal ignores unknown codes like this.
TURN_TOKENS = 0
_tok_last_emit = 0.0


def tok_tick(n=1, exact=None):
    """Count n more generated tokens (or set the exact total), report at most 4x a second."""
    global TURN_TOKENS, _tok_last_emit
    TURN_TOKENS = exact if exact is not None else TURN_TOKENS + n
    now = time.time()
    if now - _tok_last_emit >= 0.25:
        tok_emit()


def tok_emit():
    global _tok_last_emit
    _tok_last_emit = time.time()
    # only inside a Moonstone window (the dock sets CLAUDE_SESSION_LABEL for its pipes)
    if not sys.stdout.isatty() and os.environ.get("CLAUDE_SESSION_LABEL"):
        sys.stdout.write(f"\033]7717;tok={TURN_TOKENS}\007")
        sys.stdout.flush()


# ─── API cost for Moonstone's balance pill (2026-10-21) ──────────────────────
#
# Matt 10/21: the Grok balance on Moonstone's API pill should count down as he uses it. After every
# paid API reply the engine reports what it cost as an invisible code, ESC ] 7717 ; cost=<usd>;grok BEL,
# and the dock subtracts it from that account's balance (server.js balLog). xAI's own cost field is used
# when the reply carries one; otherwise tokens x xAI's list prices per 1M tokens, from
# docs.x.ai/docs/models (checked 2026-10-21): (input, cached input, output), and a prompt of 200k+
# tokens bills ALL of its tokens at the long-context row.
XAI_PRICES = {
    "grok-4.7": ((2.00, 0.50, 6.00), (4.00, 1.00, 12.00)),
    "grok-4.6": ((2.00, 0.50, 6.00), (4.00, 1.00, 12.00)),
    "grok-4.5": ((2.00, 0.30, 6.00), (4.00, 0.60, 12.00)),
    "grok-4.3": ((1.25, 0.20, 2.50), (2.50, 0.40, 5.00)),
}


def cost_emit(usage):
    if not usage or "api.x.ai" not in BASE_URL:
        return
    if sys.stdout.isatty() or not os.environ.get("CLAUDE_SESSION_LABEL"):
        return
    try:
        ticks = usage.get("cost_in_usd_ticks")
        if isinstance(ticks, (int, float)) and ticks > 0:
            usd = ticks / 1e10          # xAI bills in ticks of 1e-10 USD
        else:
            model = os.environ.get("AGENT_MODEL", "")
            rows = XAI_PRICES.get(model)
            if not rows:
                print(f"[cost] no xAI price on file for {model}; the Grok balance won't move", file=sys.stderr)
                return
            # Anthropic-format usage: input_tokens EXCLUDES cache reads/writes, which come separately
            inp = int(usage.get("input_tokens") or 0)
            cached = int(usage.get("cache_read_input_tokens") or 0)
            wrote = int(usage.get("cache_creation_input_tokens") or 0)
            out = int(usage.get("output_tokens") or 0)
            p_in, p_cached, p_out = rows[1] if inp + cached + wrote >= 200_000 else rows[0]
            usd = ((inp + wrote) * p_in + cached * p_cached + out * p_out) / 1e6
        if usd > 0:
            sys.stdout.write(f"\033]7717;cost={usd:.6f};grok\007")
            sys.stdout.flush()
    except Exception as e:
        print(f"[cost] couldn't price this reply: {e}", file=sys.stderr)


def ctx_emit(engine):
    """Context use for Moonstone's memory meter: ESC ] 7717 ; ctx=<used>/<limit> BEL (2026-10-17)."""
    if sys.stdout.isatty() or not os.environ.get("CLAUDE_SESSION_LABEL"):
        return
    try:
        used, limit = engine.ctx_used()
    except Exception:
        return
    if limit:
        sys.stdout.write(f"\033]7717;ctx={int(used)}/{int(limit)}\007")
        sys.stdout.flush()


def tool_bash(command):
    # stdin closed on purpose: an interactive program (a game, a REPL, vim)
    # would otherwise inherit the terminal and silently fight the user for
    # keystrokes while its output is captured — invisible chaos for up to
    # BASH_TIMEOUT. Closed stdin makes it EOF/crash out in milliseconds.
    # Output goes to a temp file, not a pipe: a job backgrounded with `&`
    # inherits the pipe and holds it open, so reading a pipe would hang the
    # whole chat until that job ends (2026-10-16, Bolt sweep). Its own process
    # group lets a timeout kill the command's children too.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as f:
        if WIN:   # on the PC the "Bash" tool runs PowerShell
            p = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            p = subprocess.Popen(
                command, shell=True, stdout=f, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, start_new_session=True,
            )
        try:
            p.wait(timeout=BASH_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                if WIN:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                else:
                    os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            raise
        f.seek(0)
        out = f.read()
    if p.returncode != 0:
        out += f"\n[exit {p.returncode}]"
    return _clip(out.strip()) or "[no output]"


def tool_read(path, offset=None, limit=None):
    p = Path(path).expanduser()
    if p.is_dir():
        items = sorted(p.iterdir())
        return _clip(f"{p} is a folder, not a file. Its contents (Read one of these files):\n"
                     + "\n".join(str(i) + ("/" if i.is_dir() else "") for i in items[:200]))
    if not p.exists():
        import difflib
        par = p.parent
        while not par.exists() and par != par.parent:
            par = par.parent
        near = difflib.get_close_matches(p.name, [i.name for i in par.iterdir()], n=5, cutoff=0.3) if par.exists() else []
        return (f"ERROR: no file at {p}. " + (f"Closest names in {par}: {', '.join(near)}. " if near else
                f"The nearest folder that exists is {par}. ") + "Use Glob or ls to find the real path.")
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
    # 10/30: folders count too (shown with a trailing /), and a bare name with no wildcard also
    # searches below the folder, so "promo" finds formats/promo/ instead of [no matches]
    root = Path(path).expanduser() if path else Path.cwd()
    if not root.exists():
        return f"[no such folder: {root}]"
    if Path(pattern).expanduser().is_absolute():
        pp = Path(pattern).expanduser()
        root, pattern = Path(pp.anchor), str(pp.relative_to(pp.anchor))
    hits = list(root.glob(pattern))
    if not hits and not any(ch in pattern for ch in "*?["):
        hits = list(root.glob(f"**/*{pattern}*"))[:300]
    hits = [f for f in hits if f.exists()]
    hits.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if not hits:
        return f"[no matches for {pattern} under {root}]"
    return _clip("\n".join(str(f) + ("/" if f.is_dir() else "") for f in hits[:300]))


def tool_grep(pattern, path=None, glob=None):
    cmd = ["rg", "-n", "--no-heading", "--color=never", "-S"]
    if glob:
        cmd += ["--glob", glob]
    cmd += [pattern, path or "."]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        # no ripgrep on this machine: plain grep does the same job, just slower (2026-10-16)
        cmd = ["grep", "-rnI", "-i", "-E", pattern, path or "."]
        if glob:
            cmd.insert(1, f"--include={glob}")
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   f"Get-ChildItem -Recurse -File -Path '{path or '.'}' -Include '{glob or '*'}' | "
                   f"Select-String -Pattern '{pattern}' | Select-Object -First 200 | ForEach-Object {{ \"$($_.Path):$($_.LineNumber):$($_.Line)\" }}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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
if hq_tools:
    DISPATCH.update(hq_tools.HQ_DISPATCH)


def run_tool(name, args):
    fn = DISPATCH.get(name)
    if not fn:
        # small local models drift to "bash" for "Bash" and never self-correct off the error
        # (10/26 Liquid LFM2.5: 30 of 46 calls were "bash"); match the name case-insensitively
        name = next((k for k in DISPATCH if k.lower() == str(name).lower()), name)
        fn = DISPATCH.get(name)
    if not fn:
        return f"ERROR: no tool named {name}. Available: {', '.join(DISPATCH)}"
    if isinstance(args, dict):   # 10/30: common argument-name drift, mapped instead of failing
        args = dict(args)
        for alias, real in (("file_path", "path"), ("filepath", "path"), ("file", "path"), ("filename", "path"),
                            ("cmd", "command"), ("script", "command"), ("text", "content"), ("old", "old_string"),
                            ("new", "new_string"), ("directory", "path"), ("dir", "path")):
            if alias in args and real not in args:
                args[real] = args.pop(alias)
    try:
        return fn(**args)
    except subprocess.TimeoutExpired:
        return (f"ERROR: {name} timed out after {BASH_TIMEOUT}s and was stopped. For a long job run it in the "
                "background: nohup <command> > <log> 2>&1 & then check it with: sleep 90 && tail -40 <log>")
    except TypeError as e:
        want = ", ".join(TOOLS_BY_NAME.get(name, {}).get("parameters", {}).get("properties", {}).keys())
        return f"ERROR: bad arguments for {name}: {e}. {name} takes: {want}"
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


# ─── forge_guard seat (mlx backend only) ─────────────────────────────────────
#
# This box's memory broker SIGTERMs (then SIGSTOPs) any large process that never
# asked for room — a model sitting in a GPU prefill reads ~0% CPU and looks
# exactly like an idle hog. Ask honestly, release on every exit path.

MEM_DIR = Path.home() / "SongForgeM5"
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


BIG_RSS_GB = 12  # anything resident above this is "a big model" for the one-at-a-time rule


def _other_big_model():
    """Another model already sitting in memory on this Mac, or None.

    One big model at a time (Matt, 2026-10-16). Two of them together is what
    crashed the M5 that night: a Bolt test and a Gemma test both loaded, the box
    swapped 14 GB a minute and the guard started killing things. The guard's
    seats alone don't stop it: a seat is released the moment a run ends, but the
    process keeps its ~70 GB for many seconds after, so the next model is
    granted room that is not actually free yet. So look at what is really
    resident, not just at the seats.
    """
    me = os.getpid()
    try:
        out = subprocess.run(["ps", "-axo", "pid=,rss=,command="], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, rss_gb, cmd = int(parts[0]), int(parts[1]) / 1048576, parts[2]
        if pid == me or rss_gb < BIG_RSS_GB:
            continue
        # the program itself must be Python; a shell whose command line merely
        # mentions agent.py (a test harness) is not a model
        if "ython" not in cmd.split()[0].rsplit("/", 1)[-1]:
            continue
        # Any Python this big is a model (Bolt's sweep.py counts too), except
        # Song Forge's own engines: the launchers pause those, and the guard
        # already manages them.
        if re.search(r"SongForge|songforge|forge_|acestep|ace_server", cmd):
            continue
        name = "another model"
        if _mem:
            for lease in (_mem.state() or {}).get("leases", []):
                if lease.get("name") != LEASE_NAME and lease.get("gb", 0) >= 10:
                    name = lease["name"]
        return f"{name} ({rss_gb:.0f} GB, process {pid})"
    return None


def _refuse_second_model(holder):
    print(c(" refused", RED))
    print(f"\n  {holder} is already loaded on this Mac.")
    print("  Only one big model at a time, or both crash. Close that one first,")
    print("  then open this one again.\n")
    if sys.stdin.isatty():
        try:
            input("  press Return to close… ")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(1)


def acquire_seat():
    global _lease_id, _hb_stop
    if LEASE_GB <= 0 or not _load_mem_client():
        return
    _reap_stale_seats()
    holder = _other_big_model()
    if holder:
        print(c(f"  reserving {LEASE_GB:.0f}GB…", DIM), end="", flush=True)
        _refuse_second_model(holder)
    # quiet start (2026-10-17): no "reserving… ok" / "[mem] waited" lines unless it fails
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        _lease_id, status, _ = _mem.acquire_ex(LEASE_NAME, LEASE_GB, timeout=600)
    if _lease_id:
        # The wait above can end because another model's run just finished and
        # gave its seat back while its memory is still resident. Check again.
        holder = _other_big_model()
        if holder:
            release_seat()
            _refuse_second_model(holder)
    elif status == "refused":
        # The room genuinely isn't there (Song Forge + the One AI warm pool
        # can hold most of the box). Loading 30GB anyway is how the 7/27 swap
        # panic happened — stop and say why instead.
        print(c(f"  reserving {LEASE_GB:.0f}GB… refused", RED))
        print(f"\n  Not enough free memory for this model right now.")
        print(f"  Likely holders: Song Forge engines, the One AI warm pool.")
        print(f"  Free it with:  bash \"$HOME/Desktop/PROJECTS/Local AI Setup/smart-router/warm_pool.sh\" stop")
        print(f"  then double-click this launcher again.\n")
        if sys.stdin.isatty():
            try:
                input("  press Return to close… ")
            except (EOFError, KeyboardInterrupt):
                pass
        sys.exit(1)
    else:
        pass  # no guard on this machine: proceed unguarded, silently
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

Be concise. The user is an expert; skip explanations they did not ask for.

Answer a question in ONE sentence: just the thing he asked, nothing else. If he
asks where something is, give the location and stop. Do not add related facts
you found along the way (backups, links, other copies, how it gets used). He
will ask if he wants more.

When the task is a batch ("do the emails", "the easiest one", "go through X"),
do NOT list or triage everything. Pick exactly ONE item — the single easiest —
and present only that one, then stop and wait. No lists, no recaps, no rundown
of the others. One item at a time, always.

For EMAIL specifically: pull the inbox, pick the one easiest email, and say only
who it's from, the context (what they want), and how you'd reply — a short draft
in Matt's voice. Then wait for his OK. The briefing snippet is almost always
enough to reply; do NOT rabbit-hole trying to fetch full bodies or probe API
endpoints. Never summarize the other emails."""


def build_system(model_path):
    where = os.environ.get("AGENT_WHERE") or (
        "fully locally in-process via MLX on this machine — no cloud, no "
        "external services" if BACKEND == "mlx"
        else f"behind a local gateway at {BASE_URL}"
    )
    text = SYSTEM.format(title=TITLE, model=model_path, where=where, cwd=os.getcwd())
    if BACKEND == "mlx" and DIALECT == "prompted":
        text += "\n\n" + tools_doc()
    # House rules for every model: what tools it has, how Matt wants it to
    # talk, and never to send him somewhere else. Same file for every launcher.
    for name in ("house.md", "hq-rules.md"):   # our own rules, written for these models
        f = Path(__file__).with_name(name)
        if f.exists():
            text += "\n\n" + f.read_text()
    # The map of Matt's system (rules, memory index, where everything lives), rebuilt from
    # the live files at every start by build_map.py; last good copy if that fails (2026-10-24).
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import build_map
        smap = build_map.load(os.environ.get("AGENT_CTX_LIMIT"))
        if smap:
            text += "\n\n" + smap
    except Exception:
        f = Path(__file__).with_name("system-map.md")
        if f.exists():
            text += "\n\n" + f.read_text()
    if PROMPT_FILE and Path(PROMPT_FILE).expanduser().exists():
        text += "\n\n" + Path(PROMPT_FILE).expanduser().read_text()
    # keepgoing hands over the conversation Matt was just having (agent/keepgoing.py)
    carry = os.environ.get("AGENT_CARRYOVER", "")
    if carry and Path(carry).exists():
        text += "\n\n" + Path(carry).read_text()
    return text


# ─── MLX engine (in-process) ────────────────────────────────────────────────


def _patch_vlm_config(vlm_utils):
    """Newer Qwen 3.5-family uploads name their vision tower "qwen3_5_vision";
    this mlx_vlm only knows it as "qwen3_5". Same tower, so rename on load."""
    if getattr(vlm_utils, "_agent_patched", False):
        return
    original = vlm_utils.load_config

    def load_config(model_path, **kwargs):
        config = original(model_path, **kwargs)
        vision = config.get("vision_config") or {}
        if vision.get("model_type") == "qwen3_5_vision":
            vision["model_type"] = "qwen3_5"
        return config

    vlm_utils.load_config = load_config
    vlm_utils._agent_patched = True


def _vlm_overlay(path):
    """The Qwen 3.8 upload names its image processor "Qwen3VLImageProcessor",
    which this transformers doesn't have (the class is Qwen2VLImageProcessor).
    Point at a folder of links to the real files with that one name fixed,
    instead of editing the download."""
    cfg = Path(path) / "processor_config.json"
    try:
        text = cfg.read_text()
    except OSError:
        return str(path)
    if '"Qwen3VLImageProcessor"' not in text:
        return str(path)
    overlay = Path.home() / ".cache" / "local-ai-overlays" / Path(path).name
    overlay.mkdir(parents=True, exist_ok=True)
    for f in Path(path).iterdir():
        link = overlay / f.name
        if f.name != cfg.name and not link.exists():
            link.symlink_to(f)
    (overlay / cfg.name).write_text(
        text.replace('"Qwen3VLImageProcessor"', '"Qwen2VLImageProcessor"'))
    return str(overlay)


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

        # Models with a vision tower load through mlx_vlm so a dropped picture
        # reaches the model. mlx_lm would load the same weights text-only.
        self.vision = (os.environ.get("AGENT_VISION", "1") != "0"
                       and attachments.has_vision(model_path))

        t0 = time.time()
        _load_done = threading.Event()

        def _load_ticker():
            # One line, no \r ticker: the phone/HQ viewer ignores carriage
            # returns, so a redrawn counter piled up as a wall of text.
            # A dot every 5s still shows it isn't frozen. One-shot (-p) has
            # no viewer, and this line was scored as the answer when a load
            # got cut off (watchdog, 2026-10-17).
            if os.environ.get("AGENT_QUIET_LOAD") == "1":
                _load_done.wait()
                return
            print(c(f"  loading {TITLE}", DIM), end="", flush=True)
            while not _load_done.wait(5):
                print(c(".", DIM), end="", flush=True)

        _ticker = threading.Thread(target=_load_ticker, daemon=True)
        _ticker.start()
        if self.vision:
            # mlx_vlm pulls in torch, which logs a "Redirects are currently not supported" warning
            import logging
            logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(lambda r: False)
            import mlx_vlm.utils as vlm_utils

            _patch_vlm_config(vlm_utils)
            self.model, self.processor = vlm_utils.load(
                _vlm_overlay(vlm_utils.get_model_path(model_path)))
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            self._cache_owner = self.model.language_model
        else:
            self.model, self.tokenizer = load(model_path)
            self._cache_owner = self.model
        _load_done.set()
        _ticker.join()
        print(c(f" {time.time()-t0:.0f}s ✓", DIM))

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
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        for holder in (getattr(self.model, "args", None), cfg, text_cfg):
            for key in ("max_position_embeddings", "max_context_length",
                        "context_length"):
                v = (holder.get(key) if isinstance(holder, dict)
                     else getattr(holder, key, None))
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
        cache = self._make_cache(self._cache_owner)
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

    def add_user(self, text, images=()):
        msg = {"role": "user", "content": text}
        if images:
            msg["images"] = list(images)
        self.messages.append(msg)

    def _template_messages(self):
        """The transcript as the chat template wants it: a picture becomes an
        image item ahead of its message's text."""
        out = []
        for m in self.messages:
            m = dict(m)
            images = m.pop("images", None)
            if images and self.vision:
                m["content"] = ([{"type": "image"} for _ in images]
                                + [{"type": "text", "text": m["content"]}])
            out.append(m)
        return out

    def _render(self, tokenize=True):
        # Thinking control. Qwen3.8-class templates default reasoning_effort to
        # 'xhigh' when unset — max reasoning, dumped to screen. AGENT_THINKING
        # picks the level: off (default, terse, no <think> at all) or
        # low/medium/xhigh. Templates that don't know these kwargs ignore them.
        think = os.environ.get("AGENT_THINKING", "off").lower()
        kw = {"add_generation_prompt": True, "tokenize": tokenize}
        msgs = self._template_messages()
        tmpl = self.tokenizer
        if self.vision and not getattr(self.tokenizer, "chat_template", None):
            tmpl = self.processor
        if think in ("off", "0", "false", "none"):
            kw["enable_thinking"] = False
        else:
            kw["enable_thinking"] = True
            kw["reasoning_effort"] = think if think in ("low", "medium", "xhigh") else "low"
        try:
            if DIALECT == "native":
                return tmpl.apply_chat_template(msgs, tools=TOOLS, **kw)
            return tmpl.apply_chat_template(msgs, **kw)
        except Exception:
            # older template without the thinking kwargs
            kw.pop("enable_thinking", None); kw.pop("reasoning_effort", None)
            if DIALECT == "native":
                return tmpl.apply_chat_template(msgs, tools=TOOLS, **kw)
            return tmpl.apply_chat_template(msgs, **kw)

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

    def _generate_vision(self, on_text):
        """_generate for a model loaded through mlx_vlm: same cache reuse, but the
        prompt goes through the processor so pictures become image tokens."""
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        from mlx_vlm.generate import generate_step
        from mlx_vlm.utils import prepare_inputs

        cfg = self.model.config
        images = [p for m in self.messages for p in m.get("images", ())]
        prompt = self._render(tokenize=False)
        # Gemma's template already writes <bos>; mlx_vlm's own rule.
        add_special = (getattr(self.processor, "chat_template", None) is None
                       if cfg.model_type in ("gemma3", "gemma3n", "gemma4") else True)
        inputs = prepare_inputs(
            self.processor, images=images or None, prompts=prompt,
            image_token_index=getattr(cfg, "image_token_index", None),
            add_special_tokens=add_special,
        )
        ids = inputs["input_ids"]
        pixels = inputs.get("pixel_values")
        extra = {k: v for k, v in inputs.items()
                 if k not in ("input_ids", "pixel_values", "attention_mask")}
        tokens = ids.flatten().tolist()

        n = 0
        for a, b in zip(self.cache_tokens, tokens):
            if a != b:
                break
            n += 1
        n = min(n, len(tokens) - 1)
        image_id = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        in_prefix = image_id is not None and image_id in tokens[:n]
        in_delta = image_id is not None and image_id in tokens[n:]
        reuse = n > 0 and not (in_prefix and in_delta)
        # Qwen-style models place image tokens with their own position ids, which a
        # partial prefill can't reproduce: any picture in the thread means a full one.
        if images and hasattr(self.model.language_model, "_rope_deltas"):
            reuse = False
        stale = len(self.cache_tokens) - n
        if reuse and stale > 0:
            reuse = can_trim_prompt_cache(self.cache) and trim_prompt_cache(self.cache, stale) == stale
        if not reuse:
            self.cache = self._new_cache()
            n = 0
        if n and not in_delta:
            pixels = None
        if isinstance(pixels, list):
            # Pictures of different sizes come back unstacked, and Gemma's vision
            # tower can't take them together: encode each one, hand over the result.
            import mlx.core as mx

            feats = [self.model.embed_vision(self.model.vision_tower(mx.array(p)[None]))
                     for p in pixels]
            extra["cached_image_features"] = mx.concatenate(
                [f.reshape(1, -1, f.shape[-1]) for f in feats], axis=1)
            pixels = mx.array(pixels[0])[None]
        if n:
            if SHOW_TOOLS:   # Moonstone shows the real token count; this line only cluttered the chat (10/17)
                status_println(c(f"  [{n} cached + {len(tokens) - n} new]", DIM))
        self.cache_tokens = list(tokens)

        stop = getattr(self.tokenizer, "stopping_criteria", None)
        eos = getattr(cfg, "eos_token_id", None)
        eos = set(eos if isinstance(eos, list) else [eos])
        detok = self.processor.detokenizer
        detok.reset()
        out, made = [], []
        for token, _ in generate_step(
            ids[:, n:], self.model, pixels, None,
            max_tokens=MAX_TOKENS, sampler=self.sampler, prompt_cache=self.cache,
            **extra,
        ):
            self.cache_tokens.append(token)
            tok_tick()
            if (stop(token) if stop is not None else token in eos):
                break
            made.append(token)
            detok.add_token(token)
            piece = detok.last_segment
            if piece:
                out.append(piece)
                on_text(piece)
            tail = "".join(out[-8:])
            # </function>: a call is done, anything after it is ignored.
            # </think>: Bolt's template opens <think> in the prompt, and when
            # the model never closes it the reasoning runs to the token cap
            # and repeats onto the screen. Stop the moment it closes.
            if "</function>" in tail or "</think>" in tail:
                break
            # 2026-10-18: Bolt's abliterated build falls into a loop repeating
            # the same two sentences and never emits </think>, so it ran to the
            # token cap and dumped it all on screen. Same 40-char run three
            # times means it is stuck; stop and let the caller answer instead.
            joined = "".join(out)
            if len(joined) > 400:
                chunk = joined[-40:]
                if joined.count(chunk) >= 3:
                    break
        # The streaming detokenizer can hold back the last word; the full decode
        # is the truth, so whatever it adds goes out now.
        full = self.tokenizer.decode(made)
        shown = "".join(out)
        if full.startswith(shown) and len(full) > len(shown):
            on_text(full[len(shown):])
            return full
        return shown if shown else full

    def _generate(self, on_text):
        from mlx_lm.generate import stream_generate

        if self.vision:
            return self._generate_vision(on_text)

        tokens = self._render()
        delta, reused = self._prefill_delta(tokens)
        if reused:
            if SHOW_TOOLS:
                status_println(c(f"  [{reused} cached + {len(delta)} new]", DIM))

        out = []
        for resp in stream_generate(
            self.model, self.tokenizer, prompt=delta,
            max_tokens=MAX_TOKENS, sampler=self.sampler, prompt_cache=self.cache,
        ):
            out.append(resp.text)
            on_text(resp.text)
            tok_tick()
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
        # 2026-10-16 Bolt: with AGENT_THINKING on, the template opens <think> in
        # the prompt, so the reply is "reasoning</think>answer". Keep the
        # reasoning off the screen and out of the transcript; a reply with no
        # </think> at all is shown whole at the end.
        think_on = os.environ.get("AGENT_THINKING", "off").lower() not in ("off", "0", "false", "none")
        if think_on:
            show, seen = on_text, {"done": False, "buf": ""}

            def on_text(chunk):
                if seen["done"]:
                    return show(chunk)
                seen["buf"] += chunk
                i = seen["buf"].find("</think>")
                if i != -1:
                    seen["done"] = True
                    rest = seen["buf"][i + len("</think>"):].lstrip()
                    if rest:
                        show(rest)
        text = self._generate(on_text)
        if think_on:
            if "</think>" in text:
                text = text.split("</think>", 1)[1].lstrip()
            else:
                # 2026-10-18: Bolt never closed <think> and this fallback dumped
                # the whole reasoning loop onto the screen. That text is not
                # the answer, so show nothing rather than the loop.
                text = ""
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
    """Talks to a local Anthropic-format /v1/messages server: ds4.c (:8000),
    the Free AI LiteLLM gateway (:4001), the One AI router (:4010). These all
    fronted Claude Code before, so the protocol is already proven. The server
    owns the model and its memory — no forge_guard seat needed here."""

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

    vision = True   # cloud models take pictures as image blocks; the REPL only attaches when this is set

    def add_user(self, text, images=()):
        # pictures go up as image blocks (Grok and most cloud models can see them); 2026-10-16
        import base64, mimetypes
        content = []
        for p in images or ():
            try:
                data = base64.b64encode(open(p, "rb").read()).decode()
            except OSError:
                continue
            mt = mimetypes.guess_type(p)[0] or "image/png"
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})

    def _fit(self):
        """Keep the conversation inside the model's window (2026-10-16). Old tool
        output and pictures shrink first; if that is not enough, the oldest
        exchanges go, keeping the opening message (the carryover). No extra request."""
        budget = int(self.ctx_limit * 0.7)
        if self.ctx_used()[0] <= budget:
            return
        keep_tail = 6
        for m in self.messages[1:-keep_tail]:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    for inner in b.get("content") or []:
                        t = inner.get("text", "")
                        if len(t) > 400:
                            inner["text"] = t[:400] + " …[trimmed]"
                elif b.get("type") == "image":
                    b.clear(); b.update({"type": "text", "text": "[picture removed to save room]"})
        dropped = 0
        # drop (assistant, user) pairs right after the opening message so the
        # alternation and every tool_use/tool_result pairing stay intact
        while self.ctx_used()[0] > budget and len(self.messages) > keep_tail + 3:
            if self.messages[1].get("role") != "assistant":
                break
            del self.messages[1:3]
            dropped += 2
        if dropped and not getattr(self, "_trim_noted", False):
            first = self.messages[0]
            if isinstance(first.get("content"), list):
                first["content"].append({"type": "text", "text":
                    "[Note: the middle of this conversation was trimmed to fit. Ask Matt if you need something from it.]"})
                self._trim_noted = True

    def _request(self, stream):
        import urllib.request

        self._fit()
        body = {
            "model": self.model_path,
            "max_tokens": MAX_TOKENS,
            "system": self.system,
            "messages": self.messages,
            "tools": self.tools,
            "stream": stream,
        }
        if not self.tools:
            del body["tools"]
        req = urllib.request.Request(
            f"{BASE_URL}/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": AUTH_TOKEN,
                "authorization": f"Bearer {AUTH_TOKEN}",
                # OpenRouter gates some free models to identified apps and
                # answers 403 without these (2026-10-16, Inkling)
                "HTTP-Referer": "https://ineedhemp.com",
                "X-Title": "Divine Tribe HQ",
            },
            method="POST",
        )
        # Streaming: the timeout is PER READ, so a free-tier pool that stalls or
        # dribbles keepalive whitespace ends the turn after 75s idle instead of
        # freezing the UI for 10 minutes (2026-08-20: agent sat in readline forever
        # after the gateway had already logged 200). One-shot gets a longer wall.
        return urllib.request.urlopen(req, timeout=HTTP_TIMEOUT if stream else max(240, HTTP_TIMEOUT))

    def _step_stream(self, on_text):
        blocks = []
        with self._request(stream=True) as resp:
            cur = None
            by_index = {}
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
                    by_index[ev.get("index", len(blocks) - 1)] = cur
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    # Deltas name their block by index. When a model opens two tool calls at once
                    # (Grok does), appending to the last block glued one call's arguments onto the
                    # other and left the first empty (2026-10-16: EmailThread {}, WebSearch {}).
                    tgt = by_index.get(ev.get("index"), blocks[-1] if blocks else None)
                    if d.get("type") == "text_delta" and tgt is not None:
                        tgt["text"] = tgt.get("text", "") + d.get("text", "")
                        on_text(d.get("text", ""))
                        tok_tick(max(1, len(d.get("text", "")) // 4))
                    elif d.get("type") == "input_json_delta" and tgt is not None:
                        tgt["_json"] = tgt.get("_json", "") + d.get("partial_json", "")
                        tok_tick(max(1, len(d.get("partial_json", "")) // 4))
                    elif d.get("type") == "thinking_delta":
                        tok_tick(max(1, len(d.get("thinking", "")) // 4))
                elif t == "message_start":
                    self._step_base = TURN_TOKENS   # this reply's real count lands on top of earlier replies
                elif t == "message_delta" and isinstance((ev.get("usage") or {}).get("output_tokens"), int):
                    tok_tick(exact=getattr(self, "_step_base", 0) + ev["usage"]["output_tokens"])
                elif t == "error":
                    raise RuntimeError(str(ev.get("error")))
        for b in blocks:
            if b.get("type") == "tool_use" and "_json" in b:
                raw = b.pop("_json")
                b["_merged"] = raw
                if raw.strip():
                    try:
                        b["input"] = json.loads(raw)
                    except json.JSONDecodeError:
                        b["input"] = b.get("input") or {}
        return blocks

    def _step_once(self, on_text):
        with self._request(stream=False) as resp:
            data = json.loads(resp.read().decode())
        # OpenRouter reports an upstream overload as HTTP 200 with an error
        # body (Nemotron 3 Ultra, 2026-10-17). Swallowing it scored as a
        # blank answer in 2s; raise so the busy-retry path handles it.
        if data.get("type") == "error" or data.get("error"):
            raise RuntimeError(str(data.get("error") or data))
        if isinstance((data.get("usage") or {}).get("output_tokens"), int):
            tok_tick(data["usage"]["output_tokens"])
        cost_emit(data.get("usage") or {})   # what this reply cost, for the API balance (10/21)
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "text":
                on_text(b.get("text", ""))
        return blocks

    def step(self, on_text):
        import urllib.error
        try:
            # xAI's stream merges two tool calls into one block (the second call's start never
            # comes, both argument sets land in the first), so Grok runs unstreamed and every
            # call comes back intact (2026-10-16). Another provider doing the same is caught
            # here and that step is asked again unstreamed.
            if "api.x.ai" in BASE_URL:
                blocks = self._step_once(on_text)
            else:
                blocks = self._step_stream(on_text)
                if any(b.get("type") == "tool_use" and "}{" in (b.get("_merged") or "") for b in blocks):
                    blocks = self._step_once(lambda t: None)
        except Exception as e:
            # free pools answer 429/502/503 when busy, and OpenRouter also reports
            # an upstream overload as HTTP 200 with an error body. Wait and retry,
            # then say so plainly instead of dumping a traceback (2026-10-16, Laguna
            # 429; 2026-10-17, Nemotron 3 Ultra 200 overloaded_error).
            err = str(e).lower()
            busy = isinstance(e, urllib.error.HTTPError) or any(
                w in err for w in ("overload", "rate limit", "too many", "429", "503", "busy"))
            blocks = None
            # A retry loop with no exit is a hang: the generic except below does
            # `continue` and never sets busy, so it loops forever printing
            # "stream stalled" (2026-10-20, Keep Going handoff). Cap the tries.
            tries = 0
            for wait in ((8, 20, 45) if busy else (0, 8, 20)):
                tries += 1
                if wait:
                    on_text(f"\n[the model is busy, trying again in {wait} seconds]\n")
                    time.sleep(wait)
                elif not isinstance(e, urllib.error.HTTPError):
                    on_text(f"\n[stream stalled ({type(e).__name__}) — retrying without streaming]\n")
                try:
                    blocks = self._step_once(on_text)
                    break
                except urllib.error.HTTPError as e2:
                    if e2.code not in (408, 429, 500, 502, 503, 504):
                        msg = (f"This model turned the request down (error {e2.code}); its free server "
                               "may be broken right now. Pick another one from the menu.")
                        on_text(msg)
                        return msg, []
                    busy = True
                except Exception as e2:
                    if any(w in str(e2).lower() for w in ("overload", "rate limit", "too many", "429", "503")):
                        busy = True
                    if tries >= 3:
                        on_text(f"\n[stalled {tries} times — giving up on this turn]\n")
                        raise
                    continue
            if blocks is None:
                msg = ("This model is too busy right now (OpenRouter's free pool is full). "
                       "Pick another one from the menu, or try again in a minute.")
                on_text(msg)
                return msg, []
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


# Matt wants the token counts and the spinner, not every tool call (2026-10-17).
# AGENT_SHOW_TOOLS=1 brings the tool lines back for debugging.
SHOW_TOOLS = os.environ.get("AGENT_SHOW_TOOLS") == "1"


# ─── Agent loop ──────────────────────────────────────────────────────────────


def agent_turn(engine, user_text, quiet=False, images=()):
    global TURN_TOKENS
    TURN_TOKENS = 0
    tok_emit()
    try:
        return _agent_turn(engine, user_text, quiet, images)
    finally:
        tok_emit()


def _agent_turn(engine, user_text, quiet=False, images=()):
    if hq_tools:
        hq_tools.LAST_USER = user_text
    if images:
        engine.add_user(user_text, images)
    else:
        engine.add_user(user_text)
    final = ""

    CALL_MARKS = ("<tool_call>", "<function=")
    # Loop breaker (2026-10-16): Qwen 3.8 ran the same `ls` 35 times in a row on
    # 10/15 and never answered. The same call with the same arguments a third
    # time is not run; the model is told to stop and answer instead.
    seen_calls = {}
    last_result = {}
    blocked = 0
    turn_seen = []     # every tool result this turn, for the draft check
    checked = False
    nudged = False

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

        if not calls and not prose.strip() and not nudged:
            # it ran its tools and then said nothing (2026-10-17: Bolt question got no answer)
            nudged = True
            # 10/26: "answer in one sentence" made Liquid stop mid-job and claim a step it never ran;
            # say plainly to keep working if work is left, and to report only what really ran
            engine.add_user("Your last turn ended without a tool call or any text. If the task still has "
                            "steps left, make the next tool call now. If it is truly finished, answer Matt "
                            "in one sentence, naming only what the tools actually did.")
            continue
        if not calls:
            # draft check: one chance to fix a wrong total or a banned customs word
            # before Matt reads it (2026-10-16 real-task tests)
            fix = None
            if hq_tools and not checked and hasattr(hq_tools, "draft_check"):
                try:
                    fix = hq_tools.draft_check(prose, "\n".join(turn_seen))
                except Exception:
                    fix = None
            if fix:
                checked = True
                if SHOW_TOOLS:
                    print(c("  ⟲ checking the draft, fixing it", YELLOW))
                engine.add_user(fix)
                continue
            if hq_tools:
                hq_tools.LAST_ANSWER = prose
            return prose

        results = []
        for call_id, name, args in calls:
            sig = name + json.dumps(args, sort_keys=True)
            seen_calls[sig] = seen_calls.get(sig, 0) + 1
            # a re-check of a running job (sleep && tail, status) is legitimately repeated: never skip it
            _cmd = str(args.get("command", "")) if isinstance(args, dict) else ""
            if seen_calls[sig] > 2 and BLOCK_LIMIT > 3 and ("sleep" in _cmd or "tail" in _cmd or "status" in _cmd):
                seen_calls[sig] = 2
            if seen_calls[sig] > 2:
                blocked += 1
                if SHOW_TOOLS:
                    print(c(f"  ⟲ {name} again, skipped (already ran it twice)", YELLOW))
                prev = last_result.get(sig, "")
                results.append((call_id, name,
                    ("NOT RUN: you already made this exact call twice and got the same result. "
                     "Do not repeat it. Take a different next step toward finishing the task. "
                     f"The result you already have was:\n{prev[:1500]}") if BLOCK_LIMIT > 3 else
                    ("NOT RUN: you already made this exact call twice and got the same result. "
                     "Do not repeat it. Use what you already have to answer Matt now, or try "
                     "a genuinely different approach.")))
                continue
            if SHOW_TOOLS:
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
            if SHOW_TOOLS:
                print(
                    c(f"    {head}", RED if bad else GREEN)
                    + c(f"  ({took:.1f}s, {len(result)} chars)", DIM)
                )
            results.append((call_id, name, result))
            last_result[sig] = result
            turn_seen.append(result)
            # 10/30: re-running a build after editing a file is not a repeat (Liquid's make-promo --from footage
            # was skipped after it fixed promo.json): a successful Write/Edit resets the repeat counts
            if name in ("Write", "Edit") and not result.startswith("ERROR"):
                seen_calls.clear()
        engine.add_tool_results(results)
        final = prose
        if blocked >= BLOCK_LIMIT:
            # it keeps trying anyway: one last turn with no tools to say what it found
            engine.add_user("Stop using tools. In two or three plain sentences, tell Matt what "
                            "you found and what is still unknown.")
            # really take the tools away for this turn: with them still offered, Liquid answered
            # "stop using tools" with one more tool call, no text, and Moonstone showed
            # "finished without writing a reply" (10/26)
            saved = getattr(engine, "tools", None)
            if saved is not None:
                engine.tools = []
            try:
                prose, _ = engine.step(lambda t: None if quiet else (sys.stdout.write(t), sys.stdout.flush()))
            finally:
                if saved is not None:
                    engine.tools = saved
            if not prose.strip():
                prose = ("I got stuck repeating the same tool call and stopped. "
                         "Say 'keep going' and I'll try a different approach.")
                if not quiet:
                    sys.stdout.write(prose)
            if not quiet:
                print()
            return prose

    # Out of steps. Never end silently (2026-10-18: both chats looked dead to Matt because
    # this path returned a string the REPL never prints). Say what got done and what is left.
    if not quiet:
        print(c(f"  ⟲ used all {MAX_STEPS} steps for this message, wrapping up", YELLOW))
    engine.add_user(f"You have used all {MAX_STEPS} tool steps for this message. Stop using tools. "
                    "In a few plain sentences tell Matt exactly what you finished, what is still "
                    "left to do, and that he can say 'keep going' to have you continue.")
    try:
        prose, _ = engine.step(lambda t: None if quiet else (sys.stdout.write(t), sys.stdout.flush()))
    except Exception:
        prose = ""
    if not prose.strip():
        prose = (f"I used all {MAX_STEPS} steps on this one before finishing. "
                 "Say 'keep going' and I'll pick up where I stopped.")
        if not quiet:
            sys.stdout.write(prose)
    if not quiet:
        print()
    return prose


def speak(text):
    """Narrate a final answer through Matt's speak CLI, without blocking."""
    binp = Path.home() / ".local" / "bin" / "speak"
    if text and binp.exists():
        subprocess.Popen([str(binp), text], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


BANNER = f"""{BOLD}  {TITLE}{OFF}
"""


def _save_session(engine):
    """Snapshot the conversation after a completed turn. The system prompt is
    left out on purpose: it is rebuilt from the live cwd and tool list at start,
    so a stale one from yesterday would quietly lie about the environment."""
    path = getattr(engine, "session", None)
    if path is None:
        return
    chat_store.save_snapshot(
        path, [m for m in engine.messages if m.get("role") != "system"])


def _restore_session(engine, path):
    """Load a saved conversation into either engine.

    reset() first so the MLX engine drops its prompt cache — that cache belongs
    to the abandoned conversation and reusing it against restored messages would
    feed the model someone else's KV state.
    """
    msgs = chat_store.load(path)
    if not msgs:
        return 0
    engine.reset()
    engine.messages.extend(msgs)
    engine.session = path
    return len(msgs)


def _show_sessions(rows):
    if not rows:
        print(c("  no saved conversations for this model yet", DIM))
        return
    for i, (path, mtime, n, first) in enumerate(rows, 1):
        print(c(f"  {i}. {chat_store.ago(mtime):>11} · {n:>3} messages · {first}",
                DIM))

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


# ─── The input box ──────────────────────────────────────────────────────────
#
# This is a raw-mode line editor, not input()/readline, for two reasons that
# both bit in practice:
#
#   1. PASTING.  macOS ships libedit, not GNU readline, and libedit has no
#      bracketed-paste support. Every paste — ⌘V, a middle-click, a drag and
#      drop, a trackpad gesture the terminal reads as button 2 — arrives as
#      bare keystrokes, so every newline inside it fires as a Return and the
#      text submits itself before you can look at it. Turning bracketed paste
#      ON (\033[?2004h) makes the terminal wrap pasted text in \033[200~ …
#      \033[201~; we drop the payload into the buffer as inert text. NOTHING
#      is ever sent except by pressing Return.
#
#   2. TYPING WHILE IT WORKS.  The tty echoes keys no matter what the program
#      is doing, so keystrokes during generation painted themselves over the
#      stream and then queued up behind the answer — the box looked live when
#      it wasn't. The editor now owns the echo bit: it mutes on submit and
#      unmutes (dropping whatever was typed meanwhile) at the next prompt.
#
# Also: ⌃J or ⌥Return inserts a newline, so a multi-line message is one turn.

_TTY_ORIG = None       # tty attrs as we found them, captured once
_TTY_ATEXIT = False


def mute_input():
    """Stop the tty echoing keys — held while the model is generating."""
    global _TTY_ORIG, _TTY_ATEXIT
    if termios is None or not sys.stdin.isatty():
        return
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        if _TTY_ORIG is None:
            _TTY_ORIG = attrs[:]
        if not _TTY_ATEXIT:
            atexit.register(unmute_input)
            _TTY_ATEXIT = True
        quiet = attrs[:]
        quiet[3] &= ~termios.ECHO          # lflag
        termios.tcsetattr(fd, termios.TCSADRAIN, quiet)
    except (termios.error, ValueError):
        pass


def unmute_input():
    """Echo back on, and drop anything typed while it was off."""
    if termios is None or _TTY_ORIG is None or not sys.stdin.isatty():
        return
    try:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, _TTY_ORIG)
        termios.tcflush(fd, termios.TCIFLUSH)
    except (termios.error, ValueError):
        pass


def _cw(ch):
    """Display width of one character."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _history_list():
    if not readline:
        return []
    try:
        n = readline.get_current_history_length()
        return [readline.get_history_item(i) or "" for i in range(1, n + 1)]
    except Exception:
        return []


def _remember(line):
    # Multi-line entries are skipped: the history FILE is line-based, so a
    # newline in an entry corrupts every entry after it on reload.
    if readline and "\n" not in line:
        try:
            readline.add_history(line)
        except Exception:
            pass


class _LineEditor:
    """One line (or several) typed inside the frame.

      ─── context 8% · 10.9k/131.1k ──────────────
      › the cursor lives here
      ─────────────────────────────────────────────

    The whole frame is redrawn on every keystroke from the buffer, so the
    box always matches the text — including text that wrapped or was pasted
    across many lines.
    """

    def __init__(self, engine):
        self.engine = engine
        self.buf = ""
        self.pos = 0
        self.cols = 80
        self.rows_from_top = 0     # cursor's line offset from the top rule
        self.hist = _history_list()
        self.hidx = len(self.hist)
        self.stash = ""
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")

    # — drawing —

    def _top_rule(self):
        ctx = _ctx_parts(self.engine)
        if ctx:
            pct, used, limit = ctx
            label = f" context {pct}% · {_fmt_k(used)}/{_fmt_k(limit)} "
            lcolor = RED if pct >= 90 else YELLOW if pct >= 70 else DIM
        else:
            label, lcolor = "", DIM
        pad = max(0, self.cols - 3 - len(label))
        return c("───", DIM) + c(label, lcolor) + c("─" * pad, DIM)

    def _layout(self):
        """Wrap the buffer to the window. Returns (display rows, cursor rc)."""
        rows, col, cur = ["› "], 2, None
        for i, ch in enumerate(self.buf):
            if i == self.pos:
                cur = (len(rows) - 1, col)
            if ch == "\n":
                rows.append("  ")      # continuation lines sit under the text
                col = 2
                continue
            w = _cw(ch)
            if col + w > self.cols:
                rows.append("")
                col = 0
            rows[-1] += ch
            col += w
        if cur is None:
            cur = (len(rows) - 1, col)
        return rows, cur

    @staticmethod
    def _paint(row):
        # Matt's own words sit in a soft blue, a shade off the plain-white the
        # model answers in — enough to tell "what I said" from "what it said"
        # at a glance, without the box looking like a christmas tree. The color
        # rides through close(keep=True) too, so the scrollback keeps the
        # distinction after the box is gone.
        if row.startswith("› "):
            return c("› ", BOLD + CYAN) + c(row[2:], USER_TEXT)
        return c(row, USER_TEXT)

    def draw(self, first=False):
        self.cols = max(24, shutil.get_terminal_size((80, 24)).columns)
        out = []
        if first:
            out.append("\r")
        else:
            if self.rows_from_top:
                out.append(f"\033[{self.rows_from_top}A")
            out.append("\r")
        out.append("\033[J")           # wipe the old frame, however tall it was
        rows, (crow, ccol) = self._layout()
        out.append(self._top_rule() + "\r\n")
        for row in rows:
            out.append(self._paint(row) + "\r\n")
        out.append(c("─" * self.cols, DIM))
        up = len(rows) - crow          # bottom rule → the cursor's row
        if up:
            out.append(f"\033[{up}A")
        out.append("\r")
        if ccol:
            out.append(f"\033[{ccol}C")
        self.rows_from_top = crow + 1
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def close(self, keep):
        """Take the box down. `keep` leaves the typed text on screen as a
        plain transcript line, so nothing on screen still looks editable."""
        out = []
        if self.rows_from_top:
            out.append(f"\033[{self.rows_from_top}A")
        out.append("\r\033[J")
        if keep and self.buf:
            for row in self._layout()[0]:
                out.append(self._paint(row) + "\r\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    # — editing —

    def insert(self, text):
        self.buf = self.buf[:self.pos] + text + self.buf[self.pos:]
        self.pos += len(text)

    def _wordback(self):
        i = self.pos
        while i and self.buf[i - 1].isspace():
            i -= 1
        while i and not self.buf[i - 1].isspace():
            i -= 1
        return i

    def _wordfwd(self):
        i, n = self.pos, len(self.buf)
        while i < n and self.buf[i].isspace():
            i += 1
        while i < n and not self.buf[i].isspace():
            i += 1
        return i

    def history(self, step):
        if not self.hist:
            return
        if self.hidx == len(self.hist):
            self.stash = self.buf
        idx = min(len(self.hist), max(0, self.hidx + step))
        if idx == self.hidx:
            return
        self.hidx = idx
        self.buf = self.stash if idx == len(self.hist) else self.hist[idx]
        self.pos = len(self.buf)

    # — reading —

    def _read(self, fd, timeout=None):
        if timeout is not None and not select.select([fd], [], [], timeout)[0]:
            return b""
        try:
            return os.read(fd, 4096)
        except (InterruptedError, OSError):
            return b""

    def _paste_body(self, fd, rest):
        END = b"\033[201~"
        buf = rest
        while END not in buf:
            more = self._read(fd, 2.0)
            if not more:
                break
            buf += more
        i = buf.find(END)
        body, tail = (buf, b"") if i == -1 else (buf[:i], buf[i + len(END):])
        text = body.decode("utf-8", "replace")
        return text.replace("\r\n", "\n").replace("\r", "\n"), tail

    def _escape(self, fd, pend):
        """Parse one escape sequence off the front of `pend`."""
        while len(pend) < 2:
            more = self._read(fd, 0.04)
            if not more:
                return ("key", "esc"), pend[1:]
            pend += more
        second = pend[1:2]
        if second == b"[":
            i = 2
            while True:
                while i >= len(pend):
                    more = self._read(fd, 0.05)
                    if not more:
                        return ("key", "esc"), pend[i:]
                    pend += more
                if 0x40 <= pend[i] <= 0x7E:
                    break
                i += 1
            params, final, rest = pend[2:i], pend[i:i + 1], pend[i + 1:]
            if params == b"200" and final == b"~":
                text, rest = self._paste_body(fd, rest)
                return ("paste", text), rest
            if params == b"201" and final == b"~":
                return ("key", "nop"), rest
            modded = b";" in params            # ⌥/⌃ + arrow
            key = {
                b"A": "up", b"B": "down",
                b"C": "wordright" if modded else "right",
                b"D": "wordleft" if modded else "left",
                b"H": "home", b"F": "end",
            }.get(final)
            if key is None and final == b"~":
                key = {b"1": "home", b"7": "home", b"4": "end", b"8": "end",
                       b"3": "delete"}.get(params, "nop")
            return ("key", key or "nop"), rest
        if second in (b"b", b"B"):
            return ("key", "wordleft"), pend[2:]
        if second in (b"f", b"F"):
            return ("key", "wordright"), pend[2:]
        if second == b"\r":                    # ⌥Return
            return ("key", "newline"), pend[2:]
        if second == b"\x7f":
            return ("key", "killword"), pend[2:]
        return ("key", "esc"), pend[2:]

    CTRL = {
        0x01: "home", 0x02: "left", 0x03: "intr", 0x04: "eof", 0x05: "end",
        0x06: "right", 0x08: "backspace", 0x0A: "newline", 0x0B: "killend",
        0x0C: "clear", 0x0D: "enter", 0x0E: "down", 0x10: "up",
        0x15: "killstart", 0x17: "killword", 0x7F: "backspace",
    }

    def _event(self, fd, pend):
        first = pend[0]
        if first == 0x1B:
            return self._escape(fd, pend)
        if first in self.CTRL:
            return ("key", self.CTRL[first]), pend[1:]
        if first < 0x20:
            return ("key", "nop"), pend[1:]
        i = 0
        while i < len(pend) and pend[i] >= 0x20 and pend[i] not in (0x7F,):
            i += 1
        return ("text", self._dec.decode(pend[:i])), pend[i:]

    # — the loop —

    def run(self):
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        sys.stdout.write("\033[?2004h")     # bracketed paste ON
        sys.stdout.flush()
        try:
            tty.setraw(fd)                  # we handle ⌃C ourselves
            self.draw(first=True)
            return self._loop(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            sys.stdout.write("\033[?2004l")
            sys.stdout.flush()

    def _loop(self, fd):
        pend = b""
        while True:
            if not pend:
                pend = self._read(fd)
                if not pend:                # stdin closed
                    self.close(keep=False)
                    raise EOFError
            (kind, val), pend = self._event(fd, pend)

            if kind == "text" or kind == "paste":
                if val:
                    self.insert(val)
                    self.hidx = len(self.hist)
            elif val == "enter":
                self.close(keep=bool(self.buf.strip()))
                return self.buf
            elif val == "intr":
                self.close(keep=False)
                raise KeyboardInterrupt
            elif val == "eof":
                if not self.buf:
                    self.close(keep=False)
                    raise EOFError
                self.buf = self.buf[:self.pos] + self.buf[self.pos + 1:]
            elif val == "newline":
                self.insert("\n")
            elif val == "backspace":
                if self.pos:
                    self.buf = self.buf[:self.pos - 1] + self.buf[self.pos:]
                    self.pos -= 1
            elif val == "delete":
                self.buf = self.buf[:self.pos] + self.buf[self.pos + 1:]
            elif val == "left":
                self.pos = max(0, self.pos - 1)
            elif val == "right":
                self.pos = min(len(self.buf), self.pos + 1)
            elif val == "wordleft":
                self.pos = self._wordback()
            elif val == "wordright":
                self.pos = self._wordfwd()
            elif val == "home":
                self.pos = 0
            elif val == "end":
                self.pos = len(self.buf)
            elif val == "killend":
                self.buf = self.buf[:self.pos]
            elif val == "killstart":
                self.buf = self.buf[self.pos:]
                self.pos = 0
            elif val == "killword":
                i = self._wordback()
                self.buf = self.buf[:i] + self.buf[self.pos:]
                self.pos = i
            elif val == "up":
                self.history(-1)
            elif val == "down":
                self.history(1)
            elif val == "clear":
                sys.stdout.write("\033[H\033[2J")
                self.rows_from_top = 0
                self.draw(first=True)
                continue
            self.draw()


def framed_input(engine):
    """Read one message from the framed input box (see _LineEditor)."""
    if termios is None or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input("› ")
    unmute_input()          # echo on, and drop whatever was typed while busy
    try:
        line = _LineEditor(engine).run()
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")   # land clear of the frame for whoever catches
        sys.stdout.flush()
        raise
    if line.strip():
        _remember(line)
    mute_input()            # keys pressed while it answers go nowhere
    return line


def take_images(line):
    """Split dropped pictures off a typed line, and say which ones were attached."""
    text, images = attachments.find_images(line)
    for p in images:
        print(c(f"  📎 {os.path.basename(p)}", DIM))
    if images and not text:
        text = "What's in this picture?" if len(images) == 1 else "What's in these pictures?"
    return text, images


def repl(engine):
    if readline:
        try:
            readline.read_history_file(HISTFILE)
        except OSError:
            pass
        readline.set_history_length(1000)
        atexit.register(lambda: readline.write_history_file(HISTFILE))
    # A Moonstone window keeps ONE conversation per chat (MOONSTONE_CHAT_ID from the dock): after the dock unloads an
    # idle local model and restarts it on the next message, the thread picks up where it left off (Matt 10/26)
    chat_id = re.sub(r"[^\w-]", "", os.environ.get("MOONSTONE_CHAT_ID", ""))[:64]
    moon_path = (chat_store.ROOT / f"{chat_store._slug(engine.model_path)}--moonstone-{chat_id}.json") if chat_id else None
    engine.session = moon_path or chat_store.new_snapshot(engine.model_path)
    if moon_path and moon_path.exists():
        n = _restore_session(engine, moon_path)
        if n:
            print(c(f"  picked up this chat's {n} earlier messages", DIM))
    chat_store.prune(engine.model_path)
    if not os.environ.get("AGENT_NO_BANNER"):   # launchers that print their own header (HQ ☁ Free, 10/24)
        print(BANNER)
    ctx_emit(engine)
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
            if moon_path:   # same chat, fresh thread: empty its file so a later reload doesn't bring the old one back
                engine.session = moon_path
                _save_session(engine)
            else:
                engine.session = chat_store.new_snapshot(engine.model_path)
            print(c("  conversation cleared", DIM))
            continue
        if line == "/sessions":
            _show_sessions(chat_store.sessions(engine.model_path))
            continue
        if line.split()[0] == "/resume":
            rows = chat_store.sessions(engine.model_path)
            if not rows:
                print(c("  nothing saved for this model yet", DIM))
                continue
            arg = line.split()[1:]
            try:
                pick = int(arg[0]) if arg else 1
            except ValueError:
                pick = 0
            if not 1 <= pick <= len(rows):
                print(c("  which one? /resume <number>:", DIM))
                _show_sessions(rows)
                continue
            n = _restore_session(engine, rows[pick - 1][0])
            print(c(f"  resumed {n} messages from {chat_store.ago(rows[pick-1][1])}"
                    f" — it remembers the whole thread", DIM))
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
        images = []
        if getattr(engine, "vision", False):
            line, images = take_images(line)
        t0 = time.time()
        try:
            answer = agent_turn(engine, line, images=images)
            _save_session(engine)
            ctx_emit(engine)
            if SPEAK:
                speak(answer)
        except KeyboardInterrupt:
            print(c("\n  interrupted", YELLOW))
            continue
        except Exception as e:
            print(c(f"  error: {type(e).__name__}: {e}", RED))
            continue
        # no elapsed-seconds line: Matt wants the answer only, and the narrator read "3.1s" aloud (2026-10-17)


def main():
    ap = argparse.ArgumentParser(prog="agent", add_help=True)
    ap.add_argument("-p", "--prompt", help="one-shot prompt, print result and exit")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("-C", "--dir", help="working directory")
    args = ap.parse_args()

    if args.dir:
        os.chdir(os.path.expanduser(args.dir))

    for sig in [x for x in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)) if x]:
        signal.signal(sig, lambda *_: (release_seat(), sys.exit(0)))
    atexit.register(release_seat)

    if BACKEND == "http":
        engine = HTTPEngine(args.model)
    else:
        acquire_seat()
        engine = MLXEngine(args.model)

    if args.prompt:
        print(agent_turn(engine, args.prompt, quiet=True), flush=True)  # os._exit below skips the flush
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
