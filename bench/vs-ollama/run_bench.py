#!/usr/bin/env python3
"""Claude Code on claude-code-local (MLX) vs Claude Code on Ollama's Anthropic API.

One setup x model per invocation. The script starts the server it needs,
refuses to run if the other server is up, runs every task REPS times with the
real Claude Code binary in `-p` stream-json mode, checks each result, samples
the server's memory, and writes one JSON line per run to results/raw.jsonl.

    python3 bench/vs-ollama/run_bench.py --setup mlx    --model qwen35-2b
    python3 bench/vs-ollama/run_bench.py --setup ollama --model qwen35-2b

No request may leave the machine: every run goes through HTTPS_PROXY pointed
at a local guard that refuses and logs any CONNECT, and the Claude Code
process's sockets are sampled with lsof. Both counts land in each record.
"""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from tasks import TASKS  # noqa: E402

CLAUDE = os.environ.get("BENCH_CLAUDE_BIN") or os.path.expanduser("~/.local/bin/claude")  # the real binary
MLX_PY = os.environ.get("MLX_PYTHON") or os.path.expanduser("~/.local/mlx-server/bin/python3")
OLLAMA = os.environ.get("BENCH_OLLAMA_BIN") or "/opt/homebrew/bin/ollama"
HF = os.path.expanduser("~/.cache/huggingface/hub")
WORK_ROOT = "/tmp/ccl-bench/runs"
GUARD_PORT = 18999
TIMEOUT_S = 300

# Same family, matched precision. Context: each side as its own docs ship it
# (claude-code-local small-Mac tier = 16K; Ollama's Claude Code page = 64K+).
MODELS = {
    "qwen35-2b": {"mlx": {"dir": f"{HF}/Qwen3.5-2B-MLX-8bit", "tier": "8gb"},
                  "ollama": {"tag": "qwen3.5:2b-q8_0", "tag_t02": "bench-qwen3.5:2b-q8_0-t02"}},
    "qwen35-4b": {"mlx": {"dir": f"{HF}/Qwen3.5-4B-bf16", "tier": "16gb"},
                  "ollama": {"tag": "qwen3.5:4b-bf16", "tag_t02": "bench-qwen3.5:4b-bf16-t02"}},
    "gemma4-e4b": {"mlx": {"dir": f"{HF}/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/"
                                  "4255b21bd9a9d3fc807ef7abd80373f5e3a52a73", "tier": None},
                   "ollama": {"tag": "gemma4:e4b-it-q8_0", "tag_t02": "bench-gemma4:e4b-it-q8_0-t02"}},
    "gemma4-31b": {"mlx": {"dir": f"{HF}/gemma-4-31b-it-abliterated-VL-mlx-8bit", "tier": None},
                   "ollama": {"tag": "gemma4:31b-it-q8_0"}},
}
MLX_CTX = {"8gb": 16384, "16gb": 16384, None: 65536}  # client compact window; Gemma: server has no ceiling
OLLAMA_CTX = 65536

# Text that looks like a tool call but never became a tool_use block.
LEAK_RE = re.compile(
    r"<tool_call>|</tool_call>|<function=|<parameter=|<\|tool_call|call:[A-Za-z_]+\{"
    r"|\{\s*\"name\"\s*:\s*\"(?:Bash|Write|Edit|Read|Glob|Grep)\""
)
# A tool_use Claude Code rejected before running it: unknown tool name or a
# schema violation. Errors from running a well-formed call ("String to replace
# not found", a non-zero exit code) count as valid calls with a bad outcome.
INVALID_RE = re.compile(
    r"InputValidationError|No such tool|is not available|Invalid tool|Unknown tool|"
    r"required parameter|parameter .{0,40} is missing|unexpected parameter|expected (?:string|number|boolean|object|array)",
    re.I)


# ── cloud guard ─────────────────────────────────────────────────────────────
GUARD_HITS = []


def guard_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", GUARD_PORT))
    s.listen(16)
    while True:
        c, _ = s.accept()
        try:
            c.settimeout(2)
            line = c.recv(4096).split(b"\r\n", 1)[0].decode(errors="replace")
            GUARD_HITS.append({"t": time.time(), "req": line})
            c.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        except Exception:  # noqa: BLE001
            pass
        finally:
            c.close()


# ── servers ─────────────────────────────────────────────────────────────────
def listening(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_http(url, secs):
    end = time.time() + secs
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def tier_env(tier):
    if not tier:
        return {}
    out = subprocess.run(["bash", f"{REPO}/scripts/mac-tier.sh", "--env"], env=dict(os.environ, CCL_TIER=tier),
                         capture_output=True, text=True).stdout
    env = {}
    for line in out.splitlines():
        m = re.match(r"export (\w+)=(.*)", line)
        if m:
            env[m.group(1)] = m.group(2).strip("'\"")
    return env


def start_mlx(cfg, log_path, raw=False):
    if listening(11434):
        sys.exit("Ollama is listening on 11434: stop it first (one server at a time).")
    if listening(4000):
        sys.exit("Something already listens on 4000.")
    env = dict(os.environ)
    env.update(tier_env(cfg["tier"]))
    env["MLX_MODEL"] = cfg["dir"]
    env["MLX_PORT"] = "4000"
    # mlx-vlm is not installed on this machine; the tasks are text-only.
    env["MLX_VISION"] = "0"
    if raw:
        # ablation: keep Claude Code's own system prompt (no code-mode rewrite);
        # tool-call parsing/repair is unchanged
        env["MLX_CODE_MODE"] = "0"
    log = open(log_path, "w")
    p = subprocess.Popen([MLX_PY, f"{REPO}/proxy/server.py"], env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    if not wait_http("http://127.0.0.1:4000/health", 600):
        p.kill()
        sys.exit(f"MLX server did not come up; see {log_path}")
    server_env = {k: env[k] for k in sorted(env) if k.startswith("MLX_")}
    return p, server_env


def ollama_pids():
    out = subprocess.run(["pgrep", "-f", r"/opt/homebrew/(bin/ollama serve|Cellar/ollama/)"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def start_ollama(log_path, ctx):
    if listening(4000):
        sys.exit("MLX server is listening on 4000: stop it first (one server at a time).")
    started = None
    if not listening(11434):
        env = dict(os.environ, OLLAMA_CONTEXT_LENGTH=str(ctx))
        log = open(log_path, "w")
        started = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        if not wait_http("http://127.0.0.1:11434/api/version", 60):
            sys.exit("ollama serve did not come up")
    return started


# ── memory sampler ──────────────────────────────────────────────────────────
FOOT_RE = re.compile(r"Footprint: ([\d.]+) (KB|MB|GB)")


def footprint_mb(pids):
    total = 0.0
    for pid in pids:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
        m = FOOT_RE.search(out)
        if m:
            v = float(m.group(1))
            total += v / 1024 if m.group(2) == "KB" else v * 1024 if m.group(2) == "GB" else v
    return total


def rss_mb(pids):
    if not pids:
        return 0.0
    out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / 1024


class Sampler(threading.Thread):
    def __init__(self, pid_fn, claude_pid_fn):
        super().__init__(daemon=True)
        self.pid_fn, self.claude_pid_fn = pid_fn, claude_pid_fn
        self.peak_foot = self.peak_rss = 0.0
        self.remote = set()
        self.stop = False

    def run(self):
        while not self.stop:
            pids = self.pid_fn()
            self.peak_foot = max(self.peak_foot, footprint_mb(pids))
            self.peak_rss = max(self.peak_rss, rss_mb(pids))
            cp = self.claude_pid_fn()
            if cp:
                out = subprocess.run(["lsof", "-a", "-p", str(cp), "-i", "-n", "-P"], capture_output=True,
                                     text=True).stdout
                for line in out.splitlines()[1:]:
                    m = re.search(r"->([\[\]\w:.]+):(\d+)", line)
                    if m and not re.match(r"^(127\.|\[::1\]|::1|localhost)", m.group(1)):
                        self.remote.add(f"{m.group(1)}:{m.group(2)}")
            time.sleep(2)


# ── one Claude Code run ─────────────────────────────────────────────────────
def claude_env(base_url, model_name, compact_window, cfg_dir):
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "HOME": os.path.expanduser("~"),
        "TMPDIR": "/tmp",
        "LANG": "en_US.UTF-8",
        "CLAUDE_CONFIG_DIR": cfg_dir,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "sk-local-dummy",
        "ANTHROPIC_MODEL": model_name,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_name,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_name,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_name,
        "ANTHROPIC_SMALL_FAST_MODEL": model_name,
        "CLAUDE_CODE_SUBAGENT_MODEL": model_name,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
        "HTTPS_PROXY": f"http://127.0.0.1:{GUARD_PORT}",
        "HTTP_PROXY": f"http://127.0.0.1:{GUARD_PORT}",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        # The small-Mac tier's client settings, identical on both sides.
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
        "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS": "4000",
        "BASH_MAX_OUTPUT_LENGTH": "12000",
        "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@localhost",
        "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@localhost",
    }
    if compact_window:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(compact_window)
    return env


def parse_stream(path):
    st = {"tool_use": 0, "tool_invalid": 0, "tool_exec_error": 0, "leaked_calls": 0, "tools": {},
          "thinking_blocks": 0, "result": None, "assistant_msgs": 0, "slashless_paths": 0,
          # malformed output that reached Claude Code, by kind
          "leak_gemma_full": 0, "leak_gemma_no_opener": 0, "leak_function_tag": 0, "leak_tool_call_json": 0,
          "bad_name_case": 0, "bad_name_other": 0, "bad_names": {}}
    declared = set()
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return st
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            declared = set(ev.get("tools") or [])
        if ev.get("type") == "assistant":
            st["assistant_msgs"] += 1
            for b in ev.get("message", {}).get("content", []) or []:
                if b.get("type") == "tool_use":
                    st["tool_use"] += 1
                    st["tools"][b.get("name")] = st["tools"].get(b.get("name"), 0) + 1
                    # an absolute path with its leading "/" dropped ("private/tmp/...")
                    fp = (b.get("input") or {}).get("file_path") or ""
                    if isinstance(fp, str) and re.match(r"(private/|tmp/|Users/)", fp):
                        st["slashless_paths"] += 1
                    nm = b.get("name") or ""
                    if declared and nm not in declared:
                        if nm.lower() in {d.lower() for d in declared}:
                            st["bad_name_case"] += 1
                        else:
                            st["bad_name_other"] += 1
                        st["bad_names"][nm] = st["bad_names"].get(nm, 0) + 1
                elif b.get("type") == "text":
                    txt = b.get("text", "")
                    st["leaked_calls"] += len(LEAK_RE.findall(txt))
                    st["leak_gemma_full"] += len(re.findall(r"<\|tool_call>", txt))
                    st["leak_gemma_no_opener"] += len(re.findall(r"(?<!<\|tool_call>)(?<![\w>])call:[A-Za-z_]+\{", txt))
                    st["leak_function_tag"] += len(re.findall(r"<function=", txt))
                    st["leak_tool_call_json"] += len(re.findall(r"<tool_call>|\{\s*\"name\"\s*:\s*\"(?:Bash|Write|Edit|Read|Glob|Grep)\"", txt))
                elif b.get("type") in ("thinking", "redacted_thinking"):
                    st["thinking_blocks"] += 1
        elif ev.get("type") == "user":
            content = ev.get("message", {}).get("content", [])
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                        txt = json.dumps(b.get("content"))
                        if INVALID_RE.search(txt):
                            st["tool_invalid"] += 1
                        else:
                            st["tool_exec_error"] += 1
        elif ev.get("type") == "result":
            st["result"] = {k: ev.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms",
                                                    "duration_api_ms", "usage", "result")}
            if isinstance(st["result"].get("result"), str):
                st["result"]["result"] = st["result"]["result"][:300]
    return st


def run_one(task, rep, setup, model_key, base_url, model_name, compact_window, sampler_holder):
    rid = f"{setup}-{model_key}-{task['id']}-r{rep}"
    root = os.path.join(WORK_ROOT, rid)
    shutil.rmtree(root, ignore_errors=True)
    work, cfg = os.path.join(root, "work"), os.path.join(root, "cfg")
    os.makedirs(work)
    os.makedirs(cfg)
    task["setup"](work)
    stream = os.path.join(root, "stream.jsonl")
    cmd = [CLAUDE, "-p", task["prompt"], "--output-format", "stream-json", "--verbose", "--bare",
           "--model", model_name, "--dangerously-skip-permissions", "--no-session-persistence"]
    hits_before = len(GUARD_HITS)
    t0 = time.time()
    with open(stream, "w") as out, open(os.path.join(root, "stderr.txt"), "w") as err:
        p = subprocess.Popen(cmd, cwd=work, env=claude_env(base_url, model_name, compact_window, cfg),
                             stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
        sampler_holder["claude_pid"] = p.pid
        timed_out = False
        try:
            p.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        sampler_holder["claude_pid"] = None
    wall = time.time() - t0
    passed, reason = task["check"](work)
    st = parse_stream(stream)
    usage = (st["result"] or {}).get("usage") or {}
    api_s = ((st["result"] or {}).get("duration_api_ms") or 0) / 1000
    out_toks = usage.get("output_tokens") or 0
    return {
        "run_id": rid, "setup": setup, "model": model_key, "model_name": model_name, "task": task["id"], "rep": rep,
        "passed": bool(passed), "check": reason, "timed_out": timed_out, "exit_code": p.returncode,
        "wall_s": round(wall, 1), "api_s": round(api_s, 1), "output_tokens": out_toks,
        "input_tokens": usage.get("input_tokens"),
        "out_tok_per_api_s": round(out_toks / api_s, 1) if api_s > 0 else None,
        "num_turns": (st["result"] or {}).get("num_turns"),
        "tool_use": st["tool_use"], "tool_invalid": st["tool_invalid"], "tool_exec_error": st["tool_exec_error"],
        "leaked_calls": st["leaked_calls"], "tools": st["tools"], "thinking_blocks": st["thinking_blocks"],
        "result_subtype": (st["result"] or {}).get("subtype"), "result_is_error": (st["result"] or {}).get("is_error"),
        "result_text": (st["result"] or {}).get("result"),
        "cloud_guard_hits": GUARD_HITS[hits_before:],
        "t_start": t0,
    }


def mem_free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else None


def swap_used():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()


def warmup(base_url, model_name):
    body = json.dumps({"model": model_name, "max_tokens": 16,
                       "messages": [{"role": "user", "content": "Say ok."}]}).encode()
    req = urllib.request.Request(base_url + "/v1/messages", data=body, headers={
        "content-type": "application/json", "x-api-key": "sk-local-dummy", "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        r.read()
    return round(time.time() - t0, 1)


def main():
    global TIMEOUT_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["mlx", "mlx-raw", "mlx-fix", "ollama", "ollama-16k", "ollama-t02"], required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S, help="seconds per Claude Code run")
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("BENCH_RESULTS") or os.path.join(HERE, "results"), "raw.jsonl"))
    args = ap.parse_args()
    TIMEOUT_S = args.timeout

    threading.Thread(target=guard_server, daemon=True).start()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(WORK_ROOT, exist_ok=True)
    cfg = MODELS[args.model]["ollama" if args.setup.startswith("ollama") else "mlx"]
    logdir = "/tmp/ccl-bench"
    meta = {"setup": args.setup, "model": args.model, "timeout_s": TIMEOUT_S, "claude_version": subprocess.run(
        [CLAUDE, "--version"], capture_output=True, text=True).stdout.strip(), "started": time.time(),
        "load_avg_start": os.getloadavg(), "mem_free_pct_start": mem_free_pct(), "swap_start": swap_used()}
    server_proc = None
    if args.setup == "mlx-fix":
        # same settings as "mlx"; the label marks runs on a server that has the
        # structured tool_calls fix (fix-gemma4-tool-results)
        if "def template_messages" not in open(f"{REPO}/proxy/server.py").read():
            sys.exit("mlx-fix needs the fixed proxy/server.py checked out")
    if args.setup.startswith("mlx"):
        server_proc, server_env = start_mlx(MODELS[args.model]["mlx"], f"{logdir}/{args.setup}-{args.model}.log",
                                            raw=args.setup == "mlx-raw")
        base_url, model_name, compact = "http://127.0.0.1:4000", "claude-sonnet-4-6", MLX_CTX[cfg["tier"]]
        pid_fn = lambda: [server_proc.pid]  # noqa: E731
        meta.update(server="claude-code-local proxy/server.py", server_env=server_env,
                    git_head=subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(), model_path=cfg["dir"])
    else:
        # ollama-16k: control run at the same context as the MLX small-Mac tier
        ctx = 16384 if args.setup == "ollama-16k" else OLLAMA_CTX
        server_proc = start_ollama(f"{logdir}/{args.setup}-serve.log", ctx)
        # ollama-t02: same weights, sampling set to what claude-code-local uses
        # (temperature 0.2, no top-k / top-p / presence penalty); see make_t02_tags.sh
        tag = cfg["tag_t02"] if args.setup == "ollama-t02" else cfg["tag"]
        base_url, model_name, compact = "http://127.0.0.1:11434", tag, ctx
        pid_fn = ollama_pids
        show = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:11434/api/show", data=json.dumps({"model": tag}).encode()), timeout=30).read())
        tags = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10).read())
        digest = next((m["digest"] for m in tags["models"] if m["name"] in (tag, tag + ":latest")), None)
        meta.update(server="ollama " + json.loads(urllib.request.urlopen(
            "http://127.0.0.1:11434/api/version").read())["version"], ollama_tag=tag, ollama_digest=digest,
            ollama_details=show.get("details"), ollama_parameters=show.get("parameters"),
            ollama_context_length=ctx)
    meta["warmup_s"] = warmup(base_url, model_name)

    holder = {"claude_pid": None}
    sampler = Sampler(pid_fn, lambda: holder["claude_pid"])
    sampler.start()
    wanted = [t for t in TASKS if not args.tasks or t["id"] in args.tasks.split(",")]
    results = []
    try:
        for rep in range(1, args.reps + 1):
            for task in wanted:
                sampler.remote.clear()
                rec = run_one(task, rep, args.setup, args.model, base_url, model_name, compact, holder)
                rec["remote_sockets"] = sorted(sampler.remote)
                results.append(rec)
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{rec['run_id']}: {'PASS' if rec['passed'] else 'FAIL'} {rec['wall_s']}s "
                      f"tools={rec['tool_use']} inv={rec['tool_invalid']} leak={rec['leaked_calls']} "
                      f"{rec['check'][:60]}", flush=True)
    finally:
        sampler.stop = True
        time.sleep(2.5)
        meta.update(finished=time.time(), load_avg_end=os.getloadavg(), mem_free_pct_end=mem_free_pct(), peak_footprint_mb=round(sampler.peak_foot),
                    peak_rss_mb=round(sampler.peak_rss), runs=len(results),
                    passed=sum(r["passed"] for r in results), cloud_guard_hits_total=len(GUARD_HITS))
        with open(os.path.join(os.path.dirname(args.out), "meta.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
        if args.setup.startswith("mlx") and server_proc:
            os.killpg(server_proc.pid, signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server_proc.pid, signal.SIGKILL)
        if args.setup.startswith("ollama"):
            # unload the model so the next setup starts from the same memory state
            try:
                urllib.request.urlopen(urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=json.dumps({"model": model_name, "keep_alive": 0}).encode()), timeout=60).read()
            except Exception:  # noqa: BLE001
                pass
    print(json.dumps({k: meta[k] for k in ("setup", "model", "runs", "passed", "peak_footprint_mb",
                                           "peak_rss_mb", "cloud_guard_hits_total")}))


if __name__ == "__main__":
    main()
