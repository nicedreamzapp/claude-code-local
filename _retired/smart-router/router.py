#!/usr/bin/env python3
"""
ONE AI — smart local model router.

Claude Code talks Anthropic to this router (one endpoint). The router reads each
request, picks the best local backend in microseconds (pure heuristics, no LLM
call), makes sure that backend is up, and transparently forwards/streams the
Anthropic response straight back. No translation needed — every backend already
speaks Anthropic.

Backends (all Anthropic-compatible):
  qwen     -> MLX server  :4000  (Qwen 3.6 35B-A3B 8-bit)     DEFAULT / code / agentic
  gemma    -> MLX server  :4002  (Gemma 4)                    quick / trivial
  deepseek -> ds4 server  :8000  (DeepSeek V4 Flash 284B)     huge context / hard reasoning

Note: the MLX server holds ONE model at a time, so qwen<->gemma is a swap
(restart). The decision is instant; switching the actually-loaded model costs a
load. Default stays Qwen warm so the common path never pauses.
"""
import json, os, re, subprocess, sys, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(os.environ.get("ONE_AI_PORT", "4010"))
MLX_URL   = "http://127.0.0.1:4000"
DS4_URL   = "http://127.0.0.1:8000"
SETUP     = os.path.expanduser("~/Desktop/PROJECTS/Local AI Setup")
LAUNCH_LIB = f"{SETUP}/launchers/lib/claude-local-common.sh"

# (local_path, repo_id) pairs — resolved via the launchers' resolve_mlx_model
# helper so the MLX server gets a real local path, not a bare name it tries to
# download. Matches what Qwen 3 Coder.command / Gemma 4 Code.command pass.
MLX_MODELS = {
    # DEFAULT/code lane: Qwen 3.6 35B-A3B 8-bit — reigning Agent-12 champion
    # (perfect 20/20 across every sweep, 2026-08-10), ~46 tok/s on the M5.
    # It REPLACED Qwen3-Coder-30B here on 2026-08-11: two Qwens in the same lane
    # was 30 GB of duplicate weights, and 3.6 beat the coder on the gauntlet.
    "qwen":   ('$HOME/.cache/huggingface/hub/Qwen3.6-35B-A3B-MLX-8bit',
               'lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit'),
    # Qwen3-Coder-Next 80B (Opus-4.6 reasoning-distilled) DELETED 2026-06-22 — benchmarked
    # as useless: bloats/never finishes one-shot builds (0/18 on a hard parser, no code at
    # all), and in agentic mode it won't even call a tool. The plain 30B beat it on
    # everything. 42GB reclaimed. Default coder stays the 30B "qwen" lane.
    "gemma":  ('$HOME/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx',
               'divinetribe/gemma-4-31b-it-abliterated-4bit-mlx'),
    # GLM-4.5-Air removed 2026-06-21 (outdated July-2025 model, deleted from disk).
    # Hard-reasoning lane now goes to DeepSeek V4 Flash 284B.
    # Qwen3-VL 32B: the only model that can SEE images -> vision lane.
    "qwenvl": ('$HOME/.cache/huggingface/hub/Huihui-Qwen3-VL-32B-Instruct-abliterated-4bit-mlx',
               'divinetribe/Huihui-Qwen3-VL-32B-Instruct-abliterated-4bit-mlx'),
}

BIG_CONTEXT_TOKENS = 100_000          # above this -> DeepSeek's 1M context
HARD_HINTS = re.compile(r"\b(think hard|prove|derive|architect|design the|reason through|hard problem|step by step)\b", re.I)
# Qwen is the safe default. Gemma is ONLY for obviously trivial chitchat — a very
# short greeting/ack with no technical content. Anything substantive -> Qwen.
TRIVIAL = re.compile(r"^\s*(hi|hey|hello|yo|thanks|thank you|ok|okay|cool|nice|got it|sup|good morning|good night|how are you|what'?s up)\b[\s!.?]*$", re.I)


# ----- routing brain: instant, no model call --------------------------------
def estimate_tokens(messages) -> int:
    chars = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict):
                    chars += len(json.dumps(blk))
    return chars // 4

def last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""

def has_image(messages) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "image":
                    return True
    return False

def route(body) -> tuple[str, str]:
    """Return (backend_key, reason). Microsecond heuristics."""
    messages = body.get("messages", [])
    text = last_user_text(messages)
    low = text.lower()

    # explicit overrides win
    if "/deep" in low:  return "deepseek", "override /deep"
    if "/glm"  in low:  return "deepseek", "override /glm (GLM removed -> DeepSeek)"
    if "/fast" in low:  return "gemma",    "override /fast"
    if "/code" in low:  return "qwen",     "override /code"

    # NOTE: local vision is unavailable — Qwen3-VL needs mlx-vlm, not this
    # mlx_lm text server (verified). Images fall through to the text default;
    # the model won't SEE the image. (Future: a separate mlx-vlm server.)
    toks = estimate_tokens(messages)
    if toks > BIG_CONTEXT_TOKENS:
        return "deepseek", f"large context ~{toks//1000}k tok (1M ctx)"
    if HARD_HINTS.search(text):
        return "deepseek", "hard reasoning -> DeepSeek V4 Flash 284B (local)"
    if TRIVIAL.match(text.strip()):
        return "gemma", "trivial chitchat"
    return "qwen", "default (code/agentic)"


# ----- backend management ----------------------------------------------------
def _mlx_running_model() -> str:
    try:
        with urllib.request.urlopen(f"{MLX_URL}/health", timeout=2) as r:
            return (json.load(r) or {}).get("model", "") or ""
    except Exception:
        return ""

# Warm pool: these two text models stay loaded together on their own ports, so
# switching between them is INSTANT (no load/unload). ~46 GB total.
WARM_PORTS = {"qwen": 4000, "gemma": 4002}  # 4001 is the Free AI gateway
WARM_SH = f"{SETUP}/smart-router/warm_pool.sh"
GLM_PORT = 4003   # GLM gets its own port; loading it displaces the warm pool

def _port_listening(port: int) -> bool:
    return subprocess.run(["bash", "-lc", f"lsof -i :{port} -sTCP:LISTEN >/dev/null 2>&1"]).returncode == 0

def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def _wait_health(port: int, timeout=180):
    for _ in range(timeout // 2):
        if _health_ok(port): return True
        time.sleep(2)
    return False

def _stop_ds4():
    subprocess.run(["bash", "-lc", "pkill -f ds4-server 2>/dev/null; true"], check=False)

def _stop_glm():
    subprocess.run(["bash", "-lc", f"lsof -ti :{GLM_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null; true"], check=False)

def _start_warm_pool():
    subprocess.run(["bash", WARM_SH, "start"], check=False)

def _stop_warm_pool():
    subprocess.run(["bash", WARM_SH, "stop"], check=False)

def ensure_backend(backend: str):
    """Warm pair (qwen/gemma) = instant, own ports. The 80GB giants (glm/deepseek)
    can't coexist with the pool, so reaching one unloads the pool (and vice versa)."""
    # --- warm pair: forward to its dedicated port, no swap ---
    if backend in WARM_PORTS:
        port = WARM_PORTS[backend]
        if not _port_listening(port):
            # pool not up (a giant displaced it, or cold start) -> bring it back
            if _port_listening(8000):
                sys.stderr.write("[one-ai] unloading DeepSeek to restore warm pool\n"); _stop_ds4()
            if _port_listening(GLM_PORT):
                sys.stderr.write("[one-ai] unloading GLM to restore warm pool\n"); _stop_glm()
            _start_warm_pool(); _wait_health(port)
        return f"http://127.0.0.1:{port}"
    # --- DeepSeek 284B: exclusive ---
    if backend == "deepseek":
        if _port_listening(4000) or _port_listening(4002):
            sys.stderr.write("[one-ai] unloading warm pool for DeepSeek 284B\n"); _stop_warm_pool()
        if _port_listening(GLM_PORT): _stop_glm()
        subprocess.run(["bash", "-lc", os.path.expanduser("~/.local/bin/ds4-server-up")],
                       check=False, capture_output=True)
        return DS4_URL
    # GLM-4.5-Air lane removed 2026-06-21 (model deleted; hard reasoning -> deepseek)
    # fallback (e.g. images) -> default warm qwen
    return ensure_backend("qwen")


# ----- transparent Anthropic passthrough ------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except Exception:
            body = {}

        backend, reason = route(body)

        def attempt(be):
            base = ensure_backend(be)
            req = urllib.request.Request(base + self.path, data=raw, method="POST")
            for h in ("content-type", "anthropic-version", "anthropic-beta", "authorization", "x-api-key"):
                if h in self.headers:
                    req.add_header(h, self.headers[h])
            req.add_header("content-type", "application/json")
            return urllib.request.urlopen(req, timeout=900)

        t0 = time.time()
        try:
            up = attempt(backend)
        except Exception as e:
            # a backend is unhealthy (e.g. port squatted) -> fall back to qwen,
            # which the warm pool keeps reliable. The common path never sees this.
            if backend != "qwen":
                sys.stderr.write(f"[one-ai] {backend} failed ({e}); falling back to qwen\n")
                backend, reason = "qwen", f"{backend} failed -> qwen fallback"
                try:
                    up = attempt("qwen")
                except Exception as e2:
                    self.send_response(502); self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e2)}).encode()); return
            else:
                code = getattr(e, "code", 502)
                self.send_response(code); self.end_headers()
                body_err = e.read() if hasattr(e, "read") else json.dumps({"error": str(e)}).encode()
                self.wfile.write(body_err); return
        sys.stderr.write(f"[one-ai] -> {backend:8s} ({reason})  prep {time.time()-t0:.2f}s\n")
        sys.stderr.flush()

        self.send_response(up.status)
        for k, v in up.getheaders():
            if k.lower() in ("content-length", "transfer-encoding", "connection"): continue
            self.send_header(k, v)
        self.send_header("x-one-ai-model", backend)   # tell callers which model answered
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        # stream chunks straight through (SSE for streaming responses)
        while True:
            chunk = up.read(2048)
            if not chunk: break
            self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
            self.wfile.write(b'{"status":"ok","router":"one-ai"}')
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        tests = [
            ("hi", "messages with a short greeting"),
            ("refactor this async function and fix the stack trace", "code"),
            ("/deep walk me through the proof", "deep override"),
            ("/fast what's 2+2", "fast override"),
        ]
        for txt, label in tests:
            b, r = route({"messages": [{"role": "user", "content": txt}]})
            print(f"  {label:28s} -> {b:9s} ({r})")
        # a synthetic huge-context request
        big = {"messages": [{"role": "user", "content": "x" * 500_000}]}
        b, r = route(big); print(f"  {'500k-char context':28s} -> {b:9s} ({r})")
        sys.exit(0)
    print(f"ONE AI router on :{LISTEN_PORT}  (qwen default · gemma quick · deepseek deep)")
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
