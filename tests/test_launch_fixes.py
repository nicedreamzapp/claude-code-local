"""Three launch fixes:

1. A local run tells Claude Code the real model ("local:<model>"), so its header stops saying
   "Sonnet 4.6".
2. The sk-local placeholder is pre-approved in ~/.claude.json the way Claude Code records a Yes to
   "Detected a custom API key", so the first `keep going` doesn't stop on that question.
3. Nothing stops servers by name across the machine: the launchers stop only the server listening
   on their own port, and port-based stops never hit connected clients.
"""
import http.server
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time

import pytest

from conftest import ROOT, load_module

keepgoing = load_module("keepgoing", ROOT / "bin" / "keepgoing.py")
router = load_module("router", ROOT / "smart-router" / "router.py")
BASH = shutil.which("bash")
LSOF = shutil.which("lsof")
COMMON = ROOT / "launchers" / "lib" / "claude-local-common.sh"


# ─── 1. the header names the local model ─────────────────────────────────────

@pytest.mark.parametrize("model,alias", [
    ("mlx-community/Qwen3.5-4B-bf16", "local:Qwen3.5-4B-bf16"),
    ("/Users/me/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx/",
     "local:gemma-4-31b-it-abliterated-4bit-mlx"),
    ("Qwen 3.5 4B [Q4_K_M]", "local:Qwen-3.5-4B-Q4_K_M"),
    ("C:\\models\\qwen.gguf", "local:qwen.gguf"),
    ("", "local"),
    (None, "local"),
])
def test_local_model_alias(model, alias):
    assert keepgoing.local_model_alias(model) == alias


def test_local_alias_has_no_claude_model_name():
    for model, _, _ in keepgoing.LOCAL:
        a = keepgoing.local_model_alias(model)
        assert a.startswith("local:") and " " not in a and "claude" not in a.lower()


def test_cli_prints_the_alias():
    r = subprocess.run([sys.executable, str(ROOT / "bin" / "keepgoing.py"), "--local-alias",
                        "mlx-community/Qwen3.5-4B-bf16"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout.strip() == "local:Qwen3.5-4B-bf16"


def test_servers_answer_any_model_name():
    """The alias is only safe because neither local server picks a model by name."""
    for path in (ROOT / "proxy" / "server.py", ROOT / "portable" / "portable_server.py"):
        src = path.read_text()
        assert "MODEL_MAP" not in src and "model not found" not in src.lower()


# ─── 2. sk-local pre-approved in ~/.claude.json ──────────────────────────────

def test_approve_creates_the_entry_and_keeps_every_other_key(tmp_path):
    cfg = tmp_path / ".claude.json"
    original = {"hasCompletedOnboarding": True, "mcpServers": {"x": {"command": "y"}},
                "projects": {"/p": {"allowedTools": ["Bash"]}}, "numStartups": 12,
                "customApiKeyResponses": {"approved": ["abc123"], "rejected": ["zzz"]}}
    cfg.write_text(json.dumps(original))
    os.chmod(cfg, 0o600)
    assert keepgoing.approve_local_key(str(cfg)) == "approved"
    got = json.loads(cfg.read_text())
    want = dict(original, customApiKeyResponses={"approved": ["abc123", "sk-local"], "rejected": ["zzz"]})
    assert got == want
    # backed up first, byte for byte, and the file mode is kept
    assert json.loads((tmp_path / ".claude.json.ccl-backup").read_text()) == original
    assert stat.S_IMODE(os.stat(cfg).st_mode) == 0o600 or os.name == "nt"
    # atomic write leaves no temp file behind
    assert sorted(p.name for p in tmp_path.iterdir()) == [".claude.json", ".claude.json.ccl-backup"]


def test_approve_is_a_no_op_when_already_approved(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text('{"customApiKeyResponses": {"approved": ["sk-local"], "rejected": []}}')
    before = cfg.read_bytes()
    assert keepgoing.approve_local_key(str(cfg)) == "already"
    assert cfg.read_bytes() == before
    assert not (tmp_path / ".claude.json.ccl-backup").exists()


def test_approve_stores_the_last_20_characters(tmp_path):
    cfg = tmp_path / ".claude.json"
    key = "sk-local-" + "x" * 30 + "ABCDEFGHIJ"
    assert keepgoing.approve_local_key(str(cfg), key) == "approved"
    assert json.loads(cfg.read_text())["customApiKeyResponses"]["approved"] == [key[-20:]]


def test_approve_creates_a_missing_file(tmp_path):
    cfg = tmp_path / "sub" / ".claude.json"
    assert keepgoing.approve_local_key(str(cfg)) == "approved"
    assert json.loads(cfg.read_text()) == {
        "customApiKeyResponses": {"approved": ["sk-local"], "rejected": []}}


def test_approve_respects_a_rejection(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text('{"customApiKeyResponses": {"approved": [], "rejected": ["sk-local"]}}')
    before = cfg.read_bytes()
    assert keepgoing.approve_local_key(str(cfg)) == "rejected"
    assert cfg.read_bytes() == before


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"customApiKeyResponses": "x"}'])
def test_approve_never_touches_a_file_it_cannot_read(tmp_path, content):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(content)
    assert keepgoing.approve_local_key(str(cfg)) == "skipped"
    assert cfg.read_text() == content
    assert not (tmp_path / ".claude.json.ccl-backup").exists()


def test_config_file_follows_claude_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert keepgoing.claude_config_file() == str(tmp_path / ".claude.json")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert keepgoing.claude_config_file() == os.path.join(os.path.expanduser("~"), ".claude.json")


def test_cli_approves_the_key(tmp_path):
    env = dict(os.environ, CLAUDE_CONFIG_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, str(ROOT / "bin" / "keepgoing.py"), "--approve-local-key"],
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout == ""
    assert json.loads((tmp_path / ".claude.json").read_text())["customApiKeyResponses"]["approved"] == ["sk-local"]


def test_portable_run_and_keepgoing_use_the_same_helpers(tmp_path, monkeypatch):
    ccl = load_module("ccl_portable_fixes", ROOT / "portable" / "ccl.py")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    env = ccl.local_claude_env("Qwen 3.5 4B [Q4_K_M]")
    assert env["ANTHROPIC_MODEL"] == env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "local:Qwen-3.5-4B-Q4_K_M"
    assert env["ANTHROPIC_API_KEY"] == "sk-local"
    assert json.loads((tmp_path / ".claude.json").read_text())["customApiKeyResponses"]["approved"] == ["sk-local"]
    src = (ROOT / "portable" / "ccl.py").read_text()
    assert '"claude-sonnet-4-6"' not in src


# ─── bin/keepgoing (the Mac path) end to end ─────────────────────────────────

FAKE_CLAUDE = """#!/bin/sh
{ printf '%s\\n' "$@"; echo "MODEL=$ANTHROPIC_MODEL"; echo "KEY=$ANTHROPIC_API_KEY"; } > "$KG_LOG"
"""


@pytest.mark.skipif(sys.platform != "darwin" or not BASH, reason="bin/keepgoing's MLX path is macOS")
def test_mac_keepgoing_names_the_model_and_preapproves_the_key(tmp_path):
    port = socket.socket()
    port.bind(("127.0.0.1", 0))
    p = port.getsockname()[1]
    port.close()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "model": "mlx-community/Qwen3.5-4B-bf16"}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", p), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    home = tmp_path / "home"
    work = tmp_path / "proj"
    sess = home / ".claude" / "projects" / "x"
    for d in (work, sess):
        d.mkdir(parents=True)
    lines = [{"type": "user", "cwd": str(work), "message": {"content": "hi " * 80}}] * 2
    (sess / "mac-session.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    (home / ".claude.json").write_text('{"keep": "me"}')
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "claude").write_text(FAKE_CLAUDE)
    (stubs / "claude").chmod(0o755)
    log = tmp_path / "claude.log"
    env = {"PATH": f"{stubs}:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home), "MLX_PORT": str(p),
           "KG_LOG": str(log)}
    try:
        r = subprocess.run([BASH, str(ROOT / "bin" / "keepgoing"), "--auto"], cwd=work, env=env,
                           capture_output=True, text=True, timeout=60)
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.returncode == 0, r.stdout + r.stderr
    out = log.read_text().splitlines()
    assert out[:4] == ["--model", "local:Qwen3.5-4B-bf16", "--resume", "mac-session"]
    assert "MODEL=local:Qwen3.5-4B-bf16" in out and "KEY=sk-local" in out
    cfg = json.loads((home / ".claude.json").read_text())
    assert cfg == {"keep": "me", "customApiKeyResponses": {"approved": ["sk-local"], "rejected": []}}
    assert "claude-sonnet-4-6" not in (ROOT / "bin" / "keepgoing").read_text()


# ─── 3. stop only our own server ─────────────────────────────────────────────

KILLERS = ("pkill", "killall", "taskkill")


def test_no_name_based_kills_anywhere():
    """pkill / killall / taskkill match by name across the whole machine: other users' and other
    jobs' servers die too. Code stops a server by its pid or by the port it listens on."""
    hits = []
    for path in ROOT.rglob("*"):
        if (not path.is_file() or path.suffix not in (".sh", ".py", ".command", ".ps1", ".cmd", "")
                or any(part in ("tests", "docs", ".git", "__pycache__", "repo", "up", "node_modules")
                       for part in path.relative_to(ROOT).parts)):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0] if path.suffix != ".py" else line
            if any(k + " " in code for k in KILLERS) and "`" not in code:
                hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert hits == []


def _listener(script_path, port):
    """A process listening on port whose command line contains script_path."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "import socket, sys, time\n"
        "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(5)\n"
        "conns = []\n"
        "while True:\n"
        "    c, _ = s.accept(); conns.append(c)\n")
    return subprocess.Popen([sys.executable, str(script_path), str(port)])


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_listening(port, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def _alive(proc, timeout=5):
    try:
        proc.wait(timeout=timeout)
        return False
    except subprocess.TimeoutExpired:
        return True


@pytest.mark.skipif(not BASH or not LSOF or os.name == "nt", reason="needs bash and lsof")
def test_launcher_finds_only_the_server_on_its_port(tmp_path):
    a, b, c = _free_port(), _free_port(), _free_port()
    ours = _listener(tmp_path / "a" / "mlx-native-server" / "server.py", a)
    other = _listener(tmp_path / "b" / "mlx-native-server" / "server.py", b)
    not_a_server = _listener(tmp_path / "c" / "something-else.py", c)
    try:
        for port in (a, b, c):
            assert _wait_listening(port)

        def pids(port):
            r = subprocess.run([BASH, "-c", f'source "{COMMON}"; _mlx_server_pids {port}'],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.split()
        assert pids(a) == [str(ours.pid)]
        assert pids(b) == [str(other.pid)]
        assert pids(c) == []  # something else on the port is never ours to stop
    finally:
        for p in (ours, other, not_a_server):
            p.kill()
            p.wait()


@pytest.mark.skipif(not BASH or os.name == "nt", reason="needs bash")
def test_stop_mlx_server_kills_only_what_the_port_lookup_returns(tmp_path):
    """_stop_mlx_server asks _mlx_server_pids for port 4000 and kills only those. Stubbed so the
    test never touches a real server on 4000."""
    target = subprocess.Popen(["sleep", "60"])
    bystander = subprocess.Popen(["sleep", "60"])
    try:
        script = (f'source "{COMMON}"\n'
                  f'_mlx_server_pids() {{ [ "$1" = 4000 ] && echo {target.pid}; }}\n'
                  'lsof() { return 1; }\n'
                  '_stop_mlx_server && echo stopped\n')
        r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)
        assert "stopped" in r.stdout, r.stderr
        assert not _alive(target)
        assert _alive(bystander, timeout=0.5)
    finally:
        for p in (target, bystander):
            p.kill()
            p.wait()
    assert "pkill" not in COMMON.read_text().split("_stop_mlx_server() {", 1)[1].split("\n}", 1)[0]


@pytest.mark.skipif(not BASH or not LSOF or os.name == "nt", reason="needs bash and lsof")
def test_router_stop_hits_the_listener_not_its_clients(tmp_path):
    port = _free_port()
    server = _listener(tmp_path / "ds4-server.py", port)
    assert _wait_listening(port)
    client = subprocess.Popen([sys.executable, "-c",
                               f"import socket, time; s = socket.create_connection(('127.0.0.1', {port})); "
                               "time.sleep(60)"])
    try:
        time.sleep(0.5)
        subprocess.run([BASH, "-c", router._stop_listener_cmd(port)], timeout=30, check=False)
        assert not _alive(server)
        assert _alive(client, timeout=0.5)
    finally:
        for p in (server, client):
            p.kill()
            p.wait()


def test_port_stops_only_target_listeners():
    for path in (ROOT / "smart-router" / "warm_pool.sh", ROOT / "smart-router" / "router.py",
                 *sorted((ROOT / "bench" / "vs-ollama").glob("run_*.sh"))):
        for line in path.read_text().splitlines():
            if "lsof" in line and "kill" in line and "#" not in line.split("lsof")[0]:
                assert "-sTCP:LISTEN" in line, f"{path.name}: {line.strip()}"
