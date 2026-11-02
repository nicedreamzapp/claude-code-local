"""keep going on Linux and Windows: bin/keepgoing.py's portable paths (folder matching, config
dir, server port, the installed llama.cpp model) and `ccl.py keepgoing`, which resumes the
conversation against the llama.cpp proxy or a free cloud model.

Nothing here downloads a model or starts llama.cpp. The end-to-end test runs the real
`ccl.py keepgoing` in a subprocess against a stub /health server and a stub `claude`.
"""
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

from conftest import ROOT, load_module

keepgoing = load_module("keepgoing", ROOT / "bin" / "keepgoing.py")
ccl = load_module("ccl_portable", ROOT / "portable" / "ccl.py")
BASH = shutil.which("bash")
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_session(projects, name, cwd, user_turns, mtime=None):
    d = projects / "proj"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.jsonl"
    lines = [json.dumps({"type": "system", "cwd": cwd, "pad": "x" * 200})]
    lines += [json.dumps({"type": "user", "message": "m"})] * user_turns
    path.write_text("\n".join(lines) + "\n")
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


def write_config(home, name="Qwen 3.5 4B", quant="Q4_K_M"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(json.dumps({"model_name": name, "quant": quant}))


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(keepgoing, "IS_MAC", False)
    monkeypatch.setattr(keepgoing, "IS_WINDOWS", False)


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(keepgoing, "IS_MAC", False)
    monkeypatch.setattr(keepgoing, "IS_WINDOWS", True)


# ─── path handling ───────────────────────────────────────────────────────────

def test_windows_folders_match_ignoring_case_and_slashes(windows):
    assert keepgoing.same_folder(r"C:\Users\Me\proj", r"c:\users\me\proj")
    assert keepgoing.same_folder(r"C:\Users\Me\proj", "C:/Users/Me/proj")
    assert keepgoing.same_folder(r"C:\Users\Me\proj\\", r"C:\Users\Me\proj")
    assert not keepgoing.same_folder(r"C:\Users\Me\proj", r"C:\Users\Me\proj2")
    assert not keepgoing.same_folder(None, r"C:\x")


def test_linux_and_mac_folders_match_exactly(linux):
    assert keepgoing.same_folder("/home/me/proj", "/home/me/proj")
    assert not keepgoing.same_folder("/home/me/Proj", "/home/me/proj")


def test_find_session_with_windows_paths(tmp_path, monkeypatch, windows):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    now = time.time()
    write_session(tmp_path, "older", r"C:\Users\Me\proj", 2, now - 50)
    write_session(tmp_path, "newer", r"c:\users\me\proj", 3, now)
    write_session(tmp_path, "other", r"C:\Users\Me\elsewhere", 5, now + 50)
    sid, turns, _ = keepgoing.find_session(r"C:\Users\Me\proj")
    assert (sid, turns) == ("newer", 3)


def test_projects_dir_follows_claude_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    mod = load_module("keepgoing_cfgdir", ROOT / "bin" / "keepgoing.py")
    assert mod.PROJECTS == os.path.join(str(tmp_path / "cfg"), "projects")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    mod = load_module("keepgoing_cfgdir", ROOT / "bin" / "keepgoing.py")
    assert mod.PROJECTS == os.path.join(os.path.expanduser("~/.claude"), "projects")


def test_portable_home_matches_ccl(tmp_path, monkeypatch):
    monkeypatch.setenv("CCL_HOME", str(tmp_path / "h"))
    assert keepgoing.portable_home() == ccl.home() == str(tmp_path / "h")
    monkeypatch.delenv("CCL_HOME")
    monkeypatch.setattr(keepgoing, "IS_WINDOWS", False)
    assert keepgoing.portable_home() == os.path.join(os.path.expanduser("~"), ".cache", "claude-code-local")
    monkeypatch.setattr(keepgoing, "IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert keepgoing.portable_home() == os.path.join(str(tmp_path / "AppData" / "Local"), "claude-code-local")


# ─── server URL ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("env, port", [({}, 4000), ({"CCL_PORT": "4100"}, 4100),
                                       ({"MLX_PORT": "4200", "CCL_PORT": "4100"}, 4200)])
def test_keepgoing_port(monkeypatch, env, port):
    for k in ("MLX_PORT", "CCL_PORT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = load_module("keepgoing_port", ROOT / "bin" / "keepgoing.py")
    assert mod.PORT == port


@pytest.mark.parametrize("env, proxy, llama", [({}, 4000, 8080),
                                               ({"CCL_PORT": "4100", "CCL_LLAMA_PORT": "8181"}, 4100, 8181)])
def test_ccl_ports(monkeypatch, env, proxy, llama):
    for k in ("CCL_PORT", "CCL_LLAMA_PORT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = load_module("ccl_ports", ROOT / "portable" / "ccl.py")
    assert (mod.PROXY_PORT, mod.LLAMA_PORT) == (proxy, llama)
    assert mod.PROXY_URL == f"http://localhost:{proxy}"
    assert mod.local_claude_env()["ANTHROPIC_BASE_URL"] == f"http://localhost:{proxy}"


def test_running_local_model_reads_the_portable_health(monkeypatch):
    port = free_port()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "model": "Qwen 3.5 4B [Q4_K_M]",
                               "backend": "llama.cpp"}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(keepgoing, "PORT", port)
        assert keepgoing.running_local_model() == "Qwen 3.5 4B [Q4_K_M]"
        monkeypatch.setattr(ccl, "PROXY_PORT", port)
        assert ccl.proxy_is_up()
    finally:
        srv.shutdown()
        srv.server_close()
    assert not ccl.proxy_is_up()


# ─── the installed model and the menu ────────────────────────────────────────

def test_lineup_is_the_installed_llama_model(tmp_path, monkeypatch, linux):
    monkeypatch.setenv("CCL_HOME", str(tmp_path / "h"))
    assert keepgoing.local_lineup() == []
    write_config(tmp_path / "h")
    assert keepgoing.local_lineup() == [("Qwen 3.5 4B [Q4_K_M]", "Qwen 3.5 4B [Q4_K_M]", 0)]
    (tmp_path / "h" / "config.json").write_text("{broken")
    assert keepgoing.local_lineup() == []


def test_mac_lineup_unchanged(monkeypatch):
    monkeypatch.setattr(keepgoing, "IS_MAC", True)
    assert keepgoing.local_lineup() is keepgoing.LOCAL


def test_auto_pick_with_the_portable_lineup():
    lineup = [("Qwen 3.5 4B [Q4_K_M]", "Qwen 3.5 4B [Q4_K_M]", 0)]
    assert keepgoing.auto_pick(["a:free"], None, 0, lineup) == ("cloud", "a:free", 0)
    assert keepgoing.auto_pick([], "Qwen 3.5 4B [Q4_K_M]", 0, lineup) == ("local", "Qwen 3.5 4B [Q4_K_M]", 1)
    assert keepgoing.auto_pick([], None, 0, lineup) == ("local", "Qwen 3.5 4B [Q4_K_M]", 0)
    assert keepgoing.auto_pick([], None, 0, []) is None


def portable_setup(tmp_path, monkeypatch, live=None, installed=True):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path / "projects"))
    write_session(tmp_path / "projects", "sess", str(tmp_path), 2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CCL_HOME", str(tmp_path / "h"))
    if installed:
        write_config(tmp_path / "h")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(keepgoing, "running_local_model", lambda: live)
    monkeypatch.setattr(keepgoing, "mac_ram_gb", lambda: 0)


def test_auto_on_linux_picks_the_installed_model(tmp_path, monkeypatch, capsys, linux):
    portable_setup(tmp_path, monkeypatch)
    keepgoing.main(["--auto"])
    out = capsys.readouterr()
    assert out.out == "sess\tlocal\tQwen 3.5 4B [Q4_K_M]\t0\n"
    assert "this PC" in out.err and "this Mac" not in out.err


def test_auto_on_linux_reuses_the_running_proxy(tmp_path, monkeypatch, capsys, linux):
    portable_setup(tmp_path, monkeypatch, live="Qwen 3.5 4B [Q4_K_M]")
    keepgoing.main(["--auto"])
    assert capsys.readouterr().out == "sess\tlocal\tQwen 3.5 4B [Q4_K_M]\t1\n"


def test_menu_on_windows_lists_the_installed_model_once(tmp_path, monkeypatch, capsys, windows):
    portable_setup(tmp_path, monkeypatch, live="Qwen 3.5 4B [Q4_K_M]")
    monkeypatch.setattr("builtins.input", lambda: "1")
    keepgoing.main([])
    out = capsys.readouterr()
    assert "ON THIS PC" in out.err and "Mac" not in out.err
    assert out.err.count("Qwen 3.5 4B [Q4_K_M]") == 1
    assert out.out == "sess\tlocal\tQwen 3.5 4B [Q4_K_M]\t1\n"


def test_menu_without_an_install_says_so(tmp_path, monkeypatch, capsys, linux):
    portable_setup(tmp_path, monkeypatch, installed=False)
    monkeypatch.setattr("builtins.input", lambda: "1")
    with pytest.raises(SystemExit):
        keepgoing.main([])
    assert "no local model installed yet" in capsys.readouterr().err


# ─── ccl.py keepgoing ────────────────────────────────────────────────────────

def test_split_args_passes_everything_after_dashes_to_claude():
    assert ccl.split_keepgoing_args(["--auto"]) == (["--auto"], [])
    assert ccl.split_keepgoing_args(["--auto", "--", "-p", "hi --auto"]) == (["--auto"], ["-p", "hi --auto"])
    assert ccl.split_keepgoing_args([]) == ([], [])


def test_keepgoing_env_file_loads_and_real_env_wins(tmp_path):
    f = tmp_path / "keepgoing.env"
    f.write_text("# c\nOPENROUTER_API_KEY=from-file\nexport KEEPGOING_AFTER_LOCAL='echo hi'\nX=1\n")
    env = {"X": "from-env"}
    ccl.load_keepgoing_env(str(f), env)
    assert env == {"OPENROUTER_API_KEY": "from-file", "KEEPGOING_AFTER_LOCAL": "echo hi", "X": "from-env"}
    ccl.load_keepgoing_env(str(tmp_path / "missing"), env)  # no file is fine


@pytest.fixture
def run_kg(tmp_path, monkeypatch):
    """Call ccl.cmd_keepgoing with the picker, claude and the servers stubbed."""
    calls = {"claude": [], "started": 0, "terminated": 0}
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))  # never ~/.claude.json
    monkeypatch.setattr(ccl, "KEEPGOING_ENV", str(tmp_path / "none.env"))
    monkeypatch.setattr(ccl, "load_keepgoing_env", lambda *a, **k: None)
    monkeypatch.setattr(ccl, "resolve_claude_bin", lambda: "/bin/claude")

    def fake_call(cmd, env=None, shell=False):
        calls["claude"].append((cmd, env))
        return 7
    monkeypatch.setattr(ccl.subprocess, "call", fake_call)

    class P:
        def terminate(self):
            calls["terminated"] += 1

        def wait(self, timeout=None):
            return 0

    def fake_start(cfg, log_path=None):
        calls["started"] += 1
        calls["log"] = log_path
        return P(), P()
    monkeypatch.setattr(ccl, "start_servers", fake_start)
    monkeypatch.setattr(ccl, "load_config", lambda: {})
    monkeypatch.setattr(ccl, "proxy_is_up", lambda: False)

    def go(pick, argv=("--auto",)):
        monkeypatch.setattr(ccl, "pick_for_keepgoing", lambda own: pick)
        return ccl.cmd_keepgoing(list(argv))
    go.calls = calls
    return go


def test_cmd_keepgoing_local_starts_llama_and_resumes(run_kg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-not-leak")
    rc = run_kg(("abc", "local", "Qwen 3.5 4B [Q4_K_M]", False), ["--auto", "--", "-p", "hi"])
    assert rc == 7
    cmd, env = run_kg.calls["claude"][0]
    # the header names the real local model, not a Claude model
    assert cmd == ["/bin/claude", "--model", "local:Qwen-3.5-4B-Q4_K_M", "--resume", "abc", "-p", "hi"]
    assert env["ANTHROPIC_MODEL"] == "local:Qwen-3.5-4B-Q4_K_M"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "local:Qwen-3.5-4B-Q4_K_M"
    assert env["ANTHROPIC_BASE_URL"] == ccl.PROXY_URL and env["ANTHROPIC_API_KEY"] == "sk-local"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert run_kg.calls["started"] == 1 and run_kg.calls["terminated"] == 2
    # server output goes to a log, not into Claude Code's screen
    assert run_kg.calls["log"] == os.path.join(ccl.home(), "keepgoing-server.log")


def test_wait_health_gives_up_when_the_server_exits():
    class Dead:
        def poll(self):
            return 127
    t = time.time()
    assert ccl._wait_health(f"http://127.0.0.1:{free_port()}/health", 600, Dead()) is False
    assert time.time() - t < 5


def test_cmd_keepgoing_local_already_running_starts_nothing(run_kg):
    run_kg(("abc", "local", "m", True))
    assert run_kg.calls["started"] == 0 and run_kg.calls["terminated"] == 0
    assert run_kg.calls["claude"][0][0][-2:] == ["--resume", "abc"]


def test_cmd_keepgoing_local_reuses_a_proxy_that_is_already_up(run_kg, monkeypatch):
    monkeypatch.setattr(ccl, "proxy_is_up", lambda: True)
    run_kg(("abc", "local", "m", False))
    assert run_kg.calls["started"] == 0


def test_cmd_keepgoing_cloud_talks_to_openrouter(run_kg, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    run_kg(("abc", "cloud", "big:free", False))
    cmd, env = run_kg.calls["claude"][0]
    assert cmd == ["/bin/claude", "--model", "big:free", "--resume", "abc"]
    assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "or-key" and "ANTHROPIC_API_KEY" not in env
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "big:free"
    assert run_kg.calls["started"] == 0


def test_cmd_keepgoing_hooks_run_around_a_local_session(run_kg, monkeypatch):
    monkeypatch.setenv("KEEPGOING_BEFORE_LOCAL", "before")
    monkeypatch.setenv("KEEPGOING_AFTER_LOCAL", "after")
    run_kg(("abc", "local", "m", False))
    shells = [c[0] for c in run_kg.calls["claude"] if isinstance(c[0], str)]
    assert shells == ["before", "after"]


def test_cmd_keepgoing_cancel_and_no_claude(run_kg, monkeypatch):
    assert run_kg(None) == 0
    monkeypatch.setattr(ccl, "resolve_claude_bin", lambda: None)
    assert run_kg(("abc", "local", "m", True)) == 1
    assert run_kg.calls["claude"] == []


def test_pick_for_keepgoing_reads_the_tab_line(monkeypatch):
    kg = sys.modules["keepgoing"]
    monkeypatch.setattr(kg, "main", lambda argv: print("sid\tlocal\tQwen 3.5 4B [Q4_K_M]\t1"))
    assert ccl.pick_for_keepgoing(["--auto"]) == ("sid", "local", "Qwen 3.5 4B [Q4_K_M]", True)

    def cancel(argv):
        raise SystemExit(1)
    monkeypatch.setattr(kg, "main", cancel)
    assert ccl.pick_for_keepgoing([]) is None


# ─── end to end: ccl.py keepgoing in a subprocess ────────────────────────────

FAKE_CLAUDE = """\
import json, os, sys
with open(os.environ["FAKE_CLAUDE_LOG"], "w") as f:
    json.dump({"argv": sys.argv[1:], "base": os.environ.get("ANTHROPIC_BASE_URL"),
               "key": os.environ.get("ANTHROPIC_API_KEY"),
               "model": os.environ.get("ANTHROPIC_MODEL")}, f)
"""


def test_ccl_keepgoing_end_to_end_against_a_running_proxy(tmp_path):
    port = free_port()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "model": "Qwen 3.5 4B [Q4_K_M]"}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    work = tmp_path / "my project"
    work.mkdir()
    cfgdir = tmp_path / "claude-config"
    write_session(cfgdir / "projects", "e2e-session", str(work), 3)
    write_config(tmp_path / "h")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "fake_claude.py").write_text(FAKE_CLAUDE)
    if os.name == "nt":
        (stubs / "claude.cmd").write_text(f'@"{sys.executable}" "{stubs / "fake_claude.py"}" %*\r\n')
    else:
        (stubs / "claude").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stubs / "fake_claude.py"}" "$@"\n')
        (stubs / "claude").chmod(0o755)
    log = tmp_path / "claude.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("MLX_PORT", "OPENROUTER_API_KEY", "ANTHROPIC_BASE_URL")}
    env.update({"PATH": str(stubs) + os.pathsep + env.get("PATH", ""), "CCL_PORT": str(port),
                "CLAUDE_CONFIG_DIR": str(cfgdir), "CCL_HOME": str(tmp_path / "h"),
                "HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "FAKE_CLAUDE_LOG": str(log)})
    try:
        r = subprocess.run([sys.executable, "-I", str(ROOT / "portable" / "ccl.py"), "keepgoing",
                            "--auto", "--", "-p", "what was the word?"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=60)
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.returncode == 0, r.stderr + r.stdout
    got = json.loads(log.read_text())
    assert got["argv"] == ["--model", "local:Qwen-3.5-4B-Q4_K_M", "--resume", "e2e-session",
                           "-p", "what was the word?"]
    assert got["base"] == f"http://localhost:{port}" and got["key"] == "sk-local"
    assert got["model"] == "local:Qwen-3.5-4B-Q4_K_M"
    assert "already running" in r.stderr
    # the placeholder key was pre-approved, so the first run doesn't stop on the API key question
    cfg = json.loads((cfgdir / ".claude.json").read_text())
    assert cfg["customApiKeyResponses"]["approved"] == ["sk-local"]


# ─── shell glue ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not BASH or os.name == "nt", reason="needs bash")
def test_bin_keepgoing_hands_off_to_ccl_on_linux(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "uname").write_text("#!/bin/sh\necho Linux\n")
    (stubs / "python3").write_text('#!/bin/sh\necho "$@" > "$KG_LOG"\n')
    for f in ("uname", "python3"):
        (stubs / f).chmod(0o755)
    log = tmp_path / "args"
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "KG_LOG": str(log)}
    r = subprocess.run([BASH, str(ROOT / "bin" / "keepgoing"), "--auto", "--", "-p", "x"],
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    args = log.read_text().split()
    assert args[0] == "-I" and args[1].endswith("portable/ccl.py")
    assert os.path.realpath(args[1]) == str((ROOT / "portable" / "ccl.py").resolve())
    assert args[2:] == ["keepgoing", "--auto", "--", "-p", "x"]


@pytest.mark.skipif(not BASH or os.name == "nt", reason="needs bash")
def test_keepgoing_shell_works_in_bash():
    r = subprocess.run([BASH, "-c", f'source "{ROOT}/scripts/keepgoing-shell.sh"; '
                        'echo "$_CCL_KEEPGOING"; keep nope'],
                       capture_output=True, text=True, timeout=30)
    lines = r.stdout.splitlines()
    assert os.path.realpath(lines[0]) == str((ROOT / "bin" / "keepgoing").resolve())
    assert lines[1] == "did you mean: keep going"


def test_setup_linux_installs_keepgoing():
    s = (ROOT / "portable" / "setup-linux.sh").read_text()
    assert 'ln -sf "$REPO_DIR/bin/keepgoing" "$HOME/.local/bin/keepgoing"' in s
    assert "scripts/keepgoing-shell.sh" in s and ".bashrc" in s
    u = (ROOT / "uninstall.sh").read_text()
    assert '"$HOME/.bashrc"' in u


def test_windows_shims_call_ccl_keepgoing():
    for name in ("keepgoing.cmd", "keep.cmd"):
        raw = (ROOT / "bin" / name).read_bytes()
        assert b'python -I "%~dp0..\\portable\\ccl.py" keepgoing' in raw
    assert (ROOT / ".gitattributes").read_text().count("*.cmd text eol=crlf") == 1
    ps = (ROOT / "install.ps1").read_text()
    assert "keepgoing-windows.ps1" in ps
    assert "keepgoing-windows.ps1" in (ROOT / "uninstall.ps1").read_text()


# ─── PowerShell: profile hook install/uninstall and the keep function ────────

def ps(script, env=None, timeout=60):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          env=e, capture_output=True, text=True, timeout=timeout)


@pytest.mark.skipif(not POWERSHELL, reason="PowerShell not installed")
def test_powershell_profile_hook_installs_once_and_uninstalls(tmp_path):
    p1 = tmp_path / "a" / "profile.ps1"
    p2 = tmp_path / "b" / "profile.ps1"
    p2.parent.mkdir()
    p2.write_text("Write-Host mine\n")
    helper = ROOT / "scripts" / "keepgoing-windows.ps1"
    env = {"CCL_PROFILES": f"{p1};{p2}", "CCL_NO_PATH": "1"}
    for _ in range(2):
        r = ps(f"& '{helper}' -RepoDir '{ROOT}'", env)
        assert r.returncode == 0, r.stderr
    for p in (p1, p2):
        assert p.read_text().count("keepgoing-profile.ps1") == 1
    assert "Write-Host mine" in p2.read_text()
    r = ps(f"& '{helper}' -Uninstall -RepoDir '{ROOT}'", env)
    assert r.returncode == 0, r.stderr
    assert "keepgoing-profile" not in p1.read_text()
    assert "keepgoing-profile" not in p2.read_text() and "Write-Host mine" in p2.read_text()


@pytest.mark.skipif(not POWERSHELL, reason="PowerShell not installed")
def test_powershell_keep_function(tmp_path):
    hook = ROOT / "scripts" / "keepgoing-profile.ps1"
    r = ps(f". '{hook}'; keep nope; (Get-Command keepgoing).CommandType; $global:CclKeepGoingPy")
    out = r.stdout.splitlines()
    assert out[0] == "did you mean: keep going"
    assert out[1] == "Function"
    assert os.path.normcase(os.path.realpath(out[2])) == os.path.normcase(str((ROOT / "portable" / "ccl.py").resolve()))
