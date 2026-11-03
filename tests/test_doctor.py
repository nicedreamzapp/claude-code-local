"""scripts/doctor.sh: report redaction, add-on detection, and the --fix paths.

Everything runs in a throwaway HOME with test ports and stub launchctl/systemctl,
so nothing here touches real services or real files.
"""
import os
import shutil
import socket
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT

DOCTOR = ROOT / "scripts" / "doctor.sh"
BASH = shutil.which("bash")
SYS_PATH = "/usr/bin:/bin"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    e = {
        "HOME": str(home),
        "PATH": f"{stubs}:{SYS_PATH}",
        "MLX_PORT": str(free_port()),
        "BROKER_PORT": str(free_port()),
        "DOCK_PORT": str(free_port()),
        "FIAOS_PORT": str(free_port()),
        "CCL_HOME": str(tmp_path / "ccl"),
        "CCL_DOCTOR_USER": "zedthree",
        "CCL_DOCTOR_HOST": "zed-box.local",
        "TERM": "dumb",
    }
    e["_stubs"] = str(stubs)
    e["_home"] = str(home)
    return e


def run_fn(env, snippet):
    """Source doctor.sh and run a snippet of bash against its functions."""
    e = {k: v for k, v in env.items() if not k.startswith("_")}
    r = subprocess.run([BASH, "-c", f'source "{DOCTOR}"; {snippet}'], env=e,
                       capture_output=True, text=True, timeout=60)
    return r


def run_doctor(env, *args, **extra):
    e = {k: v for k, v in env.items() if not k.startswith("_")}
    e.update(extra)
    return subprocess.run([BASH, str(DOCTOR), *args], env=e, capture_output=True,
                          text=True, timeout=120)


# ── redaction ─────────────────────────────────────────────────

LEAKY = textwrap.dedent("""\
    config at {home}/.config/thing and /Users/someoneelse/Desktop/x
    also /home/otherguy/.cache and C:\\Users\\Winuser\\AppData
    user zedthree on host zed-box.local (zed-box)
    server 192.168.1.44 and 10.0.0.7 plus fe80::1c2d:3e4f:5a6b:7c8d
    mail me at someone@example.com
    mac aa:bb:cc:dd:ee:ff
    token ghp_abcdefghijklmnop1234567890 and sk-ant-api03-abcdefghijkl and hf_abcdefghijklmnopqrst
    Authorization: Bearer eyJhbGciOi.payload.sig
    ANTHROPIC_API_KEY=sk-live-whatever MY_SECRET_THING=hunter2 PATH=/usr/bin:/bin
    password: hunter2
    deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
""")

MUST_GONE = ["zedthree", "zed-box", "someoneelse", "otherguy", "Winuser", "192.168.1.44",
             "10.0.0.7", "fe80", "someone@example.com", "aa:bb:cc:dd:ee:ff", "ghp_abc",
             "sk-ant", "hf_abc", "eyJhbGci", "sk-live", "hunter2", "/usr/bin:/bin",
             "deadbeefdeadbeef"]


def test_redact_removes_everything_personal(env):
    text = LEAKY.format(home=env["_home"])
    r = run_fn(env, "redact")  # no input yet: sanity that the function exists
    assert r.returncode == 0
    e = {k: v for k, v in env.items() if not k.startswith("_")}
    r = subprocess.run([BASH, "-c", f'source "{DOCTOR}"; redact'], env=e, input=text,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert env["_home"] not in out
    for secret in MUST_GONE:
        assert secret not in out, f"{secret!r} leaked:\n{out}"
    assert "~/.config/thing" in out


def test_redact_keeps_useful_facts(env):
    e = {k: v for k, v in env.items() if not k.startswith("_")}
    text = "llama.cpp b11669 on port 4000, Apple M4 Pro, 64 GB, claude 2.1.306\n~/.local/bin/narrate-claude missing\n"
    r = subprocess.run([BASH, "-c", f'source "{DOCTOR}"; redact'], env=e, input=text,
                       capture_output=True, text=True, timeout=30)
    assert r.stdout == text


def test_report_has_no_private_values(env):
    # A home path and user name that look like real ones, and a failing core.
    r = run_doctor(env, "--report")
    out = r.stdout
    assert out.startswith("claude-code-local doctor report")
    for needle in (env["_home"], "zedthree", "zed-box"):
        assert needle not in out
    for field in ("OS:", "Chip/GPU:", "RAM:", "Versions:", "Summary:", "Failed:", "Not installed:"):
        assert field in out
    # --report prints the block only, not the full walk-through
    assert "Doctor" not in out and "╔" not in out


def test_report_redacts_a_leaky_failure_message(env):
    # A failing check whose message contains the user's folder name must come out as ~.
    cfg = os.path.join(env["CCL_HOME"])
    os.makedirs(cfg)
    open(os.path.join(cfg, "config.json"), "w").write(
        '{"model_gguf": "%s/zedthree-models/m.gguf", "mmproj": "", "server_bin": "", "llama_tag": "b1"}'
        % env["_home"])
    r = run_doctor(env, "--report", CCL_DOCTOR_OS="Linux")
    assert "zedthree" not in r.stdout
    assert env["_home"] not in r.stdout


# ── add-on detection ──────────────────────────────────────────

def installed(env, name):
    return run_fn(env, f"addon_installed {name}").returncode == 0


def test_nothing_installed_means_not_installed_not_errors(env):
    r = run_doctor(env, CCL_DOCTOR_OS="Linux")
    for a in ("voice", "browser", "moonstone", "remote", "failover"):
        assert f"{a}: not installed" in r.stdout
    assert "phone: macOS only" in r.stdout
    assert "narrate-claude" not in r.stdout     # no voice errors for a missing add-on
    r = run_doctor(env, "--report", CCL_DOCTOR_OS="Linux")
    assert "voice" in r.stdout.split("Not installed:")[1]


@pytest.mark.parametrize("name,rel", [
    ("voice", "NarrateClaude"),
    ("browser", ".browser-agent"),
    ("browser", ".browser-broker/.git"),   # the older folder name, found the way addons.sh finds it
    ("phone", "claude-screen-to-phone"),
    ("remote", "FiaOS"),
    ("moonstone", "moonstone"),
    ("failover", ".local/share/claude-failover"),
])
def test_detects_clone_folder(env, name, rel):
    assert not installed(env, name)
    os.makedirs(os.path.join(env["_home"], rel))
    assert installed(env, name)


def test_detects_by_command_on_path(env):
    for cmd, name in (("narrate-claude", "voice"), ("agent-llm", "failover")):
        assert not installed(env, name)
        p = os.path.join(env["_stubs"], cmd)
        open(p, "w").write("#!/bin/sh\n")
        os.chmod(p, 0o755)
        assert installed(env, name)


def test_env_var_folders_are_honoured(env, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    e = dict(env, NARRATE_DIR=str(other))
    assert installed(e, "voice")


def test_running_service_counts_as_installed(env):
    srv = subprocess.Popen([sys.executable, "-m", "http.server", env["DOCK_PORT"], "--bind", "127.0.0.1"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=env["_home"])
    try:
        import time
        time.sleep(1.5)
        r = run_doctor(env, CCL_DOCTOR_OS="Linux")
        assert "Moonstone answering on port" in r.stdout
    finally:
        srv.kill()


def test_installed_but_down_addon_fails_with_hint(env):
    os.makedirs(os.path.join(env["_home"], "FiaOS"))
    r = run_doctor(env, CCL_DOCTOR_OS="Linux")
    assert "FiaOS not answering on port" in r.stdout
    assert r.returncode == 1


def test_voice_checks_name_each_missing_piece(env):
    os.makedirs(os.path.join(env["_home"], "NarrateClaude"))
    r = run_doctor(env, CCL_DOCTOR_OS="Linux")
    assert "narrate-claude" in r.stdout
    assert "no whisper model" in r.stdout
    assert "Piper" in r.stdout


def test_voice_passes_when_complete(env):
    data = os.path.join(env["_home"], "vdata")
    os.makedirs(os.path.join(data, "models"))
    for n in ("ggml-small.en.bin", "voice.onnx"):
        open(os.path.join(data, "models", n), "w").write("x")
    for cmd in ("narrate-claude", "speak"):
        p = os.path.join(env["_stubs"], cmd)
        open(p, "w").write("#!/bin/sh\n")
        os.chmod(p, 0o755)
    r = run_doctor(env, CCL_DOCTOR_OS="Linux", NARRATE_DATA=data)
    assert "narrate-claude on PATH" in r.stdout
    assert "whisper model present" in r.stdout
    assert "Piper voice and speak command present" in r.stdout


# ── --fix, only ever in the throwaway HOME ───────────────────

def test_fix_recreates_missing_symlink_and_default_run_does_not(env):
    nd = os.path.join(env["_home"], "NarrateClaude")
    os.makedirs(nd)
    open(os.path.join(nd, "narrative-claude.sh"), "w").write("#!/bin/sh\n")
    link = os.path.join(env["_home"], ".local", "bin", "narrate-claude")

    run_doctor(env, CCL_DOCTOR_OS="Linux")
    assert not os.path.lexists(link)                       # read-only by default

    r = run_doctor(env, "--fix", CCL_DOCTOR_OS="Linux")
    assert os.path.islink(link)
    assert os.readlink(link) == os.path.join(nd, "narrative-claude.sh")
    assert "recreated the symlink" in r.stdout


def test_fix_never_replaces_a_real_file(env):
    nd = os.path.join(env["_home"], "NarrateClaude")
    os.makedirs(nd)
    open(os.path.join(nd, "narrative-claude.sh"), "w").write("#!/bin/sh\n")
    keep = os.path.join(env["_home"], ".local", "bin", "narrate-claude")
    os.makedirs(os.path.dirname(keep))
    # a real file that is not executable is not "on PATH", but it exists, so link_check passes
    open(keep, "w").write("mine")
    run_doctor(env, "--fix", CCL_DOCTOR_OS="Linux")
    assert open(keep).read() == "mine" and not os.path.islink(keep)


STUB_SERVER = """\
#!/bin/sh
# stub for launchctl / systemctl: log the call, then start a tiny server on $STUB_PORT
echo "$@" >> "$STUB_LOG"
case "$*" in
  *list-unit-files*) echo "moonstone.service enabled"; exit 0 ;;
esac
nohup python3 -m http.server "$STUB_PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
echo $! >> "$STUB_PIDS"
exit 0
"""


@pytest.fixture
def stub_service(env, tmp_path):
    log, pids = tmp_path / "calls.log", tmp_path / "pids"
    env.update(STUB_PORT=env["DOCK_PORT"], STUB_LOG=str(log), STUB_PIDS=str(pids))
    yield log
    if pids.exists():
        for pid in pids.read_text().split():
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass


@pytest.mark.parametrize("which", ["mac", "linux"])
def test_fix_restarts_a_down_service(env, stub_service, which):
    os.makedirs(os.path.join(env["_home"], "moonstone"))
    stub = os.path.join(env["_stubs"], "launchctl" if which == "mac" else "systemctl")
    open(stub, "w").write(STUB_SERVER)
    os.chmod(stub, 0o755)
    if which == "mac":
        la = os.path.join(env["_home"], "Library", "LaunchAgents")
        os.makedirs(la)
        open(os.path.join(la, "com.test.moonstone.plist"), "w").write("<plist/>")
        extra = dict(CCL_DOCTOR_OS="Darwin", CCL_DOCTOR_LAUNCHCTL=stub)
    else:
        extra = dict(CCL_DOCTOR_OS="Linux")

    r = run_doctor(env, **extra)
    assert "Moonstone not answering" in r.stdout
    assert not stub_service.exists() or "restart" not in stub_service.read_text().replace("list-unit-files", "") \
        and "kickstart" not in stub_service.read_text()   # no restart without --fix
    assert "--fix" in r.stdout

    r = run_doctor(env, "--fix", **extra)
    assert "Moonstone was down; restarted your own service" in r.stdout, r.stdout
    calls = stub_service.read_text()
    assert ("kickstart -k" in calls and "com.test.moonstone" in calls) if which == "mac" \
        else "restart moonstone.service" in calls


def test_fix_without_a_service_says_so(env):
    os.makedirs(os.path.join(env["_home"], "moonstone"))
    r = run_doctor(env, "--fix", CCL_DOCTOR_OS="Linux")
    assert "no service of yours to restart" in r.stdout


# ── script hygiene ───────────────────────────────────────────

def test_doctor_scripts_parse():
    assert subprocess.run([BASH, "-n", str(DOCTOR)]).returncode == 0
    assert (ROOT / "scripts" / "doctor.ps1").exists()


def test_unknown_flag_is_rejected(env):
    r = run_doctor(env, "--nope")
    assert r.returncode == 2


@pytest.mark.skipif(not shutil.which("pwsh"), reason="PowerShell not installed")
def test_powershell_doctor_report_runs_and_redacts(env, tmp_path):
    home = tmp_path / "pshome"
    home.mkdir()
    e = {"HOME": str(home), "PATH": os.environ["PATH"], "USERPROFILE": str(home),
         "MLX_PORT": env["MLX_PORT"], "BROKER_PORT": env["BROKER_PORT"],
         "DOCK_PORT": env["DOCK_PORT"], "FIAOS_PORT": env["FIAOS_PORT"],
         "CCL_HOME": str(tmp_path / "ccl")}
    r = subprocess.run(["pwsh", "-NoProfile", "-File", str(ROOT / "scripts" / "doctor.ps1"), "-Report"],
                       env=e, capture_output=True, text=True, timeout=120)
    assert r.stdout.startswith("claude-code-local doctor report"), r.stdout + r.stderr
    assert str(home) not in r.stdout
    assert "Not installed:" in r.stdout


# ── keep going is checked on every OS ─────────────────────────

@pytest.mark.parametrize("os_name", ["Darwin", "Linux"])
def test_keepgoing_missing_is_a_warning_not_a_failure(env, os_name):
    r = run_fn(env, f'OS={os_name}; MODE_REPORT=1; check_keepgoing; printf "%s" "$RESULTS"')
    assert "keepgoing-link" in r.stdout and "keep going not installed" in r.stdout
    assert "keepgoing-shell" in r.stdout
    assert "fail|keepgoing" not in r.stdout


@pytest.mark.parametrize("os_name, rc", [("Darwin", ".zshrc"), ("Linux", ".bashrc")])
def test_keepgoing_installed_passes(env, os_name, rc):
    home = env["_home"]
    os.makedirs(os.path.join(home, ".local", "bin"))
    os.symlink(str(ROOT / "bin" / "keepgoing"), os.path.join(home, ".local", "bin", "keepgoing"))
    with open(os.path.join(home, rc), "w") as f:
        f.write(f'source "{ROOT}/scripts/keepgoing-shell.sh"\n')
    r = run_fn(env, f'OS={os_name}; MODE_REPORT=1; check_keepgoing; printf "%s" "$RESULTS"')
    assert "pass|keepgoing-link|" in r.stdout
    assert "pass|keepgoing-shell|keep going shell line in ~/" + rc in r.stdout
    assert "warn|" not in r.stdout and "fail|" not in r.stdout


def test_keepgoing_fix_recreates_the_command(env):
    link = os.path.join(env["_home"], ".local", "bin", "keepgoing")
    r = run_doctor(env, "--fix", CCL_DOCTOR_OS="Linux")
    assert os.path.islink(link)
    assert os.path.realpath(link) == str((ROOT / "bin" / "keepgoing").resolve())
    assert "recreated the symlink" in r.stdout
