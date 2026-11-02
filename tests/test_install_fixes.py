"""Fixes from the clean-install runs (Mac, Ubuntu 24.04, Windows):

- Linux: llama-server needs libgomp.so.1, which stock Ubuntu 24.04 lacks. setup-linux.sh installs
  libgomp1, and the doctor fails an install whose llama-server is missing a library.
- Small Macs: Claude Code ignores CLAUDE_CODE_AUTO_COMPACT_WINDOW under 100000, so mac-tier.sh
  doesn't set it for the 16K tiers and the server enforces the window (prompt + answer).
- Launchers: the health-wait message says the real wait, and the server log is per user.
- setup.sh installs Claude Code with the same official installer as Linux.
- The doctor and the installers call /usr/sbin/sysctl by its full path.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT, server

BASH = shutil.which("bash")
DEPS = ROOT / "portable" / "linux-deps.sh"
DOCTOR = ROOT / "scripts" / "doctor.sh"
COMMON = ROOT / "launchers" / "lib" / "claude-local-common.sh"
CLAUDE_INSTALL = ROOT / "scripts" / "install-claude-code.sh"
needs_bash = pytest.mark.skipif(not BASH or os.name == "nt", reason="needs bash")
TOOLS = ("grep", "sed", "awk", "sort", "tr", "cat", "mkdir", "chmod", "printf", "env", "dirname")


def stub_dir(tmp_path, stubs, tools=TOOLS):
    """A PATH with only these stub scripts plus the basic tools the shell code needs."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for t in tools:
        real = shutil.which(t)
        if real:
            (d / t).symlink_to(real)
    for name, body in stubs.items():
        (d / name).write_text("#!" + BASH + "\n" + body + "\n")
        (d / name).chmod(0o755)
    return d


def run(snippet, path, **env):
    e = {"PATH": str(path), "HOME": env.pop("HOME", "/nonexistent"), **env}
    return subprocess.run([BASH, "-c", snippet], env=e, capture_output=True, text=True, timeout=60)


# ─── Linux: libgomp ──────────────────────────────────────────────────────────

APT_LOG = 'echo "$*" >> "$LOG"'


@needs_bash
def test_libgomp_installed_with_sudo_when_missing(tmp_path):
    log = tmp_path / "log"
    d = stub_dir(tmp_path, {"ldconfig": "exit 0", "id": "echo 1000",
                            "apt-get": 'echo "apt-get $*" >> "$LOG"',
                            "sudo": 'echo "sudo" >> "$LOG"; "$@"'})
    r = run(f'source "{DEPS}"; ccl_ensure_libgomp', d, LOG=str(log))
    assert r.returncode == 0, r.stdout + r.stderr
    assert log.read_text().splitlines() == ["sudo", "apt-get install -y libgomp1"]
    assert "libgomp.so.1" in r.stdout


@needs_bash
def test_libgomp_as_root_needs_no_sudo(tmp_path):
    log = tmp_path / "log"
    d = stub_dir(tmp_path, {"ldconfig": "exit 0", "id": "echo 0",
                            "apt-get": 'echo "apt-get $*" >> "$LOG"'})
    r = run(f'source "{DEPS}"; ccl_ensure_libgomp', d, LOG=str(log))
    assert r.returncode == 0 and log.read_text().splitlines() == ["apt-get install -y libgomp1"]


@needs_bash
def test_libgomp_present_installs_nothing(tmp_path):
    log = tmp_path / "log"
    d = stub_dir(tmp_path, {"ldconfig": 'echo "	libgomp.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libgomp.so.1"',
                            "id": "echo 1000", "apt-get": 'echo "$*" >> "$LOG"', "sudo": '"$@"'})
    r = run(f'source "{DEPS}"; ccl_ensure_libgomp', d, LOG=str(log))
    assert r.returncode == 0 and r.stdout == "" and not log.exists()


@needs_bash
def test_libgomp_without_apt_says_what_to_do(tmp_path):
    d = stub_dir(tmp_path, {"ldconfig": "exit 0", "id": "echo 1000", "sudo": '"$@"'})
    r = run(f'source "{DEPS}"; ccl_ensure_libgomp', d)
    assert r.returncode == 1
    assert "apt-get isn't available" in r.stdout and "dnf install libgomp" in r.stdout


@needs_bash
def test_libgomp_not_root_and_no_sudo(tmp_path):
    d = stub_dir(tmp_path, {"ldconfig": "exit 0", "id": "echo 1000", "apt-get": "exit 0"})
    r = run(f'source "{DEPS}"; ccl_ensure_libgomp', d)
    assert r.returncode == 1 and "apt-get install -y libgomp1" in r.stdout


@needs_bash
def test_missing_libs_reads_ldd(tmp_path):
    d = stub_dir(tmp_path, {"ldd": 'printf "\\tlibllama.so => /x/libllama.so (0x1)\\n'
                                   '\\tlibgomp.so.1 => not found\\n\\tlibfoo.so.2 => not found\\n"'})
    r = run(f'source "{DEPS}"; ccl_missing_libs /x/llama-server', d)
    assert r.stdout.strip() == "libfoo.so.2 libgomp.so.1"


def test_setup_linux_installs_libgomp_before_llama():
    s = (ROOT / "portable" / "setup-linux.sh").read_text()
    assert s.index("ccl_ensure_libgomp") < s.index('ccl.py" install')
    assert "ccl_missing_libs" in s


def _doctor_linux(tmp_path, ldd_out):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "ldd").write_text(f"#!/bin/sh\nprintf '{ldd_out}'\n")
    (stubs / "ldd").chmod(0o755)
    h = tmp_path / "ccl"
    h.mkdir()
    for f in ("m.gguf", "mm.gguf"):
        (h / f).write_text("x")
    llama = h / "llama-server"
    llama.write_text("#!/bin/sh\n")
    llama.chmod(0o755)
    (h / "config.json").write_text(json.dumps({
        "model_gguf": str(h / "m.gguf"), "mmproj": str(h / "mm.gguf"), "server_bin": str(llama),
        "llama_tag": "b1", "model_name": "Qwen 3.5 4B", "quant": "Q4_K_M", "accel": "cpu"}))
    env = {"HOME": str(tmp_path), "PATH": f"{stubs}:/usr/bin:/bin", "CCL_HOME": str(h), "TERM": "dumb"}
    return subprocess.run([BASH, "-c", f'source "{DOCTOR}"; OS=Linux; check_models_linux; '
                                       'printf "%s" "$RESULTS"'],
                          env=env, capture_output=True, text=True, timeout=60).stdout


@needs_bash
def test_doctor_fails_a_llama_server_missing_a_library(tmp_path):
    out = _doctor_linux(tmp_path, "\\tlibgomp.so.1 => not found\\n\\tlibc.so.6 => /lib/libc.so.6\\n")
    assert "fail|llama|llama-server can't start, missing libraries: libgomp.so.1" in out
    assert "apt-get install libgomp1" in out
    assert "pass|llama" not in out


@needs_bash
def test_doctor_passes_a_llama_server_with_every_library(tmp_path):
    out = _doctor_linux(tmp_path, "\\tlibc.so.6 => /lib/libc.so.6\\n")
    assert "pass|llama|llama-server present" in out and "fail|llama" not in out


# ─── sysctl by full path ─────────────────────────────────────────────────────

@needs_bash
@pytest.mark.skipif(sys.platform != "darwin", reason="hw.memsize is a macOS sysctl")
def test_doctor_reads_memory_without_usr_sbin_on_path():
    r = subprocess.run([BASH, "-c", f'source "{DOCTOR}"; OS=Darwin; total_ram_gb'],
                       env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TERM": "dumb"},
                       capture_output=True, text=True, timeout=60)
    assert int(r.stdout.strip()) >= 8


def test_no_bare_sysctl_in_mac_scripts():
    for name in ("install.sh", "setup.sh", "scripts/mac-tier.sh"):
        for line in (ROOT / name).read_text().splitlines():
            code = line.split("#", 1)[0]
            if "sysctl -n" in code:
                assert "/usr/sbin/sysctl -n" in code, f"{name}: {line.strip()}"
    assert '"$SYSCTL" -n hw.memsize' in DOCTOR.read_text()


# ─── small Macs: the server enforces the window ──────────────────────────────

def tier(**env):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("CCL_", "MLX_", "CLAUDE_CODE_"))}
    full.update(env)
    out = subprocess.run([BASH, str(ROOT / "scripts" / "mac-tier.sh"), "--env"], env=full,
                         capture_output=True, text=True, check=True).stdout
    return dict(line.removeprefix("export ").split("=", 1) for line in out.splitlines())


@needs_bash
def test_tier_never_sets_a_compact_window_under_100k():
    for gb in ("8", "16"):
        v = tier(CCL_MEM_GB=gb)
        assert v["MLX_MAX_CONTEXT"] == "16384"
        assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in v


@needs_bash
def test_tier_keeps_your_own_compact_window():
    assert tier(CCL_MEM_GB="16", CLAUDE_CODE_AUTO_COMPACT_WINDOW="200000")[
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"


def test_fit_max_tokens():
    assert server.fit_max_tokens(1000, 32000, limit=0) == 32000          # no ceiling
    assert server.fit_max_tokens(10000, 8192, limit=16384) == 6384      # trimmed to the room left
    assert server.fit_max_tokens(1000, 500, limit=16384) == 500         # fits as asked
    assert server.fit_max_tokens(16000, 300, limit=16384) == 300        # small ask, enough room
    with pytest.raises(server.PromptTooLong, match="prompt is too long"):
        server.fit_max_tokens(15800, 8192, limit=16384)                  # under 1024 left: compact
    with pytest.raises(server.PromptTooLong):
        server.fit_max_tokens(16385, 10, limit=16384)


def test_generation_is_capped_to_the_window(fake_model, monkeypatch):
    seen = []
    real = fake_model.stream_generate

    def spy(model, tokenizer, prompt, max_tokens, **kw):
        seen.append((len(prompt), max_tokens))
        return real(model, tokenizer, prompt, max_tokens, **kw)
    monkeypatch.setattr(server, "stream_generate", spy)
    fake_model.outputs = ["ok"]
    body = {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 32000}
    prompt_len = len(server.tokenize_messages(server.convert_messages(body, vision=False, tool_ids=True)))
    monkeypatch.setattr(server, "MAX_CONTEXT", prompt_len + 2000)
    server.generate_response(body)
    assert seen and seen[0][1] == 2000


# ─── launchers: honest wait, per-user log ────────────────────────────────────

@needs_bash
def test_health_wait_message_matches_the_real_wait():
    s = COMMON.read_text()
    assert "120s" not in s and "/tmp/mlx-server.log" not in s.replace("a shared /tmp/mlx-server.log", "")
    r = subprocess.run([BASH, "-c", f'source "{COMMON}"; echo "$((MLX_HEALTH_ATTEMPTS * 2)) $MLX_LOG"'],
                       capture_output=True, text=True, timeout=30)
    secs, log = r.stdout.split()
    user = subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()
    assert secs == "360" and log == f"/tmp/mlx-server-{user}.log"
    assert 'within $((MLX_HEALTH_ATTEMPTS * 2))s' in s


def test_no_shared_server_log():
    for name in ("bin/keepgoing", "setup.sh", "launchers/lib/claude-local-common.sh"):
        code = "\n".join(l for l in (ROOT / name).read_text().splitlines() if not l.lstrip().startswith("#"))
        assert ">/tmp/mlx-server.log" not in code and "mlx-server-" in code, name


# ─── Claude Code: one official installer, Mac and Linux ──────────────────────

@needs_bash
def test_claude_found_is_used(tmp_path):
    d = stub_dir(tmp_path, {"claude": "exit 0"})
    r = run(f'source "{CLAUDE_INSTALL}"; ccl_ensure_claude', d, HOME=str(tmp_path))
    assert r.returncode == 0 and r.stdout.strip() == str(d / "claude")


@needs_bash
def test_claude_missing_is_installed_with_the_official_installer(tmp_path):
    log = tmp_path / "curl.log"
    home = tmp_path / "home"
    home.mkdir()
    installer = 'mkdir -p "$HOME/.local/bin"; printf "#!/bin/sh\\n" > "$HOME/.local/bin/claude"; chmod +x "$HOME/.local/bin/claude"'
    d = stub_dir(tmp_path, {"curl": f'echo "$*" >> "{log}"; echo \'{installer}\''},
                 tools=TOOLS + ("bash",))
    r = run(f'source "{CLAUDE_INSTALL}"; ccl_ensure_claude', d, HOME=str(home))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(home / ".local" / "bin" / "claude")
    assert log.read_text().strip() == "-fsSL https://claude.ai/install.sh"


@needs_bash
def test_claude_install_can_be_skipped(tmp_path):
    d = stub_dir(tmp_path, {})
    r = run(f'source "{CLAUDE_INSTALL}"; ccl_ensure_claude', d, HOME=str(tmp_path),
            CCL_NO_CLAUDE_INSTALL="1")
    assert r.returncode == 1 and "curl -fsSL https://claude.ai/install.sh | bash" in r.stderr


def test_mac_and_linux_setup_use_the_same_installer():
    for name in ("setup.sh", "portable/setup-linux.sh"):
        s = (ROOT / name).read_text()
        assert "install-claude-code.sh" in s and "ccl_ensure_claude" in s, name
        assert "npm install -g @anthropic-ai/claude-code" not in s, name
