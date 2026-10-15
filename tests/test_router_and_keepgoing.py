"""smart-router/router.py (request routing heuristics) and bin/keepgoing.py
(session discovery and model menu). Network and sysctl are mocked."""
import io
import json
import os
import sys
import time

import pytest

from conftest import ROOT, load_module

router = load_module("router", ROOT / "smart-router" / "router.py")
keepgoing = load_module("keepgoing", ROOT / "bin" / "keepgoing.py")


def user(content):
    return {"messages": [{"role": "user", "content": content}]}


# ─── router ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,backend", [
    ("/deep walk me through it", "deepseek"),
    ("/glm please", "glm"),
    ("/fast what's 2+2", "gemma"),
    ("/code fix it", "qwen"),
    ("hi", "gemma"),
    ("Thanks!", "gemma"),
    ("hi, can you refactor utils.py", "qwen"),
    ("think hard about this race condition", "glm"),
    ("refactor this async function", "qwen"),
    ("", "qwen"),
])
def test_route(text, backend):
    assert router.route(user(text))[0] == backend


def test_route_huge_context_goes_to_long_context_backend():
    backend, reason = router.route(user("x" * 500_000))
    assert backend == "deepseek"
    assert "125k" in reason


def test_route_reads_the_last_user_message_only():
    body = {"messages": [
        {"role": "user", "content": "/deep earlier"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]}
    assert router.route(body)[0] == "gemma"


def test_estimate_tokens_and_has_image():
    msgs = [{"role": "user", "content": "abcd" * 10},
            {"role": "user", "content": [{"type": "image", "source": {}}]}]
    assert router.estimate_tokens(msgs) == (40 + len(json.dumps(msgs[1]["content"][0]))) // 4
    assert router.has_image(msgs)
    assert not router.has_image(msgs[:1])
    assert router.last_user_text([{"role": "assistant", "content": "x"}]) == ""


# ─── keepgoing ───────────────────────────────────────────────────────────────

def write_session(projects, name, cwd, user_turns, mtime=None):
    d = projects / "proj"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.jsonl"
    lines = [json.dumps({"type": "system", "cwd": cwd, "pad": "x" * 200})]
    lines += [json.dumps({"type": "user", "message": "m"})] * user_turns
    lines += ["", "{not json"]
    path.write_text("\n".join(lines) + "\n")
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


def test_find_session_picks_newest_in_this_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    now = time.time()
    write_session(tmp_path, "old", "/work", 2, now - 100)
    write_session(tmp_path, "new", "/work", 3, now)
    write_session(tmp_path, "elsewhere", "/other", 9, now + 100)
    write_session(tmp_path, "empty", "/work", 0, now + 200)
    sid, turns, mtime = keepgoing.find_session("/work")
    assert (sid, turns) == ("new", 3)
    assert keepgoing.find_session("/nowhere") is None


def test_find_session_skips_tiny_files(tmp_path, monkeypatch):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "tiny.jsonl").write_text('{"cwd": "/work", "type": "user"}\n')
    assert keepgoing.find_session("/work") is None


@pytest.mark.parametrize("secs,expected", [(59, "0m ago"), (600, "10m ago"),
                                           (7200, "2h ago"), (3 * 86400, "3d ago")])
def test_ago(secs, expected):
    assert keepgoing.ago(secs) == expected


def test_free_cloud_models_filters_and_sorts(monkeypatch):
    data = {"data": [
        {"id": "a:free", "supported_parameters": ["tools"], "context_length": 131072},
        {"id": "b:free", "supported_parameters": ["tools"], "context_length": 1_000_000},
        {"id": "c:free", "supported_parameters": [], "context_length": 1_000_000},
        {"id": "d", "supported_parameters": ["tools"], "context_length": 1_000_000},
        {"id": "e:free", "supported_parameters": ["tools"], "context_length": 8192},
        {"id": "f:free", "supported_parameters": None, "context_length": None},
    ]}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def fake_urlopen(req, timeout):
        seen["auth"] = req.get_header("Authorization")
        return Resp(json.dumps(data).encode())

    monkeypatch.setattr(keepgoing.urllib.request, "urlopen", fake_urlopen)
    assert keepgoing.free_cloud_models("k") == ["b:free", "a:free"]
    assert seen["auth"] == "Bearer k"
    assert keepgoing.free_cloud_models("") == []


def test_free_cloud_models_offline(monkeypatch):
    def offline(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(keepgoing.urllib.request, "urlopen", offline)
    assert keepgoing.free_cloud_models("k") == []


def test_mac_ram_gb_parses_sysctl(monkeypatch):
    class R:
        stdout = f"{64 * (1 << 30)}\n"
    monkeypatch.setattr(keepgoing.subprocess, "run", lambda *a, **k: R())
    assert keepgoing.mac_ram_gb() == 64


@pytest.mark.skipif(sys.platform != "darwin", reason="hw.memsize is a macOS sysctl")
def test_mac_ram_gb_on_a_real_mac():
    assert keepgoing.mac_ram_gb() >= 8


def test_main_menu_prints_the_choice(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    write_session(tmp_path, "sess1", str(tmp_path), 4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(keepgoing, "running_local_model", lambda: None)
    monkeypatch.setattr(keepgoing, "mac_ram_gb", lambda: 16)  # Gemma 4 E4B and Hermes 4 14B fit
    monkeypatch.setattr("builtins.input", lambda: "2")
    keepgoing.main()
    out = capsys.readouterr()
    assert out.out == "sess1\tlocal\tdivinetribe/Hermes-4-14B-abliterated-4bit-mlx\t0\n"
    assert "Gemma 4 12B" not in out.err


def test_main_offers_the_running_model_first(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    write_session(tmp_path, "s", str(tmp_path), 1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(keepgoing, "running_local_model", lambda: "my/model")
    monkeypatch.setattr(keepgoing, "mac_ram_gb", lambda: 0)
    monkeypatch.setattr("builtins.input", lambda: "1")
    keepgoing.main()
    assert capsys.readouterr().out == "s\tlocal\tmy/model\t1\n"


@pytest.mark.parametrize("answer", ["q", "", "99", "abc"])
def test_main_quit_or_bad_choice_exits_1(tmp_path, monkeypatch, answer):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    write_session(tmp_path, "s", str(tmp_path), 1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(keepgoing, "running_local_model", lambda: None)
    monkeypatch.setattr(keepgoing, "mac_ram_gb", lambda: 0)
    monkeypatch.setattr(keepgoing, "free_cloud_models", lambda key: [])
    monkeypatch.setattr("builtins.input", lambda: answer)
    with pytest.raises(SystemExit) as e:
        keepgoing.main()
    assert e.value.code == 1


def test_main_without_a_session_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(keepgoing, "PROJECTS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        keepgoing.main()
    assert e.value.code == 1
