"""portable/ccl.py on Android (Termux): detection, llama.cpp asset, model sizing, pins."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PORTABLE = os.path.join(os.path.dirname(HERE), "portable")


@pytest.fixture(scope="module")
def ccl():
    spec = importlib.util.spec_from_file_location("ccl", os.path.join(PORTABLE, "ccl.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def phone(ram_gb):
    return {"os": "Android", "arch": "arm64", "accel": None, "vram_gb": 0.0, "ram_gb": ram_gb}


def test_termux_prefix_means_android(ccl, monkeypatch):
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    assert ccl.is_android()
    assert ccl.detect_hw()["os"] == "Android"


def test_plain_linux_is_not_android(ccl, monkeypatch):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(ccl.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ccl.sys, "platform", "linux")
    assert not ccl.is_android()


def test_android_gets_plain_arm64_build_not_snapdragon(ccl):
    llama, _ = ccl.load_pins()
    names = ccl.pick_llama_assets(phone(8), llama)
    assert names == [f"llama-{llama['tag']}-bin-android-arm64.tar.gz"]


def test_android_x64_refused(ccl):
    llama, _ = ccl.load_pins()
    with pytest.raises(SystemExit):
        ccl.pick_llama_assets(dict(phone(8), arch="x64"), llama)


def test_8gb_phone_runs_2b_at_bf16(ccl, monkeypatch):
    monkeypatch.delenv("CCL_BUDGET_GB", raising=False)
    monkeypatch.delenv("CCL_MODEL_ID", raising=False)
    _, catalog = ccl.load_pins()
    entry, quant, ctx, _, _ = ccl.pick_model(phone(7.75), catalog)
    assert (entry["id"], quant) == ("qwen3.5-2b", "BF16")
    assert entry["vision"]


def test_12gb_phone_never_drops_below_q8(ccl, monkeypatch):
    monkeypatch.delenv("CCL_BUDGET_GB", raising=False)
    monkeypatch.delenv("CCL_MODEL_ID", raising=False)
    _, catalog = ccl.load_pins()
    entry, quant, _, _, _ = ccl.pick_model(phone(11.5), catalog)
    assert quant in ("BF16", "Q8_0")
    assert entry["params_b"] >= 4


def test_android_pins_are_complete(ccl):
    pins = ccl.load_android_pins()
    c, m = pins["claude"], pins["musl"]
    assert "linux-arm64-musl" in c["url"] and c["version"] in c["url"]
    assert len(c["sha512"]) == 128 and c["size"] > 0
    assert m["url"].endswith(f"musl-{m['version']}.apk")
    assert len(m["sha256"]) == 64 and m["size"] > 0


def test_wrapper_drops_bionic_preload_and_updater(ccl):
    w = ccl.android_wrapper_text("/x/claude", "/x/musl")
    assert "unset LD_PRELOAD" in w
    assert "TERMUX_EXEC__SYSTEM_LINKER_EXEC__MODE=disable" in w
    assert "DISABLE_AUTOUPDATER=1" in w
    assert w.rstrip().endswith('exec "/x/claude" "$@"')


def test_catalog_phone_model_has_vision_and_tools():
    with open(os.path.join(PORTABLE, "catalog.json")) as f:
        cat = json.load(f)
    m = next(x for x in cat["models"] if x["id"] == "qwen3.5-2b")
    assert m["vision"] and "vision_config" in m["vision_evidence"]
    assert {"BF16", "Q8_0"} <= set(m["quants"])
