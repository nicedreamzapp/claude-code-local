#!/usr/bin/env python3
"""Rebuild portable/catalog.json and portable/llamacpp.json from the source.

Maintainer tool, not run by the installer. It asks Hugging Face for the exact
size and sha256 of every model file, reads each base model's config.json to
confirm it has a vision tower (and to size its KV cache), and asks GitHub for
the sha256 of every llama.cpp release asset. The installer then verifies every
download against these pinned hashes.

    python3 portable/tools/refresh_pins.py              # both files
    python3 portable/tools/refresh_pins.py --llama b12000   # move the llama.cpp pin
    python3 portable/tools/refresh_pins.py --only android --claude 2.1.400

Stdlib only. Set GITHUB_TOKEN in the environment if you hit GitHub's rate limit.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORTABLE = os.path.dirname(HERE)

# Android (Termux): Claude Code's linux-arm64-musl npm build + Alpine's musl loader.
CLAUDE_VERSION = "2.1.312"
MUSL_APK = "https://dl-cdn.alpinelinux.org/alpine/v3.24/main/aarch64/musl-1.2.6-r3.apk"

LLAMA_TAG = "b11669"  # = llama.cpp v0.7.0 (its nightly-tag.txt), 2026-10-15

# Best first. Every entry is a general coding + tool-use model WITH vision.
# "base" is the original model repo whose config.json must carry a
# vision_config; the build refuses an entry without one.
MODELS = [
    dict(id="qwen3.6-27b", name="Qwen 3.6 27B", base="Qwen/Qwen3.6-27B", repo="unsloth/Qwen3.6-27B-GGUF",
         params_b=27, active_b=27,
         quants={"BF16": ["BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf", "BF16/Qwen3.6-27B-BF16-00002-of-00002.gguf"],
                 "Q8_0": ["Qwen3.6-27B-Q8_0.gguf"], "Q6_K": ["Qwen3.6-27B-Q6_K.gguf"], "Q4_K_M": ["Qwen3.6-27B-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=["--image-min-tokens", "1024"]),
    dict(id="gemma-4-31b", name="Gemma 4 31B", base="google/gemma-4-31B-it", repo="unsloth/gemma-4-31B-it-GGUF",
         params_b=31, active_b=31,
         quants={"BF16": ["BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf", "BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf"],
                 "Q8_0": ["gemma-4-31B-it-Q8_0.gguf"], "Q6_K": ["gemma-4-31B-it-Q6_K.gguf"], "Q4_K_M": ["gemma-4-31B-it-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=[]),
    dict(id="qwen3.6-35b-a3b", name="Qwen 3.6 35B-A3B (MoE)", base="Qwen/Qwen3.6-35B-A3B", repo="unsloth/Qwen3.6-35B-A3B-GGUF",
         params_b=35, active_b=3,
         quants={"BF16": ["BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf", "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf"],
                 "Q8_0": ["Qwen3.6-35B-A3B-Q8_0.gguf"], "Q6_K": ["Qwen3.6-35B-A3B-UD-Q6_K.gguf"], "Q4_K_M": ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=["--image-min-tokens", "1024"]),
    dict(id="gemma-4-26b-a4b", name="Gemma 4 26B-A4B (MoE)", base="google/gemma-4-26B-A4B-it", repo="unsloth/gemma-4-26B-A4B-it-GGUF",
         params_b=26, active_b=4,
         quants={"BF16": ["BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf", "BF16/gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf"],
                 "Q8_0": ["gemma-4-26B-A4B-it-Q8_0.gguf"], "Q6_K": ["gemma-4-26B-A4B-it-UD-Q6_K.gguf"], "Q4_K_M": ["gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=[]),
    dict(id="qwen3.5-9b", name="Qwen 3.5 9B", base="Qwen/Qwen3.5-9B", repo="unsloth/Qwen3.5-9B-GGUF",
         params_b=9, active_b=9,
         quants={"BF16": ["Qwen3.5-9B-BF16.gguf"], "Q8_0": ["Qwen3.5-9B-Q8_0.gguf"],
                 "Q6_K": ["Qwen3.5-9B-Q6_K.gguf"], "Q4_K_M": ["Qwen3.5-9B-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=["--image-min-tokens", "1024"]),
    dict(id="gemma-4-12b", name="Gemma 4 12B", base="google/gemma-4-12B-it", repo="unsloth/gemma-4-12b-it-GGUF",
         params_b=12, active_b=12,
         quants={"BF16": ["gemma-4-12b-it-BF16.gguf"], "Q8_0": ["gemma-4-12b-it-Q8_0.gguf"],
                 "Q6_K": ["gemma-4-12b-it-Q6_K.gguf"], "Q4_K_M": ["gemma-4-12b-it-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=[]),
    dict(id="qwen3.5-4b", name="Qwen 3.5 4B", base="Qwen/Qwen3.5-4B", repo="unsloth/Qwen3.5-4B-GGUF",
         params_b=4, active_b=4,
         quants={"BF16": ["Qwen3.5-4B-BF16.gguf"], "Q8_0": ["Qwen3.5-4B-Q8_0.gguf"],
                 "Q6_K": ["Qwen3.5-4B-Q6_K.gguf"], "Q4_K_M": ["Qwen3.5-4B-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=["--image-min-tokens", "1024"]),
    dict(id="gemma-4-e4b", name="Gemma 4 E4B", base="google/gemma-4-E4B-it", repo="unsloth/gemma-4-E4B-it-GGUF",
         params_b=8, active_b=4,
         quants={"BF16": ["gemma-4-E4B-it-BF16.gguf"], "Q8_0": ["gemma-4-E4B-it-Q8_0.gguf"],
                 "Q6_K": ["gemma-4-E4B-it-Q6_K.gguf"], "Q4_K_M": ["gemma-4-E4B-it-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=[]),
    # Phone tier (Android/Termux). Smallest model that still has a vision tower
    # and Qwen's <tool_call> template; BF16 is ~3.8 GB so it runs full precision
    # on an 8 GB phone, where 4B only fits at Q8.
    dict(id="qwen3.5-2b", name="Qwen 3.5 2B", base="Qwen/Qwen3.5-2B", repo="unsloth/Qwen3.5-2B-GGUF",
         params_b=2, active_b=2,
         quants={"BF16": ["Qwen3.5-2B-BF16.gguf"], "Q8_0": ["Qwen3.5-2B-Q8_0.gguf"],
                 "Q6_K": ["Qwen3.5-2B-Q6_K.gguf"], "Q4_K_M": ["Qwen3.5-2B-Q4_K_M.gguf"]},
         mmproj="mmproj-F16.gguf", args=["--image-min-tokens", "1024"]),
]


def get_json(url):
    headers = {"User-Agent": "claude-code-local-pins"}
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return json.load(r)


def kv_shape(cfg):
    """Bytes of f16 KV cache per token of context, plus a fixed part (sliding
    window layers and linear-attention state do not grow with context)."""
    t = cfg.get("text_config", cfg)
    layers = t["num_hidden_layers"]
    types = t.get("layer_types") or ["full_attention"] * layers
    full = sum(1 for x in types if "full" in x)
    sliding = sum(1 for x in types if "sliding" in x)
    linear = layers - full - sliding
    kv_heads = t.get("num_global_key_value_heads") or t.get("num_key_value_heads") or t["num_attention_heads"]
    head_dim = t.get("global_head_dim") or t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    per_token = 2 * full * kv_heads * head_dim * 2
    swa_heads = t.get("num_key_value_heads") or kv_heads
    swa_dim = t.get("head_dim") or head_dim
    fixed = 2 * sliding * swa_heads * swa_dim * 2 * (t.get("sliding_window") or 0)
    fixed += linear * 8 * 1024 * 1024  # recurrent state, generous
    return per_token, fixed, t.get("max_position_embeddings", 32768)


def build_catalog():
    out = []
    for m in MODELS:
        cfg = get_json(f"https://huggingface.co/{m['base']}/resolve/main/config.json")
        vc = cfg.get("vision_config")
        if not vc:
            sys.exit(f"REFUSED {m['id']}: {m['base']} has no vision_config, text-only models never ship")
        tree = get_json(f"https://huggingface.co/api/models/{m['repo']}/tree/main?recursive=1")
        files = {f["path"]: f for f in tree if f.get("type") == "file"}

        def pin(path):
            f = files.get(path)
            if not f or not f.get("lfs"):
                sys.exit(f"missing {m['repo']}/{path}")
            return {"path": path, "size": f["size"], "sha256": f["lfs"]["oid"]}

        per_token, fixed, max_ctx = kv_shape(cfg)
        entry = {k: m[k] for k in ("id", "name", "repo", "params_b", "active_b")}
        entry.update({
            "moe": m["active_b"] < m["params_b"] and "E4B" not in m["name"],
            "vision": True,
            "vision_evidence": f"{m['base']} config.json vision_config.model_type={vc.get('model_type')}; mmproj {m['mmproj']}",
            "kv_bytes_per_token": per_token,
            "kv_fixed_bytes": fixed,
            "max_ctx": max_ctx,
            "llama_args": m["args"],
            "mmproj": pin(m["mmproj"]),
            "quants": {q: [pin(p) for p in paths] for q, paths in m["quants"].items()},
        })
        out.append(entry)
        print(f"  {m['id']:18s} vision={vc.get('model_type'):22s} kv/token={per_token/1024:.0f}KiB", file=sys.stderr)
    return {"_note": "Generated by portable/tools/refresh_pins.py. Best model first. Vision only.", "models": out}


def build_llama(tag):
    rel = get_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")
    assets = {}
    for a in rel["assets"]:
        if not a.get("digest", "").startswith("sha256:"):
            continue
        assets[a["name"]] = {"url": a["browser_download_url"], "size": a["size"], "sha256": a["digest"][7:]}
    return {"_note": "Generated by portable/tools/refresh_pins.py. sha256 from GitHub's release asset digests.",
            "tag": tag, "published_at": rel["published_at"], "assets": assets}


def build_android(claude_version, musl_url):
    pkg = "@anthropic-ai/claude-code-linux-arm64-musl"
    meta = get_json(f"https://registry.npmjs.org/{pkg.replace('/', '%2f')}/{claude_version}")
    algo, b64 = meta["dist"]["integrity"].split("-", 1)
    if algo != "sha512":
        sys.exit(f"unexpected npm integrity {algo}")
    tgz = meta["dist"]["tarball"]
    req = urllib.request.Request(tgz, headers={"User-Agent": "claude-code-local-pins", "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tsize = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    with urllib.request.urlopen(urllib.request.Request(musl_url, headers={"User-Agent": "claude-code-local-pins"}),
                                timeout=60) as r:
        apk = r.read()
    mver = musl_url.rsplit("/", 1)[1][len("musl-"):-len(".apk")]
    return {
        "_note": "Generated by portable/tools/refresh_pins.py. Android/Termux only.",
        "claude": {"version": claude_version, "url": tgz, "size": tsize,
                   "sha512": base64.b64decode(b64).hex()},
        "musl": {"version": mver, "url": musl_url, "size": len(apk),
                 "sha256": hashlib.sha256(apk).hexdigest()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", default=LLAMA_TAG)
    ap.add_argument("--only", choices=["catalog", "llama", "android"])
    ap.add_argument("--claude", default=CLAUDE_VERSION)
    ap.add_argument("--musl", default=MUSL_APK)
    a = ap.parse_args()
    if a.only in (None, "catalog"):
        with open(os.path.join(PORTABLE, "catalog.json"), "w") as f:
            json.dump(build_catalog(), f, indent=1)
            f.write("\n")
    if a.only in (None, "llama"):
        with open(os.path.join(PORTABLE, "llamacpp.json"), "w") as f:
            json.dump(build_llama(a.llama), f, indent=1)
            f.write("\n")
    if a.only in (None, "android"):
        android = build_android(a.claude, a.musl)
        with open(os.path.join(PORTABLE, "android.json"), "w") as f:
            json.dump(android, f, indent=1)
            f.write("\n")
    print("ok", file=sys.stderr)
