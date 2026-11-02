#!/usr/bin/env python3
"""Claude Code Local — portable installer/launcher for Linux and Windows.

One brain for "download and go" off a Mac:

    python3 ccl.py install     detect hardware, fetch the right pinned llama.cpp
                               build, pick the best VISION model that fits at the
                               highest precision it can, download it from Hugging
                               Face (resumable, checksum-verified), write config
    python3 ccl.py run         start llama-server, start the Anthropic proxy, then
                               launch Claude Code pointed at it (sk-local)
    python3 ccl.py detect      just print what it sees (hardware + the pick)

Everything is pinned in portable/llamacpp.json (llama.cpp release asset hashes)
and portable/catalog.json (model files + sha256, every entry a coding+tool-use
model WITH a vision tower). Stdlib only; run with -I so a planted module in the
download dir can't be imported. We depend on exactly two outside things: the
llama.cpp release binaries (MIT, pinned by tag) and Hugging Face for the weights.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

# On Windows the console defaults to a legacy code page (cp1252) that cannot
# encode the arrows/box characters we print, which crashes detect/install with
# UnicodeEncodeError. Force UTF-8 on our streams so output never aborts the run.
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PRECISION_ORDER = ["BF16", "Q8_0", "Q6_K", "Q4_K_M"]
GB = 1024 ** 3

# llama/ggml runtime + OS headroom we don't let the model eat into.
RUNTIME_OVERHEAD_GB = 1.5
# Context lengths to try, largest first; we shrink to make a model fit.
CTX_CHOICES = [16384, 8192, 4096]

# Ports: Claude Code talks to the Anthropic proxy (same port as the Mac server), the proxy talks
# to llama-server. Both can be moved, e.g. when something else already owns 4000 or 8080.
PROXY_PORT = int(os.environ.get("CCL_PORT") or 4000)
LLAMA_PORT = int(os.environ.get("CCL_LLAMA_PORT") or 8080)
PROXY_URL = f"http://localhost:{PROXY_PORT}"
# Same switches as the Mac launchers: with these set nothing leaves the machine.
OFFLINE_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
}


def home():
    env = os.environ.get("CCL_HOME")
    if env:
        return env
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "claude-code-local")
    return os.path.join(os.path.expanduser("~"), ".cache", "claude-code-local")


def load_pins():
    with open(os.path.join(HERE, "llamacpp.json")) as f:
        llama = json.load(f)
    with open(os.path.join(HERE, "catalog.json")) as f:
        catalog = json.load(f)
    return llama, catalog


# ─── Hardware detection ───────────────────────────────────────────────────────

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def detect_ram_gb():
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
        elif platform.system() == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / GB
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except Exception:
        return 8.0


def detect_gpu():
    """Return (accel, vram_gb). accel in {'cuda','rocm','vulkan',None}."""
    out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    vram = [int(x) for x in re.findall(r"\d+", out)]
    if vram:
        return "cuda", max(vram) / 1024  # MiB → GiB

    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"]) or _run(["rocm-smi", "--showmeminfo", "vram"])
    vram = [int(x) for x in re.findall(r"(\d{9,})", out)]  # bytes
    if vram:
        return "rocm", max(vram) / GB
    if shutil.which("rocminfo"):
        return "rocm", 0.0  # present but size unknown; sizing falls back to RAM

    out = _run(["vulkaninfo", "--summary"]) or _run(["vulkaninfo"])
    if out:
        # heap sizes print as hex, e.g. "size = 0x1e0000000"; the largest
        # DEVICE_LOCAL heap is a good proxy for VRAM.
        heaps = [int(x, 16) for x in re.findall(r"size\s*[:=]\s*0x([0-9a-fA-F]+)", out)]
        if heaps:
            return "vulkan", max(heaps) / GB
        return "vulkan", 0.0
    return None, 0.0


def detect_hw():
    osname = platform.system()
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
    accel, vram = detect_gpu()
    # Vulkan on an integrated GPU (e.g. Intel Iris Xe) advertises itself but
    # exposes no usable device-local heap to size; under 2 GB we cannot offload,
    # so treat it as CPU-only. This keeps the pinned-build pick, ngl, and the RAM
    # budget consistent instead of pulling the Vulkan binary we'd never offload to.
    if accel == "vulkan" and vram < 2:
        accel, vram = None, 0.0
    ram = detect_ram_gb()
    return {"os": osname, "arch": arch, "accel": accel,
            "vram_gb": round(vram, 1), "ram_gb": round(ram, 1)}


# ─── Pick the llama.cpp asset ─────────────────────────────────────────────────

def pick_llama_assets(hw, pins):
    """Return a list of asset names (binary first, cudart second if needed)."""
    o, a, accel = hw["os"], hw["arch"], hw["accel"]
    assets = pins["assets"]

    def has(name):
        return name in assets

    if o == "Linux":
        if accel == "cuda":
            if a == "arm64" and has("llama-b11669-bin-ubuntu-cuda-13.4-arm64.tar.gz"):
                return [find(assets, "bin-ubuntu-cuda", "arm64"), find(assets, "cudart", "ubuntu-cuda-13.4-arm64")]
            return [find(assets, "bin-ubuntu-cuda-12.8-x64"), find(assets, "cudart", "ubuntu-cuda-12.8-x64")]
        if accel == "rocm" and a == "x64":
            return [find(assets, "bin-ubuntu-rocm")]
        if accel == "vulkan":
            return [find(assets, "bin-ubuntu-vulkan", a)]
        return [find(assets, f"bin-ubuntu-{a}" if a == "arm64" else "bin-ubuntu-x64")]

    if o == "Windows":
        if accel == "cuda":
            if a == "arm64":
                return [find(assets, "bin-win-cuda-13.4-arm64"), find(assets, "cudart", "win-cuda-13.4-arm64")]
            return [find(assets, "bin-win-cuda-12.4-x64"), find(assets, "cudart", "win-cuda-12.4-x64")]
        if accel == "rocm" and a == "x64":
            return [find(assets, "bin-win-rocm")]
        if accel == "vulkan":
            return [find(assets, "bin-win-vulkan", a)]
        return [find(assets, f"bin-win-cpu-{a}")]

    raise SystemExit(f"Unsupported OS for the portable path: {o}")


def find(assets, *needles):
    # The CUDA runtime archive "cudart-llama-...-cuda-x64" contains the llama
    # binary's name as a substring, so a plain substring match for the binary
    # would grab cudart by accident. Unless "cudart" is one of the needles,
    # skip cudart archives.
    want_cudart = any("cudart" in n for n in needles)
    for name in assets:
        if not want_cudart and name.startswith("cudart"):
            continue
        if all(n in name for n in needles):
            return name
    raise SystemExit(f"No pinned llama.cpp asset matches {needles}. "
                     f"Run portable/tools/refresh_pins.py to repin.")


# ─── Pick the model ───────────────────────────────────────────────────────────

def quant_weight_bytes(entry, quant):
    return sum(f["size"] for f in entry["quants"][quant]) + entry["mmproj"]["size"]


def model_need_gb(entry, quant, ctx):
    kv = entry["kv_fixed_bytes"] + entry["kv_bytes_per_token"] * ctx
    return (quant_weight_bytes(entry, quant) + kv) / GB + RUNTIME_OVERHEAD_GB


def pick_model(hw, catalog):
    """Best coding+vision model that fits, at the highest precision it can.

    Budget is VRAM when a sized GPU is present, otherwise system RAM. We try the
    largest context first and shrink. Precision never silently drops below Q8:
    if some model fits at BF16/Q8 we take the strongest such model; only if none
    does do we fall back (and the caller prints that loudly).
    """
    if hw["accel"] in ("cuda", "rocm", "vulkan") and hw["vram_gb"] >= 2:
        budget = hw["vram_gb"] * 0.92
        budget_src = f"{hw['vram_gb']:.0f} GB VRAM ({hw['accel']})"
    else:
        budget = hw["ram_gb"] * 0.80
        budget_src = f"{hw['ram_gb']:.0f} GB system RAM (CPU)"
    if os.environ.get("CCL_BUDGET_GB"):
        budget = float(os.environ["CCL_BUDGET_GB"])
        budget_src = f"{budget:.0f} GB (CCL_BUDGET_GB override)"

    # Power-user / test override: force a specific catalog model by id. We still
    # pick the highest precision of it that fits, and still refuse to go below
    # Q8 silently (the caller prints the reason).
    forced = os.environ.get("CCL_MODEL_ID")
    if forced:
        entry = next((m for m in catalog["models"] if m["id"] == forced), None)
        if not entry:
            raise SystemExit(f"CCL_MODEL_ID={forced} is not in the catalog")
        for ctx in CTX_CHOICES:
            for q in PRECISION_ORDER:
                if q in entry["quants"] and model_need_gb(entry, q, ctx) <= budget:
                    return entry, q, ctx, budget, budget_src
        # nothing fits even at the smallest ctx — use lowest precision + ctx anyway
        q = [p for p in PRECISION_ORDER if p in entry["quants"]][-1]
        return entry, q, CTX_CHOICES[-1], budget, budget_src

    for ctx in CTX_CHOICES:
        hiq = None   # strongest model at BF16/Q8
        anyq = None  # strongest model at any precision
        for entry in catalog["models"]:
            for q in PRECISION_ORDER:
                if q not in entry["quants"]:
                    continue
                if model_need_gb(entry, q, ctx) <= budget:
                    if anyq is None:
                        anyq = (entry, q, ctx)
                    if q in ("BF16", "Q8_0") and hiq is None:
                        hiq = (entry, q, ctx)
                    break  # highest fitting precision for this model
        chosen = hiq or anyq
        if chosen:
            return chosen + (budget, budget_src)
    raise SystemExit(
        f"Nothing in the catalog fits your budget ({budget_src}). The smallest "
        f"vision model needs about {model_need_gb(catalog['models'][-1], 'Q4_K_M', CTX_CHOICES[-1]):.1f} GB.")


def precision_reason(entry, quant):
    have = [q for q in PRECISION_ORDER if q in entry["quants"]]
    higher = [q for q in have if PRECISION_ORDER.index(q) < PRECISION_ORDER.index(quant)]
    if quant == "BF16":
        return "BF16 — full precision, nothing to gain above it"
    if quant == "Q8_0" and (not higher or "BF16" in higher):
        return "Q8_0 — effectively lossless; BF16 did not fit the budget"
    if quant == "Q6_K":
        return "Q6_K — BF16/Q8 did not fit; Q6 is still high quality"
    if quant == "Q4_K_M":
        return ("Q4_K_M — ONLY this precision fit the budget; tool-calling is "
                "noticeably less reliable at 4-bit. More memory would let it run Q8+")
    return quant


# ─── Download with resume + checksum ──────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, size=None, sha256=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        if sha256 is None or sha256_file(dest) == sha256:
            print(f"  have {os.path.basename(dest)} ({size/GB:.2f} GB), skipping")
            return dest
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size and existing > size:
        existing = 0  # corrupt/overshot, restart
    headers = {"User-Agent": "claude-code-local"}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"  resuming {os.path.basename(dest)} at {existing/GB:.2f} GB")
    else:
        print(f"  downloading {os.path.basename(dest)}"
              + (f" ({size/GB:.2f} GB)" if size else ""))
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as out:
        total = existing
        last = time.time()
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if size and time.time() - last > 2:
                pct = 100 * total / size
                sys.stdout.write(f"\r    {pct:5.1f}%  {total/GB:.2f}/{size/GB:.2f} GB")
                sys.stdout.flush()
                last = time.time()
    if size:
        sys.stdout.write("\r" + " " * 50 + "\r")
    if sha256:
        got = sha256_file(dest)
        if got != sha256:
            raise SystemExit(f"CHECKSUM MISMATCH for {dest}\n  expected {sha256}\n  got      {got}")
        print(f"  verified {os.path.basename(dest)} (sha256 ok)")
    return dest


def extract(archive, into):
    os.makedirs(into, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into)


def find_llama_server(root):
    target = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for dirpath, _, files in os.walk(root):
        if target in files:
            return os.path.join(dirpath, target)
    return None


# ─── install / run ────────────────────────────────────────────────────────────

def cmd_detect():
    llama, catalog = load_pins()
    hw = detect_hw()
    print(f"OS/arch:  {hw['os']} {hw['arch']}")
    print(f"GPU:      {hw['accel'] or 'none'}  VRAM {hw['vram_gb']} GB")
    print(f"RAM:      {hw['ram_gb']} GB")
    assets = pick_llama_assets(hw, llama)
    print(f"llama.cpp: {llama['tag']} → {assets[0]}")
    entry, quant, ctx, budget, budget_src = pick_model(hw, catalog)
    print(f"Model:    {entry['name']}  [{quant}]  ctx {ctx}")
    print(f"Fits:     need {model_need_gb(entry, quant, ctx):.1f} GB of {budget:.1f} GB usable ({budget_src})")
    print(f"Precision: {precision_reason(entry, quant)}")
    print(f"Vision:   {entry['vision_evidence']}")
    return hw, entry, quant, ctx


def cmd_install(args):
    llama, catalog = load_pins()
    hw = detect_hw()
    h = home()
    print(f"Install dir: {h}")
    print(f"Hardware:    {hw['os']} {hw['arch']}, GPU {hw['accel'] or 'none'} "
          f"{hw['vram_gb']} GB, RAM {hw['ram_gb']} GB\n")

    # 1. llama.cpp
    asset_names = pick_llama_assets(hw, llama)
    bindir = os.path.join(h, "llama", llama["tag"])
    server_bin = find_llama_server(bindir)
    if not server_bin or args.force:
        for name in asset_names:
            a = llama["assets"][name]
            arc = os.path.join(h, "dl", name)
            download(a["url"], arc, a["size"], a["sha256"])
            extract(arc, bindir)
        server_bin = find_llama_server(bindir)
    if not server_bin:
        raise SystemExit(f"llama-server not found after extracting into {bindir}")
    print(f"llama.cpp:   {llama['tag']} ({hw['accel'] or 'cpu'}) → {server_bin}\n")

    # 2. model
    entry, quant, ctx, budget, budget_src = pick_model(hw, catalog)
    print(f"Model:       {entry['name']}  [{quant}]  ctx {ctx}")
    print(f"             need {model_need_gb(entry, quant, ctx):.1f} GB of "
          f"{budget:.1f} GB usable ({budget_src})")
    print(f"Precision:   {precision_reason(entry, quant)}")
    print(f"Vision:      {entry['vision_evidence']}\n")

    modeldir = os.path.join(h, "models", entry["id"])
    base = f"https://huggingface.co/{entry['repo']}/resolve/main/"
    model_paths = []
    for f in entry["quants"][quant]:
        dest = os.path.join(modeldir, os.path.basename(f["path"]))
        download(base + f["path"], dest, f["size"], f["sha256"])
        model_paths.append(dest)
    mm = entry["mmproj"]
    mmproj_path = os.path.join(modeldir, os.path.basename(mm["path"]))
    download(base + mm["path"], mmproj_path, mm["size"], mm["sha256"])

    # first shard is what llama-server loads; it auto-finds the rest
    model_paths.sort()
    cfg = {
        "home": h,
        "llama_tag": llama["tag"],
        "accel": hw["accel"] or "cpu",
        "server_bin": server_bin,
        "bindir": os.path.dirname(server_bin),
        "model_id": entry["id"],
        "model_name": entry["name"],
        "quant": quant,
        "ctx": ctx,
        "model_gguf": model_paths[0],
        "mmproj": mmproj_path,
        "ngl": 999 if hw["accel"] in ("cuda", "rocm", "vulkan") and hw["vram_gb"] >= 2 else 0,
        "llama_args": entry.get("llama_args", []),
    }
    with open(os.path.join(h, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {os.path.join(h, 'config.json')}")
    print("Done. Start it with:  python3 ccl.py run")
    return cfg


def _wait_health(url, timeout_s, proc=None):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # it exited (missing library, bad model path): no point waiting 10 min
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def resolve_claude_bin():
    c = shutil.which("claude")
    if c:
        return c
    cands = [os.path.expanduser("~/.local/bin/claude"),
             os.path.expanduser("~/.claude/local/claude")]
    if platform.system() == "Windows":
        cands += [os.path.expanduser(r"~\AppData\Local\Programs\claude\claude.exe")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def load_config():
    cfgpath = os.path.join(home(), "config.json")
    if not os.path.exists(cfgpath):
        raise SystemExit("Not installed yet. Run:  python3 ccl.py install")
    with open(cfgpath) as f:
        return json.load(f)


def start_servers(cfg, log_path=None):
    """Start llama-server and the Anthropic proxy; returns (llama_proc, proxy_proc).
    The caller terminates both. llama_proc is terminated here if anything fails.
    With log_path, both servers write there instead of this terminal (keep going does this so
    their output doesn't land in Claude Code's screen)."""
    out = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        out = open(log_path, "w")
    env = dict(os.environ)
    sep = ";" if platform.system() == "Windows" else ":"
    if platform.system() == "Windows":
        env["PATH"] = cfg["bindir"] + sep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = cfg["bindir"] + sep + env.get("LD_LIBRARY_PATH", "")

    llama_cmd = [cfg["server_bin"], "-m", cfg["model_gguf"], "--mmproj", cfg["mmproj"],
                 "-c", str(cfg["ctx"]), "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
                 "-ngl", str(cfg["ngl"]), "--jinja"] + list(cfg.get("llama_args", []))
    print(f"Starting llama-server ({cfg['model_name']} [{cfg['quant']}], "
          f"{cfg['accel']}, ngl={cfg['ngl']})...")
    llama_proc = subprocess.Popen(llama_cmd, env=env, stdout=out, stderr=out)
    try:
        if not _wait_health(f"http://127.0.0.1:{LLAMA_PORT}/health", 600, llama_proc):
            raise SystemExit(f"llama-server did not come up on :{LLAMA_PORT} within 10 min"
                             + (f" (log: {log_path})" if log_path else ""))
        print(f"llama-server is up on :{LLAMA_PORT}")

        proxy_cmd = [sys.executable, os.path.join(HERE, "portable_server.py")]
        penv = dict(env)
        penv["LLAMA_SERVER_URL"] = f"http://127.0.0.1:{LLAMA_PORT}"
        penv["CCL_MODEL_NAME"] = f"{cfg['model_name']} [{cfg['quant']}]"
        penv["CCL_PORT"] = str(PROXY_PORT)
        print(f"Starting Anthropic proxy on :{PROXY_PORT}...")
        proxy_proc = subprocess.Popen(proxy_cmd, env=penv, stdout=out, stderr=out)

        if not _wait_health(f"http://127.0.0.1:{PROXY_PORT}/health", 60):
            proxy_proc.terminate()
            raise SystemExit(f"portable proxy did not come up on :{PROXY_PORT}"
                             + (f" (log: {log_path})" if log_path else ""))
        print(f"Proxy is up. Claude Code → {PROXY_URL}\n")
        return llama_proc, proxy_proc
    except BaseException:
        llama_proc.terminate()
        raise


def _kg():
    """bin/keepgoing.py, which holds the shared keepgoing helpers."""
    bindir = os.path.join(REPO, "bin")
    if bindir not in sys.path:
        sys.path.insert(0, bindir)
    import keepgoing  # bin/keepgoing.py
    return keepgoing


def local_claude_env(model=None):
    """Claude Code's environment for a local run. With the model's label, Claude Code is told the
    real model ("local:<model>") so its header shows it instead of a Claude model; the proxy answers
    any model name. The sk-local placeholder is pre-approved in ~/.claude.json so the first run
    doesn't stop on "Detected a custom API key"."""
    kg = _kg()
    cenv = dict(os.environ)
    cenv["ANTHROPIC_BASE_URL"] = PROXY_URL
    cenv["ANTHROPIC_API_KEY"] = kg.LOCAL_KEY
    if model:
        alias = kg.local_model_alias(model)
        for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                  "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL"):
            cenv[k] = alias
    if kg.approve_local_key() == "rejected":
        print(f"  Claude Code has the local key ({kg.LOCAL_KEY}) marked as rejected. "
              "Answer Yes when it asks about the custom API key.")
    return cenv


def cmd_run(args):
    cfg = load_config()
    llama_proc, proxy_proc = start_servers(cfg)
    try:
        if args.no_claude:
            print(f"Servers running. ANTHROPIC_BASE_URL={PROXY_URL} "
                  "ANTHROPIC_API_KEY=sk-local")
            proxy_proc.wait()
            return

        claude = resolve_claude_bin()
        if not claude:
            print("Claude Code CLI not found. Install it, then run with these set:")
            print(f"  ANTHROPIC_BASE_URL={PROXY_URL}")
            print("  ANTHROPIC_API_KEY=sk-local")
            proxy_proc.wait()
            return

        label = f"{cfg['model_name']} [{cfg['quant']}]"
        print(f"Launching Claude Code ({label}, 100% local)\n")
        cenv = local_claude_env(label)
        subprocess.call([claude, "--model", cenv["ANTHROPIC_MODEL"]], env=cenv)
        proxy_proc.terminate()
    finally:
        llama_proc.terminate()


# ─── keep going (Linux + Windows) ─────────────────────────────────────────────
# The Mac runs bin/keepgoing (bash + MLX). Here the same bin/keepgoing.py finds the conversation
# and picks the model; this part starts llama.cpp if needed and resumes the conversation.

KEEPGOING_ENV = os.path.join(os.path.expanduser("~"), ".config", "claude-code-local", "keepgoing.env")


def load_keepgoing_env(path=KEEPGOING_ENV, environ=None):
    """KEY=value lines from keepgoing.env into the environment. A real environment variable wins."""
    environ = os.environ if environ is None else environ
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                environ.setdefault(k, v.strip().strip("'\""))
    except OSError:
        pass


def split_keepgoing_args(argv):
    """keepgoing's own flags, and everything after `--`, which goes to claude as is
    (e.g. `keepgoing --auto -- -p "finish the tests"`)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return list(argv), []


def pick_for_keepgoing(own_args):
    """Run bin/keepgoing.py's picker; (session_id, kind, model, already_running) or None."""
    import contextlib
    import io
    kg = _kg()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            kg.main(own_args)
    except SystemExit:
        return None
    line = out.getvalue().strip().splitlines()
    if not line:
        return None
    sid, kind, model, already = line[-1].split("\t")
    return sid, kind, model, already == "1"


def proxy_is_up():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _hook(name):
    cmd = os.environ.get(name)
    if cmd:
        subprocess.call(cmd, shell=True)


def cmd_keepgoing(argv):
    if os.name == "nt":
        os.system("")  # turns on ANSI colors in a classic Windows console for the menu
    load_keepgoing_env()
    own, extra = split_keepgoing_args(argv)
    sel = pick_for_keepgoing(own)
    if not sel:
        return 0
    sid, kind, model, already = sel
    claude = resolve_claude_bin()
    if not claude:
        print("Claude Code CLI not found. Install it, then run keep going again.")
        return 1

    if kind == "cloud":
        # OpenRouter speaks the Anthropic Messages API, so Claude Code talks to it directly.
        cenv = dict(os.environ)
        cenv.pop("ANTHROPIC_API_KEY", None)
        cenv.update({"ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
                     "ANTHROPIC_AUTH_TOKEN": os.environ.get("OPENROUTER_API_KEY", ""),
                     "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                     "ANTHROPIC_DEFAULT_SONNET_MODEL": model, "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                     "API_TIMEOUT_MS": "600000", "DISABLE_TELEMETRY": "1"})
        print(f"\n  Continuing on {model} (free cloud) …\n")
        return subprocess.call([claude, "--model", model, "--resume", sid] + extra, env=cenv)

    # Local: this install's llama.cpp model behind the Anthropic proxy.
    procs = None
    try:
        if not (already or proxy_is_up()):
            _hook("KEEPGOING_BEFORE_LOCAL")
            print(f"\n  Loading {model} on this PC …")
            log = os.path.join(home(), "keepgoing-server.log")
            print(f"  (server output: {log})")
            procs = start_servers(load_config(), log_path=log)
        cenv = local_claude_env(model)
        cenv.update(OFFLINE_ENV)
        cenv.pop("ANTHROPIC_AUTH_TOKEN", None)
        print(f"\n  Continuing on {model}, 100% on this PC …\n")
        return subprocess.call([claude, "--model", cenv["ANTHROPIC_MODEL"], "--resume", sid] + extra,
                               env=cenv)
    finally:
        if procs:
            for p in reversed(procs):
                p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=15)
                except Exception:
                    p.kill()
        if os.environ.get("KEEPGOING_AFTER_LOCAL"):
            print("\n  running your after-local command…")
            _hook("KEEPGOING_AFTER_LOCAL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Claude Code Local — portable (Linux/Windows)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    pi = sub.add_parser("install")
    pi.add_argument("--force", action="store_true", help="re-download llama.cpp even if present")
    pr = sub.add_parser("run")
    pr.add_argument("--no-claude", action="store_true", help="start servers but don't launch Claude Code")
    sub.add_parser("keepgoing", help="resume this folder's Claude Code conversation on a free "
                   "or local model ([--auto] [-- claude args])", add_help=False)
    if len(sys.argv) > 1 and sys.argv[1] == "keepgoing":
        sys.exit(cmd_keepgoing(sys.argv[2:]))
    a = ap.parse_args()
    if a.cmd == "detect":
        cmd_detect()
    elif a.cmd == "install":
        cmd_install(a)
    elif a.cmd == "run":
        cmd_run(a)
