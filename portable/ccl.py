#!/usr/bin/env python3
"""Claude Code Local — portable installer/launcher for Linux, Windows and Android.

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

Android (Termux) adds two more pins in portable/android.json: the Claude Code
linux-arm64-musl binary from npm and Alpine's musl loader to run it, because
Claude Code ships no Android build and Termux's libc is bionic, not musl.
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
# Phones: CPU-only llama.cpp needs far less scratch, but Android itself keeps
# ~2 GB, so the budget is a smaller share of RAM with a smaller overhead.
ANDROID_OVERHEAD_GB = 0.75
ANDROID_RAM_SHARE = 0.75
# Context lengths to try, largest first; we shrink to make a model fit.
CTX_CHOICES = [16384, 8192, 4096]


def is_android():
    """Termux on Android. Python 3.13+ reports platform.system() == 'Android';
    older Termux Pythons report 'Linux', so also check the Termux prefix."""
    return (platform.system() == "Android" or sys.platform == "android"
            or os.environ.get("PREFIX", "").startswith("/data/data/com.termux"))


def home():
    env = os.environ.get("CCL_HOME")
    if env:
        return env
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "claude-code-local")
    return os.path.join(os.path.expanduser("~"), ".cache", "claude-code-local")


def load_android_pins():
    with open(os.path.join(HERE, "android.json")) as f:
        return json.load(f)


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
        if platform.system() in ("Linux", "Android"):
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
    osname = "Android" if is_android() else platform.system()
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
    # Phones run llama.cpp on the CPU (the pinned android-arm64 build); no
    # nvidia-smi/vulkaninfo probing there.
    accel, vram = (None, 0.0) if osname == "Android" else detect_gpu()
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

    if o == "Android":
        if a != "arm64":
            raise SystemExit(f"Android needs an arm64 phone. Detected: {a}")
        # Exact name: the "-snapdragon" variant also contains "bin-android-arm64".
        name = f"llama-{pins['tag']}-bin-android-arm64.tar.gz"
        if name not in assets:
            raise SystemExit(f"No pinned llama.cpp asset {name}. Run portable/tools/refresh_pins.py to repin.")
        return [name]

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


def model_need_gb(entry, quant, ctx, overhead_gb=RUNTIME_OVERHEAD_GB):
    kv = entry["kv_fixed_bytes"] + entry["kv_bytes_per_token"] * ctx
    return (quant_weight_bytes(entry, quant) + kv) / GB + overhead_gb


def overhead_for(hw):
    return ANDROID_OVERHEAD_GB if hw.get("os") == "Android" else RUNTIME_OVERHEAD_GB


def pick_model(hw, catalog):
    """Best coding+vision model that fits, at the highest precision it can.

    Budget is VRAM when a sized GPU is present, otherwise system RAM. We try the
    largest context first and shrink. Precision never silently drops below Q8:
    if some model fits at BF16/Q8 we take the strongest such model; only if none
    does do we fall back (and the caller prints that loudly).
    """
    ov = overhead_for(hw)
    if hw["accel"] in ("cuda", "rocm", "vulkan") and hw["vram_gb"] >= 2:
        budget = hw["vram_gb"] * 0.92
        budget_src = f"{hw['vram_gb']:.0f} GB VRAM ({hw['accel']})"
    elif hw.get("os") == "Android":
        budget = hw["ram_gb"] * ANDROID_RAM_SHARE
        budget_src = f"{hw['ram_gb']:.0f} GB phone RAM (CPU)"
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
                if q in entry["quants"] and model_need_gb(entry, q, ctx, ov) <= budget:
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
                if model_need_gb(entry, q, ctx, ov) <= budget:
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
        f"vision model needs about {model_need_gb(catalog['models'][-1], 'Q4_K_M', CTX_CHOICES[-1], ov):.1f} GB.")


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

def sha256_file(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, size=None, sha256=None, sha512=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        if ((sha256 is None or sha256_file(dest) == sha256)
                and (sha512 is None or sha256_file(dest, "sha512") == sha512)):
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
    if sha512:
        got = sha256_file(dest, "sha512")
        if got != sha512:
            raise SystemExit(f"CHECKSUM MISMATCH for {dest}\n  expected sha512 {sha512}\n  got             {got}")
        print(f"  verified {os.path.basename(dest)} (sha512 ok)")
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


# ─── Android: Claude Code itself ──────────────────────────────────────────────
#
# Claude Code ships linux/mac/windows binaries but none for Android, and Termux's
# libc is bionic. Its linux-arm64-musl build needs only libc.musl, which is one
# file: Alpine's musl loader. So we pin both, verify both, point the binary's ELF
# interpreter at our loader (patchelf --set-interpreter, nothing else touched;
# setting an rpath too corrupts the bundled JS), and wrap it in $PREFIX/bin/claude.

def _extract_member(archive, member, dest, mode="r:gz"):
    with tarfile.open(archive, mode, ignore_zeros=True) as t:
        src = t.extractfile(member)
        if src is None:
            raise SystemExit(f"{member} not found in {archive}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
    os.replace(tmp, dest)
    os.chmod(dest, 0o755)
    return dest


def android_wrapper_text(claude_bin, musl_dir):
    return f"""#!/data/data/com.termux/files/usr/bin/bash
# Claude Code under Termux: the linux-arm64-musl build, run through a pinned
# musl loader. Written by claude-code-local (portable/ccl.py install).
# termux-exec must not route this exec through Android's linker (it is a musl
# binary), and its bionic LD_PRELOAD cannot load into a musl process.
export TERMUX_EXEC__SYSTEM_LINKER_EXEC__MODE=disable
export LD_LIBRARY_PATH="{musl_dir}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export SHELL="${{SHELL:-$PREFIX/bin/bash}}"
# The auto-updater would swap in an unpatched binary that cannot start here.
export DISABLE_AUTOUPDATER=1
unset LD_PRELOAD
exec "{claude_bin}" "$@"
"""


def install_claude_android(h, force=False):
    pins = load_android_pins()
    c, m = pins["claude"], pins["musl"]
    musl_dir = os.path.join(h, "musl", m["version"])
    loader = os.path.join(musl_dir, "ld-musl-aarch64.so.1")
    if not os.path.exists(loader) or force:
        arc = os.path.join(h, "dl", os.path.basename(m["url"]))
        download(m["url"], arc, m["size"], m["sha256"])
        _extract_member(arc, "lib/ld-musl-aarch64.so.1", loader)
    soname = os.path.join(musl_dir, "libc.musl-aarch64.so.1")
    if not os.path.lexists(soname):
        os.symlink("ld-musl-aarch64.so.1", soname)

    cdir = os.path.join(h, "claude", c["version"])
    claude_bin = os.path.join(cdir, "claude")
    if not os.path.exists(claude_bin) or force:
        arc = os.path.join(h, "dl", os.path.basename(c["url"]))
        download(c["url"], arc, c["size"], sha512=c["sha512"])
        os.makedirs(cdir, exist_ok=True)
        _extract_member(arc, "package/claude", claude_bin)
        patchelf = shutil.which("patchelf")
        if not patchelf:
            raise SystemExit("patchelf not found. In Termux:  pkg install patchelf")
        subprocess.run([patchelf, "--set-interpreter", loader, claude_bin], check=True)
        os.remove(arc)  # ~130 MB we no longer need on a phone

    prefix = os.environ.get("PREFIX") or "/data/data/com.termux/files/usr"
    wrapper = os.path.join(prefix, "bin", "claude")
    with open(wrapper, "w") as f:
        f.write(android_wrapper_text(claude_bin, musl_dir))
    os.chmod(wrapper, 0o755)
    print(f"Claude Code: {c['version']} (linux-arm64-musl via musl {m['version']}) → {wrapper}")
    return wrapper


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
    print(f"Fits:     need {model_need_gb(entry, quant, ctx, overhead_for(hw)):.1f} GB of {budget:.1f} GB usable ({budget_src})")
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
    print(f"             need {model_need_gb(entry, quant, ctx, overhead_for(hw)):.1f} GB of "
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

    claude_bin = None
    if hw["os"] == "Android":
        print("")
        claude_bin = install_claude_android(h, force=args.force)

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
        "claude_bin": claude_bin,
    }
    with open(os.path.join(h, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {os.path.join(h, 'config.json')}")
    print("Done. Start it with:  python3 ccl.py run")
    return cfg


def _wait_health(url, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
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


def cmd_run(args):
    h = home()
    cfgpath = os.path.join(h, "config.json")
    if not os.path.exists(cfgpath):
        raise SystemExit("Not installed yet. Run:  python3 ccl.py install")
    with open(cfgpath) as f:
        cfg = json.load(f)

    if is_android():
        # termux-exec otherwise launches binaries through /system/bin/linker64,
        # so /proc/self/exe is the linker and ggml finds no CPU backend .so next
        # to it ("no backends are loaded"). Termux's own builds may exec directly.
        os.environ["TERMUX_EXEC__SYSTEM_LINKER_EXEC__MODE"] = "disable"
    env = dict(os.environ)
    sep = ";" if platform.system() == "Windows" else ":"
    if platform.system() == "Windows":
        env["PATH"] = cfg["bindir"] + sep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = cfg["bindir"] + sep + env.get("LD_LIBRARY_PATH", "")

    llama_cmd = [cfg["server_bin"], "-m", cfg["model_gguf"], "--mmproj", cfg["mmproj"],
                 "-c", str(cfg["ctx"]), "--host", "127.0.0.1", "--port", "8080",
                 "-ngl", str(cfg["ngl"]), "--jinja"] + list(cfg.get("llama_args", []))
    print(f"Starting llama-server ({cfg['model_name']} [{cfg['quant']}], "
          f"{cfg['accel']}, ngl={cfg['ngl']})...")
    llama_proc = subprocess.Popen(llama_cmd, env=env)
    try:
        if not _wait_health("http://127.0.0.1:8080/health", 600):
            raise SystemExit("llama-server did not come up on :8080 within 10 min")
        print("llama-server is up on :8080")

        proxy_cmd = [sys.executable, os.path.join(HERE, "portable_server.py")]
        penv = dict(env)
        penv["LLAMA_SERVER_URL"] = "http://127.0.0.1:8080"
        penv["CCL_MODEL_NAME"] = f"{cfg['model_name']} [{cfg['quant']}]"
        print("Starting Anthropic proxy on :4000...")
        proxy_proc = subprocess.Popen(proxy_cmd, env=penv)

        if not _wait_health("http://127.0.0.1:4000/health", 60):
            raise SystemExit("portable proxy did not come up on :4000")
        print("Proxy is up. Claude Code → http://localhost:4000\n")

        if args.no_claude:
            print("Servers running. ANTHROPIC_BASE_URL=http://localhost:4000 "
                  "ANTHROPIC_API_KEY=sk-local")
            proxy_proc.wait()
            return

        claude = cfg.get("claude_bin") or resolve_claude_bin()
        if not claude:
            print("Claude Code CLI not found. Install it, then run with these set:")
            print("  ANTHROPIC_BASE_URL=http://localhost:4000")
            print("  ANTHROPIC_API_KEY=sk-local")
            proxy_proc.wait()
            return

        cenv = dict(os.environ)
        cenv["ANTHROPIC_BASE_URL"] = "http://localhost:4000"
        cenv["ANTHROPIC_API_KEY"] = "sk-local"
        if is_android():
            # musl finds no /etc/resolv.conf on Android; nothing should be
            # dialing out anyway when the model is on the phone.
            cenv["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        print(f"Launching Claude Code ({cfg['model_name']} [{cfg['quant']}], 100% local)\n")
        subprocess.call([claude, "--model", "claude-sonnet-4-6"], env=cenv)
        proxy_proc.terminate()
    finally:
        llama_proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Claude Code Local — portable (Linux/Windows/Android)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    pi = sub.add_parser("install")
    pi.add_argument("--force", action="store_true", help="re-download llama.cpp even if present")
    pr = sub.add_parser("run")
    pr.add_argument("--no-claude", action="store_true", help="start servers but don't launch Claude Code")
    a = ap.parse_args()
    if a.cmd == "detect":
        cmd_detect()
    elif a.cmd == "install":
        cmd_install(a)
    elif a.cmd == "run":
        cmd_run(a)
