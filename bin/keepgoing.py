#!/usr/bin/env python3
"""keepgoing — out of Claude credits? Carry THIS folder's conversation on with another model.

Finds the most recent Claude Code conversation that ran in the current folder, then shows what you
can continue it with right now:

  * FREE CLOUD  — free models on OpenRouter (needs a free OPENROUTER_API_KEY)
  * THIS MAC    — models from the Claude Code Local lineup that fit your RAM, served by this repo's
                  MLX server on port 4000
  * THIS PC     — on Linux and Windows, the model portable/ccl.py installed, served by llama.cpp
                  behind portable/portable_server.py on the same port (4000, or CCL_PORT)

With --auto there is no menu: it takes the first free cloud model if you have a key, else the model
already running on this Mac, else the biggest lineup model that fits your RAM (on Linux and Windows:
the installed llama.cpp model). That is what the
Claude Code hook in scripts/limit-handoff.py runs when you hit a usage limit.

The choice is printed for bin/keepgoing (macOS) or `ccl.py keepgoing` (Linux, Windows) as one
tab-separated line:
  <session_id> <kind: cloud|local> <model> <already_running: 0|1>
Everything meant for you goes to stderr.
"""
import glob
import json
import ntpath
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

# Claude Code keeps its conversations under ~/.claude (or CLAUDE_CONFIG_DIR) on every OS.
PROJECTS = os.path.join(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"),
                        "projects")
PORT = int(os.environ.get("MLX_PORT") or os.environ.get("CCL_PORT") or "4000")
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = os.name == "nt"
C = dict(dim="\033[2m", b="\033[1m", grn="\033[32m", ylw="\033[33m", red="\033[31m",
         cyn="\033[36m", off="\033[0m")

# The lineup setup.sh installs from (see README "Pick your fighter"), smallest first.
# (Hugging Face id, label, minimum Mac RAM in GB)
LOCAL = [
    ("mlx-community/Qwen3.5-2B-MLX-8bit", "Qwen 3.5 2B", 8),
    ("mlx-community/Qwen3.5-4B-bf16", "Qwen 3.5 4B", 16),
    ("divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text", "Gemma 4 12B", 32),
    ("divinetribe/gemma-4-31b-it-abliterated-4bit-mlx", "Gemma 4 31B", 64),
    ("lmstudio-community/Qwen3.8-27B-MLX-8bit", "Qwen 3.8 27B", 96),
]


# The placeholder key every local launch hands Claude Code. Our servers never check it.
LOCAL_KEY = "sk-local"


def local_model_alias(model):
    """The model name Claude Code gets on a local run, and so shows in its header: "local:" plus
    the model's own name ("mlx-community/Qwen3.5-4B-bf16" -> "local:Qwen3.5-4B-bf16", the llama.cpp
    label "Qwen 3.5 4B [Q4_K_M]" -> "local:Qwen-3.5-4B-Q4_K_M"). Both local servers answer whatever
    model name a request carries, so this only changes what you see, never which model runs."""
    name = re.split(r"[\\/]", (model or "").strip().rstrip("/\\"))[-1]
    name = re.sub(r"\s+", "-", re.sub(r"[\[\]]", "", name).strip())
    return f"local:{name}" if name else "local"


def claude_config_file():
    """Claude Code's own settings file: ~/.claude.json, or .claude.json inside CLAUDE_CONFIG_DIR."""
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~")
    return os.path.join(base, ".claude.json")


def approve_local_key(path=None, key=LOCAL_KEY):
    """Record the local placeholder key as approved, the way Claude Code does when you answer Yes
    to "Detected a custom API key": the key's last 20 characters go in
    customApiKeyResponses.approved in ~/.claude.json. Without it the first local run stops on that
    prompt. Only that one entry changes; every other key in the file is kept. The file is backed up
    to .claude.json.ccl-backup first and replaced atomically.

    Returns "already", "approved", "rejected" (you said No to this key before, so it is left
    alone) or "skipped" (the file isn't valid JSON, or couldn't be written)."""
    path = path or claude_config_file()
    tail = key[-20:]
    exists = os.path.exists(path)
    try:
        if exists:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
    except (OSError, ValueError):
        return "skipped"
    if not isinstance(data, dict):
        return "skipped"
    resp = data.get("customApiKeyResponses")
    if resp is None:
        resp = {"approved": [], "rejected": []}
    if not isinstance(resp, dict) or not isinstance(resp.get("approved", []), list):
        return "skipped"
    if tail in resp.get("approved", []):
        return "already"
    if tail in (resp.get("rejected") or []):
        return "rejected"
    resp = dict(resp)
    resp["approved"] = list(resp.get("approved", [])) + [tail]
    resp.setdefault("rejected", [])
    data["customApiKeyResponses"] = resp
    folder = os.path.dirname(os.path.abspath(path))
    tmp = None
    try:
        os.makedirs(folder, exist_ok=True)
        if exists:
            shutil.copy2(path, path + ".ccl-backup")
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".claude.json.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if exists:
            shutil.copymode(path, tmp)
        else:
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        tmp = None
    except OSError:
        return "skipped"
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return "approved"


def this_machine():
    return "this Mac" if IS_MAC else "this PC"


def say(msg=""):
    sys.stderr.write(msg + "\n")


def same_folder(a, b):
    """True if two cwd strings name the same folder. Windows paths ignore case and slash style
    (Claude Code may log C:\\x while the shell says c:\\x); everywhere else it is exact."""
    if not a or not b:
        return False
    if IS_WINDOWS:
        return ntpath.normcase(ntpath.normpath(a)) == ntpath.normcase(ntpath.normpath(b))
    return a == b


def portable_home():
    """Where portable/ccl.py keeps llama.cpp, the model and config.json (same rule as ccl.home())."""
    if os.environ.get("CCL_HOME"):
        return os.environ["CCL_HOME"]
    if IS_WINDOWS:
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                            "claude-code-local")
    return os.path.join(os.path.expanduser("~"), ".cache", "claude-code-local")


def installed_portable_model():
    """The llama.cpp model the Linux/Windows installer set up, as the label its server reports on
    /health ("<name> [<quant>]"), or None if nothing is installed yet."""
    try:
        with open(os.path.join(portable_home(), "config.json")) as f:
            cfg = json.load(f)
        return f"{cfg['model_name']} [{cfg['quant']}]"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def local_lineup():
    """(model, label, minimum RAM GB) choices for this machine. On a Mac: the MLX lineup. On Linux
    and Windows: the one model portable/ccl.py installed (it was sized to fit when installed)."""
    if IS_MAC:
        return LOCAL
    model = installed_portable_model()
    return [(model, model, 0)] if model else []


def find_session(cwd):
    """(session_id, turns, last_modified) of the newest conversation that ran in cwd."""
    best = None
    for path in glob.glob(os.path.join(PROJECTS, "*", "*.jsonl")):
        try:
            if os.path.getsize(path) < 200:
                continue
            session_cwd, turns = None, 0
            with open(path, errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if session_cwd is None and rec.get("cwd"):
                        session_cwd = rec["cwd"]
                    if rec.get("type") == "user":
                        turns += 1
            if not same_folder(session_cwd, cwd) or turns < 1:
                continue
            mtime = os.path.getmtime(path)
            if best is None or mtime > best[2]:
                best = (os.path.splitext(os.path.basename(path))[0], turns, mtime)
        except OSError:
            continue
    return best


def free_cloud_models(key, limit=3):
    """Free OpenRouter models that can use tools and hold a long conversation, biggest context first."""
    if not key:
        return []
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)["data"]
    except Exception as e:  # offline, bad key, etc. — the local list still works
        say(f"{C['dim']}  (couldn't reach OpenRouter: {e}){C['off']}")
        return []
    picks = [m for m in data
             if m.get("id", "").endswith(":free")
             and "tools" in (m.get("supported_parameters") or [])
             and (m.get("context_length") or 0) >= 128_000]
    picks.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    return [m["id"] for m in picks[:limit]]


def mac_ram_gb():
    try:
        # full path: /usr/sbin is often missing from PATH (hooks, launchd), and RAM then read as 0
        sysctl = "/usr/sbin/sysctl" if os.path.exists("/usr/sbin/sysctl") else "sysctl"
        out = subprocess.run([sysctl, "-n", "hw.memsize"], capture_output=True, text=True).stdout
        return int(out.strip()) // (1 << 30)
    except Exception:
        return 0


def running_local_model():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return json.load(r).get("model")
    except Exception:
        return None


def ago(seconds):
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


def auto_pick(cloud, live, ram, lineup=None):
    """The --auto choice as (kind, model, already_running), or None if there is nothing to use."""
    lineup = LOCAL if lineup is None else lineup
    if cloud:
        return ("cloud", cloud[0], 0)
    if live:
        return ("local", live, 1)  # already loaded: instant, and nothing else gets swapped out
    fits = [m for m, _, need in lineup if ram and ram >= need]
    if fits:
        return ("local", fits[-1], 0)  # LOCAL is smallest first
    return ("local", lineup[0][0], 0) if lineup else None  # RAM unknown: the smallest is the safe bet


def main(argv=None):
    auto = "--auto" in (sys.argv[1:] if argv is None else argv)
    cwd = os.getcwd()
    session = find_session(cwd)
    if not session:
        say(f"{C['red']}  No Claude Code conversation found for this folder:{C['off']}")
        say(f"{C['dim']}    {cwd}{C['off']}")
        say(f"{C['dim']}  cd to the folder you were working in, then run keep going again.{C['off']}")
        sys.exit(1)
    sid, turns, mtime = session

    say(f"\n{C['cyn']}{C['b']}  Picking up your conversation in {os.path.basename(cwd)}{C['off']}")
    say(f"{C['dim']}  {turns} messages · last active {ago(time.time() - mtime)}{C['off']}\n")

    if auto:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        pick = auto_pick(free_cloud_models(key, limit=1), running_local_model(), mac_ram_gb(),
                         local_lineup())
        if not pick:
            say(f"{C['red']}  Nothing to keep going with.{C['off']}")
            sys.exit(1)
        kind, model, already = pick
        where = "free cloud" if kind == "cloud" else "already running" if already else this_machine()
        say(f"{C['b']}  Keeping going automatically with {model}{C['off']} {C['dim']}({where}){C['off']}")
        print("\t".join([sid, kind, model, str(already)]))
        return

    say(f"{C['b']}  Pick a model to keep going with:{C['off']}\n")

    options = []

    key = os.environ.get("OPENROUTER_API_KEY", "")
    say(f"{C['dim']}  FREE CLOUD{C['off']}")
    cloud = free_cloud_models(key)
    for model in cloud:
        options.append(("cloud", model, 0))
        say(f"{C['grn']}  {len(options):2}{C['off']}  {model}")
    if not key:
        say(f"{C['dim']}  (add a free OpenRouter key to see these — see README){C['off']}")
    elif not cloud:
        say(f"{C['dim']}  (none available right now){C['off']}")

    say(f"\n{C['dim']}  ON {this_machine().upper()}{C['off']}")
    live = running_local_model()
    ram = mac_ram_gb()
    lineup = local_lineup()
    if live:
        options.append(("local", live, 1))
        say(f"{C['grn']}  {len(options):2}{C['off']}  {live}  {C['grn']}already running, instant{C['off']}")
    for model, label, need in lineup:
        if model == live:
            continue
        if ram and ram < need:
            continue
        options.append(("local", model, 0))
        if IS_MAC:
            say(f"{C['grn']}  {len(options):2}{C['off']}  {label}  {C['dim']}(needs {need} GB+ Mac){C['off']}")
        else:
            say(f"{C['grn']}  {len(options):2}{C['off']}  {label}  {C['dim']}(llama.cpp){C['off']}")
    if not lineup and not live:
        say(f"{C['dim']}  (no local model installed yet: run install.sh or install.ps1){C['off']}")

    say("")
    sys.stderr.write("  Number (q = quit): ")
    sys.stderr.flush()
    try:
        choice = input().strip()
    except EOFError:
        choice = ""
    if not choice or choice.lower() == "q":
        sys.exit(1)
    try:
        kind, model, already = options[int(choice) - 1]
    except (ValueError, IndexError):
        say("  Not a valid choice.")
        sys.exit(1)
    print("\t".join([sid, kind, model, str(already)]))


def cli(argv):
    """Helpers for bin/keepgoing (bash): --local-alias MODEL prints the header name for a local run;
    --approve-local-key pre-approves sk-local in ~/.claude.json. Anything else runs the picker."""
    if argv[:1] == ["--local-alias"]:
        print(local_model_alias(argv[1] if len(argv) > 1 else ""))
        return
    if argv[:1] == ["--approve-local-key"]:
        if approve_local_key() == "rejected":
            say(f"{C['ylw']}  Claude Code has the local key ({LOCAL_KEY}) marked as rejected. "
                f"Answer Yes when it asks about the custom API key.{C['off']}")
        return
    main(argv)


if __name__ == "__main__":
    cli(sys.argv[1:])
