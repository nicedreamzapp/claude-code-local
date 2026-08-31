#!/usr/bin/env python3
"""Conversation persistence for the local chat windows — so closing a terminal
stops losing the conversation.

Matt, 2026-08-12, after losing a long Gemma thread: these windows had no resume,
no transcript, and readline only ever kept the lines he TYPED (and only on a
clean exit, which a killed terminal isn't).

Design notes that matter:
  * One JSONL file per session, appended after every completed turn. A window
    that gets killed mid-thought still leaves everything up to the last reply —
    that is the whole point, so nothing is buffered.
  * Sessions are per model, because resuming a Gemma thread into Glimmer would
    silently graft one model's voice onto another's context.
  * Files are plain text and stay put. Old ones are pruned by COUNT, never by
    age, so a thread from months ago is still there if it was one of the last N.
"""
import json
import os
import time
from pathlib import Path

ROOT = Path(os.getenv("CHAT_SESSION_DIR",
                      Path.home() / ".local" / "chat-sessions"))
KEEP = int(os.getenv("CHAT_SESSION_KEEP", "40"))   # per model


def ago(ts):
    """Human gap since ts, for the session list."""
    m = (time.time() - ts) / 60
    if m < 60:
        return f"{m:.0f} min ago"
    if m < 60 * 24:
        return f"{m/60:.0f} hr ago"
    return f"{m/1440:.0f} days ago"


def _slug(model):
    return model.rstrip("/").split("/")[-1].replace(" ", "_")[:60]


def new_session(model):
    """Path for a fresh session. Not created until the first append."""
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT / f"{_slug(model)}--{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def append(path, role, content):
    """Append one message. Never raises — a failed write must not kill a chat."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"t": time.time(), "role": role,
                                "content": content}) + "\n")
    except Exception:
        pass


def new_snapshot(model):
    """Path for a snapshot-style session (the agent windows).

    The plain chat appends one line per message, but an agent turn is a whole
    burst — assistant, tool call, tool result, assistant again — and a torn
    burst is worse than no burst. So those windows rewrite the entire message
    list after each completed turn instead.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT / f"{_slug(model)}--{time.strftime('%Y%m%d-%H%M%S')}.json"


def save_snapshot(path, messages):
    """Write the whole conversation atomically. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"t": time.time(), "messages": messages}, f)
        os.replace(tmp, path)
    except Exception:
        pass


def text_of(content):
    """Flatten a message's content — the agent engines use both a plain string
    and OpenAI-style [{"type": "text", ...}] parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict))
    return ""


def sessions(model, limit=10):
    """Recent sessions for this model, newest first.

    Each entry: (path, mtime, n_messages, first_user_line). Empty or unreadable
    files are skipped rather than shown as blank rows.
    """
    out = []
    files = list(ROOT.glob(f"{_slug(model)}--*.jsonl")) + \
        list(ROOT.glob(f"{_slug(model)}--*.json"))
    for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        msgs = load(p)
        if not msgs:
            continue
        first = text_of(next((m["content"] for m in msgs
                              if m.get("role") == "user"), ""))
        out.append((p, p.stat().st_mtime, len(msgs), first.replace("\n", " ")[:60]))
        if len(out) >= limit:
            break
    return out


def load(path):
    """Messages from a session file — a snapshot, or a JSONL skipping any
    half-written trailing line."""
    if str(path).endswith(".json"):
        try:
            with open(path) as f:
                return json.load(f).get("messages", [])
        except Exception:
            return []
    msgs = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue          # torn last line from a killed window
                if d.get("role") in ("user", "assistant") and d.get("content"):
                    msgs.append({"role": d["role"], "content": d["content"]})
    except OSError:
        return []
    return msgs


def prune(model, keep=KEEP):
    """Keep the newest `keep` sessions for this model, delete older files."""
    try:
        files = sorted(list(ROOT.glob(f"{_slug(model)}--*.jsonl"))
                       + list(ROOT.glob(f"{_slug(model)}--*.json")),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep:]:
            p.unlink()
    except OSError:
        pass
