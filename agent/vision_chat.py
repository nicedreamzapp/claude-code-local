#!/usr/bin/env python3
"""Drag a picture into the terminal and ask Muse Glimmer about it.

Why this exists separately from agent.py/chat.py: those two are text only. This
is the STOCK mlx-community build (not Matt's abliterated one, which has the
vision tower stripped), loaded in-process through mlx-vlm.

Needs the SEPARATE venv at ~/.local/mlx-vlm-latest (0.6.12+, which has
muse_glimmer). Do not point it at ~/.local/mlx-server — that one is older and
Song Forge depends on it.

Usage: type a question. To ask about a picture, drag the file into the window
(Terminal pastes the path) and type your question on the same line. The picture
stays attached to follow-up questions until you type /clear.

It remembers the conversation and saves it after every reply, so a closed or
killed window loses nothing: /resume picks the thread back up, /sessions lists
them. One honest limit — resuming replays what was SAID, not the pictures
themselves, so re-drag an image if you want it looked at again.

Glimmer thinks out loud in a reasoning channel the server leaks into the reply,
so everything before the final `to=user` marker is hidden unless you /think.
"""
import os
import sys
import time
import readline  # noqa: F401 — enables line editing and history

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/SongForgeM5"))
import chat_store  # noqa: E402
from mem_client import reserve  # noqa: E402

MODEL = os.getenv("VISION_MODEL", "mlx-community/Muse-Glimmer-30B-bf16")
# Set when the launcher has already stopped Song Forge: forge_guard keeps its
# 60 GB reservation even with the engines down, so a 62 GB ask can never be
# granted and reserve() would just block for its whole timeout. Skipping the
# lease is only correct because the room was made explicitly.
NO_LEASE = os.getenv("VISION_NO_LEASE") == "1"
LEASE_GB = int(os.getenv("VISION_LEASE_GB", "24"))
MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "800"))
# Follow-ups re-send the whole thread, so an unbounded history would eventually
# blow the context on a 30B. Keep the last N exchanges.
KEEP_TURNS = int(os.getenv("VISION_KEEP_TURNS", "12"))

DIM, BOLD, CYAN, GREEN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"
# The real answer starts after this marker; everything before it is Glimmer
# talking to itself. A tuple because the server has shipped both spellings.
ANSWER_MARKERS = ("<|start|>assistant to=user<|message|>", "to=user<|message|>")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic")


class _NoLease:
    """Stand-in for reserve() when the caller already made the room."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def split_paths(line):
    """Pull image paths out of a line, returning (paths, remaining_text).

    Terminal drag-and-drop escapes spaces as '\\ ', so unescape before testing.
    """
    words, paths, rest = line.split(), [], []
    buf = ""
    for w in words:
        buf = (buf + " " + w) if buf else w
        if buf.endswith("\\"):
            continue  # escaped space, the path keeps going
        cand = buf.replace("\\ ", " ").strip("'\"")
        if cand.lower().endswith(IMAGE_EXT) and os.path.isfile(cand):
            paths.append(cand)
        else:
            rest.append(buf)
        buf = ""
    if buf:
        rest.append(buf)
    return paths, " ".join(rest).strip()


def visible(text, show_thinking):
    if show_thinking:
        return text
    for m in ANSWER_MARKERS:
        if m in text:
            return text.split(m)[-1].replace("<|eom|>", "").strip()
    return text.strip()


def show_sessions(rows):
    if not rows:
        print("%s  no saved conversations yet%s" % (DIM, RESET))
        return
    for i, (path, mtime, n, first) in enumerate(rows, 1):
        print("%s  %d. %11s · %3d messages · %s%s"
              % (DIM, i, chat_store.ago(mtime), n, first, RESET))


def main():
    show_thinking = os.getenv("VISION_SHOW_THINKING") == "1"
    print("%s  Muse Glimmer 30B · SEEING (%s)%s"
          % (BOLD, "full quality" if "bf16" in MODEL else MODEL.split("-")[-1], RESET))
    print("%s  /clear drops the picture · /reset new conversation · /resume · "
          "/sessions · /think · /quit%s" % (DIM, RESET))
    print("%s  %s%s" % (DIM, "loading, ~60 GB, give it a minute" if NO_LEASE
                        else "waiting for a memory seat from forge_guard...", RESET))

    with (_NoLease() if NO_LEASE else reserve("glimmer-vision", LEASE_GB)):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template

        t0 = time.time()
        model, processor = load(MODEL)
        print("%s  loaded in %.0fs. drag a picture in and ask about it.%s"
              % (DIM, time.time() - t0, RESET))

        chat_store.prune(MODEL)
        recent = chat_store.sessions(MODEL, limit=1)
        if recent:
            print("%s  last conversation: %d messages, %s — /resume to pick it up%s"
                  % (DIM, recent[0][2], chat_store.ago(recent[0][1]), RESET))
        print()

        session = chat_store.new_session(MODEL)
        messages, images = [], []
        while True:
            try:
                line = input("%s› %s" % (CYAN, RESET)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in ("/quit", "/exit"):
                return 0
            if line == "/clear":
                images = []
                print("%s  picture dropped.%s" % (DIM, RESET))
                continue
            if line == "/reset":
                messages, images = [], []
                session = chat_store.new_session(MODEL)
                print("%s  new conversation.%s" % (DIM, RESET))
                continue
            if line == "/think":
                show_thinking = not show_thinking
                print("%s  thinking %s.%s"
                      % (DIM, "shown" if show_thinking else "hidden", RESET))
                continue
            if line == "/sessions":
                show_sessions(chat_store.sessions(MODEL))
                continue
            if line.split()[0] == "/resume":
                rows = chat_store.sessions(MODEL)
                if not rows:
                    print("%s  nothing saved yet%s" % (DIM, RESET))
                    continue
                arg = line.split()[1:]
                try:
                    pick = int(arg[0]) if arg else 1
                except ValueError:
                    pick = 0
                if not 1 <= pick <= len(rows):
                    print("%s  which one? /resume <number>:%s" % (DIM, RESET))
                    show_sessions(rows)
                    continue
                messages = chat_store.load(rows[pick - 1][0])
                session, images = rows[pick - 1][0], []
                print("%s  resumed %d messages from %s — it remembers what was "
                      "said, re-drag a picture to show it again%s"
                      % (DIM, len(messages), chat_store.ago(rows[pick - 1][1]), RESET))
                continue

            new_images, question = split_paths(line)
            if new_images:
                images = new_images
                print("%s  looking at %s%s"
                      % (DIM, ", ".join(os.path.basename(p) for p in images), RESET))
            if not question:
                question = "What is in this picture? Describe it, and read any text you can see."

            messages.append({"role": "user", "content": question})
            prompt = apply_chat_template(
                processor, model.config, messages[-KEEP_TURNS * 2:],
                num_images=len(images),
            )
            t1 = time.time()
            try:
                out = generate(model, processor, prompt, images or None,
                               max_tokens=MAX_TOKENS, verbose=False)
            except KeyboardInterrupt:
                messages.pop()          # drop the unanswered question
                print("%s  interrupted%s" % (DIM, RESET))
                continue
            text = out.text if hasattr(out, "text") else str(out)
            answer = visible(text, show_thinking)
            messages.append({"role": "assistant", "content": answer})
            # Saved per completed turn, not at exit: these windows get closed and
            # killed, and an exit hook would lose the whole thread.
            note = ("[looking at %s] " % ", ".join(os.path.basename(p) for p in images)
                    if images else "")
            chat_store.append(session, "user", note + question)
            chat_store.append(session, "assistant", answer)

            print("\n%s%s%s" % (GREEN, answer, RESET))
            print("%s  %.0fs · %.0f tok/s%s\n"
                  % (DIM, time.time() - t1, getattr(out, "generation_tps", 0), RESET))


if __name__ == "__main__":
    sys.exit(main())
