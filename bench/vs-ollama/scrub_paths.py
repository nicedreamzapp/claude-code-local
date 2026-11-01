#!/usr/bin/env python3
"""Replace absolute home paths in results/ with <repo> or ~ (run after summarize.py).
Only path text changes; no number in any record depends on it."""
import gzip
import os
import re
import sys

RES = os.environ.get("BENCH_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPO_RE = re.compile(r"/Users/[^/\s\"'\\]+/repos/claude-code-local")
HOME_RE = re.compile(r"/Users/[^/\s\"'\\]+")


def scrub(text):
    return HOME_RE.sub("~", REPO_RE.sub("<repo>", text))


changed = 0
for root, _, files in os.walk(RES):
    for fn in files:
        p = os.path.join(root, fn)
        gz = fn.endswith(".gz")
        raw = gzip.open(p, "rt", errors="replace").read() if gz else open(p, errors="replace").read()
        new = scrub(raw)
        if new != raw:
            changed += 1
            if gz:
                with gzip.open(p, "wt") as f:
                    f.write(new)
            else:
                with open(p, "w") as f:
                    f.write(new)
print(f"scrubbed {changed} files", file=sys.stderr)
