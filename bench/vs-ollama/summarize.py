#!/usr/bin/env python3
"""Turn results/raw.jsonl + results/meta.jsonl into tables (markdown on stdout, summary.json)."""
import gzip
import json
import os
import statistics
import sys
import re
import shutil
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bench import WORK_ROOT, parse_stream  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("BENCH_RESULTS") or os.path.join(HERE, "results")


def load(name):
    p = os.path.join(RES, name)
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def fisher(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    from math import comb
    n1, n2, k, n = a + b, c + d, a + c, a + b + c + d
    pr = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n, k)  # noqa: E731
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def main():
    runs, metas = load("raw.jsonl"), load("meta.jsonl")
    # Keep every Claude Code transcript next to the numbers, and recount the
    # tool calls from them so a classifier fix applies to every run alike.
    sdir = os.path.join(RES, "streams")
    os.makedirs(sdir, exist_ok=True)
    for r in runs:
        dst = os.path.join(sdir, r["run_id"] + ".jsonl")
        src = os.path.join(WORK_ROOT, r["run_id"], "stream.jsonl")
        if not os.path.exists(dst) and os.path.exists(src):
            # drop the per-token "thinking_tokens" progress pings (no content)
            with open(src, errors="replace") as fi, open(dst, "w") as fo:
                fo.writelines(ln for ln in fi if '"subtype":"thinking_tokens"' not in ln)
        if os.path.exists(dst):
            st = parse_stream(dst)
            for k in ("tool_use", "tool_invalid", "tool_exec_error", "leaked_calls", "tools", "slashless_paths",
                      "leak_gemma_full", "leak_gemma_no_opener", "leak_function_tag", "leak_tool_call_json",
                      "bad_name_case", "bad_name_other", "bad_names"):
                r[k] = st[k]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["setup"])].append(r)
    meta = {(m["model"], m["setup"]): m for m in metas}
    summary = []
    print("| Model | Setup | Pass rate | Valid tool calls | Leaked (unparsed) calls | Invalid tool_use | "
          "Median wall s (all) | Median wall s (passed) | Median out tok/s | Timeouts | Peak server RSS / footprint |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (model, setup) in sorted(groups):
        rs = groups[(model, setup)]
        n, p = len(rs), sum(r["passed"] for r in rs)
        tu = sum(r["tool_use"] for r in rs)
        inv = sum(r["tool_invalid"] for r in rs)
        leak = sum(r["leaked_calls"] for r in rs)
        attempted = tu + leak
        valid = tu - inv
        m = meta.get((model, setup), {})
        row = {
            "model": model, "setup": setup, "runs": n, "passed": p, "pass_rate": round(p / n, 3),
            "tool_use": tu, "tool_invalid": inv, "leaked_calls": leak, "attempted": attempted,
            "valid_rate": round(valid / attempted, 3) if attempted else None,
            "median_wall_s": med([r["wall_s"] for r in rs]),
            "median_wall_passed_s": med([r["wall_s"] for r in rs if r["passed"]]),
            "median_out_tok_s": med([r["out_tok_per_api_s"] for r in rs]),
            "timeouts": sum(r["timed_out"] for r in rs),
            "slashless_paths": sum(r.get("slashless_paths", 0) for r in rs),
            "runs_with_slashless_path": sum(1 for r in rs if r.get("slashless_paths")),
            "cloud_guard_hits": sum(len(r["cloud_guard_hits"]) for r in rs),
            "remote_sockets": sorted({s for r in rs for s in r.get("remote_sockets", [])}),
            "peak_rss_mb": m.get("peak_rss_mb"), "peak_footprint_mb": m.get("peak_footprint_mb"),
            "model_name": rs[0]["model_name"], "digest": m.get("ollama_digest"),
        }
        summary.append(row)
        vr = f"{valid}/{attempted} ({row['valid_rate']:.0%})" if attempted else "n/a"
        mem = f"{(row['peak_rss_mb'] or 0)/1024:.1f} / {(row['peak_footprint_mb'] or 0)/1024:.1f} GB"
        print(f"| {model} | {setup} | {p}/{n} ({p/n:.0%}) | {vr} | {leak} | {inv} | {row['median_wall_s']} | "
              f"{row['median_wall_passed_s']} | {row['median_out_tok_s']} | {row['timeouts']} | {mem} |")

    print("\nfile_path arguments that reached Claude Code with the leading '/' dropped (written "
          "relative to the work dir, so the file lands in the wrong place):\n")
    print("| Model | Setup | Calls | Runs affected |")
    print("|---|---|---|---|")
    for row in summary:
        print(f"| {row['model']} | {row['setup']} | {row['slashless_paths']} | {row['runs_with_slashless_path']}/{row['runs']} |")

    kinds = ["leak_gemma_full", "leak_gemma_no_opener", "leak_function_tag", "leak_tool_call_json",
             "bad_name_case", "bad_name_other"]
    print("\nMalformed tool calls that reached Claude Code (from the transcripts). leak_* = tool-call "
          "markup left in plain text, never run; bad_name_* = a tool_use whose name is not a declared "
          "tool (case variant, or anything else such as an alias or Write). 'Failed runs with any' = "
          "runs that failed and contain at least one of these:\n")
    print("| Model | Setup | " + " | ".join(kinds) + " | Runs with any | Failed runs with any | Unknown names |")
    print("|---|---|" + "---|" * (len(kinds) + 3))
    for (model, setup) in sorted(groups):
        rs = groups[(model, setup)]
        tot = {k: sum(r.get(k, 0) for r in rs) for k in kinds}
        anyr = [r for r in rs if any(r.get(k, 0) for k in kinds)]
        names = {}
        for r in rs:
            for n, c in (r.get("bad_names") or {}).items():
                names[n] = names.get(n, 0) + c
        print(f"| {model} | {setup} | " + " | ".join(str(tot[k]) for k in kinds) +
              f" | {len(anyr)}/{len(rs)} | {sum(not r['passed'] for r in anyr)} | "
              + (", ".join(f"{n} x{c}" for n, c in sorted(names.items())) or "-") + " |")

    print("\nPer task (passes out of reps):\n")
    tasks = sorted({r["task"] for r in runs})
    cols = sorted(groups)
    print("| Task | " + " | ".join(f"{m} {s}" for m, s in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for t in tasks:
        cells = []
        for c in cols:
            rs = [r for r in groups[c] if r["task"] == t]
            cells.append(f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")
    # claude-code-local logs every repair it makes; count them per run set.
    pats = {
        "Gemma call parsed": r"Gemma4 tool call: ",
        "Gemma missing opener recovered": r"Gemma4 tool call with no opening marker recovered",
        "Gemma no name match": r"Gemma4 no name match",
        "Gemma no args parsed": r"Gemma4 no args parsed",
        "unrecoverable JSON": r"unrecoverable (?:tool_call|<tools>) JSON",
        "function-in-tag parsed": r"Recovered function-in-tag",
        "garbled JSON recovered": r"Recovered garbled tool call",
        "tagless call recovered": r"Recovered tagless tool call",
        "tool inferred from params": r"Inferred \w+ tool call",
        "tool name case fixed": r"Tool name case fixed",
        "tool name alias": r"Tool name alias applied",
        "argument alias": r"Tool argument alias applied",
        "path leading slash restored": r"missing its leading slash",
        "Bash timeout s->ms": r"read as seconds ->",
        "re-prompt retries": r"Retry \d/\d: tool intent",
        "retries that succeeded": r"Retry succeeded",
        "tool calls emitted": r"Tool calls: ",
    }
    ldir = os.path.join(RES, "server-logs")
    os.makedirs(ldir, exist_ok=True)
    rows = []
    for (model, setup) in sorted(groups):
        if not setup.startswith("mlx"):
            continue
        src = f"/tmp/ccl-bench/{setup}-{model}.log"
        dst = os.path.join(ldir, f"{setup}-{model}.log.gz")
        if os.path.exists(src) and not os.path.exists(dst):
            with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        if os.path.exists(dst):
            txt = gzip.open(dst, "rt", errors="replace").read()
            rows.append((f"{model} {setup}", {k: len(re.findall(v, txt)) for k, v in pats.items()}))
    if not rows:
        print("\n(No server logs in results/server-logs, so no repair counts here. The published "
              "counts are in results/tables.md.)")
    if rows:
        print("\nclaude-code-local server log, repair events (whole run set, warm-up included):\n")
        print("| Run set | " + " | ".join(pats) + " |")
        print("|---|" + "---|" * len(pats))
        for name, c in rows:
            print(f"| {name} | " + " | ".join(str(c[k]) for k in pats) + " |")
    # Each other setup against claude-code-local as shipped, same model.
    print("\nAgainst claude-code-local (mlx) on the same model: pass-rate difference with a two-sided "
          "Fisher exact p, and the per-task wall-time ratio (median over the 10 tasks of "
          "setup median / mlx median; above 1 = slower than mlx):\n")
    print("| Model | Setup | Passes vs mlx | Fisher p | Wall-time ratio vs mlx |")
    print("|---|---|---|---|---|")
    for (model, setup) in sorted(groups):
        if setup == "mlx" or (model, "mlx") not in groups:
            continue
        a, b = groups[(model, "mlx")], groups[(model, setup)]
        pa, pb = sum(r["passed"] for r in a), sum(r["passed"] for r in b)
        ratios = []
        for t in tasks:
            ta = [r["wall_s"] for r in a if r["task"] == t]
            tb = [r["wall_s"] for r in b if r["task"] == t]
            if ta and tb:
                ratios.append(statistics.median(tb) / statistics.median(ta))
        print(f"| {model} | {setup} | {pb}/{len(b)} vs {pa}/{len(a)} | {fisher(pa, len(a) - pa, pb, len(b) - pb):.2f} | "
              f"{statistics.median(ratios):.2f}x |")
    guard = sum(r["cloud_guard_hits"] for r in summary)
    remote = sorted({s for r in summary for s in r["remote_sockets"]})
    print(f"\nCloud guard CONNECT attempts: {guard}; non-local sockets seen on Claude Code: {remote or 'none'}")
    json.dump({"groups": summary, "meta": metas}, open(os.path.join(RES, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
