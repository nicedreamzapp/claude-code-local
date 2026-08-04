#!/usr/bin/env python3
"""Sweep harness — iterate over (model, draft, kv_bits, ...) configs by
spawning run_bench.py once per config in a clean subprocess.

Each subprocess loads + tears down the model itself, so there's no carryover
between runs. Output: a JSON list of all run results, plus a markdown table.

Usage:
  sweep.py [--max-tokens 256] [--workloads code,prose]
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PYTHON = "/Users/dtribe/.local/mlx-server/bin/python3"
RUN_BENCH = HERE / "run_bench.py"


# ── Configurations to sweep ───────────────────────────────────────────────
# Each config is one row in the final table. Add/remove freely.

# Llama 70B variants
TARGET_8BIT = "/Users/dtribe/.cache/huggingface/hub/Llama-3.3-70B-Instruct-abliterated-8bit-mlx"
TARGET_4BIT = "frscrcc/Llama-3.3-70B-Instruct-abliterated-mlx-4Bit"

# Draft candidates (smaller = faster guess, lower acceptance)
DRAFT_1B = "mlx-community/Llama-3.2-1B-Instruct-4bit"
DRAFT_3B = "mlx-community/Llama-3.2-3B-Instruct-4bit"


CONFIGS = [
    # ── Baseline: 8-bit, no spec-decoding, no KV-quant ────────────────────
    dict(label="8bit-baseline",       model=TARGET_8BIT, draft=None,    kv_bits=0),

    # ── Spec decoding on 8-bit (bit-exact safe) ───────────────────────────
    dict(label="8bit+spec1B-k4",      model=TARGET_8BIT, draft=DRAFT_1B, kv_bits=0, ndt=4),
    dict(label="8bit+spec3B-k4",      model=TARGET_8BIT, draft=DRAFT_3B, kv_bits=0, ndt=4),

    # ── KV-quant on 8-bit (Llama 3.3 supports it; minor quality risk at 4) ─
    dict(label="8bit+kv8",            model=TARGET_8BIT, draft=None,    kv_bits=8),

    # ── 4-bit, no spec ────────────────────────────────────────────────────
    dict(label="4bit-baseline",       model=TARGET_4BIT, draft=None,    kv_bits=0),

    # ── 4-bit + spec decoding (combo) ─────────────────────────────────────
    dict(label="4bit+spec1B-k4",      model=TARGET_4BIT, draft=DRAFT_1B, kv_bits=0, ndt=4),
    dict(label="4bit+spec3B-k4",      model=TARGET_4BIT, draft=DRAFT_3B, kv_bits=0, ndt=4),

    # ── 4-bit + spec + KV-quant (full stack) ──────────────────────────────
    dict(label="4bit+spec1B-k4+kv8",  model=TARGET_4BIT, draft=DRAFT_1B, kv_bits=8, ndt=4),
]


def run_one(cfg, workload, max_tokens):
    out_path = RESULTS_DIR / f"{cfg['label']}--{workload}.json"
    cmd = [
        PYTHON, str(RUN_BENCH),
        "--model", cfg["model"],
        "--workload", workload,
        "--max-tokens", str(max_tokens),
        "--label", cfg["label"],
        "--out", str(out_path),
    ]
    if cfg.get("draft"):
        cmd += ["--draft-model", cfg["draft"]]
    if cfg.get("ndt"):
        cmd += ["--num-draft-tokens", str(cfg["ndt"])]
    if cfg.get("kv_bits"):
        cmd += ["--kv-bits", str(cfg["kv_bits"])]

    print(f"\n>>> {cfg['label']} · {workload}", flush=True)
    print(f"    {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAILED in {elapsed:.1f}s. stderr tail:")
        print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-12:]))
        return {"label": cfg["label"], "workload": workload, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown"}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"    Non-JSON stdout, stderr tail:")
        print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-8:]))
        return {"label": cfg["label"], "workload": workload, "error": "json decode failed"}
    print(f"    {result['decode_tok_s']:.1f} tok/s decode · "
          f"{result['ttft_s']:.2f}s TTFT · "
          f"{result['peak_mem_gb']:.1f} GB · "
          f"{elapsed:.0f}s wall")
    return result


def render_table(rows, workload):
    out = [f"\n## Workload: `{workload}`\n"]
    out.append("| label | decode tok/s | prefill tok/s | TTFT (s) | peak GB | gen tok | finish |")
    out.append("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if "error" in r:
            out.append(f"| {r['label']} | — | — | — | — | — | ERROR: {r['error'][:40]} |")
            continue
        out.append(
            f"| {r['label']} | {r['decode_tok_s']:.1f} | {r['prefill_tok_s']:.0f} | "
            f"{r['ttft_s']:.2f} | {r['peak_mem_gb']:.1f} | "
            f"{r['gen_tokens']} | {r['finish_reason']} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--workloads", default="code,prose")
    ap.add_argument("--only", default=None,
                    help="Comma-separated label prefixes to run (e.g. '8bit-baseline,8bit+spec1B')")
    args = ap.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    configs = [c for c in CONFIGS if (only is None or c["label"] in only)]

    print(f"Sweeping {len(configs)} configs × {len(workloads)} workloads = "
          f"{len(configs) * len(workloads)} runs")

    all_results = []
    for workload in workloads:
        rows = []
        for cfg in configs:
            r = run_one(cfg, workload, args.max_tokens)
            rows.append(r)
            all_results.append(r)
        print(render_table(rows, workload))

    # Persist combined results + table
    summary_json = RESULTS_DIR / "summary.json"
    summary_json.write_text(json.dumps(all_results, indent=2) + "\n")

    summary_md = RESULTS_DIR / "summary.md"
    md_parts = ["# MLX inference sweep — Llama 70B on M5 Max\n",
                f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n"]
    by_workload = {}
    for r in all_results:
        by_workload.setdefault(r.get("workload", "?"), []).append(r)
    for workload, rows in by_workload.items():
        md_parts.append(render_table(rows, workload))
    summary_md.write_text("\n".join(md_parts) + "\n")
    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
