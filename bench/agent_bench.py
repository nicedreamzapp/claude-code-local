#!/usr/bin/env python3
"""Reproduce the native-engine cache numbers on your own machine.

Runs the SAME 6-turn conversation twice against the in-process MLX engine:

  cached — as shipped: the KV cache is trimmed to the shared prefix and only
           the new tokens are prefilled each turn
  naive  — the cache is wiped before every turn, so the full transcript is
           re-prefilled (what an agent without prefix reuse pays; also what
           the engine itself paid on Gemma before the RotatingKVCache fix)

Turn 3 injects a ~4k-token blob to simulate a big file read landing in the
conversation — the moment cache reuse starts to matter.

Usage:
    AGENT_MODEL=<hf-id-or-path> python3 bench/agent_bench.py

Prints per-turn: prompt tokens, delta prefilled, time-to-first-token,
generation tok/s. Results land in bench_results.json next to this file.
"""
import json
import os
import sys
import time

os.environ.setdefault("AGENT_MODEL", "divinetribe/gemma-4-31b-it-abliterated-4bit-mlx")
os.environ.setdefault("AGENT_BACKEND", "mlx")
os.environ.setdefault("AGENT_DIALECT", "prompted")
os.environ.setdefault("AGENT_MAX_TOKENS", "120")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
import agent  # noqa: E402

BLOB = ("def process(item):\n    result = transform(item)\n    "
        "if result.status == 'ok':\n        queue.push(result)\n") * 120

TURNS = [
    "In one sentence, what is a KV cache?",
    "In one sentence, why does quantization make models smaller?",
    "Here is a code file:\n\n" + BLOB + "\n\nIn one sentence, what does it do?",
    "In one sentence, name one risk of the pattern in that file.",
    "In one sentence, how would you add retry logic to it?",
    "In one sentence, summarize this whole conversation.",
]


def run_turn(engine, text, naive):
    if naive:
        engine.cache = engine._new_cache()
        engine.cache_tokens = []
    engine.add_user(text)
    tokens = engine._render()
    overlap = 0
    for a, b in zip(engine.cache_tokens, tokens):
        if a != b:
            break
        overlap += 1
    overlap = min(overlap, len(tokens) - 1)
    t0 = time.time()
    first = [None]

    def on_text(_):
        if first[0] is None:
            first[0] = time.time()

    engine.step(on_text)
    t1 = time.time()
    gen = len(engine.cache_tokens) - len(tokens)
    gtime = t1 - (first[0] or t1)
    return {
        "prompt_toks": len(tokens),
        "delta_prefilled": len(tokens) - overlap,
        "ttft_s": round((first[0] or t1) - t0, 2),
        "gen_toks": gen,
        "gen_tps": round(gen / gtime, 1) if gtime > 0 else None,
        "total_s": round(t1 - t0, 2),
    }


def main():
    agent.acquire_seat()
    t0 = time.time()
    engine = agent.MLXEngine(agent.MODEL_DEFAULT)
    results = {"model": agent.MODEL_DEFAULT, "load_s": round(time.time() - t0, 1),
               "cached": [], "naive": []}
    for mode in ("cached", "naive"):
        engine.reset()
        for text in TURNS:
            r = run_turn(engine, text, naive=(mode == "naive"))
            results[mode].append(r)
            print(f"[{mode}] {r}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out, flush=True)
    agent.release_seat()
    os._exit(0)


if __name__ == "__main__":
    main()
