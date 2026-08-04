#!/usr/bin/env python3
"""Single-config MLX inference benchmark.

Loads a target model (and optionally a draft model for speculative decoding),
runs a fixed prompt, and reports decode tok/s, prefill tok/s, TTFT, and peak
memory. Emits a single JSON blob to stdout (and optionally a results file).

Usage:
  run_bench.py \\
      --model <path-or-hf-id> \\
      [--draft-model <path-or-hf-id>] \\
      [--num-draft-tokens 4] \\
      [--kv-bits 0|4|8] \\
      [--prefill 8192] \\
      [--max-tokens 256] \\
      [--workload code|prose] \\
      [--label "8bit-baseline"] \\
      [--out results/<name>.json]

Designed to be invoked as a subprocess by sweep.py — each run is a clean
process with no carryover state between configurations.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS


def log(msg):
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def peak_mem_gb():
    """Peak resident memory of this process in GB (Darwin reports bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--num-draft-tokens", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=0,
                    help="0=disabled, else 4 or 8")
    ap.add_argument("--kv-quant-start", type=int, default=256)
    ap.add_argument("--prefill", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--workload", choices=list(PROMPTS.keys()), default="code")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup-tokens", type=int, default=8,
                    help="Discard the first N tokens from decode-rate calc to "
                         "avoid first-token compile/cache overhead")
    args = ap.parse_args()

    label = args.label or f"{Path(args.model).name}"
    log(f"loading target model: {args.model}")
    t_load0 = time.time()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    target_load_s = time.time() - t_load0
    log(f"  loaded in {target_load_s:.1f}s")

    draft_model = None
    draft_load_s = 0.0
    if args.draft_model:
        log(f"loading draft model: {args.draft_model}")
        t_d0 = time.time()
        draft_model_obj, _ = load(args.draft_model)
        mx.eval(draft_model_obj.parameters())
        draft_load_s = time.time() - t_d0
        draft_model = draft_model_obj
        log(f"  loaded in {draft_load_s:.1f}s")

    prompt_text = PROMPTS[args.workload]
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    prompt_token_count = len(prompt_tokens)
    log(f"prompt: {prompt_token_count} tokens, workload={args.workload}")

    gen_kwargs = {
        "prefill_step_size": args.prefill,
        "sampler": make_sampler(temp=0.0),  # greedy = reproducible
    }
    # Spec-decoding manages caches for target+draft internally — only pass an
    # external prompt_cache when NOT spec-decoding, to avoid layer-count
    # mismatches between the target's cache and the draft model.
    if draft_model is None:
        gen_kwargs["prompt_cache"] = make_prompt_cache(model)
    if args.kv_bits:
        gen_kwargs["kv_bits"] = args.kv_bits
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = args.kv_quant_start
    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["num_draft_tokens"] = args.num_draft_tokens

    # ── Generate, measuring carefully ─────────────────────────────────────
    t_start = time.time()
    t_first = None
    output_text = ""
    gen_token_count = 0
    decode_start_token_idx = None
    decode_start_time = None
    finish_reason = "end_turn"

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=args.max_tokens,
        **gen_kwargs,
    ):
        if t_first is None:
            t_first = time.time()
        output_text += response.text
        gen_token_count = response.generation_tokens
        if decode_start_token_idx is None and gen_token_count >= args.warmup_tokens:
            decode_start_token_idx = gen_token_count
            decode_start_time = time.time()
        if response.finish_reason == "length":
            finish_reason = "max_tokens"
        elif response.finish_reason == "stop":
            finish_reason = "end_turn"

    t_end = time.time()
    if t_first is None:
        t_first = t_end

    ttft_s = t_first - t_start
    total_s = t_end - t_start
    decode_only_s = (t_end - decode_start_time) if decode_start_time else (t_end - t_first)
    decode_only_tokens = (gen_token_count - decode_start_token_idx) if decode_start_token_idx else max(gen_token_count - 1, 0)

    decode_tok_s = decode_only_tokens / decode_only_s if decode_only_s > 0 else 0.0
    prefill_tok_s = prompt_token_count / ttft_s if ttft_s > 0 else 0.0
    overall_tok_s = gen_token_count / total_s if total_s > 0 else 0.0

    result = {
        "label": label,
        "model": args.model,
        "draft_model": args.draft_model,
        "num_draft_tokens": args.num_draft_tokens if draft_model else None,
        "kv_bits": args.kv_bits,
        "kv_quant_start": args.kv_quant_start if args.kv_bits else None,
        "prefill": args.prefill,
        "max_tokens": args.max_tokens,
        "workload": args.workload,
        "prompt_tokens": prompt_token_count,
        "gen_tokens": gen_token_count,
        "ttft_s": round(ttft_s, 4),
        "total_s": round(total_s, 4),
        "decode_only_s": round(decode_only_s, 4),
        "decode_only_tokens": decode_only_tokens,
        "decode_tok_s": round(decode_tok_s, 2),
        "prefill_tok_s": round(prefill_tok_s, 2),
        "overall_tok_s": round(overall_tok_s, 2),
        "peak_mem_gb": round(peak_mem_gb(), 2),
        "target_load_s": round(target_load_s, 2),
        "draft_load_s": round(draft_load_s, 2),
        "finish_reason": finish_reason,
        "output_first_120": output_text[:120].replace("\n", " "),
        "output_len_chars": len(output_text),
    }

    out_json = json.dumps(result, indent=2)
    print(out_json)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_json + "\n")
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
