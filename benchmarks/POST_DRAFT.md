# Llama 3.3 70B on Apple Silicon: an honest before/after

_M5 Max MacBook Pro, 128 GB unified memory, mlx-lm 0.31.2_

A tweet went around recently claiming a model "self-optimized" itself from 2.3 to 84.3 tok/s in 26 hours — a 37× kicker. The framing was overcooked. What actually happened was an agent ran an automated grid-search over inference flags, starting from a worst-case CPU-only config that left the GPU idle. The real lesson wasn't "the model became smarter." It was "ad-hoc tuning leaves throughput on the table, and you can let an agent do the legwork."

So I ran the honest version of that experiment on my own machine. One model, one harness, before-and-after numbers anyone can re-run. No 37× kicker — and as it turns out, no kicker is needed.

## Setup

| | |
|---|---|
| **Hardware** | M5 Max MacBook Pro, 128 GB unified memory |
| **Target model** | `Llama-3.3-70B-Instruct-abliterated` (8-bit MLX, ~70 GB on disk) |
| **Inference engine** | mlx-lm 0.31.2 |
| **Sampling** | Greedy (temperature 0) for reproducibility |
| **Prompt** | 152 tokens (synthetic, fixed across runs) |
| **Generation** | 256 tokens cap |

The harness is two files: `run_bench.py` (one config = one subprocess, emits JSON) and `sweep.py` (iterates configs, calls run_bench fresh each time, aggregates). Each config gets its own process — no carryover state between runs. ~250 lines of Python total.

## Baseline

```
8-bit (current production launcher):
  decode  7.05 tok/s  ·  TTFT 6.56 s  ·  peak 42 GB
```

That's not a bottleneck — it's a ceiling. M5 Max has roughly 500 GB/s memory bandwidth. An 8-bit 70B model needs to read ~70 GB per generated token. Hard upper bound around 7 tok/s, regardless of software. So we start at the physics limit. To get faster, the model has to either read fewer bytes per token (quantization) or generate more than one token per pass (speculative decoding).

## What I tried

Three orthogonal levers, then combinations:

| Lever | What it changes | Quality risk |
|---|---|---|
| **4-bit quantization** | Halves bytes-read-per-token | Real but small on reasoning |
| **Speculative decoding** | Small "draft" model proposes tokens; target verifies | None — bit-exact identical output (rejection sampling) |
| **KV-cache quantization** | Compresses past-token state | Minor at 8-bit, real at 4-bit |

The interesting one is speculative decoding. The math says: when the draft model guesses right, you get multiple tokens per target forward pass. When it guesses wrong, you fall back to one. Output distribution is provably identical to the target alone — same tokens, same order, no quality cost.

I used Llama 3.2 1B and 3B (4-bit MLX) as drafts; both share the Llama-3 tokenizer with the 70B target.

## Results

Two workloads — `code` (high entropy structure, where draft acceptance is high) and `prose` (open-ended, lower acceptance).

### Code workload (Python function generation)

| Config | Decode tok/s | TTFT (s) | Peak GB | vs 8-bit |
|---|---:|---:|---:|---:|
| `8bit-baseline` | 7.05 | 6.56 | 42.2 | 1.00× |
| `8bit+kv8` | 7.13 | 6.43 | 47.2 | 1.01× |
| `4bit-baseline` | **13.40** | **1.03** | 37.6 | **1.90×** |
| `4bit+kv8` | 12.90 | 0.97 | 37.6 | 1.83× |
| `4bit+spec3B` | 21.60 | 0.91 | 39.5 | 3.06× |
| `4bit+spec1B+kv8` | 22.60 | 0.91 | 38.4 | 3.21× |
| **`4bit+spec1B`** | **24.00** | **0.87** | 38.4 | **3.40×** |

### Prose workload (free-form description)

| Config | Decode tok/s | TTFT (s) | Peak GB | vs 8-bit |
|---|---:|---:|---:|---:|
| `8bit-baseline` | 7.17 | 6.53 | 46.2 | 1.00× |
| `8bit+kv8` | 6.39 | 5.53 | 46.2 | 0.89× |
| `4bit-baseline` | 13.30 | 0.82 | 37.6 | 1.86× |
| `4bit+spec1B` | 13.80 | 0.71 | 38.4 | 1.93× |
| **`4bit+spec3B`** | **14.40** | **0.70** | 39.5 | **2.01×** |
| `4bit+spec1B+kv8` | 12.40 | 0.67 | 38.4 | 1.73× |

## What that means

**4-bit quantization alone is the biggest win, and it's nearly free.** From 7 tok/s to 13.4 tok/s, with TTFT dropping from 6.5 seconds to 1 second — that's a 1.9× decode speedup and a 6× faster first token. Memory drops from 42 GB to 38 GB. Quality cost on this abliterated model: small but real. On the code prompt, 8-bit imported `Dict` from typing while 4-bit reached for `datetime` — both valid; different stylistic choice. On the prose prompt, 8-bit followed the third-person "describe a day" framing literally; 4-bit slipped into first-person narration. Neither is wrong; the 4-bit drifts further from the prompt's exact framing. For reasoning-heavy work, validate yourself.

**Speculative decoding is provably bit-exact, and the data confirms it.** Every 4-bit-plus-spec variant in the table — `4bit+spec1B`, `4bit+spec3B`, `4bit+spec1B+kv8` — produced **byte-identical output** to the `4bit-baseline` on both workloads. Same first 100 chars, same total length. That's the rejection-sampling guarantee in action: the draft proposes, the target verifies, only target-sampled tokens survive. You pay nothing in quality for the speed gain.

**Speculative decoding works on 4-bit, but not on 8-bit.** The first attempts paired the 8-bit Llama 70B with a Llama 3.2 1B draft and threw `IndexError: list index out of range` deep in mlx-lm's attention mask builder, then a Metal GPU Timeout after a workaround. Same error from the official `mlx_lm.generate` CLI — not a harness bug. There's an mlx-lm 0.31.2 path that doesn't reconcile flash-attention layer assumptions between Llama 3.2 and 3.3 when the target is 8-bit. The same combination on the **4-bit** target ran cleanly and topped out at **24 tok/s on code (3.4× over 8-bit baseline)**.

**Draft-model size matters and depends on workload.** On structured code, the 1B draft beats the 3B draft (24.0 vs 21.6) — even with lower acceptance, the smaller draft is fast enough that the math wins. On prose, the 3B draft wins (14.4 vs 13.8) — when acceptance is lower per token, you want a smarter guesser. Real takeaway: pick the draft based on what you actually do.

**KV-cache quantization barely moved the needle here.** It's a long-context optimization. At 152 prompt + 256 generated tokens, there isn't enough KV state to compress meaningfully. For real Claude Code sessions with 5K–20K-token system prompts, it would matter. Worth keeping on the table for that case — but not the headline.

## Why the production launcher ships 4-bit only (not 4-bit+spec)

The 24 tok/s number is real, but there's a catch the benchmark hides. To make spec decoding work, my harness has to **not** pass an external `prompt_cache` to `stream_generate` (the cache layer counts mismatch between target and draft). That's fine for a fresh-prompt benchmark. It's not fine for a real Claude Code conversation, where the same 10K-token system prompt gets re-sent on every turn. Without prompt-cache reuse, the prefill happens from scratch every turn — at ~150 tok/s prefill on the 4-bit, that's **a 67-second TTFT on every message**. Unacceptable.

So:
- **The new `Llama 70B Fast.command` ships 4-bit, no spec decoding.** Prompt-cache reuse keeps TTFT under a second on warm sessions, decode is 13.4 tok/s, total UX is roughly 2× the old launcher.
- **The 24 tok/s spec-decoding config stays in the harness** as a documented lever for one-shot, fresh-prompt use cases — and as something to revisit when mlx-lm fixes the 8-bit + spec compatibility (bit-exact, so it'd be a free win).

## What this isn't

This isn't 37×. It isn't a "self-optimizing model." The model never changed — only how it was run. The agent in that tweet started from a configuration nobody competent would ship to production (CPU-only, highest quant, no GPU offload), so most of the speedup was just "stop doing the dumbest possible thing." The remaining 2–3× — the part that's genuinely interesting — is what a sweep harness can find for any of us in roughly **15 minutes of compute time**, not 26 hours. Most of that 15 minutes was a 40 GB download.

The pattern worth borrowing isn't "let an agent run for a day." It's: define the search space, write a benchmark that measures one config cleanly, then sweep. That's it. Measurement plus iteration beats vibes.

## Reproducible

Every run lives in `benchmarks/`. One config = one subprocess = one JSON file.

```bash
# Single run
python3 run_bench.py \
  --model frscrcc/Llama-3.3-70B-Instruct-abliterated-mlx-4Bit \
  --workload code --max-tokens 256

# 4-bit + speculative decoding (the 24 tok/s number)
python3 run_bench.py \
  --model frscrcc/Llama-3.3-70B-Instruct-abliterated-mlx-4Bit \
  --draft-model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --num-draft-tokens 4 --workload code --max-tokens 256

# Full sweep
python3 sweep.py --max-tokens 256 --workloads code,prose
```

Output of every run lands in `results/<label>--<workload>.json`. `sweep.py` writes a combined `summary.md`.

## Caveats I won't pretend away

- These numbers are decode-only on this specific machine. M3 Max users will see different curves; M2 Ultra different again.
- The 152-token prompt is a deliberately short benchmark, not representative of long Claude Code sessions. KV-quant matters more as context grows.
- Greedy sampling means deterministic numbers, not real-session conditions. Real sessions use temperature > 0.
- Output coherence checked by eye on a small set, not via a benchmark suite. If your work is reasoning-heavy, validate the 4-bit yourself before using it for that.
- The spec-decoding gap between code and prose (3.4× vs 2.0× over baseline) is workload-shape-dependent. Your acceptance rates will vary.

## Bottom line

| | tok/s | TTFT | RAM | Notes |
|---|---:|---:|---:|---|
| Old launcher (8-bit) | 7.0 | 6.5s | 42 GB | The ceiling for this quant |
| New launcher (4-bit) | **13.4** | **1.0s** | 38 GB | Ships now as `Llama 70B Fast.command` |
| Theoretical max (4-bit + spec) | 24.0 | 0.9s | 38 GB | 3.4× — gated on prompt-cache work |

Two quiet weekend hours, one harness, no agent overnight. The model never changed.

---
_Hardware: M5 Max MacBook Pro, 128 GB unified memory. Software: mlx-lm 0.31.2._

_Methodology, harness, and raw JSON results in the repo. Discussion welcome — especially if you've found a way to combine prompt-cache reuse with speculative decoding in mlx-lm 0.31.x._
