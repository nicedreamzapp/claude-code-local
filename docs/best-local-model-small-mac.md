---
title: Best local model for Claude Code on a 16 GB or 8 GB Mac
description: >-
  Which local model to run with Claude Code on a 16 GB or 8 GB Apple Silicon
  Mac: Qwen 3.5 4B at bf16 for 16 GB, Qwen 3.5 2B at 8-bit for 8 GB. Measured
  memory, speed and what works, with the commands to set it up.
---

# Best local model for Claude Code on a 16 GB or 8 GB Mac

On a 16 GB Mac, run **Qwen 3.5 4B at bf16**. On an 8 GB Mac, run **Qwen 3.5 2B at 8-bit**. Both read images, handle Claude Code's tool calls and fit the memory macOS leaves free, and `setup.sh` picks the right one for you.

## The two picks

| | **16 GB Mac** (16 to 31 GB) | **8 GB Mac** (under 16 GB) |
|---|---|---|
| Model | [Qwen 3.5 4B](https://huggingface.co/mlx-community/Qwen3.5-4B-bf16) | [Qwen 3.5 2B](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-8bit) |
| Precision | bf16 | 8-bit |
| Download | 9.1 GB | 2.7 GB |
| Sees images | yes | yes |
| Context ceiling | 16,384 tokens | 16,384 tokens |
| Server memory, peak | 10.6 GB | 4.2 GB |
| Generation speed (M4 Pro) | 22 tokens/s | 92 tokens/s |
| Read, edit, bash and image tests | all pass | all pass |
| Write a program and run it | right program 5 of 5 | right program 3 of 5 |

macOS and a few apps leave roughly 10 to 11 GB free on a 16 GB Mac and 4 to 5 GB on an 8 GB Mac. Each pick was sized to that budget.

## Why these and not something bigger

The model has to handle vision, code and tool calls together, run at bf16 wherever it fits (8-bit only where bf16 cannot, never 4-bit), and peak inside the budget at a real Claude Code context.

- **16 GB:** the next size up, Qwen 3.5 9B, is 10.4 GB even at 8-bit before any cache, and Gemma 4 E4B is 15.9 GB at bf16.
- **8 GB:** the bf16 build of Qwen 3.5 2B peaked at 5.6 GB on an 8K prompt and 6.0 GB with a full screenshot, past what an 8 GB Mac has free. Qwen 3.5 4B at 8-bit is 5.1 GB of weights alone.

## Set it up

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

`setup.sh` reads your memory, picks the tier, downloads the model and puts **Claude Local.command** on your Desktop. Double-click it. Or with the repo already set up:

```bash
bash scripts/mac-tier.sh                         # shows the tier this Mac gets
bash scripts/download-and-import.sh small        # downloads that model (one time)
open "launchers/Small Mac.command"               # starts the server and Claude Code
```

To force a tier on any Mac: `CCL_TIER=8gb open "launchers/Small Mac.command"` or `CCL_TIER=16gb bash scripts/download-and-import.sh small`.

## What to expect

Both models do one clear step at a time well: read a file, fix a bug with Edit, run a script, describe a screenshot. They get lost on long, many-file jobs, especially after Claude Code compacts the conversation. Keep each request small and use `/clear` between unrelated tasks.

Thinking is on for both. With it off, the 4B got a FizzBuzz wrong in 4 of 5 runs; with it on, it was right in 11 of 11, at about twice the time. Set `MLX_SUPPRESS_THINKING=1` to trade speed for accuracy the other way.

## Older pick: Hermes 4 14B

Hermes 4 14B (4-bit, text only) was the earlier 16 GB pick and still runs:

```bash
MLX_MODEL=divinetribe/Hermes-4-14B-abliterated-4bit-mlx bash scripts/start-mlx-server.sh
```

It is no longer the default because it cannot see images and it is 4-bit.

## How these numbers were measured

On a Mac mini with an M4 Pro and 64 GB, with MLX 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.11 and Claude Code 2.1.306. A process's memory does not depend on how much the Mac has, so the memory figures carry over to 8 and 16 GB Macs; speed will not. On a base M1, M2 or M3 expect it slower. The full method and every test prompt are on the [Small Macs](SMALL-MACS.md) page.

## Related

- [Install](install.md)
- [Hit your usage limit? Keep working on a local model](usage-limit.md)
- [Claude Code with Ollama vs Claude Code Local on small Macs](vs-ollama.md)
