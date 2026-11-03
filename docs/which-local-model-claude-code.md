---
title: Which local model should I run with Claude Code?
description: >-
  Which local model to run with Claude Code, by how much memory you have: from
  Qwen 3.5 2B on an 8 GB Mac to Qwen 3.8 27B on 96 GB, plus what the Linux and
  Windows installer picks for your GPU. Measured memory and speed.
faq:
  - q: Which local model should I run on a 16 GB Mac?
    a: >-
      Qwen 3.5 4B at bf16. It reads images, runs with a 16K context and the
      server peaks at 10.6 GB. setup.sh picks it for any Mac with 16 to 31 GB.
  - q: Which local model should I run on an 8 GB Mac?
    a: >-
      Qwen 3.5 2B at 8-bit. It reads images, runs with a 16K context and the
      server peaks at 4.2 GB. The bf16 build was tested first and did not fit.
  - q: Which model is best on a big Mac?
    a: >-
      On 96 GB and up, setup.sh installs Qwen 3.8 27B at 8-bit, which passed the
      Claude Code test (write a file, run it, report) in 38 s. On 64 to 95 GB it
      installs Gemma 4 31B at 4-bit, which ran at 26 tok/s in the Agent-12 runs on an M5 Max.
  - q: What does the Linux and Windows installer pick?
    a: >-
      The strongest vision model that fits your hardware. For example, Qwen 3.5
      9B at full BF16 on a 24 GB NVIDIA card, Qwen 3.5 4B at full BF16 on a 12 GB
      card, the largest that fits in VRAM on an AMD or Vulkan GPU, and a 4B or 9B
      vision model sized to your RAM with no GPU.
  - q: Is a local model as good as Claude?
    a: >-
      No. Big models on a 64 GB or 96 GB Mac handle real multi-step work. Small
      ones on 8 or 16 GB do best with one clear step at a time. It keeps you
      moving until your limit resets.
---

# Which local model should I run with Claude Code?

Start with the one setup picks for you. It reads how much memory your machine has and installs a model that fits, so the short answer is: run the installer and take its pick. This page shows what that pick is for every size, what was measured, and how to change it.

## On a Mac

`setup.sh` reads the Mac's memory with `sysctl hw.memsize` and installs one model:

| Mac memory | Model | Hugging Face ID | Approx. size |
|---|---|---|---|
| 96 GB and up | Qwen 3.8 27B, 8-bit | `lmstudio-community/Qwen3.8-27B-MLX-8bit` | ~29 GB |
| 64 to 95 GB | Gemma 4 31B, 4-bit | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | ~18 GB |
| 32 to 63 GB | Gemma 4 12B, 4-bit | `divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text` | ~11 GB |
| 16 to 31 GB | Qwen 3.5 4B, bf16, reads images | `mlx-community/Qwen3.5-4B-bf16` | ~9 GB |
| Under 16 GB | Qwen 3.5 2B, 8-bit, reads images | `mlx-community/Qwen3.5-2B-MLX-8bit` | ~2.7 GB |

Install it with one command:

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

### What was measured

| Model | What we measured | Good for |
|---|---|---|
| Qwen 3.8 27B, 8-bit | 17.9 tok/s plain decode on an M5 Max. Passed the Claude Code test (write `hello.py`, run it, report) in 38 s | Careful coding and pictures, 96 GB Macs |
| Gemma 4 31B, 4-bit | 26 tok/s in the Agent-12 runs on an M5 Max. Agent-12: 11/12 easy, 8/8 hard | Daily coding and the browser agent |
| Gemma 4 12B, 4-bit | Not on Agent-12 yet | 32 GB Macs |
| Qwen 3.5 4B, bf16 | 22 tok/s on an M4 Pro, server peaks at 10.6 GB, passed read, edit, bash and image tests | 16 GB Macs |
| Qwen 3.5 2B, 8-bit | 92 tok/s on an M4 Pro, server peaks at 4.2 GB, passed read, edit, bash and image tests | 8 GB Macs |

Agent-12 scores come from Agent-12's own lean test harness, not from inside Claude Code. Claude Code sends the model a lot more per turn, so the same model can score and time differently there. The live scores are on the [Agent-12 leaderboard](https://nicedreamzapp.github.io/agent12/), and every table and method is in [docs/BENCHMARKS.md](https://github.com/nicedreamzapp/claude-code-local/blob/main/docs/BENCHMARKS.md).

Hermes 4 14B (4-bit, text only) was the earlier 16 GB pick and still works: a user confirmed it on a 16 GB Mac in [issue #54](https://github.com/nicedreamzapp/claude-code-local/issues/54). The small-Mac models and why they won are on [Best model for a 16 GB or 8 GB Mac](best-local-model-small-mac.md).

### Switching models on a Mac

Download a model once, then name it in `MLX_MODEL`:

```bash
bash scripts/download-and-import.sh gemma   # or qwen, gemma12, hermes, small, or any Hugging Face id
MLX_MODEL=lmstudio-community/Qwen3.8-27B-MLX-8bit bash ~/claude-code-local/launchers/"Claude Local.command"
```

Run that from a terminal. Double-clicking a launcher does not pass your shell's environment variables to it. Only MLX-format models load on a Mac; GGUF files will not.

## On Linux or Windows

Off a Mac there is no MLX, so the model runs on llama.cpp. The installer looks at your hardware (NVIDIA, AMD, or just your CPU), picks the strongest vision model that fits at the highest precision it can, and tells you why:

| Your hardware | What you get |
|---|---|
| 24 GB NVIDIA card | Qwen 3.5 9B at full BF16 |
| 12 GB NVIDIA card | Qwen 3.5 4B at full BF16 |
| AMD or Vulkan GPU | the largest that fits in VRAM |
| No GPU (CPU only) | a 4B or 9B vision model, sized to your RAM |

Every model it can pick reads images. Want a specific one?

```bash
CCL_MODEL_ID=gemma-4-12b python3 ~/claude-code-local/portable/ccl.py install
```

then run `claude-local`. The full catalog is in [`portable/catalog.json`](https://github.com/nicedreamzapp/claude-code-local/blob/main/portable/catalog.json). Setup for each system is on [Claude Code on Linux](claude-code-linux-local-model.md) and [Claude Code on Windows](claude-code-windows-local-model.md).

## Questions people ask

### Which local model should I run on a 16 GB Mac?

Qwen 3.5 4B at bf16. It reads images, runs with a 16K context and the server peaks at 10.6 GB. `setup.sh` picks it for any Mac with 16 to 31 GB.

### Which local model should I run on an 8 GB Mac?

Qwen 3.5 2B at 8-bit. It reads images, runs with a 16K context and the server peaks at 4.2 GB. The bf16 build was tested first and did not fit.

### Which model is best on a big Mac?

On 96 GB and up, `setup.sh` installs Qwen 3.8 27B at 8-bit, which passed the Claude Code test (write a file, run it, report) in 38 s. On 64 to 95 GB it installs Gemma 4 31B at 4-bit, which ran at 26 tok/s in the Agent-12 runs on an M5 Max.

### What does the Linux and Windows installer pick?

The strongest vision model that fits your hardware. For example, Qwen 3.5 9B at full BF16 on a 24 GB NVIDIA card, Qwen 3.5 4B at full BF16 on a 12 GB card, the largest that fits in VRAM on an AMD or Vulkan GPU, and a 4B or 9B vision model sized to your RAM with no GPU.

### Is a local model as good as Claude?

No. Big models on a 64 GB or 96 GB Mac handle real multi-step work. Small ones on 8 or 16 GB do best with one clear step at a time. It keeps you moving until your limit resets.

## Related

- [Claude Code on Apple Silicon with MLX](claude-code-apple-silicon-mlx.md)
- [Best model for a 16 GB or 8 GB Mac](best-local-model-small-mac.md)
- [Hit your usage limit? Keep working on a local model](usage-limit.md)
- [The code on GitHub](https://github.com/nicedreamzapp/claude-code-local)
