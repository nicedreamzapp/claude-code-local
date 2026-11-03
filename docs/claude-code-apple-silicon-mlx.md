---
title: Run Claude Code on an Apple Silicon Mac with MLX
description: >-
  Run Claude Code on an M1 or newer Mac with a local model on Apple's MLX. One
  command installs it, setup picks a model for your memory, and the server
  answers Claude Code on localhost with no API key.
faq:
  - q: Does it work on an Intel Mac?
    a: >-
      No. The Mac path uses Apple's MLX, which needs Apple Silicon (M1 or
      newer), and the installer stops on an Intel Mac.
  - q: Can I use GGUF files or a model from LM Studio?
    a: >-
      Only MLX-format models. GGUF files will not load. Point MLX_MODEL at a
      Hugging Face ID or a local MLX folder.
  - q: How fast is it?
    a: >-
      It depends on the Mac and the model. Measured on an M5 Max with 128 GB,
      Qwen 3.8 27B 8-bit runs at 17.9 tok/s and Gemma 4 31B 4-bit at 26 tok/s.
      On an M4 Pro, Gemma 4 31B runs at about 13.5 tok/s, Qwen 3.5 4B bf16 at
      22 tok/s and Qwen 3.5 2B 8-bit at 92 tok/s.
  - q: Why not run Ollama or another OpenAI-style server?
    a: >-
      Claude Code speaks Anthropic's Messages API. This server speaks that
      format itself and runs the model in the same process with mlx-lm, so
      there is no translating proxy in the middle. On the same 10 coding tasks
      the repo measured the same pass rate as Ollama and 2.8x faster on Qwen
      3.5 2B.
---

# Run Claude Code on an Apple Silicon Mac with MLX

Claude Code Local runs an open model on your Mac's own chip with Apple's [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm), and answers Claude Code on `http://localhost:4000` in Anthropic's Messages API format. Claude Code reads files, edits code and runs commands as usual. The model doing the thinking lives in the Mac's unified memory, and no API key is needed.

You need a Mac with Apple Silicon (M1 or newer). Python 3.12 and the Claude Code CLI are installed for you if they are missing. Intel Macs are not supported.

## Install with one command

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

Or read the script first:

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

Setup reads your memory with `sysctl hw.memsize`, picks a model that fits, downloads it, installs the server and puts **Claude Local.command** on your Desktop. Double-click it and you are coding locally. Every step is on the [Install](install.md) page.

## The model your Mac gets

| Mac memory | Model `setup.sh` installs | Approx. size |
|---|---|---|
| 96 GB and up | Qwen 3.8 27B, 8-bit | ~29 GB |
| 64 to 95 GB | Gemma 4 31B, 4-bit | ~18 GB |
| 32 to 63 GB | Gemma 4 12B, 4-bit | ~11 GB |
| 16 to 31 GB | Qwen 3.5 4B, bf16, reads images, 16K context | ~9 GB |
| Under 16 GB | Qwen 3.5 2B, 8-bit, reads images, 16K context | ~2.7 GB |

The full list with Hugging Face IDs and how to switch is on [Which local model to run](which-local-model-claude-code.md). On 8 GB and 16 GB Macs, see [Best model for a 16 GB or 8 GB Mac](best-local-model-small-mac.md).

## Why MLX and not a translating proxy

Most local model servers speak the OpenAI API, so a common setup puts a translator between them and Claude Code. This project's server, [`proxy/server.py`](https://github.com/nicedreamzapp/claude-code-local/blob/main/proxy/server.py), speaks Anthropic's format itself and runs the model in the same process with mlx-lm, so a request crosses one local HTTP hop and nothing else.

On top of that it does the things Claude Code needs from a local model. It swaps Claude Code's long system prompt for a short coding prompt, turns each model's own tool-call format into Claude Code tool calls, repairs garbled tool JSON with a retry, and reuses the prompt cache between turns so long sessions don't re-read everything. [How it works](how-it-works.md) walks through each step.

The repo measured it against Ollama on the same 10 coding tasks, 30 runs each: same pass rate, 2.8x faster on Qwen 3.5 2B. The setup and numbers are on [Benchmark vs Ollama](vs-ollama.md).

## Measured speed

These numbers come from the repository's own docs. Your Mac will differ.

| Mac | Model | Speed |
|---|---|---|
| M5 Max, 128 GB | Gemma 4 31B, 4-bit | 26 tok/s |
| M5 Max, 128 GB | Qwen 3.8 27B, 8-bit | 17.9 tok/s, passed the Claude Code smoke test in 38 s |
| M4 Pro, 64 GB | Gemma 4 31B | about 13.5 tok/s |
| M4 Pro | Qwen 3.5 4B, bf16 | 22 tok/s |
| M4 Pro | Qwen 3.5 2B, 8-bit | 92 tok/s |

The first reply of a session is the slowest, because two costs stack up: loading the weights into memory, then reading Claude Code's prompt. Keep the server running and later turns reuse the cache (Gemma's rotating cache is the exception, so Gemma reads each request in full). [Troubleshooting](troubleshooting.md#the-first-reply-takes-minutes) has more. The Native Engine in [`agent/agent.py`](https://github.com/nicedreamzapp/claude-code-local/blob/main/agent/agent.py) goes further: on a 4.5k token conversation with Gemma 4 31B 4-bit its time to first token dropped from 6.5 to 7.2 s down to 0.36 s.

## Memory settings worth knowing

The server reads the prompt in chunks of `MLX_PREFILL_SIZE` tokens, 512 by default. A comment in the code records the reason: on a 21.4k token prompt with Gemma 4 31B 4-bit, MLX peaked at 34.2 GB with 8192 token chunks and 20.9 GB with 1024. `MLX_CACHE_LIMIT_GB` (default 3) caps MLX's buffer pool. All the settings are on the [Configuration](configuration.md) page.

## Questions people ask

### Does it work on an Intel Mac?

No. The Mac path uses Apple's MLX, which needs Apple Silicon (M1 or newer), and the installer stops on an Intel Mac.

### Can I use GGUF files or a model from LM Studio?

Only MLX-format models. GGUF files will not load. Point `MLX_MODEL` at a Hugging Face ID or a local MLX folder.

### How fast is it?

It depends on the Mac and the model. Measured on an M5 Max with 128 GB, Qwen 3.8 27B 8-bit runs at 17.9 tok/s and Gemma 4 31B 4-bit at 26 tok/s. On an M4 Pro, Gemma 4 31B runs at about 13.5 tok/s, Qwen 3.5 4B bf16 at 22 tok/s and Qwen 3.5 2B 8-bit at 92 tok/s.

### Why not run Ollama or another OpenAI-style server?

Claude Code speaks Anthropic's Messages API. This server speaks that format itself and runs the model in the same process with mlx-lm, so there is no translating proxy in the middle. On the same 10 coding tasks the repo measured the same pass rate as Ollama and 2.8x faster on Qwen 3.5 2B.

## Related

- [Run Claude Code offline on a Mac](claude-code-offline-mac.md)
- [Which local model to run](which-local-model-claude-code.md)
- [Hit your usage limit? Keep working on a local model](usage-limit.md)
- [The code on GitHub](https://github.com/nicedreamzapp/claude-code-local)
