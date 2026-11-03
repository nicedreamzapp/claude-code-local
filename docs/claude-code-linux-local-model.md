---
title: Run Claude Code on Linux with a local model
description: >-
  Run Claude Code on Linux with a local model and no API key. One command
  detects NVIDIA, AMD/Vulkan or CPU, downloads a pinned llama.cpp build and a
  vision model that fits, and starts Claude Code against it.
---

# Run Claude Code on Linux with a local model

On Linux, Claude Code Local runs the model with [llama.cpp](https://github.com/ggml-org/llama.cpp) and puts a small Anthropic Messages API server in front of it on `http://localhost:4000`, so Claude Code works against a model on your own machine with no API key. One command installs everything and you start it with `claude-local`.

## What you need

- `git` and `python3`. The installer stops and tells you if either is missing (on Debian or Ubuntu: `sudo apt-get install -y git python3`).
- Disk space for the model, which can be large.
- The Claude Code CLI. If `claude` is not found, setup installs it with the official `curl -fsSL https://claude.ai/install.sh | bash`.

## 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

On Linux, `install.sh` clones the repo to `~/claude-code-local` (set `CCL_DIR` to change it) and runs `portable/setup-linux.sh`, which:

1. Detects your hardware: an NVIDIA card (`nvidia-smi`), an AMD or other Vulkan GPU (`vulkaninfo`), or CPU only.
2. Downloads the matching llama.cpp release, pinned by version and checksum in `portable/llamacpp.json`.
3. Picks the strongest vision model that fits your memory at the highest precision it can, and downloads it from Hugging Face. Every file is checked by size and SHA-256 from `portable/catalog.json`.
4. Writes the `claude-local` launcher to `~/.local/bin`, a desktop entry named **Claude Code Local**, and the `keep going` command.

Some examples of what it picks:

| Your hardware | What you get |
|---|---|
| 24 GB NVIDIA card | Qwen 3.5 9B at full BF16 |
| 12 GB NVIDIA card | Qwen 3.5 4B at full BF16 |
| AMD or Vulkan GPU | the largest that fits in VRAM |
| No GPU (CPU only) | a 4B or 9B vision model, sized to your RAM |

To see the pick before downloading anything:

```bash
python3 ~/claude-code-local/portable/ccl.py detect
```

## 2. Start Claude Code

```bash
claude-local
```

This starts `llama-server` on port 8080, starts the Anthropic proxy on port 4000, waits for both to answer `/health`, and opens Claude Code pointed at `http://localhost:4000` with the placeholder key `sk-local`.

If `claude-local` is not found, add `~/.local/bin` to your PATH:

```bash
export PATH=$HOME/.local/bin:$PATH
```

## 3. Pick a different model

```bash
CCL_MODEL_ID=gemma-4-12b python3 ~/claude-code-local/portable/ccl.py install
claude-local
```

`install` downloads that model and writes it into the config `claude-local` reads. The IDs are in [`portable/catalog.json`](https://github.com/nicedreamzapp/claude-code-local/blob/main/portable/catalog.json). Every model in it can read images, so screenshots you paste into Claude Code work.

## Hit your Claude limit? Keep going

When your Claude plan runs out mid-task, type `exit` in Claude Code, then in the same folder:

```bash
keep going
```

The same conversation resumes on the local model. Open a new terminal after installing so the command is on your PATH. More on the [usage limit page](usage-limit.md).

## If something does not start

- `llama-server can't start, these libraries are missing`: install GNU OpenMP (Debian or Ubuntu: `sudo apt-get install -y libgomp1`), then run `claude-local` again. Stock Ubuntu 24.04 does not ship it.
- `llama-server did not come up on :8080`: another program owns that port. Set `CCL_LLAMA_PORT` (and `CCL_PORT` for 4000).
- `bash scripts/doctor.sh` checks the model server, the model files, the llama.cpp binary and `keep going`. `--fix` repairs what it safely can.

More in [Troubleshooting](troubleshooting.md).

## Related

- [Install](install.md)
- [Run Claude Code on Windows with a local model](claude-code-windows-local-model.md)
- [Use Claude Code without an API key](claude-code-without-api-key.md)
