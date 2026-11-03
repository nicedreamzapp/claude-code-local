---
title: Run Claude Code on Windows with a local model
description: >-
  Run Claude Code on a Windows PC with a local model and no API key. One
  PowerShell line detects your GPU, downloads a pinned llama.cpp build and a
  vision model that fits, and puts Claude Local on your Desktop.
---

# Run Claude Code on Windows with a local model

On Windows, Claude Code Local runs the model with [llama.cpp](https://github.com/ggml-org/llama.cpp) and serves it to Claude Code in Anthropic's Messages API format on `http://localhost:4000`, so Claude Code works against a model on your own PC with no API key. One PowerShell line installs it, then you double-click **Claude Local** on your Desktop.

## What you need

- **Git for Windows** ([git-scm.com/download/win](https://git-scm.com/download/win)) and **Python 3** ([python.org/downloads](https://www.python.org/downloads/), check "Add to PATH"). The installer stops if `git` or `python` is not on your PATH.
- **The Claude Code CLI.** The Windows installer does not install it; if `claude` is missing it tells you to install it and launch again. One way is `npm install -g @anthropic-ai/claude-code`.
- Disk space for the model, which can be large.

## 1. Install

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.ps1 | iex
```

It does this, in order:

1. Clones the repo to `%USERPROFILE%\claude-code-local` (set `CCL_DIR` to change it), or updates an existing checkout.
2. Runs `portable\ccl.py detect`, which looks for an NVIDIA card, an AMD or other Vulkan GPU, or falls back to CPU, and prints the model it will use.
3. Runs `portable\ccl.py install`, which downloads the matching llama.cpp build (pinned by version and checksum) and the strongest vision model that fits, each file checked by size and SHA-256.
4. Writes `claude-local.cmd` in the install folder and copies it to your Desktop as **Claude Local**.
5. Sets up `keep going` (`keepgoing.cmd` / `keep.cmd` on your PATH plus a PowerShell profile line).
6. Offers optional add-ons. Press Enter for none.

Some examples of what it picks:

| Your hardware | What you get |
|---|---|
| 24 GB NVIDIA card | Qwen 3.5 9B at full BF16 |
| 12 GB NVIDIA card | Qwen 3.5 4B at full BF16 |
| AMD or Vulkan GPU | the largest that fits in VRAM |
| No GPU (CPU only) | a 4B or 9B vision model, sized to your RAM |

Installer settings, set before the one-liner: `$env:CCL_SKIP_MODEL = "1"` (skip the llama.cpp and model download), `$env:CCL_NO_DESKTOP = "1"` (no Desktop launcher), `$env:CCL_WITH = "none"` (no add-on menu).

## 2. Start Claude Code

Double-click **Claude Local** on your Desktop, or run `%USERPROFILE%\claude-code-local\claude-local.cmd`. It starts `llama-server` on port 8080 and the Anthropic proxy on port 4000, then opens Claude Code pointed at `http://localhost:4000` with the placeholder key `sk-local`.

## 3. Pick a different model

```powershell
$env:CCL_MODEL_ID = "gemma-4-12b"
python -I $env:USERPROFILE\claude-code-local\portable\ccl.py install
```

That downloads the model and makes it the one **Claude Local** starts. The IDs are in [`portable/catalog.json`](https://github.com/nicedreamzapp/claude-code-local/blob/main/portable/catalog.json). Every model in it can read images.

## Hit your Claude limit? Keep going

Type `exit` in Claude Code, then in the same folder, in PowerShell or cmd:

```powershell
keep going
```

The same conversation resumes on the local model. If a new window says `keep going` is not found, run the doctor with `-Fix` (below); `bin\keepgoing.cmd` in the install folder always works by its full path. More on the [usage limit page](usage-limit.md).

## Check and repair

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1          # what is installed and running
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1 -Fix     # put keep going back on PATH, safe repairs
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1 -Report  # privacy-safe block for a bug report
```

Run them from the install folder. `llama-server did not come up on :8080` means another program owns that port; set `CCL_LLAMA_PORT` (and `CCL_PORT` for 4000).

## Uninstall

```powershell
irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/uninstall.ps1 | iex
```

Set `$env:CCL_WITH = "all"` first to remove the add-ons too. Models and user data are kept unless `$env:CCL_PURGE = "1"`.

## Related

- [Install](install.md)
- [Run Claude Code on Linux with a local model](claude-code-linux-local-model.md)
- [Use Claude Code without an API key](claude-code-without-api-key.md)
