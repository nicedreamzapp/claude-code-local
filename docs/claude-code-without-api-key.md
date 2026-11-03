---
title: Use Claude Code without an API key
description: >-
  Use Claude Code with no Anthropic API key and no Claude subscription by
  pointing it at a free local model. Works on Apple Silicon Macs, Linux and
  Windows. The key sk-local is a placeholder the server never checks.
---

# Use Claude Code without an API key

You can run Claude Code with no Anthropic API key and no Claude subscription by pointing it at a model on your own machine. Claude Code Local answers Claude Code on `http://localhost:4000` in Anthropic's Messages API format, and the launchers pass `sk-local` as the key, which the server never checks.

You still need the Claude Code CLI itself, which is free to install. The model is open and runs on your Mac (with MLX) or your Linux or Windows PC (with llama.cpp).

## 1. Install

Mac (Apple Silicon) or Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.ps1 | iex
```

Setup looks at your hardware and downloads a model that fits. Details: [Install](install.md) (Mac), [Linux](claude-code-linux-local-model.md), [Windows](claude-code-windows-local-model.md).

## 2. Start it

- **Mac:** double-click **Claude Local** on your Desktop.
- **Linux:** run `claude-local`.
- **Windows:** double-click **Claude Local** on your Desktop.

Each one starts the local server and opens Claude Code against it. Nothing asks for a key.

## How it works without a key

Claude Code reads two environment variables to decide where to send requests. The launchers set them for you. By hand it looks like this:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

- `ANTHROPIC_BASE_URL` sends every request to the local server instead of Anthropic.
- `ANTHROPIC_API_KEY=sk-local` fills the key field. The server ignores it.
- `--model claude-sonnet-4-6` is only a name Claude Code recognizes. The server answers with the model it loaded.

Your normal `claude` command is not changed, so it still uses your Claude plan or key when you have one.

## If Claude Code asks you to sign in

Your `claude` is too old. Update it with `npm install -g @anthropic-ai/claude-code`. See [Troubleshooting](troubleshooting.md#claude-code-asks-me-to-log-in).

## Out of credits in the middle of a task?

Type `exit` in Claude Code, then `keep going` in the same folder. The same conversation carries on with a local model. See [Hit your usage limit?](usage-limit.md).

## Is it as good as Claude?

No. Big models on a 64 GB or 96 GB Mac handle real multi-step work. Small ones on an 8 or 16 GB machine do best with one clear step at a time. See [Best local model for a 16 GB or 8 GB Mac](best-local-model-small-mac.md).
