---
title: Run Claude Code offline on a Mac
description: >-
  How to run Claude Code offline on an Apple Silicon Mac: install once, download
  a local MLX model, switch off Claude Code's background traffic and check with
  lsof that the only connection is localhost.
---

# Run Claude Code offline on a Mac

Yes, you can. Claude Code Local runs an open model on your Mac's own chip with Apple's MLX and answers Claude Code on `http://localhost:4000`, so once the model is downloaded you can work with no internet connection and no Anthropic account.

You need an Apple Silicon Mac (M1 or newer). Intel Macs are not supported.

## 1. Install while you are still online

The install and the model download are the only steps that need the internet.

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

`setup.sh` (which the installer runs) reads your memory, picks a model that fits, downloads it into the Hugging Face cache and puts **Claude Local.command** on your Desktop. It also installs the Claude Code CLI if `claude` is not found. Full details are on the [Install](install.md) page.

| Mac memory | Model `setup.sh` picks |
|---|---|
| 96 GB and up | Qwen 3.8 27B, 8-bit |
| 64 to 95 GB | Gemma 4 31B, 4-bit |
| 32 to 63 GB | Gemma 4 12B, 4-bit |
| 16 to 31 GB | Qwen 3.5 4B, bf16 |
| Under 16 GB | Qwen 3.5 2B, 8-bit |

If you plan to switch models offline, download each one first:

```bash
bash scripts/download-and-import.sh gemma   # or qwen, gemma12, hermes, small, or any Hugging Face id
```

## 2. Switch off Claude Code's own background traffic

The model server makes no outbound calls and only listens on `127.0.0.1`. Claude Code itself still contacts Anthropic at startup unless these are set. Not every launcher sets them, so export them in the terminal you start from:

```bash
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 \
       CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1 CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
```

## 3. Start Claude Code against the local model

Double-click **Claude Local** on your Desktop, or start it by hand:

```bash
# Terminal 1: the model server (default Gemma 4 31B 4-bit, or name the model you downloaded)
bash ~/claude-code-local/scripts/start-mlx-server.sh divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text

# Terminal 2: Claude Code, in your project folder
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

`sk-local` is a placeholder the server never checks, and the model name is only a label Claude Code needs. The [Quickstart](quickstart.md) explains both.

## 4. Check that nothing leaves the Mac

During a session:

```bash
lsof -a -i -p "$(pgrep -d, -f claude)"
```

This lists the network connections of every Claude process. The only one should be `localhost:4000`.

## What still needs the internet

- The first download of each model.
- Picking a free cloud model in `keep going`, which sends that conversation to OpenRouter. The local choices in the same menu stay on your Mac.

## Related

- [Hit your usage limit? Keep working on a local model](usage-limit.md)
- [Best local model for a 16 GB or 8 GB Mac](best-local-model-small-mac.md)
- [Use Claude Code without an API key](claude-code-without-api-key.md)
