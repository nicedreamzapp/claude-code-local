---
title: FAQ - Claude Code with local models on a Mac
description: >-
  Answers about Claude Code Local: whether it is really offline, if it needs an
  API key, which Mac and model to use, Windows and Intel support, Ollama and
  GGUF, MCP servers, images, cost and license.
---

# FAQ

## Does it really run without the cloud?

The model runs on your Mac and the server only listens on `127.0.0.1`. The server code makes no outbound network calls. Claude Code itself contacts Anthropic on startup unless four environment variables are set, and not every launcher sets them; see [Quickstart](quickstart.md#4-check-that-nothing-leaves-the-mac) for which ones do and how to check with `lsof`. Models are downloaded from Hugging Face once, the first time you use them. The one deliberate exception is picking a free cloud model in `keep going`, which sends that conversation to OpenRouter.

## Do I need an Anthropic API key or a Claude subscription?

No. The launchers send `sk-local` as the key and the server never checks it. You do need the Claude Code CLI installed, which is free to install with `npm install -g @anthropic-ai/claude-code`.

## Why does it say `--model claude-sonnet-4-6`?

Claude Code needs a model name it recognizes. The server ignores it when choosing what to run (that is `MLX_MODEL`) and just echoes it back. `GET /v1/models` returns three Claude model names for the same reason.

## Which Mac do I need?

Any Apple Silicon Mac (M1 or newer). Memory decides the model: `setup.sh` gives 16 GB Macs Hermes 4 14B, 32 GB Gemma 4 12B, 64 GB Gemma 4 31B and 96 GB or more Qwen 3.8 27B. Under 16 GB you get Gemma 4 E4B, which the repo reports as unreliable with tools. Run `bash scripts/doctor.sh` to see what fits. The maintainer has asked for reports from 8 GB and 16 GB Macs in [#57](https://github.com/nicedreamzapp/claude-code-local/issues/57).

## Which model should I run?

Start with what `setup.sh` picks. For comparisons on real agent tasks, the maintainer runs the [Agent-12 leaderboard](https://nicedreamzapp.github.io/agent12/). Its scores come from Agent-12's own harness, not from inside Claude Code, so the same model can behave differently here.

## How fast is it?

It depends on the Mac and the model. Numbers in the repo were measured on an M5 Max with 128 GB: for example Qwen 3.8 27B 8-bit at 17.9 tok/s and Gemma 4 31B 4-bit at 26 tok/s. See [docs/BENCHMARKS.md](https://github.com/nicedreamzapp/claude-code-local/blob/main/docs/BENCHMARKS.md). An M4 Pro measurement in the same file has Gemma 4 31B at about 13.5 tok/s. The first reply of a session is the slowest; see [Troubleshooting](troubleshooting.md#the-first-reply-takes-minutes).

## Does it run on Windows, Linux or an Intel Mac?

No. It uses Apple's MLX, which needs Apple Silicon, and the installer stops on anything else ([#53](https://github.com/nicedreamzapp/claude-code-local/issues/53)). The README lists other ways to run Claude Code against models you choose:

- [Ollama](https://docs.ollama.com/api/anthropic-compatibility) speaks the Anthropic Messages API natively since v0.14 and is not limited to Macs.
- [claude-code-router](https://github.com/musistudio/claude-code-router) routes Claude Code across many providers, local or cloud.
- [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) translates Claude Code's requests to OpenAI-style APIs.

## How is this different from Ollama?

Both can serve Claude Code. This project is MLX-only, tuned for Apple Silicon, and adds Claude Code specific handling: a slim prompt in code mode, parsers for the tool-call formats local models actually emit, retries for garbled calls, and prompt-cache reuse. Early versions of this project did use Ollama behind a proxy ([#2](https://github.com/nicedreamzapp/claude-code-local/issues/2)); the repo's benchmark history records that path at 133 s for a Claude Code task in April 2026, against 17.6 s after moving to the MLX server, on the model shipped at the time.

## Can I use GGUF files or a model from LM Studio?

Only MLX-format models. GGUF files will not load. Point `MLX_MODEL` at a Hugging Face ID or a local MLX folder ([#14](https://github.com/nicedreamzapp/claude-code-local/issues/14)).

## Can I use any MLX model?

Any model mlx-lm can load as a text model should start. Whether its tool calls work depends on the format it writes; [How it works](how-it-works.md#6-parsing-tool-calls) lists the formats the parser knows. Vision-language models that need mlx-vlm do not load in this text server.

## Do MCP servers work?

Claude Code's MCP tools are passed to the model like any other tool. Code mode keeps only the built-in Bash, Read, Edit, Write, Grep, Glob and Skill tools, so to use MCP tools in a coding session run the server with `MLX_CODE_MODE=0`. Browser mode keeps only Chrome DevTools tools. `docs/MCP-SERVERS.md` has setup examples.

!!! info "TODO-VERIFY"
    That code mode removes MCP tools is read from the code (`CODE_TOOLS_ALLOW`). Whether that is intended for MCP-heavy setups is not documented in the repo.

## Can it read images?

Not through the server. Image and document blocks inside tool results are replaced with `[image omitted]`, and only text blocks of user messages are passed to the model. The repo's vision work (Muse-Glimmer, Nemotron Omni) runs through other tools such as [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) and [nemotron-omni-mlx](https://github.com/nicedreamzapp/nemotron-omni-mlx).

## What does "abliterated" mean?

Several default models are abliterated builds, with their built-in refusal direction suppressed. The repo states it is not a general capability upgrade and each model's license still applies. Explanation: [Abliteration explained](https://huggingface.co/blog/mlabonne/abliteration) by Maxime Labonne. Original, non-abliterated releases are linked on the [Credits](project-credits.md#models) page.

## Why Python and not C++?

Asked in [#8](https://github.com/nicedreamzapp/claude-code-local/issues/8). The heavy work runs in MLX's Metal kernels; Python handles HTTP, parsing and orchestration around them. TODO-VERIFY: the repo has no measurement of Python overhead.

## Can several people or sessions share one server?

Requests are handled one at a time (single-threaded `HTTPServer` plus a generation lock), and there is one shared prompt cache. Separate sessions work but wait for each other, and switching between unrelated conversations means fresh prefills.

## What does it cost?

The software is free and MIT licensed. Running it costs your Mac's electricity and disk space for models.

## What license is it under?

MIT ([LICENSE](https://github.com/nicedreamzapp/claude-code-local/blob/main/LICENSE), added after [#34](https://github.com/nicedreamzapp/claude-code-local/issues/34)). Models, Claude Code, MLX and the other upstream projects have their own licenses.

## How do I contribute?

Read [CONTRIBUTING.md](https://github.com/nicedreamzapp/claude-code-local/blob/main/CONTRIBUTING.md). The easiest useful contribution is adding your model's raw tool-call output as a test case ([#58](https://github.com/nicedreamzapp/claude-code-local/issues/58)).
