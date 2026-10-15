---
title: Credits - upstream projects, models and contributors
description: >-
  Claude Code Local is built on Anthropic's Claude Code, Apple's MLX and
  mlx-lm, and open models from Google, Qwen, Nous Research and others. Credits
  for upstream projects, model builds and every contributor.
---

# Credits

This repository is glue: a server, launchers, an agent and installers. The hard parts underneath were built by other people.

## Upstream software

| Project | By | Used for |
|---|---|---|
| [Claude Code](https://github.com/anthropics/claude-code) | Anthropic | The coding agent you run. Not modified; this project only points it at a local server. |
| [MLX](https://github.com/ml-explore/mlx) | Apple | Array framework on Apple Silicon's GPU and unified memory |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple | Model loading, chat templates, KV caches and `stream_generate`, used by the server and the Native Engine |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | Prince Canuma (Blaizzy) and contributors | Runs the vision builds (Muse-Glimmer) mentioned in the docs; not used by the server |
| [mlx-dspark](https://github.com/ARahim3/mlx-dspark) | ARahim3 | Speculative decoding engine used for the Qwen 3.8 DFlash 2 measurements |
| [Hugging Face Hub](https://huggingface.co) | Hugging Face | Where models are downloaded from |
| [OpenRouter](https://openrouter.ai) | OpenRouter | Free cloud models offered by `keep going` |
| [MkDocs](https://www.mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | Tom Christie and contributors; Martin Donath | This documentation site |

The repo's own credits name the people behind MLX and mlx-lm by commit count: Awni Hannun, Cheng, Angelos Katharopoulos, Ayaan Gazali, Jagrit Digani, Gökdeniz Gülmez, Anchen and everyone else who has contributed.

## Models

Original releases:

| Model | By |
|---|---|
| [Gemma 4](https://huggingface.co/google/gemma-4-31B-it) (31B, [12B](https://huggingface.co/google/gemma-4-12B-it), [E4B](https://huggingface.co/google/gemma-4-E4B-it)) | Google DeepMind |
| [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) and Qwen 2.5 / Qwen3-Coder | Alibaba Qwen team |
| [Hermes 4 14B](https://huggingface.co/NousResearch/Hermes-4-14B) | Nous Research |
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | Meta |
| [Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) | NVIDIA |
| [DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | Inco AI / Z Lab |

MLX conversions installed by the scripts come from [mlx-community](https://huggingface.co/mlx-community), [lmstudio-community](https://huggingface.co/lmstudio-community) and the maintainer's own uploads at [huggingface.co/divinetribe](https://huggingface.co/divinetribe). The abliterated builds start from abliterations by [huihui-ai](https://huggingface.co/huihui-ai), [null-space](https://huggingface.co/null-space), [OpenYourMind](https://huggingface.co/OpenYourMind) and [Babsie](https://huggingface.co/Babsie). The technique is explained by Maxime Labonne in [Abliteration explained](https://huggingface.co/blog/mlabonne/abliteration). Each model's own license applies.

## Contributors to this repository

| Who | What they fixed |
|---|---|
| [@0xshugo](https://github.com/0xshugo) | Client disconnects handled, retries skipped when there are no tools ([#4](https://github.com/nicedreamzapp/claude-code-local/pull/4)); diagnosed the wrong-model problem behind the launchers' model-aware restart ([#5](https://github.com/nicedreamzapp/claude-code-local/pull/5)) |
| [@asdmoment](https://github.com/asdmoment) | Gemma inference crash: auto-disable KV quantization ([#7](https://github.com/nicedreamzapp/claude-code-local/pull/7)) |
| [@kulveersingh](https://github.com/kulveersingh) | `ArraysCache` has no attribute `offset` ([#10](https://github.com/nicedreamzapp/claude-code-local/pull/10)) |
| [@tripathiprateek](https://github.com/tripathiprateek) | `uninstall.sh` ([#23](https://github.com/nicedreamzapp/claude-code-local/pull/23)) |
| [@tadrianonet](https://github.com/tadrianonet) | 16 GB Mac support, ChatML stop markers, the `<tools>` parser, server-sent events for tools and the offline launcher settings ([#32](https://github.com/nicedreamzapp/claude-code-local/pull/32)) |
| [@kevbarns](https://github.com/kevbarns) | Gemma 4 thinking suppression and slimmer tool descriptions ([#33](https://github.com/nicedreamzapp/claude-code-local/pull/33)) |
| [@KaoCSC](https://github.com/KaoCSC) | Real end-of-sequence stop and empty env values ([#41](https://github.com/nicedreamzapp/claude-code-local/pull/41)); bare JSON tool calls ([#43](https://github.com/nicedreamzapp/claude-code-local/pull/43)); the prompt-cache and KV-quantization reports ([#42](https://github.com/nicedreamzapp/claude-code-local/issues/42), [#46](https://github.com/nicedreamzapp/claude-code-local/issues/46)) |
| [@zwolf25](https://github.com/zwolf25) | Slash commands in code mode and array-shaped fenced tool calls, with a parser test ([#56](https://github.com/nicedreamzapp/claude-code-local/pull/56)) |

The server, launchers, Native Engine, `keep going` and install scripts are maintained by [Matt Macosko](https://github.com/nicedreamzapp). Bug reports and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/nicedreamzapp/claude-code-local/blob/main/CONTRIBUTING.md).
