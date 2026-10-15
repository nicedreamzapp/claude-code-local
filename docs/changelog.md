---
title: Changelog - Claude Code Local releases
description: >-
  Release history of Claude Code Local from git tags v0.1.0, v0.2.0 and v0.3.0
  and the commits since, including prompt-cache, tool-call parser, memory and
  streaming fixes.
---

# Changelog

Built from the git history and the three tags in the repository. Pull request numbers link to GitHub. Full release notes are on the [releases page](https://github.com/nicedreamzapp/claude-code-local/releases).

## Unreleased (on `main` since v0.3.0)

### Server fixes (2026-10-09)

- Think blocks are filtered out of the live stream, and the tool-call retry runs on a fresh KV cache so it no longer lands on top of the first attempt ([#71](https://github.com/nicedreamzapp/claude-code-local/pull/71)).
- `stop_sequences` and `tool_result` `is_error` are honored; tool parsing is skipped for requests without tools ([#70](https://github.com/nicedreamzapp/claude-code-local/pull/70)).
- Errors use Anthropic's error shape; a bad request body gets a 400 instead of a dropped connection ([#69](https://github.com/nicedreamzapp/claude-code-local/pull/69)).
- Arguments sent as a JSON string are decoded; Gemma 4 arguments are parsed as JSON ([#68](https://github.com/nicedreamzapp/claude-code-local/pull/68)).
- Images in `tool_result` are no longer pasted into the prompt as base64 ([#67](https://github.com/nicedreamzapp/claude-code-local/pull/67)).
- A JSON array inside `<tool_call>` tags is handled ([#66](https://github.com/nicedreamzapp/claude-code-local/pull/66)).
- Gemma 4 number and boolean arguments are kept ([#65](https://github.com/nicedreamzapp/claude-code-local/pull/65)).
- Parallel calls to the same tool with different arguments are kept ([#64](https://github.com/nicedreamzapp/claude-code-local/pull/64)).

### Code mode and parser (2026-09-14 to 09-19)

- Code mode keeps the `Skill` tool so slash commands work, and fenced JSON arrays of tool calls (Hermes 4 14B) are parsed, with a standalone parser test, by @zwolf25 ([#56](https://github.com/nicedreamzapp/claude-code-local/pull/56)).

### Features and models

- `keep going`: resume the current folder's Claude Code conversation on a free OpenRouter model or a local one, plus an optional status line reminder (2026-09-17).
- Model lineup refresh: Qwen 3.8 27B (8-bit) for 96 GB Macs and Gemma 4 for 32 and 64 GB, older models retired (2026-09-19).
- `bin/` helpers added (`fia`, `keepgoing`) (2026-09-07).
- `CONTRIBUTING.md` and an "other ways to do this" section in the README (2026-09-19).

### Docs

- README rewritten around a one-line summary and accurate setup and mode caveats (#61), plus fixes to links and the `lsof` privacy check (#60, #62).

## v0.3.0 (2026-08-22)

- **Native Engine** (`agent/agent.py`): an in-process MLX terminal agent with a fixed short prompt and a KV-cache fix for Gemma's rotating cache; 0.36 s turn starts on long sessions as measured in the repo ([#51](https://github.com/nicedreamzapp/claude-code-local/pull/51)), with `bench/agent_bench.py` to reproduce.
- **16 GB Macs** get Hermes 4 14B instead of a 4B model; one-line `install.sh` added ([#49](https://github.com/nicedreamzapp/claude-code-local/pull/49)).
- Real terminal demo GIF in the README ([#50](https://github.com/nicedreamzapp/claude-code-local/pull/50)).
- Qwen 3.8 27B and Muse-Glimmer 30B documented; Agent-12 leaderboard linked from the README.
- `MLX_KV_BITS=4` tool-argument degradation documented ([#42](https://github.com/nicedreamzapp/claude-code-local/issues/42)).

## v0.2.0 (2026-08-05)

- **Prompt cache correctness** ([#46](https://github.com/nicedreamzapp/claude-code-local/issues/46), [#47](https://github.com/nicedreamzapp/claude-code-local/pull/47)): the cache was trimmed using `.step` (a fixed allocation size) instead of `.offset`, so the trim never ran and later sessions were answered from the previous session's cache. The release notes record 1/8 correct before and 8/8 after on one long-lived server. An empty reply after a cache hit now drops the cache.
- Launchers find the `claude` binary instead of hardcoding `~/.local/bin/claude` ([#40](https://github.com/nicedreamzapp/claude-code-local/issues/40), [#48](https://github.com/nicedreamzapp/claude-code-local/pull/48)).
- Gemma 4's collapsed `<tool_call:Name{...}` opener is parsed ([#45](https://github.com/nicedreamzapp/claude-code-local/pull/45)).
- Prefill chunk lowered from 8192 to 512 tokens to cut peak memory ([#44](https://github.com/nicedreamzapp/claude-code-local/pull/44)).
- Bare top-level JSON tool calls are parsed (Qwen 2.5 Coder), by @KaoCSC ([#43](https://github.com/nicedreamzapp/claude-code-local/pull/43)).
- Generation stops on the tokenizer's real end-of-sequence token, and empty numeric environment values fall back to defaults, by @KaoCSC ([#41](https://github.com/nicedreamzapp/claude-code-local/pull/41)).
- Live token streaming for requests without tools ([#39](https://github.com/nicedreamzapp/claude-code-local/issues/39)).
- Gemma 4 thinking suppression and slimmer tool descriptions in code mode, by @kevbarns ([#33](https://github.com/nicedreamzapp/claude-code-local/pull/33)).
- MIT `LICENSE` added.

## v0.1.0 (2026-05-08)

First tagged release, mostly from @tadrianonet's [#32](https://github.com/nicedreamzapp/claude-code-local/pull/32):

- Offline launchers: the four `CLAUDE_CODE_DISABLE_*` / `DISABLE_AUTOUPDATER` variables stop Claude Code's own startup traffic to Anthropic.
- 16 GB Mac support with `Claude Chat.command` and `Claude Agentico.command` (Qwen 2.5 Coder 14B, 4-bit KV cache, login workaround).
- Server-sent events for tool responses, which Claude Code 2.1 requires.
- ChatML and Llama 3 stop markers, the Qwen 2.5 `<tools>` format and fenced JSON tool calls.
- Tool inputs filtered against each tool's schema; the empty text block that made Claude Code drop tool calls removed.
- `uninstall.sh` by @tripathiprateek ([#23](https://github.com/nicedreamzapp/claude-code-local/pull/23)).

Earlier, untagged work (March to May 2026):

- 2026-03-25: initial release, built on llama.cpp ("TurboQuant") behind a proxy.
- 2026-03-26: the MLX server that speaks the Anthropic API directly replaces the proxy.
- 2026-03-29: tool-call reliability work: garbled JSON recovery, retries, lower temperature.
- 2026-04-11: launchers share a library with model-aware restarts and local model folders; the server is installed as a symlink to the repo; Gemma KV quantization crash fixed by @asdmoment ([#7](https://github.com/nicedreamzapp/claude-code-local/pull/7)); client disconnects handled by @0xshugo ([#4](https://github.com/nicedreamzapp/claude-code-local/pull/4)).
- 2026-04-13: `ArraysCache` crash fixed by @kulveersingh ([#10](https://github.com/nicedreamzapp/claude-code-local/pull/10)).
- 2026-04-23: `scripts/doctor.sh` added.
