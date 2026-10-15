---
title: Troubleshooting Claude Code with local MLX models
description: >-
  Fixes for common Claude Code Local problems, taken from real GitHub issues and
  the exact error messages in the code: login prompts, slow first replies,
  "(No output)", out-of-memory crashes, failed downloads and empty tool calls.
---

# Troubleshooting

Start with the server log. Launchers write it to **`/tmp/mlx-server.log`**; if you started the server yourself, it is the terminal it runs in. Every request logs its prompt size, cache hits, tokens per second, parsed tool calls and peak memory.

```bash
tail -f /tmp/mlx-server.log
curl -s http://localhost:4000/health
bash ~/claude-code-local/scripts/doctor.sh
```

## Claude Code asks me to log in

**Symptom:** Claude Code shows "Select login method" or asks for an Anthropic account instead of using the local server. Reported in [#26](https://github.com/nicedreamzapp/claude-code-local/issues/26).

**Fixes, in order:**

1. Update the CLI. The README says this prompt means your `claude` is too old: `npm install -g @anthropic-ai/claude-code`.
2. Check that the `claude` being run is the one you updated: `which claude`.
3. Use `Claude Chat.command` or `Claude Agentico.command`. They set `ANTHROPIC_AUTH_TOKEN=sk-local` and `DISABLE_LOGIN_COMMAND=1` and pass a settings file with `"apiKeyHelper": "echo sk-local"`, the workaround from PR #32 for Claude Code reading the macOS keychain before the environment.
4. `docs/MAC-BASE-SETUP.md` also sets `hasCompletedOnboarding: true` in `~/.claude.json`. In #26 the maintainer suggested adding `"primaryApiKey": "sk-local"` to `~/.claude.json`.

!!! warning
    The #26 suggestion `echo '{"primaryApiKey": "sk-local"}' > ~/.claude.json` **overwrites** the whole file, including your MCP servers and other settings. Edit the file instead of replacing it.

## `ERROR: Claude Code CLI not found.`

The launcher looked on your `PATH` and in `~/.local/bin/claude`, `~/.claude/local/claude`, `/opt/homebrew/bin/claude` and `/usr/local/bin/claude`. Install Claude Code, or point the launcher at it:

```bash
CLAUDE_BIN=/path/to/claude bash launchers/"Gemma 4 Code.command"
```

Older launchers hardcoded `~/.local/bin/claude` and failed with `No such file or directory` ([#40](https://github.com/nicedreamzapp/claude-code-local/issues/40)). Fixed in v0.2.0. The Desktop launcher made by `setup.sh` stores the path it found at install time; re-run `setup.sh` if you move Claude Code.

## The first reply takes minutes

Reported in [#13](https://github.com/nicedreamzapp/claude-code-local/issues/13) (2 m 25 s for "hello" on an M4 Pro 48 GB with Gemma 4 31B) and [#28](https://github.com/nicedreamzapp/claude-code-local/issues/28) (M1 Max 64 GB).

Two costs stack up on the first request: loading the weights into memory, then prefilling Claude Code's prompt. Things that help:

- Leave code mode on (the default). It replaces Claude Code's long prompt and tool descriptions; `docs/BENCHMARKS.md` reports prefill on Gemma 4 31B dropping from about 60 s to about 2 s on an M4 Pro.
- Keep the server running between sessions. Later turns reuse the cache (look for `Cache hit: N reused` in the log). Gemma's rotating cache cannot be reused by the server, so Gemma prefills each request in full.
- Use a smaller model for your memory tier (see [Install](install.md#model-chosen-by-memory)).
- For long sessions, try the [Native Engine](how-it-works.md#the-native-engine), which keeps the cache across turns even for Gemma.

## `ERROR: MLX server failed to respond on port 4000 within 120s`

The launcher started the server but `/health` never answered. Read `/tmp/mlx-server.log`. Common causes:

- The model is still **downloading**. A first download of 5 to 30 GB can take longer than the wait. Download it first with `bash scripts/download-and-import.sh <name>`.
- The model does not fit in memory and the process was killed.
- A bad environment value, for example a non-numeric `MLX_PREFILL_SIZE`, stops the server at startup.

!!! info "TODO-VERIFY"
    The message says 120 s, but the wait in `launchers/lib/claude-local-common.sh` is 180 checks two seconds apart, about six minutes.

## `Different model is loaded (...) — restarting MLX server...`

Not an error. You launched with a different model than the one running, so the launcher stopped the old server and is starting yours. If it also prints `Warning: existing MLX server didn't exit cleanly`, something else may still hold port 4000: check with `lsof -i :4000`.

## Claude Code shows `(No output)` or empty replies

Possible causes, from the code and from closed issues:

- **Cache state from an earlier session** ([#46](https://github.com/nicedreamzapp/claude-code-local/issues/46)). Before v0.2.0, a long-lived server answered later sessions from the previous session's cache. Update. The server now logs `WARNING: empty completion after a N-token cache hit — discarding prompt cache` and recovers on the next request. Restarting the server always clears it.
- **Small models and extended thinking.** The `Claude Chat` and `Claude Agentico` launchers pass `--effort low` because, per their comments, small quantized models spend their budget thinking and return empty replies on the second call.
- **Stop token mismatch.** Fixed in v0.2.0 ([#41](https://github.com/nicedreamzapp/claude-code-local/pull/41)): some models never stopped and every request ran to `max_tokens`.

## Python crashes or the Mac swaps

Reported in [#20](https://github.com/nicedreamzapp/claude-code-local/issues/20): a 122B model on a 96 GB Mac grew to 72 GB and crashed, and Claude Code showed `API Error: Unable to connect to API (ConnectionRefused)`. The model was too big for the machine.

- Pick a model from your [memory tier](install.md#model-chosen-by-memory).
- Watch `mx_peak_gb` from `/health` or the `Memory: peak ...GB this request` log line. The code warns that `ps` RSS does not count Metal buffers, so it looks normal right up to the crash.
- Keep `MLX_PREFILL_SIZE` small (512 by default).
- On 16 GB, the `Claude Chat` and `Claude Agentico` launchers use 4-bit KV cache to avoid `kIOGPUCommandBufferCallbackErrorOutOfMemory` (see the next section for the trade-off).

## Tool calls arrive with empty arguments

[Issue #42](https://github.com/nicedreamzapp/claude-code-local/issues/42): with `MLX_KV_BITS=4`, Hermes 4 14B began returning tool calls like `Bash` with `{}` arguments partway through a session, and they did not recover until the server restarted. The retry logic does not catch this because the JSON is valid. Use `MLX_KV_BITS=0` (the server default) wherever memory allows.

## The model says it will do something but never calls a tool

Look in the log for `Tool intent detected but parser missed it`, followed by the raw text. The server retries up to `MLX_TOOL_RETRIES` times. If the model's format is new, that raw text is a good bug report: add it as a case in `scripts/test_parse_tool_calls.py` ([#58](https://github.com/nicedreamzapp/claude-code-local/issues/58)).

Slash commands that did nothing were caused by code mode dropping the `Skill` tool; fixed in [#56](https://github.com/nicedreamzapp/claude-code-local/pull/56).

## `500 ... 'ArraysCache' object has no attribute 'offset'`

Reported in [#9](https://github.com/nicedreamzapp/claude-code-local/issues/9). Fixed long ago. Current code checks for `.offset` and, if a cache type lacks it, logs `Cache has no offset (untrimmable type) — full prefill` and carries on.

## Gemma and `MLX_KV_BITS`

If you set `MLX_KV_BITS` with a Gemma model, the log says `Gemma detected: disabling KV cache quantization (RotatingKVCache NYI)`. Expected: mlx-lm cannot quantize that cache type, so the server ignores the setting ([PR #7](https://github.com/nicedreamzapp/claude-code-local/pull/7)).

## Model download fails

- **Hugging Face blocked or slow** ([#14](https://github.com/nicedreamzapp/claude-code-local/issues/14)): the maintainer suggested the mirror, `export HF_ENDPOINT=https://hf-mirror.com`, then run the download again. `scripts/persistent-download.sh` retries every 10 s until it finishes.
- **`Repository Not Found`** ([#3](https://github.com/nicedreamzapp/claude-code-local/issues/3)): an old model ID. Update the repo; current IDs are in [Install](install.md#install-by-hand).
- **`ERROR: MLX virtualenv not found at ...`**: run `setup.sh` first, or set `MLX_PYTHON`.

## I downloaded a model with LM Studio and it will not load

LM Studio often downloads GGUF files, and this server only loads **MLX** models ([#14](https://github.com/nicedreamzapp/claude-code-local/issues/14)). Use an MLX build (usually `-mlx` or from `mlx-community` / `lmstudio-community` MLX repos) and point at its folder:

```bash
MLX_MODEL=/path/to/your/mlx-model bash launchers/"Gemma 4 Code.command"
```

## The installer refuses to run

- `✗ This only runs on macOS (Apple Silicon).` or `✗ This needs Apple Silicon (M1 or newer).` Windows, Linux and Intel Macs are not supported ([#53](https://github.com/nicedreamzapp/claude-code-local/issues/53)). See the [FAQ](faq.md#does-it-run-on-windows-linux-or-an-intel-mac).
- `✗ git not found.` Run `xcode-select --install`.
- `✗ ~/claude-code-local already exists and is not a git checkout.` Move that folder or set `CCL_DIR=~/somewhere-else`.

## I saw a network connection when Claude Code started

Reported in [#11](https://github.com/nicedreamzapp/claude-code-local/issues/11). Claude Code itself contacts Anthropic on startup for telemetry, feature flags, marketplace and updates, even with `ANTHROPIC_BASE_URL` set. PR #32 found that the four `CLAUDE_CODE_DISABLE_*` / `DISABLE_AUTOUPDATER` variables stop it. Only some launchers set them; see [Quickstart](quickstart.md#4-check-that-nothing-leaves-the-mac) for the list and how to set them yourself, and how to check with `lsof`.

## `request body must be a JSON object` (HTTP 400)

The server got something that is not a JSON object, for example from a hand-written `curl`. Check the `Content-Type` and the body.

## Browser Agent

- `ERROR: Brave debug port (9222) didn't open.` Quit Brave completely and run the launcher again. It relaunches Brave with `--remote-debugging-port=9222`.
- `No page tab found — creating new tab...` is normal when every Brave window is closed.
- The agent itself is `~/.local/browser-agent/agent.py` from the separate [browser-agent](https://github.com/nicedreamzapp/browser-agent) repo; this repo does not install it.

## keep going

- `No Claude Code conversation found for this folder:` run it from the folder you were working in.
- `(add a free OpenRouter key to see these — see README)` or `(couldn't reach OpenRouter: ...)`: the cloud list needs `OPENROUTER_API_KEY` and a network connection. Local models still work.
- `The model didn't come up. Last lines of /tmp/mlx-server.log:` read those lines; usually memory or download problems as above.

## Native Engine

- `ERROR: ripgrep (rg) is not installed.` The Grep tool needs it: `brew install ripgrep`.
- `Not enough free memory for this model right now.` Only appears when `AGENT_MEM_BROKER_DIR` points at an external memory broker that refused the request. Close other model servers.
- `memory released — window may take a few seconds to close` after `/exit` is expected while macOS reclaims a large model's memory.

## Still stuck

Open an [issue](https://github.com/nicedreamzapp/claude-code-local/issues/new/choose) with your chip, memory, macOS version, the model, the output of `bash scripts/doctor.sh`, and the last lines of the server log ([CONTRIBUTING.md](https://github.com/nicedreamzapp/claude-code-local/blob/main/CONTRIBUTING.md)).
