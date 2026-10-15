---
title: Configuration - every environment variable, flag and launcher
description: >-
  Reference for every MLX_ environment variable the Claude Code Local server
  reads, its HTTP endpoints, the launchers, the Native Engine AGENT_ settings,
  keep going, and the flags of each helper script. Read from the source code.
---

# Configuration

Everything on this page was read from the source. When a default here disagrees with older docs, the code wins, and the disagreement is marked **TODO-VERIFY**.

## Server: `proxy/server.py`

All settings are environment variables read once at startup. Restart the server to change them.

### Model and network

| Variable | Default | What the code does |
|---|---|---|
| `MLX_MODEL` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | Hugging Face ID or local folder passed to `mlx_lm.utils.load()`. Downloaded on first use if it is an ID. Reported as `model` by `/health`. |
| `MLX_PORT` | `4000` | Port to listen on. The server always binds `127.0.0.1`, never all interfaces. |

### Memory and speed

| Variable | Default | What the code does |
|---|---|---|
| `MLX_PREFILL_SIZE` | `512` | Tokens per prefill chunk (`prefill_step_size`). It used to be 8192; the code comment records 34.2 GB peak at 8192 vs 20.9 GB at 1024 on a 21.4k-token prompt with Gemma 4 31B 4-bit, and notes small chunks were not slower. |
| `MLX_CACHE_LIMIT_GB` | `3` | Cap on MLX's buffer recycle pool via `mx.set_cache_limit`. `0` disables the cap. Empty means the default. |
| `MLX_KV_BITS` | `0` (full precision) | KV cache quantization bits. When set, the server also passes `kv_group_size=64`. **Forced back to 0 when the model path contains `gemma`**, because mlx-lm cannot quantize Gemma's `RotatingKVCache`. |
| `MLX_KV_QUANT_START` | `256` | Token position where KV quantization begins (`quantized_kv_start`). Only used when `MLX_KV_BITS` is non-zero. |

!!! warning "`MLX_KV_BITS=4` and tool calls"
    [Issue #42](https://github.com/nicedreamzapp/claude-code-local/issues/42) measured `MLX_KV_BITS=4` on Hermes 4 14B passing 3 to 10 of 14 tool-call tests, against 12 to 13 at `0`, with tool calls coming back with empty arguments partway through a session. Use 4-bit KV only when you need the memory. `Claude Chat.command` and `Claude Agentico.command` default to it (with `MLX_KV_QUANT_START=0`) to fit 16 GB.

!!! info "TODO-VERIFY: defaults in older docs"
    `docs/FULL-GUIDE.md` lists the defaults as `MLX_KV_BITS=8` and `MLX_KV_QUANT_START=1024`. The code defaults are `0` and `256`. This page follows the code.

`MLX_PORT`, `MLX_PREFILL_SIZE`, `MLX_KV_BITS`, `MLX_KV_QUANT_START`, `MLX_MAX_TOKENS` and `MLX_TOOL_RETRIES` are read with a helper that treats an empty value as "use the default". `MLX_CACHE_LIMIT_GB` does the same. A non-numeric value stops the server at startup.

### Generation

| Variable | Default | What the code does |
|---|---|---|
| `MLX_MAX_TOKENS` | `8192` | Output token limit when the request has no `max_tokens`. |
| `MLX_SUPPRESS_THINKING` | `1` | Only when the model path contains `gemma`: appends an empty `<|channel>thought\n<channel|>` block to the prompt so Gemma 4 skips its reasoning chain. Anything other than exactly `1` turns it off. |
| `MLX_TOOL_RETRIES` | `2` | How many times to re-prompt when the reply looks like a tool call but none could be parsed. Each retry runs on a fresh, throwaway KV cache. |

Temperature is not an environment variable. The server uses the request's `temperature`, defaulting to `0.2`.

### Prompt modes

| Variable | Default | What the code does |
|---|---|---|
| `MLX_CODE_MODE` | on | Any value except `0` keeps it on. When a request includes any of Bash, Read, Edit, Write, Grep, Glob or Skill, the server replaces Claude Code's system prompt with a short coding prompt, keeps only those seven tools and strips their descriptions down to names and parameter types. See [How it works](how-it-works.md#code-mode). |
| `MLX_BROWSER_MODE` | off | Exactly `1` turns it on. Takes priority over code mode. For requests that include Chrome DevTools MCP tools, replaces the system prompt with a browser prompt and keeps only nine `mcp__chrome-devtools__*` tools (no screenshots). Requests without those tools pass through untouched. |
| `MLX_APPEND_SYSTEM_PROMPT_FILE` | unset | Path to a text file. Its contents are appended to the system prompt after code or browser mode has run, separated by `---`. Read on every request. `Narrative Gemma.command` uses it for its narration rules. |

### Streaming and debugging

| Variable | Default | What the code does |
|---|---|---|
| `MLX_LIVE_STREAM` | on | For `stream: true` requests **with no tools**, tokens are sent as they are generated. `0` or `false` (any case) turns that off. Requests with tools are always generated in full first, parsed, then replayed as server-sent events. |
| `MLX_DEBUG_REQUEST` | unset | Any non-empty value logs the role and block types of the last six messages of each request. |
| `MLX_DEBUG_RESPONSE` | unset | Any non-empty value logs the tool schemas used for input filtering and the first 1500 characters of each response JSON. |

### Request fields the server honors

| Field | Behavior |
|---|---|
| `system` | String or list of text blocks. Replaced in code mode and browser mode. |
| `messages` | Text, `tool_use` and `tool_result` blocks are converted. Image and document blocks inside a `tool_result` become the text `[image omitted]` or `[document omitted]`. `tool_result` with `is_error: true` gets an `Error: ` prefix. |
| `tools` | Converted to the Hugging Face function format and passed to the model's chat template. If the template rejects them, they are written into the system prompt as text instead. A request with no tools never gets `tool_use` back. |
| `max_tokens` | Defaults to `MLX_MAX_TOKENS`. |
| `temperature` | Defaults to `0.2`. |
| `stop_sequences` | Generation stops at the earliest match, the text is cut there, and `stop_reason` is `stop_sequence`. |
| `stream` | Server-sent events in the Anthropic format. |
| `model` | Not used to pick a model. Echoed back in the response (default `claude-sonnet-4-6`). |

### HTTP endpoints

| Method and path | Response |
|---|---|
| `POST /v1/messages` (also `/messages`) | Anthropic Messages response, JSON or SSE. Errors are `{"type": "error", "error": {...}}`: 400 `invalid_request_error` for a body that is not a JSON object, 500 `api_error` if generation throws. |
| `POST /v1/messages/count_tokens` (also `/messages/count_tokens`) | `{"input_tokens": N}`, computed by running the same conversion and tokenizer as a real request. Falls back to characters / 4 if that fails. |
| `GET /v1/models` (also `/models`) | A fixed list: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`. |
| `GET /health` | `status`, `model`, `prefill_size`, `rss_gb`, `mx_active_gb`, `mx_cache_gb`, `mx_peak_gb`. |
| `HEAD` any path | 200 |
| Any other path | 200 with `{}` |

The server does not check `x-api-key` or `Authorization`.

## Launchers

The `.command` files in `launchers/` are bash scripts for macOS. All but the two Native Engine launchers source `launchers/lib/claude-local-common.sh`.

| Launcher | Default model | Starts the server with | Claude Code flags and extras |
|---|---|---|---|
| `Claude Local.command` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | `ensure_mlx_server` | `--permission-mode auto --bare --append-system-prompt-file ~/.claude/CLAUDE.md --mcp-config ~/.claude.json`, `ANTHROPIC_API_KEY=sk-local` |
| `Gemma 4 Code.command` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | `ensure_mlx_server` | `--permission-mode auto --settings lib/local-settings.json --append-system-prompt-file ~/.claude/CLAUDE.md --mcp-config ~/.claude.json` |
| `Qwen 3.8 Code.command` | `lmstudio-community/Qwen3.8-27B-MLX-8bit` | `ensure_mlx_server` | same as Gemma 4 Code |
| `Claude Chat.command` | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | `ensure_mlx_server`, `MLX_KV_BITS=4`, `MLX_KV_QUANT_START=0` | `--tools "" --effort low`, offline env vars, `ANTHROPIC_AUTH_TOKEN`, `DISABLE_LOGIN_COMMAND=1` |
| `Claude Agentico.command` | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | same as Chat | `--effort low --permission-mode auto`, offline env vars, `ANTHROPIC_AUTH_TOKEN`, `DISABLE_LOGIN_COMMAND=1` |
| `Narrative Gemma.command` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | `force_restart_mlx_server` with `MLX_APPEND_SYSTEM_PROMPT_FILE` | Runs inside `NarrativeGemma/` so its `CLAUDE.md` narration persona loads; starts NarrateClaude dictation if `~/NarrateClaude/dictation/bin/dictation` exists |
| `Browser Agent.command` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | `force_restart_mlx_server` with `MLX_BROWSER_MODE=1` | Starts Brave with `--remote-debugging-port=9222`, then runs `~/.local/browser-agent/agent.py` from the separate [browser-agent](https://github.com/nicedreamzapp/browser-agent) repo. No Claude Code. |
| `Gemma 4 Code (Native Engine).command` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | none, model loads in-process | `AGENT_DIALECT=prompted`, runs `agent/agent.py` from `~` |
| `Qwen 3 Coder (Native Engine).command` | `lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit` | none, model loads in-process | `AGENT_DIALECT=native`, runs `agent/agent.py` from `~` |

"Offline env vars" means `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `DISABLE_AUTOUPDATER=1`, `CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1` and `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`. `lib/local-settings.json` sets `"apiKeyHelper": "echo sk-local"`.

!!! info "TODO-VERIFY"
    - Only Chat and Agentico set the offline env vars (see [Quickstart](quickstart.md#4-check-that-nothing-leaves-the-mac)).
    - `Claude Local`, `Gemma 4 Code`, `Qwen 3.8 Code` and `Narrative Gemma` pass `~/.claude/CLAUDE.md` and `~/.claude.json` to Claude Code without checking that they exist. Behavior when they are missing is untested here.
    - `docs/MAC-BASE-SETUP.md` says `--bare` does not exist in recent CLI versions, while the README says an old CLI is the reason `--bare` fails. Only `Claude Local.command` uses it.

### Settings every launcher reads

| Variable | Default | Used by |
|---|---|---|
| `MLX_MODEL` | per launcher, see table | all |
| `MLX_SERVER` | `~/.local/mlx-native-server/server.py` | launchers using the shared library, `start-mlx-server.sh`, `keepgoing` |
| `MLX_PYTHON` | `~/.local/mlx-server/bin/python3` | all |
| `CLAUDE_BIN` | first of: `claude` on `PATH`, `~/.local/bin/claude`, `~/.claude/local/claude`, `/opt/homebrew/bin/claude`, `/usr/local/bin/claude` | Claude Code launchers |
| `MLX_MODEL_LABEL` | `Gemma 4 31B` | display name only, in `Claude Local` and `Browser Agent` |
| `MLX_KV_BITS`, `MLX_KV_QUANT_START` | empty | forwarded to the server by the shared library |

Two details of the shared library:

- **Model-aware restart.** `ensure_mlx_server` reads `/health` and compares the last path component of the running model with the one you asked for, ignoring case. If they differ, it stops the server (`pkill -f mlx-native-server/server.py`) and starts the right one. `force_restart_mlx_server` always restarts, because some settings only apply at process start.
- **Local folders first.** `resolve_mlx_model` uses a flat folder such as `~/.cache/huggingface/hub/gemma-4-31b-it-abliterated-4bit-mlx` if it exists and contains `config.json`, so mlx-lm loads from disk instead of downloading. Otherwise it passes the Hugging Face ID.

The server log goes to `/tmp/mlx-server.log`. The health wait is 180 checks two seconds apart (six minutes).

### The Desktop launcher made by `setup.sh`

`~/Desktop/Claude Local.command` is generated, not copied. It has the chosen model, server path and Python path written into it, starts the server only if nothing is on port 4000 (it does not check which model is running), and runs Claude Code with `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY=sk-local` and `--model claude-sonnet-4-6`, without the extra flags of `launchers/Claude Local.command`.

## Native Engine: `agent/agent.py`

### Command line

```text
agent.py [-p PROMPT] [--model MODEL] [-C DIR]
```

| Flag | Meaning |
|---|---|
| `-p`, `--prompt` | Run one prompt, print the final answer, exit |
| `--model` | Model ID or path (default: `AGENT_MODEL`) |
| `-C`, `--dir` | Change to this directory first |

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `AGENT_MODEL` | `lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit` | Model to load (mlx) or name to send (http) |
| `AGENT_TITLE` | last part of the model ID | Name shown in the banner and the system prompt |
| `AGENT_BACKEND` | `mlx` | `mlx` loads the model in this process. `http` sends requests to an Anthropic-format server. |
| `AGENT_DIALECT` | `native` | `native` passes the tools to the model's own chat template. `prompted` describes an XML tool format in the system prompt instead (used for Gemma 4). |
| `AGENT_BASE_URL` | empty | Server URL for the `http` backend, for example `http://localhost:4000` |
| `AGENT_AUTH_TOKEN` | `sk-local` | Sent as `x-api-key` and `Authorization: Bearer` on the `http` backend |
| `AGENT_MAX_TOKENS` | `4096` | Output limit per step |
| `AGENT_TEMP` | `0.3` | Sampling temperature |
| `AGENT_MAX_STEPS` | `40` | Tool-call steps per user turn |
| `AGENT_BASH_TIMEOUT` | `120` | Seconds before a Bash tool call is killed (stdin is closed so interactive programs exit) |
| `AGENT_CTX_LIMIT` | from the model config, else `131072` | Context size used by the `/context` meter |
| `AGENT_ROLLING_KV` | unset | `1` keeps Gemma's stock rotating KV cache instead of swapping in plain caches. See [How it works](how-it-works.md#the-native-engine). |
| `AGENT_SPEAK` | unset | `1`, `true` or `yes` speaks each final answer through `~/.local/bin/speak` |
| `AGENT_PROMPT_FILE` | unset | Extra system prompt text, read once at startup |
| `AGENT_MEM_BROKER_DIR` | unset | Directory containing an external `mem_client.py`. When set, the engine reserves memory before loading and releases it on exit. Without it, nothing happens. |
| `AGENT_LEASE_NAME` | `agent-local` | Name of that reservation |
| `AGENT_LEASE_GB` | `36` (mlx) or `0` (http) | Size of that reservation; `0` skips it |

Tool output fed back to the model is clipped to 30,000 characters. Grep needs `rg` (ripgrep) installed.

### Commands inside the agent

| Command | Effect |
|---|---|
| `/exit`, `/quit` | Quit |
| `/reset` | Clear the conversation (and the KV cache) |
| `/context` | Show tokens used out of the context limit |
| `/tools` | List the tools |
| `/cwd` | Show the working directory |
| `/model` | Show the model path |

## Keep going

`keep going` (or `keepgoing`) runs `bin/keepgoing`, which calls `bin/keepgoing.py` to find the newest Claude Code conversation for the current folder in `~/.claude/projects/*/*.jsonl` and show a menu.

| Setting | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | Free key from openrouter.ai. Unlocks the free cloud list. |
| `KEEPGOING_BEFORE_LOCAL` | Shell command run before a local model is started |
| `KEEPGOING_AFTER_LOCAL` | Shell command run when the local session ends |
| `MLX_PORT`, `MLX_PYTHON`, `MLX_SERVER` | Same meaning as for the server and launchers |

Settings can also live in `~/.config/claude-code-local/keepgoing.env`, which is sourced first.

- **Free cloud list:** up to three OpenRouter models whose ID ends in `:free`, that support tools and have at least 128,000 tokens of context, largest context first. Claude Code is pointed at `https://openrouter.ai/api` with `ANTHROPIC_AUTH_TOKEN` set to your key.
- **Local list:** the model already running on the port (if any), then the five `setup.sh` models whose memory requirement (8, 16, 32, 64, 96 GB) your Mac meets. Choosing one kills whatever is on the port, starts the server and waits up to 20 minutes for it. The local path sets the four offline env vars.

To get the reminder in Claude Code's status line, point `statusLine` in `~/.claude/settings.json` at `bash ~/claude-code-local/scripts/statusline-keepgoing.sh`.

## Helper scripts

| Script | Arguments and settings |
|---|---|
| `install.sh` | `CCL_DIR` sets the install folder (default `~/claude-code-local`) |
| `setup.sh` | none |
| `uninstall.sh` | `--keep-models`, `-h`/`--help` |
| `scripts/doctor.sh` | `--bench`; reads `HF_HOME` |
| `scripts/start-mlx-server.sh` | Model as first argument, else `MLX_MODEL`; `MLX_PORT`, `MLX_PYTHON`, `MLX_SERVER` |
| `scripts/download-and-import.sh` | Short name or Hugging Face ID ([list](install.md#install-by-hand)); `MLX_MODEL`, `MLX_PYTHON` |
| `scripts/persistent-download.sh` | Same arguments, retried every 10 s |
| `scripts/upload-mlx-quant.sh` | `<local-folder> <hf-repo-id>`; needs `hf auth login` or `HF_TOKEN`; installs `hf_transfer` into the MLX venv |
| `bin/fia` | `start`/`code`/`brain`, `gemma`, `qwen`, `voice`/`narrate`, `browser`, `phone`, `make film\|music`, `status`, `help`. A front door that runs the matching launcher or prints where to get the missing piece. |

### `smart-router/` (experimental)

`smart-router/router.py` listens on `ONE_AI_PORT` (default `4010`) and forwards each Anthropic request to one of several local backends using keyword and size rules (`/deep`, `/glm`, `/fast`, `/code` overrides; over ~100k estimated tokens to a DeepSeek server on port 8000; short greetings to Gemma; everything else to Qwen3-Coder). `warm_pool.sh` keeps two models loaded on ports 4000 and 4001.

!!! info "TODO-VERIFY"
    Both files hardcode paths under `~/Desktop/PROJECTS/Local AI Setup` and `~/.lmstudio/models`, and expect other servers (a DeepSeek `ds4` server, a GLM model) that this repo does not install. They look like the maintainer's own setup rather than something meant to run as shipped.
