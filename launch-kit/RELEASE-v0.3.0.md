# Draft release notes

> **Read this first (note for Matt, delete before publishing).**
> `v0.3.0` already exists. It was tagged on commit `184a952` and published as a GitHub release on
> 2026-08-22 ("v0.3.0, the leaderboard"). So these notes cover what landed **after** it:
> `v0.3.0..fcfe6fd`, 31 commits from 2026-09-05 to 2026-10-10. The file keeps the name you asked
> for, but the tag for this change set should probably be **v0.4.0** (it changes what `setup.sh`
> installs and what the `qwen` shortcut downloads, see "If you're upgrading"). Nothing here has been
> tagged or released. Compare link to use once a tag exists:
> `https://github.com/nicedreamzapp/claude-code-local/compare/v0.3.0...v0.4.0`

---

## v0.4.0 (draft)

Two Read calls in one reply used to become one Read call. The parser deduplicated tool calls by tool
name only, so when a model asked to read two files at once, the second request vanished and nobody got
an error. It now deduplicates on name plus arguments (#64). That was the first of eight fixes to
`proxy/server.py` that went in on Oct 9, most of them the same kind of thing: the server quietly
throwing away something the model or Claude Code sent.

The rest of that batch:

Gemma 4 kept losing its number and boolean arguments. It writes strings as `key:<|"|>value<|"|>` and
everything else bare, and once a string matched, the bare values were never read, so Edit lost
`replace_all` and Read lost `offset` and `limit` (#65). Gemma 4 arguments are now read as Gemma's own
JSON dialect first, which also fixes nested values like TodoWrite's `todos` and dashed keys like Grep's
`-i`, with the old regexes kept as a fallback (#68). Arguments sent as a JSON string, OpenAI style, are
decoded instead of arriving as empty input (#68).

A JSON array inside `<tool_call>` tags used to crash the request with a 500. It is now read as a list of
calls (#66).

When Claude Code's Read tool opened an image, the whole base64 payload went into the prompt as text.
One screenshot was enough to push a 32 GB Mac into swap. Images and documents in a tool result now
become a short marker (#67).

Errors now come back in the Anthropic shape (`{"type": "error", "error": {"type": "api_error"}}`), and a
request body that isn't valid JSON gets a 400 instead of a dropped connection (#69).

`stop_sequences` is honored on both the buffered and the streaming path, a failed tool's `is_error`
reaches the model as `Error: ...`, and a request that declares no tools no longer gets tool calls
parsed out of its JSON answer (#70).

The live stream now filters `<think>` blocks the same way the buffered path does, and the tool-call
retry runs on a throwaway cache so it no longer sees the conversation twice or changes the cache the
next request reuses (#71).

Earlier in the cycle @zwolf25 fixed two code-mode bugs from #54 (#56). Code mode's tool filter dropped
`Skill`, so slash commands could never call the skill they named. And fenced JSON tool calls shaped as
an array, which is what Hermes 4 14B emits, were skipped. Both fixed, and that PR added
`scripts/test_parse_tool_calls.py`.

### Tests you can run without a model

There are now five test scripts that stub out MLX and need no model and no running server:

```bash
python3 scripts/test_parse_tool_calls.py
python3 scripts/test_convert_messages.py
python3 scripts/test_generate_response.py
python3 scripts/test_http_errors.py
python3 scripts/test_stream_and_retry.py
```

On this commit they report 38 passing checks between them (20, 1, 6, 4 and 7).
`scripts/test_mlx_server.py` still needs a server on `localhost:4000`.

### Keep going when Claude runs out

`bin/keepgoing` finds the last Claude Code conversation in the current folder and carries it on with
another model: a free OpenRouter model if you add a key, or one of the local models through this repo's
server. `setup.sh` links it in and adds a `keep going` shell command (`scripts/keepgoing-shell.sh`).
`scripts/statusline-keepgoing.sh` is an optional status line that reminds you it's there. Picking a
cloud model in that menu sends the conversation to OpenRouter; the local ones don't leave the Mac.

### New model lineup in setup

`setup.sh` now picks by memory:

| Memory | Model |
|---|---|
| 96 GB and up | Qwen 3.8 27B, 8-bit (`lmstudio-community/Qwen3.8-27B-MLX-8bit`) |
| 64 to 95 GB | Gemma 4 31B, 4-bit |
| 32 to 63 GB | Gemma 4 12B, 4-bit |
| 16 to 31 GB | Hermes 4 14B, 4-bit |
| under 16 GB | Gemma 4 E4B, 4-bit (labeled: it failed our Claude Code smoke test) |

Qwen 3.8 27B 8-bit passed the Claude Code smoke test (write `hello.py`, run it, report) in 38 s on an
M5 Max 128 GB. On Agent-12 with the DFlash 2 drafter it went 12/12 easy in 122 s and 7/8 hard in
401 s (Sep 19 rerun). Those Agent-12 numbers come from Agent-12's own harness, not from inside Claude
Code. `scripts/doctor.sh` gives the same recommendations as setup now.

### If you're upgrading

`bash scripts/download-and-import.sh qwen` now downloads **Qwen 3.8 27B 8-bit** (~29 GB). It used to
download Qwen 3.5 122B. `gemma12` and `hermes` are new shortcuts. `llama` still works for existing
setups.

The `Llama 70B.command` launcher is gone and `Qwen 3.8 Code.command` replaces it. Qwen 3.5 122B,
Qwen 3.5 4B and DeepSeek through `ds4` are no longer offered by setup.

### Docs

The README was rewritten for people arriving for the first time, and the old long README moved,
unchanged apart from relative links, to `docs/FULL-GUIDE.md`. New `CONTRIBUTING.md`. A section on
other ways to run Claude Code against other models (Ollama's Anthropic API, claude-code-router,
claude-code-proxy). The `lsof` privacy check in the README now lists network connections only (#62).
@pratikgx fixed a broken link in `docs/PHONE-CONTROL.md` (#60). `bin/fia` is a single entry point that
runs the existing launchers.

This release stands on Anthropic's Claude Code, Apple's MLX and mlx-lm, and the model teams at Google
(Gemma), Qwen, Nous Research (Hermes), lmstudio-community and mlx-community for the builds. Thanks to
@zwolf25 and @pratikgx for the PRs, and to everyone who filed issues.
