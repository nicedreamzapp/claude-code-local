<p align="center">
  <h1 align="center">🧠⚡ Claude Code Local</h1>
  <p align="center">
    <strong>Run Claude Code 100% on-device with local AI on Apple Silicon.<br>No cloud, no API key, no proxy — an MLX-native server that speaks the Anthropic API.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/nicedreamzapp/claude-code-local/stargazers"><img src="https://img.shields.io/github/stars/nicedreamzapp/claude-code-local?style=for-the-badge&logo=github&color=f5c542&labelColor=1f2328" alt="GitHub stars"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/⚡_Qwen_3.5-65_tok%2Fs-brightgreen?style=for-the-badge" alt="Qwen 3.5 speed"></a>
    <a href="#-privacy--how-the-data-flows"><img src="https://img.shields.io/badge/🔒_Privacy-100%25_Local-success?style=for-the-badge" alt="100% Local"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/📜_License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
    <a href="https://discord.gg/ZdSqgAxUW"><img src="https://img.shields.io/discord/1497121921580404818?label=Discord&logo=discord&color=5865F2&style=for-the-badge" alt="Join the NiceDreamzApps Discord"></a>
  </p>
  <p align="center">
    <a href="#-what-is-this">🤔 What Is This</a> ·
    <a href="#-quick-start-3-commands">🚀 Quick Start</a> ·
    <a href="#-the-lineup--pick-your-fighter">🥊 Models</a> ·
    <a href="#-privacy--how-the-data-flows">🔒 Privacy</a> ·
    <a href="#-benchmarks">📊 Benchmarks</a> ·
    <a href="#-mcp-servers-work-too">🔌 MCP</a> ·
    <a href="#-whats-in-this-repo">📁 Repo Map</a>
  </p>
</p>

---

## 🤔 What Is This?

Your Mac has a powerful GPU built right into the chip. This project uses that GPU to run **large AI models — the same kind that power ChatGPT and Claude — entirely on your computer**, and plugs them into Claude Code so the whole coding experience works offline.

- 🚫 No internet needed
- 💰 No monthly subscription
- 🔒 No one sees your code or data
- ✅ Full Claude Code experience — write code, edit files, run tools, use MCP servers

```
         📱 You
          │
     🤖 Claude Code           ← the AI coding tool you know
          │  HTTP localhost:4000
     ⚡ MLX Native Server      ← this repo (~1000 lines of Python)
          │
     🧠 Local model           ← Gemma 4 31B · Llama 3.3 70B · Qwen 3.5 122B
          │
     🖥️  Apple Silicon GPU    ← your M-series chip does all the work
```

The trick: Claude Code speaks the **Anthropic API**, but most local model servers speak the OpenAI API — so everyone bolts on a translation proxy, which is slow and fragile. This server speaks Anthropic natively. One process, zero translations, **7.5× faster** than the proxy approach (17.6s vs 133s on the same Claude Code task).

The server (`proxy/server.py`) is one file. It loads the model via Apple's MLX framework, speaks the Anthropic API to Claude Code, translates each model family's tool-call format into Anthropic `tool_use` blocks (with garbled-output recovery for small models), strips local-model thinking tags in real time, reuses prompt caches across requests, and auto-swaps Claude Code's ~10K-token harness prompt for a slim local-friendly one (~28× fewer prompt tokens, prefill drops from ~60s to ~2s on Gemma 4 31B).

### 🎬 See it run

A real NDA. Llama 3.3 70B. Wi-Fi physically OFF. `lsof` running live on screen:

<p align="center">
  <a href="https://www.youtube.com/watch?v=V_J1LpNGwmY">
    <img src="https://img.youtube.com/vi/V_J1LpNGwmY/maxresdefault.jpg" width="640" alt="AirGap AI — Wi-Fi OFF NDA Demo">
  </a>
</p>

More demos on the channel: [youtube.com/@nicedreamzapps](https://www.youtube.com/@nicedreamzapps)

---

## 🚀 Quick Start (3 Commands)

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

`setup.sh` auto-detects your RAM, picks a model, downloads it, installs the MLX server, and creates a `Claude Local.command` launcher on your Desktop.

**Then double-click `Claude Local.command`.** You're coding locally.

You'll need:
- 🐍 **Python 3.12+** (for MLX)
- 🤖 **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

> 🐛 **If the launcher asks you to sign in to a Claude account:** your `claude` CLI is too old to support the `--bare` flag. Fix: `npm install -g @anthropic-ai/claude-code`.

### Or do it manually

```bash
# 1. Set up the MLX virtualenv
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm

# 2. Pick a model and download (one time, ~18-75 GB)
bash scripts/download-and-import.sh gemma   # or 'llama' or 'qwen'

# 3. Start the server
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh

# 4. Launch Claude Code
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

> 🛠️ **Note for contributors:** `setup.sh` installs the server as a symlink pointing back at this repo's `proxy/server.py`. Edit the file in the repo, restart the server, done — one source of truth, no drift.

---

## 🥊 The Lineup — Pick Your Fighter

Same MLX server, same Anthropic API — swap one env var to swap the brain. Plus DeepSeek V4 Flash via [Antirez's `ds4`](https://github.com/antirez/ds4) engine with its own native Metal runtime.

| | 🟢 **Gemma 4 31B** | 🟠 **Llama 3.3 70B** | 🔵 **Qwen 3.5 122B** | 🐳 **DeepSeek V4 Flash** |
|---|:---:|:---:|:---:|:---:|
| Build | 4-bit abliterated | 8-bit abliterated | 4-bit MoE (A10B) | 2-bit asymmetric (ds4 GGUF) |
| Speed | ~15 tok/s | ~7 tok/s | **65 tok/s** 🚀 | ~32 tok/s |
| Params | 31 B dense | 71 B dense | 122 B / 10 B active | 284 B / 37 B active |
| Context | 128 K | 128 K | 256 K | **1 M tokens** |
| RAM | ~18 GB | ~70 GB | ~75 GB | ~81 GB |
| Min Mac | 32 GB | 96 GB | 96 GB | 128 GB |
| Best at | Daily coding | Hardest reasoning | Max throughput | Long context, agentic loops |
| Launcher | `Gemma 4 Code.command` | `Llama 70B.command` | `Claude Local.command` | `DeepSeek V4 Flash.app` |

On a 16 GB Mac? [`Hermes-4-14B-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/Hermes-4-14B-abliterated-4bit-mlx) (~8 GB) is the sweet spot.

### ⭐ Our own MLX abliterated uploads

We package and upload our own abliterated MLX builds to HuggingFace so anyone running this repo can pull them with one command: [huggingface.co/divinetribe](https://huggingface.co/divinetribe). Abliteration sources: [huihui-ai](https://huggingface.co/huihui-ai) and [Babsie](https://huggingface.co/Babsie); MLX conversion + quantization by us. See [what abliteration means](https://huggingface.co/blog/mlabonne/abliteration).

> ⚠️ **Use it responsibly.** "Abliterated" suppresses the model's built-in refusal direction so it doesn't refuse benign-but-edgy requests. It is **not** a capability upgrade, and you remain bound by each upstream license (Llama 3.3, Gemma, Hermes/Qwen3).

---

## 🔒 Privacy + How the Data Flows

**Your code never leaves your Mac.** Not for a model call. Not for telemetry. Not ever. Everything runs over `localhost:4000`; the only network calls in the codebase are to localhost.

Every component audited: `server.py` (ours, zero outbound calls), mlx-lm and MLX (Apple, zero), model weights (zero at runtime), and the Claude Code CLI itself — its telemetry, feature-flag, marketplace, and autoupdater channels are all disabled by the launchers via documented Anthropic env vars:

```bash
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
DISABLE_AUTOUPDATER=1
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
```

**Verify it yourself:** run `lsof -p $(pgrep -f claude)` during a session — you'll see only `localhost:4000`. Run `lsof -i -P` while the server is up — nothing leaves your Mac.

> ⚠️ We [removed LiteLLM](https://x.com/Tahseen_Rahman/status/2035501506242240520) after supply-chain attack concerns and re-audited every dependency from scratch. If a package had unexplained network calls, it didn't ship.

| Situation | Use this? |
|-----------|:---------:|
| On a plane, no wifi | ✅ |
| NDA / sensitive client code | ✅ |
| Healthcare / legal / finance review | ✅ |
| Don't want API fees | ✅ |
| Need Claude-level reasoning | ☁️ local models are good, not Claude-level |

---

## 📊 Benchmarks

Three generations of optimization on the same task (asking Claude Code to write a function):

| Generation | Approach | Speed | Task time |
|---|---|---:|---:|
| 🐌 Gen 1 | Ollama + proxy | 30 tok/s | 133 s |
| 🏃 Gen 2 | llama.cpp + proxy | 41 tok/s | 133 s |
| 🚀 Gen 3 | **MLX Native (ours)** | **65 tok/s** | **17.6 s** |

Killing the proxy produced the entire 7.5× delta. Qwen 122B numbers measured on M5 Max 128 GB; full details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

For the tool-call reliability work (garbled-JSON recovery, retry logic, 98/98 test passes), see [docs/TOOL-CALL-RELIABILITY.md](docs/TOOL-CALL-RELIABILITY.md) and run `python3 scripts/test_mlx_server.py` yourself.

### ⚙️ Tuning

| Variable | Default | What it does |
|----------|---------|-------------|
| `MLX_MODEL` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | Pick which model to load |
| `MLX_KV_BITS` | `8` | KV cache quantization bits (4 saves memory, 8 improves coherence) |
| `MLX_KV_QUANT_START` | `1024` | Token position where KV quantization begins |
| `MLX_TOOL_RETRIES` | `2` | Max retries when a garbled tool call is detected |
| `MLX_MAX_TOKENS` | `8192` | Max output tokens per response |
| `MLX_SUPPRESS_THINKING` | `1` | Skip the model's reasoning chain (~1 min/request saved). Set `0` to let it think. |
| `MLX_BROWSER_MODE` | `0` | Optimize for chrome-devtools MCP sessions — strips a 30+ tool list down to the 9 essential browser tools (~99% fewer tokens) |

---

## 🔌 MCP Servers Work Too

Most local-LLM proxies break Claude Code's MCP plugin ecosystem — they mangle tool definitions or the streaming format. This server passes tool definitions through and translates the model's responses back into Anthropic's format, so **the whole MCP ecosystem works against your local model**. Wire servers up the normal Claude Code way:

```bash
# filesystem access
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem ~/projects

# GitHub issues/PRs/code search
claude mcp add github --env GITHUB_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github

# web search
claude mcp add brave-search --env BRAVE_API_KEY=$BRAVE_API_KEY -- npx -y @modelcontextprotocol/server-brave-search
```

---

## 📁 What's In This Repo

```
📦 claude-code-local/
 ├── ⚡ proxy/
 │   └── server.py              ← MLX Native Anthropic Server (~1000 lines)
 ├── 🚀 launchers/
 │   ├── Claude Local.command    ← Default — Claude Code + local model
 │   ├── Gemma 4 Code.command
 │   ├── Llama 70B.command
 │   ├── Browser Agent.command   ← launches the sibling browser-agent repo
 │   ├── Narrative Gemma.command ← voice-narration mode (pairs with NarrateClaude)
 │   └── lib/claude-local-common.sh
 ├── 🎭 NarrativeGemma/CLAUDE.md ← narration persona (opt-in)
 ├── 🛠️  scripts/                ← downloaders, server start, test suite, HF upload
 ├── 📊 docs/                    ← BENCHMARKS.md · TOOL-CALL-RELIABILITY.md
 └── setup.sh                    ← one-command installer
```

---

## 🧩 Sibling Repos

This repo is the **brain** of a local-first stack. Each sibling stands alone:

- 🎤 [NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude) — talk to Claude Code, hear replies in a cloned voice, both directions on-device
- 🌐 [browser-agent](https://github.com/nicedreamzapp/browser-agent) — local AI drives a real Brave browser via Chrome DevTools Protocol
- 📱 [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) — control Claude Code over iMessage, get back text/screenshots/video
- 🛟 [claude-failover](https://github.com/nicedreamzapp/claude-failover) — keep cloud Claude as primary, flip to local when you hit limits or an outage

---

## 🙏 Credits

Built on [Claude Code](https://claude.ai/claude-code) (Anthropic), [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-examples) (Apple), [Gemma](https://blog.google/technology/developers/gemma-open-models/) (Google DeepMind), [Llama](https://llama.meta.com/) (Meta), [Qwen](https://qwenlm.github.io/) (Alibaba), [ds4](https://github.com/antirez/ds4) (Antirez), and abliterations by [huihui-ai](https://huggingface.co/huihui-ai) and [Babsie](https://huggingface.co/Babsie). Tested on Apple M5 Max, 128 GB.

Built by [Matt Macosko](https://x.com/NiceDreamzApps) in Arcata, CA. More open-source at [nicedreamzwholesale.com/software](https://nicedreamzwholesale.com/software/) · demos at [youtube.com/@nicedreamzapps](https://www.youtube.com/@nicedreamzapps) · builders hang out on [Discord](https://discord.gg/ZdSqgAxUW).

Ideas, bug reports, new model launchers, edge cases on hardware I don't have — [open an issue](https://github.com/nicedreamzapp/claude-code-local/issues/new). I read them all.

**📜 MIT License** — use it however you want.
