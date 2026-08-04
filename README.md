<p align="center">
  <h1 align="center">🧠⚡ Claude Code Local</h1>
  <p align="center">
    <strong>Run Claude Code 100% on-device with local AI on Apple Silicon.<br>No cloud, no API key, no proxy — an MLX-native server that speaks the Anthropic API.<br>🥊 Pick your fighter: Gemma 4 31B · Llama 3.3 70B · Qwen 3.5 122B · DeepSeek V4 Flash (1M context via <a href="#-deepseek-v4-flash-via-ds4"><code>ds4</code></a>).</strong>
  </p>
  <p align="center">
    <a href="https://github.com/nicedreamzapp/claude-code-local/stargazers"><img src="https://img.shields.io/github/stars/nicedreamzapp/claude-code-local?style=for-the-badge&logo=github&color=f5c542&labelColor=1f2328" alt="GitHub stars"></a>
    <a href="https://github.com/nicedreamzapp/claude-code-local/network/members"><img src="https://img.shields.io/github/forks/nicedreamzapp/claude-code-local?style=for-the-badge&logo=github&color=4c9a2a&labelColor=1f2328" alt="GitHub forks"></a>
    <a href="#-the-lineup--pick-your-fighter"><img src="https://img.shields.io/badge/🥊_Lineup-4_Models-red?style=for-the-badge" alt="4 Models"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/⚡_Qwen_3.5-65_tok%2Fs-brightgreen?style=for-the-badge" alt="Qwen 3.5 speed"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/🚀_Claude_Code_Task-17.6s-blue?style=for-the-badge" alt="Claude Code task time"></a>
    <a href="#-privacy--how-the-data-flows"><img src="https://img.shields.io/badge/🔒_Privacy-100%25_Local-success?style=for-the-badge" alt="100% Local"></a>
    <a href="#-hands-free-voice-mode"><img src="https://img.shields.io/badge/🎤_Voice-Hands_Free-orange?style=for-the-badge" alt="Hands-Free Voice"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/📜_License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
    <a href="https://discord.gg/ZdSqgAxUW"><img src="https://img.shields.io/discord/1497121921580404818?label=NiceDreamzApps&logo=discord&color=5865F2&style=for-the-badge" alt="Join the NiceDreamzApps Discord"></a>
  </p>
  <p align="center">
    <a href="#-what-is-this">🤔 What Is This</a> ·
    <a href="#-quick-start-3-commands">🚀 Quick Start</a> ·
    <a href="#-the-lineup--pick-your-fighter">🥊 Lineup</a> ·
    <a href="#-the-modes">🎮 Modes</a> ·
    <a href="#-privacy--how-the-data-flows">🔒 Privacy</a> ·
    <a href="#-benchmarks">📊 Benchmarks</a> ·
    <a href="#-hands-free-voice-mode">🎤 Voice</a> ·
    <a href="#-browser-agent">🌐 Browser</a> ·
    <a href="#-control-from-your-phone">📱 Phone</a> ·
    <a href="#-mcp-servers--claude-codes-plugin-ecosystem-100-local">🔌 MCP</a> ·
    <a href="#-whats-next">🛣️ Roadmap</a>
  </p>
</p>

---

## 🤔 What Is This?

Your Mac has a powerful GPU built right into the chip. This project uses that GPU to run **massive AI models — the same kind that power ChatGPT and Claude — entirely on your computer**, and plugs them into Claude Code so the whole coding experience works offline.

🚫 No internet needed
💰 No monthly subscription
🔒 No one sees your code or data
✅ Full Claude Code experience — write code, edit files, manage projects, control your browser, or run a full hands-free voice session

```
         📱 You (Mac or Phone)
          │
     🤖 Claude Code           ← the AI coding tool you know
          │  HTTP localhost:4000
     ⚡ MLX Native Server      ← this repo (~1000 lines of Python)
          │
     🥊 Pick your fighter     ← Gemma 4 31B · Llama 3.3 70B · Qwen 3.5 122B
          │
     🖥️  Apple Silicon GPU    ← your M-series chip does all the work
```

**The trick:** Claude Code speaks the **Anthropic API**. Local model servers speak the **OpenAI API**. So everyone bolts a translation proxy in between — and the proxy is slow and fragile. This server speaks Anthropic natively. One process, zero translations:

| 🐌 What everyone else does | 🚀 What we did |
|---|---|
| Claude Code → **Proxy** → Ollama → Model | Claude Code → **Our Server** → Model |
| 3 processes, 2 API translations | **1 process, 0 translations** |
| 133 seconds per task | **17.6 seconds per task** |

> 🎯 That one change — **eliminating the proxy** — made it **7.5× faster**.

---

## 🎬 Watch It Run — AirGap AI

**A real NDA. Llama 3.3 70B. Wi-Fi physically OFF. `lsof` running live.** Watch a 70-billion-parameter model audit a confidential legal document, on-device, with the receipts on screen.

<p align="center">
  <a href="https://www.youtube.com/watch?v=V_J1LpNGwmY">
    <img src="https://img.youtube.com/vi/V_J1LpNGwmY/maxresdefault.jpg" width="720" alt="AirGap AI — Wi-Fi OFF NDA Demo">
  </a>
</p>

<p align="center">
  <em>AirGap is this whole build running as one private workstation — a capability, not a product. Everything you need is in this repo. If your firm needs one built, <a href="https://nicedreamzwholesale.com/airgap/">here's what it looks like</a>.</em>
</p>

**More local-AI demos on the channel:**

| Video | What happens |
|---|---|
| [🌌 The Rematch](https://www.youtube.com/watch?v=03KVQmEx13Q) | 4 AI engines build northern lights, 3 fully local — the local challenger painted the best aurora |
| [🏁 Hexagon Shootout](https://www.youtube.com/watch?v=2KeTDDodE0A) | Gemma 31B vs Llama 70B vs cloud Claude, same physics prompt, live counters — 2 of 3 with zero cloud calls |
| [🐳 DeepSeek Three-Way](https://youtu.be/7l8-s8xkpms) | DeepSeek V4 Flash local beats cloud Claude on wall-clock, same MacBook |
| [🎤 NarrateClaude](https://www.youtube.com/watch?v=4ETqEjjopUk) | Speak to Claude Code, hear replies in a cloned voice — 100% on-device |
| [🏠 Mac mini as home AI](https://www.youtube.com/watch?v=PLbV4QtFmFY) | Chat with the Mac mini at home from any browser on any phone |

---

## 🥊 The Lineup — Pick Your Fighter

We started with one model. Now we ship a **roster**. Same MLX server, same Anthropic API — swap one env var and you swap the brain. Plus the `ds4` engine for DeepSeek V4 Flash via its own native Metal runtime.

| | 🟢 **Gemma 4 31B** | 🟠 **Llama 3.3 70B** | 🔵 **Qwen 3.5 122B** | 🐳 **DeepSeek V4 Flash** ⭐ |
|---|:---:|:---:|:---:|:---:|
| Nickname | The Quick One | The Wise One | The Beast | The 1M-Context Whale |
| Build | 4-bit IT abliterated | 8-bit abliterated | 4-bit MoE (A10B) | 2-bit asymmetric (ds4 GGUF) |
| Speed | ~15 tok/s | ~7 tok/s | **65 tok/s** 🚀 | ~32 tok/s |
| Params | 31 B dense | 71 B dense | 122 B / 10 B active | **284 B / 37 B active** |
| Context | 128 K | 128 K | 256 K | **1 M tokens** |
| RAM | ~18 GB | ~70 GB | ~75 GB | ~81 GB |
| Min RAM to run | 32 GB | 96 GB | 96 GB | 128 GB |
| Best at | Daily coding | Hardest reasoning, full precision | Max throughput, active sparsity | Long context, agentic loops |
| Engine | MLX Native | MLX Native | MLX Native | [`antirez/ds4`](https://github.com/antirez/ds4) |
| Launcher | `Gemma 4 Code.command` | `Llama 70B.command` | `Claude Local.command` | `DeepSeek V4 Flash.app` |

> 💡 **Fun fact:** Qwen wins raw speed because it's an MoE — only 10B of 122B params activate per token. DeepSeek V4 Flash is even bigger (284B) but only ~37B active per token, *and* it ships with on-disk KV cache so a 25k-token Claude Code system prompt prefills exactly once, ever.

### 🐳 DeepSeek V4 Flash via `ds4`

We tested it the day Antirez (the Redis guy) shipped `ds4`. **Local DeepSeek beat cloud Claude on wall-clock time** on the same MacBook, same prompt — [watch the three-way](https://youtu.be/7l8-s8xkpms).

| | |
|---|---|
| 🧠 **Engine** | [`antirez/ds4`](https://github.com/antirez/ds4) — pure C + Metal kernels, ~few thousand lines |
| 🤗 **Weights** | [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) (q2: 81 GB, q4: 153 GB) |
| 📦 **Server wrapper** | `~/.local/bin/ds4-server-up` (boots on demand) |
| 🚀 **Claude Code wrapper** | `~/.local/bin/claude-ds4` (drop-in replacement for `claude`) |
| 📏 **Context** | 1 M tokens; 200 K is sane for most agent runs |
| 💾 **Disk KV cache** | Persists across restarts — first prefill is the only one that ever happens |

### ⭐ Our Own MLX Abliterated Uploads

The models in this lineup aren't from generic mirrors — **we package and upload our own abliterated MLX builds** to HuggingFace so anyone running this repo can pull them with one command. Browse the full set at [huggingface.co/divinetribe](https://huggingface.co/divinetribe).

```bash
# Llama 3.3 70B — full-precision feel
MLX_MODEL=divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx \
  bash scripts/start-mlx-server.sh

# Gemma 4 31B — fast daily driver
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh

# Hermes 4 14B — sweet spot for 16/32 GB Macs
MLX_MODEL=divinetribe/Hermes-4-14B-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh
```

| Model | Quant | Disk | Params | Context | Best for |
|---|---|---|---|---|---|
| [`Llama-3.3-70B-Instruct-abliterated-8bit-mlx`](https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx) | 8-bit, g64 | ~75 GB | 71 B dense | 128 K | Hardest reasoning on 96 GB+ Macs |
| [`gemma-4-31b-it-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/gemma-4-31b-it-abliterated-4bit-mlx) | 4-bit, g64 | ~17 GB | 31 B dense | 128 K | Daily coding on a 32 GB+ Mac |
| [`Hermes-4-14B-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/Hermes-4-14B-abliterated-4bit-mlx) | 4-bit, g64 | ~8 GB | 14 B dense (Qwen3 base) | 40 K | 16 GB Macs, instruction-following, tool use |

**Abliteration sources:** [huihui-ai](https://huggingface.co/huihui-ai) (Llama, Gemma) and [Babsie](https://huggingface.co/Babsie) (Hermes). MLX conversion + quantization by us. See [what abliteration means](https://huggingface.co/blog/mlabonne/abliteration).

> ⚠️ **Use it responsibly.** "Abliterated" suppresses the model's built-in refusal direction so it doesn't refuse benign-but-edgy requests. It is **not** a general capability upgrade, and you remain bound by each upstream license (Llama 3.3, Gemma, Hermes/Qwen3).

---

## 🎮 The Modes

Four ways to run the lineup. Each one is a double-clickable launcher in `launchers/`.

| Mode | What it does | Launcher |
|---|---|---|
| 🤖 **Code** | Run Claude Code with a local model — same UX, no API key | `Claude Local.command`, `Gemma 4 Code.command`, `Llama 70B.command` |
| 🌐 **Browser** | Local AI controls real Brave browser via Chrome DevTools | `Browser Agent.command` |
| 🎤 **Hands-Free Voice** | Speak in, hear replies in your cloned voice — full loop, 100% on-device | `Narrative Gemma.command` + [NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude) |
| 📱 **Phone** | iMessage in → text/image/video out, via [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) | `~/.claude/imessage-*.sh` |

---

## 💻 What You Need

| Your Mac | RAM | What You Can Run |
|----------|-----|-------------------|
| M1/M2/M3/M4 (base) | 8-16 GB | 🟡 Small models (Hermes 4 14B) |
| M1/M2/M3/M4 Pro | 18-36 GB | 🟠 Gemma 4 31B (tight) |
| M2/M3/M4/M5 Max | 64-128 GB | 🟢 **Gemma 4 31B** + 🔵 Qwen 3.5 122B |
| M2/M3/M4 Ultra | 128-192 GB | 🔵 All four fighters, incl. 🐳 DeepSeek |

Also need:
- 🐍 **Python 3.12+** (for MLX)
- 🤖 **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

---

## 🚀 Quick Start (3 Commands)

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

`setup.sh` auto-detects your RAM, picks a model from the lineup, downloads it, installs the MLX server, and creates a `Claude Local.command` launcher on your Desktop.

**Then double-click `Claude Local.command`.** You're coding locally.

> 🐛 **If the launcher asks you to sign in to a Claude account:** your `claude` CLI is too old. The launchers pass `--bare` to force local-only API-key auth; older CLIs don't support it. Fix: `npm install -g @anthropic-ai/claude-code`

> 🛠️ **Note for contributors:** `setup.sh` installs the server as a **symlink** at `~/.local/mlx-native-server/server.py` pointing back at this repo's `proxy/server.py`. Edit the file in the repo, restart the server, done — one source of truth, no silent drift.

### Or do it manually

```bash
# 1. Set up the MLX virtualenv
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm

# 2. Pick a fighter and download (one time, ~18-75 GB)
bash scripts/download-and-import.sh gemma   # or 'llama' or 'qwen'

# 3. Start the server
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh

# 4. Launch Claude Code
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

---

## 🔧 How It Works

```
┌──────────────────────────────────────────────────┐
│              Your MacBook (M-series)             │
│                                                  │
│  📝 You type ──> 🤖 Claude Code                  │
│                      │                           │
│                      ▼                           │
│                 ⚡ MLX Server (port 4000)        │
│                      │                           │
│                      ▼                           │
│                 🥊 Local model ──> 🖥️  GPU        │
│                 (Gemma·Llama·Qwen)               │
│                      │                           │
│                      ▼                           │
│  📝 Answer <─── ✨ Clean response                │
│                                                  │
│         🔒 Nothing leaves this box. Ever.        │
└──────────────────────────────────────────────────┘
```

The server (`proxy/server.py`) is **one file, ~1000 lines**. It does six things:

1. 📦 **Loads the model** — Apple's MLX framework, native Metal GPU, unified memory. Handles Gemma's `RotatingKVCache` quirk automatically.
2. 🔌 **Speaks Anthropic API** — Claude Code thinks it's talking to Anthropic's cloud. It's not.
3. 🔧 **Translates tool use** — Three tool-call formats in and out: Gemma 4 native, Llama 3.3 raw JSON, and HuggingFace `<tool_call>` JSON (Qwen and others). All converted ↔ Anthropic `tool_use` blocks, with garbled-output recovery for small models.
4. 🧹 **Cleans the output** — A real-time `ThinkingFilter` strips `<think>` blocks token-by-token during generation, then `clean_response` handles stop markers and reasoning preamble.
5. ⚡ **Reuses prompt caches across requests** — Claude Code's system prompt doesn't get re-prefilled every turn. Huge speedup for short questions.
6. 🎯 **Code mode** — auto-detects Claude Code coding sessions, swaps the ~10K-token harness prompt for a slim ~150-token one, and strips verbose tool descriptions to name + parameter types. A **28× prompt reduction** that cuts prefill from ~60 s to ~2 s on Gemma 4 31B.

### 🛤️ The Journey

We didn't start here. Three generations in one night:

| Gen | What We Tried | Speed | 💡 What We Learned |
|:---:|---|:---:|---|
| 1️⃣ | Ollama + custom proxy | 30 tok/s | Ollama works but Claude Code can't talk to it directly |
| 2️⃣ | llama.cpp TurboQuant + proxy | 41 tok/s | TurboQuant compresses KV cache 4.9x, but the proxy is the bottleneck |
| 3️⃣ | **MLX native server** | **65 tok/s** | **Kill the proxy. Speak Anthropic API directly. 7.5x faster.** |
| 4️⃣ | **The lineup** | 65 / 15 / 7 tok/s | Three brains, one server — swap one env var to change the fighter |

---

## 🔒 Privacy + How the Data Flows

This is the part we're proudest of. **Your code never leaves your Mac.** Not for a model call. Not for telemetry. Not for "anonymous analytics". Not ever.

```
   ┌─────────────────────────────────────────────────────────────┐
   │                    🖥️  YOUR MACBOOK                          │
   │                                                             │
   │   📝 Your code ──> 🤖 Claude Code ──> ⚡ MLX Server          │
   │                     (localhost:4000)      │                 │
   │                                           ▼                 │
   │                    🧠 Local model ──> 🖥️  Apple GPU          │
   │                                                             │
   │             🚫 ZERO outbound network calls                  │
   │             🚫 ZERO telemetry                               │
   │             🚫 ZERO phone-home                              │
   └─────────────────────────────────────────────────────────────┘
                   │
                   ✗  ←  Nothing from our code crosses this line.
                   │
   ┌─────────────────────────────────────────────────────────────┐
   │                    ☁️  THE INTERNET                          │
   │                  (your code never goes here)                 │
   └─────────────────────────────────────────────────────────────┘
```

### 🔍 What We Audited (Every Component)

| Component | Source | Outbound calls | Verdict |
|-----------|--------|:---:|:---:|
| **server.py** (ours) | We wrote it line by line | **0** | ✅ Safe |
| **browser agent** | [nicedreamzapp/browser-agent](https://github.com/nicedreamzapp/browser-agent) — we wrote it | **0** (localhost CDP only) | ✅ Safe |
| **mlx-lm** | Apple ML team | **0** | ✅ Safe |
| **MLX framework** | Apple | **0** | ✅ Safe |
| **Model weights** | HuggingFace verified repos | **0** at runtime | ✅ Safe |
| **Claude Code CLI** | Anthropic (closed-source binary) | **0** with our launchers — `lsof`-verified | ✅ Safe |

> ✅ **Verified offline.** Claude Code's own binary previously reached out to `api.anthropic.com` on startup for telemetry, statsig feature flags, marketplace auto-install, and the autoupdater. The launchers plug all four channels via documented Anthropic env vars (thanks [@tadrianonet](https://github.com/tadrianonet), PR #32):
>
> ```bash
> CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
> DISABLE_AUTOUPDATER=1
> CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1
> CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
> ```
>
> **Verify it yourself:** run `lsof -p $(pgrep -f claude)` during a session — you'll see only `localhost:4000`. Run `lsof -i -P` while the server is up — nothing leaves your Mac.

> ⚠️ We **[removed LiteLLM](https://x.com/Tahseen_Rahman/status/2035501506242240520)** after supply-chain attack concerns. Every dependency was re-audited from scratch. If a package had unexplained network calls, it didn't ship.

### ✈️ When To Use This

| Situation | Use This? | Why |
|-----------|:---------:|-----|
| On a plane (no wifi) | ✅ | Full AI coding, no internet needed |
| NDA / sensitive client code | ✅ | Nothing leaves your machine — air-gapped, `lsof`-verified |
| Healthcare / legal / finance review | ✅ | 100% on-device, audit-friendly |
| Don't want API fees | ✅ | $0/month forever |
| Want fastest possible | ☁️ | Cloud Sonnet is still slightly faster |
| Need Claude-level reasoning | ☁️ | Local models are good, not Claude-level |

---

## 📊 Benchmarks

Three generations of optimization. Each one got faster.

### ⚡ Speed Comparison

| Generation | Approach | Speed | Real Claude Code task |
|---|---|---:|---:|
| 🐌 Gen 1 | Ollama + Proxy | 30 tok/s | 133 s |
| 🏃 Gen 2 | llama.cpp + Proxy | 41 tok/s | 133 s |
| 🚀 Gen 3 | **MLX Native (ours)** | **65 tok/s** | **17.6 s** |

### 🥊 Lineup Comparison

| Model | tok/s | RAM | Best For |
|---|:---:|:---:|---|
| 🟢 Gemma 4 31B Abliterated | ~15 | ~18 GB | Daily coding on a 64 GB Mac |
| 🟠 Llama 3.3 70B Abliterated | ~7 | ~70 GB | Hardest reasoning, full precision |
| 🔵 **Qwen 3.5 122B-A10B** | **65** | ~75 GB | Maximum throughput, MoE sparsity |

### ☁️ vs Cloud APIs

| | 🖥️ **Our Local Setup** | ☁️ Claude Sonnet | ☁️ Claude Opus |
|---|:---:|:---:|:---:|
| Speed | 65 tok/s | ~80 tok/s | ~40 tok/s |
| Monthly cost | **$0** 🎉 | $20-100+ | $20-100+ |
| Privacy | **100% local** 🔒 | Cloud | Cloud |
| Works offline | **Yes** ✈️ | No | No |

> 💡 Our local setup **beats cloud Opus on raw speed** (65 vs 40 tok/s) at $0/month. Qwen numbers measured on M5 Max 128 GB — full details in [BENCHMARKS.md](docs/BENCHMARKS.md).

---

## 🔧 Tool-Call Reliability

Local models don't format tool calls perfectly. They *want* to call a tool but mix XML and JSON syntax — Claude Code sees no valid tool call, re-prompts, and the model garbles it the same way again. The result: **infinite loops where the AI says "let me do that" but never does anything.**

We fixed this with 4 changes to `server.py`:

| Change | What | Why |
|--------|------|-----|
| **KV Cache** | 4-bit → 8-bit, quantization starts at token 1024 | Model retains conversation context |
| **Temperature** | 0.7 → 0.2 | Less randomness = more consistent tool formatting |
| **Garbled Recovery** | `recover_garbled_tool_json()` | Catches XML-in-JSON hybrids, infers tool names from parameter keys |
| **Retry Logic** | Up to 2 retries when tool intent is detected but parsing fails | Re-prompts with explicit formatting instructions |

🧪 **Results: 98/98 tests passed across 7 consecutive runs. Zero failures.** The multi-step scenario that used to trigger infinite loops — create 12 month folders, delete all but September, verify — now passes every time. Run it yourself:

```bash
python3 scripts/test_mlx_server.py
```

### ⚙️ Tuning

| Variable | Default | What It Does |
|----------|---------|-------------|
| `MLX_MODEL` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | Pick which fighter to load |
| `MLX_KV_BITS` | `8` | KV cache quantization bits (4 saves memory, 8 improves coherence) |
| `MLX_KV_QUANT_START` | `1024` | Token position where KV quantization begins |
| `MLX_TOOL_RETRIES` | `2` | Max retries when a garbled tool call is detected |
| `MLX_MAX_TOKENS` | `8192` | Max output tokens per response |
| `MLX_SUPPRESS_THINKING` | `1` | Skip the model's reasoning chain (~1 min/request saved). Set `0` to let it think. |
| `MLX_BROWSER_MODE` | `0` | Optimize for chrome-devtools MCP sessions — keeps only the 9 essential browser tools (~99% fewer tokens) |

---

## 🔌 MCP Servers — Claude Code's plugin ecosystem, 100% local

> **The only way to run Claude Code's full MCP plugin ecosystem 100% local on Apple Silicon.**

Most local-LLM proxies break MCP — they strip tool definitions, mangle `tool_use` blocks, or refuse to forward the streaming format Claude Code expects. This server passes tool definitions through to your local model and translates the responses back into Anthropic's format, across all three model families. From Claude Code's perspective it's talking to Anthropic. From your MCP server's perspective, nothing changed.

Wire servers up the normal Claude Code way:

```bash
# Filesystem — let the local model read/write a folder
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem ~/projects

# GitHub — issues, PRs, code search
claude mcp add github --env GITHUB_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github

# Web search — for when the local model needs fresh info
claude mcp add brave-search --env BRAVE_API_KEY=$BRAVE_API_KEY -- npx -y @modelcontextprotocol/server-brave-search
```

The whole 200+ server MCP universe works the same against your local Gemma or Qwen — just running on your machine instead of someone else's.

---

## 🎤 Hands-Free Voice Mode

Talk to your Mac. It talks back in your own cloned voice. **Nothing touches the internet in either direction** — most "AI voice" demos use cloud STT and cloud TTS; this runs both sides of the loop fully on-device.

```
┌─────────────────────────────────────────────────────┐
│              YOUR MACBOOK (M-series)                │
│                                                     │
│  🎙️  Your voice                                     │
│      ▼                                              │
│  🎧 listen   ← Apple SFSpeechRecognizer, on-device  │
│      ▼         continuous, stability-based cutoff   │
│  ⌨️  inject  ← AppleScript → Terminal window        │
│      ▼                                              │
│  🤖 claude → ⚡ MLX Server → 🥊 Gemma 4 31B          │
│      ▼                                              │
│  🔊 speak    ← cloned-voice TTS (Pocket TTS/Piper)  │
│      ▼                                              │
│  👂 You hear it — and keep talking                  │
│                                                     │
│   🔒 Your voice never leaves this box. Ever.        │
└─────────────────────────────────────────────────────┘
```

- 🎙️ **Speech-in** — a compiled Swift binary wraps Apple's on-device `SFSpeechRecognizer` in a continuous listening loop. End of utterance = transcript stable for 2.5s — way more robust than silence heuristics against fans or music.
- 🔊 **Speech-out** — `~/.local/bin/speak` wraps a cloned-voice TTS. Any TTS that takes text and plays audio slots in: macOS `say`, Piper, Pocket TTS.
- 🔁 **Feedback-loop prevention** — the listener auto-pauses during playback so the model never hears itself.
- 🛡️ **Production hardening** — 10-minute preventive recycle (dodges a known `SFSpeech` daemon wedge), queue-backlog detection. Runs unattended for hours.

**The two halves:**
- 🗣️ **Speak-and-think (this repo):** `launchers/Narrative Gemma.command` boots the MLX server with the narration persona (`NarrativeGemma/CLAUDE.md`) so Gemma narrates every tool call and result out loud.
- 🎧 **Listen-and-inject ([NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude), sibling repo):** the Swift listener, dispatch pipeline, and one-click `narrative-claude.sh` launcher.

```bash
# 1. This repo — MLX server + Narrative launcher
git clone https://github.com/nicedreamzapp/claude-code-local.git && cd claude-code-local && bash setup.sh

# 2. Sibling repo — the listening pipeline
git clone https://github.com/nicedreamzapp/NarrateClaude.git ~/NarrateClaude
cd ~/NarrateClaude && chmod +x dictation/bin/* narrative-claude.sh
./dictation/bin/dictation setup

# 3. Launch the full hands-free loop
bash ~/NarrateClaude/narrative-claude.sh
```

---

## 🌐 Browser Agent

A standalone agent that controls your **real Brave browser** via Chrome DevTools Protocol — powered entirely by local AI. Lives in its own repo: [`nicedreamzapp/browser-agent`](https://github.com/nicedreamzapp/browser-agent). The `Browser Agent.command` launcher here starts the MLX server, opens Brave with remote debugging, and drops you into the agent.

```
     📝 Your task
      ▼
 🤖 agent.py              ← autonomous browser agent (separate repo)
      ▼
 ⚡ MLX Server             ← local AI decides what to do
      ▼
 🌐 Brave (CDP port 9222) ← clicks, types, navigates your real browser
      ▼
 📊 Context Meter          ← color-coded memory usage after each step
```

---

## 📱 Control From Your Phone

You don't have to be at your Mac to use this. Text a command, get back a full video.

```
📱 Your iPhone                    💻 Your Mac
     │                                │
     │── "find me an article  ──────>│── imessage-receive.sh reads it
     │    and send me a video"        │── local model plans the task
     │                                │── Brave browser finds the article
     │                                │── speak narrates in your voice
     │                                │── Studio Record captures it all
     │<── 🎥 video in iMessage ──────│── imessage-send-video.sh ships it
     │                                │
   🛋️  From your couch            🖥️  At your desk
```

**Everything works — text, images, and video:**

| Command | What happens | You get |
|---|---|---|
| "summarize this article" | Local model reads + replies | 💬 Text |
| "send me a screenshot of X" | Claude screenshots | 📸 Image in iMessage |
| "screen record you doing Y" | Records + sends | 🎥 Video in iMessage |
| "make me a produced video" | Full edit pipeline | 🎬 Title card + subs |

**Full pipeline repo:** [nicedreamzapp/claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone)
→ Clone it, run `setup.sh`, fill in your phone number. Works with this local AI stack or Claude cloud. Docs in [IMESSAGE_MEDIA_PIPELINE.md](IMESSAGE_MEDIA_PIPELINE.md).

---

## 📁 What's In This Repo

```
📦 claude-code-local/
 ├── ⚡ proxy/
 │   └── server.py              ← MLX Native Anthropic Server with tool-call recovery (~1000 lines)
 ├── 🚀 launchers/
 │   ├── Claude Local.command    ← Default fighter — Claude Code + local model
 │   ├── Gemma 4 Code.command    ← 🟢 THE QUICK ONE
 │   ├── Llama 70B.command       ← 🟠 THE WISE ONE
 │   ├── Browser Agent.command   ← 🌐 Autonomous Brave browser control
 │   ├── Narrative Gemma.command ← 🎭 Auto-narration mode
 │   └── lib/claude-local-common.sh ← Shared: model-aware restart, cache resolver, health-wait
 ├── 🎭 NarrativeGemma/
 │   └── CLAUDE.md              ← Narration persona (sanitized, generic, opt-in)
 ├── 🛠️  scripts/
 │   ├── download-and-import.sh ← Download a fighter (`gemma` / `llama` / `qwen`)
 │   ├── persistent-download.sh ← Auto-retry downloader for big models
 │   ├── start-mlx-server.sh    ← Server start helper
 │   ├── test_mlx_server.py     ← Tool-call reliability test suite
 │   └── upload-mlx-quant.sh    ← Publish your own MLX-quantized uploads to HF
 ├── 📊 docs/
 │   └── BENCHMARKS.md          ← Detailed speed comparisons
 └── setup.sh                    ← One-command installer
```

---

## 🧩 The Local-First Stack

`claude-code-local` is the **brain**. It pairs with sibling repos — each stands alone, together they take Claude Code off the keyboard and off the screen:

| Repo | Role | What it does |
|---|---|---|
| 🤖 **claude-code-local** | Brain *(you are here)* | MLX Anthropic server · launcher lineup · tool-call translation |
| 🎤 [NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude) | Ears + Mouth | Talk to Claude, hear replies in your cloned voice — both directions on-device |
| 🌐 [browser-agent](https://github.com/nicedreamzapp/browser-agent) | Hands | Drives real Brave via CDP — iframes, Shadow DOM, ProseMirror |
| 📱 [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) | Remote | iPhone → Claude Code over iMessage; text/screenshots/videos back |
| 🛟 [claude-failover](https://github.com/nicedreamzapp/claude-failover) | Backstop | Keep cloud Claude primary, flip one command to local when limits pinch or Anthropic is down |

---

## 🛣️ What's Next

We ship fast and in public. If any of these excite you, hit **Watch** to get the release ping.

- 🟡 **Full Qwen 3.5 122B benchmark suite** — reliability, tool-call pass rate, long-context behavior vs Gemma
- 🟡 **Fully-local Whisper fallback** — alternative to the Apple `SFSpeechRecognizer` path for older Macs and non-English voices
- 🟡 **One-click DMG installer** — no terminal needed
- 🟡 **`MLX_MODEL=<hf-url>`** — point at any HuggingFace repo and auto-register a new fighter
- 🟡 **More fighters** — open to PRs adding launchers for DeepSeek, Mistral, Phi, anything MLX-compatible

> 💡 Want something that's not on this list? [**Open an issue →**](https://github.com/nicedreamzapp/claude-code-local/issues/new) Every serious request gets read and usually replied to within 24h.

## 🤝 Contributing

Ideas, bug reports, a new launcher for a model I don't run, a better code-mode prompt — open an issue or a PR, I read them all. Especially interested in: folks on older Apple Silicon (M1/M2, 16–36 GB) who know which models actually fit; anyone stress-testing the voice loop on different hardware or accents; TTS recipes beyond Pocket TTS (Piper, MLX-TTS, Kyutai Moshi); and edge cases I'll never hit on an M5 Max with 128 GB.

---

## 🙏 Credits

Built on the shoulders of giants:

| Project | What It Does | By |
|---------|-------------|-----|
| 🤖 [Claude Code](https://claude.ai/claude-code) | AI coding agent | Anthropic |
| 🍎 [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-examples) | Apple Silicon ML framework + inference | Apple |
| 🟢 [Gemma](https://blog.google/technology/developers/gemma-open-models/) | The 31B fighter (base weights) | Google DeepMind |
| 🟠 [Llama](https://llama.meta.com/) | The 70B fighter (base weights) | Meta |
| 🔵 [Qwen 3.5](https://qwenlm.github.io/) | The 122B fighter | Alibaba |
| 🐳 [ds4](https://github.com/antirez/ds4) | DeepSeek V4 Flash Metal engine | Antirez |
| 🔧 [huihui-ai](https://huggingface.co/huihui-ai) + [Babsie](https://huggingface.co/Babsie) | Abliterations we build on | — |
| 📖 [Abliteration explained](https://huggingface.co/blog/mlabonne/abliteration) | The technique | Maxime Labonne |

Tested on **Apple M5 Max** with **128 GB unified memory**.

Built by [Matt Macosko](https://x.com/NiceDreamzApps) in Arcata, CA — part of [Nice Dreamz LLC](https://nicedreamzwholesale.com). More open-source at [nicedreamzwholesale.com/software](https://nicedreamzwholesale.com/software/) · demos at [youtube.com/@nicedreamzapps](https://www.youtube.com/@nicedreamzapps).

<p>
  <a href="https://x.com/NiceDreamzApps"><img src="https://img.shields.io/badge/X-@NiceDreamzApps-000000?style=flat-square&logo=x&logoColor=white" alt="X"></a>
  <a href="https://www.youtube.com/@nicedreamzapps"><img src="https://img.shields.io/badge/YouTube-@nicedreamzapps-FF0000?style=flat-square&logo=youtube&logoColor=white" alt="YouTube"></a>
  <a href="https://github.com/nicedreamzapp"><img src="https://img.shields.io/badge/GitHub-@nicedreamzapp-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
</p>

---

<p align="center">
  <strong>📜 MIT License</strong> — Use it however you want.<br><br>
  💬 Builders hang out on <a href="https://discord.gg/ZdSqgAxUW">Discord</a> — share what you're building, swap MLX tips.<br><br>
  ⭐ <strong>Star this repo if it helped you!</strong> ⭐
</p>
