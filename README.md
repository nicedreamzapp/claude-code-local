<p align="center">
  <h1 align="center">🧠⚡ Claude Code Local — The Lineup</h1>
  <p align="center">
    <strong>Three local AI brains. Four modes. One MacBook. Zero cloud.<br>Pick your fighter and run Claude Code 100% on-device.</strong>
  </p>
  <p align="center">
    <a href="#-the-lineup--pick-your-fighter"><img src="https://img.shields.io/badge/🥊_Lineup-3_Models-red?style=for-the-badge" alt="3 Models"></a>
    <a href="#-the-modes"><img src="https://img.shields.io/badge/🎮_Modes-4-purple?style=for-the-badge" alt="4 Modes"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/⚡_Top_Speed-65_tok%2Fs-brightgreen?style=for-the-badge" alt="Speed"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/🚀_Claude_Code-17.6s_per_task-blue?style=for-the-badge" alt="Claude Code"></a>
    <a href="#-safety--how-the-data-flows"><img src="https://img.shields.io/badge/🔒_Privacy-100%25_Local-success?style=for-the-badge" alt="100% Local"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/📜_License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
  </p>
</p>

---

## 🥊 The Lineup — Pick Your Fighter

We started with one model. Now we ship a **roster**. Same MLX server, same Anthropic API, swap one env var and you swap the brain.

```
   ╔══════════════════════╦═══════════════════════╦══════════════════════╗
   ║  🟢 GEMMA 4 31B      ║   🟠 LLAMA 3.3 70B    ║  🔵 QWEN 3.5 122B    ║
   ║   ABLITERATED        ║    ABLITERATED        ║    A10B MoE          ║
   ║  ──────────────────  ║  ───────────────────  ║  ──────────────────  ║
   ║   THE QUICK ONE      ║    THE WISE ONE       ║    THE BEAST         ║
   ║                      ║                       ║                      ║
   ║  Speed   ~15 tok/s   ║  Speed   ~7 tok/s     ║  Speed  65 tok/s 🚀  ║
   ║  Params  31 B dense  ║  Params  71 B dense   ║  Params 122 B / 10B  ║
   ║  Quant   4-bit IT    ║  Quant   8-bit affine ║  Quant  4-bit MoE    ║
   ║  RAM     ~18 GB      ║  RAM     ~75 GB       ║  RAM    ~75 GB       ║
   ║  Disk    18 GB       ║  Disk    75 GB        ║  Disk   65 GB        ║
   ║                      ║                       ║                      ║
   ║                      ║  ⭐ Uploaded by us!    ║                      ║
   ║                      ║                       ║                      ║
   ║  🎯 Daily coding     ║  🎯 Hardest reasoning ║  🎯 Max throughput   ║
   ║  💪 Fits 64 GB Mac   ║  💪 Full precision    ║  💪 Active sparsity  ║
   ╚══════════════════════╩═══════════════════════╩══════════════════════╝
```

| Pick This Model | If You Want… | Min RAM | Launcher |
|---|---|:---:|---|
| 🟢 **Gemma 4 31B** | Daily coding, low RAM, fast loop | 32 GB | `Gemma 4 Code.command` |
| 🟠 **Llama 3.3 70B** ⭐ | Hardest reasoning at full 8-bit precision | 96 GB | `Llama 70B.command` |
| 🔵 **Qwen 3.5 122B** | Max tok/s, biggest brain | 96 GB | `Claude Local.command` |

> 💡 **Fun fact:** Qwen wins raw speed because it's an MoE — only 10B of 122B params activate per token. Llama 70B is the slowest *and* the smartest because it's full-precision dense weights. Gemma is the lightweight champ that fits where the others can't.

### ⭐ Our Own Abliterated Upload

The Llama 3.3 70B in this lineup isn't from a generic mirror — **we packaged and uploaded our own 8-bit MLX abliterated build** to HuggingFace so anyone running this repo can pull it with one command:

```bash
MLX_MODEL=divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx \
  bash scripts/start-mlx-server.sh
```

| | |
|---|---|
| 🤗 **HuggingFace** | [`divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx`](https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx) |
| 📐 **Quant** | 8-bit affine, group size 64 |
| 💾 **Disk** | ~75 GB (15 safetensors shards) |
| 🧠 **Params** | 71 B dense |
| 📏 **Context** | 128 K tokens |
| 🔓 **Abliteration base** | [huihui-ai abliterated build](https://huggingface.co/huihui-ai) of Meta's Llama 3.3 70B Instruct ([what abliteration means](https://huggingface.co/blog/mlabonne/abliteration)) |
| 🍎 **MLX conversion + 8-bit pack** | by us — chosen to preserve quality over minimal footprint |

> ⚠️ **Use it responsibly.** "Abliterated" suppresses the model's built-in refusal direction so it doesn't refuse benign-but-edgy requests. It is **not** a general capability upgrade, and you remain bound by the upstream Llama 3.3 license.

---

## 🎮 The Modes

Four ways to run the lineup. Each one is a double-clickable launcher in `launchers/`.

```
   ┌─────────────────────────────┬─────────────────────────────┐
   │  🤖 CODE MODE               │  🌐 BROWSER MODE             │
   │  ──────────                 │  ─────────────               │
   │  Full Claude Code + local   │  Autonomous Brave agent      │
   │  model. Same UX, no cloud.  │  via Chrome DevTools         │
   │  → Claude Local.command     │  Protocol. Clicks, types,    │
   │  → Gemma 4 Code.command     │  navigates iframes & Shadow  │
   │  → Llama 70B.command        │  DOM with 0 cloud calls.     │
   │                             │  → Browser Agent.command     │
   ├─────────────────────────────┼─────────────────────────────┤
   │  🎭 NARRATIVE MODE          │  📱 PHONE MODE               │
   │  ─────────────              │  ────────────                │
   │  Gemma speaks every reply   │  Text from your couch.       │
   │  out loud through your TTS  │  iMessage in, code/video     │
   │  or cloned voice. Includes  │  out. Full screen-record +   │
   │  a sanitized persona file.  │  send-back pipeline.         │
   │  → Narrative Gemma.command  │  → ~/.claude/imessage-*.sh   │
   └─────────────────────────────┴─────────────────────────────┘
```

| Mode | What it does | Launcher |
|---|---|---|
| 🤖 **Code** | Run Claude Code with a local model — same UX, no API key | `Claude Local.command`, `Gemma 4 Code.command`, `Llama 70B.command` |
| 🌐 **Browser** | Local AI controls real Brave browser via Chrome DevTools | `Browser Agent.command` |
| 🎭 **Narrative** | Every reply spoken aloud through your TTS / cloned voice | `Narrative Gemma.command` |
| 📱 **Phone** | iMessage in → text/image/video out, full pipeline | `~/.claude/imessage-*.sh` |

---

## 🤔 What Is This?

Your MacBook has a powerful GPU built right into the chip. This project uses that GPU to run **massive AI models — the same kind that power ChatGPT and Claude — entirely on your computer**.

🚫 No internet needed
💰 No monthly subscription
🔒 No one sees your code or data
✅ Full Claude Code experience — write code, edit files, manage projects, control your browser, even narrate replies in your own voice

```
         📱 You (Mac or Phone)
          │
     🤖 Claude Code           ← the AI coding tool you know
          │
     ⚡ MLX Native Server      ← our server (~800 lines of Python)
          │
     🥊 Pick your fighter     ← Gemma 4 31B · Llama 3.3 70B · Qwen 3.5 122B
          │
     🖥️  Apple Silicon GPU    ← your M-series chip does all the work
```

---

## 🔒 Safety + How the Data Flows

This is the part we're proudest of. **Your code never leaves your Mac.** Not for a model call. Not for telemetry. Not for "anonymous analytics". Not ever.

### 🛡️ The Data-Flow Diagram

```
   ┌─────────────────────────────────────────────────────────────┐
   │                    🖥️  YOUR MACBOOK                          │
   │                                                             │
   │   📝 Your code         ┌────────────────────┐               │
   │       │                │  🤖 Claude Code     │               │
   │       └───────────────▶│  (CLI on your Mac)  │               │
   │                        └────────┬───────────┘               │
   │                                 │  HTTP localhost:4000       │
   │                                 ▼                            │
   │                        ┌────────────────────┐               │
   │                        │  ⚡ MLX Server      │               │
   │                        │  (Python, ours)    │               │
   │                        └────────┬───────────┘               │
   │                                 │  Metal API                 │
   │                                 ▼                            │
   │                        ┌────────────────────┐               │
   │                        │  🧠 Local model     │               │
   │                        │  (Gemma·Llama·Qwen)│               │
   │                        └────────┬───────────┘               │
   │                                 │                            │
   │                                 ▼                            │
   │                        ┌────────────────────┐               │
   │                        │  🖥️  Apple GPU      │               │
   │                        │  (unified memory)  │               │
   │                        └────────────────────┘               │
   │                                                             │
   │             🚫 ZERO outbound network calls                  │
   │             🚫 ZERO telemetry                               │
   │             🚫 ZERO phone-home                              │
   └─────────────────────────────────────────────────────────────┘
                   │
                   ✗  ←  Nothing crosses this line. Ever.
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
| **agent.py** (browser agent) | We wrote it | **0** (talks to localhost CDP only) | ✅ Safe |
| **mlx-lm** | Apple ML team | **0** | ✅ Safe |
| **MLX framework** | Apple | **0** | ✅ Safe |
| **Model weights** | HuggingFace verified mlx-community repos | **0** at runtime | ✅ Safe |
| **iMessage scripts** | Pure shell + AppleScript | localhost only (Studio Record port 17494) | ✅ Safe |

### 🚫 What We Ripped Out

> ⚠️ We **[removed LiteLLM](https://x.com/Tahseen_Rahman/status/2035501506242240520)** after supply-chain attack concerns. Every dependency was re-audited from scratch. If a package had unexplained network calls, it didn't ship.

### ✅ What This Means in Practice

| Scenario | Cloud Claude | This Repo |
|---|:---:|:---:|
| Working with NDA / proprietary code | ❌ Risky | ✅ Air-gapped |
| Coding on a plane (no wifi) | ❌ Doesn't work | ✅ Works |
| Running on a kill-switch firewall | ❌ Blocked | ✅ Works |
| Healthcare / legal / finance review | ⚠️ Compliance burden | ✅ Stays on-device |
| Worry about training-data leakage | ⚠️ Trust required | ✅ Mathematically impossible |

> 🔒 **The math is simple:** if there are no outbound HTTP calls, your data cannot leak. We grep'd every file for `requests`, `urllib`, `urlopen`, `httpx`, `socket.connect` — the only network calls in the entire codebase are to `localhost`. Run `lsof -i -P` while it's running. You'll see nothing leaving your Mac.

---

## 📊 Benchmarks

Three generations of optimization. Each one got faster.

### ⚡ Speed Comparison

```
                         Tokens per Second
  🐌 Ollama (Gen 1)      ██████████████████████████████ 30 tok/s
  🏃 llama.cpp (Gen 2)   █████████████████████████████████████████ 41 tok/s
  🚀 MLX Native (Gen 3)  ████████████████████████████████████████████████████████████████ 65 tok/s
```

### ⏱️ Real-World Claude Code Task

How long to ask Claude Code to write a function:

```
  😴 Ollama + Proxy          ████████████████████████████████████████████ 133 seconds
  😐 llama.cpp + Proxy       ████████████████████████████████████████████ 133 seconds
  🔥 MLX Native (no proxy)   ██████ 17.6 seconds

                              7.5x faster ⚡
```

### 📋 Three-Generation Side-by-Side

| | 🐌 Ollama | 🏃 llama.cpp + TurboQuant | 🚀 **MLX Native (ours)** |
|---|:---:|:---:|:---:|
| **Speed** | 30 tok/s | 41 tok/s | **65 tok/s** |
| **Claude Code task** | 133s | 133s | **17.6s** |
| **Needs a proxy?** | ❌ Yes | ❌ Yes | ✅ **No** |
| **Lines of code** | N/A | N/A (C++ fork) | **~800 Python** |
| **Apple native?** | ❌ Generic | ❌ Ported | ✅ **MLX** |

### 🥊 Lineup Comparison

| Model | tok/s | RAM | Best For |
|---|:---:|:---:|---|
| 🟢 Gemma 4 31B Abliterated | ~15 | ~18 GB | Daily coding on a 64 GB Mac |
| 🟠 Llama 3.3 70B Abliterated | ~7 | ~70 GB | Hardest reasoning, full precision |
| 🔵 **Qwen 3.5 122B-A10B** | **65** | ~75 GB | Maximum throughput, MoE sparsity |

> Qwen 122B numbers are measured on M5 Max 128 GB. Gemma and Llama are observed real-world approximations. Full benchmarks for all three pending — see [BENCHMARKS.md](docs/BENCHMARKS.md).

### ☁️ vs Cloud APIs

| | 🖥️ **Our Local Setup** | ☁️ Claude Sonnet | ☁️ Claude Opus |
|---|:---:|:---:|:---:|
| Speed | 65 tok/s | ~80 tok/s | ~40 tok/s |
| Monthly cost | **$0** 🎉 | $20-100+ | $20-100+ |
| Privacy | **100% local** 🔒 | Cloud | Cloud |
| Works offline | **Yes** ✈️ | No | No |
| Data leaves your Mac | **Never** | Always | Always |

> 💡 Our local setup **beats cloud Opus on raw speed** (65 vs 40 tok/s) at $0/month.

---

## 🔧 Tool-Call Reliability (v2 — March 2026)

Local models don't format tool calls perfectly. They *want* to call a tool but mix XML and JSON syntax. Claude Code sees no valid tool call, re-prompts, and the model does it again. The result: **infinite loops where the AI says "let me do that" but never actually does anything.**

We fixed this. Here's what was happening and what we did about it.

### 🐛 The Problem

The model was generating garbled tool calls like this:
```
<tool_call>
<function=Bash><parameter=command>rm -rf /tmp/old</parameter></function>
</tool_call>
```

Instead of the correct JSON format Claude Code expects:
```json
<tool_call>
{"name": "Bash", "arguments": {"command": "rm -rf /tmp/old"}}
</tool_call>
```

The JSON parser choked, Claude Code saw no tool call, re-prompted the model, and the model garbled it the exact same way again — creating an infinite loop.

### ✅ The Fix (4 changes to `server.py`)

| Change | What | Why |
|--------|------|-----|
| **KV Cache** | 4-bit → 8-bit, quantization starts at token 1024 | Model retains conversation context instead of "forgetting" earlier messages |
| **Temperature** | 0.7 → 0.2 | Less randomness = more consistent tool formatting |
| **Garbled Recovery** | New `recover_garbled_tool_json()` function | Catches XML-in-JSON hybrids, `<function=X><parameter=Y>` inside `<tool_call>` tags, and infers tool names from parameter keys |
| **Retry Logic** | Up to 2 retries when tool intent is detected but parsing fails | Re-prompts with explicit formatting instructions before giving up |

### 🧪 Test Results

We built an automated test suite (`scripts/test_mlx_server.py`) that sends real Anthropic API requests to the server simulating multi-step tasks — the exact kind that were failing before.

```
Test Suite: 14 tests per run
─────────────────────────────
  ✅ Simple Bash commands
  ✅ Directory creation (mkdir -p)
  ✅ File reading (Read tool)
  ✅ Complex Bash with pipes
  ✅ File editing (Edit tool with find/replace)
  ✅ Multi-tool sequences (Glob → Read)
  ✅ 5 rapid-fire sequential commands
  ✅ Multi-step calendar scenario (create → delete → verify)
```

**Results: 98/98 tests passed across 7 consecutive runs. Zero failures.**

The multi-step calendar scenario — create 12 month folders, delete all but September, verify — was the exact task that triggered infinite loops before the fix. Now it passes every time.

```bash
# Run the test suite yourself:
python3 scripts/test_mlx_server.py
```

### ⚙️ Tuning

You can override defaults with environment variables:

| Variable | Default | What It Does |
|----------|---------|-------------|
| `MLX_MODEL` | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | Pick which fighter to load |
| `MLX_KV_BITS` | `8` | KV cache quantization bits (4 saves memory, 8 improves coherence) |
| `MLX_KV_QUANT_START` | `1024` | Token position where KV quantization begins |
| `MLX_TOOL_RETRIES` | `2` | Max retries when a garbled tool call is detected |
| `MLX_MAX_TOKENS` | `8192` | Max output tokens per response |

---

## 📱 Control From Your Phone — Full Media Pipeline

You don't have to be at your Mac to use this. Text a command, get back a full video.

```
📱 Your iPhone                    💻 Your Mac
     │                                │
     │── "find me an article  ──────>│── imessage-receive.sh reads it
     │    and send me a video"        │── local model plans the task
     │                                │── Brave browser finds the article
     │                                │── speak narrates in your voice
     │                                │── Studio Record captures it all
     │                                │── build_production_video.py edits it
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
→ Clone it, run `setup.sh`, fill in your phone number. Works with this local AI stack or Claude cloud.

We built this **before** Anthropic shipped their Dispatch feature. Same concept, but ours uses iMessage, runs on your local model, and can send back media — not just text.

> 💡 **Pro tip:** Anthropic's Dispatch doesn't read your CLAUDE.md. Mention it in your message or it'll miss your custom setup. Our iMessage system doesn't have this problem.

---

## 💡 How We Got Here

Most people trying to run Claude Code locally hit the same wall:

> Claude Code speaks **Anthropic API**. Local models speak **OpenAI API**. Different languages. 🤷

So everyone builds a **proxy** to translate between them. That proxy adds latency, complexity, and breaks things.

**We took a different approach:**

| 🐌 What everyone else does | 🚀 What we did |
|---|---|
| Claude Code → **Proxy** → Ollama → Model | Claude Code → **Our Server** → Model |
| 3 processes, 2 API translations | **1 process, 0 translations** |
| 133 seconds per task | **17.6 seconds per task** |

> 🎯 That one change — **eliminating the proxy** — made it **7.5x faster**.

---

## 💻 What You Need

| Your Mac | RAM | What You Can Run |
|----------|-----|-------------------|
| M1/M2/M3/M4 (base) | 8-16 GB | 🟡 Small models (4B) |
| M1/M2/M3/M4 Pro | 18-36 GB | 🟠 Gemma 4 31B (tight) |
| M2/M3/M4/M5 Max | 64-128 GB | 🟢 **Gemma 4 31B** + 🔵 Qwen 3.5 122B |
| M2/M3/M4 Ultra | 128-192 GB | 🔵 Multiple large models, all three fighters |

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

> 💡 **Or just double-click a launcher** in `launchers/`. They do all of this automatically.

---

## 🔧 How It Works

```
┌──────────────────────────────────────────────────┐
│              Your MacBook (M5 Max)               │
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

The server (`proxy/server.py`) is **one file, ~800 lines**. It does four things:

1. 📦 **Loads the model** — Apple's MLX framework, native Metal GPU, unified memory
2. 🔌 **Speaks Anthropic API** — Claude Code thinks it's talking to Anthropic's cloud. It's not.
3. 🔧 **Translates tool use** — Converts Anthropic tool definitions ↔ HuggingFace `<tool_call>` format, parses tool calls back into Anthropic `tool_use` blocks
4. 🧹 **Cleans the output** — Local models think out loud in `<think>` tags. We strip those.

---

## 🌐 Browser Agent

A standalone browser agent (`agent.py`) that controls your **real Brave browser** via Chrome DevTools Protocol — powered entirely by local AI. No Claude Code wrapper needed.

```
         📝 Your task
          │
     🤖 agent.py              ← autonomous browser agent
          │
     ⚡ MLX Server             ← local AI decides what to do
     (Gemma · Llama · Qwen)
          │
     🌐 Brave (CDP port 9222) ← clicks, types, navigates your real browser
          │
     📊 Context Meter          ← shows memory usage so you know its limits
```

**Context memory pipeline** — the agent doesn't forget what it's doing:

| | 🐌 Old Behavior | 🚀 New Pipeline |
|---|---|---|
| **Memory** | Hard drop after 5 steps | Smart trim at 60% of 32K budget |
| **When trimming** | Deletes old steps entirely | Compresses into summary |
| **Original task** | Lost after step 6+ | Re-injected every cycle |
| **Visibility** | None — flying blind | Color-coded context meter |
| **Response tokens** | 1,024 | 2,048 |

The context meter shows green/yellow/red after each step:
```
  Step 5 snapshot() 2.2s
         → [101] heading "The Best Coffee Cake Recipe"...
  [Context: 18% ████░░░░░░░░░░░░░░░░ 6K/32K tokens]    ← green = plenty of room
```

> 💡 **Double-click** `Browser Agent.command` to launch. It starts the MLX server, opens Brave with remote debugging, and drops you into the agent.

---

## 🎭 Narrative Mode

A new mode where **every reply gets spoken aloud** through your speakers. Pair Gemma 4 31B with a TTS CLI (your cloned voice, ElevenLabs, Piper, or just macOS `say`) and the model narrates everything it's doing in real time — "opening your notes file now", "found 12 TODOs across 4 files", etc.

```
   You ask a question
        │
        ▼
   🎭 Gemma 4 31B (narrative persona loaded)
        │
        ├──> 🔊 ~/.local/bin/speak "Sure, opening your notes file now."
        │
        ├──> 📂 [Read tool call]
        │
        ├──> 🔊 ~/.local/bin/speak "Got it — your notes are about the auth migration."
        │
        └──> 💬 Text reply: "Read it. Want a summary?"
```

The persona file (`NarrativeGemma/CLAUDE.md`) is loaded as a system prompt at launch. It enforces the **speak-first-reply-second** rule on every turn.

**No fancy TTS?** The persona doc shows you how to stub `~/.local/bin/speak` in three lines using macOS's built-in `say` command.

> 💡 **Double-click** `Narrative Gemma.command` to launch. It boots Gemma 4 31B, injects the narration rules into the system prompt, and opens Claude Code in narration mode.

---

## ✈️ When To Use This

| Situation | Use This? | Why |
|-----------|:---------:|-----|
| On a plane | ✅ | Full AI coding, no internet needed |
| Sensitive client code | ✅ | Nothing leaves your machine |
| Don't want API fees | ✅ | $0/month forever |
| Want fastest possible | ☁️ | Cloud Sonnet is still slightly faster |
| Need Claude-level reasoning | ☁️ | Local models are good, not Claude-level |
| Controlling from phone | ✅ | iMessage pipeline works offline |
| Healthcare / legal / finance review | ✅ | 100% on-device, audit-friendly |

---

## 📁 What's In This Repo

```
📦 claude-code-local/
 ├── ⚡ proxy/
 │   ├── server.py              ← MLX Native Anthropic Server with tool-call recovery (~800 lines)
 │   └── proxy.py               ← LEGACY Ollama path (kept for users on the old route)
 ├── 🌐 agent.py                ← Standalone browser agent with context memory pipeline
 ├── 🚀 launchers/
 │   ├── Claude Local.command    ← Default fighter — Claude Code + local model
 │   ├── Gemma 4 Code.command    ← 🟢 THE QUICK ONE
 │   ├── Llama 70B.command       ← 🟠 THE WISE ONE
 │   ├── Browser Agent.command   ← 🌐 Autonomous Brave browser control
 │   └── Narrative Gemma.command ← 🎭 Auto-narration mode
 ├── 🎭 NarrativeGemma/
 │   └── CLAUDE.md              ← Narration persona (sanitized, generic, opt-in)
 ├── 🛠️  scripts/
 │   ├── download-and-import.sh ← Download a fighter (`gemma` / `llama` / `qwen`)
 │   ├── persistent-download.sh ← Auto-retry downloader for big models
 │   ├── start-mlx-server.sh    ← Server start helper
 │   └── test_mlx_server.py     ← Tool-call reliability test suite
 ├── 📊 docs/
 │   ├── BENCHMARKS.md          ← Detailed speed comparisons
 │   └── TWITTER-THREAD.md      ← Social media content
 ├── 📱 IMESSAGE_MEDIA_PIPELINE.md ← Phone control + media sending docs
 └── setup.sh                    ← One-command installer
```

---

## 🛤️ The Journey

We didn't start here. We went through three generations in one night:

| Gen | What We Tried | Speed | 💡 What We Learned |
|:---:|---|:---:|---|
| 1️⃣ | Ollama + custom proxy | 30 tok/s | Ollama works but Claude Code can't talk to it directly |
| 2️⃣ | llama.cpp TurboQuant + proxy | 41 tok/s | TurboQuant compresses KV cache 4.9x, but the proxy is the bottleneck |
| 3️⃣ | **MLX native server** | **65 tok/s** | **Kill the proxy. Speak Anthropic API directly. 7.5x faster.** |
| 4️⃣ | **The lineup** | varies | One server, three brains. Pick your fighter. |

> 🎯 Each generation taught us something. Killing the proxy made it fast. Adding the lineup made it flexible.

---

## 🙏 Credits

Built on the shoulders of giants:

| Project | What It Does | By |
|---------|-------------|-----|
| 🤖 [Claude Code](https://claude.ai/claude-code) | AI coding agent | Anthropic |
| 🍎 [MLX](https://github.com/ml-explore/mlx) | Apple Silicon ML framework | Apple |
| 📦 [mlx-lm](https://github.com/ml-explore/mlx-examples) | Model loading + inference | Apple |
| 🟢 [Gemma](https://blog.google/technology/developers/gemma-open-models/) | The 31B fighter | Google DeepMind |
| 🟠 [Llama](https://llama.meta.com/) | The 70B fighter (base weights) | Meta |
| ⭐ [Llama 3.3 70B Abliterated 8-bit MLX](https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx) | **Our own MLX-packed abliterated upload** — the THE WISE ONE in the lineup | divinetribe (us) |
| 🔧 [huihui-ai](https://huggingface.co/huihui-ai) | Original abliteration of Llama 3.3 70B Instruct | huihui-ai |
| 📖 [Abliteration explained](https://huggingface.co/blog/mlabonne/abliteration) | The technique we built on | Maxime Labonne |
| 🔵 [Qwen 3.5](https://qwenlm.github.io/) | The 122B fighter | Alibaba |
| ⚡ [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | KV cache compression research | Google Research |

Tested on **Apple M5 Max** with **128 GB unified memory**.

---

<p align="center">
  <strong>📜 MIT License</strong> — Use it however you want.<br><br>
  ⭐ <strong>Star this repo if it helped you!</strong> ⭐
</p>
