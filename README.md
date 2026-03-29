<p align="center">
  <h1 align="center">🧠 Claude Code Local</h1>
  <p align="center">
    <strong>Run a 122 billion parameter AI on your MacBook.<br>No cloud. No fees. No data leaves your machine.</strong>
  </p>
  <p align="center">
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/⚡_Speed-65_tok%2Fs-brightgreen?style=for-the-badge" alt="Speed"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/🚀_Claude_Code-17.6s_per_task-blue?style=for-the-badge" alt="Claude Code"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/📈_vs_llama.cpp-7.5x_faster-orange?style=for-the-badge" alt="7.5x faster"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/📜_License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
  </p>
</p>

---

## 🤔 What Is This?

Your MacBook has a powerful GPU built right into the chip. This project uses that GPU to run a **massive AI model** — the same kind that powers ChatGPT and Claude — **entirely on your computer**.

🚫 No internet needed
💰 No monthly subscription
🔒 No one sees your code or data
✅ Full Claude Code experience — write code, edit files, manage projects, control your browser

```
         📱 You (Mac or Phone)
          │
     🤖 Claude Code         ← the AI coding tool you know
          │
     ⚡ MLX Native Server    ← our server (200 lines of Python)
          │
     🧠 Qwen 3.5 122B       ← 122 billion parameter brain
          │
     🖥️ Apple Silicon GPU    ← your M-series chip does all the work
```

---

## 📱 Control From Your Phone

You don't have to be at your Mac to use this. We built a remote control pipeline:

```
📱 Your iPhone                    💻 Your Mac
     │                                │
     │── iMessage ──────────────────>│
     │                                │── Claude Code
     │                                │── MLX Server
     │                                │── Qwen 3.5 122B
     │                                │── (does the work)
     │<── iMessage response ────────│
     │                                │
   🛋️ From your couch            🖥️ At your desk
```

**How it works:**
- 📲 Send a message from your phone via iMessage
- 🤖 Claude Code receives it and runs the task on your local AI
- 💬 Response comes back to your phone
- ✈️ Works anywhere your Mac has power — even offline for the AI part

We built this **before** Anthropic shipped their Dispatch feature. Same concept, but ours uses iMessage and runs on your local model instead of cloud.

> 💡 **Pro tip:** Anthropic's Dispatch doesn't read your CLAUDE.md. Mention it in your message or it'll miss your custom setup. Our iMessage system doesn't have this problem.

---

## 📊 Benchmarks

We built and tested three different approaches. Each one got faster.

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

### 📋 Side-by-Side

| | 🐌 Ollama | 🏃 llama.cpp + TurboQuant | 🚀 **MLX Native (ours)** |
|---|:---:|:---:|:---:|
| **Speed** | 30 tok/s | 41 tok/s | **65 tok/s** |
| **Claude Code task** | 133s | 133s | **17.6s** |
| **Needs a proxy?** | ❌ Yes | ❌ Yes | ✅ **No** |
| **Lines of code** | N/A | N/A (C++ fork) | **~200 Python** |
| **Apple native?** | ❌ Generic | ❌ Ported | ✅ **MLX** |

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
| M1/M2/M3/M4 Pro | 18-36 GB | 🟠 Medium models (35B) |
| M2/M3/M4/M5 Max | 64-128 GB | 🟢 **Large models (122B)** |
| M2/M3/M4 Ultra | 128-192 GB | 🔵 Multiple large models |

Also need:
- 🐍 **Python 3.12+** (for MLX)
- 🤖 **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

---

## 🚀 Quick Start (4 Steps)

### 1️⃣ Set up Python environment

```bash
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm
```

### 2️⃣ Download the AI model

First run downloads ~50 GB (one time only):

```bash
~/.local/mlx-server/bin/python3 -c "
from mlx_lm.utils import load
load('mlx-community/Qwen3.5-122B-A10B-4bit')
print('Done!')
"
```

### 3️⃣ Start the server

```bash
~/.local/mlx-server/bin/python3 proxy/server.py
```

### 4️⃣ Launch Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

> 💡 **Or just double-click** `Claude Local.command` on your Desktop. It does all of this automatically.

---

## 🔧 How It Works

```
┌──────────────────────────────────────────────────┐
│              Your MacBook (M5 Max)               │
│                                                  │
│  📱 You type ──> 🤖 Claude Code                  │
│                      │                           │
│                      ▼                           │
│                 ⚡ MLX Server (port 4000)         │
│                      │                           │
│                      ▼                           │
│                 🧠 Qwen 3.5 122B ──> 🖥️ GPU      │
│                      │                           │
│                      ▼                           │
│  📱 Answer <─── ✨ Clean response                │
│                                                  │
│         🔒 Nothing leaves this box. Ever.        │
└──────────────────────────────────────────────────┘
```

The server (`proxy/server.py`) is **one file, ~600 lines**. It does four things:

1. 📦 **Loads the model** — Apple's MLX framework, native Metal GPU, unified memory
2. 🔌 **Speaks Anthropic API** — Claude Code thinks it's talking to Anthropic's cloud. It's not.
3. 🔧 **Translates tool use** — Converts Anthropic tool definitions ↔ Qwen's native format, parses `<tool_call>` tags back into Anthropic `tool_use` blocks
4. 🧹 **Cleans the output** — Qwen thinks out loud in `<think>` tags. We strip those.

---

## 🌐 Browser Agent

A standalone browser agent (`agent.py`) that controls your **real Brave browser** via Chrome DevTools Protocol — powered entirely by local AI. No Claude Code wrapper needed.

```
         📝 Your task
          │
     🤖 agent.py              ← autonomous browser agent
          │
     ⚡ MLX Server (Qwen 122B) ← local AI decides what to do
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

## ✈️ When To Use This

| Situation | Use This? | Why |
|-----------|:---------:|-----|
| On a plane | ✅ | Full AI coding, no internet needed |
| Sensitive client code | ✅ | Nothing leaves your machine |
| Don't want API fees | ✅ | $0/month forever |
| Want fastest possible | ☁️ | Cloud API is still faster |
| Need Claude-level reasoning | ☁️ | Local model is good, not Claude-level |
| Controlling from phone | ✅ | iMessage pipeline works offline |

---

## 📁 What's In This Repo

```
📦 claude-code-local/
 ├── ⚡ proxy/
 │   └── server.py              ← MLX server with Anthropic API + tool use (~600 lines)
 ├── 🌐 agent.py                ← Standalone browser agent with context memory pipeline
 ├── 🚀 launchers/
 │   ├── Claude Local.command    ← Double-click to start Claude Code locally
 │   └── Browser Agent.command   ← Double-click for autonomous browser control
 ├── 🛠️ scripts/
 │   ├── download-and-import.sh  ← Download models
 │   ├── persistent-download.sh  ← Auto-retry downloader
 │   └── start-mlx-server.sh    ← Alternative config
 ├── 📊 docs/
 │   ├── BENCHMARKS.md           ← Detailed speed comparisons
 │   └── TWITTER-THREAD.md       ← Social media content
 └── setup.sh                    ← One-command installer
```

---

## 🔒 Security

We audited every component before running it:

| Component | Source | Network Calls | Verdict |
|-----------|--------|:---:|:---:|
| **server.py** | We wrote it | 0 | ✅ Safe |
| **MLX framework** | Apple | 0 | ✅ Safe |
| **Qwen 3.5 model** | HuggingFace verified | 0 | ✅ Safe |

🚫 No telemetry
🚫 No analytics
🚫 No phone-home
🚫 No sketchy pip packages

> ⚠️ We [removed LiteLLM](https://x.com/Tahseen_Rahman/status/2035501506242240520) after supply chain attack concerns were raised. Every dependency was audited.

---

## 🛤️ The Journey

We didn't start here. We went through three generations in one night:

| Gen | What We Tried | Speed | 💡 What We Learned |
|:---:|---|:---:|---|
| 1️⃣ | Ollama + custom proxy | 30 tok/s | Ollama works but Claude Code can't talk to it directly |
| 2️⃣ | llama.cpp TurboQuant + proxy | 41 tok/s | TurboQuant compresses KV cache 4.9x, but the proxy is the bottleneck |
| 3️⃣ | **MLX native server** | **65 tok/s** | **Kill the proxy. Speak Anthropic API directly. 7.5x faster.** |

> 🎯 Each generation taught us something. The final insight — the proxy was the bottleneck, not the model — changed everything.

---

## 🙏 Credits

Built on the shoulders of giants:

| Project | What It Does | By |
|---------|-------------|-----|
| 🤖 [Claude Code](https://claude.ai/claude-code) | AI coding agent | Anthropic |
| 🍎 [MLX](https://github.com/ml-explore/mlx) | Apple Silicon ML framework | Apple |
| 📦 [mlx-lm](https://github.com/ml-explore/mlx-examples) | Model loading + inference | Apple |
| 🧠 [Qwen 3.5](https://qwenlm.github.io/) | The 122B model | Alibaba |
| ⚡ [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | KV cache compression research | Google Research |

Tested on **Apple M5 Max** with **128 GB unified memory**.

---

<p align="center">
  <strong>📜 MIT License</strong> — Use it however you want.<br><br>
  ⭐ <strong>Star this repo if it helped you!</strong> ⭐
</p>
