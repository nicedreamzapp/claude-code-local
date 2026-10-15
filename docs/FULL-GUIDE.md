> 📖 **This is the full technical guide** — every benchmark, model table and design note that used to be the
> main README. The friendly quick version is the [README](../README.md). Only relative links were adjusted
> so they still work from this folder.

<p align="center">
  <h1 align="center">🧠⚡ Claude Code Local</h1>
  <p align="center">
    <strong>Run Claude Code 100% on-device with local AI on Apple Silicon.<br>No cloud, no API key, no proxy — an MLX-native server that speaks the Anthropic API.<br>🥊 Pick your model: Qwen 3.8 27B · Gemma 4 31B · Gemma 4 12B · Hermes 4 14B · Muse-Glimmer 30B · Nemotron 3 Nano Omni.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/nicedreamzapp/claude-code-local/stargazers"><img src="https://img.shields.io/github/stars/nicedreamzapp/claude-code-local?style=for-the-badge&logo=github&color=f5c542&labelColor=1f2328" alt="GitHub stars"></a>
    <a href="https://github.com/nicedreamzapp/claude-code-local/network/members"><img src="https://img.shields.io/github/forks/nicedreamzapp/claude-code-local?style=for-the-badge&logo=github&color=4c9a2a&labelColor=1f2328" alt="GitHub forks"></a>
    <a href="#-the-lineup--pick-your-fighter"><img src="https://img.shields.io/badge/🥊_Lineup-7_Models-red?style=for-the-badge" alt="7 Models"></a>
    <a href="https://nicedreamzapp.github.io/agent12/"><img src="https://img.shields.io/badge/🏆_Agent--12-Local_Agent_Leaderboard-gold?style=for-the-badge" alt="Agent-12 leaderboard"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/⚡_Qwen_3.8_+_DFlash_2-39.9_tok%2Fs-brightgreen?style=for-the-badge" alt="Qwen 3.8 speed"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/🚀_Claude_Code_Task-17.6s-blue?style=for-the-badge" alt="Claude Code task time"></a>
    <a href="#-privacy--how-the-data-flows"><img src="https://img.shields.io/badge/🔒_Privacy-100%25_Local-success?style=for-the-badge" alt="100% Local"></a>
    <a href="VOICE-MODE.md"><img src="https://img.shields.io/badge/🎤_Voice-Hands_Free-orange?style=for-the-badge" alt="Hands-Free Voice"></a>
    <a href="../LICENSE"><img src="https://img.shields.io/badge/📜_License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
    <a href="https://discord.gg/ZdSqgAxUW"><img src="https://img.shields.io/discord/1497121921580404818?label=NiceDreamzApps&logo=discord&color=5865F2&style=for-the-badge" alt="Join the NiceDreamzApps Discord"></a>
  </p>
  <p align="center">
    <a href="#-hit-your-claude-usage-limit">🛑 Usage Limit</a> ·
    <a href="#-what-is-this">🤔 What Is This</a> ·
    <a href="#-quick-start-one-command">🚀 Quick Start</a> ·
    <a href="#-which-model-should-i-run--the-agent-12-leaderboard">🏆 Leaderboard</a> ·
    <a href="#-the-lineup--pick-your-fighter">🥊 Lineup</a> ·
    <a href="#-the-modes">🎮 Modes</a> ·
    <a href="#-privacy--how-the-data-flows">🔒 Privacy</a> ·
    <a href="#-benchmarks">📊 Benchmarks</a> ·
    <a href="VOICE-MODE.md">🎤 Voice</a> ·
    <a href="BROWSER-AGENT.md">🌐 Browser</a> ·
    <a href="PHONE-CONTROL.md">📱 Phone</a> ·
    <a href="MCP-SERVERS.md">🔌 MCP</a> ·
    <a href="#-whats-next">🛣️ Roadmap</a>
  </p>
</p>

---

## 🛑 Hit your Claude usage limit?

If Claude Code just told you **"you've reached your usage limit"** and gave you a reset time hours away, that's what this is for. You keep working — same Claude Code, same terminal, same project — except the model answering is running on your own Mac.

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

No API key. No second subscription. No waiting until 3pm. It works on a **16 GB MacBook** and gets better the more RAM you have — [see what runs on your Mac](#-what-you-need).

<p align="center">
  <img src="../assets/demo.gif" width="900" alt="Claude Code editing a file with Gemma 4 31B running locally on a Mac, no cloud">
</p>

<p align="center">
  <em>Real session, unedited. Claude Code reads and edits the file — the model answering is Gemma 4 31B on the laptop.</em>
</p>

---

## 🤔 What Is This?

Your Mac has a powerful GPU built right into the chip. This project uses that GPU to run **massive AI models — the same kind that power ChatGPT and Claude — entirely on your computer**, and plugs them into Claude Code so the whole coding experience works offline.

No internet, no subscription, nobody sees your code — and it's the full Claude Code experience: edit files, manage projects, drive your browser, or run a hands-free voice session.

**The trick:** Claude Code speaks the **Anthropic API**. Local model servers speak the **OpenAI API**. So everyone bolts a translation proxy in between — and the proxy is slow and fragile. This server speaks Anthropic natively. One process, zero translations:

| 🐌 What everyone else does | 🚀 What we did |
|---|---|
| Claude Code → **Proxy** → Ollama → Model | Claude Code → **Our Server** → Model |
| 3 processes, 2 API translations | **1 process, 0 translations** |
| 133 seconds per task | **17.6 seconds per task** |

> 🎯 That one change — **eliminating the proxy** — made it **7.5× faster**.

---

## 🎬 Watch It Run — AirGap AI

**A real NDA. Wi-Fi physically OFF. `lsof` running live.** Watch a local model audit a confidential legal document, on-device, with the receipts on screen. (Recorded in spring 2026 on the model we ran then; the privacy setup is unchanged, and today's models are in [the lineup](#-the-lineup--pick-your-fighter).)

<p align="center">
  <a href="https://www.youtube.com/watch?v=V_J1LpNGwmY">
    <img src="https://img.youtube.com/vi/V_J1LpNGwmY/maxresdefault.jpg" width="720" alt="AirGap AI — Wi-Fi OFF NDA Demo">
  </a>
</p>

<p align="center">
  <em>AirGap is this whole build running as one private workstation — a capability, not a product. Everything you need is in this repo. If your firm needs one built, <a href="https://nicedreamzwholesale.com/airgap/">here's what it looks like</a>.</em>
</p>

**More local-AI demos on the channel** (dated recordings; each shows the models we ran at the time):

| Video | What happens |
|---|---|
| [🌌 The Rematch](https://www.youtube.com/watch?v=03KVQmEx13Q) | 4 AI engines build northern lights, 3 fully local — the local challenger painted the best aurora |
| [🏁 Hexagon Shootout](https://www.youtube.com/watch?v=2KeTDDodE0A) | Gemma 31B vs Llama 70B vs cloud Claude, same physics prompt, live counters — 2 of 3 with zero cloud calls |
| [🐳 DeepSeek Three-Way](https://youtu.be/7l8-s8xkpms) | DeepSeek V4 Flash local beats cloud Claude on wall-clock, same MacBook |
| [🎤 NarrateClaude](https://www.youtube.com/watch?v=4ETqEjjopUk) | Speak to Claude Code, hear replies in a cloned voice — 100% on-device |
| [🏠 Mac mini as home AI](https://www.youtube.com/watch?v=PLbV4QtFmFY) | Chat with the Mac mini at home from any browser on any phone |

---

## 🏆 Which model should I run? — The Agent-12 Leaderboard

The question every issue here eventually asks. So we measure it. **[Agent-12](https://nicedreamzapp.github.io/agent12/)** runs each local model through real agent tasks in a sandboxed working directory, judged by what ends up on the filesystem, never by the model's prose. Temperature 0, fixed caps, fresh sandbox per task, one variable moved per comparison. Tasks, judges, runner and the judge-validation gate are all open at [github.com/nicedreamzapp/agent12](https://github.com/nicedreamzapp/agent12).

The two models we recommend, as measured there:

| Model | Easy (12) | time | Hard (8) | time | tok/s |
|---|:---:|:---:|:---:|:---:|:---:|
| **Gemma 4 31B** (MLX 4-bit) | 11/12 | 92s | 8/8 | 348s | 26 |
| **Qwen3.8-27B** (MLX 8-bit, dense) | 12/12 | 351s | 7/8 | 1156s | — |
| *Claude Sonnet 5 (cloud, reference only)* | 12/12 | 122s | 8/8 | 131s | — |

Every other model we have run is on [the full board](https://nicedreamzapp.github.io/agent12/).

**Where these numbers come from:** every row was measured inside Agent-12's own lean reference harness (Anvil, a ~550-token native terminal engine with per-model tool-call dialects), **not inside Claude Code itself.** Claude Code as installed sends the model far more per turn (with MCP servers connected, tens of thousands of tokens of tool definitions on the first turn), and the same model scores and times differently there. A contributor measured that gap on the same model and engine in [agent12 #1](https://github.com/nicedreamzapp/agent12/issues/1): 8/8 hard in 105s under Anvil vs 6/8 in 310s inside Claude Code as installed. The harness is the variable; the board holds it fixed so models can be compared against each other. All local rows: Apple M5 Max, 128 GB.

Qwen 3.8 reaches 8/8 hard when given an 8000-token thinking budget; its misses at the standard budget were the cap cutting it off mid-thought ([writeup](https://github.com/nicedreamzapp/agent12/blob/main/writeups/qwen38_vs_qwen36.md)). Muse-Glimmer 30B and Nemotron Omni are listed on the board in a separate **vendor-reported** section with Meta's and NVIDIA's own published numbers, credited and linked, until they get a real Agent-12 run.

**Browser test (Sep 16, 2026).** Four real jobs in four real browser tabs, each checked by script: Gemma 4 31B (bf16) got 8/8 in 123 s one at a time and 110 s all at once; Qwen 3.8 27B (bf16) got 8/8 in 175 s and 176 s. Gemma is the faster browser driver.

---

## 🥊 The Lineup — Pick Your Fighter

Same MLX server, same Anthropic API. Swap one env var and you swap the brain. Every speed here was measured by us on an M5 Max 128 GB unless marked.

| | 🟣 **Qwen 3.8 27B** | 🟢 **Gemma 4 31B** | 🟢 **Gemma 4 12B** | 🟡 **Hermes 4 14B** | 🟠 **Gemma 4 E4B** | ✨ **Muse-Glimmer 30B** | 👁️ **Nemotron 3 Nano Omni** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Build | 8-bit ([lmstudio-community](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-8bit)), or bf16 + DFlash 2 | 4-bit abliterated | 4-bit abliterated | 4-bit abliterated | 4-bit ([mlx-community](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit)) | bf16 / 8-bit abliterated (in-house) | bf16 / 8-bit / 4-bit abliterated (in-house) |
| Speed | 17.9 tok/s at 8-bit, **39.9** with DFlash 2; bf16 + DFlash 2: 36.5 | 26 tok/s | not measured | not measured | not measured | ~18 tok/s (8-bit) | see [nemotron-omni-mlx](https://github.com/nicedreamzapp/nemotron-omni-mlx) |
| Agent-12 | 12/12 easy, 7/8 hard (8/8 with a bigger budget) | 11/12, 8/8 | not yet | not yet | not yet | vendor-reported only | vendor-reported only |
| Claude Code smoke test (Sep 19) | ✅ 38 s | ✅ (the default launcher) | — | ✅ on a 16 GB Mac ([#54](https://github.com/nicedreamzapp/claude-code-local/issues/54)) | ❌ claimed to run a file it never wrote | — | — |
| RAM | ~29 GB (8-bit) | ~18 GB | ~11 GB | ~8 GB | ~5 GB | ~30–60 GB | ~20–66 GB |
| `setup.sh` gives it to | 96 GB+ | 64–95 GB | 32–63 GB | 16–31 GB | under 16 GB | — | — |
| Launcher | `Qwen 3.8 Code.command` | `Gemma 4 Code.command`, `Claude Local.command` | `Claude Local.command` | `Claude Local.command` | `Claude Local.command` | *coming* | *coming* |

> 🟣 **Qwen 3.8 27B at full precision.** Alibaba's 27B dense model (Apache 2.0, native image + video input, 262K context). The trick that makes bf16 livable is **[DFlash 2](https://inco.ai/blog/dflash2/)**, a speculative-decoding drafter from Inco AI / Z Lab: a 2B draft model proposes a block of tokens, the 27B verifies the block in one pass, and the output is **byte-identical** to plain decoding. Measured on our M5 Max 128 GB, greedy: **9.7 → 36.5 tok/s at bf16** and **17.9 → 39.9 tok/s at 8-bit** with the drafter. Weights: [`mlx-community/Qwen3.8-27B-bf16`](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) + drafter [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), served by [`mlx-dspark`](https://github.com/ARahim3/mlx-dspark) (`pip install mlx-dspark`, OpenAI-compatible API). Two tips that survived a night of testing: keep the draft block at 5 on Metal, and drop the community **[Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)** into the model folder. It fixes the stock template's empty-think aborts, defaults reasoning to `medium` instead of `xhigh`, and makes the model lead with the answer.
>
> ✨ **Muse-Glimmer (Aug 2026)**: Meta's agentic 30B, [abliterated in-house](#-our-own-mlx-abliterated-uploads). ~18 tok/s at 8-bit. 👁️ Vision via the [`-MM-bf16`](https://huggingface.co/divinetribe/Muse-Glimmer-30B-Abliterated-MM-bf16) build: **[watch it read a chart, a neon sign and a blurred car badge](https://www.youtube.com/watch?v=5fs_FfkCaDA)**.

> 💻 **Got a 16 GB MacBook Air?** `setup.sh` picks Hermes 4 14B for you automatically.

### 🤗 Good builds from other people

Original releases and MLX builds of everything above. Credit to the teams who made them.

- **Qwen 3.8 27B:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [lmstudio-community MLX 8-bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-8bit) · [mlx-community 8-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit) · [mlx-community bf16](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) · [incoai DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- **Gemma 4:** [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) · [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) · [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) · [mlx-community 31B 4-bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) · [mlx-community 31B 8-bit](https://huggingface.co/mlx-community/gemma-4-31b-it-8bit) · [mlx-community E4B 4-bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit)
- **Hermes 4 14B:** [NousResearch/Hermes-4-14B](https://huggingface.co/NousResearch/Hermes-4-14B)
- **Muse Glimmer 30B:** [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) · [mlx-community bf16](https://huggingface.co/mlx-community/Muse-Glimmer-30B-bf16)
- **Nemotron 3 Nano Omni:** [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) · [mlx-community 4-bit](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit)

### ⭐ Our Own MLX Abliterated Uploads

We also package and upload our own abliterated MLX builds to HuggingFace so anyone running this repo can pull them with one command. Browse the full set at [huggingface.co/divinetribe](https://huggingface.co/divinetribe).

```bash
# Gemma 4 31B — fast daily driver
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh

# Hermes 4 14B — for 16 GB Macs
MLX_MODEL=divinetribe/Hermes-4-14B-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh
```

Current public models on [huggingface.co/divinetribe](https://huggingface.co/divinetribe). Sizes are the real on-disk totals from the Hub. All are MLX-format; the 👁️ ones take images.

**Text models**

| Model | Quant | Disk | Params | Context | Best for |
|---|---|---|---|---|---|
| [`gemma-4-31b-it-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/gemma-4-31b-it-abliterated-4bit-mlx) | 4-bit, g64 | 17.3 GB | 31 B dense | 128 K | Daily coding on a 64 GB Mac (the default launcher) |
| [`Huihui-gemma-4-31B-it-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/Huihui-gemma-4-31B-it-abliterated-4bit-mlx) | 4-bit, g64 | 17.3 GB | 31 B dense | 128 K | Same model, huihui-ai's abliteration instead of null-space's |
| [`gemma-4-12B-it-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/gemma-4-12B-it-abliterated-4bit-mlx) | 4-bit | 11.0 GB | 12 B dense | 128 K | 32 GB Macs |
| [`gemma-4-12B-it-abliterated-4bit-mlx-text`](https://huggingface.co/divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text) | 4-bit | 11.0 GB | 12 B dense | 128 K | Same, vision tower stripped — loads in plain `mlx-lm` |
| [`Hermes-4-14B-abliterated-4bit-mlx`](https://huggingface.co/divinetribe/Hermes-4-14B-abliterated-4bit-mlx) | 4-bit, g64 | 8.3 GB | 14 B dense | 40 K | 16 GB Macs |

**Vision + text models** 👁️

| Model | Quant | Disk | Params | Context | Best for |
|---|---|---|---|---|---|
| [`Nemotron-3-Nano-Omni-30B-Abliterated-MM-bf16`](https://huggingface.co/divinetribe/Nemotron-3-Nano-Omni-30B-Abliterated-MM-bf16) 👁️🎧 | bf16 | 66.0 GB | 30 B / 3 B active MoE | 128 K | NVIDIA's tri-modal (text + vision + **audio**) Omni, full precision; runs via [`nemotron-omni-mlx`](https://github.com/nicedreamzapp/nemotron-omni-mlx) |
| [`Nemotron-3-Nano-Omni-30B-Abliterated-MM-8bit`](https://huggingface.co/divinetribe/Nemotron-3-Nano-Omni-30B-Abliterated-MM-8bit) 👁️🎧 | 8-bit | 35.8 GB | 30 B / 3 B active MoE | 128 K | Same, 48 GB+ Macs |
| [`Nemotron-3-Nano-Omni-30B-Abliterated-MM-4bit`](https://huggingface.co/divinetribe/Nemotron-3-Nano-Omni-30B-Abliterated-MM-4bit) 👁️🎧 | 4-bit | 19.7 GB | 30 B / 3 B active MoE | 128 K | Same, 32 GB Macs |
| [`Muse-Glimmer-30B-Abliterated-MM-bf16`](https://huggingface.co/divinetribe/Muse-Glimmer-30B-Abliterated-MM-bf16) 👁️ | bf16 | 59.6 GB | ~30 B | 128 K | Meta's agentic 30B with the full vision tower, abliterated in-house; runs via [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) |

Older uploads are still on the Hub for anyone who depends on them, but we no longer recommend them: Llama 3.3 70B, Qwen3-8B, Qwen3-Coder-Next distilled, Qwen3-VL-32B and Qwen 3.6 27B. Also on the Hub: [`yolov8n-oiv7-coreml`](https://huggingface.co/divinetribe/yolov8n-oiv7-coreml), the 601-class CoreML detector behind [RealTimeAICam](https://github.com/nicedreamzapp/RealTimeAICam).

**Abliteration sources:** [huihui-ai](https://huggingface.co/huihui-ai) (one of the Gemma 31Bs), [null-space](https://huggingface.co/null-space) (Gemma 4 31B), [OpenYourMind](https://huggingface.co/OpenYourMind) (Gemma 4 12B) and [Babsie](https://huggingface.co/Babsie) (Hermes). **Muse-Glimmer and Nemotron Omni we abliterated ourselves**: refusal direction removed across every residual-writing layer (layer 26 on Glimmer), on the freshly-released bf16 weights, with the vision (and for Nemotron, audio) towers kept intact. We believe Glimmer-MM was the first abliterated multimodal model running on Apple MLX. MLX conversion + quantization by us. See [what abliteration means](https://huggingface.co/blog/mlabonne/abliteration).

> ⚠️ **Use it responsibly.** "Abliterated" suppresses the model's built-in refusal direction so it doesn't refuse benign-but-edgy requests. It is **not** a general capability upgrade, and you remain bound by each upstream license (Gemma, Hermes, Muse-Glimmer, Nemotron).

---

## 🎮 The Modes

Four ways to run the lineup. Each one is a double-clickable launcher in `launchers/`.

| Mode | What it does | Launcher |
|---|---|---|
| 🤖 **Code** | Run Claude Code with a local model — same UX, no API key | `Claude Local.command`, `Gemma 4 Code.command`, `Qwen 3.8 Code.command` |
| ⚡ **Native Engine** *(new)* | Our own ~900-line agent, model loaded in-process — replies start in ~0.3 s deep into long sessions ([details](#-gen-4--the-native-engine-added-aug-7-2026)) | `Gemma 4 Code (Native Engine).command`, `Qwen 3 Coder (Native Engine).command` |
| 🌐 **Browser** | Local AI controls real Brave browser via Chrome DevTools | `Browser Agent.command` |
| 🎤 **Hands-Free Voice** | Speak in, hear replies in your cloned voice — full loop, 100% on-device | `Narrative Gemma.command` + [NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude) |
| 📱 **Phone** | iMessage in → text/image/video out, via [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) | `~/.claude/imessage-*.sh` |

---

## 💻 What You Need

| Your Mac | RAM | What `setup.sh` installs for you |
|----------|-----|-------------------|
| Base M-series | under 16 GB | 🟠 Gemma 4 E4B (tool calls will be unreliable) |
| MacBook Air / base M1-M5 | **16 GB** | 🟡 **Hermes 4 14B** |
| M1-M5 Pro | 32-63 GB | 🟢 Gemma 4 12B |
| M2-M5 Max | 64-95 GB | 🟢 **Gemma 4 31B** |
| M3-M5 Max · Ultra | 96 GB+ | 🟣 **Qwen 3.8 27B** (8-bit) |

Also need:
- 🐍 **Python 3.12+** (for MLX)
- 🤖 **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

---

## 🚀 Quick Start (One Command)

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

Or clone it yourself if you'd rather read the script first:

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

# 2. Pick a model and download it (one time, 5-30 GB)
bash scripts/download-and-import.sh gemma   # or 'qwen', 'gemma12', 'hermes'

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
│                 (Gemma·Qwen·Hermes)              │
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
3. 🔧 **Translates tool use** — Three tool-call formats in and out: Gemma 4 native, Llama-style raw JSON, and HuggingFace `<tool_call>` JSON (Qwen and others). All converted ↔ Anthropic `tool_use` blocks, with garbled-output recovery for small models.
4. 🧹 **Cleans the output** — A real-time `ThinkingFilter` strips `<think>` blocks token-by-token during generation, then `clean_response` handles stop markers and reasoning preamble.
5. ⚡ **Reuses prompt caches across requests** — Claude Code's system prompt doesn't get re-prefilled every turn. Huge speedup for short questions.
6. 🎯 **Code mode** — auto-detects Claude Code coding sessions, swaps the ~10K-token harness prompt for a slim ~150-token one, and strips verbose tool descriptions to name + parameter types. A **28× prompt reduction** that cuts prefill from ~60 s to ~2 s on Gemma 4 31B.

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
| **browser broker** | [nicedreamzapp/browser-broker](https://github.com/nicedreamzapp/browser-broker) — we wrote it | **0** (localhost :9223 only) | ✅ Safe |
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

---

## 📊 Benchmarks

### ⚡ Speed Comparison (history)

How the server got fast, April 2026. Gens 1-3 were measured with the big MoE model we shipped then; today's model speeds are in the lineup table above.

| Generation | Approach | Speed | Real Claude Code task |
|---|---|---:|---:|
| 🐌 Gen 1 | Ollama + Proxy | 30 tok/s | 133 s |
| 🏃 Gen 2 | llama.cpp + Proxy | 41 tok/s | 133 s |
| 🚀 Gen 3 | **MLX Native (ours)** | **65 tok/s** | **17.6 s** |
| ⚡ Gen 4 | **Native engine** — same models, no Claude Code in the loop | first token in **0.36 s** at 4.5k ctx | see below |

### ⚡ Gen 4 — The Native Engine (added Aug 7, 2026)

**New, optional, and additive.** Every Claude Code launcher in this repo
works exactly as it did yesterday. This is a second way in, for people who
want the fastest possible turnaround out of the same local models.

**Why we built it.** Claude Code is a great harness, but it was designed for
cloud models: its system prompt is tens of thousands of tokens, and parts of
the prompt change every turn. A local model pays for that twice — once in
prefill time, and again because a mutating prompt head defeats KV-cache reuse
entirely. The native engine (`agent/agent.py`, ~900 lines, zero dependencies
beyond mlx-lm) flips the design: a ~550-token system prompt that never
changes, the same tools (Bash/Read/Write/Edit/Glob/Grep), and a KV cache that
is trimmed to the shared prefix each turn so only the *new* tokens are ever
prefilled.

**What we found on the way (useful even if you never run this).** While
benchmarking we discovered that Gemma-family models hand their sliding-window
layers a `RotatingKVCache` capped at the window (1024 tokens on Gemma 4). The
moment a conversation outgrows the window, those caches report untrimmable and
prompt-cache reuse silently dies — every turn re-prefills the entire
transcript, exactly in the long sessions where caching matters most. The
engine swaps in plain `KVCache` on every layer: the sliding-window attention
*mask* still enforces the window, so outputs are byte-identical (we verified
with a greedy A/B comparison), but the cache stays trimmable forever.
`AGENT_ROLLING_KV=1` restores stock behavior if you want the lower memory
ceiling instead.

**Measured** (Gemma 4 31B 4-bit, 4.5k-token conversation, Apple Silicon):

| | time to first token |
|---|---:|
| stock rotating cache (full re-prefill every turn) | 6.5–7.2 s |
| native engine with the cache fix | **0.36 s** |

Roughly **18× faster turn starts** on long sessions, with generation steady at
26–28 tok/s. Reproduce it yourself: `python3 bench/agent_bench.py`.

**Terminal UX**: a framed input area with a live context-percentage meter in
the top rule, a thinking/running spinner so silence never means "hung", and
readline history across sessions.

**Run it**: double-click a `(Native Engine)` launcher, or:

```bash
AGENT_MODEL=lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit \
AGENT_DIALECT=native \
python3 agent/agent.py
```

`AGENT_DIALECT=native` uses the model's own tool-call template (Qwen3-Coder's
XML with raw-text values); `prompted` teaches the same XML dialect in the
system prompt for models whose native format re-introduces JSON escaping
(Gemma). `AGENT_BACKEND=http` points the same agent at any local
Anthropic-protocol server, including this repo's proxy.

This builds directly on the prompt-cache trim work contributed in #46 — that
fix is what made the deeper rotating-cache problem visible. Thank you.

### ☁️ vs Cloud APIs

| | 🖥️ **Our Local Setup** | ☁️ Claude Sonnet | ☁️ Claude Opus |
|---|:---:|:---:|:---:|
| Speed | 17.9–39.9 tok/s (Qwen 3.8 27B), 26 (Gemma 4 31B) | ~80 tok/s | ~40 tok/s |
| Monthly cost | **$0** 🎉 | $20-100+ | $20-100+ |
| Privacy | **100% local** 🔒 | Cloud | Cloud |
| Works offline | **Yes** ✈️ | No | No |

> Local speeds measured on an M5 Max 128 GB. More in [BENCHMARKS.md](BENCHMARKS.md).

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
| `MLX_KV_BITS` | `8` | KV cache quantization bits (4 saves memory, 8 improves coherence). Known: `4` can drop ~1 in 14 tool calls with empty arguments, and it gets noisier when prompt-cache reuse is on (see [#42](https://github.com/nicedreamzapp/claude-code-local/issues/42)). Default `8` is unaffected; use `4` only when you need the memory. |
| `MLX_KV_QUANT_START` | `1024` | Token position where KV quantization begins |
| `MLX_TOOL_RETRIES` | `2` | Max retries when a garbled tool call is detected |
| `MLX_MAX_TOKENS` | `8192` | Max output tokens per response |
| `MLX_SUPPRESS_THINKING` | `1` | Skip the model's reasoning chain (~1 min/request saved). Set `0` to let it think. |
| `MLX_BROWSER_MODE` | `0` | Optimize for chrome-devtools MCP sessions — keeps only the 9 essential browser tools (~99% fewer tokens) |

---

## 📚 More

Everything above gets you running. These live in [`docs/`](.) so this page stays short:

| | |
|---|---|
| 🎤 [Hands-Free Voice Mode](VOICE-MODE.md) | Talk to Claude Code, hear it answer in a cloned voice |
| 🌐 [Browser Agent](BROWSER-AGENT.md) | Let the local model drive your real browser |
| 📱 [Control From Your Phone](PHONE-CONTROL.md) | Run a session on your Mac from anywhere |
| 🔌 [MCP Servers](MCP-SERVERS.md) | Claude Code's whole plugin ecosystem, 100% local |
| 📁 [What's In This Repo](REPO-LAYOUT.md) | File-by-file tour |
| 📊 [Benchmarks](BENCHMARKS.md) · 🔧 [Tool-Call Reliability](TOOL-CALL-RELIABILITY.md) | The numbers and how they were measured |
| 📱 [Apps From The Same Workshop](OTHER-APPS.md) · 🙏 [Credits](CREDITS.md) | Everything else |

---

## 🧩 The Local-First Stack

`claude-code-local` is the **brain**. It pairs with sibling repos — each stands alone, together they take Claude Code off the keyboard and off the screen:

| Repo | Role | What it does |
|---|---|---|
| 🤖 **claude-code-local** | Brain *(you are here)* | MLX Anthropic server · launcher lineup · tool-call translation |
| 🎤 [NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude) | Ears + Mouth | Talk to Claude, hear replies in your cloned voice — both directions on-device |
| 🌐 [browser-agent](https://github.com/nicedreamzapp/browser-agent) | Hands | Drives real Brave via CDP — iframes, Shadow DOM, ProseMirror |
| 🚦 [browser-broker](https://github.com/nicedreamzapp/browser-broker) | Traffic cop | Leases each agent its own hidden tab in your real browser — many agents, nobody fights, nobody grabs the tab you're reading |
| 📱 [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) | Remote | iPhone → Claude Code over iMessage; text/screenshots/videos back |
| 🛟 [claude-failover](https://github.com/nicedreamzapp/claude-failover) | Backstop | Keep cloud Claude primary, flip one command to local when limits pinch or Anthropic is down |

---

## 🔬 More Local-AI Work

Not part of this stack — separate projects, same rule: **the model runs on your machine, not on somebody's server.**

| Repo | What it is |
|---|---|
| 🧠 [nemotron-omni-mlx](https://github.com/nicedreamzapp/nemotron-omni-mlx) | NVIDIA's tri-modal **Nemotron Omni** — text, vision *and* audio — in pure MLX. The vision and audio towers had no Apple Silicon runtime, so I wrote one. **23/23 parity tests** against NVIDIA's PyTorch reference. 67 tok/s with an image, wifi off. |
| 🎬 [story-forge](https://github.com/nicedreamzapp/story-forge) | A script goes in, a finished film comes out — stills, motion, voices, music, grade and sound, start to finish on one laptop. Flux · LTX-2 · Piper · ACE-Step · ffmpeg. |
| 🎙️ [song-forge](https://github.com/nicedreamzapp/song-forge) | Describe a song, get a finished original in any of 19 languages. ACE-Step + Gemma + seed-vc voice swap, rendered locally, never stored. |
| 👁️ [VisionBuilder](https://github.com/nicedreamzapp/VisionBuilder) | Label photos and train your own on-device vision model straight from the camera roll. The dataset never leaves the phone. |
| 🗓️ [Family-Planner](https://github.com/nicedreamzapp/Family-Planner) | Self-hosted family command center for a spare iPad — voice control, OCR document scanning, meal planning. No cloud, no subscription. |

📱 There are four apps on the App Store and Google Play too — [see them here](OTHER-APPS.md).

---

## 🛣️ What's Next

We ship fast and in public. If any of these excite you, hit **Watch** to get the release ping.

- 🟣 **DFlash 2 in the Claude Code path** — `Qwen 3.8 Code.command` runs 8-bit through this server today; the drafter (2.2× faster) still runs through `mlx-dspark` on its own port
- 🟡 **Agent-12 runs for Gemma 4 12B, Hermes 4 14B and Muse-Glimmer** so every model we ship has our own numbers
- 🟡 **Fully-local Whisper fallback** — alternative to the Apple `SFSpeechRecognizer` path for older Macs and non-English voices
- 🟡 **One-click DMG installer** — no terminal needed
- 🟡 **`MLX_MODEL=<hf-url>`** — point at any HuggingFace repo and auto-register a new fighter
- 🟡 **More models** — open to PRs adding launchers for anything MLX-compatible

> 💡 Want something that's not on this list? [**Open an issue →**](https://github.com/nicedreamzapp/claude-code-local/issues/new) Every serious request gets read and usually replied to within 24h.

## 🤝 Contributing

Ideas, bug reports, a new launcher for a model I don't run, a better code-mode prompt — open an issue or a PR, I read them all. Especially interested in: folks on older Apple Silicon (M1/M2, 16–36 GB) who know which models actually fit; anyone stress-testing the voice loop on different hardware or accents; TTS recipes beyond Pocket TTS (Piper, MLX-TTS, Kyutai Moshi); and edge cases I'll never hit on an M5 Max with 128 GB.

---

## 🙏 Credits

### 🧑‍🔧 Contributors

Every one of these landed on hardware I don't own, on a bug I hadn't hit. Thank you.

| Who | What they fixed |
|---|---|
| [@0xshugo](https://github.com/0xshugo) | Client disconnects handled, retries skipped when there are no tools ([#4](https://github.com/nicedreamzapp/claude-code-local/pull/4)) |
| [@asdmoment](https://github.com/asdmoment) | Gemma inference crash — auto-disable KV quantization ([#7](https://github.com/nicedreamzapp/claude-code-local/pull/7)) |
| [@kulveersingh](https://github.com/kulveersingh) | `ArraysCache` has no attribute `offset` ([#10](https://github.com/nicedreamzapp/claude-code-local/pull/10)) |
| [@tripathiprateek](https://github.com/tripathiprateek) | `uninstall.sh` — reverses `setup.sh` cleanly ([#23](https://github.com/nicedreamzapp/claude-code-local/pull/23)) |
| [@tadrianonet](https://github.com/tadrianonet) | Mac base/Pro 16 GB support: Qwen 2.5 14B, ChatML stop markers, `<tools>` parser, offline leak fix ([#32](https://github.com/nicedreamzapp/claude-code-local/pull/32)) |
| [@kevbarns](https://github.com/kevbarns) | Gemma 4 thinking suppression + slimmer tool descriptions — ~4× latency cut ([#33](https://github.com/nicedreamzapp/claude-code-local/pull/33)) |
| [@KaoCSC](https://github.com/KaoCSC) | Stop on the tokenizer's real EOS, and tolerate empty env ints ([#41](https://github.com/nicedreamzapp/claude-code-local/pull/41)) · bare JSON tool calls, which took Qwen 2.5 Coder from 0/12 to 14/14 ([#43](https://github.com/nicedreamzapp/claude-code-local/pull/43)) |
| [@zwolf25](https://github.com/zwolf25) | Slash commands in code mode (`Skill` was being filtered out) and array-shaped fenced tool calls from Hermes 4 14B, with a standalone parser test ([#56](https://github.com/nicedreamzapp/claude-code-local/pull/56)) |
| [@pratikgx](https://github.com/pratikgx) | Broken `IMESSAGE_MEDIA_PIPELINE.md` link in `docs/PHONE-CONTROL.md` ([#60](https://github.com/nicedreamzapp/claude-code-local/pull/60)) |

Tested on **Apple M5 Max** with **128 GB unified memory**.

Built by [Matt Macosko](https://x.com/NiceDreamzApps) in Humboldt, CA. The MLX server, the [leaderboard](https://nicedreamzapp.github.io/agent12/), the [abliterated uploads](https://huggingface.co/divinetribe), the voice loop and the phone bridge are all his. **Open to work** on local-AI and Apple Silicon inference — matt@ineedhemp.com. Part of [Nice Dreamz LLC](https://nicedreamzwholesale.com). More open-source at [nicedreamzwholesale.com/software](https://nicedreamzwholesale.com/software/) · demos at [youtube.com/@nicedreamzapps](https://www.youtube.com/@nicedreamzapps).

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

Upstream projects this is built on are listed in [docs/CREDITS.md](CREDITS.md).

---