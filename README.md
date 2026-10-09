<div align="center">

# 🧠⚡ Claude Code Local

### Claude Code, running on your own Mac. No cloud. No API key. No waiting.

**Your Mac's chip runs the AI · Claude Code works exactly like it always does · nothing leaves your computer**

<a href="https://github.com/nicedreamzapp/claude-code-local/stargazers"><img src="https://img.shields.io/github/stars/nicedreamzapp/claude-code-local?style=for-the-badge&logo=github&color=f5c542&labelColor=1f2328" alt="GitHub stars"></a>
<a href="#-what-you-need"><img src="https://img.shields.io/badge/Mac-Apple_Silicon-111111?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon"></a>
<a href="#-your-code-stays-home"><img src="https://img.shields.io/badge/🔒_Privacy-100%25_Local-success?style=for-the-badge" alt="100% Local"></a>
<a href="#-use-it-with-trinidad-head"><img src="https://img.shields.io/badge/🌊_Best_with-Trinidad_Head-a855f7?style=for-the-badge" alt="Best with Trinidad Head"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/📜_License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
<a href="https://discord.gg/ZdSqgAxUW"><img src="https://img.shields.io/discord/1497121921580404818?label=Discord&logo=discord&color=5865F2&style=for-the-badge" alt="Discord"></a>

</div>

**In one sentence:** a small Python server ([`proxy/server.py`](proxy/server.py)) runs open AI models on your Mac's chip with Apple's MLX and answers Claude Code in Anthropic's own API format, so Claude Code works with no cloud model and no API key.

Other things I built that go with it are under [More from me](#more-from-me).

<p align="center">
  <img src="assets/demo.gif" width="860" alt="Claude Code editing a file with Gemma 4 31B running locally on a Mac, no cloud">
  <br><em>A real session, unedited. Claude Code reads and edits the file, and the AI answering is running on the laptop.</em>
</p>

---

## 🛠️ What I built

Matt Macosko wrote the glue that makes local models usable inside Claude Code. The pieces in this repo:

- ⚡ **[`proxy/server.py`](proxy/server.py)**: the MLX server. Speaks the Anthropic Messages API with streaming, turns each model's own tool-call format (Gemma 4, Qwen, Llama JSON) into Claude Code tool calls, repairs garbled tool JSON, and reuses the prompt cache between turns so long sessions don't re-read everything.
- 🧪 **[`scripts/test_parse_tool_calls.py`](scripts/test_parse_tool_calls.py)** and **[`scripts/test_mlx_server.py`](scripts/test_mlx_server.py)**: tests for the tool-call parser (no model needed) and for multi-step tool calls against a running server.
- 🏎️ **[`agent/agent.py`](agent/agent.py)**: the Native Engine, a one-file terminal coding agent with a short fixed system prompt so the cache survives every turn. [`bench/agent_bench.py`](bench/agent_bench.py) reproduces its cache numbers.
- 🔋 **[`bin/keepgoing`](bin/keepgoing)** + [`bin/keepgoing.py`](bin/keepgoing.py): finds your last Claude Code conversation in this folder and resumes it on a free cloud or local model.
- 🚀 **[`install.sh`](install.sh)**, **[`setup.sh`](setup.sh)** and **[`scripts/doctor.sh`](scripts/doctor.sh)**: check the Mac, pick a model that fits its memory, install everything and make a Desktop launcher.
- 🎮 **[`launchers/`](launchers/)** with the shared [`launchers/lib/claude-local-common.sh`](launchers/lib/claude-local-common.sh): the double-click modes below.

**Upstream, not mine:** [Claude Code](https://github.com/anthropics/claude-code) (Anthropic), [MLX and mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple), and the models themselves (Google, Qwen, Nous Research and the others credited [below](#-good-builds-from-other-people)). Contributors who fixed things are named at the bottom.

---

## 🛑 Hit your Claude usage limit?

Claude Code just said **"you've reached your usage limit"** and the reset is hours away? Keep working:
same Claude Code, same project, but the AI answering runs on your own Mac.

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

That's it. It even works on a **16 GB MacBook**, and gets better the more memory your Mac has.

Once it's running, [the local-first family](#-the-local-first-family) takes it off the keyboard: your browser, your voice, your phone, all on the same machine.

---

## 🌊 Use it with Trinidad Head

**Trinidad Head is the recommended terminal for running local models, on Mac and PC.** It's built
from scratch by the same person who made this project, and it makes long local-AI sessions nicer:

- 🪟 **Glass windows with a soft neon glow**, rounded and easy on the eyes
- 🌈 **A different color for every window**, so you can tell your local-model windows apart at a glance
- 🔠 **Bigger, easy-to-read text**
- 💬 **Your own prompts stand out** in a soft bubble, so you can find what you asked when you scroll back
- 🖱️ **Copy and paste just work**, even under Claude Code's calm focus view

<p align="center">
  <img src="https://raw.githubusercontent.com/nicedreamzapp/trinidad-head/main/docs/images/colors.jpg" width="860" alt="Three Trinidad Head windows, each glowing a different color">
</p>

👉 **[Get Trinidad Head](https://github.com/nicedreamzapp/trinidad-head)** · [project page](https://nicedreamzwholesale.com/software/trinidad-head/)

---

## 🔋 Out of Claude credits? Keep going.

When Claude runs out, type **`exit`** in Claude Code, then **`keep going`**. You get a short menu,
and **the same conversation carries on** with whichever model you pick:

- ☁️ **Free cloud models** from OpenRouter (add a free key, see below)
- 💻 **Models on your Mac** from the lineup below, sized to fit your memory

<p align="center">
  <img src="https://raw.githubusercontent.com/nicedreamzapp/trinidad-head/main/docs/images/keep-going.jpg" width="860" alt="The keep going menu: free cloud and local models to continue a Claude conversation">
</p>

`setup.sh` installs it for you. To add it by hand, or to turn on the free cloud models:

```bash
echo 'source "$HOME/claude-code-local/scripts/keepgoing-shell.sh"' >> ~/.zshrc   # the "keep going" command
mkdir -p ~/.config/claude-code-local
echo 'OPENROUTER_API_KEY=your-free-key' >> ~/.config/claude-code-local/keepgoing.env   # optional
```

Want a reminder at the bottom of every Claude window? Point your status line at
`scripts/statusline-keepgoing.sh`. It says *"💡 Out of Claude credits? Type exit, then keep going to
carry on with a free or local model."*

> 💡 Free cloud models can get busy later in the day. The models on your Mac are always there.

---

## 🤔 What is this, really?

Your Mac has a powerful graphics chip built in. This project uses it to run **big AI models right on
your computer** and plugs them into Claude Code, so the whole coding experience works offline:
editing files, managing projects, even driving your browser or talking out loud.

**The trick:** Claude Code speaks one language (Anthropic's), and local AI servers usually speak
another (OpenAI's). Most setups put a slow translator in between. This project's server speaks
Claude Code's language directly, so there's **nothing in the middle**, and that's what makes it fast.

```
📝 You  ──▶  🤖 Claude Code  ──▶  ⚡ this server (on your Mac)  ──▶  🧠 local model  ──▶  🖥️ your Mac's chip
```

---

## 💻 What you need

- 🍎 A Mac with **Apple Silicon** (M1 or newer)
- 🐍 **Python 3.12** (setup installs it for you)
- 🤖 **Claude Code**: `npm install -g @anthropic-ai/claude-code`

Setup looks at your Mac's memory and picks a model that fits:

| Your Mac's memory | The model you get |
|---|---|
| **Under 16 GB** (8 GB) | Gemma 4 E4B. It loads and chats, but in our Claude Code test it claimed to run a file it never wrote, so expect tool trouble |
| **16 GB** (MacBook Air, base models) | 🟡 Hermes 4 14B, confirmed on a 16 GB Mac by a user ([#54](https://github.com/nicedreamzapp/claude-code-local/issues/54)) |
| **32–63 GB** (Pro) | 🟢 Gemma 4 12B |
| **64–95 GB** (Max) | 🟢 Gemma 4 31B |
| **96 GB and up** (Max, Ultra) | 🟣 Qwen 3.8 27B, 8-bit. Passed our Claude Code test (write a file, run it, report) in 38 s |

---

## 🚀 Get started

**One command:**

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

**Or read the script first:**

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

Setup picks your model, downloads it, installs the server, and puts **`Claude Local.command`** on
your Desktop. **Double-click it** and you're coding locally. 🎉

> 🐛 **Asked to sign in to a Claude account?** Your `claude` is too old. Update it:
> `npm install -g @anthropic-ai/claude-code`

<details>
<summary>🛠️ Prefer to do it by hand?</summary>

```bash
# 1. Set up the MLX environment
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm

# 2. Pick a model and download it (one time)
bash scripts/download-and-import.sh gemma   # or 'qwen', 'gemma12', 'hermes'

# 3. Start the server (MLX_SERVER points at the repo copy, since setup.sh didn't install one)
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
MLX_SERVER="$PWD/proxy/server.py" \
  bash scripts/start-mlx-server.sh

# 4. Launch Claude Code against it
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

`setup.sh` installs the server as a **symlink** back to `proxy/server.py` in this repo, so if you edit
it, just restart the server.

</details>

---

## 🥊 Pick your AI

Same server, same Claude Code. Swap one setting and you swap the brain. Links to every build are just below the table.

Every number below was measured by us on an M5 Max (128 GB), except the rows marked vendor-reported.

| | Model | What we measured | Good for |
|---|---|---|---|
| 🟣 | **Qwen 3.8 27B** | Agent-12 with the DFlash 2 drafter (8-bit, Sep 19): 12/12 easy in 122 s, 7/8 hard in 401 s, 20-29 tok/s inside the harness ([writeup](https://github.com/nicedreamzapp/agent12/blob/main/writeups/qwen38_dflash_rerun.md)). Browser test (Sep 16, bf16): 8/8 | Careful coding and pictures, 96 GB Macs |
| 🟢 | **Gemma 4 31B** (4-bit) | Agent-12: 11/12, 8/8. Browser test (Sep 16): 8/8, about 30% faster than Qwen 3.8 | Daily coding and the browser agent |
| 🟢 | **Gemma 4 12B** (4-bit) | Not on Agent-12 yet | 32 GB Macs |
| 🟡 | **Hermes 4 14B** (4-bit) | Not on Agent-12 yet. Runs on a 16 GB Mac ([#54](https://github.com/nicedreamzapp/claude-code-local/issues/54)) | 16 GB MacBooks |
| ✨ | **Muse Glimmer 30B**, **Nemotron 3 Nano Omni** | Vendor-reported numbers only, until we run them ourselves | Pictures (and audio, for Nemotron) |

**Which one should I run?** We test them on real agent tasks in the open:
**[the Agent-12 leaderboard](https://nicedreamzapp.github.io/agent12/)**. Honest note: those scores
come from Agent-12's own lean test harness (Anvil), **not from inside Claude Code**. Claude Code
sends the model a lot more per turn, so the same model can score and time differently here.

Sizes, speeds and memory needs for every model are in the **[full guide](docs/FULL-GUIDE.md)**.

### 🤗 Good builds from other people

These are the original releases and the MLX builds we have run ourselves. Credit to the teams who
made them.

- **Qwen 3.8 27B:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (original) · [lmstudio-community MLX 8-bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-8bit) (what `setup.sh` installs) · [mlx-community 8-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit) · [mlx-community bf16](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) · [incoai DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) · [Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
- **Gemma 4:** [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) · [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) · [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) · [mlx-community 31B 4-bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) · [mlx-community 31B 8-bit](https://huggingface.co/mlx-community/gemma-4-31b-it-8bit) · [mlx-community E4B 4-bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit)
- **Hermes 4 14B:** [NousResearch/Hermes-4-14B](https://huggingface.co/NousResearch/Hermes-4-14B)
- **Muse Glimmer 30B:** [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) · [mlx-community bf16](https://huggingface.co/mlx-community/Muse-Glimmer-30B-bf16)
- **Nemotron 3 Nano Omni:** [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) · [mlx-community 4-bit](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit)
- **Our own abliterated builds:** [huggingface.co/divinetribe](https://huggingface.co/divinetribe)

> ⚠️ **"Abliterated"** models have their built-in refusals turned down. That's not a general upgrade,
> and each model's own license still applies. Please use them responsibly.

---

## 🎮 Ways to use it

Code and Native Engine need only what setup installs (a launcher for a model you don't have yet downloads it on first launch). The other three need one extra piece that `setup.sh` does not install, named in each row.

| | Mode | What it does |
|---|---|---|
| 🤖 | **Code** | Claude Code with a local model: `Claude Local`, `Gemma 4 Code`, `Qwen 3.8 Code` |
| ⚡ | **Native Engine** | Our own lightweight agent for the fastest replies: the `(Native Engine)` launchers |
| 🌐 | **Browser** | The local AI drives your real Brave browser: `Browser Agent` ([guide](docs/BROWSER-AGENT.md)). Needs [browser-agent](https://github.com/nicedreamzapp/browser-agent) installed |
| 🎤 | **Hands-free voice** | Talk to it and hear it answer in your own voice: `Narrative Gemma` ([guide](docs/VOICE-MODE.md)). Needs a `speak` command (macOS `say` works as a stand-in) |
| 📱 | **Phone** | Text your Mac from your iPhone and get answers back ([guide](docs/PHONE-CONTROL.md)). No launcher here: the iMessage scripts live in [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) |

---

## 🧩 The local-first family

The modes above aren't bolted on. Each one is its own repo that works on its own, and they all run
on your hardware. If Claude Code on your own Mac was the part you wanted, this is the rest of it.

| | Project | What it does |
|---|---|---|
| 🌊 | **[Trinidad Head](https://github.com/nicedreamzapp/trinidad-head)** | The terminal to run it all in, on Mac and PC |
| 🎤 | [NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude) | Talk to Claude and hear it answer in your cloned voice |
| 🌐 | [browser-agent](https://github.com/nicedreamzapp/browser-agent) | Drives your real browser |
| 🚦 | [browser-broker](https://github.com/nicedreamzapp/browser-broker) | Gives each AI agent its own browser tab so nobody fights |
| 📱 | [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) | Control Claude Code from your iPhone |
| 🌙 | [Moonstone](https://github.com/nicedreamzapp/moonstone) | Every Claude Code session on every machine in one window, with push-to-talk and spoken answers |
| 🛟 | [claude-failover](https://github.com/nicedreamzapp/claude-failover) | Keep cloud Claude first, flip to local with one command |
| 👁️ | [nemotron-omni-mlx](https://github.com/nicedreamzapp/nemotron-omni-mlx) | NVIDIA's tri-modal Nemotron seeing and hearing on Apple Silicon. The only open MLX runtime for its vision and audio towers, 23/23 parity tests against NVIDIA's own reference |
| 🏁 | [Agent-12](https://github.com/nicedreamzapp/agent12) | The open leaderboard the model table above is scored on. Filesystem-judged, no AI grader |
| 🎬 | [story-forge](https://github.com/nicedreamzapp/story-forge) | A finished animated film on one Mac. Stills, motion, voices, music, sound, no cloud |

---

## 🔒 Your code stays home

**Your code never leaves your Mac.** No cloud model, no tracking, no "anonymous analytics".

The launchers switch off Claude Code's own background internet traffic with Anthropic's documented
settings (thanks [@tadrianonet](https://github.com/tadrianonet), PR #32):

```bash
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
DISABLE_AUTOUPDATER=1
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
```

**Check it yourself:** during a session, `lsof -p $(pgrep -f claude)` shows only `localhost:4000`.

> 🧹 We **removed LiteLLM** after supply-chain concerns and re-checked every dependency. Anything
> with unexplained network calls didn't ship. The full audit is in the [full guide](docs/FULL-GUIDE.md).
>
> ☁️ The one exception is your own choice: picking a **free cloud** model in `keep going` sends that
> conversation to OpenRouter.

---

## 📚 Want the details?

| | |
|---|---|
| 📖 **[Full technical guide](docs/FULL-GUIDE.md)** | Every model table, benchmark, tuning setting and design note |
| 📊 [Benchmarks](docs/BENCHMARKS.md) · 🔧 [Tool-call reliability](docs/TOOL-CALL-RELIABILITY.md) | The numbers and how they were measured |
| 🎤 [Voice mode](docs/VOICE-MODE.md) · 🌐 [Browser agent](docs/BROWSER-AGENT.md) · 📱 [Phone control](docs/PHONE-CONTROL.md) · 🔌 [MCP servers](docs/MCP-SERVERS.md) | Everything the launchers can do |
| 📁 [What's in this repo](docs/REPO-LAYOUT.md) · 🙏 [Credits](docs/CREDITS.md) | The tour and the thank-yous |

---

## 🧭 Other ways to do this

This isn't the only way to run Claude Code against models you choose, and some of these may fit you better:

- **[Ollama](https://docs.ollama.com/api/anthropic-compatibility)** speaks the Anthropic Messages API natively since v0.14, so Claude Code can point straight at it. The simplest route if you already use Ollama, and it isn't limited to Macs.
- **[claude-code-router](https://github.com/musistudio/claude-code-router)** routes Claude Code's requests across many providers and models, local or cloud. Good if you want to mix models per task.
- **[claude-code-proxy](https://github.com/1rgs/claude-code-proxy)** translates Claude Code's requests to OpenAI-style APIs. Handy if your model already sits behind an OpenAI-compatible server.

What this repo adds is an MLX-native server tuned for Apple Silicon and for the tool-call formats local models actually emit.

---

## More from me

Other projects of mine that go well with this one:

- **[Trinidad Head](https://github.com/nicedreamzapp/trinidad-head)**: a terminal for Mac and Windows that I wrote from scratch in Rust. Every window gets its own glow color and your own prompts sit in a bubble, so you can tell your local model windows apart and find what you asked in a long Claude Code session.
- **[NarrateClaude](https://github.com/nicedreamzapp/NarrateClaude)**: a hands-free voice loop for Claude Code on macOS. It turns your speech into text on the Mac, types it into the Terminal running `claude`, and reads the replies aloud. It is the listening half of the voice mode here.
- **[story-forge](https://github.com/nicedreamzapp/story-forge)**: turns a written story into a finished video on an Apple Silicon Mac. Local models make the stills, motion, voices and music, and the pipeline cuts them together, if you want to see what else your Mac can do without a cloud service.
- **[RealTimeAICam](https://github.com/nicedreamzapp/RealTimeAICam)**: a free iPhone and Android app that names objects, reads text and describes the scene out loud, with the models running on the phone. Same offline idea as this repo, in your pocket.
- **[claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone)**: macOS scripts that let you text commands from your iPhone into a Claude Code session on your Mac and get text, images and video back in iMessage. The Phone mode above uses them.
- **[FiaOS](https://github.com/nicedreamzapp/FiaOS)**: a small self-hosted Python web server that gives you a computer's live screen and a real terminal in any browser, so you can run `claude` on your Mac from another device.
- **[nemotron-omni-mlx](https://github.com/nicedreamzapp/nemotron-omni-mlx)**: an MLX runtime for the vision and audio towers of NVIDIA's Nemotron 3 Nano Omni. If you want the Nemotron model listed above to look at images and listen to audio on your Mac, not only read text, this is how.
- **[browser-broker](https://github.com/nicedreamzapp/browser-broker)**: a local proxy on Chrome's debug port that gives each AI agent its own tab in your real, logged-in browser. Useful if you run the Browser mode here next to other browser agents and want them to stop grabbing each other's tabs or yours.

---

## 🤝 Help make it better

Ideas, bug reports, a launcher for a model we don't run: open an
[issue](https://github.com/nicedreamzapp/claude-code-local/issues/new) or a PR, every one gets read.
[CONTRIBUTING.md](CONTRIBUTING.md) has how to run it from the repo and the tests.
We especially love hearing from people on older or smaller Macs about which models really fit.

**Thank you, contributors:** [@0xshugo](https://github.com/0xshugo) ·
[@asdmoment](https://github.com/asdmoment) · [@kulveersingh](https://github.com/kulveersingh) ·
[@tripathiprateek](https://github.com/tripathiprateek) · [@tadrianonet](https://github.com/tadrianonet) ·
[@kevbarns](https://github.com/kevbarns) · [@KaoCSC](https://github.com/KaoCSC) ·
[@zwolf25](https://github.com/zwolf25). What each of them fixed is in
the [full guide](docs/FULL-GUIDE.md#-credits).

---

<div align="center">

Built by **[Matt Macosko](https://x.com/NiceDreamzApps)** in Humboldt, California.
**Open to work** on local AI and Apple Silicon: matt@ineedhemp.com

[Nice Dreamz software](https://nicedreamzwholesale.com/software/) · [YouTube demos](https://www.youtube.com/@nicedreamzapps) · [Discord](https://discord.gg/ZdSqgAxUW)

📜 **MIT License**, use it however you want. ⭐ **Star the repo if it helped you!**

</div>