# 🎤 Hands-Free Voice Mode

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

[← back to the README](../README.md)
