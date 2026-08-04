# 🎤 NarrateClaude Launch Playbook

## The Hook (one sentence)

> "Anthropic just shipped push-to-talk voice for Claude Code. I built always-on ambient mode that talks back in my own cloned voice — 100% local, no spacebar, no cloud."

---

## Why You Win

| Feature | Anthropic Official | VoiceMode MCP | **NarrateClaude** |
|---|---|---|---|
| Always-on / ambient | ❌ push-to-talk only | ❌ push-to-talk | ✅ continuous |
| Talks back in your voice | ❌ no TTS | ❌ generic Kokoro voice | ✅ cloned voice |
| 100% local / offline | ❌ cloud transcription | ⚠️ partial | ✅ fully on-device |
| No spacebar required | ❌ | ❌ | ✅ |
| Part of a full local stack | ❌ | ❌ | ✅ brain + ears + hands |

---

## The 90-Second Demo Video

### Shot list (in order)

1. **You, not at your desk** — walking around the room, making coffee, doing something physical. No keyboard visible.
2. **You speak naturally** — no button press, no spacebar, just talk to the room. Something real: *"Hey, what's the status on the Flask endpoint?"*
3. **Claude responds out loud in your cloned voice** — this is the jaw-drop moment. Let it play out.
4. **Cut to screen** — show it actually ran something. Code written, task done, terminal output visible.
5. **Back to you** — still moving, still not touching anything.

### Key rules
- Show the **absence of the keyboard** — that's the whole demo
- Keep it under 90 seconds
- No intro titles or fluff — start talking in the first 3 seconds
- Subtitles help for silent autoplay on social

---

## X (Twitter) Post

### Version A — punchy
```
I built what Anthropic hasn't shipped yet.

Their voice mode: hold spacebar → talk → release.
Mine: just... talk. It talks back. In my own cloned voice.
100% local. No cloud. Works on a plane.

This is what ambient computing feels like in 2026 👇

[VIDEO]

github.com/nicedreamzapp/NarrateClaude
```

### Version B — story
```
Anthropic shipped push-to-talk voice for Claude Code last month.

Cool. But I wanted more.

So I built:
→ Always-on ambient listening (no spacebar)
→ Claude talks back in my cloned voice
→ 100% on-device, zero cloud
→ Part of a 3-repo local AI stack

I'm walking around my house while it writes code. 

[VIDEO]

Full stack: EARS+MOUTH → BRAIN → HANDS
github.com/nicedreamzapp/NarrateClaude
```

---

## Reddit Posts

### r/ClaudeAI
**Title:** I built ambient voice mode for Claude Code before Anthropic did — always-on, talks back in your own cloned voice, 100% local

### r/LocalLLaMA  
**Title:** NarrateClaude — always-on ambient voice loop for Claude Code on Apple Silicon. Your cloned voice, local MLX inference, no push-to-talk, no cloud.

### r/MachineLearning
**Title:** Built a 3-repo ambient computing stack: local LLM brain + voice clone ears/mouth + CDP browser hands — all on M5 Max, zero cloud

---

## Hacker News — Show HN

```
Show HN: NarrateClaude – ambient voice mode for Claude Code, talks back in your cloned voice

I got tired of being hunched over a keyboard. So I built a 3-part local-first stack:

- NarrateClaude (this repo): always-on mic using Apple's on-device speech engine → local LLM → your cloned voice reads the reply back. No push-to-talk. No cloud. Works offline.
- claude-code-local: runs Claude Code against local MLX models (122B at 41 tok/s on M5 Max)
- browser-agent: CDP-based browser automation that handles cross-origin iframes and Shadow DOM

The result: I walk around my house, talk to Claude Code, it talks back in my voice, and writes code. I haven't touched a keyboard in hours at a time.

Anthropic's official voice mode for Claude Code shipped last month — it's push-to-talk only, no TTS, no voice clone. This is the version I wanted.

Demo video in the README.

github.com/nicedreamzapp/NarrateClaude
```

---

## Timing

| Action | When |
|---|---|
| Film the demo video | This weekend |
| Post to X | Monday morning (9–11am PT) |
| Post to Reddit (r/ClaudeAI + r/LocalLLaMA) | Monday same day |
| Show HN | Tuesday morning |
| Reply to Anthropic's March voice mode announcement thread | Monday with video |

---

## The Bigger Narrative (for interviews / longer posts)

> "Screens-and-mice as the only interface is ending. I'm building what comes after — a local-first ambient computing stack where the keyboard is optional, your data never leaves your house, and the AI sounds like you because it is you."

The three repos together tell that story:
- **claude-code-local** — the brain (670+ stars, proven traction)
- **NarrateClaude** — the ears and mouth
- **browser-agent** — the hands

Stack them and you have a solo developer's ambient AI workstation running entirely on a MacBook Pro. No subscriptions beyond the M5 Max you already own. No data leaving the house.

That's the story. The video is what makes people believe it.
