---
title: Automatic keep going when Claude hits a usage limit
description: >-
  A Claude Code hook that notices when you hit your usage limit and carries the
  same conversation on with a free cloud model or a model on your Mac.
---

# Automatic keep going

`keep going` normally waits for you: you type `exit`, then `keep going`, then pick a model. With the
hook on this page, it happens by itself. When Claude Code stops because you hit your usage limit, a
new window opens in the same folder and the same conversation carries on.

## What it picks

The hook runs `keepgoing --auto`, which skips the menu and takes, in order:

1. The first free cloud model on OpenRouter, if you have added a free key.
2. The model already running on this Mac, if there is one.
3. The biggest model from the lineup that fits your Mac's memory.

Everything after that is the same as the manual path: the conversation resumes with
`claude --resume`, pointed at that model.

## Turn it on

`setup.sh` prints this for you but never edits your settings. Add it to `~/.claude/settings.json`
yourself. If you already have a `"hooks"` block, add `StopFailure` inside it.

```json
{
  "hooks": {
    "StopFailure": [
      {
        "matcher": "rate_limit",
        "hooks": [
          { "type": "command", "command": "python3 ~/claude-code-local/scripts/limit-handoff.py" }
        ]
      }
    ]
  }
}
```

Change the path if you cloned the repo somewhere else. Open a new Claude Code window afterwards so it
reads the new setting.

## When it fires

Only when Claude Code stops with a `rate_limit` error. Claude Code already retries short hiccups
itself, so by the time this error stops a turn, the work has really halted. Every other error is
written to the log and left alone.

It also fires at most once per folder every 30 minutes. A capped account hits the limit in every
open chat, and without this you could get a pile of new windows at once.

## Where the new window opens

- **Moonstone**, if you use [Moonstone](https://github.com/nicedreamzapp/moonstone) and a dock is
  answering at `MOONSTONE_URL` (default `http://127.0.0.1:8899`). The hook asks it to start the
  launcher named in `MOONSTONE_LAUNCHER` (default `keep-going`) and sends the folder as the first
  message. Moonstone launchers each have their own fixed folder, so this only helps if you have set
  up a launcher for it. If the dock is not running or has no such launcher, the hook moves on.
- **Terminal**, otherwise. A new macOS Terminal window opens in the folder and runs
  `keepgoing --auto`.

## Settings

Put these in your environment or in `~/.config/claude-code-local/keepgoing.env`.

| Setting | What it does |
|---|---|
| `KEEPGOING_AUTO=0` | Turns the hook off without touching `settings.json` |
| `MOONSTONE=0` | Skips Moonstone and always uses Terminal |
| `MOONSTONE_URL` | Where to look for a Moonstone dock |
| `MOONSTONE_LAUNCHER` | Which Moonstone launcher to start |
| `OPENROUTER_API_KEY` | Unlocks the free cloud models, same as for the menu |

## Checking what happened

Each time the hook runs it adds a line to `~/.claude/logs/limit-handoff.log`: the payload Claude Code
sent, whether it tried Moonstone or Terminal, and how that went.

## Honest limits

- The hook opens its window through Moonstone (any OS) or, without Moonstone, a macOS Terminal
  window via `osascript`. On Linux and Windows without Moonstone, type `exit`, then `keep going`
  yourself (or `keepgoing --auto` for no menu); the manual command works on all three.
- Free cloud models get busy, mostly later in the day. If the first one is overloaded, type `exit`
  and run `keep going` to pick another, or use a model on your Mac.
- The old window stays open. Leave it alone or close it, so two windows are not writing to the same
  conversation.
- If two Claude chats share one folder, it resumes the one that was active most recently.
- Picking a free cloud model sends that conversation to OpenRouter. Only the local models keep it on
  your Mac.
