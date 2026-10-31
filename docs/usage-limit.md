---
title: Claude Code usage limit reached? Keep working on a local model
description: >-
  Hit the Claude Code 5-hour limit, the weekly limit, a rate limit, or ran out of
  credits? Switch Claude Code to a free local model on your Mac, Linux or Windows
  PC in one command, keep the same conversation, and switch back when it resets.
---

# Claude Code usage limit reached? Keep working on a local model

You're in the middle of something and Claude Code stops with one of these:

- **"5-hour limit reached - resets [time]"**
- **"You've reached your weekly limit. It resets [day] at [time]."**
- a **rate limit** error, or **"credit balance is too low"** on an API key

You ran out of Claude usage, not out of work. Claude Code can talk to any server that speaks
Anthropic's Messages API, and this project runs one on your own machine with an open model behind
it. Same Claude Code, same project, no API key, no usage limit.

## Switch to local in three steps

1. **Install it once.** Mac or Linux:

    ```bash
    curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
    ```

    Windows (PowerShell):

    ```powershell
    irm https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.ps1 | iex
    ```

    Setup looks at your memory and graphics hardware and picks a model that fits.

2. **Switch.** On a Mac, type `exit` in Claude Code, then `keep going`. You get a short menu and the
   same conversation carries on with the model you pick. Or start a fresh session: double-click
   **Claude Local** on your Desktop (Mac or Windows), or run `claude-local` on Linux.

3. **Switch back when the limit resets.** The install never changes plain `claude`, so it still uses
   your Claude plan.

## Make it happen by itself

On a Mac you can add one Claude Code hook. The moment you hit your usage limit, a new window opens
in the same folder and carries on with a free cloud model or a model on your Mac, no menu. Setup and
honest limits are in [Automatic keep going](AUTO-HANDOFF.md).

Running background `claude -p` scripts? [claude-failover](https://github.com/nicedreamzapp/claude-failover)
is a one-command switch that sends them to a local model until you flip it back.

## Questions people ask

### When does the Claude Code usage limit reset?

The message shows the time. Run `/usage` inside Claude Code to see each limit and its reset time.
Plans have a 5-hour session limit and a separate weekly limit, so the weekly one can mean days.

### Do I need an API key or more credits for the local model?

No. The launchers send `sk-local` as the key and the server never checks it. You only need the
Claude Code CLI, which is free to install.

### What happens if Claude is down?

The model runs on your machine, so an Anthropic outage doesn't stop it. Claude Code itself still
makes some background calls to Anthropic at startup unless you switch them off; the
[Quickstart](quickstart.md#4-check-that-nothing-leaves-the-mac) lists the environment variables.
Check [status.claude.com](https://status.claude.com) to see whether it's an outage.

### Is a local model as good as Claude?

No. Big models on a 64 GB or 96 GB Mac handle real multi-step work. Small ones on an 8 or 16 GB
machine do best with one clear step at a time (see [Small Macs](SMALL-MACS.md)). It keeps you moving
until your limit resets.

### Does my code leave my computer?

With a local model, no: the server only listens on `127.0.0.1`. The one exception is picking a free
cloud model in `keep going`, which sends that conversation to OpenRouter.

More in the [FAQ](faq.md) and [Troubleshooting](troubleshooting.md).
