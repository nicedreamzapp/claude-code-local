#!/bin/bash
# Local AI — fresh Claude session scoped to local-AI work.
# (Cloud Claude with topic-scoping seed prompt — NOT a local model session.
# For a local-MLX session use Claude Local.command, Gemma 4 Code.command,
# or Qwen 3 Coder.command instead.)

cd "$HOME/Desktop/Local AI Setup" 2>/dev/null || cd "$HOME" || exit 1

export CLAUDE_SESSION_LABEL='Local AI'
exec "$HOME/.local/bin/claude" "Focus this session only on local AI work (models, inference, voice, talking-head, mesh nodes). Ignore HQ/firm/business threads unless I explicitly bring them up."
