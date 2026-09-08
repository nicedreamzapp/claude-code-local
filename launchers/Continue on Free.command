#!/bin/bash
# Continue on Free — you hit a Claude Code usage limit; pick up the SAME
# conversation on the best currently-free OpenRouter model instead of waiting.
# Claude Code stores every conversation locally (~/.claude/projects/*.jsonl),
# so the model is just an env var — the history comes along untouched.
set -a; . "$HOME/.config/free-api/keys.env"; set +a
printf '\033]0;Continue on Free\007'
clear
echo "  Continue on Free — resume a Claude Code conversation on a free model"

SEL="$(python3 "$HOME/.config/free-api/pick_session.py")" || exit 0
DIR="${SEL%%$'\t'*}"; SID="${SEL##*$'\t'}"

if [ -n "$FREE_MODEL" ]; then
  MODEL="$FREE_MODEL"; echo "  Using pinned FREE_MODEL=$MODEL"
else
  MODEL="$(python3 "$HOME/.config/free-api/pick_free_model.py")"
fi

export CLAUDE_SESSION_LABEL="Continue on Free"
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
CTX="$(python3 -c "import json;print(json.load(open('$HOME/.config/free-api/free_model_state.json'))['ctx'])" 2>/dev/null || echo 200000)"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CTX" API_TIMEOUT_MS=180000 DISABLE_TELEMETRY=1
export DT_NO_IMESSAGE=1   # Matt is at this keyboard — see the HQ Free launcher

cd "$DIR" || { echo "  Could not cd to $DIR"; exit 1; }
echo ""
echo "  Resuming in $DIR"
echo "  Loading $MODEL …"
exec claude --resume "$SID" \
  --append-system-prompt-file "$HOME/.config/free-api/mac-guardrails.md" "$@"
