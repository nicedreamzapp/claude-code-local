#!/bin/bash
# Divine Tribe HQ (Free) — YOUR terminal for running the business, on a free
# cloud model. Double-click to launch.
#
# Runs on your own agent engine (../agent/agent.py), the same one behind Local AI
# and DeepSeek V4 Flash — no Claude Code, no Anthropic CLI, no telemetry. The
# engine's HTTP backend speaks the plain Messages protocol, which is exactly what
# OpenRouter serves, so the only thing that comes from Anthropic here is the
# wire format. Free = 50 requests/day, resets 5pm PT.

set -a; . "$HOME/.config/free-api/keys.env"; set +a
cd "$HOME/Desktop/PROJECTS/ineedhemp website" || cd "$HOME"
printf '\033]0;Divine Tribe HQ (Free)\007'
echo "  Divine Tribe HQ (Free) — your terminal on OpenRouter free · workspace: ineedhemp website"
echo ""

# Model is chosen LIVE at every launch by pick_free_model.py: best currently-free,
# tool-capable, >=128K-ctx model on OpenRouter. It reports anything that died since
# last launch. Pin one with FREE_MODEL=<id> to skip the lookup.
#
# FREE_HARNESS=0 tells the picker to introduce itself as this machine's own agent
# rather than as Claude Code. OpenRouter gates a few free models (thinkingmachines
# /inkling today) to harnesses registered on openrouter.ai/apps, and claiming to be
# one we are not is not how we get access — so those simply drop out of the pool
# and the picker returns something that actually answers us.
export FREE_HARNESS=0
if [ -n "$FREE_MODEL" ]; then
  MODEL="$FREE_MODEL"; echo "  Using pinned FREE_MODEL=$MODEL"
else
  MODEL="$(python3 "$HOME/.config/free-api/pick_free_model.py")"
fi
[ -n "$MODEL" ] || { echo "  no free model answered — try again in a minute."; read -n 1; exit 1; }

# The engine takes ONE system-prompt file, so the business knowledge and the
# Mac corrections get concatenated into one. mac-guardrails.md goes LAST so its
# "these override anything in CLAUDE.md that contradicts them" actually holds.
PROMPT="$HOME/.config/free-api/hq-free-prompt.md"
cat "$HOME/Desktop/PROJECTS/ineedhemp website/CLAUDE.md" \
    "$HOME/.config/free-api/mac-guardrails.md" > "$PROMPT" 2>/dev/null

CTX="$(python3 -c "import json;print(json.load(open('$HOME/.config/free-api/free_model_state.json'))['ctx'])" 2>/dev/null || echo 200000)"

export AGENT_TITLE="Divine Tribe HQ (Free)"
export AGENT_MODEL="$MODEL"
export AGENT_BACKEND="http"
export AGENT_BASE_URL="https://openrouter.ai/api"
export AGENT_AUTH_TOKEN="$OPENROUTER_API_KEY"
export AGENT_WHERE="on OpenRouter's free tier, reached over the network"
export AGENT_PROMPT_FILE="$PROMPT"
export AGENT_CTX_LIMIT="$CTX"
export AGENT_TEMP="0"

echo "  Loading $MODEL …"
exec python3 "$HOME/Desktop/PROJECTS/Local AI Setup/agent/agent.py"
