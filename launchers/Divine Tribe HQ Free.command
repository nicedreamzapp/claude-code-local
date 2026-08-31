#!/bin/bash
# Divine Tribe HQ (Free) — run the business with the REAL Claude Code CLI on a free
# cloud model. Same workspace as Divine Tribe HQ.app. No middleware: Claude Code talks
# straight to OpenRouter's Anthropic-compatible API. Free = 50 req/day, resets 5pm PT.
set -a; . "$HOME/.config/free-api/keys.env"; set +a
cd "$HOME/Desktop/PROJECTS/ineedhemp website" || cd "$HOME"
MODEL="${FREE_MODEL:-stealth/ox-alpha}"   # Ox Alpha stealth, free thru ~Aug 27 2026, 1M ctx; prior default: nvidia/nemotron-3-super-120b-a12b:free
export CLAUDE_SESSION_LABEL="Divine Tribe HQ (Free)"
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000 API_TIMEOUT_MS=180000 DISABLE_TELEMETRY=1
rem=$(curl -s -m 6 -D - -o /dev/null https://openrouter.ai/api/v1/messages -H "x-api-key: $OPENROUTER_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" -d '{"model":"nvidia/nemotron-nano-9b-v2:free","max_tokens":1,"messages":[{"role":"user","content":"."}]}' | grep -i '^x-ratelimit-remaining:' | tr -dc '0-9')
[ -n "$rem" ] && echo "  OpenRouter free requests left today: $rem / 50 (resets 5pm PT)"
printf '\033]0;Divine Tribe HQ (Free)\007'
echo "  Divine Tribe HQ (Free) — real Claude Code on OpenRouter free · workspace: ineedhemp website"
exec claude --model "$MODEL" "$@"
