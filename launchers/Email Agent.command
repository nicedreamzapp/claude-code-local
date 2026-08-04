#!/bin/bash
# EMAIL AGENT — Divine Tribe's local customer-email handler.
# Drafts every reply in Matt's voice off the local model (fast), you approve each send.
# Double-click. Pulls your reply queue, easiest first.

# make sure the writing model (Qwen on :4000) is warm
bash "$HOME/Desktop/PROJECTS/Local AI Setup/smart-router/warm_pool.sh" start >/dev/null 2>&1
echo -n "  warming the local model"
for i in $(seq 1 90); do
  curl -s "http://127.0.0.1:4000/health" >/dev/null 2>&1 && break
  echo -n "."; sleep 1
done
echo " ready."

clear
echo ""
echo "  ✉️  EMAIL AGENT — local customer-email handler"
echo "  → drafts in Matt's voice, easiest first · you approve every send"
echo "  → [g]send  [e]dit  [r]e-draft  [s]kip  [q]uit"
echo ""

exec python3 "$HOME/.local/email-agent/agent.py"
