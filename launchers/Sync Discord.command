#!/bin/bash
# Sync Discord — push the live repo list to the NiceDreamzApps #projects showcase.
# Double-click to run. Posts the first time, edits the same message after that
# (no spam). Pulls every public repo from the nicedreamzapp GitHub account.

SHOWCASE="$HOME/HQ/scripts/discord_projects_showcase.py"
WH_FILE="$HOME/HQ/scripts/.discord_projects_webhook"
PYTHON="$(command -v python3 || echo /opt/homebrew/bin/python3)"

clear
echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Sync Discord                                 ║"
echo "  ║  NiceDreamzApps · #projects showcase          ║"
echo "  ║  All public repos · edits in place            ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Load the webhook URL (sourced from a 600-perm config file)
WH=""
if [[ -f "$WH_FILE" ]]; then
  WH="$(grep -E '^DISCORD_PROJECTS_WEBHOOK=' "$WH_FILE" | tail -1 | cut -d= -f2-)"
fi

if [[ -z "$WH" || "$WH" == "PASTE_WEBHOOK_URL_HERE" ]]; then
  echo "  ⚠️  No webhook configured yet."
  echo ""
  echo "  1. In Discord, open the channel you want the showcase in."
  echo "  2. Edit Channel → Integrations → Webhooks → New Webhook → Copy URL."
  echo "  3. Paste it into:"
  echo "       $WH_FILE"
  echo "     (replace PASTE_WEBHOOK_URL_HERE)"
  echo ""
  echo "  Then double-click this launcher again."
  echo ""
  read -n 1 -s -r -p "  Press any key to close..."
  echo ""
  exit 1
fi

echo "  Pulling repos and updating the showcase..."
echo ""
"$PYTHON" "$SHOWCASE" --webhook "$WH"
STATUS=$?
echo ""
if [[ $STATUS -eq 0 ]]; then
  echo "  ✅ Done."
else
  echo "  ❌ Something went wrong (exit $STATUS). Scroll up for details."
fi
echo ""
read -n 1 -s -r -p "  Press any key to close..."
echo ""
