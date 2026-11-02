#!/bin/bash
# Installs the Claude Code CLI if it is missing, the same way on macOS and Linux: Anthropic's
# official native installer (it puts `claude` in ~/.local/bin). Sourced by setup.sh and
# portable/setup-linux.sh. CCL_NO_CLAUDE_INSTALL=1 skips the install and only prints the command.
CCL_CLAUDE_INSTALL_CMD="curl -fsSL https://claude.ai/install.sh | bash"

ccl_find_claude() {
  local c
  for c in "$(command -v claude 2>/dev/null)" "$HOME/.local/bin/claude" "$HOME/.claude/local/claude" \
           /opt/homebrew/bin/claude /usr/local/bin/claude; do
    if [ -n "$c" ] && [ -x "$c" ]; then printf '%s\n' "$c"; return 0; fi
  done
  return 1
}

# Prints the claude path on stdout (empty if it still isn't there); messages go to stderr.
ccl_ensure_claude() {
  local found
  if found="$(ccl_find_claude)"; then printf '%s\n' "$found"; return 0; fi
  if [ "${CCL_NO_CLAUDE_INSTALL:-0}" = "1" ]; then
    echo "Claude Code CLI not found (CCL_NO_CLAUDE_INSTALL=1). Install it with:" >&2
    echo "  $CCL_CLAUDE_INSTALL_CMD" >&2
    return 1
  fi
  echo "Installing the Claude Code CLI (official installer: $CCL_CLAUDE_INSTALL_CMD) ..." >&2
  if command -v curl >/dev/null 2>&1 && curl -fsSL https://claude.ai/install.sh | bash >&2; then
    if found="$(ccl_find_claude)"; then
      echo "Claude Code installed → $found" >&2
      printf '%s\n' "$found"; return 0
    fi
  fi
  echo "WARNING: could not install the Claude Code CLI. Install it with:" >&2
  echo "  $CCL_CLAUDE_INSTALL_CMD" >&2
  return 1
}
