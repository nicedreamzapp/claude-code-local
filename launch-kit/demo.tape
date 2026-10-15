# claude-code-local demo, terminal part (see launch-kit/DEMO-SCRIPT.md)
#
# Needs: macOS on Apple Silicon, vhs (brew install vhs), Claude Code, and the
# MLX server already running and warm on localhost:4000. Run from the repo root:
#
#   vhs launch-kit/demo.tape
#
# Run `claude` once in ~/demo by hand first so the trust-folder prompt is gone.

Output launch-kit/demo.gif
Output launch-kit/demo.mp4

Require claude
Require curl
Require python3

Set Shell "bash"
Set FontSize 20
Set Width 1280
Set Height 760
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set PlaybackSpeed 1.0

# Setup the viewer doesn't need to see: a fresh fib.py and the same env the
# launchers use (point Claude Code at the local server, turn off its
# non-essential traffic as documented in the README).
Hide
Type "mkdir -p ~/demo && cd ~/demo && rm -f test_fib.py" Enter
Type `printf 'def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n' > fib.py` Enter
Type "export ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=sk-local" Enter
Type "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1" Enter
Type "export CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1 CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1" Enter
Type "clear" Enter
Sleep 500ms
Show

# 0:03  the model answering is on this Mac
Type "curl -s localhost:4000/health"
Sleep 300ms
Enter
Sleep 2.5s

# 0:07  the file before
Type "cat fib.py"
Enter
Sleep 2s

# 0:10  Claude Code, unchanged, pointed at localhost
Type "claude --model claude-sonnet-4-6"
Enter
Sleep 4s

# 0:14  the task
Type "add a docstring to fib, write test_fib.py that checks fib(10) == 55, and run it"
Sleep 500ms
Enter

# 0:17  model works: Read, Edit, Write, Bash. TODO-VERIFY the timing on your Mac.
Sleep 20s

# 0:34  leave Claude Code
Type "/exit"
Enter
Sleep 1.5s

# 0:36  the file after
Type "cat fib.py"
Enter
Sleep 3s

# 0:41  and the test passes
Type "python3 test_fib.py && echo ok"
Enter
Sleep 3s
