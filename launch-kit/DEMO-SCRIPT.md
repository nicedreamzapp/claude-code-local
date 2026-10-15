# Demo script (30 to 45 s)

One take that shows the three things a visitor wants to know: the model is on the Mac, Claude Code
works normally against it, and the edit is real. It follows the same idea as the existing
`assets/demo.gif` (recorded with vhs against a live server, per commit `ac7ffbe`), but adds a test run
and the Wi-Fi-off proof.

Honesty rules for the cut: don't speed up the model's part without saying so on screen. If you trim
waiting time, put a small "sped up 2x" (or whatever it is) caption over that span. Don't show a
token/s figure in a caption unless it's on screen from the tool itself.

## Before you record

- Model already loaded and warm: start the server first and send one request, so the recording doesn't
  include a cold load. Use the Gemma 4 31B launcher model or whatever `setup.sh` picked.
  ```bash
  MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
  MLX_SERVER="$PWD/proxy/server.py" bash scripts/start-mlx-server.sh
  ```
- Make a scratch folder `~/demo` and run `claude` in it once by hand, so the "trust this folder" prompt
  and any first-run screens are already accepted. They shouldn't appear in the take.
- Terminal at about 110 x 30 characters, font 20 to 22 pt, a dark theme. Trinidad Head works if you want
  it to match the README; any terminal is fine.
- Turn Wi-Fi off for the whole take. That's the point of shot 1.

## Shot list

| Time | Shot | What you type | What appears | Cut |
|---|---|---|---|---|
| 0:00 to 0:03 | Screen recorder, not the tape. Mac menu bar, cursor clicks the Wi-Fi icon, Wi-Fi shows **off**. | nothing | Wi-Fi off | Hard cut to terminal |
| 0:03 to 0:07 | Terminal | `curl -s localhost:4000/health` | One JSON line with `"status": "ok"` and the model id, e.g. `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx`, plus memory fields | Hold 1 s on the model id |
| 0:07 to 0:10 | Terminal | `cat fib.py` | A 4-line recursive `fib(n)` with no docstring | none |
| 0:10 to 0:14 | Terminal | `claude --model claude-sonnet-4-6` (env vars are already exported in the hidden setup; the model name is just what the launchers pass, the server answers with the local model) | Claude Code's normal start screen | none |
| 0:14 to 0:17 | Claude Code | `add a docstring to fib, write test_fib.py that checks fib(10) == 55, and run it` then Enter | Prompt submitted | none |
| 0:17 to 0:34 | Claude Code | nothing | Read `fib.py`, Edit `fib.py` (diff visible), Write `test_fib.py`, Bash `python3 test_fib.py`, then a one-line answer | If the model takes longer than ~17 s, trim the thinking spinner only and caption the span "sped up Nx" |
| 0:34 to 0:36 | Claude Code | `/exit` | Back at the shell | none |
| 0:36 to 0:41 | Terminal | `cat fib.py` | The file with the docstring the model wrote | Hold 1.5 s |
| 0:41 to 0:44 | Terminal | `python3 test_fib.py && echo ok` | `ok` | End card |
| 0:44 to 0:45 | End card | | `github.com/nicedreamzapp/claude-code-local` | |

Total 45 s. To get it nearer 30 s, drop the `/exit` and `cat fib.py` shots (the diff in Claude Code
already shows the edit) and end on the test output inside Claude Code.

Optional extra shot for a longer cut (about 5 s): in a second terminal while Claude Code is still open,
`lsof -a -i -p "$(pgrep -d, -f claude)"`. The README says the only connection listed is
`localhost:4000`. Only use it if that's what your screen shows on the day.

## The vhs tape

`launch-kit/demo.tape` reproduces the terminal part (0:03 to 0:44). It doesn't do the Wi-Fi shot or the
end card; add those in the editor. Run it from the repo root on the Mac with the server already up:

```bash
brew install vhs
vhs launch-kit/demo.tape     # writes launch-kit/demo.gif and demo.mp4
```

The model's part is timed with `Sleep 20s`. Gemma 4 31B on an M5 Max should fit, but this hasn't been
timed for this exact task (TODO-VERIFY). If it runs long, raise the sleep rather than speeding up the
recording, or switch to vhs's `Wait` on screen text if your vhs version has it.

Keep the output small. The current `assets/demo.gif` is 107 KB (commit `ac7ffbe`), so it loads
instantly from Google.

## Upstream credit for the video description

Claude Code by Anthropic. Model inference with Apple's MLX and mlx-lm. Gemma 4 by Google (abliterated
4-bit MLX build from `huggingface.co/divinetribe`). Recorded with vhs by Charmbracelet.
