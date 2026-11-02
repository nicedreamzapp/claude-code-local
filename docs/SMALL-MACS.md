# Small Macs: 8 GB and 16 GB

One path for an Apple Silicon Mac with 8 GB or 16 GB of memory (anything under 32 GB), from
nothing to Claude Code running on a local model that can read files, edit them, run commands and
look at images. Every number on this page was measured; how is at the bottom.

## The short version

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

`setup.sh` reads your memory with `sysctl hw.memsize`, picks the tier below, installs `mlx-lm`
and `mlx-vlm`, downloads the model, and puts **Claude Local.command** on your Desktop. Double-click
it. That is the whole install.

Already set up? Three commands do the same by hand:

```bash
bash scripts/mac-tier.sh                         # shows the tier this Mac gets
bash scripts/download-and-import.sh small        # downloads that model (one time)
open "launchers/Small Mac.command"               # starts the server and Claude Code
```

## The two tiers

| | **16 GB tier** (16 to 31 GB) | **8 GB tier** (under 16 GB) |
|---|---|---|
| Model | [Qwen 3.5 4B](https://huggingface.co/mlx-community/Qwen3.5-4B-bf16) | [Qwen 3.5 2B](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-8bit) |
| Precision | **bf16** (nothing quantized) | **8-bit** (bf16 did not fit, see below) |
| Download | 9.1 GB | 2.7 GB |
| Sees images | yes | yes |
| Context ceiling | 16,384 tokens | 16,384 tokens |
| Server memory, just loaded | 8.9 GB | 3.0 GB |
| Server memory, peak | **10.6 GB** (15.7K-token prompt) | **4.2 GB** (15.7K-token prompt or a full screenshot) |
| Generation speed (M4 Pro) | 22 tokens/s | 92 tokens/s |
| Prompt reading speed (M4 Pro) | about 670 tokens/s | about 1,650 tokens/s |
| Claude Code test: read, edit, bash, image | all pass | all pass |
| Claude Code test: write a program and run it | right program 5 of 5 | right program 3 of 5 |

Memory is what Activity Monitor shows in its Memory column for the server process
(`phys_footprint`, in the same GB units Activity Monitor uses). Claude Code itself adds roughly
0.15 to 0.25 GB on top (measured headless). macOS and a few apps leave roughly 10 to 11 GB free on
a 16 GB Mac and 4 to 5 GB on an 8 GB Mac, which is the budget each tier was sized to.

### Why these two models

The rules: the model has to handle **vision, code and tool calls together** (a vision tower in its
`config.json` and a tool-call chat template, so no text-only models), **bf16 wherever it fits, 8-bit
only where bf16 cannot, never 4-bit**, and the peak has to fit the budget with the KV cache at a real
Claude Code context.

- **16 GB: Qwen 3.5 4B, bf16.** 9.1 GB of weights, and Qwen 3.5 keeps a full KV cache on only 1 layer
  in 4 (the others are linear attention), so 16K of context costs well under 1 GB. The next size up,
  Qwen 3.5 9B, is 10.4 GB even at 8-bit before any cache, and Gemma 4 E4B is 15.9 GB at bf16.
- **8 GB: Qwen 3.5 2B, 8-bit.** The bf16 build was tested first and did not fit: 4.6 GB just
  loaded, 5.6 GB at peak on an 8K prompt and 6.0 GB with a full screenshot, past the ~5 GB an
  8 GB Mac has free. The 8-bit build peaks at 4.2 GB with twice the context. Nothing with vision
  and tool calls fits 8 GB at bf16; nothing bigger than 2B fits it at 8-bit (Qwen 3.5 4B 8-bit is
  5.1 GB of weights alone).

## What works and what does not

Tested through this repo's own path: `proxy/server.py` on the tier's settings, real Claude Code
(`claude --bare`, the way the launchers run it), real tool calls on real files.

| Task | 16 GB (4B bf16) | 8 GB (2B 8-bit) |
|---|---|---|
| Read a file and answer from it | ✅ 6 to 8 s | ✅ 2 s |
| Fix a bug with the Edit tool | ✅ 12 to 16 s | ✅ 3 s |
| Run a script with Bash and report the output | ✅ 7 to 9 s | ✅ 2 s |
| Read a `.png` and describe it | ✅ 5 of 5, 9 to 12 s | ✅ 5 of 5, 2 s |
| Write `fizz.py`, run it, report the last line | ✅ tool flow 5 of 5, program right 5 of 5, 15 to 26 s | tool flow 5 of 5, program right 3 of 5, 3 to 4 s |
| A task that overflows the 16K context | ✅ compacts, then often loses the thread | same |

Times are wall clock for the whole Claude Code run on an M4 Pro. On a base M1, M2 or M3 expect
them to be slower; generation speed roughly follows the chip's memory bandwidth. We did not have
one to measure, so there is no number for it here.

**Where they slip.** The 2B wrote a FizzBuzz that checks 3 before 15 in two of five runs (it ran
it and reported the wrong last line truthfully). Both models do one clear step at a time well and
get lost on long, many-file jobs, especially after a compaction. Keep each request small, and use
`/clear` between unrelated tasks.

## How memory stays inside the budget

`scripts/mac-tier.sh` sets these, and every one can be overridden from your shell:

| Setting | 16 GB | 8 GB | What it does |
|---|---|---|---|
| `MLX_MODEL` | Qwen3.5-4B-bf16 | Qwen3.5-2B-MLX-8bit | the model folder |
| `MLX_VISION` | 1 | 1 | load through mlx-vlm so images reach the model |
| `MLX_MAX_CONTEXT` | 16384 | 16384 | the server refuses a longer prompt with the API's own "prompt is too long" 400 (Claude Code then compacts) and trims each answer so prompt + answer fits |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | not set | not set | Claude Code 2.1.306 treats anything under 100000 as 100000, so a 16K value would do nothing; the server's ceiling enforces the window instead |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 8192 | 4096 | caps one answer |
| `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS` | 4000 | 4000 | a big file is read in pieces instead of filling the window in one turn |
| `BASH_MAX_OUTPUT_LENGTH` | 12000 | 12000 | same for command output |
| `MLX_MAX_IMAGE_SIDE` | 1024 | 1024 | screenshots are shrunk to this longest side before the vision tower |
| `MLX_CACHE_LIMIT_GB` | 1 | 0.5 | MLX's buffer pool, which counts toward the process's memory |
| `MLX_SUPPRESS_THINKING` | 0 | 0 | lets the model think before it answers (see below) |

**What happens when the context fills.** In our runs Claude Code did not compact on its own before
the window filled. It compacted when the server answered "prompt is too long" (16,392 tokens > 16,384),
and it did that each time in our runs. The server then answers that compaction request with no tools and, if
the conversation is itself over the ceiling, drops the oldest turns so the summary fits instead of
failing again. That keeps the session alive and the memory flat. The summary a 2B or 4B model
writes is thin, which is why long jobs drift after it.

**Thinking is on for these tiers.** With thinking off, the 4B wrote a FizzBuzz with Fizz and Buzz
swapped in 4 of 5 runs; with it on, it was right in 11 of 11, at about twice the time per task.
Set `MLX_SUPPRESS_THINKING=1` to trade the other way.

## Server changes this tier needed

These are in `proxy/server.py` and help any model, not just these two:

- **Vision.** With mlx-vlm installed and a `vision_config` in the model's `config.json`, the model is
  loaded whole and images in the newest message (a Read on a `.png`, a pasted screenshot) go to it.
  Older images become `[image omitted]`. `MLX_VISION=0` turns it off; text-only models are unchanged.
- **Typed tool arguments.** Qwen 3.5 writes every value as text (`"allow_large": "True"`), and Claude
  Code rejected the whole call. Values are now converted to the type the tool's schema declares.
- **Bash timeouts in seconds.** Models wrote `"timeout": 5` meaning seconds; Claude Code read 5 ms
  and every command died at once. A value under 1000 is read as seconds.
- **The working directory is kept in code mode,** and the tool list in the short prompt matches the
  tools actually offered (`claude --bare` offers Bash, Edit and Read, not Write). Without it the 2B
  invented `/Users/username/fizz.py`.
- **A dropped leading slash is put back** when the path then lands inside the working directory
  (the 2B writes `private/tmp/proj/notes.txt`).
- **Reasoning stays out of the answer.** Qwen 3.5's template opens `<think>` in the prompt, so the
  output starts mid-thought; everything before the lone `</think>` is now dropped instead of being
  sent as part of the reply.
- **Compaction requests** are answered as text, never with a tool call, and trimmed to fit.
- **`MLX_MAX_CONTEXT`**, the hard ceiling described above.
- The log now shows prefill and decode speed separately, and `/health` reports `vision` and `max_context`.

## Picking a tier by hand

```bash
CCL_TIER=8gb  open "launchers/Small Mac.command"   # force the 8 GB tier on any Mac
CCL_TIER=16gb bash scripts/download-and-import.sh small
CCL_TIER=none bash setup.sh                         # skip the tier, use the old lineup
```

Hermes 4 14B (4-bit, text only), the earlier 16 GB pick that a user confirmed in
[#54](https://github.com/nicedreamzapp/claude-code-local/issues/54), still runs:
`MLX_MODEL=divinetribe/Hermes-4-14B-abliterated-4bit-mlx bash scripts/start-mlx-server.sh`. It is
no longer the default because it cannot see images and it is 4-bit.

## How it was measured

- **Machine:** Mac mini, Apple M4 Pro, 64 GB. Not an 8 GB or 16 GB Mac. A process's own memory
  does not depend on how much the Mac has, so the memory figures carry over; speed will not.
- **Software:** MLX 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.11, Claude Code 2.1.306, Python 3.12.
- **Memory:** `footprint -p <server pid>`, reading `phys_footprint` (now) and
  `phys_footprint_peak` (since start), after each step: model loaded, prompts of about 2K, 8K and
  15.7K tokens, a 400x200 image, and a 1568x980 screenshot. The server's own log line
  (`Memory: peak ... this request`, from `mx.get_peak_memory()`) agrees within about 1 GB; the
  difference is Python, the tokenizer and MLX's buffer pool, which MLX does not count.
- **Speed:** the server's `Speed: prefill ... decode ...` log line over the 27 requests of one
  test run; the table shows the median.
- **Claude Code test:** in an empty folder holding `notes.txt`, a broken `calc.py` and a
  `badge.png` (a red circle and the text "BLUE 7"), each prompt ran as
  `claude --bare -p "<prompt>"` with the tier's environment, then the files were checked:
  1. "Read notes.txt and tell me the secret word."
  2. "calc.py has a bug: add() subtracts. Fix it with the Edit tool so add returns a + b."
  3. "Run python3 calc.py with the Bash tool and tell me exactly what it printed."
  4. "Use the Read tool to open badge.png (an image). What shape and color is on the left, and
     what text is on the right?"
  5. "Create fizz.py that prints FizzBuzz for 1 to 15, one per line. Then run it with Bash and tell
     me the last line it printed."

  The five tests ran twice per tier (once more after the last code change), and tests 4 and 5
  three more times each. The launcher itself was run end to end once
  (`Small Mac.command` on the 8 GB tier, port 4000): it started the server with the tier's model
  and settings, and Claude Code read the image and ran the script through it.
- **Context overflow:** seven ~2,200-token files read one per turn with the 16 GB tier, and three
  read in one turn with the ceiling lowered to 6,000.

Run it on your own 8 GB or 16 GB Mac and post what you see in
[#57](https://github.com/nicedreamzapp/claude-code-local/issues/57): `bash scripts/doctor.sh --report`
prints a privacy-safe block to paste.
