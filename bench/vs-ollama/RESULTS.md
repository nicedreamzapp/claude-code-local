# Claude Code on claude-code-local vs Claude Code on Ollama

Run on 2026-10-31 on one Mac mini (M4 Pro, 64 GB, macOS 27.2): Qwen from 10:48 to 14:15,
Gemma 4 E4B from 14:47 to 15:40, fixed-server re-run 17:00 to 17:35. Ten small agentic coding tasks, each run 3 times per setup,
all checked automatically. 450 Claude Code runs in total (60 of them on the fixed server), plus one aborted Gemma 31B run
(see "Not tested").

**Two rounds:** on Qwen 3.5, claude-code-local is faster and slightly ahead. On Gemma 4 E4B,
the first run lost to Ollama 20/30 vs 27/30 because of a claude-code-local bug: with the stock
Gemma 4 chat template, every tool result was dropped from the prompt. After the fix
(PR #89, commit `144f0a6` on main) it ties Ollama
at **27/30**, at about the same speed. See "Gemma 4 E4B round" below.

## Headline

Both setups as shipped: claude-code-local's small-Mac tier (16K context, its own sampling and
code mode) against Ollama configured the way its Claude Code page says (64K context, the
model's own sampling defaults).

| Model | Setup | Tasks passed | Valid tool calls | Median time per task |
|---|---|---|---|---|
| Qwen 3.5 2B, 8-bit | claude-code-local | **28/30 (93%)** | 165/166 (99%) | **9.1 s** |
| Qwen 3.5 2B, 8-bit | Ollama | 25/30 (83%) | 320/322 (99%) | 23.2 s |
| Qwen 3.5 4B, bf16 | claude-code-local | 30/30 (100%) | 100/100 (100%) | **20.0 s** |
| Qwen 3.5 4B, bf16 | Ollama | 30/30 (100%) | 118/118 (100%) | 28.6 s |

What this shows, and what it does not:

- **Speed is the clear result.** On the 2B the median task took 2.8x as long on Ollama
  (median of the per-task ratios), and 1.2x as long on the 4B. Most of the gap is fewer turns:
  the 2B on Ollama took 352 turns over 30 runs against 147 on claude-code-local.
- **Pass rates are close.** 28/30 vs 25/30 on the 2B is not a significant difference at this
  sample size (two-sided Fisher exact p = 0.42), and the 4B passed every run on both.
- **The 2B's most common failure is a dropped leading "/".** It writes `private/tmp/...` for
  `/private/tmp/...`. claude-code-local put the slash back 25 times in its 30 runs, so none of
  these paths reached Claude Code. On Ollama 30 such paths reached Claude Code, in 15 of the
  30 runs, and they explain 3 of Ollama's 5 failures (the file was written to the wrong
  directory and the model reported success).
- **Tool-call validity is the same on both** for these models. Qwen 3.5 writes its calls as
  `<function=Name><parameter=...>`. claude-code-local parsed 100 calls of that form for each
  model, and Ollama parsed them just as well (no leaked or unparsed calls on Ollama).
  The Gemma-specific repairs (a call missing its opener, tool-name case, aliases like
  Run/Execute to Bash) **never fired** in these runs. This benchmark does not measure them.
- **Ollama wins on raw generation speed for the 4B:** 21.0 output tokens per second of API
  time against 17.8 on claude-code-local. It still finished the tasks slower, because it took
  more turns.

## Controls (same machine, same tasks, 30 runs each)

These separate what claude-code-local's server does from its other settings.

| Model | Setup | What changed | Passed | Valid calls | Median s | Time vs claude-code-local |
|---|---|---|---|---|---|---|
| 2B | claude-code-local, code mode off | Claude Code's own system prompt kept | 29/30 | 167/175 (95%) | 9.8 | 1.24x |
| 2B | Ollama at 16K | same context as claude-code-local | 25/30 | 239/242 (99%) | 14.7 | 1.77x |
| 2B | Ollama at temperature 0.2 | claude-code-local's sampling, 64K | **20/30** | 814/815 (100%) | 12.4 | 1.82x |
| 4B | claude-code-local, code mode off | Claude Code's own system prompt kept | 30/30 | 130/130 (100%) | 28.3 | 1.19x |
| 4B | Ollama at 16K | same context as claude-code-local | 28/30 | 138/138 (100%) | 29.6 | 1.29x |
| 4B | Ollama at temperature 0.2 | claude-code-local's sampling, 64K | 29/30 | 113/113 (100%) | 23.8 | **0.97x** |

- Giving Ollama claude-code-local's sampling did **not** close the gap on the 2B: it made it
  worse (20/30, Fisher p = 0.02 against claude-code-local's 28/30). At temperature 0.2 the 2B
  looped: 815 tool calls over 30 runs, 4 runs hit the 5-minute limit, and 159 dropped-slash
  paths reached Claude Code. On the 4B it brought Ollama level on time (0.97x, a tie).
- Code mode (claude-code-local's shorter system prompt) is worth about 1.2x in time on both
  models. It does not change the pass rate. The server-side repairs stay on with code mode off.
- Context length explains part of Ollama's 2B time (14.7 s at 16K vs 23.2 s at 64K), not all.

## Per task (passes out of 3)

| Task | 2B ccl | 2B Ollama | 4B ccl | 4B Ollama |
|---|---|---|---|---|
| t01 write a file | 3 | 2 | 3 | 3 |
| t02 edit one value | 3 | 3 | 3 | 3 |
| t03 fix code from a failing test | 3 | 3 | 3 | 3 |
| t04 run a script, record output | 3 | 3 | 3 | 3 |
| t05 create, run, save output | 2 | 2 | 3 | 3 |
| t06 rename across two files | 3 | 3 | 3 | 3 |
| t07 edit JSON, keep keys | 3 | 3 | 3 | 3 |
| t08 search files, count, write | 2 | 1 | 3 | 3 |
| t09 fix a syntax error | 3 | 3 | 3 | 3 |
| t10 package + test + run test | 3 | 2 | 3 | 3 |

`results/tables.md` has every setup, including the controls, and the repair counts from the
server log.

## Fairness: what was matched and what was not

Matched:
- **The same Claude Code binary for every run** (2.1.306), invoked directly (not through a
  shell alias) as `claude -p <task> --output-format stream-json --verbose --bare
  --dangerously-skip-permissions --no-session-persistence`. Each run got a fresh empty
  `CLAUDE_CONFIG_DIR` and a fresh work dir under `/tmp`. The client environment was the same
  on both sides (the small-Mac tier's output and read caps), except for the auto-compact window,
  which followed each server's context.
- **The same model family and precision class.** 2B: MLX 8-bit (`mlx-community/Qwen3.5-2B-MLX-8bit`)
  vs GGUF Q8_0 (`qwen3.5:2b-q8_0`, digest `0689d440…b130a`). 4B: MLX bf16
  (`mlx-community/Qwen3.5-4B-bf16`) vs GGUF BF16 (`qwen3.5:4b-bf16`, digest `530cc2ae…80b5`).
  The 4B is bf16 on both sides, both converted from the same release. The 2B 8-bit formats are different
  quantizers: MLX uses group-wise affine 8-bit and Q8_0 uses 32-weight blocks with one scale.
  Both are about 8.5 bits per weight.
- **One server at a time.** The harness refuses to start if the other server's port is open.
  Ollama's model is unloaded at the end of each set and the MLX server is stopped. Each set
  starts with the same one-line warm-up request. The order was claude-code-local 2B, Ollama 2B,
  Ollama 4B, then claude-code-local 4B, so neither side always ran first.
- **No cloud traffic.** Every run went through `HTTPS_PROXY` set to a local guard that refuses
  and logs every CONNECT. The Claude Code process's sockets were sampled every 2 s with
  `lsof`. Result: 0 CONNECT attempts and no non-local sockets in any of the 300 runs.
  Nonessential traffic and telemetry were also turned off.
- **Tasks frozen before the first run.** None was changed after results came in. The only
  post-hoc change was to the classifier for "invalid" tool calls. The first version counted
  any `<tool_use_error>` (for example "String to replace not found") as invalid. It now counts
  only calls Claude Code refused to run: an unknown tool or a schema error. Every run was
  recounted from its saved transcript with the same rule.

Not matched (each side as its own docs ship it):
- **Context:** claude-code-local 16K (`MLX_MAX_CONTEXT`, small-Mac tier), Ollama 64K (its
  Claude Code page asks for at least 64K). The "Ollama at 16K" control covers this.
- **Sampling:** Claude Code sends no temperature. claude-code-local then uses 0.2. Ollama uses
  the model's Modelfile defaults (temperature 1.0, top_k 20, top_p 0.95, presence_penalty 1.5).
  The "temperature 0.2" control covers this.
- **System prompt:** claude-code-local's code mode replaces Claude Code's system prompt with a
  shorter one that includes the working directory. Ollama passes Claude Code's prompt through.
  The "code mode off" control covers this.
- **Thinking:** both servers let Qwen think (the tier sets `MLX_SUPPRESS_THINKING=0`).
  Ollama returns thinking blocks to Claude Code. claude-code-local strips them, so its token
  counts and turn transcripts do not show the thinking.
- **Vision:** off on the MLX side (`MLX_VISION=0`, because mlx-vlm is not installed on this
  machine). Ollama's GGUFs load their vision projector. The tasks are text only.
- **`--bare` gives Claude Code only Bash, Edit and Read** (no Write, Glob or Grep). This is the
  same on both sides and matches how the repo's launchers run Claude Code. Some invalid calls
  in the tables are the model asking for Write.

## Measurement notes

- "Valid tool calls" = tool_use blocks Claude Code ran, divided by (all tool_use blocks plus
  tool-call markup that leaked into plain text without becoming a tool_use).
- "Median time" is the wall time of the whole `claude -p` run. The time ratio is the median,
  over the 10 tasks, of (setup median / claude-code-local median), so a single slow task does
  not dominate it.
- Tokens per second = output tokens / Claude Code's `duration_api_ms`. That includes prefill
  and, on Ollama, thinking tokens. Treat it as rough.
- **Treat the memory numbers as rough; they are not apples to apples.** The peaks were sampled
  every 2 s over the server's processes: claude-code-local 2B 2.3 GB RSS / 3.4 GB footprint,
  4B 8.2 / 9.1 GB; Ollama 2B 7.4 / 9.5 GB, 4B 12.3 / 4.7 GB. Ollama memory-maps its weights, so
  RSS counts file pages and the footprint leaves them out. `ollama ps` reported 3.6 GB for the
  2B at 64K. Do not publish a memory comparison from these numbers without a better meter.
- **Ollama's runner (llama-server) reloaded the model mid-set:** 3 times in the 64K set, 3 in
  the 16K set and 2 in the temperature-0.2 set, beyond the one load per model. One reload in
  each set followed a SIGKILL of the runner during a run. The others came right after a Claude
  Code run was killed at the 5-minute limit. The reload time counts against Ollama, as a user
  would see it.
- **The machine was not idle.** The owner's usual always-on services ran throughout (a Python service at
  about 70% of one core with 3.5 GB, a browser, another Claude Code session). The 1-minute load
  average was 3 to 19 at set boundaries. Both sides ran under the same background load, but
  the times have that noise in them.
- 12 of the 300 runs hit the 5-minute limit. 4 of them still passed, because the files were
  right when the model was stopped. They count as passes, with time recorded as 300 s.

## Gemma 4 E4B round

This round exists to test whether the Gemma tool-call repairs matter in practice: the
missing-opener recovery, `<function=Name><parameter>` parsing, and tool-name case and alias
fixes. The 31B was left for a bigger machine.

The models:
- claude-code-local: `mlx-community/gemma-4-e4b-it-8bit`, revision `4255b21bd9a9d3fc807ef7abd80373f5e3a52a73`.
  This is stock Google weights, not abliterated, 8.9 GB. bf16 would have been 15.9 GB, with
  only about 3 GB free before the round, so it ran at 8-bit.
- Ollama: `gemma4:e4b-it-q8_0`, digest `3721c812…53f3`, 9.0 GB.

Setup:
- The same 10 tasks × 3 runs, the same no-cloud guard and the same client environment as
  the Qwen round.
- Ollama ran first this time, because the Qwen round started with MLX.
- Each side ran as it ships. claude-code-local has no E4B launcher, so it used the server
  defaults, as the Gemma launchers do. That means temperature 0.2, Gemma thinking suppressed
  and no context ceiling, with the client's compact window at 64K. Ollama ran at 64K with the
  model's defaults (temperature 1, top_k 64, top_p 0.95, thinking on).
- There is also an Ollama temperature-0.2 control.

Free memory (`memory_pressure`) was 69% before the Ollama set, 60% before claude-code-local and
60% before the control. Swap was 3.5 of 4 GB used throughout. 0 CONNECT attempts, no
non-local sockets.

| Model | Setup | Tasks passed | Valid tool calls | Median time per task | Time vs claude-code-local |
|---|---|---|---|---|---|
| Gemma 4 E4B, 8-bit | claude-code-local | 20/30 (67%) | 142/144 (99%) | 19.9 s | 1.00x |
| Gemma 4 E4B, 8-bit | Ollama | **27/30 (90%)** | 77/77 (100%) | 20.8 s | 1.05x |
| Gemma 4 E4B, 8-bit | Ollama, temperature 0.2 | **29/30 (97%)** | 81/81 (100%) | 20.4 s | 1.02x |

Ollama passed more (p = 0.06 as shipped, p = 0.01 for the control) at about the same speed.

| Task | ccl | Ollama | Ollama t0.2 |
|---|---|---|---|
| t01 write a file | 3 | 2 | 3 |
| t02 edit one value | 2 | 3 | 3 |
| t03 fix code from a failing test | **0** | 3 | 3 |
| t04 run a script, record output | 3 | 3 | 3 |
| t05 create, run, save output | 3 | 3 | 3 |
| t06 rename across two files | **1** | 3 | 3 |
| t07 edit JSON, keep keys | 3 | 2 | 3 |
| t08 search files, count, write | 3 | 2 | 2 |
| t09 fix a syntax error | **0** | 3 | 3 |
| t10 package + test + run test | 2 | 3 | 3 |

### Why claude-code-local lost: tool results never reach the model

claude-code-local flattens Claude Code's history before it applies the chat template. Each
assistant tool call becomes plain text (`<tool_call>{json}</tool_call>`), and each tool result
becomes a separate `role: "tool"` message (`proxy/server.py`, `convert_messages`). The stock
Gemma 4 template (`chat_template.jinja` in the checkpoint) skips `tool` messages in its main
loop. It only renders a tool response when the assistant message before it has structured
`tool_calls`. So **every tool result is dropped.**

Reproduced directly with the checkpoint's tokenizer: a history with a Read call and its
result renders with no trace of the result text.

The transcripts show what the model sees:
- In 18 of the 30 runs it says it has not received the tool output, for example "I have
  repeatedly called `Read` on `broken.py` but have not received the content yet."
- It then guesses at file contents.
- 8 of claude-code-local's 10 failures are among those runs.
- The tasks it still passed are mostly the ones that need no tool output (write a file, run
  and record, create and run).

On Ollama no run says this.

A second, smaller issue: the visible text of all 30 claude-code-local runs ends in a stray
`<channel|>`. That is the thinking-suppression prefill not matching how E4B closes its thought
channel. Gemma's thinking ends up as plain text instead of being skipped.

These are bugs in claude-code-local, not properties of the model. **The first run measured
them, not the repairs.** The harness was not changed to work around them. The server was fixed
instead, and that side was re-run.

### After the fix

**The fix.** PR #89, merged to main as `144f0a6`. The `mlx-fix` sets ran on the same change
before it was merged; their `git_head` in `meta.jsonl` (`fe437f5`) is a local benchmark
commit with that change applied.
- `convert_messages` keeps each tool call's id and name.
- Once per chat template, the server renders a probe conversation both ways. If the text-form
  call loses the tool result and the structured form keeps it, calls go to the template as
  structured `tool_calls`, with matching `tool_call_id`/`name` on the results.
- Every other template gets exactly the same messages as before. Qwen 3.5 2B and 4B render
  byte for byte the same; this was checked with the real tokenizers.
- A lone `<channel|>` now drops the reasoning before it, the same way the server already
  handles a lone `</think>`.
- Checked with the E4B tokenizer: both tool results are now in the prompt (before the fix,
  neither was).

**What the fix does not cover.** The cached 31B (`gemma-4-31b-it-abliterated-VL-mlx-8bit`) has
a different chat template, and that template already renders text-form tool results. Its
probe says no switch is needed, so its prompts are unchanged. The bug did not affect it.

**The re-run.** Same 10 × 3, same guard, same settings, and the Ollama sets were reused. Free
memory was 55% at the start, and swap was 3.5 of 4 GB used.

| Gemma 4 E4B, 8-bit | Passed | Valid tool calls | Median time | Says it got no tool output | Stray `<channel|>` |
|---|---|---|---|---|---|
| claude-code-local, before fix | 20/30 (67%) | 142/144 (99%) | 19.9 s | 18 runs | 30 runs |
| claude-code-local, after fix | **27/30 (90%)** | 185/185 (100%) | 21.4 s | **0** | **0** |
| Ollama | 27/30 (90%) | 77/77 (100%) | 20.8 s | 0 | 0 |
| Ollama, temperature 0.2 | 29/30 (97%) | 81/81 (100%) | 20.4 s | 0 | 0 |

| Task | ccl before | ccl after | Ollama | Ollama t0.2 |
|---|---|---|---|---|
| t01 write a file | 3 | 3 | 2 | 3 |
| t02 edit one value | 2 | 3 | 3 | 3 |
| t03 fix code from a failing test | 0 | **3** | 3 | 3 |
| t04 run a script, record output | 3 | 3 | 3 | 3 |
| t05 create, run, save output | 3 | 3 | 3 | 3 |
| t06 rename across two files | 1 | **3** | 3 | 3 |
| t07 edit JSON, keep keys | 3 | 2 | 2 | 3 |
| t08 search files, count, write | 3 | 1 | 2 | 2 |
| t09 fix a syntax error | 0 | **3** | 3 | 3 |
| t10 package + test + run test | 2 | 3 | 3 | 3 |

- With the fix, claude-code-local and Ollama as shipped tie on E4B: 27/30 each, at about the
  same speed (21.4 s vs 20.8 s median).
- **Ollama with temperature 0.2 is the best E4B setup in this round, at 29/30.** That gap is 2
  runs and is not significant.
- The fixed server's 3 failures are wrong work, the same kinds as Ollama's: hand-written
  invalid JSON (t07), a count of 0, and a 5-minute run that wrote a grep listing instead of a
  number (t08).
- **The Gemma repairs still never fired after the fix:** 0 missing-opener recoveries, 0
  function-in-tag parses, 0 case fixes, 0 aliases. All 188 tool calls came through the normal
  Gemma `<|tool_call>call:Name{...}` parse. No malformed call reached Claude Code on either
  side.

**Qwen 3.5 2B regression check.** The fixed server, same settings, 10 × 3: **28/30**, the same
as before the fix (Fisher p = 1.0). Median 8.8 s against 9.1 s before. 188/188 valid tool
calls. The server's Qwen repairs fired as before: 109 function-in-tag parses, 25 leading
slashes restored, 3 garbled calls recovered. Per task, t05 went from 2/3 to 1/3 and t08 from
2/3 to 3/3. That is noise at n = 3, and Qwen's prompts are byte-identical with or without the
fix.

Tests: 367 existing tests pass, plus 11 new ones in `tests/test_tool_result_templates.py`
(378 passed, 4 skipped for missing ripgrep, PowerShell, PIL and a live server). The ruff
syntax check passes. The standalone `scripts/test_*.py` checks pass, except
`test_mlx_server.py`, which needs a live server.

### Did the Gemma repairs fire?

**No.** On the claude-code-local side, from the server log, over 30 runs:

| Repair | Times fired |
|---|---|
| Gemma call missing its opener, recovered | **0** |
| `<function=Name><parameter>` (function-in-tag) parsed | **0** |
| Tool name case fixed | **0** |
| Tool name alias (Run/Execute → Bash and similar) | **0** |
| Tool argument alias | 0 |
| Garbled JSON / tagless call recovered | 0 |
| Re-prompt retries (none succeeded) | 2 |
| Normal Gemma `<\|tool_call>call:Name{...}` calls parsed | 39 (of 136 tool calls the log shows emitted) |

On the Ollama side, from the transcripts: no tool-call markup leaked into text (no
`<|tool_call>`, no opener-less `call:Name{`, no `<function=`), and no tool_use came back with a
wrong-case or unknown name. That holds in both Ollama sets (60 runs). None of Ollama's 4
failures involved a malformed call. Two were wrong work: invalid JSON written by hand, and a
count of 0 (in each Ollama set). In the third, the first reply was a single 4m59s generation
that never finished before the 5-minute limit.

On claude-code-local, one `<|tool_call>` leaked into text, in a run that passed.

So with Gemma 4 E4B on these tasks, neither server saw the malformed output the repairs exist
for. This benchmark cannot show that they help. They may matter on the 31B, on longer
sessions, or on older templates. That needs its own test, after the tool-result bug is fixed.

### Gemma round caveats

- **Precision:** MLX 8-bit (group-wise affine) vs GGUF Q8_0. The same release, same class.
- **Thinking:** suppressed on claude-code-local (server default for Gemma), on by default on
  Ollama (106 thinking blocks in its 30 runs). Each side as it ships.
- **Memory numbers are rough,** as in the Qwen round. Ollama memory-maps its weights.
- **Vision and audio towers:** off on MLX (mlx-vlm is not installed), loaded by Ollama.

## Not tested: Gemma 4 31B 8-bit

This pair was the one meant to exercise the Gemma repairs (a call missing its opener,
tool-name case, aliases). Ollama has a comparable tag (`gemma4:31b-it-q8_0`, 33 GB, pulled and
left on the machine). The local MLX copy is a different checkpoint
(`gemma-4-31b-it-abliterated-VL-mlx-8bit`, abliterated and not stock), so even this pair would
not have been the same weights.

It did not fit next to what the mini already runs. With the MLX copy loaded, swap reached
15.3 GB of 16 GB, free memory fell to 18%, and the first task timed out on a 626-token prompt.
That run was stopped. It is kept in `results/aborted-gemma.jsonl` and is not counted anywhere
above. To test the Gemma repairs fairly, run this pair on a machine with nothing else
resident, using a stock MLX 8-bit Gemma 4 31B so the weights match.

## Reproduce

You need Claude Code, Ollama, this repo's MLX server environment (`~/.local/mlx-server`, from
`setup.sh`) and the models listed above. New runs append to `results/`, so point them at an
empty folder to keep them apart from the published numbers:

```bash
export BENCH_RESULTS=/tmp/my-bench-results
python3 bench/vs-ollama/run_bench.py --setup mlx    --model qwen35-2b   # one set
bash bench/vs-ollama/run_all.sh 3       # mlx / mlx-raw / ollama for 2B and 4B, ABBA order
bash bench/vs-ollama/run_phase2.sh 3    # Ollama at 16K (add WITH_GEMMA=1 for the Gemma pair)
bash bench/vs-ollama/run_phase3.sh 3    # Ollama at temperature 0.2
bash bench/vs-ollama/run_gemma_e4b.sh 3 # Gemma 4 E4B: Ollama, claude-code-local, Ollama t0.2
bash bench/vs-ollama/run_gemma_fix.sh 3 3 # fixed server: Gemma 4 E4B + Qwen 3.5 2B regression
python3 bench/vs-ollama/scrub_paths.py  # replace home paths in results/ with ~ / <repo>
python3 bench/vs-ollama/summarize.py    # tables from results/
```

Files:
- `tasks.py`: the 10 tasks and their checks
- `run_bench.py`: the harness
- `results/raw.jsonl`: one line per run
- `results/meta.jsonl`: one line per set, with versions, digests, server env, load and peak memory
- `results/tables.md`: every table, including the server-side repair counts
- `results/summary.json`: the same numbers as JSON

Left out of the repo to keep it light: the 450 Claude Code transcripts (9 MB) and the server
logs (0.7 MB). The tool-call counts from the transcripts are stored per run in `raw.jsonl`,
and the repair counts from the server logs are in `tables.md`. A fresh run writes both
next to its results again (`summarize.py` copies them in).
