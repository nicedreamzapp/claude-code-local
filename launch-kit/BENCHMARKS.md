# Benchmarks (launch-kit draft)

Every number on this page was copied from somewhere already in this repo or its history. Each row
names its source. Where a number someone will ask for doesn't exist yet, the row says
**TODO-VERIFY** and stays empty. Nothing here was measured for this page, except the no-model unit
test counts in the last section, which were run on commit `fcfe6fd`.

Unless a row says otherwise, the machine is a MacBook Pro with an **Apple M5 Max and 128 GB** of
unified memory (`docs/BENCHMARKS.md`, "System").

Three different harnesses produced these numbers, and they are not interchangeable:

- **Claude Code**: the real `claude` CLI talking to `proxy/server.py`.
- **Agent-12**: the open leaderboard at [nicedreamzapp.github.io/agent12](https://nicedreamzapp.github.io/agent12/).
  Its scores come from its own lean harness (Anvil), **not from inside Claude Code**.
- **Native Engine**: `agent/agent.py`, this repo's own small agent, with the model loaded in-process.

---

## 1. Current lineup

| Model (build) | Weights in RAM | Decode speed | Agent-12 easy / hard | Claude Code smoke test | Source |
|---|---|---|---|---|---|
| Qwen 3.8 27B, 8-bit (`lmstudio-community/Qwen3.8-27B-MLX-8bit`) | ~29 GB | 17.9 tok/s plain, 39.9 with DFlash 2 (short prompt, 400 tokens, Aug 20); 20.2 to 28.7 tok/s inside Agent-12 | 12/12 in 122.0 s, 7/8 in 401.4 s (Sep 19, DFlash 2) | Pass, 38 s (Sep 19) | `docs/BENCHMARKS.md` lineup table and methodology; commit `b803186` |
| Qwen 3.8 27B, bf16 (`mlx-community/Qwen3.8-27B-bf16`) | 54.7 GB on disk | 9.7 tok/s plain, 36.5 with DFlash 2 (600 tokens, Aug 20) | 8/8 hard with an 8000-token thinking budget | TODO-VERIFY | `docs/BENCHMARKS.md`; `docs/FULL-GUIDE.md` line 121 |
| Gemma 4 31B abliterated, 4-bit (`divinetribe/gemma-4-31b-it-abliterated-4bit-mlx`) | ~18 GB | 26 tok/s (from the Agent-12 runs) | 11/12 in 92 s, 8/8 in 348 s | Default launcher; no timed run recorded: TODO-VERIFY | `docs/FULL-GUIDE.md` lines 113, 136; `docs/BENCHMARKS.md` methodology |
| Gemma 4 12B abliterated, 4-bit | ~11 GB | TODO-VERIFY | TODO-VERIFY (README: "not on Agent-12 yet") | TODO-VERIFY | `docs/FULL-GUIDE.md` lines 134 to 137 |
| Hermes 4 14B abliterated, 4-bit | ~8 GB | TODO-VERIFY | TODO-VERIFY (README: "not on Agent-12 yet") | Works on a 16 GB Mac, user report in [#54](https://github.com/nicedreamzapp/claude-code-local/issues/54); no timing | README "Pick your AI" |
| Gemma 4 E4B, 4-bit (`mlx-community/gemma-4-e4b-it-4bit`) | ~5 GB | TODO-VERIFY | TODO-VERIFY | **Fail**: reported running a file it never wrote | `docs/BENCHMARKS.md` lineup table |
| Muse Glimmer 30B abliterated, 8-bit | ~30 to 60 GB | ~18 tok/s ("M-series Max") | Vendor-reported only | TODO-VERIFY | commit `00d252a`; `docs/FULL-GUIDE.md` lines 134, 143 |
| Nemotron 3 Nano Omni | ~20 to 66 GB | 67 tok/s with an image (measured in the separate nemotron-omni-mlx repo) | Vendor-reported only | TODO-VERIFY | `docs/FULL-GUIDE.md` line 515 |
| Claude Sonnet 5 via API (cloud reference, same Agent-12 tasks) | n/a | n/a | 12/12 in 122 s, 8/8 in 131 s | n/a | `docs/FULL-GUIDE.md` line 115 |

The smoke test (`docs/BENCHMARKS.md`): `claude -p` against this repo's server in an empty folder,
asked to create `hello.py` that prints 2+2, run it and report. Pass means the file exists and prints 4.

## 2. Same model, different harness

One contributor ran the same model and engine both ways ([agent12#1](https://github.com/nicedreamzapp/agent12/issues/1),
quoted in `docs/FULL-GUIDE.md` line 119):

| Harness | Agent-12 hard | Time |
|---|---|---|
| Anvil (Agent-12's harness) | 8/8 | 105 s |
| Claude Code as installed | 6/8 | 310 s |

Model name and hardware for that run: TODO-VERIFY (not stated in this repo; see agent12#1).

## 3. Browser test (Sep 16, 2026)

Four jobs in four real browser tabs, checked by script (`docs/BENCHMARKS.md`).

| Model | Score | One at a time | All at once |
|---|---|---|---|
| Gemma 4 31B, bf16 | 8/8 | 123 s | 110 s |
| Qwen 3.8 27B, bf16 | 8/8 | 175 s | 176 s |

## 4. Qwen 3.8 27B bf16, plain decode vs DFlash 2 (Aug 20, 2026)

`docs/BENCHMARKS.md`. Engine `mlx-dspark` 0.14 on MLX 0.32, drafter `incoai/Qwen3.8-27B-DFlash2`,
one prompt, 600 new tokens, temperature 0.

| Mode | Tokens | Wall | tok/s | Notes |
|---|---|---|---|---|
| Plain decode | 600 | 62.1 s | 9.7 | |
| DFlash 2 | 603 | 16.5 s | 36.5 | 4.43 accepted tokens per round, 137 target forwards, output identical |

Block size 5 was fastest on Metal; 4, 6, 7 and 10 were slower. DFlash 2 is from Inco AI / Z Lab and
`mlx-dspark` is ARahim3's project; neither is part of this repo.

## 5. Native Engine: time to first token on a long session

`docs/FULL-GUIDE.md` lines 400 to 409 and `assets/native-engine-ttft.png`. Gemma 4 31B 4-bit,
4.5k-token conversation.

| Cache behavior | Time to first token |
|---|---|
| Stock rotating cache (full re-prefill every turn) | 6.5 to 7.2 s |
| Native engine with the cache fix | 0.36 s |

Generation 26 to 28 tok/s. Chip and memory for this run: TODO-VERIFY (the docs say "Apple Silicon",
the image says "same MacBook"). The guide calls it "roughly 18x", the image says "~20x"; both follow
from 6.5 or 7.2 s against 0.36 s.

## 6. Server correctness fixes with before/after numbers

| Fix | Before | After | Source |
|---|---|---|---|
| Prompt cache trimmed by `.offset`, not `.step` (#46, #47) | 1/8 correct answers across independent sessions | 8/8, then 20/20 on a rerun | v0.2.0 release notes |
| Gemma 4 collapsed `<tool_call:Name{...}` opener (#45) | 7/14 | 14/14 | v0.2.0 release notes, commit `236bf72` |
| Bare top-level JSON tool calls, Qwen 2.5 Coder (#43) | 0/12 | 14/14 | v0.2.0 release notes, `docs/FULL-GUIDE.md` line 558 |
| Tool-call reliability v2 (garbled-JSON recovery, retries, temp 0.2, 8-bit KV) | infinite tool loops | 98/98 across 7 runs of `scripts/test_mlx_server.py` | `docs/TOOL-CALL-RELIABILITY.md` (March 2026, on the model of that time; current lineup: TODO-VERIFY) |
| `MLX_KV_BITS=4` (not the default) | about 1 in 14 tool calls with empty arguments | default 8-bit unaffected | v0.3.0 release notes, #42 |

## 7. History (April 2026, models no longer in the lineup)

Kept for the record in `docs/BENCHMARKS.md`. These were measured on the April 2026 flagship model, which
setup no longer installs. Don't quote them as current speeds.

| Stack | Decode | Real Claude Code task |
|---|---|---|
| Ollama + proxy | 30 tok/s | 133 s |
| llama.cpp TurboQuant + proxy | 41 tok/s | 133 s |
| MLX native server (this repo) | 65 tok/s | 17.6 s |

Gemma 4 31B on an **M4 Pro, 64 GB** (`docs/BENCHMARKS.md`): decode ~13.5 tok/s; code-mode tool
description slimming took prefill from ~60 s to ~2 s, and thinking suppression took generation for a
"hello" from ~40 s to ~1 s, for ~120 s down to ~3 to 5 s end to end.

**Not carried over from `docs/BENCHMARKS.md`:** its "vs Cloud APIs" table gives Claude Sonnet ~80 tok/s,
Claude Opus ~40 tok/s and cloud task times of ~10 s and ~15 s with no source or date. TODO-VERIFY before
anyone quotes them; they are also against the April model.

---

## How to run these yourself

All of this needs an Apple Silicon Mac except step 1.

**1. Unit tests, no model, any machine with Python 3.**

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
for t in parse_tool_calls convert_messages generate_response http_errors stream_and_retry; do
  python3 scripts/test_$t.py | tail -1
done
```

Each prints `all checks passed`. On commit `fcfe6fd` (Linux, Python 3.13) they ran 38 checks: 20, 1,
6, 4 and 7.

**2. Set up the server and a model.**

```bash
bash setup.sh                                  # picks a model for your RAM
# or by hand, see CONTRIBUTING.md:
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm
bash scripts/download-and-import.sh gemma      # or qwen, gemma12, hermes
bash scripts/doctor.sh                         # what your Mac should run
```

**3. Start the server and check it.**

```bash
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
MLX_SERVER="$PWD/proxy/server.py" \
  bash scripts/start-mlx-server.sh
curl -s localhost:4000/health                  # model, prefill size, memory
```

**4. Tool-call reliability suite** (sends Claude Code style requests to `localhost:4000`):

```bash
python3 scripts/test_mlx_server.py
```

**5. The Claude Code smoke test** (the one Qwen 3.8 passed in 38 s):

```bash
mkdir -p /tmp/smoke && cd /tmp/smoke
time ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=sk-local \
  claude --model claude-sonnet-4-6 -p "Create hello.py that prints 2+2, run it, and tell me the output."
python3 hello.py    # pass = prints 4
```

The exact prompt wording used on Sep 19 isn't recorded in the repo: TODO-VERIFY.

**6. Native Engine cache numbers.**

```bash
AGENT_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx python3 bench/agent_bench.py
```

It runs the same 6-turn conversation cached and uncached and prints per-turn prompt tokens, time to
first token and tok/s. Results go to `bench/bench_results.json`.

**7. Agent-12.** Tasks, judges and runner are in
[nicedreamzapp/agent12](https://github.com/nicedreamzapp/agent12); follow that repo's README.

When you post a result, include chip, memory, macOS, model build, and which harness it came from.

---

## TODO-VERIFY list

1. Gemma 4 12B: decode tok/s, Agent-12, Claude Code smoke test.
2. Hermes 4 14B: decode tok/s on a 16 GB Mac, Agent-12, timed smoke test.
3. Gemma 4 E4B: decode tok/s, Agent-12.
4. Gemma 4 31B: timed Claude Code smoke test.
5. Qwen 3.8 27B bf16: Claude Code smoke test.
6. Muse Glimmer 30B and Nemotron Omni: any run through Claude Code; their own Agent-12 runs.
7. agent12#1: model and hardware for the 105 s vs 310 s comparison.
8. Native Engine TTFT: chip and memory of the machine.
9. `scripts/test_mlx_server.py` 98/98: rerun on a current lineup model.
10. Cloud comparison speeds (~80 and ~40 tok/s): source and date, or drop them.
11. Exact prompt used for the Sep 19 smoke test.
12. Anything on M1 to M4 hardware for the current lineup (issue #57 asks for 16 GB and 8 GB reports).
