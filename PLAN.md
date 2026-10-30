# Phone version: plan and prototype (branch `phone-android`)

Goal: Claude Code running against a model on the phone itself, no cloud.

## What we depend on

| Outside thing | Why we can't build it | How we'd replace it |
|---|---|---|
| Termux (app, GPL, F-Droid/GitHub) | It is the Linux userland on Android: shell, Python, pkg | Our own Android app that embeds the same pieces (see "Real release") |
| llama.cpp `android-arm64` release build (MIT, pinned by tag + sha256) | Same engine the Linux/Windows path already uses | Build it ourselves from source in CI with the NDK |
| Hugging Face (Qwen 3.5 2B GGUF, Apache 2.0, pinned sha256) | The weights | Mirror the pinned files on our own server |
| Claude Code `linux-arm64-musl` npm build (pinned sha512) | Claude Code itself; there is no Android build | Nothing to replace; it is the client this kit exists for |
| Alpine `musl` loader, one 700 KB file (MIT, pinned sha256) | Claude Code's musl build needs only musl libc; Android's libc is bionic | Build musl from source (tiny, one `make`) |
| Termux `patchelf` | Points Claude Code's ELF interpreter at our loader | ~60 lines of Python that rewrite PT_INTERP |

## Chosen path: Termux on Android

`curl -fsSL .../install-android.sh | bash` inside Termux. It installs git, python,
patchelf and ripgrep, clones the repo, and runs `portable/setup-android.sh`, which
calls the same `portable/ccl.py` the Linux/Windows path uses, now Android aware:

- detects Termux, picks the pinned `llama-<tag>-bin-android-arm64` build (exact name, not the Snapdragon variant)
- sizes models against phone RAM (75% of RAM, 0.75 GB runtime overhead)
- downloads Claude Code's musl build + Alpine's musl loader, verifies both, `patchelf --set-interpreter` (nothing else; setting an rpath too corrupts the bundled JS), wraps it as `$PREFIX/bin/claude`
- runs llama-server with Termux's linker-exec mode off, or ggml finds no CPU backend

Why this path: Node is not needed (Claude Code is now a native binary), nothing is
rooted, and every byte is pinned and checksummed. An app is the real release, but
the app would ship exactly these pieces, so this proves the stack first.

## Model pick

Qwen 3.5 2B (`Qwen/Qwen3.5-2B`: `vision_config.model_type=qwen3_5`, chat template has `tools` and `<tool_call>`), added to the catalog as the phone tier.

- 8 GB phone: 2B at **BF16** (needs 5.2 of 5.8 GB)
- 6 GB phone: 2B at **Q8_0** (BF16 does not fit)
- 12 GB+: 4B at Q8 or better
- Never below Q8; picker unchanged

## Measured on the mini's emulator (Android 16, arm64, 6 cores, headless)

- **Install:** the one-liner finishes on a fresh Termux; `claude --version` = 2.1.312
- **Server:** llama-server + proxy healthy about 6 s after start
- **End to end:** `claude -p` wrote `add.py`, ran it, answered "The output is **5**". 8 turns, 49 s, Write and Bash tool calls
- **Vision:** red test image answered "red"
- **Q8_0 (5 GB guest):** prompt 242 tok/s, generation 51 tok/s (llama-bench pp512/tg64). Inside Claude Code: 92 to 206 tok/s prompt, about 48 tok/s generation
- **BF16 (same build):** prompt 5.1 tok/s, generation 3.5 tok/s. The CPU variant it loads has no BF16 matmul path, so BF16 runs about 47x slower on prompts
- The emulator runs on the mini's Apple cores, so phone speeds will be lower. Treat these numbers as a ceiling

Where the emulator stopped: an 8 GB guest kept getting SIGTERMed by the mini's
Forge Memory Guard, because Song Forge reserves 60 of the mini's 64 GB. The full
run was done with a 5 GB guest, so it ran Q8. The BF16 tier was only benchmarked.

## Real phone: Cricket U680AC (4 GB)

Android 13, 8 ARMv8.0 cores (2.2 + 1.6 GHz, no dotprod/i8mm), 3.7 GB RAM with about
1.7 GB actually free. Below the 6 GB tier, so this run used the new catalog entry
**Qwen 3.5 0.8B** (`Qwen/Qwen3.5-0.8B`: `vision_config.model_type=qwen3_5`, `<tool_call>` template)
at **Q8_0** (`CCL_MODEL_ID=qwen3.5-0.8b CCL_BUDGET_GB=2.4`), ctx 16384.

- **Termux:** official v0.118.3 GitHub arm64 APK, sha256 matched the release's `sha256sums`. `adb install` was refused (`INSTALL_FAILED_VERIFICATION_FAILURE`) until adb-install verification was switched off for the install, then restored. The GitHub build is debuggable, so `adb shell run-as com.termux` drives it with no screen taps
- **Install:** the one-liner finished in **213 s** on Wi-Fi (pkg upgrade, clone, llama.cpp b11669, 0.76 GB model, mmproj, Claude Code 2.1.312 + musl loader)
- **Server start:** llama-server + proxy healthy in **14 s**
- **llama-bench (8 threads):** prompt **15.2 tok/s** (pp512), generation **7.5 tok/s** (tg64)
- **Inside Claude Code:** first turn 3084 prompt tokens at **11.4 tok/s** (270 s), generation **2.5 to 2.9 tok/s**. The second try reused the prompt cache (4 new tokens), so it took 57 s instead of 316 s
- **RAM:** free memory went from 1.73 GB to a low of **1.09 GB** during the run. No OOM, Termux was not killed
- **End to end: failed at the tool call, twice.** Both tries the model answered "I'll create add.py..." and emitted `<function=Write><file_path>...</file_path><content>...` instead of `<parameter=file_path>`. The proxy logged `function-in-tag but no params` and returned plain text, so Claude Code ended after 1 turn; add.py was never written or run

What this tells us:
1. The whole stack installs and runs on a real 4 GB phone; the weak link is the 0.8B model's tool format, not the phone
2. The proxy could accept `<function=X><arg>value</arg>` as a fallback (the intent is clear in the output); worth trying before ruling 0.8B out
3. With no override the picker chooses 0.8B at **BF16** on this phone (2.68 of 2.78 GB budget), but only 1.7 GB is free and BF16 has no fast CPU path here. On Android the budget should come from MemAvailable, not MemTotal
4. 4 GB phones work as a server-start test, not as a coding agent; the 6 GB floor stands

**Re-test after parser fix (c064370):** the Write call now parses and **add.py was written** (`print(2 + 3)`, prints 5 when run by hand), but the model never ran it. Try 1 (338 s, 4 turns): after Write it called tools named `Run` and `Execute`, which don't exist, then said it had no tool to run Python; the answer never said 5. Try 2: same Write, then `Run`/`Execute` and 30+ calls to lowercase `bash` (tool names are case sensitive), so I stopped it at about 20 min. Server start 12 to 17 s, prompt 12.9 tok/s (3062 tokens, 237 s first turn), generation 5.5 to 6.0 tok/s, free RAM low 1.04 GB. Next fix to try: match tool names case-insensitively and map `Run`/`Execute` to Bash in the proxy

**Run 3 (tool-name aliases, f7f04a0): passed.** Try 1 hit the 15 min cap with no finished turn: the model ran on for about 1,700 tokens, the proxy's 600 s llama timeout fired and Claude Code's retry did the same. Try 2 (694 s, including about 4 min waiting for try 1's abandoned request to drain from llama-server): Write made add.py, the model called `Run`, the log shows `Tool name alias applied: Run -> Bash`, it ran and answered "The output is: `2 + 3 = 5`" in 3 turns. Server start 14.6 s, prompt 12.1 tok/s (3062 tokens, 253 s first turn), generation 4.1 to 5.3 tok/s, free RAM low 1.13 GB. Only the Run alias fired; no argument aliases. Left: cap max_tokens or add a stop for runaway turns, and cancel the llama request when the client disconnects

## Left for a real release

1. Run it on a real 8 GB and a real 12 GB phone: speed, heat, battery, whether Android's low-memory killer leaves Termux alone
2. BF16 speed: try the `armv8.6`/`armv9.2` CPU variants (i8mm + bf16) on phones that have them, or the Snapdragon OpenCL/Hexagon build. This decides whether 8 GB phones really get BF16
3. Termux has to stay in the foreground (or hold a wake-lock) or Android cuts its network. `claude-local` should run `termux-wake-lock`
4. Claude Code's Bash tool runs without termux-exec, so `#!/usr/bin/env` scripts fail inside it. Fix with `CLAUDE_CODE_SHELL_PREFIX`
5. Wire `install.sh` to hand off to the Android path on Termux, add a README section, mirror the weights
6. The app: one APK that ships llama.cpp (NDK build), the proxy, Claude Code and the musl loader, with no Termux needed

## iOS

No build here. iOS has no terminal, and apps cannot fork/exec other binaries or
load executable code at runtime. Claude Code's binary therefore cannot run on an
iPhone in any form.

What is possible:
- an iOS app running llama.cpp (Metal) with the proxy compiled in, serving the Anthropic API on the local network so a Mac or Android Claude Code can use the iPhone's model
- a native chat/agent UI inside the app (our own agent loop, like `agent/agent.py`), not Claude Code itself

Either way it is an App Store app, with review, not a script.
