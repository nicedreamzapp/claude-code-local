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
