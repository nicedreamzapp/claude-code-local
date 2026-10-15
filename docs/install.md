---
title: Install Claude Code Local on an Apple Silicon Mac
description: >-
  Install Claude Code Local with one command or by hand. Covers requirements,
  what setup.sh changes on your Mac, model downloads, the doctor script and how
  to uninstall.
---

# Install

## Requirements

| Need | Why | Checked by |
|---|---|---|
| macOS on Apple Silicon (M1 or newer) | MLX runs on Apple's GPU and unified memory | `install.sh` stops unless `uname -s` is `Darwin` and `uname -m` is `arm64`; `setup.sh` and `doctor.sh` stop unless `arm64` |
| `git` | `install.sh` clones the repo | `install.sh` stops and suggests `xcode-select --install` |
| Homebrew | Installs Python 3.12 | `setup.sh` installs Homebrew if `brew` is missing |
| Python 3.12 | The MLX virtualenv is created with `python3.12` | `setup.sh` runs `brew install python@3.12` if missing |
| Claude Code CLI | The thing you actually use | `setup.sh` warns if `claude` is not found; install with `npm install -g @anthropic-ai/claude-code` |
| Disk space | Models are 5 to 30 GB | `doctor.sh` warns under 25 GB free |

!!! warning "Old Claude Code versions"
    If a launcher asks you to sign in to a Claude account, your `claude` is too old. Update it with `npm install -g @anthropic-ai/claude-code`. See [Troubleshooting](troubleshooting.md#claude-code-asks-me-to-log-in).

## One-command install

```bash
curl -fsSL https://raw.githubusercontent.com/nicedreamzapp/claude-code-local/main/install.sh | bash
```

`install.sh` does this, in order:

1. Refuses to run on anything that is not macOS on `arm64`, or if `git` is missing.
2. Prints the chip and memory. Under 16 GB it warns that tool calling will be less reliable.
3. Clones the repo to `~/claude-code-local` with `git clone --depth 1`, or runs `git pull --ff-only` if a checkout is already there. Set `CCL_DIR` to use a different folder. If the folder exists and is not a git checkout, it stops.
4. Runs `setup.sh` from the checkout.

## Install from a clone

If you want to read the scripts first:

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

## What `setup.sh` changes on your Mac

| Step | Result |
|---|---|
| Homebrew | Installed if missing |
| Python | `brew install python@3.12` if `python3.12` is missing |
| MLX virtualenv | `~/.local/mlx-server`, with `pip` and `mlx-lm` upgraded inside it |
| Model | Picked by memory (table below) and downloaded into the Hugging Face cache by calling `mlx_lm.utils.load()` once |
| Server | `~/.local/mlx-native-server/server.py`, a **symlink** to `proxy/server.py` in your checkout. Edit the repo copy and restart the server; there is no second copy to drift. |
| Desktop launcher | `~/Desktop/Claude Local.command`, generated with the chosen model baked in |
| `keep going` | `~/.local/bin/keepgoing` symlinked to `bin/keepgoing`, and a `source .../scripts/keepgoing-shell.sh` line appended to `~/.zshrc` if it is not there yet |
| Phone scripts (optional) | Copies `scripts/imessage-*.sh` to `~/.claude/` only if they exist in the checkout. They are not bundled in this repo, so setup prints a pointer to [claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone) instead. |

### Model chosen by memory

| `hw.memsize` | Model ID | Label printed by setup |
|---|---|---|
| 96 GB and up | `lmstudio-community/Qwen3.8-27B-MLX-8bit` | Qwen 3.8 27B, 8-bit (~29 GB) |
| 64 to 95 GB | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` | Gemma 4 31B Abliterated, 4-bit (~18 GB) |
| 32 to 63 GB | `divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text` | Gemma 4 12B Abliterated, 4-bit (~11 GB) |
| 16 to 31 GB | `divinetribe/Hermes-4-14B-abliterated-4bit-mlx` | Hermes 4 14B Abliterated, 4-bit (~8 GB) |
| Under 16 GB | `mlx-community/gemma-4-e4b-it-4bit` | Gemma 4 E4B, 4-bit (~5 GB, may struggle with tool calls) |

!!! note "About “abliterated” models"
    Several defaults are abliterated builds, which have their built-in refusals turned down. The README is clear that this is not a general upgrade and that each model's own license still applies. Original releases from Google, Qwen and Nous Research are linked on the [Credits](project-credits.md#models) page if you would rather run those.

## Check your Mac first: `doctor.sh`

```bash
bash scripts/doctor.sh          # what fits, what is installed, what is cached
bash scripts/doctor.sh --bench  # also time a 100-token generation on the recommended model
```

It reports chip, memory, cores, macOS version, the Hugging Face cache (`$HF_HOME` or `~/.cache/huggingface`) and its size, free disk, whether `mlx-lm` is installed in `~/.local/mlx-server`, every model folder in the cache, and a recommendation by memory tier. Its tiers match `setup.sh`.

## Install by hand

```bash
# 1. MLX virtualenv
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm

# 2. Download a model (one time)
bash scripts/download-and-import.sh gemma   # or qwen, gemma12, hermes, llama, or any HF id

# 3. Start the server from the repo copy (setup.sh did not install one)
MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
MLX_SERVER="$PWD/proxy/server.py" \
  bash scripts/start-mlx-server.sh

# 4. In another terminal, point Claude Code at it
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

`download-and-import.sh` accepts these short names:

| Argument | Model |
|---|---|
| `gemma`, `gemma31`, `31b`, or nothing | `divinetribe/gemma-4-31b-it-abliterated-4bit-mlx` |
| `qwen`, `qwen38`, `27b` | `lmstudio-community/Qwen3.8-27B-MLX-8bit` |
| `gemma12`, `12b` | `divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text` |
| `hermes`, `14b` | `divinetribe/Hermes-4-14B-abliterated-4bit-mlx` |
| `llama`, `llama70`, `70b` | `divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx` (older, ~75 GB, kept for existing users) |
| anything else | Treated as a Hugging Face model ID |

`MLX_MODEL` overrides the model for any of the named choices. On a flaky connection, `bash scripts/persistent-download.sh <same arguments>` re-runs the download every 10 seconds until it succeeds.

## Uninstall

```bash
bash uninstall.sh                # removes everything below, including downloaded models
bash uninstall.sh --keep-models  # keeps the Hugging Face model weights
```

It removes `~/Desktop/Claude Local.command`, `~/.local/mlx-native-server`, `~/.local/mlx-server`, the `imessage-*.sh` phone scripts and `screen-to-phone-config.sh` in `~/.claude/`, kills the process listening on port 4000, deletes `/tmp/mlx-server.log`, and, unless you pass `--keep-models`, deletes from `~/.cache/huggingface/hub` the five models `setup.sh` can pick plus two older picks (`mlx-community/Qwen3.5-122B-A10B-4bit`, `mlx-community/Qwen3.5-4B-4bit`). Models you downloaded any other way are left alone. It leaves Homebrew, Python 3.12 and the Claude Code CLI alone.

!!! info "TODO-VERIFY"
    `uninstall.sh` does not remove the `~/.local/bin/keepgoing` symlink or the `keepgoing-shell.sh` line that `setup.sh` adds to `~/.zshrc`. Remove those by hand if you want them gone.

Next: [Quickstart](quickstart.md).
