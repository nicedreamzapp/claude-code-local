# Contributing

Thanks for taking a look. Most of the best fixes in this repo came from people running it on
hardware I don't own, so a bug report from your Mac is genuinely useful even if you never send code.

## Reporting a problem

Open an [issue](https://github.com/nicedreamzapp/claude-code-local/issues/new/choose). The things
that make it quick to fix:

- your Mac (chip and RAM) and macOS version
- the model you ran
- what `bash scripts/doctor.sh` prints
- the last lines of the server log, and what you typed in Claude Code when it went wrong

## Running the server from the repo

```bash
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm

MLX_MODEL=divinetribe/gemma-4-31b-it-abliterated-4bit-mlx \
  bash scripts/start-mlx-server.sh

ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=sk-local \
  claude --model claude-sonnet-4-6
```

If you installed with `setup.sh`, the running server is a symlink back to `proxy/server.py` in
this repo, so edit it here and restart the server.

## Tests

The suite in `tests/` runs anywhere, Linux included. MLX is stubbed, so it needs no model, no GPU
and no network, and it also runs the standalone `scripts/test_*.py` checks. CI runs it on every
push and pull request.

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

- `python3 scripts/test_parse_tool_calls.py` checks the tool-call parser. It needs no model and no
  running server, so run it before any change to `parse_tool_calls`.
- `python3 scripts/test_mlx_server.py` sends real Claude Code style requests to a server on
  `localhost:4000` and checks the model can make multi-step tool calls.

If you fix a parsing bug, adding the model's exact output as a new case in
`test_parse_tool_calls.py` is the most helpful thing you can do.

## Pull requests

Small and focused is easiest to review. Say which Mac and model you tested on. If you reproduced
the bug straight against the server (curl or the test script) instead of through Claude Code,
mention it, that makes it much faster to confirm.

## Credit

Everyone whose PR lands is named in the README contributor line and in the credits table in
[docs/FULL-GUIDE.md](docs/FULL-GUIDE.md#-credits), with what they fixed.
