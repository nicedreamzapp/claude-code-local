# Claude Code Local — Run AI Coding Agents Entirely On Your Mac

Run Claude Code with a 122 billion parameter AI model on Apple Silicon. No cloud. No API fees. No data leaves your machine.

**48 tokens/second. 17 seconds per Claude Code task. 7.5x faster than llama.cpp.**

```
You → Claude Code → MLX Native Server → Qwen 3.5 122B → Apple Silicon GPU
         (no proxy, no translation — direct Anthropic API)
```

## Why This Exists

Claude Code is the best AI coding agent available. But it requires an internet connection and an API subscription. This project bridges that gap — it lets you run Claude Code powered by a local model when you're offline, want privacy, or just don't want to pay per token.

**What you get:**
- Full Claude Code experience (Cowork, projects, tools, file editing) powered by local AI
- 122B parameter model at **48 tokens/second** on Apple Silicon
- **7.5x faster** Claude Code tasks vs llama.cpp setups
- Native Anthropic Messages API — no proxy layer, no translation overhead
- 4-bit KV cache quantization on Metal GPU
- Everything runs on-device. Your code never touches a server.

## Benchmarks — Qwen 3.5 122B on M5 Max (128 GB)

### MLX Native vs llama.cpp TurboQuant

| Test | llama.cpp TurboQuant | MLX Native (ours) | Improvement |
|------|---------------------|-------------------|-------------|
| Code generation | 41.0 tok/s | **48.3 tok/s** | **+18%** |
| Claude Code E2E | 133s | **17.6s** | **7.5x faster** |

### Three Generations of Optimization

| Gen | Server | Speed | Claude Code Task |
|-----|--------|-------|-----------------|
| 1 | Ollama + proxy | 30 tok/s | ~133s |
| 2 | llama.cpp TurboQuant + proxy | 41 tok/s | ~133s |
| **3** | **MLX Native (direct)** | **48 tok/s** | **17.6s** |

The proxy was the bottleneck. Eliminating it changed everything.

## Requirements

- **Mac with Apple Silicon** (M1 Pro/Max or later)
- **Memory:** 96+ GB unified memory for 122B model (M2/M3/M4/M5 Max or Ultra)
- **Python 3.12+** with mlx-lm installed
- **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

## Quick Start

### 1. Set up the MLX environment

```bash
# Create a venv with mlx-lm
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm
```

### 2. Download the model

First run will auto-download from HuggingFace (~50 GB):

```bash
~/.local/mlx-server/bin/python3 -c "
from mlx_lm.utils import load
load('mlx-community/Qwen3.5-122B-A10B-4bit')
print('Model ready')
"
```

### 3. Start the server

```bash
~/.local/mlx-server/bin/python3 proxy/server.py
```

### 4. Launch Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

### Or: Double-click the launcher

Copy `launchers/Claude Local.command` to your Desktop. Double-click. Done.

## How It Works

Most local AI setups for Claude Code look like this:
```
Claude Code → Proxy (translates API) → Ollama/llama.cpp → Model
```

We eliminated the middle layers:
```
Claude Code → MLX Native Server (speaks Anthropic API directly) → Model
```

The server (`proxy/server.py`) is a single Python file (~250 lines) that:
1. Loads the model via Apple's MLX framework (native Metal GPU acceleration)
2. Serves the Anthropic Messages API that Claude Code expects
3. Handles Qwen 3.5's thinking/reasoning mode (strips `<think>` tags)
4. Compresses KV cache to 4-bit on the Metal GPU
5. No proxy, no translation, no overhead

## Architecture

```
Claude Code                         MLX Native Server (:4000)
    |                                     |
    |--- Anthropic Messages POST -------->|
    |                                     |--- mlx-lm generate --->  Model (GPU)
    |                                     |     (4-bit KV cache)
    |                                     |<-- tokens + text ------
    |                                     |   strip <think> tags
    |<-- Anthropic response --------------|
```

One hop. One process. One file.

## Browser Control (CDP)

Two browser options, each for different use cases:

| Tool | Browser | Use Case |
|------|---------|----------|
| **chrome-devtools-mcp** | Your real Brave/Chrome | Logged-in tasks, real sessions |
| **playwright** | Sandboxed instance | Automated jobs, scraping |

CDP controls your actual browser — already logged into everything. No re-authenticating.

## Using With Claude Max

- **Online:** Claude Code with Anthropic API (fastest, most capable)
- **Offline/Private:** Double-click `Claude Local` (48 tok/s, fully private)
- **From phone:** Dispatch or iMessage to control either mode

## Project Structure

```
├── proxy/
│   └── server.py            # MLX Native Anthropic Server (the whole thing)
├── launchers/
│   ├── Claude Local.command  # Double-click launcher
│   └── Browser Agent.command # Browser automation
├── scripts/
│   ├── download-and-import.sh
│   ├── persistent-download.sh
│   └── start-mlx-server.sh
├── docs/
│   ├── BENCHMARKS.md         # Full benchmark comparison
│   └── TWITTER-THREAD.md
├── setup.sh
└── README.md
```

## Security

Every component was audited:
- **Server** — our code, ~250 lines, zero network dependencies
- **Model** — loaded from HuggingFace's verified mlx-community
- **MLX** — Apple's official framework
- No pip packages from strangers. No telemetry. No phone-home.

## Credits

- [Claude Code](https://claude.ai/claude-code) by Anthropic
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [mlx-lm](https://github.com/ml-explore/mlx-examples) for model serving
- [Qwen 3.5](https://qwenlm.github.io/) by Alibaba
- Benchmarked on Apple M5 Max with 128 GB unified memory

## License

MIT
