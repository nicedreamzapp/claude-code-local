# Benchmarks — Claude Code Local (MLX Native Server)

## System

| | |
|---|---|
| **Machine** | MacBook Pro M5 Max |
| **Chip** | Apple M5 Max |
| **Memory** | 128 GB Unified |
| **Server** | MLX Native Anthropic Server (custom, ~800 lines Python) |
| **KV cache** | 4-bit / 8-bit quantized (MLX Metal GPU) |
| **Claude Code** | v2.1.84 |

---

## 🥊 The Lineup — Three Fighters, One Server

The same MLX server runs all three. Just swap the `MLX_MODEL` env var.

| Model | Tier | Architecture | Disk | RAM | tok/s | Best at |
|---|---|---|:---:|:---:|:---:|---|
| **Qwen 3.5 122B-A10B** | 🔵 THE BEAST | MoE 122B / 10B active, 4-bit | 65 GB | ~75 GB | **65** | Maximum throughput |
| **Llama 3.3 70B Abliterated** ⭐ | 🟠 THE WISE ONE | Dense 71B, 8-bit affine, group 64, 128K ctx | 75 GB | ~75 GB | ~7 | Hardest reasoning |
| **Gemma 4 31B Abliterated** | 🟢 THE QUICK ONE | Dense 31B, 4-bit IT | 18 GB | ~18 GB | ~15 | Daily coding, low RAM |

> ⭐ The Llama 3.3 70B build is **our own MLX-packed upload**: [`divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx`](https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx). 8-bit affine quantization with group size 64, chosen to preserve quality over minimal footprint. Built on top of [huihui-ai's abliteration](https://huggingface.co/huihui-ai) of Meta's Llama 3.3 70B Instruct.

> Qwen wins raw tok/s because only 10B of its 122B params activate per token (MoE). Llama is the slowest but the smartest at 8-bit full precision. Gemma is the lightweight champion — fits on a 64 GB Mac with room to spare.

---

## Generation Speed (Qwen 3.5 122B — measured)

| Max Tokens | Output Tokens | Time | **Tokens/sec** |
|:---:|:---:|:---:|:---:|
| 100 | 100 | 2.2s | **45.0 tok/s** |
| 500 | 500 | 7.7s | **64.8 tok/s** |
| 1000 | 1000 | 15.3s | **65.4 tok/s** |

Sustained generation at 65 tok/s. Short requests are slower (45 tok/s) due to prompt-processing overhead amortized over fewer tokens.

---

## Three Generations — Our Optimization Journey

```
Generation Speed (tok/s):

  Gen 1: Ollama + proxy              ████████████████████████████████ 30
  Gen 2: llama.cpp TurboQuant        █████████████████████████████████████████ 41
  Gen 3: MLX Native (no proxy)       █████████████████████████████████████████████████████████████████ 65
```

```
Claude Code Task Time (seconds):

  Gen 1: Ollama + proxy              █████████████████████████████████████████████████████████ 133s
  Gen 2: llama.cpp TurboQuant        █████████████████████████████████████████████████████████ 133s
  Gen 3: MLX Native (no proxy)       ████████ 17.6s
```

| Generation | Stack | tok/s | Claude Code E2E | What Changed |
|:---:|---|:---:|:---:|---|
| 1 | Ollama → Proxy → Claude Code | 30 | 133s | Baseline |
| 2 | llama.cpp TurboQuant → Proxy → Claude Code | 41 | 133s | +37% speed, 4.9x KV compression |
| **3** | **MLX Server → Claude Code (direct)** | **65** | **17.6s** | **+117% speed, eliminated proxy** |

---

## vs Cloud APIs

| | **MLX Native (Local)** | Claude Sonnet (Cloud) | Claude Opus (Cloud) |
|---|:---:|:---:|:---:|
| **Generation speed** | 65 tok/s | ~80 tok/s | ~40 tok/s |
| **Claude Code task** | 17.6s | ~10s | ~15s |
| **Cost / million tokens** | **$0** | $3 / $15 | $15 / $75 |
| **Privacy** | **100% on-device** | Cloud | Cloud |
| **Works offline** | **Yes** | No | No |
| **Monthly cost** | **$0** | $20-100+ | $20-100+ |

Our local setup **beats cloud Opus on speed** (65 vs 40 tok/s) and is within striking distance of Sonnet.

---

## Why MLX Native is Faster

| Factor | Impact |
|--------|--------|
| **No proxy** | Eliminated API translation overhead — the #1 bottleneck |
| **MLX framework** | Apple's own ML framework, built for Metal GPU + unified memory |
| **Native Anthropic API** | Server speaks Claude Code's language directly |
| **Unified memory** | Zero-copy between CPU and GPU — model weights stay in place |
| **MoE efficiency (Qwen 122B)** | Only 10B of 122B params activate per token — fast on unified memory |

---

## Gemma 4 31B — Prompt Latency Fix (M4 Pro, 64 GB)

Gemma 4 is a reasoning model that produces `<|channel>thought\n…<channel|>` blocks before every response. Combined with Claude Code's verbose tool descriptions, the effective prompt was ~5 600 tokens per turn — causing ~60 s of prefill before the first output token.

Two server-side fixes cut E2E latency by ~4×:

| Fix | Mechanism | Tokens saved | Latency before | Latency after |
|---|---|:---:|:---:|:---:|
| **Tool description slimming** | Strip all text from tool definitions in code mode, keep only name + param types | ~5 400 tok | ~60 s prefill | ~2 s prefill |
| **Thinking suppression** (`MLX_SUPPRESS_THINKING=1`) | Pre-fill an empty `<\|channel>thought\n<channel\|>` block so the model skips its reasoning chain | ~300–500 tok generated | ~40 s generation | ~1 s generation |

```
Gemma 4 31B "hello" latency (M4 Pro, warm server):

  Before fixes    ████████████████████████████████████████████████████████████████████████████ ~120 s
  After fixes     ████ ~3-5 s
```

Generation speed is hardware-bound at ~13.5 tok/s on M4 Pro (memory bandwidth limit for a 17 GB model). The latency reduction comes entirely from eliminating unnecessary tokens — not from changing the model or quantization.

---

## Methodology

- All benchmarks run on a warm server (model already loaded)
- Each test run once (not averaged — these are representative single runs)
- Claude Code E2E includes full Claude Code startup, system prompt processing, and generation
- KV cache quantized via MLX's built-in `QuantizedKVCache`
- Temperature: 0.2 for tool-call reliability runs, 0.7 for raw generation runs
- Qwen 122B numbers are measured. Gemma 4 31B (~15 tok/s) and Llama 3.3 70B (~7 tok/s) are observed approximations from real-world Claude Code usage on the same M5 Max — full benchmarks pending.
