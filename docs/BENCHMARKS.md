# Benchmarks — Claude Code Local

**Machine:** MacBook Pro M5 Max, 128 GB Unified Memory
**Date:** March 26, 2026
**Model:** Qwen 3.5 122B-A10B (4-bit, MLX native)
**Server:** MLX Native Anthropic Server (our custom build)
**Claude Code:** 2.1.84

## MLX Native vs llama.cpp TurboQuant — Same Model, Same Hardware

| Test | llama.cpp TurboQuant | MLX Native | Improvement |
|------|---------------------|------------|-------------|
| Code generation | 41.0 tok/s | **48.3 tok/s** | **+18%** |
| Q&A | 43.7 tok/s | 42.3 tok/s | ~same |
| Claude Code E2E | 133s | **17.6s** | **7.5x faster** |

The raw token generation is 18% faster. But the real win is Claude Code end-to-end: **7.5x faster** because the MLX server eliminates the proxy layer and handles prompt processing more efficiently.

## Evolution — Three Generations

| Generation | Server | Speed | Claude Code E2E | Architecture |
|-----------|--------|-------|-----------------|--------------|
| Gen 1 | Ollama | 30 tok/s | ~133s | Ollama → Proxy → Claude Code |
| Gen 2 | llama.cpp TurboQuant | 41 tok/s | ~133s | llama-server → Proxy → Claude Code |
| **Gen 3** | **MLX Native** | **48 tok/s** | **17.6s** | **MLX Server → Claude Code (direct)** |

From 30 tok/s to 48 tok/s (+60%). From 133s to 17.6s per Claude Code task (+7.5x).

## What Makes MLX Native Faster

1. **No proxy** — the server speaks Anthropic Messages API directly. Zero translation overhead.
2. **Apple Silicon native** — MLX is Apple's own ML framework, optimized for M-series unified memory.
3. **Efficient prompt handling** — processes the 32K Claude Code system prompt faster than llama.cpp.
4. **4-bit KV cache quantization** — built into MLX, runs on Metal GPU natively.

## Model Details

| Property | Value |
|----------|-------|
| Model | Qwen 3.5 122B-A10B |
| Architecture | Mixture of Experts (122B total, 10B active per token) |
| Quantization | 4-bit (MLX native) |
| Size on disk | ~50 GB |
| Format | MLX safetensors |
| KV cache | 4-bit quantized |

## Cloud API Comparison

| | MLX Native (Local) | Claude Sonnet (API) | Claude Opus (API) |
|---|---|---|---|
| Speed | 48 tok/s | ~80 tok/s | ~40 tok/s |
| Claude Code task | 17.6s | ~10s | ~15s |
| Cost per M tokens | $0 | $3/$15 | $15/$75 |
| Privacy | 100% local | Cloud | Cloud |
| Works offline | Yes | No | No |
| Monthly cost | $0 | $20-100+ | $20-100+ |

The local MLX setup is now **competitive with cloud Opus on speed** and beats it on cost. Permanently.
