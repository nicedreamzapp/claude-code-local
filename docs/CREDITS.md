# 🙏 Credits

Built on the shoulders of giants:

| Project | What It Does | By |
|---------|-------------|-----|
| 🤖 [Claude Code](https://claude.ai/claude-code) | AI coding agent | Anthropic |
| 🍎 [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-examples) | Apple Silicon ML framework + inference | Apple |
| 🟢 [Gemma](https://blog.google/technology/developers/gemma-open-models/) | The 31B fighter (base weights) | Google DeepMind |
| 🟠 [Llama](https://llama.meta.com/) | The 70B fighter (base weights) | Meta |
| 🔵 [Qwen 3.5](https://qwenlm.github.io/) | The 122B fighter | Alibaba |
| 🐳 [ds4](https://github.com/antirez/ds4) | DeepSeek V4 Flash Metal engine | Antirez |
| 🔧 [huihui-ai](https://huggingface.co/huihui-ai) + [Babsie](https://huggingface.co/Babsie) | Abliterations we build on | — |
| 📖 [Abliteration explained](https://huggingface.co/blog/mlabonne/abliteration) | The technique | Maxime Labonne |

### 🧑‍🔧 Contributors

Every one of these landed on hardware I don't own, on a bug I hadn't hit. Thank you.

| Who | What they fixed |
|---|---|
| [@0xshugo](https://github.com/0xshugo) | Client disconnects handled, retries skipped when there are no tools ([#4](https://github.com/nicedreamzapp/claude-code-local/pull/4)) |
| [@asdmoment](https://github.com/asdmoment) | Gemma inference crash — auto-disable KV quantization ([#7](https://github.com/nicedreamzapp/claude-code-local/pull/7)) |
| [@kulveersingh](https://github.com/kulveersingh) | `ArraysCache` has no attribute `offset` ([#10](https://github.com/nicedreamzapp/claude-code-local/pull/10)) |
| [@tripathiprateek](https://github.com/tripathiprateek) | `uninstall.sh` — reverses `setup.sh` cleanly ([#23](https://github.com/nicedreamzapp/claude-code-local/pull/23)) |
| [@tadrianonet](https://github.com/tadrianonet) | Mac base/Pro 16 GB support: Qwen 2.5 14B, ChatML stop markers, `<tools>` parser, offline leak fix ([#32](https://github.com/nicedreamzapp/claude-code-local/pull/32)) |
| [@kevbarns](https://github.com/kevbarns) | Gemma 4 thinking suppression + slimmer tool descriptions — ~4× latency cut ([#33](https://github.com/nicedreamzapp/claude-code-local/pull/33)) |
| [@KaoCSC](https://github.com/KaoCSC) | Stop on the tokenizer's real EOS, and tolerate empty env ints ([#41](https://github.com/nicedreamzapp/claude-code-local/pull/41)) · bare JSON tool calls, which took Qwen 2.5 Coder from 0/12 to 14/14 ([#43](https://github.com/nicedreamzapp/claude-code-local/pull/43)) |

Tested on **Apple M5 Max** with **128 GB unified memory**.

Built by [Matt Macosko](https://x.com/NiceDreamzApps) in Arcata, CA — part of [Nice Dreamz LLC](https://nicedreamzwholesale.com). More open-source at [nicedreamzwholesale.com/software](https://nicedreamzwholesale.com/software/) · demos at [youtube.com/@nicedreamzapps](https://www.youtube.com/@nicedreamzapps).

<p>
  <a href="https://x.com/NiceDreamzApps"><img src="https://img.shields.io/badge/X-@NiceDreamzApps-000000?style=flat-square&logo=x&logoColor=white" alt="X"></a>
  <a href="https://www.youtube.com/@nicedreamzapps"><img src="https://img.shields.io/badge/YouTube-@nicedreamzapps-FF0000?style=flat-square&logo=youtube&logoColor=white" alt="YouTube"></a>
  <a href="https://github.com/nicedreamzapp"><img src="https://img.shields.io/badge/GitHub-@nicedreamzapp-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
</p>

---

<p align="center">
  <strong>📜 MIT License</strong> — Use it however you want.<br><br>
  💬 Builders hang out on <a href="https://discord.gg/ZdSqgAxUW">Discord</a> — share what you're building, swap MLX tips.<br><br>
  ⭐ <strong>Star this repo if it helped you!</strong> ⭐
</p>

---

[← back to the README](../README.md)
