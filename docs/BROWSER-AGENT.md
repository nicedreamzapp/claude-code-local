# 🌐 Browser Agent

A standalone agent that controls your **real Brave browser** via Chrome DevTools Protocol — powered entirely by local AI. Lives in its own repo: [`nicedreamzapp/browser-agent`](https://github.com/nicedreamzapp/browser-agent). The `Browser Agent.command` launcher here starts the MLX server, opens Brave with remote debugging, and drops you into the agent.

```
     📝 Your task
      ▼
 🤖 agent.py              ← autonomous browser agent (separate repo)
      ▼
 ⚡ MLX Server             ← local AI decides what to do
      ▼
 🌐 Brave (CDP port 9222) ← clicks, types, navigates your real browser
      ▼
 📊 Context Meter          ← color-coded memory usage after each step
```

---

[← back to the README](../README.md)
