# 🔌 MCP Servers — 100% Local

> **The only way to run Claude Code's full MCP plugin ecosystem 100% local on Apple Silicon.**

Most local-LLM proxies break MCP — they strip tool definitions, mangle `tool_use` blocks, or refuse to forward the streaming format Claude Code expects. This server passes tool definitions through to your local model and translates the responses back into Anthropic's format, across all three model families. From Claude Code's perspective it's talking to Anthropic. From your MCP server's perspective, nothing changed.

Wire servers up the normal Claude Code way:

```bash
# Filesystem — let the local model read/write a folder
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem ~/projects

# GitHub — issues, PRs, code search
claude mcp add github --env GITHUB_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github

# Web search — for when the local model needs fresh info
claude mcp add brave-search --env BRAVE_API_KEY=$BRAVE_API_KEY -- npx -y @modelcontextprotocol/server-brave-search
```

The whole 200+ server MCP universe works the same against your local Gemma or Qwen — just running on your machine instead of someone else's.

---

[← back to the README](../README.md)
