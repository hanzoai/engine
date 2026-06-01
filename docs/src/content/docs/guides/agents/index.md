---
title: Build agents
description: Tool calling, code execution, web search, MCP, generated media, and persistent sessions. The pieces that turn a chat model into something that takes action.
---

hanzo can run the tool loop for you, expose standard OpenAI tool calls to your client, or act as the local runtime behind an agent app. Unlike a plain OpenAI-compatible model server, hanzo can execute tools locally and stream model text, tool progress, files, media, and session state from the same request. [Tutorial 5](/hanzo/tutorials/05-build-an-agent/) is the end-to-end walkthrough.

The agent system has three layers:

- **Tool protocol:** how a model requests external work.
- **Tool execution:** built-in code execution, web search, MCP tools, callbacks, or external dispatch.
- **App runtime:** streaming model output, tool progress, files, generated media, and sessions as one local surface.

## Choose a path

**Learn the basics**

- [Tool calling basics](/hanzo/guides/agents/tool-calling-basics/): OpenAI tool calls and the hanzo server-side loop.
- [Strict tool calling](/hanzo/guides/agents/strict-tool-calling/): constrain tool arguments to the declared JSON Schema.

**Use built-in tools**

- [Enable code execution](/hanzo/guides/agents/enable-code-execution/): Python execution, file outputs, media capture, and [sandbox isolation](/hanzo/reference/sandbox/).
- [Web search](/hanzo/guides/agents/web-search/): built-in search, extraction, and embedding reranking.

**Build an app runtime**

- [Agentic runtime for apps](/hanzo/guides/agents/agentic-runtime/): streaming model output, tool progress, files, generated media, and sessions.
- [Persist agent sessions](/hanzo/guides/agents/persist-sessions/): state across HTTP requests.

**Bring or expose tools**

- [Connect to an MCP server](/hanzo/guides/agents/connect-mcp-server/): use external MCP tools.
- [Configure the tool loop](/hanzo/guides/agents/configure-tool-loop/): round caps and dispatch URLs.
- [Expose hanzo as an MCP server](/hanzo/guides/agents/expose-as-mcp/): let other agents call your models.

## Recommended order

1. [Tool calling basics](/hanzo/guides/agents/tool-calling-basics/)
2. [Strict tool calling](/hanzo/guides/agents/strict-tool-calling/)
3. [Enable code execution](/hanzo/guides/agents/enable-code-execution/)
4. [Web search](/hanzo/guides/agents/web-search/)
5. [Agentic runtime for apps](/hanzo/guides/agents/agentic-runtime/)
6. [Persist agent sessions](/hanzo/guides/agents/persist-sessions/)
7. [Connect to an MCP server](/hanzo/guides/agents/connect-mcp-server/)
8. [Configure the tool loop](/hanzo/guides/agents/configure-tool-loop/)
9. [Expose hanzo as an MCP server](/hanzo/guides/agents/expose-as-mcp/)

For design rationale (server-side tool loops, session splicing), see the [explanation section](/hanzo/explanation/agentic-loop/).
