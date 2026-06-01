---
title: Expose hanzo as an MCP server
description: Run hanzo as an MCP server other agents can call.
sidebar:
  order: 9
---

hanzo can expose itself as an MCP server.

## Starting an MCP server

```bash
hanzo serve --mcp-port 3030 -m Qwen/Qwen3-4B
```

`--mcp-port` enables the MCP server on a separate port. The endpoint is `POST /mcp` over HTTP with JSON-RPC 2.0.

The HTTP API on `--port` continues to run alongside.

## Client configuration

Clients connect to `http://<host>:<mcp_port>/mcp`.

## Authentication

The MCP endpoint has no built-in authentication. For non-localhost use, place an authenticating proxy in front.
