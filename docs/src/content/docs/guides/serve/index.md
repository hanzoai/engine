---
title: Serve models
description: Run hanzo as an HTTP server, with one or more models, with the web UI, and with OpenAI-compatible APIs.
---

[Tutorial 2](/hanzo/tutorials/02-serve-an-api/) covers basic single-model serving. These guides cover the configuration needed beyond a single local server.

## Choose by task

| If you need to... | Start here |
|---|---|
| Change host, port, CORS, request limits, or authentication | [HTTP server configuration](/hanzo/guides/serve/http-server/) |
| Serve more than one model from one process | [Running multiple models](/hanzo/guides/serve/multiple-models/) |
| Use the browser chat interface | [Using the web UI](/hanzo/guides/serve/with-web-ui/) |
| Use the newer OpenAI Responses endpoint | [OpenAI Responses API](/hanzo/guides/serve/openai-responses-api/) |

For operational concerns (reverse proxy, Docker, health checks, TLS), see the [deployment guides](/hanzo/guides/deploy/).
