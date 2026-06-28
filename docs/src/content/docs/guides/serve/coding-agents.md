---
title: Use Codex and Claude Code
description: Configure coding agents to use a local Hanzo Engine server.
sidebar:
  order: 7
---

Hanzo Engine can back coding-agent clients through the compatibility APIs those
clients already speak.

| Client | API surface | Base URL to configure |
|---|---|---|
| Codex | OpenAI Responses | `http://localhost:1234/v1` |
| Claude Code | Anthropic Messages | `http://localhost:1234` |

Use `default` for a single-model server. With multi-model serving, use a model
id exactly as it appears in `GET /v1/models`.

## Start Hanzo Engine

Start a model that is tuned for tool use and code:

```bash
hanzo serve -p 1234 -m Qwen/Qwen3-Coder-Next
```

Check that the server is reachable:

```bash
curl http://localhost:1234/v1/models
```

For Codex and Claude Code, let the coding agent own normal file edits, shell
commands, and repository inspection. Enable Hanzo Engine server-side agent tools
only when you deliberately want web search or Python code execution to happen
inside the Hanzo Engine server as well.

## Codex

Codex uses the OpenAI Responses wire API for custom providers. Put provider
configuration in your user-level `~/.codex/config.toml`; Codex ignores provider
configuration in project-local `.codex/config.toml` files.

```toml
model = "default"
model_provider = "hanzo"
model_context_window = 32768
model_reasoning_summary = "none"
model_supports_reasoning_summaries = false

[model_providers.hanzo]
name = "Hanzo Engine"
base_url = "http://localhost:1234/v1"
wire_api = "responses"
request_max_retries = 1
stream_max_retries = 0
stream_idle_timeout_ms = 300000

[profiles.hanzo]
model = "default"
model_provider = "hanzo"
```

Then launch Codex with the `hanzo` profile:

```bash
codex --profile hanzo
```

If a reverse proxy enforces authentication, add `env_key = "HANZO_API_KEY"`
under `[model_providers.hanzo]` and export that variable before launching
Codex. The local Hanzo Engine server itself does not validate API keys.

Codex tool calls arrive through `/v1/responses` as Responses function tools.
Hanzo Engine routes them through the same tool-calling path used by Chat
Completions. For direct Responses examples, see the
[OpenAI Responses API guide](/hanzo/guides/serve/openai-responses-api/).

## Claude Code

Claude Code should use the Anthropic-compatible root URL, without `/v1`.
Claude Code appends `/v1/messages` and `/v1/messages/count_tokens` itself.

For a persistent local setup, add this to `~/.claude/settings.json` or to a
project-local `.claude/settings.local.json`:

```json
{
  "model": "sonnet",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:1234",
    "ANTHROPIC_API_KEY": "not-used",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "default",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "default",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "default",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "default",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Hanzo Engine default",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "Local model served by Hanzo Engine",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

Or set the same values for one shell session:

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_API_KEY=not-used
export ANTHROPIC_MODEL=sonnet
export ANTHROPIC_DEFAULT_SONNET_MODEL=default
export ANTHROPIC_DEFAULT_OPUS_MODEL=default
export ANTHROPIC_DEFAULT_HAIKU_MODEL=default
export ANTHROPIC_CUSTOM_MODEL_OPTION=default
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

Mapping the Sonnet, Opus, and Haiku defaults to `default` makes the `sonnet`
setting and Claude Code background or planning calls use the single loaded
Hanzo Engine model. If you serve several models, map each Claude Code default to
the local model id you want for that role.

Claude Code client tools arrive as Anthropic tool definitions and later
`tool_result` content blocks. Hanzo Engine translates these to its internal tool
format. For direct Anthropic examples, see the
[Anthropic Messages API guide](/hanzo/guides/serve/anthropic-messages-api/).

## Server-side agent tools

The coding clients already provide their own editing and shell tools. Server-side
Hanzo Engine tools are separate:

| Feature | How it reaches Hanzo Engine |
|---|---|
| Client-side tool use | Codex Responses function tools or Claude Code Anthropic tools |
| Server web search | `web_search_options` or Anthropic `web_search_*` server tools |
| Server code execution | `enable_code_execution` or Anthropic `code_execution_*` server tools |

Server-side web search and code execution require the corresponding Hanzo Engine
agentic runtime flags at server startup. Use them when you want the model server
to perform web search or Python execution independently of the coding client's
own terminal tools.

## Examples

Copyable config snippets live in `examples/server/`:

| File | What it shows |
|---|---|
| `codex_config.toml` | User-level Codex provider config for `/v1/responses`. |
| `claude_code_settings.json` | Claude Code settings for `/v1/messages`. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Codex returns 404 | Include `/v1` in the Codex provider `base_url`. |
| Claude Code returns 404 | Remove `/v1` from `ANTHROPIC_BASE_URL`. |
| The client requests an Anthropic model id | Use `default`, or map Claude Code default model env vars to your local ids. |
| A remote server accepts any key | Put authentication in a reverse proxy. Hanzo Engine does not validate compatibility API keys. |
| Claude Code sends beta fields your proxy rejects | Set `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` in Claude Code. |
| Long streams time out | Raise Codex `stream_idle_timeout_ms` or Claude Code `API_TIMEOUT_MS`. |

## Upstream references

- [Codex configuration reference](https://developers.openai.com/codex/config-reference)
- [Claude Code environment variables](https://code.claude.com/docs/en/env-vars)
- [Claude Code LLM gateway configuration](https://code.claude.com/docs/en/llm-gateway)
