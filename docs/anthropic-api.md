# Anthropic Messages API (native)

`hanzo-server` exposes a native, drop-in **Anthropic Messages API** surface so that
Anthropic SDK clients (Claude Code, `anthropic-sdk-*`, raw `curl`) can talk to a locally
served model without changes. Requests are translated into the internal chat pipeline (the
same path as `/v1/chat/completions`) and translated back to the Anthropic response shape.

Implementation: `hanzo-server-core/src/anthropic.rs`. Route declarations:
`hanzo-server-core/src/route_registry.rs`. Route wiring: `hanzo-server-core/src/router.rs`.

## Supported routes

| Method | Path | Handler | Purpose |
|---|---|---|---|
| `POST` | `/v1/messages` | `anthropic::messages` | Chat completion (non-streaming JSON or SSE stream) |
| `POST` | `/v1/messages/count_tokens` | `anthropic::count_tokens` | Input token count, no generation |

CORS allows the Anthropic headers `x-api-key`, `anthropic-version`, `anthropic-beta`.

## Model routing

Anthropic model ids (`claude-*`) are routed to the single loaded model (`model_id = None`,
the engine default). Any other id routes by name to a specifically-loaded model. The
`/v1/messages` response echoes the **originally requested** model id verbatim.

---

## POST /v1/messages

### Request

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | string | yes | Echoed back in the response |
| `max_tokens` | integer ≥ 1 | yes | Missing/`0` → `400 invalid_request_error` |
| `messages` | array | yes | ≥ 1 message; each `role` ∈ {`user`, `assistant`} |
| `system` | string \| block[] | no | String, or array of content blocks (text collected) |
| `stream` | bool | no | `true` → SSE (see below) |
| `temperature` | number | no | |
| `top_p` | number | no | |
| `stop_sequences` | string[] | no | Passed through as OpenAI `stop` |
| `tools` | object[] | no | Anthropic tool defs (`name`, `description`, `input_schema`) |
| `tool_choice` | object | no | Passed through |

Each message's `content` is a bare string **or** an array of content blocks:
`text`, `tool_use` (`id`, `name`, `input`), `tool_result` (`tool_use_id`, `content`).
`tool_use`/`tool_result` blocks are translated bidirectionally so full agentic
(Claude Code) tool loops work. Image blocks are accepted but only their text is counted
(see Deviations).

### Response (non-streaming)

```json
{
  "id": "msg_<id>",
  "type": "message",
  "role": "assistant",
  "model": "<echoed request model>",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 123, "output_tokens": 45}
}
```

- `content` is one `text` block, or one `tool_use` block per tool call (preceded by any
  text block). Never empty (falls back to an empty `text` block).
- `usage` counts are **real** — taken from the pipeline's prompt/completion token counts.

### stop_reason mapping

The internal `StopReason` collapses to an OpenAI `finish_reason` string, which is then
mapped to the Anthropic enum:

| internal `StopReason` | OpenAI `finish_reason` | Anthropic `stop_reason` |
|---|---|---|
| `Length` / `ModelLength` | `length` | `max_tokens` |
| `ToolCalls` | `tool_calls` | `tool_use` |
| `Eos` / `StopTok` / `StopString` | `stop` | `end_turn` |

See Deviations for why `stop_sequence` is not distinguished.

### Streaming (SSE)

`stream: true` emits the Anthropic event sequence, each as a named SSE event:

```
message_start        {message: {id, type:"message", role, content:[], model, stop_reason:null, stop_sequence:null, usage:{input_tokens, output_tokens:0}}}
ping
content_block_start  {index, content_block:{type:"text", text:""}}          # or tool_use
content_block_delta  {index, delta:{type:"text_delta", text}}               # or input_json_delta / partial_json
content_block_stop   {index}
message_delta        {delta:{stop_reason, stop_sequence:null}, usage:{output_tokens}}
message_stop
```

Text blocks stream `text_delta`; tool_use blocks stream `input_json_delta` with
incremental `partial_json`. On error, an `error` event carries
`{type:"error", error:{type, message}}` and the stream ends.

---

## POST /v1/messages/count_tokens

Counts the input tokens of the fully-rendered prompt (system + messages + tools) **without
running generation**. The request body is the same shape as `/v1/messages` **minus the
`max_tokens` requirement** — Anthropic's count_tokens endpoint does not require it (it is
accepted and ignored if present).

### Request

| Field | Type | Required |
|---|---|---|
| `model` | string | yes |
| `messages` | array | yes (≥ 1, roles `user`/`assistant`) |
| `system` | string \| block[] | no |
| `tools` | object[] | no |
| `tool_choice` | object | no |
| `max_tokens`, `stream` | — | accepted, ignored |

### Response

```json
{"input_tokens": 2095}
```

### How it counts

The handler renders the same message + tool structure `/v1/messages` sends, then issues a
`Request::Tokenize` to the engine. The engine applies the model's chat template
(`add_generation_prompt = true`, `add_special_tokens = true` — identical to the generation
path) and tokenizes. `input_tokens` therefore equals the `usage.input_tokens` a real
`/v1/messages` call would report for the same body. A loaded model is required (its
tokenizer + chat template); with no model loaded the endpoint returns `404 not_found_error`.

---

## Errors

All non-streaming error responses use the Anthropic error envelope, so the Anthropic SDKs'
typed exceptions classify them correctly:

```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}
```

| HTTP | `error.type` | When |
|---|---|---|
| 400 | `invalid_request_error` | malformed body, missing `max_tokens`, bad role, empty messages |
| 404 | `not_found_error` | no model loaded to serve/tokenize the request |
| 500 | `api_error` | internal tokenization / pipeline error |

Streaming errors are delivered as an SSE `error` event with the same inner shape.

---

## Deviations from Anthropic's API (honest list)

- **`stop_reason: "stop_sequence"` is not distinguished.** The engine's internal
  `StopReason` maps EOS, stop-token, and custom stop-string matches all to the OpenAI
  `finish_reason` `"stop"`, and the matched stop string is stripped from the output before
  the Anthropic layer sees it. There is no signal to recover `stop_sequence` at this
  boundary, so custom-stop stops report `end_turn` with `stop_sequence: null`. Fixing this
  would require threading a distinct signal through the engine response without changing the
  OpenAI route's `finish_reason`.
- **Image / non-text content blocks are counted as text only.** `count_tokens` renders the
  text of a message; image blocks contribute their placeholder text, not
  dimension-based image tokens. `/v1/messages` itself does parse image/audio/video content
  for multimodal models via the shared chat pipeline.
- **`usage` omits `cache_creation_input_tokens` / `cache_read_input_tokens`.** The engine
  has no prompt-cache accounting to report; these optional Anthropic fields are absent.
- **No server-side tool execution, batches, files, or beta features.** Only the two
  Messages routes above are implemented. Tool calls are surfaced to the client
  (`tool_use` blocks) for client-side execution, exactly like Anthropic's default tool use.

## Tests

Unit tests live in `hanzo-server-core/src/anthropic.rs` (`mod tests`). They cover the
translation, validation, and serialization layers that all of this new logic lives in:
count_tokens request parsing (with/without `max_tokens`), string- and block-content
template-map rendering, tool translation, the count_tokens response shape, `/v1/messages`
request validation (missing `max_tokens`, bad role, empty messages), model routing, and
`stop_reason` mapping — plus the pre-existing streaming/response-block tests.

End-to-end tokenization (`Request::Tokenize` → chat template → encode) requires a loaded
pipeline and is exercised by the engine's model-loading path, not re-run on CPU without a
model.
