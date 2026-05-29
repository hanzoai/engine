# Punch list — features to port from antirez/ds4 into hanzo-engine

Audit comparing `antirez/ds4` (DeepSeek V4 Flash native engine, `~/work/zen/zen5`) against `hanzo-engine` (thin shim over `mistralrs-server`, this repo). Items ordered by **value-to-difficulty ratio**, highest first. DeepSeek/DSML-specific items skipped — only carry features that generalize to mistral-rs's model zoo.

## Port progress

### Original 8-item audit

| # | Item | Status |
|---|---|---|
| 1 | SHA1-keyed rendered-prefix disk KV cache | ⬜ pending (1–2 weeks; unblocks #3 + 3 of 4 newer items) |
| 2 | Anthropic-compatible `/v1/messages` | ✅ landed in `mistralrs-server-core/src/anthropic.rs` |
| 3 | Exact tool-call replay map (tool-id → sampled bytes) | ⬜ pending (waits on #1 for KVC tail-section) |
| 4 | Streaming tool-call argument deltas | ✅ landed in `mistralrs-core/src/tools/streaming_parser.rs` + `pipeline/sampling.rs` |
| 5 | Thinking-mode streaming separation in OAI | ✅ already wired via `get_think_tag_reasoning_delta` / `get_think_tag_content_delta` |
| 6 | Greedy-during-protocol / sampled-during-payload split | ⬜ pending (sampler state machine + grammar classifier) |
| 7 | Cross-session live-checkpoint eviction-to-disk | ⬜ pending (subset of #1) |
| 8 | Single-direction activation steering hooks | ⬜ pending (research-tier; lower priority) |

### Post-launch upstream commits we've reviewed

| Commit | Item | Status |
|---|---|---|
| `037ee39` | Ignore tool calls inside `<think>` | ✅ ported — `Sequence::is_currently_in_think_block()` guard in `sampling.rs` |
| `613e9b2` | Default sampling to min-p 0.05 | ✅ ported — applied in all three handlers (chat_completion / responses / anthropic) |
| `be43477` | Standardize context-length errors | ✅ ported (pragmatic) — `classify_error_kind()` in `anthropic.rs` promotes generic `api_error` to `invalid_request_error` / `not_found_error` / `overloaded_error` based on message pattern. Engine-side classification still TBD; this is the protocol-shape side |
| `312935e` | Opt-in CORS | ✅ already covered — mistral-rs uses `tower_http::cors::CorsLayer` in `router_builder.rs`, more configurable than ds4's binary flag |
| `950e8e6` | Preserve literal tool-result text | ✅ already covered — mistral-rs's Jinja-based chat templates render tool results literally; our `flatten_message_content` for `ToolResult` does no escaping |
| `7b68234` | Prepend tool schemas to system prompt | ⬜ pending — Jinja-template rework; the cache-stability motivation lands once #1 is in |
| `f074c7b` | Anchor cold KV checkpoints at chat task boundary | ⬜ pending — needs #1 |
| `d0357ec` + `b62292c` | KV-cache hit-count decay for eviction | ⬜ pending — needs #1 |
| `5bc1e6d` | Flash graph correctness fixes | ⬜ pending — DS4-specific (compressed indexer); lands when we flesh out `models/zen5.rs` indexer path |
| `c9dd949` | CUDA compressed-prefill RoPE | ⬜ pending — DS4-specific |

### What's blocking what

```
disk-KV cache (#1)
    ├── #3 exact tool-call replay map
    ├── #7 eviction-to-disk
    ├── 7b68234 prepend tool schemas (gains cache-stability value)
    ├── f074c7b anchor cold KV at chat-task boundary
    └── d0357ec + b62292c hit-count decay
```



## 1. SHA1-keyed rendered-prefix disk KV cache

**Status:** absent. `mistralrs-core/src/prefix_cacher.rs` is in-RAM token-prefix only.

**What ds4 does:** keyed by `SHA1(rendered_text_prefix)`, stored as `<sha1>.kv` files, with a fixed 48-byte `KVC` header (magic, version, quant bits, save reason, ext flags, token count, hit count, ctx size, creation/lastused times, payload size). Saves at four moments: `cold` (after first stable prefix), `continued` (every ~10k tokens at aligned frontiers), `evict` (before unrelated session replaces live KV), `shutdown`. Cold saves trim 32 tail tokens and align down to 2048-token chunks to avoid BPE-boundary retokenization misses.

**Why it matters:** stateless agent clients (Claude Code, Codex CLI, opencode) resend the full conversation every turn. Without on-disk KV, the engine re-prefills 25k+ tokens per turn from scratch after any session switch. With it, sessions survive process restarts.

**Cost:** 1–2 weeks. File format, hash lookup, write-on-cold/continued/evict/shutdown hooks, prefix-rebuild on partial hit, tool-id map tail section.

**Where to land:** new module `mistralrs-core/src/disk_kv_cache.rs`, wired into `PrefixCacheManagerV2`. CLI flag `--kv-disk-dir DIR --kv-disk-space-mb N` on `mistralrs-server`.

## 2. Anthropic-compatible `/v1/messages` endpoint

**Status:** absent. `mistralrs-server/src/zap_server.rs:30` references `"messages" => "/v1/messages"` as a forwarding label but the actual axum router (`mistralrs-server-core/.../router_builder.rs`) has no `/v1/messages` route. Claude Code / Anthropic SDK clients cannot talk to hanzo-engine today (they fall back through Hanzo Node, which has `claude.rs` provider — but that's a different layer).

**What ds4 does:** native Anthropic Messages API surface, including `system`, `messages`, `tools`, `tool_choice`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, thinking controls. Streams thinking and text live, emits structured `tool_use` blocks when complete. Maps internally to the same chat-completion pipeline.

**Cost:** low–moderate. Translate Anthropic JSON → internal chat-completion request, emit Anthropic SSE event types. Pair with item #5 (thinking-mode streaming separation).

**Where to land:** `mistralrs-server-core/src/handlers/anthropic_messages.rs`, registered in `mistralrs_server_router_builder.rs`.

## 3. Exact tool-call replay map (tool-id → sampled bytes)

**Status:** absent. `mistralrs-core/src/tools/{mod,response}.rs` parses/re-emits JSON tool calls but does not memoize raw sampled bytes per tool-id. The prefix cache invalidates on every tool turn because canonical JSON rendering rarely round-trips byte-perfectly.

**What ds4 does:** every tool call gets an unguessable API tool ID. The server stores `tool id → exact sampled DSML block` in a bounded in-memory map (default 100k IDs, tunable via `--tool-memory-max-ids`). The map is persisted in KV cache files as a `KTM` tail section, so exact replay survives restarts.

**Why it matters:** without exact replay, every tool-call turn forces a KV cache miss on the next turn. In a long agentic loop (Claude Code, Codex CLI) this is the difference between a 100ms turn and a 30s turn.

**Cost:** moderate. Bounded HashMap + radix-trie + optional KVC tail section. Implementation is small (~500 LOC) but interaction with the canonical-rendering fallback (`--disable-exact-dsml-tool-replay`) needs care.

**Where to land:** `mistralrs-core/src/tools/exact_replay.rs`, fed by the streaming tool parser.

## 4. Streaming tool-call argument deltas

**Status:** absent. `mistralrs-server-core/src/streaming.rs` does not emit `tool_calls[].function.arguments` deltas; clients see the tool only when the call closes.

**What ds4 does:** as soon as the DSML invocation is recognized in the token stream, the tool header is sent first via SSE, then parameter bytes are forwarded as `tool_calls[].function.arguments` deltas while generation continues. The Anthropic endpoint streams thinking and text live and emits structured `tool_use` blocks when the generated tool block is complete.

**Why it matters:** latency for tool-heavy agents. Today the agent waits for the full tool call to complete before invoking. With streaming deltas, a long argument body (e.g., a file-write call with kilobytes of content) can begin invocation immediately.

**Cost:** low. Hook the existing tool parser to push deltas through the SSE writer.

**Where to land:** `mistralrs-server-core/src/streaming.rs` + `mistralrs-core/src/tools/streaming_parser.rs`.

## 5. Native thinking-mode separation in streamed output

**Status:** partial. `responses.rs` accumulates `reasoning_content` into a buffer and includes it on completion, but the chat-completion streaming path mixes reasoning into final text.

**What ds4 does:** in thinking mode, reasoning is streamed in the native API shape (Anthropic `thinking` blocks; OpenAI Responses `reasoning_content` events) instead of being mixed into final text. The Responses endpoint streams `response.output_text.delta`, function-call argument events, and terminal `response.completed` / `response.incomplete` / `response.failed`.

**Why it matters:** most modern models (Qwen3-Thinking, DeepSeek-V4-Flash, GLM-4) emit `<think>` blocks. Mixing them into final text degrades the client UX and breaks Claude-Code-style "show me your reasoning separately" panels.

**Cost:** low–moderate. Extend existing reasoning extractor to the chat streamer.

**Where to land:** `mistralrs-server-core/src/handlers/{chat_completions,responses}.rs`.

## 6. Greedy-during-protocol-syntax / sampled-during-payload split

**Status:** absent. Single global sampler regardless of where in the tool-call grammar we are.

**What ds4 does:** when the model emits stable protocol structure — DSML tags, parameter headers, JSON punctuation, closing markers — sampling is forced to `temperature=0` so the tool call stays parseable. This greedy mode does *not* apply to argument payloads: `string=true` parameter bodies and JSON string values (including file contents and edit text) use the request's normal sampling settings.

**Why it matters:** parse-reliability. Without this, a high-temperature creative request produces unparseable tool JSON. The split is also useful for other structured-output formats (function calls, MCP, OpenAPI schemas).

**Cost:** moderate. Sampler state machine driven by a small tool-grammar token classifier.

**Where to land:** `mistralrs-core/src/sampler/structured.rs`.

## 7. Cross-session live-checkpoint eviction-to-disk

**Status:** absent.

**What ds4 does:** even when full disk KV cache (#1) is disabled, writes the currently-live in-RAM prefix to disk on eviction, so the next unrelated request doesn't lose the previous session's work permanently. Cheap subset of #1.

**Why it matters:** standalone shipping path if #1 is deferred. Single user with two terminals (Claude Code + Codex CLI) currently loses one prefix each time they switch.

**Cost:** small. Single write hook in the prefix cache eviction path.

## 8. Single-direction activation steering hooks

**Status:** absent.

**What ds4 does:** `dir-steering/` subsystem implementing the technique from "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., arXiv:2406.11717). Project hidden-states orthogonally to a refusal direction at inference time, with knobs to make the model more or less compliant on dual-use prompts. Inference-time control without retraining.

**Why it matters:** useful research-tool and a substitute for fine-tuning when a quick attitude adjustment is needed. Pairs naturally with Zen5's planned abliteration pipeline (Section §4 of `zen5_whitepaper.tex`) which bakes the same idea into the weights for distribution.

**Cost:** moderate. Hook into hidden-state path of each model architecture in `mistralrs-core/src/models/`. Lower priority than 1–7 — these are research hooks, not user-facing features.

---

## Sequencing recommendation

Phase 1 (2 weeks): items **2** (`/v1/messages`), **4** (tool-call deltas), **5** (thinking-mode streaming). These are all SSE / handler work; one engineer can land all three in one milestone.

Phase 2 (2 weeks): item **1** (disk KV cache). Highest impact, biggest commit.

Phase 3 (1 week): item **3** (tool-id exact replay), which is most valuable on top of #1.

Phase 4 (1 week): items **6** (structured sampler), **7** (eviction-to-disk), **8** (steering hooks). Optional polish.

## Skipped — DeepSeek/DSML-specific or already covered

- DSML rendering and `--disable-exact-dsml-tool-replay` flag (DSML-specific, but tool-id replay (#3) is the portable subset).
- MTP speculative decoding — mistral-rs has its own speculative-decoding path.
- `--dump-logprobs` — already supported via the `logprobs` API field.
- GGUF-specific custom loaders — mistral-rs has its own loader.
- 2-bit asymmetric routed-MoE quantization (`IQ2_XXS` up/gate, `Q2_K` down) — mistral-rs uses its own quant zoo via candle.
- q2/q4 cross-quant cache reuse — solved by #1 once `(model_id, sha1)` keying is in place.
