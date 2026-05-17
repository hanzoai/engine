# Multimodal router design — `hanzo-engine` + Zoo Desktop

Goal: one `hanzo-engine` process holds Zen5 (the 81 GB DeepSeek V4 Flash reasoning core, via `zen5-server`) plus sibling modality experts (Zen-VL, Zen-3D, Zen-Foley, Zen-Musician, Zen-Video, Jin), exposing a single OpenAI/Anthropic-compatible endpoint to Zoo Desktop. The engine routes each request to the right expert (or composition of experts) per modality.

Companion to `PUNCH_LIST_FROM_DS4.md` and `~/work/zen/zenlm/papers/zen5_whitepaper.tex` §5.

## Constraints and budget

Apple M4 Max 128 GB with `iogpu.wired_limit_mb=122880` (120 GB GPU-wired) is the reference machine. Budget at q-default quants:

| Expert | Quant | Resident size |
|---|---|---|
| Zen5 (DeepSeek V4 Flash) | `IQ2_XXS`/`Q2_K` imatrix | 81 GB |
| Zen-VL-8B | Q4_K_M | 5 GB |
| Zen-3D | F16 | 4 GB |
| Jin (diffusion-MoE, 8 experts × 220M active 2/8) | F16 | 8 GB |
| Zen-Foley | F16 | 2 GB |
| Zen-Musician | Q4_K_M | 4 GB |
| Zen-Video | F16 | 6 GB |
| **Subtotal** | | **110 GB** |
| Headroom (KV, scratch, OS) | | 10 GB |

Tight but feasible. On 96 GB machines, Zen5 + 1–2 siblings; on Mac Studio 512 GB everything plus Jin-Large.

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │             Zoo Desktop (Tauri)              │
                │  one provider entry: http://127.0.0.1:36900  │
                └──────────────────┬───────────────────────────┘
                                   │ OpenAI / Anthropic JSON
                                   ▼
                ┌──────────────────────────────────────────────┐
                │         hanzo-engine HTTP front           36900│
                │  /v1/chat/completions  /v1/messages          │
                │  /v1/images/generations  /v1/audio/speech    │
                │  /v1/models  /v1/system/info                 │
                └──────────────────┬───────────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────────────┐
                │           ModalityRouter (V1: rules)         │
                │  inspects message content-types + intent     │
                │  emits ExpertPlan = list[(expert_id, slice)] │
                └─────┬──────┬──────┬──────┬──────┬──────┬─────┘
                      │      │      │      │      │      │
                      ▼      ▼      ▼      ▼      ▼      ▼
                ┌─────────┐┌───────┐┌──────┐┌──────┐┌──────┐┌────────┐
                │  Zen5   ││ Zen-VL││Zen-3D││ Jin  ││Foley ││Zen-Mus.│
                │ (proxy) ││  ...  ││ ...  ││ ...  ││ ...  ││  ...   │
                └────┬────┘└───┬───┘└──┬───┘└──┬───┘└──┬───┘└────┬───┘
                     │         │       │       │       │         │
                     ▼         ▼       ▼       ▼       ▼         ▼
                ┌──────────┐ ┌──────────────────────────────────┐
                │zen5-server│ │   mistralrs-core ModelRegistry  │
                │  port 8000│ │ Metal device | multiple loaded  │
                │  (DS4 fmt)│ │ async per-model queues          │
                └──────────┘ └──────────────────────────────────┘
```

`zen5-server` stays a separate process because its model format (DS4-specific GGUF) and engine code (forked `ds4.c`) are incompatible with mistral-rs's loader. `hanzo-engine` proxies `Zen5` requests over loopback HTTP and treats `zen5-server` as one more "expert" registered in the `ModelRegistry`.

## ModelRegistry

New module on top of `mistralrs-core::Engine`. Three responsibilities:

```rust
pub struct ModelRegistry {
    experts: HashMap<ExpertId, RegisteredExpert>,
    vram_budget_mb: u64,
    vram_used_mb: u64,
}

pub struct RegisteredExpert {
    id: ExpertId,
    backend: ExpertBackend, // InProcess(ModelHandle) | RemoteHttp(Url) | Subprocess(PathBuf)
    modalities: ModalitySet, // bitset: TEXT | IMAGE_IN | IMAGE_OUT | AUDIO_IN | AUDIO_OUT | VIDEO | MODEL3D | TOOL
    resident_size_mb: u64,
    queue: AsyncQueue<ExpertRequest>,
    quant: QuantSpec,
}

impl ModelRegistry {
    pub async fn register(&mut self, spec: ExpertSpec) -> Result<()>;
    pub async fn dispatch(&self, plan: ExpertPlan, req: Request) -> Response;
}
```

**VRAM budgeting.** Each `register()` declares `resident_size_mb`; the registry refuses if it would exceed `vram_budget_mb`. The default budget is derived from `iogpu.wired_limit_mb` minus a 10 GB OS reserve. CLI flag `--vram-budget-mb` to override.

**Per-model async queues.** Each expert has its own request queue and graph worker. Cross-expert parallelism is bound only by GPU compute and memory bandwidth. Within an expert, requests are serialized through one graph (consistent with `mistralrs-core`'s current design — that constraint is per-model, not per-process).

**Shared KV semantics.** Disk KV cache is keyed by `(expert_id, sha1(rendered_text))` so that sibling switches on consecutive turns do not cross-corrupt cached prefixes. Depends on disk KV cache landing first (`PUNCH_LIST_FROM_DS4.md` item #1).

## ModalityRouter V1 — rule-based

Deterministic. Inspects three signals:

1. **Content-type discriminator.** The incoming OpenAI/Anthropic JSON has `content` arrays with typed parts. Presence of `{ "type": "image_url" }` → vision expert candidate. `{ "type": "audio" }` → audio expert candidate. `{ "type": "video" }` → video. `{ "type": "model3d" }` (custom extension) → 3D.
2. **Intent prefix regex.** First user turn matched against a small lookup table:
   - `^Render (an? )?image\b` → Jin / Zen-Artist
   - `^Generate (an? )?audio\b|^Compose music\b` → Zen-Musician / Zen-Foley
   - `^Create (an? )?(video|animation)\b` → Zen-Video
   - `^Build (an? )?3D\b|\.glb|\.obj|\.usd` → Zen-3D
3. **Tool list inspection.** Tool names like `render_image`, `transcribe_audio`, `synthesize_speech` route by tool, with Zen5 acting as the orchestrating LLM that interleaves siblings into a single response.

Routing result is an `ExpertPlan`:

```rust
pub struct ExpertPlan {
    primary: ExpertId,        // who composes the final response
    siblings: Vec<SiblingCall>, // who else is invoked, and with what slice
}

pub struct SiblingCall {
    expert: ExpertId,
    slice: ContentSlice,      // which parts of the request go here
    inject_back_as: InjectKind, // tool_result | content_part | system_context
}
```

Zen5 is the default `primary` unless the request is "pure generation" (e.g. image-only render with no surrounding chat). For mixed requests, siblings run first; their results are injected into Zen5's context as tool results, and Zen5 composes the final reply.

## ModalityRouter V2 — learned

Two-tower classifier (~50M parameters) trained on V1 traces. Inputs: full conversation prefix + attachment metadata; outputs: multi-label sibling assignment with confidence. The hanzoai/zen `RoutingPolicy` framework is the architectural template — we add the modality dimension to it.

Training data: V1 routing decisions logged with the model's final composed response. A human reviewer flags incorrect routes; the corrected labels become V2's supervised set. Initial training corpus target: 100k decisions.

Replacement happens via feature flag. V1 always remains available as fallback when V2 confidence < threshold.

## Zoo Desktop integration

Zoo Desktop already supports adding an LLM provider with a custom `external_url`. Configuration:

```json
{
  "providers": [{
    "id": "zen5-multimodal",
    "name": "Zen5 (multimodal)",
    "external_url": "http://127.0.0.1:36900/v1",
    "api_key": "zen-local",
    "models": [{
      "id": "zen5",
      "name": "Zen5",
      "capabilities": ["text", "vision", "audio_in", "audio_out", "video", "model3d", "tools"],
      "context": 100000,
      "max_output": 32000
    }]
  }]
}
```

From the user's perspective there is one Zen model that handles all modalities. The desktop client doesn't need to know about the sibling roster — that's an engine-internal optimization. Capabilities advertised on `/v1/models` are the *union* of capabilities across loaded siblings, so the UI can show appropriate input controls (image upload, audio record, 3D file upload).

## Engine startup

```sh
hanzo-engine \
  --port 36900 \
  --vram-budget-mb 110000 \
  --register zen5:proxy:http://127.0.0.1:8000/v1 \
  --register zen-vl:gguf:~/work/zen/models/zen-vl-8b/model.gguf \
  --register zen-3d:gguf:~/work/zen/models/zen-3d/model.gguf \
  --register jin:diffusion:~/work/zen/models/jin/jin-small.safetensors \
  --register zen-foley:gguf:~/work/zen/models/zen-foley/model.gguf \
  --register zen-musician:gguf:~/work/zen/models/zen-musician/model.gguf \
  --register zen-video:diffusion:~/work/zen/models/zen-video/model.safetensors \
  --kv-disk-dir /tmp/hanzo-engine-kv --kv-disk-space-mb 16384 \
  --router rules
```

`zen5-server` must be running on port 8000 separately (process supervisor: `launchd` plist on macOS, `systemd` on Linux). A wrapper script `hanzo-engine-up.sh` brings up both processes in the right order.

## Milestones

**M1 — single-model parity (1 week).** `hanzo-engine` loads one model via `--register`, no router, no registry. Validate equivalence to `mistralrs-server`.

**M2 — ModelRegistry + multi-model loading (2 weeks).** Two models loaded simultaneously, per-model queues, VRAM budgeting. No router yet; client picks model via `model` field. Disk KV cache (`PUNCH_LIST_FROM_DS4.md` #1) must land here.

**M3 — `/v1/messages` + thinking-mode streaming (1 week).** `PUNCH_LIST_FROM_DS4.md` items #2 and #5. Pre-requisite for Claude Code as a Zoo Desktop alternative.

**M4 — ModalityRouter V1 (1 week).** Rule-based router. Smoke test: text+image request routes to (Zen-VL → Zen5 compose); image-render request routes to Jin directly.

**M5 — Zen5 proxy + zen5-server supervisor (3 days).** `--register zen5:proxy:http://...` backend. Health-checks. Wrapper script.

**M6 — Full sibling roster (2 weeks).** Register Zen-3D, Zen-Foley, Zen-Musician, Zen-Video, Jin. Validate VRAM budget on M4 Max 128 GB. End-to-end smoke test from Zoo Desktop UI.

**M7 — ModalityRouter V2 learning loop (4 weeks).** Trace collection, classifier training, A/B harness for V1 vs V2.

Total to a usable end-to-end on M4 Max: **~6 weeks** with one engineer focused, assuming disk KV cache is the parallel critical path.

## Open questions

1. **Diffusion model integration.** mistral-rs is LLM-first; Jin (diffusion transformer) is a different inference loop. Options: (a) carry diffusion as a separate runtime that `ModelRegistry` invokes via FFI; (b) treat Jin as a remote sibling and run it in its own subprocess. (b) is simpler for M1; we can collapse to (a) later.
2. **Audio in vs audio out.** `mistralrs-audio` exists in the workspace and presumably handles audio-in; audio-out (Zen-Foley, Zen-Musician) is generative and needs its own runtime. May share the diffusion runtime answer from (1).
3. **Streaming back through the router.** When Zen5 is the primary and a sibling is invoked, do we stream the sibling's intermediate output through the client's SSE channel, or wait for the sibling to finish and only stream Zen5's composed reply? For tool calls this is already solved (tool-results inject as messages, no streaming). For inline content (e.g., Zen5 quotes a Zen-VL image description as it generates) we need a content-part SSE protocol. Defer to M7.

## Status / next concrete action

- [x] Punch list captured (`PUNCH_LIST_FROM_DS4.md`).
- [x] Design captured (this file).
- [ ] M1 implementation start.
- [ ] Decide on diffusion-runtime strategy (open question 1).
- [ ] Cross-check VRAM budget table against actual on-disk file sizes for each sibling.
