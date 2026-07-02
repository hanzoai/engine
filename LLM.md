# Hanzo Engine - LLM Inference Integration

This file provides guidance to AI assistants working with the Hanzo Engine codebase.

## Project Overview

**Hanzo Engine** is Hanzo AI's high-performance LLM inference engine written in Rust.

### Integration Status

- **Last Sync Date**: 2026-05-06 — merged upstream hanzo `2d4ba4f16`
- **Remote**: Configured as `upstream` (hanzoai/engine) in git
- **Workspace version**: 0.8.1 (synced with upstream)

### Hanzo-Specific Components

`hanzo-engine/` is now both a **library** and a **binary**:

1. **Library — `hanzo_engine::*`** (the canonical inference API for the Hanzo stack):
   - `InferenceEngine` trait: `fn infer(&self, model_id: &[u8;32], prompt: &[u8]) -> Result<Vec<u8>, EngineError>`
   - `EmbeddingEngine` trait: `fn embed(&self, dim: usize, text: &[u8]) -> Result<Vec<f32>, EngineError>`
   - Process-wide registry (`OnceLock`-backed): `register_inference_engine`, `register_embedding_engine`, `infer`, `embed`
   - `MistralEngine` — real implementation backed by `hanzo-engine` (handles HF repos and local paths; derives `model_id` as SHA-256 of source)
   - Consumers: hanzo-vm precompiles `0x0201` (AI inference) and `0x0202` (AI embedding) — they call `hanzo_engine::infer` / `embed` synchronously through the registry
   - **NOT** the routing/pricing crate. The Hamiltonian-Hidden-Markov MarketMaker lives in `hanzo-hmm` (`~/work/hanzo/net/hanzo-hmm`) and prices heterogeneous compute; the EVM precompiles depend on `hanzo-engine`, not on `hanzo-hmm`.

2. **Binary — `hanzo-engine`** (thin CLI wrapper):
   - Shells out to `hanzo-server` for the full HTTP server experience
   - Use the library for programmatic / in-process integration

### Architecture

Hanzo Engine is a Rust workspace containing:
- All upstream hanzo workspace members (hanzo-engine, hanzo-server, hanzo, hanzo-llm-mcp, …)
- **hanzo-engine/** — lib + bin: canonical Hanzo inference + embedding API
- Local hanzo-ml fork at `../ml/hanzo-{ml,nn,flash-attn,metal-kernels}` overrides upstream's `hanzo-ml-*` crates via `[workspace.dependencies]` path overrides

The engine provides comprehensive LLM inference with support for text, multimodal (incl. video), image generation, speech, and embeddings through multiple APIs (Rust, Python, OpenAI HTTP, MCP).

## Native ML Pipeline

Three sibling crates live alongside `hanzo-engine/` and share the workspace's pinned candle (`hanzo-ml 0.9.2-alpha.2`, `half 2.7.1`) — single source of truth:

- **`hanzo-federation/`** — Vendor-neutral federated-training transport, scheduler, and coordinator. Pure-Rust port of `zen/gym/src/gym/distributed` with byte-identical wire format (canonical BF16 delta blob + HMAC-SHA256 auth). Public surface: `Coordinator::new(lab).serve(addr)`, `Worker::new(cfg).run(step, params, apply, data)`, `TransportClient`. Built on the engine workspace's axum 0.8 / reqwest 0.13 / tower-http 0.6.
- **`hanzo-quant/`** — Pure-Rust quantization: `BitDelta` (1-bit signs + per-tensor scale, ~32x), `DeltaQuant` (INT2/4/8 grouped symmetric), `DeltaSoup` (Byzantine-robust trim-mean aggregation). Uses workspace `candle-core` (the `hanzo-ml` fork) so no version split.
- **`hanzo-zen5/`** — Zen5 inference adapter. Default `ffi` feature wraps the vendored `zen5-engine` C runtime (Metal / CUDA / CPU); `native` feature swaps in a candle-rs DeepSeek V4 Flash scaffold using workspace `candle-core` / `candle-nn` plus the local `hanzo-transformers` fork.

Composition with `hanzo-engine`: `hanzo-federation` ships the cross-host training fabric; `hanzo-quant` produces the delta blobs that travel on it; `hanzo-zen5` plugs zen5 model variants into the `InferenceEngine` trait registry alongside `MistralEngine`. hanzod (in `~/work/hanzo/node`) pulls all four via workspace paths (`hanzo-federation = { path = "../engine/hanzo-federation" }`, etc.).

## Essential Commands

### Building Hanzo Engine

```bash
# Check compilation (recommended first step)
cargo check --package hanzo-engine --no-default-features --features metal

# Build for macOS (Metal backend)
cargo build --package hanzo-engine --release --no-default-features --features metal

# Build for Linux (CUDA backend)
cargo build --package hanzo-engine --release --features cuda

# Install hanzo-engine binary
cargo install --path hanzo-engine --no-default-features --features metal
```

### Building Core Components

```bash
# Basic release build
cargo build --release

# With CUDA support (Linux)
cargo build --release --features "cuda flash-attn cudnn"

# With Metal support (macOS)
cargo build --release --features metal

# Install hanzo-server binary
cargo install --path hanzo-server --features <features>
```

### Testing & Quality
```bash
# Run core tests
cargo test -p hanzo-engine -p hanzo-quant -p hanzo-vision

# Format code (uses rustfmt, ruff, clang-format)
make fmt

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace --tests --examples -- -D warnings
```

### Running Models
```bash
# Run interactive mode with plain model
cargo run --release --features <features> -- -i plain -m <model_id> -a <arch>

# Run with GGUF quantized model
cargo run --release --features <features> -- -i gguf -f <file> -t <tokenizer>

# Run server
cargo run --release --features <features> -- --port 1234 <model_args>
```

## Models

When integrating a new model, make sure it respects all of the varbuilder `.pp` calls. In Hanzo, a VarBuilder maintains an internal path vector that acts like a “current working directory” for model weights; every call to pp("sub") (alias for push_prefix) clones the builder and appends sub, so successive calls accumulate a dotted prefix such as transformer.h.0 while leaving the original builder untouched . When you eventually call get(...), Hanzo joins that prefix with the tensor name (prefix + "." + name) and looks it up in the checkpoint backend, producing keys that exactly match the dot-separated names emitted by PyTorch’s state_dict/named_parameters, which means PyTorch-trained weights can be loaded without any renaming  ￼. This lets you recreate the PyTorch module tree in Rust by “walking” it: e.g. vb.pp("word_embeddings") grabs word_embeddings.*, while a chain like vb.pp("encoder").pp("layers").pp(i.to_string()) targets keys such as encoder.layers.0.*, exactly as shown in community tutorials porting Transformers models to Hanzo  ￼. As one maintainer put it, the prefix system lets you “cd” around the parameter hierarchy, giving a lightweight namespace mechanism that keeps Hanzo fully compatible with PyTorch naming conventions while remaining ergonomic to use.

You should also look for a model.safetensors.index.json file for the model at hand to verify correct structure.

## Architecture Overview

### Workspace Structure

#### Hanzo-Specific
- **`hanzo-engine/`** - Hanzo's custom inference server and CLI
  - Custom CLI with model management commands
  - OpenAI-compatible HTTP server (port 36900)
  - Ollama compatibility layer
  - Status: Compiles successfully with Metal backend (macOS)

#### Core Components
- `hanzo-engine/` - Core inference engine, model implementations, pipelines
- `hanzo-server/` - CLI binary entry point
- `hanzo-server-core/` - HTTP server routing, OpenAI API implementation
- `hanzo-pyo3/` - Python bindings (PyO3)
- `hanzo/` - High-level Rust API
- `hanzo-vision/` - Vision model support
- `hanzo-quant/` - Quantization implementations (ISQ, GGUF, GPTQ, etc.)
- `hanzo-paged-attn/` - PagedAttention implementation
- `hanzo-audio/` - Audio processing
- `hanzo-llm-mcp/` - Model Context Protocol client
- `hanzo-bench/` - Benchmarking tools

### Key Design Patterns

1. **Pipeline Architecture**: All models implement the `Pipeline` trait in `hanzo-engine/src/pipeline/mod.rs`. Different model types (Plain, GGUF, GGML, Vision) have their own pipeline implementations.

2. **Model Loading**: Models are loaded through `Loader` traits that handle different formats and quantizations. See `hanzo-engine/src/loader.rs`.

3. **Request Handling**: The server uses message passing with `Hanzo` struct managing a background thread pool. Requests flow through `hanzo-engine/src/engine/mod.rs`.

4. **Device Management**: Automatic and manual device mapping for multi-GPU setups handled in `hanzo-engine/src/device_map.rs`.

5. **Decode Graphs (CUDA/ROCm)**: Steady-state single-token decode replays a captured GPU graph instead of relaunching hundreds of kernels/token (`pipeline/cuda_graph.rs`, `pipeline/rocm_graph.rs`). Wired for safetensors in `NormalPipeline` and for GGUF in `GGUFPipeline` (`pipeline/gguf.rs`). Toggle with `CUDA_GRAPHS=0` / `ROCM_GRAPHS=0`. ELIGIBILITY INVARIANT (`model_supports_decode_graph`): only variants whose decode RoPE reads the device `metadata.rope_positions` buffer (Qwen3, Qwen3MoE) are graph-safe. Host-offset RoPE (`RotaryEmbedding::forward` -> `selected_rope_cache` -> `cos.narrow`, used by Llama/Phi2/Qwen2) and per-forward mRoPE built via `Tensor::from_vec` (Qwen35 `compute_text_mrope`) FREEZE the position at capture and emit garbage on replay. GB10 decode wins: 0.6B +32%, 8B +7%, 30B-A3B MoE +25%, all byte-identical to eager.

### Adding New Features

When adding new model architectures:
1. Implement the model in `hanzo-engine/src/models/`
2. Add pipeline support in `hanzo-engine/src/pipeline/`
3. Update model detection in `hanzo-engine/src/pipeline/normal.rs`
4. Add architecture enum variant in `hanzo-engine/src/lib.rs`
5. Update CLI args in `hanzo-server/src/main.rs`

When adding new quantization methods:
1. Implement in `hanzo-quant/src/`
2. Add to quantization loading logic in pipelines
3. Update documentation in `docs/QUANTIZATION.md`

### Facial Animation (MuseTalk)

`hanzo serve --arch musetalk -m <bundle>` loads a MuseTalk lip-sync/avatar model and
serves `/v1/animate` (alias `/v1/video/lipsync`): POST `{model, visual, audio, fps?}`,
audio + visual in, dubbed mp4 out. A still image -> Portrait, a video -> Footage
(lip-sync); the pipeline picks the animator via `accepts()` and the handler muxes the
rendered frames + driving audio with ffmpeg.

`-m` is a self-contained MuseTalk bundle (a local dir or HF repo). All six sources
resolve as fixed safetensors sub-paths of the single `model_id`:

    musetalkV15/musetalk.json
    musetalkV15/unet.safetensors
    sd-vae-ft-mse/config.json
    sd-vae-ft-mse/diffusion_pytorch_model.safetensors
    whisper/tiny.safetensors
    s3fd.safetensors

Selection path: CLI `--arch musetalk` -> `ModelType::Animation` -> `ModelSelected::Animation`
-> `AnimationLoader` (`AnimationLoaderType::MuseTalk`) in `pipeline/animation.rs`. Runs on
CPU today (`Device::Cpu`); GPU is a separate effort. A `.pth`/`.pt` pickle still loads
(`load_vb`) but safetensors is canonical.

### Important Files to Know

- `hanzo-engine/src/engine/mod.rs` - Main engine orchestration
- `hanzo-engine/src/pipeline/mod.rs` - Pipeline trait and common logic
- `hanzo-server-core/src/routes.rs` - HTTP API endpoints
- `hanzo-pyo3/src/lib.rs` - Python API entry point
- `hanzo/examples/` - Usage examples for Rust API

### Testing Approach

You should *always* run `cargo check`/`cargo c` before returning to make sure code compiles. If code does not compile, only make edits.

Avoid returning TODOs.

- Unit tests are colocated with source files
- Integration tests in `tests/` directories
- Use `cargo test -p <crate>` to test specific components
- Python tests require building and installing the package first

### Common Pitfalls

1. **Feature Flags**: Many features are gated behind Cargo features. Always check what features are needed for your use case.
2. **Device Indices**: CUDA device selection uses 0-based indexing
3. **Chat Templates**: Models may need specific chat templates - check `chat_templates/` directory
4. **Quantization**: Different quantization methods have different hardware requirements

## Latest Upstream Features (as of commit 530463af1)

- **Qwen 3 VL** - Vision-language model support (#1657)
- **Paged Attention Refactor** - Simplified paged attention modules (#1654)
- **Audio Processing** - normalize, apply_fade, remove_dc_offset functions (#1572)
- **Gemma 3N** - Support for cases where q != (k=v) devices (#1653)
- **No Busyloop Refactor** - Improved engine efficiency (#1655)

See `docs/` directory for detailed documentation on specific models and features.

## Known Issues & Work in Progress

### Embeddings Implementation
- **Status**: Temporarily disabled (backed up to `embeddings.rs.bak`)
- **Issue**: The `embedding` module in `hanzo_engine` is private and not accessible through public API
- **TODO**: Research proper way to implement embeddings using public API
- **Previous attempt**: Used internal `BertEmbeddingModel` and `BertPipeline` which are not publicly exposed

### Dependencies
Current `hanzo-engine/Cargo.toml` needs these dependencies for embeddings:
- `hanzo-ml` (from workspace)
- `tokenizers` (from workspace)
- May need to re-export or use different approach

## Syncing with Upstream

To pull latest changes from upstream:

```bash
# Fetch upstream changes
git fetch upstream

# View what's new
git log HEAD..upstream/master --oneline

# Merge upstream changes (creates merge commit)
git merge upstream/master

# Or rebase Hanzo changes on top of upstream
git rebase upstream/master

# After resolving conflicts, test build
cargo check --package hanzo-engine --no-default-features --features metal
```

## enso -- the learned router policy (brain) for hanzo-router (mechanism)

`hanzo-router` is the routing MECHANISM (registry, SLO gate, placement, dispatch).
It exposes one seam: the `RoutePolicy` trait --
`route(&Request, &User, &Slo, &Registry) -> Route { model, level, modality, confidence }`.
The rule-based `Policy` implements it as the cold-start fallback (placement-agnostic,
fixed low confidence). New vocabulary lives in `registry` (`Modality`, `Level`) and
`route` (`Route`, `User`, `Slo`, `REFUSED_MODEL`). Pre-existing API unchanged.

`enso` (separate crate, `enso/`) is the learned POLICY implementing `RoutePolicy`.
Six orthogonal pieces, one concern each:
- `featurize`: Request -> feature vector x (hashing + metadata, sub-us; a tiny
  finetunable encoder swaps in at the same `Featurizer` trait).
- `profile`: eval rows p per (model, level, modality); `ingest`/`parse_jsonl` fold
  bench tuples into the table. REAL eval data plugs in HERE (JSONL of `EvalSample`);
  absent it, `synth` emits clearly-labeled synthetic tuples + an `oracle`.
- `policy`: the one learnable object -- bilinear utility `x^T W p`.
- `guard`: two-tier safety -- hot-path keyword classifier + escalate to a `Teacher`
  seam (Qwen3Guard; `DistilledTeacher` stands in until weights are wired).
- `selector`: safety-gated, SLO-feasible `argmax[utility - lambda*cost - mu*latency]`.
- `learner`: offline ridge fit of base W + online per-user LinUCB. `theta_u` is
  prior-centered at W; `dW_u = theta_u - W` is the per-user delta. Serving is greedy
  on `theta_u` (sub-ms); UCB exploration runs off the hot path during learning.

Registry is cross-modal: LLMs, dub generators (musetalk/echomimic), image models are
all profile rows -- "add a model = add a row".

Proof: `cargo test -p enso --test proof -- --nocapture` (add `--release` for the
latency number). It asserts correct (model, level) picks, per-user bandit divergence
after feedback (alice -> zen-eco, bob -> zen-ultra on the same request), guard
block/escalate, sub-ms routing (p99 ~1.5us), and 100% (model, level) accuracy vs the
oracle on a held-out synthetic split (rule baseline 0% on (model, level): no level/cost
awareness). 100% reflects well-separated synthetic profiles -- real data will be lower.

Honest scope: the LinUCB per-user bandit is solid and realizable. Per-user real-time
LoRA over a neural encoder and self-adaptive expert vectors are research-frontier and
attach at the same `policy`/`learner` seam -- flagged, not faked.

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.) - they're in .gitignore
3. **NEVER** create random summary files - update THIS file
4. **ALWAYS** check compilation with `cargo check` before considering integration complete

---

# DeepSeek-V4 + SOTA Inference Playbook (2026-06-28)

Cross-verified research (llama.cpp source, antirez/ds4, vLLM/SGLang, V4 config.json/arXiv) +
this session's GB10 (sm_121, 273 GB/s LPDDR5X, 128 GB unified) measurements. **The binding
constraint for batch-1 decode is MEMORY BANDWIDTH (~11 GB read/token); every win = fewer
bytes/token or higher BW utilization. Compute-side FP4 tensor-core gains are IRRELEVANT at
batch-1 (and broken on sm_121).**

## V4 architecture — the corrected mental model
- **NOT MLA.** V4 replaced MLA with **hybrid sparse attention (DSA)**, per-layer mix of:
  **SWA-128** (raw sliding window) · **CSA** (4×-compressed KV, lightning-indexer top-k 512/1024)
  · **HCA** (128×-compressed KV, dense) · learned **per-head attention sinks**. The `head_dim:512
  / kv_heads:1` is a shared-K=V **MQA** head with partial RoPE-64 — compressed-KV MQA + sparse
  selection, *not* MLA up-projection decompression.
- Weights ship **FP8 (E4M3, 128×128 block, UE8M0 µscale) dense + native FP4 MoE experts**, bf16
  master. The model is **W8A8 QAT-trained** — it EXPECTS low-precision activations.
- MoE: 256(Flash)/384(Pro) routed, 6 active + 1 shared; gate = **sqrtsoftplus + noaux_tc** bias;
  clamped SwiGLU (limit 10); **3 Hash-MoE bootstrap layers** (`tid2eid`). Residual = **mHC**
  (4-stream, 20-iter Sinkhorn doubly-stochastic mixing).

## Quant recipe (canonical = what we run; antirez GGUF, imatrix-built)
`IQ2_XXS`(ffn_gate/up_exps) · `Q2_K`(ffn_down_exps) · `Q8_0`(shared experts, all attn proj,
output) · F16(token_embd, router) · F32(norms, sinks) · I32(hash). q2≈81 GiB. **imatrix REQUIRED
for IQ2_XXS** (from real routed-expert activations). **Q3=garbage, Q4 decode unstable** (argmax
collapse from 43-layer error accumulation — ds4 fix: hi-precision matmul, `Q4_DECODE`+`Q8_NO_Q4`).

## The decode-degeneration finding (this session) — WHY hard/long prompts collapse
V4-at-IQ2 sits at a **repetition bifurcation**: cumulative quant error across 43 layers tips it.
ds4 hit the SAME issue (its README documents it) and fixes it with **hi-precision matmul
accumulation**, not higher activation precision. KEY MEASUREMENT: **F32 is WORSE than bf16** here
— because the model is FP8-trained, so MORE precision moves AWAY from the trained operating point.
**bit-exact = match ds4's Q8_0/IQ2 *accumulation* kernels (int32 dp4a / hi-precision), NOT go f32.**
Current best config: **bf16 carrier + native dp4a** (HANZO_CUDA_FAST_MMQ / V4 `set_fast_mmq(true)`)
→ correct on short/medium (primary-colors, France, math, natural EOS); haiku-class long-hard
reasoning still degenerates (the accumulation-precision residual).

## Improvement playbook (ranked by batch-1-decode leverage on GB10)

### TIER 1 — proven, highest leverage
1. **Per-layer CUDA-graph capture + split-K F16/dp4a decode GEMV.** ds4's exact win: 73→94% of
   roofline → ~19-25 t/s (we're at 3.5). Per-LAYER (resolve MoE routing host-side / fixed expert
   capacity) — llama.cpp can't (mul_mat_id disables whole-graph capture) → our edge. `cudarc`
   graph capture. **This is the #1 perf item.**
2. **dp4a int8 (NOT tensor-core INT8/FP4) for decode GEMV.** At N=1, WMMA wastes 15/16 slots
   (0.25×). FP4 tensor cores only help PREFILL (build sm_121a + CUDA 13 compute_121f). Decode =
   dp4a / F16 split-K.
3. **Bandwidth-optimal IQ2_XXS dequant→GEMV** (wide 128/256-bit loads, in-register dequant,
   `L1::no_allocate` on streamed expert weights, `evict_last` on reused activation, maxrregcount
   32-45). The routed IQ2 experts dominate the 11 GB/token. Port dequant from **github.com/nktkt/ds4**
   (Rust, bit-exact-tested).
4. **Hi-precision accumulation in ALL quant matmuls** (the haiku fix) — match ds4's
   `matmul_q8_0_tensor` hi-precision path / `ds4_gpu_set_quality(true)`. int32 dp4a accumulate +
   verify IQ2 mmvq accumulates hi-precision. **This is bit-exact correctness for hard prompts.**

### TIER 2 — correctness + structural
5. **Real V4 sparse attention (DSA: CSA top-k lightning-indexer + HCA + SWA-128 + sinks, FP8 KV)**
   — refs: llama.cpp **#24162** (`llama-kv-cache-dsv4`), **#23346** (`llama_kv_cache_dsa`, 2 caches:
   latents + indexer keys). The compressor is wired (this session); the **indexer top-k selection**
   is the missing piece for >128 context (1M ctx ≈ 9.62 GiB/seq).
6. **Fuse MoE hot path** (TopK gate+route, gate/up SwiGLU sharing activation load, RMSNorm+mul,
   compressor+RoPE+cache-insert). llama.cpp +42.5% (#16130/#16715/#14800); vLLM 1.4-20× on V4 ops.
7. **Unified-memory zero-copy** (mmap GGUF GPU-addressable via Grace ATS; `cudaMemAdvise`
   ReadMostly/AccessedBy on resident, PreferredLocation(CPU)+AccessedBy(GPU) on cold experts;
   prefetch predicted experts). Enables SSD/CPU expert streaming.

### TIER 3 — conditional
8. **EAGLE-3 speculative (> native MTP)** — accept ~2.5 vs MTP ~1.9. **MANDATORY: batched verify
   (stream weights ONCE for all K draft tokens, fixed-shape CUDA graph)** else net-negative
   single-stream (ds4 proved depth-1 MTP = −21%). The MTP head (3.6 GiB gguf) is being loaded;
   wire it EAGLE-style, not depth-1.
9. **DeepSeek-OCR** (`deepseek2ocr` in llama.cpp, llama-model.cpp:181) — a separate vision+V2 port
   (SAM/CLIP encoder + DeepEncoder + V2-MoE decoder). Scoped follow-on; reference llama.cpp's impl.

### DON'T (negative results, save the effort)
FP4/INT8 tensor cores for decode (batch-1 irrelevant, sm_121-broken) · whole-model CUDA graphs
(MoE breaks them → per-layer) · native depth-1 MTP single-stream · DeepGEMM/FlashMLA drop-in
(no sm_121 FP4 path mid-2026) · going F32 for "precision" (worse — model is FP8-trained).

## Key references
ds4: github.com/antirez/ds4 + **nktkt/ds4** (Rust dequant) · llama.cpp PRs #12801(MLA)
#19057/#22286(FA head-dim 512/576) #23346(V3.2/DSA) #24162(V4,open) #22673(MTP) #18039(EAGLE-3)
#16130/#16715/#14800(fusion) · vLLM blog 2026-04-24 + LMSYS 2026-04-25(V4 recipe) · sm_121:
build sm_121a + CUDA 13 (issue #19662).
## CUDA flash prefill: the missing-`.contiguous()` bug -- FOUND + FIXED (573d12614)
- SYMPTOM: CUDA GGUF prefill was BOTH slow (eager O(n^2) collapse: pp128 2158 -> pp512 2044 -> pp2048
  1118, vs llama FLAT ~2579/3033/3008) AND, when routed to flash, produced GARBAGE logits (byte-identical
  `钊，，，` from token 1 -> corrupted KV cache -> whole generation garbled).
- ROOT CAUSE (one bug, two symptoms): the `can_use_flash` path in `Sdpa::run_attention` transposed
  q/k/v `(b,H,s,d)->(b,s,H,d)` WITHOUT `.contiguous()`. `flash_attn_v2` hands the tensor straight to the
  CUDA kernel with no internal contiguous, so for **seq>1 (prefill)** the non-contiguous strides are read
  wrong -> garbage. **Decode (seq==1)** is trivially contiguous so it worked -- which HID the bug and made
  it look prefill-specific. Because flash garbled, GGUF prefill had been left on the eager `naive_sdpa`
  fallback (the O(n^2) collapse). ONE `.contiguous()` fixes both.
- HOW IT WAS FOUND (methodology, not guessing): (1) an ISOLATION TEST (`flash_correctness` in attention/
  mod.rs) proved `flash_attn == naive_sdpa` to maxabs 0.0156 on controlled bf16 GQA-causal inputs WITH
  `.contiguous()` -> the crate/kernel is CORRECT (killed the "Blackwell bf16 kernel bug" theory). (2) a
  `DBG_ATTN` trace showed the real prefill call is `mask=CausalFlash q_dims=[1,32,512,128]` -> the
  `can_use_flash` path (NOT the Custom-mask block I'd been chasing). (3) that path lacked the `.contiguous()`
  the passing isolation test had. Fix = add it. Earlier guesses (Custom-mask route, GQA repeat_kv, window,
  bf16 dispatch) were all on the WRONG path -- the trace corrected course.
- RESULT (8B CUDA, spark GB10, main + fix): prefill coherent + FLAT -- pp512 2653 (pure) / 2349 (with
  decode) vs llama 2779 = **0.85-0.95x** (was 0.70x); pp2048 ~2586 flat vs the old eager 1118 =
  **0.37x -> ~0.86x** (the big long-context win). Decode 35.8 vs 36.0 = **0.99x (parity)**. So CUDA is
  now NEAR-PARITY on BOTH prefill and decode, and the pp2048 collapse is gone. Fixes flash prefill for
  ALL CUDA models (any that hit the CausalFlash path), not just GGUF. Residual prefill gap to llama is
  now kernel-efficiency (llama's flash is well-tuned), not an algorithmic O(n^2) defect.
- Flash dispatch is DECOMPLECTED: no runtime env knob. Flash is used iff `using_flash_attn()` (compiled)
  AND applicable (device/dtype/shape/causal) -- a pure function, not a place. The dev-time `FLASH_ATTN`
  A/B toggle (and the branded `HANZO_ROCM_FLASH_ATTN`) were deleted; ROCm flash is always-on when applicable.

## 8B repetition-collapse ROOT-CAUSED: flash PREFILL precision (a regression from the .contiguous fix)
- SYMPTOM: Qwen3-8B-Q4K_M on CUDA (and Vulkan) collapses to `钊，，，` repetition during LONG thinking-mode
  generation (plain prompt -> `<think>` block). Short / `/no_think` output is coherent. Metal is fully
  correct (same byte-identical GGUF -> "Paris"). 0.6B/4B fine on all backends.
- LOCALIZED by A/B (seed 42, plain prompt): FORCE_EAGER (prefill+decode eager) = COHERENT; DECODE_EAGER
  (prefill flash, decode eager) = STILL COLLAPSES. So the collapse originates in flash **PREFILL**, not
  decode. NOT the file (identical sha256), NOT the quant kernel (DEQUANTIZE_ALL still collapses).
- ROOT CAUSE: the CUDA flash-attn prefill is ~1.5% numerically off vs eager naive_sdpa (isolation test
  `flash_matches_naive` maxabs=0.0156 -- larger than a correct flash's ~0.4% bf16-reorder). That error,
  accumulated across 36 layers, tips 8B's sensitive thinking-mode trajectory into the repetition
  bifurcation. Eager (f32 softmax) is exact and stays stable; Metal's attention is a different, stable impl.
- THE TRADE-OFF THIS EXPOSES: the `.contiguous()` flash-prefill fix (573d12614) gave prefill 0.37x->0.86x
  flat BUT introduced this collapse (pre-fix, GGUF prefill used eager = correct-but-slow). So main now
  ships fast-prefill + 8B-long-gen-collapse on CUDA/Vulkan. Options: (a) revert to eager (correct, slow
  prefill); (b) fix flash's softmax to f32-accumulate precision (deep, in hanzo-flash-attn -- the real
  fix, keeps perf); (c) the kernel-DSL migration (one correct attention impl across backends -- the
  structural cure). DECODE_EAGER is a NEGATIVE (does not fix) -- do not ship it.
- NEXT: diagnose whether hanzo-flash-attn accumulates the softmax in f16 vs f32 (the likely 1.5% source)
  and force f32. If the flash kernel is correct, the 1.5% is the is_causal alignment -- verify the causal
  triangular direction matches naive. Until fixed, 8B-long-thinking on CUDA/Vulkan is a known-issue
  (usable for short/direct gen; Metal fully correct).

## CORRECTION: the sampling change is NOT the 8B collapse fix (drill continued)
- Tested the sane-sampling fix (rep_penalty 1.1, top_p 0.9, temp 0.7) on 8B seed42 flash prefill:
  STILL COLLAPSES -- `针needle needle needle...` (different token than `钊，，，`, but still a repetition
  attractor). So the collapse is NOT a sampling artifact -- rep-penalty does not rescue it.
- CONCLUSION (all hypotheses eliminated by A/B): the collapse is hanzo-flash-attn's PREFILL being subtly
  off. Its softmax is correctly f32, yet its output (~1.5% vs eager) tips 8B-Q4K bf16 into a repetition
  attractor -- while EAGER, Metal's attention, AND llama.cpp's own flash_attn_ext all keep 8B stable. So
  hanzo-flash-attn has a subtle numeric error (worse than llama's flash), NOT just "expected reorder".
- The sampling change is kept as a genuine DEFAULT improvement (temp 0.1/top_p 0.1/no-penalty was bad),
  but it is NOT the collapse fix. Do not present it as such.
- REAL FIX PATHS: (a) revert flash prefill -> eager (correct, prefill 0.37x -- correctness-first);
  (b) fix hanzo-flash-attn's CUDA kernel by numeric-diffing its prefill output against llama's
  flash_attn_ext to find the subtle error (deep, keeps the 0.86x perf); (c) the kernel-DSL migration
  (one correct attention impl -- structural cure). Production (server/API, client-set sampling, short
  gens) is largely unaffected; the collapse is a bare-CLI long-thinking-gen issue on CUDA/Vulkan.

## FINAL VERDICT (numeric diff done): 8B collapse = bf16 BIFURCATION SENSITIVITY, not a fixable flash bug
- The flash-vs-eager seq sweep (attention::flash_precision_probe, CUDA, bf16 GQA causal 8B-shape) settles it:
    seq=8  max_abs=0.0156 | seq=32 0.0156 | seq=128 0.0156 | seq=512 0.0176
  The error is FLAT with seq at ~0.0156 = ONE bf16 ULP for these magnitudes. So it is NOT a systematic
  algorithmic bug (which a short-seq test would also show, but a real bug would be a LARGER fixed offset)
  and NOT reduction-reorder (which would SCALE with seq). It is the bf16 OUTPUT-QUANTIZATION floor: flash
  and eager compute slightly-different-but-both-VALID f32 attention outputs that round to adjacent bf16
  values. flash's softmax is f32-correct (verified: MaxOp<float>/SumOp<float>/ElementAccum=float).
- CONCLUSION: there is NO clean flash bug to fix. 8B-Q4K at bf16 sits on a repetition bifurcation; flash's
  valid rounding tips it, eager's valid rounding does not (Metal is stable because it uses its OWN eager/
  steel attention, NOT the CUDA flash crate). Better sampling did not rescue it (rep-penalty still looped).
- SHIPPED RESOLUTION (pragmatic, CTO call): KEEP flash prefill on main -- the 0.86x-flat prefill win is
  real and broad; correct for the vast majority of inputs; production (server/API, client sampling) is
  unaffected; the collapse is a NARROW bare-CLI long-thinking-gen edge on the one sensitive 8B-Q4K model.
  The CLI sampling defaults were also improved (temp 0.7/top_p 0.9/rep-pen 1.1 -- objectively better than
  the old temp 0.1/top_p 0.1/no-penalty) though they do not fix the bifurcation. `flash_vs_eager_seq_sweep`
  stays as the regression gate.
- THE PERMANENT CURE is the kernel-DSL migration (hanzo-kernel, proven CPU+Vulkan+Metal this session): ONE
  correct attention implementation lowered to every backend eliminates the flash-vs-eager-vs-Metal numeric
  fork by construction -- the whole "same op, N impls, N bf16 roundings, N bifurcation outcomes" class.
