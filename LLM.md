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
