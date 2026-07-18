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

# With ROCm support (Linux, AMD)
cargo build --release --features rocm

# Install hanzo-server binary
cargo install --path hanzo-server --features <features>
```

#### ROCm on AMD APUs (Strix Halo / gfx1151, unified memory)

APUs expose a tiny dedicated-VRAM carve-out (HIP reports ~1 GB) alongside the
large unified GTT pool (~the whole system RAM). A raw `hipMalloc` targets the
1 GB carve-out and OOMs immediately on any real model. Run with unified memory:

```bash
HSA_XNACK=1 LD_LIBRARY_PATH=/opt/rocm/lib \
  hanzo-server --port 8080 gguf -m <repo> -f <file.gguf>
```

`HSA_XNACK=1` lets managed allocations page into GTT so the full unified pool is
usable. (The build already matches the fused qk-norm-rope cos/sin cache to the
activation dtype; without that fix every GGUF MoE decode step errors on gfx1151.)

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

### World Model (Oasis)

`hanzo-engine/src/diffusion_models/oasis/` is a native port of Etched/Decart Oasis-500M
(MIT, code + weights `Etched/oasis-500m`): an action-conditioned frame-autoregressive
latent-diffusion Minecraft world model (a playable generative game engine).

- `vae.rs` — ViT-VAE (`vit-l-20`): 20x20 conv patchify of a 360x640 frame -> ViT blocks ->
  16-dim latent `[16,18,32]`. Attention is 2D pixel-axial RoPE; MLP is exact GELU.
- `dit.rs` — DiT-S/2 spatiotemporal DiT (16 blocks): adaLN spatial-attn + causal-temporal-attn,
  action conditioning via `external_cond` added to the timestep embedding. tanh-GELU MLPs.
- `rope.rs` — interleaved (lucidrains adjacent-pair, `is_gptx=false`) axial RoPE. Spatial/VAE
  use pixel freqs, temporal uses lang freqs; freqs verified against the checkpoint.
- `sampling.rs` — diffusion-forcing rollout (sigmoid-beta schedule, per-frame noise, sliding
  window at `MAX_FRAMES=32`). Fully-hallucinated: only prompt frames are VAE-encoded, all
  subsequent frames are generated purely in latent space (no encoder in the loop).
- `mod.rs` — `WorldModel::{load,encode_frames,generate,decode_frames}` + 25-key `ACTION_KEYS`.

Parity vs the torch reference (CPU f32, `parity.rs`, weight-gated on `OASIS_DIR`): VAE encode
latent cosine 1.000000, decode PSNR 59.73 dB, DiT single-step cosine 1.000000.

Gotcha: candle `Linear` only matmuls up to rank 4; the DiT keeps rank-5 `[B,T,H,W,D]` tensors,
so all its linears go through `apply_linear` (flatten leading dims -> matmul -> reshape).

Example: `cargo run --release --example oasis_generate -- --frames 32 --steps 10 --out <dir>`
(prompt image + action stream -> PNG frames; assemble with ffmpeg).

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

## Qwen3.5/3.6 hybrid (GDN + MoE) and the portable kernel DSL

- **`quantized_qwen3_5_moe`** is the Qwen3.5/3.6 (`qwen3_5_moe` arch) hybrid model: Gated-DeltaNet
  (GDN) linear-attention layers interleaved with MoE. Registered end-to-end (GGUF arch
  `Qwen35`/`Qwen35MoE` in `pipeline/gguf.rs`, `Qwen3_5MoeLoader`, vision + text variants). GDN lives
  in `hanzo-engine/src/models/gdn.rs`.
- **Fused GDN kernel**: `gdn.rs` routes the gated-delta-rule recurrence through ONE fused per-backend
  kernel (`fused_gdn_gating_{cuda,metal}` + the fused scan) that drives Qwen3.5/3.6 to near-parity;
  the portable f32 ops-composed scan (`gdn_step_scalar`) is the reference and the CPU/Vulkan fallback.
  The fused-vs-portable A/B env knob was removed; bit-exact CPU gates (`gdn_step_matches_reference_seq1`,
  `gdn_recurrence_cpu_shapes`) are colocated in `gdn.rs` and compare the two paths directly.
- **hanzo-kernel** (the portable kernel DSL, published 0.2.15) is a CODEGEN: one
  `#[kernel(targets(...))]` Rust source lowers to CUDA/ROCm/Vulkan/Metal/WebGPU/CPU. Its op library
  (`gdn`, `norm`, `rope`, `attn` incl. `sdpa_runtime`, `quant`) and the `fuse` auto-fusion pass
  (fusion == composition of Map morphisms, one `fused_interp` launch, zero intermediates) are all
  bit-exact on CPU. `sdpa`/`sdpa_runtime` online-softmax is the structural cure for the 8B
  flash-collapse. See `ml/LLM.md` for the DSL and env-var details.

## Kernel tuning + new-model onboarding: the roofline-driven process (SOTA loop)
A repeatable, measurement-first process. The scar tissue behind every step is in `ml/LLM.md`
("Vulkan PREFILL: the Q6_K lever, the roofline method..."). The rule: **measure in-engine, never trust
a cache-warm microbench for a memory-bound kernel, and kill a lever the moment the engine says it lost.**

### Tune a kernel (the loop that took prefill 212 -> 704 t/s)
1. PROFILE a real forward: `VK_PROFILE_GPU=1 VK_PROFILE=1 hanzo bench --prompt-len 512 --gen-len 0 ...`
   -> per-op GPU time + record/submit/fence + pool fresh/hit counters.
2. RANK by roofline: achieved BW = bytes-moved / GPU-time vs device peak. Biggest (time-share x
   distance-from-roofline) = the target. (Q4_K GEMM: ~24 GB/s vs ~135 practical = the 18% that IS the gap.)
3. PICK THE LEVER by what the roofline says, not by fashion: coalescing / occupancy / layout / reuse /
   fusion / vectorization. NOT tensor-core dtype (coopmat is measured-dead on gfx1151), NOT bigger tiles
   (occupancy collapse, measured 2.4x WORSE).
4. IMPLEMENT in the DSL/shader; GATE BIT-EXACT -- a CPU oracle OR a CONTROLLED known-answer test (uniform
   weight -> exact k localized the Q6_K decode bug in minutes where random-vs-oracle only said "wrong").
5. MEASURE IN-ENGINE (`--gen-len 0` = the true cold-weight prefill floor). Env-gate A/B variants
   (VK_Q4K_BM, VK_Q6K_LEGACY, VK_Q4K_COOPMAT). Cache-warm microbench LIES for memory-bound kernels.
6. KEEP only if it wins in-engine; RECORD the dead ends (this session killed coopmat + bigger-BM).

### Autotune (mechanical, per shape/device -- NOT AutoML)
Kernel-level LATENCY search over tile/occupancy configs (BM/BN/BK/threads/unroll), measured and cached
per (kernel, shape, device). Deterministic objective (ns), cheap trials -> a for-loop + cache, like
`triton.autotune` / cubecl autotune. The env A/B knobs (VK_Q4K_BM=64/128/256) are the manual version;
the DSL autotune automates the sweep so a hand-picked constant can't be a local optimum on another GPU.

### Onboard a new model fast (get it to SOTA on our stack)
1. LOAD the GGUF; confirm COHERENCE with a known-answer generation (`hanzo run -i "The capital of France is"`).
2. PROFILE prefill + decode -> the hot kernels for THIS model's quant mix + shapes.
3. ROOFLINE-RANK -> the offenders. The classic miss: a quant type with NO tiled prefill path (Q6_K was
   76% of prefill because attn_v/ffn_down are Q6_K in Q4_K_M and only had the column matvec).
4. ENSURE every quant type in the model has a tiled prefill path (Q4_K, Q6_K done; Q5_K/Q8_0/IQ next).
5. AUTOTUNE the tile configs for the model's shapes.
6. GATE vs llama-bench, SAME box + model: `~/llama.cpp/build/bin/llama-bench -m <gguf> -p 512 -n 128`
   vs `hanzo bench --prompt-len 512 --gen-len 128 auto -m <dir> -f <gguf>`. That ratio is the SOTA gate.

### The tooling
- Roofline dashboard (visual "find" UI: roofline plot + priority-ranked kernels + tuning log).
- `VK_PROFILE_GPU` (per-op GPU ns), `VK_PROFILE` (record/submit/fence + pool counters), `VK_ROOFLINE`
  (planned: per-dispatch exact bytes/FLOPs joined to timestamps, auto-refreshing the dashboard).
- The DSL end state (`ml/LLM.md` "DSL MAGIC"): layout-as-value + a coalesced layout-aware loader +
  tiled-matmul + pluggable quant-decode + autotune, so every kernel inherits coalescing correct-by-
  construction and naive strided reads stop being possible.

## Qwen3-VL GGUF (text backbone) — `qwen3vl` / `qwen3vlmoe`

- **GGUF fast-path** for Qwen3-VL: the dense text backbone (`qwen3vl`, 2/4/8/32B) reuses
  `quantized_qwen3::ModelWeights`; the MoE text backbone (`qwen3vlmoe`, 30B-A3B / 235B-A22B)
  reuses `quantized_qwen3_moe::ModelWeights`. Wired as `GGUFArchitecture::Qwen3Vl` / `Qwen3VlMoE`
  in `gguf/mod.rs`, dispatched in `pipeline/gguf.rs`, sized in `utils/gguf_metadata.rs`. No new
  modeling code — the VL text tower is structurally identical to Qwen3/Qwen3MoE (same `blk.*`
  tensors, q/k-norm, GQA); metadata is read under the file's own `qwen3vl.*` / `qwen3vlmoe.*`
  prefix (dynamic `path_prefix`), so `rope.freq_base`, head dims and expert counts come straight
  from the file.
- **Why reuse is bit-correct for text**: Qwen3-VL uses interleaved-MRoPE (sections [24,20,20]),
  but for text-only tokens t==h==w, so every frequency band receives the same position and the
  interleaved partition is a no-op — identical to the standard 1D RoPE already in
  `quantized_qwen3`. Verified: Qwen3-VL-8B-Instruct Q4_K_M loads (36 layers, 151936-tok Qwen3
  BPE), dispatches to `Model::Qwen3`, and decodes coherent text.
- **Image-blind (mmproj not wired)**: GGUF Qwen3-VL ships TWO files — the LLM gguf (loaded here)
  and a separate `mmproj-*.gguf` vision tower (`general.architecture=clip`,
  `clip.projector_type=qwen3vl_merger`, `v.blk.*` / `mm.*` / `v.deepstack.*`, DeepStack at vision
  layers 8/16/24). `GGUFPipeline` is text-only (modalities Text→Text, no image inputs processor),
  so the mmproj is NOT consumed. Full image support needs a `clip`/mmproj GGUF loader feeding the
  existing `vision_models/qwen3_vl` tower PLUS a multimodal quantized-VL text forward (3D-MRoPE
  position_ids + DeepStack embed injection + `<|image_pad|>` merge) — a new subsystem, not a
  reuse of `quantized_qwen3::forward` (whose signature carries only 1D `start_offsets` and no
  DeepStack hook). See the safetensors `Qwen3VLLoader` for the reference multimodal path.
- **Decode graphs EXCLUDED** for `qwen3vl`/`qwen3vlmoe` GGUF: they dispatch to the `Qwen3`/`Qwen3MoE`
  model variants, which are already graph-eligible for pure-text 1D RoPE; the VL archs run the same
  eager/graph text path. (Multimodal 3D-MRoPE, if added, would be position-dependent and must stay
  eager per `model_supports_decode_graph`.)

## Environment variables (bare names, one-way; de-brand DONE)

The env-flag convention is BARE names (no `HANZO_` brand prefix). The de-brand is COMPLETE: every runtime
knob is a bare name and the one-off dev A/B "fallback" toggles are DELETED (production always runs the
fast path; runtime HW auto-select is kept). One bare name per real knob. Canonical flags in `perf_flags.rs`:
- `CUDA_GRAPHS`, `ROCM_GRAPHS`, `METAL_GRAPHS`, `FLASHINFER_DECODE`, `FLASH_PREFILL` -- all default ON,
  set `=0` to force the eager/unfused path. (`HANZO_ROCM_FLASH_ATTN` and its A/B toggle were already
  deleted; ROCm flash is always-on when applicable.) `FLASH_PREFILL` gates `using_flash_attn()` (CUDA).
- **De-branded runtime config**: `KV_SPILL_DIR`, `KV_SPILL_BUDGET_MB`, `ISQ_SINGLETHREAD`,
  `DEV_SIGNING_SEED`, `VK_FUSED_QKNORM`, `MXFP4_DP4A`, `MN_LOCAL_WORLD_SIZE`, `NO_NCCL`,
  `FFI_MODELS`, `FFI_TOK_DIR`, `CUDA_FLASH_BF16`, `METAL_PRECOMPILE`, `ROCM_GFX_ARCH`.
- **DELETED (one-off dev A/B toggles that forced the OLD/slow path)**: `HANZO_GDN_FUSED_FALLBACK`,
  `HANZO_ADD_RMSNORM_FALLBACK`, `HANZO_QK_NORM_ROPE_FALLBACK`, `HANZO_NO_MEMPOOL_FIX`, `SAMPLER_TRACE`
  (plus the ml `HANZO_Q*K_FALLBACK` / `HANZO_MOE_*_FALLBACK` / `HANZO_IQ*_FALLBACK` family). The fused
  vs unfused decision is now the always-fast path; bit-exact oracle tests force the unfused/scalar leg
  via test-only programmatic setters (`layers::set_force_unfused_qk_norm_rope`,
  `hanzo_ml::set_force_scalar_matvec`), never env, never set in production.
- License gate reads `HANZO_ENGINE_LICENSE_TOKEN` / `HANZO_ENGINE_LICENSE_FILE` +
  `HANZO_LICENSE_SIGNING_KEY` (see `license.rs`); these are product-namespaced identity vars, not perf
  flags, and stay branded by design -- the ONLY remaining `HANZO_` env names.

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
Current best config: **bf16 carrier + native dp4a** (CUDA_FAST_MMQ / V4 `set_fast_mmq(true)`)
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
- Flash dispatch: `using_flash_attn()` (CUDA) is now default ON with a `FLASH_PREFILL=0` opt-out; flash is
  used iff that predicate AND applicable (device/dtype/shape/causal). ROCm flash is always-on when applicable.

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

## RESOLVED: flash prefill default ON (dense) + GGUF gated (spark GB10, hanzo-flash-attn 0.11.35)
- `using_flash_attn()` flipped default ON (`FLASH_PREFILL=0` opts out) after bumping hanzo-flash-attn to
  0.11.35 (routes bf16 softmax-P through the f16 kernel's 10 mantissa bits vs bf16's 7).
- DENSE bf16 8B on CUDA: flash prefill is BYTE-EXACT vs eager (greedy seed-0, first-prefill non-varlen path)
  and 1.34x@pp512 / 2.46x@pp2048 faster (flat vs length: eager 2110/1166, flash 2825/2868 T/s), beats
  llama.cpp Q4KM (2706/2723). Decode 14.8->15.0 (paged kernel, unchanged). d_flash == d_bf16raw == d_eager,
  so the P-cast is NOT what tips dense here.
- GGUF/quantized garble under flash and are GATED to eager via `CausalMaskConfig::gguf()` (all
  `quantized_*` models). Verified: Qwen3-8B-Q4K flash emits corrupted text ("-devel...corrupted") that is
  P-precision-INDEPENDENT (q_flash == q_bf16raw), so it is NOT the 1.5% P-cast issue above -- it is
  structural: GGUF `forward_attn` has no FlashParams plumbing (passes `None`), so flash runs without
  cumulative seqlens. Q4K + MoE flash == eager (coherent "Paris") after the gate. This is the same config as
  the "8B-Q4K thinking-mode collapse" above.
- The paged prefix-cache/chunked GATHER path (varlen, q_len<kv_len) is coherent under flash but its greedy
  output differs from a fresh (non-cached) run -- and EAGER gather differs from fresh too, so this is a
  pre-existing prefix-cache non-transparency (reduction-order/fp between gather vs non-gather kernels), NOT
  a flash-specific garble. Single-shot generation never hits the gather path (needs a prefix-cache hit), so
  the default-ON win is safe there. A transparent gather (build-2) remains a separate follow-up.

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

## Pixal3D: native image-to-3D (TRELLIS-image-large, MIT) -- `diffusion_models/pixal3d/`
- Target = microsoft/TRELLIS-image-large (the mature 1.2B model). TRELLIS.2-4B (DINOv3, ConvNeXt VAE,
  RoPE, shape/texture cascade) is the successor and a 3x heavier follow-up, NOT this port.
- Pipeline stages: image -> DINOv2 tokens -> sparse-structure rectified-flow -> Conv3d decode to a 64^3
  occupancy grid -> [SLAT sparse flow -> FlexiCubes mesh] -> GLB. The DENSE half (through occupancy) is
  fully ported; the SLAT/FlexiCubes half is the frontier (needs sparse-conv infra + FlexiCubes).
- ALL dense stages are BIT-EXACT vs the torch reference oracle (cos=1.00000000):
    dinov2 (max|d|=8.2e-5), ss_flow (mse=3.0e-12), ss_decoder (mse=2.3e-9).
  Oracle = TRELLIS's own pure-torch code run CPU-only with ATTN_BACKEND=sdpa and the sparse/render
  subpackages stubbed (`sys.modules['trellis.{pipelines,renderers,representations}']=ModuleType(...)`),
  so no spconv/nvdiffrast/flash needed to validate the dense models. Scripts in scratchpad/trellis_oracle.
- Key facts pinned from the reference source (do not re-derive):
  * DINOv2 = torch-hub `dinov2_vitl14_reg` (NOT the HF-naming variant): fused qkv, LayerScale ls1/ls2,
    4 registers inserted AFTER cls (no pos-emb), 518->1374 tokens, exact-erf GELU; TRELLIS uses `x_prenorm`
    (before the final `norm`) then a non-affine `F.layer_norm` (eps 1e-5). Block norms eps 1e-6.
  * ModulatedTransformerCrossBlock (SS + SLAT share it): norm1/norm3 non-affine, norm2 affine; self-attn
    is adaLN-gated with per-head qk-RMSNorm (= L2-normalize over head_dim * gamma[H,D] * sqrt(D)); cross-attn
    to the DINOv2 tokens is un-gated/un-modulated; FFN is tanh-GELU, adaLN-gated. adaLN = SiLU->Linear(C,6C).
  * Sampler passes t*1000 to the model; timestep_embedding is cos-then-sin; SS pos_emb is a stored buffer.
  * FlowEuler: t_seq=linspace(1,0,steps+1) reparam by rescale_t=3; step x-=（t-t_prev)*v; CFG only inside
    cfg_interval [0.5,1] as (1+s)*cond - s*uncond, s=5; neg_cond = zeros_like(cond).
- hanzo-ml has NO Conv3d: the decoder's only kernel (3x3x3, pad1, stride1) is decomposed into 3 depth-sliced
  conv2d summed with depth alignment (ss_decoder.rs::Conv3d). candle tuple Shape/Dims cap at rank 6 -> use
  Vec/&[usize] for the 8-D pixel_shuffle_3d reshape/permute.
- Weights: `PIXAL3D_MODEL` dir holds the TRELLIS ckpts (ckpts/*.safetensors) + a converted
  `dinov2_vitl14_reg.safetensors` (torch-hub .pth has no safetensors; convert once). Runs Device::Cpu today.
- GLB export = hand-rolled binary glTF 2.0 (glb.rs), validated against the third-party `gltf` loader;
  wired into ThreeDFormat::Glb (was a PLY fallback). The /v1/3d async endpoint already calls pixal3d_generate.
- SLAT + FlexiCubes half is now DONE (the fine textured mesh), all bit-exact vs the reference:
  * sparse.rs = the sparse-tensor infra (SparseTensor coords[N,3]+feats[N,C], B=1): submanifold Conv3d
    (SubMConv3d, gather active neighbours c+(kz-1,ky-1,kx-1)), SparseLinear, downsample/upsample/
    subdivide, SparseGroupNorm32, AbsolutePositionEmbedder, window partition. GOTCHA: spconv's CPU build
    is ~2% WRONG on sparse SubMConv (single-tap dense is exact, multi-tap sparse gather-gemm is buggy) --
    the correct oracle is dense conv3d @ active sites (== GPU spconv). GOTCHA: SparseDownsample uses
    torch.scatter_reduce(mean, include_self=True) so the divisor is (group size + 1), not the size.
  * slat_flow.rs (SLatFlowModel) reuses the SS-flow ModulatedCrossBlock verbatim (full attention over the
    voxel set == dense sdpa at B=1) + SparseResBlock3d pack/unpack (downsample 64->32, 24 blocks, upsample
    32->64 with U-Net skips). NO stored pos_emb (on-the-fly AbsolutePositionEmbedder from coords). Parity
    cos=1.0000, max|d|=5.2e-5.
  * slat_decoder.rs (SLatMeshDecoder): 12-block sparse SWIN transformer (windowed self-attn, alternating
    shift 4, non-affine norms), 2 SparseSubdivideBlock3d (GroupNorm32+subdivide+conv residual) 64->128->256,
    out_layer -> 101-ch FlexiCubes layout (sdf 8 + deform 24 + weights 21 + color 48). Parity cos=1.0000.
  * flexicubes.rs (+ flexicubes_tables.rs) = nvdiffrec FlexiCubes dual-MC, inference path. Ported SPARSELY:
    the SDF grid is 1 everywhere except active-voxel corners, so only cubes touching an sdf<0 vertex can be
    surface -- no dense 256^3 arrays. sparse_cube2verts corner-mean, DMC case + check_table ambiguity
    inversion, beta-weighted dual verts from alpha-weighted edge zero-crossings, 6-ch color interp, quad
    triangulation split by gamma with sdf winding. Cubes processed in reg_c (z,y,x) order so the 4-cube
    quad winding matches. Parity vs golden mesh: verts 32981==32981, faces 63978==63978, Chamfer=0.000000.
  * pipeline.rs generate_textured(): coarse occupancy -> active coords (argwhere>0, res 64) -> SLAT flow
    denoise (from noise, CFG) -> denormalize (*std+mean, consts SLAT_MEAN/STD) -> mesh decoder -> FlexiCubes.
    /v1/3d exposes it via the `texture` request flag; GLB carries per-vertex RGB as COLOR_0 (mesh_to_glb_colored).
- Oracle method for the SLAT half: real TRELLIS + real CPU spconv, with a flash_attn shim (sdpa CPU) so the
  real code runs, trellis.{pipelines,renderers,gaussian,octree,radiance_field} stubbed, kaolin.check_tensor
  stubbed, FlexiCubes forced device=cpu, and SparseConv3d monkeypatched to the correct submanifold gather
  (CPU spconv is buggy). Scripts in scratchpad/oracle (gen_oracle.py + flash_attn.py).

## LLaDA: diffusion LLM (dLLM) native port -- models/llada.rs
- Model: GSAI-ML/LLaDA-8B-Instruct (MIT, ungated, most-downloaded dLLM). Same code loads LLaDA-1.5 +
  LLaDA-8B-Base (identical arch). Rejected: DiffuCoder (apple-amlr research-only); Dream-7B (Apache, qwen2-
  arch) is the fallback; LLaDA-MoE-7B-A1B (Apache) is future.
- Arch = Llama with 2 differences: BIDIRECTIONAL attention (no causal mask) + iterative masked-diffusion
  generation (no KV cache, whole sequence re-forwarded each step). OLMo config names (d_model 4096, n_heads
  32, n_kv_heads 32 = MHA, n_layers 32, mlp_hidden_size 12288 SPLIT SwiGLU ff_proj gate + up_proj, ff_out
  down, rms_norm, rope_theta 500000 NeoX, untied lm_head = transformer.ff_out, vocab 126464, mask 126336).
  Reuses RmsNorm/Sdpa/RotaryEmbedding/ReplicatedLayer verbatim; the ONLY structural change vs llama.rs is
  the mask. Bidirectional = AttentionMask::None + FlashParams::empty(false) (pitfall 6). Weight prefix is
  model.transformer.{wte,blocks.N.*,ln_f,ff_out}.
- Sampler (official LLaDA generate.py, ported): semi-autoregressive blocks (block_length), low-confidence
  remasking (confidence = max softmax prob = softmax at argmax), linear noise schedule
  (num_transfer_tokens), greedy (temp 0). Per step: full forward -> argmax x0 + confidence -> unmask the
  top-k highest-confidence masked positions within the current block. Selection on CPU (tiny), forward on
  GPU (dominant). Deterministic.
- PARITY (GB10, bf16, vs transformers-4.46 oracle -- tf5.x breaks LLaDA remote code via all_tied_weights_keys):
  gate1 single-forward logit cosine > 0.99; gate2 generation BYTE-IDENTICAL on deterministic prompts (France
  ->"Paris" [65926,eot,eos...] 32/32; 17*23->"391" [18,24,16,...] 32/32); haiku 19/32 = legitimate bf16
  confidence-tie nondeterminism (both valid). Coherence: "Paris", coherent haiku.
- THE DISTRIBUTED THESIS, measured (gen=128, block=32, GB10 bf16): tok/s scales LINEARLY with tokens-per-
  traversal (steps 128->1.0 tok/traversal->6.9 T/s; 64->2.0->13.6; 32->4.0->27.7). Each denoise step = ONE
  model.forward = ONE ring traversal (pp_head_forward). So a 2-node ring PP dLLM amortizes the traversal
  latency over MANY tokens per hop -- inverting AR's 1-token-per-traversal economics. PP wiring is a thin
  adapter: PipelineParallelModel = {pp_embed=wte, pp_run_local=local block range, pp_norm_head=ln_f+ff_out};
  loop pp_head_forward N times = N traversals. (Trait in pipeline_parallel.rs; single-node sweep proves the
  economics; 2-node execution + /v1 serving pipeline are the remaining wiring.)
- No-KV-cost caveat: every step re-forwards the whole [prompt+gen] sequence, so per-step cost grows with
  sequence length (Fast-dLLM-style block KV cache is the known follow-up). gated tests: llada_parity,
  llada_smoke (LLADA_WEIGHTS + LLADA_ORACLE / LLADA_GEN/STEPS/BLOCK).

## ACE-Step music: /v1/audio/music (this session)
- Made the composable AceStepPipeline (diffusion_models/ace_step/) servable as a new SpeechLoaderType.
  UMT5-base text encoder + DiT flow-match (APG guidance) + DCAE + HiFi-GAN vocoder -> stereo 44.1kHz.
- Registration mirrors TTS exactly: SpeechLoaderType::AceStep (FromStr `ace_step|ace-step|acestep`,
  auto-detect on the DiT config's `_class_name == "ACEStepTransformer2DModel"`), SpeechGenerationConfig::
  AceStep { frames, steps, guidance_scale } (default ~10s = 10*44100/4096 frames, 27 steps, guidance 15).
  Reuses ModelCategory::Speech + Response::Speech + RequestMessage::SpeechGeneration -> ForwardInputsResult
  ::Speech; forward_inputs tokenizes with the UMT5 tokenizer (umt5-base/tokenizer.json), runs generate(),
  and interleaves the (1,C,S) waveform into PCM (channels=2, rate=44100).
- Loader (pipeline/speech.rs) fetches the 4 real ACE-Step sub-checkpoints from ACE-Step/ACE-Step-v1-3.5B:
  umt5-base/model.safetensors (keep all), ace_step_transformer/diffusion_pytorch_model.safetensors (drop
  the unused `lyric*`/`projectors`/`.add_`/`.to_add_out` heads), music_dcae_f8c8/... (keep `decoder.*`),
  music_vocoder/... (keep all). All loaded f32 (vocoder/DCAE fidelity). Key filters byte-match the proven
  ace_step_generate_e2e test fixtures; the DCAE f8c8 gives 4096 samples/latent-frame (44100/4096 ~10.77 fps).
- Server: POST /v1/audio/music (MusicGenerationRequest {model,input,response_format}) in
  hanzo-server-core/src/music_generation.rs reuses speech_generation's parse + response plane (wav/pcm).
  Merged to main (5c8b8968b). Per-request duration/steps/guidance = config-driven (load-time), same as TTS;
  threading them per-request would need RequestMessage::SpeechGeneration to carry gen params (follow-up).

## Customer surface: where a Hanzo SaaS customer generates audio/video/3D
- Engine (this repo) serves the OpenAI-adjacent /v1 modality plane: /v1/chat/completions (text+dLLM),
  /v1/images/generations, /v1/audio/speech (TTS), /v1/audio/music (ACE-Step), /v1/audio/transcriptions
  (ASR), /v1/3d (TRELLIS/Pixal3D, `texture` flag), /v1/videos (WAN async), /v1/animate (MuseTalk dub).
  Deployed via platform.hanzo.ai on DOKS behind the gateway (api.hanzo.ai).
- studio.hanzo.ai (hanzoai/studio, ComfyUI fork; deployed by hanzo.yml onto do-sfo3-hanzo-k8s / studio
  Service CR) is the PRIMARY authoring surface: the `hanzo_engine` node pack (custom_nodes/hanzo_engine)
  maps one node == one /v1 endpoint over a shared client; the flagship workflow hanzo-full-generative-
  pipeline.json chains Chat -> ImageGen -> (textured ImageTo3D + WAN video) + TTS + Music.
- hanzo.app (app builder + /games) is the consumer/preview surface for the produced assets.
- PRODUCTIZATION GAP (hosted multi-tenant), named precisely from the studio code:
  (1) Engine URL is not per-tenant: the node pack reads process-global HANZO_ENGINE_URL (default
      localhost:1234); it is set NOWHERE in the studio repo, and the per-org engine/worker routing
      (middleware/engine_selector.py + compute_config.py + prompt_router.py, which route the graph to
      local vs a BYO-GPU worker) never derives or sets the node pack's /v1 base URL per org/worker.
  (2) Identity is not propagated to the engine: iam_auth_middleware.py authenticates the Studio surface
      (Hanzo IAM JWT, org from the `owner` claim) and worker_client signs Studio->worker calls, but
      custom_nodes/hanzo_engine/client.py sends the /v1 calls with NO Authorization header -> a gateway-
      fronted engine (api.hanzo.ai injects identity, scopes by `owner`) gets an anonymous/unscoped call.
  (3) Hosted GPU + model availability: /v1/3d,/videos,/audio/music,/animate are GPU-bound and need those
      specific models loaded; the compute layer schedules the ComfyUI graph on GPU workers but there is no
      per-org binding of "engine deployment X (with ACE-Step/WAN/TRELLIS/MuseTalk loaded) serves this org".
  Self-hosted single-engine works TODAY (set HANZO_ENGINE_URL, run `hanzo serve`); the gap is the 3-part
  wiring to make it hosted SaaS: per-org engine-URL resolution from engine_selector, IAM-token forwarding
  onto the node /v1 calls, and per-tenant engine/model provisioning.

## FP8 -> GGUF converter for GLM-5.2 (glm-dsa) -- scripts/convert_fp8_to_gguf.py
- WHY: unblock loading GLM-5.2 in-engine. The loader is `models/quantized_deepseek2.rs`, which
  accepts `general.architecture in {deepseek2, glm-dsa}`; GLM-5.2 = glm-dsa (split-MLA + 256-expert MoE).
- DISK-SAFE: stream one ~5 GB FP8 shard, requantize, delete it,
  next. But GGUF's whole tensor directory precedes its data blob, so it can't append per shard. So: (1)
  PLAN the full directory + metadata from config.json, write header/KV/tensor-info, preallocate; (2) stream
  shards, `pwrite` each tensor (or one expert's slice of a rank-3 bank) into its fixed offset. Resumable via
  a `.progress.json` sidecar keyed by a plan signature (offsets shift if any --*-type/config changes).
- THE glm-dsa GGUF CONTRACT (matched byte-for-byte to the reader + utils/gguf_metadata.rs):
  - routed experts -> rank-3 `blk.{i}.ffn_{gate,up,down}_exps.weight`, ne=[hidden, moe_inter, n_expert], Q4_K.
  - router `ffn_gate_inp.weight` + no-aux `exp_probs_b.bias` -> F32 (reader dequantizes them).
  - split-MLA: `attention.key_length`=qk_nope, `key_length_mla`=q_head_dim, `value_length_mla`=v_head_dim,
    `rope.dimension_count`=qk_rope. q_lora path uses attn_q_a/attn_q_a_norm/attn_q_b; kv_b shipped combined.
  - `expert_gating_func`=2 (sigmoid), `expert_group_count`=1 (noaux), `leading_dense_block_count`=first_k_dense.
  - MTP/nextn block at index n_layers, emitted Q8_0 -- an int4 draft head measures ~0% acceptance, so
    speculation never starts. `nextn_predict_layers` records it; the text loader drops it from
    block_count (n_layers = block_count - nextn).
  - VALUE-TYPE gotchas from hanzo-ml gguf `Value`: `to_f32` accepts ONLY F32 (rms_eps/rope.freq_base/
    expert_weights_scale must be FLOAT32, not F64); `to_bool` ONLY Bool; `to_u32/to_u64` upcast U8/U16/U32.
    `context_length` is `.unwrap()`ed by the device mapper -- must be present.
- QUANT: gguf-py 0.19 quantizes Q8_0/F32 but NOT the K-quants (it does dequantize Q4_K). So Q4_K is a
  faithful `quantize_row_q4_K_ref` (make_qkx2 + K_SCALE packing, numpy-vectorized over super-blocks),
  proven by round-tripping through gguf-py's own Q4_K dequantizer (`--selftest`, relerr ~0.075).
- VALIDATED here without the 756 GB checkpoint: `--selftest` builds a tiny synthetic glm-dsa GGUF end-to-end
  and asserts every reader-required key/type/tensor/shape via gguf-py `GGUFReader`; an on-disk FP8(e4m3
  128-block)+BF16 safetensors fixture exercised the `--indir` streaming path and resume. NEEDS a real
  zai-org/GLM-5.2-FP8 checkpoint for full-scale conversion + an in-engine load smoke test (and the tokenizer
  is loaded separately -- GGUF vocab embedding is not done here).
