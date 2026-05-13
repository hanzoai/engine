# Hanzo Engine - LLM Inference Integration

This file provides guidance to AI assistants working with the Hanzo Engine codebase.

## Project Overview

**Hanzo Engine** is Hanzo AI's high-performance LLM inference engine written in Rust.

### Integration Status

- **Last Sync Date**: 2026-05-06 — merged upstream mistral.rs `2d4ba4f16`
- **Remote**: Configured as `upstream` (EricLBuehler/mistral.rs) in git
- **Workspace version**: 0.8.1 (synced with upstream)

### Hanzo-Specific Components

`hanzo-engine/` is now both a **library** and a **binary**:

1. **Library — `hanzo_engine::*`** (the canonical inference API for the Hanzo stack):
   - `InferenceEngine` trait: `fn infer(&self, model_id: &[u8;32], prompt: &[u8]) -> Result<Vec<u8>, EngineError>`
   - `EmbeddingEngine` trait: `fn embed(&self, dim: usize, text: &[u8]) -> Result<Vec<f32>, EngineError>`
   - Process-wide registry (`OnceLock`-backed): `register_inference_engine`, `register_embedding_engine`, `infer`, `embed`
   - `MistralEngine` — real implementation backed by `mistralrs-core` (handles HF repos and local paths; derives `model_id` as SHA-256 of source)
   - Consumers: hanzo-vm precompiles `0x0201` (AI inference) and `0x0202` (AI embedding) — they call `hanzo_engine::infer` / `embed` synchronously through the registry
   - **NOT** the routing/pricing crate. The Hamiltonian-Hidden-Markov MarketMaker lives in `hanzo-hmm` (`~/work/hanzo/net/hanzo-hmm`) and prices heterogeneous compute; the EVM precompiles depend on `hanzo-engine`, not on `hanzo-hmm`.

2. **Binary — `hanzo-engine`** (thin CLI wrapper):
   - Shells out to `mistralrs-server` for the full HTTP server experience
   - Use the library for programmatic / in-process integration

### Architecture

Hanzo Engine is a Rust workspace containing:
- All upstream mistral.rs workspace members (mistralrs-core, mistralrs-server, mistralrs, mistralrs-mcp, …)
- **hanzo-engine/** — lib + bin: canonical Hanzo inference + embedding API
- Local candle fork at `../ml/hanzo-{ml,nn,flash-attn,metal-kernels}` overrides upstream's `candle-*` crates via `[workspace.dependencies]` path overrides

The engine provides comprehensive LLM inference with support for text, multimodal (incl. video), image generation, speech, and embeddings through multiple APIs (Rust, Python, OpenAI HTTP, MCP).

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

# Install mistralrs-server binary
cargo install --path mistralrs-server --features <features>
```

### Testing & Quality
```bash
# Run core tests
cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision

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

When integrating a new model, make sure it respects all of the varbuilder `.pp` calls. In Candle, a VarBuilder maintains an internal path vector that acts like a “current working directory” for model weights; every call to pp("sub") (alias for push_prefix) clones the builder and appends sub, so successive calls accumulate a dotted prefix such as transformer.h.0 while leaving the original builder untouched . When you eventually call get(...), Candle joins that prefix with the tensor name (prefix + "." + name) and looks it up in the checkpoint backend, producing keys that exactly match the dot-separated names emitted by PyTorch’s state_dict/named_parameters, which means PyTorch-trained weights can be loaded without any renaming  ￼. This lets you recreate the PyTorch module tree in Rust by “walking” it: e.g. vb.pp("word_embeddings") grabs word_embeddings.*, while a chain like vb.pp("encoder").pp("layers").pp(i.to_string()) targets keys such as encoder.layers.0.*, exactly as shown in community tutorials porting Transformers models to Candle  ￼. As one maintainer put it, the prefix system lets you “cd” around the parameter hierarchy, giving a lightweight namespace mechanism that keeps Candle fully compatible with PyTorch naming conventions while remaining ergonomic to use.

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
- `mistralrs-core/` - Core inference engine, model implementations, pipelines
- `mistralrs-server/` - CLI binary entry point
- `mistralrs-server-core/` - HTTP server routing, OpenAI API implementation
- `mistralrs-pyo3/` - Python bindings (PyO3)
- `mistralrs/` - High-level Rust API
- `mistralrs-vision/` - Vision model support
- `mistralrs-quant/` - Quantization implementations (ISQ, GGUF, GPTQ, etc.)
- `mistralrs-paged-attn/` - PagedAttention implementation
- `mistralrs-audio/` - Audio processing
- `mistralrs-mcp/` - Model Context Protocol client
- `mistralrs-bench/` - Benchmarking tools

### Key Design Patterns

1. **Pipeline Architecture**: All models implement the `Pipeline` trait in `mistralrs-core/src/pipeline/mod.rs`. Different model types (Plain, GGUF, GGML, Vision) have their own pipeline implementations.

2. **Model Loading**: Models are loaded through `Loader` traits that handle different formats and quantizations. See `mistralrs-core/src/loader.rs`.

3. **Request Handling**: The server uses message passing with `MistralRs` struct managing a background thread pool. Requests flow through `mistralrs-core/src/engine/mod.rs`.

4. **Device Management**: Automatic and manual device mapping for multi-GPU setups handled in `mistralrs-core/src/device_map.rs`.

### Adding New Features

When adding new model architectures:
1. Implement the model in `mistralrs-core/src/models/`
2. Add pipeline support in `mistralrs-core/src/pipeline/`
3. Update model detection in `mistralrs-core/src/pipeline/normal.rs`
4. Add architecture enum variant in `mistralrs-core/src/lib.rs`
5. Update CLI args in `mistralrs-server/src/main.rs`

When adding new quantization methods:
1. Implement in `mistralrs-quant/src/`
2. Add to quantization loading logic in pipelines
3. Update documentation in `docs/QUANTIZATION.md`

### Important Files to Know

- `mistralrs-core/src/engine/mod.rs` - Main engine orchestration
- `mistralrs-core/src/pipeline/mod.rs` - Pipeline trait and common logic
- `mistralrs-server-core/src/routes.rs` - HTTP API endpoints
- `mistralrs-pyo3/src/lib.rs` - Python API entry point
- `mistralrs/examples/` - Usage examples for Rust API

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
- **Issue**: The `embedding` module in `mistralrs_core` is private and not accessible through public API
- **TODO**: Research proper way to implement embeddings using public API
- **Previous attempt**: Used internal `BertEmbeddingModel` and `BertPipeline` which are not publicly exposed

### Dependencies
Current `hanzo-engine/Cargo.toml` needs these dependencies for embeddings:
- `candle-core` (from workspace)
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
