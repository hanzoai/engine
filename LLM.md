# Hanzo Engine - LLM Inference Integration

This file provides guidance to AI assistants working with the Hanzo Engine codebase.

## Project Overview

**Hanzo Engine** is Hanzo AI's high-performance LLM inference engine, built on top of [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - a blazing-fast LLM inference engine written in Rust.

### Integration Status

- **Upstream Sync**: Fully synchronized with mistral.rs at commit `530463af1` (Implement Qwen 3 VL! #1657)
- **Last Sync Date**: 2025-10-26
- **Upstream Repository**: https://github.com/EricLBuehler/mistral.rs
- **Remote**: Configured as `upstream` in git

### Hanzo-Specific Components

This repository extends mistral.rs with:

1. **hanzo-engine/** - Custom CLI tool and server with Hanzo-specific features:
   - `serve` - Start inference server on port 36900
   - `pull` - Download models from HuggingFace, Ollama, MLX, or URLs
   - `list` - Manage downloaded models
   - `chat` - Interactive chat interface
   - `embed` - Generate embeddings (WIP - needs reimplementation)

2. **Features**:
   - OpenAI-compatible API endpoints
   - Ollama compatibility mode
   - Model management and caching
   - Custom model directory support

### Architecture

Hanzo Engine is a Rust workspace containing:
- All standard mistral.rs workspace members
- **hanzo-engine/** (NEW) - Custom inference server and CLI

mistral.rs provides comprehensive LLM inference with support for text, vision, image generation, and speech models through multiple APIs (Rust, Python, OpenAI HTTP, MCP).

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

### Building Upstream mistral.rs Components

```bash
# Basic release build
cargo build --release

# With CUDA support (Linux)
cargo build --release --features "cuda flash-attn cudnn"

# With Metal support (macOS)
cargo build --release --features metal

# Install upstream mistralrs-server binary
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

#### Upstream mistral.rs Components
- `mistralrs-core/` - Core inference engine, model implementations, pipelines
- `mistralrs-server/` - CLI binary entry point (upstream)
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
- **TODO**: Research proper way to implement embeddings using public mistralrs API
- **Previous attempt**: Used internal `BertEmbeddingModel` and `BertPipeline` which are not publicly exposed

### Dependencies
Current `hanzo-engine/Cargo.toml` needs these dependencies for embeddings:
- `candle-core` (from workspace)
- `tokenizers` (from workspace)
- May need to re-export or use different approach

## Syncing with Upstream

To pull latest changes from upstream mistral.rs:

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

## GRPO (Group Relative Policy Optimization) and Experience-Based Learning Analysis

### Executive Summary

The Hanzo Engine (mistral.rs-based) is **currently a pure inference engine** with no integrated reinforcement learning, GRPO, or experience-based training capabilities. The architecture is focused on:
- **High-performance inference** across multiple model types (text, vision, diffusion, audio, speech)
- **Batch request scheduling** and multiplexing
- **Token-by-token generation** with sampling
- **Stateless pipelines** - no persistent state across requests

**Current capabilities are NOT suitable for GRPO** which requires:
- Multi-step trajectory collection
- Experience/rollout storage and replay
- Reward computation and tracking
- Policy gradient optimization
- Online learning or iterative refinement

### Current Architecture Analysis

#### 1. Request/Response Flow Architecture

**Entry Point**: `mistralrs-core/src/request.rs`
```rust
pub struct NormalRequest {
    pub messages: RequestMessage,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,  // One-shot response channel
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    pub constraint: Constraint,
    pub suffix: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    // ... logits_processors, custom behaviors
}
```

**Request Lifecycle**:
1. Requests enter scheduler (`mistralrs-core/src/scheduler/default_scheduler.rs`)
2. Scheduler batches sequences for efficiency
3. Pipeline executes `step()` function per batch
4. Responses sent via one-shot channels to clients
5. **No persistent state** between requests

#### 2. Sequence Management

**File**: `mistralrs-core/src/sequence.rs`

Sequences represent individual generation continuations:
```rust
pub enum SequenceState {
    Done(StopReason),
    RunningPrompt,
    RunningCompletion,
    Waiting,
    Error,
}

pub enum StopReason {
    Eos,
    StopTok(u32),
    Length(usize),
    ModelLength(usize),
    StopString { ... },
    Canceled,
    GeneratedImage,
    GeneratedSpeech,
    ToolCalls,
}
```

**Sequence Tracking**:
- Token history stored inline
- Cache management (KV cache) per sequence
- Sampling logs optional (logprobs)
- **No trajectory accumulation** - state cleared on completion

#### 3. Sampling and Generation

**File**: `mistralrs-core/src/sampler.rs`

```rust
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub top_n_logprobs: usize,        // Optional logprobs return
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub dry_params: Option<DrySamplingParams>,
}
```

**Generation Step** (`mistralrs-core/src/pipeline/mod.rs:383-500`):
```rust
async fn step(
    &mut self,
    input_seqs: &mut [&mut Sequence],
    is_prompt: bool,                  // Prompt vs completion
    return_raw_logits: bool,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    backend_metadata: CacheBackendMetadata,
) -> Result<Duration, candle_core::Error>
```

Process:
1. Input processor converts messages/tokens to embeddings
2. Cache management (KV cache clone in/out)
3. Forward pass through model
4. Sampling applied to logits
5. Token appended to sequence
6. Response sent immediately

#### 4. Pipeline Trait

**File**: `mistralrs-core/src/pipeline/mod.rs:365-392`

The core abstraction - all models implement `Pipeline`:
```rust
pub trait Pipeline:
    Send + Sync
    + PreProcessingMixin
    + IsqPipelineMixin
    + CacheManagerMixin
    + MetadataMixin
    + AnyMoePipelineMixin
{
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;
    
    async fn step(...) -> Result<Duration>;
}
```

**Implementations**:
- `NormalPipeline` - Standard text models
- `VisionPipeline` - Vision + text models
- `SpeculativePipeline` - Speculative decoding
- `DiffusionPipeline` - Image generation
- `SpeechPipeline` - Audio generation

### 2. Experience Storage Systems

**FINDING: NONE PRESENT**

There are NO experience storage, rollout collection, or trajectory management systems in the codebase:
- No experience buffer/replay implementation
- No trajectory recording structures
- No episode/rollout types
- No persistent state machine for multi-turn learning

**What exists**:
- Per-sequence token history (cleared on completion)
- Optional logprobs tracking (for debugging, not learning)
- Sampling parameters (but not results/decisions stored)

### 3. Rollout Orchestration

**FINDING: SCHEDULER-BASED BATCHING ONLY**

The scheduler (`mistralrs-core/src/scheduler/default_scheduler.rs`) handles:
- **Bucketing** sequences by length for efficient batch packing
- **Priority scheduling** (FCFS with bucketing)
- **Waiting queue management** when GPU memory constrained
- **Single-pass generation** - no rollout collection

**Does NOT support**:
- Parallel trajectory collection
- Multi-step lookahead
- Rollout storage and retrieval
- Experience prioritization

### 4. Trajectory Analysis and Critique

**FINDING: NONE PRESENT**

No systems for:
- Trajectory comparison/scoring
- Reward computation
- Action value estimation
- Trajectory clustering or analysis
- Critique/feedback systems

**What exists**:
- Tool calling support (`mistralrs-core/src/tools/`)
- Constraint-guided generation (llguidance)
- Grammar-based sampling
- But NO learning from results

### 5. RL/Policy Optimization References

**FINDING: ZERO RL INFRASTRUCTURE**

Grep of entire codebase for RL-related terms:
- ❌ No "policy", "gradient", "reward", "value", "loss"
- ❌ No "experience", "trajectory", "episode", "rollout"
- ❌ No "optimization", "training" in RL context
- ✓ Only `amoe_pre_train()` in `mistralrs-core/src/pipeline/amoe.rs` for MoE gate training (not RL)

---

## Integration Architecture for GRPO

### Recommended Approach

#### Phase 1: Experience Collection Module (Standalone)

Create new module: `mistralrs-core/src/learning/`

```
mistralrs-core/src/learning/
├── mod.rs                      # Core traits
├── experience.rs               # Experience storage
├── trajectory.rs               # Trajectory types
├── rollout.rs                  # Rollout orchestration
├── reward.rs                   # Reward functions
└── replay.rs                   # Replay buffer
```

**File**: `mistralrs-core/src/learning/trajectory.rs`
```rust
pub struct Trajectory {
    pub id: String,
    pub model_id: String,
    pub tokens: Vec<u32>,           // Complete token sequence
    pub logprobs: Vec<f32>,         // Log prob at each step
    pub logits: Vec<Arc<Tensor>>,   // Optional: save logits
    pub actions: Vec<u32>,          // Sampled token actions
    pub values: Vec<f32>,           // Estimated values (if critic)
    pub rewards: Vec<f32>,          // Per-step or final reward
    pub dones: Vec<bool>,           // Step termination flags
    pub metadata: TrajectoryMetadata,
}

pub struct TrajectoryMetadata {
    pub prompt: String,
    pub timestamp: SystemTime,
    pub model_name: String,
    pub sampling_params: SamplingParams,
    pub terminal_reward: f32,       // Final reward
    pub rollout_length: usize,
}
```

#### Phase 2: Rollout Collector Integration

Extend `mistralrs-core/src/engine/mod.rs`:

```rust
pub struct RolloutConfig {
    pub collect_rollouts: bool,
    pub collect_logits: bool,
    pub store_experiences: bool,
    pub experience_dir: PathBuf,
    pub max_trajectory_length: usize,
}

pub trait ExperienceCollector: Send + Sync {
    async fn record_trajectory(&self, trajectory: Trajectory) -> Result<()>;
    async fn get_batch(&self, batch_size: usize) -> Result<Vec<Trajectory>>;
}
```

**Integration Point in Engine**:
```rust
// In MistralRs struct (mistralrs-core/src/engine/mod.rs:~150)
pub struct MistralRs {
    // ... existing fields ...
    experience_collector: Option<Arc<dyn ExperienceCollector>>,
    rollout_config: Option<RolloutConfig>,
}

// Modify sequence completion to optionally collect trajectory
async fn complete_sequence(&self, seq: Sequence, reason: StopReason) {
    // Send response as usual
    // THEN if experience collection enabled:
    if let Some(collector) = &self.experience_collector {
        let trajectory = self.sequence_to_trajectory(seq);
        collector.record_trajectory(trajectory).await?;
    }
}
```

#### Phase 3: Sampling Extensions

Extend sampling to record decisions:

**File**: `mistralrs-core/src/sampler.rs`

```rust
#[derive(Clone)]
pub struct DecisionRecord {
    pub token_id: u32,
    pub probability: f32,
    pub top_k_probs: Vec<(u32, f32)>,
    pub temperature_used: f32,
    pub penalty_applied: bool,
}

impl Sampler {
    pub fn sample_with_record(
        &mut self,
        logits: &Tensor,
        temperature: Option<f64>,
        record: bool,
    ) -> Result<(u32, Option<DecisionRecord>)> {
        // Existing sampling logic
        let token = self.sample(logits, temperature)?;
        
        if record {
            let record = DecisionRecord { /* ... */ };
            Ok((token, Some(record)))
        } else {
            Ok((token, None))
        }
    }
}
```

#### Phase 4: Policy Optimization Module

Create: `mistralrs-core/src/learning/grpo.rs`

```rust
pub struct GRPOOptimizer {
    model: Arc<Mutex<dyn Pipeline>>,
    critic: Option<Arc<Mutex<dyn CriticModel>>>,
    config: GRPOConfig,
}

pub struct GRPOConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub gamma: f32,              // Discount factor
    pub beta: f32,               // KL divergence weight
    pub group_size: usize,       // Group relative size
}

impl GRPOOptimizer {
    pub async fn optimize_batch(
        &mut self,
        trajectories: Vec<Trajectory>,
    ) -> Result<TrainingMetrics> {
        // 1. Group trajectories by prompt
        let groups = self.group_trajectories(trajectories);
        
        // 2. Compute group relative rewards
        for group in &groups {
            self.compute_group_rewards(group)?;
        }
        
        // 3. Compute policy gradients
        let gradients = self.compute_policy_gradients(&groups)?;
        
        // 4. Update model parameters
        self.apply_gradients(gradients).await?;
        
        Ok(TrainingMetrics { /* ... */ })
    }
    
    fn group_trajectories(&self, trajs: Vec<Trajectory>) -> Vec<Vec<Trajectory>> {
        let mut groups: HashMap<String, Vec<Trajectory>> = HashMap::new();
        for traj in trajs {
            let key = traj.metadata.prompt.clone();
            groups.entry(key).or_default().push(traj);
        }
        groups.into_values().collect()
    }
    
    fn compute_group_rewards(&self, group: &[Trajectory]) -> Result<Vec<f32>> {
        // Group relative reward formula from GRPO paper
        let rewards: Vec<f32> = group.iter()
            .map(|t| t.metadata.terminal_reward)
            .collect();
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        Ok(rewards.iter().map(|r| r - mean).collect())
    }
}

pub trait CriticModel: Send + Sync {
    async fn estimate_value(
        &self,
        tokens: &[u32],
    ) -> Result<f32>;
    
    async fn update(
        &mut self,
        trajectories: &[Trajectory],
    ) -> Result<()>;
}
```

#### Phase 5: Request/Response Extensions

Extend `mistralrs-core/src/request.rs`:

```rust
pub struct NormalRequest {
    // ... existing fields ...
    pub collect_trajectory: bool,
    pub reward_signal: Option<RewardFunction>,
}

pub enum RewardFunction {
    Custom(Arc<dyn Fn(&str) -> f32 + Send + Sync>),
    LengthBased { 
        min_len: usize, 
        max_len: usize 
    },
    Callback(String),  // Callback endpoint for reward
}
```

### Integration Points Summary

| Component | File | Modification | Effort |
|-----------|------|--------------|--------|
| Experience Collection | `learning/experience.rs` | NEW | Med |
| Trajectory Types | `learning/trajectory.rs` | NEW | Low |
| Rollout Config | `engine/mod.rs` | Extension | Low |
| Sequence→Trajectory | `sequence.rs` | Extension | Med |
| Sampling Records | `sampler.rs` | Extension | Low |
| GRPO Optimizer | `learning/grpo.rs` | NEW | High |
| Request Types | `request.rs` | Extension | Low |
| Critic Model | `learning/critic.rs` | NEW | High |
| Server Routes | `mistralrs-server-core/src/routes.rs` | Extension | Med |

### Critical Design Decisions

1. **Stateless Inference Philosophy Conflict**
   - Current design: Pure request→response with no persistent state
   - GRPO needs: Persistent trajectory storage and optimizer state
   - **Solution**: Optional experience collector that operates asynchronously, doesn't block inference

2. **Memory Overhead**
   - Storing full logits per trajectory is expensive
   - **Solution**: Make logit storage optional, default to logprobs only

3. **Backward Compatibility**
   - Existing code should work unchanged
   - **Solution**: All GRPO features behind feature flags and optional components

4. **Batching Complexity**
   - Current scheduler batches by length for inference efficiency
   - GRPO optimization needs grouped trajectories
   - **Solution**: Separate "learning scheduler" for optimization passes

### Feature Flags for GRPO Integration

```toml
[features]
default = []
# ... existing features ...
grpo = ["learning", "reward"]
learning = []
reward = []
```

---

## Implementation Roadmap

### Step 1: Minimal Viable GRPO (Week 1-2)
- [ ] Trajectory structure and serialization
- [ ] File-based experience storage (SQLite or JSON Lines)
- [ ] Collection hooks in sequence completion
- [ ] Basic GRPO optimizer with group relative rewards

### Step 2: Efficient Batching (Week 2-3)
- [ ] Experience replay buffer with prioritization
- [ ] Batch sampling for optimization
- [ ] Streaming trajectory collection

### Step 3: Critic Model (Week 3-4)
- [ ] Value function training
- [ ] Advantage estimation
- [ ] GAE (Generalized Advantage Estimation)

### Step 4: Full Pipeline Integration (Week 4-5)
- [ ] Async optimizer background task
- [ ] Reward computation frameworks
- [ ] Training metrics and logging
- [ ] Checkpoint management

### Step 5: Production Hardening (Week 5-6)
- [ ] Distributed trajectory collection
- [ ] Model serving during optimization
- [ ] Failure recovery
- [ ] Comprehensive testing

---

## Key Files to Modify

### Core Changes Required

1. **`mistralrs-core/src/lib.rs`** - Add public learning module
   ```rust
   #[cfg(feature = "grpo")]
   pub mod learning;
   ```

2. **`mistralrs-core/src/engine/mod.rs`** - Add experience collector
   ```rust
   pub experience_collector: Option<Arc<dyn ExperienceCollector>>,
   
   // In MistralRs::complete_sequence()
   if let Some(collector) = &self.experience_collector {
       // Record trajectory
   }
   ```

3. **`mistralrs-core/src/sequence.rs`** - Add trajectory conversion
   ```rust
   pub fn to_trajectory(self) -> Trajectory { /* ... */ }
   ```

4. **`mistralrs-core/src/sampler.rs`** - Track decisions
   ```rust
   pub fn sample_with_record(...) -> (Token, Option<DecisionRecord>)
   ```

### New Files to Create

- `mistralrs-core/src/learning/mod.rs`
- `mistralrs-core/src/learning/experience.rs`
- `mistralrs-core/src/learning/trajectory.rs`
- `mistralrs-core/src/learning/grpo.rs`
- `mistralrs-core/src/learning/critic.rs`
- `mistralrs-core/src/learning/reward.rs`

---

## Current Gaps vs GRPO Requirements

| GRPO Requirement | Current Status | Gap |
|------------------|----------------|-----|
| Trajectory collection | No infrastructure | Must build experience collector |
| Rollout storage | No persistence | Must add trajectory DB/storage |
| Reward computation | No framework | Must implement reward signals |
| Group relative rewards | Not applicable | Must compute groups, normalize |
| Value estimation | No critic model | Must build value network |
| Policy gradients | Sampling only | Must implement PG computation |
| KL divergence tracking | Not tracked | Must track policy changes |
| Experience replay | No buffer | Must implement replay logic |
| Online learning loop | No loop | Must coordinate collection + optimization |
| Distributed scaling | Not applicable | Consider for production |

