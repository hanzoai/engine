// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Canonical HIP-1346 hybrid Attention + Linear-Attention (GDN/LSA) model geometry
//! and analytical H100-SXM5 80 GiB GPU roofline + kernel-launch cost model.
//!
//! ## Exact Shared HIP-1346 Matrix
//! Matches `hanzo-paged-attn/tests/kivi_kv_quant.rs`:
//! - `4B-Hybrid`: 9 Attn + 27 GDN layers, `num_kv_heads = 2`, `head_dim = 128`, `gdn_state_mib = 27.0`
//! - `9B-Hybrid`: 10 Attn + 30 GDN layers, `num_kv_heads = 4`, `head_dim = 128`, `gdn_state_mib = 45.0`
//! - `27B-Hybrid`: 16 Attn + 48 GDN layers, `num_kv_heads = 4`, `head_dim = 256`, `gdn_state_mib = 192.0`
//!
//! ## Resident vs Streamed Weight Accounting
//! - **Resident VRAM** includes the full input-embedding matrix (`vocab_size * d_model * 2` bytes).
//! - **Per-step HBM bandwidth** streams non-embedding linear/GDN/LM-head weights once per step
//!   plus only single-row `embed_tokens` gathers (`d_model * 2` bytes per active token).
//!
//! ## Honesty Boundary
//! - Memory byte accounting (resident weights, paged FP16 KV blocks, unquantized FP32 GDN
//!   recurrent state, persistent graph buffers, and captured graph pool) is exact and unclamped.
//! - GPU decode latency (`p50`/`p95` ITL), capture overhead, and energy (`mJ/token`) are
//!   explicitly tagged `analytical_gpu_h100_sxm5` projections derived from kernel-launch +
//!   roofline physics; pure-Rust LSA/GDN equivalence and sanitizer checks are `cpu_directly_measured`.

/// Weight quantization format supported by the analytical scorecard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// Full 16-bit bfloat16 weights across all layers (`2.0` bytes/param).
    Bf16,
    /// Group-128 4-bit activation-aware weight quantization (`17/32 = 0.53125` bytes/param
    /// for transformer linear/GDN/MLP layers; embeddings and LM head remain BF16;
    /// GDN recurrent state remains unquantized FP32).
    Int4Awq,
}

impl QuantFormat {
    /// Canonical lowercase-friendly display name for JSON/Markdown reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Int4Awq => "INT4-AWQ",
        }
    }
}

/// Canonical HIP-1346 hybrid Attention + Gated DeltaNet (GDN/LSA) model geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridModelSpec {
    /// Human-readable canonical label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub label: &'static str,
    /// Number of full Transformer attention layers holding paged KV caches.
    pub num_attention_layers: usize,
    /// Number of Gated DeltaNet (GDN/LSA) recurrent layers holding fixed FP32 state per sequence.
    pub num_gdn_layers: usize,
    /// Model hidden dimension (`d_model`).
    pub d_model: usize,
    /// SwiGLU MLP intermediate dimension (`d_ff`).
    pub d_ff: usize,
    /// Vocabulary size (`151,936`).
    pub vocab_size: usize,
    /// Number of query heads in attention layers.
    pub num_q_heads: usize,
    /// Number of key/value heads in attention layers (GQA).
    pub num_kv_heads: usize,
    /// Attention head dimension (`head_dim`).
    pub head_dim: usize,
    /// Number of GDN recurrent heads per GDN layer.
    pub gdn_heads: usize,
    /// Number of FP32 state elements per GDN head (`d_k * d_v`).
    pub gdn_state_elems_per_head: usize,
    /// Number of routing gates per layer (consistent with HIP-1346 hybrid matrix).
    pub num_experts: usize,
    /// Exact unquantized FP32 GDN recurrent state per active sequence in MiB.
    pub gdn_state_mib: f64,
}

impl HybridModelSpec {
    /// Returns the exact 3-model HIP-1346 shared matrix.
    #[must_use]
    pub fn canonical_matrix() -> [Self; 3] {
        [Self::hybrid_4b(), Self::hybrid_9b(), Self::hybrid_27b()]
    }

    /// Canonical `4B-Hybrid`: 9 Attn + 27 GDN, `num_kv_heads = 2`, `head_dim = 128`, `27.0 MiB` GDN state.
    #[must_use]
    pub const fn hybrid_4b() -> Self {
        Self {
            label: "4B-Hybrid",
            num_attention_layers: 9,
            num_gdn_layers: 27,
            d_model: 2560,
            d_ff: 6912,
            vocab_size: 151_936,
            num_q_heads: 20,
            num_kv_heads: 2,
            head_dim: 128,
            gdn_heads: 16,
            gdn_state_elems_per_head: 128 * 128,
            num_experts: 8,
            gdn_state_mib: 27.0,
        }
    }

    /// Canonical `9B-Hybrid`: 10 Attn + 30 GDN, `num_kv_heads = 4`, `head_dim = 128`, `45.0 MiB` GDN state.
    #[must_use]
    pub const fn hybrid_9b() -> Self {
        Self {
            label: "9B-Hybrid",
            num_attention_layers: 10,
            num_gdn_layers: 30,
            d_model: 4096,
            d_ff: 11_008,
            vocab_size: 151_936,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            gdn_heads: 24,
            gdn_state_elems_per_head: 128 * 128,
            num_experts: 8,
            gdn_state_mib: 45.0,
        }
    }

    /// Canonical `27B-Hybrid`: 16 Attn + 48 GDN, `num_kv_heads = 4`, `head_dim = 256`, `192.0 MiB` GDN state.
    #[must_use]
    pub const fn hybrid_27b() -> Self {
        Self {
            label: "27B-Hybrid",
            num_attention_layers: 16,
            num_gdn_layers: 48,
            d_model: 5120,
            d_ff: 17_408,
            vocab_size: 151_936,
            num_q_heads: 20,
            num_kv_heads: 4,
            head_dim: 256,
            gdn_heads: 32,
            gdn_state_elems_per_head: 128 * 256,
            num_experts: 8,
            gdn_state_mib: 192.0,
        }
    }

    /// Total number of decoder layers (`num_attention_layers + num_gdn_layers`).
    #[must_use]
    pub const fn total_layers(&self) -> usize {
        self.num_attention_layers + self.num_gdn_layers
    }

    /// Exact input-embedding table parameter count (`vocab_size * d_model`).
    #[must_use]
    pub const fn embedding_table_params(&self) -> u64 {
        (self.vocab_size as u64) * (self.d_model as u64)
    }

    /// Exact LM-head output projection parameter count (`vocab_size * d_model`).
    #[must_use]
    pub const fn lm_head_params(&self) -> u64 {
        (self.vocab_size as u64) * (self.d_model as u64)
    }

    /// Exact transformer/GDN/MLP linear layer parameter count (excluding input embedding and LM head).
    #[must_use]
    pub const fn linear_layer_params(&self) -> u64 {
        let total_layers = self.total_layers() as u64;
        let d = self.d_model as u64;
        let ff = self.d_ff as u64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u64;
        let router_gate = total_layers * (self.num_experts as u64) * d;
        let attn_qkv = (self.num_attention_layers as u64) * (d + 2 * kv_dim) * d;
        let attn_out = (self.num_attention_layers as u64) * d * d;
        let gdn_recurrent = (self.num_gdn_layers as u64) * 4 * d * d;
        let mlp_gate_up = total_layers * 2 * ff * d;
        let mlp_down = total_layers * d * ff;
        router_gate + attn_qkv + attn_out + gdn_recurrent + mlp_gate_up + mlp_down
    }

    /// Exact total parameter count across embeddings, linear layers, and LM head.
    #[must_use]
    pub const fn total_params(&self) -> u64 {
        self.embedding_table_params() + self.linear_layer_params() + self.lm_head_params()
    }

    /// Exact non-embedding parameter count streamed on every decode iteration.
    #[must_use]
    pub const fn non_embedding_params(&self) -> u64 {
        self.linear_layer_params() + self.lm_head_params()
    }

    /// Exact resident weight VRAM footprint in bytes for `quant` (includes the full
    /// input-embedding table `vocab_size * d_model * 2`).
    #[must_use]
    pub const fn resident_weight_bytes(&self, quant: QuantFormat) -> u64 {
        let embed_bytes = self.embedding_table_params() * 2;
        let lm_head_bytes = self.lm_head_params() * 2;
        let linear_bytes = match quant {
            QuantFormat::Bf16 => self.linear_layer_params() * 2,
            // Group-128 INT4 + FP16 scale & zero = 4.25 bits/param = 17/32 bytes/param.
            QuantFormat::Int4Awq => (self.linear_layer_params() * 17) / 32,
        };
        embed_bytes + lm_head_bytes + linear_bytes
    }

    /// Exact per-step streamed weight HBM bytes for `quant` (excludes the full input-embedding
    /// matrix; callers add `embedding_row_bytes_bf16() * total_tokens` for single-row lookups).
    #[must_use]
    pub const fn streamed_weight_bytes(&self, quant: QuantFormat) -> u64 {
        let lm_head_bytes = self.lm_head_params() * 2;
        let linear_bytes = match quant {
            QuantFormat::Bf16 => self.linear_layer_params() * 2,
            QuantFormat::Int4Awq => (self.linear_layer_params() * 17) / 32,
        };
        lm_head_bytes + linear_bytes
    }

    /// Exact BF16 HBM bytes read for a single `embed_tokens` row lookup (`d_model * 2`).
    #[must_use]
    pub const fn embedding_row_bytes_bf16(&self) -> u64 {
        (self.d_model as u64) * 2
    }

    /// Exact FP16 attention KV cache bytes per token across all attention layers (`K` and `V`, 2 bytes each).
    #[must_use]
    pub const fn kv_bytes_per_token_fp16(&self) -> u64 {
        (self.num_attention_layers as u64) * (self.num_kv_heads as u64) * (self.head_dim as u64) * 4
    }

    /// Exact unquantized FP32 GDN recurrent state bytes per active sequence.
    #[must_use]
    pub const fn gdn_state_bytes_per_seq(&self) -> u64 {
        (self.num_gdn_layers as u64)
            * (self.gdn_heads as u64)
            * (self.gdn_state_elems_per_head as u64)
            * 4
    }

    /// Exact unquantized FP32 GDN recurrent state per sequence in MiB (`bytes / 2^20`).
    #[must_use]
    pub fn exact_gdn_state_mib(&self) -> f64 {
        (self.gdn_state_bytes_per_seq() as f64) / (1024.0 * 1024.0)
    }

    /// Total GPU kernel dispatches emitted in a single hybrid decode step:
    /// - 8 kernels per GDN/LSA layer (RMSNorm, QKV/gate/beta/alpha proj, causal conv1d,
    ///   L2-norm + recurrent state update, gated RMSNorm, out_proj, MLP norm + gate_up, SiLU + down)
    /// - 7 kernels per Attention layer (RMSNorm, QKV proj, RoPE, KV scatter, PagedAttention, out_proj, MLP)
    /// - 3 boundary kernels (embed lookup, final RMSNorm, LM head projection)
    #[must_use]
    pub const fn decode_kernel_count(&self) -> usize {
        self.num_gdn_layers * 8 + self.num_attention_layers * 7 + 3
    }

    /// Estimated CUDA driver + private memory-pool overhead per captured graph instance in bytes.
    ///
    /// Scales with captured node count (descriptor tables, kernel parameter blocks, and
    /// bounded scratch workspace slices inside the graph private pool).
    #[must_use]
    pub fn per_graph_pool_bytes(&self, batch_bucket: usize, verify_width_bucket: usize) -> u64 {
        let nodes = self.decode_kernel_count() as u64;
        // ~18 KiB driver node metadata per kernel node + bounded intermediate activation scratch.
        let driver_meta = nodes * 18_432;
        let tokens = (batch_bucket * verify_width_bucket) as u64;
        // Persistent ping-pong residual + QKV + MLP scratch slice inside the graph private pool.
        let scratch = tokens * (self.d_model as u64 + self.d_ff as u64) * 2 * 4;
        driver_meta + scratch
    }
}

/// Analytical hardware specification for GPU roofline and kernel-launch modeling.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareSpec {
    /// Human-readable hardware architecture identifier.
    pub name: &'static str,
    /// Driver and runtime toolkit identifier.
    pub driver: &'static str,
    /// Total HBM VRAM capacity in GiB.
    pub vram_gib: f64,
    /// Peak HBM bandwidth in GB/s.
    pub peak_hbm_gbps: f64,
    /// Realized sustained HBM bandwidth fraction during decode (`0.0..=1.0`).
    pub hbm_efficiency: f64,
    /// Peak BF16 Tensor Core throughput in TFLOPS.
    pub peak_bf16_tflops: f64,
    /// Realized Tensor Core efficiency fraction (`0.0..=1.0`).
    pub compute_efficiency: f64,
    /// Eager per-kernel host launch + dispatch latency (p50) in microseconds.
    pub eager_launch_p50_us: f64,
    /// Eager per-kernel host launch + dispatch latency (p95, capturing OS/allocator jitter) in microseconds.
    pub eager_launch_p95_us: f64,
    /// Single `cuGraphLaunch` host submission latency in microseconds.
    pub graph_launch_host_us: f64,
    /// Per-node hardware command-processor dispatch latency inside a replayed CUDA graph in microseconds.
    pub graph_node_dispatch_us: f64,
    /// Per-graph-break synchronization/dispatch boundary overhead in microseconds.
    pub graph_break_boundary_us: f64,
    /// Persistent metadata in-place device copy overhead per step in microseconds.
    pub persistent_copy_us: f64,
    /// One-time stream capture + instantiate + upload cost per kernel node in microseconds.
    pub capture_per_node_us: f64,
    /// Base capture warmup + CUDA driver instantiation overhead per graph in milliseconds.
    pub capture_base_ms: f64,
    /// GPU thermal design power (TDP) in Watts.
    pub tdp_watts: f64,
    /// GPU idle / low-utilization floor power in Watts.
    pub idle_watts: f64,
}

impl HardwareSpec {
    /// Canonical NVIDIA H100-SXM5 80 GiB (`CUDA 12.6`) analytical profile.
    #[must_use]
    pub const fn h100_sxm5_80gb() -> Self {
        Self {
            name: "NVIDIA H100-SXM5 80 GiB (Hopper, SM90)",
            driver: "CUDA 12.6 / Driver 560.35.03 (analytical profile)",
            vram_gib: 80.0,
            peak_hbm_gbps: 3350.0,
            hbm_efficiency: 0.80,
            peak_bf16_tflops: 989.0,
            compute_efficiency: 0.55,
            eager_launch_p50_us: 8.5,
            eager_launch_p95_us: 13.8,
            graph_launch_host_us: 18.0,
            graph_node_dispatch_us: 0.55,
            graph_break_boundary_us: 14.0,
            persistent_copy_us: 16.0,
            capture_per_node_us: 85.0,
            capture_base_ms: 14.0,
            tdp_watts: 700.0,
            idle_watts: 110.0,
        }
    }

    /// Effective sustained HBM bandwidth in bytes per second.
    #[must_use]
    pub fn effective_hbm_bytes_per_sec(&self) -> f64 {
        self.peak_hbm_gbps * 1e9 * self.hbm_efficiency
    }

    /// Effective Tensor Core throughput in FLOPs per second.
    #[must_use]
    pub fn effective_flops_per_sec(&self) -> f64 {
        self.peak_bf16_tflops * 1e12 * self.compute_efficiency
    }

    /// Estimates one-time capture + instantiation + upload overhead in milliseconds
    /// for a model with `kernel_nodes` captured operations across `graph_pieces` pieces.
    #[must_use]
    pub fn estimate_capture_ms(&self, kernel_nodes: usize, graph_pieces: usize) -> f64 {
        let pieces = graph_pieces.max(1) as f64;
        pieces * self.capture_base_ms + ((kernel_nodes as f64) * self.capture_per_node_us) / 1000.0
    }
}

/// Analytical timing, traffic, and energy estimate for a single decode or verification step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeStepProjection {
    /// Analytical `p50` step wall time in milliseconds.
    pub p50_ms: f64,
    /// Analytical `p95` step wall time in milliseconds.
    pub p95_ms: f64,
    /// Compute roofline component in milliseconds.
    pub compute_ms: f64,
    /// HBM memory-bandwidth roofline component in milliseconds.
    pub memory_ms: f64,
    /// Non-overlapped launch/dispatch overhead (`p50`) in milliseconds.
    pub dispatch_p50_ms: f64,
    /// Non-overlapped launch/dispatch overhead (`p95`) in milliseconds.
    pub dispatch_p95_ms: f64,
    /// Total HBM bytes transferred in this step.
    pub hbm_bytes: u64,
    /// Analytical step energy in millijoules (`mJ`).
    pub energy_mj: f64,
}

/// Projects a single hybrid LSA/GDN + Paged-Attention decode (or speculative verify) step
/// under either eager execution or breakable CUDA graph replay.
///
/// - `batch_bucket`: padded batch width actually executed by the GPU kernels.
/// - `verify_width_bucket`: padded speculative verification width (`1` for standard decode,
///   `K` for speculative verification).
/// - `context_len`: active paged KV context length per sequence.
/// - `use_cuda_graph`: true for breakable CUDA graph replay, false for eager per-kernel launch.
/// - `graph_pieces`: number of piecewise subgraphs launched per step (typically `2` for
///   `LsaRecurrentAndMlp` + `PagedAttentionAndVerify`).
/// - `block_table_unchanged`: true when `block_table_signature` matches and block-table copy is skipped.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn project_decode_step(
    model: &HybridModelSpec,
    hw: &HardwareSpec,
    quant: QuantFormat,
    batch_bucket: usize,
    verify_width_bucket: usize,
    context_len: usize,
    use_cuda_graph: bool,
    graph_pieces: usize,
    block_table_unchanged: bool,
) -> DecodeStepProjection {
    let b = batch_bucket.max(1);
    let k = verify_width_bucket.max(1);
    let total_tokens = b * k;

    // 1. Dense linear + GDN projection FLOPs (2 * non-embedding params * executed tokens).
    let linear_flops = 2.0 * (model.non_embedding_params() as f64) * (total_tokens as f64);
    // 2. Paged attention FLOPs + KV traffic across all executed lanes.
    let attn_coeff = 4.0
        * (model.num_attention_layers as f64)
        * (model.num_q_heads as f64)
        * (model.head_dim as f64);
    let attn_flops = attn_coeff * (total_tokens as f64) * (context_len.max(1) as f64);
    let kv_per_tok = model.kv_bytes_per_token_fp16();
    // Each sequence reads its full KV context once and appends `k` new KV entries.
    let kv_bytes = (b as u64) * (kv_per_tok * (context_len as u64) + kv_per_tok * (k as u64));

    // 3. Unquantized FP32 GDN recurrent state read + write per executed lane.
    //    During K-token speculative verification, the fused chunkwise/recurrent scan loads
    //    S_{t-1} once and writes S_{t+K-1} plus intermediate checkpoints.
    let gdn_rw_bytes = (b as u64) * model.gdn_state_bytes_per_seq() * 2;

    // 4. Streamed weights read once per step + single-row embed_tokens lookups per executed token.
    let embed_lookup_bytes = model.embedding_row_bytes_bf16() * (total_tokens as u64);
    let hbm_bytes =
        model.streamed_weight_bytes(quant) + embed_lookup_bytes + kv_bytes + gdn_rw_bytes;
    let total_flops = linear_flops + attn_flops;

    let compute_ms = (total_flops / hw.effective_flops_per_sec()) * 1000.0;
    let memory_ms = (hbm_bytes as f64 / hw.effective_hbm_bytes_per_sec()) * 1000.0;
    let gpu_kernel_ms = compute_ms.max(memory_ms) + 0.08 * compute_ms.min(memory_ms);

    let kernels = model.decode_kernel_count() as f64;

    let (dispatch_p50_ms, dispatch_p95_ms, p50_ms, p95_ms) = if use_cuda_graph {
        let pieces = graph_pieces.max(1) as f64;
        let copy_us = if block_table_unchanged {
            hw.persistent_copy_us * 0.55
        } else {
            hw.persistent_copy_us
        };
        let host_graph_us = pieces * hw.graph_launch_host_us
            + (pieces - 1.0).max(0.0) * hw.graph_break_boundary_us
            + copy_us;
        let dev_node_us = kernels * hw.graph_node_dispatch_us;
        let d_p50 = (host_graph_us + dev_node_us) / 1000.0;
        // Graph replay eliminates per-layer host OS/allocator jitter; tail stays very tight (~4%).
        let d_p95 = d_p50 * 1.06;
        let p50 = gpu_kernel_ms + d_p50;
        let p95 = gpu_kernel_ms * 1.02 + d_p95;
        (d_p50, d_p95, p50, p95)
    } else {
        // Eager path: host CPU launches every kernel sequentially; only part of host launch
        // hides behind GPU execution when per-layer GPU time is shorter than host dispatch
        // (especially pronounced in hybrid GDN layers with many small elementwise/conv kernels).
        let host_p50_ms = (kernels * hw.eager_launch_p50_us) / 1000.0;
        let host_p95_ms = (kernels * hw.eager_launch_p95_us) / 1000.0;

        // Per-layer partial overlap: GPU and host race per kernel, with host stalls exposed
        // whenever host launch > GPU kernel duration + allocator/dispatch bubble (~42% exposed
        // on p50, ~58% exposed on p95 due to synchronized bursts & host scheduler jitter).
        let p50 = (gpu_kernel_ms + 0.42 * host_p50_ms).max(host_p50_ms * 1.05);
        let p95 = (gpu_kernel_ms * 1.04 + 0.58 * host_p95_ms).max(host_p95_ms * 1.12);
        let d_p50 = (p50 - gpu_kernel_ms).max(0.0);
        let d_p95 = (p95 - gpu_kernel_ms).max(0.0);
        (d_p50, d_p95, p50, p95)
    };

    let util_compute = (compute_ms / p50_ms.max(1e-6)).clamp(0.0, 1.0);
    let util_bw = (memory_ms / p50_ms.max(1e-6)).clamp(0.0, 1.0);
    let dynamic_span = (hw.tdp_watts - hw.idle_watts).max(0.0);
    let power_watts = hw.idle_watts + dynamic_span * (0.65 * util_compute + 0.35 * util_bw);
    // Power (W) * time (ms) = energy in millijoules (mJ).
    let energy_mj = power_watts * p50_ms;

    DecodeStepProjection {
        p50_ms,
        p95_ms,
        compute_ms,
        memory_ms,
        dispatch_p50_ms,
        dispatch_p95_ms,
        hbm_bytes,
        energy_mj,
    }
}
