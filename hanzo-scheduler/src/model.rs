// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact HIP-1346 shared `4B / 9B / 27B` hybrid Transformer + GDN model matrix
//! and roofline analytical GPU step cost model.
//!
//! ## Exact Alignment with `hanzo-paged-attn/tests/kivi_kv_quant.rs`
//! - `4B-Hybrid`: 9 Attn + 27 GDN layers, `num_kv_heads = 2`, `head_dim = 128`,
//!   `gdn_heads = 16`, `gdn_state_elems_per_head = 128 * 128`, `gdn_state_mib = 27.0`
//! - `9B-Hybrid`: 10 Attn + 30 GDN layers, `num_kv_heads = 4`, `head_dim = 128`,
//!   `gdn_heads = 24`, `gdn_state_elems_per_head = 128 * 128`, `gdn_state_mib = 45.0`
//! - `27B-Hybrid`: 16 Attn + 48 GDN layers, `num_kv_heads = 4`, `head_dim = 256`,
//!   `gdn_heads = 32`, `gdn_state_elems_per_head = 128 * 256`, `gdn_state_mib = 192.0`
//!
//! ## Honesty Boundary
//! - Memory byte accounting (weights, paged FP16 KV blocks, unquantized FP32 GDN recurrent state)
//!   is exact and never clamped.
//! - GPU iteration latency, throughput, and energy (`mJ/token`) are **explicitly tagged
//!   analytical H100-SXM5 80 GiB roofline projections**; CPU scheduler loop wall time and
//!   token-level causal verification are **directly measured on the host CPU**.

/// Canonical HIP-1346 hybrid Attention + Gated DeltaNet (GDN) model geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridModelSpec {
    /// Human-readable canonical label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub label: &'static str,
    /// Number of full Transformer attention layers holding paged KV caches.
    pub num_attention_layers: usize,
    /// Number of Gated DeltaNet (GDN) recurrent layers holding fixed FP32 state per active sequence.
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
    /// Number of MoE/routing gates per layer (consistent with HIP-1346 hybrid matrix).
    pub num_experts: usize,
    /// Exact unquantized FP32 GDN recurrent state per active sequence in MiB.
    pub gdn_state_mib: f64,
}

impl HybridModelSpec {
    /// Returns the exact 3-model HIP-1346 shared matrix matching
    /// `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`.
    #[must_use]
    pub fn canonical_matrix() -> [Self; 3] {
        [Self::hybrid_4b(), Self::hybrid_9b(), Self::hybrid_27b()]
    }

    /// Canonical `4B-Hybrid` geometry: 9 Attn + 27 GDN, `num_kv_heads = 2`, `head_dim = 128`, `27.0 MiB` GDN state.
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

    /// Canonical `9B-Hybrid` geometry: 10 Attn + 30 GDN, `num_kv_heads = 4`, `head_dim = 128`, `45.0 MiB` GDN state.
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

    /// Canonical `27B-Hybrid` geometry: 16 Attn + 48 GDN, `num_kv_heads = 4`, `head_dim = 256`, `192.0 MiB` GDN state.
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

    /// Exact total parameter count derived from the architectural geometry.
    #[must_use]
    pub const fn total_params(&self) -> u64 {
        let total_layers = (self.num_attention_layers + self.num_gdn_layers) as u64;
        let d = self.d_model as u64;
        let ff = self.d_ff as u64;
        let vocab = self.vocab_size as u64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u64;
        let embedding = vocab * d;
        let lm_head = vocab * d;
        let router_gate = total_layers * (self.num_experts as u64) * d;
        let attn_qkv = (self.num_attention_layers as u64) * (d + 2 * kv_dim) * d;
        let attn_out = (self.num_attention_layers as u64) * d * d;
        let gdn_recurrent = (self.num_gdn_layers as u64) * 4 * d * d;
        let mlp_gate_up = total_layers * 2 * ff * d;
        let mlp_down = total_layers * d * ff;
        embedding
            + lm_head
            + router_gate
            + attn_qkv
            + attn_out
            + gdn_recurrent
            + mlp_gate_up
            + mlp_down
    }

    /// Exact input-embedding table parameter count (`vocab_size * d_model`).
    #[must_use]
    pub const fn embedding_table_params(&self) -> u64 {
        (self.vocab_size as u64) * (self.d_model as u64)
    }

    /// Exact non-embedding linear/projection parameter count streamed on every forward step.
    #[must_use]
    pub const fn non_embedding_params(&self) -> u64 {
        self.total_params() - self.embedding_table_params()
    }

    /// Exact BF16 resident model weight storage in bytes (`2` bytes per parameter, including
    /// the full input-embedding table for VRAM-capacity accounting).
    #[must_use]
    pub const fn weight_bytes_bf16(&self) -> u64 {
        self.total_params() * 2
    }

    /// Exact BF16 input-embedding table storage in bytes (`vocab_size * d_model * 2`).
    #[must_use]
    pub const fn embedding_table_bytes_bf16(&self) -> u64 {
        self.embedding_table_params() * 2
    }

    /// Exact BF16 streamed linear/GDN/LM-head weight traffic per forward iteration
    /// (excludes the full input-embedding matrix, which is accessed via row gather).
    #[must_use]
    pub const fn streamed_weight_bytes_bf16(&self) -> u64 {
        self.non_embedding_params() * 2
    }

    /// Exact BF16 HBM bytes read for one `embed_tokens` row lookup (`d_model * 2`).
    #[must_use]
    pub const fn embedding_row_bytes_bf16(&self) -> u64 {
        (self.d_model as u64) * 2
    }

    /// Exact FP16 attention KV cache bytes per token across all attention layers (`K` and `V`, 2 bytes each).
    #[must_use]
    pub const fn kv_bytes_per_token_fp16(&self) -> u64 {
        (self.num_attention_layers as u64) * (self.num_kv_heads as u64) * (self.head_dim as u64) * 4
    }

    /// Exact FP16 attention KV cache bytes per paged block of `block_size` tokens.
    #[must_use]
    pub const fn kv_bytes_per_block(&self, block_size: usize) -> u64 {
        self.kv_bytes_per_token_fp16() * (block_size as u64)
    }

    /// Exact unquantized FP32 GDN recurrent state bytes per active sequence.
    ///
    /// Matches `num_gdn_layers * gdn_heads * gdn_state_elems_per_head * 4`.
    #[must_use]
    pub const fn gdn_state_bytes_per_seq(&self) -> u64 {
        (self.num_gdn_layers as u64)
            * (self.gdn_heads as u64)
            * (self.gdn_state_elems_per_head as u64)
            * 4
    }

    /// Exact GDN state per active sequence in MiB (`bytes / 2^20`).
    #[must_use]
    pub fn exact_gdn_state_mib(&self) -> f64 {
        (self.gdn_state_bytes_per_seq() as f64) / (1024.0 * 1024.0)
    }

    /// Exact per-sequence KV + GDN memory footprint in bytes for `seq_len` tokens at `block_size`.
    #[must_use]
    pub const fn sequence_state_bytes(&self, seq_len: usize, block_size: usize) -> u64 {
        let blocks = seq_len.div_ceil(block_size);
        (blocks as u64) * self.kv_bytes_per_block(block_size) + self.gdn_state_bytes_per_seq()
    }
}

/// Hardware accelerator envelope for explicitly tagged analytical GPU projections.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareSpec {
    /// Accelerator identifier (e.g. `"H100-SXM5-80GB-Analytical"`).
    pub name: &'static str,
    /// Total HBM capacity in bytes.
    pub vram_bytes: u64,
    /// Fixed runtime activation / CUDA graph workspace bytes.
    pub workspace_bytes: u64,
    /// Peak BF16 Tensor Core throughput in TFLOP/s.
    pub peak_bf16_tflops: f64,
    /// Realizable GEMM/attention compute efficiency factor (`0.0..=1.0`).
    pub compute_efficiency: f64,
    /// Peak HBM bandwidth in GB/s.
    pub hbm_bandwidth_gbps: f64,
    /// Realizable HBM bandwidth efficiency factor (`0.0..=1.0`).
    pub bandwidth_efficiency: f64,
    /// PCIe host-to-device / device-to-host swap bandwidth in GB/s.
    pub pcie_swap_gbps: f64,
    /// Base per-iteration kernel dispatch + synchronization overhead in microseconds.
    pub kernel_launch_overhead_us: f64,
    /// Additional metadata/packing overhead for ragged mixed prefill+decode steps in microseconds.
    pub mixed_batch_overhead_us: f64,
    /// Idle power draw in Watts.
    pub idle_watts: f64,
    /// Thermal design power (TDP) in Watts.
    pub tdp_watts: f64,
}

impl HardwareSpec {
    /// Analytical NVIDIA H100-SXM5 80 GiB envelope (aligned with HIP-1346 dossiers).
    #[must_use]
    pub const fn h100_sxm5_80gb() -> Self {
        Self {
            name: "H100-SXM5-80GB-Analytical",
            vram_bytes: 80 * 1024 * 1024 * 1024,
            workspace_bytes: 2 * 1024 * 1024 * 1024,
            peak_bf16_tflops: 989.0,
            compute_efficiency: 0.55,
            hbm_bandwidth_gbps: 3350.0,
            bandwidth_efficiency: 0.78,
            pcie_swap_gbps: 48.0,
            kernel_launch_overhead_us: 180.0,
            mixed_batch_overhead_us: 45.0,
            idle_watts: 120.0,
            tdp_watts: 700.0,
        }
    }

    /// Available VRAM bytes remaining for the paged KV block pool and GDN recurrent state pool.
    #[must_use]
    pub const fn available_kv_and_gdn_bytes(&self, model: &HybridModelSpec) -> u64 {
        let reserved = model.weight_bytes_bf16() + self.workspace_bytes;
        self.vram_bytes.saturating_sub(reserved)
    }

    /// Effective sustained BF16 compute throughput in FLOP/s.
    #[must_use]
    pub fn effective_flops_per_sec(&self) -> f64 {
        self.peak_bf16_tflops * 1e12 * self.compute_efficiency
    }

    /// Effective sustained HBM bandwidth in bytes/s.
    #[must_use]
    pub fn effective_hbm_bytes_per_sec(&self) -> f64 {
        self.hbm_bandwidth_gbps * 1e9 * self.bandwidth_efficiency
    }
}

/// Summary of one scheduled prefill chunk passed to the roofline step cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunkCostInput {
    /// Number of already-prefilled prefix tokens prior to this chunk.
    pub prefix_len: usize,
    /// Number of newly processed tokens in this chunk.
    pub chunk_len: usize,
}

/// Analytical roofline timing, byte traffic, and energy output for one forward iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCostEstimate {
    /// Analytical step wall time in milliseconds.
    pub step_ms: f64,
    /// Compute-bound component in milliseconds.
    pub compute_ms: f64,
    /// HBM-bandwidth-bound component in milliseconds.
    pub memory_ms: f64,
    /// Kernel launch and batch-packing overhead in milliseconds.
    pub overhead_ms: f64,
    /// Total HBM bytes read and written during this iteration.
    pub hbm_bytes: u64,
    /// Total floating-point operations executed during this iteration.
    pub total_flops: f64,
    /// Analytical step energy in Joules.
    pub energy_joules: f64,
    /// True if the step is compute-bound (`compute_ms >= memory_ms`).
    pub compute_bound: bool,
}

/// First-principles roofline cost model for fused prefill + decode iterations.
///
/// Accounts for BOTH:
/// - Compute-bound dense GEMM + causal FlashAttention FLOPs (`2 * P_active * T + attn_flops`), and
/// - Memory-bandwidth-bound HBM traffic (shared model weight read once per iteration,
///   attention KV reads/writes for prefix + decode contexts, and fixed FP32 GDN state
///   read+write per active sequence in the step).
#[derive(Debug, Clone)]
pub struct StepCostModel {
    /// Hybrid model specification.
    pub model: HybridModelSpec,
    /// Analytical hardware specification.
    pub hardware: HardwareSpec,
}

impl StepCostModel {
    /// Creates a new analytical roofline step cost model.
    #[must_use]
    pub const fn new(model: HybridModelSpec, hardware: HardwareSpec) -> Self {
        Self { model, hardware }
    }

    /// Estimates wall time, HBM traffic, FLOPs, and energy for a single forward iteration.
    ///
    /// - `prefill_chunks`: active prefill chunks processed in this iteration.
    /// - `decode_context_lens`: current KV context length for each decoding sequence (1 token each).
    /// - `single_user_fast_path`: when true, omits ragged mixed-batch packing overhead.
    #[must_use]
    pub fn estimate_step(
        &self,
        prefill_chunks: &[PrefillChunkCostInput],
        decode_context_lens: &[usize],
        single_user_fast_path: bool,
    ) -> StepCostEstimate {
        let prefill_tokens: usize = prefill_chunks.iter().map(|c| c.chunk_len).sum();
        let decode_tokens = decode_context_lens.len();
        let total_tokens = prefill_tokens + decode_tokens;
        if total_tokens == 0 {
            return StepCostEstimate {
                step_ms: 0.0,
                compute_ms: 0.0,
                memory_ms: 0.0,
                overhead_ms: 0.0,
                hbm_bytes: 0,
                total_flops: 0.0,
                energy_joules: 0.0,
                compute_bound: false,
            };
        }

        // 1. Dense linear + GDN projection FLOPs (2 * non-embedding active params * tokens).
        let linear_flops = 2.0 * (self.model.non_embedding_params() as f64) * (total_tokens as f64);

        // 2. Causal attention FLOPs + KV HBM traffic.
        let attn_coeff = 4.0
            * (self.model.num_attention_layers as f64)
            * (self.model.num_q_heads as f64)
            * (self.model.head_dim as f64);
        let kv_per_tok = self.model.kv_bytes_per_token_fp16();
        let gdn_per_seq = self.model.gdn_state_bytes_per_seq();

        let mut attn_flops = 0.0f64;
        let mut kv_bytes: u64 = 0;

        for chunk in prefill_chunks {
            let c = chunk.chunk_len as f64;
            let mean_ctx = (chunk.prefix_len as f64) + 0.5 * c;
            attn_flops += attn_coeff * c * mean_ctx;
            // Read cached prefix KV (if chunk follows prior chunks) + write new chunk KV.
            kv_bytes += kv_per_tok * (chunk.prefix_len as u64);
            kv_bytes += kv_per_tok * (chunk.chunk_len as u64);
        }

        for &ctx_len in decode_context_lens {
            attn_flops += attn_coeff * (ctx_len.max(1) as f64);
            kv_bytes += kv_per_tok * (ctx_len as u64);
            kv_bytes += kv_per_tok;
        }

        // 3. Fixed FP32 GDN recurrent state read + write once per active sequence in the step.
        let active_seqs = (prefill_chunks.len() + decode_context_lens.len()) as u64;
        let gdn_rw_bytes = active_seqs * gdn_per_seq * 2;

        // 4. Streamed linear/GDN/LM-head weights are loaded once per fused iteration;
        //    input embeddings are gathered per token (one `embed_tokens` row per token).
        let embed_lookup_bytes = self.model.embedding_row_bytes_bf16() * (total_tokens as u64);
        let hbm_bytes =
            self.model.streamed_weight_bytes_bf16() + embed_lookup_bytes + kv_bytes + gdn_rw_bytes;
        let total_flops = linear_flops + attn_flops;

        let compute_ms = (total_flops / self.hardware.effective_flops_per_sec()) * 1000.0;
        let memory_ms = (hbm_bytes as f64 / self.hardware.effective_hbm_bytes_per_sec()) * 1000.0;
        let is_mixed_or_chunked = !single_user_fast_path
            && ((!prefill_chunks.is_empty() && !decode_context_lens.is_empty())
                || prefill_chunks.iter().any(|c| c.prefix_len > 0));
        let overhead_us = self.hardware.kernel_launch_overhead_us
            + if is_mixed_or_chunked {
                self.hardware.mixed_batch_overhead_us
            } else {
                0.0
            };
        let overhead_ms = overhead_us / 1000.0;

        // Smooth roofline: dominant term + 10% non-overlapped secondary tail + dispatch overhead.
        let step_ms = compute_ms.max(memory_ms) + 0.10 * compute_ms.min(memory_ms) + overhead_ms;

        // Dynamic GPU power scales with realized compute and HBM utilization.
        let util_compute = (compute_ms / step_ms).clamp(0.0, 1.0);
        let util_bw = (memory_ms / step_ms).clamp(0.0, 1.0);
        let dynamic_span = (self.hardware.tdp_watts - self.hardware.idle_watts).max(0.0);
        let power_watts =
            self.hardware.idle_watts + dynamic_span * (0.65 * util_compute + 0.35 * util_bw);
        let energy_joules = power_watts * (step_ms / 1000.0);

        StepCostEstimate {
            step_ms,
            compute_ms,
            memory_ms,
            overhead_ms,
            hbm_bytes,
            total_flops,
            energy_joules,
            compute_bound: compute_ms >= memory_ms,
        }
    }

    /// Estimates host-device PCIe transfer stall time in milliseconds for swapping `bytes`.
    #[must_use]
    pub fn estimate_swap_ms(&self, bytes: u64) -> f64 {
        if bytes == 0 {
            return 0.0;
        }
        (bytes as f64 / (self.hardware.pcie_swap_gbps * 1e9)) * 1000.0
    }

    /// Computes the maximum safe prefill chunk token count that keeps a fused iteration
    /// within `target_step_ms` given `active_decode_lens`, clamped to `[min_chunk, max_chunk]`.
    #[must_use]
    pub fn max_chunk_for_step_budget(
        &self,
        prefix_len: usize,
        remaining_prompt: usize,
        decode_context_lens: &[usize],
        target_step_ms: f64,
        min_chunk: usize,
        max_chunk: usize,
    ) -> usize {
        let upper = remaining_prompt.min(max_chunk).max(1);
        let lower = min_chunk.min(upper).max(1);
        if lower >= upper {
            return upper;
        }
        // Binary search on [lower, upper] for largest chunk staying under target_step_ms.
        let mut lo = lower;
        let mut hi = upper;
        let mut best = lower;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let est = self.estimate_step(
                &[PrefillChunkCostInput {
                    prefix_len,
                    chunk_len: mid,
                }],
                decode_context_lens,
                false,
            );
            if est.step_ms <= target_step_ms {
                best = mid;
                lo = mid.saturating_add(1);
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }
        best
    }
}
