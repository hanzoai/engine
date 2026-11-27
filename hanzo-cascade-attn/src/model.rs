// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Canonical HIP-1346 hybrid Attention + Linear-Attention (GDN/LSA) model geometry
//! and exact memory/FLOP accounting.
//!
//! ## Exact Shared HIP-1346 Matrix
//! Matches `hanzo-paged-attn/tests/kivi_kv_quant.rs`:
//! - `4B-Hybrid`: 9 Attn + 27 GDN layers, `num_kv_heads = 2`, `head_dim = 128`, `gdn_state_mib = 27.0`
//! - `9B-Hybrid`: 10 Attn + 30 GDN layers, `num_kv_heads = 4`, `head_dim = 128`, `gdn_state_mib = 45.0`
//! - `27B-Hybrid`: 16 Attn + 48 GDN layers, `num_kv_heads = 4`, `head_dim = 256`, `gdn_state_mib = 192.0`
//!
//! ## Resident vs Streamed Weight Accounting
//! - **Resident VRAM** includes the full input-embedding matrix (`vocab_size * d_model * 2` bytes),
//!   the output LM-head matrix, and all attention/GDN/MLP linear layers.
//! - **Per-step memory bandwidth** streams non-embedding linear/GDN/LM-head weights once per step
//!   plus only single-row `embed_tokens` gathers (`d_model * 2` bytes per active token).

/// Weight quantization format evaluated across the HIP-1346 serving matrix.
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
    /// Canonical display string for scorecards and JSON reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Int4Awq => "INT4-AWQ",
        }
    }

    /// Returns both supported quantization formats in canonical order.
    #[must_use]
    pub const fn all() -> [Self; 2] {
        [Self::Bf16, Self::Int4Awq]
    }
}

/// Canonical HIP-1346 hybrid Attention + Gated DeltaNet (GDN/LSA) model geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridModelSpec {
    /// Human-readable canonical label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub label: &'static str,
    /// Number of full grouped-query attention layers holding paged KV caches.
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

    /// Grouped-query attention ratio (`num_q_heads / num_kv_heads`).
    #[must_use]
    pub const fn gqa_group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    /// Fraction of decoder layers that use full paged attention (`0.25` across all 3 models).
    #[must_use]
    pub fn attention_layer_fraction(&self) -> f64 {
        (self.num_attention_layers as f64) / (self.total_layers() as f64)
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

    /// Attention linear projection parameters (Q, K, V, O across all attention layers).
    #[must_use]
    pub const fn attention_proj_params(&self) -> u64 {
        let d = self.d_model as u64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u64;
        let qkv = (self.num_attention_layers as u64) * (d + 2 * kv_dim) * d;
        let out = (self.num_attention_layers as u64) * d * d;
        qkv + out
    }

    /// GDN/LSA linear projection parameters across all GDN layers.
    #[must_use]
    pub const fn gdn_proj_params(&self) -> u64 {
        let d = self.d_model as u64;
        (self.num_gdn_layers as u64) * 4 * d * d
    }

    /// Dense SwiGLU MLP + router gate parameters across all layers.
    #[must_use]
    pub const fn mlp_and_router_params(&self) -> u64 {
        let total_layers = self.total_layers() as u64;
        let d = self.d_model as u64;
        let ff = self.d_ff as u64;
        let router_gate = total_layers * (self.num_experts as u64) * d;
        let mlp_gate_up = total_layers * 2 * ff * d;
        let mlp_down = total_layers * d * ff;
        router_gate + mlp_gate_up + mlp_down
    }

    /// Exact transformer/GDN/MLP linear layer parameter count (excluding input embedding and LM head).
    #[must_use]
    pub const fn linear_layer_params(&self) -> u64 {
        self.attention_proj_params() + self.gdn_proj_params() + self.mlp_and_router_params()
    }

    /// Exact total parameter count across embeddings, linear layers, and LM head.
    #[must_use]
    pub const fn total_params(&self) -> u64 {
        self.embedding_table_params() + self.linear_layer_params() + self.lm_head_params()
    }

    /// Exact non-embedding parameter count streamed on every step.
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
        let linear_bytes = Self::quantized_bytes(self.linear_layer_params(), quant);
        embed_bytes + lm_head_bytes + linear_bytes
    }

    /// Exact per-step streamed weight bytes for `quant` (excludes the full input-embedding
    /// matrix; callers add `embedding_row_bytes_bf16() * total_tokens` for single-row lookups).
    #[must_use]
    pub const fn streamed_weight_bytes(&self, quant: QuantFormat) -> u64 {
        let lm_head_bytes = self.lm_head_params() * 2;
        let linear_bytes = Self::quantized_bytes(self.linear_layer_params(), quant);
        lm_head_bytes + linear_bytes
    }

    /// Streamed bytes for attention linear projections (`Q, K, V, O`) alone.
    #[must_use]
    pub const fn streamed_attn_proj_bytes(&self, quant: QuantFormat) -> u64 {
        Self::quantized_bytes(self.attention_proj_params(), quant)
    }

    /// Streamed bytes for GDN/LSA linear projections alone.
    #[must_use]
    pub const fn streamed_gdn_proj_bytes(&self, quant: QuantFormat) -> u64 {
        Self::quantized_bytes(self.gdn_proj_params(), quant)
    }

    /// Streamed bytes for dense MLP + router + LM head (excluding full input embeddings).
    #[must_use]
    pub const fn streamed_dense_mlp_and_head_bytes(&self, quant: QuantFormat) -> u64 {
        Self::quantized_bytes(self.mlp_and_router_params(), quant) + self.lm_head_params() * 2
    }

    /// Converts a parameter count to quantized bytes.
    #[must_use]
    const fn quantized_bytes(params: u64, quant: QuantFormat) -> u64 {
        match quant {
            QuantFormat::Bf16 => params * 2,
            // Group-128 INT4 + FP16 scale & zero = 4.25 bits/param = 17/32 bytes/param.
            QuantFormat::Int4Awq => (params * 17) / 32,
        }
    }

    /// Exact BF16 bytes read for a single `embed_tokens` row lookup (`d_model * 2`).
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

    /// Exact attention dot-product (`Q*K^T` + `P*V`) FLOPs for a single sequence with
    /// `q_tokens` query tokens attending over `past_kv_len` cached tokens plus causal self-attention
    /// over `q_tokens`, summed across all attention layers.
    #[must_use]
    pub fn attention_core_flops(&self, q_tokens: usize, past_kv_len: usize) -> f64 {
        let q = q_tokens as f64;
        let past = past_kv_len as f64;
        // Effective mean attended KV length per query token under causal masking:
        // past_kv_len + (q_tokens + 1) / 2.
        let mean_kv = past + 0.5 * (q + 1.0);
        // 2 FLOPs per MAC for Q*K^T + 2 FLOPs per MAC for P*V = 4 * num_q_heads * head_dim * q * mean_kv.
        4.0 * (self.num_attention_layers as f64)
            * (self.num_q_heads as f64)
            * (self.head_dim as f64)
            * q
            * mean_kv
    }

    /// Exact GDN recurrent state update FLOPs (`S_t = decay * S_{t-1} + beta * k_t (x) (v_t - S^T k_t)`
    /// and `y_t = S_t^T q_t`) for `tokens` active tokens across all GDN layers.
    #[must_use]
    pub fn gdn_recurrent_flops(&self, tokens: usize) -> f64 {
        // 6 FLOPs per state element per token (decay, read-query, read-key, outer-product update).
        6.0 * (self.num_gdn_layers as f64)
            * (self.gdn_heads as f64)
            * (self.gdn_state_elems_per_head as f64)
            * (tokens as f64)
    }

    /// Exact dense linear projection FLOPs (`2 * params * tokens`) for attention projections (`Q,K,V,O`).
    #[must_use]
    pub fn attn_proj_flops(&self, tokens: usize) -> f64 {
        2.0 * (self.attention_proj_params() as f64) * (tokens as f64)
    }

    /// Exact GDN linear projection FLOPs (`2 * params * tokens`).
    #[must_use]
    pub fn gdn_proj_flops(&self, tokens: usize) -> f64 {
        2.0 * (self.gdn_proj_params() as f64) * (tokens as f64)
    }

    /// Exact dense SwiGLU MLP + router + LM-head FLOPs for `tokens` forward tokens and `logit_tokens`
    /// tokens that compute final vocabulary logits.
    #[must_use]
    pub fn dense_mlp_and_head_flops(&self, tokens: usize, logit_tokens: usize) -> f64 {
        2.0 * (self.mlp_and_router_params() as f64) * (tokens as f64)
            + 2.0 * (self.lm_head_params() as f64) * (logit_tokens as f64)
    }
}
