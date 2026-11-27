// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Canonical HIP-1346/1350 hybrid Transformer + GDN model geometry matrix and
//! deterministic executable Transformer + MoE micro-model for live peer-to-peer
//! distributed inference verification.
//!
//! ## Exact Matrix Alignment (`hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`)
//! - `4B-Hybrid`: 9 Attn + 27 GDN (36 layers), `num_kv_heads = 2`, `head_dim = 128`,
//!   `gdn_heads = 16`, `gdn_state_elems_per_head = 128 * 128`, `gdn_state_mib = 27.0`
//! - `9B-Hybrid`: 10 Attn + 30 GDN (40 layers), `num_kv_heads = 4`, `head_dim = 128`,
//!   `gdn_heads = 24`, `gdn_state_elems_per_head = 128 * 128`, `gdn_state_mib = 45.0`
//! - `27B-Hybrid`: 16 Attn + 48 GDN (64 layers), `num_kv_heads = 4`, `head_dim = 256`,
//!   `gdn_heads = 32`, `gdn_state_elems_per_head = 128 * 256`, `gdn_state_mib = 192.0`
//!
//! ## Resident vs Streamed Weight Accounting
//! - VRAM capacity counts the full resident input-embedding table (`vocab_size * d_model * 2`).
//! - Per-step forward bandwidth streams non-embedding linear/GDN/expert/LM-head weights plus
//!   one `embed_tokens` row gather (`d_model * 2`) per token.

use crate::sha256::{Sha256, SplitMix64, DIGEST_LEN};
use std::collections::HashMap;

/// Canonical HIP-1346/1350 hybrid Attention + Gated DeltaNet (GDN) + MoE model geometry.
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
    /// SwiGLU MLP / expert intermediate dimension (`d_ff`).
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
    /// Number of routed experts per layer.
    pub num_experts: usize,
    /// Number of active experts routed per token (`top_k`).
    pub top_k_experts: usize,
    /// Exact unquantized FP32 GDN recurrent state per active sequence in MiB.
    pub gdn_state_mib: f64,
}

impl HybridModelSpec {
    /// Returns the exact 3-model HIP-1346/1350 shared matrix.
    #[must_use]
    pub fn canonical_matrix() -> [Self; 3] {
        [Self::hybrid_4b(), Self::hybrid_9b(), Self::hybrid_27b()]
    }

    /// Canonical `4B-Hybrid` geometry: 9 Attn + 27 GDN, `num_kv_heads = 2`, `27.0 MiB` GDN state.
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
            top_k_experts: 2,
            gdn_state_mib: 27.0,
        }
    }

    /// Canonical `9B-Hybrid` geometry: 10 Attn + 30 GDN, `num_kv_heads = 4`, `45.0 MiB` GDN state.
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
            top_k_experts: 2,
            gdn_state_mib: 45.0,
        }
    }

    /// Canonical `27B-Hybrid` geometry: 16 Attn + 48 GDN, `num_kv_heads = 4`, `192.0 MiB` GDN state.
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
            top_k_experts: 2,
            gdn_state_mib: 192.0,
        }
    }

    /// Total decoder layers (`num_attention_layers + num_gdn_layers`).
    #[must_use]
    pub const fn total_layers(&self) -> usize {
        self.num_attention_layers + self.num_gdn_layers
    }

    /// Exact total parameter count derived from architectural geometry (including resident
    /// input-embedding table and output LM head).
    #[must_use]
    pub const fn total_params(&self) -> u64 {
        let total_layers = self.total_layers() as u64;
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

    /// Exact non-embedding parameter count streamed during forward passes.
    #[must_use]
    pub const fn non_embedding_params(&self) -> u64 {
        self.total_params() - self.embedding_table_params()
    }

    /// Exact BF16 resident model weight storage in bytes (`total_params * 2`),
    /// including the full input-embedding table for VRAM-capacity accounting.
    #[must_use]
    pub const fn resident_weight_bytes_bf16(&self) -> u64 {
        self.total_params() * 2
    }

    /// Exact BF16 resident input-embedding table storage in bytes (`vocab_size * d_model * 2`).
    #[must_use]
    pub const fn embedding_table_bytes_bf16(&self) -> u64 {
        self.embedding_table_params() * 2
    }

    /// Exact BF16 streamed non-embedding weight bytes per forward pass
    /// (excludes the full input-embedding matrix, accessed via row gather).
    #[must_use]
    pub const fn streamed_weight_bytes_bf16(&self) -> u64 {
        self.non_embedding_params() * 2
    }

    /// Exact BF16 bytes read for one `embed_tokens` row lookup (`d_model * 2`).
    #[must_use]
    pub const fn embedding_row_bytes_bf16(&self) -> u64 {
        (self.d_model as u64) * 2
    }

    /// Exact BF16 weight bytes for a single transformer/GDN decoder layer on average
    /// (excludes input-embedding table and final LM head).
    #[must_use]
    pub const fn avg_decoder_layer_bytes_bf16(&self) -> u64 {
        let lm_head = (self.vocab_size as u64) * (self.d_model as u64);
        let decoder_params = self.non_embedding_params() - lm_head;
        (decoder_params * 2) / (self.total_layers() as u64)
    }

    /// Exact FP16 attention KV cache bytes per token across all attention layers.
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

    /// Exact GDN state per active sequence in MiB (`bytes / 2^20`).
    #[must_use]
    pub fn exact_gdn_state_mib(&self) -> f64 {
        (self.gdn_state_bytes_per_seq() as f64) / (1024.0 * 1024.0)
    }

    /// Activation tensor wire bytes per token in BF16 (`d_model * 2`).
    #[must_use]
    pub const fn activation_bytes_bf16(&self) -> u64 {
        (self.d_model as u64) * 2
    }

    /// Activation tensor wire bytes per token in INT8 per-tensor (`d_model + 4` scale bytes).
    #[must_use]
    pub const fn activation_bytes_int8(&self) -> u64 {
        (self.d_model as u64) + 4
    }

    /// Per-layer resident VRAM bytes for `layer_count` decoder layers at `seq_len` context
    /// and `active_seqs` concurrent sequences, plus optional embedding/LM-head residency.
    #[must_use]
    pub const fn shard_vram_bytes(
        &self,
        layer_count: usize,
        seq_len: usize,
        active_seqs: usize,
        holds_embedding: bool,
        holds_lm_head: bool,
    ) -> u64 {
        let total = self.total_layers() as u64;
        let lc = layer_count as u64;
        let weights = lc * self.avg_decoder_layer_bytes_bf16()
            + if holds_embedding {
                self.embedding_table_bytes_bf16()
            } else {
                0
            }
            + if holds_lm_head {
                self.embedding_table_bytes_bf16()
            } else {
                0
            };
        let kv =
            (self.kv_bytes_per_token_fp16() * (seq_len as u64) * (active_seqs as u64) * lc) / total;
        let gdn = (self.gdn_state_bytes_per_seq() * (active_seqs as u64) * lc) / total;
        weights + kv + gdn
    }
}

/// Truncates an IEEE-754 `f32` to BF16 (`u16`, upper 16 bits with round-to-nearest-even).
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fffu32.wrapping_add(lsb);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Expands a BF16 `u16` back into an exact IEEE-754 `f32`.
#[must_use]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Rounds every element of a slice in-place to canonical BF16 precision.
pub fn round_bf16_in_place(vals: &mut [f32]) {
    for v in vals.iter_mut() {
        *v = bf16_to_f32(f32_to_bf16(*v));
    }
}

/// Encodes an activation vector as little-endian BF16 bytes (`2 * len` bytes).
#[must_use]
pub fn encode_bf16(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    out
}

/// Decodes little-endian BF16 bytes into an `f32` activation vector.
#[must_use]
pub fn decode_bf16(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for &[lo, hi] in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes([lo, hi]);
        out.push(bf16_to_f32(bits));
    }
    Some(out)
}

/// Encodes an activation vector with symmetric per-tensor INT8 quantization
/// (`4`-byte `f32` scale + `len` signed `i8` codes).
#[must_use]
pub fn encode_int8(vals: &[f32]) -> Vec<u8> {
    let mut max_abs = 0.0f32;
    for &v in vals {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs < 1e-12 {
        1.0f32
    } else {
        max_abs / 127.0
    };
    let inv_scale = 1.0 / scale;
    let mut out = Vec::with_capacity(4 + vals.len());
    out.extend_from_slice(&scale.to_le_bytes());
    for &v in vals {
        let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        out.push(q as u8);
    }
    out
}

/// Decodes symmetric per-tensor INT8 bytes (`4`-byte scale + `i8` codes) back to `f32`.
#[must_use]
pub fn decode_int8(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 4 {
        return None;
    }
    let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if !scale.is_finite() {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() - 4);
    for &b in &bytes[4..] {
        out.push((b as i8) as f32 * scale);
    }
    Some(out)
}

/// Cosine similarity between two equal-length float vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        dot / denom
    }
}

/// Configuration for the live executable deterministic Transformer + MoE micro-model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicroModelConfig {
    /// Total number of Transformer + MoE layers (e.g. `6` or `12`).
    pub num_layers: usize,
    /// Hidden state width (`d_model`).
    pub d_model: usize,
    /// Number of causal attention heads (`d_model` must be divisible by `num_heads`).
    pub num_heads: usize,
    /// Number of routed MoE experts per layer.
    pub num_experts: usize,
    /// Top-K active experts per token.
    pub top_k: usize,
    /// Expert MLP intermediate width.
    pub expert_hidden: usize,
    /// Vocabulary size for embedding lookup and LM head logits.
    pub vocab_size: usize,
    /// Deterministic PRNG seed for canonical weight initialization.
    pub weight_seed: u64,
}

impl MicroModelConfig {
    /// Default 6-layer Transformer + MoE micro-model for live 3-node TCP execution and failover.
    #[must_use]
    pub const fn default_live_6layer() -> Self {
        Self {
            num_layers: 6,
            d_model: 64,
            num_heads: 4,
            num_experts: 4,
            top_k: 2,
            expert_hidden: 64,
            vocab_size: 256,
            weight_seed: 1350,
        }
    }

    /// Head dimension (`d_model / num_heads`).
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}

/// Cryptographic SHA-256 commitment over a contiguous stage's exact layer weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightsCommitment {
    /// Inclusive starting layer index.
    pub layer_start: usize,
    /// Exclusive ending layer index.
    pub layer_end: usize,
    /// SHA-256 digest of all layer parameters in canonical little-endian byte order.
    pub digest: [u8; DIGEST_LEN],
}

/// A single routed MoE expert MLP (`W_up: [expert_hidden, d_model]`, `W_down: [d_model, expert_hidden]`).
#[derive(Debug, Clone)]
pub struct MoeExpertWeights {
    /// Up-projection matrix flattened row-major (`expert_hidden * d_model`).
    pub w_up: Vec<f32>,
    /// Down-projection matrix flattened row-major (`d_model * expert_hidden`).
    pub w_down: Vec<f32>,
}

/// One deterministic Transformer + Top-K MoE decoder layer.
#[derive(Debug, Clone)]
pub struct TransformerMoeLayer {
    /// Zero-based global layer index.
    pub layer_idx: usize,
    /// Hidden width.
    pub d_model: usize,
    /// Attention head count.
    pub num_heads: usize,
    /// Top-K experts activated per token.
    pub top_k: usize,
    /// Expert intermediate width.
    pub expert_hidden: usize,
    /// Pre-attention RMSNorm scale (`d_model`).
    pub attn_norm: Vec<f32>,
    /// Query projection (`d_model * d_model`).
    pub w_q: Vec<f32>,
    /// Key projection (`d_model * d_model`).
    pub w_k: Vec<f32>,
    /// Value projection (`d_model * d_model`).
    pub w_v: Vec<f32>,
    /// Output projection (`d_model * d_model`).
    pub w_o: Vec<f32>,
    /// Post-attention RMSNorm scale (`d_model`).
    pub ffn_norm: Vec<f32>,
    /// MoE router gate (`num_experts * d_model`).
    pub router_gate: Vec<f32>,
    /// Routed expert MLPs (`num_experts`).
    pub experts: Vec<MoeExpertWeights>,
}

impl TransformerMoeLayer {
    /// Deterministically synthesizes layer `layer_idx` from `cfg`.
    #[must_use]
    pub fn from_config(cfg: &MicroModelConfig, layer_idx: usize) -> Self {
        let mut rng = SplitMix64::new(
            cfg.weight_seed
                .wrapping_add((layer_idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        );
        let d = cfg.d_model;
        let eh = cfg.expert_hidden;
        let scale_d = 1.0 / (d as f32).sqrt();
        let scale_eh = 1.0 / (eh as f32).sqrt();

        let mut gen_mat = |len: usize, scale: f32| -> Vec<f32> {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(rng.next_f32_signed() * scale);
            }
            v
        };

        let attn_norm = vec![1.0f32; d];
        let w_q = gen_mat(d * d, scale_d * 0.5);
        let w_k = gen_mat(d * d, scale_d * 0.5);
        let w_v = gen_mat(d * d, scale_d * 0.5);
        let w_o = gen_mat(d * d, scale_d * 0.5);
        let ffn_norm = vec![1.0f32; d];
        let router_gate = gen_mat(cfg.num_experts * d, scale_d);

        let mut experts = Vec::with_capacity(cfg.num_experts);
        for _ in 0..cfg.num_experts {
            experts.push(MoeExpertWeights {
                w_up: gen_mat(eh * d, scale_d * 0.4),
                w_down: gen_mat(d * eh, scale_eh * 0.4),
            });
        }

        Self {
            layer_idx,
            d_model: d,
            num_heads: cfg.num_heads,
            top_k: cfg.top_k,
            expert_hidden: eh,
            attn_norm,
            w_q,
            w_k,
            w_v,
            w_o,
            ffn_norm,
            router_gate,
            experts,
        }
    }

    /// Feeds canonical little-endian weight bytes into a SHA-256 hasher.
    pub fn hash_into(&self, hasher: &mut Sha256) {
        hasher.update(&(self.layer_idx as u32).to_le_bytes());
        hash_f32_slice(hasher, &self.attn_norm);
        hash_f32_slice(hasher, &self.w_q);
        hash_f32_slice(hasher, &self.w_k);
        hash_f32_slice(hasher, &self.w_v);
        hash_f32_slice(hasher, &self.w_o);
        hash_f32_slice(hasher, &self.ffn_norm);
        hash_f32_slice(hasher, &self.router_gate);
        for exp in &self.experts {
            hash_f32_slice(hasher, &exp.w_up);
            hash_f32_slice(hasher, &exp.w_down);
        }
    }

    /// Executes one causal decode/prefill step at position `seq_pos`, appending `(K, V)` into
    /// `kv_cache` and returning the updated hidden state plus selected top-K expert indices.
    #[must_use]
    pub fn forward_token(
        &self,
        hidden: &[f32],
        kv_cache: &mut LayerKvState,
    ) -> (Vec<f32>, Vec<usize>) {
        let d = self.d_model;
        let normed = rms_norm(hidden, &self.attn_norm, 1e-5);
        let q = matvec(&self.w_q, &normed, d, d);
        let k = matvec(&self.w_k, &normed, d, d);
        let v = matvec(&self.w_v, &normed, d, d);

        kv_cache.keys.push(k);
        kv_cache.values.push(v);

        let seq_len = kv_cache.keys.len();
        let head_dim = d / self.num_heads;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
        let mut attn_concat = vec![0.0f32; d];

        for h in 0..self.num_heads {
            let offset = h * head_dim;
            let mut scores = Vec::with_capacity(seq_len);
            let mut max_score = f32::NEG_INFINITY;
            for key_vec in &kv_cache.keys {
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q[offset + i] * key_vec[offset + i];
                }
                let s = dot * inv_sqrt_hd;
                if s > max_score {
                    max_score = s;
                }
                scores.push(s);
            }
            let mut denom = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                denom += *s;
            }
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for t in 0..seq_len {
                let prob = scores[t] * inv_denom;
                for i in 0..head_dim {
                    attn_concat[offset + i] += prob * kv_cache.values[t][offset + i];
                }
            }
        }

        let attn_proj = matvec(&self.w_o, &attn_concat, d, d);
        let mut residual = vec![0.0f32; d];
        for i in 0..d {
            residual[i] = hidden[i] + attn_proj[i];
        }

        let normed2 = rms_norm(&residual, &self.ffn_norm, 1e-5);
        let router_logits = matvec(&self.router_gate, &normed2, self.experts.len(), d);
        let routed = top_k_softmax(&router_logits, self.top_k);

        let mut moe_out = vec![0.0f32; d];
        let mut chosen_experts = Vec::with_capacity(routed.len());
        for (exp_idx, weight) in routed {
            chosen_experts.push(exp_idx);
            let exp = &self.experts[exp_idx];
            let mut up = matvec(&exp.w_up, &normed2, self.expert_hidden, d);
            for u in &mut up {
                // SiLU activation: x / (1 + exp(-x))
                *u = *u / (1.0 + (-*u).exp());
            }
            let down = matvec(&exp.w_down, &up, d, self.expert_hidden);
            for i in 0..d {
                moe_out[i] += weight * down[i];
            }
        }

        for i in 0..d {
            residual[i] += moe_out[i];
        }
        (residual, chosen_experts)
    }
}

/// Per-layer KV cache holding cached key and value vectors for one sequence.
#[derive(Debug, Clone, Default)]
pub struct LayerKvState {
    /// Cached key vectors indexed by sequence position.
    pub keys: Vec<Vec<f32>>,
    /// Cached value vectors indexed by sequence position.
    pub values: Vec<Vec<f32>>,
}

/// Per-session KV cache held by a pipeline stage covering `[layer_start, layer_end)`.
#[derive(Debug, Clone)]
pub struct StageKvCache {
    /// One `LayerKvState` per local layer in the stage.
    pub layers: Vec<LayerKvState>,
}

impl StageKvCache {
    /// Allocates an empty KV cache for `num_local_layers`.
    #[must_use]
    pub fn new(num_local_layers: usize) -> Self {
        Self {
            layers: vec![LayerKvState::default(); num_local_layers],
        }
    }

    /// Number of tokens currently cached in this stage.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.keys.len())
    }
}

/// Contiguous slice of Transformer + MoE layers owned by one peer stage.
#[derive(Debug, Clone)]
pub struct LayerSlice {
    /// Inclusive starting layer index.
    pub layer_start: usize,
    /// Exclusive ending layer index.
    pub layer_end: usize,
    /// Model configuration metadata.
    pub config: MicroModelConfig,
    /// Concrete layers belonging to `[layer_start, layer_end)`.
    pub layers: Vec<TransformerMoeLayer>,
    /// Cryptographic SHA-256 commitment over all layer weights in this slice.
    pub commitment: WeightsCommitment,
}

impl LayerSlice {
    /// Builds a deterministic contiguous layer shard `[layer_start, layer_end)`.
    #[must_use]
    pub fn from_config(cfg: &MicroModelConfig, layer_start: usize, layer_end: usize) -> Self {
        assert!(
            layer_start < layer_end && layer_end <= cfg.num_layers,
            "invalid layer slice [{layer_start}, {layer_end}) for {} layers",
            cfg.num_layers
        );
        let mut layers = Vec::with_capacity(layer_end - layer_start);
        let mut hasher = Sha256::new();
        hasher.update(b"hanzo-p2p-layer-slice-v1");
        hasher.update(&(layer_start as u32).to_le_bytes());
        hasher.update(&(layer_end as u32).to_le_bytes());
        for idx in layer_start..layer_end {
            let layer = TransformerMoeLayer::from_config(cfg, idx);
            layer.hash_into(&mut hasher);
            layers.push(layer);
        }
        let commitment = WeightsCommitment {
            layer_start,
            layer_end,
            digest: hasher.finalize(),
        };
        Self {
            layer_start,
            layer_end,
            config: *cfg,
            layers,
            commitment,
        }
    }

    /// Executes all layers in this slice sequentially for one token, updating `cache`.
    #[must_use]
    pub fn forward_token(
        &self,
        input: &[f32],
        cache: &mut StageKvCache,
    ) -> (Vec<f32>, Vec<Vec<usize>>) {
        let mut hidden = input.to_vec();
        let mut routed_per_layer = Vec::with_capacity(self.layers.len());
        for (local_idx, layer) in self.layers.iter().enumerate() {
            let (next_hidden, experts) = layer.forward_token(&hidden, &mut cache.layers[local_idx]);
            hidden = next_hidden;
            routed_per_layer.push(experts);
        }
        (hidden, routed_per_layer)
    }
}

/// Complete monolithic deterministic Transformer + MoE micro-model serving as the
/// single-node reference oracle (embedding table, all decoder layers, final norm, LM head).
#[derive(Debug, Clone)]
pub struct MicroModel {
    /// Model geometry and seed.
    pub config: MicroModelConfig,
    /// Input embedding table (`vocab_size * d_model`).
    pub embed_tokens: Vec<f32>,
    /// All decoder layers (`0..num_layers`).
    pub layers: Vec<TransformerMoeLayer>,
    /// Final RMSNorm scale (`d_model`).
    pub final_norm: Vec<f32>,
    /// Output LM head projection (`vocab_size * d_model`).
    pub lm_head: Vec<f32>,
}

impl MicroModel {
    /// Deterministically initializes the full monolithic model from `cfg`.
    #[must_use]
    pub fn from_config(cfg: MicroModelConfig) -> Self {
        let mut rng = SplitMix64::new(cfg.weight_seed);
        let d = cfg.d_model;
        let scale_d = 1.0 / (d as f32).sqrt();

        let mut embed_tokens = Vec::with_capacity(cfg.vocab_size * d);
        for _ in 0..(cfg.vocab_size * d) {
            embed_tokens.push(rng.next_f32_signed() * 0.5);
        }

        let final_norm = vec![1.0f32; d];
        let mut lm_head = Vec::with_capacity(cfg.vocab_size * d);
        for _ in 0..(cfg.vocab_size * d) {
            lm_head.push(rng.next_f32_signed() * scale_d);
        }

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for idx in 0..cfg.num_layers {
            layers.push(TransformerMoeLayer::from_config(&cfg, idx));
        }

        Self {
            config: cfg,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        }
    }

    /// Single-token `embed_tokens` row lookup (`d_model` elements, BF16-canonicalizable).
    #[must_use]
    pub fn embed_token(&self, token_id: u32) -> Vec<f32> {
        let d = self.config.d_model;
        let idx = (token_id as usize) % self.config.vocab_size;
        self.embed_tokens[idx * d..(idx + 1) * d].to_vec()
    }

    /// Projects the final hidden state through `final_norm` and `lm_head` to vocabulary logits.
    #[must_use]
    pub fn project_logits(&self, final_hidden: &[f32]) -> Vec<f32> {
        let normed = rms_norm(final_hidden, &self.final_norm, 1e-5);
        matvec(
            &self.lm_head,
            &normed,
            self.config.vocab_size,
            self.config.d_model,
        )
    }

    /// Returns greedy argmax token ID from `logits` (ties broken deterministically by lowest index).
    #[must_use]
    pub fn argmax_token(logits: &[f32]) -> u32 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &val) in logits.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        best_idx as u32
    }

    /// Runs monolithic reference generation with optional BF16 boundary rounding at `stage_boundaries`
    /// so single-node reference outputs can be compared bit-exactly against a multi-stage BF16 pipeline.
    #[must_use]
    pub fn generate_reference(
        &self,
        prompt_tokens: &[u32],
        gen_tokens: usize,
        stage_boundaries: &[usize],
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut kv_cache = StageKvCache::new(self.config.num_layers);
        let mut last_logits = Vec::new();
        let mut all_gen_tokens = Vec::with_capacity(gen_tokens);
        let mut all_gen_logits = Vec::with_capacity(gen_tokens);

        let forward_one = |tok: u32, cache: &mut StageKvCache| -> Vec<f32> {
            let mut hidden = self.embed_token(tok);
            if !stage_boundaries.is_empty() {
                round_bf16_in_place(&mut hidden);
            }
            for (l_idx, layer) in self.layers.iter().enumerate() {
                let (next_h, _) = layer.forward_token(&hidden, &mut cache.layers[l_idx]);
                hidden = next_h;
                if stage_boundaries.contains(&(l_idx + 1)) {
                    round_bf16_in_place(&mut hidden);
                }
            }
            self.project_logits(&hidden)
        };

        for &tok in prompt_tokens {
            last_logits = forward_one(tok, &mut kv_cache);
        }

        for _ in 0..gen_tokens {
            let next_tok = Self::argmax_token(&last_logits);
            all_gen_tokens.push(next_tok);
            all_gen_logits.push(last_logits.clone());
            last_logits = forward_one(next_tok, &mut kv_cache);
        }

        (all_gen_tokens, all_gen_logits)
    }
}

/// Multi-session KV cache map stored inside each live worker node.
pub type SessionKvStore = HashMap<u64, StageKvCache>;

fn hash_f32_slice(hasher: &mut Sha256, slice: &[f32]) {
    for &v in slice {
        hasher.update(&v.to_bits().to_le_bytes());
    }
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / (x.len() as f32) + eps).sqrt();
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        out.push(x[i] * inv_rms * weight[i]);
    }
    out
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for (r, out_val) in out.iter_mut().enumerate() {
        let row_start = r * cols;
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += w[row_start + c] * x[c];
        }
        *out_val = acc;
    }
    out
}

fn top_k_softmax(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let k = top_k.min(logits.len()).max(1);
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k);
    let max_v = indexed[0].1;
    let mut sum = 0.0f32;
    for (_, v) in &mut indexed {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for (_, v) in &mut indexed {
            *v /= sum;
        }
    }
    indexed
}
