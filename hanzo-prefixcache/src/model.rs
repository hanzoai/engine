// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shared `4B / 9B / 27B` hybrid Attention + GDN model matrix and transparent
//! analytical GPU/IO/energy performance model for HIP-1346 scorecards.
//!
//! All architectural layer counts match `docs/HIP-1346-kivi-kv-compression.md`:
//! - `4B-Hybrid`: 9 Attention + 27 GDN layers (`head_dim = 128`, `kv_heads = 4`)
//! - `9B-Hybrid`: 10 Attention + 30 GDN layers (`head_dim = 128`, `kv_heads = 8`)
//! - `27B-Hybrid`: 16 Attention + 48 GDN layers (`head_dim = 256`, `kv_heads = 8`)
//!
//! Energy per token (`analytical_mj_per_token`) and GPU prefill/decode timings are
//! explicitly modeled from compute FLOPs, HBM/PCIe/NIC bandwidth, and TDP power
//! envelopes, combined directly with real host-measured SHA-256, CRC32, radix lookup,
//! and Tier 1/Tier 2 filesystem snapshot I/O timings.

use crate::crypto::Sha256;
use crate::key::{
    CompatibilityProfile, ModelCheckpointId, QuantFormat, RopePosEncoding, RopeScheme,
    RuntimeConfigIdentity, TokenizerIdentity,
};

/// Architectural and analytical hardware parameters for a model in the shared 4B/9B/27B matrix.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// Human-readable model label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub name: &'static str,
    /// Nominal active parameter count in billions.
    pub params_billions: f64,
    /// Number of paged attention layers.
    pub attn_layers: u16,
    /// Number of recurrent Gated DeltaNet (GDN) layers.
    pub gdn_layers: u16,
    /// Number of KV heads per attention layer.
    pub kv_heads: u16,
    /// Attention head dimension.
    pub head_dim: u16,
    /// Recurrent GDN state dimension per head (`FP32` unquantized).
    pub gdn_state_dim: u16,
    /// Number of tokens per prefix cache block.
    pub block_size: u32,
    /// Active KV cache quantization format.
    pub quant_format: QuantFormat,
    /// Reference GPU prefill throughput in tokens/sec (analytical H100/RTX-class model).
    pub gpu_prefill_tok_per_s: f64,
    /// Reference GPU decode base latency per generated token in ms (weights + active compute).
    pub gpu_decode_base_itl_ms: f64,
    /// Additional decode ITL per 1,024 resident context tokens (HBM KV scan bandwidth term, ms).
    pub gpu_decode_itl_per_1k_ctx_ms: f64,
    /// Reference GPU active power draw during prefill/decode in Watts (for analytical `mJ/tok`).
    pub gpu_power_watts: f64,
    /// Reference host DRAM / NVMe + PCIe transfer bandwidth for Tier 1 restore (bytes/sec).
    pub tier1_restore_bw_bps: f64,
    /// Reference nominal Tier 2 cold object-store network bandwidth (bytes/sec).
    pub tier2_nominal_bw_bps: f64,
    /// Reference Tier 2 per-fetch RTT overhead (seconds).
    pub tier2_rtt_sec: f64,
    /// Host + NVMe + NIC incremental I/O power draw during snapshot transfers (Watts).
    pub io_subsystem_watts: f64,
    /// Tight HBM prefix cache budget in blocks (triggers realistic eviction under HBM-only regime).
    pub tight_hbm_block_budget: u64,
}

impl ModelSpec {
    /// `4B-Hybrid` specification (`9` Attn + `27` GDN layers, `d=128`, `4` KV heads, `FP16` KV + `FP32` GDN).
    #[must_use]
    pub fn hybrid_4b() -> Self {
        Self {
            name: "4B-Hybrid",
            params_billions: 4.0,
            attn_layers: 9,
            gdn_layers: 27,
            kv_heads: 4,
            head_dim: 128,
            gdn_state_dim: 128,
            block_size: 64,
            quant_format: QuantFormat::Fp16,
            gpu_prefill_tok_per_s: 28_000.0,
            gpu_decode_base_itl_ms: 4.2,
            gpu_decode_itl_per_1k_ctx_ms: 0.14,
            gpu_power_watts: 185.0,
            tier1_restore_bw_bps: 12.0e9, // 12 GB/s PCIe/NVMe-DRAM warm tier
            tier2_nominal_bw_bps: 2.5e9,  // 20 Gbps (2.5 GB/s) cluster object store
            tier2_rtt_sec: 0.0018,        // 1.8 ms RTT
            io_subsystem_watts: 22.0,
            tight_hbm_block_budget: 28,
        }
    }

    /// `9B-Hybrid` specification (`10` Attn + `30` GDN layers, `d=128`, `8` KV heads, `FP16` KV + `FP32` GDN).
    #[must_use]
    pub fn hybrid_9b() -> Self {
        Self {
            name: "9B-Hybrid",
            params_billions: 9.0,
            attn_layers: 10,
            gdn_layers: 30,
            kv_heads: 8,
            head_dim: 128,
            gdn_state_dim: 128,
            block_size: 64,
            quant_format: QuantFormat::Fp16,
            gpu_prefill_tok_per_s: 14_500.0,
            gpu_decode_base_itl_ms: 8.6,
            gpu_decode_itl_per_1k_ctx_ms: 0.24,
            gpu_power_watts: 290.0,
            tier1_restore_bw_bps: 12.0e9,
            tier2_nominal_bw_bps: 2.5e9,
            tier2_rtt_sec: 0.0018,
            io_subsystem_watts: 24.0,
            tight_hbm_block_budget: 28,
        }
    }

    /// `27B-Hybrid` specification (`16` Attn + `48` GDN layers, `d=256`, `8` KV heads, `FP16` KV + `FP32` GDN).
    #[must_use]
    pub fn hybrid_27b() -> Self {
        Self {
            name: "27B-Hybrid",
            params_billions: 27.0,
            attn_layers: 16,
            gdn_layers: 48,
            kv_heads: 8,
            head_dim: 256,
            gdn_state_dim: 256,
            block_size: 64,
            quant_format: QuantFormat::Fp16,
            gpu_prefill_tok_per_s: 5_200.0,
            gpu_decode_base_itl_ms: 21.5,
            gpu_decode_itl_per_1k_ctx_ms: 0.58,
            gpu_power_watts: 520.0,
            tier1_restore_bw_bps: 12.0e9,
            tier2_nominal_bw_bps: 2.5e9,
            tier2_rtt_sec: 0.0020,
            io_subsystem_watts: 28.0,
            tight_hbm_block_budget: 28,
        }
    }

    /// Return the canonical shared `4B / 9B / 27B` matrix.
    #[must_use]
    pub fn matrix() -> [Self; 3] {
        [Self::hybrid_4b(), Self::hybrid_9b(), Self::hybrid_27b()]
    }

    /// Exact full-scale KV attention bytes per block of `block_size` tokens.
    ///
    /// `2 (K+V) * attn_layers * kv_heads * head_dim * block_size * (kv_bits_per_elem / 8)`.
    #[must_use]
    pub fn full_kv_bytes_per_block(&self) -> u64 {
        let elems = 2u64
            * u64::from(self.attn_layers)
            * u64::from(self.kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.block_size);
        ((elems as f64) * (self.quant_format.kv_bits_per_elem() / 8.0)).ceil() as u64
    }

    /// Exact full-scale unquantized `FP32` GDN recurrent state checkpoint bytes at a block boundary.
    ///
    /// Matches `docs/HIP-1346-kivi-kv-compression.md` (`4B`: `27 MiB`, `9B`: `45 MiB`, `27B`: `192 MiB`
    /// across all GDN layers in full `FP32`; amortized per checkpointed block boundary when
    /// only the deepest matched boundary's GDN checkpoint needs to be restored!).
    #[must_use]
    pub fn full_gdn_checkpoint_bytes(&self) -> u64 {
        // 4B: 27 layers * 1.0 MiB = 27 MiB
        // 9B: 30 layers * 1.5 MiB = 45 MiB
        // 27B: 48 layers * 4.0 MiB = 192 MiB
        let bytes_per_layer: u64 = match self.name {
            "4B-Hybrid" => 1024 * 1024,
            "9B-Hybrid" => 1536 * 1024,
            "27B-Hybrid" => 4 * 1024 * 1024,
            _ => {
                u64::from(self.kv_heads)
                    * u64::from(self.gdn_state_dim)
                    * u64::from(self.gdn_state_dim)
                    * 4
            }
        };
        u64::from(self.gdn_layers) * bytes_per_layer
    }

    /// Accounted HBM / storage bytes per cached prefix block:
    /// full KV page bytes for the block plus amortized GDN boundary checkpoint storage
    /// (1 boundary checkpoint retained per 8 blocks / 512 tokens to bound snapshot bloat while
    /// preserving exact GDN restoration at block boundaries).
    #[must_use]
    pub fn accounted_bytes_per_block(&self) -> u64 {
        self.full_kv_bytes_per_block() + (self.full_gdn_checkpoint_bytes() / 8)
    }

    /// Total full-scale snapshot bytes transferred when restoring `matched_blocks` KV blocks
    /// plus the single latest block-boundary GDN recurrent state checkpoint.
    #[must_use]
    pub fn full_restore_transfer_bytes(&self, restored_blocks: usize) -> u64 {
        if restored_blocks == 0 {
            return 0;
        }
        (restored_blocks as u64) * self.full_kv_bytes_per_block() + self.full_gdn_checkpoint_bytes()
    }

    /// Baseline local prefill time per token (seconds).
    #[must_use]
    pub fn prefill_sec_per_token(&self) -> f64 {
        1.0 / self.gpu_prefill_tok_per_s
    }

    /// Exact analytical Tier 2 crossover bandwidth (MB/s) below which fetching `blocks` KV blocks
    /// + 1 GDN boundary checkpoint from Tier 2 takes longer than locally recomputing prefill.
    ///
    /// Derived from:
    /// `rtt + restore_bytes / bw_bps == (blocks * block_size) * prefill_sec_per_token`
    #[must_use]
    pub fn tier2_crossover_bandwidth_mbps(&self, blocks: usize) -> f64 {
        let tokens = (blocks as f64) * f64::from(self.block_size);
        let recompute_sec = tokens * self.prefill_sec_per_token();
        let net_budget_sec = (recompute_sec - self.tier2_rtt_sec).max(1e-6);
        let restore_bytes = self.full_restore_transfer_bytes(blocks) as f64;
        (restore_bytes / net_budget_sec) / 1.0e6
    }

    /// Analytical prefill compute duration in milliseconds for `recomputed_tokens`
    /// at total context depth `total_prompt_tokens` (linear FFN + mild quadratic attention term).
    #[must_use]
    pub fn analytical_prefill_ms(
        &self,
        recomputed_tokens: usize,
        total_prompt_tokens: usize,
    ) -> f64 {
        if recomputed_tokens == 0 {
            return 0.0;
        }
        let base_ms = (recomputed_tokens as f64 / self.gpu_prefill_tok_per_s) * 1000.0;
        // Mild quadratic attention factor (~1.0 at 1K, ~1.28 at 8K context).
        let ctx_scale = 1.0 + 0.035 * (total_prompt_tokens as f64 / 1024.0);
        base_ms * ctx_scale
    }

    /// Analytical per-token decode inter-token latency (ITL) in milliseconds at `context_tokens`.
    #[must_use]
    pub fn analytical_decode_itl_ms(&self, context_tokens: usize) -> f64 {
        self.gpu_decode_base_itl_ms
            + self.gpu_decode_itl_per_1k_ctx_ms * (context_tokens as f64 / 1024.0)
    }

    /// Build the canonical 5-field [`CompatibilityProfile`] for this model specification.
    #[must_use]
    pub fn compatibility_profile(&self, tokenizer_bytes: &[u8]) -> CompatibilityProfile {
        let checkpoint_manifest = format!(
            "hanzo-engine:{}:attn={}:gdn={}:kv_heads={}:d={}",
            self.name, self.attn_layers, self.gdn_layers, self.kv_heads, self.head_dim
        );
        CompatibilityProfile {
            model_checkpoint: ModelCheckpointId::from_manifest(
                format!("hanzo/{}", self.name.to_lowercase()),
                checkpoint_manifest.as_bytes(),
            ),
            tokenizer: TokenizerIdentity::from_vocab_bytes("llama3-chat-template", tokenizer_bytes),
            quant_format: self.quant_format.clone(),
            rope_pos_encoding: RopePosEncoding::new(RopeScheme::Llama3, 500_000.0, 1.0, 131_072),
            runtime_config: RuntimeConfigIdentity {
                block_size: self.block_size,
                attn_layers: self.attn_layers,
                gdn_layers: self.gdn_layers,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                gdn_state_dim: self.gdn_state_dim,
                runtime_flags_hash: Sha256::digest(b"gdn=fp32_unquantized;radix=v1;det=1"),
            },
        }
    }
}
