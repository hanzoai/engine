//! Exact HIP-1346 hybrid Transformer + Gated DeltaNet (GDN) model specifications,
//! compressed-weight residency accounting, and unclamped memory footprint calculations.
//!
//! All three model geometries (`4B-Hybrid`, `9B-Hybrid`, `27B-Hybrid`) match
//! `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478` byte-for-byte.

/// Number of bytes in 1 MiB (`2^20`).
pub const MIB: u64 = 1024 * 1024;
/// Number of bytes in 1 GiB (`2^30`).
pub const GIB: u64 = 1024 * 1024 * 1024;
/// Standard 128K long-context evaluation length (`131,072` tokens).
pub const CONTEXT_128K: usize = 131_072;

/// On-disk / in-memory weight storage representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// Uncompressed 16-bit bfloat16 weights (`2.0 B/param`).
    Bf16,
    /// Group-128 packed INT8 weights (`8b + FP16 scale` on 90% linear/MLP layers,
    /// FP16 on 10% sensitive embeddings/norms/head -> `1.128125 B/param`).
    Int8Packed,
    /// Group-128 packed INT4 weights (`4b + FP16 scale + FP16 zero` on 90% linear/MLP
    /// layers, FP16 on 10% sensitive embeddings/norms/head -> `0.678125 B/param`).
    Int4Packed,
}

impl WeightFormat {
    /// Stable lowercase identifier for reports and JSON receipts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Int8Packed => "int8_packed",
            Self::Int4Packed => "int4_packed",
        }
    }

    /// Exact average bytes per parameter including group-128 quantization metadata
    /// and unquantized embedding/norm/head weights (10% of parameters kept in FP16).
    pub fn effective_bytes_per_param(self) -> f64 {
        match self {
            Self::Bf16 => 2.0,
            // 0.90 * (1.0 + 4.0 / 128.0) + 0.10 * 2.0 = 1.128125 B/param
            Self::Int8Packed => 0.90 * (1.0 + 4.0 / 128.0) + 0.10 * 2.0,
            // 0.90 * (0.5 + 4.0 / 128.0) + 0.10 * 2.0 = 0.678125 B/param
            Self::Int4Packed => 0.90 * (0.5 + 4.0 / 128.0) + 0.10 * 2.0,
        }
    }

    /// Exact compression ratio relative to uncompressed BF16.
    pub fn compression_ratio_vs_bf16(self) -> f64 {
        2.0 / self.effective_bytes_per_param()
    }

    /// Dequantization compute/register overhead multiplier on GPU/CPU matvec kernels.
    pub fn dequant_overhead_factor(self) -> f64 {
        match self {
            Self::Bf16 => 1.00,
            Self::Int8Packed => 1.04,
            Self::Int4Packed => 1.08,
        }
    }
}

/// Exact fit regime of a model + context workload relative to a device's usable VRAM
/// or unified wired memory budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FitRegime {
    /// Total exact working set uses `< 78%` of usable device memory.
    FullyFits,
    /// Total exact working set uses `78%..=100%` of usable device memory (tight headroom).
    BarelyFits,
    /// Total exact working set exceeds `100%` of usable device memory (requires offload
    /// or compressed residency).
    ExceedsVram,
}

impl FitRegime {
    /// Stable lowercase identifier for reports and JSON receipts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullyFits => "fully_fits",
            Self::BarelyFits => "barely_fits",
            Self::ExceedsVram => "exceeds_vram",
        }
    }

    /// Classifies an exact unclamped memory demand against usable device memory.
    pub fn classify(demand_bytes: u64, usable_device_bytes: u64) -> Self {
        let ratio = demand_bytes as f64 / usable_device_bytes.max(1) as f64;
        if ratio > 1.0 {
            Self::ExceedsVram
        } else if ratio >= 0.78 {
            Self::BarelyFits
        } else {
            Self::FullyFits
        }
    }
}

/// Exact HIP-1346 hybrid Attention + Gated DeltaNet (GDN) model geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridModelSpec {
    /// Canonical model label (`4B-Hybrid`, `9B-Hybrid`, `27B-Hybrid`).
    pub label: &'static str,
    /// Number of standard softmax attention layers (holding paged FP16 KV cache).
    pub num_attention_layers: usize,
    /// Number of Gated DeltaNet recurrent layers (holding unquantized FP32 state).
    pub num_gdn_layers: usize,
    /// Number of KV heads per attention layer.
    pub num_kv_heads: usize,
    /// Head dimension for attention KV heads.
    pub head_dim: usize,
    /// Number of GDN recurrent heads per GDN layer.
    pub gdn_heads: usize,
    /// Number of FP32 state elements per GDN head.
    pub gdn_state_elems_per_head: usize,
    /// Model hidden dimension.
    pub hidden_dim: usize,
    /// SwiGLU / MoE FFN intermediate width per layer.
    pub intermediate_dim: usize,
    /// Logical neuron / expert blocks per layer tracked by the PowerInfer profiler.
    pub neuron_blocks_per_layer: usize,
    /// Total parameter count across all layers + embeddings.
    pub total_params: u64,
    /// Fraction of parameters belonging to MLP / MoE feed-forward blocks (eligible for
    /// sparse PowerInfer hot/cold neuron partitioning).
    pub mlp_param_fraction: f64,
    /// Fixed runtime activation + kernel workspace bytes (never clamped).
    pub runtime_workspace_bytes: u64,
}

impl HybridModelSpec {
    /// Canonical `4B-Hybrid` geometry matching `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-458`.
    pub const fn model_4b_hybrid() -> Self {
        Self {
            label: "4B-Hybrid",
            num_attention_layers: 9,
            num_gdn_layers: 27,
            num_kv_heads: 2,
            head_dim: 128,
            gdn_heads: 16,
            gdn_state_elems_per_head: 128 * 128,
            hidden_dim: 2560,
            intermediate_dim: 9216,
            neuron_blocks_per_layer: 144,
            total_params: 4_000_000_000,
            mlp_param_fraction: 0.70,
            runtime_workspace_bytes: 512 * MIB,
        }
    }

    /// Canonical `9B-Hybrid` geometry matching `hanzo-paged-attn/tests/kivi_kv_quant.rs:459-467`.
    pub const fn model_9b_hybrid() -> Self {
        Self {
            label: "9B-Hybrid",
            num_attention_layers: 10,
            num_gdn_layers: 30,
            num_kv_heads: 4,
            head_dim: 128,
            gdn_heads: 24,
            gdn_state_elems_per_head: 128 * 128,
            hidden_dim: 4096,
            intermediate_dim: 14336,
            neuron_blocks_per_layer: 224,
            total_params: 9_000_000_000,
            mlp_param_fraction: 0.70,
            runtime_workspace_bytes: 1280 * MIB,
        }
    }

    /// Canonical `27B-Hybrid` geometry matching `hanzo-paged-attn/tests/kivi_kv_quant.rs:468-476`.
    pub const fn model_27b_hybrid() -> Self {
        Self {
            label: "27B-Hybrid",
            num_attention_layers: 16,
            num_gdn_layers: 48,
            num_kv_heads: 4,
            head_dim: 256,
            gdn_heads: 32,
            gdn_state_elems_per_head: 128 * 256,
            hidden_dim: 5120,
            intermediate_dim: 24576,
            neuron_blocks_per_layer: 384,
            total_params: 27_000_000_000,
            mlp_param_fraction: 0.70,
            runtime_workspace_bytes: 1536 * MIB,
        }
    }

    /// Returns all three canonical HIP-1346 hybrid models in ascending scale order.
    pub const fn hip_1346_matrix() -> [Self; 3] {
        [
            Self::model_4b_hybrid(),
            Self::model_9b_hybrid(),
            Self::model_27b_hybrid(),
        ]
    }

    /// Total transformer + GDN layers (`num_attention_layers + num_gdn_layers`).
    pub const fn total_layers(&self) -> usize {
        self.num_attention_layers + self.num_gdn_layers
    }

    /// Exact unquantized FP32 GDN recurrent state bytes for a single active sequence.
    ///
    /// Matches `kivi_kv_quant.rs`: `27.0 MiB` (`4B`), `45.0 MiB` (`9B`), `192.0 MiB` (`27B`).
    pub const fn gdn_state_bytes(&self) -> u64 {
        (self.num_gdn_layers as u64)
            * (self.gdn_heads as u64)
            * (self.gdn_state_elems_per_head as u64)
            * 4
    }

    /// Exact unquantized FP32 GDN recurrent state in MiB.
    pub fn gdn_state_mib(&self) -> f64 {
        self.gdn_state_bytes() as f64 / MIB as f64
    }

    /// Exact FP16 K+V cache bytes per token across all attention layers.
    ///
    /// Matches `kivi_kv_quant.rs`: `9,216 B/tok` (`4B`), `20,480 B/tok` (`9B`),
    /// `65,536 B/tok` (`27B`).
    pub const fn kv_bytes_per_token(&self) -> u64 {
        (self.num_attention_layers as u64)
            * 2
            * (self.num_kv_heads as u64)
            * (self.head_dim as u64)
            * 2
    }

    /// Exact FP16 attention KV cache bytes for `context_tokens` tokens across `batch_size`
    /// concurrent sequences.
    pub const fn kv_cache_bytes(&self, context_tokens: usize, batch_size: usize) -> u64 {
        self.kv_bytes_per_token() * (context_tokens as u64) * (batch_size as u64)
    }

    /// Exact model weight bytes under `format` (never clamped).
    pub fn weight_bytes(&self, format: WeightFormat) -> u64 {
        (self.total_params as f64 * format.effective_bytes_per_param()).round() as u64
    }

    /// Exact per-layer weight bytes under `format`.
    pub fn layer_weight_bytes(&self, format: WeightFormat) -> u64 {
        self.weight_bytes(format) / (self.total_layers() as u64)
    }

    /// Exact non-MLP weight bytes (attention projections, GDN projections, embeddings, norms)
    /// that must remain GPU-resident under PowerInfer hot/cold partitioning.
    pub fn core_non_mlp_weight_bytes(&self, format: WeightFormat) -> u64 {
        let total = self.weight_bytes(format) as f64;
        (total * (1.0 - self.mlp_param_fraction)).round() as u64
    }

    /// Exact MLP / MoE weight bytes eligible for hot/cold neuron partitioning.
    pub fn mlp_weight_bytes(&self, format: WeightFormat) -> u64 {
        self.weight_bytes(format)
            .saturating_sub(self.core_non_mlp_weight_bytes(format))
    }

    /// Exact unclamped full working-set memory demand (weights + KV + FP32 GDN + workspace).
    pub fn total_working_set_bytes(
        &self,
        format: WeightFormat,
        context_tokens: usize,
        batch_size: usize,
    ) -> u64 {
        self.weight_bytes(format)
            + self.kv_cache_bytes(context_tokens, batch_size)
            + self.gdn_state_bytes() * (batch_size as u64)
            + self.runtime_workspace_bytes
    }

    /// Approximate FLOPs per token for a single forward pass (`2 * total_params` dense MACs
    /// plus attention & GDN state FLOPs).
    pub fn flops_per_token(&self, context_tokens: usize) -> f64 {
        let linear_flops = 2.0 * self.total_params as f64;
        let attn_flops = 4.0
            * (self.num_attention_layers as f64)
            * (self.num_kv_heads as f64)
            * (self.head_dim as f64)
            * (context_tokens as f64);
        let gdn_flops = 4.0
            * (self.num_gdn_layers as f64)
            * (self.gdn_heads as f64)
            * (self.gdn_state_elems_per_head as f64);
        linear_flops + attn_flops + gdn_flops
    }
}
