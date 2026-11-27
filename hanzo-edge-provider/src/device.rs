// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Edge/mobile unified-memory hardware profiles and canonical HIP-1346/1350
//! `4B / 9B / 27B` hybrid Attention + Gated DeltaNet (GDN) model residency and
//! bandwidth accounting.
//!
//! ## Resident vs Streamed Weight Accounting
//! - Unified-memory residency capacity counts the full input-embedding table
//!   (`vocab_size * d_model * 2` bytes in BF16) plus quantized non-embedding
//!   parameters, active attention KV cache, unquantized FP32 GDN recurrent state,
//!   and runtime activation workspace.
//! - Per-step decode bandwidth streams non-embedding weights at the active weight
//!   precision plus a single-token `embed_tokens` row lookup (`d_model * 2` bytes),
//!   active attention KV cache read, and GDN state read/write (`2 * gdn_state_bytes`).

use std::fmt;

/// Bytes in one GiB (`1024^3`).
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Bytes in one MiB (`1024^2`).
pub const MIB: f64 = 1024.0 * 1024.0;

/// Weight storage precision for non-embedding transformer/GDN tensors.
///
/// The input-embedding lookup table (`vocab_size * d_model * 2`) stays resident in
/// BF16 to preserve token-representation fidelity and avoid dequantization jitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightPrecision {
    /// Group-128 INT4 (`4.5` effective bits/param = `9/16` bytes/param incl. FP16 scales/zeros).
    Int4,
    /// Group-128 INT8 (`8.5` effective bits/param = `17/16` bytes/param incl. FP16 scales).
    Int8,
    /// Native unquantized BF16 (`16` bits/param = `2` bytes/param).
    Bf16,
}

impl WeightPrecision {
    /// Canonical wire/manifest identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Bf16 => "bf16",
        }
    }

    /// Parses a wire/manifest precision label.
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "int4" => Some(Self::Int4),
            "int8" => Some(Self::Int8),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Numerator over 16 for exact integer byte accounting per non-embedding parameter.
    #[must_use]
    pub const fn sixteenths_of_byte_per_param(self) -> u64 {
        match self {
            Self::Int4 => 9,  // 0.5625 B/param (4-bit weight + group-128 FP16 scale+zero)
            Self::Int8 => 17, // 1.0625 B/param (8-bit weight + group-128 FP16 scale)
            Self::Bf16 => 32, // 2.0000 B/param
        }
    }

    /// Effective bytes per non-embedding parameter as `f64`.
    #[must_use]
    pub const fn bytes_per_param(self) -> f64 {
        (self.sixteenths_of_byte_per_param() as f64) / 16.0
    }
}

impl fmt::Display for WeightPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Canonical HIP-1346/1350 hybrid Attention + Gated DeltaNet (GDN) model geometry.
///
/// Exact layer counts, KV head counts, head sizes, and unquantized FP32 GDN state sizes
/// match `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridModelSpec {
    /// Canonical model label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub label: &'static str,
    /// Number of full attention layers holding paged KV caches.
    pub num_attention_layers: usize,
    /// Number of Gated DeltaNet (GDN) recurrent layers holding fixed FP32 state per sequence.
    pub num_gdn_layers: usize,
    /// Model hidden dimension (`d_model`).
    pub d_model: usize,
    /// SwiGLU MLP intermediate dimension (`d_ff`).
    pub d_ff: usize,
    /// Vocabulary size (`151,936`).
    pub vocab_size: usize,
    /// Number of attention query heads.
    pub num_q_heads: usize,
    /// Number of attention key/value heads (GQA).
    pub num_kv_heads: usize,
    /// Attention head dimension (`head_dim`).
    pub head_dim: usize,
    /// Number of GDN recurrent heads per GDN layer.
    pub gdn_heads: usize,
    /// Number of FP32 state elements per GDN head (`d_k * d_v`).
    pub gdn_state_elems_per_head: usize,
    /// Exact unquantized FP32 GDN recurrent state per active sequence in MiB.
    pub gdn_state_mib: f64,
}

impl HybridModelSpec {
    /// Returns the canonical 3-model HIP-1346/1350 hybrid matrix.
    #[must_use]
    pub const fn canonical_matrix() -> [Self; 3] {
        [Self::hybrid_4b(), Self::hybrid_9b(), Self::hybrid_27b()]
    }

    /// Canonical `4B-Hybrid`: 9 Attn + 27 GDN, `num_kv_heads = 2`, `27.0 MiB` GDN state.
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
            gdn_state_mib: 27.0,
        }
    }

    /// Canonical `9B-Hybrid`: 10 Attn + 30 GDN, `num_kv_heads = 4`, `45.0 MiB` GDN state.
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
            gdn_state_mib: 45.0,
        }
    }

    /// Canonical `27B-Hybrid`: 16 Attn + 48 GDN, `num_kv_heads = 4`, `192.0 MiB` GDN state.
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
            gdn_state_mib: 192.0,
        }
    }

    /// Looks up a canonical hybrid model spec by label.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "4B-Hybrid" => Some(Self::hybrid_4b()),
            "9B-Hybrid" => Some(Self::hybrid_9b()),
            "27B-Hybrid" => Some(Self::hybrid_27b()),
            _ => None,
        }
    }

    /// Total decoder layers (`num_attention_layers + num_gdn_layers`).
    #[must_use]
    pub const fn total_layers(&self) -> usize {
        self.num_attention_layers + self.num_gdn_layers
    }

    /// Exact total parameter count derived from model geometry (including full input
    /// embedding table and output LM head).
    #[must_use]
    pub const fn total_params(&self) -> u64 {
        let total_layers = self.total_layers() as u64;
        let d = self.d_model as u64;
        let ff = self.d_ff as u64;
        let vocab = self.vocab_size as u64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u64;

        let embedding = vocab * d;
        let lm_head = vocab * d;
        let attn_qkv = (self.num_attention_layers as u64) * (d + 2 * kv_dim) * d;
        let attn_out = (self.num_attention_layers as u64) * d * d;
        let gdn_recurrent = (self.num_gdn_layers as u64) * 4 * d * d;
        let mlp_gate_up = total_layers * 2 * ff * d;
        let mlp_down = total_layers * d * ff;

        embedding + lm_head + attn_qkv + attn_out + gdn_recurrent + mlp_gate_up + mlp_down
    }

    /// Input-embedding table parameter count (`vocab_size * d_model`).
    #[must_use]
    pub const fn embedding_table_params(&self) -> u64 {
        (self.vocab_size as u64) * (self.d_model as u64)
    }

    /// Non-embedding parameter count streamed during every forward step.
    #[must_use]
    pub const fn non_embedding_params(&self) -> u64 {
        self.total_params() - self.embedding_table_params()
    }

    /// Full resident BF16 input-embedding table size in bytes (`vocab_size * d_model * 2`),
    /// charged against unified-memory / VRAM capacity.
    #[must_use]
    pub const fn embedding_table_bytes_bf16(&self) -> u64 {
        self.embedding_table_params() * 2
    }

    /// Single-token `embed_tokens` row lookup size in bytes (`d_model * 2`),
    /// charged against per-step decode memory bandwidth.
    #[must_use]
    pub const fn embedding_row_bytes_bf16(&self) -> u64 {
        (self.d_model as u64) * 2
    }

    /// Quantized non-embedding weight storage in bytes at `precision`.
    #[must_use]
    pub const fn non_embedding_weight_bytes(&self, precision: WeightPrecision) -> u64 {
        (self.non_embedding_params() * precision.sixteenths_of_byte_per_param()).div_ceil(16)
    }

    /// Full resident model weight storage in bytes at `precision`
    /// (`embedding_table_bytes_bf16 + non_embedding_weight_bytes(precision)`).
    #[must_use]
    pub const fn resident_weight_bytes(&self, precision: WeightPrecision) -> u64 {
        self.embedding_table_bytes_bf16() + self.non_embedding_weight_bytes(precision)
    }

    /// Per-step streamed weight bytes during single-token decode at `precision`
    /// (`embedding_row_bytes_bf16 + non_embedding_weight_bytes(precision)`).
    ///
    /// Explicitly excludes the full `vocab_size * d_model * 2` embedding table because
    /// autoregressive decode only gathers one token row per step.
    #[must_use]
    pub const fn streamed_weight_bytes_per_step(&self, precision: WeightPrecision) -> u64 {
        self.embedding_row_bytes_bf16() + self.non_embedding_weight_bytes(precision)
    }

    /// Unquantized FP32 GDN recurrent state size per active sequence in bytes.
    #[must_use]
    pub const fn gdn_state_bytes(&self) -> u64 {
        (self.num_gdn_layers as u64)
            * (self.gdn_heads as u64)
            * (self.gdn_state_elems_per_head as u64)
            * 4
    }

    /// Exact BF16 attention KV cache bytes for one sequence of length `seq_len`.
    #[must_use]
    pub const fn attention_kv_bytes_bf16(&self, seq_len: usize) -> u64 {
        (self.num_attention_layers as u64)
            * (seq_len as u64)
            * 2
            * (self.num_kv_heads as u64)
            * (self.head_dim as u64)
            * 2
    }

    /// Total memory bytes streamed per decode token at `seq_len` (weights + single embed row
    /// + attention KV read + GDN state read/write).
    #[must_use]
    pub const fn total_streamed_bytes_per_decode_token(
        &self,
        precision: WeightPrecision,
        seq_len: usize,
    ) -> u64 {
        self.streamed_weight_bytes_per_step(precision)
            + self.attention_kv_bytes_bf16(seq_len)
            + 2 * self.gdn_state_bytes()
    }

    /// Approximate FLOPs per token (`2 * non_embedding_params + attention dot-products`).
    #[must_use]
    pub const fn flops_per_token(&self, seq_len: usize) -> u64 {
        let linear_flops = 2 * self.non_embedding_params();
        let attn_flops = 4
            * (self.num_attention_layers as u64)
            * (self.num_q_heads as u64)
            * (self.head_dim as u64)
            * (seq_len as u64);
        linear_flops + attn_flops
    }
}

/// Supported edge and mobile hardware device classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeDeviceClass {
    /// Apple Silicon M4 Pro laptop/workstation with 24 GiB unified memory (`273 GB/s`).
    AppleSiliconM4Pro24G,
    /// Apple Silicon M4 Max laptop/studio with 64 GiB unified memory (`546 GB/s`).
    AppleSiliconM4Max64G,
    /// AMD Ryzen AI Max+ 395 (Strix Halo) with 128 GiB LPDDR5X unified memory (`256 GB/s`).
    AmdStrixHalo128G,
    /// Mobile-class SoC (flagship phone/tablet) with 12 GiB LPDDR5X (`68 GB/s`).
    MobileClass12G,
}

impl EdgeDeviceClass {
    /// Returns all four canonical edge/mobile hardware profiles.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::AppleSiliconM4Pro24G,
            Self::AppleSiliconM4Max64G,
            Self::AmdStrixHalo128G,
            Self::MobileClass12G,
        ]
    }

    /// Canonical wire/manifest identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AppleSiliconM4Pro24G => "apple_silicon_m4_pro_24g",
            Self::AppleSiliconM4Max64G => "apple_silicon_m4_max_64g",
            Self::AmdStrixHalo128G => "amd_strix_halo_128g",
            Self::MobileClass12G => "mobile_class_12g",
        }
    }

    /// Parses a wire/manifest device class identifier.
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "apple_silicon_m4_pro_24g" => Some(Self::AppleSiliconM4Pro24G),
            "apple_silicon_m4_max_64g" => Some(Self::AppleSiliconM4Max64G),
            "amd_strix_halo_128g" => Some(Self::AmdStrixHalo128G),
            "mobile_class_12g" => Some(Self::MobileClass12G),
            _ => None,
        }
    }

    /// Returns the full hardware & thermal profile for this device class.
    #[must_use]
    pub const fn profile(self) -> EdgeDeviceProfile {
        match self {
            Self::AppleSiliconM4Pro24G => EdgeDeviceProfile {
                class: self,
                label: "Apple Silicon M4 Pro (24 GiB Unified)",
                unified_memory_gib: 24.0,
                user_reserved_memory_gib: 8.0,
                memory_bandwidth_gbs: 273.0,
                bandwidth_efficiency: 0.78,
                compute_tflops_fp16: 22.0,
                compute_efficiency: 0.52,
                sustained_ac_watts: 30.0,
                battery_cap_watts: 18.0,
                idle_watts: 4.5,
                soft_throttle_temp_c: 85.0,
                hard_throttle_temp_c: 90.0,
                withdraw_temp_c: 95.0,
                ambient_temp_c: 24.0,
                thermal_resistance_c_per_w: 2.35,
                thermal_capacitance_j_per_c: 42.0,
                battery_capacity_wh: 72.4,
                default_min_battery_pct: 25.0,
                runtime_workspace_base_mib: 512.0,
                hardware_capex_usd: 1999.0,
                is_mobile_skin_limited: false,
            },
            Self::AppleSiliconM4Max64G => EdgeDeviceProfile {
                class: self,
                label: "Apple Silicon M4 Max (64 GiB Unified)",
                unified_memory_gib: 64.0,
                user_reserved_memory_gib: 16.0,
                memory_bandwidth_gbs: 546.0,
                bandwidth_efficiency: 0.80,
                compute_tflops_fp16: 48.0,
                compute_efficiency: 0.55,
                sustained_ac_watts: 55.0,
                battery_cap_watts: 30.0,
                idle_watts: 7.0,
                soft_throttle_temp_c: 85.0,
                hard_throttle_temp_c: 91.0,
                withdraw_temp_c: 96.0,
                ambient_temp_c: 24.0,
                thermal_resistance_c_per_w: 1.28,
                thermal_capacitance_j_per_c: 65.0,
                battery_capacity_wh: 100.0,
                default_min_battery_pct: 25.0,
                runtime_workspace_base_mib: 512.0,
                hardware_capex_usd: 3899.0,
                is_mobile_skin_limited: false,
            },
            Self::AmdStrixHalo128G => EdgeDeviceProfile {
                class: self,
                label: "AMD Strix Halo Ryzen AI Max+ 395 (128 GiB Unified)",
                unified_memory_gib: 128.0,
                user_reserved_memory_gib: 32.0,
                memory_bandwidth_gbs: 256.0,
                bandwidth_efficiency: 0.76,
                compute_tflops_fp16: 59.0,
                compute_efficiency: 0.50,
                sustained_ac_watts: 85.0,
                battery_cap_watts: 45.0,
                idle_watts: 9.0,
                soft_throttle_temp_c: 82.0,
                hard_throttle_temp_c: 89.0,
                withdraw_temp_c: 95.0,
                ambient_temp_c: 24.0,
                thermal_resistance_c_per_w: 0.76,
                thermal_capacitance_j_per_c: 95.0,
                battery_capacity_wh: 99.9,
                default_min_battery_pct: 20.0,
                runtime_workspace_base_mib: 512.0,
                hardware_capex_usd: 2499.0,
                is_mobile_skin_limited: false,
            },
            Self::MobileClass12G => EdgeDeviceProfile {
                class: self,
                label: "Mobile-Class SoC (12 GiB LPDDR5X)",
                unified_memory_gib: 12.0,
                user_reserved_memory_gib: 5.5,
                memory_bandwidth_gbs: 68.0,
                bandwidth_efficiency: 0.72,
                compute_tflops_fp16: 6.5,
                compute_efficiency: 0.45,
                sustained_ac_watts: 6.0,
                battery_cap_watts: 3.5,
                idle_watts: 0.8,
                soft_throttle_temp_c: 42.0,
                hard_throttle_temp_c: 44.0,
                withdraw_temp_c: 46.0,
                ambient_temp_c: 24.0,
                thermal_resistance_c_per_w: 3.80,
                thermal_capacitance_j_per_c: 18.0,
                battery_capacity_wh: 18.5,
                default_min_battery_pct: 30.0,
                runtime_workspace_base_mib: 256.0,
                hardware_capex_usd: 999.0,
                is_mobile_skin_limited: true,
            },
        }
    }
}

impl fmt::Display for EdgeDeviceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Complete physical memory, compute, power, and thermal envelope for an edge/mobile device.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeDeviceProfile {
    /// Device class identifier.
    pub class: EdgeDeviceClass,
    /// Human-readable hardware description.
    pub label: &'static str,
    /// Total installed unified physical RAM in GiB.
    pub unified_memory_gib: f64,
    /// Mandatory OS + user foreground reserved memory headroom in GiB (never invaded by inference).
    pub user_reserved_memory_gib: f64,
    /// Peak unified memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// Achievable sustained decode bandwidth fraction `(0, 1]`.
    pub bandwidth_efficiency: f64,
    /// Peak FP16/BF16 matrix throughput in TFLOPS.
    pub compute_tflops_fp16: f64,
    /// Achievable sustained prefill compute fraction `(0, 1]`.
    pub compute_efficiency: f64,
    /// Maximum sustained power budget on AC wall power in Watts.
    pub sustained_ac_watts: f64,
    /// Maximum power budget on battery or quiet laptop mode in Watts.
    pub battery_cap_watts: f64,
    /// Baseline idle system power draw in Watts.
    pub idle_watts: f64,
    /// Soft DVFS thermal throttle onset temperature in °C (SoC die or mobile skin).
    pub soft_throttle_temp_c: f64,
    /// Hard DVFS thermal throttle onset temperature in °C.
    pub hard_throttle_temp_c: f64,
    /// Automatic market withdrawal temperature ceiling in °C.
    pub withdraw_temp_c: f64,
    /// Ambient room temperature baseline in °C.
    pub ambient_temp_c: f64,
    /// Effective package-to-ambient thermal resistance `R_th` in °C/W.
    pub thermal_resistance_c_per_w: f64,
    /// Effective lumped thermal capacitance `C_th` in J/°C.
    pub thermal_capacitance_j_per_c: f64,
    /// Battery pack capacity in Watt-hours (Wh).
    pub battery_capacity_wh: f64,
    /// Default owner minimum battery state-of-charge floor (%).
    pub default_min_battery_pct: f64,
    /// Fixed runtime graph/command-buffer workspace overhead in MiB.
    pub runtime_workspace_base_mib: f64,
    /// Representative consumer retail purchase price in USD (for amortized TCO analysis).
    pub hardware_capex_usd: f64,
    /// True if thermal thresholds represent mobile enclosure skin temperature rather than die temp.
    pub is_mobile_skin_limited: bool,
}

impl EdgeDeviceProfile {
    /// Maximum unified memory available for inference without invading user-reserved RAM.
    #[must_use]
    pub fn usable_memory_gib(&self) -> f64 {
        (self.unified_memory_gib - self.user_reserved_memory_gib).max(0.0)
    }

    /// Effective power budget given whether the machine is on AC or battery.
    #[must_use]
    pub const fn active_power_cap_watts(&self, on_ac_power: bool) -> f64 {
        if on_ac_power {
            self.sustained_ac_watts
        } else {
            self.battery_cap_watts
        }
    }

    /// Effective sustained memory bandwidth in bytes/sec under a given DVFS clock scale.
    #[must_use]
    pub fn effective_bandwidth_bytes_per_sec(&self, clock_scale: f64) -> f64 {
        self.memory_bandwidth_gbs * 1.0e9 * self.bandwidth_efficiency * clock_scale.clamp(0.0, 1.0)
    }

    /// Effective sustained compute throughput in FLOPs/sec under a given DVFS clock scale.
    #[must_use]
    pub fn effective_compute_flops_per_sec(&self, clock_scale: f64) -> f64 {
        self.compute_tflops_fp16 * 1.0e12 * self.compute_efficiency * clock_scale.clamp(0.0, 1.0)
    }
}

/// Centralized datacenter H100-SXM5 80 GiB reference baseline for 5-axis HIP-1350 comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct DatacenterH100Baseline {
    /// Human-readable label.
    pub label: &'static str,
    /// Usable HBM3 VRAM in GiB.
    pub vram_gib: f64,
    /// Peak HBM3 bandwidth in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// Sustained bandwidth utilization fraction.
    pub bandwidth_efficiency: f64,
    /// Peak FP16 tensor throughput in TFLOPS.
    pub compute_tflops_fp16: f64,
    /// Sustained prefill utilization fraction.
    pub compute_efficiency: f64,
    /// Board TDP in Watts.
    pub tdp_watts: f64,
    /// On-demand cloud rental price in USD/hour.
    pub hourly_rental_usd: f64,
    /// Typical active batch concurrency for shared cloud serving.
    pub typical_batch_size: usize,
}

impl Default for DatacenterH100Baseline {
    fn default() -> Self {
        Self {
            label: "Centralized H100-SXM5 80 GiB (Cloud Baseline)",
            vram_gib: 80.0,
            memory_bandwidth_gbs: 3350.0,
            bandwidth_efficiency: 0.78,
            compute_tflops_fp16: 989.0,
            compute_efficiency: 0.55,
            tdp_watts: 700.0,
            hourly_rental_usd: 3.20,
            typical_batch_size: 8,
        }
    }
}

/// Exact unclamped unified-memory residency verification report for `(device, model, precision, context)`.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryFitReport {
    /// Device class evaluated.
    pub device_class: EdgeDeviceClass,
    /// Canonical model label.
    pub model_label: &'static str,
    /// Weight precision evaluated.
    pub precision: WeightPrecision,
    /// Maximum context length reserved in KV cache.
    pub max_context_tokens: usize,
    /// Full resident BF16 input-embedding table size in GiB (`vocab_size * d_model * 2`).
    pub resident_embedding_gib: f64,
    /// Non-embedding weight size in GiB at `precision`.
    pub non_embedding_weights_gib: f64,
    /// Attention KV cache size in GiB for `max_context_tokens`.
    pub attention_kv_gib: f64,
    /// Unquantized FP32 GDN recurrent state size in GiB.
    pub gdn_state_gib: f64,
    /// Runtime activation and graph workspace in GiB.
    pub runtime_workspace_gib: f64,
    /// Total required unified memory in GiB (unclamped sum).
    pub total_required_gib: f64,
    /// Device usable unified memory cap in GiB (excluding user-reserved headroom).
    pub usable_memory_gib: f64,
    /// Remaining headroom (`usable_memory_gib - total_required_gib`); negative indicates OOM deficit.
    pub headroom_gib: f64,
    /// Whether the model fits strictly inside the advertised usable memory cap.
    pub fits: bool,
    /// Explicit rejection reason when `fits == false`.
    pub rejection_reason: Option<String>,
}

/// Evaluates unclamped unified-memory residency for an open-weight Zen hybrid model on an edge device.
///
/// Guarantees that oversized models (such as `27B-Hybrid` INT4 on `MobileClass12G`) are
/// cleanly rejected rather than clamped or allowed to OOM the user's laptop/phone.
#[must_use]
pub fn check_model_residency(
    profile: &EdgeDeviceProfile,
    model: &HybridModelSpec,
    precision: WeightPrecision,
    max_context_tokens: usize,
    override_usable_cap_gib: Option<f64>,
) -> MemoryFitReport {
    let embed_bytes = model.embedding_table_bytes_bf16() as f64;
    let non_embed_bytes = model.non_embedding_weight_bytes(precision) as f64;
    let kv_bytes = model.attention_kv_bytes_bf16(max_context_tokens) as f64;
    let gdn_bytes = model.gdn_state_bytes() as f64;
    // Runtime scratchpad: base driver/graph overhead + 4% of weights for activations/dequant buffers.
    let workspace_bytes =
        profile.runtime_workspace_base_mib * MIB + 0.04 * (embed_bytes + non_embed_bytes);

    let resident_embedding_gib = embed_bytes / GIB;
    let non_embedding_weights_gib = non_embed_bytes / GIB;
    let attention_kv_gib = kv_bytes / GIB;
    let gdn_state_gib = gdn_bytes / GIB;
    let runtime_workspace_gib = workspace_bytes / GIB;
    let total_required_gib = resident_embedding_gib
        + non_embedding_weights_gib
        + attention_kv_gib
        + gdn_state_gib
        + runtime_workspace_gib;

    // Never allow an advertised cap to exceed the physical usable budget after user reservation.
    let physical_usable = profile.usable_memory_gib();
    let usable_memory_gib = match override_usable_cap_gib {
        Some(cap) => cap.min(physical_usable).max(0.0),
        None => physical_usable,
    };

    let headroom_gib = usable_memory_gib - total_required_gib;
    let fits = headroom_gib >= 0.0;
    let rejection_reason = if fits {
        None
    } else {
        Some(format!(
            "InsufficientUnifiedMemory: {} ({}) at ctx={} requires {:.3} GiB > {:.3} GiB usable cap on {} (deficit {:.3} GiB; {:.1} GiB reserved for user/OS)",
            model.label,
            precision,
            max_context_tokens,
            total_required_gib,
            usable_memory_gib,
            profile.class,
            -headroom_gib,
            profile.user_reserved_memory_gib
        ))
    };

    MemoryFitReport {
        device_class: profile.class,
        model_label: model.label,
        precision,
        max_context_tokens,
        resident_embedding_gib,
        non_embedding_weights_gib,
        attention_kv_gib,
        gdn_state_gib,
        runtime_workspace_gib,
        total_required_gib,
        usable_memory_gib,
        headroom_gib,
        fits,
        rejection_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_matrix_matches_exact_gdn_state_sizes() {
        let m4 = HybridModelSpec::hybrid_4b();
        let m9 = HybridModelSpec::hybrid_9b();
        let m27 = HybridModelSpec::hybrid_27b();
        assert_eq!(m4.num_kv_heads, 2);
        assert_eq!(m9.num_kv_heads, 4);
        assert_eq!(m27.num_kv_heads, 4);
        assert!(((m4.gdn_state_bytes() as f64) / MIB - 27.0).abs() < 1e-9);
        assert!(((m9.gdn_state_bytes() as f64) / MIB - 45.0).abs() < 1e-9);
        assert!(((m27.gdn_state_bytes() as f64) / MIB - 192.0).abs() < 1e-9);
    }

    #[test]
    fn resident_vs_streamed_embedding_accounting_is_distinct() {
        let m9 = HybridModelSpec::hybrid_9b();
        assert_eq!(m9.embedding_table_bytes_bf16(), 151_936 * 4096 * 2);
        assert_eq!(m9.embedding_row_bytes_bf16(), 4096 * 2);
        assert!(
            m9.resident_weight_bytes(WeightPrecision::Int4)
                > m9.streamed_weight_bytes_per_step(WeightPrecision::Int4) + 1_000_000_000
        );
    }

    #[test]
    fn mobile_rejects_27b_int4_cleanly() {
        let mobile = EdgeDeviceClass::MobileClass12G.profile();
        let m27 = HybridModelSpec::hybrid_27b();
        let fit = check_model_residency(&mobile, &m27, WeightPrecision::Int4, 2048, None);
        assert!(!fit.fits);
        assert!(fit.total_required_gib > 14.0);
        assert!(fit.rejection_reason.is_some());
    }
}
