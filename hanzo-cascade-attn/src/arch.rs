// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Per-architecture hardware profiles for strict, non-extrapolated serving evaluation.
//!
//! Each target architecture is modeled independently with its own memory subsystem,
//! sustained bandwidth, tensor-core throughput, asynchronous pipeline capabilities,
//! VRAM capacity limit, and power envelope:
//! - [`HardwareArch::HopperH100Sxm5`]: 80 GiB HBM3 (`3,350 GB/s`, `989 TFLOPS` BF16)
//! - [`HardwareArch::BlackwellB200`]: 192 GiB HBM3e (`8,000 GB/s`, `2,250 TFLOPS` BF16)
//! - [`HardwareArch::GraceBlackwellGb10`]: 128 GiB Unified LPDDR5X (`273 GB/s`, `250 TFLOPS` BF16)
//! - [`HardwareArch::ConsumerRtx4090`]: 24 GiB GDDR6X (`1,008 GB/s`, `330 TFLOPS` BF16)
//!
//! ## Per-Architecture Honesty Rule
//! Blackwell B200 numbers are NEVER extrapolated onto Grace Blackwell GB10 or Consumer RTX.
//! GB10's `273 GB/s` unified LPDDR5X subsystem gives it a `915.8 FLOP/byte` ridge point
//! (over `3.2x` more bandwidth-bound than B200's `281.3 FLOP/byte`), so compute-pipelined
//! attention kernels that deliver large prefill gains on B200 produce minimal or negative
//! end-to-end gain on bandwidth-bound GB10 hybrid workloads.

/// Supported hardware architectures evaluated independently in the HIP-1346 scorecard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareArch {
    /// Hopper H100-SXM5 80 GiB HBM3 datacenter GPU with asynchronous WGMMA/TMA pipeline.
    HopperH100Sxm5,
    /// Blackwell B200 192 GiB HBM3e datacenter GPU with 5th-gen asynchronous tensor-memory
    /// dual-warp pipelining and overlapped exponential/softmax execution.
    BlackwellB200,
    /// Grace Blackwell GB10 128 GiB unified LPDDR5X desktop/edge SoC (`273 GB/s` memory
    /// bandwidth — severely bandwidth-bound compared to datacenter HBM3e).
    GraceBlackwellGb10,
    /// Consumer RTX 4090 24 GiB GDDR6X desktop GPU with strict 24 GiB VRAM ceiling
    /// (lacks Hopper/Blackwell asynchronous tensor-memory hardware pipelines).
    ConsumerRtx4090,
}

impl HardwareArch {
    /// Returns all four hardware architectures in canonical report order.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::HopperH100Sxm5,
            Self::BlackwellB200,
            Self::GraceBlackwellGb10,
            Self::ConsumerRtx4090,
        ]
    }

    /// Stable machine-readable identifier for JSON artifacts.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::HopperH100Sxm5 => "hopper_h100_sxm5",
            Self::BlackwellB200 => "blackwell_b200",
            Self::GraceBlackwellGb10 => "grace_blackwell_gb10",
            Self::ConsumerRtx4090 => "consumer_rtx_4090",
        }
    }

    /// Explicit analytical provenance tag naming the exact architecture modeled.
    #[must_use]
    pub const fn provenance_tag(self) -> &'static str {
        match self {
            Self::HopperH100Sxm5 => "analytical_gpu_hopper_h100_sxm5",
            Self::BlackwellB200 => "analytical_gpu_blackwell_b200",
            Self::GraceBlackwellGb10 => "analytical_gpu_grace_blackwell_gb10",
            Self::ConsumerRtx4090 => "analytical_gpu_consumer_rtx_4090",
        }
    }

    /// Returns the complete hardware performance and capacity profile for `self`.
    #[must_use]
    pub const fn profile(self) -> HardwareProfile {
        match self {
            Self::HopperH100Sxm5 => HardwareProfile {
                arch: Self::HopperH100Sxm5,
                display_name: "Hopper H100-SXM5 80GiB",
                memory_kind: "HBM3",
                driver_label: "CUDA 12.6 (R560)",
                vram_capacity_bytes: 80 * 1024 * 1024 * 1024,
                peak_bandwidth_gb_s: 3_350.0,
                sustained_bw_efficiency: 0.80,
                peak_bf16_tflops: 989.0,
                gemm_compute_efficiency: 0.58,
                baseline_attn_compute_efficiency: 0.42,
                warp_pipelined_attn_compute_efficiency: 0.64,
                kernel_launch_us: 4.8,
                cascade_merge_overhead_us: 18.0,
                ragged_dispatch_overhead_us: 8.0,
                warp_pipeline_setup_us: 20.0,
                warp_pipeline_extra_staging_bytes_per_layer: 64 * 1024,
                tdp_watts: 700.0,
                idle_watts: 110.0,
                supports_tma_wgmma: true,
                supports_tmem_dual_warp_pipeline: false,
                is_bandwidth_bound_unified_memory: false,
            },
            Self::BlackwellB200 => HardwareProfile {
                arch: Self::BlackwellB200,
                display_name: "Blackwell B200 192GiB",
                memory_kind: "HBM3e",
                driver_label: "CUDA 13.0 (R580)",
                vram_capacity_bytes: 192 * 1024 * 1024 * 1024,
                peak_bandwidth_gb_s: 8_000.0,
                sustained_bw_efficiency: 0.82,
                peak_bf16_tflops: 2_250.0,
                gemm_compute_efficiency: 0.62,
                baseline_attn_compute_efficiency: 0.44,
                warp_pipelined_attn_compute_efficiency: 0.78,
                kernel_launch_us: 4.2,
                cascade_merge_overhead_us: 15.0,
                ragged_dispatch_overhead_us: 7.0,
                warp_pipeline_setup_us: 14.0,
                warp_pipeline_extra_staging_bytes_per_layer: 48 * 1024,
                tdp_watts: 1_000.0,
                idle_watts: 145.0,
                supports_tma_wgmma: true,
                supports_tmem_dual_warp_pipeline: true,
                is_bandwidth_bound_unified_memory: false,
            },
            Self::GraceBlackwellGb10 => HardwareProfile {
                arch: Self::GraceBlackwellGb10,
                display_name: "Grace Blackwell GB10 128GiB",
                memory_kind: "LPDDR5X Unified",
                driver_label: "CUDA 13.0 (R580-aarch64)",
                vram_capacity_bytes: 128 * 1024 * 1024 * 1024,
                peak_bandwidth_gb_s: 273.0,
                sustained_bw_efficiency: 0.78,
                peak_bf16_tflops: 250.0,
                gemm_compute_efficiency: 0.48,
                baseline_attn_compute_efficiency: 0.36,
                warp_pipelined_attn_compute_efficiency: 0.48,
                kernel_launch_us: 7.5,
                cascade_merge_overhead_us: 32.0,
                ragged_dispatch_overhead_us: 14.0,
                warp_pipeline_setup_us: 68.0,
                warp_pipeline_extra_staging_bytes_per_layer: 512 * 1024,
                tdp_watts: 140.0,
                idle_watts: 28.0,
                supports_tma_wgmma: true,
                supports_tmem_dual_warp_pipeline: true,
                is_bandwidth_bound_unified_memory: true,
            },
            Self::ConsumerRtx4090 => HardwareProfile {
                arch: Self::ConsumerRtx4090,
                display_name: "Consumer RTX 4090 24GiB",
                memory_kind: "GDDR6X",
                driver_label: "CUDA 12.6 (R560)",
                vram_capacity_bytes: 24 * 1024 * 1024 * 1024,
                peak_bandwidth_gb_s: 1_008.0,
                sustained_bw_efficiency: 0.78,
                peak_bf16_tflops: 330.0,
                gemm_compute_efficiency: 0.54,
                baseline_attn_compute_efficiency: 0.40,
                warp_pipelined_attn_compute_efficiency: 0.46,
                kernel_launch_us: 6.2,
                cascade_merge_overhead_us: 24.0,
                ragged_dispatch_overhead_us: 11.0,
                warp_pipeline_setup_us: 28.0,
                warp_pipeline_extra_staging_bytes_per_layer: 128 * 1024,
                tdp_watts: 450.0,
                idle_watts: 45.0,
                supports_tma_wgmma: false,
                supports_tmem_dual_warp_pipeline: false,
                is_bandwidth_bound_unified_memory: false,
            },
        }
    }
}

/// Complete physical specification for one target GPU/SoC architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareProfile {
    /// Architecture enum identifier.
    pub arch: HardwareArch,
    /// Human-readable hardware name.
    pub display_name: &'static str,
    /// Memory technology label (`"HBM3"`, `"HBM3e"`, `"LPDDR5X Unified"`, `"GDDR6X"`).
    pub memory_kind: &'static str,
    /// Reference driver/runtime label recorded in every scorecard.
    pub driver_label: &'static str,
    /// Total accessible device/unified memory capacity in bytes (hard fit ceiling).
    pub vram_capacity_bytes: u64,
    /// Peak memory bandwidth in GB/s (`10^9` bytes/s).
    pub peak_bandwidth_gb_s: f64,
    /// Sustained fraction of peak memory bandwidth achieved under steady streaming.
    pub sustained_bw_efficiency: f64,
    /// Peak dense BF16 tensor-core throughput in TFLOPS (`10^12` FLOP/s).
    pub peak_bf16_tflops: f64,
    /// Sustained efficiency for dense GEMM projections (QKV, O, GDN, SwiGLU MLP).
    pub gemm_compute_efficiency: f64,
    /// Sustained attention tensor-core efficiency for standard tiled online-softmax paged attention.
    pub baseline_attn_compute_efficiency: f64,
    /// Sustained attention tensor-core efficiency for warp-specialized async pipelined attention.
    pub warp_pipelined_attn_compute_efficiency: f64,
    /// Per-kernel host dispatch latency in microseconds.
    pub kernel_launch_us: f64,
    /// Additional launch + synchronization overhead for the two-level cascade LSE merge per attention layer (us).
    pub cascade_merge_overhead_us: f64,
    /// Additional metadata/indptr dispatch overhead for fused ragged attention per attention layer (us).
    pub ragged_dispatch_overhead_us: f64,
    /// Extra descriptor/barrier/workspace setup latency for warp-specialized async pipeline per attention layer (us).
    pub warp_pipeline_setup_us: f64,
    /// Additional per-layer workspace staging bytes touched by heavy async pipeline kernels.
    pub warp_pipeline_extra_staging_bytes_per_layer: u64,
    /// Thermal design power (Watts) under full compute saturation.
    pub tdp_watts: f64,
    /// Base power (Watts) under memory/launch-limited execution.
    pub idle_watts: f64,
    /// Whether hardware exposes Hopper-class async tensor memory copy + warp-group MMA.
    pub supports_tma_wgmma: bool,
    /// Whether hardware exposes Blackwell-class 5th-gen tensor-memory dual-warp pipelining.
    pub supports_tmem_dual_warp_pipeline: bool,
    /// Whether the architecture uses bandwidth-constrained unified LPDDR5X memory (e.g. GB10).
    pub is_bandwidth_bound_unified_memory: bool,
}

impl HardwareProfile {
    /// Total accessible memory capacity in MiB (`bytes / 2^20`).
    #[must_use]
    pub fn vram_capacity_mib(&self) -> f64 {
        (self.vram_capacity_bytes as f64) / (1024.0 * 1024.0)
    }

    /// Sustained memory bandwidth in bytes/second.
    #[must_use]
    pub fn sustained_bandwidth_bytes_per_s(&self) -> f64 {
        self.peak_bandwidth_gb_s * 1.0e9 * self.sustained_bw_efficiency
    }

    /// Sustained dense GEMM throughput in FLOP/second.
    #[must_use]
    pub fn sustained_gemm_flops_per_s(&self) -> f64 {
        self.peak_bf16_tflops * 1.0e12 * self.gemm_compute_efficiency
    }

    /// Sustained baseline attention compute throughput in FLOP/second.
    #[must_use]
    pub fn sustained_baseline_attn_flops_per_s(&self) -> f64 {
        self.peak_bf16_tflops * 1.0e12 * self.baseline_attn_compute_efficiency
    }

    /// Sustained warp-pipelined attention compute throughput in FLOP/second.
    #[must_use]
    pub fn sustained_pipelined_attn_flops_per_s(&self) -> f64 {
        self.peak_bf16_tflops * 1.0e12 * self.warp_pipelined_attn_compute_efficiency
    }

    /// Arithmetic-intensity ridge point in FLOP/byte (`peak_tflops * 1e3 / peak_bw_gb_s`).
    #[must_use]
    pub fn ridge_point_flops_per_byte(&self) -> f64 {
        (self.peak_bf16_tflops * 1_000.0) / self.peak_bandwidth_gb_s
    }

    /// Whether adaptive dispatch should enable heavy warp-specialized async pipelining
    /// on compute-bound prefill tiles for this architecture.
    ///
    /// Bandwidth-bound unified-memory SoCs (`GraceBlackwellGb10`) disable heavy async
    /// workspace staging by default because LPDDR5X bandwidth and setup overhead erase
    /// isolated attention TFLOPS gains on hybrid models.
    #[must_use]
    pub const fn prefers_async_pipeline_on_compute_prefill(&self) -> bool {
        (self.supports_tma_wgmma || self.supports_tmem_dual_warp_pipeline)
            && !self.is_bandwidth_bound_unified_memory
    }

    /// Estimates average power draw (Watts) given compute-to-total time fraction `compute_frac`.
    #[must_use]
    pub fn estimated_power_watts(&self, compute_frac: f64) -> f64 {
        let clamped = compute_frac.clamp(0.0, 1.0);
        // Memory-bound workloads draw ~62% of TDP above idle; compute-bound draw ~95% of TDP.
        let load_frac = 0.62 + 0.33 * clamped;
        self.idle_watts + (self.tdp_watts - self.idle_watts) * load_frac
    }
}
