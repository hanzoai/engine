//! Heterogeneous hardware topology profiles for discrete PCIe GPUs and
//! zero-copy unified-memory accelerators (`hanzoai/engine#57`).

use crate::model::GIB;

/// Memory address-space architecture of the target machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryArchitecture {
    /// Separate GPU VRAM and CPU host DRAM connected via PCIe.
    /// Moving weights or activations between CPU and GPU requires explicit DMA copies.
    DiscretePcie,
    /// Shared physical LPDDR5X unified memory pool accessed coherently by CPU and GPU.
    /// Zero-copy views eliminate both duplicate residency and host<->device copy traffic.
    UnifiedZeroCopy,
}

impl MemoryArchitecture {
    /// Stable lowercase identifier for reports and JSON receipts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DiscretePcie => "discrete_pcie",
            Self::UnifiedZeroCopy => "unified_zero_copy",
        }
    }
}

/// Complete physical topology specification for a discrete or unified-memory inference host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardwareProfile {
    /// Stable identifier (e.g., `DiscreteGpu24G`, `UnifiedMemory24G`).
    pub id: &'static str,
    /// Human-readable hardware class description.
    pub description: &'static str,
    /// Discrete PCIe vs zero-copy unified memory.
    pub architecture: MemoryArchitecture,
    /// Physical GPU VRAM (or total shared unified memory pool).
    pub vram_bytes: u64,
    /// Usable GPU VRAM (or Metal/GPU wired unified memory cap after OS/driver reserve).
    pub usable_vram_bytes: u64,
    /// Physical host DRAM (equals `vram_bytes` on unified memory — NOT a separate pool).
    pub host_ram_bytes: u64,
    /// Usable host DRAM (equals `usable_vram_bytes` on unified memory).
    pub usable_host_ram_bytes: u64,
    /// Peak sustained GPU HBM/GDDR/LPDDR bandwidth in GB/s.
    pub gpu_mem_bw_gbs: f64,
    /// Effective sustained GPU BF16/FP16 tensor throughput in TFLOPS.
    pub gpu_compute_tflops: f64,
    /// Peak sustained CPU DRAM bandwidth in GB/s.
    pub cpu_mem_bw_gbs: f64,
    /// Effective sustained CPU vector/AMX compute throughput in TFLOPS.
    pub cpu_compute_tflops: f64,
    /// Sustained unidirectional CPU<->GPU interconnect bandwidth in GB/s (PCIe or unified bus).
    pub interconnect_bw_gbs: f64,
    /// One-way CPU<->GPU handoff / DMA launch latency in microseconds.
    pub interconnect_latency_us: f64,
    /// Sequential NVMe storage read bandwidth in GB/s.
    pub nvme_seq_bw_gbs: f64,
    /// Minor 4 KiB page-fault / page-table bind cost in microseconds.
    pub minor_fault_us: f64,
    /// Major 4 KiB NVMe page-fault latency (batched effective) in microseconds.
    pub major_fault_us: f64,
    /// GPU board TDP in watts.
    pub gpu_tdp_watts: f64,
    /// Host CPU package TDP in watts.
    pub cpu_tdp_watts: f64,
    /// Energy per byte transferred over host<->device copy path in picojoules/byte.
    /// Zero for true zero-copy pointer views on unified memory.
    pub copy_pj_per_byte: f64,
}

impl HardwareProfile {
    /// RTX 4090 / L4-class 24 GiB discrete GPU + 64 GiB DDR5 over PCIe Gen4 x16.
    ///
    /// On this topology at 128K context:
    /// - `4B-Hybrid` BF16 (`9.10 GiB`) **fully fits**
    /// - `9B-Hybrid` BF16 (`20.56 GiB`) **barely fits** (`22.50 GiB` usable)
    /// - `27B-Hybrid` (`59.98 GiB` BF16, `26.74 GiB` INT4) **exceeds VRAM**
    pub const fn discrete_gpu_24g() -> Self {
        Self {
            id: "DiscreteGpu24G",
            description: "24 GiB Discrete GPU (RTX 4090/L4 class) + 64 GiB DDR5, PCIe Gen4 x16",
            architecture: MemoryArchitecture::DiscretePcie,
            vram_bytes: 24 * GIB,
            usable_vram_bytes: 22 * GIB + 512 * 1024 * 1024, // 22.5 GiB
            host_ram_bytes: 64 * GIB,
            usable_host_ram_bytes: 56 * GIB,
            gpu_mem_bw_gbs: 1008.0,
            gpu_compute_tflops: 165.0,
            cpu_mem_bw_gbs: 85.0,
            cpu_compute_tflops: 2.4,
            interconnect_bw_gbs: 26.0,
            interconnect_latency_us: 18.0,
            nvme_seq_bw_gbs: 7.0,
            minor_fault_us: 1.8,
            major_fault_us: 12.0,
            gpu_tdp_watts: 420.0,
            cpu_tdp_watts: 125.0,
            copy_pj_per_byte: 32.0,
        }
    }

    /// H100-SXM5 80 GiB discrete GPU + 256 GiB DDR5 over PCIe Gen5 x16.
    pub const fn discrete_gpu_80g() -> Self {
        Self {
            id: "DiscreteGpu80G",
            description: "80 GiB Discrete GPU (H100-SXM5 class) + 256 GiB DDR5, PCIe Gen5 x16",
            architecture: MemoryArchitecture::DiscretePcie,
            vram_bytes: 80 * GIB,
            usable_vram_bytes: 76 * GIB,
            host_ram_bytes: 256 * GIB,
            usable_host_ram_bytes: 232 * GIB,
            gpu_mem_bw_gbs: 3350.0,
            gpu_compute_tflops: 989.0,
            cpu_mem_bw_gbs: 200.0,
            cpu_compute_tflops: 4.8,
            interconnect_bw_gbs: 52.0,
            interconnect_latency_us: 12.0,
            nvme_seq_bw_gbs: 12.0,
            minor_fault_us: 1.4,
            major_fault_us: 9.0,
            gpu_tdp_watts: 700.0,
            cpu_tdp_watts: 180.0,
            copy_pj_per_byte: 24.0,
        }
    }

    /// Apple M4 Pro 24 GiB unified memory (LPDDR5X 273 GB/s shared CPU/GPU pool).
    pub const fn unified_memory_24g() -> Self {
        Self {
            id: "UnifiedMemory24G",
            description: "24 GiB Unified Memory (Apple M4 Pro class, LPDDR5X 273 GB/s)",
            architecture: MemoryArchitecture::UnifiedZeroCopy,
            vram_bytes: 24 * GIB,
            usable_vram_bytes: 20 * GIB,
            host_ram_bytes: 24 * GIB,
            usable_host_ram_bytes: 20 * GIB,
            gpu_mem_bw_gbs: 273.0,
            gpu_compute_tflops: 18.0,
            cpu_mem_bw_gbs: 273.0,
            cpu_compute_tflops: 3.2,
            interconnect_bw_gbs: 273.0,
            interconnect_latency_us: 0.4,
            nvme_seq_bw_gbs: 6.5,
            minor_fault_us: 1.2,
            major_fault_us: 11.0,
            gpu_tdp_watts: 42.0,
            cpu_tdp_watts: 18.0,
            copy_pj_per_byte: 6.5,
        }
    }

    /// Apple M4 Max 64 GiB unified memory (LPDDR5X 546 GB/s shared CPU/GPU pool).
    pub const fn unified_memory_64g() -> Self {
        Self {
            id: "UnifiedMemory64G",
            description: "64 GiB Unified Memory (Apple M4 Max class, LPDDR5X 546 GB/s)",
            architecture: MemoryArchitecture::UnifiedZeroCopy,
            vram_bytes: 64 * GIB,
            usable_vram_bytes: 54 * GIB,
            host_ram_bytes: 64 * GIB,
            usable_host_ram_bytes: 54 * GIB,
            gpu_mem_bw_gbs: 546.0,
            gpu_compute_tflops: 36.0,
            cpu_mem_bw_gbs: 546.0,
            cpu_compute_tflops: 4.2,
            interconnect_bw_gbs: 546.0,
            interconnect_latency_us: 0.4,
            nvme_seq_bw_gbs: 7.4,
            minor_fault_us: 1.1,
            major_fault_us: 10.0,
            gpu_tdp_watts: 78.0,
            cpu_tdp_watts: 28.0,
            copy_pj_per_byte: 6.5,
        }
    }

    /// Returns all four canonical hardware topologies evaluated by HIP-1346 `#57`.
    pub const fn all_profiles() -> [Self; 4] {
        [
            Self::discrete_gpu_24g(),
            Self::discrete_gpu_80g(),
            Self::unified_memory_24g(),
            Self::unified_memory_64g(),
        ]
    }

    /// Returns `true` if the hardware supports true zero-copy shared CPU/GPU views.
    pub const fn supports_zero_copy(&self) -> bool {
        matches!(self.architecture, MemoryArchitecture::UnifiedZeroCopy)
    }

    /// Exact total physical memory across the machine (never double-counts unified RAM).
    pub const fn total_physical_memory_bytes(&self) -> u64 {
        match self.architecture {
            MemoryArchitecture::DiscretePcie => self.vram_bytes + self.host_ram_bytes,
            MemoryArchitecture::UnifiedZeroCopy => self.vram_bytes,
        }
    }

    /// Exact total usable memory across GPU + host pools (never double-counts unified RAM).
    pub const fn total_usable_memory_bytes(&self) -> u64 {
        match self.architecture {
            MemoryArchitecture::DiscretePcie => self.usable_vram_bytes + self.usable_host_ram_bytes,
            MemoryArchitecture::UnifiedZeroCopy => self.usable_vram_bytes,
        }
    }
}
