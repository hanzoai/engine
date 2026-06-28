//! Memory snapshot + the fit check. A *value*, not a query: hanzo-node fills it
//! from `hanzo_engine::MemoryUsage`/`DeviceMemory` (the same `available`/`is_unified`
//! the auto device-mapper uses), and the router decides purely against the value —
//! no hardware access here, so it's deterministic and unit-testable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemSnapshot {
    /// Currently-available bytes the device may use (system RAM on a unified
    /// device, free VRAM on a discrete GPU).
    pub available_bytes: u64,
    /// Total device/system memory.
    pub total_bytes: u64,
    /// True on unified-memory devices (GB10/DGX-Spark, Apple Silicon, Strix Halo)
    /// where weights + KV + activations share one pool.
    pub unified: bool,
}

impl MemSnapshot {
    /// Usable budget for resident weights. Mirrors the engine's iGPU-fraction
    /// policy: on unified memory a high fraction of *available* is safe; on a
    /// discrete GPU leave a reserve for activations/KV. The caller can override
    /// `fraction` (the engine's `IGPU_MEMORY_FRACTION`).
    pub fn budget(&self, fraction: f64) -> u64 {
        (self.available_bytes as f64 * fraction.clamp(0.0, 1.0)) as u64
    }

    /// Does a model of `est_bytes` resident footprint fit the budget?
    pub fn fits(&self, est_bytes: u64, fraction: f64) -> bool {
        est_bytes <= self.budget(fraction)
    }
}

/// Default memory fraction: high on unified memory (one shared pool), more
/// conservative on a discrete GPU (reserve for activations/KV). Matches the
/// engine's `igpu_memory_fraction` default for unified devices.
pub fn default_fraction(unified: bool) -> f64 {
    if unified {
        0.85
    } else {
        0.7
    }
}
