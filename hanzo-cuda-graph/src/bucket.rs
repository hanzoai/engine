// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Bounded shape bucketing and decode-graph identity keys (`CudaDecodeGraphKey`).
//!
//! Variable batch sizes (`B`), speculative verification widths (`K`), and paged KV context
//! lengths would cause combinatorial capture storms and multi-gigabyte graph pool bloat if
//! every exact `(B, K, ctx_len)` triple captured a separate CUDA graph.
//!
//! This module maps dynamic runtime shapes into deterministic bounded buckets with explicit
//! active-lane masking (`lane_mask`) so padded lanes execute safely without mutating active
//! GDN recurrent state or paged KV cache entries.

use crate::lifetime::DeviceLocation;
use crate::model::QuantFormat;

/// Default bounded decode batch buckets (`B ∈ {1, 2, 4, 8, 16, 32}`).
pub const DEFAULT_BATCH_BUCKETS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// Default bounded speculative verification width buckets (`K ∈ {1, 2, 4, 8}`).
/// `K = 1` is standard autoregressive decode; `K = 2, 4, 8` verify `K` draft+target tokens
/// in a single bucketed graph replay.
pub const DEFAULT_VERIFY_BUCKETS: [usize; 4] = [1, 2, 4, 8];

/// Default bounded paged KV context buckets covering `8K`, `16K`, `30K`, and `64K` contexts
/// (block-size-aligned to 16-token paged KV blocks; `30_720` covers `30_000` and `30_720`).
pub const DEFAULT_CONTEXT_BUCKETS: [usize; 4] = [8_192, 16_384, 30_720, 65_536];

/// Bucketing policy controlling how dynamic shapes map to captured graph geometries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BucketPolicy {
    /// Bounded default buckets (`B ∈ {1,2,4,8,16,32}`, `K ∈ {1,2,4,8}`, `ctx ∈ {8K,16K,30K,64K}`).
    Bounded {
        /// Allowed batch-size buckets (sorted ascending).
        batch_buckets: Vec<usize>,
        /// Allowed speculative verification width buckets (sorted ascending).
        verify_buckets: Vec<usize>,
        /// Allowed paged KV context buckets (sorted ascending).
        context_buckets: Vec<usize>,
    },
    /// Negative-control policy: captures an exact graph for every distinct `(B, K, ctx_len)`
    /// with no bucketing, triggering capture storms and graph pool VRAM explosion.
    UnboundedExact,
    /// Negative-control policy: forces all decode requests (including `B = 1`) into a single
    /// oversized batch bucket (e.g., `B = 32`), wasting compute and HBM bandwidth.
    OversizedMonolithic {
        /// Fixed monolithic batch size.
        fixed_batch: usize,
        /// Fixed verification width.
        fixed_verify_width: usize,
        /// Fixed context bucket.
        fixed_context: usize,
    },
}

impl Default for BucketPolicy {
    fn default() -> Self {
        Self::bounded_default()
    }
}

impl BucketPolicy {
    /// Returns the production bounded shape-bucketing policy.
    #[must_use]
    pub fn bounded_default() -> Self {
        Self::Bounded {
            batch_buckets: DEFAULT_BATCH_BUCKETS.to_vec(),
            verify_buckets: DEFAULT_VERIFY_BUCKETS.to_vec(),
            context_buckets: DEFAULT_CONTEXT_BUCKETS.to_vec(),
        }
    }

    /// Resolves `(active_batch, active_verify_width, active_context)` into a [`ShapeBucket`].
    ///
    /// Returns `None` if the active request exceeds the maximum configured bucket (signaling
    /// that the caller must fall back to eager execution via a breakable graph boundary).
    #[must_use]
    pub fn resolve(
        &self,
        active_batch: usize,
        active_verify_width: usize,
        active_context: usize,
    ) -> Option<ShapeBucket> {
        if active_batch == 0 || active_verify_width == 0 {
            return None;
        }
        match self {
            Self::Bounded {
                batch_buckets,
                verify_buckets,
                context_buckets,
            } => {
                let b = *batch_buckets.iter().find(|&&cap| cap >= active_batch)?;
                let k = *verify_buckets
                    .iter()
                    .find(|&&cap| cap >= active_verify_width)?;
                let c = *context_buckets
                    .iter()
                    .find(|&&cap| cap >= active_context.max(1))?;
                Some(ShapeBucket {
                    batch_bucket: b,
                    verify_width_bucket: k,
                    context_bucket: c,
                    active_batch,
                    active_verify_width,
                    active_context,
                })
            }
            Self::UnboundedExact => Some(ShapeBucket {
                batch_bucket: active_batch,
                verify_width_bucket: active_verify_width,
                context_bucket: active_context.max(1),
                active_batch,
                active_verify_width,
                active_context,
            }),
            Self::OversizedMonolithic {
                fixed_batch,
                fixed_verify_width,
                fixed_context,
            } => {
                if active_batch > *fixed_batch
                    || active_verify_width > *fixed_verify_width
                    || active_context > *fixed_context
                {
                    return None;
                }
                Some(ShapeBucket {
                    batch_bucket: *fixed_batch,
                    verify_width_bucket: *fixed_verify_width,
                    context_bucket: *fixed_context,
                    active_batch,
                    active_verify_width,
                    active_context,
                })
            }
        }
    }

    /// Maximum supported batch size under this policy (`None` if unbounded).
    #[must_use]
    pub fn max_batch(&self) -> Option<usize> {
        match self {
            Self::Bounded { batch_buckets, .. } => batch_buckets.iter().copied().max(),
            Self::UnboundedExact => None,
            Self::OversizedMonolithic { fixed_batch, .. } => Some(*fixed_batch),
        }
    }

    /// Maximum supported context length under this policy (`None` if unbounded).
    #[must_use]
    pub fn max_context(&self) -> Option<usize> {
        match self {
            Self::Bounded {
                context_buckets, ..
            } => context_buckets.iter().copied().max(),
            Self::UnboundedExact => None,
            Self::OversizedMonolithic { fixed_context, .. } => Some(*fixed_context),
        }
    }
}

/// Resolved bounded shape bucket pairing active request dimensions with padded graph geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeBucket {
    /// Padded batch size captured by the graph (`>= active_batch`).
    pub batch_bucket: usize,
    /// Padded speculative verification width captured by the graph (`>= active_verify_width`).
    pub verify_width_bucket: usize,
    /// Padded paged KV context length bucket (`>= active_context`).
    pub context_bucket: usize,
    /// Actual active sequences in this step.
    pub active_batch: usize,
    /// Actual active tokens per sequence (`1` for decode, `K` for speculative verify).
    pub active_verify_width: usize,
    /// Actual maximum sequence context length in this step.
    pub active_context: usize,
}

impl ShapeBucket {
    /// Number of padded (inactive) batch lanes in this bucket.
    #[must_use]
    pub const fn padding_lanes(&self) -> usize {
        self.batch_bucket.saturating_sub(self.active_batch)
    }

    /// Fraction of executed token slots that are padding (`0.0` = exact fit, close to `1.0` = wasteful).
    #[must_use]
    pub fn padding_waste_ratio(&self) -> f64 {
        let total = (self.batch_bucket * self.verify_width_bucket).max(1) as f64;
        let active = (self.active_batch * self.active_verify_width) as f64;
        ((total - active) / total).clamp(0.0, 1.0)
    }

    /// Builds the boolean active-lane mask of length `batch_bucket`.
    /// Lanes `0..active_batch` are `true`; padded lanes `active_batch..batch_bucket` are `false`.
    #[must_use]
    pub fn lane_mask(&self) -> Vec<bool> {
        (0..self.batch_bucket)
            .map(|idx| idx < self.active_batch)
            .collect()
    }
}

/// Canonical identity key for a cached CUDA decode or verification graph.
///
/// Mirrors Hanzo Engine's `CudaDecodeGraphKey` (`hanzo-engine/src/pipeline/cuda_graph.rs`)
/// by keying on device placement, bucketed input shape `(batch_bucket, verify_width_bucket)`,
/// bucketed `max_context_len`, quantization format, and piecewise graph segment tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaDecodeGraphKey {
    /// Target device placement.
    pub device: DeviceLocation,
    /// Padded batch bucket (`B`).
    pub batch_bucket: usize,
    /// Padded speculative verification width bucket (`K`).
    pub verify_width_bucket: usize,
    /// Bucketed `max_context_len` aligned to paged KV block size.
    pub max_context_len: usize,
    /// Weight quantization format (changes kernel dispatch graph).
    pub quant: QuantFormat,
    /// Piecewise subgraph tag (e.g. `"lsa_recurrent_mlp"`, `"paged_attn_verify"`, `"fused_decode"`).
    pub piece_tag: &'static str,
}

impl CudaDecodeGraphKey {
    /// Constructs a canonical key from a resolved [`ShapeBucket`].
    #[must_use]
    pub const fn from_bucket(
        device: DeviceLocation,
        bucket: &ShapeBucket,
        quant: QuantFormat,
        piece_tag: &'static str,
    ) -> Self {
        Self {
            device,
            batch_bucket: bucket.batch_bucket,
            verify_width_bucket: bucket.verify_width_bucket,
            max_context_len: bucket.context_bucket,
            quant,
            piece_tag,
        }
    }
}
