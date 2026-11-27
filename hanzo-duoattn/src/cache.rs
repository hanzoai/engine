// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dual-regime KV cache storage (Full Retrieval KV + O(1) Ring-Buffered Streaming KV)
//! with exact memory accounting and fail-closed profile validation.
//!
//! Each `(attn_layer, kv_head)` slot is independently backed by either:
//! - [`HeadKvStorage::Full`]: retains every appended `(key, value)` pair (`O(L)` memory), or
//! - [`HeadKvStorage::Streaming`]: retains only the first `sink_tokens` attention sinks plus
//!   a constant-capacity ring buffer of the most recent `window_tokens` tokens (`O(1)` memory).

use crate::error::DuoAttnError;
use crate::profile::{HeadCalibrationProfile, HeadRole, ModelIdentity};

/// KV cache element storage dtype used for resident byte accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvDtype {
    /// 32-bit IEEE float (`4` bytes/element).
    F32,
    /// 16-bit IEEE float (`2` bytes/element — standard Hanzo GPU KV baseline).
    F16,
    /// 16-bit bfloat (`2` bytes/element).
    Bf16,
    /// 8-bit floating point (`1` byte/element).
    Fp8,
    /// 8-bit integer (`1` byte/element).
    Int8,
}

impl KvDtype {
    /// Bytes per scalar KV element.
    #[must_use]
    pub const fn bytes_per_elem(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Fp8 | Self::Int8 => 1,
        }
    }

    /// Canonical display tag.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "FP32",
            Self::F16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Fp8 => "FP8",
            Self::Int8 => "INT8",
        }
    }
}

/// Matched KV-cache regime executed in dual-cache benchmarks and runtime comparisons.
#[derive(Clone, Debug, PartialEq)]
pub enum CacheRegime {
    /// 100% full KV history across all heads (`O(L)` memory and decode bandwidth).
    FullKvBaseline,
    /// Existing `hanzo-engine` LSA bucket-clustered sparse attention (retains 100% full KV +
    /// bucket centroids; attends to `sink + window + top_k` distant keys per query).
    LsaBucketSparse {
        /// Leading attention sink tokens always attended.
        sink_tokens: usize,
        /// Recent sliding-window tokens always attended.
        window_tokens: usize,
        /// Mean keys per bucket used to form centroid summaries.
        bucket_size: usize,
        /// Maximum distant keys unpacked from top-scoring buckets per query.
        top_k: usize,
    },
    /// 100% uniform streaming/sliding-window + sink across all heads (`O(1)` memory, collapses retrieval).
    UniformLsaStreaming {
        /// Leading attention sink tokens retained.
        sink_tokens: usize,
        /// Recent sliding-window tokens retained.
        window_tokens: usize,
    },
    /// Same retrieval-head budget `r%`, but randomly assigned heads without offline calibration.
    RandomHeadSelection {
        /// Deterministic PRNG seed controlling random head selection.
        seed: u64,
        /// Leading attention sink tokens for streaming heads.
        sink_tokens: usize,
        /// Recent sliding-window tokens for streaming heads.
        window_tokens: usize,
        /// Fraction of KV heads randomly assigned to `Retrieval`.
        retrieval_ratio: f32,
    },
    /// Calibrated DuoAttention: verified per-model profile assigns top-`r%` retrieval heads
    /// to full KV and `(1 - r)%` streaming heads to `sink_tokens + window_tokens`.
    DuoAttentionCalibrated {
        /// Leading attention sink tokens for streaming heads.
        sink_tokens: usize,
        /// Recent sliding-window tokens for streaming heads.
        window_tokens: usize,
        /// Fraction of KV heads assigned to `Retrieval`.
        retrieval_ratio: f32,
    },
    /// Negative-control ablation ONLY: cross-model transferred profile bypassing model-ID match.
    CrossModelWrongProfileAblation {
        /// Source model checkpoint ID whose profile was misapplied.
        source_model: String,
        /// Leading attention sink tokens.
        sink_tokens: usize,
        /// Recent sliding-window tokens.
        window_tokens: usize,
        /// Retrieval ratio.
        retrieval_ratio: f32,
    },
}

impl CacheRegime {
    /// Canonical benchmark name for tables and JSON output.
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::FullKvBaseline => "full_kv_baseline",
            Self::LsaBucketSparse { .. } => "lsa_existing_hanzo",
            Self::UniformLsaStreaming { .. } => "uniform_lsa_streaming",
            Self::RandomHeadSelection { .. } => "random_head_selection_control",
            Self::DuoAttentionCalibrated { .. } => "duoattention_calibrated",
            Self::CrossModelWrongProfileAblation { .. } => "cross_model_wrong_profile",
        }
    }
}

/// Per-KV-head storage engine: either full append-only KV or constant-memory sink+window ring.
#[derive(Clone, Debug)]
pub enum HeadKvStorage {
    /// Full KV cache retaining every token appended (`O(L)`).
    Full {
        /// Attention head dimension.
        head_dim: usize,
        /// Flat keys `[len, head_dim]`.
        keys: Vec<f32>,
        /// Flat values `[len, head_dim]`.
        values: Vec<f32>,
        /// Logical token positions `[len]`.
        positions: Vec<usize>,
        /// Number of tokens stored.
        len: usize,
    },
    /// Constant-memory streaming KV cache retaining `[0..sink_tokens)` + recent `window_tokens`.
    Streaming {
        /// Attention head dimension.
        head_dim: usize,
        /// Maximum leading sink tokens retained permanently.
        sink_tokens: usize,
        /// Maximum recent sliding-window tokens retained in a circular buffer.
        window_tokens: usize,
        /// Sink keys `[sink_len, head_dim]`.
        sink_k: Vec<f32>,
        /// Sink values `[sink_len, head_dim]`.
        sink_v: Vec<f32>,
        /// Sink token positions `[sink_len]`.
        sink_pos: Vec<usize>,
        /// Current filled sink slots (`<= sink_tokens`).
        sink_len: usize,
        /// Ring-buffer window keys `[window_tokens, head_dim]`.
        win_k: Vec<f32>,
        /// Ring-buffer window values `[window_tokens, head_dim]`.
        win_v: Vec<f32>,
        /// Ring-buffer window token positions `[window_tokens]`.
        win_pos: Vec<usize>,
        /// Next write slot inside `[0..window_tokens)`.
        win_write: usize,
        /// Current populated slots in the ring buffer (`<= window_tokens`).
        win_len: usize,
        /// Total logical tokens seen by this head (including evicted middle tokens).
        total_seen: usize,
        /// Number of middle tokens evicted from this streaming head.
        evicted_tokens: usize,
    },
}

impl HeadKvStorage {
    /// Allocates a full-KV head store.
    #[must_use]
    pub fn new_full(head_dim: usize) -> Self {
        Self::Full {
            head_dim,
            keys: Vec::new(),
            values: Vec::new(),
            positions: Vec::new(),
            len: 0,
        }
    }

    /// Allocates a constant-memory streaming KV head store (`sink_tokens + window_tokens`).
    #[must_use]
    pub fn new_streaming(head_dim: usize, sink_tokens: usize, window_tokens: usize) -> Self {
        Self::Streaming {
            head_dim,
            sink_tokens,
            window_tokens,
            sink_k: Vec::with_capacity(sink_tokens * head_dim),
            sink_v: Vec::with_capacity(sink_tokens * head_dim),
            sink_pos: Vec::with_capacity(sink_tokens),
            sink_len: 0,
            win_k: vec![0.0_f32; window_tokens * head_dim],
            win_v: vec![0.0_f32; window_tokens * head_dim],
            win_pos: vec![0usize; window_tokens],
            win_write: 0,
            win_len: 0,
            total_seen: 0,
            evicted_tokens: 0,
        }
    }

    /// Returns the role of this head storage.
    #[must_use]
    pub const fn role(&self) -> HeadRole {
        match self {
            Self::Full { .. } => HeadRole::Retrieval,
            Self::Streaming { .. } => HeadRole::Streaming,
        }
    }

    /// Appends one `(key, value)` token at logical `position`.
    ///
    /// For [`HeadKvStorage::Streaming`], once `sink_len == sink_tokens` and
    /// `win_len == window_tokens`, the oldest window slot is overwritten in place so
    /// memory remains strictly `O(sink_tokens + window_tokens)`.
    pub fn append(
        &mut self,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), DuoAttnError> {
        let head_dim = self.head_dim();
        if key.len() != head_dim {
            return Err(DuoAttnError::DimensionMismatch {
                context: "HeadKvStorage::append(key)",
                expected: head_dim,
                actual: key.len(),
            });
        }
        if value.len() != head_dim {
            return Err(DuoAttnError::DimensionMismatch {
                context: "HeadKvStorage::append(value)",
                expected: head_dim,
                actual: value.len(),
            });
        }

        match self {
            Self::Full {
                keys,
                values,
                positions,
                len,
                ..
            } => {
                keys.extend_from_slice(key);
                values.extend_from_slice(value);
                positions.push(position);
                *len += 1;
            }
            Self::Streaming {
                head_dim,
                sink_tokens,
                window_tokens,
                sink_k,
                sink_v,
                sink_pos,
                sink_len,
                win_k,
                win_v,
                win_pos,
                win_write,
                win_len,
                total_seen,
                evicted_tokens,
            } => {
                *total_seen += 1;
                if *sink_len < *sink_tokens {
                    sink_k.extend_from_slice(key);
                    sink_v.extend_from_slice(value);
                    sink_pos.push(position);
                    *sink_len += 1;
                } else if *window_tokens > 0 {
                    let slot = *win_write;
                    let off = slot * *head_dim;
                    win_k[off..off + *head_dim].copy_from_slice(key);
                    win_v[off..off + *head_dim].copy_from_slice(value);
                    win_pos[slot] = position;
                    *win_write = (slot + 1) % *window_tokens;
                    if *win_len < *window_tokens {
                        *win_len += 1;
                    } else {
                        *evicted_tokens += 1;
                    }
                } else {
                    *evicted_tokens += 1;
                }
            }
        }
        Ok(())
    }

    /// Head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        match self {
            Self::Full { head_dim, .. } | Self::Streaming { head_dim, .. } => *head_dim,
        }
    }

    /// Number of currently resident KV tokens stored in this head.
    #[must_use]
    pub const fn resident_tokens(&self) -> usize {
        match self {
            Self::Full { len, .. } => *len,
            Self::Streaming {
                sink_len, win_len, ..
            } => *sink_len + *win_len,
        }
    }

    /// Total logical sequence length observed by this head.
    #[must_use]
    pub const fn total_seen(&self) -> usize {
        match self {
            Self::Full { len, .. } => *len,
            Self::Streaming { total_seen, .. } => *total_seen,
        }
    }

    /// Number of evicted middle tokens (always `0` for `Full`).
    #[must_use]
    pub const fn evicted_tokens(&self) -> usize {
        match self {
            Self::Full { .. } => 0,
            Self::Streaming { evicted_tokens, .. } => *evicted_tokens,
        }
    }

    /// Resident KV bytes under the specified `dtype` (`2 * resident_tokens * head_dim * bytes_per_elem`).
    #[must_use]
    pub const fn resident_bytes(&self, dtype: KvDtype) -> usize {
        2 * self.resident_tokens() * self.head_dim() * dtype.bytes_per_elem()
    }

    /// Clears all stored tokens without changing role or streaming capacity.
    pub fn clear(&mut self) {
        match self {
            Self::Full {
                keys,
                values,
                positions,
                len,
                ..
            } => {
                keys.clear();
                values.clear();
                positions.clear();
                *len = 0;
            }
            Self::Streaming {
                sink_k,
                sink_v,
                sink_pos,
                sink_len,
                win_write,
                win_len,
                total_seen,
                evicted_tokens,
                ..
            } => {
                sink_k.clear();
                sink_v.clear();
                sink_pos.clear();
                *sink_len = 0;
                *win_write = 0;
                *win_len = 0;
                *total_seen = 0;
                *evicted_tokens = 0;
            }
        }
    }

    /// Iterates resident `(key_slice, value_slice, logical_position)` pairs in chronological order.
    pub fn for_each_resident<F>(&self, mut f: F)
    where
        F: FnMut(&[f32], &[f32], usize),
    {
        match self {
            Self::Full {
                head_dim,
                keys,
                values,
                positions,
                len,
            } => {
                for (i, &pos) in positions.iter().enumerate().take(*len) {
                    let off = i * *head_dim;
                    f(
                        &keys[off..off + *head_dim],
                        &values[off..off + *head_dim],
                        pos,
                    );
                }
            }
            Self::Streaming {
                head_dim,
                window_tokens,
                sink_k,
                sink_v,
                sink_pos,
                sink_len,
                win_k,
                win_v,
                win_pos,
                win_write,
                win_len,
                ..
            } => {
                for (i, &pos) in sink_pos.iter().enumerate().take(*sink_len) {
                    let off = i * *head_dim;
                    f(
                        &sink_k[off..off + *head_dim],
                        &sink_v[off..off + *head_dim],
                        pos,
                    );
                }
                if *win_len > 0 && *window_tokens > 0 {
                    let start = if *win_len == *window_tokens {
                        *win_write
                    } else {
                        0
                    };
                    for step in 0..*win_len {
                        let slot = (start + step) % *window_tokens;
                        let off = slot * *head_dim;
                        f(
                            &win_k[off..off + *head_dim],
                            &win_v[off..off + *head_dim],
                            win_pos[slot],
                        );
                    }
                }
            }
        }
    }
}

/// Multi-layer, multi-head DuoAttention KV cache supporting all matched regimes.
#[derive(Clone, Debug)]
pub struct DualKvCache {
    model: ModelIdentity,
    regime: CacheRegime,
    dtype: KvDtype,
    /// Storage indexed by `[attn_layer_idx][kv_head_idx]`.
    layers: Vec<Vec<HeadKvStorage>>,
    /// High-water mark of resident KV bytes observed during the cache lifetime.
    peak_resident_bytes: usize,
    /// Number of retrieval heads configured across all attention layers.
    retrieval_heads: usize,
    /// Number of streaming heads configured across all attention layers.
    streaming_heads: usize,
    /// Streaming budget (`sink_tokens + window_tokens`), or `0` for `FullKvBaseline`.
    streaming_budget: usize,
}

impl DualKvCache {
    /// Creates a `FullKvBaseline` cache where 100% of KV heads keep full history.
    pub fn new_full(model: &ModelIdentity, dtype: KvDtype) -> Result<Self, DuoAttnError> {
        model.validate()?;
        let mut layers = Vec::with_capacity(model.num_attn_layers);
        for _ in 0..model.num_attn_layers {
            let mut heads = Vec::with_capacity(model.num_kv_heads);
            for _ in 0..model.num_kv_heads {
                heads.push(HeadKvStorage::new_full(model.head_dim));
            }
            layers.push(heads);
        }
        Ok(Self {
            model: model.clone(),
            regime: CacheRegime::FullKvBaseline,
            dtype,
            layers,
            peak_resident_bytes: 0,
            retrieval_heads: model.total_kv_heads(),
            streaming_heads: 0,
            streaming_budget: 0,
        })
    }

    /// Creates an `LsaBucketSparse` cache matching `hanzo-engine`'s LSA backend: retains 100% of
    /// full KV history across all heads plus bucket centroid summaries, and routes decode queries
    /// to `sink_tokens + window_tokens + top_k` distant keys from top-scoring key buckets.
    pub fn new_lsa_bucket_sparse(
        model: &ModelIdentity,
        sink_tokens: usize,
        window_tokens: usize,
        bucket_size: usize,
        top_k: usize,
        dtype: KvDtype,
    ) -> Result<Self, DuoAttnError> {
        model.validate()?;
        if sink_tokens == 0 || window_tokens == 0 || bucket_size == 0 || top_k == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "LSA sink_tokens, window_tokens, bucket_size, and top_k must all be >= 1"
                    .to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(model.num_attn_layers);
        for _ in 0..model.num_attn_layers {
            let mut heads = Vec::with_capacity(model.num_kv_heads);
            for _ in 0..model.num_kv_heads {
                heads.push(HeadKvStorage::new_full(model.head_dim));
            }
            layers.push(heads);
        }
        Ok(Self {
            model: model.clone(),
            regime: CacheRegime::LsaBucketSparse {
                sink_tokens,
                window_tokens,
                bucket_size,
                top_k,
            },
            dtype,
            layers,
            peak_resident_bytes: 0,
            retrieval_heads: model.total_kv_heads(),
            streaming_heads: 0,
            streaming_budget: 0,
        })
    }

    /// Creates a `UniformLsaStreaming` cache where 100% of KV heads use `sink + window` only.
    pub fn new_uniform_streaming(
        model: &ModelIdentity,
        sink_tokens: usize,
        window_tokens: usize,
        dtype: KvDtype,
    ) -> Result<Self, DuoAttnError> {
        model.validate()?;
        if sink_tokens == 0 || window_tokens == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "sink_tokens and window_tokens must be >= 1".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(model.num_attn_layers);
        for _ in 0..model.num_attn_layers {
            let mut heads = Vec::with_capacity(model.num_kv_heads);
            for _ in 0..model.num_kv_heads {
                heads.push(HeadKvStorage::new_streaming(
                    model.head_dim,
                    sink_tokens,
                    window_tokens,
                ));
            }
            layers.push(heads);
        }
        Ok(Self {
            model: model.clone(),
            regime: CacheRegime::UniformLsaStreaming {
                sink_tokens,
                window_tokens,
            },
            dtype,
            layers,
            peak_resident_bytes: 0,
            retrieval_heads: 0,
            streaming_heads: model.total_kv_heads(),
            streaming_budget: sink_tokens + window_tokens,
        })
    }

    /// Creates a `RandomHeadSelection` control cache with the same retrieval-head budget
    /// `retrieval_ratio`, but heads assigned uniformly at random (no offline calibration).
    pub fn new_random_control(
        model: &ModelIdentity,
        sink_tokens: usize,
        window_tokens: usize,
        retrieval_ratio: f32,
        seed: u64,
        dtype: KvDtype,
    ) -> Result<Self, DuoAttnError> {
        model.validate()?;
        if sink_tokens == 0 || window_tokens == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "sink_tokens and window_tokens must be >= 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&retrieval_ratio) || retrieval_ratio.is_nan() {
            return Err(DuoAttnError::InvalidConfig(format!(
                "retrieval_ratio must be in [0.0, 1.0], got {retrieval_ratio}"
            )));
        }

        let total = model.total_kv_heads();
        let target_ret = if retrieval_ratio <= 0.0 {
            0
        } else if retrieval_ratio >= 1.0 {
            total
        } else {
            ((retrieval_ratio as f64) * (total as f64))
                .round()
                .clamp(1.0, total as f64) as usize
        };

        // Deterministic Fisher-Yates shuffle using SplitMix64.
        let mut indices: Vec<usize> = (0..total).collect();
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        for i in (1..total).rev() {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let j = ((z ^ (z >> 31)) as usize) % (i + 1);
            indices.swap(i, j);
        }
        let mut is_ret = vec![false; total];
        for &idx in indices.iter().take(target_ret) {
            is_ret[idx] = true;
        }

        let mut layers = Vec::with_capacity(model.num_attn_layers);
        for l in 0..model.num_attn_layers {
            let mut heads = Vec::with_capacity(model.num_kv_heads);
            for h in 0..model.num_kv_heads {
                let idx = l * model.num_kv_heads + h;
                if is_ret[idx] {
                    heads.push(HeadKvStorage::new_full(model.head_dim));
                } else {
                    heads.push(HeadKvStorage::new_streaming(
                        model.head_dim,
                        sink_tokens,
                        window_tokens,
                    ));
                }
            }
            layers.push(heads);
        }

        Ok(Self {
            model: model.clone(),
            regime: CacheRegime::RandomHeadSelection {
                seed,
                sink_tokens,
                window_tokens,
                retrieval_ratio,
            },
            dtype,
            layers,
            peak_resident_bytes: 0,
            retrieval_heads: target_ret,
            streaming_heads: total - target_ret,
            streaming_budget: sink_tokens + window_tokens,
        })
    }

    /// Creates a `DuoAttentionCalibrated` cache after fail-closed verification of `profile`
    /// against `model` ([`HeadCalibrationProfile::verify_for_model`]).
    pub fn new_calibrated(
        model: &ModelIdentity,
        profile: &HeadCalibrationProfile,
        dtype: KvDtype,
    ) -> Result<Self, DuoAttnError> {
        profile.verify_for_model(model)?;
        Self::build_from_profile(
            model,
            profile,
            dtype,
            CacheRegime::DuoAttentionCalibrated {
                sink_tokens: profile.sink_tokens,
                window_tokens: profile.window_tokens,
                retrieval_ratio: profile.retrieval_ratio,
            },
        )
    }

    /// Negative-control / ablation constructor that intentionally skips `model_checkpoint_id`
    /// matching while still verifying profile integrity and compatible geometry.
    ///
    /// Exists exclusively to measure Negative Regime 1 (cross-model profile transfer failure)
    /// in benchmarks and tests. Production callers MUST always use [`Self::new_calibrated`].
    pub fn new_cross_model_ablation_only(
        model: &ModelIdentity,
        wrong_model_profile: &HeadCalibrationProfile,
        dtype: KvDtype,
    ) -> Result<Self, DuoAttnError> {
        model.validate()?;
        wrong_model_profile.verify_integrity()?;
        if wrong_model_profile.num_attn_layers != model.num_attn_layers {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "num_attn_layers",
                profile_value: wrong_model_profile.num_attn_layers,
                target_value: model.num_attn_layers,
            });
        }
        if wrong_model_profile.num_kv_heads != model.num_kv_heads {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "num_kv_heads",
                profile_value: wrong_model_profile.num_kv_heads,
                target_value: model.num_kv_heads,
            });
        }
        Self::build_from_profile(
            model,
            wrong_model_profile,
            dtype,
            CacheRegime::CrossModelWrongProfileAblation {
                source_model: wrong_model_profile.model_checkpoint_id.clone(),
                sink_tokens: wrong_model_profile.sink_tokens,
                window_tokens: wrong_model_profile.window_tokens,
                retrieval_ratio: wrong_model_profile.retrieval_ratio,
            },
        )
    }

    fn build_from_profile(
        model: &ModelIdentity,
        profile: &HeadCalibrationProfile,
        dtype: KvDtype,
        regime: CacheRegime,
    ) -> Result<Self, DuoAttnError> {
        let mut layers = Vec::with_capacity(model.num_attn_layers);
        let mut retrieval_heads = 0usize;
        let mut streaming_heads = 0usize;
        for l in 0..model.num_attn_layers {
            let mut heads = Vec::with_capacity(model.num_kv_heads);
            for h in 0..model.num_kv_heads {
                match profile.role_for(l, h)? {
                    HeadRole::Retrieval => {
                        retrieval_heads += 1;
                        heads.push(HeadKvStorage::new_full(model.head_dim));
                    }
                    HeadRole::Streaming => {
                        streaming_heads += 1;
                        heads.push(HeadKvStorage::new_streaming(
                            model.head_dim,
                            profile.sink_tokens,
                            profile.window_tokens,
                        ));
                    }
                }
            }
            layers.push(heads);
        }
        Ok(Self {
            model: model.clone(),
            regime,
            dtype,
            layers,
            peak_resident_bytes: 0,
            retrieval_heads,
            streaming_heads,
            streaming_budget: profile.sink_tokens + profile.window_tokens,
        })
    }

    /// Returns the model identity bound to this cache.
    #[must_use]
    pub fn model(&self) -> &ModelIdentity {
        &self.model
    }

    /// Returns the active cache regime.
    #[must_use]
    pub fn regime(&self) -> &CacheRegime {
        &self.regime
    }

    /// Returns the KV element dtype.
    #[must_use]
    pub const fn dtype(&self) -> KvDtype {
        self.dtype
    }

    /// Returns the number of configured retrieval KV heads.
    #[must_use]
    pub const fn retrieval_heads(&self) -> usize {
        self.retrieval_heads
    }

    /// Returns the number of configured streaming KV heads.
    #[must_use]
    pub const fn streaming_heads(&self) -> usize {
        self.streaming_heads
    }

    /// Appends one token's KV states for all `num_kv_heads` of `attn_layer_idx`.
    ///
    /// `keys` and `values` each have flat shape `[num_kv_heads * head_dim]`.
    pub fn append_layer_step(
        &mut self,
        attn_layer_idx: usize,
        position: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), DuoAttnError> {
        let expected = self.model.num_kv_heads * self.model.head_dim;
        if keys.len() != expected {
            return Err(DuoAttnError::DimensionMismatch {
                context: "DualKvCache::append_layer_step(keys)",
                expected,
                actual: keys.len(),
            });
        }
        if values.len() != expected {
            return Err(DuoAttnError::DimensionMismatch {
                context: "DualKvCache::append_layer_step(values)",
                expected,
                actual: values.len(),
            });
        }
        let layer = self.layers.get_mut(attn_layer_idx).ok_or_else(|| {
            DuoAttnError::InvalidConfig(format!(
                "attn_layer_idx {attn_layer_idx} out of range (num_attn_layers = {})",
                self.model.num_attn_layers
            ))
        })?;
        let d = self.model.head_dim;
        for (h, storage) in layer.iter_mut().enumerate() {
            let off = h * d;
            storage.append(position, &keys[off..off + d], &values[off..off + d])?;
        }
        let current = self.resident_kv_bytes();
        if current > self.peak_resident_bytes {
            self.peak_resident_bytes = current;
        }
        Ok(())
    }

    /// Appends using a global hybrid layer index, rejecting recurrent GDN layers fail-closed.
    pub fn append_global_layer_step(
        &mut self,
        global_layer_idx: usize,
        position: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), DuoAttnError> {
        let attn_idx = self.model.check_attention_layer(global_layer_idx)?;
        self.append_layer_step(attn_idx, position, keys, values)
    }

    /// Accesses storage for a specific `(attn_layer_idx, kv_head_idx)`.
    pub fn head_storage(
        &self,
        attn_layer_idx: usize,
        kv_head_idx: usize,
    ) -> Result<&HeadKvStorage, DuoAttnError> {
        self.layers
            .get(attn_layer_idx)
            .and_then(|l| l.get(kv_head_idx))
            .ok_or_else(|| {
                DuoAttnError::InvalidConfig(format!(
                    "head ({attn_layer_idx}, {kv_head_idx}) out of bounds"
                ))
            })
    }

    /// Resets all stored KV tokens while keeping head roles intact.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            for head in layer {
                head.clear();
            }
        }
        self.peak_resident_bytes = 0;
    }

    /// Total logical sequence length seen by layer 0 head 0.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.first())
            .map_or(0, HeadKvStorage::total_seen)
    }

    /// Returns `true` when the sequence fits completely inside `sink_tokens + window_tokens`,
    /// where streaming eviction yields zero KV memory reduction (`1.0x`).
    #[must_use]
    pub fn is_short_context(&self) -> bool {
        self.streaming_budget > 0 && self.seq_len() <= self.streaming_budget
    }

    /// Sum of resident KV tokens across all `(attn_layer, kv_head)` pairs.
    #[must_use]
    pub fn total_resident_kv_tokens(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.iter())
            .map(HeadKvStorage::resident_tokens)
            .sum()
    }

    /// Total resident KV cache bytes under `self.dtype` (plus bucket centroid metadata for LSA).
    #[must_use]
    pub fn resident_kv_bytes(&self) -> usize {
        let base: usize = self
            .layers
            .iter()
            .flat_map(|l| l.iter())
            .map(|h| h.resident_bytes(self.dtype))
            .sum();
        if let CacheRegime::LsaBucketSparse { bucket_size, .. } = self.regime {
            let meta: usize = self
                .layers
                .iter()
                .flat_map(|l| l.iter())
                .map(|h| {
                    let buckets = h.resident_tokens() / bucket_size;
                    buckets * (self.model.head_dim * self.dtype.bytes_per_elem() + 8)
                })
                .sum();
            base + meta
        } else {
            base
        }
    }

    /// Peak resident KV bytes observed since creation or last [`Self::clear`].
    #[must_use]
    pub const fn peak_resident_kv_bytes(&self) -> usize {
        self.peak_resident_bytes
    }

    /// Equivalent full-KV baseline bytes for the same logical sequence length.
    #[must_use]
    pub fn full_kv_baseline_bytes(&self) -> usize {
        let total_seen: usize = self
            .layers
            .iter()
            .flat_map(|l| l.iter())
            .map(HeadKvStorage::total_seen)
            .sum();
        2 * total_seen * self.model.head_dim * self.dtype.bytes_per_elem()
    }

    /// KV memory reduction factor (`full_kv_baseline_bytes / resident_kv_bytes`).
    ///
    /// Returns `1.0` for empty caches or short contexts (`L <= sink_tokens + window_tokens`).
    #[must_use]
    pub fn memory_reduction_ratio(&self) -> f64 {
        let resident = self.resident_kv_bytes();
        let baseline = self.full_kv_baseline_bytes();
        if resident == 0 || baseline == 0 {
            1.0
        } else {
            baseline as f64 / resident as f64
        }
    }
}
