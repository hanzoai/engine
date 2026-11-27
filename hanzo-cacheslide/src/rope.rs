// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact Rotary Position Embedding (RoPE) forward, inverse, and relative-shift math.
//!
//! # Mathematical Foundation (CacheSlide Position Shifting)
//!
//! In a RoPE Transformer, a key vector projected at token position $p$ is rotated by a
//! block-diagonal $SO(2)^{d_r/2}$ orthogonal matrix $R(p)$:
//!
//! $$K(p) = R(p)\,K_{\text{raw}}$$
//!
//! Because 2D rotations in each frequency plane $\theta_i$ form an abelian group under
//! matrix multiplication,
//!
//! $$R(a)\,R(b) = R(a + b), \qquad R(0) = I, \qquad R(-\Delta) = R(\Delta)^{\top} = R(\Delta)^{-1}.$$
//!
//! Consequently, when a cached KV segment originally computed at start offset $p_{\text{orig}}$
//! is reused at a new offset $p_{\text{new}}$, every token $t \in [0, L)$ in that segment moves
//! from $p_{\text{orig}} + t$ to $p_{\text{new}} + t$ with the **identical** shift
//! $\Delta p = p_{\text{new}} - p_{\text{orig}}$:
//!
//! $$K_{\text{shifted}}[t] = R(\Delta p)\,K_{\text{orig}}[t] = R(p_{\text{new}} + t)\,K_{\text{raw}}[t].$$
//!
//! A single precomputed [`RopeShiftTable`] of $d_r / 2$ `(cos, sin)` pairs rotates the entire
//! segment across all tokens, KV heads, and layers without re-running Q/K/V or FFN projections.
//! Values $V$ are position-invariant under RoPE and require zero transformation.
//!
//! Supports both RoPE layouts used in `hanzo-engine/src/layers.rs`:
//! - [`RopeLayout::GptNeoX`]: half-split `(x[i], x[i + rotary_dim/2])` (Llama-3, Qwen2/3, Gemma3, Phi3/4).
//! - [`RopeLayout::Interleaved`]: adjacent pairs `(x[2*i], x[2*i + 1])` (DeepSeek-V2/V3, classic RoFormer).

/// Memory layout of rotated 2D planes inside each attention head vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeLayout {
    /// Half-split GPT-NeoX layout (`is_gpt_neox = true` in `hanzo-engine/src/layers.rs`):
    /// plane `i` pairs `(x[i], x[i + half_rotary])`.
    GptNeoX,
    /// Interleaved layout (`is_gpt_neox = false` in `hanzo-engine/src/layers.rs`):
    /// plane `i` pairs `(x[2*i], x[2*i + 1])`.
    Interleaved,
}

/// Configuration for Rotary Position Embeddings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    /// Full attention head dimension (`head_dim`).
    pub head_dim: usize,
    /// Number of leading dimensions rotated by RoPE (`rotary_dim <= head_dim`, even).
    /// Supports partial RoPE (`partial_rotary_factor < 1.0`, e.g. Phi/Qwen3-VL/GLM).
    pub rotary_dim: usize,
    /// RoPE base frequency $\theta$ (e.g. `10_000.0`, `500_000.0`, `1_000_000.0`).
    pub theta: f64,
    /// Linear position scaling factor (`1.0` for standard RoPE, `> 1.0` for linear context extension).
    pub freq_scale: f64,
    /// 2D plane pairing layout.
    pub layout: RopeLayout,
}

impl RopeConfig {
    /// Construct a standard full-head RoPE configuration.
    ///
    /// # Panics
    /// Panics if `head_dim` is 0 or odd, or if `theta <= 0.0`.
    #[must_use]
    pub fn standard(head_dim: usize, theta: f64, layout: RopeLayout) -> Self {
        assert!(
            head_dim >= 2 && head_dim.is_multiple_of(2),
            "head_dim must be positive and even"
        );
        assert!(theta > 0.0 && theta.is_finite(), "theta must be positive");
        Self {
            head_dim,
            rotary_dim: head_dim,
            theta,
            freq_scale: 1.0,
            layout,
        }
    }

    /// Construct a partial-RoPE configuration (`rotary_dim <= head_dim`, even).
    ///
    /// # Panics
    /// Panics if `rotary_dim` is 0, odd, or exceeds `head_dim`, or if `theta`/`freq_scale` are invalid.
    #[must_use]
    pub fn partial(
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        freq_scale: f64,
        layout: RopeLayout,
    ) -> Self {
        assert!(
            rotary_dim >= 2 && rotary_dim.is_multiple_of(2) && rotary_dim <= head_dim,
            "rotary_dim must be positive, even, and <= head_dim"
        );
        assert!(theta > 0.0 && theta.is_finite(), "theta must be positive");
        assert!(
            freq_scale > 0.0 && freq_scale.is_finite(),
            "freq_scale must be positive"
        );
        Self {
            head_dim,
            rotary_dim,
            theta,
            freq_scale,
            layout,
        }
    }

    /// Number of 2D rotation planes (`rotary_dim / 2`).
    #[inline]
    #[must_use]
    pub const fn num_planes(&self) -> usize {
        self.rotary_dim / 2
    }

    /// Compute inverse frequencies $\omega_i = \theta^{-2i / d_r} / \text{freq\_scale}$ in `f64`
    /// to avoid single-precision phase drift at large positions.
    #[must_use]
    pub fn inv_freqs(&self) -> Vec<f64> {
        let half = self.num_planes();
        let d_r = self.rotary_dim as f64;
        (0..half)
            .map(|i| self.theta.powf(-(2.0 * i as f64) / d_r) / self.freq_scale)
            .collect()
    }
}

/// Precomputed `(cos, sin)` rotation factors for a single signed shift $\Delta p$.
///
/// Computing trigonometric functions once per segment shift makes rotating an entire
/// cached KV block (`seq_len * num_kv_heads * num_layers` vectors) pure multiply-add work.
#[derive(Debug, Clone)]
pub struct RopeShiftTable {
    delta_pos: i64,
    head_dim: usize,
    rotary_dim: usize,
    layout: RopeLayout,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeShiftTable {
    /// Precompute $R(\Delta p)$ from `config` and signed shift `delta_pos = p_new - p_orig`.
    #[must_use]
    pub fn new(config: &RopeConfig, delta_pos: i64) -> Self {
        let inv_freqs = config.inv_freqs();
        let mut cos = Vec::with_capacity(inv_freqs.len());
        let mut sin = Vec::with_capacity(inv_freqs.len());
        if delta_pos == 0 {
            cos.resize(inv_freqs.len(), 1.0);
            sin.resize(inv_freqs.len(), 0.0);
        } else {
            let dp = delta_pos as f64;
            for &inv_f in &inv_freqs {
                let angle = dp * inv_f;
                let (s, c) = angle.sin_cos();
                cos.push(c as f32);
                sin.push(s as f32);
            }
        }
        Self {
            delta_pos,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            layout: config.layout,
            cos,
            sin,
        }
    }

    /// Signed position delta this table encodes.
    #[inline]
    #[must_use]
    pub const fn delta_pos(&self) -> i64 {
        self.delta_pos
    }

    /// Returns `true` when `delta_pos == 0` (identity transform).
    #[inline]
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        self.delta_pos == 0
    }

    /// Apply $R(\Delta p)$ in place to a single head vector of length `head_dim`.
    ///
    /// # Panics
    /// Panics if `vec.len() != self.head_dim`.
    #[inline]
    pub fn apply_in_place(&self, vec: &mut [f32]) {
        assert_eq!(vec.len(), self.head_dim, "head vector length mismatch");
        if self.delta_pos == 0 {
            return;
        }
        let half = self.rotary_dim / 2;
        match self.layout {
            RopeLayout::GptNeoX => {
                for i in 0..half {
                    let x0 = vec[i];
                    let x1 = vec[i + half];
                    let c = self.cos[i];
                    let s = self.sin[i];
                    vec[i] = x0 * c - x1 * s;
                    vec[i + half] = x0 * s + x1 * c;
                }
            }
            RopeLayout::Interleaved => {
                for i in 0..half {
                    let x0 = vec[2 * i];
                    let x1 = vec[2 * i + 1];
                    let c = self.cos[i];
                    let s = self.sin[i];
                    vec[2 * i] = x0 * c - x1 * s;
                    vec[2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    }

    /// Apply $R(\Delta p)$ in place to a contiguous KV key block laid out as
    /// `[seq_len, num_kv_heads, head_dim]`.
    ///
    /// # Panics
    /// Panics if `k_block.len() != seq_len * num_kv_heads * self.head_dim`.
    pub fn apply_to_k_block(&self, k_block: &mut [f32], seq_len: usize, num_kv_heads: usize) {
        let expected = seq_len * num_kv_heads * self.head_dim;
        assert_eq!(
            k_block.len(),
            expected,
            "k_block length {actual} != expected {expected}",
            actual = k_block.len()
        );
        if self.delta_pos == 0 {
            return;
        }
        for head_vec in k_block.chunks_exact_mut(self.head_dim) {
            self.apply_in_place(head_vec);
        }
    }

    /// Return a shifted copy of a contiguous KV key block `[seq_len, num_kv_heads, head_dim]`.
    #[must_use]
    pub fn shift_k_block_cloned(
        &self,
        k_block: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
    ) -> Vec<f32> {
        let mut out = k_block.to_vec();
        self.apply_to_k_block(&mut out, seq_len, num_kv_heads);
        out
    }
}

/// Apply forward RoPE $R(\text{pos})$ in place to a single head vector.
pub fn apply_rope_in_place(config: &RopeConfig, vec: &mut [f32], pos: i64) {
    let table = RopeShiftTable::new(config, pos);
    table.apply_in_place(vec);
}

/// Apply inverse RoPE $R(-\text{pos}) = R(\text{pos})^{\top}$ in place to a single head vector.
pub fn apply_inverse_rope_in_place(config: &RopeConfig, vec: &mut [f32], pos: i64) {
    let table = RopeShiftTable::new(config, -pos);
    table.apply_in_place(vec);
}

/// Shift a cached key vector from `orig_pos` to `new_pos` directly via $R(p_{\text{new}} - p_{\text{orig}})$.
///
/// Returns `None` if `new_pos - orig_pos` overflows `i64` (fail-closed, mirroring
/// `hanzo-kv-fabric/src/composition.rs`).
#[must_use]
pub fn checked_position_shift(orig_pos: usize, new_pos: usize) -> Option<i64> {
    let orig = i64::try_from(orig_pos).ok()?;
    let new = i64::try_from(new_pos).ok()?;
    new.checked_sub(orig)
}

/// Precomputed forward RoPE cache for absolute positions `[0..max_seq_len)`.
#[derive(Debug, Clone)]
pub struct RopeCache {
    config: RopeConfig,
    max_seq_len: usize,
    /// Flattened `[max_seq_len, num_planes]`.
    cos: Vec<f32>,
    /// Flattened `[max_seq_len, num_planes]`.
    sin: Vec<f32>,
}

impl RopeCache {
    /// Precompute cos/sin tables for positions `0..max_seq_len`.
    #[must_use]
    pub fn new(config: RopeConfig, max_seq_len: usize) -> Self {
        let inv_freqs = config.inv_freqs();
        let half = inv_freqs.len();
        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);
        for pos in 0..max_seq_len {
            let p = pos as f64;
            for &inv_f in &inv_freqs {
                let (s, c) = (p * inv_f).sin_cos();
                cos.push(c as f32);
                sin.push(s as f32);
            }
        }
        Self {
            config,
            max_seq_len,
            cos,
            sin,
        }
    }

    /// Underlying [`RopeConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &RopeConfig {
        &self.config
    }

    /// Apply forward RoPE $R(\text{pos})$ to a single head vector using cached tables
    /// (falling back to exact on-the-fly trig if `pos >= max_seq_len`).
    #[inline]
    pub fn apply_at_pos(&self, vec: &mut [f32], pos: usize) {
        assert_eq!(vec.len(), self.config.head_dim);
        if pos >= self.max_seq_len {
            apply_rope_in_place(&self.config, vec, pos as i64);
            return;
        }
        let half = self.config.num_planes();
        let base = pos * half;
        let cos = &self.cos[base..base + half];
        let sin = &self.sin[base..base + half];
        match self.config.layout {
            RopeLayout::GptNeoX => {
                for i in 0..half {
                    let x0 = vec[i];
                    let x1 = vec[i + half];
                    let c = cos[i];
                    let s = sin[i];
                    vec[i] = x0 * c - x1 * s;
                    vec[i + half] = x0 * s + x1 * c;
                }
            }
            RopeLayout::Interleaved => {
                for i in 0..half {
                    let x0 = vec[2 * i];
                    let x1 = vec[2 * i + 1];
                    let c = cos[i];
                    let s = sin[i];
                    vec[2 * i] = x0 * c - x1 * s;
                    vec[2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut x = seed ^ 0x9e37_79b9_7f4a_7c15;
        (0..dim)
            .map(|_| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
                u * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn l2_norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn rope_group_composition_identity_neox_and_interleaved() {
        for layout in [RopeLayout::GptNeoX, RopeLayout::Interleaved] {
            for theta in [10_000.0, 500_000.0] {
                let cfg = RopeConfig::standard(64, theta, layout);
                let raw = sample_vec(64, 42);

                for &(a, b) in &[
                    (0i64, 0i64),
                    (17, 256),
                    (512, 256),
                    (768, -256),
                    (2048, -1900),
                    (10_000, 4_096),
                ] {
                    // Apply R(b) then R(a)
                    let mut composed = raw.clone();
                    apply_rope_in_place(&cfg, &mut composed, b);
                    apply_rope_in_place(&cfg, &mut composed, a);

                    // Apply R(a + b) directly
                    let mut direct = raw.clone();
                    apply_rope_in_place(&cfg, &mut direct, a + b);

                    let err = max_abs_err(&composed, &direct);
                    assert!(
                        err < 1e-5,
                        "group property R({a})R({b}) == R({}) violated for {layout:?} theta={theta}: err={err}",
                        a + b
                    );
                }
            }
        }
    }

    #[test]
    fn rope_inverse_and_norm_preservation() {
        let cfg = RopeConfig::standard(128, 500_000.0, RopeLayout::GptNeoX);
        let raw = sample_vec(128, 1346);
        let raw_norm = l2_norm(&raw);

        let mut v = raw.clone();
        apply_rope_in_place(&cfg, &mut v, 1_536);
        assert!((l2_norm(&v) - raw_norm).abs() < 1e-5);

        apply_inverse_rope_in_place(&cfg, &mut v, 1_536);
        assert!(max_abs_err(&v, &raw) < 1e-5);
    }

    #[test]
    fn rope_segment_shift_matches_fresh_position_encoding() {
        let cfg = RopeConfig::partial(64, 32, 10_000.0, 1.0, RopeLayout::GptNeoX);
        let seq_len = 48;
        let num_kv_heads = 4;
        let p_orig = 512usize;
        let p_new = 768usize;

        let raw_k = sample_vec(seq_len * num_kv_heads * cfg.head_dim, 999);

        // Compute K_orig at p_orig..p_orig+seq_len
        let mut k_orig = raw_k.clone();
        for t in 0..seq_len {
            for h in 0..num_kv_heads {
                let start = (t * num_kv_heads + h) * cfg.head_dim;
                apply_rope_in_place(
                    &cfg,
                    &mut k_orig[start..start + cfg.head_dim],
                    (p_orig + t) as i64,
                );
            }
        }

        // Shift K_orig directly by delta = p_new - p_orig (+256)
        let delta = checked_position_shift(p_orig, p_new).expect("valid shift");
        let shift_table = RopeShiftTable::new(&cfg, delta);
        let k_shifted = shift_table.shift_k_block_cloned(&k_orig, seq_len, num_kv_heads);

        // Compute ground-truth K_fresh at p_new..p_new+seq_len
        let mut k_fresh = raw_k.clone();
        for t in 0..seq_len {
            for h in 0..num_kv_heads {
                let start = (t * num_kv_heads + h) * cfg.head_dim;
                apply_rope_in_place(
                    &cfg,
                    &mut k_fresh[start..start + cfg.head_dim],
                    (p_new + t) as i64,
                );
            }
        }

        let err = max_abs_err(&k_shifted, &k_fresh);
        assert!(err < 1e-5, "shifted K block error {err} >= 1e-5");

        // Verify unrotated trailing dims [rotary_dim..head_dim) remained bit-identical!
        for t in 0..seq_len {
            for h in 0..num_kv_heads {
                let base = (t * num_kv_heads + h) * cfg.head_dim;
                assert_eq!(
                    &k_shifted[base + cfg.rotary_dim..base + cfg.head_dim],
                    &raw_k[base + cfg.rotary_dim..base + cfg.head_dim]
                );
            }
        }
    }

    #[test]
    fn rope_relative_inner_product_invariance() {
        // Core RoPE theorem: <R(p+d) q, R(p) k> depends ONLY on d, not on p!
        let cfg = RopeConfig::standard(64, 10_000.0, RopeLayout::GptNeoX);
        let q = sample_vec(64, 7);
        let k = sample_vec(64, 11);

        let dot_at = |p_q: i64, p_k: i64| -> f32 {
            let mut qr = q.clone();
            let mut kr = k.clone();
            apply_rope_in_place(&cfg, &mut qr, p_q);
            apply_rope_in_place(&cfg, &mut kr, p_k);
            qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum()
        };

        let ref_dot = dot_at(100, 37);
        for shift in [-30i64, 15, 256, 4096] {
            let shifted_dot = dot_at(100 + shift, 37 + shift);
            assert!(
                (shifted_dot - ref_dot).abs() < 1e-4,
                "relative dot-product invariance broken at shift={shift}: {shifted_dot} vs {ref_dot}"
            );
        }
    }

    #[test]
    fn checked_position_shift_fails_closed_on_overflow_and_scaled_rope_composes() {
        assert_eq!(checked_position_shift(512, 768), Some(256));
        assert_eq!(checked_position_shift(768, 512), Some(-256));
        assert_eq!(checked_position_shift(0, usize::MAX), None);
        assert_eq!(checked_position_shift(usize::MAX, 0), None);

        // Linear-scaled partial RoPE (freq_scale = 2.0, rotary_dim = 32 / head_dim = 64).
        let cfg = RopeConfig::partial(64, 32, 500_000.0, 2.0, RopeLayout::Interleaved);
        let raw = sample_vec(64, 2026);
        let mut hops = raw.clone();
        apply_rope_in_place(&cfg, &mut hops, 300);
        apply_rope_in_place(&cfg, &mut hops, 700);
        apply_rope_in_place(&cfg, &mut hops, -450);

        let mut direct = raw.clone();
        apply_rope_in_place(&cfg, &mut direct, 550);
        assert!(max_abs_err(&hops, &direct) < 1e-5);
    }
}
