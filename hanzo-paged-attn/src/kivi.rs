//! Asymmetric 2-bit and 4-bit KV-cache quantization with a full-precision sliding residual
//! window (KIVI, Zirui Liu et al., ICML 2024, arXiv:2402.02750).
//!
//! While symmetric per-token `int8` quantization ([`crate::quant`]) keeps reconstruction error
//! within `1/254` even when key channels contain persistent magnitude outliers, dropping to
//! 4-bit or 2-bit with a single symmetric per-token scale collapses non-outlier channels toward
//! zero. KIVI resolves this asymmetry without calibration data:
//!
//! - **Key cache (per-channel within token groups of `G`)**: Fixed key channels exhibit
//!   persistent large magnitudes across sequence positions. Quantizing each channel `c`
//!   independently across a completed group of `G` tokens (`G = 32`, `64`, or `128`) isolates
//!   channel bias into `zero_point[c]` and dynamic range into `scale[c]`.
//! - **Value cache (per-token)**: Value activations lack persistent cross-token channel
//!   outliers, and sparse softmax weights concentrate on individual tokens. Quantizing each
//!   token vector independently (or in sub-head slices) preserves per-token fidelity.
//! - **Sliding full-precision residual window (`R` recent tokens)**: Newly generated tokens
//!   remain in unquantized `FP16` or `FP32` storage and flush atomically into bit-packed `u32`
//!   pages only when a complete group of `G` tokens ages past the residual window.
//! - **Recurrent / GDN state exemption**: Hybrid models interleave paged attention layers with
//!   recurrent Gated DeltaNet (GDN) state layers. Because recurrent states update
//!   multiplicatively at every step, KV quantization recipes are unsafe for GDN state;
//!   [`LayerStateKind::RecurrentGdn`] is explicitly held in full precision.

use half::f16;
use thiserror::Error;

/// Maximum per-head dimension supported by stack-resident fused attention buffers.
pub const MAX_HEAD_DIM: usize = 512;

/// Maximum token group size supported by stack-resident online softmax buffers.
pub const MAX_GROUP_SIZE: usize = 128;

/// Maximum unsigned integer code for asymmetric 4-bit quantization (`2^4 - 1`).
pub const QMAX_INT4: u32 = 15;

/// Maximum unsigned integer code for asymmetric 2-bit quantization (`2^2 - 1`).
pub const QMAX_INT2: u32 = 3;

const MAX_VALUE_SUBGROUPS: usize = MAX_HEAD_DIM / 16;
const F16_FINITE_MAX: f32 = 65504.0;

/// Errors returned by KIVI quantization, cache management, and fused attention routines.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum KiviError {
    /// `head_size` must be a non-zero multiple of 16 and at most [`MAX_HEAD_DIM`].
    #[error("invalid head_size {head_size}: expected a multiple of 16 in 16..={MAX_HEAD_DIM}")]
    InvalidHeadSize {
        /// Requested per-head dimension.
        head_size: usize,
    },
    /// `group_size` must be a non-zero multiple of 16 and at most [`MAX_GROUP_SIZE`].
    #[error("invalid group_size {group_size}: expected a multiple of 16 in 16..={MAX_GROUP_SIZE}")]
    InvalidGroupSize {
        /// Requested token group size.
        group_size: usize,
    },
    /// `value_group_size` must be a multiple of 16 that evenly divides `head_size`.
    #[error(
        "invalid value_group_size {value_group_size} for head_size {head_size}: \
         expected a multiple of 16 dividing head_size"
    )]
    InvalidValueGroupSize {
        /// Requested per-token channel subgroup width.
        value_group_size: usize,
        /// Per-head dimension.
        head_size: usize,
    },
    /// Query / KV head counts do not form a valid MHA / GQA / MQA ratio.
    #[error(
        "invalid head topology (num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}): \
         both must be non-zero and num_q_heads must be a multiple of num_kv_heads"
    )]
    InvalidHeadTopology {
        /// Number of query attention heads.
        num_q_heads: usize,
        /// Number of key/value heads.
        num_kv_heads: usize,
    },
    /// Slice length or head index did not match the configured layout.
    #[error("dimension mismatch in {context}: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected element count or bound.
        expected: usize,
        /// Actual element count or index.
        actual: usize,
        /// Operation context label.
        context: &'static str,
    },
    /// Attempted to instantiate a quantized KIVI KV cache on a recurrent GDN layer.
    #[error("recurrent GDN state layers are exempt from KIVI KV quantization")]
    RecurrentLayerExempt,
}

/// Bit-width selector for KIVI asymmetric quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KiviBits {
    /// 4-bit unsigned affine codes (`0..=15`), packed 8 codes per `u32` word.
    Int4,
    /// 2-bit unsigned affine codes (`0..=3`), packed 16 codes per `u32` word.
    Int2,
}

impl KiviBits {
    /// Bit width per quantized element (`4` or `2`).
    #[inline]
    pub const fn bits(self) -> usize {
        match self {
            Self::Int4 => 4,
            Self::Int2 => 2,
        }
    }

    /// Maximum integer quantization code (`15` for `Int4`, `3` for `Int2`).
    #[inline]
    pub const fn qmax(self) -> u32 {
        match self {
            Self::Int4 => QMAX_INT4,
            Self::Int2 => QMAX_INT2,
        }
    }

    /// Maximum integer quantization code as `f32`.
    #[inline]
    pub const fn qmax_f32(self) -> f32 {
        match self {
            Self::Int4 => 15.0,
            Self::Int2 => 3.0,
        }
    }

    /// Number of quantized codes packed into a single little-endian `u32` word.
    #[inline]
    pub const fn codes_per_word(self) -> usize {
        match self {
            Self::Int4 => 8,
            Self::Int2 => 16,
        }
    }

    /// Worst-case span-relative round-off bound `1 / (2 * QMAX)`.
    #[inline]
    pub const fn span_error_bound(self) -> f32 {
        match self {
            Self::Int4 => 1.0 / 30.0,
            Self::Int2 => 1.0 / 6.0,
        }
    }

    /// Number of `u32` words needed to pack `elem_count` codes (`elem_count` must be a multiple
    /// of [`Self::codes_per_word`]).
    #[inline]
    pub const fn packed_words(self, elem_count: usize) -> usize {
        elem_count / self.codes_per_word()
    }
}

/// Storage precision for the unquantized recent-token residual window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ResidualPrecision {
    /// Store recent residual `K` and `V` vectors in IEEE-754 `f16` (2 bytes/element).
    #[default]
    Fp16,
    /// Store recent residual `K` and `V` vectors in `f32` (4 bytes/element).
    Fp32,
}

impl ResidualPrecision {
    /// Bytes per scalar element in the residual buffer.
    #[inline]
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::Fp16 => 2,
            Self::Fp32 => 4,
        }
    }
}

/// Decoder layer state classification for hybrid attention + recurrent architectures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerStateKind {
    /// Standard, GQA, MQA, or MLA paged attention KV cache layer.
    AttentionKv,
    /// Recurrent Gated DeltaNet (GDN) or linear-attention state layer (must stay unquantized).
    RecurrentGdn,
}

/// Configuration for a [`KiviKvCache`] instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KiviConfig {
    /// Target quantization bit-width (`Int4` or `Int2`).
    pub bits: KiviBits,
    /// Sequence group length `G` for per-channel Key quantization (`16..=128`, multiple of 16).
    pub key_group_size: usize,
    /// Optional channel subgroup width for per-token Value quantization. `None` uses one
    /// `(scale, zero_point)` pair per token across the full `head_size`.
    pub value_group_size: Option<usize>,
    /// Maximum number of recent tokens `R` retained in the unquantized residual window before
    /// the oldest completed group of `G` tokens flushes atomically into quantized pages.
    pub residual_length: usize,
    /// Element format used inside the sliding residual buffer.
    pub residual_precision: ResidualPrecision,
}

impl Default for KiviConfig {
    fn default() -> Self {
        Self::int4_default()
    }
}

impl KiviConfig {
    /// Create a KIVI configuration with `FP16` residual storage and whole-token Value grouping.
    #[inline]
    pub const fn new(bits: KiviBits, key_group_size: usize, residual_length: usize) -> Self {
        Self {
            bits,
            key_group_size,
            value_group_size: None,
            residual_length,
            residual_precision: ResidualPrecision::Fp16,
        }
    }

    /// Production default for 4-bit KIVI (`G = 64`, `R = 64`, `FP16` residual).
    #[inline]
    pub const fn int4_default() -> Self {
        Self::new(KiviBits::Int4, 64, 64)
    }

    /// Production default for 2-bit KIVI (`G = 64`, `R = 64`, `FP16` residual).
    #[inline]
    pub const fn int2_default() -> Self {
        Self::new(KiviBits::Int2, 64, 64)
    }

    /// Resolve and validate the effective Value channel group width for `head_size`.
    pub fn validate(&self, head_size: usize) -> Result<usize, KiviError> {
        validate_head_size(head_size)?;
        validate_group_size(self.key_group_size)?;
        let v_group = self.value_group_size.unwrap_or(head_size);
        if v_group < 16 || !v_group.is_multiple_of(16) || !head_size.is_multiple_of(v_group) {
            return Err(KiviError::InvalidValueGroupSize {
                value_group_size: v_group,
                head_size,
            });
        }
        Ok(v_group)
    }
}

/// Resolved per-layer compression plan preserving recurrent GDN layers in full precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerCompressionPlan {
    /// Apply KIVI asymmetric KV compression on this attention layer.
    QuantizedAttention(KiviConfig),
    /// Leave recurrent GDN state in full precision without KV quantization.
    UnquantizedRecurrentGdn,
}

/// Resolve the safe compression policy for a decoder layer.
#[inline]
pub const fn resolve_layer_plan(
    layer_kind: LayerStateKind,
    requested: KiviConfig,
) -> LayerCompressionPlan {
    match layer_kind {
        LayerStateKind::AttentionKv => LayerCompressionPlan::QuantizedAttention(requested),
        LayerStateKind::RecurrentGdn => LayerCompressionPlan::UnquantizedRecurrentGdn,
    }
}

#[inline]
fn validate_head_size(head_size: usize) -> Result<(), KiviError> {
    if head_size < 16 || head_size > MAX_HEAD_DIM || !head_size.is_multiple_of(16) {
        return Err(KiviError::InvalidHeadSize { head_size });
    }
    Ok(())
}

#[inline]
fn validate_group_size(group_size: usize) -> Result<(), KiviError> {
    if group_size < 16 || group_size > MAX_GROUP_SIZE || !group_size.is_multiple_of(16) {
        return Err(KiviError::InvalidGroupSize { group_size });
    }
    Ok(())
}

#[inline]
fn sanitize_f32(val: f32) -> f32 {
    if val.is_finite() {
        val
    } else {
        0.0
    }
}

#[inline]
fn sanitize_f16_clamped(val: f32) -> f32 {
    sanitize_f32(val).clamp(-F16_FINITE_MAX, F16_FINITE_MAX)
}

#[inline]
fn to_f16_clamped(val: f32) -> f16 {
    f16::from_f32(sanitize_f16_clamped(val))
}

/// Pack `(scale, zero_point)` as two `f16` values inside one `u32` word (low 16 bits = `scale`,
/// high 16 bits = `zero_point`), returning `(packed_word, scale_f32, zero_f32)`.
#[inline]
pub fn pack_affine_params_f16(scale: f32, zero_point: f32) -> (u32, f32, f32) {
    let s_f16 = f16::from_f32(sanitize_f16_clamped(scale).max(0.0));
    let z_f16 = f16::from_f32(sanitize_f16_clamped(zero_point));
    let word = u32::from(s_f16.to_bits()) | (u32::from(z_f16.to_bits()) << 16);
    (word, s_f16.to_f32(), z_f16.to_f32())
}

/// Unpack `(scale, zero_point)` from a `u32` word written by [`pack_affine_params_f16`].
#[inline]
pub fn unpack_affine_params_f16(word: u32) -> (f32, f32) {
    let scale = f16::from_bits((word & 0xFFFF) as u16).to_f32();
    let zero = f16::from_bits((word >> 16) as u16).to_f32();
    (scale, zero)
}

/// Quantize a single vector with exact `f32` `(scale, zero_point)` metadata into bit-packed
/// little-endian `u32` words. Used as the backend-agnostic numerical oracle across arbitrary
/// `f32` dynamic ranges (`1e-6..1e6`).
pub fn quantize_affine_f32(bits: KiviBits, values: &[f32], packed_out: &mut [u32]) -> (f32, f32) {
    let n = values.len();
    let cpw = bits.codes_per_word();
    debug_assert_eq!(n % cpw, 0);
    debug_assert_eq!(packed_out.len(), n / cpw);

    if n == 0 {
        return (0.0, 0.0);
    }

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &raw in values {
        let v = sanitize_f32(raw);
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let span = (max_val - min_val).max(0.0);
    let qmax = bits.qmax_f32();
    let scale = if span > 0.0 { span / qmax } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    match bits {
        KiviBits::Int4 => {
            for (chunk, out_word) in values.chunks_exact(8).zip(packed_out.iter_mut()) {
                let mut word = 0u32;
                for (lane, &raw) in chunk.iter().enumerate() {
                    let v = sanitize_f32(raw);
                    let q = if inv_scale > 0.0 {
                        ((v - min_val) * inv_scale).round().clamp(0.0, qmax) as u32
                    } else {
                        0
                    };
                    word |= (q & 0x0F) << (lane * 4);
                }
                *out_word = word;
            }
        }
        KiviBits::Int2 => {
            for (chunk, out_word) in values.chunks_exact(16).zip(packed_out.iter_mut()) {
                let mut word = 0u32;
                for (lane, &raw) in chunk.iter().enumerate() {
                    let v = sanitize_f32(raw);
                    let q = if inv_scale > 0.0 {
                        ((v - min_val) * inv_scale).round().clamp(0.0, qmax) as u32
                    } else {
                        0
                    };
                    word |= (q & 0x03) << (lane * 2);
                }
                *out_word = word;
            }
        }
    }

    (scale, min_val)
}

/// Dequantize bit-packed codes produced by [`quantize_affine_f32`] back to `f32`.
pub fn dequantize_affine_f32(
    bits: KiviBits,
    packed: &[u32],
    scale: f32,
    zero_point: f32,
    out: &mut [f32],
) {
    let cpw = bits.codes_per_word();
    debug_assert_eq!(out.len(), packed.len() * cpw);

    match bits {
        KiviBits::Int4 => {
            for (&word, out_chunk) in packed.iter().zip(out.chunks_exact_mut(8)) {
                for (lane, slot) in out_chunk.iter_mut().enumerate() {
                    let q = (word >> (lane * 4)) & 0x0F;
                    *slot = zero_point + (q as f32) * scale;
                }
            }
        }
        KiviBits::Int2 => {
            for (&word, out_chunk) in packed.iter().zip(out.chunks_exact_mut(16)) {
                for (lane, slot) in out_chunk.iter_mut().enumerate() {
                    let q = (word >> (lane * 2)) & 0x03;
                    *slot = zero_point + (q as f32) * scale;
                }
            }
        }
    }
}

/// Span-relative round-trip reconstruction error `max_i |x_i - x_hat_i| / (max_i x_i - min_i x_i)`
/// under [`quantize_affine_f32`]. Bounded above by `1/30` (`Int4`) and `1/6` (`Int2`).
pub fn span_relative_error_f32(bits: KiviBits, values: &[f32]) -> f32 {
    let n = values.len();
    let mut packed = vec![0u32; bits.packed_words(n)];
    let (scale, zero_point) = quantize_affine_f32(bits, values, &mut packed);
    let mut back = vec![0.0f32; n];
    dequantize_affine_f32(bits, &packed, scale, zero_point, &mut back);

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut max_err = 0.0f32;
    for (&raw, &rec) in values.iter().zip(&back) {
        let v = sanitize_f32(raw);
        min_val = min_val.min(v);
        max_val = max_val.max(v);
        max_err = max_err.max((v - rec).abs());
    }
    let span = max_val - min_val;
    if span == 0.0 {
        max_err
    } else {
        max_err / span
    }
}

/// Bit-packed per-channel quantized Key page covering one completed group of `G` tokens for a
/// single KV head.
#[derive(Clone, Debug)]
pub struct KiviKeyPage {
    bits: KiviBits,
    group_size: usize,
    head_size: usize,
    /// Length `head_size`; each entry packs `(f16 scale_c, f16 zero_c)` for channel `c`.
    channel_params: Vec<u32>,
    /// Token-major packed codes of length `group_size * (head_size / codes_per_word)`.
    packed_codes: Vec<u32>,
}

impl KiviKeyPage {
    /// Quantize a completed `[group_size, head_size]` row-major Key block per-channel.
    pub fn quantize(
        bits: KiviBits,
        keys: &[f32],
        group_size: usize,
        head_size: usize,
    ) -> Result<Self, KiviError> {
        validate_group_size(group_size)?;
        validate_head_size(head_size)?;
        let expected = group_size * head_size;
        if keys.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: keys.len(),
                context: "KiviKeyPage::quantize",
            });
        }

        let mut ch_min = [f32::INFINITY; MAX_HEAD_DIM];
        let mut ch_max = [f32::NEG_INFINITY; MAX_HEAD_DIM];
        for row in keys.chunks_exact(head_size) {
            for (c, &raw) in row.iter().enumerate() {
                let v = sanitize_f16_clamped(raw);
                ch_min[c] = ch_min[c].min(v);
                ch_max[c] = ch_max[c].max(v);
            }
        }

        let qmax = bits.qmax_f32();
        let mut channel_params = vec![0u32; head_size];
        let mut zero_hat = [0.0f32; MAX_HEAD_DIM];
        let mut inv_scale = [0.0f32; MAX_HEAD_DIM];

        for (c, param_slot) in channel_params.iter_mut().enumerate() {
            let z_f16 = f16::from_f32(ch_min[c]);
            let z_val = z_f16.to_f32();
            let raw_scale = ((ch_max[c] - z_val).max(0.0) / qmax).min(F16_FINITE_MAX);
            let s_f16 = f16::from_f32(raw_scale);
            let s_val = s_f16.to_f32();
            *param_slot = u32::from(s_f16.to_bits()) | (u32::from(z_f16.to_bits()) << 16);
            zero_hat[c] = z_val;
            inv_scale[c] = if s_val > 0.0 { 1.0 / s_val } else { 0.0 };
        }

        let wpt = bits.packed_words(head_size);
        let mut codes = vec![0u32; group_size * wpt];

        match bits {
            KiviBits::Int4 => {
                for (row, tok_w) in keys
                    .chunks_exact(head_size)
                    .zip(codes.chunks_exact_mut(wpt))
                {
                    for (w_idx, (chunk, dst)) in row.chunks_exact(8).zip(tok_w).enumerate() {
                        let base = w_idx * 8;
                        let mut word = 0u32;
                        for (lane, &raw) in chunk.iter().enumerate() {
                            let c = base + lane;
                            let inv = inv_scale[c];
                            let q = if inv > 0.0 {
                                ((sanitize_f16_clamped(raw) - zero_hat[c]) * inv)
                                    .round()
                                    .clamp(0.0, qmax) as u32
                            } else {
                                0
                            };
                            word |= (q & 0x0F) << (lane * 4);
                        }
                        *dst = word;
                    }
                }
            }
            KiviBits::Int2 => {
                for (row, tok_w) in keys
                    .chunks_exact(head_size)
                    .zip(codes.chunks_exact_mut(wpt))
                {
                    for (w_idx, (chunk, dst)) in row.chunks_exact(16).zip(tok_w).enumerate() {
                        let base = w_idx * 16;
                        let mut word = 0u32;
                        for (lane, &raw) in chunk.iter().enumerate() {
                            let c = base + lane;
                            let inv = inv_scale[c];
                            let q = if inv > 0.0 {
                                ((sanitize_f16_clamped(raw) - zero_hat[c]) * inv)
                                    .round()
                                    .clamp(0.0, qmax) as u32
                            } else {
                                0
                            };
                            word |= (q & 0x03) << (lane * 2);
                        }
                        *dst = word;
                    }
                }
            }
        }

        Ok(Self {
            bits,
            group_size,
            head_size,
            channel_params,
            packed_codes: codes,
        })
    }

    /// Total storage footprint of this Key page in bytes (packed codes + co-located metadata).
    #[inline]
    pub fn byte_footprint(&self) -> usize {
        (self.packed_codes.len() + self.channel_params.len()) * core::mem::size_of::<u32>()
    }

    /// Raw packed `u32` code words (token-major).
    #[inline]
    pub fn packed_codes(&self) -> &[u32] {
        &self.packed_codes
    }

    /// Co-located per-channel `(f16 scale, f16 zero_point)` words.
    #[inline]
    pub fn channel_params(&self) -> &[u32] {
        &self.channel_params
    }

    /// Dequantize the entire `[group_size, head_size]` block to `f32` (verification helper; not
    /// invoked on the fused attention path).
    pub fn dequantize(&self, out: &mut [f32]) -> Result<(), KiviError> {
        let expected = self.group_size * self.head_size;
        if out.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: out.len(),
                context: "KiviKeyPage::dequantize",
            });
        }

        let mut scales = [0.0f32; MAX_HEAD_DIM];
        let mut zeros = [0.0f32; MAX_HEAD_DIM];
        for (c, &param) in self.channel_params.iter().enumerate() {
            let (s, z) = unpack_affine_params_f16(param);
            scales[c] = s;
            zeros[c] = z;
        }

        let wpt = self.bits.packed_words(self.head_size);
        match self.bits {
            KiviBits::Int4 => {
                for (tok_w, out_row) in self
                    .packed_codes
                    .chunks_exact(wpt)
                    .zip(out.chunks_exact_mut(self.head_size))
                {
                    for (w_idx, (&word, dst)) in
                        tok_w.iter().zip(out_row.chunks_exact_mut(8)).enumerate()
                    {
                        let base = w_idx * 8;
                        for (lane, slot) in dst.iter_mut().enumerate() {
                            let c = base + lane;
                            let q = (word >> (lane * 4)) & 0x0F;
                            *slot = zeros[c] + (q as f32) * scales[c];
                        }
                    }
                }
            }
            KiviBits::Int2 => {
                for (tok_w, out_row) in self
                    .packed_codes
                    .chunks_exact(wpt)
                    .zip(out.chunks_exact_mut(self.head_size))
                {
                    for (w_idx, (&word, dst)) in
                        tok_w.iter().zip(out_row.chunks_exact_mut(16)).enumerate()
                    {
                        let base = w_idx * 16;
                        for (lane, slot) in dst.iter_mut().enumerate() {
                            let c = base + lane;
                            let q = (word >> (lane * 2)) & 0x03;
                            *slot = zeros[c] + (q as f32) * scales[c];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute scaled attention logits `out_scores[t] = softmax_scale * (query . K_hat[t])`
    /// directly from bit-packed codes without materializing dequantized keys.
    ///
    /// Hoists `sum_c (softmax_scale * query[c] * zero[c])` and `q_scaled[c]` once across the
    /// entire `group_size`-token page.
    pub fn score_query(
        &self,
        query: &[f32],
        softmax_scale: f32,
        out_scores: &mut [f32],
    ) -> Result<(), KiviError> {
        if query.len() != self.head_size {
            return Err(KiviError::DimensionMismatch {
                expected: self.head_size,
                actual: query.len(),
                context: "KiviKeyPage::score_query(query)",
            });
        }
        if out_scores.len() != self.group_size {
            return Err(KiviError::DimensionMismatch {
                expected: self.group_size,
                actual: out_scores.len(),
                context: "KiviKeyPage::score_query(out_scores)",
            });
        }

        let mut zero_bias = 0.0f32;
        let mut q_scaled = [0.0f32; MAX_HEAD_DIM];
        for (c, (&q_raw, &param)) in query.iter().zip(&self.channel_params).enumerate() {
            let (scale_c, zero_c) = unpack_affine_params_f16(param);
            let q_val = sanitize_f32(q_raw) * softmax_scale;
            zero_bias += q_val * zero_c;
            q_scaled[c] = q_val * scale_c;
        }

        let wpt = self.bits.packed_words(self.head_size);
        let codes = &self.packed_codes;
        match self.bits {
            KiviBits::Int4 => {
                for (tok_w, score_slot) in codes.chunks_exact(wpt).zip(out_scores.iter_mut()) {
                    let mut dot = zero_bias;
                    for (w_idx, &word) in tok_w.iter().enumerate() {
                        let qs = &q_scaled[w_idx * 8..w_idx * 8 + 8];
                        dot += ((word & 0x0F) as f32) * qs[0]
                            + (((word >> 4) & 0x0F) as f32) * qs[1]
                            + (((word >> 8) & 0x0F) as f32) * qs[2]
                            + (((word >> 12) & 0x0F) as f32) * qs[3]
                            + (((word >> 16) & 0x0F) as f32) * qs[4]
                            + (((word >> 20) & 0x0F) as f32) * qs[5]
                            + (((word >> 24) & 0x0F) as f32) * qs[6]
                            + (((word >> 28) & 0x0F) as f32) * qs[7];
                    }
                    *score_slot = dot;
                }
            }
            KiviBits::Int2 => {
                for (tok_w, score_slot) in codes.chunks_exact(wpt).zip(out_scores.iter_mut()) {
                    let mut dot = zero_bias;
                    for (w_idx, &word) in tok_w.iter().enumerate() {
                        let qs = &q_scaled[w_idx * 16..w_idx * 16 + 16];
                        dot += ((word & 0x03) as f32) * qs[0]
                            + (((word >> 2) & 0x03) as f32) * qs[1]
                            + (((word >> 4) & 0x03) as f32) * qs[2]
                            + (((word >> 6) & 0x03) as f32) * qs[3]
                            + (((word >> 8) & 0x03) as f32) * qs[4]
                            + (((word >> 10) & 0x03) as f32) * qs[5]
                            + (((word >> 12) & 0x03) as f32) * qs[6]
                            + (((word >> 14) & 0x03) as f32) * qs[7]
                            + (((word >> 16) & 0x03) as f32) * qs[8]
                            + (((word >> 18) & 0x03) as f32) * qs[9]
                            + (((word >> 20) & 0x03) as f32) * qs[10]
                            + (((word >> 22) & 0x03) as f32) * qs[11]
                            + (((word >> 24) & 0x03) as f32) * qs[12]
                            + (((word >> 26) & 0x03) as f32) * qs[13]
                            + (((word >> 28) & 0x03) as f32) * qs[14]
                            + (((word >> 30) & 0x03) as f32) * qs[15];
                    }
                    *score_slot = dot;
                }
            }
        }
        Ok(())
    }
}

/// Bit-packed per-token quantized Value page covering `G` tokens for a single KV head.
#[derive(Clone, Debug)]
pub struct KiviValuePage {
    bits: KiviBits,
    group_size: usize,
    head_size: usize,
    value_group_size: usize,
    /// Length `group_size * (head_size / value_group_size)`; packed `(f16 scale, f16 zero)`.
    token_params: Vec<u32>,
    /// Token-major packed codes of length `group_size * (head_size / codes_per_word)`.
    packed_codes: Vec<u32>,
}

impl KiviValuePage {
    /// Quantize a `[group_size, head_size]` row-major Value block per-token (with channel
    /// subgroup width `value_group_size`).
    pub fn quantize(
        bits: KiviBits,
        values: &[f32],
        group_size: usize,
        head_size: usize,
        value_group_size: usize,
    ) -> Result<Self, KiviError> {
        validate_group_size(group_size)?;
        validate_head_size(head_size)?;
        if value_group_size < 16
            || !value_group_size.is_multiple_of(16)
            || !head_size.is_multiple_of(value_group_size)
        {
            return Err(KiviError::InvalidValueGroupSize {
                value_group_size,
                head_size,
            });
        }
        let expected = group_size * head_size;
        if values.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: values.len(),
                context: "KiviValuePage::quantize",
            });
        }

        let subgroups = head_size / value_group_size;
        let wpt = bits.packed_words(head_size);
        let words_per_sub = bits.packed_words(value_group_size);
        let qmax = bits.qmax_f32();

        let mut token_params = vec![0u32; group_size * subgroups];
        let mut packed_codes = vec![0u32; group_size * wpt];

        for ((row, row_params), row_words) in values
            .chunks_exact(head_size)
            .zip(token_params.chunks_exact_mut(subgroups))
            .zip(packed_codes.chunks_exact_mut(wpt))
        {
            for ((sub_slice, param_slot), sub_words) in row
                .chunks_exact(value_group_size)
                .zip(row_params.iter_mut())
                .zip(row_words.chunks_exact_mut(words_per_sub))
            {
                let mut v_min = f32::INFINITY;
                let mut v_max = f32::NEG_INFINITY;
                for &raw in sub_slice {
                    let v = sanitize_f16_clamped(raw);
                    v_min = v_min.min(v);
                    v_max = v_max.max(v);
                }
                let z_f16 = f16::from_f32(v_min);
                let z_val = z_f16.to_f32();
                let raw_scale = ((v_max - z_val).max(0.0) / qmax).min(F16_FINITE_MAX);
                let s_f16 = f16::from_f32(raw_scale);
                let s_val = s_f16.to_f32();
                let inv = if s_val > 0.0 { 1.0 / s_val } else { 0.0 };
                *param_slot = u32::from(s_f16.to_bits()) | (u32::from(z_f16.to_bits()) << 16);

                match bits {
                    KiviBits::Int4 => {
                        for (chunk, dst) in sub_slice.chunks_exact(8).zip(sub_words.iter_mut()) {
                            let mut word = 0u32;
                            for (lane, &raw) in chunk.iter().enumerate() {
                                let q = if inv > 0.0 {
                                    ((sanitize_f16_clamped(raw) - z_val) * inv)
                                        .round()
                                        .clamp(0.0, qmax) as u32
                                } else {
                                    0
                                };
                                word |= (q & 0x0F) << (lane * 4);
                            }
                            *dst = word;
                        }
                    }
                    KiviBits::Int2 => {
                        for (chunk, dst) in sub_slice.chunks_exact(16).zip(sub_words.iter_mut()) {
                            let mut word = 0u32;
                            for (lane, &raw) in chunk.iter().enumerate() {
                                let q = if inv > 0.0 {
                                    ((sanitize_f16_clamped(raw) - z_val) * inv)
                                        .round()
                                        .clamp(0.0, qmax) as u32
                                } else {
                                    0
                                };
                                word |= (q & 0x03) << (lane * 2);
                            }
                            *dst = word;
                        }
                    }
                }
            }
        }

        Ok(Self {
            bits,
            group_size,
            head_size,
            value_group_size,
            token_params,
            packed_codes,
        })
    }

    /// Total storage footprint of this Value page in bytes (packed codes + co-located metadata).
    #[inline]
    pub fn byte_footprint(&self) -> usize {
        (self.packed_codes.len() + self.token_params.len()) * core::mem::size_of::<u32>()
    }

    /// Dequantize the `[group_size, head_size]` Value block back to `f32` (verification helper).
    pub fn dequantize(&self, out: &mut [f32]) -> Result<(), KiviError> {
        let expected = self.group_size * self.head_size;
        if out.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: out.len(),
                context: "KiviValuePage::dequantize",
            });
        }

        let subgroups = self.head_size / self.value_group_size;
        let wpt = self.bits.packed_words(self.head_size);
        let words_per_sub = self.bits.packed_words(self.value_group_size);

        for ((row_params, row_words), out_row) in self
            .token_params
            .chunks_exact(subgroups)
            .zip(self.packed_codes.chunks_exact(wpt))
            .zip(out.chunks_exact_mut(self.head_size))
        {
            for ((&param, sub_words), out_sub) in row_params
                .iter()
                .zip(row_words.chunks_exact(words_per_sub))
                .zip(out_row.chunks_exact_mut(self.value_group_size))
            {
                let (scale, zero) = unpack_affine_params_f16(param);
                dequantize_affine_f32(self.bits, sub_words, scale, zero, out_sub);
            }
        }
        Ok(())
    }

    /// Accumulate `out_accum[0..head_size] += sum_t weights[t] * V_hat[t]` directly from packed
    /// codes without materializing a dequantized `[group_size, head_size]` tensor.
    ///
    /// Hoists `sum_t (weights[t] * zero_v[t, sub])` into a stack scalar per subgroup so the
    /// inner loop performs only `w_scaled * code` FMAs.
    pub fn accumulate_weighted(
        &self,
        weights: &[f32],
        out_accum: &mut [f32],
    ) -> Result<(), KiviError> {
        if weights.len() != self.group_size {
            return Err(KiviError::DimensionMismatch {
                expected: self.group_size,
                actual: weights.len(),
                context: "KiviValuePage::accumulate_weighted(weights)",
            });
        }
        if out_accum.len() != self.head_size {
            return Err(KiviError::DimensionMismatch {
                expected: self.head_size,
                actual: out_accum.len(),
                context: "KiviValuePage::accumulate_weighted(out_accum)",
            });
        }

        let subgroups = self.head_size / self.value_group_size;
        let wpt = self.bits.packed_words(self.head_size);
        let words_per_sub = self.bits.packed_words(self.value_group_size);
        let mut zero_accum = [0.0f32; MAX_VALUE_SUBGROUPS];

        for ((&w_t, row_params), row_words) in weights
            .iter()
            .zip(self.token_params.chunks_exact(subgroups))
            .zip(self.packed_codes.chunks_exact(wpt))
        {
            if w_t == 0.0 {
                continue;
            }
            for (sub_idx, ((&param, sub_words), out_sub)) in row_params
                .iter()
                .zip(row_words.chunks_exact(words_per_sub))
                .zip(out_accum.chunks_exact_mut(self.value_group_size))
                .enumerate()
            {
                let (scale_v, zero_v) = unpack_affine_params_f16(param);
                zero_accum[sub_idx] += w_t * zero_v;
                let w_scaled = w_t * scale_v;
                if w_scaled == 0.0 {
                    continue;
                }
                match self.bits {
                    KiviBits::Int4 => {
                        for (&word, dst) in sub_words.iter().zip(out_sub.chunks_exact_mut(8)) {
                            dst[0] += w_scaled * ((word & 0x0F) as f32);
                            dst[1] += w_scaled * (((word >> 4) & 0x0F) as f32);
                            dst[2] += w_scaled * (((word >> 8) & 0x0F) as f32);
                            dst[3] += w_scaled * (((word >> 12) & 0x0F) as f32);
                            dst[4] += w_scaled * (((word >> 16) & 0x0F) as f32);
                            dst[5] += w_scaled * (((word >> 20) & 0x0F) as f32);
                            dst[6] += w_scaled * (((word >> 24) & 0x0F) as f32);
                            dst[7] += w_scaled * (((word >> 28) & 0x0F) as f32);
                        }
                    }
                    KiviBits::Int2 => {
                        for (&word, dst) in sub_words.iter().zip(out_sub.chunks_exact_mut(16)) {
                            dst[0] += w_scaled * ((word & 0x03) as f32);
                            dst[1] += w_scaled * (((word >> 2) & 0x03) as f32);
                            dst[2] += w_scaled * (((word >> 4) & 0x03) as f32);
                            dst[3] += w_scaled * (((word >> 6) & 0x03) as f32);
                            dst[4] += w_scaled * (((word >> 8) & 0x03) as f32);
                            dst[5] += w_scaled * (((word >> 10) & 0x03) as f32);
                            dst[6] += w_scaled * (((word >> 12) & 0x03) as f32);
                            dst[7] += w_scaled * (((word >> 14) & 0x03) as f32);
                            dst[8] += w_scaled * (((word >> 16) & 0x03) as f32);
                            dst[9] += w_scaled * (((word >> 18) & 0x03) as f32);
                            dst[10] += w_scaled * (((word >> 20) & 0x03) as f32);
                            dst[11] += w_scaled * (((word >> 22) & 0x03) as f32);
                            dst[12] += w_scaled * (((word >> 24) & 0x03) as f32);
                            dst[13] += w_scaled * (((word >> 26) & 0x03) as f32);
                            dst[14] += w_scaled * (((word >> 28) & 0x03) as f32);
                            dst[15] += w_scaled * (((word >> 30) & 0x03) as f32);
                        }
                    }
                }
            }
        }

        let sub_w = self.value_group_size;
        for (&bias, out_sub) in zero_accum[..subgroups]
            .iter()
            .zip(out_accum.chunks_exact_mut(sub_w))
        {
            if bias != 0.0 {
                for slot in out_sub {
                    *slot += bias;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum ResidualStorage {
    Fp16 { keys: Vec<f16>, values: Vec<f16> },
    Fp32 { keys: Vec<f32>, values: Vec<f32> },
}

impl ResidualStorage {
    fn with_capacity(precision: ResidualPrecision, elem_cap: usize) -> Self {
        match precision {
            ResidualPrecision::Fp16 => Self::Fp16 {
                keys: Vec::with_capacity(elem_cap),
                values: Vec::with_capacity(elem_cap),
            },
            ResidualPrecision::Fp32 => Self::Fp32 {
                keys: Vec::with_capacity(elem_cap),
                values: Vec::with_capacity(elem_cap),
            },
        }
    }

    #[inline]
    fn token_len(&self, head_size: usize) -> usize {
        match self {
            Self::Fp16 { keys, .. } => keys.len() / head_size,
            Self::Fp32 { keys, .. } => keys.len() / head_size,
        }
    }

    #[inline]
    fn byte_footprint(&self) -> usize {
        match self {
            Self::Fp16 { keys, values } => (keys.len() + values.len()) * 2,
            Self::Fp32 { keys, values } => (keys.len() + values.len()) * 4,
        }
    }

    fn push_slice(&mut self, key: &[f32], value: &[f32]) {
        match self {
            Self::Fp16 { keys, values } => {
                keys.extend(key.iter().copied().map(to_f16_clamped));
                values.extend(value.iter().copied().map(to_f16_clamped));
            }
            Self::Fp32 { keys, values } => {
                keys.extend(key.iter().copied().map(sanitize_f32));
                values.extend(value.iter().copied().map(sanitize_f32));
            }
        }
    }

    fn drain_prefix_to_f32(&mut self, elem_count: usize, k_out: &mut [f32], v_out: &mut [f32]) {
        match self {
            Self::Fp16 { keys, values } => {
                for (dst, src) in k_out.iter_mut().zip(keys.drain(..elem_count)) {
                    *dst = src.to_f32();
                }
                for (dst, src) in v_out.iter_mut().zip(values.drain(..elem_count)) {
                    *dst = src.to_f32();
                }
            }
            Self::Fp32 { keys, values } => {
                for (dst, src) in k_out.iter_mut().zip(keys.drain(..elem_count)) {
                    *dst = src;
                }
                for (dst, src) in v_out.iter_mut().zip(values.drain(..elem_count)) {
                    *dst = src;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct HeadKvState {
    key_pages: Vec<KiviKeyPage>,
    value_pages: Vec<KiviValuePage>,
    residual: ResidualStorage,
}

/// Resident memory breakdown of a [`KiviKvCache`] compared against uncompressed baselines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KiviMemoryFootprint {
    /// Total cached sequence length (`quantized_tokens + residual_tokens`).
    pub seq_len: usize,
    /// Number of tokens stored in bit-packed quantized pages.
    pub quantized_tokens: usize,
    /// Number of recent tokens stored in the full-precision residual buffer.
    pub residual_tokens: usize,
    /// Number of completed `G`-token pages per KV head.
    pub num_pages_per_head: usize,
    /// Resident bytes occupied by quantized Key pages (codes + channel scales/zeros).
    pub quantized_key_bytes: usize,
    /// Resident bytes occupied by quantized Value pages (codes + token scales/zeros).
    pub quantized_value_bytes: usize,
    /// Resident bytes occupied by the full-precision residual window across all KV heads.
    pub residual_bytes: usize,
    /// Total resident bytes (`quantized_key_bytes + quantized_value_bytes + residual_bytes`).
    pub total_bytes: usize,
    /// Uncompressed `FP16` KV cache footprint for the same `(num_kv_heads, seq_len, head_size)`.
    pub fp16_baseline_bytes: usize,
    /// Uncompressed `FP32` KV cache footprint for the same `(num_kv_heads, seq_len, head_size)`.
    pub fp32_baseline_bytes: usize,
}

impl KiviMemoryFootprint {
    /// Memory reduction multiple relative to an `FP16` KV cache (`fp16_bytes / total_bytes`).
    #[inline]
    pub fn compression_ratio_vs_fp16(&self) -> f64 {
        if self.total_bytes == 0 {
            1.0
        } else {
            self.fp16_baseline_bytes as f64 / self.total_bytes as f64
        }
    }

    /// Memory reduction multiple relative to an `FP32` KV cache (`fp32_bytes / total_bytes`).
    #[inline]
    pub fn compression_ratio_vs_fp32(&self) -> f64 {
        if self.total_bytes == 0 {
            1.0
        } else {
            self.fp32_baseline_bytes as f64 / self.total_bytes as f64
        }
    }
}

/// Multi-head paged KIVI KV cache with sliding residual window and zero-allocation online fused
/// attention execution.
#[derive(Clone, Debug)]
pub struct KiviKvCache {
    num_kv_heads: usize,
    head_size: usize,
    value_group_size: usize,
    config: KiviConfig,
    heads: Vec<HeadKvState>,
}

impl KiviKvCache {
    /// Construct an empty KIVI KV cache for an attention layer.
    pub fn new(
        num_kv_heads: usize,
        head_size: usize,
        config: KiviConfig,
    ) -> Result<Self, KiviError> {
        if num_kv_heads == 0 {
            return Err(KiviError::InvalidHeadTopology {
                num_q_heads: 0,
                num_kv_heads: 0,
            });
        }
        let value_group_size = config.validate(head_size)?;
        let cap_tokens = config.residual_length.max(config.key_group_size) + config.key_group_size;
        let elem_cap = cap_tokens * head_size;

        let heads = (0..num_kv_heads)
            .map(|_| HeadKvState {
                key_pages: Vec::new(),
                value_pages: Vec::new(),
                residual: ResidualStorage::with_capacity(config.residual_precision, elem_cap),
            })
            .collect();

        Ok(Self {
            num_kv_heads,
            head_size,
            value_group_size,
            config,
            heads,
        })
    }

    /// Construct a KIVI KV cache for `layer_kind`, explicitly rejecting recurrent GDN layers.
    pub fn new_for_layer(
        layer_kind: LayerStateKind,
        num_kv_heads: usize,
        head_size: usize,
        config: KiviConfig,
    ) -> Result<Self, KiviError> {
        if layer_kind == LayerStateKind::RecurrentGdn {
            return Err(KiviError::RecurrentLayerExempt);
        }
        Self::new(num_kv_heads, head_size, config)
    }

    /// Active KIVI configuration.
    #[inline]
    pub const fn config(&self) -> KiviConfig {
        self.config
    }

    /// Number of KV heads managed by this cache.
    #[inline]
    pub const fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Per-head dimension.
    #[inline]
    pub const fn head_size(&self) -> usize {
        self.head_size
    }

    /// Number of quantized pages per KV head.
    #[inline]
    pub fn num_pages(&self) -> usize {
        self.heads.first().map_or(0, |h| h.key_pages.len())
    }

    /// Number of tokens currently stored in quantized bit-packed pages.
    #[inline]
    pub fn quantized_len(&self) -> usize {
        self.num_pages() * self.config.key_group_size
    }

    /// Number of recent tokens currently held in the unquantized residual buffer.
    #[inline]
    pub fn residual_len(&self) -> usize {
        let d = self.head_size;
        self.heads.first().map_or(0, |h| h.residual.token_len(d))
    }

    /// Total cached sequence length (`quantized_len + residual_len`).
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.quantized_len() + self.residual_len()
    }

    /// Append one token across all KV heads (`key` and `value` each have length
    /// `num_kv_heads * head_size`). Atomically flushes one `G`-token page per head if the
    /// residual buffer exceeds `residual_length` with a complete group available.
    pub fn append_token(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KiviError> {
        let d = self.head_size;
        let stride = self.num_kv_heads * d;
        if key.len() != stride {
            return Err(KiviError::DimensionMismatch {
                expected: stride,
                actual: key.len(),
                context: "KiviKvCache::append_token(key)",
            });
        }
        if value.len() != stride {
            return Err(KiviError::DimensionMismatch {
                expected: stride,
                actual: value.len(),
                context: "KiviKvCache::append_token(value)",
            });
        }

        let bits = self.config.bits;
        let group_size = self.config.key_group_size;
        let residual_cap = self.config.residual_length;
        let v_group = self.value_group_size;
        let group_elems = group_size * d;
        let mut flushed_pages = 0usize;

        let mut k_buf = Vec::new();
        let mut v_buf = Vec::new();

        for (h_idx, head_state) in self.heads.iter_mut().enumerate() {
            let off = h_idx * d;
            head_state
                .residual
                .push_slice(&key[off..off + d], &value[off..off + d]);

            let mut rem_tokens = head_state.residual.token_len(d);
            while rem_tokens >= group_size && rem_tokens > residual_cap {
                if k_buf.is_empty() {
                    k_buf.resize(group_elems, 0.0);
                    v_buf.resize(group_elems, 0.0);
                }
                head_state
                    .residual
                    .drain_prefix_to_f32(group_elems, &mut k_buf, &mut v_buf);
                let k_page = KiviKeyPage::quantize(bits, &k_buf, group_size, d)?;
                let v_page = KiviValuePage::quantize(bits, &v_buf, group_size, d, v_group)?;
                head_state.key_pages.push(k_page);
                head_state.value_pages.push(v_page);
                if h_idx == 0 {
                    flushed_pages += 1;
                }
                rem_tokens = head_state.residual.token_len(d);
            }
        }
        Ok(flushed_pages)
    }

    /// Append a batch of `num_tokens` tokens (`keys` and `values` laid out token-major as
    /// `[num_tokens, num_kv_heads, head_size]`). Produces the exact same page and residual state
    /// as `num_tokens` sequential [`Self::append_token`] calls without growing the residual
    /// allocation beyond `residual_length + key_group_size`.
    pub fn append_batch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        num_tokens: usize,
    ) -> Result<usize, KiviError> {
        let d = self.head_size;
        let token_stride = self.num_kv_heads * d;
        let expected = num_tokens * token_stride;
        if keys.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: keys.len(),
                context: "KiviKvCache::append_batch(keys)",
            });
        }
        if values.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: values.len(),
                context: "KiviKvCache::append_batch(values)",
            });
        }
        if num_tokens == 0 {
            return Ok(0);
        }

        let bits = self.config.bits;
        let group_size = self.config.key_group_size;
        let residual_cap = self.config.residual_length;
        let v_group = self.value_group_size;
        let group_elems = group_size * d;
        let round_via_f16 = self.config.residual_precision == ResidualPrecision::Fp16;

        let prev_res = self.residual_len();
        let total_unquant = prev_res + num_tokens;
        let flushes = total_unquant
            .saturating_sub(residual_cap)
            .div_ceil(group_size)
            .min(total_unquant / group_size);

        let mut k_buf = vec![0.0f32; group_elems];
        let mut v_buf = vec![0.0f32; group_elems];

        for (h_idx, head_state) in self.heads.iter_mut().enumerate() {
            let head_offset = h_idx * d;
            let mut batch_cursor = 0usize;

            for _ in 0..flushes {
                let res_now = head_state.residual.token_len(d);
                let from_res = res_now.min(group_size);
                if from_res > 0 {
                    let elem_res = from_res * d;
                    head_state.residual.drain_prefix_to_f32(
                        elem_res,
                        &mut k_buf[..elem_res],
                        &mut v_buf[..elem_res],
                    );
                }
                let from_batch = group_size - from_res;
                for local_t in 0..from_batch {
                    let src_base = (batch_cursor + local_t) * token_stride + head_offset;
                    let dst_base = (from_res + local_t) * d;
                    for c in 0..d {
                        let kv = keys[src_base + c];
                        let vv = values[src_base + c];
                        if round_via_f16 {
                            k_buf[dst_base + c] = to_f16_clamped(kv).to_f32();
                            v_buf[dst_base + c] = to_f16_clamped(vv).to_f32();
                        } else {
                            k_buf[dst_base + c] = sanitize_f32(kv);
                            v_buf[dst_base + c] = sanitize_f32(vv);
                        }
                    }
                }
                batch_cursor += from_batch;

                let k_page = KiviKeyPage::quantize(bits, &k_buf, group_size, d)?;
                let v_page = KiviValuePage::quantize(bits, &v_buf, group_size, d, v_group)?;
                head_state.key_pages.push(k_page);
                head_state.value_pages.push(v_page);
            }

            while batch_cursor < num_tokens {
                let src = batch_cursor * token_stride + head_offset;
                head_state
                    .residual
                    .push_slice(&keys[src..src + d], &values[src..src + d]);
                batch_cursor += 1;
            }
        }

        Ok(flushes)
    }

    /// Compute the exact resident byte breakdown of this KV cache.
    pub fn memory_footprint(&self) -> KiviMemoryFootprint {
        let seq_len = self.seq_len();
        let quantized_tokens = self.quantized_len();
        let residual_tokens = self.residual_len();
        let num_pages_per_head = self.num_pages();

        let mut quantized_key_bytes = 0usize;
        let mut quantized_value_bytes = 0usize;
        let mut residual_bytes = 0usize;

        for head in &self.heads {
            for kp in &head.key_pages {
                quantized_key_bytes += kp.byte_footprint();
            }
            for vp in &head.value_pages {
                quantized_value_bytes += vp.byte_footprint();
            }
            residual_bytes += head.residual.byte_footprint();
        }

        let total_bytes = quantized_key_bytes + quantized_value_bytes + residual_bytes;
        let total_elements = 2 * self.num_kv_heads * seq_len * self.head_size;

        KiviMemoryFootprint {
            seq_len,
            quantized_tokens,
            residual_tokens,
            num_pages_per_head,
            quantized_key_bytes,
            quantized_value_bytes,
            residual_bytes,
            total_bytes,
            fp16_baseline_bytes: total_elements * 2,
            fp32_baseline_bytes: total_elements * 4,
        }
    }

    /// Execute online page-streaming fused attention for a single query vector against `kv_head`
    /// without allocating `O(seq_len)` logit buffers or materializing dequantized `K`/`V` pages.
    pub fn fused_decode_head(
        &self,
        kv_head: usize,
        query: &[f32],
        softmax_scale: f32,
        out: &mut [f32],
    ) -> Result<(), KiviError> {
        let head = self
            .heads
            .get(kv_head)
            .ok_or(KiviError::DimensionMismatch {
                expected: self.num_kv_heads,
                actual: kv_head,
                context: "KiviKvCache::fused_decode_head(kv_head)",
            })?;
        let d = self.head_size;
        if query.len() != d {
            return Err(KiviError::DimensionMismatch {
                expected: d,
                actual: query.len(),
                context: "KiviKvCache::fused_decode_head(query)",
            });
        }
        if out.len() != d {
            return Err(KiviError::DimensionMismatch {
                expected: d,
                actual: out.len(),
                context: "KiviKvCache::fused_decode_head(out)",
            });
        }

        out.fill(0.0);
        if self.seq_len() == 0 {
            return Ok(());
        }

        let group_size = self.config.key_group_size;
        let mut page_scores = [0.0f32; MAX_GROUP_SIZE];
        let mut page_weights = [0.0f32; MAX_GROUP_SIZE];
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;

        for (k_page, v_page) in head.key_pages.iter().zip(&head.value_pages) {
            let scores = &mut page_scores[..group_size];
            let weights = &mut page_weights[..group_size];
            k_page.score_query(query, softmax_scale, scores)?;

            let page_max = scores.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
            let next_max = running_max.max(page_max);
            if running_sum > 0.0 && next_max > running_max {
                let rescale = (running_max - next_max).exp();
                running_sum *= rescale;
                for slot in out.iter_mut() {
                    *slot *= rescale;
                }
            }
            running_max = next_max;

            let mut page_sum = 0.0f32;
            for (&s, w_slot) in scores.iter().zip(weights.iter_mut()) {
                let w = (s - running_max).exp();
                *w_slot = w;
                page_sum += w;
            }
            running_sum += page_sum;
            v_page.accumulate_weighted(weights, out)?;
        }

        match &head.residual {
            ResidualStorage::Fp16 { keys, values } => {
                for (k_tok, v_tok) in keys.chunks_exact(d).zip(values.chunks_exact(d)) {
                    let mut dot = 0.0f32;
                    for (&q_v, &k_v) in query.iter().zip(k_tok) {
                        dot += sanitize_f32(q_v) * k_v.to_f32();
                    }
                    let logit = dot * softmax_scale;
                    let next_max = running_max.max(logit);
                    if running_sum > 0.0 && next_max > running_max {
                        let rescale = (running_max - next_max).exp();
                        running_sum *= rescale;
                        for slot in out.iter_mut() {
                            *slot *= rescale;
                        }
                    }
                    running_max = next_max;
                    let w = (logit - running_max).exp();
                    running_sum += w;
                    for (slot, &v_val) in out.iter_mut().zip(v_tok) {
                        *slot += w * v_val.to_f32();
                    }
                }
            }
            ResidualStorage::Fp32 { keys, values } => {
                for (k_tok, v_tok) in keys.chunks_exact(d).zip(values.chunks_exact(d)) {
                    let mut dot = 0.0f32;
                    for (&q_v, &k_v) in query.iter().zip(k_tok) {
                        dot += sanitize_f32(q_v) * k_v;
                    }
                    let logit = dot * softmax_scale;
                    let next_max = running_max.max(logit);
                    if running_sum > 0.0 && next_max > running_max {
                        let rescale = (running_max - next_max).exp();
                        running_sum *= rescale;
                        for slot in out.iter_mut() {
                            *slot *= rescale;
                        }
                    }
                    running_max = next_max;
                    let w = (logit - running_max).exp();
                    running_sum += w;
                    for (slot, &v_val) in out.iter_mut().zip(v_tok) {
                        *slot += w * v_val;
                    }
                }
            }
        }

        if running_sum > 0.0 {
            let inv_sum = 1.0 / running_sum;
            for slot in out.iter_mut() {
                *slot *= inv_sum;
            }
        }
        Ok(())
    }

    /// Execute fused Grouped-Query Attention (GQA) decode across all `num_q_heads` query heads.
    /// `queries` and `out` have shape `[num_q_heads, head_size]`.
    pub fn fused_gqa_decode(
        &self,
        queries: &[f32],
        num_q_heads: usize,
        softmax_scale: f32,
        out: &mut [f32],
    ) -> Result<(), KiviError> {
        if num_q_heads == 0 || !num_q_heads.is_multiple_of(self.num_kv_heads) {
            return Err(KiviError::InvalidHeadTopology {
                num_q_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }
        let d = self.head_size;
        let expected = num_q_heads * d;
        if queries.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: queries.len(),
                context: "KiviKvCache::fused_gqa_decode(queries)",
            });
        }
        if out.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: out.len(),
                context: "KiviKvCache::fused_gqa_decode(out)",
            });
        }

        let q_per_kv = num_q_heads / self.num_kv_heads;
        for (q_idx, (q_slice, out_slice)) in queries
            .chunks_exact(d)
            .zip(out.chunks_exact_mut(d))
            .enumerate()
        {
            let kv_head = q_idx / q_per_kv;
            self.fused_decode_head(kv_head, q_slice, softmax_scale, out_slice)?;
        }
        Ok(())
    }

    /// Two-pass materialized reference attention over dequantized `K_hat` and `V_hat` tensors
    /// (intended solely for unit-test equivalence validation of [`Self::fused_decode_head`]).
    pub fn materialized_decode_head_reference(
        &self,
        kv_head: usize,
        query: &[f32],
        softmax_scale: f32,
        out: &mut [f32],
    ) -> Result<(), KiviError> {
        let head = self
            .heads
            .get(kv_head)
            .ok_or(KiviError::DimensionMismatch {
                expected: self.num_kv_heads,
                actual: kv_head,
                context: "KiviKvCache::materialized_decode_head_reference(kv_head)",
            })?;
        let d = self.head_size;
        if query.len() != d || out.len() != d {
            return Err(KiviError::DimensionMismatch {
                expected: d,
                actual: query.len().min(out.len()),
                context: "KiviKvCache::materialized_decode_head_reference",
            });
        }

        out.fill(0.0);
        let total_len = self.seq_len();
        if total_len == 0 {
            return Ok(());
        }

        let group_elems = self.config.key_group_size * d;
        let mut full_k = Vec::with_capacity(total_len * d);
        let mut full_v = Vec::with_capacity(total_len * d);
        let mut page_buf = vec![0.0f32; group_elems];

        for kp in &head.key_pages {
            kp.dequantize(&mut page_buf)?;
            full_k.extend_from_slice(&page_buf);
        }
        for vp in &head.value_pages {
            vp.dequantize(&mut page_buf)?;
            full_v.extend_from_slice(&page_buf);
        }
        match &head.residual {
            ResidualStorage::Fp16 { keys, values } => {
                full_k.extend(keys.iter().copied().map(f16::to_f32));
                full_v.extend(values.iter().copied().map(f16::to_f32));
            }
            ResidualStorage::Fp32 { keys, values } => {
                full_k.extend_from_slice(keys);
                full_v.extend_from_slice(values);
            }
        }

        let mut logits = Vec::with_capacity(total_len);
        let mut max_logit = f32::NEG_INFINITY;
        for k_row in full_k.chunks_exact(d) {
            let dot: f32 = query
                .iter()
                .zip(k_row)
                .map(|(&q, &k)| sanitize_f32(q) * k)
                .sum();
            let s = dot * softmax_scale;
            max_logit = max_logit.max(s);
            logits.push(s);
        }

        let mut denom = 0.0f32;
        for s in &mut logits {
            *s = (*s - max_logit).exp();
            denom += *s;
        }
        let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        for (&w, v_row) in logits.iter().zip(full_v.chunks_exact(d)) {
            let prob = w * inv_denom;
            for (dst, &v) in out.iter_mut().zip(v_row) {
                *dst += prob * v;
            }
        }
        Ok(())
    }
}

/// Uniform symmetric per-token KV cache baseline (`Int4` on `-7..=7`, `Int2` mid-riser on
/// `{-3, -1, +1, +3}`) used in controlled evaluations to demonstrate why channel-outlier keys
/// require KIVI's asymmetric per-channel Key quantization.
#[derive(Clone, Debug)]
pub struct UniformSymmetricKvCache {
    bits: KiviBits,
    num_kv_heads: usize,
    head_size: usize,
    key_scales: Vec<Vec<f32>>,
    key_codes: Vec<Vec<u32>>,
    val_scales: Vec<Vec<f32>>,
    val_codes: Vec<Vec<u32>>,
}

impl UniformSymmetricKvCache {
    /// Create an empty uniform symmetric per-token baseline cache.
    pub fn new(bits: KiviBits, num_kv_heads: usize, head_size: usize) -> Result<Self, KiviError> {
        if num_kv_heads == 0 {
            return Err(KiviError::InvalidHeadTopology {
                num_q_heads: 0,
                num_kv_heads: 0,
            });
        }
        validate_head_size(head_size)?;
        Ok(Self {
            bits,
            num_kv_heads,
            head_size,
            key_scales: vec![Vec::new(); num_kv_heads],
            key_codes: vec![Vec::new(); num_kv_heads],
            val_scales: vec![Vec::new(); num_kv_heads],
            val_codes: vec![Vec::new(); num_kv_heads],
        })
    }

    /// Append a batch of `[num_tokens, num_kv_heads, head_size]` tokens using symmetric per-token
    /// quantization on both keys and values.
    pub fn append_batch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        num_tokens: usize,
    ) -> Result<(), KiviError> {
        let d = self.head_size;
        let stride = self.num_kv_heads * d;
        let expected = num_tokens * stride;
        if keys.len() != expected || values.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: keys.len().min(values.len()),
                context: "UniformSymmetricKvCache::append_batch",
            });
        }

        let wpt = self.bits.packed_words(d);
        let mut word_buf = vec![0u32; wpt];

        for t in 0..num_tokens {
            let base = t * stride;
            for h in 0..self.num_kv_heads {
                let h_off = base + h * d;
                let k_slice = &keys[h_off..h_off + d];
                let k_scale = quantize_symmetric_token(self.bits, k_slice, &mut word_buf);
                self.key_scales[h].push(k_scale);
                self.key_codes[h].extend_from_slice(&word_buf);

                let v_slice = &values[h_off..h_off + d];
                let v_scale = quantize_symmetric_token(self.bits, v_slice, &mut word_buf);
                self.val_scales[h].push(v_scale);
                self.val_codes[h].extend_from_slice(&word_buf);
            }
        }
        Ok(())
    }

    /// Resident byte footprint of the uniform symmetric cache.
    pub fn byte_footprint(&self) -> usize {
        let mut words = 0usize;
        for h in 0..self.num_kv_heads {
            words += self.key_scales[h].len() + self.key_codes[h].len();
            words += self.val_scales[h].len() + self.val_codes[h].len();
        }
        words * 4
    }

    /// Decode GQA attention across `num_q_heads` query heads.
    pub fn decode_gqa(
        &self,
        queries: &[f32],
        num_q_heads: usize,
        softmax_scale: f32,
        out: &mut [f32],
    ) -> Result<(), KiviError> {
        if num_q_heads == 0 || !num_q_heads.is_multiple_of(self.num_kv_heads) {
            return Err(KiviError::InvalidHeadTopology {
                num_q_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }
        let d = self.head_size;
        let expected = num_q_heads * d;
        if queries.len() != expected || out.len() != expected {
            return Err(KiviError::DimensionMismatch {
                expected,
                actual: queries.len().min(out.len()),
                context: "UniformSymmetricKvCache::decode_gqa",
            });
        }

        let wpt = self.bits.packed_words(d);
        let q_per_kv = num_q_heads / self.num_kv_heads;
        let mut row_buf = [0.0f32; MAX_HEAD_DIM];

        for (q_idx, (q_slice, out_slice)) in queries
            .chunks_exact(d)
            .zip(out.chunks_exact_mut(d))
            .enumerate()
        {
            out_slice.fill(0.0);
            let h = q_idx / q_per_kv;
            if self.key_scales[h].is_empty() {
                continue;
            }

            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let row = &mut row_buf[..d];

            for (((&k_scale, k_words), &v_scale), v_words) in self.key_scales[h]
                .iter()
                .zip(self.key_codes[h].chunks_exact(wpt))
                .zip(&self.val_scales[h])
                .zip(self.val_codes[h].chunks_exact(wpt))
            {
                dequantize_symmetric_token(self.bits, k_words, k_scale, row);
                let dot: f32 = q_slice
                    .iter()
                    .zip(&*row)
                    .map(|(&q, &k)| sanitize_f32(q) * k)
                    .sum();
                let logit = dot * softmax_scale;
                let next_max = running_max.max(logit);
                if running_sum > 0.0 && next_max > running_max {
                    let rescale = (running_max - next_max).exp();
                    running_sum *= rescale;
                    for slot in out_slice.iter_mut() {
                        *slot *= rescale;
                    }
                }
                running_max = next_max;
                let w = (logit - running_max).exp();
                running_sum += w;

                dequantize_symmetric_token(self.bits, v_words, v_scale, row);
                for (dst, &v) in out_slice.iter_mut().zip(&*row) {
                    *dst += w * v;
                }
            }

            if running_sum > 0.0 {
                let inv = 1.0 / running_sum;
                for slot in out_slice.iter_mut() {
                    *slot *= inv;
                }
            }
        }
        Ok(())
    }
}

/// Quantize a single vector symmetrically per token (`Int4` codes in `-7..=7` biased by `+7`;
/// `Int2` 4-level mid-riser codes `{-3, -1, +1, +3}` mapped to `0..=3`).
pub fn quantize_symmetric_token(bits: KiviBits, row: &[f32], out_words: &mut [u32]) -> f32 {
    let amax = row
        .iter()
        .fold(0.0f32, |m, &v| m.max(sanitize_f32(v).abs()));
    match bits {
        KiviBits::Int4 => {
            let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            let inv = if amax > 0.0 { 7.0 / amax } else { 0.0 };
            for (chunk, dst) in row.chunks_exact(8).zip(out_words.iter_mut()) {
                let mut word = 0u32;
                for (lane, &raw) in chunk.iter().enumerate() {
                    let q_signed = (sanitize_f32(raw) * inv).round().clamp(-7.0, 7.0) as i32;
                    let biased = (q_signed + 7) as u32;
                    word |= (biased & 0x0F) << (lane * 4);
                }
                *dst = word;
            }
            scale
        }
        KiviBits::Int2 => {
            // 4-level symmetric mid-riser grid {-3, -1, +1, +3} * scale with scale = amax / 3.
            let scale = if amax > 0.0 { amax / 3.0 } else { 0.0 };
            let inv = if amax > 0.0 { 3.0 / amax } else { 0.0 };
            for (chunk, dst) in row.chunks_exact(16).zip(out_words.iter_mut()) {
                let mut word = 0u32;
                for (lane, &raw) in chunk.iter().enumerate() {
                    let scaled = sanitize_f32(raw) * inv;
                    let code = if scaled < -2.0 {
                        0u32
                    } else if scaled < 0.0 {
                        1u32
                    } else if scaled < 2.0 {
                        2u32
                    } else {
                        3u32
                    };
                    word |= code << (lane * 2);
                }
                *dst = word;
            }
            scale
        }
    }
}

/// Dequantize a single token vector written by [`quantize_symmetric_token`].
pub fn dequantize_symmetric_token(bits: KiviBits, words: &[u32], scale: f32, out: &mut [f32]) {
    match bits {
        KiviBits::Int4 => {
            for (&word, dst) in words.iter().zip(out.chunks_exact_mut(8)) {
                for (lane, slot) in dst.iter_mut().enumerate() {
                    let biased = ((word >> (lane * 4)) & 0x0F) as i32;
                    *slot = ((biased - 7) as f32) * scale;
                }
            }
        }
        KiviBits::Int2 => {
            for (&word, dst) in words.iter().zip(out.chunks_exact_mut(16)) {
                for (lane, slot) in dst.iter_mut().enumerate() {
                    let code = ((word >> (lane * 2)) & 0x03) as i32;
                    let level = 2 * code - 3; // {-3, -1, +1, +3}
                    *slot = (level as f32) * scale;
                }
            }
        }
    }
}

/// Architecture parameters for hybrid attention + recurrent GDN memory footprint projection.
#[derive(Clone, Copy, Debug)]
pub struct HybridModelSpec {
    /// Model family label (e.g. `"4B"`, `"9B"`, `"27B"`).
    pub label: &'static str,
    /// Number of paged-attention decoder layers.
    pub num_attention_layers: usize,
    /// Number of recurrent Gated DeltaNet (GDN) layers (always held in full precision).
    pub num_gdn_layers: usize,
    /// Number of KV heads per attention layer.
    pub num_kv_heads: usize,
    /// Per-head dimension on attention layers.
    pub head_size: usize,
    /// Number of recurrent GDN heads per GDN layer.
    pub gdn_heads: usize,
    /// Recurrent state matrix size `state_dim * head_dim` per GDN head (in `FP16`/`FP32` scalars).
    pub gdn_state_elems_per_head: usize,
}

/// Projected memory summary for a hybrid attention + recurrent GDN model at `seq_len`.
#[derive(Clone, Copy, Debug)]
pub struct HybridMemoryProjection {
    /// Uncompressed `FP16` attention KV bytes + `FP32` GDN state bytes.
    pub baseline_total_bytes: usize,
    /// KIVI-compressed attention KV bytes + unquantized `FP32` GDN state bytes.
    pub compressed_total_bytes: usize,
    /// Unquantized recurrent GDN state bytes (identical before and after KIVI).
    pub unquantized_gdn_bytes: usize,
    /// Attention-only compression multiple vs `FP16`.
    pub attention_kv_ratio: f64,
    /// End-to-end state + KV compression multiple.
    pub total_state_ratio: f64,
}

/// Compute exact hybrid model state footprint at `seq_len`, keeping all recurrent GDN layers in
/// full precision (`4` bytes/scalar `FP32` recurrent state) while compressing attention KV layers
/// according to `config`.
pub fn project_hybrid_model_memory(
    spec: &HybridModelSpec,
    seq_len: usize,
    config: KiviConfig,
) -> Result<HybridMemoryProjection, KiviError> {
    let d = spec.head_size;
    let g = config.key_group_size;
    let v_group = config.validate(d)?;
    let flushes = seq_len
        .saturating_sub(config.residual_length)
        .div_ceil(g)
        .min(seq_len / g);
    let quant_tokens = flushes * g;
    let res_tokens = seq_len - quant_tokens;

    let code_words = config.bits.packed_words(g * d);
    let key_page_words = code_words + d;
    let val_subgroups = d / v_group;
    let val_page_words = code_words + g * val_subgroups;
    let bytes_per_head_quant = flushes * (key_page_words + val_page_words) * 4;
    let res_elem_bytes = config.residual_precision.bytes_per_element();
    let bytes_per_head_res = 2 * res_tokens * d * res_elem_bytes;

    let kv_heads_total = spec.num_attention_layers * spec.num_kv_heads;
    let attn_compressed_bytes = kv_heads_total * (bytes_per_head_quant + bytes_per_head_res);
    let attn_fp16_bytes = kv_heads_total * 2 * seq_len * d * 2;
    let unquantized_gdn_bytes =
        spec.num_gdn_layers * spec.gdn_heads * spec.gdn_state_elems_per_head * 4;

    let baseline_total_bytes = attn_fp16_bytes + unquantized_gdn_bytes;
    let compressed_total_bytes = attn_compressed_bytes + unquantized_gdn_bytes;

    let attention_kv_ratio = if attn_compressed_bytes > 0 {
        attn_fp16_bytes as f64 / attn_compressed_bytes as f64
    } else {
        1.0
    };
    let total_state_ratio = if compressed_total_bytes > 0 {
        baseline_total_bytes as f64 / compressed_total_bytes as f64
    } else {
        1.0
    };

    Ok(HybridMemoryProjection {
        baseline_total_bytes,
        compressed_total_bytes,
        unquantized_gdn_bytes,
        attention_kv_ratio,
        total_state_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*state >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }

    #[test]
    fn affine_roundtrip_respects_tight_int4_and_int2_bounds() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for &bits in &[KiviBits::Int4, KiviBits::Int2] {
            let bound = bits.span_error_bound() + 1e-6;
            for &head_size in &[64usize, 80, 96, 112, 128, 192, 256, 512] {
                for _ in 0..128 {
                    let mag = 10f32.powf(lcg(&mut st) * 6.0);
                    let shift = lcg(&mut st) * mag * 3.0;
                    let mut row = Vec::with_capacity(head_size);
                    for _ in 0..head_size {
                        row.push(shift + lcg(&mut st) * mag);
                    }
                    let err = span_relative_error_f32(bits, &row);
                    assert!(
                        err <= bound,
                        "bits={bits:?} head_size={head_size} err={err} > bound={bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn constant_nonzero_channel_is_exact() -> Result<(), KiviError> {
        let group_size = 32;
        let head_size = 64;
        let mut keys = vec![0.0f32; group_size * head_size];
        for row in keys.chunks_exact_mut(head_size) {
            row[7] = 142.5;
            row[31] = -88.25;
        }
        let page = KiviKeyPage::quantize(KiviBits::Int2, &keys, group_size, head_size)?;
        let mut back = vec![0.0f32; group_size * head_size];
        page.dequantize(&mut back)?;
        for row in back.chunks_exact(head_size) {
            assert!((row[7] - 142.5).abs() < 1e-3);
            assert!((row[31] - (-88.25)).abs() < 1e-3);
            assert_eq!(row[0], 0.0);
        }
        Ok(())
    }

    #[test]
    fn online_fused_attention_matches_materialized_reference() -> Result<(), KiviError> {
        let mut st = 0xfeed_face_cafe_babeu64;
        let num_kv_heads = 2;
        let head_size = 128;
        let seq_len = 173; // 2 pages of 64 + 45 residual tokens

        for &bits in &[KiviBits::Int4, KiviBits::Int2] {
            let cfg = KiviConfig::new(bits, 64, 64);
            let mut cache = KiviKvCache::new(num_kv_heads, head_size, cfg)?;
            let total = seq_len * num_kv_heads * head_size;
            let mut keys = Vec::with_capacity(total);
            for idx in 0..total {
                let outlier = if idx % 128 == 11 { 35.0 } else { 0.0 };
                keys.push(lcg(&mut st) + outlier);
            }
            let vals: Vec<f32> = (0..total).map(|_| lcg(&mut st) * 2.0).collect();
            cache.append_batch(&keys, &vals, seq_len)?;
            assert_eq!(cache.num_pages(), 2);
            assert_eq!(cache.residual_len(), 45);

            let query: Vec<f32> = (0..head_size).map(|_| lcg(&mut st)).collect();
            let scale = 1.0 / (head_size as f32).sqrt();
            let mut out_fused = vec![0.0f32; head_size];
            let mut out_ref = vec![0.0f32; head_size];
            cache.fused_decode_head(1, &query, scale, &mut out_fused)?;
            cache.materialized_decode_head_reference(1, &query, scale, &mut out_ref)?;

            for (&a, &b) in out_fused.iter().zip(&out_ref) {
                assert!((a - b).abs() < 1e-5, "bits={bits:?} fused={a} vs ref={b}");
            }
        }
        Ok(())
    }

    #[test]
    fn append_batch_matches_token_by_token_streaming_exactly() -> Result<(), KiviError> {
        let mut st = 0x5151_7373_9595_b7b7u64;
        let num_kv_heads = 2;
        let head_size = 64;
        let seq_len = 139;
        let cfg = KiviConfig::new(KiviBits::Int4, 32, 64);

        let mut streaming = KiviKvCache::new(num_kv_heads, head_size, cfg)?;
        let mut batched = KiviKvCache::new(num_kv_heads, head_size, cfg)?;

        let stride = num_kv_heads * head_size;
        let keys: Vec<f32> = (0..seq_len * stride).map(|_| lcg(&mut st) * 12.0).collect();
        let vals: Vec<f32> = (0..seq_len * stride).map(|_| lcg(&mut st) * 3.0).collect();

        for t in 0..seq_len {
            let off = t * stride;
            streaming.append_token(&keys[off..off + stride], &vals[off..off + stride])?;
        }
        batched.append_batch(&keys, &vals, seq_len)?;

        assert_eq!(streaming.num_pages(), batched.num_pages());
        assert_eq!(streaming.residual_len(), batched.residual_len());

        let query: Vec<f32> = (0..head_size).map(|_| lcg(&mut st)).collect();
        let mut out_a = vec![0.0f32; head_size];
        let mut out_b = vec![0.0f32; head_size];
        streaming.fused_decode_head(0, &query, 0.125, &mut out_a)?;
        batched.fused_decode_head(0, &query, 0.125, &mut out_b)?;
        assert_eq!(out_a, out_b);
        Ok(())
    }

    #[test]
    fn recurrent_gdn_layers_are_strictly_exempt() {
        let cfg = KiviConfig::int2_default();
        assert_eq!(
            resolve_layer_plan(LayerStateKind::RecurrentGdn, cfg),
            LayerCompressionPlan::UnquantizedRecurrentGdn
        );
        let res = KiviKvCache::new_for_layer(LayerStateKind::RecurrentGdn, 4, 128, cfg);
        assert_eq!(res.unwrap_err(), KiviError::RecurrentLayerExempt);
    }
}
