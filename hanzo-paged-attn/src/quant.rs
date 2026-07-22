//! Symmetric per-token int8 quantization for the paged KV cache (KIVI / KVQuant style).
//!
//! Each cached token's per-head vector `x[0..head_size]` is quantized independently to
//! signed int8 with a single f32 scale (symmetric, zero-point free):
//!
//! ```text
//!   scale = amax / 127          (amax = max_i |x_i|)
//!   q_i   = clamp(round(x_i / scale), -127, 127)
//!   x̂_i   = q_i * scale
//! ```
//!
//! An all-zero vector takes `scale = 1` and all-zero codes so it dequantizes to exactly 0
//! (avoids 0/0). The range is `±127` (not `-128`) so the grid is symmetric, matching the
//! Q8_0 weight convention the dp4a matmul path already uses.
//!
//! Grouping is **per token** (one scale per per-head vector). This is the accuracy-critical
//! choice for the value cache in KIVI (arXiv:2402.02750) and is ~lossless at 8 bits for the
//! key cache as well: the 256-level grid absorbs a token's channel outliers within its own
//! dynamic range, so the per-channel key refinement KIVI needs at 2-4 bits is unnecessary at
//! 8 bits (this matches the ~lossless int8 KV reported by KVQuant, arXiv:2401.18079). The
//! per-channel key path is reserved for int4.
//!
//! Codes are packed 4-per-u32, little-endian by lane (byte `l` holds lane `l`) — the SAME
//! convention as `quantize_act_q8`, so packing is shared across the engine. The scale rides
//! in one trailing u32 (its f32 bit pattern), co-located after the code words, so a quantized
//! token is self-describing and paged/addressable by slot with no side tensor.

/// Symmetric int8 code range. 127, not 128, keeps `-QMAX..=QMAX` symmetric.
pub const QMAX: f32 = 127.0;

/// u32 words to hold `head_size` int8 codes (4 per word) plus the trailing scale word.
/// `head_size` must be a multiple of 4 (every supported head dim — 64,80,96,112,128,192,
/// 256,512 — is).
#[inline]
pub const fn words_per_token(head_size: usize) -> usize {
    head_size / 4 + 1
}

#[inline]
fn amax(row: &[f32]) -> f32 {
    row.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

/// Quantize one per-token vector into `out`, which must be exactly
/// `words_per_token(row.len())` u32 words: the packed int8 codes followed by the scale's
/// f32 bit pattern. Returns the scale.
#[inline]
pub fn quantize_token(row: &[f32], out: &mut [u32]) -> f32 {
    let n = row.len();
    debug_assert_eq!(n % 4, 0, "head_size must be a multiple of 4");
    debug_assert_eq!(out.len(), words_per_token(n));

    let a = amax(row);
    let scale = if a > 0.0 { a / QMAX } else { 1.0 };
    let inv = if a > 0.0 { QMAX / a } else { 0.0 };

    let nwords = n / 4;
    for w in 0..nwords {
        let mut word = 0u32;
        for l in 0..4 {
            let q = (row[w * 4 + l] * inv).round().clamp(-QMAX, QMAX) as i32;
            word |= ((q as u32) & 0xFF) << (l * 8);
        }
        out[w] = word;
    }
    out[nwords] = scale.to_bits();
    scale
}

/// Dequantize one per-token vector written by [`quantize_token`] back to f32 into `out`,
/// which must be exactly `head_size` long. `words` must be `words_per_token(head_size)`.
#[inline]
pub fn dequantize_token(words: &[u32], head_size: usize, out: &mut [f32]) {
    debug_assert_eq!(head_size % 4, 0);
    debug_assert_eq!(words.len(), words_per_token(head_size));
    debug_assert_eq!(out.len(), head_size);

    let nwords = head_size / 4;
    let scale = f32::from_bits(words[nwords]);
    for w in 0..nwords {
        let word = words[w];
        for l in 0..4 {
            // Extract byte l, reinterpret as signed int8 (sign-extend), dequantize.
            let code = ((word >> (l * 8)) & 0xFF) as u8 as i8;
            out[w * 4 + l] = code as f32 * scale;
        }
    }
}

/// Scale-relative reconstruction error `max_i |x_i - x̂_i| / max_i |x_i|` for one per-token
/// vector under the round-trip above. Scale-relative (not per-element relative) is the
/// correct gate for symmetric/affine quantization: a per-element relative error explodes on
/// near-zero cancellation and produces false failures. The value is bounded above by
/// `1/254 ≈ 0.00394` (the symmetric int8 round-off `scale/2` divided by `amax`), independent
/// of the vector's content.
pub fn scale_relative_error(row: &[f32]) -> f32 {
    let n = row.len();
    let mut packed = vec![0u32; words_per_token(n)];
    quantize_token(row, &mut packed);
    let mut back = vec![0.0f32; n];
    dequantize_token(&packed, n, &mut back);

    let denom = amax(row);
    if denom == 0.0 {
        // All-zero input dequantizes to exactly zero.
        return row
            .iter()
            .zip(&back)
            .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
    }
    let max_abs_err = row
        .iter()
        .zip(&back)
        .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_abs_err / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Theoretical worst-case scale-relative error for symmetric int8: the round-off is at
    /// most `scale/2 = amax/254`, so the error over `amax` is at most `1/254`. A tiny f32
    /// epsilon is allowed on top for the `amax/127` scale not being exactly representable.
    const BOUND: f32 = 1.0 / 254.0 + 1e-6;

    fn lcg(state: &mut u64) -> f32 {
        // Deterministic pseudo-random in [-1, 1); no external rng dependency.
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*state >> 40) as u32; // 24 bits
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }

    #[test]
    fn roundtrip_is_within_symmetric_int8_bound() {
        let mut st = 0x1234_5678_9abc_def0u64;
        for &head_size in &[64usize, 80, 96, 112, 128, 192, 256, 512] {
            for _ in 0..256 {
                // Random vector scaled by a random magnitude so amax varies widely.
                let mag = 10f32.powf(lcg(&mut st) * 6.0); // 1e-6 .. 1e6
                let row: Vec<f32> = (0..head_size).map(|_| lcg(&mut st) * mag).collect();
                let err = scale_relative_error(&row);
                assert!(
                    err <= BOUND,
                    "head_size={head_size} err={err} exceeds symmetric int8 bound {BOUND}"
                );
            }
        }
    }

    #[test]
    fn channel_outlier_stays_bounded() {
        // KIVI's motivating case: keys have a few large-magnitude channels. Per-token int8
        // must still reconstruct within the symmetric bound (the outlier sets amax; the small
        // channels get coarse steps but the SCALE-relative error stays bounded — this is why
        // int8 needs no per-channel key treatment).
        let head_size = 128;
        let mut row = vec![0.0f32; head_size];
        for (i, v) in row.iter_mut().enumerate() {
            *v = 0.01 * ((i % 7) as f32 - 3.0); // small "normal" channels
        }
        row[42] = 250.0; // one dominant outlier channel
        row[99] = -180.0; // another
        let err = scale_relative_error(&row);
        assert!(
            err <= BOUND,
            "channel outlier err={err} exceeds bound {BOUND}"
        );
    }

    #[test]
    fn near_zero_cancellation_does_not_false_fail() {
        // Values straddling zero: per-ELEMENT relative error would explode here. The
        // scale-relative gate must stay bounded, proving we chose the right metric.
        let row: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1e-7 } else { -1e-7 })
            .collect();
        let err = scale_relative_error(&row);
        assert!(err <= BOUND, "near-zero err={err} exceeds bound {BOUND}");
    }

    #[test]
    fn all_zero_is_exact() {
        let row = vec![0.0f32; 128];
        let mut packed = vec![0u32; words_per_token(128)];
        let scale = quantize_token(&row, &mut packed);
        assert_eq!(scale, 1.0, "all-zero must take unit scale");
        let mut back = vec![9.0f32; 128];
        dequantize_token(&packed, 128, &mut back);
        assert!(
            back.iter().all(|&x| x == 0.0),
            "all-zero must dequantize to 0"
        );
    }

    #[test]
    fn amax_element_is_exact_endpoint() {
        // The max-magnitude element maps to code ±127 with no clamp saturation, so it
        // reconstructs to itself within one scale step. Verifies the endpoint math.
        let head_size = 64;
        let mut row = vec![0.5f32; head_size];
        row[10] = -4.0; // amax = 4.0 -> scale = 4/127
        let mut packed = vec![0u32; words_per_token(head_size)];
        let scale = quantize_token(&row, &mut packed);
        assert!((scale - 4.0 / 127.0).abs() < 1e-9);
        let mut back = vec![0.0f32; head_size];
        dequantize_token(&packed, head_size, &mut back);
        // -4.0 / scale = -127 exactly -> code -127 -> back = -127 * (4/127) = -4.0.
        assert!(
            (back[10] - (-4.0)).abs() < 1e-5,
            "endpoint {} != -4.0",
            back[10]
        );
    }

    #[test]
    fn packing_is_little_endian_by_lane() {
        // Byte l of each word must hold lane l (shared convention with quantize_act_q8).
        let row = vec![127.0, 63.5, -127.0, 0.0]; // amax=127 -> scale=1, codes 127,64,-127,0
        let mut packed = vec![0u32; words_per_token(4)];
        quantize_token(&row, &mut packed);
        let w = packed[0];
        assert_eq!((w & 0xFF) as u8 as i8, 127);
        assert_eq!(((w >> 8) & 0xFF) as u8 as i8, 64); // round(63.5)=64
        assert_eq!(((w >> 16) & 0xFF) as u8 as i8, -127);
        assert_eq!(((w >> 24) & 0xFF) as u8 as i8, 0);
    }
}
