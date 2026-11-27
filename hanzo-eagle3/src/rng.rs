//! Deterministic PRNG (SplitMix64), clean-room FIPS 180-4 SHA-256, and numerical helpers.

/// Deterministic SplitMix64 PRNG for reproducible training, sampling, and benchmarks.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform `f32` in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Standard normal `N(0, 1)` via Box-Muller transform.
    pub fn normal_f32(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos()) as f32
    }

    /// Sample an index from a normalized probability distribution `probs`.
    pub fn sample_categorical(&mut self, probs: &[f32]) -> usize {
        if probs.is_empty() {
            return 0;
        }
        let u = self.next_f64();
        let mut acc = 0.0_f64;
        for (i, &p) in probs.iter().enumerate() {
            acc += (p.max(0.0)) as f64;
            if u < acc {
                return i;
            }
        }
        // Fallback to largest probability token if floating-point rounding fell short.
        argmax_f32(probs)
    }
}

/// Deterministic argmax with lowest-index tie-breaking.
#[inline]
pub fn argmax_f32(xs: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Numerically stable softmax with temperature `temp` (`temp > 0`).
pub fn softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let inv_t = 1.0 / temp.max(1e-6);
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
    let mut out = Vec::with_capacity(logits.len());
    let mut sum = 0.0_f64;
    for &x in logits {
        let e = ((x * inv_t - max_v) as f64).exp();
        out.push(e as f32);
        sum += e;
    }
    let inv_sum = if sum > 0.0 {
        1.0 / sum
    } else {
        1.0 / (logits.len().max(1) as f64)
    };
    for p in &mut out {
        *p = ((*p as f64) * inv_sum) as f32;
    }
    out
}

/// Shannon entropy in nats of a probability distribution.
pub fn shannon_entropy(probs: &[f32]) -> f64 {
    let mut h = 0.0_f64;
    for &p in probs {
        let pf = p as f64;
        if pf > 1e-12 {
            h -= pf * pf.ln();
        }
    }
    h
}

/// Total Variation distance `0.5 * sum_i |p[i] - q[i]|`.
pub fn total_variation_distance(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut s = 0.0_f64;
    for (&a, &b) in p.iter().zip(q.iter()) {
        s += (a - b).abs();
    }
    0.5 * s
}

/// SiLU activation `x / (1 + exp(-x))`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMSNorm `out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0_f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / ((sum_sq / (n as f32)) + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/// Compute percentile `p` in `[0.0, 100.0]` from a slice of values.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * ((sorted.len() - 1) as f64)).clamp(0.0, (sorted.len() - 1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Clean-room FIPS 180-4 SHA-256 returning 32 raw digest bytes.
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Clean-room SHA-256 returning lowercase 64-char hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let bytes = sha256_bytes(data);
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}
