//! DeltaQuant: INT2 / INT4 / INT8 grouped symmetric quantization of weight
//! deltas (`weight - base`).
//!
//! Symmetric per-group quant:
//!
//! ```text
//! qmax       = 2^(bits-1) - 1            // 1 for INT2, 7 for INT4, 127 for INT8
//! scale[g]   = max(|x[g]|) / qmax
//! q[i]       = round(x[i] / scale[g])    // in [-qmax, qmax]
//! ```
//!
//! Zero point is always 0 (true symmetric — `_simple` in the Python ref). Asym
//! per-channel is filed under future work; v1 matches what `deltasoup.py` uses.
//!
//! Packing:
//! - INT8: one byte per element, sign-preserving (`x as i8 as u8`).
//! - INT4: two values per byte. Low nibble = even index, high nibble = odd.
//!   Values are stored as i4 in two's complement (so -1 == 0b1111, 7 == 0b0111).
//! - INT2: four values per byte. Bits 0-1 = idx 0, 2-3 = idx 1, etc.
//!   Two's complement i2 (-2..=1 with the sign bit at position 1).
//!
//! Group_size default is 128 (matches DeltaQuant Python default); last group
//! may be shorter (no padding, unlike BitDelta).

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Supported bit widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bits {
    Int2,
    Int4,
    Int8,
}

impl Bits {
    pub fn n(&self) -> u8 {
        match self {
            Bits::Int2 => 2,
            Bits::Int4 => 4,
            Bits::Int8 => 8,
        }
    }
    pub fn qmax(&self) -> i32 {
        match self {
            Bits::Int2 => 1, // [-2..=1] -> use 1 as the divisor; mirrors INT4 logic
            Bits::Int4 => 7,
            Bits::Int8 => 127,
        }
    }
    pub fn qmin(&self) -> i32 {
        match self {
            Bits::Int2 => -2,
            Bits::Int4 => -8,
            Bits::Int8 => -128,
        }
    }
    pub fn try_from_u8(b: u8) -> Result<Self> {
        match b {
            2 => Ok(Bits::Int2),
            4 => Ok(Bits::Int4),
            8 => Ok(Bits::Int8),
            _ => Err(Error::InvalidBits(b)),
        }
    }
}

/// Grouped symmetric quantized delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaQuant {
    /// Bit width per element.
    pub bits: Bits,
    /// Group size used at encode time. Last group may be shorter.
    pub group_size: usize,
    /// One f32 scale per group: `scales[g] = max(|delta[g]|) / qmax`.
    pub scales: Vec<f32>,
    /// Zero point per group. Always 0 in this implementation (true symmetric);
    /// kept as a field so the wire format stays stable if we add asymmetric.
    pub zeros: Vec<i32>,
    /// Packed quantized values (see module docs).
    pub packed: Vec<u8>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Original number of elements (since packed may be padded).
    pub numel: usize,
}

impl DeltaQuant {
    /// Encode `delta = weight - base` to INT2/4/8 packed form. Default group
    /// size is 128 if `group_size = None`.
    pub fn encode(
        weight: &Tensor,
        base: &Tensor,
        bits: Bits,
        group_size: Option<usize>,
    ) -> Result<Self> {
        let ws = weight.dims().to_vec();
        let bs = base.dims().to_vec();
        if ws != bs {
            return Err(Error::ShapeMismatch { base: bs, weight: ws });
        }
        let delta = weight.sub(base)?;
        Self::encode_delta(&delta, bits, group_size)
    }

    /// Encode an already-computed delta tensor.
    pub fn encode_delta(delta: &Tensor, bits: Bits, group_size: Option<usize>) -> Result<Self> {
        let shape = delta.dims().to_vec();
        let flat: Vec<f32> = delta.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let numel = flat.len();
        if numel == 0 {
            return Err(Error::Empty("DeltaQuant: delta has 0 elements"));
        }

        let group_size = group_size.unwrap_or(128).max(1);
        let num_groups = numel.div_ceil(group_size);
        let qmax = bits.qmax() as f32;
        let qmin_i = bits.qmin();
        let qmax_i = bits.qmax();

        let mut scales = Vec::with_capacity(num_groups);
        let mut quantized = Vec::with_capacity(numel);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(numel);
            let group = &flat[start..end];

            // Symmetric per-group: scale = max(|x|)/qmax, clamped from 0.
            let abs_max = group.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = if abs_max < 1e-12 { 1e-8 } else { abs_max / qmax };
            scales.push(scale);

            for &x in group {
                let q = (x / scale).round() as i32;
                let q = q.clamp(qmin_i, qmax_i);
                quantized.push(q as i8);
            }
        }

        let packed = pack(bits, &quantized);
        let zeros = vec![0i32; num_groups];

        Ok(Self { bits, group_size, scales, zeros, packed, shape, numel })
    }

    /// Decode back to a full-precision delta tensor on `device`.
    pub fn decode(&self, device: &Device) -> Result<Tensor> {
        let unpacked = unpack(self.bits, &self.packed, self.numel);
        let mut out = Vec::with_capacity(self.numel);
        for i in 0..self.numel {
            let g = i / self.group_size;
            let scale = self.scales[g];
            let zero = self.zeros[g] as f32;
            out.push((unpacked[i] as f32 - zero) * scale);
        }
        let t = Tensor::from_vec(out, self.shape.as_slice(), device)?;
        Ok(t)
    }

    /// Reconstruct the fine-tuned weight: `base + decode()`.
    pub fn apply(&self, base: &Tensor) -> Result<Tensor> {
        let delta = self.decode(base.device())?;
        Ok(base.add(&delta)?)
    }

    /// Compression ratio vs storing the delta as raw f32. Counts packed bytes,
    /// per-group scales (32-bit), and zeros (32-bit, always 0 here but counted).
    pub fn compression_ratio(&self) -> f32 {
        let original_bits = self.numel as f32 * 32.0;
        let compressed_bits = self.packed.len() as f32 * 8.0
            + self.scales.len() as f32 * 32.0
            + self.zeros.len() as f32 * 32.0
            + (self.shape.len() as f32 * 32.0)
            + 64.0; // numel + bits + group_size header
        original_bits / compressed_bits
    }
}

/// Pack i8 values into bits-per-element byte string. For INT8 this is a
/// straight reinterpret; INT4 packs two values per byte, INT2 packs four.
fn pack(bits: Bits, values: &[i8]) -> Vec<u8> {
    match bits {
        Bits::Int8 => values.iter().map(|&x| x as u8).collect(),
        Bits::Int4 => {
            let nbytes = values.len().div_ceil(2);
            let mut out = vec![0u8; nbytes];
            for (i, &v) in values.iter().enumerate() {
                // Mask to 4 bits, preserving two's complement.
                let nibble = (v as u8) & 0x0F;
                if i % 2 == 0 {
                    out[i / 2] |= nibble; // low nibble
                } else {
                    out[i / 2] |= nibble << 4; // high nibble
                }
            }
            out
        }
        Bits::Int2 => {
            let nbytes = values.len().div_ceil(4);
            let mut out = vec![0u8; nbytes];
            for (i, &v) in values.iter().enumerate() {
                let twobit = (v as u8) & 0x03;
                let shift = (i % 4) * 2;
                out[i / 4] |= twobit << shift;
            }
            out
        }
    }
}

/// Inverse of [`pack`]. Returns exactly `numel` signed values.
fn unpack(bits: Bits, packed: &[u8], numel: usize) -> Vec<i8> {
    match bits {
        Bits::Int8 => packed.iter().take(numel).map(|&b| b as i8).collect(),
        Bits::Int4 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let byte = packed[i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                // Sign-extend 4-bit to 8-bit.
                let signed = if nibble & 0x08 != 0 { (nibble | 0xF0) as i8 } else { nibble as i8 };
                out.push(signed);
            }
            out
        }
        Bits::Int2 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let byte = packed[i / 4];
                let shift = (i % 4) * 2;
                let twobit = (byte >> shift) & 0x03;
                // Sign-extend 2-bit (bit 1 is sign).
                let signed = if twobit & 0x02 != 0 { (twobit | 0xFC) as i8 } else { twobit as i8 };
                out.push(signed);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn int8_round_trip_is_close_to_lossless() {
        let dev = cpu();
        let v: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.001).collect();
        let t = Tensor::from_vec(v.clone(), 256, &dev).unwrap();
        let zero = Tensor::zeros(256, DType::F32, &dev).unwrap();
        let dq = DeltaQuant::encode(&t, &zero, Bits::Int8, Some(128)).unwrap();
        let back: Vec<f32> = dq.decode(&dev).unwrap().to_vec1().unwrap();
        // INT8 with abs_max ~0.127 and 127 levels -> error <= 0.001.
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "int8 err too large: {} vs {}", a, b);
        }
    }

    #[test]
    fn int4_packing_two_per_byte() {
        let vals: Vec<i8> = vec![1, -1, 7, -8];
        let packed = pack(Bits::Int4, &vals);
        assert_eq!(packed.len(), 2);
        // idx0=+1 (0001), idx1=-1 (1111) -> byte0 = 0xF1
        assert_eq!(packed[0], 0xF1);
        // idx2=+7 (0111), idx3=-8 (1000) -> byte1 = 0x87
        assert_eq!(packed[1], 0x87);
        let unp = unpack(Bits::Int4, &packed, 4);
        assert_eq!(unp, vals);
    }

    #[test]
    fn int2_packing_four_per_byte() {
        // i2 range: -2..=1
        let vals: Vec<i8> = vec![1, -1, -2, 0];
        let packed = pack(Bits::Int2, &vals);
        assert_eq!(packed.len(), 1);
        // 0=00, -2=10, -1=11, 1=01 reading low->high: bits 0-1=01, 2-3=11, 4-5=10, 6-7=00
        // -> 0b00_10_11_01 = 0x2D
        assert_eq!(packed[0], 0x2D);
        let unp = unpack(Bits::Int2, &packed, 4);
        assert_eq!(unp, vals);
    }

    #[test]
    fn int4_round_trip_under_max_abs() {
        let dev = cpu();
        // Stay within +/- 0.7 so qmax=7 with scale ~0.1 gives clean integer rounding.
        let v: Vec<f32> = (0..128).map(|i| ((i % 15) as f32 - 7.0) * 0.1).collect();
        let t = Tensor::from_vec(v.clone(), 128, &dev).unwrap();
        let zero = Tensor::zeros(128, DType::F32, &dev).unwrap();
        let dq = DeltaQuant::encode(&t, &zero, Bits::Int4, Some(128)).unwrap();
        let back: Vec<f32> = dq.decode(&dev).unwrap().to_vec1().unwrap();
        // INT4 -> quantization error bounded by 0.5 * scale.
        let max_scale = dq.scales.iter().cloned().fold(0.0_f32, f32::max);
        for (a, b) in v.iter().zip(back.iter()) {
            let err = (a - b).abs();
            assert!(err <= 0.5 * max_scale + 1e-6, "err {} > tol", err);
        }
    }

    #[test]
    fn last_group_can_be_partial() {
        let dev = cpu();
        let v: Vec<f32> = (0..130).map(|i| (i as f32) * 0.01).collect();
        let t = Tensor::from_vec(v, 130, &dev).unwrap();
        let zero = Tensor::zeros(130, DType::F32, &dev).unwrap();
        let dq = DeltaQuant::encode(&t, &zero, Bits::Int4, Some(128)).unwrap();
        assert_eq!(dq.scales.len(), 2);
        let back = dq.decode(&dev).unwrap();
        assert_eq!(back.dims(), &[130]);
    }
}
