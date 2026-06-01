//! BitDelta: 1-bit quantization of fine-tune deltas.
//!
//! Algorithm (from `bitdelta.py`, simplified to the canonical case described in
//! the ZIP-7 paper / task brief):
//!
//! 1. `delta = weight - base`
//! 2. `scale = mean(|delta|)`  (single f32 per tensor)
//! 3. `sign_bit[i] = 1 if delta[i] >= 0 else 0`  (packed 8 per byte, LE)
//! 4. To reconstruct: `delta_hat[i] = (sign_bit[i] ? +1 : -1) * scale`
//!
//! Compression: 32-bit floats -> 1-bit signs + one f32 scale, ~32x raw or ~10x
//! once you account for an unquantized base and metadata overhead.
//!
//! ## Wire format (for [`BitDelta::to_bytes`] / [`BitDelta::from_bytes`])
//!
//! ```text
//! [0..4]   u32 LE  : magic = 0x42444C54 ("BDLT")
//! [4..8]   u32 LE  : numel (number of elements)
//! [8..12]  f32 LE  : scale
//! [12..15] u8 LE   : ndim
//! [15..]   u32 LE * ndim : shape
//! [..]     u8 * ceil(numel/8) : packed sign bits, bit i of byte i/8 = sign of elt i
//! ```

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const MAGIC: u32 = 0x4244_4C54; // "BDLT"

/// 1-bit delta. Hold onto the [`shape`](Self::shape) so we can round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitDelta {
    /// Packed sign bits. `sign_bits[i/8] >> (i%8) & 1` is the sign of elt `i`
    /// (1 => +, 0 => -). Length is `ceil(numel / 8)`.
    pub sign_bits: Vec<u8>,
    /// Per-tensor scale: `mean(|delta|)`. Always >= a tiny epsilon.
    pub scale: f32,
    /// Original number of elements (since `sign_bits` is bit-padded).
    pub numel: usize,
    /// Original tensor shape, so we can reshape on decode.
    pub shape: Vec<usize>,
}

impl BitDelta {
    /// Encode `delta = weight - base` to a [`BitDelta`].
    ///
    /// Both tensors must be `f32` and the same shape; convert with
    /// `tensor.to_dtype(DType::F32)?` first if needed.
    pub fn encode(weight: &Tensor, base: &Tensor) -> Result<Self> {
        let ws = weight.dims().to_vec();
        let bs = base.dims().to_vec();
        if ws != bs {
            return Err(Error::ShapeMismatch { base: bs, weight: ws });
        }

        let delta = weight.sub(base)?;
        Self::encode_delta(&delta)
    }

    /// Encode an already-computed delta tensor. Useful when you already have
    /// `delta = w_ft - w_base` and don't want to recompute.
    pub fn encode_delta(delta: &Tensor) -> Result<Self> {
        let shape = delta.dims().to_vec();
        let flat = delta.flatten_all()?.to_dtype(DType::F32)?;
        let values: Vec<f32> = flat.to_vec1()?;
        let numel = values.len();
        if numel == 0 {
            return Err(Error::Empty("delta tensor has 0 elements"));
        }

        // scale = mean(|delta|), clamped away from zero.
        let abs_sum: f32 = values.iter().map(|x| x.abs()).sum();
        let scale = (abs_sum / numel as f32).max(1e-8);

        // Pack 8 signs per byte, LE (bit 0 = element 0).
        let nbytes = numel.div_ceil(8);
        let mut sign_bits = vec![0u8; nbytes];
        for (i, &v) in values.iter().enumerate() {
            if v >= 0.0 {
                sign_bits[i / 8] |= 1u8 << (i % 8);
            }
        }

        Ok(Self { sign_bits, scale, numel, shape })
    }

    /// Decode back to a full-precision delta tensor on `device`.
    pub fn decode(&self, device: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(self.numel);
        for i in 0..self.numel {
            let bit = (self.sign_bits[i / 8] >> (i % 8)) & 1;
            // bit=1 => +scale, bit=0 => -scale
            let s = if bit == 1 { self.scale } else { -self.scale };
            out.push(s);
        }
        let t = Tensor::from_vec(out, self.shape.as_slice(), device)?;
        Ok(t)
    }

    /// Reconstruct the fine-tuned weight: `base + decode()`.
    pub fn apply(&self, base: &Tensor) -> Result<Tensor> {
        let bs = base.dims().to_vec();
        if bs != self.shape {
            return Err(Error::ShapeMismatch { base: bs, weight: self.shape.clone() });
        }
        let delta = self.decode(base.device())?;
        Ok(base.add(&delta)?)
    }

    /// Compression ratio vs storing the delta as raw f32. Counts the scale,
    /// the packed sign bytes, and the shape metadata (24 bytes amortized).
    pub fn compression_ratio(&self) -> f32 {
        let original_bits = self.numel as f32 * 32.0;
        let compressed_bits = self.sign_bits.len() as f32 * 8.0 // signs
            + 32.0                                                // scale
            + (self.shape.len() as f32 * 32.0)                    // shape u32s
            + 32.0; // numel
        original_bits / compressed_bits
    }

    /// Serialize to the compact binary format documented in the module header.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.shape.len() * 4 + self.sign_bits.len());
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&(self.numel as u32).to_le_bytes());
        buf.extend_from_slice(&self.scale.to_le_bytes());
        buf.push(self.shape.len() as u8);
        for &d in &self.shape {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        buf.extend_from_slice(&self.sign_bits);
        buf
    }

    /// Deserialize from the wire format. Validates the magic but does not
    /// otherwise sanity-check bit lengths beyond the obvious.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 13 {
            return Err(Error::Empty("BitDelta::from_bytes: buffer < header"));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(Error::Empty("BitDelta::from_bytes: bad magic"));
        }
        let numel = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let scale = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let ndim = bytes[12] as usize;
        let shape_end = 13 + ndim * 4;
        if bytes.len() < shape_end {
            return Err(Error::Empty("BitDelta::from_bytes: shape truncated"));
        }
        let mut shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let off = 13 + i * 4;
            shape.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize);
        }
        let want_bytes = numel.div_ceil(8);
        if bytes.len() < shape_end + want_bytes {
            return Err(Error::Empty("BitDelta::from_bytes: sign bits truncated"));
        }
        let sign_bits = bytes[shape_end..shape_end + want_bytes].to_vec();
        Ok(Self { sign_bits, scale, numel, shape })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn round_trip_preserves_signs_and_scale() {
        let dev = cpu();
        let base = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        // delta has alternating signs and magnitudes.
        let raw: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 0.1 } else { -0.3 }).collect();
        let weight = Tensor::from_vec(raw.clone(), (4, 4), &dev).unwrap();

        let bd = BitDelta::encode(&weight, &base).unwrap();

        // scale = mean(|delta|) = (8*0.1 + 8*0.3)/16 = 0.2
        assert!((bd.scale - 0.2).abs() < 1e-6, "scale was {}", bd.scale);

        let reconstructed = bd.decode(&dev).unwrap();
        let out: Vec<f32> = reconstructed.flatten_all().unwrap().to_vec1().unwrap();
        for (i, v) in out.iter().enumerate() {
            // Reconstruction is sign(raw[i]) * scale, NOT raw[i] itself.
            // Sign should match.
            assert!(v.signum() == raw[i].signum() || raw[i] == 0.0);
            assert!((v.abs() - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn sign_packing_is_little_endian() {
        let dev = cpu();
        // 9 elements: 8 positives then 1 negative -> first byte = 0xFF, second = 0x00.
        let v: Vec<f32> = vec![1.0; 8].into_iter().chain([-1.0]).collect();
        let bd = BitDelta::encode_delta(&Tensor::from_vec(v, 9, &dev).unwrap()).unwrap();
        assert_eq!(bd.sign_bits, vec![0xFF, 0x00]);
        assert_eq!(bd.numel, 9);
    }

    #[test]
    fn wire_format_round_trips() {
        let dev = cpu();
        let raw: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
        let bd = BitDelta::encode_delta(&Tensor::from_vec(raw, (10, 10), &dev).unwrap()).unwrap();
        let bytes = bd.to_bytes();
        let bd2 = BitDelta::from_bytes(&bytes).unwrap();
        assert_eq!(bd.sign_bits, bd2.sign_bits);
        assert_eq!(bd.scale.to_bits(), bd2.scale.to_bits());
        assert_eq!(bd.numel, bd2.numel);
        assert_eq!(bd.shape, bd2.shape);
    }

    #[test]
    fn apply_against_base_yields_base_plus_delta() {
        let dev = cpu();
        let base = Tensor::ones((3, 3), DType::F32, &dev).unwrap();
        let weight = (&base + 0.5f64).unwrap(); // all positive deltas
        let bd = BitDelta::encode(&weight, &base).unwrap();
        let w_hat = bd.apply(&base).unwrap();
        let v: Vec<f32> = w_hat.flatten_all().unwrap().to_vec1().unwrap();
        // All deltas were +0.5, mean(|.|)=0.5, so reconstruction is exactly base + 0.5.
        for x in v {
            assert!((x - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn compression_ratio_about_32x_for_large_tensors() {
        let dev = cpu();
        let n = 4096;
        let v: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.001).collect();
        let bd = BitDelta::encode_delta(&Tensor::from_vec(v, n, &dev).unwrap()).unwrap();
        let ratio = bd.compression_ratio();
        // Should be close to 32 (1 bit per element) once metadata is amortized.
        assert!(ratio > 30.0 && ratio < 33.0, "ratio = {ratio}");
    }
}
