//! Unified quantization interface.
//!
//! Use this when the caller doesn't care which backend is used — pick a
//! [`Backend`] up front and dispatch through the [`Quantize`] trait.
//!
//! ```no_run
//! use candle_core::{Device, Tensor, DType};
//! use hanzo_quant::{Backend, Quantize, QuantizedDelta};
//!
//! let dev = Device::Cpu;
//! let base   = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
//! let weight = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
//!
//! let q = Backend::BitDelta.encode(&weight, &base).unwrap();
//! let w_hat = q.apply(&base).unwrap();
//! ```

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::{bitdelta::BitDelta, deltaquant::{Bits, DeltaQuant}, Result};

/// Choose which compression backend to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Backend {
    /// 1-bit, fastest decode, ~32x raw compression.
    BitDelta,
    /// Multi-bit with `bits` per element and `group_size` per scale.
    DeltaQuant {
        bits: Bits,
        group_size: usize,
    },
}

impl Backend {
    pub fn deltaquant_int4_default() -> Self {
        Backend::DeltaQuant { bits: Bits::Int4, group_size: 128 }
    }
    pub fn deltaquant_int8_default() -> Self {
        Backend::DeltaQuant { bits: Bits::Int8, group_size: 128 }
    }

    /// Encode `weight - base` using this backend.
    pub fn encode(self, weight: &Tensor, base: &Tensor) -> Result<QuantizedDelta> {
        match self {
            Backend::BitDelta => Ok(QuantizedDelta::BitDelta(BitDelta::encode(weight, base)?)),
            Backend::DeltaQuant { bits, group_size } => Ok(QuantizedDelta::DeltaQuant(
                DeltaQuant::encode(weight, base, bits, Some(group_size))?,
            )),
        }
    }
}

/// Backend-tagged quantized delta. Round-tripping through serde_json gives you
/// a portable on-disk format; for tight binary representations call into
/// [`crate::bitdelta::BitDelta::to_bytes`] etc. directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum QuantizedDelta {
    BitDelta(BitDelta),
    DeltaQuant(DeltaQuant),
}

impl QuantizedDelta {
    pub fn decode(&self, device: &Device) -> Result<Tensor> {
        match self {
            QuantizedDelta::BitDelta(b) => b.decode(device),
            QuantizedDelta::DeltaQuant(d) => d.decode(device),
        }
    }

    pub fn apply(&self, base: &Tensor) -> Result<Tensor> {
        match self {
            QuantizedDelta::BitDelta(b) => b.apply(base),
            QuantizedDelta::DeltaQuant(d) => d.apply(base),
        }
    }

    pub fn compression_ratio(&self) -> f32 {
        match self {
            QuantizedDelta::BitDelta(b) => b.compression_ratio(),
            QuantizedDelta::DeltaQuant(d) => d.compression_ratio(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            QuantizedDelta::BitDelta(b) => &b.shape,
            QuantizedDelta::DeltaQuant(d) => &d.shape,
        }
    }
}

/// One-call surface. Implement if you want a custom backend on top of
/// `hanzo-quant`'s primitives.
pub trait Quantize {
    fn encode(&self, weight: &Tensor, base: &Tensor) -> Result<QuantizedDelta>;
}

impl Quantize for Backend {
    fn encode(&self, weight: &Tensor, base: &Tensor) -> Result<QuantizedDelta> {
        (*self).encode(weight, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn dispatch_through_backend() {
        let dev = Device::Cpu;
        let base = Tensor::zeros((8, 8), DType::F32, &dev).unwrap();
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let weight = Tensor::from_vec(v, (8, 8), &dev).unwrap();

        for backend in [
            Backend::BitDelta,
            Backend::deltaquant_int4_default(),
            Backend::deltaquant_int8_default(),
        ] {
            let q = backend.encode(&weight, &base).unwrap();
            assert_eq!(q.shape(), &[8, 8]);
            let w_hat = q.apply(&base).unwrap();
            assert_eq!(w_hat.dims(), &[8, 8]);
            assert!(q.compression_ratio() > 1.0);
        }
    }

    #[test]
    fn serde_round_trip() {
        let dev = Device::Cpu;
        let base = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let weight = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
        let q = Backend::BitDelta.encode(&weight, &base).unwrap();
        let json = serde_json::to_string(&q).unwrap();
        let q2: QuantizedDelta = serde_json::from_str(&json).unwrap();
        let w1: Vec<f32> = q.apply(&base).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let w2: Vec<f32> = q2.apply(&base).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w1, w2);
    }
}
