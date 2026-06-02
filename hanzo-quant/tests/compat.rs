//! Cross-implementation compatibility tests.
//!
//! Verifies the Rust BitDelta wire format is self-consistent and matches the
//! invariants the Python `bitdelta.py` would produce for the same input.
//! Since the Python reference doesn't pin a wire format (it pickles via
//! torch.save), we instead validate the canonical packing:
//!
//! - 8 signs per byte, little-endian (bit 0 = element 0, 1 => positive).
//! - Per-tensor scale = mean(|delta|), clamped above 1e-8.
//! - Reconstruction always lands on `+/- scale`.

use candle_core::{DType, Device, Tensor};
use hanzo_quant::{BitDelta, Backend, QuantizedDelta};

#[test]
fn bitdelta_byte_layout_matches_python_invariants() {
    let dev = Device::Cpu;
    // 16-element delta: 8 positive, 8 negative; expected packed = [0xFF, 0x00].
    let v: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { -1.0 }).collect();
    let delta = Tensor::from_vec(v, 16, &dev).unwrap();
    let bd = BitDelta::encode_delta(&delta).unwrap();

    assert_eq!(bd.sign_bits, vec![0xFF, 0x00]);
    assert_eq!(bd.numel, 16);
    assert_eq!(bd.shape, vec![16]);
    assert!((bd.scale - 1.0).abs() < 1e-6);

    // Reconstruction is +/- scale, never 0, never the original magnitude.
    let back: Vec<f32> = bd.decode(&dev).unwrap().to_vec1().unwrap();
    for (i, x) in back.iter().enumerate() {
        if i < 8 {
            assert!((x - 1.0).abs() < 1e-6, "elt {} expected +1, got {}", i, x);
        } else {
            assert!((x + 1.0).abs() < 1e-6, "elt {} expected -1, got {}", i, x);
        }
    }
}

#[test]
fn bitdelta_scale_is_per_tensor_mean_abs() {
    let dev = Device::Cpu;
    // Mix of magnitudes so mean(|.|) != max(|.|).
    let v: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    // mean(|.|) = (0.1+0.2+...+0.8)/8 = 3.6/8 = 0.45
    let delta = Tensor::from_vec(v, 8, &dev).unwrap();
    let bd = BitDelta::encode_delta(&delta).unwrap();
    assert!((bd.scale - 0.45).abs() < 1e-6, "scale = {}", bd.scale);
}

#[test]
fn unified_serde_json_round_trip_preserves_apply() {
    let dev = Device::Cpu;
    let base = Tensor::zeros((16, 16), DType::F32, &dev).unwrap();
    let v: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let weight = Tensor::from_vec(v, (16, 16), &dev).unwrap();

    let q = Backend::deltaquant_int4_default().encode(&weight, &base).unwrap();
    let json = serde_json::to_string(&q).unwrap();
    let q2: QuantizedDelta = serde_json::from_str(&json).unwrap();

    let w1: Vec<f32> = q.apply(&base).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let w2: Vec<f32> = q2.apply(&base).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(w1, w2);
}

#[test]
fn bitdelta_apply_then_reextract_is_idempotent() {
    // Encode once, apply to base, then re-encode that result against base.
    // Because BitDelta is lossy (clamps to +/- scale), the second pass should
    // produce *exactly* the same signs and a scale equal to the prior scale.
    let dev = Device::Cpu;
    let base = Tensor::zeros((32,), DType::F32, &dev).unwrap();
    let v: Vec<f32> = (0..32).map(|i| if i % 3 == 0 { 0.5 } else { -0.2 }).collect();
    let weight = Tensor::from_vec(v, 32, &dev).unwrap();

    let bd1 = BitDelta::encode(&weight, &base).unwrap();
    let w1 = bd1.apply(&base).unwrap();
    let bd2 = BitDelta::encode(&w1, &base).unwrap();

    assert_eq!(bd1.sign_bits, bd2.sign_bits);
    assert!(
        (bd1.scale - bd2.scale).abs() < 1e-6,
        "scales drifted: {} vs {}",
        bd1.scale,
        bd2.scale
    );
}
