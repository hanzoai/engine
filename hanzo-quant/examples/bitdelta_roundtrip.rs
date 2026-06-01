//! BitDelta round-trip + compression ratio benchmark.
//!
//! ```sh
//! cargo run --release --example bitdelta_roundtrip
//! ```

use candle_core::{DType, Device, Tensor};
use hanzo_quant::BitDelta;

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;

    // Simulate a 4096x1024 weight matrix with a small fine-tune delta.
    let rows = 4096usize;
    let cols = 1024usize;
    let numel = rows * cols;

    // Base: deterministic pseudo-random in [-1, 1).
    let base_v: Vec<f32> = (0..numel)
        .map(|i| ((i.wrapping_mul(2654435761) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let base = Tensor::from_vec(base_v.clone(), (rows, cols), &dev)?;

    // Fine-tune delta: small, zero-mean-ish.
    let delta_v: Vec<f32> = (0..numel)
        .map(|i| {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            s * ((i as f32).sin() * 0.05).abs()
        })
        .collect();
    let weight_v: Vec<f32> = base_v.iter().zip(delta_v.iter()).map(|(a, b)| a + b).collect();
    let weight = Tensor::from_vec(weight_v.clone(), (rows, cols), &dev)?;

    // Encode + measure.
    let bd = BitDelta::encode(&weight, &base)?;
    let ratio = bd.compression_ratio();
    let nbytes_packed = bd.sign_bits.len();
    let nbytes_raw = numel * 4;

    println!("BitDelta benchmark — {} x {} ({} elements)", rows, cols, numel);
    println!("  raw f32 delta:  {} bytes", nbytes_raw);
    println!("  packed signs:   {} bytes (+ 1 scale f32)", nbytes_packed);
    println!("  scale:          {:.6e}", bd.scale);
    println!("  compression:    {:.2}x", ratio);

    // Round-trip and report mean abs reconstruction error vs original delta.
    let w_hat = bd.apply(&base)?;
    let w_hat_v: Vec<f32> = w_hat.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let mean_abs_err: f32 = weight_v
        .iter()
        .zip(w_hat_v.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / numel as f32;
    println!("  mean |err|:     {:.4e}", mean_abs_err);

    // Sanity: reconstruction is bounded by max(|delta|).
    let max_delta: f32 = delta_v.iter().map(|x| x.abs()).fold(0.0, f32::max);
    assert!(
        mean_abs_err < max_delta,
        "reconstruction error {} exceeded max delta {}",
        mean_abs_err,
        max_delta
    );

    // Wire-format round trip.
    let bytes = bd.to_bytes();
    let bd2 = BitDelta::from_bytes(&bytes)?;
    assert_eq!(bd.sign_bits, bd2.sign_bits);
    assert_eq!(bd.scale.to_bits(), bd2.scale.to_bits());
    println!("  wire bytes:     {} (round-trip OK)", bytes.len());

    Ok(())
}
