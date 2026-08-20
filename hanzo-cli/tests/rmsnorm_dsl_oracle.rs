// Cross-repo gate for the fused (x + residual) -> rmsnorm that quantized_qwen/_qwen3/_qwen3_moe and
// quantized_minimax take every layer: engine's `QRmsNorm::forward_of_sum` vs
// `hanzo_kernel::norm::add_rmsnorm_ref`, the kernel DSL's authoritative CPU oracle. `rmsnorm_numeric.rs`
// gates the same ROCm kernel against an engine-internal composite (a separate add, then
// `hanzo_nn::ops::rms_norm`), so both of ITS sides live in this dependency tree and a drift from the
// DSL's definition of the op is invisible to it. hanzo-kernel gates its lowered `add_rmsnorm_blk`
// against `add_rmsnorm_ref` per backend; composing the two gives engine-production == DSL-oracle ==
// DSL-on-every-backend, the precondition for engine's norm ever dispatching to the DSL twin. Mirrors
// `gdn_recurrence_portable_matches_dsl_oracle`, which closes this loop for GDN.
//
// The drift is not hypothetical: `rms_norm_residual` (hanzo-engine/src/cuda/sort.cu:216) normalizes x
// and THEN adds the residual, while `add_rmsnorm_blk` adds first and normalizes the sum. Same name
// shape, different arithmetic. Only `rms_norm_of_sum` / `forward_of_sum` is the add-then-normalize
// twin, so it is the one gated here.
//
// `forward_of_sum` dispatches ROCm -> CUDA -> Vulkan -> portable tensor ops, so whichever backend this
// build carries is the one under test; no `#![cfg]`, the portable fallback is worth gating too.
//
// F32 only, deliberately. `add_rmsnorm_ref` reduces in f32, and so does every path here, so the only
// difference is reduction ORDER and the comparison stays exact enough to catch a real drift. The ROCm
// arm also accepts F16, but there the incumbent squares the sum AFTER rounding it to the tensor dtype
// while the DSL kernel squares the pre-cast f32 -- a genuine semantic split that an f32 oracle cannot
// adjudicate, and that the DSL side does not yet cover (`add_rmsnorm_blk_run` is f32-only, unlike the
// dtype-generic `rms_norm_blk_run`). Gating F16 needs an f16 DSL twin first.

use hanzo_engine::layers::QRmsNorm;
use hanzo_ml::quantized::{GgmlDType, QStorage, QTensor};
use hanzo_ml::{DType, Device, Tensor};

const EPS: f32 = 1e-5;

/// The accelerator this build carries, else CPU (which exercises the portable fallback).
fn device() -> Device {
    #[cfg(feature = "rocm")]
    if let Ok(d) = Device::new_rocm(0) {
        return d;
    }
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

/// Deterministic values in roughly [-3, 3], varying per index so every row differs.
fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    (x - 0.5) * 6.0
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("tensor -> Vec<f32>")
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-4))
        .fold(0.0f32, f32::max)
}

/// Runs one shape and returns `(sum_bit_exact, y_max_rel)`.
fn check(dev: &Device, rows: usize, cols: usize) -> (bool, f32) {
    let n = rows * cols;
    let x_v: Vec<f32> = (0..n).map(val).collect();
    let r_v: Vec<f32> = (0..n).map(|i| val(i.wrapping_mul(7) + 3)).collect();
    // Norm weight around 1.0, some negative, to catch a dropped sign.
    let w_v: Vec<f32> = (0..cols).map(|i| 1.0 + 0.25 * val(i)).collect();

    let x = Tensor::from_vec(x_v.clone(), (rows, cols), dev).expect("x");
    let r = Tensor::from_vec(r_v.clone(), (rows, cols), dev).expect("r");
    // Quantize on CPU then upload: `quantize_onto` has no device-side arm for a ROCm destination
    // ("Invalid quantize source storage locations: not on cpu"). F32 keeps block_size 1, so the narrow
    // and non-power-of-2 shapes below stay legal.
    let w_cpu = Tensor::from_vec(w_v, (cols,), &Device::Cpu).expect("w");
    let qt_cpu = QTensor::quantize(&w_cpu, GgmlDType::F32).expect("quantize weight");
    let bytes = qt_cpu.data().expect("weight bytes");
    let storage = QStorage::from_data(bytes, dev, GgmlDType::F32).expect("upload weight");
    let qw = QTensor::new(storage, (cols,)).expect("weight qtensor");
    let norm = QRmsNorm::new(qw, EPS).expect("QRmsNorm");

    // Oracle against the weight the kernel ACTUALLY holds, so the gate measures the kernel and never
    // the quantizer round-trip.
    let alpha: Vec<f32> = to_f32(norm.weight()).iter().map(|w| w * 1.01).collect();
    let (want_s, want_y) =
        hanzo_kernel::norm::add_rmsnorm_ref(&x_v, &r_v, &alpha, rows, cols, EPS);

    let (got_s, got_y) = norm.forward_of_sum(&x, &r).expect("forward_of_sum");

    // `sum` is a plain f32 add on every path, so it must agree to the bit; only the normalization
    // carries a reduction-order difference (block/shuffle reduce vs the oracle's sequential sum).
    (to_f32(&got_s) == want_s, max_rel(&to_f32(&got_y), &want_y))
}

#[test]
fn rms_norm_of_sum_matches_dsl_oracle() {
    let dev = device();
    // Decode (rows=1), batch>1, under a warp, just over the 1024 block-size switch, and wide rows at
    // the shipping hidden sizes.
    let shapes = [
        (1usize, 2048usize),
        (1, 4096),
        (4, 2048),
        (3, 31),
        (5, 1025),
        (2, 12288),
    ];
    let mut worst = 0.0f32;
    let mut bad = Vec::new();
    for &(rows, cols) in &shapes {
        let (sum_exact, y_rel) = check(&dev, rows, cols);
        eprintln!(
            "[of_sum vs dsl-oracle] {dev:?} {rows}x{cols}  sum_bit_exact={sum_exact}  y_rel={y_rel:.2e}"
        );
        worst = worst.max(y_rel);
        // 1e-5 relative is ~2 orders above the f32 reduction-order noise measured on gfx1151 for this
        // op (~1e-6), and far below anything a real semantic drift could hide under.
        if !sum_exact || !(y_rel < 1e-5) {
            bad.push(format!("{rows}x{cols} sum_bit_exact={sum_exact} y_rel={y_rel:.3e}"));
        }
    }
    assert!(
        bad.is_empty(),
        "engine's forward_of_sum diverged from hanzo_kernel::norm::add_rmsnorm_ref on {dev:?}: {}",
        bad.join("; ")
    );
    eprintln!("[of_sum vs dsl-oracle] {dev:?} OK, worst y_rel={worst:.2e}");
}
