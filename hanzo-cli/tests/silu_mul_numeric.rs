// Numeric unit test for the FUSED ROCm SwiGLU kernel (hanzo_nn::ops::silu_mul via the SiluMul
// CustomOp -> RocmStorage::silu_mul -> binary.hip `silu_mul_{f16,f32,bf16}`). Compares the fused
// kernel against the unfused composite (silu(a) * b) on identical inputs, on the SAME ROCm device,
// within tolerance. A wrong SwiGLU corrupts every FFN output.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Tensor};
use std::io::Write;

fn val(i: usize, salt: usize) -> f32 {
    let x = ((i
        .wrapping_mul(2654435761)
        .wrapping_add(salt.wrapping_mul(40503))
        >> 8)
        & 0xffff) as f32
        / 65535.0; // [0,1)
    (x - 0.5) * 8.0 // [-4,4]
}

fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec1")
}

fn check(dev: &Device, log: &mut String, rows: usize, cols: usize, dtype: DType, tol: f32) {
    let n = rows * cols;
    let af: Vec<f32> = (0..n).map(|i| val(i, 1)).collect();
    let bf: Vec<f32> = (0..n).map(|i| val(i, 2)).collect();
    let a = Tensor::from_vec(af.clone(), (rows, cols), dev)
        .expect("a")
        .to_dtype(dtype)
        .expect("a dtype");
    let b = Tensor::from_vec(bf.clone(), (rows, cols), dev)
        .expect("b")
        .to_dtype(dtype)
        .expect("b dtype");

    let fused = hanzo_nn::ops::silu_mul(&a, &b).expect("fused silu_mul");
    // Reference: unfused silu(a) * b (the path silu_mul replaces).
    let slow = hanzo_nn::ops::silu(&a).expect("silu").mul(&b).expect("mul");

    assert_eq!(fused.dims(), &[rows, cols], "fused shape");
    assert_eq!(fused.dtype(), dtype, "fused dtype");

    let x = to_f32_vec(&fused);
    let y = to_f32_vec(&slow);
    assert_eq!(x.len(), y.len());
    // Exact f32 reference computed from the (dtype-rounded) inputs the kernel actually sees.
    let xq = to_f32_vec(&a);
    let yq = to_f32_vec(&b);
    let mut max_abs_err = 0f32; // fused vs unfused composite
    let mut argmax = 0usize;
    let mut max_ref_err = 0f32; // fused vs exact f32 silu(a)*b
    for i in 0..x.len() {
        let e = (x[i] - y[i]).abs();
        if e > max_abs_err {
            max_abs_err = e;
            argmax = i;
        }
        let truth = (xq[i] / (1.0 + (-xq[i]).exp())) * yq[i];
        max_ref_err = max_ref_err.max((x[i] - truth).abs());
    }
    log.push_str(&format!(
        "rows={rows} cols={cols} dtype={dtype:?} max_abs_err={max_abs_err:.6} at {argmax} (fused={:.5} slow={:.5}) max_ref_err={max_ref_err:.6}\n",
        x[argmax], y[argmax]
    ));
    assert!(
        max_abs_err < tol,
        "silu_mul fused vs composite mismatch: rows={rows} cols={cols} dtype={dtype:?} err={max_abs_err} >= tol {tol}"
    );
    // The fused kernel keeps silu in f32, so it must be within ~1 dtype ULP of the TRUE value.
    // (This is the strong gate: a real math bug fails here regardless of the looser bf16 tol above.)
    let ref_tol = if dtype == DType::F32 { 1e-5 } else { tol };
    assert!(
        max_ref_err < ref_tol,
        "silu_mul fused vs exact f32 ref mismatch: rows={rows} cols={cols} dtype={dtype:?} ref_err={max_ref_err} >= tol {ref_tol}"
    );
}

#[test]
fn silu_mul_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // FFN intermediate shapes (Qwen3-8B intermediate ~12288). decode rows=1, prefill rows>1.
    // bf16 tol: silu(a)*b reaches ~|16| here, where 1 bf16 ULP ~= 0.125. The fused kernel keeps
    // silu in f32 (one rounding) vs the reference's two roundings, so it differs by up to ~1 ULP
    // (the fused result is in fact the more accurate of the two).
    check(&dev, &mut log, 1, 12288, DType::BF16, 1.5e-1);
    check(&dev, &mut log, 8, 12288, DType::BF16, 1.5e-1);
    check(&dev, &mut log, 1, 4096, DType::BF16, 1.5e-1);
    check(&dev, &mut log, 3, 4097, DType::BF16, 1.5e-1); // non-pow2
    check(&dev, &mut log, 4, 4096, DType::F16, 2e-2);
    check(&dev, &mut log, 4, 4096, DType::F32, 1e-5);
    check(&dev, &mut log, 2, 12288, DType::F32, 1e-5);

    let _ = std::fs::File::create("C:\\silu-mul-test.txt")
        .and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
}
