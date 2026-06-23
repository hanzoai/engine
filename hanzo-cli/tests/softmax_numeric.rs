// Numeric unit test for the FUSED ROCm softmax kernel (hanzo_nn::ops::softmax_last_dim via the
// SoftmaxLastDim CustomOp -> RocmStorage::softmax_last_dim -> reduce.hip `softmax_{f16,f32,bf16}`).
// Compares the fused kernel against the KNOWN-GOOD composite (hanzo_nn::ops::softmax over the last
// dim: max_keepdim/broadcast_sub/exp/sum_keepdim/broadcast_div) on identical inputs, on the SAME
// ROCm device, within tolerance. A wrong softmax = wrong attention = gibberish.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Tensor, D};
use std::io::Write;

fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0; // [0,1)
    (x - 0.5) * 10.0 // [-5,5]: a realistic logit spread
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
    let xf: Vec<f32> = (0..n).map(val).collect();
    let x = Tensor::from_vec(xf, (rows, cols), dev)
        .expect("x")
        .to_dtype(dtype)
        .expect("x dtype");

    let fused = hanzo_nn::ops::softmax_last_dim(&x).expect("fused softmax");
    // Reference: the generic composite softmax over the last dim (runs the unfused path on rocm).
    let slow = hanzo_nn::ops::softmax(&x, D::Minus1).expect("composite softmax");

    assert_eq!(fused.dims(), &[rows, cols], "fused shape");
    assert_eq!(fused.dtype(), dtype, "fused dtype");

    let a = to_f32_vec(&fused);
    let b = to_f32_vec(&slow);
    assert_eq!(a.len(), b.len());

    // Each row must sum to 1 (probability simplex) and match the reference elementwise.
    let mut max_abs_err = 0f32;
    let mut argmax = 0usize;
    for i in 0..a.len() {
        let e = (a[i] - b[i]).abs();
        if e > max_abs_err {
            max_abs_err = e;
            argmax = i;
        }
    }
    let mut max_sum_err = 0f32;
    for r in 0..rows {
        let s: f32 = a[r * cols..(r + 1) * cols].iter().sum();
        max_sum_err = max_sum_err.max((s - 1.0).abs());
    }
    log.push_str(&format!(
        "rows={rows} cols={cols} dtype={dtype:?} max_abs_err={max_abs_err:.6} at {argmax} max_sum_err={max_sum_err:.6}\n"
    ));
    assert!(
        max_abs_err < tol,
        "softmax fused vs composite mismatch: rows={rows} cols={cols} dtype={dtype:?} err={max_abs_err} >= tol {tol}"
    );
    // bf16/f16 rows sum to ~1 within accumulation error.
    assert!(
        max_sum_err < tol * 4.0 + 1e-3,
        "softmax row sum off simplex: rows={rows} cols={cols} dtype={dtype:?} sum_err={max_sum_err}"
    );
}

#[test]
fn softmax_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // Attention-score shapes: rows = heads*positions, cols = key length.
    check(&dev, &mut log, 32, 16, DType::BF16, 2e-2); // decode-ish: short context
    check(&dev, &mut log, 32, 512, DType::BF16, 2e-2); // prefill context
    check(&dev, &mut log, 1, 4096, DType::BF16, 2e-2); // long row
    check(&dev, &mut log, 7, 33, DType::BF16, 2e-2); // non-pow2
    check(&dev, &mut log, 8, 31, DType::F16, 1e-2); // < warp
    check(&dev, &mut log, 16, 128, DType::F16, 1e-2);
    check(&dev, &mut log, 16, 512, DType::F32, 1e-5);
    check(&dev, &mut log, 4, 4097, DType::F32, 1e-5); // non-pow2 long

    let _ =
        std::fs::File::create("C:\\softmax-test.txt").and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
}
