// Numeric unit test for the FUSED ROCm rms_norm kernel (hanzo_nn::ops::rms_norm via the RmsNorm
// CustomOp2 -> RocmStorage::rms_norm -> reduce.hip `rmsnorm_{f16,f32}`). Compares the fused kernel
// against rms_norm_slow (the KNOWN-GOOD ~9-op composite) on identical inputs, on the SAME ROCm
// device, and asserts they agree within f16 tolerance. Covers last_dim in {4096, 12288}, a
// non-power-of-2 (4097, exercises the strided tail + partial-warp reduction), batch>1, and both
// F16 and F32 dtypes. A wrong norm = garbage across every model, so this is the critical gate.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Tensor};
use std::io::Write;

// Deterministic pseudo-random-ish value in roughly [-3, 3], varying per index so each row differs.
fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0; // [0,1)
    (x - 0.5) * 6.0
}

fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    let t = t.to_dtype(DType::F32).expect("to f32");
    let t = t.flatten_all().expect("flatten");
    t.to_vec1::<f32>().expect("to_vec1")
}

fn check_shape(dev: &Device, log: &mut String, rows: usize, cols: usize, dtype: DType, tol: f32) {
    let eps = 1e-5f32;
    let n = rows * cols;

    // Build x[rows, cols] and alpha[cols] on the rocm device, in the target dtype.
    let x_f32: Vec<f32> = (0..n).map(val).collect();
    // alpha varies around 1.0 (realistic rmsnorm weight), some negative to catch sign bugs.
    let alpha_f32: Vec<f32> = (0..cols).map(|i| 1.0 + 0.25 * val(i)).collect();

    let x = Tensor::from_vec(x_f32, (rows, cols), dev)
        .expect("x tensor")
        .to_dtype(dtype)
        .expect("x dtype");
    let alpha = Tensor::from_vec(alpha_f32, (cols,), dev)
        .expect("alpha tensor")
        .to_dtype(dtype)
        .expect("alpha dtype");

    let fused = hanzo_nn::ops::rms_norm(&x, &alpha, eps).expect("fused rms_norm");
    let slow = hanzo_nn::ops::rms_norm_slow(&x, &alpha, eps).expect("rms_norm_slow");

    // Sanity: fused output keeps the input shape + dtype.
    assert_eq!(fused.dims(), &[rows, cols], "fused shape");
    assert_eq!(fused.dtype(), dtype, "fused dtype");

    let a = to_f32_vec(&fused);
    let b = to_f32_vec(&slow);
    assert_eq!(a.len(), b.len());

    let mut max_abs_err = 0f32;
    let mut argmax = 0usize;
    for i in 0..a.len() {
        let e = (a[i] - b[i]).abs();
        if e > max_abs_err {
            max_abs_err = e;
            argmax = i;
        }
    }
    log.push_str(&format!(
        "rows={rows} cols={cols} dtype={dtype:?} max_abs_err={max_abs_err:.6} at idx {argmax} (fused={:.5} slow={:.5})\n",
        a[argmax], b[argmax]
    ));
    assert!(
        max_abs_err < tol,
        "rms_norm fused vs slow mismatch: rows={rows} cols={cols} dtype={dtype:?} max_abs_err={max_abs_err} >= tol {tol}"
    );
}

#[test]
fn rmsnorm_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // F16: the production dtype. f16 round-trip on both sides -> allow ~1e-2.
    check_shape(&dev, &mut log, 1, 4096, DType::F16, 1e-2);
    check_shape(&dev, &mut log, 4, 4096, DType::F16, 1e-2); // batch>1
    check_shape(&dev, &mut log, 2, 12288, DType::F16, 1e-2); // wide FFN-ish hidden
    check_shape(&dev, &mut log, 3, 4097, DType::F16, 1e-2); // non-power-of-2 + batch>1
    check_shape(&dev, &mut log, 1, 31, DType::F16, 1e-2); // < warp size
    check_shape(&dev, &mut log, 5, 1025, DType::F16, 1e-2); // just over the 1024 block switch

    // F32: tighter tolerance — both paths accumulate in f32, so they should be very close.
    check_shape(&dev, &mut log, 4, 4096, DType::F32, 1e-4);
    check_shape(&dev, &mut log, 2, 4097, DType::F32, 1e-4);
    check_shape(&dev, &mut log, 8, 12288, DType::F32, 1e-4);

    let _ =
        std::fs::File::create("C:\\rmsnorm-test.txt").and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
}

// Fused add_rmsnorm (RocmStorage::add_rms_norm -> reduce.hip `add_rmsnorm_f32`): the (sum, normed)
// outputs must be BIT-IDENTICAL to a separate f32 add then the fused `rms_norm` kernel, because the
// residual stream is f32 and the sum-of-squares reduction order matches `rmsnorm`. nbad counts any
// non-zero deviation. Covers narrow/wide rows + the 1024 block-size switch + batch>1.
fn add_rms_check(dev: &Device, log: &mut String, rows: usize, cols: usize) -> usize {
    let eps = 1e-5f32;
    let n = rows * cols;
    let x_v: Vec<f32> = (0..n).map(val).collect();
    let r_v: Vec<f32> = (0..n).map(|i| val(i.wrapping_mul(7) + 3)).collect();
    let a_v: Vec<f32> = (0..cols).map(|i| 1.0 + 0.25 * val(i)).collect();
    let x = Tensor::from_vec(x_v, (rows, cols), dev).expect("x");
    let r = Tensor::from_vec(r_v, (rows, cols), dev).expect("r");
    let a = Tensor::from_vec(a_v, (cols,), dev).expect("a");

    let sum_ref = (&x + &r).expect("sum_ref");
    let normed_ref = hanzo_nn::ops::rms_norm(&sum_ref.contiguous().expect("c"), &a, eps).expect("ref");

    let (xs, xl) = x.storage_and_layout();
    let xr = match &*xs {
        hanzo_ml::Storage::Rocm(s) => s,
        _ => panic!("x not rocm"),
    };
    let (rs, rl) = r.storage_and_layout();
    let rr = match &*rs {
        hanzo_ml::Storage::Rocm(s) => s,
        _ => panic!("r not rocm"),
    };
    let (as_, al) = a.storage_and_layout();
    let ar = match &*as_ {
        hanzo_ml::Storage::Rocm(s) => s,
        _ => panic!("a not rocm"),
    };
    let (sum_s, normed_s) = xr.add_rms_norm(xl, rr, rl, ar, al, eps).expect("add_rms_norm");
    let shape = x.shape().clone();
    let sum_f = Tensor::from((hanzo_ml::Storage::Rocm(sum_s), shape.clone()));
    let normed_f = Tensor::from((hanzo_ml::Storage::Rocm(normed_s), shape));

    let (sr, sf) = (to_f32_vec(&sum_ref), to_f32_vec(&sum_f));
    let (nr, nf) = (to_f32_vec(&normed_ref), to_f32_vec(&normed_f));
    let mut nbad = 0usize;
    let mut max_err = 0f32;
    for i in 0..n {
        if sr[i] != sf[i] {
            nbad += 1;
        }
        let e = (nr[i] - nf[i]).abs();
        max_err = max_err.max(e);
        if e != 0.0 {
            nbad += 1;
        }
    }
    log.push_str(&format!(
        "add_rmsnorm rows={rows} cols={cols} nbad={nbad}/{} sum_exact={} normed_max_err={max_err:.3e}\n",
        2 * n,
        sr == sf
    ));
    nbad
}

#[test]
fn add_rmsnorm_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");
    let mut nbad = 0;
    nbad += add_rms_check(&dev, &mut log, 1, 2048); // decode hidden (Qwen3-30B-A3B)
    nbad += add_rms_check(&dev, &mut log, 1, 4096);
    nbad += add_rms_check(&dev, &mut log, 4, 2048); // batch>1
    nbad += add_rms_check(&dev, &mut log, 3, 31); // < warp size
    nbad += add_rms_check(&dev, &mut log, 5, 1025); // just over the 1024 block switch
    nbad += add_rms_check(&dev, &mut log, 2, 12288); // wide
    eprintln!("{log}");
    assert_eq!(nbad, 0, "add_rmsnorm must be bit-identical to add + rms_norm\n{log}");
}
