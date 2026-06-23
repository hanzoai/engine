// Numeric unit test for the FUSED ROCm rope kernel (hanzo_nn::rotary_emb::rope via the RotaryEmb
// CustomOp3 -> RocmStorage::rope -> reduce.hip `rope_{f16,f32,bf16}`). Compares the fused kernel
// against rope_slow (the KNOWN-GOOD ~7-op composite: cat/cat/broadcast_mul/rotate_half/add) on
// identical inputs, on the SAME ROCm device, within f16 tolerance. A wrong rope = positional
// gibberish across every token, so this is a hard correctness gate.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Tensor};
use std::io::Write;

fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0; // [0,1)
    (x - 0.5) * 4.0
}

fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec1")
}

fn check(
    dev: &Device,
    log: &mut String,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    dtype: DType,
    tol: f32,
) {
    let n = b * h * t * d;
    let xf: Vec<f32> = (0..n).map(val).collect();
    // cos/sin are [t, d/2]; build a realistic rotary table (theta_j over positions).
    let half = d / 2;
    let mut cosf = vec![0f32; t * half];
    let mut sinf = vec![0f32; t * half];
    for p in 0..t {
        for j in 0..half {
            let freq = 1.0f32 / 10000f32.powf(2.0 * j as f32 / d as f32);
            let ang = p as f32 * freq;
            cosf[p * half + j] = ang.cos();
            sinf[p * half + j] = ang.sin();
        }
    }

    let x = Tensor::from_vec(xf, (b, h, t, d), dev)
        .expect("x")
        .to_dtype(dtype)
        .expect("x dtype");
    let cos = Tensor::from_vec(cosf, (t, half), dev)
        .expect("cos")
        .to_dtype(dtype)
        .expect("cos dtype");
    let sin = Tensor::from_vec(sinf, (t, half), dev)
        .expect("sin")
        .to_dtype(dtype)
        .expect("sin dtype");

    let fused = hanzo_nn::rotary_emb::rope(&x, &cos, &sin).expect("fused rope");
    let slow = hanzo_nn::rotary_emb::rope_slow(&x, &cos, &sin).expect("rope_slow");

    assert_eq!(fused.dims(), &[b, h, t, d], "fused shape");
    assert_eq!(fused.dtype(), dtype, "fused dtype");

    let a = to_f32_vec(&fused);
    let bb = to_f32_vec(&slow);
    assert_eq!(a.len(), bb.len());
    let mut max_abs_err = 0f32;
    let mut argmax = 0usize;
    for i in 0..a.len() {
        let e = (a[i] - bb[i]).abs();
        if e > max_abs_err {
            max_abs_err = e;
            argmax = i;
        }
    }
    log.push_str(&format!(
        "b={b} h={h} t={t} d={d} dtype={dtype:?} max_abs_err={max_abs_err:.6} at {argmax} (fused={:.5} slow={:.5})\n",
        a[argmax], bb[argmax]
    ));
    assert!(
        max_abs_err < tol,
        "rope fused vs slow mismatch: b={b} h={h} t={t} d={d} dtype={dtype:?} err={max_abs_err} >= tol {tol}"
    );
}

#[test]
fn rope_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // Decode shape: t=1 (single new token), Qwen3-8B head dim 128.
    check(&dev, &mut log, 1, 32, 1, 128, DType::BF16, 2e-2);
    check(&dev, &mut log, 1, 8, 1, 128, DType::BF16, 2e-2); // kv heads
                                                            // Prefill-ish: many positions.
    check(&dev, &mut log, 1, 32, 512, 128, DType::BF16, 2e-2);
    check(&dev, &mut log, 2, 4, 7, 64, DType::BF16, 2e-2); // batch>1, small
                                                           // F16 + F32 dtypes.
    check(&dev, &mut log, 1, 32, 16, 128, DType::F16, 2e-2);
    check(&dev, &mut log, 1, 32, 16, 128, DType::F32, 1e-4);
    check(&dev, &mut log, 1, 16, 33, 96, DType::F32, 1e-4); // non-pow2 t, d=96

    let _ =
        std::fs::File::create("C:\\rope-test.txt").and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
}
