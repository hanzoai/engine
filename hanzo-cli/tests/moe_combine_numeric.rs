// Numeric unit test for the FUSED ROCm MoE expert-combine (hanzo_ml::quantized::moe_combine ->
// RocmStorage::moe_combine -> quant.hip `moe_combine_{f16,bf16,f32}`). Compares the fused kernel
// against the composite it replaces (ys.broadcast_mul(scores).sum(Minus2)) AND an exact f32 oracle,
// on the SAME ROCm device. A wrong combine corrupts every MoE layer's residual.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Tensor, D};

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

fn check(dev: &Device, log: &mut String, t: usize, topk: usize, n: usize, dtype: DType, tol: f32) {
    let ysf: Vec<f32> = (0..t * topk * n).map(|i| val(i, 1)).collect();
    let scf: Vec<f32> = (0..t * topk).map(|i| (val(i, 7) + 4.0) / 8.0).collect(); // [0,1)
    let ys = Tensor::from_vec(ysf, (t, topk, n), dev)
        .expect("ys")
        .to_dtype(dtype)
        .expect("ys dtype");
    let scores = Tensor::from_vec(scf, (t, topk), dev).expect("scores"); // f32

    let fused = hanzo_ml::quantized::moe_combine(&ys, &scores).expect("fused moe_combine");
    // Composite the fused kernel replaces: ys -> f32, weighted by f32 scores, summed, rounded to dtype.
    let slow = ys
        .to_dtype(DType::F32)
        .expect("ys f32")
        .broadcast_mul(&scores.unsqueeze(D::Minus1).expect("unsqueeze"))
        .expect("bmul")
        .sum(D::Minus2)
        .expect("sum")
        .to_dtype(dtype)
        .expect("slow dtype");

    assert_eq!(fused.dims(), &[t, n], "fused shape");
    assert_eq!(fused.dtype(), dtype, "fused dtype");

    let x = to_f32_vec(&fused);
    let y = to_f32_vec(&slow);
    let ysq = to_f32_vec(&ys); // dtype-rounded ys the kernel actually sees
    let scq = to_f32_vec(&scores);

    let mut nbad = 0usize;
    let mut max_err = 0f32; // fused vs composite
    let mut max_ref = 0f32; // fused vs exact f32 oracle
    for i in 0..t {
        for j in 0..n {
            let idx = i * n + j;
            let mut truth = 0f32;
            for e in 0..topk {
                truth += scq[i * topk + e] * ysq[(i * topk + e) * n + j];
            }
            let e_comp = (x[idx] - y[idx]).abs();
            let e_ref = (x[idx] - truth).abs();
            max_err = max_err.max(e_comp);
            max_ref = max_ref.max(e_ref);
            if e_ref > tol {
                nbad += 1;
            }
        }
    }
    log.push_str(&format!(
        "t={t} topk={topk} n={n} dtype={dtype:?} nbad={nbad}/{} max_err(vs composite)={max_err:.6} max_ref={max_ref:.6}\n",
        t * n
    ));
    assert_eq!(
        nbad, 0,
        "moe_combine {dtype:?} t={t} topk={topk} n={n}: nbad={nbad} max_ref={max_ref}"
    );
}

#[test]
fn moe_combine_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");

    // Qwen3-30B-A3B MoE: topk=8, hidden=2048. Prefill (t>1) + decode (t=1) + non-pow2 tails.
    check(&dev, &mut log, 1, 8, 2048, DType::F16, 6e-2);
    check(&dev, &mut log, 1, 8, 2048, DType::BF16, 3e-1);
    check(&dev, &mut log, 128, 8, 2048, DType::F16, 6e-2);
    check(&dev, &mut log, 128, 8, 2048, DType::BF16, 3e-1);
    check(&dev, &mut log, 1024, 8, 2048, DType::F16, 6e-2);
    check(&dev, &mut log, 37, 8, 257, DType::F16, 6e-2); // non-pow2 t and n
    check(&dev, &mut log, 64, 6, 768, DType::F32, 1e-4);
    check(&dev, &mut log, 200, 4, 160, DType::F32, 1e-4);

    eprintln!("{log}");
}
