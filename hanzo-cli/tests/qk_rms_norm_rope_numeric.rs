// Numeric gate for the FUSED ROCm q/k-RMSNorm + positions-RoPE kernel (rope.hip
// `rope_norm_positions`, wired via layers::rocm_qk_rms_norm_rope_positions). Drives the SAME engine
// entry `qk_rms_norm_rope_positions` twice on identical inputs: once with the fused kernel (default)
// and once with QK_NORM_ROPE_FALLBACK=1 (the proven rms_norm + rope_positions chain). A wrong
// norm/rope = positional gibberish across every token, so byte-closeness here is a hard gate. The two
// paths both accumulate the rms sum in f32; only the reduction ORDER differs, so the tolerance covers
// f32/f16 reorder, not a layout/scale bug. Run with --test-threads=1 (process-global FALLBACK env).
#![cfg(feature = "rocm")]

use hanzo_engine::layers::qk_rms_norm_rope_positions;
use hanzo_ml::{DType, Device, Tensor};

fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    (x - 0.5) * 4.0
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn check(dev: &Device, log: &mut String, b: usize, h: usize, kh: usize, t: usize, d: usize, dtype: DType, tol: f32) {
    let half = d / 2;
    let max_pos = 256usize;
    let mk = |n: usize, off: usize| {
        Tensor::from_vec((0..n).map(|i| val(i + off)).collect::<Vec<_>>(), n, dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    };
    let q = mk(b * h * t * d, 1).reshape((b, h, t, d)).unwrap();
    let k = mk(b * kh * t * d, 7).reshape((b, kh, t, d)).unwrap();
    // RMSNorm weights near 1.0 (realistic scale).
    let qw = Tensor::from_vec((0..d).map(|i| 1.0 + 0.1 * val(i)).collect::<Vec<_>>(), d, dev).unwrap().to_dtype(dtype).unwrap();
    let kw = Tensor::from_vec((0..d).map(|i| 1.0 + 0.1 * val(i + 3)).collect::<Vec<_>>(), d, dev).unwrap().to_dtype(dtype).unwrap();
    let mut cosf = vec![0f32; max_pos * half];
    let mut sinf = vec![0f32; max_pos * half];
    for p in 0..max_pos {
        for j in 0..half {
            let freq = 1.0f32 / 10000f32.powf(2.0 * j as f32 / d as f32);
            let ang = p as f32 * freq;
            cosf[p * half + j] = ang.cos();
            sinf[p * half + j] = ang.sin();
        }
    }
    let cos = Tensor::from_vec(cosf, (max_pos, half), dev).unwrap().to_dtype(dtype).unwrap();
    let sin = Tensor::from_vec(sinf, (max_pos, half), dev).unwrap().to_dtype(dtype).unwrap();
    let positions = Tensor::from_vec((0..b).map(|i| (3 + 5 * i) as u32).collect::<Vec<_>>(), b, dev).unwrap();
    let eps = 1e-6f64;

    std::env::remove_var("QK_NORM_ROPE_FALLBACK");
    let (qf, kf) = qk_rms_norm_rope_positions(&q, &k, &qw, &kw, eps, eps, &cos, &sin, true, &positions).unwrap();
    std::env::set_var("QK_NORM_ROPE_FALLBACK", "1");
    let (qs, ks) = qk_rms_norm_rope_positions(&q, &k, &qw, &kw, eps, eps, &cos, &sin, true, &positions).unwrap();
    std::env::remove_var("QK_NORM_ROPE_FALLBACK");

    let mut me = 0f32;
    for (a, bb) in to_f32(&qf).iter().zip(to_f32(&qs).iter()).chain(to_f32(&kf).iter().zip(to_f32(&ks).iter())) {
        me = me.max((a - bb).abs());
    }
    log.push_str(&format!("b={b} h={h} kh={kh} t={t} d={d} {dtype:?} max_abs_err={me:.6} (tol {tol})\n"));
    assert!(me < tol, "qk_rms_norm_rope fused vs fallback: b={b} h={h} t={t} d={d} {dtype:?} err={me} >= {tol}");
}

#[test]
fn qk_rms_norm_rope_numeric() {
    let mut log = String::new();
    let dev = Device::new_rocm(0).expect("rocm device");
    // Decode (t=1, head_dim 128) -- the production path -- plus prefill-ish t and other dims/dtypes.
    check(&dev, &mut log, 1, 32, 8, 1, 128, DType::F32, 2e-3);
    check(&dev, &mut log, 1, 32, 8, 1, 128, DType::F16, 3e-2);
    check(&dev, &mut log, 1, 32, 8, 16, 128, DType::F32, 2e-3);
    check(&dev, &mut log, 2, 16, 4, 7, 128, DType::F32, 2e-3);
    check(&dev, &mut log, 1, 8, 8, 1, 64, DType::F32, 2e-3);
    eprintln!("{log}");
}
