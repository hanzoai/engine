// Numeric unit test for the BF16-native decode matvec (qmatvec_q8_0_bf16). The decode cast-chain
// fusion makes RocmDevice::matvec_q8_0 accept bf16 activations directly (returning bf16) so the
// decode path skips the bf16->f32->f16->bf16 cast detour. A silent math bug here corrupts every
// decoded token, so this gate compares, on the SAME ROCm device:
//   1. the bf16 matvec vs an EXACT f32 CPU reference (dequant W * x), and
//   2. the bf16 matvec vs the proven f16 matvec on identical inputs.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::{RocmDevice, RocmStorage};

// Deterministic value in roughly [-3, 3], varying per index.
fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0; // [0,1)
    (x - 0.5) * 6.0
}

fn to_f32(s: &RocmStorage) -> Vec<f32> {
    match s.to_cpu_storage().expect("to cpu") {
        hanzo_ml::CpuStorage::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        hanzo_ml::CpuStorage::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
        hanzo_ml::CpuStorage::F32(v) => v,
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}

fn check(dev: &RocmDevice, log: &mut String, n: usize, k: usize) {
    let nblk = k / 32;

    // Activation x[k] and the same values as f16 + bf16.
    let xf: Vec<f32> = (0..k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xb: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xb).expect("upload x bf16");

    // W[n,k] Q8_0 bytes: per-block f16 scale d varies, 32 int8 quants in -4..4.
    let mut wq_bytes: Vec<u8> = Vec::with_capacity(n * nblk * 34);
    for nn in 0..n {
        for b in 0..nblk {
            let d = 0.1 + 0.05 * ((nn + b) % 7) as f32;
            wq_bytes.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            for kk in 0..32 {
                let q = (((nn * 3 + b * 32 + kk) % 9) as i32 - 4) as i8;
                wq_bytes.push(q as u8);
            }
        }
    }
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    let y_h = dev.matvec_q8_0(&wst, &xst_h, n, k).expect("matvec f16");
    let y_b = dev.matvec_q8_0(&wst, &xst_b, n, k).expect("matvec bf16");
    assert_eq!(y_b.dtype(), hanzo_ml::DType::BF16, "bf16 matvec keeps bf16");
    let yh = to_f32(&y_h);
    let yb = to_f32(&y_b);

    // Exact f32 reference: dequant(W) . x  with the bf16-rounded activation (the kernel reads bf16).
    let mut refy = vec![0f32; n];
    for nn in 0..n {
        let mut acc = 0f32;
        for b in 0..nblk {
            let d = f16::from_le_bytes([
                wq_bytes[(nn * nblk + b) * 34],
                wq_bytes[(nn * nblk + b) * 34 + 1],
            ])
            .to_f32();
            let mut s = 0f32;
            for kk in 0..32 {
                let wqv = wq_bytes[(nn * nblk + b) * 34 + 2 + kk] as i8 as f32;
                // reference uses the bf16-rounded x, matching what the bf16 kernel sees.
                s += wqv * xb[b * 32 + kk].to_f32();
            }
            acc += d * s;
        }
        refy[nn] = acc;
    }

    let mut max_err_ref = 0f32;
    let mut max_err_fb = 0f32;
    for i in 0..n {
        max_err_ref = max_err_ref.max((yb[i] - refy[i]).abs());
        max_err_fb = max_err_fb.max((yb[i] - yh[i]).abs());
    }
    // Tolerances: bf16 has ~3 sig figs; outputs here are O(1..10). Scale tol by magnitude.
    let scale = refy.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    let tol_ref = 0.02 * scale; // bf16 round-trip on x + bf16 output rounding
    let tol_fb = 0.04 * scale; // f16 vs bf16 differ only by activation/output rounding
    log.push_str(&format!(
        "n={n} k={k} max_err_vs_ref={max_err_ref:.5} (tol {tol_ref:.5}) max_err_bf16_vs_f16={max_err_fb:.5} (tol {tol_fb:.5}) scale={scale:.3}\n"
    ));
    assert!(
        max_err_ref < tol_ref,
        "bf16 matvec vs f32 ref mismatch: n={n} k={k} err={max_err_ref} >= tol {tol_ref}"
    );
    assert!(
        max_err_fb < tol_fb,
        "bf16 vs f16 matvec mismatch: n={n} k={k} err={max_err_fb} >= tol {tol_fb}"
    );
}

#[test]
fn qmatvec_bf16_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");

    check(&dev, &mut log, 64, 128); // small, exact tiles
    check(&dev, &mut log, 4096, 4096); // production decode shape (Qwen3-8B hidden)
    check(&dev, &mut log, 1024, 12288); // wide-k FFN-ish
    check(&dev, &mut log, 17, 4096); // n not a multiple of 8 (partial last block)

    eprintln!("{log}");
}
