// Numeric gate for the native Q4_K decode matvec (qmatvec_q4k_{f16,bf16}). Q4_K decode keeps the
// GGML super-blocks (144 B / 256 weights) resident in VRAM and dequantizes on-the-fly
// (ASYMMETRIC: weight = d·sc·q − dmin·m), reading ~7x fewer bytes than dense f16. A silent
// layout/packing/nibble/scale-unpack bug here corrupts every decoded token, so this gate compares,
// on the SAME ROCm device:
//   1. the f16/bf16 Q4_K matvec vs an EXACT f32 CPU reference that replicates the CPU oracle
//      k_quants::BlockQ4K::to_float dequant (same get_scale_min_k4, same nibble order) · x, and
//   2. the bf16 matvec vs the f16 matvec on identical inputs.
// Both the kernel and the reference accumulate in f32 in the same per-super-block, per-chunk order,
// so any divergence beyond f32 rounding is a real bug. `nbad` counts outputs exceeding a tiny
// magnitude-scaled tolerance; the gate requires nbad == 0. Lives in hanzo-cli so it links the HIP
// runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::{RocmDevice, RocmStorage};
use std::io::Write;

const QK_K: usize = 256;
const Q4K_BYTES: usize = 144; // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)

// Deterministic activation value in roughly [-3, 3], varying per index (same generator as the
// Q8_0 bf16 gate).
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

// get_scale_min_k4 from k_quants/utils.rs — the exact CPU oracle scale/min unpack. `s` = the 12
// packed scale bytes of a super-block, j = sub-block index 0..7. Returns (sc, m), each 6-bit.
fn get_scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        let d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        let m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        (d, m)
    }
}

// Build one Q4_K super-block's 144 bytes for output row `nn`, super-block `b`. d/dmin are small,
// exactly-f16-representable, nonuniform scales; the 12 scale bytes and 128 quant bytes are varied
// so all 8 sub-blocks, both nibble lanes, and the full 0..15 quant range are exercised.
fn build_block(nn: usize, b: usize, out: &mut Vec<u8>) {
    let d = 0.0625f32 + 0.03125f32 * (((nn + b) % 5) as f32); // exact f16
    let dmin = 0.03125f32 + 0.015625f32 * (((nn * 2 + b) % 4) as f32); // exact f16
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
    // 12 scale bytes: spread across 0..255 so the 6-bit sc/m unpack (incl. the j>=4 cross-byte
    // path) covers a wide range.
    for sb in 0..12 {
        let v = ((nn * 7 + b * 13 + sb * 29 + 3) % 251) as u8;
        out.push(v);
    }
    // 128 quant bytes: each holds two independent nibbles in 0..15.
    for qb in 0..128 {
        let lo = ((nn + b * 3 + qb * 5) % 16) as u8;
        let hi = ((nn * 2 + b + qb * 7 + 1) % 16) as u8;
        out.push((lo & 0x0F) | ((hi & 0x0F) << 4));
    }
}

// Exact f32 reference for output row `nn`: replicates k_quants::BlockQ4K::to_float dequant
// (per 64-weight chunk: scale-pair (2c, 2c+1), qs bytes [c*32, c*32+32); byte l low nibble ->
// weight c*64+l, high nibble -> weight c*64+32+l; weight = d·sc·q − dmin·m) dotted with the
// activation `x` (already rounded to the dtype the kernel reads). Accumulation order matches the
// kernel: outer super-block, inner chunk, inner 32.
fn reference_row(wq: &[u8], nn: usize, nblk: usize, x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for b in 0..nblk {
        let base = (nn * nblk + b) * Q4K_BYTES;
        let d = f16::from_le_bytes([wq[base], wq[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([wq[base + 2], wq[base + 3]]).to_f32();
        let scales = &wq[base + 4..base + 16];
        let qs = &wq[base + 16..base + 144];
        let xb = &x[b * QK_K..b * QK_K + QK_K];
        for c in 0..4 {
            let (sc1, m1i) = get_scale_min_k4(2 * c, scales);
            let (sc2, m2i) = get_scale_min_k4(2 * c + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1i as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2i as f32;
            for l in 0..32 {
                let qb = qs[c * 32 + l] as u32;
                let wlo = d1 * (qb & 0xF) as f32 - m1;
                let whi = d2 * (qb >> 4) as f32 - m2;
                acc += wlo * xb[c * 64 + l];
                acc += whi * xb[c * 64 + 32 + l];
            }
        }
    }
    acc
}

fn check(dev: &RocmDevice, log: &mut String, n: usize, k: usize) {
    assert!(k % QK_K == 0, "k must be a multiple of 256");
    let nblk = k / QK_K;

    // Activation x[k] as f32, and the same values rounded to f16 + bf16 (what each kernel reads).
    let xf: Vec<f32> = (0..k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xb: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xf_h: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let xf_b: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xb).expect("upload x bf16");

    // W[n,k] Q4_K bytes.
    let mut wq_bytes: Vec<u8> = Vec::with_capacity(n * nblk * Q4K_BYTES);
    for nn in 0..n {
        for b in 0..nblk {
            build_block(nn, b, &mut wq_bytes);
        }
    }
    assert_eq!(wq_bytes.len(), n * nblk * Q4K_BYTES);
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    let y_h = dev.matvec_q4k(&wst, &xst_h, n, k).expect("matvec_q4k f16");
    let y_b = dev.matvec_q4k(&wst, &xst_b, n, k).expect("matvec_q4k bf16");
    assert_eq!(y_b.dtype(), hanzo_ml::DType::BF16, "bf16 matvec keeps bf16");
    assert_eq!(y_h.dtype(), hanzo_ml::DType::F16, "f16 matvec keeps f16");
    let yh = to_f32(&y_h);
    let yb = to_f32(&y_b);

    // Exact f32 references using the matching dtype-rounded activation.
    let mut ref_h = vec![0f32; n];
    let mut ref_b = vec![0f32; n];
    for nn in 0..n {
        ref_h[nn] = reference_row(&wq_bytes, nn, nblk, &xf_h);
        ref_b[nn] = reference_row(&wq_bytes, nn, nblk, &xf_b);
    }

    // Magnitude scale for tolerances (outputs are O(1..10s)).
    let scale = ref_b
        .iter()
        .chain(ref_h.iter())
        .fold(0f32, |m, &v| m.max(v.abs()))
        .max(1.0);
    // f16/bf16 output rounding + f32 reduction-order differences only. 1% of magnitude is far below
    // any layout/packing/scale-unpack bug (those flip signs or scale by integer factors) but well
    // above f32 reorder noise.
    let tol = 0.01 * scale;

    let mut max_err_h = 0f32;
    let mut max_err_b = 0f32;
    let mut max_err_fb = 0f32;
    let mut nbad = 0usize;
    for i in 0..n {
        let eh = (yh[i] - ref_h[i]).abs();
        let eb = (yb[i] - ref_b[i]).abs();
        let efb = (yb[i] - yh[i]).abs();
        max_err_h = max_err_h.max(eh);
        max_err_b = max_err_b.max(eb);
        max_err_fb = max_err_fb.max(efb);
        if eh > tol || eb > tol {
            nbad += 1;
        }
    }
    log.push_str(&format!(
        "n={n} k={k} nbad={nbad}/{n} max_err_f16_vs_ref={max_err_h:.5} max_err_bf16_vs_ref={max_err_b:.5} max_err_bf16_vs_f16={max_err_fb:.5} (tol {tol:.5}) scale={scale:.3}\n"
    ));
    assert!(
        nbad == 0,
        "qmatvec_q4k mismatch: n={n} k={k} nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b} tol={tol}"
    );
}

#[test]
fn qmatvec_q4k_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");

    check(&dev, &mut log, 64, 256); // single super-block per row, exact warp tile
    check(&dev, &mut log, 4096, 4096); // production decode shape (hidden 4096 = 16 super-blocks)
    check(&dev, &mut log, 1024, 12288); // wide-k FFN-ish (48 super-blocks)
    check(&dev, &mut log, 17, 4096); // n not a multiple of 8 (partial last warp-row block)
    check(&dev, &mut log, 151936, 4096); // vocab-sized output (lm_head shape), partial tail

    let _ = std::fs::File::create("C:\\qmatvec-q4k-test.txt")
        .and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
}
