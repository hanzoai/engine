// Numeric unit test for the int8 Q4_0 gemm (qmmq_q4_0) — the first new native type riding the
// unified iu8 WMMA core. Compares the GPU kernel against an EXACT CPU reference computed from the
// same quantized inputs: X is quantized to int8 exactly as the kernel does (per-32-block
// absmax/127, round-half-away), and the Q4_0 weight is dequantized as (nibble - 8) * d. Any
// mismatch is a pure layout/packing/nibble-order/signedness bug, not quantization error. Writes a
// diagnostic to C:\qmmq-q4_0-test.txt. Lives in hanzo-cli so it links the HIP runtime via
// hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::f16;
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::{RocmDevice, RocmStorage};
use std::io::Write;

#[test]
fn qmmq_q4_0_numeric() {
    let mut log = String::new();
    // Exercises multiple tiles (M,N > 128 in both dims), all 4 waves + 4 fragments, partial tiles
    // (not a multiple of 128), and double-buffering over several K-blocks.
    let m = 200usize;
    let n = 160usize;
    let k = 128usize;
    let nblk = k / 32;

    let dev = RocmDevice::new(0).expect("rocm device");

    // X[m,k] f16 — small varied values (same generator as the Q8_0 test).
    let xh: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 7) as i32 - 3) as f32))
        .collect();
    let xst = dev.storage_from_slice(&xh).expect("upload x");

    // W[n,k] Q4_0 bytes: per 18-byte block = f16 d + 16 nibble-pairs (32 weights). Vary d per
    // (row,block) and vary nibbles across the full 0..15 range so both nibble lanes are exercised.
    // GGML Q4_0 element order: byte j low nibble -> weight[j] (j<16); byte j high nibble ->
    // weight[j+16]. weight = (nibble - 8) * d.
    let mut wq_bytes: Vec<u8> = Vec::with_capacity(n * nblk * 18);
    let mut d_of = vec![0f32; n * nblk]; // per (row,block) f16 scale, kept for the reference
    for nn in 0..n {
        for b in 0..nblk {
            // A small, exactly-f16-representable, nonuniform scale.
            let dval = 0.25f32 + 0.0625f32 * (((nn + b) % 4) as f32);
            let dh = f16::from_f32(dval);
            d_of[nn * nblk + b] = dh.to_f32();
            wq_bytes.extend_from_slice(&dh.to_le_bytes());
            for j in 0..16 {
                // Two independent nibbles, spread over 0..15.
                let lo = ((nn + b * 3 + j * 5) % 16) as u8;
                let hi = ((nn * 2 + b + j * 7 + 1) % 16) as u8;
                wq_bytes.push((lo & 0x0F) | ((hi & 0x0F) << 4));
            }
        }
    }
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    // GPU Q4_0 gemm: activation is quantized to int8 inside the launcher (quantize_q8).
    let y = dev.qmmq_q4_0(&xst, &wst, m, n, k).expect("qmmq_q4_0");
    let y_cpu = to_f32(&y);

    // Exact reference: quantize X on CPU exactly like the kernel (per-32-block absmax/127,
    // round-half-away-from-zero, clamp ±127), dequantize Q4_0 as (nibble-8), symmetric accumulate
    // scaled by (amax/127) * d  (== xd[block] * wd[block], the same dA·dB as Q8_0).
    let mut refy = vec![0f32; m * n];
    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0f32;
            for b in 0..nblk {
                let mut amax = 0f32;
                for kk in 0..32 {
                    amax = amax.max(xh[mm * k + b * 32 + kk].to_f32().abs());
                }
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let mut isum = 0i32;
                for kk in 0..32 {
                    let xv = xh[mm * k + b * 32 + kk].to_f32();
                    let xqv = (xv * inv).round().clamp(-127.0, 127.0) as i32;
                    // Q4_0 weight int8: low nibble of byte kk for kk<16, high nibble of byte kk-16
                    // for kk>=16, each centered by -8.
                    let byte = wq_bytes[(nn * nblk + b) * 18 + 2 + (kk % 16)];
                    let nib = if kk < 16 {
                        (byte & 0x0F) as i32
                    } else {
                        (byte >> 4) as i32
                    };
                    let wqv = nib - 8;
                    isum += xqv * wqv;
                }
                acc += isum as f32 * (amax / 127.0) * d_of[nn * nblk + b];
            }
            refy[mm * n + nn] = acc;
        }
    }

    let mut max_err = 0f32;
    let mut nbad = 0;
    for i in 0..m * n {
        let e = (y_cpu[i] - refy[i]).abs();
        if e > 0.5 {
            nbad += 1;
        }
        if e > max_err {
            max_err = e;
        }
    }
    log.push_str(&format!(
        "m={m} n={n} k={k} max_err={max_err} nbad={nbad}/{}\n",
        m * n
    ));
    log.push_str("row0 gpu: ");
    for nn in 0..n.min(48) {
        log.push_str(&format!("{:.1} ", y_cpu[nn]));
    }
    log.push_str("\nrow0 ref: ");
    for nn in 0..n.min(48) {
        log.push_str(&format!("{:.1} ", refy[nn]));
    }
    log.push_str("\ncol0 gpu: ");
    for mm in 0..m.min(48) {
        log.push_str(&format!("{:.1} ", y_cpu[mm * n]));
    }
    log.push_str("\ncol0 ref: ");
    for mm in 0..m.min(48) {
        log.push_str(&format!("{:.1} ", refy[mm * n]));
    }
    log.push('\n');

    let _ =
        std::fs::File::create("C:\\qmmq-q4_0-test.txt").and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
    assert!(nbad == 0, "qmmq_q4_0 mismatch: {nbad} bad, max_err {max_err}");
}

fn to_f32(s: &RocmStorage) -> Vec<f32> {
    match s.to_cpu_storage().expect("to cpu") {
        hanzo_ml::CpuStorage::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        hanzo_ml::CpuStorage::F32(v) => v,
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}
