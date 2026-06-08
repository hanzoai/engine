// Numeric unit test for the int8 Q8_0 gemm (qmmq_q8_0) — isolates the undocumented RDNA3 iu8
// WMMA fragment conventions. Compares the GPU kernel against an EXACT CPU reference computed from
// the same quantized inputs (xq/xd read back from the GPU), so any mismatch is a pure
// layout/packing/signedness bug, not quantization error. Writes a diagnostic to C:\qmmq-test.txt.
// Lives in hanzo-cli (engine workspace) so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::f16;
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::{RocmDevice, RocmStorage};
use std::io::Write;

#[test]
fn qmmq_numeric() {
    let mut log = String::new();
    // Exercises multiple tiles (M,N > 64), all 4 waves + 4 fragments, partial tiles (not mult-64),
    // and double-buffering over 4 K-blocks.
    let m = 200usize;
    let n = 160usize;
    let k = 128usize;
    let nblk = k / 32;

    let dev = RocmDevice::new(0).expect("rocm device");

    // X[m,k] f16 — small varied values.
    let xh: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 7) as i32 - 3) as f32))
        .collect();
    let xst = dev.storage_from_slice(&xh).expect("upload x");

    // W[n,k] Q8_0 bytes: per block d=0.5 (f16) + 32 int8 in -2..2.
    let mut wq_bytes: Vec<u8> = Vec::with_capacity(n * nblk * 34);
    for nn in 0..n {
        for b in 0..nblk {
            wq_bytes.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
            for kk in 0..32 {
                let q = (((nn + b * 32 + kk) % 5) as i32 - 2) as i8;
                wq_bytes.push(q as u8);
            }
        }
    }
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    // Fused gemm: the activation is quantized to int8 inside the kernel.
    let y = dev.qmmq_q8_0(&xst, &wst, m, n, k).expect("qmmq");
    let y_cpu = to_f32(&y);

    // Exact reference: quantize X on CPU the same way the kernel does (per-32-block absmax/127).
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
                    let wqv = wq_bytes[(nn * nblk + b) * 34 + 2 + kk] as i8 as i32;
                    isum += xqv * wqv;
                }
                acc += isum as f32 * (amax / 127.0) * 0.5f32;
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
    for nn in 0..n {
        log.push_str(&format!("{:.0} ", y_cpu[nn]));
    }
    log.push_str("\nrow0 ref: ");
    for nn in 0..n {
        log.push_str(&format!("{:.0} ", refy[nn]));
    }
    log.push_str("\ncol0 gpu: ");
    for mm in 0..m {
        log.push_str(&format!("{:.0} ", y_cpu[mm * n]));
    }
    log.push_str("\ncol0 ref: ");
    for mm in 0..m {
        log.push_str(&format!("{:.0} ", refy[mm * n]));
    }
    log.push('\n');

    let _ = std::fs::File::create("C:\\qmmq-test.txt").and_then(|mut f| f.write_all(log.as_bytes()));
    eprintln!("{log}");
    assert!(nbad == 0, "qmmq mismatch: {nbad} bad, max_err {max_err}");
}

fn to_f32(s: &RocmStorage) -> Vec<f32> {
    match s.to_cpu_storage().expect("to cpu") {
        hanzo_ml::CpuStorage::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        hanzo_ml::CpuStorage::F32(v) => v,
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}

fn to_u8(s: &RocmStorage) -> Vec<u8> {
    match s.to_cpu_storage().expect("to cpu") {
        hanzo_ml::CpuStorage::U8(v) => v,
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}
