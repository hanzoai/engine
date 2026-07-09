// Numeric gate for the IQ4_NL native ROCm decode (qmatvec_core<DW_IQ4_NL> in quant.hip). IQ4_NL is
// the 32-element nonlinear-codebook 4-bit type: y = d * KVALUES_IQ4NL[nibble] (the same 16-entry int8
// LUT IQ4_XS rides, but a single per-block f16 scale and a 32-elem block like Q4_0). The reference is
// the REAL CPU `BlockIQ4nl::to_float`, reinterpreting the exact GGML bytes the GPU reads. Codebook
// quants are integer-exact, so the only error is f16/bf16 activation rounding + f32 reorder; the gate
// requires nbad == 0 for the f16 AND bf16 matvec, single-expert and indexed-MoE. Lives in hanzo-cli so
// it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::quantized::k_quants::BlockIQ4nl;
use hanzo_ml::quantized::GgmlType;
use hanzo_ml::{RocmDevice, RocmQuantType, RocmStorage};

fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    (x - 0.5) * 6.0
}

fn byte(seed: usize) -> u8 {
    (seed.wrapping_mul(2654435761).wrapping_add(seed >> 7) & 0xff) as u8
}

fn to_f32(s: &RocmStorage) -> Vec<f32> {
    match s.to_cpu_storage().expect("to cpu") {
        hanzo_ml::CpuStorage::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        hanzo_ml::CpuStorage::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
        hanzo_ml::CpuStorage::F32(v) => v,
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}

fn as_blocks<T: Clone>(bytes: &[u8]) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    assert_eq!(
        bytes.len() % sz,
        0,
        "byte len {} not a multiple of {}",
        bytes.len(),
        sz
    );
    let n = bytes.len() / sz;
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * sz);
        v.set_len(n);
    }
    v
}

fn build_bytes<F: Fn(&mut [u8], usize, usize)>(
    n: usize,
    k: usize,
    blk: usize,
    tsz: usize,
    fill: F,
) -> Vec<u8> {
    assert_eq!(k % blk, 0);
    let nblk = k / blk;
    let mut out = vec![0u8; n * nblk * tsz];
    for r in 0..n {
        for b in 0..nblk {
            let base = (r * nblk + b) * tsz;
            let block = &mut out[base..base + tsz];
            for (i, by) in block.iter_mut().enumerate() {
                *by = byte(r * 7919 + b * 131 + i * 17 + 3);
            }
            fill(block, r, b);
        }
    }
    out
}

// Small exact-f16 scale at byte `off` so the reference and kernel f16 scales agree to the last bit.
fn put_d(block: &mut [u8], off: usize, r: usize, b: usize, lo: f32, step: f32, m: usize) {
    let d = lo + step * (((r + b) % m) as f32);
    let bytes = f16::from_f32(d).to_le_bytes();
    block[off] = bytes[0];
    block[off + 1] = bytes[1];
}

fn reference_row<T: GgmlType>(
    wq: &[u8],
    nn: usize,
    nblk: usize,
    blk: usize,
    tsz: usize,
    x: &[f32],
) -> f32 {
    let mut acc = 0f32;
    let mut deq = vec![0f32; blk];
    for b in 0..nblk {
        let base = (nn * nblk + b) * tsz;
        let blocks = as_blocks::<T>(&wq[base..base + tsz]);
        T::to_float(&blocks, &mut deq);
        let xb = &x[b * blk..b * blk + blk];
        for j in 0..blk {
            acc += deq[j] * xb[j];
        }
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn check<T: GgmlType, F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice,
    log: &mut String,
    name: &str,
    qt: RocmQuantType,
    n: usize,
    k: usize,
    blk: usize,
    tsz: usize,
    fill: F,
) {
    assert_eq!(k % blk, 0, "{name}: k must be a multiple of {blk}");
    let nblk = k / blk;
    let xf: Vec<f32> = (0..k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xbf: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xf_h: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let xf_b: Vec<f32> = xbf.iter().map(|v| v.to_f32()).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xbf).expect("upload x bf16");

    let wq_bytes = build_bytes(n, k, blk, tsz, fill);
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    let y_h = dev
        .matvec_quant(qt, &wst, &xst_h, n, k)
        .expect("matvec f16");
    let y_b = dev
        .matvec_quant(qt, &wst, &xst_b, n, k)
        .expect("matvec bf16");
    assert_eq!(
        y_h.dtype(),
        hanzo_ml::DType::F16,
        "{name}: f16 matvec keeps f16"
    );
    assert_eq!(
        y_b.dtype(),
        hanzo_ml::DType::BF16,
        "{name}: bf16 matvec keeps bf16"
    );
    let yh = to_f32(&y_h);
    let yb = to_f32(&y_b);

    let mut ref_h = vec![0f32; n];
    let mut ref_b = vec![0f32; n];
    for nn in 0..n {
        ref_h[nn] = reference_row::<T>(&wq_bytes, nn, nblk, blk, tsz, &xf_h);
        ref_b[nn] = reference_row::<T>(&wq_bytes, nn, nblk, blk, tsz, &xf_b);
    }
    let scale = ref_b
        .iter()
        .chain(ref_h.iter())
        .fold(0f32, |m, &v| m.max(v.abs()))
        .max(1.0);
    let tol = 0.01 * scale;
    let tol_b = 0.02 * scale;
    let mut max_err_h = 0f32;
    let mut max_err_b = 0f32;
    let mut nbad = 0usize;
    for i in 0..n {
        let eh = (yh[i] - ref_h[i]).abs();
        let eb = (yb[i] - ref_b[i]).abs();
        max_err_h = max_err_h.max(eh);
        max_err_b = max_err_b.max(eb);
        if eh > tol || eb > tol_b {
            nbad += 1;
        }
    }
    log.push_str(&format!(
        "{name:8} n={n} k={k} nbad={nbad}/{n} max_err_f16={max_err_h:.5} max_err_bf16={max_err_b:.5} (tol {tol:.5}) scale={scale:.3}\n"
    ));
    assert!(nbad == 0, "{name} IQ4_NL decode mismatch: nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b} tol={tol}");
}

#[allow(clippy::too_many_arguments)]
fn moe_check<T: GgmlType, F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice,
    log: &mut String,
    name: &str,
    qt: RocmQuantType,
    e_cnt: usize,
    nrows: usize,
    n: usize,
    k: usize,
    blk: usize,
    tsz: usize,
    fill: F,
) {
    let nblk = k / blk;
    let expert_bytes = n * nblk * tsz;
    let mut bank: Vec<u8> = Vec::with_capacity(e_cnt * expert_bytes);
    for e in 0..e_cnt {
        bank.extend_from_slice(&build_bytes(n, k, blk, tsz, |b, r, bb| {
            fill(b, r * (e + 1) + e, bb)
        }));
    }
    let wbank = dev.storage_from_slice(&bank).expect("upload bank");
    let ids: Vec<u32> = (0..nrows).map(|s| ((s * 3 + 1) % e_cnt) as u32).collect();
    let ids_dev = dev.storage_from_slice(&ids).expect("upload ids");
    let xf: Vec<f32> = (0..nrows * k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xbf: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xf_h: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let xf_b: Vec<f32> = xbf.iter().map(|v| v.to_f32()).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xbf).expect("upload x bf16");

    let y_h = dev
        .moe_matvec_quant(qt, &wbank, &xst_h, &ids_dev, nrows, n, k)
        .expect("moe f16");
    let y_b = dev
        .moe_matvec_quant(qt, &wbank, &xst_b, &ids_dev, nrows, n, k)
        .expect("moe bf16");
    let yh = to_f32(&y_h);
    let yb = to_f32(&y_b);

    let mut nbad = 0usize;
    let mut max_err_h = 0f32;
    let mut max_err_b = 0f32;
    let mut scale = 1f32;
    for (s, &eid) in ids.iter().enumerate() {
        let eb = &bank[eid as usize * expert_bytes..(eid as usize + 1) * expert_bytes];
        let xrow_h = &xf_h[s * k..s * k + k];
        let xrow_b = &xf_b[s * k..s * k + k];
        for r in 0..n {
            let rh = reference_row::<T>(eb, r, nblk, blk, tsz, xrow_h);
            let rb = reference_row::<T>(eb, r, nblk, blk, tsz, xrow_b);
            scale = scale.max(rh.abs()).max(rb.abs());
            let eh = (yh[s * n + r] - rh).abs();
            let ebb = (yb[s * n + r] - rb).abs();
            max_err_h = max_err_h.max(eh);
            max_err_b = max_err_b.max(ebb);
            let tol = 0.02 * scale.max(1.0);
            if eh > tol || ebb > tol {
                nbad += 1;
            }
        }
    }
    log.push_str(&format!(
        "{name:8} MoE E={e_cnt} nrows={nrows} n={n} k={k} nbad={nbad}/{} max_err_f16={max_err_h:.5} max_err_bf16={max_err_b:.5} scale={scale:.3}\n",
        nrows * n
    ));
    assert!(
        nbad == 0,
        "{name} IQ4_NL MoE mismatch: nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b}"
    );
}

#[test]
fn qmatvec_iq4nl_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");
    // Single block per row, production decode, wide-k FFN, partial last warp-row (17), vocab-sized out.
    let shapes: &[(usize, usize)] = &[
        (64, 32),
        (4096, 4096),
        (1024, 3072),
        (17, 4096),
        (4096, 256),
    ];
    // IQ4_NL: 18 B, 32 elems, d (f16) at byte 0. Small exact-f16 d keeps the +-127 codebook outputs
    // inside f16 range (same reasoning as the IQ4_XS gate -- shared KVALUES_IQ4NL LUT).
    for &(n, k) in shapes {
        check::<BlockIQ4nl, _>(
            &dev,
            &mut log,
            "IQ4_NL",
            RocmQuantType::IQ4_NL,
            n,
            k,
            32,
            18,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
            },
        );
    }
    // MoE: 8 experts, 16 routed slots, n=128 out, k=512 in (16 blocks of 32).
    moe_check::<BlockIQ4nl, _>(
        &dev,
        &mut log,
        "IQ4_NL",
        RocmQuantType::IQ4_NL,
        8,
        16,
        128,
        512,
        32,
        18,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
        },
    );
    eprintln!("{log}");
}
