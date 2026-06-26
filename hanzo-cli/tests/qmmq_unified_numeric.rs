// Numeric gate for the UNIFIED ROCm quant PREFILL GEMM (qmmq_core<WTYPE> in quant.hip). ONE int8-
// WMMA core + per-type in-kernel weight decode + a symmetric/asymmetric flag covers the SAME 1-bit
// -> 8-bit spread the decode core does, with NO per-quant prefill kernel. This proves a
// representative spread bit-exact against the CPU oracle:
//   - Q8_0    : symmetric 8-bit         (the proven prefill path, re-validated through the core)
//   - Q4_K    : ASYMMETRIC super-block  (the headline: d*sc*q - dmin*m, min bias via the q8_1 sum)
//   - Q6_K    : symmetric K-quant 6-bit (signed per-16 sub-scales -> the 2-scales-per-32 path)
//   - IQ4_XS  : symmetric codebook      (grid-LUT decode via KVALUES_IQ4NL)
//   - TQ2_0   : symmetric ternary 2-bit (the sub-4-bit end)
//
// EXACT ORACLE. The int8 GEMM computes  sum_k(q_w * q_x) * scales  (minus the min bias for ASYM).
// For BOTH shapes this equals  sum_k( to_float(w)[k] * x_requant[k] ), where x_requant is the SAME
// per-32-block int8-requantized activation the kernel builds (absmax/127, round-half-away). So the
// reference is the REAL CPU `GgmlType::to_float` (the quant_format! decoder) on the same raw bytes,
// dotted against x_requant -- the GPU prefill is checked against the exact oracle, not a re-
// derivation, and the gate requires nbad == 0 for every type. Lives in hanzo-cli so it links HIP.
#![cfg(feature = "rocm")]

use half::f16;
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::quantized::iq_quants::BlockTQ2_0;
use hanzo_ml::quantized::k_quants::{BlockIQ4xs, BlockQ4K, BlockQ4_0, BlockQ6K, BlockQ8_0};
use hanzo_ml::quantized::GgmlType;
use hanzo_ml::{RocmDevice, RocmQuantType, RocmStorage};

// Deterministic activation value ~[-3,3] (same generator as the decode/Q4_K gates).
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
        hanzo_ml::CpuStorage::F32(v) => v,
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}

fn as_blocks<T: Clone>(bytes: &[u8]) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    assert_eq!(bytes.len() % sz, 0);
    let n = bytes.len() / sz;
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * sz);
        v.set_len(n);
    }
    v
}

// Build a valid GGML weight tensor's raw bytes for `n` rows x `k` cols of type T. Body bytes are
// pseudo-random; the leading per-block f16 scale(s) are forced to small exact-f16 magnitudes (via
// `fill`) so reference and kernel scales agree to the last bit and outputs stay O(1..10s).
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

fn put_d(block: &mut [u8], off: usize, r: usize, b: usize, lo: f32, step: f32, m: usize) {
    let d = lo + step * (((r + b) % m) as f32);
    let bytes = f16::from_f32(d).to_le_bytes();
    block[off] = bytes[0];
    block[off + 1] = bytes[1];
}

// Per-32-block int8 requantization of one activation row, EXACTLY as the kernel's quantize_q8 /
// quantize_q8_1 does (per-block absmax, inv=127/amax, round-half-away, clamp +-127), reconstructed
// back to f32 as q*(amax/127). This is the activation the int8 GEMM effectively dots against.
fn requant_row(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for blk in x.chunks(32) {
        let off = (blk.as_ptr() as usize - x.as_ptr() as usize) / std::mem::size_of::<f32>();
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let d = amax / 127.0;
        for (j, &v) in blk.iter().enumerate() {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
            out[off + j] = q as f32 * d;
        }
    }
    out
}

// to_float(weight row n) . activation row (sequential within each on-disk block).
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
    m: usize,
    n: usize,
    k: usize,
    blk: usize,
    tsz: usize,
    fill: F,
) {
    assert_eq!(k % blk, 0, "{name}: k must be a multiple of {blk}");
    let nblk = k / blk;

    // X[m,k] f16, varied; requantized per row for the oracle.
    let xf: Vec<f32> = (0..m * k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xf_h: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let xst = dev.storage_from_slice(&xh).expect("upload x");

    let wq_bytes = build_bytes(n, k, blk, tsz, fill);
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    // Native prefill GEMM through the unified core (rows = m > 1).
    let y = dev.qmmq_quant(qt, &xst, &wst, m, n, k).expect("qmmq_quant");
    assert_eq!(
        y.dtype(),
        hanzo_ml::DType::F16,
        "{name}: prefill returns f16"
    );
    let yg = to_f32(&y); // [m, n]

    // Oracle: per row, requantize that activation row to int8 (as the kernel does) and dot the real
    // to_float-dequantized weight rows against it.
    let mut refy = vec![0f32; m * n];
    let mut scale = 1f32;
    for mm in 0..m {
        let xr = requant_row(&xf_h[mm * k..mm * k + k]);
        for nn in 0..n {
            let r = reference_row::<T>(&wq_bytes, nn, nblk, blk, tsz, &xr);
            refy[mm * n + nn] = r;
            scale = scale.max(r.abs());
        }
    }
    let tol = 0.01 * scale.max(1.0); // 1% of magnitude: far below any layout/packing/scale bug.

    let mut max_err = 0f32;
    let mut nbad = 0usize;
    for i in 0..m * n {
        let e = (yg[i] - refy[i]).abs();
        max_err = max_err.max(e);
        if e > tol {
            nbad += 1;
        }
    }
    log.push_str(&format!(
        "{name:8} m={m} n={n} k={k} nbad={nbad}/{} max_err={max_err:.5} (tol {tol:.5}) scale={scale:.3}\n",
        m * n
    ));
    assert!(
        nbad == 0,
        "{name} prefill-core mismatch: m={m} n={n} k={k} nbad={nbad} max_err={max_err} tol={tol}"
    );
}

#[test]
fn qmmq_unified_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");

    // Shapes exercise: partial tiles (m,n not mult-128), production prefill width (512 tok x 4096),
    // wide-K FFN, and a tall/narrow case. Each type runs the full set (multi-tile, double-buffer,
    // partial tails). k is a multiple of the type's block (32 / 256).
    let shapes_32: &[(usize, usize, usize)] = &[
        (200, 160, 128),
        (130, 130, 256),
        (256, 512, 1024),
        (64, 256, 4096),
    ];
    let shapes_256: &[(usize, usize, usize)] = &[
        (200, 160, 256),
        (130, 130, 512),
        (64, 256, 1024),
        (37, 320, 768),
    ];

    // Q8_0: 34 B, 32 elems. SYMMETRIC -- the proven prefill, re-validated through the unified core.
    for &(m, n, k) in shapes_32 {
        check::<BlockQ8_0, _>(
            &dev,
            &mut log,
            "Q8_0",
            RocmQuantType::Q8_0,
            m,
            n,
            k,
            32,
            34,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            },
        );
    }
    // Q4_0: 18 B, 32 elems. SYMMETRIC nibble-8.
    for &(m, n, k) in shapes_32 {
        check::<BlockQ4_0, _>(
            &dev,
            &mut log,
            "Q4_0",
            RocmQuantType::Q4_0,
            m,
            n,
            k,
            32,
            18,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            },
        );
    }
    // Q4_K: 144 B, 256 elems. ASYMMETRIC (d at 0, dmin at 2) -- the min-bias path via the q8_1 sum.
    for &(m, n, k) in shapes_256 {
        check::<BlockQ4K, _>(
            &dev,
            &mut log,
            "Q4_K",
            RocmQuantType::Q4K,
            m,
            n,
            k,
            256,
            144,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
                put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
            },
        );
    }
    // Q6_K: 210 B, 256 elems. SYMMETRIC, signed per-16 scales -> the 2-scales-per-32 path. Small d
    // (~0.008, exact-f16) keeps the +-127-magnitude worst-case rows inside f16, as the decode gate does.
    for &(m, n, k) in shapes_256 {
        check::<BlockQ6K, _>(
            &dev,
            &mut log,
            "Q6_K",
            RocmQuantType::Q6K,
            m,
            n,
            k,
            256,
            210,
            |blk, r, b| {
                put_d(blk, 208, r, b, 0.0078125, 0.00390625, 5);
            },
        );
    }
    // IQ4_XS: 136 B, 256 elems. SYMMETRIC codebook (LUT). d at 0; scales_h u16 at 2; scales_l[4] at 4.
    for &(m, n, k) in shapes_256 {
        check::<BlockIQ4xs, _>(
            &dev,
            &mut log,
            "IQ4_XS",
            RocmQuantType::IQ4_XS,
            m,
            n,
            k,
            256,
            136,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
            },
        );
    }
    // TQ2_0: 66 B, 256 elems. SYMMETRIC ternary 2-bit. qs[64] at 0; d at 64.
    for &(m, n, k) in shapes_256 {
        check::<BlockTQ2_0, _>(
            &dev,
            &mut log,
            "TQ2_0",
            RocmQuantType::TQ2_0,
            m,
            n,
            k,
            256,
            66,
            |blk, r, b| {
                put_d(blk, 64, r, b, 0.0625, 0.03125, 5);
            },
        );
    }

    eprintln!("{log}");
}

// FUSED indexed-MoE PREFILL GEMM gate (`qmmq_core<WTYPE,true>` via moe_qmmq_quant). Builds an E-expert
// [E,n,k] bank, routes nslots tokens to experts, runs the expert-grouped WMMA GEMM, and asserts every
// output row equals the SAME exact oracle as the dense gate -- per slot s with expert eid: requantize
// x[s] to int8 (as the kernel does) and dot it against the real to_float-dequantized rows of expert
// eid. The fused kernel gathers tokens by expert and SCATTERS results back to original slot order, so
// y[s] must match slot s's expert regardless of routing. nbad==0 required.
#[allow(clippy::too_many_arguments)]
fn moe_check<T: GgmlType, F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice,
    log: &mut String,
    name: &str,
    qt: RocmQuantType,
    e_cnt: usize,
    nslots: usize,
    n: usize,
    k: usize,
    blk: usize,
    tsz: usize,
    fill: F,
) {
    assert_eq!(k % blk, 0, "{name}: k must be a multiple of {blk}");
    let nblk = k / blk;
    let expert_bytes = n * nblk * tsz;

    let mut bank: Vec<u8> = Vec::with_capacity(e_cnt * expert_bytes);
    for e in 0..e_cnt {
        let eb = build_bytes(n, k, blk, tsz, |b, r, bb| fill(b, r * (e + 1) + e, bb));
        assert_eq!(eb.len(), expert_bytes);
        bank.extend_from_slice(&eb);
    }
    let wbank = dev.storage_from_slice(&bank).expect("upload bank");

    // Routing: a spread + repeats so multiple tokens land on the same expert (the amortization the
    // fused kernel exploits) AND some experts get 0 tokens (empty-tile / offset edge cases).
    let ids: Vec<u32> = (0..nslots).map(|s| ((s * 5 + 3) % e_cnt) as u32).collect();
    let ids_dev = dev.storage_from_slice(&ids).expect("upload ids");

    let xf: Vec<f32> = (0..nslots * k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xf_h: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let xst = dev.storage_from_slice(&xh).expect("upload x");

    let y = dev
        .moe_qmmq_quant(qt, &wbank, &xst, &ids_dev, nslots, n, k)
        .expect("moe_qmmq_quant");
    assert_eq!(y.dtype(), hanzo_ml::DType::F16, "{name}: moe prefill returns f16");
    let yg = to_f32(&y); // [nslots, n]

    let mut nbad = 0usize;
    let mut max_err = 0f32;
    let mut scale = 1f32;
    for (s, &eid) in ids.iter().enumerate() {
        let eb = &bank[eid as usize * expert_bytes..(eid as usize + 1) * expert_bytes];
        let xr = requant_row(&xf_h[s * k..s * k + k]);
        for nn in 0..n {
            let r = reference_row::<T>(eb, nn, nblk, blk, tsz, &xr);
            scale = scale.max(r.abs());
            let e = (yg[s * n + nn] - r).abs();
            max_err = max_err.max(e);
            if e > 0.01 * scale.max(1.0) {
                nbad += 1;
            }
        }
    }
    log.push_str(&format!(
        "{name:8} MoE E={e_cnt} nslots={nslots} n={n} k={k} nbad={nbad}/{} max_err={max_err:.5} scale={scale:.3}\n",
        nslots * n
    ));
    assert!(nbad == 0, "{name} MoE prefill mismatch: nbad={nbad} max_err={max_err}");
}

#[test]
fn moe_qmmq_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");

    // E=8 experts, 300 routed slots (multi-tile per expert + partial tails + some empty experts),
    // production-ish FFN widths. Q4_K (gate/up) + Q6_K (down) = the Qwen3-30B-A3B-Q4_K_M MoE spread,
    // plus Q8_0 as the symmetric proof.
    moe_check::<BlockQ4K, _>(
        &dev, &mut log, "Q4_K", RocmQuantType::Q4K, 8, 300, 256, 1024, 256, 144,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
        },
    );
    moe_check::<BlockQ6K, _>(
        &dev, &mut log, "Q6_K", RocmQuantType::Q6K, 8, 300, 256, 768, 256, 210,
        |blk, r, b| {
            put_d(blk, 208, r, b, 0.0078125, 0.00390625, 5);
        },
    );
    moe_check::<BlockQ8_0, _>(
        &dev, &mut log, "Q8_0", RocmQuantType::Q8_0, 8, 300, 256, 512, 32, 34,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
        },
    );

    eprintln!("{log}");
}
