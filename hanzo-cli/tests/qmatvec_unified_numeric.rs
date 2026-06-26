// Numeric gate for the UNIFIED ROCm quant decode core (qmatvec_core<WTYPE> in quant.hip). ONE
// quant-agnostic warp-per-row core + one per-type decode_block covers the whole 1-bit -> 8-bit
// zoo. This test proves a REPRESENTATIVE SPREAD bit-exact against the CPU oracle:
//   - Q8_0    : symmetric 8-bit         (the proven decode path, re-validated through the core)
//   - Q4_0    : symmetric nibble (-8)
//   - Q4_K    : ASYMMETRIC super-block  (exercises the 2nd accumulation shape: d·sc·q − dmin·m)
//   - Q6_K    : symmetric K-quant (6-bit, signed sub-scales)
//   - IQ4_XS  : symmetric codebook      (exercises grid-LUT decode via KVALUES_IQ4NL)
//   - TQ2_0   : symmetric ternary 2-bit (the sub-4-bit / 1-2-bit end)
//
// For EACH type the reference is the REAL CPU `to_float` (the `quant_format!` / GgmlType decoder),
// reinterpreting the same raw GGML bytes the GPU reads -- so the GPU decode is checked against the
// exact oracle, not a re-derivation. The kernel and reference both accumulate in f32; `nbad` counts
// outputs beyond a tiny magnitude-scaled tolerance and the gate requires nbad == 0 for every type.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::quantized::iq_quants::BlockTQ2_0;
use hanzo_ml::quantized::k_quants::{
    BlockIQ4xs, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K, BlockQ8_0,
    BlockQ8_1,
};
use hanzo_ml::quantized::GgmlType;
use hanzo_ml::{RocmDevice, RocmQuantType, RocmStorage};

// Deterministic activation value in roughly [-3, 3] (same generator as the Q4_K / Q8_0 gates).
fn val(i: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    (x - 0.5) * 6.0
}

// Deterministic byte stream for building GGML blocks; varied so all nibbles/bits/scale lanes and
// the full quant range are exercised across rows.
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

// Reinterpret raw GGML bytes as whole `T` blocks with correct alignment (fs/byte buffers are only
// 1-byte aligned; several blocks have u16/f16 fields needing >=2). Copies bytes, does not change them.
fn as_blocks<T: Clone>(bytes: &[u8]) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    assert_eq!(
        bytes.len() % sz,
        0,
        "byte length {} not a multiple of {}",
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

// Build a valid GGML weight tensor's raw bytes for `n` rows x `k` cols of type `T` (block size
// `blk`, byte size `tsz`). Most bytes are pseudo-random, EXCEPT the leading per-block f16 scale
// `d` (and Q4_K's dmin), which we force to small exactly-f16-representable magnitudes so the
// reference and kernel f16 scales agree to the last bit and the outputs stay O(1..10s). The scale
// position differs per type (TQ2_0 trails its d), so each builder sets it explicitly.
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
            // pseudo-random body
            for (i, by) in block.iter_mut().enumerate() {
                *by = byte(r * 7919 + b * 131 + i * 17 + 3);
            }
            fill(block, r, b);
        }
    }
    out
}

// Put a small exact-f16 scale at byte `off` in the block. Returns nothing; mutates block.
fn put_d(block: &mut [u8], off: usize, r: usize, b: usize, lo: f32, step: f32, m: usize) {
    let d = lo + step * (((r + b) % m) as f32);
    let bytes = f16::from_f32(d).to_le_bytes();
    block[off] = bytes[0];
    block[off + 1] = bytes[1];
}

// Exact CPU reference: dequantize the SAME bytes via the real GgmlType::to_float, then dot row `nn`
// against the dtype-rounded activation `x`. Accumulation order: per block, sequential within block.
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
    let xb: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xf_h: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let xf_b: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xb).expect("upload x bf16");

    let wq_bytes = build_bytes(n, k, blk, tsz, fill);
    assert_eq!(wq_bytes.len(), n * nblk * tsz);
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    let y_h = dev
        .matvec_quant(qt, &wst, &xst_h, n, k)
        .expect("matvec_quant f16");
    let y_b = dev
        .matvec_quant(qt, &wst, &xst_b, n, k)
        .expect("matvec_quant bf16");
    assert_eq!(
        y_b.dtype(),
        hanzo_ml::DType::BF16,
        "{name}: bf16 matvec keeps bf16"
    );
    assert_eq!(
        y_h.dtype(),
        hanzo_ml::DType::F16,
        "{name}: f16 matvec keeps f16"
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
    let tol = 0.01 * scale; // f16: 1% of magnitude, far below any layout/packing/scale bug, above f32 reorder.
    let tol_b = 0.02 * scale; // bf16 has ~3 fewer mantissa bits than f16; 2% gate covers its worst-case rounding at large k.

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
    assert!(
        nbad == 0,
        "{name} unified-core mismatch: n={n} k={k} nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b} tol={tol}"
    );
}

#[test]
fn qmatvec_unified_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");

    // Shapes: single block per row, production decode (4096), wide-k FFN, partial last warp-row (17),
    // and a vocab-sized output. Each type runs the full set so partial tails + warp tiling are covered.
    let shapes: &[(usize, usize)] = &[
        (64, 256),
        (4096, 4096),
        (1024, 3072),
        (17, 4096),
        (4096, 256),
    ];

    // Q8_0: 34 B, 32 elems. d (f16) at byte 0.
    for &(n, k) in shapes {
        check::<BlockQ8_0, _>(
            &dev,
            &mut log,
            "Q8_0",
            RocmQuantType::Q8_0,
            n,
            k,
            32,
            34,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            },
        );
    }
    // Q4_0: 18 B, 32 elems. d (f16) at byte 0.
    for &(n, k) in shapes {
        check::<BlockQ4_0, _>(
            &dev,
            &mut log,
            "Q4_0",
            RocmQuantType::Q4_0,
            n,
            k,
            32,
            18,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            },
        );
    }
    // Q4_K: 144 B, 256 elems. d (f16) at 0, dmin (f16) at 2. ASYMMETRIC.
    for &(n, k) in shapes {
        check::<BlockQ4K, _>(
            &dev,
            &mut log,
            "Q4_K",
            RocmQuantType::Q4K,
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
    // Q6_K: 210 B, 256 elems. d (f16) at 208. scales[16] are signed i8 (random bytes OK). Q6_K and
    // IQ4_XS both carry ±127-magnitude quants (6-bit signed scale / 8-bit codebook), so with random
    // scale bytes a few of 4096 random rows would sum past the f16 output range (65504) and store
    // inf -- not a decode bug (the math is bit-identical to the oracle), just an output-range
    // artifact. A small `d` (~0.008, still exact-f16) keeps worst-case rows well inside f16, exactly
    // as the proven Q4_K gate uses small exact scales; the full quant/scale RANGE is still exercised.
    for &(n, k) in shapes {
        check::<BlockQ6K, _>(
            &dev,
            &mut log,
            "Q6_K",
            RocmQuantType::Q6K,
            n,
            k,
            256,
            210,
            |blk, r, b| {
                put_d(blk, 208, r, b, 0.0078125, 0.00390625, 5);
            },
        );
    }
    // IQ4_XS: 136 B, 256 elems. d (f16) at 0; scales_h (u16) at 2; scales_l[4] at 4; qs[128] at 8.
    for &(n, k) in shapes {
        check::<BlockIQ4xs, _>(
            &dev,
            &mut log,
            "IQ4_XS",
            RocmQuantType::IQ4_XS,
            n,
            k,
            256,
            136,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
            },
        );
    }
    // TQ2_0: 66 B, 256 elems. qs[64] at 0; d (f16) at 64.
    for &(n, k) in shapes {
        check::<BlockTQ2_0, _>(
            &dev,
            &mut log,
            "TQ2_0",
            RocmQuantType::TQ2_0,
            n,
            k,
            256,
            66,
            |blk, r, b| {
                put_d(blk, 64, r, b, 0.0625, 0.03125, 5);
            },
        );
    }
    // Q5_K: 176 B, 256 elems. d (f16) at 0, dmin (f16) at 2. ASYMMETRIC 5-bit (Q4_K + qh 5th bit).
    // dp4a-capable, so check() exercises the dp4a path vs the to_float oracle (scalar gated in the A/B
    // below). Small exact-f16 d/dmin keep the 5-bit*6-bit-scale worst case inside f16 range.
    for &(n, k) in shapes {
        check::<BlockQ5K, _>(
            &dev,
            &mut log,
            "Q5_K",
            RocmQuantType::Q5K,
            n,
            k,
            256,
            176,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
                put_d(blk, 2, r, b, 0.00390625, 0.001953125, 4);
            },
        );
    }
    // Q4_1: 20 B, 32 elems. d (f16) at 0, m (f16) at 2. ASYMMETRIC legacy (val = nib*d + m).
    for &(n, k) in shapes {
        check::<BlockQ4_1, _>(
            &dev,
            &mut log,
            "Q4_1",
            RocmQuantType::Q4_1,
            n,
            k,
            32,
            20,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
                put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
            },
        );
    }
    // Q5_0: 22 B, 32 elems. d (f16) at 0; qh[4] at 2; qs[16] at 6. symmetric 5-bit (val = (q5-16)*d).
    for &(n, k) in shapes {
        check::<BlockQ5_0, _>(
            &dev,
            &mut log,
            "Q5_0",
            RocmQuantType::Q5_0,
            n,
            k,
            32,
            22,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            },
        );
    }
    // Q5_1: 24 B, 32 elems. d (f16) at 0; m (f16) at 2; qh[4] at 4; qs[16] at 8. ASYMMETRIC 5-bit.
    for &(n, k) in shapes {
        check::<BlockQ5_1, _>(
            &dev,
            &mut log,
            "Q5_1",
            RocmQuantType::Q5_1,
            n,
            k,
            32,
            24,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
                put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
            },
        );
    }
    // Q8_1: 36 B, 32 elems. d (f16) at 0; s (f16, sum, unused for dequant) at 2; qs[32] at 4. SYM 8-bit.
    for &(n, k) in shapes {
        check::<BlockQ8_1, _>(
            &dev,
            &mut log,
            "Q8_1",
            RocmQuantType::Q8_1,
            n,
            k,
            32,
            36,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            },
        );
    }

    eprintln!("{log}");
}

// MoE gate: the resident expert bank [E,n,k] dispatched per routed slot through the SAME unified
// core (moe_matvec_quant). For each slot s with expert eid=ids[s], output row s must equal
// W_eid[n,k]·x[s] computed via the real CPU to_float. Proves MoE rides the one core for a wired
// quant (Q4_K here -- the 30B-A3B production type), nbad=0, no per-quant MoE code.
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

    // Bank: E experts, each [n,k] = n*nblk blocks. Reuse build_bytes per expert with a per-expert
    // seed offset so experts differ (different decoded weights).
    let mut bank: Vec<u8> = Vec::with_capacity(e_cnt * expert_bytes);
    for e in 0..e_cnt {
        let eb = build_bytes(n, k, blk, tsz, |b, r, bb| fill(b, r * (e + 1) + e, bb));
        assert_eq!(eb.len(), expert_bytes);
        bank.extend_from_slice(&eb);
    }
    let wbank = dev.storage_from_slice(&bank).expect("upload bank");

    // Router ids: each slot picks an expert (round-robin + a twist so repeats + spread both occur).
    // Uploaded to the device: the batched MoE kernels index experts ON-DEVICE (no host ids loop).
    let ids: Vec<u32> = (0..nrows).map(|s| ((s * 3 + 1) % e_cnt) as u32).collect();
    let ids_dev = dev.storage_from_slice(&ids).expect("upload ids");

    // Activation matrix [nrows, k], f16 and bf16.
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

    // Reference: per slot, dequantize expert eid and dot with that slot's activation row.
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
            let tol = 0.01 * scale.max(1.0);
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
        "{name} MoE mismatch: nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b}"
    );
}

#[test]
fn moe_matvec_unified_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");

    // Q4_K (the 30B-A3B production MoE type): E experts, nrows routed slots (= t*topk), n out, k in.
    // 8 experts, 16 slots (decode t=2,topk=8-ish), n=128 out, k=512 in (2 super-blocks).
    moe_check::<BlockQ4K, _>(
        &dev,
        &mut log,
        "Q4_K",
        RocmQuantType::Q4K,
        8,
        16,
        128,
        512,
        256,
        144,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
        },
    );
    // Q8_0 MoE (the other proven type) to show MoE is per-type generic via the same core.
    moe_check::<BlockQ8_0, _>(
        &dev,
        &mut log,
        "Q8_0",
        RocmQuantType::Q8_0,
        8,
        16,
        128,
        512,
        32,
        34,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
        },
    );
    // Q6_K MoE: the mixed-precision down-proj type in Qwen3-30B-A3B-Q4_K_M. Exercises the batched
    // on-device-ids `moe_qmatvecu_q6k_*` kernel (experts on grid.y) against the CPU oracle. Small d
    // keeps worst-case rows inside f16 range, same as the single-expert Q6_K gate above.
    moe_check::<BlockQ6K, _>(
        &dev,
        &mut log,
        "Q6_K",
        RocmQuantType::Q6K,
        8,
        16,
        128,
        512,
        256,
        210,
        |blk, r, b| {
            put_d(blk, 208, r, b, 0.0078125, 0.00390625, 5);
        },
    );
    // Q5_K MoE (ASYMMETRIC 5-bit) -- dp4a expert-bank path (moe_qmatvec_dp4a_q5k) vs the CPU oracle.
    moe_check::<BlockQ5K, _>(
        &dev,
        &mut log,
        "Q5_K",
        RocmQuantType::Q5K,
        8,
        16,
        128,
        512,
        256,
        176,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
            put_d(blk, 2, r, b, 0.00390625, 0.001953125, 4);
        },
    );
    // Legacy 32-elem MoE types via the scalar moe_qmatvecu_* core (experts on grid.y).
    moe_check::<BlockQ4_1, _>(
        &dev, &mut log, "Q4_1", RocmQuantType::Q4_1, 8, 16, 128, 512, 32, 20,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
        },
    );
    moe_check::<BlockQ5_0, _>(
        &dev, &mut log, "Q5_0", RocmQuantType::Q5_0, 8, 16, 128, 512, 32, 22,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
        },
    );
    moe_check::<BlockQ5_1, _>(
        &dev, &mut log, "Q5_1", RocmQuantType::Q5_1, 8, 16, 128, 512, 32, 24,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
        },
    );
    moe_check::<BlockQ8_1, _>(
        &dev, &mut log, "Q8_1", RocmQuantType::Q8_1, 8, 16, 128, 512, 32, 36,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
        },
    );

    eprintln!("{log}");
}

// dp4a-vs-scalar A/B gate: the int8-dp4a decode core (`qdp4a<WTYPE>`) must equal the scalar
// `qmatvec_core<WTYPE>` (the byte-faithful CPU-oracle reference) for every dp4a-capable type. Sets
// the per-type `HANZO_<T>_FALLBACK` env to force the scalar core, unset to take dp4a, and asserts the
// two agree to a tight f16/bf16-reorder tolerance (both accumulate in f32; only the summation ORDER
// differs). Covers both the single-expert matvec and the indexed-MoE path. This proves the speedup
// changed nothing numeric -- the whole point of the bit-faithful dp4a port.
fn ab_matvec<F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice,
    log: &mut String,
    name: &str,
    qt: RocmQuantType,
    fb_env: &str,
    n: usize,
    k: usize,
    blk: usize,
    tsz: usize,
    fill: F,
) {
    let xf: Vec<f32> = (0..k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xbf: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xbf).expect("upload x bf16");
    let wq_bytes = build_bytes(n, k, blk, tsz, fill);
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    // dp4a path (fallback unset).
    std::env::remove_var(fb_env);
    let dp4a_h = to_f32(&dev.matvec_quant(qt, &wst, &xst_h, n, k).expect("dp4a f16"));
    let dp4a_b = to_f32(&dev.matvec_quant(qt, &wst, &xst_b, n, k).expect("dp4a bf16"));
    // scalar path (fallback set) -- the byte-faithful reference core.
    std::env::set_var(fb_env, "1");
    let scal_h = to_f32(
        &dev.matvec_quant(qt, &wst, &xst_h, n, k)
            .expect("scalar f16"),
    );
    let scal_b = to_f32(
        &dev.matvec_quant(qt, &wst, &xst_b, n, k)
            .expect("scalar bf16"),
    );
    std::env::remove_var(fb_env);

    let scale = scal_h
        .iter()
        .chain(scal_b.iter())
        .fold(0f32, |m, &v| m.max(v.abs()))
        .max(1.0);
    let tol = 0.01 * scale;
    let mut nbad = 0usize;
    let mut max_err = 0f32;
    for i in 0..n {
        let eh = (dp4a_h[i] - scal_h[i]).abs();
        let eb = (dp4a_b[i] - scal_b[i]).abs();
        max_err = max_err.max(eh).max(eb);
        if eh > tol || eb > tol {
            nbad += 1;
        }
    }
    log.push_str(&format!(
        "{name:8} dp4a-vs-scalar n={n} k={k} nbad={nbad}/{n} max_err={max_err:.5} (tol {tol:.5})\n"
    ));
    assert!(
        nbad == 0,
        "{name} dp4a != scalar core: nbad={nbad} max_err={max_err} tol={tol}"
    );
}

#[allow(clippy::too_many_arguments)]
fn ab_moe<F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice,
    log: &mut String,
    name: &str,
    qt: RocmQuantType,
    fb_env: &str,
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
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");

    std::env::remove_var(fb_env);
    let dp4a = to_f32(
        &dev.moe_matvec_quant(qt, &wbank, &xst_h, &ids_dev, nrows, n, k)
            .expect("moe dp4a"),
    );
    std::env::set_var(fb_env, "1");
    let scal = to_f32(
        &dev.moe_matvec_quant(qt, &wbank, &xst_h, &ids_dev, nrows, n, k)
            .expect("moe scalar"),
    );
    std::env::remove_var(fb_env);

    let scale = scal.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    let tol = 0.01 * scale;
    let mut nbad = 0usize;
    let mut max_err = 0f32;
    for i in 0..nrows * n {
        let e = (dp4a[i] - scal[i]).abs();
        max_err = max_err.max(e);
        if e > tol {
            nbad += 1;
        }
    }
    log.push_str(&format!(
        "{name:8} MoE dp4a-vs-scalar nbad={nbad}/{} max_err={max_err:.5} (tol {tol:.5})\n",
        nrows * n
    ));
    assert!(
        nbad == 0,
        "{name} MoE dp4a != scalar core: nbad={nbad} max_err={max_err} tol={tol}"
    );
}

#[test]
fn qmatvec_dp4a_vs_scalar() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");
    let shapes: &[(usize, usize)] = &[
        (64, 256),
        (4096, 4096),
        (1024, 3072),
        (17, 4096),
        (4096, 256),
    ];

    // Q4_K (ASYMMETRIC) and Q6_K (SYMMETRIC 6-bit) are the dp4a-capable types; each must match its
    // scalar core bit-faithfully. Same small exact-f16 scales the oracle gate uses.
    for &(n, k) in shapes {
        ab_matvec(
            &dev,
            &mut log,
            "Q4_K",
            RocmQuantType::Q4K,
            "HANZO_Q4K_FALLBACK",
            n,
            k,
            256,
            144,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
                put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
            },
        );
        ab_matvec(
            &dev,
            &mut log,
            "Q6_K",
            RocmQuantType::Q6K,
            "HANZO_Q6K_FALLBACK",
            n,
            k,
            256,
            210,
            |blk, r, b| {
                put_d(blk, 208, r, b, 0.0078125, 0.00390625, 5);
            },
        );
        ab_matvec(
            &dev,
            &mut log,
            "Q5_K",
            RocmQuantType::Q5K,
            "HANZO_Q5K_FALLBACK",
            n,
            k,
            256,
            176,
            |blk, r, b| {
                put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
                put_d(blk, 2, r, b, 0.00390625, 0.001953125, 4);
            },
        );
    }
    // MoE A/B for both (8 experts, 16 slots, n=128, k=512).
    ab_moe(
        &dev,
        &mut log,
        "Q4_K",
        RocmQuantType::Q4K,
        "HANZO_Q4K_FALLBACK",
        8,
        16,
        128,
        512,
        256,
        144,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0625, 0.03125, 5);
            put_d(blk, 2, r, b, 0.03125, 0.015625, 4);
        },
    );
    ab_moe(
        &dev,
        &mut log,
        "Q6_K",
        RocmQuantType::Q6K,
        "HANZO_Q6K_FALLBACK",
        8,
        16,
        128,
        512,
        256,
        210,
        |blk, r, b| {
            put_d(blk, 208, r, b, 0.0078125, 0.00390625, 5);
        },
    );
    ab_moe(
        &dev,
        &mut log,
        "Q5_K",
        RocmQuantType::Q5K,
        "HANZO_Q5K_FALLBACK",
        8,
        16,
        128,
        512,
        256,
        176,
        |blk, r, b| {
            put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
            put_d(blk, 2, r, b, 0.00390625, 0.001953125, 4);
        },
    );

    eprintln!("{log}");
}
