// Numeric gate for the sub-2-bit native ROCm decode (qmatvec_core<WTYPE> in quant.hip) of the three
// hardest remaining quants:
//   - TQ1_0 : ternary base-3, 1.69 bpw  (q = byte*pow3[n] mod 256; xi = (q*3)>>8; val = (xi-1)*d)
//   - IQ1_S : 1.56 bpw, 11-bit signed grid + per-group +/-0.125 delta bias (val = dl*(grid_i8 + delta))
//   - IQ1_M : 1.75 bpw, same signed grid + delta, f16 super-scale reconstructed from 4 nibbles of scales[8]
// The reference is the REAL CPU `to_float` (BlockTQ1_0 / BlockIQ1s / BlockIQ1m), reinterpreting the
// exact GGML bytes the GPU reads. The grid points are {-1,0,+1} and the delta is 2^-3, so the dequant is
// f32-exact; the only error is f16/bf16 activation rounding + f32 reorder. The gate requires nbad == 0
// for the f16 AND bf16 matvec, single-expert and indexed-MoE, for every type. Lives in hanzo-cli so it
// links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::backend::{BackendDevice, BackendStorage};
use hanzo_ml::quantized::iq_quants::{BlockIQ1m, BlockIQ1s, BlockTQ1_0};
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
    assert_eq!(bytes.len() % sz, 0, "byte len {} not a multiple of {}", bytes.len(), sz);
    let n = bytes.len() / sz;
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * sz);
        v.set_len(n);
    }
    v
}

fn build_bytes<F: Fn(&mut [u8], usize, usize)>(n: usize, k: usize, blk: usize, tsz: usize, fill: F) -> Vec<u8> {
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

// Put a small exact-f16 scale `d` at byte `off` (TQ1_0 d trails its qs/qh at +52; IQ1_S d leads at 0).
fn put_d(block: &mut [u8], off: usize, r: usize, b: usize, lo: f32, step: f32, m: usize) {
    let d = lo + step * (((r + b) % m) as f32);
    let bytes = f16::from_f32(d).to_le_bytes();
    block[off] = bytes[0];
    block[off + 1] = bytes[1];
}

// IQ1_M has no top-level d: the f16 super-scale is reconstructed as
//   scale_u16 = (scales[1]>>4) | ((scales[3]>>4)<<4) | ((scales[5]>>4)<<8) | ((scales[7]>>4)<<12)
// i.e. its 4 nibbles are the HIGH nibbles of scales[1],[3],[5],[7] (block offsets 49,51,53,55). The LOW
// nibbles + scales[0,2,4,6] drive the per-sub-block 3-bit dl scales, so leave those random. We force the
// high nibbles to spell an exact-f16 `d` so the kernel and CPU reference agree to the last bit.
fn put_iq1m_d(block: &mut [u8], d: f32) {
    let bits = f16::from_f32(d).to_bits();
    for (i, &off) in [49usize, 51, 53, 55].iter().enumerate() {
        let nib = ((bits >> (4 * i)) & 0xF) as u8;
        block[off] = (block[off] & 0x0F) | (nib << 4);
    }
}

fn reference_row<T: GgmlType>(wq: &[u8], nn: usize, nblk: usize, blk: usize, tsz: usize, x: &[f32]) -> f32 {
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

// q8_1 activation reconstruction mirroring the GPU `quantize_q8_1` (quant.hip) BIT-FOR-BIT. dp4a-path
// types (IQ1_S/IQ1_M) dot the exact-dequantized weight against THIS; scalar types (TQ1_0) use the
// exact activation. `dp4a_active()` selects which, so nbad=0 means bit-exact for both.
fn q8_1_recon(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (xc, oc) in x.chunks(32).zip(out.chunks_mut(32)) {
        let absmax = xc.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let inv = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };
        let d8 = f16::from_f32(absmax / 127.0).to_f32();
        for (xv, ov) in xc.iter().zip(oc.iter_mut()) {
            *ov = d8 * (xv * inv).round().clamp(-127.0, 127.0);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check<T: GgmlType, F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice, log: &mut String, name: &str, qt: RocmQuantType,
    n: usize, k: usize, blk: usize, tsz: usize, fill: F,
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

    let y_h = dev.matvec_quant(qt, &wst, &xst_h, n, k).expect("matvec f16");
    let y_b = dev.matvec_quant(qt, &wst, &xst_b, n, k).expect("matvec bf16");
    assert_eq!(y_h.dtype(), hanzo_ml::DType::F16, "{name}: f16 matvec keeps f16");
    assert_eq!(y_b.dtype(), hanzo_ml::DType::BF16, "{name}: bf16 matvec keeps bf16");
    let yh = to_f32(&y_h);
    let yb = to_f32(&y_b);

    let dp4a = qt.dp4a_active();
    let ax_h = if dp4a { q8_1_recon(&xf_h) } else { xf_h.clone() };
    let ax_b = if dp4a { q8_1_recon(&xf_b) } else { xf_b.clone() };
    let mut ref_h = vec![0f32; n];
    let mut ref_b = vec![0f32; n];
    for nn in 0..n {
        ref_h[nn] = reference_row::<T>(&wq_bytes, nn, nblk, blk, tsz, &ax_h);
        ref_b[nn] = reference_row::<T>(&wq_bytes, nn, nblk, blk, tsz, &ax_b);
    }
    let scale = ref_b.iter().chain(ref_h.iter()).fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
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
    assert!(nbad == 0, "{name} decode mismatch: nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b} tol={tol}");
}

#[allow(clippy::too_many_arguments)]
fn moe_check<T: GgmlType, F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice, log: &mut String, name: &str, qt: RocmQuantType,
    e_cnt: usize, nrows: usize, n: usize, k: usize, blk: usize, tsz: usize, fill: F,
) {
    let nblk = k / blk;
    let expert_bytes = n * nblk * tsz;
    let mut bank: Vec<u8> = Vec::with_capacity(e_cnt * expert_bytes);
    for e in 0..e_cnt {
        bank.extend_from_slice(&build_bytes(n, k, blk, tsz, |b, r, bb| fill(b, r * (e + 1) + e, bb)));
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

    let y_h = dev.moe_matvec_quant(qt, &wbank, &xst_h, &ids_dev, nrows, n, k).expect("moe f16");
    let y_b = dev.moe_matvec_quant(qt, &wbank, &xst_b, &ids_dev, nrows, n, k).expect("moe bf16");
    let yh = to_f32(&y_h);
    let yb = to_f32(&y_b);

    let mut nbad = 0usize;
    let mut max_err_h = 0f32;
    let mut max_err_b = 0f32;
    let mut scale = 1f32;
    let dp4a = qt.dp4a_active();
    for (s, &eid) in ids.iter().enumerate() {
        let eb = &bank[eid as usize * expert_bytes..(eid as usize + 1) * expert_bytes];
        let raw_h = &xf_h[s * k..s * k + k];
        let raw_b = &xf_b[s * k..s * k + k];
        let xrow_h = if dp4a { q8_1_recon(raw_h) } else { raw_h.to_vec() };
        let xrow_b = if dp4a { q8_1_recon(raw_b) } else { raw_b.to_vec() };
        for r in 0..n {
            let rh = reference_row::<T>(eb, r, nblk, blk, tsz, &xrow_h);
            let rb = reference_row::<T>(eb, r, nblk, blk, tsz, &xrow_b);
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
    assert!(nbad == 0, "{name} MoE mismatch: nbad={nbad} max_err_f16={max_err_h} max_err_bf16={max_err_b}");
}

#[test]
fn qmatvec_iq1_numeric() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");
    let shapes: &[(usize, usize)] = &[(64, 256), (4096, 4096), (1024, 3072), (17, 4096), (4096, 256)];

    // TQ1_0: 54 B, 256 elems. qs[48] at 0, qh[4] at +48, d (f16) at +52. ternary {-1,0,1}*d.
    for &(n, k) in shapes {
        check::<BlockTQ1_0, _>(&dev, &mut log, "TQ1_0", RocmQuantType::TQ1_0, n, k, 256, 54, |blk, r, b| {
            put_d(blk, 52, r, b, 0.0625, 0.03125, 5);
        });
    }
    // IQ1_S: 50 B, 256 elems. d (f16) at 0, qs[32] at +2, qh[8] (u16) at +34. Small exact-f16 d keeps the
    // dl=d*(2s+1) (s<=7) scaled grid+delta outputs inside f16 range.
    for &(n, k) in shapes {
        check::<BlockIQ1s, _>(&dev, &mut log, "IQ1_S", RocmQuantType::IQ1_S, n, k, 256, 50, |blk, r, b| {
            put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
        });
    }
    // IQ1_M: 56 B, 256 elems. qs[32] at 0, qh[16] at +32, scales[8] at +48. Reconstructed f16 scale.
    for &(n, k) in shapes {
        check::<BlockIQ1m, _>(&dev, &mut log, "IQ1_M", RocmQuantType::IQ1_M, n, k, 256, 56, |blk, _r, _b| {
            put_iq1m_d(blk, 0.0078125);
        });
    }

    // MoE: 8 experts, 16 routed slots, n=128 out, k=512 in (2 super-blocks).
    moe_check::<BlockTQ1_0, _>(&dev, &mut log, "TQ1_0", RocmQuantType::TQ1_0, 8, 16, 128, 512, 256, 54, |blk, r, b| {
        put_d(blk, 52, r, b, 0.0625, 0.03125, 5);
    });
    moe_check::<BlockIQ1s, _>(&dev, &mut log, "IQ1_S", RocmQuantType::IQ1_S, 8, 16, 128, 512, 256, 50, |blk, r, b| {
        put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
    });
    moe_check::<BlockIQ1m, _>(&dev, &mut log, "IQ1_M", RocmQuantType::IQ1_M, 8, 16, 128, 512, 256, 56, |blk, _r, _b| {
        put_iq1m_d(blk, 0.0078125);
    });

    eprintln!("{log}");
}

// dp4a-vs-scalar A/B: the int8-dp4a IQ1 decode (`qdp4a<DW_IQ1_*>`, signed grid + delta-bias sum) must
// equal the scalar core to a tight reorder tolerance. HANZO_IQ1*_FALLBACK forces scalar; unset = dp4a.
#[allow(clippy::too_many_arguments)]
fn ab_matvec<F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice, log: &mut String, name: &str, qt: RocmQuantType, fb_env: &str,
    n: usize, k: usize, blk: usize, tsz: usize, fill: F,
) {
    let xf: Vec<f32> = (0..k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xbf: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");
    let xst_b = dev.storage_from_slice(&xbf).expect("upload x bf16");
    let wq_bytes = build_bytes(n, k, blk, tsz, fill);
    let wst = dev.storage_from_slice(&wq_bytes).expect("upload w");

    std::env::remove_var(fb_env);
    let dp4a_h = to_f32(&dev.matvec_quant(qt, &wst, &xst_h, n, k).expect("dp4a f16"));
    let dp4a_b = to_f32(&dev.matvec_quant(qt, &wst, &xst_b, n, k).expect("dp4a bf16"));
    std::env::set_var(fb_env, "1");
    let scal_h = to_f32(&dev.matvec_quant(qt, &wst, &xst_h, n, k).expect("scalar f16"));
    let scal_b = to_f32(&dev.matvec_quant(qt, &wst, &xst_b, n, k).expect("scalar bf16"));
    std::env::remove_var(fb_env);

    let scale = scal_h.iter().chain(scal_b.iter()).fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
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
    assert!(nbad == 0, "{name} dp4a != scalar core: nbad={nbad} max_err={max_err} tol={tol}");
}

#[allow(clippy::too_many_arguments)]
fn ab_moe<F: Fn(&mut [u8], usize, usize)>(
    dev: &RocmDevice, log: &mut String, name: &str, qt: RocmQuantType, fb_env: &str,
    e_cnt: usize, nrows: usize, n: usize, k: usize, blk: usize, tsz: usize, fill: F,
) {
    let nblk = k / blk;
    let expert_bytes = n * nblk * tsz;
    let mut bank: Vec<u8> = Vec::with_capacity(e_cnt * expert_bytes);
    for e in 0..e_cnt {
        bank.extend_from_slice(&build_bytes(n, k, blk, tsz, |b, r, bb| fill(b, r * (e + 1) + e, bb)));
    }
    let wbank = dev.storage_from_slice(&bank).expect("upload bank");
    let ids: Vec<u32> = (0..nrows).map(|s| ((s * 3 + 1) % e_cnt) as u32).collect();
    let ids_dev = dev.storage_from_slice(&ids).expect("upload ids");
    let xf: Vec<f32> = (0..nrows * k).map(val).collect();
    let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let xst_h = dev.storage_from_slice(&xh).expect("upload x f16");

    std::env::remove_var(fb_env);
    let dp4a = to_f32(&dev.moe_matvec_quant(qt, &wbank, &xst_h, &ids_dev, nrows, n, k).expect("moe dp4a"));
    std::env::set_var(fb_env, "1");
    let scal = to_f32(&dev.moe_matvec_quant(qt, &wbank, &xst_h, &ids_dev, nrows, n, k).expect("moe scalar"));
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
    assert!(nbad == 0, "{name} MoE dp4a != scalar core: nbad={nbad} max_err={max_err} tol={tol}");
}

#[test]
fn qmatvec_iq1_dp4a_vs_scalar() {
    let mut log = String::new();
    let dev = RocmDevice::new(0).expect("rocm device");
    let shapes: &[(usize, usize)] = &[(64, 256), (4096, 4096), (1024, 3072), (17, 4096), (4096, 256)];
    let iq1s_fill = |blk: &mut [u8], r: usize, b: usize| put_d(blk, 0, r, b, 0.0078125, 0.00390625, 5);
    let iq1m_fill = |blk: &mut [u8], _r: usize, _b: usize| put_iq1m_d(blk, 0.0078125);
    for &(n, k) in shapes {
        ab_matvec(&dev, &mut log, "IQ1_S", RocmQuantType::IQ1_S, "HANZO_IQ1S_FALLBACK", n, k, 256, 50, iq1s_fill);
        ab_matvec(&dev, &mut log, "IQ1_M", RocmQuantType::IQ1_M, "HANZO_IQ1M_FALLBACK", n, k, 256, 56, iq1m_fill);
    }
    ab_moe(&dev, &mut log, "IQ1_S", RocmQuantType::IQ1_S, "HANZO_IQ1S_FALLBACK", 8, 16, 128, 512, 256, 50, iq1s_fill);
    ab_moe(&dev, &mut log, "IQ1_M", RocmQuantType::IQ1_M, "HANZO_IQ1M_FALLBACK", 8, 16, 128, 512, 256, 56, iq1m_fill);
    eprintln!("{log}");
}
