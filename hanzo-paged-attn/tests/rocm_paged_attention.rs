//! Numeric correctness gate for the ROCm/HIP PagedAttention v1 decode kernel.
//!
//! Builds a single decode query, a small paged K/V cache spanning a few blocks,
//! a block_table, and a DEVICE-side `context_lens` array, runs
//! `paged_attention` on the ROCm backend, and compares against a naive host
//! reference `softmax(scale * Q.K^T) @ V` over the valid context. Passes iff
//! every element is within f16 tolerance (nbad == 0).
//!
//! The context length lives in device memory and is read inside the kernel, so
//! the launch parameters do not encode it (device_side_seqlen). The second test
//! exercises this directly: two different device-side context lengths are run
//! with IDENTICAL launch shapes and an identical `max_context_len` (the cache
//! capacity), and each result must match its own reference — i.e. only the
//! device array changed, not the launch parameters. Run with:
//!   cargo +nightly test -p hanzo-paged-attn --features rocm

#![cfg(feature = "rocm")]

use half::f16;
use hanzo_ml::{DType, Device, Tensor};

// Scenario dimensions.
const NUM_SEQS: usize = 1;
const NUM_HEADS: usize = 2; // query heads
const NUM_KV_HEADS: usize = 2; // == NUM_HEADS here (no GQA grouping)
const HEAD_SIZE: usize = 64;
const BLOCK_SIZE: usize = 16;
const X: usize = 8; // 16 / sizeof(f16) == 8
const NUM_BLOCKS: usize = 8; // a few blocks in the pool

fn softmax(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

/// Run the ROCm v1 decode kernel for a given live `context_len` against a naive
/// host reference. `max_context_len` is the *capacity* used to size the launch
/// (constant across decode steps); the live length lives in a DEVICE tensor.
/// Returns (nbad, max_err).
fn run_case(
    dev: &Device,
    context_len: usize,
    max_context_len: usize,
) -> Result<(usize, f32), Box<dyn std::error::Error>> {
    assert!(context_len <= max_context_len);
    let scale = 1.0f32 / (HEAD_SIZE as f32).sqrt();

    // Deterministic pseudo-random host data (seed derived from context_len so
    // the two cases use different data).
    let mut seed: u32 = 0x1234_5678 ^ (context_len as u32).wrapping_mul(2_654_435_761);
    let mut rnd = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5 // [-0.5, 0.5)
    };

    let q_host: Vec<f32> = (0..NUM_SEQS * NUM_HEADS * HEAD_SIZE)
        .map(|_| rnd())
        .collect();

    let mut k_ref = vec![vec![vec![0f32; HEAD_SIZE]; context_len]; NUM_KV_HEADS];
    let mut v_ref = vec![vec![vec![0f32; HEAD_SIZE]; context_len]; NUM_KV_HEADS];
    for h in 0..NUM_KV_HEADS {
        for t in 0..context_len {
            for d in 0..HEAD_SIZE {
                k_ref[h][t][d] = rnd();
                v_ref[h][t][d] = rnd();
            }
        }
    }

    // The launch is sized to capacity, so the block table always covers
    // max_context_len logical blocks. Physical blocks are a non-identity
    // permutation to exercise the indirection.
    let max_logical_blocks = max_context_len.div_ceil(BLOCK_SIZE);
    assert!(max_logical_blocks <= NUM_BLOCKS);
    let phys_of_logical: Vec<usize> = (0..max_logical_blocks)
        .map(|i| (i * 3 + 2) % NUM_BLOCKS)
        .collect();

    let kc_elems = NUM_BLOCKS * NUM_KV_HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X;
    let vc_elems = NUM_BLOCKS * NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE;
    let mut kc_host = vec![0f32; kc_elems];
    let mut vc_host = vec![0f32; vc_elems];

    let kc_idx = |blk: usize, h: usize, d: usize, off: usize| -> usize {
        let x_idx = d / X;
        let x_off = d % X;
        blk * NUM_KV_HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X
            + h * (HEAD_SIZE / X) * BLOCK_SIZE * X
            + x_idx * BLOCK_SIZE * X
            + off * X
            + x_off
    };
    let vc_idx = |blk: usize, h: usize, d: usize, off: usize| -> usize {
        blk * NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE
            + h * HEAD_SIZE * BLOCK_SIZE
            + d * BLOCK_SIZE
            + off
    };

    for t in 0..context_len {
        let blk = phys_of_logical[t / BLOCK_SIZE];
        let off = t % BLOCK_SIZE;
        for h in 0..NUM_KV_HEADS {
            for d in 0..HEAD_SIZE {
                // Round-trip through f16 so the cache and the reference agree.
                let kv = f16::from_f32(k_ref[h][t][d]).to_f32();
                let vv = f16::from_f32(v_ref[h][t][d]).to_f32();
                k_ref[h][t][d] = kv;
                v_ref[h][t][d] = vv;
                kc_host[kc_idx(blk, h, d, off)] = kv;
                vc_host[vc_idx(blk, h, d, off)] = vv;
            }
        }
    }

    // Host reference over the valid [0, context_len) span.
    let q_f16: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let mut out_ref = vec![vec![0f32; HEAD_SIZE]; NUM_HEADS];
    for h in 0..NUM_HEADS {
        let kvh = h * NUM_KV_HEADS / NUM_HEADS;
        let mut logits = vec![0f32; context_len];
        for t in 0..context_len {
            let mut dotp = 0f32;
            for d in 0..HEAD_SIZE {
                dotp += q_f16[h * HEAD_SIZE + d] * k_ref[kvh][t][d];
            }
            logits[t] = scale * dotp;
        }
        let w = softmax(&logits);
        for d in 0..HEAD_SIZE {
            let mut acc = 0f32;
            for t in 0..context_len {
                acc += w[t] * v_ref[kvh][t][d];
            }
            out_ref[h][d] = acc;
        }
    }

    // Device tensors. The block table is sized to capacity (max_logical_blocks),
    // so the launch shape is identical regardless of the live context_len.
    let q = Tensor::from_vec(
        q_host.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>(),
        (NUM_SEQS, NUM_HEADS, HEAD_SIZE),
        dev,
    )?;
    let key_cache = Tensor::from_vec(
        kc_host
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect::<Vec<_>>(),
        (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X),
        dev,
    )?;
    let value_cache = Tensor::from_vec(
        vc_host
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect::<Vec<_>>(),
        (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
        dev,
    )?;
    let bt_host: Vec<i32> = phys_of_logical.iter().map(|&p| p as i32).collect();
    let block_tables = Tensor::from_vec(bt_host, (NUM_SEQS, max_logical_blocks), dev)?;

    // DEVICE-side context length — this is the only thing that differs between
    // the two cases; the kernel reads it from device memory.
    let context_lens = Tensor::from_vec(vec![context_len as i32], (NUM_SEQS,), dev)?;

    let out = hanzo_paged_attn::paged_attention(
        &q,
        None,
        None,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lens,
        None,
        max_context_len, // capacity (constant per decode step)
        scale,
        1.0,
        None,
    )?;
    let out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    let atol = 2e-2f32;
    let rtol = 2e-2f32;
    let mut nbad = 0usize;
    let mut max_err = 0f32;
    for h in 0..NUM_HEADS {
        for d in 0..HEAD_SIZE {
            let got = out[h * HEAD_SIZE + d];
            let want = out_ref[h][d];
            let err = (got - want).abs();
            max_err = max_err.max(err);
            if err > atol + rtol * want.abs() {
                nbad += 1;
                if nbad <= 8 {
                    eprintln!(
                        "mismatch ctx={context_len} h={h} d={d}: got={got} want={want} err={err}"
                    );
                }
            }
        }
    }
    Ok((nbad, max_err))
}

#[test]
fn rocm_paged_attention_v1_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    // context_len = 40 spans 3 blocks (16 + 16 + 8), last block partial.
    let (nbad, max_err) = run_case(&dev, 40, 40)?;
    eprintln!("nbad={nbad} max_err={max_err}");
    assert_eq!(
        nbad, 0,
        "ROCm paged_attention v1 mismatch (max_err={max_err})"
    );
    Ok(())
}

/// Proves the device-side-seqlen invariant: identical launch shapes and an
/// identical capacity `max_context_len`, but two different DEVICE context
/// lengths, each matching its own reference. Only the device array changed.
#[test]
fn rocm_paged_attention_v1_device_side_seqlen() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let capacity = NUM_BLOCKS * BLOCK_SIZE; // 128 — fixed launch sizing
    for &ctx in &[17usize, 96usize] {
        let (nbad, max_err) = run_case(&dev, ctx, capacity)?;
        eprintln!("device_side_seqlen ctx={ctx} cap={capacity} nbad={nbad} max_err={max_err}");
        assert_eq!(
            nbad, 0,
            "device-side seqlen case ctx={ctx} mismatch (max_err={max_err})"
        );
    }
    Ok(())
}
