//! Numeric correctness gate for the ROCm/HIP PagedAttention decode kernels.
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
//! device array changed, not the launch parameters.
//!
//! The long-context tests run past what v1 can hold in shared memory (a little
//! under 16K tokens on gfx1151), so they are served by the partitioned v2. Run with:
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
    spike: Option<usize>,
    atol: f32,
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
    // (i * 3 + 2) mod n is a permutation exactly when 3 does not divide n, so the
    // pool is the smallest such n that holds the capacity -- 8 for the short cases.
    let num_blocks = (NUM_BLOCKS.max(max_logical_blocks)..)
        .find(|n| n % 3 != 0)
        .unwrap();
    let phys_of_logical: Vec<usize> = (0..max_logical_blocks)
        .map(|i| (i * 3 + 2) % num_blocks)
        .collect();

    let kc_elems = num_blocks * NUM_KV_HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X;
    let vc_elems = num_blocks * NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE;
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

    // A spike key: aligned with the query and forty times its length, so its logit
    // dominates every other and the answer is essentially V at that position. Placed
    // late in a long context it sits in a late partition, so a v2 that drops or
    // mis-scales one lands far from the reference instead of inside the tolerance.
    if let Some(at) = spike {
        assert!(at < context_len);
        for h in 0..NUM_KV_HEADS {
            let qh = h * NUM_HEADS / NUM_KV_HEADS;
            for d in 0..HEAD_SIZE {
                k_ref[h][at][d] = 40.0 * q_host[qh * HEAD_SIZE + d];
            }
        }
    }

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
        (num_blocks, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X),
        dev,
    )?;
    let value_cache = Tensor::from_vec(
        vc_host
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect::<Vec<_>>(),
        (num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
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

    // A tolerance only means something if the answer is larger than it: attention
    // spread evenly over 20K tokens averages the values to a few thousandths, and an
    // all-zero output would sit inside a loose bound. Refuse such a case outright.
    let max_ref = out_ref.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        max_ref > 10.0 * atol,
        "ctx={context_len}: the reference (max |x| = {max_ref}) is too small for atol {atol} to discriminate"
    );
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
    // context_len = 40 spans 3 blocks (16 + 16 + 8), last block partial. A short context averages
    // its values to about 0.2, so the tolerance sits two orders below the answer.
    let (nbad, max_err) = run_case(&dev, 40, 40, None, 2e-3)?;
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
        let (nbad, max_err) = run_case(&dev, ctx, capacity, None, 2e-3)?;
        eprintln!("device_side_seqlen ctx={ctx} cap={capacity} nbad={nbad} max_err={max_err}");
        assert_eq!(
            nbad, 0,
            "device-side seqlen case ctx={ctx} mismatch (max_err={max_err})"
        );
    }
    Ok(())
}

/// Past what v1 can hold in shared memory: 20000 tokens is ~78 KiB of v1 logits
/// against a 64 KiB workgroup, which v1 cannot launch at all. v2 serves it. The
/// even case checks the ordinary merge at a tolerance tight enough that an
/// all-zero answer fails; the spike case puts the dominant key in a late
/// partition, where a dropped or mis-scaled partition cannot hide.
#[test]
fn rocm_paged_attention_serves_a_context_v1_cannot() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    for (ctx, spike, atol) in [
        (20_000usize, None, 2e-4f32),
        (20_000, Some(17_321), 2e-2),
        // 16384 = 32 partitions exactly, so the spike is the ONLY token of the 33rd:
        // a boundary v2 must merge, at a length v1 cannot launch.
        (16_385, Some(16_384), 2e-2),
    ] {
        let (nbad, max_err) = run_case(&dev, ctx, ctx, spike, atol)?;
        eprintln!("long ctx={ctx} spike={spike:?} nbad={nbad} max_err={max_err}");
        assert_eq!(
            nbad, 0,
            "ctx={ctx} spike={spike:?} mismatch (max_err={max_err})"
        );
    }
    Ok(())
}

/// The capacity a captured decode graph is launched with is the bucket, not the
/// live length: partitions past this sequence's end must do nothing and must not
/// be read. Two live lengths under one long capacity, each against its own reference.
#[test]
fn rocm_paged_attention_long_capacity_short_context() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let capacity = 24_576;
    for (ctx, spike) in [(16_500usize, Some(16_400usize)), (700, Some(3))] {
        let (nbad, max_err) = run_case(&dev, ctx, capacity, spike, 2e-2)?;
        eprintln!("capacity={capacity} ctx={ctx} nbad={nbad} max_err={max_err}");
        assert_eq!(
            nbad, 0,
            "capacity={capacity} ctx={ctx} mismatch (max_err={max_err})"
        );
    }
    Ok(())
}
