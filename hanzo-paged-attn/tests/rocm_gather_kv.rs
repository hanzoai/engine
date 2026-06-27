//! Bit-exact correctness gate for the ROCm `gather_kv_cache` un-paging path.
//!
//! Builds a small paged K/V cache whose every element holds its own contiguous
//! flat index, plus a 2-sequence block_table with a non-identity physical-block
//! permutation and a partial last block. Gathers it and asserts each output
//! element equals the value at the hand-computed source index (so any error in
//! the block/slot lookup or the K-cache x-split is caught exactly, nbad == 0).
//! Run with: cargo test -p hanzo-paged-attn --features rocm

#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::{DType, Device, Tensor};

const NUM_BLOCKS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_SIZE: usize = 4;
const X: usize = 2; // head_size folds into (head_size/x, .., x)
const BLOCK_SIZE: usize = 2;

// Two sequences: lens 3 and 2 (so seq0 spans 2 blocks with a partial second
// block). cu_seq_lens = [0, 3, 5], total 5 tokens.
const SEQ_LENS: [usize; 2] = [3, 2];
// Physical block per (seq, logical-block); non-identity so the gather must
// actually follow the table. Block 0 is left unused.
const BLOCK_TABLE: [[u32; 2]; 2] = [[3, 1], [2, 0]];

fn kc_idx(blk: usize, h: usize, d: usize, off: usize) -> usize {
    let x_idx = d / X;
    let x_off = d % X;
    blk * NUM_KV_HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X
        + h * (HEAD_SIZE / X) * BLOCK_SIZE * X
        + x_idx * BLOCK_SIZE * X
        + off * X
        + x_off
}

fn vc_idx(blk: usize, h: usize, d: usize, off: usize) -> usize {
    blk * NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE + h * HEAD_SIZE * BLOCK_SIZE + d * BLOCK_SIZE + off
}

fn typed_1d(
    vals: &[f32],
    dtype: DType,
    dev: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let n = vals.len();
    let t = match dtype {
        DType::F16 => Tensor::from_vec(
            vals.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>(),
            n,
            dev,
        )?,
        DType::BF16 => Tensor::from_vec(
            vals.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
            n,
            dev,
        )?,
        DType::F32 => Tensor::from_vec(vals.to_vec(), n, dev)?,
        other => return Err(format!("unsupported test dtype {other:?}").into()),
    };
    Ok(t)
}

fn run_case(
    dev: &Device,
    cache_dtype: DType,
    out_dtype: DType,
) -> Result<(usize, f32), Box<dyn std::error::Error>> {
    let kc_elems = NUM_BLOCKS * NUM_KV_HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X;
    let vc_elems = NUM_BLOCKS * NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE;
    // cache[i] == i, so a gathered element equals its source flat index exactly
    // (all indices < 256, representable in f16/bf16/f32 without rounding).
    let kc_host: Vec<f32> = (0..kc_elems).map(|i| i as f32).collect();
    let vc_host: Vec<f32> = (0..vc_elems).map(|i| i as f32).collect();

    let key_cache = typed_1d(&kc_host, cache_dtype, dev)?.reshape((
        NUM_BLOCKS,
        NUM_KV_HEADS,
        HEAD_SIZE / X,
        BLOCK_SIZE,
        X,
    ))?;
    let value_cache = typed_1d(&vc_host, cache_dtype, dev)?.reshape((
        NUM_BLOCKS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
    ))?;

    let bt_host: Vec<u32> = BLOCK_TABLE.iter().flatten().copied().collect();
    let block_table = Tensor::from_vec(bt_host, (SEQ_LENS.len(), BLOCK_TABLE[0].len()), dev)?;

    let mut cu_host = vec![0i32];
    for &len in &SEQ_LENS {
        cu_host.push(cu_host.last().unwrap() + len as i32);
    }
    let num_tokens = *cu_host.last().unwrap() as usize;
    let cu_seq_lens = Tensor::from_vec(cu_host.clone(), (cu_host.len(),), dev)?;

    let (k_out, v_out) = hanzo_paged_attn::gather_kv_cache(
        &key_cache,
        &value_cache,
        None,
        None,
        &block_table,
        &cu_seq_lens,
        out_dtype,
    )?;
    assert_eq!(k_out.dims3()?, (num_tokens, NUM_KV_HEADS, HEAD_SIZE));
    assert_eq!(v_out.dims3()?, (num_tokens, NUM_KV_HEADS, HEAD_SIZE));

    let k_got = k_out
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let v_got = v_out
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut nbad = 0usize;
    let mut max_err = 0f32;
    for (seq, &seq_len) in SEQ_LENS.iter().enumerate() {
        let seq_start = cu_host[seq] as usize;
        for offset in 0..seq_len {
            let token_id = seq_start + offset;
            let blk = BLOCK_TABLE[seq][offset / BLOCK_SIZE] as usize;
            let slot = offset % BLOCK_SIZE;
            for h in 0..NUM_KV_HEADS {
                for d in 0..HEAD_SIZE {
                    let pos = (token_id * NUM_KV_HEADS + h) * HEAD_SIZE + d;
                    let k_want = kc_host[kc_idx(blk, h, d, slot)];
                    let v_want = vc_host[vc_idx(blk, h, d, slot)];
                    let k_err = (k_got[pos] - k_want).abs();
                    let v_err = (v_got[pos] - v_want).abs();
                    max_err = max_err.max(k_err).max(v_err);
                    if k_err != 0.0 || v_err != 0.0 {
                        nbad += 1;
                        if nbad <= 8 {
                            eprintln!(
                                "mismatch t={token_id} h={h} d={d}: k got={} want={k_want}; v got={} want={v_want}",
                                k_got[pos], v_got[pos]
                            );
                        }
                    }
                }
            }
        }
    }
    Ok((nbad, max_err))
}

#[test]
fn rocm_gather_kv_bit_exact_f16() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, DType::F16, DType::F16)?;
    eprintln!("f16->f16 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "f16 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_bit_exact_bf16() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, DType::BF16, DType::BF16)?;
    eprintln!("bf16->bf16 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "bf16 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_bit_exact_f32() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, DType::F32, DType::F32)?;
    eprintln!("f32->f32 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "f32 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_cast_f16_to_f32() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, DType::F16, DType::F32)?;
    eprintln!("f16->f32 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "f16->f32 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_empty() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let key_cache = typed_1d(
        &vec![0f32; NUM_KV_HEADS * (HEAD_SIZE / X) * BLOCK_SIZE * X],
        DType::F16,
        &dev,
    )?
    .reshape((1, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X))?;
    let value_cache = typed_1d(
        &vec![0f32; NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE],
        DType::F16,
        &dev,
    )?
    .reshape((1, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE))?;
    let block_table = Tensor::from_vec(vec![0u32, 0], (1, 2), &dev)?;
    let cu_seq_lens = Tensor::from_vec(vec![0i32, 0], (2,), &dev)?;
    let (k_out, v_out) = hanzo_paged_attn::gather_kv_cache(
        &key_cache,
        &value_cache,
        None,
        None,
        &block_table,
        &cu_seq_lens,
        DType::F16,
    )?;
    assert_eq!(k_out.dims(), [0, NUM_KV_HEADS, HEAD_SIZE]);
    assert_eq!(v_out.dims(), [0, NUM_KV_HEADS, HEAD_SIZE]);
    Ok(())
}
