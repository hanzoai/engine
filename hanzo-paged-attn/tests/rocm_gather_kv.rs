//! Bit-exact correctness gate for the ROCm `gather_kv_cache` un-paging path.
//!
//! Fills a paged K/V cache so every element holds its own contiguous flat index,
//! then gathers via a multi-sequence block_table with a non-identity physical-block
//! permutation and a partial last block, and asserts each output element equals the
//! value at the hand-computed source index. Any error in the block/slot lookup or
//! the K-cache x-split is caught exactly (nbad == 0). Covers both small mixed-dtype
//! shapes and the real Qwen3 shape (head_size 128, x 8, block_size 32, multi-block).
//! Run with: cargo test -p hanzo-paged-attn --features rocm

#![cfg(feature = "rocm")]

use half::{bf16, f16};
use hanzo_ml::{DType, Device, Tensor};

struct Scenario<'a> {
    num_blocks: usize,
    num_kv_heads: usize,
    head_size: usize,
    x: usize,
    block_size: usize,
    seq_lens: &'a [usize],
    block_table: &'a [&'a [usize]], // [seq][logical_block] -> physical block
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
    sc: &Scenario,
    cache_dtype: DType,
    out_dtype: DType,
) -> Result<(usize, f32), Box<dyn std::error::Error>> {
    let Scenario {
        num_blocks,
        num_kv_heads,
        head_size,
        x,
        block_size,
        seq_lens,
        block_table,
    } = *sc;
    let head_size_over_x = head_size / x;
    let kc_elems = num_blocks * num_kv_heads * head_size_over_x * block_size * x;
    let vc_elems = num_blocks * num_kv_heads * head_size * block_size;
    // cache[i] == i, so a gathered element equals its source flat index exactly
    // (f32 is exact for integers up to 2^24, covering every shape here).
    let kc_host: Vec<f32> = (0..kc_elems).map(|i| i as f32).collect();
    let vc_host: Vec<f32> = (0..vc_elems).map(|i| i as f32).collect();

    let key_cache = typed_1d(&kc_host, cache_dtype, dev)?.reshape((
        num_blocks,
        num_kv_heads,
        head_size_over_x,
        block_size,
        x,
    ))?;
    let value_cache = typed_1d(&vc_host, cache_dtype, dev)?.reshape((
        num_blocks,
        num_kv_heads,
        head_size,
        block_size,
    ))?;

    let max_blocks = block_table.iter().map(|r| r.len()).max().unwrap();
    let mut bt_host = Vec::with_capacity(block_table.len() * max_blocks);
    for row in block_table {
        for b in 0..max_blocks {
            bt_host.push(*row.get(b).unwrap_or(&0) as u32);
        }
    }
    let block_table_t = Tensor::from_vec(bt_host, (block_table.len(), max_blocks), dev)?;

    let mut cu_host = vec![0i32];
    for &len in seq_lens {
        cu_host.push(cu_host.last().unwrap() + len as i32);
    }
    let num_tokens = *cu_host.last().unwrap() as usize;
    let cu_seq_lens = Tensor::from_vec(cu_host.clone(), (cu_host.len(),), dev)?;

    let (k_out, v_out) = hanzo_paged_attn::gather_kv_cache(
        &key_cache,
        &value_cache,
        None,
        None,
        &block_table_t,
        &cu_seq_lens,
        out_dtype,
    )?;
    assert_eq!(k_out.dims3()?, (num_tokens, num_kv_heads, head_size));
    assert_eq!(v_out.dims3()?, (num_tokens, num_kv_heads, head_size));
    if num_tokens == 0 {
        return Ok((0, 0.0)); // empty gather: shape is the whole contract, no elements to read back
    }

    let k_got = k_out
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let v_got = v_out
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let kc_idx = |blk: usize, h: usize, d: usize, off: usize| -> usize {
        blk * num_kv_heads * head_size_over_x * block_size * x
            + h * head_size_over_x * block_size * x
            + (d / x) * block_size * x
            + off * x
            + (d % x)
    };
    let vc_idx = |blk: usize, h: usize, d: usize, off: usize| -> usize {
        blk * num_kv_heads * head_size * block_size
            + h * head_size * block_size
            + d * block_size
            + off
    };

    let mut nbad = 0usize;
    let mut max_err = 0f32;
    for (seq, &seq_len) in seq_lens.iter().enumerate() {
        let seq_start = cu_host[seq] as usize;
        for offset in 0..seq_len {
            let token_id = seq_start + offset;
            let blk = block_table[seq][offset / block_size];
            let slot = offset % block_size;
            for h in 0..num_kv_heads {
                for d in 0..head_size {
                    let pos = (token_id * num_kv_heads + h) * head_size + d;
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

// Small shape, exercises the x-split + indirection + a partial block across 2 seqs.
fn small() -> Scenario<'static> {
    Scenario {
        num_blocks: 4,
        num_kv_heads: 2,
        head_size: 4,
        x: 2,
        block_size: 2,
        seq_lens: &[3, 2],
        block_table: &[&[3, 1], &[2, 0]],
    }
}

// Real Qwen3 shape: head_size 128, x 8 (f16), block_size 32, prefill spanning
// multiple physical blocks with a non-identity permutation. seq0 = 70 tokens (3
// blocks: 32+32+6), seq1 = 40 tokens (2 blocks: 32+8).
fn qwen3_like() -> Scenario<'static> {
    Scenario {
        num_blocks: 8,
        num_kv_heads: 2,
        head_size: 128,
        x: 8,
        block_size: 32,
        seq_lens: &[70, 40],
        block_table: &[&[5, 1, 6], &[3, 7]],
    }
}

#[test]
fn rocm_gather_kv_bit_exact_f16() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, &small(), DType::F16, DType::F16)?;
    eprintln!("f16->f16 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "f16 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_bit_exact_bf16() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, &small(), DType::BF16, DType::BF16)?;
    eprintln!("bf16->bf16 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "bf16 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_bit_exact_f32() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, &small(), DType::F32, DType::F32)?;
    eprintln!("f32->f32 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "f32 gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_cast_f16_to_f32() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, &small(), DType::F16, DType::F32)?;
    eprintln!("f16->f32 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "f16->f32 gather mismatch (max_err={max_err})");
    Ok(())
}

// The real-shape gather (head_size 128, x 8, block_size 32, multi-block prefill)
// in f32 so every flat index (< 65536) is represented exactly.
#[test]
fn rocm_gather_kv_bit_exact_qwen3_shape() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let (nbad, max_err) = run_case(&dev, &qwen3_like(), DType::F32, DType::F32)?;
    eprintln!("qwen3-shape f32 nbad={nbad} max_err={max_err}");
    assert_eq!(nbad, 0, "qwen3-shape gather mismatch (max_err={max_err})");
    Ok(())
}

#[test]
fn rocm_gather_kv_empty() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_rocm(0)?;
    let sc = Scenario {
        num_blocks: 1,
        num_kv_heads: 2,
        head_size: 4,
        x: 2,
        block_size: 2,
        seq_lens: &[0],
        block_table: &[&[0]],
    };
    let (nbad, max_err) = run_case(&dev, &sc, DType::F16, DType::F16)?;
    assert_eq!(
        nbad, 0,
        "empty gather should yield no elements (max_err={max_err})"
    );
    Ok(())
}
