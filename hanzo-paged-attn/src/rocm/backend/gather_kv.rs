// ROCm port of cuda/backend/gather_kv.rs: un-page the paged KV cache into
// contiguous `[total_tokens, kv_heads, head_size]` K/V. Expressed with hanzo-ml
// tensor ops (one index_select per cache over a source-index map) rather than a
// HIP kernel: the per-token source offsets are tiny and the gather runs once per
// prefill, not per decode token, so the host read of block_table / cu_seq_lens is
// off the hot path. The index math is identical to the CUDA/Metal gather kernels.

use hanzo_ml::{DType, Device, Result, Tensor};

fn read_index_vec(t: &Tensor) -> Result<Vec<i64>> {
    let t = t.to_device(&Device::Cpu)?.contiguous()?;
    Ok(match t.dtype() {
        DType::I32 => t.to_vec1::<i32>()?.into_iter().map(|v| v as i64).collect(),
        DType::U32 => t.to_vec1::<u32>()?.into_iter().map(|v| v as i64).collect(),
        dt => hanzo_ml::bail!("gather_kv_cache cu_seq_lens must be i32/u32 (got {dt:?})"),
    })
}

fn read_index_rows(t: &Tensor) -> Result<Vec<Vec<i64>>> {
    let t = t.to_device(&Device::Cpu)?.contiguous()?;
    let rows = match t.dtype() {
        DType::I32 => t
            .to_vec2::<i32>()?
            .into_iter()
            .map(|r| r.into_iter().map(|v| v as i64).collect())
            .collect(),
        DType::U32 => t
            .to_vec2::<u32>()?
            .into_iter()
            .map(|r| r.into_iter().map(|v| v as i64).collect())
            .collect(),
        dt => hanzo_ml::bail!("gather_kv_cache block_table must be i32/u32 (got {dt:?})"),
    };
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache(
    key_cache: &Tensor,   // [num_blocks, kv_heads, head_size/x, block_size, x]
    value_cache: &Tensor, // [num_blocks, kv_heads, head_size, block_size]
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    block_table: &Tensor, // [batch, max_blocks]
    cu_seq_lens: &Tensor, // [batch + 1]
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let cache_dtype = key_cache.dtype();
    if value_cache.dtype() != cache_dtype {
        hanzo_ml::bail!(
            "gather_kv_cache expects matching cache dtypes, got {:?} and {:?}",
            cache_dtype,
            value_cache.dtype()
        );
    }
    // FP8 KV cache (hence any non-identity k/v scale) is rejected on ROCm, matching
    // the cache_dtype gate in paged_attention.rs; f16/bf16/f32 reach here with
    // identity scales so the gather is a pure copy.
    if matches!(cache_dtype, DType::F8E4M3) {
        hanzo_ml::bail!("FP8 KV cache is not supported on the rocm backend");
    }
    let _ = (k_scale, v_scale);

    if !matches!(block_table.dtype(), DType::I32 | DType::U32) {
        hanzo_ml::bail!(
            "gather_kv_cache expects i32/u32 block_table (got {:?})",
            block_table.dtype()
        );
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        hanzo_ml::bail!(
            "gather_kv_cache expects i32/u32 cu_seq_lens (got {:?})",
            cu_seq_lens.dtype()
        );
    }

    let device = key_cache.device();
    let (num_blocks, num_kv_heads, head_size_over_x, block_size, x) = key_cache.dims5()?;
    let head_size = head_size_over_x * x;
    if value_cache.dims4()? != (num_blocks, num_kv_heads, head_size, block_size) {
        hanzo_ml::bail!(
            "gather_kv_cache shape mismatch key_cache {:?} value_cache {:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let cu = read_index_vec(cu_seq_lens)?;
    let num_seqs = cu.len() - 1;
    let num_tokens = cu[num_seqs] as usize;

    if num_tokens == 0 {
        let k_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, device)?;
        let v_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, device)?;
        return Ok((k_out, v_out));
    }

    let block_table_rows = read_index_rows(block_table)?;

    // Contiguous element strides of the paged caches; K folds head_size into
    // (head_size/x, .., x), so a head dim d maps to (d/x, .., d%x).
    let k_block_stride = (num_kv_heads * head_size_over_x * block_size * x) as i64;
    let k_head_stride = (head_size_over_x * block_size * x) as i64;
    let v_block_stride = (num_kv_heads * head_size * block_size) as i64;
    let v_head_stride = (head_size * block_size) as i64;
    let x = x as i64;
    let block_size_i = block_size as i64;

    // Per-(head, d) source offsets, identical for every token.
    let n = num_kv_heads * head_size;
    let mut hd_k = Vec::with_capacity(n);
    let mut hd_v = Vec::with_capacity(n);
    for head_idx in 0..num_kv_heads as i64 {
        for d in 0..head_size as i64 {
            hd_k.push(head_idx * k_head_stride + (d / x) * block_size_i * x + (d % x));
            hd_v.push(head_idx * v_head_stride + d * block_size_i);
        }
    }

    // Per-token source base: physical block + slot within block.
    let mut tb_k = vec![0i64; num_tokens];
    let mut tb_v = vec![0i64; num_tokens];
    for batch_id in 0..num_seqs {
        let seq_start = cu[batch_id] as usize;
        let seq_len = cu[batch_id + 1] as usize - seq_start;
        let row = &block_table_rows[batch_id];
        for offset in 0..seq_len {
            let token_id = seq_start + offset;
            let block_id = row[offset / block_size];
            let slot = (offset % block_size) as i64;
            tb_k[token_id] = block_id * k_block_stride + slot * x;
            tb_v[token_id] = block_id * v_block_stride + slot;
        }
    }

    let k_idx = Tensor::from_vec(tb_k, (num_tokens, 1), device)?
        .broadcast_add(&Tensor::from_vec(hd_k, (1, n), device)?)?
        .flatten_all()?;
    let v_idx = Tensor::from_vec(tb_v, (num_tokens, 1), device)?
        .broadcast_add(&Tensor::from_vec(hd_v, (1, n), device)?)?
        .flatten_all()?;

    let k_out = key_cache
        .flatten_all()?
        .index_select(&k_idx, 0)?
        .reshape((num_tokens, num_kv_heads, head_size))?
        .to_dtype(out_dtype)?;
    let v_out = value_cache
        .flatten_all()?
        .index_select(&v_idx, 0)?
        .reshape((num_tokens, num_kv_heads, head_size))?
        .to_dtype(out_dtype)?;

    Ok((k_out, v_out))
}
