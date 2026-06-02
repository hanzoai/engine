// Vulkan gather_kv_cache: CPU-side gather. Gathers all cached K/V for each sequence out of the
// paged blocks into contiguous [total_kv, kv_heads, head_size] tensors, used by the prefix-cache /
// donor-gather prompt path (hanzo-engine paged_attention/layers).
//
// The paged layout (matching reshape_and_cache.comp / the CUDA kernel) is
//   key_cache  : [num_blocks, kv_heads, head_size/x, block_size, x]
//   value_cache: [num_blocks, kv_heads, head_size, block_size]
// The Vulkan backend stores the cache as f32 (f16/bf16 are upcast on upload) and the engine always
// passes unit k/v scales, so this is a plain copy: no fp8 dequant. We read the cache and the (tiny)
// index tensors to the host, walk every (seq, token) -> physical slot, copy the kv_heads*head_size
// vector into the packed output, then upload and cast to out_dtype on the cache device. A real
// device-side gather kernel can replace this later, but the indexing is identical.

use hanzo_ml::{DType, Device, Result, Tensor};

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
    if cache_dtype == DType::F8E4M3 {
        // fp8 cache needs a scaled dequant; the Vulkan scaffold has no fp8 cache, so this can't
        // happen on the happy path. TODO(vulkan-pagedattn): implement when an fp8 KV cache lands.
        hanzo_ml::bail!("vulkan gather_kv_cache: fp8 KV cache not yet supported");
    }
    if !matches!(out_dtype, DType::F16 | DType::BF16 | DType::F32) {
        hanzo_ml::bail!("vulkan gather_kv_cache: only f16/bf16/f32 output (got {out_dtype:?})");
    }

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
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

    let (num_blocks, num_kv_heads, head_size_over_x, block_size, x) = key_cache.dims5()?;
    let head_size = head_size_over_x * x;
    let (_, block_table_stride) = block_table.dims2()?;

    let cu_seq_lens_host = to_i64_vec(&cu_seq_lens)?;
    let cu_seq_lens_len = cu_seq_lens_host.len();
    let num_seqs = cu_seq_lens_len - 1;
    let num_tokens = cu_seq_lens_host[cu_seq_lens_len - 1].max(0) as usize;

    let cache_dev = key_cache.device();
    if num_tokens == 0 {
        let k_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, cache_dev)?;
        let v_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, cache_dev)?;
        return Ok((k_out, v_out));
    }

    let _ = (k_scale, v_scale); // unit scales only for the f32 cache; nothing to apply

    let block_table_host = to_i64_vec(&block_table)?;

    // Pull the full cache to host once. These are flat f32 buffers in the paged layout.
    let key_cache_host = to_f32_vec(key_cache)?;
    let value_cache_host = to_f32_vec(value_cache)?;

    let k_block_stride = num_kv_heads * head_size_over_x * block_size * x;
    let k_head_stride = head_size_over_x * block_size * x;
    let v_block_stride = num_kv_heads * head_size * block_size;
    let v_head_stride = head_size * block_size;

    let n = num_kv_heads * head_size;
    let mut k_out_host = vec![0f32; num_tokens * n];
    let mut v_out_host = vec![0f32; num_tokens * n];

    for token_id in 0..num_tokens {
        // Largest batch_id with cu_seq_lens[batch_id] <= token_id (cu_seq_lens is monotonic).
        let batch_id = cu_seq_lens_host[..num_seqs]
            .iter()
            .take_while(|&&c| c <= token_id as i64)
            .count()
            .saturating_sub(1);
        let batch_offset = token_id as i64 - cu_seq_lens_host[batch_id];
        let block_table_id = (batch_offset as usize) / block_size;
        let slot = (batch_offset as usize) % block_size;
        let block_id = block_table_host[batch_id * block_table_stride + block_table_id] as usize;
        if block_id >= num_blocks {
            hanzo_ml::bail!(
                "gather_kv_cache: block_id {block_id} out of range (num_blocks {num_blocks})"
            );
        }

        let out_base = token_id * n;
        for i in 0..n {
            let head_idx = i / head_size;
            let d = i % head_size;
            let x_idx = d / x;
            let x_offset = d % x;
            let k_src = block_id * k_block_stride
                + head_idx * k_head_stride
                + x_idx * block_size * x
                + slot * x
                + x_offset;
            let v_src =
                block_id * v_block_stride + head_idx * v_head_stride + d * block_size + slot;
            k_out_host[out_base + i] = key_cache_host[k_src];
            v_out_host[out_base + i] = value_cache_host[v_src];
        }
    }

    let k_out = Tensor::from_vec(k_out_host, (num_tokens, num_kv_heads, head_size), &Device::Cpu)?
        .to_device(cache_dev)?
        .to_dtype(out_dtype)?;
    let v_out = Tensor::from_vec(v_out_host, (num_tokens, num_kv_heads, head_size), &Device::Cpu)?
        .to_device(cache_dev)?
        .to_dtype(out_dtype)?;
    Ok((k_out, v_out))
}

fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn to_i64_vec(t: &Tensor) -> Result<Vec<i64>> {
    let t = t.to_device(&Device::Cpu)?.flatten_all()?;
    match t.dtype() {
        DType::I32 => Ok(t.to_vec1::<i32>()?.into_iter().map(|v| v as i64).collect()),
        DType::U32 => Ok(t.to_vec1::<u32>()?.into_iter().map(|v| v as i64).collect()),
        other => hanzo_ml::bail!("gather_kv_cache: expected i32/u32 index tensor, got {other:?}"),
    }
}
