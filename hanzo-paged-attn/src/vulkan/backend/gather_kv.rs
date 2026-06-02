// Vulkan gather_kv_cache: STUB. Gathers all cached K/V for each sequence out of the paged
// blocks into contiguous [total_kv, kv_heads, head_size] tensors, used by the prefix-cache /
// donor-gather prompt path (hanzo-engine paged_attention/layers). Not yet implemented on Vulkan.
//
// TODO: port the CUDA gather_kv_cache_kernel.cu / Metal gather_kv_cache.metal -- one kernel that,
// per output token, looks up its physical block via block_table and copies the kv_heads*head_size
// vector out of the (rank-5 key / rank-4 value) paged layout into the packed output. This unblocks
// continuous batching with prefix sharing; until then the engine prompt path must not enter the
// gather branch on Vulkan (no num_cached_tokens / prefix cache).

use hanzo_ml::{DType, Result, Tensor};

pub fn gather_kv_cache(
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    _block_table: &Tensor,
    _cu_seq_lens: &Tensor,
    _out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    hanzo_ml::bail!("vulkan gather_kv_cache is not yet implemented (prefix-cache gather path)")
}
