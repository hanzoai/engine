// Vulkan kv_scale_update: STUB. Updates running fp8 K/V quantization scales from observed
// key/value magnitudes (CUDA update_kvscales.cu / Metal kv_scale_update.metal). Only invoked by
// the engine when the KV cache dtype is F8E4M3; the Vulkan scaffold uses an f32 cache, so this is
// never reached on the happy path. Bails if called.
//
// TODO: implement alongside an fp8 Vulkan KV cache (a reduce-max kernel over key/value, writing the
// per-tensor scale), at which point paged_attention/reshape_and_cache also need fp8 dequant/quant.

use hanzo_ml::{Result, Tensor};

pub fn kv_scale_update(
    _key: &Tensor,
    _value: &Tensor,
    _k_scales: &Tensor,
    _v_scales: &Tensor,
) -> Result<()> {
    hanzo_ml::bail!("vulkan kv_scale_update is not yet implemented (fp8 KV cache path)")
}
