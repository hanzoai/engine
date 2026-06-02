// Vulkan kv_scale_update: bailing stub. Updates running fp8 K/V quantization scales from observed
// key/value magnitudes (mirrors CUDA update_kvscales.cu / Metal kv_scale_update.metal). The engine
// only calls this when the KV cache dtype is F8E4M3; the Vulkan scaffold uses an f32 cache, so this
// is never reached on the happy path.
//
// TODO(vulkan-pagedattn): needs a real GPU reduce-max kernel over key/value writing the per-tensor
// scale, landed alongside an fp8 Vulkan KV cache (paged_attention/reshape_and_cache would then also
// need in-shader fp8 dequant/quant). Not expressible with existing Tensor ops in-place.

use hanzo_ml::{Result, Tensor};

pub fn kv_scale_update(
    _key: &Tensor,
    _value: &Tensor,
    _k_scales: &Tensor,
    _v_scales: &Tensor,
) -> Result<()> {
    hanzo_ml::bail!("vulkan kv_scale_update is not yet implemented (fp8 KV cache path)")
}
