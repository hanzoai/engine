// Vulkan PagedAttention backend. Mirrors the CUDA/Metal backends but dispatches the
// `paged_attn` / `reshape_and_cache` SPIR-V kernels through the public VulkanDevice API
// (hanzo_ml::vulkan::{paged_attention_vk, reshape_and_cache_vk}).
//
// SCOPE (scaffold): f32 storage only (the Vulkan backend upcasts f16/bf16 to f32 on
// upload, so DType::{F16,BF16,F32} all arrive here as f32 buffers). No fp8 cache, no
// alibi, no softcapping, no attention sinks yet -- those bail. v1 (non-partitioned)
// kernel only.

use hanzo_ml::vulkan::{PagedAttnArgs, ReshapeCacheArgs};
use hanzo_ml::{
    backend::BackendStorage, CpuStorage, DType, Layout, Result, Shape, Storage, Tensor,
    VulkanStorage,
};

fn as_vulkan<'a>(storage: &'a Storage, what: &str) -> Result<&'a VulkanStorage> {
    match storage {
        Storage::Vulkan(s) => Ok(s),
        _ => hanzo_ml::bail!("{what} must be a vulkan tensor"),
    }
}

struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    alibi_slopes: Option<Tensor>,
    max_context_len: usize,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    sinks: Option<Tensor>,
}

impl hanzo_ml::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        hanzo_ml::bail!("no cpu support for paged-attention")
    }

    fn vulkan_fwd(&self, q: &VulkanStorage, q_l: &Layout) -> Result<(VulkanStorage, Shape)> {
        if !matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            hanzo_ml::bail!("vulkan paged-attention only supports f32/f16/bf16, got {:?}", q.dtype());
        }
        if self.alibi_slopes.is_some() {
            hanzo_ml::bail!("vulkan paged-attention: alibi_slopes not yet supported");
        }
        if self.sinks.is_some() {
            hanzo_ml::bail!("vulkan paged-attention: attention sinks not yet supported");
        }
        if self.k_scale.is_some() || self.v_scale.is_some() {
            // f32 cache only for now; fp8 scales would require a quantized cache + dequant in-shader.
            // The engine always passes Some(1.0) scales, so only bail for a real (non-unit) fp8 cache.
            if self.key_cache.dtype() == DType::F8E4M3 {
                hanzo_ml::bail!("vulkan paged-attention: fp8 KV cache not yet supported");
            }
        }
        if (self.softcapping - 1.0).abs() > f32::EPSILON {
            hanzo_ml::bail!("vulkan paged-attention: softcapping not yet supported");
        }

        let out_shape = q_l.shape().clone();
        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = as_vulkan(&kc, "key_cache")?;
        let (vc, _vc_l) = self.value_cache.storage_and_layout();
        let vc = as_vulkan(&vc, "value_cache")?;
        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = as_vulkan(&bt, "block_tables")?;
        let (cl, _cl_l) = self.context_lens.storage_and_layout();
        let cl = as_vulkan(&cl, "context_lens")?;

        // Cache layout: key_cache [num_blocks, num_kv_heads, head_size/x, block_size, x].
        let (_num_blocks, num_kv_heads, _hs_div_x, block_size, x) = kc_l.shape().dims5()?;
        let (_num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

        // Strides in ELEMENTS (== f32 slots on the vulkan backend, 1 logical elem per f32).
        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let args = PagedAttnArgs {
            q,
            key_cache: kc,
            value_cache: vc,
            block_tables: bt,
            context_lens: cl,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            max_num_blocks_per_seq,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            x,
            max_context_len: self.max_context_len,
            scale: self.softmax_scale,
        };
        let out = q.device().paged_attention_vk(&args)?;
        Ok((out, out_shape))
    }
}

/// PagedAttention decode: `softmax(scale * Q.K^T) . V` over the paged KV cache.
/// See the CUDA backend doc for the full tensor-shape contract. Vulkan scaffold: f32 only.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    let op = PagedAttention {
        softmax_scale,
        softcapping,
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        alibi_slopes: alibi_slopes.cloned(),
        max_context_len,
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
        sinks: sinks.map(|s| s.to_dtype(DType::F32)).transpose()?,
    };
    q.apply_op1(op)
}

/// Write new K/V into the paged cache at the slot_mapping positions. Vulkan scaffold: f32 only.
/// Maps the engine's i64 slot_mapping (>=0, -1 = pad) to the u32 sentinel layout the shader expects.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if !matches!(key.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        hanzo_ml::bail!("vulkan reshape_and_cache only supports f32/f16/bf16, got {:?}", key.dtype());
    }
    if key_cache.dtype() == DType::F8E4M3 {
        hanzo_ml::bail!("vulkan reshape_and_cache: fp8 KV cache not yet supported");
    }

    let (num_tokens, num_heads, head_size) = key.shape().dims3()?;

    // slot_mapping arrives as i64; map to u32 with -1 -> 0xFFFFFFFF (pad sentinel the shader skips).
    // TODO: avoid the CPU round-trip by adding an i64->u32-with-sentinel cast kernel, or have the
    // engine build the u32 slot mapping directly for the vulkan backend.
    let slot_u32: Vec<u32> = slot_mapping
        .to_device(&hanzo_ml::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|s| if s < 0 { u32::MAX } else { s as u32 })
        .collect();
    let slot_dev = Tensor::from_vec(slot_u32, num_tokens, key.device())?;

    let (k, k_l) = key.storage_and_layout();
    let k = as_vulkan(&k, "key")?;
    let (v, _v_l) = value.storage_and_layout();
    let v = as_vulkan(&v, "value")?;
    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = as_vulkan(&kc, "key_cache")?;
    let (vc, _vc_l) = value_cache.storage_and_layout();
    let vc = as_vulkan(&vc, "value_cache")?;
    let (sm, _sm_l) = slot_dev.storage_and_layout();
    let sm = as_vulkan(&sm, "slot_mapping")?;

    let (_num_blocks, _num_kv_heads, _hs_div_x, block_size, x) = kc_l.shape().dims5()?;
    let key_stride = k_l.stride()[0];
    let value_stride = value.layout().stride()[0];

    let args = ReshapeCacheArgs {
        key: k,
        value: v,
        key_cache: kc,
        value_cache: vc,
        slot_mapping: sm,
        num_tokens,
        num_heads,
        head_size,
        block_size,
        key_stride,
        value_stride,
        x,
    };
    k.device().reshape_and_cache_vk(&args)
}
