#![cfg(feature = "cuda")]

use hanzo_ml::cuda::cudarc::driver::DevicePtr;
use hanzo_ml::quantized::QTensor;
use hanzo_ml::{DType, Result, Storage, Tensor};

use crate::layers::Activation;

/// Grouped-prefill is only worth its host-side counting sort past this many routed tokens; below it
/// the per-slot matvec (decode kernel) wins. Matches the MoE experts fast-path threshold.
pub(crate) const GROUPED_MIN_TOKENS: usize = 32;

fn glu_activation(act: Activation) -> Option<hanzo_quant::GluActivationType> {
    Some(match act {
        Activation::Silu | Activation::Swish => hanzo_quant::GluActivationType::Silu,
        Activation::NewGelu | Activation::GeluPytorchTanh => hanzo_quant::GluActivationType::Gelu,
        Activation::Gelu => hanzo_quant::GluActivationType::GeluErf,
        Activation::Relu => hanzo_quant::GluActivationType::Relu,
        _ => return None,
    })
}

/// Fused expert-grouped MMQ MoE for CUDA prefill (llama `mul_mat_id`): quantize the 16-bit activation
/// to q8_1 ONCE, run gate+up and the GLU-fused down through the int8 tensor-core MMQ kernels, and fold
/// the routing weights into the reduce. Keeps the activation 16-bit end to end (no f32 round-trip) and
/// shares one dispatch/quantize across the three projections. Returns `[num_tokens, hidden]` in
/// `out_dtype`, or `None` when the shapes/dtypes are outside the fused path (caller falls back).
/// `topk_weights` must already carry the router's scaling (norm + routed_scaling_factor).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_grouped_prefill(
    gate_w: &QTensor,
    up_w: &QTensor,
    down_w: &QTensor,
    xs_flat: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    act: Activation,
    num_experts: usize,
    topk: usize,
    out_dtype: DType,
) -> Result<Option<Tensor>> {
    if gate_w.dtype() != up_w.dtype()
        || !hanzo_quant::supports_mmq(gate_w.dtype())
        || !hanzo_quant::supports_mmq(down_w.dtype())
    {
        return Ok(None);
    }
    let Some(glu) = glu_activation(act) else {
        return Ok(None);
    };
    if !xs_flat.device().is_cuda() {
        return Ok(None);
    }
    let dev = xs_flat.device().as_cuda_device()?;
    let num_tokens = xs_flat.dim(0)?;
    let total_assignments = num_tokens * topk;

    let topk_ids_flat = topk_ids.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;
    let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
    let Storage::Cuda(ti_cuda) = &*ti_storage else {
        return Ok(None);
    };
    assert!(ti_layout.start_offset() == 0);
    let ti_u32 = ti_cuda.as_cuda_slice::<u32>()?;

    let (expert_bounds, sorted_token_ids, sorted_source_ids) =
        hanzo_quant::moe_dispatch_build(ti_u32, total_assignments, num_experts, topk, dev)?;

    let (gate, up) = hanzo_quant::grouped_moe_mmq_pair(
        gate_w,
        up_w,
        xs_flat,
        &sorted_source_ids,
        &sorted_token_ids,
        &expert_bounds,
        total_assignments,
        topk,
        num_experts,
        dev,
    )?;

    let tw_f32 = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (tw_storage, tw_layout) = tw_f32.storage_and_layout();
    let Storage::Cuda(tw_cuda) = &*tw_storage else {
        return Ok(None);
    };
    let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;
    let tw_ptr = tw_slice
        .slice(tw_layout.start_offset()..)
        .device_ptr(tw_slice.stream())
        .0 as *const f32;

    let down = hanzo_quant::grouped_moe_mmq_from_glu_pair(
        down_w,
        &gate,
        &up,
        &sorted_token_ids,
        &sorted_token_ids,
        &expert_bounds,
        total_assignments,
        num_tokens,
        num_experts,
        glu as i32,
        dev,
    )?;

    let reduced = if out_dtype == DType::BF16 {
        unsafe { hanzo_quant::moe_weighted_reduce_flat_bf16(&down, tw_ptr, num_tokens, topk, dev)? }
    } else {
        unsafe { hanzo_quant::moe_weighted_reduce_flat(&down, tw_ptr, num_tokens, topk, dev)? }
    };

    if reduced.dtype() == out_dtype {
        Ok(Some(reduced))
    } else {
        Ok(Some(reduced.to_dtype(out_dtype)?))
    }
}
