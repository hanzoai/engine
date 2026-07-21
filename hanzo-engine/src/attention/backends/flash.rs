use hanzo_ml::{Result, Tensor};

use crate::attention::SdpaParams;

#[cfg(feature = "flash-attn")]
fn flash_attn_v2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let window_size_left = sdpa_params.sliding_window;
    let default_causal = seq_len > 1;
    let use_varlen = b_sz > 1 || seq_len != k.dim(1)?;

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some(cumulative_seqlens_q) =
                params.cumulative_seqlens_q.get(&q.device().location())
            {
                let k_meta = &params.logical_k;
                let cumulative_seqlens_k = &k_meta.cumulative_seqlens[&q.device().location()];

                let window_size_right = if params.causal { Some(0) } else { None };
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                if let Some(softcap) = sdpa_params.softcap {
                    return hanzo_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
                        &q,
                        &k,
                        &v,
                        None,
                        cumulative_seqlens_q,
                        cumulative_seqlens_k,
                        params.max_q as usize,
                        k_meta.max as usize,
                        sdpa_params.softmax_scale,
                        window_size_left,
                        window_size_right,
                        softcap,
                    )?
                    .reshape(qshape);
                } else {
                    return hanzo_flash_attn::flash_attn_varlen_windowed(
                        &q,
                        &k,
                        &v,
                        cumulative_seqlens_q,
                        cumulative_seqlens_k,
                        params.max_q as usize,
                        k_meta.max as usize,
                        sdpa_params.softmax_scale,
                        window_size_left,
                        window_size_right,
                    )?
                    .reshape(qshape);
                }
            }
        }
    }

    let causal = flash_params.map_or(default_causal, |p| p.causal);
    let window_size_right = if causal { Some(0) } else { None };
    if let Some(softcap) = sdpa_params.softcap {
        hanzo_flash_attn::flash_attn_alibi_windowed_softcap(
            q,
            k,
            v,
            None,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            softcap,
        )
    } else {
        hanzo_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
        )
    }
}

#[cfg(feature = "flash-attn-v3")]
fn flash_attn_v3(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let default_causal = seq_len > 1;
    let use_varlen = b_sz > 1 || seq_len != k.dim(1)?;

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some(cumulative_seqlens_q) =
                params.cumulative_seqlens_q.get(&q.device().location())
            {
                let k_meta = &params.logical_k;
                let cumulative_seqlens_k = &k_meta.cumulative_seqlens[&q.device().location()];
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                let window_size_left = sdpa_params.sliding_window;
                let window_size_right = if params.causal { Some(0) } else { None };

                return hanzo_flash_attn_v3::flash_attn_varlen_windowed(
                    &q,
                    &k,
                    &v,
                    cumulative_seqlens_q,
                    cumulative_seqlens_k,
                    params.max_q as usize,
                    k_meta.max as usize,
                    sdpa_params.softmax_scale,
                    window_size_left,
                    window_size_right,
                    true,
                )?
                .reshape(qshape);
            }
        }
    }

    let causal = flash_params.map_or(default_causal, |p| p.causal);
    hanzo_flash_attn_v3::flash_attn(q, k, v, sdpa_params.softmax_scale, causal, true)
}

/// Live CUDA compute capability `(major, minor)` for the device a tensor is on,
/// cached per ordinal. `None` for non-CUDA devices or when the query fails.
/// Attention runs this every layer, so it is cheap after the first call.
#[cfg(feature = "flash-attn-v3")]
fn cuda_compute_cap(device: &hanzo_ml::Device) -> Option<(u32, u32)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    let gpu_id = match device.location() {
        hanzo_ml::DeviceLocation::Cuda { gpu_id } => gpu_id,
        _ => return None,
    };
    static CACHE: OnceLock<Mutex<HashMap<usize, Option<(u32, u32)>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    *guard
        .entry(gpu_id)
        .or_insert_with(|| crate::diagnostics::get_cuda_compute_capability(gpu_id))
}

/// FA3 / FA4 auto-selection by datacenter GPU architecture.
///
/// FlashAttention-3 (arXiv:2407.08608) is compiled for `sm_90a` only and its
/// cubins do not execute on any other architecture, so the choice is made from
/// the live device compute capability, not a build flag:
///
/// * `9.0` (Hopper H100/H200) -> FA3, for the head dims FA3 supports.
/// * `10.x` (datacenter Blackwell B200/GB200) -> FA4 when vendored, else the
///   FA2-class fallback — `sm_90a` FA3 does NOT run on `sm_100`.
/// * anything else (Ampere/Ada, consumer Blackwell sm_120/121, unknown) -> the
///   shipped FA2-class kernel, unchanged.
#[cfg(feature = "flash-attn-v3")]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let head_dim = q.dims4()?.3;
    match cuda_compute_cap(q.device()) {
        // Hopper. FA3 covers head dims 64/128/256/512; other dims (96/160/192/
        // 224) have no FA3 kernel and take the FA2-class path.
        Some((9, 0)) if matches!(head_dim, 64 | 128 | 256 | 512) => {
            flash_attn_v3(q, k, v, flash_params, sdpa_params)
        }
        // Datacenter Blackwell -> FA4 hook (falls through to FA2-class today).
        Some((10, _)) => flash_attn_datacenter_blackwell(q, k, v, flash_params, sdpa_params),
        // Everything else keeps the shipped kernel.
        _ => flash_attn_non_hopper(q, k, v, flash_params, sdpa_params),
    }
}

/// `sm_100a` (datacenter Blackwell) dispatch hook.
///
/// Dao-AILab has not released a Blackwell FA4 kernel set we can vendor as of
/// 2026-07. When it lands, port it behind a `flash-attn-v4` feature and dispatch
/// it here. Until then Blackwell datacenter uses the FA2-class fallback, because
/// the `sm_90a` FA3 cubins are architecture-specific and do not run on `sm_100`.
#[cfg(feature = "flash-attn-v3")]
fn flash_attn_datacenter_blackwell(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    flash_attn_non_hopper(q, k, v, flash_params, sdpa_params)
}

/// FA2-class fallback for every non-Hopper CUDA GPU (Ampere/Ada, consumer
/// Blackwell, and — until FA4 is vendored — datacenter Blackwell).
#[cfg(all(feature = "flash-attn-v3", feature = "flash-attn"))]
fn flash_attn_non_hopper(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    flash_attn_v2(q, k, v, flash_params, sdpa_params)
}

#[cfg(all(feature = "flash-attn-v3", not(feature = "flash-attn")))]
fn flash_attn_non_hopper(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    _sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    hanzo_ml::bail!(
        "this build has only the Hopper FA3 kernels (`flash-attn-v3`) but the device is not \
         Hopper (sm_90a); rebuild with `--features flash-attn` for the FA2-class fallback"
    )
}

#[cfg(all(feature = "flash-attn", not(feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    flash_attn_v2(q, k, v, flash_params, sdpa_params)
}

#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    _: &SdpaParams,
) -> Result<Tensor> {
    unimplemented!("Compile with `--features flash-attn` or `--features flash-attn-v3`.")
}
