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

/// FlashAttention kernel family, resolved from the live CUDA compute capability
/// rather than a build flag.
///
/// FlashAttention-3 (arXiv:2407.08608) is compiled for `sm_90a` only and its
/// cubins do not execute on any other architecture, so which family runs is an
/// architecture question, not a compile-time one:
///
/// * `V3Hopper` — `9.0` (Hopper H100/H200), for the head dims FA3 has kernels
///   for (64/128/256/512). Other Hopper head dims (96/160/192/224) have no FA3
///   kernel and take `Fallback`.
/// * `DatacenterBlackwell` — `10.x` (B200/GB200): the FA4 hook. `sm_90a` FA3
///   does NOT run on `sm_100`, so until an FA4 kernel set is vendored this also
///   resolves to the FA2-class path at dispatch.
/// * `Fallback` — everything else (Ampere/Ada, consumer Blackwell sm_120/121,
///   unknown, or non-CUDA): the shipped FA2-class kernel, unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "flash-attn-v3"), allow(dead_code))]
enum FaRoute {
    V3Hopper,
    DatacenterBlackwell,
    Fallback,
}

/// Pure architecture -> kernel-family decision. Deliberately free of any tensor
/// or device handle so the entire routing table is unit-testable on any host,
/// with no CUDA toolkit and no Hopper silicon (see `fa_route_tests`).
#[cfg_attr(not(feature = "flash-attn-v3"), allow(dead_code))]
fn fa_route(compute_cap: Option<(u32, u32)>, head_dim: usize) -> FaRoute {
    match compute_cap {
        Some((9, 0)) if matches!(head_dim, 64 | 128 | 256 | 512) => FaRoute::V3Hopper,
        Some((10, _)) => FaRoute::DatacenterBlackwell,
        _ => FaRoute::Fallback,
    }
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

/// FA3 / FA4 auto-selection by datacenter GPU architecture. See [`FaRoute`].
#[cfg(feature = "flash-attn-v3")]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let head_dim = q.dims4()?.3;
    match fa_route(cuda_compute_cap(q.device()), head_dim) {
        FaRoute::V3Hopper => flash_attn_v3(q, k, v, flash_params, sdpa_params),
        FaRoute::DatacenterBlackwell => {
            flash_attn_datacenter_blackwell(q, k, v, flash_params, sdpa_params)
        }
        FaRoute::Fallback => flash_attn_non_hopper(q, k, v, flash_params, sdpa_params),
    }
}

/// `sm_100a` (datacenter Blackwell) dispatch hook.
///
/// No Blackwell FA4 kernel set is vendored yet. When one lands, port it behind a
/// `flash-attn-v4` feature and dispatch it here; until then datacenter Blackwell
/// takes the FA2-class fallback, because the `sm_90a` FA3 cubins are
/// architecture-specific and do not run on `sm_100`.
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

/// FA3-only build on non-Hopper silicon: there is no FA2-class kernel linked to
/// fall back to, so this is a configuration error rather than a silent slow path.
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

#[cfg(test)]
mod fa_route_tests {
    //! The FA3/FA4 dispatch table is a pure function of `(compute_cap, head_dim)`,
    //! so the full routing contract is pinned here with no CUDA toolkit and no
    //! Hopper silicon. FlashAttention-3 is `sm_90a`-only (arXiv:2407.08608); this
    //! is the guard that no future edit silently routes a non-Hopper device onto
    //! cubins that would not execute there.
    use super::{fa_route, FaRoute};

    #[test]
    fn hopper_takes_fa3_only_for_supported_head_dims() {
        for hd in [64, 128, 256, 512] {
            assert_eq!(
                fa_route(Some((9, 0)), hd),
                FaRoute::V3Hopper,
                "sm_90 hd={hd}"
            );
        }
        // Head dims with no FA3 kernel fall back even on Hopper.
        for hd in [96, 160, 192, 224] {
            assert_eq!(
                fa_route(Some((9, 0)), hd),
                FaRoute::Fallback,
                "sm_90 hd={hd}"
            );
        }
    }

    #[test]
    fn datacenter_blackwell_takes_the_fa4_hook() {
        for cap in [(10, 0), (10, 1), (10, 3)] {
            assert_eq!(
                fa_route(Some(cap), 128),
                FaRoute::DatacenterBlackwell,
                "{cap:?}"
            );
        }
    }

    #[test]
    fn every_other_arch_and_non_cuda_falls_back() {
        // Consumer Blackwell (sm_120/121), Ada (sm_89), Ampere (sm_80/86), an
        // unknown future major, and non-CUDA (None) all keep the FA2-class path.
        for cap in [
            Some((12, 0)),
            Some((12, 1)),
            Some((8, 9)),
            Some((8, 6)),
            Some((8, 0)),
            Some((7, 5)),
            Some((11, 0)),
            None,
        ] {
            assert_eq!(fa_route(cap, 128), FaRoute::Fallback, "{cap:?}");
        }
    }

    #[test]
    fn hopper_minor_nonzero_is_not_sm90a_fa3() {
        // FA3 cubins are `sm_90a` specifically; a hypothetical 9.x (x>0) is not it.
        assert_eq!(fa_route(Some((9, 1)), 128), FaRoute::Fallback);
    }
}
