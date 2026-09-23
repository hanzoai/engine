#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared Gated Delta Net (GDN) implementation for hybrid models.
//!
//! Used by both Qwen3 Next (text-only) and Qwen3.5 MoE (multimodal) models.

use hanzo_ml::{DType, Device, IndexOp, Module, Result, Tensor, D};
use hanzo_nn::Linear;
use hanzo_quant::{QuantMethod, QuantizedConfig, RowParallelLayer, ShardedVarBuilder};
use std::sync::Arc;

use crate::{
    device_map::DeviceMapper,
    kv_cache::{RecurrentStatePool, RecurrentTrail},
    utils::unvarbuilder::UnVarBuilder,
};

// ====================== GDN Config Trait ======================

/// Trait abstracting over config differences between Qwen3 Next and Qwen3.5 MoE.
#[allow(dead_code)]
pub trait GdnConfig {
    fn hidden_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn linear_key_head_dim(&self) -> usize;
    fn linear_value_head_dim(&self) -> usize;
    fn linear_num_key_heads(&self) -> usize;
    fn linear_num_value_heads(&self) -> usize;
    fn quantization_config(&self) -> &Option<QuantizedConfig>;

    fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads() * self.linear_key_head_dim()
    }
    fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim()
    }
    fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

// ====================== RMSNorm Gated ======================

/// The gate's activation. vLLM `RMSNormGated` accepts silu or sigmoid (`layernorm.py:248-249`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Act {
    Silu,
    Sigmoid,
}

/// RMSNorm with gating: `rms_norm(x) * weight * act(gate)`, the norm taken before the gate
/// (vLLM `RMSNormGated`, `norm_before_gate=True`, `layernorm.py:243-269`). The gate is silu
/// unless [`RmsNormGated::sigmoid`] switches it.
pub struct RmsNormGated {
    pub weight: Tensor,
    eps: f64,
    act: Act,
}

impl RmsNormGated {
    pub fn new(
        size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get(size, "weight")?;
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self::from_weight(weight, eps))
    }

    /// Build directly from an already-materialized weight (e.g. a dequantized GGUF tensor).
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            eps,
            act: Act::Silu,
        }
    }

    /// Gate with `sigmoid(gate)`, as a GDN with `output_gate_type = "sigmoid"` does
    /// (vLLM `qwen_gdn_linear_attn.py:471-484`).
    pub fn sigmoid(self) -> Self {
        Self {
            act: Act::Sigmoid,
            ..self
        }
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let gate = gate.to_dtype(DType::F32)?;
        let gate = match self.act {
            Act::Silu => hanzo_nn::ops::silu(&gate)?,
            Act::Sigmoid => sigmoid(&gate)?,
        };
        let weight = self.weight.to_dtype(DType::F32)?;
        let normed = hanzo_nn::ops::rms_norm(&x, &weight, self.eps as f32)?;
        normed.broadcast_mul(&gate)?.to_dtype(dtype)
    }
}

// ====================== GDN layer cache ======================

/// The state after each position of one forward, batch-major.
#[derive(Debug, Clone, Default)]
pub struct GdnTrail {
    pub conv: Vec<Tensor>,
    pub recurrent: Vec<Tensor>,
}

#[derive(Debug)]
pub struct GdnLayerCache {
    /// Conv state: (batch, conv_dim, kernel_size)
    pub conv_state: Tensor,
    /// Recurrent state: (batch, num_v_heads, head_k_dim, head_v_dim)
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
    /// `Some` asks `forward` to fill it.
    pub trail: Option<GdnTrail>,
}

#[allow(dead_code)]
impl GdnLayerCache {
    pub fn new(cfg: &dyn GdnConfig, dtype: DType, device: &Device) -> Result<Self> {
        let conv_dim = cfg.linear_conv_dim();
        let conv_state = Tensor::zeros((1, conv_dim, cfg.linear_conv_kernel_dim()), dtype, device)?;
        let recurrent_state = Tensor::zeros(
            (
                1,
                cfg.linear_num_value_heads(),
                cfg.linear_key_head_dim(),
                cfg.linear_value_head_dim(),
            ),
            dtype,
            device,
        )?;
        Ok(Self {
            conv_state,
            recurrent_state,
            seqlen_offset: 0,
            trail: None,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

impl GdnLayerCache {
    /// Note the conv state after each position of `inputs`, the raw conv inputs of this forward
    /// as (batch, seq, conv_dim). Call before the conv advances `conv_state`. No-op without a
    /// trail.
    pub fn trail_conv(&mut self, inputs: &Tensor) -> Result<()> {
        let Some(trail) = self.trail.as_mut() else {
            return Ok(());
        };
        // The conv state is the last `kernel` raw inputs, so the state after position `t` is a
        // window over the old state followed by the new inputs.
        let kernel = self.conv_state.dim(D::Minus1)?;
        let window = Tensor::cat(
            &[
                &self.conv_state.to_dtype(inputs.dtype())?,
                &inputs.transpose(1, 2)?,
            ],
            D::Minus1,
        )?;
        trail.conv = (0..inputs.dim(1)?)
            .map(|t| window.narrow(D::Minus1, t + 1, kernel)?.contiguous())
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// The gated delta rule over this cache's recurrent state, noting the state after each
    /// position when a trail was asked for.
    pub fn recurrence(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let Some(trail) = self.trail.as_mut() else {
            return gated_delta_rule_recurrence(q, k, v, g, beta, &mut self.recurrent_state);
        };
        // One position at a time, copying the state after each. The fused kernels advance the
        // state in place, so a handle kept across steps would alias the final state.
        let seq_len = q.dim(1)?;
        trail.recurrent = Vec::with_capacity(seq_len);
        let mut ys = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let at = |x: &Tensor| x.narrow(1, t, 1)?.contiguous();
            ys.push(gated_delta_rule_recurrence(
                &at(q)?,
                &at(k)?,
                &at(v)?,
                &at(g)?,
                &at(beta)?,
                &mut self.recurrent_state,
            )?);
            trail.recurrent.push(self.recurrent_state.copy()?);
        }
        Tensor::cat(&ys, 1)
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            seqlen_offset: self.seqlen_offset,
            trail: self.trail.clone(),
        }
    }
}

// ====================== Pooled state ======================

/// The pool slots one forward reads and writes.
pub enum PoolSlots<'a> {
    /// One sequence at a host-known slot, `offset` tokens in. Access is constant-offset
    /// `narrow`/`slice_set` with no device sync, which keeps the decode step capturable by a
    /// CUDA/HIP graph.
    One { slot: usize, offset: usize },
    /// A batch, gathered and scattered through a device index tensor.
    Many(&'a Tensor),
}

/// Run `forward` over pooled state: load the batch's slots, let it advance them, write them back.
/// With `trail` the pool also keeps the state after each position, so a partly accepted verify can
/// rewind. Without it the pool's previous trail is dropped, since it no longer matches the state.
pub fn forward_pooled(
    pool: &mut RecurrentStatePool,
    slots: PoolSlots<'_>,
    layer_idx: usize,
    trail: bool,
    forward: impl FnOnce(&mut GdnLayerCache) -> Result<Tensor>,
) -> Result<Tensor> {
    let (slot_ids, start_offsets) = match &slots {
        PoolSlots::One { slot, offset } => (vec![*slot as u32], vec![*offset]),
        PoolSlots::Many(indices) => {
            let ids: Vec<u32> = indices.to_vec1()?;
            let offsets: Vec<usize> = ids
                .iter()
                .map(|&id| pool.get_seqlen_offset(id as usize))
                .collect();
            (ids, offsets)
        }
    };
    let Some(&first_offset) = start_offsets.first() else {
        hanzo_ml::bail!("Hybrid recurrent state indices are empty.");
    };
    // A layer forward asks one thing of the offset: is this the start of a sequence, which
    // zero-pads the conv, or a continuation, which carries its state. Sequences of different
    // lengths batch freely; a new one cannot share a forward with a continuing one.
    if start_offsets
        .iter()
        .any(|&o| (o == 0) != (first_offset == 0))
    {
        hanzo_ml::bail!(
            "Hybrid layer {layer_idx}: a new sequence shares a forward with a continuing one."
        );
    }
    let (conv_state, recurrent_state) = match &slots {
        PoolSlots::One { slot, .. } => (
            pool.conv_state.narrow(0, *slot, 1)?,
            pool.recurrent_state.narrow(0, *slot, 1)?,
        ),
        PoolSlots::Many(indices) => (
            pool.gather_conv_state(indices)?,
            pool.gather_recurrent_state(indices)?,
        ),
    };
    let mut cache = GdnLayerCache {
        conv_state,
        recurrent_state,
        seqlen_offset: first_offset,
        trail: trail.then(GdnTrail::default),
    };
    let out = forward(&mut cache)?;

    // A conv-only pool's recurrent state never moves; on the `One` path it is still a view of the
    // pool, which `slice_set` cannot write into itself.
    let recurrent = !pool.conv_only();
    match &slots {
        PoolSlots::One { slot, .. } => {
            let conv = cache.conv_state.to_dtype(pool.conv_state.dtype())?;
            pool.conv_state.slice_set(&conv.contiguous()?, 0, *slot)?;
            if recurrent {
                let state = cache
                    .recurrent_state
                    .to_dtype(pool.recurrent_state.dtype())?;
                pool.recurrent_state
                    .slice_set(&state.contiguous()?, 0, *slot)?;
            }
        }
        PoolSlots::Many(indices) => {
            pool.scatter_conv_state(indices, &cache.conv_state)?;
            if recurrent {
                pool.scatter_recurrent_state(indices, &cache.recurrent_state)?;
            }
        }
    }
    let advanced = cache.seqlen_offset.saturating_sub(first_offset);
    for (&id, &offset) in slot_ids.iter().zip(&start_offsets) {
        pool.set_seqlen_offset(id as usize, offset + advanced);
    }
    pool.set_trail(cache.trail.map(|t| RecurrentTrail {
        slots: slot_ids,
        start_offsets,
        conv: t.conv,
        recurrent: t.recurrent,
    }));
    Ok(out)
}

// ====================== GDN math functions ======================

pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // x * rsqrt(sum(x^2) + eps), expressed via the fused rms_norm kernel
    // (x * rsqrt(mean(x^2) + e) * w): w = 1/sqrt(n), e = eps/n makes it exact.
    let n = x.dim(D::Minus1)?;
    let scale = (n as f64).sqrt().recip();
    let weight = Tensor::ones(n, x.dtype(), x.device())?.affine(scale, 0.0)?;
    hanzo_nn::ops::rms_norm(&x.contiguous()?, &weight, (eps / n as f64) as f32)
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

// ROCm eager has no fused Sigmoid op; compose it from exp there (affine/exp/broadcast_div all
// already lower on ROCm via softmax/l2_norm). Other backends keep the fused kernel.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    if x.device().is_rocm() {
        let denom = x.affine(-1.0, 0.0)?.exp()?.affine(1.0, 1.0)?;
        Tensor::ones_like(x)?.broadcast_div(&denom)
    } else {
        hanzo_nn::ops::sigmoid(x)
    }
}

/// A/B knob: `GDN_FUSED_FALLBACK=1` forces the portable ops-composed scan on every backend,
/// isolating the fused per-backend recurrence kernel's contribution. Default (unset) uses the fused
/// kernel where one exists. Read once, cached.
fn gdn_force_portable() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| {
        std::env::var("GDN_FUSED_FALLBACK").is_ok_and(|v| v != "0" && !v.is_empty())
    })
}

/// Recurrent gated delta rule for prefill and decode. q,k: (b, s, v_heads, k_dim); v: (b, s,
/// v_heads, v_dim); g,beta: (b, s, v_heads); state: (b, v_heads, k_dim, v_dim), updated in place.
/// Returns (b, s, v_heads, v_dim). Single dispatch point: routes to the fused per-backend kernel
/// where one exists, else the portable scan (which also serves Vulkan/CPU). Dispatch is once per
/// call, so callers stay one backend-agnostic line at no per-element cost.
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    if gdn_force_portable() {
        return recurrence_portable(q, k, v, g, beta, state);
    }
    #[cfg(feature = "cuda")]
    if state.device().is_cuda() {
        return recurrence_cuda(q, k, v, g, beta, state);
    }
    #[cfg(feature = "metal")]
    if state.device().is_metal() {
        // seq==1 decode: skip the flatten/unflatten transposes, step the fused kernel in place.
        if q.dim(1)? == 1 && q.dim(0)? == 1 && state.dtype() == DType::F32 {
            return recurrence_metal_step(q, k, v, g, beta, state);
        }
        return recurrence_metal(q, k, v, g, beta, state);
    }
    #[cfg(feature = "rocm")]
    if state.device().is_rocm() {
        return recurrence_rocm(q, k, v, g, beta, state);
    }
    // The Vulkan single-step kernel (gdn_step_vulkan) isn't ported to canonical hanzo-ml yet, so
    // don't intercept the Vulkan decode path -- fall through to recurrence_portable, which is
    // documented to "serve CPU, Vulkan, and any backend without a fused kernel".
    recurrence_portable(q, k, v, g, beta, state)
}

/// Native Vulkan single decode step (seq==1). q,k,v,g,beta arrive (1, 1, v_heads, ..); the state
/// (1, v_heads, k_dim, v_dim) is updated in place in VRAM. Applies the 1/sqrt(k_dim) q-scale that
/// the portable scan does internally, then returns y (1, 1, v_heads, v_dim).
#[allow(dead_code)]
fn recurrence_vulkan_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (b, _s, nh, kd) = q.dims4()?;
    let vd = v.dim(D::Minus1)?;
    let bh = b * nh;
    let scale = 1.0 / (kd as f64).sqrt();
    let q = (q.reshape((bh, kd))?.to_dtype(DType::F32)? * scale)?.contiguous()?;
    let k = k.reshape((bh, kd))?.to_dtype(DType::F32)?.contiguous()?;
    let v = v.reshape((bh, vd))?.to_dtype(DType::F32)?.contiguous()?;
    let g = g.reshape(bh)?.to_dtype(DType::F32)?.contiguous()?;
    let beta = beta.reshape(bh)?.to_dtype(DType::F32)?.contiguous()?;
    let mut s = state.reshape((bh, kd, vd))?;
    let y = crate::vulkan::gdn::gdn_step_vulkan(&q, &k, &v, &g, &beta, &mut s)?;
    *state = s.reshape((b, nh, kd, vd))?;
    y.reshape((b, 1, nh, vd))
}

/// Flatten (b, s, heads, dim) -> (b*heads, s, dim) in f32 contiguous, the layout the fused
/// CUDA/Metal kernels expect. state (b, heads, k, v) flattens to (b*heads, k, v) the same way.
#[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
fn recurrence_flatten(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (b, s, nh, kd) = q.dims4()?;
    let vd = v.dim(D::Minus1)?;
    let bh = b * nh;
    let scale = 1.0 / (kd as f64).sqrt();
    let seq_dim = |t: &Tensor, d: usize| -> Result<Tensor> {
        t.transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((bh, s, d))?
            .contiguous()
    };
    let scalar = |t: &Tensor| -> Result<Tensor> {
        t.to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh, s))?
            .contiguous()
    };
    Ok((
        (seq_dim(q, kd)? * scale)?,
        seq_dim(k, kd)?,
        seq_dim(v, vd)?,
        scalar(g)?,
        scalar(beta)?,
        state
            .to_dtype(DType::F32)?
            .reshape((bh, kd, vd))?
            .contiguous()?,
    ))
}

/// Reshape a fused kernel's (b*heads, s, v_dim) output back to (b, s, heads, v_dim), write the
/// (b*heads, k, v) state back into `state`, and restore the input dtype.
#[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
fn recurrence_unflatten(
    out_bh: &Tensor,
    state_flat: &Tensor,
    q: &Tensor,
    v: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (b, s, nh, kd) = q.dims4()?;
    let vd = v.dim(D::Minus1)?;
    *state = state_flat
        .reshape((b, nh, kd, vd))?
        .to_dtype(state.dtype())?;
    out_bh
        .reshape((b, nh, s, vd))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(q.dtype())
}

/// Fused ROCm recurrence: ONE `gdn_scan` launch (a thread per (b*head, v) column, sequential
/// over the sequence, state in registers) replaces the host-side per-token ops loop. Prefill
/// and decode both take this path; the state is flattened to f32, updated in place by the
/// kernel, and folded back by `recurrence_unflatten`.
#[cfg(feature = "rocm")]
fn recurrence_rocm(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (qf, kf, vf, gf, bf, statef) = recurrence_flatten(q, k, v, g, beta, state)?;
    let out = hanzo_ml::rocm_backend::gdn_scan_rocm(&qf, &kf, &vf, &gf, &bf, &statef)?;
    recurrence_unflatten(&out, &statef, q, v, state)
}

/// Fused CUDA recurrence: chunked scan for prefill (seq >= 64), single-pass for short/decode.
#[cfg(feature = "cuda")]
fn recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    const CHUNK_THRESHOLD: usize = 64;
    let (b, s_len, nh, kd) = q.dims4()?;
    let vd = v.dim(D::Minus1)?;

    // Layout-native prefill fast path: for K in {64,128} the fused kernel reads
    // q/k/v/g/beta straight out of the model's [B,S,H,D] layout, so we skip the
    // transpose+contiguous reshuffle (the large `ucopy_f32` copies) that
    // `recurrence_flatten` does. Only the (small) state is flattened to [B*H,K,V].
    if s_len >= CHUNK_THRESHOLD && (kd == 64 || kd == 128) {
        let dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;
        let mut s = state
            .to_dtype(DType::F32)?
            .reshape((b * nh, kd, vd))?
            .contiguous()?;
        let out = crate::cuda::gdn::chunked_gated_delta_rule_recurrence_native_cuda(
            &q, &k, &v, &g, &beta, &mut s,
        )?;
        *state = s.reshape((b, nh, kd, vd))?.to_dtype(state.dtype())?;
        return out.to_dtype(dtype);
    }

    let (q_bh, k_bh, v_bh, g_bh, beta_bh, mut s) = recurrence_flatten(q, k, v, g, beta, state)?;
    let out_bh = if q.dim(1)? >= CHUNK_THRESHOLD {
        crate::cuda::gdn::chunked_gated_delta_rule_recurrence_cuda(
            &q_bh, &k_bh, &v_bh, &g_bh, &beta_bh, &mut s,
        )?
    } else {
        crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
            &q_bh, &k_bh, &v_bh, &g_bh, &beta_bh, &mut s,
        )?
    };
    recurrence_unflatten(&out_bh, &s, q, v, state)
}

/// Fused Metal recurrence (mirrors the CUDA path).
#[cfg(feature = "metal")]
fn recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    const CHUNK_THRESHOLD: usize = 64;
    let (q_bh, k_bh, v_bh, g_bh, beta_bh, mut s) = recurrence_flatten(q, k, v, g, beta, state)?;
    let out_bh = if q.dim(1)? >= CHUNK_THRESHOLD {
        crate::metal::gdn::chunked_gated_delta_rule_recurrence_metal(
            &q_bh, &k_bh, &v_bh, &g_bh, &beta_bh, &mut s,
        )?
    } else {
        crate::metal::gdn::gated_delta_rule_recurrence_metal(
            &q_bh, &k_bh, &v_bh, &g_bh, &beta_bh, &mut s,
        )?
    };
    recurrence_unflatten(&out_bh, &s, q, v, state)
}

/// Native Metal single decode step (seq==1, batch==1). Mirrors `recurrence_vulkan_step`: cheap
/// reshapes (no transpose) into the [BH, ..] layout, applies the 1/sqrt(k_dim) q-scale, steps the
/// fused kernel with the state updated in place, and reshapes y back to (1, 1, v_heads, v_dim).
#[cfg(feature = "metal")]
fn recurrence_metal_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (b, _s, nh, kd) = q.dims4()?;
    let vd = v.dim(D::Minus1)?;
    let bh = b * nh;
    let scale = 1.0 / (kd as f64).sqrt();
    let q = (q.reshape((bh, kd))?.to_dtype(DType::F32)? * scale)?.contiguous()?;
    let k = k.reshape((bh, kd))?.to_dtype(DType::F32)?.contiguous()?;
    let v = v.reshape((bh, vd))?.to_dtype(DType::F32)?.contiguous()?;
    let g = g.reshape(bh)?.to_dtype(DType::F32)?.contiguous()?;
    let beta = beta.reshape(bh)?.to_dtype(DType::F32)?.contiguous()?;
    let mut s = state.reshape((bh, kd, vd))?.contiguous()?;
    let y = crate::metal::gdn::gated_delta_rule_step_metal(&q, &k, &v, &g, &beta, &mut s)?;
    *state = s.reshape((b, nh, kd, vd))?.to_dtype(state.dtype())?;
    y.reshape((b, 1, nh, vd))
}

/// Portable f32 reference scan. Serves CPU, Vulkan, and any backend without a fused kernel.
fn recurrence_portable(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let k_head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (k_head_dim as f64).sqrt();

    // Transpose to (batch, heads, seq, dim) and cast to f32
    let q = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    // g, beta: (batch, seq, heads) -> (batch, heads, seq)
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let mut s = state.to_dtype(DType::F32)?;
    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        // q_t, k_t: (batch, heads, k_dim); v_t: (batch, heads, v_dim)
        let q_t = q.i((.., .., i, ..))?;
        let k_t = k.i((.., .., i, ..))?;
        let v_t = v.i((.., .., i, ..))?;
        // g_t, beta_t: (batch, heads)
        let g_t = g.i((.., .., i))?;
        let beta_t = beta.i((.., .., i))?;

        // s = s * exp(g_t)
        let decay = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        s = s.broadcast_mul(&decay)?;

        // kv_mem = (s * k_t[:,:,:,None]).sum(dim=2) -> (batch, heads, v_dim)
        let k_exp = k_t.unsqueeze(D::Minus1)?; // (batch, heads, k_dim, 1)
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(2)?;

        // delta = (v_t - kv_mem) * beta_t[:,:,None]
        let beta_exp = beta_t.unsqueeze(D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_exp)?;

        // s = s + k_t[:,:,:,None] * delta[:,:,None,:]
        let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
        s = (s + outer)?;

        // y_t = (s * q_t[:,:,:,None]).sum(dim=2) -> (batch, heads, v_dim)
        let q_exp = q_t.unsqueeze(D::Minus1)?;
        let y_t = s.broadcast_mul(&q_exp)?.sum(2)?;

        outputs.push(y_t);
    }

    *state = s.to_dtype(state.dtype())?;

    // Stack: (batch, heads, v_dim) * seq -> (batch, heads, seq, v_dim)
    let out = Tensor::stack(&outputs, 2)?;
    // Transpose back to (batch, seq, heads, v_dim)
    out.transpose(1, 2)?.contiguous()?.to_dtype(dtype)
}

// ====================== Gated Delta Net layer ======================

/// The two checkpoint layouts of the delta-net input projections, in the form each one is usable in.
pub enum GdnInProj {
    /// One interleaved grouped-head matrix per pair, which only exists dense.
    Merged { qkvz: Linear, ba: Linear },
    /// The HF section-major projections, kept quantized because nothing has to be re-indexed.
    Split {
        qkv: Arc<dyn QuantMethod>,
        z: Arc<dyn QuantMethod>,
        b: Arc<dyn QuantMethod>,
        a: Arc<dyn QuantMethod>,
    },
}

pub struct GatedDeltaNet {
    pub in_proj: GdnInProj,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

/// Whether to try merged weight names first or separate HF names with fallback.
pub enum GdnWeightMode {
    /// Only load merged weight names (in_proj_qkvz, in_proj_ba)
    MergedOnly,
    /// Try merged first, fall back to separate HF names (in_proj_qkv + in_proj_z, in_proj_b + in_proj_a)
    MergedWithFallback,
}

/// Rows produced by the conv state spliced onto the left of a continuation are context, not output.
fn trim_carried(out: &Tensor, carried: usize) -> Result<Tensor> {
    if carried == 0 {
        return Ok(out.clone());
    }
    let len = out.dim(1)?;
    out.narrow(1, carried, len - carried)
}

impl GatedDeltaNet {
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &dyn GdnConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<hanzo_quant::Comm>,
        weight_mode: GdnWeightMode,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };

        let num_k_heads = cfg.linear_num_key_heads();
        let num_v_heads = cfg.linear_num_value_heads();
        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim();
        let hidden_size = cfg.hidden_size();

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        // ISQ stages vb_la on the CPU and only moves what it claims; the input projections are not
        // ISQ targets and a QuantMethod cannot be moved after loading, so they load on the device.
        let vb_proj = mapper.set_device(layer_idx, vb.pp("linear_attn"), false);

        // ModelOpt ships in_proj_* as per-tensor FP8, which a raw get() hands to a matmul as fp8
        // bytes. Everything here goes through the quantized loader instead.
        let proj = |name: &str, out_dim: usize| -> Result<Arc<dyn QuantMethod>> {
            hanzo_quant::linear_no_bias(
                hidden_size,
                out_dim,
                cfg.quantization_config(),
                vb_proj.pp(name),
            )
        };
        let dense = |name: &str, out_dim: usize| -> Result<Linear> {
            Ok(Linear::new(
                proj(name, out_dim)?
                    .dequantize_w()?
                    .to_dtype(vb_la.dtype())?,
                None,
            ))
        };

        let merged = match weight_mode {
            GdnWeightMode::MergedOnly => true,
            GdnWeightMode::MergedWithFallback => vb_la.contains_tensor("in_proj_qkvz.weight"),
        };
        let in_proj = if merged {
            GdnInProj::Merged {
                qkvz: dense("in_proj_qkvz", key_dim * 2 + value_dim * 2)?,
                ba: dense("in_proj_ba", num_v_heads * 2)?,
            }
        } else {
            GdnInProj::Split {
                qkv: proj("in_proj_qkv", key_dim * 2 + value_dim)?,
                z: proj("in_proj_z", value_dim)?,
                b: proj("in_proj_b", num_v_heads)?,
                a: proj("in_proj_a", num_v_heads)?,
            }
        };

        let conv_dim = key_dim * 2 + value_dim;
        let mut conv1d_weight = vb_la.get((conv_dim, 1, conv_kernel_size), "conv1d.weight")?;
        let mut dt_bias = vb_la.get(num_v_heads, "dt_bias")?;
        let mut a_log = vb_la.get(num_v_heads, "A_log")?;

        if let Some(ref target_dev) = isq_target_device {
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
        }

        let norm = RmsNormGated::new(
            head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;

        let out_proj = RowParallelLayer::new(
            value_dim,
            hidden_size,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        })
    }

    /// Records what `load` reads back, so a UQFF residual round trips into the variant it came from.
    pub fn add_residual_tensors(&self, uvb_la: &UnVarBuilder) {
        match &self.in_proj {
            GdnInProj::Merged { qkvz, ba } => {
                uvb_la
                    .pp("in_proj_qkvz")
                    .add_tensor("weight", qkvz.weight().clone());
                uvb_la
                    .pp("in_proj_ba")
                    .add_tensor("weight", ba.weight().clone());
            }
            GdnInProj::Split { qkv, z, b, a } => {
                uvb_la.pp("in_proj_qkv").add(qkv);
                uvb_la.pp("in_proj_z").add(z);
                uvb_la.pp("in_proj_b").add(b);
                uvb_la.pp("in_proj_a").add(a);
            }
        }
        uvb_la.add_tensor("conv1d.weight", self.conv1d_weight.clone());
        uvb_la.add_tensor("dt_bias", self.dt_bias.clone());
        uvb_la.add_tensor("A_log", self.a_log.clone());
        uvb_la
            .pp("norm")
            .add_tensor("weight", self.norm.weight.clone());
    }

    /// (q, k, v) flat over key_dim/value_dim, z as (b, s, v_heads, head_v_dim), b and a as (b, s, v_heads).
    fn project_in(&self, x: &Tensor) -> Result<[Tensor; 6]> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;
        match &self.in_proj {
            GdnInProj::Merged { qkvz, ba } => {
                let v_group = v_per_group * self.head_v_dim;
                let group_size_qkvz = 2 * self.head_k_dim + 2 * v_group;
                let mixed_qkvz = qkvz.forward(x)?.reshape((
                    batch_size,
                    seq_len,
                    self.num_k_heads,
                    group_size_qkvz,
                ))?;
                let mixed_ba = ba.forward(x)?.reshape((
                    batch_size,
                    seq_len,
                    self.num_k_heads,
                    2 * v_per_group,
                ))?;
                Ok([
                    mixed_qkvz.narrow(D::Minus1, 0, self.head_k_dim)?.reshape((
                        batch_size,
                        seq_len,
                        self.key_dim,
                    ))?,
                    mixed_qkvz
                        .narrow(D::Minus1, self.head_k_dim, self.head_k_dim)?
                        .reshape((batch_size, seq_len, self.key_dim))?,
                    mixed_qkvz
                        .narrow(D::Minus1, 2 * self.head_k_dim, v_group)?
                        .reshape((batch_size, seq_len, self.value_dim))?,
                    mixed_qkvz
                        .narrow(D::Minus1, 2 * self.head_k_dim + v_group, v_group)?
                        .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?,
                    mixed_ba.narrow(D::Minus1, 0, v_per_group)?.reshape((
                        batch_size,
                        seq_len,
                        self.num_v_heads,
                    ))?,
                    mixed_ba
                        .narrow(D::Minus1, v_per_group, v_per_group)?
                        .reshape((batch_size, seq_len, self.num_v_heads))?,
                ])
            }
            GdnInProj::Split { qkv, z, b, a } => {
                let qkv = qkv.forward(x)?;
                Ok([
                    qkv.narrow(D::Minus1, 0, self.key_dim)?,
                    qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?,
                    qkv.narrow(D::Minus1, 2 * self.key_dim, self.value_dim)?,
                    z.forward(x)?.reshape((
                        batch_size,
                        seq_len,
                        self.num_v_heads,
                        self.head_v_dim,
                    ))?,
                    b.forward(x)?,
                    a.forward(x)?,
                ])
            }
        }
    }

    pub fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        let [q, k, v_flat, z, b, a] = self.project_in(x)?;

        // 1. Concatenate q, k, v for conv1d
        let mixed_qkv = Tensor::cat(&[&q, &k, &v_flat], D::Minus1)?;

        cache.trail_conv(&mixed_qkv)?;

        // 2. Apply causal conv1d (includes silu activation)
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // 3. Split back after conv and reshape to per-head
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 4. Compute beta and g
        let (beta, g) = {
            #[cfg(feature = "cuda")]
            {
                if b.device().is_cuda() {
                    let b_flat = b.contiguous()?.flatten_all()?;
                    let a_flat = a.contiguous()?.flatten_all()?;
                    let a_log_f32 = self.a_log.to_dtype(DType::F32)?.contiguous()?;
                    let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?.contiguous()?;
                    let (beta_flat, g_flat) = crate::cuda::gdn::fused_gdn_gating_cuda(
                        &b_flat,
                        &a_flat,
                        &a_log_f32,
                        &dt_bias_f32,
                    )?;
                    let shape = b.shape();
                    (beta_flat.reshape(shape)?, g_flat.reshape(shape)?)
                } else {
                    self.compute_beta_g_cpu(&b, &a, dtype)?
                }
            }
            #[cfg(feature = "metal")]
            {
                if b.device().is_metal() {
                    let b_flat = b.contiguous()?.flatten_all()?;
                    let a_flat = a.contiguous()?.flatten_all()?;
                    let a_log_f32 = self.a_log.to_dtype(DType::F32)?.contiguous()?;
                    let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?.contiguous()?;
                    let (beta_flat, g_flat) = crate::metal::gdn::fused_gdn_gating_metal(
                        &b_flat,
                        &a_flat,
                        &a_log_f32,
                        &dt_bias_f32,
                    )?;
                    let shape = b.shape();
                    (beta_flat.reshape(shape)?, g_flat.reshape(shape)?)
                } else {
                    self.compute_beta_g_cpu(&b, &a, dtype)?
                }
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                self.compute_beta_g_cpu(&b, &a, dtype)?
            }
        };

        // 5. If num_v_heads > num_k_heads, repeat_interleave q and k
        let (q, k) = if v_per_group > 1 {
            let q = q
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // 6. L2-normalize q and k
        let q = l2_norm(&q, 1e-6)?;
        let k = l2_norm(&k, 1e-6)?;

        // 7. Apply recurrence
        let y = cache.recurrence(&q, &k, &v, &g, &beta)?;

        cache.seqlen_offset += seq_len;

        // 8. Apply RMSNormGated
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.value_dim))?;

        // 9. Output projection
        let y_proj = y;
        let res = self.out_proj.forward(&y_proj)?;
        Ok(res)
    }

    fn compute_beta_g_cpu(&self, b: &Tensor, a: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let beta = hanzo_nn::ops::sigmoid(b)?;
        let a_f = a.to_dtype(DType::F32)?;
        let dt_bias_expanded = self
            .dt_bias
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let g = self
            .a_log
            .to_dtype(DType::F32)?
            .exp()?
            .neg()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?
            .to_dtype(dtype)?;
        Ok((beta, g))
    }

    /// Single-step causal conv1d update for decode.
    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;

        // Native Vulkan single decode step: conv_state stays in VRAM, no per-token readback.
        if x.device().is_vulkan() && x.dtype() == DType::F32 && x.dim(0)? == 1 && x.dim(1)? == 1 {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let conv_dim = weight.dim(0)?;
            let x_flat = x.reshape(conv_dim)?.contiguous()?;
            let mut s = cache
                .conv_state
                .reshape((conv_dim, self.conv_kernel_size))?;
            let out = crate::vulkan::gdn::gdn_conv1d_step_vulkan(&mut s, &x_flat, &weight)?;
            cache.conv_state = s.reshape((1, conv_dim, self.conv_kernel_size))?;
            return out.reshape((1, 1, conv_dim));
        }

        let x_t = x.transpose(1, 2)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let conv_state = cache.conv_state.contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &conv_state,
                self.conv_kernel_size,
                true,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        #[cfg(feature = "metal")]
        if x_t.device().is_metal() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let conv_state = cache.conv_state.contiguous()?;
            let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
                &x_t,
                &weight,
                &conv_state,
                true,
                self.conv_kernel_size,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        // CPU fallback
        let state_len = cache.conv_state.dim(2)?;
        let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = self
            .conv1d_weight
            .squeeze(1)?
            .to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - self.conv_kernel_size, self.conv_kernel_size)?;
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }

    /// Full sequence causal conv1d for prefill.
    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        // The full kernel has no conv_state argument and zero-pads its left edge, which is right
        // only at the start of a sequence. A continuation (a later prefill chunk, or a speculative
        // replay) must see the previous tokens, so splice them on and drop their outputs after.
        let carried = if cache.seqlen_offset > 0 {
            self.conv_kernel_size - 1
        } else {
            0
        };
        let x_t = if carried > 0 {
            let left = cache
                .conv_state
                .narrow(D::Minus1, 1, carried)?
                .to_dtype(x.dtype())?;
            Tensor::cat(&[&left, &x.transpose(1, 2)?], D::Minus1)?.contiguous()?
        } else {
            x.transpose(1, 2)?.contiguous()?
        };
        let seq_len = seq_len + carried;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                false,
            )?;
            cache.conv_state = new_conv_state;
            return trim_carried(&output.transpose(1, 2)?, carried);
        }

        #[cfg(feature = "metal")]
        if x_t.device().is_metal() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
                &x_t,
                &weight,
                &cache.conv_state,
                false,
                self.conv_kernel_size,
            )?;
            cache.conv_state = new_conv_state;
            return trim_carried(&output.transpose(1, 2)?, carried);
        }

        // CPU fallback
        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        cache.conv_state = if pad_width > 0 {
            let zeros =
                Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[zeros, x_t.clone()], 2)?
        } else {
            x_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };

        let padded_t = Tensor::cat(
            &[
                Tensor::zeros(
                    (batch_size, conv_dim, self.conv_kernel_size - 1),
                    x_t.dtype(),
                    x_t.device(),
                )?,
                x_t,
            ],
            2,
        )?;

        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

        let mut conv_outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        trim_carried(&out.transpose(1, 2)?, carried)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::kv_cache::RecurrentLayerConfig;
    use hanzo_quant::{QuantMethodConfig, UnquantLinear};

    fn synthetic(n: usize, seed: usize, dev: &Device) -> Result<Tensor> {
        let v = (0..n)
            .map(|i| (((i * 2654435761 + seed * 40503) % 1009) as f32 / 504.0) - 1.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(v, n, dev)
    }

    fn unquant(w: Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(w, None)),
        )?))
    }

    /// vLLM `RMSNormGated.forward_static` with `norm_before_gate=True` and one group, in f64
    /// (`layernorm.py:243-269`): each row is `x * rsqrt(mean(x²) + eps) * w * act(z)`.
    fn gated_norm_reference(x: &[f64], z: &[f64], w: &[f64], eps: f64, sigmoid: bool) -> Vec<f64> {
        let n = w.len();
        let mut out = Vec::with_capacity(x.len());
        for (xr, zr) in x.chunks(n).zip(z.chunks(n)) {
            let variance = xr.iter().map(|v| v * v).sum::<f64>() / n as f64;
            let inv = 1.0 / (variance + eps).sqrt();
            for ((a, g), wj) in xr.iter().zip(zr).zip(w) {
                let s = 1.0 / (1.0 + (-g).exp());
                let act = if sigmoid { s } else { g * s };
                out.push(a * inv * wj * act);
            }
        }
        out
    }

    /// Both gates against the reference: `from_weight` gates with silu, `.sigmoid()` with sigmoid.
    #[test]
    fn rms_norm_gated_matches_reference() -> Result<()> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let dev = Device::Cpu;
        let (rows, n, eps) = (6usize, 128usize, 1e-6);
        let mut rng = StdRng::seed_from_u64(0x676e_6f72);
        let mut draw = |len: usize, scale: f32| -> Vec<f32> {
            (0..len).map(|_| rng.random_range(-scale..scale)).collect()
        };
        let (x, z, w) = (draw(rows * n, 2.0), draw(rows * n, 4.0), draw(n, 1.5));
        let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<_>>();

        for sigmoid in [false, true] {
            let norm = RmsNormGated::from_weight(Tensor::from_vec(w.clone(), n, &dev)?, eps);
            let norm = if sigmoid { norm.sigmoid() } else { norm };
            let got = norm
                .forward(
                    &Tensor::from_vec(x.clone(), (rows, n), &dev)?,
                    &Tensor::from_vec(z.clone(), (rows, n), &dev)?,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = gated_norm_reference(&wide(&x), &wide(&z), &wide(&w), eps, sigmoid);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, r)| (f64::from(*g) - r).abs() / r.abs().max(1e-3))
                .fold(0f64, f64::max);
            assert!(
                worst < 1e-5,
                "sigmoid={sigmoid}: max relative error {worst:.3e}"
            );
        }
        Ok(())
    }

    // The split projections and the merged grouped-head matrix must be the same linear map. The merge
    // recipe written out here is the HF layout's definition, not a call into the code under test.
    #[test]
    fn gdn_split_in_proj_matches_merged() -> Result<()> {
        let dev = Device::Cpu;
        let (num_k_heads, num_v_heads, head_k_dim, head_v_dim) = (2usize, 4usize, 6usize, 8usize);
        let (hidden, conv_kernel_size) = (10usize, 4usize);
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let v_per_group = num_v_heads / num_k_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let rows = |n: usize, seed: usize| -> Result<Tensor> {
            synthetic(n * hidden, seed, &dev)?.reshape((n, hidden))
        };

        let qkv_w = rows(key_dim * 2 + value_dim, 1)?;
        let z_w = rows(value_dim, 2)?;
        let b_w = rows(num_v_heads, 3)?;
        let a_w = rows(num_v_heads, 4)?;

        let group = |t: &Tensor, per_head: usize| -> Result<Tensor> {
            t.reshape((num_k_heads, per_head, hidden))
        };
        let qkvz_w = Tensor::cat(
            &[
                group(&qkv_w.narrow(0, 0, key_dim)?, head_k_dim)?,
                group(&qkv_w.narrow(0, key_dim, key_dim)?, head_k_dim)?,
                group(
                    &qkv_w.narrow(0, key_dim * 2, value_dim)?,
                    v_per_group * head_v_dim,
                )?,
                group(&z_w, v_per_group * head_v_dim)?,
            ],
            1,
        )?
        .reshape((key_dim * 2 + value_dim * 2, hidden))?;
        let ba_w = Tensor::cat(&[group(&b_w, v_per_group)?, group(&a_w, v_per_group)?], 1)?
            .reshape((num_v_heads * 2, hidden))?;

        let conv1d_weight = synthetic(conv_dim * conv_kernel_size, 5, &dev)?.reshape((
            conv_dim,
            1,
            conv_kernel_size,
        ))?;
        let dt_bias = synthetic(num_v_heads, 6, &dev)?;
        let a_log = synthetic(num_v_heads, 7, &dev)?;
        let norm_weight = synthetic(head_v_dim, 8, &dev)?;
        let out_w = synthetic(hidden * value_dim, 9, &dev)?.reshape((hidden, value_dim))?;

        let build = |in_proj: GdnInProj| -> Result<GatedDeltaNet> {
            Ok(GatedDeltaNet {
                in_proj,
                conv1d_weight: conv1d_weight.clone(),
                dt_bias: dt_bias.clone(),
                a_log: a_log.clone(),
                norm: RmsNormGated::from_weight(norm_weight.clone(), 1e-6),
                out_proj: unquant(out_w.clone())?,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_kernel_size,
                key_dim,
                value_dim,
            })
        };
        let merged = build(GdnInProj::Merged {
            qkvz: Linear::new(qkvz_w, None),
            ba: Linear::new(ba_w, None),
        })?;
        let split = build(GdnInProj::Split {
            qkv: unquant(qkv_w)?,
            z: unquant(z_w)?,
            b: unquant(b_w)?,
            a: unquant(a_w)?,
        })?;

        let fresh_cache = || -> Result<GdnLayerCache> {
            Ok(GdnLayerCache {
                conv_state: Tensor::zeros((1, conv_dim, conv_kernel_size), DType::F32, &dev)?,
                recurrent_state: Tensor::zeros(
                    (1, num_v_heads, head_k_dim, head_v_dim),
                    DType::F32,
                    &dev,
                )?,
                seqlen_offset: 0,
                trail: None,
            })
        };
        let mut merged_cache = fresh_cache()?;
        let mut split_cache = fresh_cache()?;
        let mut worst = 0f32;
        // Prefill then decode: the second step reads the conv and recurrent state the first one wrote.
        for (step, seq_len) in [5usize, 1].into_iter().enumerate() {
            let x = synthetic(seq_len * hidden, 20 + step, &dev)?.reshape((1, seq_len, hidden))?;
            let ym = merged
                .forward(&x, &mut merged_cache)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ys = split
                .forward(&x, &mut split_cache)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(ym.len(), ys.len());
            worst = ym
                .iter()
                .zip(&ys)
                .fold(worst, |acc, (m, s)| acc.max((m - s).abs()));
        }
        eprintln!("[gdn split-vs-merged] max_abs={worst:.3e}");
        assert!(worst < 1e-5, "split != merged, max_abs={worst}");
        Ok(())
    }

    /// A prompt served in chunks must equal the same prompt served whole. The full conv kernel has
    /// no conv_state argument and zero-pads its left edge, so before the carry a second chunk
    /// convolved its first kernel_size-1 rows against zeros. Chunked prefill is the live path: a
    /// long prompt arrives in 4096-token pieces.
    #[test]
    fn gdn_chunked_prefill_matches_contiguous() -> Result<()> {
        let dev = Device::Cpu;
        let (num_k_heads, num_v_heads, head_k_dim, head_v_dim) = (2usize, 4usize, 6usize, 8usize);
        let (hidden, conv_kernel_size) = (10usize, 4usize);
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let v_per_group = num_v_heads / num_k_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let qkvz_w = synthetic((key_dim * 2 + value_dim * 2) * hidden, 31, &dev)?
            .reshape((key_dim * 2 + value_dim * 2, hidden))?;
        let ba_w =
            synthetic(num_v_heads * 2 * hidden, 32, &dev)?.reshape((num_v_heads * 2, hidden))?;
        let gdn = GatedDeltaNet {
            in_proj: GdnInProj::Merged {
                qkvz: Linear::new(qkvz_w, None),
                ba: Linear::new(ba_w, None),
            },
            conv1d_weight: synthetic(conv_dim * conv_kernel_size, 33, &dev)?.reshape((
                conv_dim,
                1,
                conv_kernel_size,
            ))?,
            dt_bias: synthetic(num_v_heads, 34, &dev)?,
            a_log: synthetic(num_v_heads, 35, &dev)?,
            norm: RmsNormGated::from_weight(synthetic(head_v_dim, 36, &dev)?, 1e-6),
            out_proj: unquant(
                synthetic(hidden * value_dim, 37, &dev)?.reshape((hidden, value_dim))?,
            )?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        };
        let _ = v_per_group;

        let fresh = || -> Result<GdnLayerCache> {
            Ok(GdnLayerCache {
                conv_state: Tensor::zeros((1, conv_dim, conv_kernel_size), DType::F32, &dev)?,
                recurrent_state: Tensor::zeros(
                    (1, num_v_heads, head_k_dim, head_v_dim),
                    DType::F32,
                    &dev,
                )?,
                seqlen_offset: 0,
                trail: None,
            })
        };

        let total = 8usize;
        let x = synthetic(total * hidden, 38, &dev)?.reshape((1, total, hidden))?;

        let mut whole_cache = fresh()?;
        let whole = gdn.forward(&x, &mut whole_cache)?;

        // The split lands past kernel_size, so the second chunk's left edge is real context.
        let split_at = 5usize;
        let mut chunk_cache = fresh()?;
        let first = x.narrow(1, 0, split_at)?;
        gdn.forward(&first, &mut chunk_cache)?;
        chunk_cache.seqlen_offset += split_at;
        let second = x.narrow(1, split_at, total - split_at)?;
        let tail = gdn.forward(&second, &mut chunk_cache)?;

        let want: Vec<f32> = whole
            .narrow(1, split_at, total - split_at)?
            .flatten_all()?
            .to_vec1()?;
        let got: Vec<f32> = tail.flatten_all()?.to_vec1()?;
        assert_eq!(want.len(), got.len());
        let worst = want
            .iter()
            .zip(&got)
            .fold(0f32, |acc, (w, g)| acc.max((w - g).abs()));
        eprintln!("[gdn chunked-vs-contiguous] max_abs={worst:.3e}");
        assert!(
            worst < 1e-5,
            "chunked prefill != contiguous, max_abs={worst}"
        );
        Ok(())
    }

    /// A small merged-projection layer on synthetic weights, with the shapes its state takes.
    struct Tiny {
        gdn: GatedDeltaNet,
        hidden: usize,
        conv_dim: usize,
        conv_kernel_size: usize,
        state_dims: [usize; 3],
    }

    fn tiny_gdn(dev: &Device) -> Result<Tiny> {
        let (num_k_heads, num_v_heads, head_k_dim, head_v_dim) = (2usize, 4usize, 6usize, 8usize);
        let (hidden, conv_kernel_size) = (10usize, 4usize);
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let gdn = GatedDeltaNet {
            in_proj: GdnInProj::Merged {
                qkvz: Linear::new(
                    synthetic((key_dim * 2 + value_dim * 2) * hidden, 51, dev)?
                        .reshape((key_dim * 2 + value_dim * 2, hidden))?,
                    None,
                ),
                ba: Linear::new(
                    synthetic(num_v_heads * 2 * hidden, 52, dev)?
                        .reshape((num_v_heads * 2, hidden))?,
                    None,
                ),
            },
            conv1d_weight: synthetic(conv_dim * conv_kernel_size, 53, dev)?.reshape((
                conv_dim,
                1,
                conv_kernel_size,
            ))?,
            dt_bias: synthetic(num_v_heads, 54, dev)?,
            a_log: synthetic(num_v_heads, 55, dev)?,
            norm: RmsNormGated::from_weight(synthetic(head_v_dim, 56, dev)?, 1e-6),
            out_proj: unquant(
                synthetic(hidden * value_dim, 57, dev)?.reshape((hidden, value_dim))?,
            )?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        };
        Ok(Tiny {
            gdn,
            hidden,
            conv_dim,
            conv_kernel_size,
            state_dims: [num_v_heads, head_k_dim, head_v_dim],
        })
    }

    /// Speculative rollback restores a checkpoint and replays the accepted prefix in one forward.
    /// That wide continuation takes the full conv path plus the carried state, while decoding the same
    /// tokens one at a time takes the update path, so the two are independent implementations. Width 2
    /// sits below the kernel, where a wrong saved window hides from any check on the output alone.
    #[test]
    fn gdn_replay_after_rewind_matches_stepwise() -> Result<()> {
        let dev = Device::Cpu;
        let Tiny {
            gdn,
            hidden,
            conv_dim,
            conv_kernel_size,
            state_dims: [num_v_heads, head_k_dim, head_v_dim],
        } = tiny_gdn(&dev)?;
        let snapshot = |c: &GdnLayerCache| GdnLayerCache {
            conv_state: c.conv_state.clone(),
            recurrent_state: c.recurrent_state.clone(),
            seqlen_offset: c.seqlen_offset,
            trail: None,
        };
        let max_abs = |a: &Tensor, b: &Tensor| -> Result<f32> {
            (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
        };

        let (prompt_len, drafted, accepted) = (5usize, 4usize, 2usize);
        let all = synthetic((prompt_len + drafted) * hidden, 58, &dev)?.reshape((
            1,
            prompt_len + drafted,
            hidden,
        ))?;
        let draft = all.narrow(1, prompt_len, drafted)?;

        let mut cache = GdnLayerCache {
            conv_state: Tensor::zeros((1, conv_dim, conv_kernel_size), DType::F32, &dev)?,
            recurrent_state: Tensor::zeros(
                (1, num_v_heads, head_k_dim, head_v_dim),
                DType::F32,
                &dev,
            )?,
            seqlen_offset: 0,
            trail: None,
        };
        gdn.forward(&all.narrow(1, 0, prompt_len)?, &mut cache)?;
        cache.seqlen_offset = prompt_len;
        let checkpoint = snapshot(&cache);

        // advance over the whole draft, then roll back and replay only what was accepted
        gdn.forward(&draft, &mut cache)?;
        let mut replay = snapshot(&checkpoint);
        let replayed = gdn.forward(&draft.narrow(1, 0, accepted)?, &mut replay)?;

        let mut truth = snapshot(&checkpoint);
        let mut rows = Vec::with_capacity(accepted);
        for i in 0..accepted {
            truth.seqlen_offset = prompt_len + i;
            rows.push(gdn.forward(&draft.narrow(1, i, 1)?, &mut truth)?);
        }
        let stepwise = Tensor::cat(&rows, 1)?;

        let out = max_abs(&replayed, &stepwise)?;
        let conv = max_abs(&replay.conv_state, &truth.conv_state)?;
        let rec = max_abs(&replay.recurrent_state, &truth.recurrent_state)?;
        eprintln!(
            "[gdn replay-vs-stepwise] out={out:.3e} conv_state={conv:.3e} recurrent={rec:.3e}"
        );
        assert!(out < 1e-5, "replayed output != stepwise, max_abs={out}");
        assert!(
            conv < 1e-5,
            "replayed conv_state != stepwise, max_abs={conv}"
        );
        assert!(
            rec < 1e-5,
            "replayed recurrent_state != stepwise, max_abs={rec}"
        );
        Ok(())
    }

    /// A verify runs the anchor and every draft through the layer. Rejecting the tail and rewinding
    /// must leave the layer where plain decoding of the accepted tokens alone would: the same state,
    /// and the same output for every token after. Two sequences of different lengths share the
    /// forward, sit in slots that differ from their batch rows, and reject different amounts, so the
    /// drafts here are wrong on purpose.
    #[test]
    fn gdn_rewind_after_verify_matches_plain_decode() -> Result<()> {
        let dev = Device::Cpu;
        let Tiny {
            gdn,
            hidden,
            conv_dim,
            conv_kernel_size,
            state_dims,
        } = tiny_gdn(&dev)?;
        let max_abs = |a: &Tensor, b: &Tensor| -> Result<f32> {
            (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
        };
        let run = |pool: &mut RecurrentStatePool, slots: &[u32], x: &Tensor, trail: bool| {
            let indices = Tensor::from_vec(slots.to_vec(), slots.len(), &dev)?;
            forward_pooled(pool, PoolSlots::Many(&indices), 0, trail, |cache| {
                gdn.forward(x, cache)
            })
        };

        const VERIFY: usize = 4;
        // (slot, prompt length, tokens of the verify that were right). Batch order is `seqs` order.
        let seqs = [(1u32, 3usize, 3usize), (0u32, 5usize, 1usize)];
        let order: Vec<u32> = seqs.iter().map(|s| s.0).collect();
        let streams = seqs
            .iter()
            .map(|&(slot, prompt, _)| {
                synthetic((prompt + VERIFY) * hidden, 60 + slot as usize, &dev)?.reshape((
                    1,
                    prompt + VERIFY,
                    hidden,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let wrong = synthetic(seqs.len() * VERIFY * hidden, 70, &dev)?.reshape((
            seqs.len(),
            VERIFY,
            hidden,
        ))?;
        // The token each sequence truly decodes `step` tokens after its prompt.
        let truth = |row: usize, step: usize| streams[row].narrow(1, seqs[row].1 + step, 1);

        let prefilled = || -> Result<RecurrentStatePool> {
            let mut pool = RecurrentStatePool::new(
                RecurrentLayerConfig {
                    conv_dim,
                    conv_width: conv_kernel_size,
                    state_dims: state_dims.to_vec(),
                    conv_dtype: DType::F32,
                    state_dtype: DType::F32,
                },
                &dev,
            )?;
            assert_eq!((pool.allocate(), pool.allocate()), (Some(0), Some(1)));
            for (row, &(slot, prompt, _)) in seqs.iter().enumerate() {
                run(
                    &mut pool,
                    &[slot],
                    &streams[row].narrow(1, 0, prompt)?,
                    false,
                )?;
            }
            Ok(pool)
        };

        // Plain decode, one true token per step, keeping the outputs and the state after each.
        let mut plain = prefilled()?;
        let mut plain_out = Vec::with_capacity(VERIFY);
        let mut plain_state = Vec::with_capacity(VERIFY);
        for step in 0..VERIFY {
            let x = Tensor::cat(&[truth(0, step)?, truth(1, step)?], 0)?;
            plain_out.push(run(&mut plain, &order, &x, false)?);
            plain_state.push((plain.conv_state.copy()?, plain.recurrent_state.copy()?));
        }

        // One verify forward: each row is right for its first `kept` tokens, wrong after.
        let mut spec = prefilled()?;
        let rows = seqs
            .iter()
            .enumerate()
            .map(|(row, &(_, prompt, kept))| {
                Tensor::cat(
                    &[
                        streams[row].narrow(1, prompt, kept)?,
                        wrong.narrow(0, row, 1)?.narrow(1, kept, VERIFY - kept)?,
                    ],
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let verified = run(&mut spec, &order, &Tensor::cat(&rows, 0)?, true)?;

        for (row, &(slot, prompt, kept)) in seqs.iter().enumerate() {
            for step in 0..kept {
                let got = verified.narrow(0, row, 1)?.narrow(1, step, 1)?;
                let want = plain_out[step].narrow(0, row, 1)?;
                let err = max_abs(&got, &want)?;
                assert!(
                    err < 1e-5,
                    "verify row {row} step {step} != plain, max_abs={err}"
                );
            }

            spec.rewind(slot as usize, VERIFY - kept)?;

            assert_eq!(spec.get_seqlen_offset(slot as usize), prompt + kept);
            let (conv, recurrent) = &plain_state[kept - 1];
            let slot = slot as usize;
            let conv_err = max_abs(&spec.conv_state.i(slot)?, &conv.i(slot)?)?;
            let rec_err = max_abs(&spec.recurrent_state.i(slot)?, &recurrent.i(slot)?)?;
            eprintln!(
                "[gdn rewind] row {row} kept {kept}: conv={conv_err:.3e} recurrent={rec_err:.3e}"
            );
            assert!(
                conv_err < 1e-5,
                "rewound conv state != plain, max_abs={conv_err}"
            );
            assert!(
                rec_err < 1e-5,
                "rewound recurrent state != plain, max_abs={rec_err}"
            );
        }

        // The sequences now sit at different depths. Their next true tokens decode as plain did.
        let x = Tensor::cat(&[truth(0, seqs[0].2)?, truth(1, seqs[1].2)?], 0)?;
        let next = run(&mut spec, &order, &x, false)?;
        for (row, &(_, _, kept)) in seqs.iter().enumerate() {
            let err = max_abs(
                &next.narrow(0, row, 1)?,
                &plain_out[kept].narrow(0, row, 1)?,
            )?;
            assert!(
                err < 1e-5,
                "decode after rewind, row {row} != plain, max_abs={err}"
            );
        }
        Ok(())
    }

    // Scalar reimplementation of the Vulkan gdn_step.comp single-step math (one (bh, v) state column,
    // looping k). q is pre-scaled by the caller, exactly as gated_delta_rule_recurrence applies the
    // 1/sqrt(k_dim) scale internally and as our engine wrapper does before the kernel. State layout is
    // [bh][k][v] at k*v_dim + v, matching the CPU reference's (heads, k_dim, v_dim) contiguous order.
    #[allow(clippy::too_many_arguments)]
    fn gdn_step_scalar(
        q: &[f32],         // [bh, k]  (pre-scaled)
        k: &[f32],         // [bh, k]
        v: &[f32],         // [bh, v]
        g: &[f32],         // [bh]
        beta: &[f32],      // [bh]
        state: &mut [f32], // [bh, k, v]
        bh: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; bh * v_dim];
        for b in 0..bh {
            let qk_base = b * k_dim;
            let state_base = b * k_dim * v_dim;
            let decay = g[b].exp();
            let beta_t = beta[b];
            for v_idx in 0..v_dim {
                let v_t = v[b * v_dim + v_idx];
                let mut s = vec![0f32; k_dim];
                let mut kv_mem = 0f32;
                for j in 0..k_dim {
                    let sj = state[state_base + j * v_dim + v_idx] * decay;
                    s[j] = sj;
                    kv_mem += sj * k[qk_base + j];
                }
                let delta = (v_t - kv_mem) * beta_t;
                let mut y_t = 0f32;
                for j in 0..k_dim {
                    let sj = s[j] + k[qk_base + j] * delta;
                    state[state_base + j * v_dim + v_idx] = sj;
                    y_t += sj * q[qk_base + j];
                }
                out[b * v_dim + v_idx] = y_t;
            }
        }
        out
    }

    // The Vulkan single-step kernel must reproduce gated_delta_rule_recurrence for seq_len==1. This
    // pins the shader's arithmetic against the portable reference on the CPU (the shader is a literal
    // GLSL transliteration of gdn_step_scalar), independent of any GPU.
    #[test]
    fn gdn_step_matches_reference_seq1() -> Result<()> {
        let dev = Device::Cpu;
        let heads = 4usize;
        let k_dim = 6usize;
        let v_dim = 5usize;

        // Deterministic pseudo-random inputs in [-1, 1]-ish ranges.
        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed * 12345 + 7) % 2000) as f32 / 1000.0) - 1.0)
                .collect()
        };

        // Reference tensors: q,k (1,1,heads,k_dim); v (1,1,heads,v_dim); g,beta (1,1,heads);
        // state (1,heads,k_dim,v_dim).
        let q_v = gen(heads * k_dim, 1);
        let k_v = gen(heads * k_dim, 2);
        let v_v = gen(heads * v_dim, 3);
        let g_v: Vec<f32> = gen(heads, 4).iter().map(|x| x * 0.5 - 0.5).collect(); // g < 0 (decay<1)
        let beta_v: Vec<f32> = gen(heads, 5).iter().map(|x| (x + 1.0) * 0.5).collect(); // [0,1]
        let state_v = gen(heads * k_dim * v_dim, 6);

        let q = Tensor::from_vec(q_v.clone(), (1, 1, heads, k_dim), &dev)?;
        let k = Tensor::from_vec(k_v.clone(), (1, 1, heads, k_dim), &dev)?;
        let v = Tensor::from_vec(v_v.clone(), (1, 1, heads, v_dim), &dev)?;
        let g = Tensor::from_vec(g_v.clone(), (1, 1, heads), &dev)?;
        let beta = Tensor::from_vec(beta_v.clone(), (1, 1, heads), &dev)?;
        let mut state_ref = Tensor::from_vec(state_v.clone(), (1, heads, k_dim, v_dim), &dev)?;

        let y_ref = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_ref)?;
        let y_ref = y_ref.flatten_all()?.to_vec1::<f32>()?;
        let state_ref = state_ref.flatten_all()?.to_vec1::<f32>()?;

        // Shader-equivalent scalar path: q pre-scaled by 1/sqrt(k_dim) (the reference does this
        // internally; the engine wrapper does it before the kernel).
        let scale = 1.0 / (k_dim as f32).sqrt();
        let q_scaled: Vec<f32> = q_v.iter().map(|x| x * scale).collect();
        let mut state_shader = state_v.clone();
        let y_shader = gdn_step_scalar(
            &q_scaled,
            &k_v,
            &v_v,
            &g_v,
            &beta_v,
            &mut state_shader,
            heads,
            k_dim,
            v_dim,
        );

        for (a, b) in y_ref.iter().zip(y_shader.iter()) {
            assert!((a - b).abs() < 1e-5, "y mismatch: ref={a} shader={b}");
        }
        for (a, b) in state_ref.iter().zip(state_shader.iter()) {
            assert!((a - b).abs() < 1e-5, "state mismatch: ref={a} shader={b}");
        }
        Ok(())
    }

    // The Vulkan conv1d single-step kernel must reproduce causal_conv1d_update for seq_len==1 with a
    // conv_state of width k (drop oldest column, append x), including the silu. Pins that arithmetic
    // on the CPU; the shader is a literal transliteration.
    #[test]
    fn gdn_conv1d_step_matches_reference_seq1() -> Result<()> {
        let dev = Device::Cpu;
        let conv_dim = 7usize;
        let k = 4usize;

        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 2654435761 + seed * 40503 + 11) % 2000) as f32 / 1000.0) - 1.0)
                .collect()
        };
        let cs_v = gen(conv_dim * k, 1); // conv_state [conv_dim, k]
        let x_v = gen(conv_dim, 2); // new column [conv_dim]
        let w_v = gen(conv_dim * k, 3); // weight [conv_dim, k]

        // Reference: replicate causal_conv1d_update for seq=1 via tensor ops. conv_state (1,conv_dim,k),
        // x (1,1,conv_dim). window = [conv_state | x][:, :, 1..k+1]; out = silu(sum(window*weight)).
        let conv_state = Tensor::from_vec(cs_v.clone(), (1, conv_dim, k), &dev)?;
        let x_t = Tensor::from_vec(x_v.clone(), (1, conv_dim, 1), &dev)?;
        let weight = Tensor::from_vec(w_v.clone(), (conv_dim, k), &dev)?;
        let hidden = Tensor::cat(&[&conv_state, &x_t], 2)?; // (1, conv_dim, k+1)
        let window = hidden.narrow(2, 1, k)?; // (1, conv_dim, k)
        let out_ref = (window.clone() * weight.unsqueeze(0)?)?.sum(D::Minus1)?; // (1, conv_dim)
        let out_ref = hanzo_nn::ops::silu(&out_ref)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let new_state_ref = window.flatten_all()?.to_vec1::<f32>()?; // new conv_state == window

        // Shader-equivalent scalar path (matches gdn_conv1d_step.comp).
        let mut cs = cs_v.clone();
        let mut out_shader = vec![0f32; conv_dim];
        for c in 0..conv_dim {
            let base = c * k;
            let mut win = vec![0f32; k];
            for j in 0..k - 1 {
                win[j] = cs[base + j + 1];
            }
            win[k - 1] = x_v[c];
            let mut acc = 0f32;
            for j in 0..k {
                acc += win[j] * w_v[base + j];
            }
            out_shader[c] = acc / (1.0 + (-acc).exp());
            cs[base..(k + base)].copy_from_slice(&win[..k]);
        }

        for (a, b) in out_ref.iter().zip(out_shader.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "conv out mismatch: ref={a} shader={b}"
            );
        }
        for (a, b) in new_state_ref.iter().zip(cs.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "conv state mismatch: ref={a} shader={b}"
            );
        }
        Ok(())
    }

    // Tiny prefill (seq>1) reproduction of the recurrence shape contract on a given device. Built so a
    // human can repro the qwen35moe Vulkan `unexpected rank, expected: 2, got: 1 ([2048])` prompt-step
    // crash in seconds instead of loading a 22GB GGUF. Inputs are constructed on CPU (deterministic,
    // identical per device) then moved to `dev`, so the CPU and Vulkan runs are bit-for-bit comparable.
    fn run_gdn_recurrence_shapes(dev: &Device) -> Result<()> {
        let (batch, nvh, hkd, hvd, seq) = (1usize, 4usize, 8usize, 4usize, 3usize);

        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed * 12345 + 7) % 2000) as f32 / 1000.0) - 1.0)
                .collect()
        };
        let on = |v: Vec<f32>, shape: (usize, usize, usize, usize)| -> Result<Tensor> {
            Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(dev)
        };
        let on3 = |v: Vec<f32>, shape: (usize, usize, usize)| -> Result<Tensor> {
            Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(dev)
        };

        let q = on(gen(batch * seq * nvh * hkd, 1), (batch, seq, nvh, hkd))?;
        let k = on(gen(batch * seq * nvh * hkd, 2), (batch, seq, nvh, hkd))?;
        let v = on(gen(batch * seq * nvh * hvd, 3), (batch, seq, nvh, hvd))?;
        let g = on3(
            gen(batch * seq * nvh, 4)
                .iter()
                .map(|x| x * 0.5 - 0.5)
                .collect(),
            (batch, seq, nvh),
        )?;
        let beta = on3(
            gen(batch * seq * nvh, 5)
                .iter()
                .map(|x| (x + 1.0) * 0.5)
                .collect(),
            (batch, seq, nvh),
        )?;
        let mut state = Tensor::from_vec(
            gen(batch * nvh * hkd * hvd, 6),
            (batch, nvh, hkd, hvd),
            &Device::Cpu,
        )?
        .to_device(dev)?;
        let state_shape_before = state.dims().to_vec();

        let y = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state)?;

        assert_eq!(
            y.dims(),
            &[batch, seq, nvh, hvd],
            "recurrence output rank/shape wrong on {dev:?}"
        );
        assert_eq!(
            state.dims(),
            state_shape_before.as_slice(),
            "recurrence mutated state shape on {dev:?}"
        );
        // Mirror the qwen35moe post-recurrence reshape (model step 8): collapse to (tokens, head_v_dim)
        // then read the rank. This is where the GGUF forward consumes the output, so it catches a
        // collapsed/extra dim escaping the recurrence as well as a bug inside it.
        let y2 = y.reshape(((), hvd))?;
        assert_eq!(
            y2.dims().len(),
            2,
            "post-recurrence reshape lost a dim on {dev:?}"
        );
        assert_eq!(
            y2.dims(),
            &[batch * seq * nvh, hvd],
            "post-recurrence shape wrong on {dev:?}"
        );

        // Force a readback so any deferred Vulkan dispatch actually executes (lazy backends can defer
        // the bad op until the buffer is read); also pins that the result is finite.
        let host = y.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        assert_eq!(host.len(), batch * seq * nvh * hvd);
        assert!(
            host.iter().all(|x| x.is_finite()),
            "non-finite recurrence output on {dev:?}"
        );
        Ok(())
    }

    // CPU twin: must always pass. Directly comparable to the Vulkan test below.
    #[test]
    fn gdn_recurrence_cpu_shapes() -> Result<()> {
        run_gdn_recurrence_shapes(&Device::Cpu)
    }

    // `recurrence_portable` is the semantics of record for GDN, and hanzo-kernel's `gdn_scan` claims it
    // as its bit-exact target. That claim was only ever checked against the DSL's own oracle, so a drift
    // between the two repos could not be caught here. This closes the loop: same inputs, this crate's
    // scan vs `hanzo_kernel::gdn::gdn_scan_ref`, which `gdn_scan_cpu_bit_exact` gates the lowered kernel
    // against. Composing the two gives engine-portable == DSL-on-every-backend, which is the
    // precondition for GDN ever dispatching to the DSL (today ROCm has no fused arm and lands here, at
    // ~10 tensor launches per timestep).
    //
    // Layout: this crate takes (batch, seq, heads, dim) and scales q by 1/sqrt(k_dim) internally; the
    // oracle takes bh-major [bh, seq, dim] with bh = batch*heads and q pre-scaled. Hence the transpose
    // to (batch, heads, seq, dim) before flattening, and the explicit scale on the oracle's q.
    #[test]
    fn gdn_recurrence_portable_matches_dsl_oracle() -> Result<()> {
        let dev = Device::Cpu;
        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed * 12345 + 7) % 2000) as f32 / 1000.0) - 1.0)
                .collect()
        };
        let max_rel = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs() / x.abs().max(1e-4))
                .fold(0.0f32, f32::max)
        };
        // bh-major flatten of a (batch, seq, heads, ..) tensor, matching the oracle's indexing.
        let bh_major = |t: &Tensor| -> Result<Vec<f32>> {
            t.transpose(1, 2)?.contiguous()?.flatten_all()?.to_vec1()
        };
        // Decode, short prefill, a real prefill chunk at the shipping head dim, and non-square dims.
        for &(batch, nvh, hkd, hvd, seq) in &[
            (1usize, 4usize, 128usize, 128usize, 1usize),
            (1, 8, 128, 128, 7),
            (1, 4, 128, 128, 64),
            (2, 3, 48, 32, 40),
        ] {
            let q = Tensor::from_vec(
                gen(batch * seq * nvh * hkd, 1),
                (batch, seq, nvh, hkd),
                &dev,
            )?;
            let k = Tensor::from_vec(
                gen(batch * seq * nvh * hkd, 2),
                (batch, seq, nvh, hkd),
                &dev,
            )?;
            let v = Tensor::from_vec(
                gen(batch * seq * nvh * hvd, 3),
                (batch, seq, nvh, hvd),
                &dev,
            )?;
            // g < 0 (decay in (0,1]) and beta in [0,1] are the physical GDN gate ranges.
            let g = Tensor::from_vec(
                gen(batch * seq * nvh, 4)
                    .iter()
                    .map(|x| x * 0.5 - 0.5)
                    .collect::<Vec<_>>(),
                (batch, seq, nvh),
                &dev,
            )?;
            let beta = Tensor::from_vec(
                gen(batch * seq * nvh, 5)
                    .iter()
                    .map(|x| (x + 1.0) * 0.5)
                    .collect::<Vec<_>>(),
                (batch, seq, nvh),
                &dev,
            )?;
            let state0 = Tensor::from_vec(
                gen(batch * nvh * hkd * hvd, 6),
                (batch, nvh, hkd, hvd),
                &dev,
            )?;

            let mut s_port = state0.clone();
            let y_port = recurrence_portable(&q, &k, &v, &g, &beta, &mut s_port)?;

            let scale = 1.0f32 / (hkd as f32).sqrt();
            let q_ref: Vec<f32> = bh_major(&q)?.iter().map(|x| x * scale).collect();
            let mut s_ref = state0.flatten_all()?.to_vec1::<f32>()?;
            let y_ref = hanzo_kernel::gdn::gdn_scan_ref(
                &q_ref,
                &bh_major(&k)?,
                &bh_major(&v)?,
                &g.transpose(1, 2)?.contiguous()?.flatten_all()?.to_vec1()?,
                &beta
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1()?,
                &mut s_ref,
                batch * nvh,
                seq,
                hkd,
                hvd,
            );

            let ry = max_rel(&bh_major(&y_port)?, &y_ref);
            let rs = max_rel(&s_port.flatten_all()?.to_vec1::<f32>()?, &s_ref);
            eprintln!(
                "[gdn portable-vs-dsl-oracle] b{batch} h{nvh} k{hkd} v{hvd} s{seq}  y_rel={ry:.2e} state_rel={rs:.2e}"
            );
            assert!(
                ry < 1e-4 && rs < 1e-4,
                "portable!=dsl oracle b{batch} h{nvh} k{hkd} v{hvd} s{seq} y_rel={ry} state_rel={rs}"
            );
        }
        Ok(())
    }

    // The fused CUDA recurrence (cuda/gdn.cu: tiled decode kernel for seq<64, warp-per-column prefill
    // kernel for seq>=64) must match the portable ops-composed scan bit-for-bit within f32 reorder.
    // This is the correctness gate for the fused-vs-ops-composed path the GDN_FUSED_FALLBACK knob
    // selects. Covers decode (seq=1 -> tiled), short prefill (seq=7 -> tiled), and a real prefill chunk
    // (seq=64 -> warp), at the shipping head dim 128. Skips cleanly with no CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_recurrence_cuda_matches_portable() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skip: no cuda device");
            return Ok(());
        };
        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed * 12345 + 7) % 2000) as f32 / 1000.0) - 1.0)
                .collect()
        };
        let max_rel = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs() / x.abs().max(1e-4))
                .fold(0.0f32, f32::max)
        };
        for &(batch, nvh, hkd, hvd, seq) in &[
            (1usize, 4usize, 128usize, 128usize, 1usize),
            (1, 8, 128, 128, 7),
            (1, 4, 128, 128, 64),
        ] {
            let on4 = |v: Vec<f32>, s: (usize, usize, usize, usize)| -> Result<Tensor> {
                Tensor::from_vec(v, s, &Device::Cpu)?.to_device(&dev)
            };
            let on3 = |v: Vec<f32>, s: (usize, usize, usize)| -> Result<Tensor> {
                Tensor::from_vec(v, s, &Device::Cpu)?.to_device(&dev)
            };
            let q = on4(gen(batch * seq * nvh * hkd, 1), (batch, seq, nvh, hkd))?;
            let k = on4(gen(batch * seq * nvh * hkd, 2), (batch, seq, nvh, hkd))?;
            let v = on4(gen(batch * seq * nvh * hvd, 3), (batch, seq, nvh, hvd))?;
            let g = on3(
                gen(batch * seq * nvh, 4)
                    .iter()
                    .map(|x| x * 0.5 - 0.5)
                    .collect(),
                (batch, seq, nvh),
            )?;
            let beta = on3(
                gen(batch * seq * nvh, 5)
                    .iter()
                    .map(|x| (x + 1.0) * 0.5)
                    .collect(),
                (batch, seq, nvh),
            )?;
            let state0 = Tensor::from_vec(
                gen(batch * nvh * hkd * hvd, 6),
                (batch, nvh, hkd, hvd),
                &Device::Cpu,
            )?
            .to_device(&dev)?;

            let mut s_fused = state0.clone();
            let y_fused = recurrence_cuda(&q, &k, &v, &g, &beta, &mut s_fused)?;
            let mut s_port = state0.clone();
            let y_port = recurrence_portable(&q, &k, &v, &g, &beta, &mut s_port)?;

            let yf = y_fused.flatten_all()?.to_vec1::<f32>()?;
            let yp = y_port.flatten_all()?.to_vec1::<f32>()?;
            let sf = s_fused.flatten_all()?.to_vec1::<f32>()?;
            let sp = s_port.flatten_all()?.to_vec1::<f32>()?;
            let ry = max_rel(&yp, &yf);
            let rs = max_rel(&sp, &sf);
            eprintln!(
                "[gdn cuda-vs-portable] b{batch} h{nvh} k{hkd} v{hvd} s{seq}  y_rel={ry:.2e} state_rel={rs:.2e}"
            );
            assert!(
                ry < 1e-4 && rs < 1e-4,
                "fused!=portable b{batch} s{seq} y_rel={ry} state_rel={rs}"
            );
        }
        Ok(())
    }

    // Fast Vulkan reproduction of the qwen35moe prompt-step recurrence shape crash. Skips cleanly with
    // no GPU. Run on a ROCm box (evo) with:
    //   cargo test --features rocm -p hanzo-engine gdn_scan_rocm_matches_portable -- --nocapture
    #[test]
    #[cfg_attr(not(feature = "rocm"), ignore = "requires rocm feature")]
    fn gdn_scan_rocm_matches_portable() -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            let Ok(dev) = Device::new_rocm(0) else {
                eprintln!("skip: no rocm device");
                return Ok(());
            };
            return run_gdn_scan_rocm_matches_portable(&dev);
        }
        #[cfg(not(feature = "rocm"))]
        {
            eprintln!("skip: built without the rocm feature");
            Ok(())
        }
    }

    #[cfg(feature = "rocm")]
    fn run_gdn_scan_rocm_matches_portable(dev: &Device) -> Result<()> {
        let (batch, nvh, hkd, hvd, seq) = (1usize, 4usize, 128usize, 128usize, 32usize);
        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 1103515245 + seed * 12345 + 7) % 2000) as f32 / 1000.0) - 1.0)
                .collect()
        };
        let on = |v: Vec<f32>, shape: (usize, usize, usize, usize)| -> Result<Tensor> {
            Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(dev)
        };
        let on3 = |v: Vec<f32>, shape: (usize, usize, usize)| -> Result<Tensor> {
            Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(dev)
        };
        let q = on(gen(batch * seq * nvh * hkd, 1), (batch, seq, nvh, hkd))?;
        let k = on(gen(batch * seq * nvh * hkd, 2), (batch, seq, nvh, hkd))?;
        let v = on(gen(batch * seq * nvh * hvd, 3), (batch, seq, nvh, hvd))?;
        let g = on3(
            gen(batch * seq * nvh, 4)
                .iter()
                .map(|x| x * 0.5 - 0.5)
                .collect(),
            (batch, seq, nvh),
        )?;
        let beta = on3(
            gen(batch * seq * nvh, 5)
                .iter()
                .map(|x| (x + 1.0) * 0.5)
                .collect(),
            (batch, seq, nvh),
        )?;
        let cpu = |t: &Tensor| -> Result<Tensor> { t.to_device(&Device::Cpu) };
        let (qc, kc, vc, gc, bc) = (cpu(&q)?, cpu(&k)?, cpu(&v)?, cpu(&g)?, cpu(&beta)?);
        let sc = Tensor::from_vec(
            gen(batch * nvh * hkd * hvd, 6),
            (batch, nvh, hkd, hvd),
            &Device::Cpu,
        )?;
        let state_dev = sc.to_device(dev)?;

        let mut state_fused = state_dev.clone();
        let y_fused = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_fused)?;
        let mut state_ref = sc.clone();
        let y_ref = recurrence_portable(&qc, &kc, &vc, &gc, &bc, &mut state_ref)?;

        let a = y_fused.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let b = y_ref.to_vec1::<f32>()?;
        let max_rel = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs() / y.abs().max(1e-3))
            .fold(0f32, f32::max);
        assert!(
            max_rel < 1e-4,
            "gdn_scan_rocm vs portable max_rel {max_rel}"
        );

        let sf = state_fused.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let sr = state_ref.to_vec1::<f32>()?;
        let state_rel = sf
            .iter()
            .zip(&sr)
            .map(|(x, y)| (x - y).abs() / y.abs().max(1e-3))
            .fold(0f32, f32::max);
        assert!(
            state_rel < 1e-4,
            "gdn_scan_rocm state divergence {state_rel}"
        );
        Ok(())
    }

    // no GPU. Run on a Vulkan box with:
    //   cargo test --features vulkan -p hanzo-engine gdn_recurrence_vulkan_shapes -- --nocapture --include-ignored
    #[test]
    #[cfg_attr(not(feature = "vulkan"), ignore = "requires vulkan feature")]
    fn gdn_recurrence_vulkan_shapes() -> Result<()> {
        #[cfg(feature = "vulkan")]
        {
            let Ok(dev) = Device::new_vulkan(0) else {
                eprintln!("skip: no vulkan device");
                return Ok(());
            };
            return run_gdn_recurrence_shapes(&dev);
        }
        #[cfg(not(feature = "vulkan"))]
        {
            eprintln!("skip: built without the vulkan feature");
            Ok(())
        }
    }
}
