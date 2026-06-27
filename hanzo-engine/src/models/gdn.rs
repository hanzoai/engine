#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared Gated Delta Net (GDN) implementation for hybrid models.
//!
//! Used by both Qwen3 Next (text-only) and Qwen3.5 MoE (multimodal) models.

use hanzo_ml::{DType, Device, IndexOp, Module, Result, Tensor, D};
use hanzo_nn::Linear;
use hanzo_quant::{QuantMethod, QuantizedConfig, RowParallelLayer, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;

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

/// RMSNorm with gating: `rms_norm(x) * weight * silu(gate)`
pub struct RmsNormGated {
    pub weight: Tensor,
    eps: f64,
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
        Ok(Self { weight, eps })
    }

    /// Build directly from an already-materialized weight (e.g. a dequantized GGUF tensor).
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let gate = hanzo_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        let normed = hanzo_nn::ops::rms_norm(&x, &weight, self.eps as f32)?;
        normed.broadcast_mul(&gate)?.to_dtype(dtype)
    }
}

// ====================== GDN layer cache ======================

#[derive(Debug)]
pub struct GdnLayerCache {
    /// Conv state: (batch, conv_dim, kernel_size)
    pub conv_state: Tensor,
    /// Recurrent state: (batch, num_v_heads, head_k_dim, head_v_dim)
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
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
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            seqlen_offset: self.seqlen_offset,
        }
    }
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
    // The Vulkan single-step kernel (gdn_step_vulkan) isn't ported to canonical hanzo-ml yet, so
    // don't intercept the Vulkan decode path -- fall through to recurrence_portable, which is
    // documented to "serve CPU, Vulkan, and any backend without a fused kernel".
    recurrence_portable(q, k, v, g, beta, state)
}

/// Native Vulkan single decode step (seq==1). q,k,v,g,beta arrive (1, 1, v_heads, ..); the state
/// (1, v_heads, k_dim, v_dim) is updated in place in VRAM. Applies the 1/sqrt(k_dim) q-scale that
/// the portable scan does internally, then returns y (1, 1, v_heads, v_dim).
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
#[cfg(any(feature = "cuda", feature = "metal"))]
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
#[cfg(any(feature = "cuda", feature = "metal"))]
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

pub struct GatedDeltaNet {
    pub in_proj_qkvz: Linear,
    pub in_proj_ba: Linear,
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
        let v_per_group = num_v_heads / num_k_heads;

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        // Load qkvz and ba projections
        let qkvz_out = key_dim * 2 + value_dim * 2;
        let mut qkvz_w = match weight_mode {
            GdnWeightMode::MergedOnly => {
                vb_la.get((qkvz_out, hidden_size), "in_proj_qkvz.weight")?
            }
            GdnWeightMode::MergedWithFallback => {
                if vb_la.contains_tensor("in_proj_qkvz.weight") {
                    vb_la.get((qkvz_out, hidden_size), "in_proj_qkvz.weight")?
                } else {
                    // Load separate HF weights and interleave into grouped layout
                    let qkv_w =
                        vb_la.get((key_dim * 2 + value_dim, hidden_size), "in_proj_qkv.weight")?;
                    let z_w = vb_la.get((value_dim, hidden_size), "in_proj_z.weight")?;
                    let q_w = qkv_w.narrow(0, 0, key_dim)?;
                    let k_w = qkv_w.narrow(0, key_dim, key_dim)?;
                    let v_w = qkv_w.narrow(0, key_dim * 2, value_dim)?;
                    let q_grouped = q_w.reshape((num_k_heads, head_k_dim, hidden_size))?;
                    let k_grouped = k_w.reshape((num_k_heads, head_k_dim, hidden_size))?;
                    let v_grouped =
                        v_w.reshape((num_k_heads, v_per_group * head_v_dim, hidden_size))?;
                    let z_grouped =
                        z_w.reshape((num_k_heads, v_per_group * head_v_dim, hidden_size))?;
                    let merged = Tensor::cat(&[q_grouped, k_grouped, v_grouped, z_grouped], 1)?;
                    merged.reshape((qkvz_out, hidden_size))?
                }
            }
        };

        let mut ba_w = match weight_mode {
            GdnWeightMode::MergedOnly => {
                vb_la.get((num_v_heads * 2, hidden_size), "in_proj_ba.weight")?
            }
            GdnWeightMode::MergedWithFallback => {
                if vb_la.contains_tensor("in_proj_ba.weight") {
                    vb_la.get((num_v_heads * 2, hidden_size), "in_proj_ba.weight")?
                } else {
                    let b_w = vb_la.get((num_v_heads, hidden_size), "in_proj_b.weight")?;
                    let a_w = vb_la.get((num_v_heads, hidden_size), "in_proj_a.weight")?;
                    let b_grouped = b_w.reshape((num_k_heads, v_per_group, hidden_size))?;
                    let a_grouped = a_w.reshape((num_k_heads, v_per_group, hidden_size))?;
                    let merged = Tensor::cat(&[b_grouped, a_grouped], 1)?;
                    merged.reshape((num_v_heads * 2, hidden_size))?
                }
            }
        };

        let conv_dim = key_dim * 2 + value_dim;
        let mut conv1d_weight = vb_la.get((conv_dim, 1, conv_kernel_size), "conv1d.weight")?;
        let mut dt_bias = vb_la.get(num_v_heads, "dt_bias")?;
        let mut a_log = vb_la.get(num_v_heads, "A_log")?;

        if let Some(ref target_dev) = isq_target_device {
            qkvz_w = qkvz_w.to_device(target_dev)?;
            ba_w = ba_w.to_device(target_dev)?;
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
        }

        let in_proj_qkvz = Linear::new(qkvz_w, None);
        let in_proj_ba = Linear::new(ba_w, None);

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
            in_proj_qkvz,
            in_proj_ba,
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

    pub fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Project input
        let mixed_qkvz = self.in_proj_qkvz.forward(x)?;
        let mixed_ba = self.in_proj_ba.forward(x)?;

        // 2. Grouped head layout
        let group_size_qkvz = 2 * self.head_k_dim + 2 * v_per_group * self.head_v_dim;
        let mixed_qkvz =
            mixed_qkvz.reshape((batch_size, seq_len, self.num_k_heads, group_size_qkvz))?;

        let group_size_ba = 2 * v_per_group;
        let mixed_ba = mixed_ba.reshape((batch_size, seq_len, self.num_k_heads, group_size_ba))?;

        // Split within each group
        let mut offset = 0;
        let q = mixed_qkvz.narrow(D::Minus1, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let k = mixed_qkvz.narrow(D::Minus1, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let v = mixed_qkvz.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;
        offset += v_per_group * self.head_v_dim;
        let z = mixed_qkvz.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;

        let b = mixed_ba.narrow(D::Minus1, 0, v_per_group)?;
        let a = mixed_ba.narrow(D::Minus1, v_per_group, v_per_group)?;

        // Reshape v, z -> (batch, seq, num_v_heads, head_v_dim)
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // Reshape b, a -> (batch, seq, num_v_heads)
        let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

        // Flatten q, k, v for conv1d
        let q = q.reshape((batch_size, seq_len, self.key_dim))?;
        let k = k.reshape((batch_size, seq_len, self.key_dim))?;
        let v_flat = v.reshape((batch_size, seq_len, self.value_dim))?;

        // 3. Concatenate q, k, v for conv1d
        let mixed_qkv = Tensor::cat(&[&q, &k, &v_flat], D::Minus1)?;

        // 4. Apply causal conv1d (includes silu activation)
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // 5. Split back after conv and reshape to per-head
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 6. Compute beta and g
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

        // 7. If num_v_heads > num_k_heads, repeat_interleave q and k
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

        // 8. L2-normalize q and k
        let q = l2_norm(&q, 1e-6)?;
        let k = l2_norm(&k, 1e-6)?;

        // 9. Apply recurrence
        let y = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)?;

        cache.seqlen_offset += seq_len;

        // 10. Apply RMSNormGated
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.value_dim))?;

        // 11. Output projection
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
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }

    /// Full sequence causal conv1d for prefill.
    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

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
            return output.transpose(1, 2);
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
            return output.transpose(1, 2);
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
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Fast Vulkan reproduction of the qwen35moe prompt-step recurrence shape crash. Skips cleanly with
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
