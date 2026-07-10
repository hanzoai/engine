//! Shared DiT primitives for the TRELLIS flow models.
//!
//! Both the sparse-structure flow (dense 16^3 grid) and the SLAT flow (sparse active voxels) use the
//! identical `ModulatedTransformerCrossBlock`: adaLN-modulated self-attention (optional per-head
//! qk-RMSNorm), un-gated cross-attention to the DINOv2 tokens, adaLN-modulated FFN. The only thing
//! that differs upstream is how tokens/pos-embeddings are formed, so the block lives here once.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use hanzo_ml::{DType, Result, Tensor, D};
use hanzo_nn::{LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::layers::{layer_norm, linear};

const NORM_EPS: f64 = 1e-6; // block LayerNorm32 eps
const NORMALIZE_EPS: f64 = 1e-12; // F.normalize eps

/// Sinusoidal timestep embedding, `cos` then `sin` (TRELLIS/glide ordering). `t` is [B], scaled by
/// the caller. Returns [B, dim].
pub fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    const MAX_PERIOD: f64 = 10000.0;
    let half = dim / 2;
    let dev = t.device();
    let arange = Tensor::arange(0u32, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?; // [half]
    let args = t
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?; // [B, half]
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1) // [B, dim]
}

/// Non-affine LayerNorm over the last dim.
pub fn nonaffine_layernorm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// `h * (1 + scale) + shift` with `scale`/`shift` shaped [B, C] broadcast over the sequence.
fn modulate(h: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    h.broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
        .broadcast_add(&shift.unsqueeze(1)?)
}

/// Per-head RMS-style norm used on q/k: L2-normalize over the head dim, then scale by learned
/// `gamma[H,D]` and `sqrt(D)`.
struct MultiHeadRmsNorm {
    gamma: Tensor, // [1, 1, H, D]
    scale: f64,
}

impl MultiHeadRmsNorm {
    fn new(head_dim: usize, heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let gamma = vb
            .get((heads, head_dim), "gamma")?
            .reshape((1, 1, heads, head_dim))?;
        Ok(Self {
            gamma,
            scale: (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [B, L, H, 1]
        let x = x.broadcast_div(&(norm + NORMALIZE_EPS)?)?;
        x.broadcast_mul(&self.gamma)? * self.scale
    }
}

/// Naive scaled-dot-product attention. q [B,Lq,H,D], k/v [B,Lk,H,D] -> [B,Lq,H,D].
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f64).sqrt();
    let q = q.transpose(1, 2)?.contiguous()?; // [B,H,Lq,D]
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;
    let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    let attn = hanzo_nn::ops::softmax_last_dim(&attn)?;
    let out = attn.matmul(&v)?; // [B,H,Lq,D]
    out.transpose(1, 2)?.contiguous()
}

struct SelfAttention {
    to_qkv: Linear,
    to_out: Linear,
    q_rms: Option<MultiHeadRmsNorm>,
    k_rms: Option<MultiHeadRmsNorm>,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(
        channels: usize,
        heads: usize,
        qk_rms_norm: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let head_dim = channels / heads;
        let (q_rms, k_rms) = if qk_rms_norm {
            (
                Some(MultiHeadRmsNorm::new(head_dim, heads, vb.pp("q_rms_norm"))?),
                Some(MultiHeadRmsNorm::new(head_dim, heads, vb.pp("k_rms_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_qkv: linear(channels, channels * 3, vb.pp("to_qkv"))?,
            to_out: linear(channels, channels, vb.pp("to_out"))?,
            q_rms,
            k_rms,
            heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let qkv = x
            .apply(&self.to_qkv)?
            .reshape((b, l, 3, self.heads, self.head_dim))?;
        let mut q = qkv.narrow(2, 0, 1)?.squeeze(2)?; // [B,L,H,D]
        let mut k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        if let (Some(qn), Some(kn)) = (&self.q_rms, &self.k_rms) {
            q = qn.forward(&q)?;
            k = kn.forward(&k)?;
        }
        let h = sdpa(&q, &k, &v)?.reshape((b, l, c))?;
        h.apply(&self.to_out)
    }
}

struct CrossAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn new(channels: usize, ctx: usize, heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: linear(channels, channels, vb.pp("to_q"))?,
            to_kv: linear(ctx, channels * 2, vb.pp("to_kv"))?,
            to_out: linear(channels, channels, vb.pp("to_out"))?,
            heads,
            head_dim: channels / heads,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let lkv = context.dim(1)?;
        let q = x
            .apply(&self.to_q)?
            .reshape((b, l, self.heads, self.head_dim))?;
        let kv = context
            .apply(&self.to_kv)?
            .reshape((b, lkv, 2, self.heads, self.head_dim))?;
        let k = kv.narrow(2, 0, 1)?.squeeze(2)?;
        let v = kv.narrow(2, 1, 1)?.squeeze(2)?;
        let h = sdpa(&q, &k, &v)?.reshape((b, l, c))?;
        h.apply(&self.to_out)
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(channels: usize, mlp_ratio: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = channels * mlp_ratio;
        Ok(Self {
            fc1: linear(channels, hidden, vb.pp("mlp.0"))?,
            fc2: linear(hidden, channels, vb.pp("mlp.2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?.gelu()?.apply(&self.fc2) // GELU(approximate="tanh")
    }
}

/// TRELLIS `ModulatedTransformerCrossBlock`: gated self-attn, un-gated cross-attn, gated FFN.
pub struct ModulatedCrossBlock {
    norm2: LayerNorm, // the only affine norm (cross-attn input)
    self_attn: SelfAttention,
    cross_attn: CrossAttention,
    mlp: FeedForward,
    ada: Linear, // adaLN_modulation.1: Linear(C, 6C)
}

impl ModulatedCrossBlock {
    pub fn new(
        channels: usize,
        ctx: usize,
        heads: usize,
        mlp_ratio: usize,
        qk_rms_norm: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm2: layer_norm(channels, NORM_EPS, vb.pp("norm2"))?,
            self_attn: SelfAttention::new(channels, heads, qk_rms_norm, vb.pp("self_attn"))?,
            cross_attn: CrossAttention::new(channels, ctx, heads, vb.pp("cross_attn"))?,
            mlp: FeedForward::new(channels, mlp_ratio, vb.pp("mlp"))?,
            ada: linear(channels, 6 * channels, vb.pp("adaLN_modulation.1"))?,
        })
    }

    /// `x` [B,L,C], `t_emb` [B,C], `context` [B,Lkv,ctx].
    pub fn forward(&self, x: &Tensor, t_emb: &Tensor, context: &Tensor) -> Result<Tensor> {
        let m = t_emb.silu()?.apply(&self.ada)?; // [B, 6C]
        let c = m.chunk(6, D::Minus1)?;
        let (shift_msa, scale_msa, gate_msa) = (&c[0], &c[1], &c[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&c[3], &c[4], &c[5]);

        let h = nonaffine_layernorm(x, NORM_EPS)?;
        let h = modulate(&h, scale_msa, shift_msa)?;
        let h = self_attn_gate(&self.self_attn.forward(&h)?, gate_msa)?;
        let x = (x + h)?;

        let h = self.cross_attn.forward(&x.apply(&self.norm2)?, context)?;
        let x = (x + h)?;

        let h = nonaffine_layernorm(&x, NORM_EPS)?;
        let h = modulate(&h, scale_mlp, shift_mlp)?;
        let h = self_attn_gate(&self.mlp.forward(&h)?, gate_mlp)?;
        x + h
    }
}

fn self_attn_gate(h: &Tensor, gate: &Tensor) -> Result<Tensor> {
    h.broadcast_mul(&gate.unsqueeze(1)?)
}

/// Timestep MLP: sinusoidal(256) -> Linear -> SiLU -> Linear.
pub struct TimestepEmbedder {
    fc0: Linear,
    fc2: Linear,
    freq_dim: usize,
}

impl TimestepEmbedder {
    pub fn new(hidden: usize, vb: ShardedVarBuilder) -> Result<Self> {
        const FREQ_DIM: usize = 256;
        Ok(Self {
            fc0: linear(FREQ_DIM, hidden, vb.pp("mlp.0"))?,
            fc2: linear(hidden, hidden, vb.pp("mlp.2"))?,
            freq_dim: FREQ_DIM,
        })
    }

    /// `t` is [B], already scaled (TRELLIS scales by 1000 at the call site).
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let f = timestep_embedding(t, self.freq_dim)?;
        f.apply(&self.fc0)?.silu()?.apply(&self.fc2)
    }
}
