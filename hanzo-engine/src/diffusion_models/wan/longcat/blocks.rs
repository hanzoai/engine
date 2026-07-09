#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! The 48-block LongCat avatar DiT core. Each block: adaLN self-attn (fused qkv, RMSNorm q/k, 3D-RoPE)
//! -> text cross-attn -> audio cross-attn (in `audio.rs`) -> SwiGLU FFN, each residual.

use hanzo_ml::{IndexOp, Result, Tensor, D};
use hanzo_nn::{LayerNorm, Linear, Module, RmsNorm};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::diffusion_models::wan::longcat::audio::AudioCrossAttn;
use crate::diffusion_models::wan::longcat::common::{
    apply_rope, no_affine_layer_norm, ModulationOut, LN_EPS, RMS_EPS,
};
use crate::diffusion_models::wan::longcat::dit::LongCatConfig;
use crate::layers;
use crate::pipeline::text_models_inputs_processor::FlashParams;

const SELF_ATTN_MULT: usize = 3; // fused qkv
const CROSS_KV_MULT: usize = 2; // fused kv
const ADALN_CHUNKS: usize = 6; // (shift,scale,gate) for self-attn and for ffn

/// Cached condition-token K/V for one block's self-attn (post-RoPE), reused across denoise steps.
pub(crate) type KvSlot = (Tensor, Tensor);

/// Per-block forward inputs shared by every block; `cos`/`sin` cover the current query span.
pub(crate) struct BlockCtx<'a> {
    pub cond: &'a Tensor,
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub text_kv: &'a Tensor,
    pub audio_kv: &'a Tensor,
}

/// Dense scaled-dot-product attention via the fused Sdpa flash path (bidirectional or masked),
/// falling back to a flatten-and-matmul eager path. q,k,v: `[B, H, seq, D]`.
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let params = SdpaParams {
        n_kv_groups: 1,
        sliding_window: None,
        softcap: None,
        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        sinks: None,
    };
    let flash = FlashParams::empty(false);
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    match Sdpa.run_attention(&q, &k, &v, mask, Some(&flash), &params) {
        Ok(out) => Ok(out),
        Err(_) => naive_attention(&q, &k, &v, mask),
    }
}

fn naive_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
    let (b, h, n, d) = q.dims4()?;
    let nk = k.dim(2)?;
    let scale = 1.0 / (d as f64).sqrt();
    let q = q.reshape((b * h, n, d))?;
    let k = k.reshape((b * h, nk, d))?;
    let v = v.reshape((b * h, nk, d))?;
    let mut att = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
    if let AttentionMask::Custom(m) = mask {
        att = att.broadcast_add(&m.reshape((1, n, nk))?)?;
    }
    let att = hanzo_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v)?.reshape((b, h, n, d))
}

/// Self-attention with fused qkv, RMSNorm(fp32) q/k, 3D-RoPE, and a block-causal KV-cache seam.
pub(crate) struct SelfAttention {
    qkv: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let qkv = layers::linear(dim, SELF_ATTN_MULT * dim, vb.pp("qkv"))?;
        let q_norm = RmsNorm::new(vb.get(cfg.head_dim, "q_norm.weight")?, RMS_EPS);
        let k_norm = RmsNorm::new(vb.get(cfg.head_dim, "k_norm.weight")?, RMS_EPS);
        let proj = layers::linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            q_norm,
            k_norm,
            proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    // x: [B, N, hidden] -> normed q,k and raw v, each [B, H, N, D]; no RoPE yet.
    fn proj_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, n, _) = x.dims3()?;
        let qkv =
            x.apply(&self.qkv)?
                .reshape((b, n, SELF_ATTN_MULT, self.num_heads, self.head_dim))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        Ok((q.apply(&self.q_norm)?, k.apply(&self.k_norm)?, v))
    }

    fn merge_heads(&self, attn: &Tensor) -> Result<Tensor> {
        let (b, _, n, _) = attn.dims4()?;
        attn.transpose(1, 2)?
            .reshape((b, n, self.num_heads * self.head_dim))?
            .apply(&self.proj)
    }

    // Dense full-sequence self-attn (mask carries the block-causal structure).
    fn dense(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &AttentionMask,
    ) -> Result<Tensor> {
        let (q, k, v) = self.proj_qkv(x)?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        self.merge_heads(&sdpa(&q, &k, &v, mask)?)
    }

    // Dense self-attn that also captures the first `n_cond` keys/values (post-RoPE) for the cache.
    fn dense_capture(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &AttentionMask,
        n_cond: usize,
    ) -> Result<(Tensor, KvSlot)> {
        let (q, k, v) = self.proj_qkv(x)?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let slot = (
            k.narrow(2, 0, n_cond)?.contiguous()?,
            v.narrow(2, 0, n_cond)?.contiguous()?,
        );
        Ok((self.merge_heads(&sdpa(&q, &k, &v, mask)?)?, slot))
    }

    // Noisy-only self-attn: noisy queries attend [cached cond K/V; this step's noisy K/V], unmasked.
    fn cached(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, cond_kv: &KvSlot) -> Result<Tensor> {
        let (q, k, v) = self.proj_qkv(x)?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let k = Tensor::cat(&[&cond_kv.0, &k], 2)?;
        let v = Tensor::cat(&[&cond_kv.1, &v], 2)?;
        self.merge_heads(&sdpa(&q, &k, &v, &AttentionMask::None)?)
    }
}

/// Cross-attention shared by the text and audio paths: `q_linear` over video tokens, fused `kv_linear`
/// over the context (UMT5 text or projected audio), RMSNorm(fp32) q/k. Not adaLN-modulated itself.
pub(crate) struct CrossAttention {
    q_linear: Linear,
    kv_linear: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    pub(crate) fn new(cfg: &LongCatConfig, kv_in: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let q_linear = layers::linear(dim, dim, vb.pp("q_linear"))?;
        let kv_linear = layers::linear(kv_in, CROSS_KV_MULT * dim, vb.pp("kv_linear"))?;
        let q_norm = RmsNorm::new(vb.get(cfg.head_dim, "q_norm.weight")?, RMS_EPS);
        let k_norm = RmsNorm::new(vb.get(cfg.head_dim, "k_norm.weight")?, RMS_EPS);
        let proj = layers::linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            q_linear,
            kv_linear,
            q_norm,
            k_norm,
            proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    // q_src: [B, Nq, hidden]; kv_src: [B, Nk, kv_in] -> [B, Nq, hidden].
    pub(crate) fn forward(&self, q_src: &Tensor, kv_src: &Tensor) -> Result<Tensor> {
        let (b, nq, _) = q_src.dims3()?;
        let nk = kv_src.dim(1)?;
        let q = q_src
            .apply(&self.q_linear)?
            .reshape((b, nq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let kv = kv_src.apply(&self.kv_linear)?.reshape((
            b,
            nk,
            CROSS_KV_MULT,
            self.num_heads,
            self.head_dim,
        ))?;
        let k = kv.i((.., .., 0))?.transpose(1, 2)?;
        let v = kv.i((.., .., 1))?.transpose(1, 2)?;
        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;
        let out = sdpa(&q, &k, &v, &AttentionMask::None)?;
        out.transpose(1, 2)?
            .reshape((b, nq, self.num_heads * self.head_dim))?
            .apply(&self.proj)
    }
}

/// SwiGLU FFN: `w2(silu(w1 x) * w3 x)`, all bias-free.
struct SwiGlu {
    w1: Linear,
    w3: Linear,
    w2: Linear,
}

impl SwiGlu {
    fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let (h, f) = (cfg.hidden_size, cfg.ffn_dim);
        Ok(Self {
            w1: layers::linear_no_bias(h, f, vb.pp("w1"))?,
            w3: layers::linear_no_bias(h, f, vb.pp("w3"))?,
            w2: layers::linear_no_bias(f, h, vb.pp("w2"))?,
        })
    }
}

impl Module for SwiGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (x.apply(&self.w1)?.silu()? * x.apply(&self.w3)?)?.apply(&self.w2)
    }
}

// SiLU + Linear(tembed -> 6*hidden), chunked into (self-attn, ffn) modulation triples.
struct AdaLn6 {
    lin: Linear,
}

impl AdaLn6 {
    fn new(tembed: usize, hidden: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            lin: layers::linear(tembed, ADALN_CHUNKS * hidden, vb.pp("1"))?,
        })
    }

    fn forward(&self, cond: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = cond
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(ADALN_CHUNKS, D::Minus1)?;
        let m1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let m2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((m1, m2))
    }
}

/// One LongCat avatar transformer block.
pub(crate) struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    ada_ln: AdaLn6,
    self_attn: SelfAttention,
    pre_crs_attn_norm: LayerNorm,
    text_cross: CrossAttention,
    audio: AudioCrossAttn,
    ffn: SwiGlu,
}

impl Block {
    pub(crate) fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device().clone();
        let norm1 = no_affine_layer_norm(cfg.hidden_size, dtype, &dev)?;
        let norm2 = no_affine_layer_norm(cfg.hidden_size, dtype, &dev)?;
        let ada_ln = AdaLn6::new(
            cfg.adaln_tembed_dim,
            cfg.hidden_size,
            vb.pp("adaLN_modulation"),
        )?;
        let self_attn = SelfAttention::new(cfg, vb.pp("attn"))?;
        let pre_crs_attn_norm =
            LayerNorm::new_no_bias(vb.get(cfg.hidden_size, "pre_crs_attn_norm.weight")?, LN_EPS);
        let text_cross = CrossAttention::new(cfg, cfg.caption_channels, vb.pp("cross_attn"))?;
        let ffn = SwiGlu::new(cfg, vb.pp("ffn"))?;
        let audio = AudioCrossAttn::new(cfg, vb)?;
        Ok(Self {
            norm1,
            norm2,
            ada_ln,
            self_attn,
            pre_crs_attn_norm,
            text_cross,
            audio,
            ffn,
        })
    }

    // Post-self-attn residual chain: gate self-attn -> text cross-attn -> audio cross-attn -> FFN.
    fn finish(
        &self,
        x: &Tensor,
        self_attn_out: &Tensor,
        m1: &ModulationOut,
        m2: &ModulationOut,
        ctx: &BlockCtx,
    ) -> Result<Tensor> {
        let x = (x + m1.gate(self_attn_out)?)?;
        let x = (&x
            + self
                .text_cross
                .forward(&x.apply(&self.pre_crs_attn_norm)?, ctx.text_kv)?)?;
        let x = (&x + self.audio.forward(&x, ctx.audio_kv, ctx.cond)?)?;
        let h = m2.scale_shift(&x.apply(&self.norm2)?)?;
        x + m2.gate(&h.apply(&self.ffn)?)?
    }

    /// Dense forward over the full token sequence; `mask` carries block-causality.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        ctx: &BlockCtx,
        mask: &AttentionMask,
    ) -> Result<Tensor> {
        let (m1, m2) = self.ada_ln.forward(ctx.cond)?;
        let h = m1.scale_shift(&x.apply(&self.norm1)?)?;
        let a = self.self_attn.dense(&h, ctx.cos, ctx.sin, mask)?;
        self.finish(x, &a, &m1, &m2, ctx)
    }

    /// Step-0 forward: dense over full seq, returning this block's cached condition K/V.
    pub(crate) fn prime(
        &self,
        x: &Tensor,
        ctx: &BlockCtx,
        mask: &AttentionMask,
        n_cond: usize,
    ) -> Result<(Tensor, KvSlot)> {
        let (m1, m2) = self.ada_ln.forward(ctx.cond)?;
        let h = m1.scale_shift(&x.apply(&self.norm1)?)?;
        let (a, slot) = self
            .self_attn
            .dense_capture(&h, ctx.cos, ctx.sin, mask, n_cond)?;
        Ok((self.finish(x, &a, &m1, &m2, ctx)?, slot))
    }

    /// Steps 1..N: forward noisy tokens only, reusing cached condition K/V (`ctx` spans noisy tokens).
    pub(crate) fn forward_noisy(
        &self,
        x: &Tensor,
        ctx: &BlockCtx,
        cond_kv: &KvSlot,
    ) -> Result<Tensor> {
        let (m1, m2) = self.ada_ln.forward(ctx.cond)?;
        let h = m1.scale_shift(&x.apply(&self.norm1)?)?;
        let a = self.self_attn.cached(&h, ctx.cos, ctx.sin, cond_kv)?;
        self.finish(x, &a, &m1, &m2, ctx)
    }
}
