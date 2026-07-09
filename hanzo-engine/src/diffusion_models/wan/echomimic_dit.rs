#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{LayerNorm, Linear, Module, RmsNorm};
use hanzo_quant::ShardedVarBuilder;
use serde::Deserialize;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{self, MatMul};
use crate::pipeline::text_models_inputs_processor::FlashParams;

use super::conv3d::CausalConv3d;

pub const THETA: usize = 10000;
// CLIP-ViT-H image context token count prepended to the text context (open-clip vit-huge).
const CLIP_TOKENS: usize = 257;
// Audio window half-extent and per-window padding from EchoMimic's split_audio_sequence.
const AUDIO_EXPAND: usize = 4;

#[derive(Debug, Clone, Deserialize)]
pub struct EchoMimicConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    #[serde(default = "default_clip_dim")]
    pub clip_dim: usize,
    #[serde(default = "default_audio_dim")]
    pub audio_dim: usize,
    pub eps: f64,
}

fn default_clip_dim() -> usize {
    1280
}
fn default_audio_dim() -> usize {
    768
}

impl EchoMimicConfig {
    pub fn echomimic_v3() -> Self {
        Self {
            dim: 1536,
            ffn_dim: 8960,
            num_layers: 30,
            num_heads: 12,
            in_dim: 36,
            out_dim: 16,
            freq_dim: 256,
            text_dim: 4096,
            clip_dim: 1280,
            audio_dim: 768,
            eps: 1e-6,
        }
    }

    fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }
}

// WanLayerNorm: normalize in f32, no affine (norm1/norm2). A ones-weight no-bias LayerNorm.
fn wan_layer_norm(dim: usize, eps: f64, dev: &Device, dtype: DType) -> Result<LayerNorm> {
    let w = Tensor::ones(dim, dtype, dev)?;
    Ok(LayerNorm::new_no_bias(w, eps))
}

// Wan sinusoidal time embedding: [cos(t*f), sin(t*f)], f = 10000^(-i/half). No 1000x time factor.
fn sinusoidal_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    const MAX_PERIOD: f64 = 10000.;
    let dev = t.device();
    let half = dim / 2;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

// One rope axis: (cos, sin) table of shape (n, dim/2, 2) over positions `index`.
fn rope_axis(index: &[i64], dim: usize, theta: usize, dev: &Device) -> Result<Tensor> {
    let theta = theta as f64;
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| (1f64 / theta.powf(i as f64 / dim as f64)) as f32)
        .collect();
    let half = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
    let idx = Tensor::from_vec(
        index.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (index.len(), 1),
        dev,
    )?;
    let freqs = idx.broadcast_mul(&inv_freq)?;
    Tensor::stack(&[&freqs.cos()?, &freqs.sin()?], D::Minus1)
}

// 3D rope table over (frame, h, w) -> [seq, head_dim/2, 2]. Wan splits head_dim into
// [d-4*(d//6), 2*(d//6), 2*(d//6)] for (T,H,W); for d=128 that is [44,42,42].
fn rope_freqs(f: usize, h: usize, w: usize, head_dim: usize, dev: &Device) -> Result<Tensor> {
    let dh = head_dim / 6;
    let t_dim = head_dim - 4 * dh;
    let s_dim = 2 * dh;
    let ft = rope_axis(&(0..f as i64).collect::<Vec<_>>(), t_dim, THETA, dev)?;
    let fh = rope_axis(&(0..h as i64).collect::<Vec<_>>(), s_dim, THETA, dev)?;
    let fw = rope_axis(&(0..w as i64).collect::<Vec<_>>(), s_dim, THETA, dev)?;
    let (dt, dhh, dw) = (ft.dim(1)?, fh.dim(1)?, fw.dim(1)?);
    let ft = ft
        .reshape((f, 1, 1, dt, 2))?
        .broadcast_as((f, h, w, dt, 2))?;
    let fh = fh
        .reshape((1, h, 1, dhh, 2))?
        .broadcast_as((f, h, w, dhh, 2))?;
    let fw = fw
        .reshape((1, 1, w, dw, 2))?
        .broadcast_as((f, h, w, dw, 2))?;
    Tensor::cat(&[&ft, &fh, &fw], 3)?
        .contiguous()?
        .reshape((f * h * w, dt + dhh + dw, 2))
}

// Complex rope on head-major q/k [b, n, seq, d] with freqs [1, 1, seq, d/2, 2].
fn apply_rope(x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let (b, n, l, d) = x.dims4()?;
    let x = x.reshape((b, n, l, d / 2, 2))?;
    let x_re = x.narrow(D::Minus1, 0, 1)?;
    let x_im = x.narrow(D::Minus1, 1, 1)?;
    let cos = freqs.narrow(D::Minus1, 0, 1)?;
    let sin = freqs.narrow(D::Minus1, 1, 1)?;
    let out_re = (x_re.broadcast_mul(&cos)? - x_im.broadcast_mul(&sin)?)?;
    let out_im = (x_re.broadcast_mul(&sin)? + x_im.broadcast_mul(&cos)?)?;
    Tensor::cat(&[out_re, out_im], D::Minus1)?.reshape((b, n, l, d))
}

// Head-major SDPA (flash path with eager fallback), q/k/v [B, n_heads, L, d].
fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
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
    match Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, Some(&flash), &params) {
        Ok(o) => Ok(o),
        Err(_) => eager_attend(&q, &k, &v),
    }
}

fn eager_attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f64).sqrt();
    let attn = (MatMul.matmul(q, &k.t()?.contiguous()?)? * scale)?;
    MatMul.matmul(&hanzo_nn::ops::softmax_last_dim(&attn)?, v)
}

struct AudioProjModel {
    proj: Linear, // 768 -> 1536, bias=false
    norm: LayerNorm,
}

impl AudioProjModel {
    fn new(audio_dim: usize, dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let proj = layers::linear_no_bias(audio_dim, dim, vb.pp("proj"))?;
        let norm = layers::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        Ok(Self { proj, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.proj)?.apply(&self.norm)
    }
}

// img_emb MLPProj: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm (1280 -> 1536).
struct MlpProj {
    ln_in: LayerNorm,
    lin1: Linear,
    lin2: Linear,
    ln_out: LayerNorm,
}

impl MlpProj {
    fn new(in_dim: usize, dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let p = vb.pp("proj");
        let ln_in = layers::layer_norm(in_dim, 1e-5, p.pp(0))?;
        let lin1 = layers::linear(in_dim, in_dim, p.pp(1))?;
        let lin2 = layers::linear(in_dim, dim, p.pp(3))?;
        let ln_out = layers::layer_norm(dim, 1e-5, p.pp(4))?;
        Ok(Self {
            ln_in,
            lin1,
            lin2,
            ln_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.ln_in)?
            .apply(&self.lin1)?
            .gelu()?
            .apply(&self.lin2)?
            .apply(&self.ln_out)
    }
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    n_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(cfg: &EchoMimicConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.dim;
        Ok(Self {
            q: layers::linear(d, d, vb.pp("q"))?,
            k: layers::linear(d, d, vb.pp("k"))?,
            v: layers::linear(d, d, vb.pp("v"))?,
            o: layers::linear(d, d, vb.pp("o"))?,
            norm_q: RmsNorm::new(vb.get(d, "norm_q.weight")?, cfg.eps),
            norm_k: RmsNorm::new(vb.get(d, "norm_k.weight")?, cfg.eps),
            n_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (n, d) = (self.n_heads, self.head_dim);
        let q = self.norm_q.forward(&x.apply(&self.q)?)?;
        let k = self.norm_k.forward(&x.apply(&self.k)?)?;
        let v = x.apply(&self.v)?;
        let q = q.reshape((b, s, n, d))?.transpose(1, 2)?;
        let k = k.reshape((b, s, n, d))?.transpose(1, 2)?;
        let v = v.reshape((b, s, n, d))?.transpose(1, 2)?;
        let q = apply_rope(&q, freqs)?;
        let k = apply_rope(&k, freqs)?;
        let out = attend(&q, &k, &v)?;
        let out = out.transpose(1, 2)?.reshape((b, s, n * d))?;
        out.apply(&self.o)
    }
}

// CDCA: shared query attends text, clip-image, and per-frame audio K/V; outputs summed.
struct CrossAttentionAudio {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    k_img: Linear,
    v_img: Linear,
    norm_k_img: RmsNorm,
    k_audio: Linear, // bias=false
    v_audio: Linear, // bias=false
    norm_k_audio: RmsNorm,
    n_heads: usize,
    head_dim: usize,
}

impl CrossAttentionAudio {
    fn new(cfg: &EchoMimicConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.dim;
        Ok(Self {
            q: layers::linear(d, d, vb.pp("q"))?,
            k: layers::linear(d, d, vb.pp("k"))?,
            v: layers::linear(d, d, vb.pp("v"))?,
            o: layers::linear(d, d, vb.pp("o"))?,
            norm_q: RmsNorm::new(vb.get(d, "norm_q.weight")?, cfg.eps),
            norm_k: RmsNorm::new(vb.get(d, "norm_k.weight")?, cfg.eps),
            k_img: layers::linear(d, d, vb.pp("k_img"))?,
            v_img: layers::linear(d, d, vb.pp("v_img"))?,
            norm_k_img: RmsNorm::new(vb.get(d, "norm_k_img.weight")?, cfg.eps),
            k_audio: layers::linear_no_bias(d, d, vb.pp("k_audio"))?,
            v_audio: layers::linear_no_bias(d, d, vb.pp("v_audio"))?,
            norm_k_audio: RmsNorm::new(vb.get(d, "norm_k_audio.weight")?, cfg.eps),
            n_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        x.reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)
    }

    fn forward(&self, x: &Tensor, ctx: &CrossContext) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let (n, d) = (self.n_heads, self.head_dim);
        let context_img = ctx.context.narrow(1, 0, CLIP_TOKENS)?;
        let text_len = ctx.context.dim(1)? - CLIP_TOKENS;
        let context_txt = ctx.context.narrow(1, CLIP_TOKENS, text_len)?;

        let q = self.norm_q.forward(&x.apply(&self.q)?)?;
        let k = self.norm_k.forward(&context_txt.apply(&self.k)?)?;
        let v = context_txt.apply(&self.v)?;
        let k_img = self.norm_k_img.forward(&context_img.apply(&self.k_img)?)?;
        let v_img = context_img.apply(&self.v_img)?;

        let txt_x = attend(&self.split(&q)?, &self.split(&k)?, &self.split(&v)?)?
            .transpose(1, 2)?
            .reshape((b, seq, n * d))?;
        let img_x = attend(&self.split(&q)?, &self.split(&k_img)?, &self.split(&v_img)?)?
            .transpose(1, 2)?
            .reshape((b, seq, n * d))?;
        let audio_x = self.audio_attend(&q, ctx, b, seq)?;
        let out = ((txt_x + img_x)? + (audio_x * ctx.audio_scale)?)?;
        out.apply(&self.o)
    }

    // Per-latent-frame audio cross-attn: each frame's spatial queries attend only its own
    // audio window; ip_mask gates which spatial tokens receive audio.
    fn audio_attend(&self, q: &Tensor, ctx: &CrossContext, b: usize, seq: usize) -> Result<Tensor> {
        let (n, d) = (self.n_heads, self.head_dim);
        let lt = ctx.latent_t;
        let n_q = seq / lt;
        let k_a = self
            .norm_k_audio
            .forward(&ctx.audio.apply(&self.k_audio)?)?;
        let v_a = ctx.audio.apply(&self.v_audio)?;
        let bt = ctx.audio.dim(0)?; // b * latent_t
        let win = ctx.audio.dim(1)?;
        let k_a = k_a.reshape((bt, win, n, d))?.transpose(1, 2)?;
        let v_a = v_a.reshape((bt, win, n, d))?.transpose(1, 2)?;
        let q_a = q.reshape((bt, n_q, n, d))?.transpose(1, 2)?;
        let a = attend(&q_a, &k_a, &v_a)?.transpose(1, 2)?; // [bt, n_q, n, d]
        let a = a.broadcast_mul(&ctx.audio_mask)?;
        a.reshape((b, seq, n * d))
    }
}

struct AttentionBlock {
    norm1: LayerNorm,
    self_attn: SelfAttention,
    norm3: LayerNorm,
    cross_attn: CrossAttentionAudio,
    norm2: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    modulation: Tensor, // [1, 6, dim]
}

impl AttentionBlock {
    fn new(
        cfg: &EchoMimicConfig,
        dev: &Device,
        dtype: DType,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let d = cfg.dim;
        let norm1 = wan_layer_norm(d, cfg.eps, dev, dtype)?;
        let norm2 = wan_layer_norm(d, cfg.eps, dev, dtype)?;
        let norm3 = layers::layer_norm(d, cfg.eps, vb.pp("norm3"))?;
        Ok(Self {
            norm1,
            self_attn: SelfAttention::new(cfg, vb.pp("self_attn"))?,
            norm3,
            cross_attn: CrossAttentionAudio::new(cfg, vb.pp("cross_attn"))?,
            norm2,
            ffn1: layers::linear(d, cfg.ffn_dim, vb.pp("ffn").pp(0))?,
            ffn2: layers::linear(cfg.ffn_dim, d, vb.pp("ffn").pp(2))?,
            modulation: vb.get((1, 6, d), "modulation")?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        e0: &Tensor,
        freqs: &Tensor,
        ctx: &CrossContext,
    ) -> Result<Tensor> {
        // e: 6 modulation vectors [b, 1, dim] from (modulation + e0).
        let e = self.modulation.broadcast_add(e0)?;
        let m: Vec<Tensor> = (0..6)
            .map(|i| e.narrow(1, i, 1))
            .collect::<Result<Vec<_>>>()?;
        let normed = x.apply(&self.norm1)?;
        let temp = normed
            .broadcast_mul(&(&m[1] + 1.0)?)?
            .broadcast_add(&m[0])?;
        let y = self.self_attn.forward(&temp, freqs)?;
        let x = (x + y.broadcast_mul(&m[2])?)?;

        let cross = self.cross_attn.forward(&x.apply(&self.norm3)?, ctx)?;
        let x = (x + cross)?;

        let temp = x
            .apply(&self.norm2)?
            .broadcast_mul(&(&m[4] + 1.0)?)?
            .broadcast_add(&m[3])?;
        let y = temp.apply(&self.ffn1)?.gelu()?.apply(&self.ffn2)?;
        x + y.broadcast_mul(&m[5])?
    }
}

struct Head {
    norm: LayerNorm,
    head: Linear,
    modulation: Tensor, // [1, 2, dim]
}

impl Head {
    fn new(
        cfg: &EchoMimicConfig,
        patch: usize,
        dev: &Device,
        dtype: DType,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let norm = wan_layer_norm(cfg.dim, cfg.eps, dev, dtype)?;
        let head = layers::linear(cfg.dim, patch * cfg.out_dim, vb.pp("head"))?;
        Ok(Self {
            norm,
            head,
            modulation: vb.get((1, 2, cfg.dim), "modulation")?,
        })
    }

    fn forward(&self, x: &Tensor, e: &Tensor) -> Result<Tensor> {
        let e = self.modulation.broadcast_add(&e.unsqueeze(1)?)?;
        let shift = e.narrow(1, 0, 1)?;
        let scale = e.narrow(1, 1, 1)?;
        let x = x
            .apply(&self.norm)?
            .broadcast_mul(&(&scale + 1.0)?)?
            .broadcast_add(&shift)?;
        x.apply(&self.head)
    }
}

// Per-block cross-attention conditioning (text+clip context, windowed audio, frame gate).
pub struct CrossContext {
    pub context: Tensor,    // [b, CLIP_TOKENS + text_len, dim]
    pub audio: Tensor,      // [b*latent_t, win, dim]
    pub audio_mask: Tensor, // [b*latent_t, n_q, 1, 1]
    pub latent_t: usize,
    pub audio_scale: f64,
}

pub struct EchoMimicDiT {
    patch_embedding: CausalConv3d,
    text0: Linear,
    text2: Linear,
    time0: Linear,
    time2: Linear,
    time_proj: Linear, // dim -> 6*dim
    img_emb: MlpProj,
    audio2token: AudioProjModel,
    blocks: Vec<AttentionBlock>,
    head: Head,
    cfg: EchoMimicConfig,
    device: Device,
    patch_t: usize,
    patch_hw: usize,
}

impl EchoMimicDiT {
    pub fn new(
        cfg: EchoMimicConfig,
        vb: ShardedVarBuilder,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let d = cfg.dim;
        let (pt, ph, pw) = (1usize, 2usize, 2usize);
        let patch_embedding = CausalConv3d::new(
            cfg.in_dim,
            d,
            (pt, ph, pw),
            (pt, ph, pw),
            (0, 0, 0),
            true,
            vb.pp("patch_embedding"),
        )?;
        let text0 = layers::linear(cfg.text_dim, d, vb.pp("text_embedding").pp(0))?;
        let text2 = layers::linear(d, d, vb.pp("text_embedding").pp(2))?;
        let time0 = layers::linear(cfg.freq_dim, d, vb.pp("time_embedding").pp(0))?;
        let time2 = layers::linear(d, d, vb.pp("time_embedding").pp(2))?;
        let time_proj = layers::linear(d, 6 * d, vb.pp("time_projection").pp(1))?;
        let img_emb = MlpProj::new(cfg.clip_dim, d, vb.pp("img_emb"))?;
        let audio2token = AudioProjModel::new(cfg.audio_dim, d, vb.pp("audio2token"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(AttentionBlock::new(&cfg, &device, dtype, vb_b.pp(i))?);
        }
        let head = Head::new(&cfg, pt * ph * pw, &device, dtype, vb.pp("head"))?;
        Ok(Self {
            patch_embedding,
            text0,
            text2,
            time0,
            time2,
            time_proj,
            img_emb,
            audio2token,
            blocks,
            head,
            cfg,
            device,
            patch_t: pt,
            patch_hw: ph,
        })
    }

    pub fn config(&self) -> &EchoMimicConfig {
        &self.cfg
    }

    // x: [b, in_dim, F, H, W] noisy+masked latent; t: [b] timestep; text: [b, L, text_dim];
    // clip: [b, 257, clip_dim]; audio: [b, 2*Fpix, audio_dim]; ip_mask: [b, n_q].
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        text: &Tensor,
        clip: &Tensor,
        audio: &Tensor,
        ip_mask: &Tensor,
        audio_scale: f64,
    ) -> Result<Tensor> {
        let dtype = x.dtype();
        let b = x.dim(0)?;
        let x = self.patch_embedding.forward(x, None)?; // [b, dim, F, Hp, Wp]
        let (_, dim, f, hp, wp) = x.dims5()?;
        let xs = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // [b, seq, dim]

        let e = sinusoidal_embedding(t, self.cfg.freq_dim)?
            .to_dtype(dtype)?
            .apply(&self.time0)?
            .silu()?
            .apply(&self.time2)?; // [b, dim]
        let e0 = e.silu()?.apply(&self.time_proj)?.reshape((b, 6, dim))?;

        let text_ctx = text.apply(&self.text0)?.gelu()?.apply(&self.text2)?;
        let clip_ctx = self.img_emb.forward(clip)?;
        let context = Tensor::cat(&[&clip_ctx, &text_ctx], 1)?;

        let audio_tok = self.audio2token.forward(audio)?; // [b, 2*Fpix, dim]
        let (audio_win, latent_t) = window_audio(&audio_tok, AUDIO_EXPAND)?;
        if latent_t != f {
            hanzo_ml::bail!("audio windows {latent_t} != latent frames {f}");
        }
        let audio_mask = build_audio_mask(ip_mask, latent_t)?;

        let freqs = rope_freqs(f, hp, wp, self.cfg.head_dim(), &self.device)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(dtype)?;

        let ctx = CrossContext {
            context,
            audio: audio_win,
            audio_mask,
            latent_t,
            audio_scale,
        };

        let mut h = xs;
        for block in &self.blocks {
            h = block.forward(&h, &e0, &freqs, &ctx)?;
        }
        let h = self.head.forward(&h, &e)?; // [b, seq, patch*out_dim]
        self.unpatchify(&h, b, f, hp, wp)
    }

    fn unpatchify(&self, x: &Tensor, b: usize, f: usize, hp: usize, wp: usize) -> Result<Tensor> {
        let c = self.cfg.out_dim;
        let (pt, ph) = (self.patch_t, self.patch_hw);
        let u = x.reshape(&[b, f, hp, wp, pt, ph, ph, c])?;
        // 'fhwpqrc->cfphqwr' with batch: (b,f,h,w,p,q,r,c) -> (b,c,f,p,h,q,w,r)
        let u = u.permute([0usize, 7, 1, 4, 2, 5, 3, 6])?.contiguous()?;
        u.reshape((b, c, f * pt, hp * ph, wp * ph))
    }
}

// Map [b, 2*Fpix, dim] audio tokens into one window per latent frame: [b*latent_t, win, dim].
// Mirrors EchoMimic split_audio_sequence + split_tensor_with_padding (2 tokens/frame, expand).
fn window_audio(audio: &Tensor, expand: usize) -> Result<(Tensor, usize)> {
    let (b, len, dim) = audio.dims3()?;
    let num_frames = len / 2;
    let tpf = len as f64 / num_frames as f64;
    let half = (tpf * 4.0 / 2.0) as i64;
    let n_win = (num_frames - 1) / 4 + 1;
    let mut pos: Vec<i64> = Vec::with_capacity(n_win);
    for i in 0..n_win {
        if i == 0 {
            pos.push(0);
        } else {
            let start = tpf * (((i as f64) - 1.0) * 4.0 + 1.0) - 4.0;
            let end = tpf * ((i as f64) * 4.0 + 1.0) + 4.0;
            pos.push(((start + end) / 2.0) as i64 - 1);
        }
    }
    let mut ranges: Vec<(i64, i64)> = pos.iter().map(|&p| (p - half, p + half)).collect();
    if n_win > 1 {
        let r1 = ranges[1].0;
        ranges[0] = (-(half * 2 - r1), r1);
    }
    let exp = expand as i64;
    let max_idx = len as i64 - 1;
    let mut wins = Vec::with_capacity(n_win);
    for (s, e) in ranges {
        let (s, e) = (s - exp, e + exp);
        let pad_front = (-s).max(0) as usize;
        let pad_back = (e - max_idx).max(0) as usize;
        let vs = s.max(0) as usize;
        let ve = e.min(max_idx);
        let valid = if (vs as i64) <= ve {
            audio.narrow(1, vs, (ve as usize) - vs + 1)?
        } else {
            Tensor::zeros((b, 0, dim), audio.dtype(), audio.device())?
        };
        let padded = valid.pad_with_zeros(1, pad_front, pad_back)?;
        wins.push(padded.unsqueeze(1)?);
    }
    let windowed = Tensor::cat(&wins, 1)?; // [b, n_win, win, dim]
    let win = windowed.dim(2)?;
    Ok((windowed.reshape((b * n_win, win, dim))?, n_win))
}

// ip_mask [b, n_q] -> [b*latent_t, n_q, 1, 1] (per-frame repeat) for gating audio cross-attn.
fn build_audio_mask(ip_mask: &Tensor, latent_t: usize) -> Result<Tensor> {
    let (b, n_q) = ip_mask.dims2()?;
    ip_mask
        .unsqueeze(1)?
        .broadcast_as((b, latent_t, n_q))?
        .contiguous()?
        .reshape((b * latent_t, n_q, 1, 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Shape;
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::ShardedSafeTensors;

    struct RandnBackend;
    impl SimpleBackend for RandnBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("weight") && s.rank() == 1 {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, 0.02, s, dev)?.to_dtype(dtype)
            }
        }
        fn get_unchecked(&self, _n: &str, _d: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("needs shape")
        }
        fn contains_tensor(&self, _n: &str) -> bool {
            true
        }
    }

    fn tiny_cfg() -> EchoMimicConfig {
        EchoMimicConfig {
            dim: 24,
            ffn_dim: 48,
            num_layers: 2,
            num_heads: 2,
            in_dim: 36,
            out_dim: 16,
            freq_dim: 16,
            text_dim: 32,
            clip_dim: 20,
            audio_dim: 18,
            eps: 1e-6,
        }
    }

    #[test]
    fn dit_structural_forward_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let vb = ShardedSafeTensors::wrap(Box::new(RandnBackend), dtype, dev.clone());
        let cfg = tiny_cfg();
        let dit = EchoMimicDiT::new(cfg.clone(), vb, dev.clone(), dtype)?;

        // F_lat=2 latent frames; patch stride 2: 16x16 -> 8x8 (n_q=64). audio: 2*Fpix, Fpix=5.
        let (f_lat, h, w) = (2usize, 16usize, 16usize);
        let f_pix = 4 * f_lat - 3; // 5
        let x = Tensor::randn(0f64, 1.0, (1, cfg.in_dim, f_lat, h, w), &dev)?.to_dtype(dtype)?;
        let t = Tensor::from_vec(vec![500f32], 1, &dev)?;
        let text = Tensor::randn(0f64, 1.0, (1, 12, cfg.text_dim), &dev)?.to_dtype(dtype)?;
        let clip =
            Tensor::randn(0f64, 1.0, (1, CLIP_TOKENS, cfg.clip_dim), &dev)?.to_dtype(dtype)?;
        let audio =
            Tensor::randn(0f64, 1.0, (1, 2 * f_pix, cfg.audio_dim), &dev)?.to_dtype(dtype)?;
        let n_q = (h / 2) * (w / 2);
        let ip_mask = Tensor::ones((1, n_q), dtype, &dev)?;

        let out = dit.forward(&x, &t, &text, &clip, &audio, &ip_mask, 2.9)?;
        assert_eq!(out.dims(), &[1, cfg.out_dim, f_lat, h, w]);
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite DiT output");
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(mx != mn, "degenerate constant DiT output");
        Ok(())
    }

    #[test]
    fn audio_windowing_matches_latent_frames() -> Result<()> {
        let dev = Device::Cpu;
        // Fpix=9 -> latent_t=(9-1)/4+1=3 windows; audio length 18.
        let audio = Tensor::randn(0f64, 1.0, (1, 18, 8), &dev)?.to_dtype(DType::F32)?;
        let (win, lt) = window_audio(&audio, AUDIO_EXPAND)?;
        assert_eq!(lt, 3);
        assert_eq!(win.dim(0)?, 3); // b*latent_t
        assert_eq!(win.dim(2)?, 8);
        Ok(())
    }
}
