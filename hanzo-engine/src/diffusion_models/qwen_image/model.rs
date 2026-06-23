#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{LayerNorm, Linear, Module, RmsNorm};
use hanzo_quant::ShardedVarBuilder;
use serde::Deserialize;

use crate::layers::{self, MatMul};

pub const THETA: usize = 10000;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub joint_attention_dim: usize,
    pub guidance_embeds: bool,
    pub axes_dims_rope: Vec<usize>,
}

impl Config {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

fn layer_norm_no_affine(dim: usize, vb: &ShardedVarBuilder) -> LayerNorm {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device()).unwrap();
    LayerNorm::new_no_bias(ws, 1e-6)
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (MatMul.matmul(&q, &k.t()?)? * scale_factor)?;
    let attn_scores = MatMul.matmul(&hanzo_nn::ops::softmax_last_dim(&attn_weights)?, &v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
}

// Complex rope (diffusers `apply_rotary_emb_qwen` with use_real=False).
// `freqs_cis` carries the cos/sin pair in the last dim: shape (..., seq, head_dim/2, 2).
fn apply_rope_qwen(x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    let x = x.reshape((b, h, l, d / 2, 2))?;
    let x_re = x.narrow(D::Minus1, 0, 1)?;
    let x_im = x.narrow(D::Minus1, 1, 1)?;
    let cos = freqs_cis.narrow(D::Minus1, 0, 1)?;
    let sin = freqs_cis.narrow(D::Minus1, 1, 1)?;
    // (a + bi)(cos + i sin) = (a cos - b sin) + i (a sin + b cos)
    let out_re = (x_re.broadcast_mul(&cos)? - x_im.broadcast_mul(&sin)?)?;
    let out_im = (x_re.broadcast_mul(&sin)? + x_im.broadcast_mul(&cos)?)?;
    Tensor::cat(&[out_re, out_im], D::Minus1)?.reshape((b, h, l, d))
}

// Computes the (cos, sin) rope table for one axis: shape (n, dim/2, 2).
fn rope_axis(index: &[i64], dim: usize, theta: usize, dev: &Device) -> Result<Tensor> {
    let theta = theta as f64;
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| (1f64 / theta.powf(i as f64 / dim as f64)) as f32)
        .collect();
    let half = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
    let index = Tensor::from_vec(
        index.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (index.len(), 1),
        dev,
    )?;
    let freqs = index.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Tensor::stack(&[&cos, &sin], D::Minus1)
}

#[derive(Debug, Clone)]
pub struct EmbedRope {
    theta: usize,
    axes_dim: Vec<usize>,
    scale_rope: bool,
    device: Device,
}

impl EmbedRope {
    fn new(theta: usize, axes_dim: Vec<usize>, scale_rope: bool, device: Device) -> Self {
        Self {
            theta,
            axes_dim,
            scale_rope,
            device,
        }
    }

    // Builds the per-axis rope table for a span of `n` positions starting at 0.
    fn pos_table(&self, axis: usize, n: usize) -> Result<Tensor> {
        let index: Vec<i64> = (0..n as i64).collect();
        rope_axis(&index, self.axes_dim[axis], self.theta, &self.device)
    }

    // Negative (centered) indices for scaled h/w rope.
    fn neg_table(&self, axis: usize, n: usize) -> Result<Tensor> {
        let index: Vec<i64> = (0..n as i64).rev().map(|i| -i - 1).collect();
        rope_axis(&index, self.axes_dim[axis], self.theta, &self.device)
    }

    // Returns (vid_freqs, txt_freqs) each as (seq, sum(axes_dim)/2, 2).
    fn forward(
        &self,
        frame: usize,
        height: usize,
        width: usize,
        txt_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (h2, w2) = (height, width);
        let f_freq = self.pos_table(0, frame)?; // (frame, d0/2, 2)
        let (h_freq, w_freq) = if self.scale_rope {
            let h = Tensor::cat(
                &[
                    &self.neg_table(1, h2 / 2)?,
                    &self.pos_table(1, h2 / 2 + h2 % 2)?,
                ],
                0,
            )?;
            let w = Tensor::cat(
                &[
                    &self.neg_table(2, w2 / 2)?,
                    &self.pos_table(2, w2 / 2 + w2 % 2)?,
                ],
                0,
            )?;
            (h, w)
        } else {
            (self.pos_table(1, h2)?, self.pos_table(2, w2)?)
        };
        let d_f = f_freq.dim(1)?;
        let d_h = h_freq.dim(1)?;
        let d_w = w_freq.dim(1)?;
        let f = f_freq
            .reshape((frame, 1, 1, d_f, 2))?
            .broadcast_as((frame, h2, w2, d_f, 2))?;
        let h = h_freq
            .reshape((1, h2, 1, d_h, 2))?
            .broadcast_as((frame, h2, w2, d_h, 2))?;
        let w = w_freq
            .reshape((1, 1, w2, d_w, 2))?
            .broadcast_as((frame, h2, w2, d_w, 2))?;
        let vid = Tensor::cat(&[&f, &h, &w], 3)?.reshape((frame * h2 * w2, d_f + d_h + d_w, 2))?;

        let max_vid_index = if self.scale_rope {
            (h2 / 2).max(w2 / 2)
        } else {
            h2.max(w2)
        };
        let txt_index: Vec<i64> = (0..txt_len as i64)
            .map(|i| i + max_vid_index as i64)
            .collect();
        let total_dim: usize = self.axes_dim.iter().sum();
        let txt = rope_axis(&txt_index, total_dim, self.theta, &self.device)?;
        Ok((vid, txt))
    }
}

fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        hanzo_ml::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    // diffusers Timesteps uses flip_sin_to_cos=True: [cos, sin].
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

#[derive(Debug, Clone)]
struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedder {
    fn new(in_dim: usize, h_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let linear_1 = layers::linear(in_dim, h_dim, vb.pp("linear_1"))?;
        let linear_2 = layers::linear(h_dim, h_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl Module for TimestepEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear_1)?.silu()?.apply(&self.linear_2)
    }
}

#[derive(Debug, Clone)]
pub struct TimeTextEmbed {
    timestep_embedder: TimestepEmbedder,
    #[allow(dead_code)]
    inner_dim: usize,
}

impl TimeTextEmbed {
    fn new(inner_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let timestep_embedder = TimestepEmbedder::new(256, inner_dim, vb.pp("timestep_embedder"))?;
        Ok(Self {
            timestep_embedder,
            inner_dim,
        })
    }

    fn forward(&self, timestep: &Tensor, dtype: DType) -> Result<Tensor> {
        let proj = timestep_embedding(timestep, 256, dtype)?;
        proj.apply(&self.timestep_embedder)
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

// SiLU then Linear(dim -> 6*dim); chunked into (mod1, mod2) each (shift, scale, gate).
#[derive(Debug, Clone)]
struct Modulation {
    lin: Linear,
}

impl Modulation {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let lin = layers::linear(dim, 6 * dim, vb.pp("1"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            hanzo_ml::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let inner = dim * 4;
        let proj_in = layers::linear(dim, inner, vb.pp("net").pp("0").pp("proj"))?;
        let proj_out = layers::linear(inner, dim, vb.pp("net").pp("2"))?;
        Ok(Self { proj_in, proj_out })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.proj_in)?.gelu()?.apply(&self.proj_out)
    }
}

// Joint dual-stream attention (QwenDoubleStreamAttnProcessor2_0).
#[derive(Debug, Clone)]
struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    to_out: Linear,
    to_add_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(dim: usize, num_heads: usize, head_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let to_q = layers::linear(dim, dim, vb.pp("to_q"))?;
        let to_k = layers::linear(dim, dim, vb.pp("to_k"))?;
        let to_v = layers::linear(dim, dim, vb.pp("to_v"))?;
        let add_q_proj = layers::linear(dim, dim, vb.pp("add_q_proj"))?;
        let add_k_proj = layers::linear(dim, dim, vb.pp("add_k_proj"))?;
        let add_v_proj = layers::linear(dim, dim, vb.pp("add_v_proj"))?;
        let norm_q = RmsNorm::new(vb.get(head_dim, "norm_q.weight")?, 1e-6);
        let norm_k = RmsNorm::new(vb.get(head_dim, "norm_k.weight")?, 1e-6);
        let norm_added_q = RmsNorm::new(vb.get(head_dim, "norm_added_q.weight")?, 1e-6);
        let norm_added_k = RmsNorm::new(vb.get(head_dim, "norm_added_k.weight")?, 1e-6);
        let to_out = layers::linear(dim, dim, vb.pp("to_out").pp("0"))?;
        let to_add_out = layers::linear(dim, dim, vb.pp("to_add_out"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            add_q_proj,
            add_k_proj,
            add_v_proj,
            norm_q,
            norm_k,
            norm_added_q,
            norm_added_k,
            to_out,
            to_add_out,
            num_heads,
            head_dim,
        })
    }

    fn split_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        xs.reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_freqs: &Tensor,
        txt_freqs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let seq_txt = txt.dim(1)?;
        let img_q = self.split_heads(&img.apply(&self.to_q)?)?;
        let img_k = self.split_heads(&img.apply(&self.to_k)?)?;
        let img_v = self.split_heads(&img.apply(&self.to_v)?)?;
        let txt_q = self.split_heads(&txt.apply(&self.add_q_proj)?)?;
        let txt_k = self.split_heads(&txt.apply(&self.add_k_proj)?)?;
        let txt_v = self.split_heads(&txt.apply(&self.add_v_proj)?)?;

        let img_q = img_q.apply(&self.norm_q)?;
        let img_k = img_k.apply(&self.norm_k)?;
        let txt_q = txt_q.apply(&self.norm_added_q)?;
        let txt_k = txt_k.apply(&self.norm_added_k)?;

        let img_q = apply_rope_qwen(&img_q, img_freqs)?;
        let img_k = apply_rope_qwen(&img_k, img_freqs)?;
        let txt_q = apply_rope_qwen(&txt_q, txt_freqs)?;
        let txt_k = apply_rope_qwen(&txt_k, txt_freqs)?;

        let q = Tensor::cat(&[&txt_q, &img_q], 2)?.contiguous()?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?.contiguous()?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?.contiguous()?;

        let attn = scaled_dot_product_attention(&q, &k, &v)?;
        let attn = attn.transpose(1, 2)?.flatten_from(2)?;

        let txt_attn = attn.narrow(1, 0, seq_txt)?;
        let img_attn = attn.narrow(1, seq_txt, attn.dim(1)? - seq_txt)?;

        let img_out = img_attn.contiguous()?.apply(&self.to_out)?;
        let txt_out = txt_attn.contiguous()?.apply(&self.to_add_out)?;
        Ok((img_out, txt_out))
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    img_mod: Modulation,
    img_norm1: LayerNorm,
    img_norm2: LayerNorm,
    img_mlp: FeedForward,
    txt_mod: Modulation,
    txt_norm1: LayerNorm,
    txt_norm2: LayerNorm,
    txt_mlp: FeedForward,
    attn: JointAttention,
}

impl TransformerBlock {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let img_mod = Modulation::new(dim, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm_no_affine(dim, &vb);
        let img_norm2 = layer_norm_no_affine(dim, &vb);
        let img_mlp = FeedForward::new(dim, vb.pp("img_mlp"))?;
        let txt_mod = Modulation::new(dim, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm_no_affine(dim, &vb);
        let txt_norm2 = layer_norm_no_affine(dim, &vb);
        let txt_mlp = FeedForward::new(dim, vb.pp("txt_mlp"))?;
        let attn = JointAttention::new(
            dim,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            vb.pp("attn"),
        )?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_norm2,
            txt_mlp,
            attn,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        img_freqs: &Tensor,
        txt_freqs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?;

        let img_modulated = img_mod1.scale_shift(&img.apply(&self.img_norm1)?)?;
        let txt_modulated = txt_mod1.scale_shift(&txt.apply(&self.txt_norm1)?)?;

        let (img_attn, txt_attn) =
            self.attn
                .forward(&img_modulated, &txt_modulated, img_freqs, txt_freqs)?;

        let img = (img + img_mod1.gate(&img_attn)?)?;
        let txt = (txt + txt_mod1.gate(&txt_attn)?)?;

        let img_mlp = img_mod2
            .scale_shift(&img.apply(&self.img_norm2)?)?
            .apply(&self.img_mlp)?;
        let img = (&img + img_mod2.gate(&img_mlp)?)?;

        let txt_mlp = txt_mod2
            .scale_shift(&txt.apply(&self.txt_norm2)?)?
            .apply(&self.txt_mlp)?;
        let txt = (&txt + txt_mod2.gate(&txt_mlp)?)?;

        Ok((img, txt))
    }
}

// AdaLayerNormContinuous(inner_dim, inner_dim) then Linear to patch output.
#[derive(Debug, Clone)]
pub struct LastLayer {
    norm: LayerNorm,
    ada_lin: Linear,
    proj_out: Linear,
}

impl LastLayer {
    fn new(inner_dim: usize, patch_out: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let norm = layer_norm_no_affine(inner_dim, &vb);
        let ada_lin = layers::linear(inner_dim, 2 * inner_dim, vb.pp("norm_out").pp("linear"))?;
        let proj_out = layers::linear(inner_dim, patch_out, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            ada_lin,
            proj_out,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor) -> Result<Tensor> {
        let chunks = vec_.silu()?.apply(&self.ada_lin)?.chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.proj_out)
    }
}

#[derive(Debug, Clone)]
pub struct QwenImageTransformer {
    img_in: Linear,
    txt_norm: RmsNorm,
    txt_in: Linear,
    time_text_embed: TimeTextEmbed,
    pos_embed: EmbedRope,
    blocks: Vec<TransformerBlock>,
    norm_out: LastLayer,
    #[allow(dead_code)]
    cfg: Config,
    #[allow(dead_code)]
    device: Device,
}

impl QwenImageTransformer {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder, device: Device) -> Result<Self> {
        let inner_dim = cfg.inner_dim();
        let img_in = layers::linear(
            cfg.in_channels,
            inner_dim,
            vb.pp("img_in").set_device(device.clone()),
        )?;
        let txt_norm = RmsNorm::new(
            vb.pp("txt_norm")
                .set_device(device.clone())
                .get(cfg.joint_attention_dim, "weight")?,
            1e-6,
        );
        let txt_in = layers::linear(
            cfg.joint_attention_dim,
            inner_dim,
            vb.pp("txt_in").set_device(device.clone()),
        )?;
        let time_text_embed = TimeTextEmbed::new(
            inner_dim,
            vb.pp("time_text_embed").set_device(device.clone()),
        )?;
        let pos_embed = EmbedRope::new(THETA, cfg.axes_dims_rope.clone(), true, device.clone());

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_b = vb.pp("transformer_blocks");
        for idx in 0..cfg.num_layers {
            blocks.push(TransformerBlock::new(cfg, vb_b.pp(idx))?);
        }

        let patch_out = cfg.patch_size * cfg.patch_size * cfg.out_channels;
        let norm_out = LastLayer::new(inner_dim, patch_out, vb.set_device(device.clone()))?;

        Ok(Self {
            img_in,
            txt_norm,
            txt_in,
            time_text_embed,
            pos_embed,
            blocks,
            norm_out,
            cfg: cfg.clone(),
            device,
        })
    }

    #[allow(dead_code)]
    pub fn device(&self) -> &Device {
        &self.device
    }

    // img: (b, img_seq, in_channels), txt: (b, txt_seq, joint_attention_dim),
    // img_shape: (frame, packed_h, packed_w), timestep: (b,)
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timestep: &Tensor,
        img_shape: (usize, usize, usize),
        _guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dtype = img.dtype();
        let txt_len = txt.dim(1)?;
        let (frame, ph, pw) = img_shape;

        let mut img = img.apply(&self.img_in)?;
        let mut txt = txt.apply(&self.txt_norm)?.apply(&self.txt_in)?;

        let vec_ = self.time_text_embed.forward(timestep, dtype)?;

        let (vid_freqs, txt_freqs) = self.pos_embed.forward(frame, ph, pw, txt_len)?;
        let vid_freqs = vid_freqs.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?;
        let txt_freqs = txt_freqs.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?;

        for block in self.blocks.iter() {
            (img, txt) = block.forward(&img, &txt, &vec_, &vid_freqs, &txt_freqs)?;
        }

        self.norm_out.forward(&img, &vec_)
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &Config {
        &self.cfg
    }
}
