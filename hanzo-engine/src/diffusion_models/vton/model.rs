//! FASHN VTON v1.5 pixel-space MMDiT. Faithful port of the reference `TryOnModel`
//! (fashn-ai/fashn-vton-1.5, Apache-2.0), whose blocks are themselves adapted from FLUX.1.
//! The RoPE / joint-attention / timestep-embedding idioms match the engine's FLUX port exactly.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{IndexOp, Module, Result, Tensor, D};
use hanzo_nn::{Conv2d, LayerNorm, Linear, RmsNorm};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{self, MatMul};
use crate::pipeline::text_models_inputs_processor::FlashParams;

use super::config::VtonConfig;

const LN_EPS: f64 = 1e-6;
const RMS_EPS: f64 = 1e-6;
const TIME_EMBED_DIM: usize = 256;

fn layer_norm(dim: usize, vb: &ShardedVarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, LN_EPS))
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

fn sdpa_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let sdpa_params = SdpaParams {
        n_kv_groups: 1,
        sliding_window: None,
        softcap: None,
        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        sinks: None,
    };
    let flash_params = FlashParams::empty(false);
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    match Sdpa.run_attention(
        &q,
        &k,
        &v,
        &AttentionMask::None,
        Some(&flash_params),
        &sdpa_params,
    ) {
        Ok(out) => Ok(out),
        Err(_) => scaled_dot_product_attention(&q, &k, &v),
    }
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        hanzo_ml::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = sdpa_attention(&q, &k, v)?;
    x.transpose(1, 2)?.flatten_from(2)
}

fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    let dev = t.device();
    let dtype = t.dtype();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(hanzo_ml::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(hanzo_ml::DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
struct EmbedNd {
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            emb.push(rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?);
        }
        Tensor::cat(&emb, 2)?.unsqueeze(1)
    }
}

#[derive(Debug, Clone)]
struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            in_layer: layers::linear(in_sz, h_sz, vb.pp("in_layer"))?,
            out_layer: layers::linear(h_sz, h_sz, vb.pp("out_layer"))?,
        })
    }
}

impl Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

#[derive(Debug, Clone)]
struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            query_norm: RmsNorm::new(vb.get(dim, "query_norm.scale")?, RMS_EPS),
            key_norm: RmsNorm::new(vb.get(dim, "key_norm.scale")?, RMS_EPS),
        })
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

fn modulation_chunks(lin: &Linear, vec_: &Tensor, n: usize) -> Result<Vec<Tensor>> {
    vec_.silu()?.apply(lin)?.unsqueeze(1)?.chunk(n, D::Minus1)
}

#[derive(Debug, Clone)]
struct Modulation {
    lin: Linear,
    double: bool,
}

impl Modulation {
    fn new(dim: usize, double: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let mult = if double { 6 } else { 3 };
        Ok(Self {
            lin: layers::linear(dim, mult * dim, vb.pp("lin"))?,
            double,
        })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, Option<ModulationOut>)> {
        let n = if self.double { 6 } else { 3 };
        let ys = modulation_chunks(&self.lin, vec_, n)?;
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = self.double.then(|| ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        });
        Ok((mod1, mod2))
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn new(dim: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: layers::linear_b(dim, dim * 3, true, vb.pp("qkv"))?,
            norm: QkNorm::new(head_dim, vb.pp("norm"))?,
            proj: layers::linear(dim, dim, vb.pp("proj"))?,
            num_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv
            .i((.., .., 0))?
            .transpose(1, 2)?
            .apply(&self.norm.query_norm)?;
        let k = qkv
            .i((.., .., 1))?
            .transpose(1, 2)?
            .apply(&self.norm.key_norm)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        Ok((q, k, v))
    }
}

fn mlp_forward(lin1: &Linear, lin2: &Linear, xs: &Tensor) -> Result<Tensor> {
    xs.apply(lin1)?.gelu()?.apply(lin2)
}

#[derive(Debug, Clone)]
struct DoubleStreamBlock {
    img_mod: Modulation,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp1: Linear,
    img_mlp2: Linear,
    txt_mod: Modulation,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp1: Linear,
    txt_mlp2: Linear,
}

impl DoubleStreamBlock {
    fn new(cfg: &VtonConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp = cfg.mlp_hidden();
        Ok(Self {
            img_mod: Modulation::new(h, true, vb.pp("img_mod"))?,
            img_norm1: layer_norm(h, &vb.pp("img_norm1"))?,
            img_attn: SelfAttention::new(h, cfg.num_heads, vb.pp("img_attn"))?,
            img_norm2: layer_norm(h, &vb.pp("img_norm2"))?,
            img_mlp1: layers::linear(h, mlp, vb.pp("img_mlp").pp("0"))?,
            img_mlp2: layers::linear(mlp, h, vb.pp("img_mlp").pp("2"))?,
            txt_mod: Modulation::new(h, true, vb.pp("txt_mod"))?,
            txt_norm1: layer_norm(h, &vb.pp("txt_norm1"))?,
            txt_attn: SelfAttention::new(h, cfg.num_heads, vb.pp("txt_attn"))?,
            txt_norm2: layer_norm(h, &vb.pp("txt_norm2"))?,
            txt_mlp1: layers::linear(h, mlp, vb.pp("txt_mlp").pp("0"))?,
            txt_mlp2: layers::linear(mlp, h, vb.pp("txt_mlp").pp("2"))?,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?;
        let img_mod2 = img_mod2.unwrap();
        let txt_mod2 = txt_mod2.unwrap();

        let img_modulated = img_mod1.scale_shift(&img.apply(&self.img_norm1)?)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt_mod1.scale_shift(&txt.apply(&self.txt_norm1)?)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_len = txt.dim(1)?;
        let txt_attn = attn.narrow(1, 0, txt_len)?;
        let img_attn = attn.narrow(1, txt_len, attn.dim(1)? - txt_len)?;

        let img = (img + img_mod1.gate(&img_attn.apply(&self.img_attn.proj)?))?;
        let img = (&img
            + img_mod2.gate(&mlp_forward(
                &self.img_mlp1,
                &self.img_mlp2,
                &img_mod2.scale_shift(&img.apply(&self.img_norm2)?)?,
            )?)?)?;

        let txt = (txt + txt_mod1.gate(&txt_attn.apply(&self.txt_attn.proj)?))?;
        let txt = (&txt
            + txt_mod2.gate(&mlp_forward(
                &self.txt_mlp1,
                &self.txt_mlp2,
                &txt_mod2.scale_shift(&txt.apply(&self.txt_norm2)?)?,
            )?)?)?;

        Ok((img, txt))
    }
}

#[derive(Debug, Clone)]
struct SingleStreamBlock {
    linear1: Linear,
    linear2: Linear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &VtonConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp = cfg.mlp_hidden();
        Ok(Self {
            linear1: layers::linear(h, h * 3 + mlp, vb.pp("linear1"))?,
            linear2: layers::linear(h + mlp, h, vb.pp("linear2"))?,
            norm: QkNorm::new(cfg.head_dim(), vb.pp("norm"))?,
            pre_norm: layer_norm(h, &vb.pp("pre_norm"))?,
            modulation: Modulation::new(h, false, vb.pp("modulation"))?,
            h_sz: h,
            mlp_sz: mlp,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (mod_, _) = self.modulation.forward(vec_)?;
        let x_mod = mod_
            .scale_shift(&xs.apply(&self.pre_norm)?)?
            .apply(&self.linear1)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv
            .i((.., .., 0))?
            .transpose(1, 2)?
            .apply(&self.norm.query_norm)?;
        let k = qkv
            .i((.., .., 1))?
            .transpose(1, 2)?
            .apply(&self.norm.key_norm)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output = Tensor::cat(&[attn, mlp.gelu()?], 2)?.apply(&self.linear2)?;
        xs + mod_.gate(&output)
    }
}

#[derive(Debug, Clone)]
struct LastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln: Linear,
}

impl LastLayer {
    fn new(cfg: &VtonConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            norm_final: layer_norm(h, &vb.pp("norm_final"))?,
            linear: layers::linear(h, cfg.patch_out_dim(), vb.pp("linear"))?,
            ada_ln: layers::linear(h, 2 * h, vb.pp("adaLN_modulation.1"))?,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor) -> Result<Tensor> {
        let chunks = vec_.silu()?.apply(&self.ada_ln)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        xs.apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?
            .apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(in_c: usize, embed_dim: usize, patch: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = hanzo_nn::Conv2dConfig {
            stride: patch,
            ..Default::default()
        };
        Ok(Self {
            proj: layers::conv2d(in_c, embed_dim, patch, cfg, vb.pp("proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)?.flatten_from(2)?.transpose(1, 2)
    }
}

/// The VTON try-on transformer. Inputs are pixel-space RGB (already normalized to [-1, 1]) plus
/// single-channel pose maps; conditioning is category ids. Output is the flow-matching velocity.
#[derive(Debug, Clone)]
pub struct TryOnModel {
    t_embedder: MlpEmbedder,
    y_embedder: Tensor,
    x_embedder: PatchEmbed,
    garment_embedder: PatchEmbed,
    x_patch_mixer: Vec<SingleStreamBlock>,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
    x_ids: Tensor,
    x_pe: Tensor,
    core_pe: Tensor,
    cfg: VtonConfig,
}

impl TryOnModel {
    pub fn new(cfg: &VtonConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let dev = vb.device().clone();

        let t_embedder = MlpEmbedder::new(TIME_EMBED_DIM, h, vb.pp("t_embedder").pp("mlp"))?;
        let y_embedder = vb.get((cfg.n_classes + 1, h), "y_embedder.weight")?;
        let x_embedder =
            PatchEmbed::new(cfg.x_in_channels(), h, cfg.patch_size, vb.pp("x_embedder"))?;
        let garment_embedder = PatchEmbed::new(
            cfg.garment_in_channels(),
            h,
            cfg.patch_size,
            vb.pp("garment_embedder"),
        )?;

        let mut x_patch_mixer = Vec::with_capacity(cfg.patch_mixer_depth);
        let vb_pm = vb.pp("x_patch_mixer");
        for i in 0..cfg.patch_mixer_depth {
            x_patch_mixer.push(SingleStreamBlock::new(cfg, vb_pm.pp(i))?);
        }
        let mut double_blocks = Vec::with_capacity(cfg.double_blocks_depth);
        let vb_d = vb.pp("double_blocks");
        for i in 0..cfg.double_blocks_depth {
            double_blocks.push(DoubleStreamBlock::new(cfg, vb_d.pp(i))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.single_blocks_depth);
        let vb_s = vb.pp("single_blocks");
        for i in 0..cfg.single_blocks_depth {
            single_blocks.push(SingleStreamBlock::new(cfg, vb_s.pp(i))?);
        }
        let final_layer = LastLayer::new(cfg, vb.pp("final_layer"))?;

        let x_ids = patch_ids(cfg.grid_h(), cfg.grid_w(), &dev)?;
        let pe_embedder = EmbedNd {
            theta: cfg.theta,
            axes_dim: cfg.axes_dim.to_vec(),
        };
        let x_pe = pe_embedder.forward(&x_ids)?;
        let core_pe = Tensor::cat(&[&x_pe, &x_pe], 2)?;

        Ok(Self {
            t_embedder,
            y_embedder,
            x_embedder,
            garment_embedder,
            x_patch_mixer,
            double_blocks,
            single_blocks,
            final_layer,
            x_ids,
            x_pe,
            core_pe,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &VtonConfig {
        &self.cfg
    }

    /// Conditioning vector: timestep embedding + category (class) embedding.
    fn condition_vec(&self, times: &Tensor, categories: &Tensor) -> Result<Tensor> {
        let t = timestep_embedding(times, TIME_EMBED_DIM)?.apply(&self.t_embedder)?;
        let y = self.y_embedder.index_select(&categories.contiguous()?, 0)?;
        t + y
    }

    /// One denoising forward. All conditioning tensors are already final (CFG masking, i.e. the
    /// null/unconditional zeroing, is applied by the caller). Returns the velocity, shape (b, 3, H, W).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        times: &Tensor,
        ca_images: &Tensor,
        garment_images: &Tensor,
        person_poses: &Tensor,
        garment_poses: &Tensor,
        categories: &Tensor,
    ) -> Result<Tensor> {
        let x_in = Tensor::cat(&[x, ca_images, person_poses], 1)?;
        let mut img = self.x_embedder.forward(&x_in)?;

        let g_in = Tensor::cat(&[garment_images, garment_poses], 1)?;
        let mut txt = self.garment_embedder.forward(&g_in)?;

        let vec_ = self.condition_vec(times, categories)?;

        for block in &self.x_patch_mixer {
            img = block.forward(&img, &vec_, &self.x_pe)?;
        }

        for block in &self.double_blocks {
            (img, txt) = block.forward(&img, &txt, &vec_, &self.core_pe)?;
        }

        let txt_len = txt.dim(1)?;
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single_blocks {
            img = block.forward(&img, &vec_, &self.core_pe)?;
        }
        let img = img.i((.., txt_len..))?;

        let x = self.final_layer.forward(&img, &vec_)?;
        self.unpatchify(&x)
    }

    fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _seq, _c) = x.dims3()?;
        let (gh, gw) = (self.cfg.grid_h(), self.cfg.grid_w());
        let (ch, p) = (self.cfg.channels_in, self.cfg.patch_size);
        x.reshape((b, gh, gw, ch, p, p))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((b, ch, gh * p, gw * p))
    }

    pub fn x_ids(&self) -> &Tensor {
        &self.x_ids
    }
}

/// Positional ids for a (grid_h x grid_w) patch grid: axis 0 = 0, axis 1 = row, axis 2 = col.
fn patch_ids(grid_h: usize, grid_w: usize, dev: &hanzo_ml::Device) -> Result<Tensor> {
    let mut ids = vec![0f32; grid_h * grid_w * 3];
    for h in 0..grid_h {
        for w in 0..grid_w {
            let base = (h * grid_w + w) * 3;
            ids[base + 1] = h as f32;
            ids[base + 2] = w as f32;
        }
    }
    Tensor::from_vec(ids, (1, grid_h * grid_w, 3), dev)
}
