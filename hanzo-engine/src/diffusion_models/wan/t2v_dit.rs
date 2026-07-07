#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Wan2.2 T2V DiT (`WanTransformer3DModel`), the text-to-video backbone. Flow-matching velocity
//! predictor over a 3D latent `[B, C, T, H, W]`: patchify -> N adaLN blocks (self-attn + text
//! cross-attn + GELU FFN, 3D-RoPE, RMSNorm q/k) -> modulated final -> unpatchify.
//!
//! Arch verified against `Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer/config.json`:
//! dim 3072 (24 heads * 128), ffn 14336, 30 layers, in/out 48ch, patch [1,2,2], text_dim 4096,
//! qk_norm rms_norm_across_heads, cross_attn_norm true. Key names mirror the diffusers checkpoint
//! (`patch_embedding`, `condition_embedder.{time_embedder,time_proj,text_embedder}`,
//! `blocks.{i}.{attn1,attn2,norm2,ffn,scale_shift_table}`, top-level `scale_shift_table`, `proj_out`).

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module, RmsNorm};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::AttentionMask;
use crate::diffusion_models::wan::longcat::blocks::sdpa;
use crate::diffusion_models::wan::longcat::common::{
    apply_rope, no_affine_layer_norm, timestep_embedding, Rope3D, ROPE_THETA,
};
use crate::layers;

const SELF_ATTN: usize = 1;
const CROSS_ATTN: usize = 2;
const BLOCK_MOD_CHUNKS: usize = 6; // shift/scale/gate for (self-attn, ffn)
const FINAL_MOD_CHUNKS: usize = 2; // shift/scale
const TIME_SCALE: f64 = 1000.0; // flow-match t in [0,1000] -> sinusoid input in [0,1]

/// Wan2.2 T2V DiT configuration. `ti2v_5b()` is the shipped 5B model; `tiny()` is for tests.
#[derive(Debug, Clone)]
pub struct Wan2Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: [usize; 3],
    pub text_dim: usize,
    pub freq_dim: usize,
    pub rope_axes_dim: [usize; 3],
    pub eps: f64,
}

impl Wan2Config {
    /// Wan-AI/Wan2.2-TI2V-5B: 30 layers, hidden 3072, 24 heads, head_dim 128, ffn 14336, 48ch.
    /// head_dim 128 splits across (t,h,w) rope axes as 44/42/42 (Wan convention, sums to 128).
    pub fn ti2v_5b() -> Self {
        Self {
            hidden_size: 3072,
            num_layers: 30,
            num_heads: 24,
            head_dim: 128,
            ffn_dim: 14336,
            in_channels: 48,
            out_channels: 48,
            patch_size: [1, 2, 2],
            text_dim: 4096,
            freq_dim: 256,
            rope_axes_dim: [44, 42, 42],
            eps: 1e-6,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hidden_size: 256,
            num_layers: 2,
            num_heads: 4,
            head_dim: 64,
            ffn_dim: 512,
            in_channels: 48,
            out_channels: 48,
            patch_size: [1, 2, 2],
            text_dim: 64,
            freq_dim: 64,
            rope_axes_dim: [24, 20, 20],
            eps: 1e-6,
        }
    }

    pub fn patch_out(&self) -> usize {
        self.patch_size.iter().product::<usize>() * self.out_channels
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_size != self.num_heads * self.head_dim {
            hanzo_ml::bail!(
                "hidden_size {} != num_heads {} * head_dim {}",
                self.hidden_size,
                self.num_heads,
                self.head_dim
            );
        }
        if self.rope_axes_dim.iter().sum::<usize>() != self.head_dim {
            hanzo_ml::bail!(
                "rope_axes_dim {:?} must sum to head_dim {}",
                self.rope_axes_dim,
                self.head_dim
            );
        }
        Ok(())
    }
}

/// nn.Conv3d(in, hidden, (1,ph,pw)) folded to a per-frame strided conv2d (temporal patch is 1).
struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(cfg: &Wan2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let [pt, ph, pw] = cfg.patch_size;
        if pt != 1 {
            hanzo_ml::bail!("Wan2 patch_embedding temporal patch must be 1, got {pt}");
        }
        let w = vb
            .get((cfg.hidden_size, cfg.in_channels, pt, ph, pw), "weight")?
            .squeeze(2)?;
        let bias = vb.get(cfg.hidden_size, "bias")?;
        let cfg2 = Conv2dConfig {
            padding: 0,
            stride: ph,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            proj: Conv2d::new(w, Some(bias), cfg2),
        })
    }

    /// `[B, C, T, H, W]` -> tokens `[B, T*Hp*Wp, hidden]` plus the (T, Hp, Wp) latent grid.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, (usize, usize, usize))> {
        let (b, c, t, h, w) = x.dims5()?;
        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, c, h, w))?
            .apply(&self.proj)?;
        let (_, hid, hp, wp) = x.dims4()?;
        let x = x
            .reshape((b, t, hid, hp * wp))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((b, t * hp * wp, hid))?;
        Ok((x, (t, hp, wp)))
    }
}

/// condition_embedder: TimestepEmbedding (sinusoid -> linear_1 -> silu -> linear_2) producing the
/// per-block modulation seed via `time_proj` (Linear dim -> 6*dim), plus a PixArt text projection.
struct ConditionEmbedder {
    time_l1: Linear,
    time_l2: Linear,
    time_proj: Linear,
    text_l1: Linear,
    text_l2: Linear,
    freq_dim: usize,
    hidden: usize,
}

impl ConditionEmbedder {
    fn new(cfg: &Wan2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let te = vb.pp("time_embedder");
        let time_l1 = layers::linear(cfg.freq_dim, h, te.pp("linear_1"))?;
        let time_l2 = layers::linear(h, h, te.pp("linear_2"))?;
        let time_proj = layers::linear(h, BLOCK_MOD_CHUNKS * h, vb.pp("time_proj"))?;
        let txt = vb.pp("text_embedder");
        let text_l1 = layers::linear(cfg.text_dim, h, txt.pp("linear_1"))?;
        let text_l2 = layers::linear(h, h, txt.pp("linear_2"))?;
        Ok(Self {
            time_l1,
            time_l2,
            time_proj,
            text_l1,
            text_l2,
            freq_dim: cfg.freq_dim,
            hidden: h,
        })
    }

    // timestep [B] -> temb [B, hidden] and per-block modulation seed [B, 6, hidden].
    fn time(&self, t: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let temb = timestep_embedding(&(t / TIME_SCALE)?, self.freq_dim, dtype)?
            .apply(&self.time_l1)?
            .silu()?
            .apply(&self.time_l2)?;
        let seed = temb
            .silu()?
            .apply(&self.time_proj)?
            .reshape((temb.dim(0)?, BLOCK_MOD_CHUNKS, self.hidden))?;
        Ok((temb, seed))
    }

    // encoder_hidden_states [B, L, text_dim] -> [B, L, hidden] (GELU-approx PixArt projection).
    fn text(&self, txt: &Tensor) -> Result<Tensor> {
        txt.apply(&self.text_l1)?.gelu()?.apply(&self.text_l2)
    }
}

// adaLN shift/scale/gate for one stream, from a [B, hidden] modulation slice.
struct Mod {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl Mod {
    // x [B, N, hidden]: norm(x) * (1 + scale) + shift.
    fn scale_shift(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&(&self.scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&self.shift.unsqueeze(1)?)
    }

    fn gate(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.gate.unsqueeze(1)?)
    }
}

/// Multi-head attention with fused-free q/k/v Linears, RMSNorm(fp32) q/k, and optional 3D-RoPE.
/// `attn1` is self-attn over video tokens (rope); `attn2` is cross-attn over text (no rope).
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Wan2Config, kv_in: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let to_q = layers::linear(dim, dim, vb.pp("to_q"))?;
        let to_k = layers::linear(kv_in, dim, vb.pp("to_k"))?;
        let to_v = layers::linear(kv_in, dim, vb.pp("to_v"))?;
        let to_out = layers::linear(dim, dim, vb.pp("to_out").pp("0"))?;
        let norm_q = RmsNorm::new(vb.get(cfg.head_dim, "norm_q.weight")?, cfg.eps);
        let norm_k = RmsNorm::new(vb.get(cfg.head_dim, "norm_k.weight")?, cfg.eps);
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm_q,
            norm_k,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn split_heads(&self, x: &Tensor, b: usize, n: usize) -> Result<Tensor> {
        x.reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)
    }

    // q_src [B, Nq, hidden]; kv_src [B, Nk, kv_in]. rope (cos,sin) [Nq, head_dim] or None.
    fn forward(
        &self,
        q_src: &Tensor,
        kv_src: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, nq, _) = q_src.dims3()?;
        let nk = kv_src.dim(1)?;
        let q = self
            .split_heads(&q_src.apply(&self.to_q)?, b, nq)?
            .apply(&self.norm_q)?;
        let k = self
            .split_heads(&kv_src.apply(&self.to_k)?, b, nk)?
            .apply(&self.norm_k)?;
        let v = self.split_heads(&kv_src.apply(&self.to_v)?, b, nk)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?),
            None => (q, k),
        };
        let out = sdpa(&q, &k, &v, &AttentionMask::None)?;
        out.transpose(1, 2)?
            .reshape((b, nq, self.num_heads * self.head_dim))?
            .apply(&self.to_out)
    }
}

/// FeedForward: Linear -> GELU(approx) -> Linear (diffusers `net.0.proj`, `net.2`).
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn new(cfg: &Wan2Config, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            proj: layers::linear(cfg.hidden_size, cfg.ffn_dim, vb.pp("net").pp("0").pp("proj"))?,
            out: layers::linear(cfg.ffn_dim, cfg.hidden_size, vb.pp("net").pp("2"))?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.proj)?.gelu()?.apply(&self.out)
    }
}

/// One Wan2.2 transformer block. Modulation = `scale_shift_table (1,6,dim)` + time seed, chunked to
/// (self-attn shift/scale/gate, ffn shift/scale/gate). norm1/norm3 are non-affine; norm2 is affine.
struct Block {
    scale_shift_table: Tensor,
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ffn: FeedForward,
}

impl Block {
    fn new(cfg: &Wan2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device().clone();
        let h = cfg.hidden_size;
        let scale_shift_table = vb.get((1, BLOCK_MOD_CHUNKS, h), "scale_shift_table")?;
        let norm1 = no_affine_layer_norm(h, dtype, &dev)?;
        let attn1 = Attention::new(cfg, h, vb.pp("attn1"))?;
        let norm2 = LayerNorm::new_no_bias(vb.get(h, "norm2.weight")?, cfg.eps);
        let attn2 = Attention::new(cfg, h, vb.pp("attn2"))?;
        let norm3 = no_affine_layer_norm(h, dtype, &dev)?;
        let ffn = FeedForward::new(cfg, vb.pp("ffn"))?;
        Ok(Self {
            scale_shift_table,
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ffn,
        })
    }

    // seed [B, 6, hidden]: block modulation = scale_shift_table + seed, chunked into 2 Mods.
    fn modulation(&self, seed: &Tensor) -> Result<(Mod, Mod)> {
        let m = self
            .scale_shift_table
            .to_dtype(seed.dtype())?
            .broadcast_add(seed)?;
        let take = |i: usize| m.i((.., i)).and_then(|t| t.contiguous());
        Ok((
            Mod {
                shift: take(0)?,
                scale: take(1)?,
                gate: take(2)?,
            },
            Mod {
                shift: take(3)?,
                scale: take(4)?,
                gate: take(5)?,
            },
        ))
    }

    fn forward(
        &self,
        x: &Tensor,
        seed: &Tensor,
        text: &Tensor,
        rope: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let (msa, ffn) = self.modulation(seed)?;
        let h = msa.scale_shift(&x.apply(&self.norm1)?)?;
        let x = (x + msa.gate(&self.attn1.forward(&h, &h, Some(rope))?)?)?;
        let x = (&x + self.attn2.forward(&x.apply(&self.norm2)?, text, None)?)?;
        let h = ffn.scale_shift(&x.apply(&self.norm3)?)?;
        x + ffn.gate(&self.ffn.forward(&h)?)?
    }
}

/// Final adaLN: `norm_out` (non-affine) modulated by top-level `scale_shift_table (1,2,dim)` + temb,
/// then `proj_out` Linear(hidden -> patch_t*patch_h*patch_w*out).
struct FinalLayer {
    scale_shift_table: Tensor,
    norm: LayerNorm,
    proj: Linear,
}

impl FinalLayer {
    fn new(cfg: &Wan2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device().clone();
        let h = cfg.hidden_size;
        let scale_shift_table = vb.get((1, FINAL_MOD_CHUNKS, h), "scale_shift_table")?;
        let norm = no_affine_layer_norm(h, dtype, &dev)?;
        let proj = layers::linear(h, cfg.patch_out(), vb.pp("proj_out"))?;
        Ok(Self {
            scale_shift_table,
            norm,
            proj,
        })
    }

    // x [B, N, hidden]; temb [B, hidden] -> [B, N, patch_out].
    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let m = self
            .scale_shift_table
            .to_dtype(temb.dtype())?
            .broadcast_add(&temb.unsqueeze(1)?)?;
        let shift = m.i((.., 0))?.contiguous()?.unsqueeze(1)?;
        let scale = m.i((.., 1))?.contiguous()?.unsqueeze(1)?;
        x.apply(&self.norm)?
            .broadcast_mul(&(&scale + 1.0)?)?
            .broadcast_add(&shift)?
            .apply(&self.proj)
    }
}

/// The assembled Wan2.2 T2V DiT.
pub struct Wan2TransformerDiT {
    patch_embed: PatchEmbed,
    condition: ConditionEmbedder,
    blocks: Vec<Block>,
    final_layer: FinalLayer,
    rope: Rope3D,
    cfg: Wan2Config,
    device: Device,
    dtype: DType,
}

impl Wan2TransformerDiT {
    pub fn new(cfg: Wan2Config, vb: ShardedVarBuilder, device: Device) -> Result<Self> {
        cfg.validate()?;
        let dtype = vb.dtype();
        let patch_embed =
            PatchEmbed::new(&cfg, vb.pp("patch_embedding").set_device(device.clone()))?;
        let condition =
            ConditionEmbedder::new(&cfg, vb.pp("condition_embedder").set_device(device.clone()))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(&cfg, vb_b.pp(i).set_device(device.clone()))?);
        }
        let final_layer = FinalLayer::new(&cfg, vb.set_device(device.clone()))?;
        let rope = Rope3D::new(cfg.rope_axes_dim, ROPE_THETA, device.clone())?;
        Ok(Self {
            patch_embed,
            condition,
            blocks,
            final_layer,
            rope,
            cfg,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &Wan2Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    fn unpatchify(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, hp, wp) = grid;
        let [pt, ph, pw] = self.cfg.patch_size;
        let oc = self.cfg.out_channels;
        let b = x.dim(0)?;
        x.reshape(&[b, t, hp, wp, pt, ph, pw, oc])?
            .permute([0, 7, 1, 4, 2, 5, 3, 6])?
            .contiguous()?
            .reshape((b, oc, t * pt, hp * ph, wp * pw))
    }

    /// Flow-matching velocity for the latent `[B, C, T, H, W]` given `timestep [B]` and the encoded
    /// text `[B, L, text_dim]`. Returns velocity `[B, C, T, H, W]`.
    pub fn forward(&self, latent: &Tensor, timestep: &Tensor, text: &Tensor) -> Result<Tensor> {
        let (temb, seed) = self.condition.time(timestep, self.dtype)?;
        let text = self.condition.text(text)?;
        let (mut x, grid) = self.patch_embed.forward(latent)?;
        let (t, hp, wp) = grid;
        let (cos, sin) = self.rope.table(t, hp, wp)?;
        for block in &self.blocks {
            x = block.forward(&x, &seed, &text, (&cos, &sin))?;
        }
        self.unpatchify(&self.final_layer.forward(&x, &temb)?, grid)
    }
}
