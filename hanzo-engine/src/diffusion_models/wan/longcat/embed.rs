#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! LongCat avatar DiT embedders: 3D patchify (temporal kernel 1 -> per-frame conv2d), sinusoidal
//! timestep MLP, UMT5 caption MLP, and the fp32 final adaLN + unpatch projection.

use hanzo_ml::{DType, Result, Tensor, D};
use hanzo_nn::{Conv2d, Conv2dConfig, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::diffusion_models::wan::longcat::common::{no_affine_layer_norm, timestep_embedding};
use crate::diffusion_models::wan::longcat::dit::LongCatConfig;
use crate::layers;

const FINAL_ADALN_CHUNKS: usize = 2; // shift, scale

/// Conv3d(in, hidden, (1,2,2)) folded to a per-frame strided conv2d (temporal kernel is 1).
pub(crate) struct PatchEmbed3D {
    proj: Conv2d,
}

impl PatchEmbed3D {
    pub(crate) fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let [pt, ph, pw] = cfg.patch_size;
        if pt != 1 {
            hanzo_ml::bail!("PatchEmbed3D temporal patch must be 1, got {pt}");
        }
        if ph != pw {
            hanzo_ml::bail!("PatchEmbed3D expects a square spatial patch, got {ph}x{pw}");
        }
        let w = vb
            .get(
                (cfg.hidden_size, cfg.in_channels, pt, ph, pw),
                "proj.weight",
            )?
            .squeeze(2)?;
        let bias = vb.get(cfg.hidden_size, "proj.bias")?;
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

    /// `[B, C, T, H, W]` -> tokens `[B, T*Hp*Wp, hidden]` plus the (T, Hp, Wp) latent grid for RoPE.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<(Tensor, (usize, usize, usize))> {
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

/// Sinusoid(freq_dim) -> Linear -> SiLU -> Linear(tembed): the adaLN conditioning vector.
pub(crate) struct TimestepEmbedder {
    mlp0: Linear,
    mlp2: Linear,
    freq_dim: usize,
}

impl TimestepEmbedder {
    pub(crate) fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mlp0 = layers::linear(
            cfg.freq_embed_dim,
            cfg.adaln_tembed_dim,
            vb.pp("mlp").pp("0"),
        )?;
        let mlp2 = layers::linear(
            cfg.adaln_tembed_dim,
            cfg.adaln_tembed_dim,
            vb.pp("mlp").pp("2"),
        )?;
        Ok(Self {
            mlp0,
            mlp2,
            freq_dim: cfg.freq_embed_dim,
        })
    }

    pub(crate) fn forward(&self, t: &Tensor, dtype: DType) -> Result<Tensor> {
        timestep_embedding(t, self.freq_dim, dtype)?
            .apply(&self.mlp0)?
            .silu()?
            .apply(&self.mlp2)
    }
}

/// UMT5 caption projection: Linear -> GELU(tanh) -> Linear; output feeds text cross-attn K/V.
pub(crate) struct CaptionEmbedder {
    y0: Linear,
    y2: Linear,
}

impl CaptionEmbedder {
    pub(crate) fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.caption_channels;
        Ok(Self {
            y0: layers::linear(d, d, vb.pp("y_proj").pp("0"))?,
            y2: layers::linear(d, d, vb.pp("y_proj").pp("2"))?,
        })
    }

    pub(crate) fn forward(&self, txt: &Tensor) -> Result<Tensor> {
        txt.apply(&self.y0)?.gelu()?.apply(&self.y2)
    }
}

/// fp32 final layer: LayerNorm + adaLN(shift, scale) + Linear(hidden -> patch_t*patch_h*patch_w*out).
pub(crate) struct FinalLayer {
    norm: hanzo_nn::LayerNorm,
    ada_ln: Linear,
    linear: Linear,
}

impl FinalLayer {
    pub(crate) fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device().clone();
        let norm = no_affine_layer_norm(cfg.hidden_size, dtype, &dev)?;
        let ada_ln = layers::linear(
            cfg.adaln_tembed_dim,
            FINAL_ADALN_CHUNKS * cfg.hidden_size,
            vb.pp("adaLN_modulation").pp("1"),
        )?;
        let linear = layers::linear(cfg.hidden_size, cfg.patch_out(), vb.pp("linear"))?;
        Ok(Self {
            norm,
            ada_ln,
            linear,
        })
    }

    // x: [B, N, hidden]; cond: [B, tembed] -> [B, N, patch_out].
    pub(crate) fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let chunks = cond
            .silu()?
            .apply(&self.ada_ln)?
            .chunk(FINAL_ADALN_CHUNKS, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        x.apply(&self.norm)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?
            .apply(&self.linear)
    }
}
