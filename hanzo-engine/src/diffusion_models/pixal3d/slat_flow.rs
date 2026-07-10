//! TRELLIS `SLatFlowModel`: the sparse rectified-flow denoiser over the structured latent.
//!
//! The active voxels of the 64^3 occupancy grid carry an 8-channel latent. A sparse ResBlock stack
//! packs (downsamples 64->32) into the transformer resolution, 24 shared `ModulatedCrossBlock`s run
//! full self-attention + DINOv2 cross-attention (B=1, so full attention == dense over the voxel set),
//! and a paired ResBlock stack unpacks (upsamples 32->64) with U-Net skips back to the 8-channel
//! velocity. Only the pack/unpack path is sparse-conv specific; the torso reuses the SS-flow block.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use hanzo_ml::{Result, Tensor, D};
use hanzo_nn::{Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use super::sparse::{
    downsample2, upsample2, AbsolutePositionEmbedder, DownCache, Sparse, SparseLinear, SubMConv3d,
};
use super::transformer::{nonaffine_layernorm, ModulatedCrossBlock, TimestepEmbedder};
use crate::layers::{layer_norm, linear};

const NORM_EPS: f64 = 1e-6; // block LayerNorm32 eps
const FINAL_LN_EPS: f64 = 1e-5; // F.layer_norm default eps
const TIME_SCALE: f64 = 1000.0;

#[derive(Debug, Clone)]
pub struct SlatFlowConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub io_block_channels: usize, // single-stage patch_size=2
    pub num_io_res_blocks: usize,
    pub qk_rms_norm: bool,
}

impl Default for SlatFlowConfig {
    /// `slat_flow_img_dit_L_64l8p2`.
    fn default() -> Self {
        Self {
            in_channels: 8,
            out_channels: 8,
            model_channels: 1024,
            cond_channels: 1024,
            num_blocks: 24,
            num_heads: 16,
            mlp_ratio: 4,
            io_block_channels: 128,
            num_io_res_blocks: 2,
            qk_rms_norm: true,
        }
    }
}

enum UpDown {
    None,
    Down,
    Up,
}

/// TRELLIS `SparseResBlock3d`: affine-norm -> conv -> modulated-norm -> conv + skip, with optional
/// average-pool downsample / paired upsample applied first.
struct SparseResBlock3d {
    norm1: hanzo_nn::LayerNorm, // affine
    conv1: SubMConv3d,
    conv2: SubMConv3d,
    emb: Linear, // emb_layers.1: Linear(emb_channels, 2*out)
    skip: Option<SparseLinear>,
    updown: UpDown,
    #[allow(dead_code)]
    out_channels: usize,
}

impl SparseResBlock3d {
    fn new(
        channels: usize,
        emb_channels: usize,
        out_channels: usize,
        updown: UpDown,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let skip = if channels != out_channels {
            Some(SparseLinear::new(
                channels,
                out_channels,
                vb.pp("skip_connection"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: layer_norm(channels, NORM_EPS, vb.pp("norm1"))?,
            conv1: SubMConv3d::new(channels, out_channels, 3, true, vb.pp("conv1").pp("conv"))?,
            conv2: SubMConv3d::new(
                out_channels,
                out_channels,
                3,
                true,
                vb.pp("conv2").pp("conv"),
            )?,
            emb: linear(emb_channels, 2 * out_channels, vb.pp("emb_layers").pp("1"))?,
            skip,
            updown,
            out_channels,
        })
    }

    fn forward(&self, x: &Sparse, emb: &Tensor, cache: &mut Option<DownCache>) -> Result<Sparse> {
        let m = emb.silu()?.apply(&self.emb)?; // [1, 2*out]
        let parts = m.chunk(2, D::Minus1)?;
        let (scale, shift) = (&parts[0], &parts[1]); // [1, out]

        let x = match self.updown {
            UpDown::None => x.clone(),
            UpDown::Down => {
                let (d, c) = downsample2(x)?;
                *cache = Some(c);
                d
            }
            UpDown::Up => upsample2(x, cache.as_ref().expect("upsample needs paired downsample"))?,
        };

        let h = self.norm1.forward(&x.feats)?.silu()?;
        let h = self.conv1.forward_feats(&x.coords, &h)?;
        let h = nonaffine_layernorm(&h, NORM_EPS)?;
        let h = h
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(shift)?
            .silu()?;
        let h = self.conv2.forward_feats(&x.coords, &h)?;
        let skip = match &self.skip {
            Some(s) => s.forward_feats(&x.feats)?,
            None => x.feats.clone(),
        };
        Ok(Sparse::new(x.coords.clone(), (h + skip)?))
    }
}

pub struct SlatFlow {
    input_layer: SparseLinear,
    input_blocks: Vec<SparseResBlock3d>,
    pos_embedder: AbsolutePositionEmbedder,
    t_embedder: TimestepEmbedder,
    blocks: Vec<ModulatedCrossBlock>,
    out_blocks: Vec<SparseResBlock3d>,
    out_layer: SparseLinear,
    model_channels: usize,
}

impl SlatFlow {
    pub fn new(cfg: &SlatFlowConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mc = cfg.model_channels;
        let io = cfg.io_block_channels;
        let input_layer = SparseLinear::new(cfg.in_channels, io, vb.pp("input_layer"))?;

        // single stage (patch_size=2): (num_io_res_blocks-1) plain blocks then a downsample block.
        let vb_i = vb.pp("input_blocks");
        let mut input_blocks = Vec::new();
        let mut idx = 0;
        for _ in 0..cfg.num_io_res_blocks - 1 {
            input_blocks.push(SparseResBlock3d::new(
                io,
                mc,
                io,
                UpDown::None,
                vb_i.pp(idx),
            )?);
            idx += 1;
        }
        input_blocks.push(SparseResBlock3d::new(
            io,
            mc,
            mc,
            UpDown::Down,
            vb_i.pp(idx),
        )?);

        let t_embedder = TimestepEmbedder::new(mc, vb.pp("t_embedder"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.num_blocks);
        for i in 0..cfg.num_blocks {
            blocks.push(ModulatedCrossBlock::new(
                mc,
                cfg.cond_channels,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.qk_rms_norm,
                vb_b.pp(i),
            )?);
        }

        // paired unpack: one upsample block (in = 2*mc skip-concat) then plain blocks (in = 2*io).
        let vb_o = vb.pp("out_blocks");
        let mut out_blocks = Vec::new();
        let mut idx = 0;
        out_blocks.push(SparseResBlock3d::new(
            mc * 2,
            mc,
            io,
            UpDown::Up,
            vb_o.pp(idx),
        )?);
        idx += 1;
        for _ in 0..cfg.num_io_res_blocks - 1 {
            out_blocks.push(SparseResBlock3d::new(
                io * 2,
                mc,
                io,
                UpDown::None,
                vb_o.pp(idx),
            )?);
            idx += 1;
        }

        let out_layer = SparseLinear::new(io, cfg.out_channels, vb.pp("out_layer"))?;
        Ok(Self {
            input_layer,
            input_blocks,
            pos_embedder: AbsolutePositionEmbedder::new(mc),
            t_embedder,
            blocks,
            out_blocks,
            out_layer,
            model_channels: mc,
        })
    }

    /// `x` sparse latent [N,8] at res 64, `t` [1] in (0,1], `cond` [1, N_ctx, 1024]. Returns the
    /// velocity [N,8] at the same active coords.
    pub fn forward(&self, x: &Sparse, t: &Tensor, cond: &Tensor) -> Result<Sparse> {
        let dev = x.feats.device().clone();
        let t_emb = self.t_embedder.forward(&(t * TIME_SCALE)?)?; // [1, mc]

        let mut h = self.input_layer.forward(x)?;
        let mut cache: Option<DownCache> = None;
        let mut skips: Vec<Tensor> = Vec::new();
        for blk in &self.input_blocks {
            h = blk.forward(&h, &t_emb, &mut cache)?;
            skips.push(h.feats.clone());
        }

        let n = h.coords.len();
        let pe = self.pos_embedder.forward(&h.coords, &dev)?; // [N, mc]
                                                              // torso: full attention over the voxel set (B=1 -> dense [1, N, C]).
        let mut feats3 = (h.feats + pe)?.reshape((1, n, self.model_channels))?;
        for blk in &self.blocks {
            feats3 = blk.forward(&feats3, &t_emb, cond)?;
        }
        h = Sparse::new(h.coords.clone(), feats3.reshape((n, self.model_channels))?);

        // unpack with U-Net skips (reversed).
        for (blk, skip) in self.out_blocks.iter().zip(skips.iter().rev()) {
            let cat = Tensor::cat(&[&h.feats, skip], D::Minus1)?;
            h = blk.forward(&Sparse::new(h.coords.clone(), cat), &t_emb, &mut cache)?;
        }

        let feats = nonaffine_layernorm(&h.feats, FINAL_LN_EPS)?;
        self.out_layer
            .forward(&Sparse::new(h.coords.clone(), feats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion_models::pixal3d::sparse::cos_stats;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device};
    use std::path::PathBuf;
    use std::sync::Arc;

    // TRELLIS_FIX=/oracle/fixtures PIXAL3D_MODEL=/model \
    //   cargo test -p hanzo-engine pixal3d::slat_flow -- --ignored --nocapture
    #[test]
    #[ignore = "needs TRELLIS_FIX + slat_flow weights"]
    fn slat_flow_parity() {
        let dir = std::env::var("TRELLIS_FIX").expect("TRELLIS_FIX");
        let wdir = std::env::var("PIXAL3D_MODEL").expect("PIXAL3D_MODEL");
        let dev = Device::Cpu;
        let w = format!("{wdir}/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors");
        let vb = from_mmaped_safetensors(
            vec![PathBuf::from(w)],
            Vec::new(),
            Some(DType::F32),
            &dev,
            vec![None],
            true,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        let model = SlatFlow::new(&SlatFlowConfig::default(), vb).unwrap();

        let io =
            hanzo_ml::safetensors::load(format!("{dir}/slat_flow_io.safetensors"), &dev).unwrap();
        let coords: Vec<[i32; 3]> = io["coords"]
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| [r[1] as i32, r[2] as i32, r[3] as i32])
            .collect();
        let x = Sparse::new(coords, io["feats"].clone());
        let out = model.forward(&x, &io["t"], &io["cond"]).unwrap();
        let (cos, mx, mse) = cos_stats(&out.feats, &io["velocity"]);
        println!("slat_flow cos={cos:.8} max|d|={mx:.3e} mse={mse:.3e}");
        assert!(cos > 0.999, "cosine {cos} < 0.999");
    }
}
