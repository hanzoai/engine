//! TRELLIS `SLatMeshDecoder`: decodes the SLAT latent into the per-cube FlexiCubes features.
//!
//! A 12-block sparse SWIN transformer runs at res 64, then two `SparseSubdivideBlock3d`s upsample
//! 64 -> 128 -> 256 (each voxel -> 2^3 children through a sparse-conv residual), and a final linear
//! projects to the 101-channel FlexiCubes layout (sdf 8 + deform 24 + weights 21 + color 48). The
//! surface is then extracted by the flexicubes module.

use hanzo_ml::{Result, Tensor, D};
use hanzo_nn::{Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use super::sparse::{
    subdivide, window_partition, AbsolutePositionEmbedder, Sparse, SparseGroupNorm32, SparseLinear,
    SubMConv3d,
};
use super::transformer::{nonaffine_layernorm, sdpa};
use crate::layers::linear;

const NORM_EPS: f64 = 1e-6;
const WINDOW: i32 = 8;
const GROUPS: usize = 32;

/// Windowed multi-head self-attention over the active voxels (TRELLIS swin): partition into
/// `WINDOW`^3 cells (optionally shifted), full attention inside each cell.
struct WindowAttn {
    to_qkv: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl WindowAttn {
    fn new(channels: usize, heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            to_qkv: linear(channels, channels * 3, vb.pp("to_qkv"))?,
            to_out: linear(channels, channels, vb.pp("to_out"))?,
            heads,
            head_dim: channels / heads,
        })
    }

    fn forward(&self, feats: &Tensor, coords: &[[i32; 3]], shift: i32) -> Result<Tensor> {
        let n = feats.dim(0)?;
        let (h, d) = (self.heads, self.head_dim);
        let dev = feats.device();
        let qkv = feats.apply(&self.to_qkv)?.reshape((n, 3, h, d))?;
        let q = qkv.narrow(1, 0, 1)?.reshape((n, h, d))?;
        let k = qkv.narrow(1, 1, 1)?.reshape((n, h, d))?;
        let v = qkv.narrow(1, 2, 1)?.reshape((n, h, d))?;

        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut outs: Vec<Tensor> = Vec::new();
        for group in window_partition(coords, WINDOW, shift) {
            let w = group.len();
            let idx = Tensor::from_vec(group.clone(), w, dev)?;
            let qg = q.index_select(&idx, 0)?.reshape((1, w, h, d))?;
            let kg = k.index_select(&idx, 0)?.reshape((1, w, h, d))?;
            let vg = v.index_select(&idx, 0)?.reshape((1, w, h, d))?;
            outs.push(sdpa(&qg, &kg, &vg)?.reshape((w, h, d))?);
            order.extend(group);
        }
        let cat = Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0)?; // [N,h,d] group order
        // invert the group ordering back to the original voxel order.
        let mut inv = vec![0u32; n];
        for (pos, &orig) in order.iter().enumerate() {
            inv[orig as usize] = pos as u32;
        }
        let out = cat.index_select(&Tensor::from_vec(inv, n, dev)?, 0)?.reshape((n, h * d))?;
        out.apply(&self.to_out)
    }
}

/// Plain sparse transformer block: non-affine-norm -> windowed self-attn -> non-affine-norm -> FFN.
struct TransformerBlock {
    attn: WindowAttn,
    fc1: Linear,
    fc2: Linear,
    shift: i32,
}

impl TransformerBlock {
    fn new(channels: usize, heads: usize, mlp_ratio: usize, shift: i32, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            attn: WindowAttn::new(channels, heads, vb.pp("attn"))?,
            fc1: linear(channels, channels * mlp_ratio, vb.pp("mlp.mlp.0"))?,
            fc2: linear(channels * mlp_ratio, channels, vb.pp("mlp.mlp.2"))?,
            shift,
        })
    }

    fn forward(&self, x: &Sparse) -> Result<Sparse> {
        let h = nonaffine_layernorm(&x.feats, NORM_EPS)?;
        let h = self.attn.forward(&h, &x.coords, self.shift)?;
        let x_feats = (&x.feats + h)?;
        let h = nonaffine_layernorm(&x_feats, NORM_EPS)?;
        let h = h.apply(&self.fc1)?.gelu()?.apply(&self.fc2)?;
        Ok(x.replace((x_feats + h)?))
    }
}

/// TRELLIS `SparseSubdivideBlock3d`: groupnorm+silu, subdivide, conv-norm-silu-conv residual + skip.
struct SubdivideBlock {
    act_norm: SparseGroupNorm32,
    conv1: SubMConv3d,
    norm: SparseGroupNorm32,
    conv2: SubMConv3d,
    skip: Option<SubMConv3d>, // 1x1x1 conv when channels change
    out_channels: usize,
}

impl SubdivideBlock {
    fn new(channels: usize, out_channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let skip = if channels != out_channels {
            Some(SubMConv3d::new(channels, out_channels, 1, true, vb.pp("skip_connection").pp("conv"))?)
        } else {
            None
        };
        Ok(Self {
            act_norm: SparseGroupNorm32::new(GROUPS, channels, vb.pp("act_layers").pp("0"))?,
            conv1: SubMConv3d::new(channels, out_channels, 3, true, vb.pp("out_layers").pp("0").pp("conv"))?,
            norm: SparseGroupNorm32::new(GROUPS, out_channels, vb.pp("out_layers").pp("1"))?,
            conv2: SubMConv3d::new(out_channels, out_channels, 3, true, vb.pp("out_layers").pp("3").pp("conv"))?,
            skip,
            out_channels,
        })
    }

    fn forward(&self, x: &Sparse) -> Result<Sparse> {
        let h = self.act_norm.forward_feats(&x.feats)?.silu()?;
        let h = subdivide(&Sparse::new(x.coords.clone(), h))?; // [8N, C]
        let xs = subdivide(x)?; // subdivided original for the skip
        let hc = self.conv1.forward_feats(&h.coords, &h.feats)?;
        let hc = self.norm.forward_feats(&hc)?.silu()?;
        let hc = self.conv2.forward_feats(&h.coords, &hc)?;
        let skip = match &self.skip {
            Some(s) => s.forward_feats(&xs.coords, &xs.feats)?,
            None => xs.feats.clone(),
        };
        Ok(Sparse::new(h.coords, (hc + skip)?))
    }
}

#[derive(Debug, Clone)]
pub struct SlatDecoderConfig {
    pub latent_channels: usize,
    pub model_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub out_channels: usize, // FlexiCubes feats: 101 with color
}

impl Default for SlatDecoderConfig {
    /// `slat_dec_mesh_swin8_B_64l8m256c` (use_color=true -> 101 channels).
    fn default() -> Self {
        Self {
            latent_channels: 8,
            model_channels: 768,
            num_blocks: 12,
            num_heads: 12,
            mlp_ratio: 4,
            out_channels: 101,
        }
    }
}

pub struct SlatMeshDecoder {
    input_layer: SparseLinear,
    pos_embedder: AbsolutePositionEmbedder,
    blocks: Vec<TransformerBlock>,
    upsample: Vec<SubdivideBlock>,
    out_layer: SparseLinear,
}

impl SlatMeshDecoder {
    pub fn new(cfg: &SlatDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mc = cfg.model_channels;
        let input_layer = SparseLinear::new(cfg.latent_channels, mc, vb.pp("input_layer"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.num_blocks);
        for i in 0..cfg.num_blocks {
            let shift = if i % 2 == 1 { WINDOW / 2 } else { 0 };
            blocks.push(TransformerBlock::new(mc, cfg.num_heads, cfg.mlp_ratio, shift, vb_b.pp(i))?);
        }
        let vb_u = vb.pp("upsample");
        let upsample = vec![
            SubdivideBlock::new(mc, mc / 4, vb_u.pp(0))?,
            SubdivideBlock::new(mc / 4, mc / 8, vb_u.pp(1))?,
        ];
        let out_layer = SparseLinear::new(mc / 8, cfg.out_channels, vb.pp("out_layer"))?;
        Ok(Self {
            input_layer,
            pos_embedder: AbsolutePositionEmbedder::new(mc),
            blocks,
            upsample,
            out_layer,
        })
    }

    /// `x` SLAT latent [N,8] at res 64 -> per-cube FlexiCubes features [M,101] at res 256.
    pub fn forward(&self, x: &Sparse) -> Result<Sparse> {
        let dev = x.feats.device().clone();
        let mut h = self.input_layer.forward(x)?;
        let pe = self.pos_embedder.forward(&h.coords, &dev)?;
        h = h.replace((&h.feats + pe)?);
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        for blk in &self.upsample {
            h = blk.forward(&h)?;
        }
        self.out_layer.forward(&h)
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

    #[test]
    #[ignore = "needs TRELLIS_FIX + mesh decoder weights"]
    fn slat_decoder_parity() {
        let dir = std::env::var("TRELLIS_FIX").expect("TRELLIS_FIX");
        let wdir = std::env::var("PIXAL3D_MODEL").expect("PIXAL3D_MODEL");
        let dev = Device::Cpu;
        let w = format!("{wdir}/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors");
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
        let model = SlatMeshDecoder::new(&SlatDecoderConfig::default(), vb).unwrap();

        let io = hanzo_ml::safetensors::load(format!("{dir}/mesh_dec_io.safetensors"), &dev).unwrap();
        let coords: Vec<[i32; 3]> = io["coords"]
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| [r[1] as i32, r[2] as i32, r[3] as i32])
            .collect();
        let x = Sparse::new(coords, io["feats"].clone());
        let out = model.forward(&x).unwrap();
        assert_eq!(out.n(), io["out_coords"].dim(0).unwrap());
        let (cos, mx, mse) = cos_stats(&out.feats, &io["out_feats"]);
        println!("slat_decoder cos={cos:.8} max|d|={mx:.3e} mse={mse:.3e}");
        assert!(cos > 0.999, "cosine {cos} < 0.999");
    }
}
