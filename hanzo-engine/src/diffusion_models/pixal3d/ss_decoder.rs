//! TRELLIS `SparseStructureDecoder`: 3D-conv upsampler from the 16^3x8 latent to a 64^3
//! occupancy-logit grid.
//!
//! hanzo-ml has no Conv3d, so the only kernel the decoder uses (3x3x3, pad 1, stride 1) is
//! decomposed into three depth-sliced conv2d calls summed with the right depth alignment - reusing
//! the optimized 2D conv. Norm is a channel-last affine LayerNorm; upsampling is Conv3d(c -> 8c)
//! then a 3D pixel shuffle.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use hanzo_ml::{Result, Tensor};
use hanzo_nn::LayerNorm;
use hanzo_quant::ShardedVarBuilder;

use crate::layers::layer_norm;

const NORM_EPS: f64 = 1e-5; // ChannelLayerNorm32 -> nn.LayerNorm default

#[derive(Debug, Clone)]
pub struct SsDecoderConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub channels: Vec<usize>,
    pub num_res_blocks: usize,
    pub num_res_blocks_middle: usize,
}

impl Default for SsDecoderConfig {
    /// `ss_dec_conv3d_16l8`.
    fn default() -> Self {
        Self {
            latent_channels: 8,
            out_channels: 1,
            channels: vec![512, 128, 32],
            num_res_blocks: 2,
            num_res_blocks_middle: 2,
        }
    }
}

fn dims5(x: &Tensor) -> (usize, usize, usize, usize, usize) {
    let d = x.dims();
    (d[0], d[1], d[2], d[3], d[4])
}

/// 3x3x3, pad 1, stride 1 conv, decomposed over the depth axis into three conv2d.
struct Conv3d {
    weight: Tensor, // [Cout, Cin, 3, 3, 3]
    bias: Tensor,   // [Cout]
}

impl Conv3d {
    fn new(cin: usize, cout: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get((cout, cin, 3, 3, 3), "weight")?,
            bias: vb.get(cout, "bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, cin, d, h, w) = dims5(x);
        let cout = self.weight.dim(0)?;
        let xp = x.pad_with_zeros(2, 1, 1)?; // [B, Cin, D+2, H, W]
        let mut acc: Option<Tensor> = None;
        for dz in 0..3 {
            let xs = xp
                .narrow(2, dz, d)?
                .permute((0, 2, 1, 3, 4))?
                .contiguous()?
                .reshape((b * d, cin, h, w))?;
            let w2 = self.weight.narrow(2, dz, 1)?.squeeze(2)?.contiguous()?; // [Cout, Cin, 3, 3]
            let y = xs
                .conv2d(&w2, 1, 1, 1, 1)?
                .reshape((b, d, cout, h, w))?
                .permute((0, 2, 1, 3, 4))?
                .contiguous()?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        acc.unwrap()
            .broadcast_add(&self.bias.reshape((1, cout, 1, 1, 1))?)
    }
}

/// Affine LayerNorm over the channel axis of a [B, C, D, H, W] volume.
struct ChannelLayerNorm {
    ln: LayerNorm,
}

impl ChannelLayerNorm {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            ln: layer_norm(channels, NORM_EPS, vb)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.permute((0, 2, 3, 4, 1))?
            .contiguous()?
            .apply(&self.ln)?
            .permute((0, 4, 1, 2, 3))?
            .contiguous()
    }
}

struct ResBlock {
    norm1: ChannelLayerNorm,
    conv1: Conv3d,
    norm2: ChannelLayerNorm,
    conv2: Conv3d,
}

impl ResBlock {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: ChannelLayerNorm::new(channels, vb.pp("norm1"))?,
            conv1: Conv3d::new(channels, channels, vb.pp("conv1"))?,
            norm2: ChannelLayerNorm::new(channels, vb.pp("norm2"))?,
            conv2: Conv3d::new(channels, channels, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;
        h + x
    }
}

/// 3D pixel shuffle by 2: [B, 8c, H, W, D] -> [B, c, 2H, 2W, 2D].
fn pixel_shuffle_3d(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w, d) = dims5(x);
    let c_ = c / 8;
    // 8-D reshape/permute: tuple Shape/Dims only go to rank 6, so use Vec/slice forms.
    x.reshape(vec![b, c_, 2, 2, 2, h, w, d])?
        .permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?
        .contiguous()?
        .reshape((b, c_, h * 2, w * 2, d * 2))
}

struct Upsample {
    conv: Conv3d,
}

impl Upsample {
    fn new(cin: usize, cout: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            conv: Conv3d::new(cin, cout * 8, vb.pp("conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        pixel_shuffle_3d(&self.conv.forward(x)?)
    }
}

enum Block {
    Res(ResBlock),
    Up(Upsample),
}

pub struct SsDecoder {
    input_layer: Conv3d,
    middle: Vec<ResBlock>,
    blocks: Vec<Block>,
    out_norm: ChannelLayerNorm,
    out_conv: Conv3d,
}

impl SsDecoder {
    pub fn new(cfg: &SsDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let ch = &cfg.channels;
        let input_layer = Conv3d::new(cfg.latent_channels, ch[0], vb.pp("input_layer"))?;

        let vb_m = vb.pp("middle_block");
        let mut middle = Vec::with_capacity(cfg.num_res_blocks_middle);
        for i in 0..cfg.num_res_blocks_middle {
            middle.push(ResBlock::new(ch[0], vb_m.pp(i))?);
        }

        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::new();
        let mut idx = 0;
        for (i, &c) in ch.iter().enumerate() {
            for _ in 0..cfg.num_res_blocks {
                blocks.push(Block::Res(ResBlock::new(c, vb_b.pp(idx))?));
                idx += 1;
            }
            if i < ch.len() - 1 {
                blocks.push(Block::Up(Upsample::new(c, ch[i + 1], vb_b.pp(idx))?));
                idx += 1;
            }
        }

        let last = *ch.last().unwrap();
        let out_norm = ChannelLayerNorm::new(last, vb.pp("out_layer.0"))?;
        let out_conv = Conv3d::new(last, cfg.out_channels, vb.pp("out_layer.2"))?;
        Ok(Self {
            input_layer,
            middle,
            blocks,
            out_norm,
            out_conv,
        })
    }

    /// `x` [B, latent, 16, 16, 16] -> occupancy logits [B, out, 64, 64, 64].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.input_layer.forward(x)?;
        for rb in &self.middle {
            h = rb.forward(&h)?;
        }
        for blk in &self.blocks {
            h = match blk {
                Block::Res(b) => b.forward(&h)?,
                Block::Up(b) => b.forward(&h)?,
            };
        }
        self.out_conv.forward(&self.out_norm.forward(&h)?.silu()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device};
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    #[ignore = "needs ss_dec oracle (TRELLIS_ORACLE dir)"]
    fn ss_decoder_parity_vs_reference() {
        let dir = std::env::var("TRELLIS_ORACLE").expect("set TRELLIS_ORACLE");
        let dev = Device::Cpu;
        let weights = format!("{dir}/trellis_dl/ckpts/ss_dec_conv3d_16l8_fp16.safetensors");
        let vb = from_mmaped_safetensors(
            vec![PathBuf::from(weights)],
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
        let model = SsDecoder::new(&SsDecoderConfig::default(), vb).unwrap();

        let io = hanzo_ml::safetensors::load(format!("{dir}/ssdec_io.safetensors"), &dev).unwrap();
        let out = model.forward(&io["z"]).unwrap();
        assert_eq!(out.dims(), &[1, 1, 64, 64, 64]);

        let a = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = io["logits"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (mut dot, mut na, mut nb, mut se, mut maxabs) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            se += (x - y).powi(2);
            maxabs = maxabs.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!(
            "ss_dec cos={cos:.8} mse={:.3e} max|d|={maxabs:.3e}",
            se / a.len() as f64
        );
        assert!(cos > 0.999, "cosine {cos} < 0.999");
    }
}
