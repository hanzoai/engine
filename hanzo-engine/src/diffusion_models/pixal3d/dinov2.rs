//! DINOv2 ViT-L/14 with registers - the TRELLIS image conditioner.
//!
//! TRELLIS conditions both flow stages on `dinov2_vitl14_reg`'s `x_prenorm` tokens (the full
//! sequence before the final norm), then applies a non-affine LayerNorm over the channel dim. The
//! sequence is [CLS, 4 registers, 1369 patches] = 1374 tokens at 518x518; registers are inserted
//! after CLS with no positional embedding. GELU is exact-erf (torch `nn.GELU`), norms are eps 1e-6,
//! and TRELLIS's outer LayerNorm is eps 1e-5.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use hanzo_ml::{IndexOp, Result, Tensor, D};
use hanzo_nn::{Conv2d, Conv2dConfig, LayerNorm};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{conv2d, layer_norm, linear};

const BLOCK_EPS: f64 = 1e-6;
const FINAL_LN_EPS: f64 = 1e-5;

#[derive(Debug, Clone)]
pub struct DinoV2Config {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub img_size: usize,
    pub num_register_tokens: usize,
    pub mlp_ratio: usize,
}

impl Default for DinoV2Config {
    /// `dinov2_vitl14_reg` as loaded by TRELLIS.
    fn default() -> Self {
        Self {
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            patch_size: 14,
            img_size: 518,
            num_register_tokens: 4,
            mlp_ratio: 4,
        }
    }
}

impl DinoV2Config {
    fn num_patches(&self) -> usize {
        (self.img_size / self.patch_size).pow(2)
    }
}

struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(cfg: &DinoV2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let proj = conv2d(
            3,
            cfg.embed_dim,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = Convolution.forward_2d(&self.proj, x)?; // [B, C, H/p, W/p]
        let (b, c, h, w) = x.dims4()?;
        x.reshape((b, c, h * w))?.transpose(1, 2) // [B, N, C]
    }
}

struct Attention {
    qkv: hanzo_nn::Linear,
    proj: hanzo_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &DinoV2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        Ok(Self {
            qkv: linear(d, 3 * d, vb.pp("qkv"))?,
            proj: linear(d, d, vb.pp("proj"))?,
            num_heads: cfg.num_heads,
            head_dim: d / cfg.num_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let qkv = x
            .apply(&self.qkv)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?; // [3, B, H, N, D]
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = hanzo_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [B, H, N, D]
        out.transpose(1, 2)?.reshape((b, n, c))?.apply(&self.proj)
    }
}

struct Mlp {
    fc1: hanzo_nn::Linear,
    fc2: hanzo_nn::Linear,
}

impl Mlp {
    fn new(cfg: &DinoV2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        Ok(Self {
            fc1: linear(d, d * cfg.mlp_ratio, vb.pp("fc1"))?,
            fc2: linear(d * cfg.mlp_ratio, d, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: Tensor,
}

impl Block {
    fn new(cfg: &DinoV2Config, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(cfg.embed_dim, BLOCK_EPS, vb.pp("norm1"))?,
            attn: Attention::new(cfg, vb.pp("attn"))?,
            ls1: vb.get(cfg.embed_dim, "ls1.gamma")?,
            norm2: layer_norm(cfg.embed_dim, BLOCK_EPS, vb.pp("norm2"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            ls2: vb.get(cfg.embed_dim, "ls2.gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self
            .attn
            .forward(&x.apply(&self.norm1)?)?
            .broadcast_mul(&self.ls1)?;
        let x = (x + attn)?;
        let mlp = self
            .mlp
            .forward(&x.apply(&self.norm2)?)?
            .broadcast_mul(&self.ls2)?;
        x + mlp
    }
}

/// Non-affine LayerNorm over the last dim (TRELLIS's outer `F.layer_norm(x, [C])`).
fn layer_norm_nonaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

pub struct DinoV2 {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    register_tokens: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    cfg: DinoV2Config,
}

impl DinoV2 {
    pub fn new(cfg: &DinoV2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let cls_token = vb.get((1, 1, cfg.embed_dim), "cls_token")?;
        let register_tokens = vb.get(
            (1, cfg.num_register_tokens, cfg.embed_dim),
            "register_tokens",
        )?;
        let pos_embed = vb.get((1, cfg.num_patches() + 1, cfg.embed_dim), "pos_embed")?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::new(cfg, vb_b.pp(i))?);
        }
        Ok(Self {
            patch_embed,
            cls_token,
            register_tokens,
            pos_embed,
            blocks,
            cfg: cfg.clone(),
        })
    }

    /// `pixel_values` is [B, 3, 518, 518], ImageNet-normalized. Returns [B, 1374, 1024] conditioning
    /// tokens (TRELLIS `x_prenorm` after the non-affine outer LayerNorm).
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        let d = self.cfg.embed_dim;
        let patches = self.patch_embed.forward(pixel_values)?; // [B, 1369, C]
        let cls = self.cls_token.broadcast_as((b, 1, d))?;
        let x = Tensor::cat(&[&cls, &patches], 1)?; // [B, 1370, C]
        let x = x.broadcast_add(&self.pos_embed)?;
        let reg = self
            .register_tokens
            .broadcast_as((b, self.cfg.num_register_tokens, d))?;
        let mut x = Tensor::cat(&[&x.i((.., ..1))?, &reg, &x.i((.., 1..))?], 1)?; // [B, 1374, C]
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        layer_norm_nonaffine(&x, FINAL_LN_EPS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device};
    use std::path::PathBuf;
    use std::sync::Arc;

    // Run with the oracle from scratchpad/trellis_oracle/build_dinov2_oracle.sh:
    //   TRELLIS_ORACLE=/path/to/trellis_oracle \
    //   cargo test -p hanzo-engine pixal3d::dinov2 -- --ignored --nocapture
    fn oracle_dir() -> String {
        std::env::var("TRELLIS_ORACLE").expect("set TRELLIS_ORACLE to the oracle dir")
    }

    fn cosine(got: &Tensor, want: &Tensor) -> f64 {
        let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            maxabs = maxabs.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!("dinov2 cos={cos:.8}  max|d|={maxabs:.3e}");
        cos
    }

    #[test]
    #[ignore = "needs dinov2 oracle (TRELLIS_ORACLE dir)"]
    fn dinov2_parity_vs_reference() {
        let dir = oracle_dir();
        let dev = Device::Cpu;
        let vb = from_mmaped_safetensors(
            vec![PathBuf::from(format!(
                "{dir}/dinov2_vitl14_reg.safetensors"
            ))],
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
        let model = DinoV2::new(&DinoV2Config::default(), vb).unwrap();

        let inp =
            hanzo_ml::safetensors::load(format!("{dir}/dinov2_input.safetensors"), &dev).unwrap();
        let want =
            hanzo_ml::safetensors::load(format!("{dir}/dinov2_patchtokens.safetensors"), &dev)
                .unwrap();
        let out = model.forward(&inp["x"]).unwrap();
        assert_eq!(out.dims(), &[1, 1374, 1024]);
        let cos = cosine(&out, &want["patchtokens"]);
        assert!(cos > 0.999, "cosine {cos} < 0.999");
    }
}
