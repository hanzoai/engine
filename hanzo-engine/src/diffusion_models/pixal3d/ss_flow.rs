//! TRELLIS `SparseStructureFlowModel`: a dense DiT over the 16^3 sparse-structure latent.
//!
//! patch_size 1, so the 16^3x8 latent is flattened to 4096 tokens, run through 24
//! cross-attention DiT blocks conditioned on the DINOv2 tokens, and projected back to a
//! 16^3x8 velocity for the rectified-flow sampler. Positional embedding is the stored `pos_emb`
//! (an AbsolutePositionEmbedder over the 16^3 grid, baked into the checkpoint).

use hanzo_ml::{Result, Tensor};
use hanzo_nn::{Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use super::transformer::{nonaffine_layernorm, ModulatedCrossBlock, TimestepEmbedder};
use crate::layers::linear;

const FINAL_LN_EPS: f64 = 1e-5;
const TIME_SCALE: f64 = 1000.0;

#[derive(Debug, Clone)]
pub struct SsFlowConfig {
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub qk_rms_norm: bool,
}

impl Default for SsFlowConfig {
    /// `ss_flow_img_dit_L_16l8`.
    fn default() -> Self {
        Self {
            resolution: 16,
            in_channels: 8,
            out_channels: 8,
            model_channels: 1024,
            cond_channels: 1024,
            num_blocks: 24,
            num_heads: 16,
            mlp_ratio: 4,
            qk_rms_norm: true,
        }
    }
}

pub struct SsFlow {
    input_layer: Linear,
    pos_emb: Tensor, // [N, C]
    t_embedder: TimestepEmbedder,
    blocks: Vec<ModulatedCrossBlock>,
    out_layer: Linear,
    out_channels: usize,
}

impl SsFlow {
    pub fn new(cfg: &SsFlowConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let n = cfg.resolution.pow(3);
        let input_layer = linear(cfg.in_channels, cfg.model_channels, vb.pp("input_layer"))?;
        let pos_emb = vb.get((n, cfg.model_channels), "pos_emb")?;
        let t_embedder = TimestepEmbedder::new(cfg.model_channels, vb.pp("t_embedder"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.num_blocks);
        for i in 0..cfg.num_blocks {
            blocks.push(ModulatedCrossBlock::new(
                cfg.model_channels,
                cfg.cond_channels,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.qk_rms_norm,
                vb_b.pp(i),
            )?);
        }
        let out_layer = linear(cfg.model_channels, cfg.out_channels, vb.pp("out_layer"))?;
        Ok(Self {
            input_layer,
            pos_emb,
            t_embedder,
            blocks,
            out_layer,
            out_channels: cfg.out_channels,
        })
    }

    /// `x` [B, C_in, R, R, R], `t` [B] in (0,1], `cond` [B, N_ctx, cond_ch]. Returns the velocity
    /// [B, C_out, R, R, R].
    pub fn forward(&self, x: &Tensor, t: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (b, c, r) = (dims[0], dims[1], dims[2]);
        let n = r * r * r;
        let mut h = x.reshape((b, c, n))?.transpose(1, 2)?.contiguous()?; // [B, N, C_in]
        h = h.apply(&self.input_layer)?; // [B, N, model]
        h = h.broadcast_add(&self.pos_emb.unsqueeze(0)?)?;
        let t_emb = self.t_embedder.forward(&(t * TIME_SCALE)?)?; // [B, model]
        for blk in &self.blocks {
            h = blk.forward(&h, &t_emb, cond)?;
        }
        h = nonaffine_layernorm(&h, FINAL_LN_EPS)?;
        h = h.apply(&self.out_layer)?; // [B, N, C_out]
        h.transpose(1, 2)?
            .contiguous()?
            .reshape((b, self.out_channels, r, r, r))
    }
}
