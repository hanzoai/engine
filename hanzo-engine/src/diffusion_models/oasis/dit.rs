#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Oasis-500M DiT (`DiT-S/2`): a spatiotemporal diffusion transformer over frame-latents
//! `[B, T, 16, 18, 32]`. Patchify (conv 2x2) -> 16 blocks, each an adaLN spatial-attn + spatial-MLP
//! then adaLN causal-temporal-attn + temporal-MLP -> adaLN final -> unpatchify to a velocity latent.
//! Conditioning `c = timestep_embed(t) + action_embed(a)` is per (batch, frame). Spatial attention uses
//! full-head pixel axial RoPE; temporal attention uses full-head lang RoPE with a causal mask.

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::AttentionMask;
use crate::diffusion_models::oasis::rope::RotaryTable;
use crate::diffusion_models::wan::longcat::blocks::sdpa;
use crate::diffusion_models::wan::longcat::common::no_affine_layer_norm;
use crate::layers;

const DIM: usize = 1024;
const DEPTH: usize = 16;
const HEADS: usize = 16;
const HEAD_DIM: usize = DIM / HEADS; // 64
const FFN: usize = DIM * 4;
const PATCH: usize = 2;
const IN_CH: usize = 16;
const COND_DIM: usize = 25;
const FREQ_EMB: usize = 256;
const SPATIAL_MAX_FREQ: f64 = 256.0;
pub const MAX_FRAMES: usize = 32;
pub const LATENT_H: usize = 18;
pub const LATENT_W: usize = 32;
const GRID_H: usize = LATENT_H / PATCH; // 9
const GRID_W: usize = LATENT_W / PATCH; // 16
const PATCH_OUT: usize = PATCH * PATCH * IN_CH; // 64
const MASK_NEG: f64 = -1e9;

// Precomputed at init (pitfall #5: no Tensor::arange/from_vec in the per-forward hot loop).
fn timestep_freqs(dev: &Device) -> Result<Tensor> {
    let half = FREQ_EMB / 2;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    (arange * (-(10000f64.ln()) / half as f64))?.exp()
}

// oasis sinusoidal timestep features: discrete noise indices used directly (no /1000 flow scaling).
fn timestep_embedding(t: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let args = t
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

// Full [1,1,MAX_FRAMES,MAX_FRAMES] additive causal mask; a leading [tt,tt] slice is the tt-frame mask.
fn causal_full_mask(dtype: DType, dev: &Device) -> Result<Tensor> {
    let n = MAX_FRAMES;
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            v[i * n + j] = MASK_NEG as f32;
        }
    }
    Tensor::from_vec(v, (1, 1, n, n), dev)?.to_dtype(dtype)
}

// adaLN modulation: x [B,T,H,W,D] * (1+scale) + shift; scale/shift/gate broadcast [B,T,1,1,D].
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

fn gate(x: &Tensor, g: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(g)
}

// candle Linear only matmuls up to rank 4; apply over the last dim of an arbitrary-rank tensor.
fn apply_linear(x: &Tensor, l: &Linear) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d = dims[dims.len() - 1];
    let out = x.contiguous()?.reshape((x.elem_count() / d, d))?.apply(l)?;
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(out.dim(1)?);
    out.reshape(shape)
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: layers::linear(DIM, FFN, vb.pp("fc1"))?,
            fc2: layers::linear(FFN, DIM, vb.pp("fc2"))?,
        })
    }

    // tanh-approx GELU (DiT's explicit `nn.GELU(approximate="tanh")`).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        apply_linear(&apply_linear(x, &self.fc1)?.gelu()?, &self.fc2)
    }
}

// SiLU + Linear(D, 6D), split into (shift,scale,gate) for attn and (shift,scale,gate) for mlp.
struct AdaLn {
    lin: Linear,
}

impl AdaLn {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            lin: layers::linear(DIM, 6 * DIM, vb.pp("1"))?,
        })
    }

    fn six(&self, c: &Tensor) -> Result<[Tensor; 6]> {
        let (b, t, _) = c.dims3()?;
        let m = c.silu()?.apply(&self.lin)?;
        let mut out: Vec<Tensor> = Vec::with_capacity(6);
        for i in 0..6 {
            out.push(
                m.narrow(D::Minus1, i * DIM, DIM)?
                    .reshape((b, t, 1, 1, DIM))?,
            );
        }
        Ok(out.try_into().unwrap())
    }
}

// Axial attention over one grouping; `spatial` selects the (H,W)-flatten vs (T)-per-position layout.
struct AxialAttention {
    qkv: Linear,
    proj: Linear,
    spatial: bool,
}

impl AxialAttention {
    fn new(spatial: bool, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: layers::linear_no_bias(DIM, DIM * 3, vb.pp("to_qkv"))?,
            proj: layers::linear(DIM, DIM, vb.pp("to_out"))?,
            spatial,
        })
    }

    // x [B,T,H,W,D] -> [B,T,H,W,D]. rope is the spatial or per-window temporal table.
    fn forward(&self, x: &Tensor, rope: &RotaryTable, mask: &AttentionMask) -> Result<Tensor> {
        let (b, t, h, w, _) = x.dims5()?;
        let qkv = apply_linear(x, &self.qkv)?;
        // narrow one of q/k/v then fold into the attention batch layout.
        let heads = |i: usize| -> Result<Tensor> {
            let g = qkv.narrow(D::Minus1, i * DIM, DIM)?;
            if self.spatial {
                // (B T) HEADS (H W) hd
                g.reshape((b, t, h * w, HEADS, HEAD_DIM))?
                    .transpose(2, 3)?
                    .reshape((b * t, HEADS, h * w, HEAD_DIM))?
                    .contiguous()
            } else {
                // (B H W) HEADS T hd
                g.reshape((b, t, h, w, HEADS, HEAD_DIM))?
                    .permute([0, 2, 3, 4, 1, 5])?
                    .reshape((b * h * w, HEADS, t, HEAD_DIM))?
                    .contiguous()
            }
        };
        let q = rope.apply(&heads(0)?)?;
        let k = rope.apply(&heads(1)?)?;
        let v = heads(2)?;
        let out = sdpa(&q, &k, &v, mask)?;
        let merged = if self.spatial {
            out.reshape((b, t, HEADS, h * w, HEAD_DIM))?
                .transpose(2, 3)?
                .reshape((b, t, h, w, DIM))?
        } else {
            out.reshape((b, h, w, HEADS, t, HEAD_DIM))?
                .permute([0, 4, 1, 2, 3, 5])?
                .reshape((b, t, h, w, DIM))?
        };
        apply_linear(&merged, &self.proj)
    }
}

struct SpatioTemporalBlock {
    s_norm1: LayerNorm,
    s_attn: AxialAttention,
    s_norm2: LayerNorm,
    s_mlp: Mlp,
    s_ada: AdaLn,
    t_norm1: LayerNorm,
    t_attn: AxialAttention,
    t_norm2: LayerNorm,
    t_mlp: Mlp,
    t_ada: AdaLn,
}

impl SpatioTemporalBlock {
    fn new(dtype: DType, dev: &Device, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            s_norm1: no_affine_layer_norm(DIM, dtype, dev)?,
            s_attn: AxialAttention::new(true, vb.pp("s_attn"))?,
            s_norm2: no_affine_layer_norm(DIM, dtype, dev)?,
            s_mlp: Mlp::new(vb.pp("s_mlp"))?,
            s_ada: AdaLn::new(vb.pp("s_adaLN_modulation"))?,
            t_norm1: no_affine_layer_norm(DIM, dtype, dev)?,
            t_attn: AxialAttention::new(false, vb.pp("t_attn"))?,
            t_norm2: no_affine_layer_norm(DIM, dtype, dev)?,
            t_mlp: Mlp::new(vb.pp("t_mlp"))?,
            t_ada: AdaLn::new(vb.pp("t_adaLN_modulation"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        s_rope: &RotaryTable,
        t_rope: &RotaryTable,
        t_mask: &AttentionMask,
    ) -> Result<Tensor> {
        let [ss, sc, sg, ss2, sc2, sg2] = self.s_ada.six(c)?;
        let x = (x + gate(
            &self.s_attn.forward(
                &modulate(&x.apply(&self.s_norm1)?, &ss, &sc)?,
                s_rope,
                &AttentionMask::None,
            )?,
            &sg,
        )?)?;
        let x = (&x
            + gate(
                &self
                    .s_mlp
                    .forward(&modulate(&x.apply(&self.s_norm2)?, &ss2, &sc2)?)?,
                &sg2,
            )?)?;
        let [ts, tc, tg, ts2, tc2, tg2] = self.t_ada.six(c)?;
        let x = (&x
            + gate(
                &self.t_attn.forward(
                    &modulate(&x.apply(&self.t_norm1)?, &ts, &tc)?,
                    t_rope,
                    t_mask,
                )?,
                &tg,
            )?)?;
        &x + gate(
            &self
                .t_mlp
                .forward(&modulate(&x.apply(&self.t_norm2)?, &ts2, &tc2)?)?,
            &tg2,
        )?
    }
}

struct FinalLayer {
    norm: LayerNorm,
    ada: Linear,
    linear: Linear,
}

impl FinalLayer {
    fn new(dtype: DType, dev: &Device, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            norm: no_affine_layer_norm(DIM, dtype, dev)?,
            ada: layers::linear(DIM, 2 * DIM, vb.pp("adaLN_modulation").pp("1"))?,
            linear: layers::linear(DIM, PATCH_OUT, vb.pp("linear"))?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let (b, t, _) = c.dims3()?;
        let m = c.silu()?.apply(&self.ada)?;
        let shift = m.narrow(D::Minus1, 0, DIM)?.reshape((b, t, 1, 1, DIM))?;
        let scale = m.narrow(D::Minus1, DIM, DIM)?.reshape((b, t, 1, 1, DIM))?;
        apply_linear(
            &modulate(&x.apply(&self.norm)?, &shift, &scale)?,
            &self.linear,
        )
    }
}

/// The assembled Oasis DiT velocity predictor.
pub struct Dit {
    x_proj: Conv2d,
    t_mlp0: Linear,
    t_mlp2: Linear,
    external_cond: Linear,
    blocks: Vec<SpatioTemporalBlock>,
    final_layer: FinalLayer,
    s_rope: RotaryTable,
    t_rope: RotaryTable,
    time_freqs: Tensor,
    causal_full: Tensor,
    device: Device,
    dtype: DType,
}

impl Dit {
    pub fn new(vb: ShardedVarBuilder, device: Device) -> Result<Self> {
        let dtype = vb.dtype();
        let xw = vb.get((DIM, IN_CH, PATCH, PATCH), "x_embedder.proj.weight")?;
        let xb = vb.get(DIM, "x_embedder.proj.bias")?;
        let x_proj = Conv2d::new(
            xw,
            Some(xb),
            Conv2dConfig {
                padding: 0,
                stride: PATCH,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        let te = vb.pp("t_embedder").pp("mlp");
        let t_mlp0 = layers::linear(FREQ_EMB, DIM, te.pp("0"))?;
        let t_mlp2 = layers::linear(DIM, DIM, te.pp("2"))?;
        let external_cond = layers::linear(COND_DIM, DIM, vb.pp("external_cond"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            blocks.push(SpatioTemporalBlock::new(dtype, &device, vb_b.pp(i))?);
        }
        let final_layer = FinalLayer::new(dtype, &device, vb.pp("final_layer"))?;
        Ok(Self {
            x_proj,
            t_mlp0,
            t_mlp2,
            external_cond,
            blocks,
            final_layer,
            s_rope: RotaryTable::spatial(GRID_H, GRID_W, HEAD_DIM, SPATIAL_MAX_FREQ, &device)?,
            t_rope: RotaryTable::temporal(MAX_FRAMES, HEAD_DIM, &device)?,
            time_freqs: timestep_freqs(&device)?,
            causal_full: causal_full_mask(dtype, &device)?,
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // t [B,T] noise levels -> conditioning c [B,T,DIM] (timestep + action embeddings).
    fn condition(&self, t: &Tensor, action: &Tensor) -> Result<Tensor> {
        let (b, tt) = t.dims2()?;
        let temb = timestep_embedding(&t.reshape(b * tt)?, &self.time_freqs)?
            .to_dtype(self.dtype)?
            .apply(&self.t_mlp0)?
            .silu()?
            .apply(&self.t_mlp2)?
            .reshape((b, tt, DIM))?;
        let aemb = action.to_dtype(self.dtype)?.apply(&self.external_cond)?;
        temb + aemb
    }

    fn unpatchify(&self, x: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        // x [B,T,GRID_H,GRID_W,PATCH_OUT] -> [B,T,16,18,32]
        x.reshape((b * t, GRID_H, GRID_W, PATCH, PATCH, IN_CH))?
            .permute([0, 5, 1, 3, 2, 4])? // n c h p w q
            .contiguous()?
            .reshape((b, t, IN_CH, LATENT_H, LATENT_W))
    }

    /// Velocity prediction for `x [B,T,16,18,32]` at noise levels `t [B,T]` under actions
    /// `action [B,T,25]`. Returns velocity `[B,T,16,18,32]`.
    pub fn forward(&self, x: &Tensor, t: &Tensor, action: &Tensor) -> Result<Tensor> {
        let (b, tt, _, _, _) = x.dims5()?;
        let c = self.condition(t, action)?;
        // patchify per (b*t): conv 2x2, keep [.,H,W,D]
        let xin = x
            .reshape((b * tt, IN_CH, LATENT_H, LATENT_W))?
            .to_dtype(self.dtype)?;
        let feat = xin.apply(&self.x_proj)?; // [BT, DIM, 9, 16]
        let mut h = feat
            .permute([0, 2, 3, 1])? // [BT, 9, 16, DIM]
            .contiguous()?
            .reshape((b, tt, GRID_H, GRID_W, DIM))?;
        let t_rope = self.t_rope.slice(tt)?;
        let t_mask = AttentionMask::Custom(
            self.causal_full.narrow(2, 0, tt)?.narrow(3, 0, tt)?.contiguous()?,
        );
        for blk in &self.blocks {
            h = blk.forward(&h, &c, &self.s_rope, &t_rope, &t_mask)?;
        }
        let out = self.final_layer.forward(&h, &c)?; // [B,T,9,16,64]
        self.unpatchify(&out, b, tt)
    }
}
