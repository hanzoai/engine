#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! ACE-Step DiT (`ACEStepTransformer2DModel`) decode path: velocity prediction over
//! DCAE music latents. Port of acestep/models/ace_step_transformer.py + attention.py +
//! customer_attention_processor.py (Apache-2.0). adaLN-single, LiteLA linear self-attn,
//! softmax cross-attn to text/speaker context, GLUMBConv FFN.

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Conv1d, GroupNorm, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::layers::{conv1d, conv1d_no_bias, group_norm, linear};

const INNER_DIM: usize = 2560;
const NUM_LAYERS: usize = 24;
const NUM_HEADS: usize = 20;
const HEAD_DIM: usize = 128;
const IN_CHANNELS: usize = 8;
const PATCH_H: usize = 16;
const FF_HIDDEN: usize = 6400; // int(2560 * 2.5)
const PROJ_MID: usize = IN_CHANNELS * 256; // 2048
const TIME_PROJ_DIM: usize = 256;
const ROPE_THETA: f64 = 1_000_000.0;
const RMS_EPS: f64 = 1e-6;
const GN_EPS: f64 = 1e-6;
const LITELA_EPS: f64 = 1e-15;
const TIME_MAX_PERIOD: f64 = 10_000.0;

// diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) freq basis [1, 128].
// Constant across the whole sample loop; build once (from_vec is a htod sync, keep it off the hot path).
fn timestep_freqs(device: &Device) -> Result<Tensor> {
    let half = TIME_PROJ_DIM / 2;
    let exps: Vec<f32> = (0..half)
        .map(|i| (-(TIME_MAX_PERIOD.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    Tensor::from_vec(exps, (1, half), device)
}

// [B] timestep -> [B, 256] using the precomputed freq basis.
fn timestep_embedding(t: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let t = t.to_dtype(DType::F32)?.reshape((t.dim(0)?, 1))?; // [B, 1]
    let args = t.broadcast_mul(freqs)?; // [B, 128]
    Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1) // [B, 256]
}

// Per-generation constants: rope tables (latent + text), timestep freq basis, LiteLA normaliser row.
// Every DiT forward in the sampler shares these, so they are built once, not per forward.
pub struct SampleCtx {
    cos_l: Tensor,
    sin_l: Tensor,
    cos_e: Tensor,
    sin_e: Tensor,
    ts_freqs: Tensor,
    litela_ones: Tensor,
}

// Qwen2 rotary cos/sin caches, each [seq, HEAD_DIM]; emb = cat(freqs, freqs).
fn rope_tables(seq: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / ROPE_THETA.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let inv = Tensor::from_vec(inv_freq, (1, half), device)?;
    let t: Vec<f32> = (0..seq).map(|p| p as f32).collect();
    let t = Tensor::from_vec(t, (seq, 1), device)?;
    let freqs = t.broadcast_mul(&inv)?; // [seq, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [seq, HEAD_DIM]
    Ok((emb.cos()?, emb.sin()?))
}

// Interleaved-pair rotate against half-split cos/sin (the ACE-Step RoPE quirk). x [B,H,S,D].
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let cos = cos.reshape((1, 1, s, d))?;
    let sin = sin.reshape((1, 1, s, d))?;
    let xp = x.reshape((b, h, s, d / 2, 2))?;
    let x_real = xp.narrow(4, 0, 1)?.squeeze(4)?;
    let x_imag = xp.narrow(4, 1, 1)?.squeeze(4)?;
    let neg_imag = x_imag.neg()?;
    let x_rot = Tensor::stack(&[&neg_imag, &x_real], 4)?.reshape((b, h, s, d))?;
    x.broadcast_mul(&cos)?.add(&x_rot.broadcast_mul(&sin)?)
}

// RMSNorm without affine params (norm1/norm2/norm_final), over the last dim.
fn rms_noaffine(x: &Tensor) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(var + RMS_EPS)?.sqrt()?)
}

// x * (1 + scale) + shift, broadcasting scale/shift [B,1,C] over [B,S,C].
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

#[derive(Debug, Clone)]
struct LiteLA {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl LiteLA {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: linear(INNER_DIM, INNER_DIM, vb.pp("to_q"))?,
            to_k: linear(INNER_DIM, INNER_DIM, vb.pp("to_k"))?,
            to_v: linear(INNER_DIM, INNER_DIM, vb.pp("to_v"))?,
            to_out: linear(INNER_DIM, INNER_DIM, vb.pp("to_out").pp("0"))?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &SampleCtx) -> Result<Tensor> {
        let (cos, sin) = (&ctx.cos_l, &ctx.sin_l);
        let (b, s, _) = x.dims3()?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;
        // q,v -> [B,H,D,S]; k -> [B,H,S,D]
        let q = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, NUM_HEADS, HEAD_DIM, s))?;
        let k = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, NUM_HEADS, HEAD_DIM, s))?
            .transpose(2, 3)?
            .contiguous()?;
        let v = v
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, NUM_HEADS, HEAD_DIM, s))?;
        let q = q.permute((0, 1, 3, 2))?.contiguous()?; // [B,H,S,D]
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let q = q.permute((0, 1, 3, 2))?.contiguous()?.relu()?; // [B,H,D,S]
        let k = k.relu()?; // [B,H,S,D]
        let v = v.contiguous()?;
        let v = Tensor::cat(&[&v, &ctx.litela_ones], 2)?; // [B,H,D+1,S]
        let vk = v.matmul(&k)?; // [B,H,D+1,D]
        let out = vk.matmul(&q)?; // [B,H,D+1,S]
        let num = out.narrow(2, 0, HEAD_DIM)?;
        let den = out.narrow(2, HEAD_DIM, 1)?;
        let out = num.broadcast_div(&(den + LITELA_EPS)?)?; // [B,H,D,S]
        let out = out
            .contiguous()?
            .reshape((b, NUM_HEADS * HEAD_DIM, s))?
            .permute((0, 2, 1))?
            .contiguous()?; // [B,S,2560]
        self.to_out.forward(&out)
    }
}

#[derive(Debug, Clone)]
struct CrossAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl CrossAttn {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        // add_q/k/v_proj and to_add_out are dead weights in inference; skip.
        Ok(Self {
            to_q: linear(INNER_DIM, INNER_DIM, vb.pp("to_q"))?,
            to_k: linear(INNER_DIM, INNER_DIM, vb.pp("to_k"))?,
            to_v: linear(INNER_DIM, INNER_DIM, vb.pp("to_v"))?,
            to_out: linear(INNER_DIM, INNER_DIM, vb.pp("to_out").pp("0"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        ehs: &Tensor,
        add_mask: Option<&Tensor>,
        ctx: &SampleCtx,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let se = ehs.dim(1)?;
        let q = self
            .to_q
            .forward(x)?
            .reshape((b, s, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .to_k
            .forward(ehs)?
            .reshape((b, se, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .to_v
            .forward(ehs)?
            .reshape((b, se, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = apply_rope(&q, &ctx.cos_l, &ctx.sin_l)?;
        let k = apply_rope(&k, &ctx.cos_e, &ctx.sin_e)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [B,H,S,Se]
        if let Some(m) = add_mask {
            scores = scores.broadcast_add(m)?; // m [B,1,1,Se]
        }
        let attn = hanzo_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?; // [B,H,S,D]
        let out = out.transpose(1, 2)?.reshape((b, s, NUM_HEADS * HEAD_DIM))?;
        self.to_out.forward(&out)
    }
}

// Depthwise (groups == channels) conv1d, kernel 3 / stride 1 / pad 1, vectorised across channels.
// candle lowers a grouped conv to one conv per group (here 12800 launches); this is the identical
// arithmetic expressed as 3 shifted channel-broadcast multiply-adds -> ~10 ops, not ~38k.
// x [B,C,L], w [1,C,3], b [1,C,1].
fn depthwise_k3_p1(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let l = x.dim(2)?;
    let xp = x.pad_with_zeros(2, 1, 1)?; // [B,C,L+2]
    let o0 = xp.narrow(2, 0, l)?.broadcast_mul(&w.narrow(2, 0, 1)?)?;
    let o1 = xp.narrow(2, 1, l)?.broadcast_mul(&w.narrow(2, 1, 1)?)?;
    let o2 = xp.narrow(2, 2, l)?.broadcast_mul(&w.narrow(2, 2, 1)?)?;
    ((o0 + o1)? + o2)?.broadcast_add(b)
}

#[derive(Debug, Clone)]
struct GlumbFf {
    inverted: Conv1d,
    depth_w: Tensor, // [1, FF_HIDDEN*2, 3]
    depth_b: Tensor, // [1, FF_HIDDEN*2, 1]
    point: Conv1d,
}

impl GlumbFf {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let inverted = conv1d(
            INNER_DIM,
            FF_HIDDEN * 2,
            1,
            Default::default(),
            vb.pp("inverted_conv").pp("conv"),
        )?;
        let dvb = vb.pp("depth_conv").pp("conv");
        let depth_w = dvb
            .get((FF_HIDDEN * 2, 1, 3), "weight")?
            .reshape((1, FF_HIDDEN * 2, 3))?;
        let depth_b = dvb
            .get(FF_HIDDEN * 2, "bias")?
            .reshape((1, FF_HIDDEN * 2, 1))?;
        let point = conv1d_no_bias(
            FF_HIDDEN,
            INNER_DIM,
            1,
            Default::default(),
            vb.pp("point_conv").pp("conv"),
        )?;
        Ok(Self {
            inverted,
            depth_w,
            depth_b,
            point,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // [B,2560,S]
        let x = self.inverted.forward(&x)?.silu()?; // [B,12800,S]
        let x = depthwise_k3_p1(&x, &self.depth_w, &self.depth_b)?;
        let parts = x.chunk(2, 1)?;
        let x = (&parts[0] * parts[1].silu()?)?; // [B,6400,S]
        let x = self.point.forward(&x)?; // [B,2560,S]
        x.transpose(1, 2)?.contiguous()
    }
}

#[derive(Debug, Clone)]
struct Block {
    scale_shift_table: Tensor, // [6, 2560]
    attn: LiteLA,
    cross_attn: CrossAttn,
    ff: GlumbFf,
}

impl Block {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            scale_shift_table: vb.get((6, INNER_DIM), "scale_shift_table")?,
            attn: LiteLA::new(vb.pp("attn"))?,
            cross_attn: CrossAttn::new(vb.pp("cross_attn"))?,
            ff: GlumbFf::new(vb.pp("ff"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        ehs: &Tensor,
        add_mask: Option<&Tensor>,
        ctx: &SampleCtx,
        temb: &Tensor,
    ) -> Result<Tensor> {
        let b = hidden.dim(0)?;
        let sst = self.scale_shift_table.reshape((1, 6, INNER_DIM))?;
        let mods = sst.broadcast_add(&temb.reshape((b, 6, INNER_DIM))?)?; // [B,6,2560]
        let m = mods.chunk(6, 1)?; // each [B,1,2560]
        let (shift_msa, scale_msa, gate_msa) = (&m[0], &m[1], &m[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&m[3], &m[4], &m[5]);

        let norm = modulate(&rms_noaffine(hidden)?, shift_msa, scale_msa)?;
        let attn_out = self.attn.forward(&norm, ctx)?;
        let hidden = (attn_out.broadcast_mul(gate_msa)? + hidden)?;

        let cross = self.cross_attn.forward(&hidden, ehs, add_mask, ctx)?;
        let hidden = (cross + hidden)?;

        let norm = modulate(&rms_noaffine(&hidden)?, shift_mlp, scale_mlp)?;
        let ff = self.ff.forward(&norm)?;
        hidden.add(&ff.broadcast_mul(gate_mlp)?)
    }
}

// PatchEmbed: Conv2d(8->2048, kernel=(16,1), stride=(16,1)) + GroupNorm(32) + Conv2d(1x1)->2560.
// The height-16 kernel with width kernel/stride 1 is exactly a linear map over the flattened
// (channel, height) = 128 column per width position; the 1x1 conv is a channel-wise linear.
#[derive(Debug, Clone)]
struct PatchEmbed {
    patch: Linear,
    norm: GroupNorm,
    proj: Linear,
}

impl PatchEmbed {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let ec = vb.pp("early_conv_layers");
        let w0 = ec
            .pp("0")
            .get((PROJ_MID, IN_CHANNELS, PATCH_H, 1), "weight")?
            .reshape((PROJ_MID, IN_CHANNELS * PATCH_H))?;
        let b0 = ec.pp("0").get(PROJ_MID, "bias")?;
        let patch = Linear::new(w0, Some(b0));
        let norm = group_norm(32, PROJ_MID, GN_EPS, ec.pp("1"))?;
        let w2 = ec
            .pp("2")
            .get((INNER_DIM, PROJ_MID, 1, 1), "weight")?
            .reshape((INNER_DIM, PROJ_MID))?;
        let b2 = ec.pp("2").get(INNER_DIM, "bias")?;
        let proj = Linear::new(w2, Some(b2));
        Ok(Self { patch, norm, proj })
    }

    fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, _, _, t) = latent.dims4()?;
        // [B,8,16,T] -> [B,T,8,16] -> [B,T,128]
        let flat =
            latent
                .permute((0, 3, 1, 2))?
                .contiguous()?
                .reshape((b, t, IN_CHANNELS * PATCH_H))?;
        let x = self.patch.forward(&flat)?; // [B,T,2048]
                                            // GroupNorm couples across width, so normalise in [B,2048,T] layout.
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = self.norm.forward(&x)?;
        let x = x.transpose(1, 2)?.contiguous()?; // [B,T,2048]
        self.proj.forward(&x) // [B,T,2560]
    }
}

const TEXT_DIM: usize = 768;
const SPEAKER_DIM: usize = 512;

#[derive(Debug, Clone)]
pub struct AceStepTransformer {
    proj_in: PatchEmbed,
    ts_linear1: Linear,
    ts_linear2: Linear,
    t_block: Linear,
    genre_embedder: Linear,
    speaker_embedder: Linear,
    blocks: Vec<Block>,
    final_norm_shift_scale: Tensor, // [2, 2560]
    final_linear: Linear,
}

impl AceStepTransformer {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let proj_in = PatchEmbed::new(vb.pp("proj_in"))?;
        let te = vb.pp("timestep_embedder");
        let ts_linear1 = linear(TIME_PROJ_DIM, INNER_DIM, te.pp("linear_1"))?;
        let ts_linear2 = linear(INNER_DIM, INNER_DIM, te.pp("linear_2"))?;
        let t_block = linear(INNER_DIM, 6 * INNER_DIM, vb.pp("t_block").pp("1"))?;
        let genre_embedder = linear(TEXT_DIM, INNER_DIM, vb.pp("genre_embedder"))?;
        let speaker_embedder = linear(SPEAKER_DIM, INNER_DIM, vb.pp("speaker_embedder"))?;
        let tb = vb.pp("transformer_blocks");
        let blocks = (0..NUM_LAYERS)
            .map(|i| Block::new(tb.pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let fl = vb.pp("final_layer");
        let final_norm_shift_scale = fl.get((2, INNER_DIM), "scale_shift_table")?;
        let final_linear = linear(INNER_DIM, PATCH_H * IN_CHANNELS, fl.pp("linear"))?;
        Ok(Self {
            proj_in,
            ts_linear1,
            ts_linear2,
            t_block,
            genre_embedder,
            speaker_embedder,
            blocks,
            final_norm_shift_scale,
            final_linear,
        })
    }

    /// Assemble cross-attn context: [speaker(1), text(T_text)] projected to 2560.
    /// text_hidden (B, T_text, 768) UMT5 last hidden; speaker (B, 512). Lyric stream omitted
    /// (masked-out lyric contributes nothing to softmax cross-attn).
    pub fn encode(&self, text_hidden: &Tensor, speaker: &Tensor) -> Result<Tensor> {
        let spk = self.speaker_embedder.forward(speaker)?.unsqueeze(1)?;
        let text = self.genre_embedder.forward(text_hidden)?;
        Tensor::cat(&[&spk, &text], 1)
    }

    /// latent (B, 8, 16, T), encoder_hidden_states (B, S_enc, 2560), optional encoder mask
    /// (B, S_enc) with 1 = keep, timestep (B,) on the 0..1000 scale -> velocity (B, 8, 16, T).
    /// Per-generation constants (rope tables, timestep freq basis, LiteLA normaliser). `seq` is the
    /// latent time dim, `se` the encoder-context length; both fixed across the whole sampler loop.
    pub fn sample_ctx(&self, seq: usize, se: usize, device: &Device) -> Result<SampleCtx> {
        let (cos_l, sin_l) = rope_tables(seq, device)?;
        let (cos_e, sin_e) = rope_tables(se, device)?;
        Ok(SampleCtx {
            cos_l,
            sin_l,
            cos_e,
            sin_e,
            ts_freqs: timestep_freqs(device)?,
            litela_ones: Tensor::ones((1, NUM_HEADS, 1, seq), DType::F32, device)?,
        })
    }

    pub fn decode(
        &self,
        latent: &Tensor,
        ehs: &Tensor,
        encoder_mask: Option<&Tensor>,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let ctx = self.sample_ctx(latent.dim(3)?, ehs.dim(1)?, latent.device())?;
        self.decode_with_ctx(latent, ehs, encoder_mask, timestep, &ctx)
    }

    /// Hot-path decode reusing a shared `SampleCtx` so no per-forward htod syncs are issued.
    pub fn decode_with_ctx(
        &self,
        latent: &Tensor,
        ehs: &Tensor,
        encoder_mask: Option<&Tensor>,
        timestep: &Tensor,
        ctx: &SampleCtx,
    ) -> Result<Tensor> {
        let embedded_ts = {
            let e = timestep_embedding(timestep, &ctx.ts_freqs)?;
            self.ts_linear2
                .forward(&self.ts_linear1.forward(&e)?.silu()?)?
        };
        let temb = self.t_block.forward(&embedded_ts.silu()?)?; // [B, 15360]

        let hidden = self.proj_in.forward(latent)?; // [B,T,2560]
        let seq = hidden.dim(1)?;
        let se = ehs.dim(1)?;

        // Additive cross-attn mask [B,1,1,Se] from a 0/1 keep mask.
        // Additive mask: 0 where keep(1), large-negative where drop(0). (mk-1)*1e30 avoids 0*-inf NaN.
        let add_mask = match encoder_mask {
            None => None,
            Some(mk) => {
                let b = mk.dim(0)?;
                let neg = ((mk.to_dtype(DType::F32)? - 1.0)? * 1e30)?;
                Some(neg.reshape((b, 1, 1, se))?)
            }
        };

        let mut h = hidden;
        for block in &self.blocks {
            h = block.forward(&h, ehs, add_mask.as_ref(), ctx, &temb)?;
        }

        // final layer: adaLN from embedded_timestep, then linear + unpatchify.
        let b = h.dim(0)?;
        let sst = self.final_norm_shift_scale.reshape((1, 2, INNER_DIM))?;
        let ss = sst.broadcast_add(&embedded_ts.reshape((b, 1, INNER_DIM))?)?; // [B,2,2560]
        let parts = ss.chunk(2, 1)?;
        let (shift, scale) = (&parts[0], &parts[1]);
        let x = modulate(&rms_noaffine(&h)?, shift, scale)?;
        let x = self.final_linear.forward(&x)?; // [B,T,128]
        unpatchfy(&x, seq)
    }
}

// [B,S,128] -> [B,8,16,S]  (patch_size 16x1, out_channels 8).
fn unpatchfy(x: &Tensor, seq: usize) -> Result<Tensor> {
    let b = x.dim(0)?;
    let x = x.reshape((b, 1, seq, PATCH_H, 1, IN_CHANNELS))?;
    x.permute((0, 5, 1, 3, 2, 4))?
        .contiguous()?
        .reshape((b, IN_CHANNELS, PATCH_H, seq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn read_f32_le(p: &str) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..n {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (d / (na.sqrt() * nb.sqrt())) as f32
    }

    // One decode() forward with the real DiT weights vs the torch reference (cosine > 0.99).
    #[test]
    fn ace_dit_decode_matches_reference() {
        let st = std::env::var("ACE_DIT_ST")
            .unwrap_or_else(|_| "/home/z/work/hanzo/ace-fixtures/dit.safetensors".to_string());
        let fix = std::env::var("ACE_FIX_DIR")
            .unwrap_or_else(|_| "/home/z/work/hanzo/ace-fixtures".to_string());
        if !PathBuf::from(&st).is_file() || !PathBuf::from(format!("{fix}/dit_out.f32")).is_file() {
            eprintln!("ACE-Step DiT fixtures absent; skipping decode validation");
            return;
        }
        let device = Device::Cpu;
        let vb = from_mmaped_safetensors(
            vec![PathBuf::from(&st)],
            Vec::new(),
            Some(DType::F32),
            &device,
            vec![None],
            true,
            None,
            |n: String| {
                n.starts_with("proj_in.")
                    || n.starts_with("timestep_embedder.")
                    || n.starts_with("t_block.")
                    || n.starts_with("genre_embedder.")
                    || n.starts_with("speaker_embedder.")
                    || n.starts_with("final_layer.")
                    || (n.starts_with("transformer_blocks.")
                        && !n.contains(".add_")
                        && !n.contains(".to_add_out"))
            },
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        let model = AceStepTransformer::new(vb).unwrap();

        let latent = Tensor::from_vec(
            read_f32_le(&format!("{fix}/dit_latent.f32")),
            (1, 8, 16, 32),
            &device,
        )
        .unwrap();
        let ehs = Tensor::from_vec(
            read_f32_le(&format!("{fix}/dit_ehs.f32")),
            (1, 10, INNER_DIM),
            &device,
        )
        .unwrap();
        let ts = Tensor::from_vec(vec![500.0f32], (1,), &device).unwrap();

        let out = model.decode(&latent, &ehs, None, &ts).unwrap();
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let want = read_f32_le(&format!("{fix}/dit_out.f32"));
        let c = cosine(&got, &want);
        println!(
            "ace dit decode cosine = {c:.6} (n_got={}, n_want={})",
            got.len(),
            want.len()
        );
        assert!(c > 0.99, "dit decode cosine {c} below 0.99");
    }
}
