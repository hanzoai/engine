#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Avatar-1.5 audio conditioning: `AudioProjModel` maps grouped Whisper-large-v3 features to per-frame
//! context tokens, and `AudioCrossAttn` injects them into each DiT block (video tokens query, audio
//! K/V), gated by its own adaLN. Single-person path; L-RoPE multi-person binding is a follow-up.

use hanzo_ml::{Result, Tensor, D};
use hanzo_nn::{LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::diffusion_models::wan::longcat::blocks::CrossAttention;
use crate::diffusion_models::wan::longcat::common::{ModulationOut, LN_EPS};
use crate::diffusion_models::wan::longcat::dit::LongCatConfig;
use crate::layers;

const AUDIO_ADALN_CHUNKS: usize = 3; // shift, scale, gate

/// Projects grouped Whisper features `[B, T, block, channel]` to context tokens `[B, T, ctx, out]`.
/// The standard branch windows `audio_window` frames; `forward_first_group` uses the wider `_vf`
/// window for the first latent group (carrying `vae_scale-1` extra audio sub-frames).
pub struct AudioProjModel {
    proj1: Linear,
    proj1_vf: Linear,
    proj2: Linear,
    proj3: Linear,
    norm: LayerNorm,
    window: usize,
    vf_window: usize,
    context_tokens: usize,
    output_dim: usize,
}

impl AudioProjModel {
    pub fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let group = cfg.audio_block * cfg.audio_channel;
        let window = cfg.audio_window;
        let vf_window = cfg.vae_scale * 2;
        let inter = cfg.audio_intermediate;
        let out = cfg.audio_output_dim;
        let proj1 = layers::linear(window * group, inter, vb.pp("proj1"))?;
        let proj1_vf = layers::linear(vf_window * group, inter, vb.pp("proj1_vf"))?;
        let proj2 = layers::linear(inter, inter, vb.pp("proj2"))?;
        let proj3 = layers::linear(inter, cfg.audio_context_tokens * out, vb.pp("proj3"))?;
        let norm = LayerNorm::new(
            vb.get(out, "norm.weight")?,
            vb.get(out, "norm.bias")?,
            LN_EPS,
        );
        Ok(Self {
            proj1,
            proj1_vf,
            proj2,
            proj3,
            norm,
            window,
            vf_window,
            context_tokens: cfg.audio_context_tokens,
            output_dim: out,
        })
    }

    // Replicate-pad time then gather `win` neighbor frames into the channel dim: [B,T,F] -> [B,T,win*F].
    fn window_time(flat: &Tensor, win: usize) -> Result<Tensor> {
        let (b, t, f) = flat.dims3()?;
        let half = win / 2;
        let front = flat.narrow(1, 0, 1)?.broadcast_as((b, half, f))?;
        let back = flat.narrow(1, t - 1, 1)?.broadcast_as((b, half, f))?;
        let padded = Tensor::cat(&[&front, flat, &back], 1)?;
        let mut wins = Vec::with_capacity(win);
        for k in 0..win {
            wins.push(padded.narrow(1, k, t)?);
        }
        Tensor::cat(&wins, D::Minus1)
    }

    fn run_proj(&self, windowed: &Tensor, vf: bool) -> Result<Tensor> {
        let (b, t, _) = windowed.dims3()?;
        let proj1 = if vf { &self.proj1_vf } else { &self.proj1 };
        let h = windowed.apply(proj1)?.relu()?;
        let h = h.apply(&self.proj2)?.relu()?;
        let h = h
            .apply(&self.proj3)?
            .reshape((b, t, self.context_tokens, self.output_dim))?;
        h.apply(&self.norm)
    }

    pub fn forward(&self, audio_emb: &Tensor) -> Result<Tensor> {
        let (b, t, blk, ch) = audio_emb.dims4()?;
        let flat = audio_emb.reshape((b, t, blk * ch))?;
        let windowed = Self::window_time(&flat, self.window)?;
        self.run_proj(&windowed, false)
    }

    pub fn forward_first_group(&self, audio_emb: &Tensor) -> Result<Tensor> {
        let (b, t, blk, ch) = audio_emb.dims4()?;
        let flat = audio_emb.reshape((b, t, blk * ch))?;
        let windowed = Self::window_time(&flat, self.vf_window)?;
        self.run_proj(&windowed, true)
    }
}

/// Per-block audio cross-attn: video tokens query projected audio tokens, output gated by adaLN.
pub(crate) struct AudioCrossAttn {
    attn: CrossAttention,
    pre_video_norm: LayerNorm,
    pre_audio_norm: LayerNorm,
    ada_ln: Linear,
}

impl AudioCrossAttn {
    pub(crate) fn new(cfg: &LongCatConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let attn = CrossAttention::new(cfg, cfg.audio_output_dim, vb.pp("audio_cross_attn"))?;
        let pre_video_norm = LayerNorm::new_no_bias(
            vb.get(cfg.hidden_size, "pre_video_crs_attn_norm.weight")?,
            LN_EPS,
        );
        let pre_audio_norm = LayerNorm::new_no_bias(
            vb.get(cfg.audio_output_dim, "pre_audio_crs_attn_norm.weight")?,
            LN_EPS,
        );
        let ada_ln = layers::linear(
            cfg.adaln_tembed_dim,
            AUDIO_ADALN_CHUNKS * cfg.hidden_size,
            vb.pp("audio_adaLN_modulation").pp("1"),
        )?;
        Ok(Self {
            attn,
            pre_video_norm,
            pre_audio_norm,
            ada_ln,
        })
    }

    // x: [B,N,hidden]; audio_kv: [B,La,audio_out]; cond: [B,tembed] -> gated delta [B,N,hidden].
    pub(crate) fn forward(&self, x: &Tensor, audio_kv: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let ys = cond
            .silu()?
            .apply(&self.ada_ln)?
            .unsqueeze(1)?
            .chunk(AUDIO_ADALN_CHUNKS, D::Minus1)?;
        let m = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let xq = m.scale_shift(&x.apply(&self.pre_video_norm)?)?;
        let akv = audio_kv.apply(&self.pre_audio_norm)?;
        m.gate(&self.attn.forward(&xq, &akv)?)
    }
}
