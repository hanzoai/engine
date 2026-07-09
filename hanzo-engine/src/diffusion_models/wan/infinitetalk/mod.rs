#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! InfiniteTalk (Wan2.1-14B + MultiTalk audio + infinite streaming).
//!
//! The DiT *is* Wan2.1-14B and the VAE is `AutoencoderKLWan` -- both shared, owned by the `wan`
//! module (the EchoMimic port builds the Wan DiT at 1.3B; here it is instantiated at 14B). This
//! module owns only the InfiniteTalk-specific value:
//! - [`streaming`]: the infinite-length sampler (81->21 chunks, 9->3 re-noised context, 72 new).
//! - [`audio`]: the MultiTalk `AudioProjModel` + injectable per-block audio cross-attn + L-RoPE.
//!
//! Integration: the Wan DiT @ 14B implements [`ChunkDenoiser`] (calling [`audio::SingleStreamAudioAttn`]
//! inside each block), the shared `WanVae` decodes the streamed latents, and a wav2vec2 encoder feeds
//! 12-layer-stacked features into [`AudioProjModel`]. See `wan/infinitetalk/README` notes in the port.

pub mod audio;
pub mod streaming;

pub use audio::{
    apply_rope_1d, audio_stream_label, build_audio_windows, video_token_labels, AudioProjModel,
    SingleStreamAudioAttn, AUDIO_LABEL_P1, AUDIO_LABEL_P2, CONTEXT_TOKENS, WAV2VEC_DIM,
    WAV2VEC_LAYERS,
};
pub use streaming::{
    frames_to_latents, latents_to_frames, plan_for_frames, stream, ChunkCond, ChunkDenoiser,
    ChunkPlan, CondTemplate, StreamConfig, CHUNK_FRAMES, CHUNK_LATENTS, CONTEXT_LATENTS,
    NEW_FRAMES, NEW_LATENTS,
};

use hanzo_ml::{Result, Tensor};

use crate::diffusion_models::animation::VisualKind;

/// Wan2.1-14B DiT dims (the InfiniteTalk runtime config, `wan_multitalk_14B`). Carried for the shared
/// Wan DiT instantiation at integration; `audio_seq_len_vf` sizes [`AudioProjModel::proj1_vf`].
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct InfiniteTalkConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    pub text_len: usize,
    pub eps: f64,
    pub audio_seq_len_vf: usize,
}

impl InfiniteTalkConfig {
    pub fn wan_14b() -> Self {
        Self {
            dim: 5120,
            ffn_dim: 13824,
            num_layers: 40,
            num_heads: 40,
            head_dim: 128,
            in_dim: 36,
            out_dim: 16,
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            audio_seq_len_vf: WAV2VEC_LAYERS,
        }
    }
}

/// DriveEncoder stage: 12-layer-stacked wav2vec2 features `[B,T,12,768]` (latent-frame-aligned; the
/// caller groups 25fps wav2vec frames to latent frames) -> the 5-frame `[B,T,5,12,768]` windows
/// [`AudioProjModel`] consumes. The wav2vec2 forward is precomputed offline and never ported (spec
/// sec.4), so this stage is just the windowing.
pub struct Wav2Vec2StackEncoder;

impl Wav2Vec2StackEncoder {
    pub fn windows(stacked: &Tensor) -> Result<Tensor> {
        build_audio_windows(stacked)
    }
}

/// The InfiniteTalk `Generator`: composes the shared Wan DiT @ 14B (any [`ChunkDenoiser`]), the
/// MultiTalk [`AudioProjModel`], and the streaming sampler. Clip-level (whole-video) generation --
/// the streaming context is cross-frame state, so generation is not per-frame.
pub struct InfiniteTalkGenerator<D: ChunkDenoiser> {
    denoiser: D,
    audio_proj: AudioProjModel,
    cfg: StreamConfig,
}

impl<D: ChunkDenoiser> InfiniteTalkGenerator<D> {
    pub fn new(denoiser: D, audio_proj: AudioProjModel, cfg: StreamConfig) -> Self {
        Self {
            denoiser,
            audio_proj,
            cfg,
        }
    }

    pub fn accepts(&self) -> VisualKind {
        VisualKind::Portrait
    }

    pub fn config(&self) -> &StreamConfig {
        &self.cfg
    }

    /// `ref_latent` [B,C,1,H,W] (portrait VAE latent), `windows` [B,T,5,12,768] -> latent video
    /// [B,C,L,H,W]. The caller VAE-decodes the result to RGB frames (shared `WanVae`).
    pub fn generate_latents(
        &self,
        ref_latent: &Tensor,
        windows: &Tensor,
        cond: &CondTemplate,
    ) -> Result<Tensor> {
        let tokens = self.audio_proj.forward(windows)?;
        stream(&self.denoiser, ref_latent, &tokens, cond, &self.cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers;
    use hanzo_ml::{DType, Device, Shape};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::{Init, Linear};
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

    struct RandnBackend;
    impl SimpleBackend for RandnBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("weight") && s.rank() == 1 {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, 0.02, s, dev)?.to_dtype(dtype)
            }
        }
        fn get_unchecked(&self, _n: &str, _d: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("needs explicit shape")
        }
        fn contains_tensor(&self, _n: &str) -> bool {
            true
        }
    }

    fn vb(dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone())
    }

    fn finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    }

    // Stand-in for the shared Wan DiT @ 14B: a depth-2 channel MLP over the latent. Exercises real
    // matmuls + 5D shape handling so the streaming/audio plumbing is validated end-to-end.
    struct TinyDit {
        l1: Linear,
        l2: Linear,
    }
    impl TinyDit {
        fn new(c: usize, hidden: usize, vb: ShardedVarBuilder) -> Result<Self> {
            Ok(Self {
                l1: layers::linear(c, hidden, vb.pp("l1"))?,
                l2: layers::linear(hidden, c, vb.pp("l2"))?,
            })
        }
    }
    impl ChunkDenoiser for TinyDit {
        fn denoise_chunk(
            &self,
            l: &Tensor,
            _a: &Tensor,
            _c: &ChunkCond,
            _t: f64,
        ) -> Result<Tensor> {
            let (b, c, t, h, w) = l.dims5()?;
            let x = l.permute((0, 2, 3, 4, 1))?.reshape((b * t * h * w, c))?;
            let x = x.apply(&self.l1)?.gelu()?.apply(&self.l2)?;
            x.reshape((b, t, h, w, c))?
                .permute((0, 4, 1, 2, 3))?
                .contiguous()
        }
    }

    #[test]
    fn config_14b_dims() {
        let c = InfiniteTalkConfig::wan_14b();
        assert_eq!(c.dim / c.num_heads, c.head_dim);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.in_dim, 36);
    }

    #[test]
    fn generator_produces_latent_video() -> Result<()> {
        let dev = Device::Cpu;
        let audio_proj = AudioProjModel::new(WAV2VEC_LAYERS, vb(&dev))?;
        let dit = TinyDit::new(4, 16, vb(&dev))?;
        let cfg = StreamConfig {
            steps: 2,
            max_frames: 153,
            ..Default::default()
        };
        let gen = InfiniteTalkGenerator::new(dit, audio_proj, cfg);
        let plan = plan_for_frames(153);

        let ref_latent = Tensor::randn(0f32, 1.0, (1usize, 4, 1, 8, 8), &dev)?;
        let feats = Tensor::randn(
            0f32,
            1.0,
            (1usize, plan.total_latents, WAV2VEC_LAYERS, WAV2VEC_DIM),
            &dev,
        )?;
        let windows = Wav2Vec2StackEncoder::windows(&feats)?;
        let out = gen.generate_latents(&ref_latent, &windows, &CondTemplate::none())?;

        assert_eq!(out.dims(), &[1, 4, plan.total_latents, 8, 8]);
        assert!(finite(&out));
        assert_eq!(gen.accepts(), VisualKind::Portrait);
        Ok(())
    }
}
