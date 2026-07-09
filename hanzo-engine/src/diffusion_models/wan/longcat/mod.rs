#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! LongCat-Video-Avatar-1.5: a homegrown 15.85B flow-matching DiT for portrait-driven full-frame
//! audio talking-head video. Wired into the decomplected animator as a `Generator` over
//! `VisualKind::Portrait`. The shared Wan VAE (`wan/vae.rs`, EchoMimic-owned) plugs in via the
//! `WanVae` trait below; the whisper-large-v3 front-end is external (precomputed `[T,5,1280]`).

pub mod audio;
pub mod blocks;
pub mod common;
pub mod dit;
pub mod embed;

pub use dit::{FlowMatchEuler, KvCache, LongCatAvatarDiT, LongCatConfig};

use hanzo_ml::{DType, Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, RgbImage};

use crate::diffusion_models::animation::{
    AnimationOutput, AnimationRequest, Animator, DriveEncoder, DrivingAudio, FacialAnimator,
    FullFrame, Generator, VisualKind, WholeFrame,
};
use dit::LongCatAvatarDiT as Dit;

const DEFAULT_STEPS: usize = 50;
const DMD_STEPS: usize = 8;
const DEFAULT_HEIGHT: usize = 480;
const DEFAULT_WIDTH: usize = 832;
const N_REF_FRAMES: usize = 1; // single reference portrait -> one condition latent frame

/// The shared Wan 3D causal VAE (`AutoencoderKLWan`), owned by the EchoMimic port in `wan/vae.rs`.
/// LongCat consumes it behind this trait so the two ports integrate without re-porting the VAE; at
/// merge, the concrete VAE gets a thin `impl WanVae`.
pub trait WanVae: Send + Sync {
    /// `[B, 3, T, H, W]` pixels -> `[B, z, Tl, Hl, Wl]` latent (z=16, 4x8x8 compression).
    fn encode(&self, frames: &Tensor) -> Result<Tensor>;
    /// `[B, z, Tl, Hl, Wl]` latent -> `[B, 3, T, H, W]` pixels.
    fn decode(&self, latent: &Tensor) -> Result<Tensor>;
}

/// Generation knobs. `distill` selects the DMD few-step `FlowMatchEuler` schedule (the speed lever).
#[derive(Clone, Copy, Debug)]
pub struct LongCatOptions {
    pub steps: usize,
    pub distill: bool,
    pub height: usize,
    pub width: usize,
}

impl Default for LongCatOptions {
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEPS,
            distill: false,
            height: DEFAULT_HEIGHT,
            width: DEFAULT_WIDTH,
        }
    }
}

impl LongCatOptions {
    /// 8-step DMD2-distilled preset (`--use_distill`).
    pub fn dmd() -> Self {
        Self {
            steps: DMD_STEPS,
            distill: true,
            ..Self::default()
        }
    }
}

/// The LongCat avatar `Generator`: VAE-encode the reference portrait, run the (DMD few-step)
/// flow-match denoise with the block-causal KV-cache, VAE-decode. `generate_latents` is the PRIMARY
/// clip-level path (diffusion backbones are sequence-level, not per-frame); the per-frame
/// `Generator::generate` is a thin adapter for the existing `Animator` trait surface.
pub struct LongCatGenerator {
    dit: Dit,
    vae: Box<dyn WanVae>,
    opts: LongCatOptions,
    device: Device,
    dtype: DType,
}

impl LongCatGenerator {
    pub fn new(dit: Dit, vae: Box<dyn WanVae>, opts: LongCatOptions) -> Self {
        let device = dit.device().clone();
        let dtype = dit.dtype();
        Self {
            dit,
            vae,
            opts,
            device,
            dtype,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    // Flow-match denoise of the noisy latent frames in `latent_full = [cond; noisy]`. Step 0 primes
    // the KV-cache over the full sequence; later steps run noisy tokens only against cached cond K/V.
    // The per-step primitive (`dit.forward*`: latent + timestep + cond -> velocity) matches the shape
    // EchoMimic/InfiniteTalk converge on as a shared `ChunkDenoiser` seam at integration.
    fn denoise_chunk(
        &self,
        latent_full: &Tensor,
        text: &Tensor,
        audio_emb: &Tensor,
        n_cond_frames: usize,
    ) -> Result<Tensor> {
        let sched = if self.opts.distill {
            FlowMatchEuler::distilled(self.opts.steps)
        } else {
            FlowMatchEuler::new(self.opts.steps)
        };
        let mut cache = KvCache::new(self.dit.depth());
        let b = latent_full.dim(0)?;
        let n_noisy = latent_full.dim(2)? - n_cond_frames;
        let cond_part = latent_full.narrow(2, 0, n_cond_frames)?.contiguous()?;
        let mut noisy = latent_full
            .narrow(2, n_cond_frames, n_noisy)?
            .contiguous()?;
        for i in 0..sched.num_steps() {
            let ts = (Tensor::ones(b, DType::F32, &self.device)? * sched.sigma(i))?;
            let v_noisy = if i == 0 {
                let full = Tensor::cat(&[&cond_part, &noisy], 2)?;
                self.dit
                    .forward_prime(&full, &ts, text, audio_emb, n_cond_frames, &mut cache)?
                    .narrow(2, n_cond_frames, n_noisy)?
            } else {
                self.dit
                    .forward_cached(&noisy, &ts, text, audio_emb, n_cond_frames, &cache)?
            };
            noisy = sched.step(&noisy, &v_noisy.contiguous()?, i)?;
        }
        Tensor::cat(&[&cond_part, &noisy], 2)
    }

    /// PRIMARY clip-level entry: reference portrait `[1,3,H,W]` + UMT5 text `[1,Lt,caption]` + full
    /// grouped audio `[1,T,5,1280]` -> generated latents `[1, z, T_noisy, Hl, Wl]` (chunked
    /// flow-match denoise, VAE-decode-free). Cross-chunk autoregressive continuation is a follow-up.
    pub fn generate_latents(
        &self,
        portrait: &Tensor,
        text: &Tensor,
        audio_emb: &Tensor,
    ) -> Result<Tensor> {
        let dtype = self.dtype;
        let portrait = portrait.to_dtype(dtype)?;
        let text = text.to_dtype(dtype)?;
        let audio_emb = audio_emb.to_dtype(dtype)?;
        let ref_latent = self.vae.encode(&portrait.unsqueeze(2)?)?;
        let (b, c, _tc, hl, wl) = ref_latent.dims5()?;
        let t_noisy = (audio_emb.dim(1)? / self.dit.config().vae_scale).max(1);
        let noise =
            Tensor::randn(0f32, 1f32, (b, c, t_noisy, hl, wl), &self.device)?.to_dtype(dtype)?;
        let latent_full = Tensor::cat(&[&ref_latent, &noise], 2)?;
        let clean = self.denoise_chunk(&latent_full, &text, &audio_emb, N_REF_FRAMES)?;
        clean.narrow(2, N_REF_FRAMES, t_noisy)
    }

    /// Pixel path: `generate_latents` then VAE-decode -> frames `[1, 3, Tv, H, W]`.
    pub fn generate_video(
        &self,
        portrait: &Tensor,
        text: &Tensor,
        audio_emb: &Tensor,
    ) -> Result<Tensor> {
        self.vae
            .decode(&self.generate_latents(portrait, text, audio_emb)?)
    }
}

impl Generator for LongCatGenerator {
    fn generate(&self, visual: &DynamicImage, audio: &Tensor) -> Result<DynamicImage> {
        let img = image_to_tensor(
            visual,
            self.opts.height,
            self.opts.width,
            self.dtype,
            &self.device,
        )?;
        let (_, blk, ch) = audio.dims3()?;
        let audio4 = audio.reshape((1, 1, blk, ch))?;
        let text = Tensor::zeros(
            (1, 1, self.dit.config().caption_channels),
            self.dtype,
            &self.device,
        )?;
        let frames = self.generate_video(&img, &text, &audio4)?;
        tensor_to_image(&frames.narrow(2, 0, 1)?.squeeze(2)?)
    }

    fn accepts(&self) -> VisualKind {
        VisualKind::Portrait
    }
}

/// whisper-large-v3 grouped features `[T, 5, 1280]` as the `DriveEncoder`. The mel front-end, full
/// encoder, 5-layer grouping, and MDX-Net vocal pre-separation run externally (out of scope here);
/// this carries the precomputed result and serves it per request.
pub struct WhisperLargeEncoder {
    features: Tensor,
}

impl WhisperLargeEncoder {
    pub fn new(features: Tensor) -> Self {
        Self { features }
    }
}

impl DriveEncoder for WhisperLargeEncoder {
    fn encode(&self, _audio: &DrivingAudio, _fps: f64) -> Result<Tensor> {
        Ok(self.features.clone())
    }
}

/// Portrait-driven full-frame avatar animator: WholeFrame locate + whisper-large drive + LongCat
/// generate + FullFrame compose. Mirrors `MuseTalkAnimator`; a new backbone is just a `Generator`.
pub struct LongCatAvatarAnimator(
    Animator<WholeFrame, WhisperLargeEncoder, LongCatGenerator, FullFrame>,
);

impl LongCatAvatarAnimator {
    pub fn new(gen: LongCatGenerator, whisper: WhisperLargeEncoder) -> Self {
        let device = gen.device().clone();
        Self(Animator::new(
            WholeFrame, whisper, gen, FullFrame, device, 1,
        ))
    }
}

impl FacialAnimator for LongCatAvatarAnimator {
    fn animate(&mut self, req: &AnimationRequest) -> Result<AnimationOutput> {
        self.0.animate(req)
    }

    fn device(&self) -> &Device {
        self.0.device()
    }

    fn accepts(&self) -> VisualKind {
        self.0.accepts()
    }
}

/// `DynamicImage` -> `[1, 3, h, w]` RGB in [0,1], cast to `dtype`.
fn image_to_tensor(
    img: &DynamicImage,
    h: usize,
    w: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let rgb = img
        .resize_exact(w as u32, h as u32, FilterType::Triangle)
        .to_rgb8();
    let raw = rgb.into_raw();
    let plane = h * w;
    let mut data = vec![0f32; 3 * plane];
    for i in 0..plane {
        data[i] = raw[i * 3] as f32 / 255.0;
        data[plane + i] = raw[i * 3 + 1] as f32 / 255.0;
        data[2 * plane + i] = raw[i * 3 + 2] as f32 / 255.0;
    }
    Tensor::from_vec(data, (1, 3, h, w), dev)?.to_dtype(dtype)
}

/// `[1, 3, h, w]` (RGB, [0,1]) -> `DynamicImage`.
fn tensor_to_image(t: &Tensor) -> Result<DynamicImage> {
    let (_, c, h, w) = t.dims4()?;
    debug_assert_eq!(c, 3);
    let v = t
        .to_dtype(DType::F32)?
        .clamp(0f32, 1f32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let plane = h * w;
    let mut buf = vec![0u8; 3 * plane];
    for i in 0..plane {
        buf[i * 3] = (v[i] * 255.0) as u8;
        buf[i * 3 + 1] = (v[plane + i] * 255.0) as u8;
        buf[i * 3 + 2] = (v[2 * plane + i] * 255.0) as u8;
    }
    let img = RgbImage::from_raw(w as u32, h as u32, buf)
        .ok_or_else(|| hanzo_ml::Error::msg("frame buffer size mismatch"))?;
    Ok(DynamicImage::ImageRgb8(img))
}

#[cfg(test)]
mod tests {
    use super::dit::{merge_lora, merged_linear};
    use super::*;
    use crate::diffusion_models::wan::longcat::audio::AudioProjModel;
    use hanzo_ml::{Shape, D};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::{Init, Linear};
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
    use image::GenericImageView;

    const WEIGHT_STD: f64 = 0.02;

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
                Tensor::randn(0f64, WEIGHT_STD, s, dev)?.to_dtype(dtype)
            }
        }

        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("RandnBackend requires an explicit shape")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn random_vb(dtype: DType, dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), dtype, dev.clone())
    }

    fn assert_finite(t: &Tensor) -> Result<()> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(!v.is_empty());
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "found {bad} non-finite values");
        let max = v.iter().cloned().fold(f32::MIN, f32::max);
        let min = v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max != min, "output is constant (degenerate)");
        Ok(())
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len());
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max))
    }

    // Shape-faithful Wan VAE stub: strided 8x spatial downsample / nearest upsample, 3<->16 channels.
    struct StubWanVae {
        space: usize,
        latent_c: usize,
    }

    impl WanVae for StubWanVae {
        fn encode(&self, frames: &Tensor) -> Result<Tensor> {
            let (b, c, t, h, w) = frames.dims5()?;
            let p = self.space;
            let ds = frames
                .reshape(&[b, c, t, h / p, p, w / p, p])?
                .narrow(4, 0, 1)?
                .narrow(6, 0, 1)?
                .contiguous()?
                .reshape((b, c, t, h / p, w / p))?;
            let reps = self.latent_c.div_ceil(c);
            let tiled = vec![&ds; reps];
            Tensor::cat(&tiled, 1)?.narrow(1, 0, self.latent_c)
        }

        fn decode(&self, latent: &Tensor) -> Result<Tensor> {
            let (b, _lc, t, hl, wl) = latent.dims5()?;
            let p = self.space;
            latent
                .narrow(1, 0, 3)?
                .reshape(&[b, 3, t, hl, 1, wl, 1])?
                .broadcast_as(&[b, 3, t, hl, p, wl, p])?
                .contiguous()?
                .reshape((b, 3, t, hl * p, wl * p))
        }
    }

    fn tiny_dit(dev: &Device, dtype: DType) -> Result<LongCatAvatarDiT> {
        LongCatAvatarDiT::new(LongCatConfig::tiny(), random_vb(dtype, dev), dev.clone())
    }

    #[test]
    fn dit_dense_forward_shapes_and_finite() -> Result<()> {
        let dev = Device::Cpu;
        let dt = DType::F32;
        let cfg = LongCatConfig::tiny();
        let dit = tiny_dit(&dev, dt)?;
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 3, 8, 8), &dev)?;
        let ts = (Tensor::ones(1, dt, &dev)? * 0.5)?;
        let text = Tensor::randn(0f32, 1f32, (1, 5, cfg.caption_channels), &dev)?;
        let audio = Tensor::randn(0f32, 1f32, (1, 4, cfg.audio_block, cfg.audio_channel), &dev)?;
        let v = dit.forward(&latent, &ts, &text, &audio, 0)?;
        assert_eq!(v.dims(), &[1, cfg.out_channels, 3, 8, 8]);
        assert_finite(&v)?;
        Ok(())
    }

    #[test]
    fn kv_cache_matches_dense() -> Result<()> {
        let dev = Device::Cpu;
        let dt = DType::F32;
        let cfg = LongCatConfig::tiny();
        let dit = tiny_dit(&dev, dt)?;
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 3, 8, 8), &dev)?;
        let ts = (Tensor::ones(1, dt, &dev)? * 0.5)?;
        let text = Tensor::randn(0f32, 1f32, (1, 5, cfg.caption_channels), &dev)?;
        let audio = Tensor::randn(0f32, 1f32, (1, 4, cfg.audio_block, cfg.audio_channel), &dev)?;
        let n_cond = 1;

        let dense = dit.forward(&latent, &ts, &text, &audio, n_cond)?;
        let mut cache = KvCache::new(dit.depth());
        let prime = dit.forward_prime(&latent, &ts, &text, &audio, n_cond, &mut cache)?;
        let noisy = latent.narrow(2, n_cond, 2)?.contiguous()?;
        let cached = dit.forward_cached(&noisy, &ts, &text, &audio, n_cond, &cache)?;

        assert!(
            max_abs_diff(&prime, &dense)? < 1e-4,
            "prime must equal dense"
        );
        let dense_noisy = dense.narrow(2, n_cond, 2)?;
        assert!(
            max_abs_diff(&cached, &dense_noisy)? < 1e-3,
            "cached noisy velocity must match the dense block-causal forward"
        );
        Ok(())
    }

    #[test]
    fn flow_match_schedules_monotonic() -> Result<()> {
        let dev = Device::Cpu;
        for sched in [FlowMatchEuler::new(50), FlowMatchEuler::distilled(8)] {
            let n = sched.num_steps();
            assert!((sched.sigma(0) - 1.0).abs() < 1e-6);
            assert!(sched.sigma(n).abs() < 1e-6);
            for i in 0..n {
                assert!(sched.sigma(i) > sched.sigma(i + 1), "sigmas must decrease");
            }
        }
        assert_eq!(FlowMatchEuler::distilled(8).num_steps(), 8);
        let x = Tensor::randn(0f32, 1f32, (2, 3), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (2, 3), &dev)?;
        assert_eq!(FlowMatchEuler::new(50).step(&x, &v, 0)?.dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn lora_merge_math() -> Result<()> {
        let dev = Device::Cpu;
        let (out, inn, r) = (6, 4, 2);
        let w = Tensor::randn(0f32, 1f32, (out, inn), &dev)?;
        let down = Tensor::randn(0f32, 1f32, (r, inn), &dev)?;
        let up = Tensor::randn(0f32, 1f32, (out, r), &dev)?;
        let alpha = 0.5;
        let merged = merge_lora(&w, &down, &up, alpha)?;
        let expect = (&w + (up.matmul(&down)? * alpha)?)?;
        assert!(max_abs_diff(&merged, &expect)? < 1e-5);
        let lin = Linear::new(w.clone(), None);
        let ml = merged_linear(&lin, &down, &up, alpha)?;
        assert!(max_abs_diff(ml.weight(), &merged)? < 1e-6);
        Ok(())
    }

    #[test]
    fn audio_proj_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let dt = DType::F32;
        let cfg = LongCatConfig::tiny();
        let ap = AudioProjModel::new(&cfg, random_vb(dt, &dev))?;
        let audio = Tensor::randn(0f32, 1f32, (1, 6, cfg.audio_block, cfg.audio_channel), &dev)?;
        let out = ap.forward(&audio)?;
        assert_eq!(
            out.dims(),
            &[1, 6, cfg.audio_context_tokens, cfg.audio_output_dim]
        );
        let vf = ap.forward_first_group(&audio)?;
        assert_eq!(
            vf.dims(),
            &[1, 6, cfg.audio_context_tokens, cfg.audio_output_dim]
        );
        assert_finite(&out)?;
        assert_finite(&vf)?;
        Ok(())
    }

    #[test]
    fn generator_video_and_per_frame() -> Result<()> {
        let dev = Device::Cpu;
        let dt = DType::F32;
        let cfg = LongCatConfig::tiny();
        let dit = tiny_dit(&dev, dt)?;
        let vae = Box::new(StubWanVae {
            space: 8,
            latent_c: cfg.in_channels,
        });
        let opts = LongCatOptions {
            steps: 3,
            distill: true,
            height: 32,
            width: 32,
        };
        let gen = LongCatGenerator::new(dit, vae, opts);

        let portrait = Tensor::rand(0f32, 1f32, (1, 3, 32, 32), &dev)?;
        let text = Tensor::randn(0f32, 1f32, (1, 5, cfg.caption_channels), &dev)?;
        let audio = Tensor::randn(0f32, 1f32, (1, 4, cfg.audio_block, cfg.audio_channel), &dev)?;

        // Primary clip-level path: 4 video frames / vae_scale 4 -> 1 latent frame at 32/8 = 4 px.
        let latents = gen.generate_latents(&portrait, &text, &audio)?;
        assert_eq!(latents.dims(), &[1, cfg.in_channels, 1, 4, 4]);
        assert_finite(&latents)?;

        let frames = gen.generate_video(&portrait, &text, &audio)?;
        assert_eq!(frames.dim(0)?, 1);
        assert_eq!(frames.dim(1)?, 3);
        assert_eq!(frames.dim(D::Minus1)?, 32);
        assert_finite(&frames)?;

        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, image::Rgb([120, 90, 60])));
        let audio1 = Tensor::randn(0f32, 1f32, (1, cfg.audio_block, cfg.audio_channel), &dev)?;
        let out = Generator::generate(&gen, &img, &audio1)?;
        assert_eq!(out.dimensions(), (32, 32));
        assert_eq!(gen.accepts(), VisualKind::Portrait);
        Ok(())
    }
}
