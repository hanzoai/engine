#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, RgbImage};

use crate::diffusion_models::animation::{
    AnimationOutput, AnimationRequest, Animator, DriveEncoder, DrivingAudio, FacialAnimator,
    FullFrame, Generator, VisualKind, WholeFrame,
};

use super::echomimic_dit::EchoMimicDiT;
use super::vae::AutoencoderKLWan;

// EchoMimic regenerates the whole frame each call, so a stale locate box never matters.
const REDETECT_EVERY: usize = 1;

// EchoMimic latent layout: 16 noise + 4 mask + 16 masked-reference = 36 DiT input channels.
const Z: usize = 16;
const MASK_CH: usize = 4;
const NUM_TRAIN_TIMESTEPS: f64 = 1000.0;
const VAE_SPATIAL: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct EchoMimicOptions {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub shift: f64,
    pub text_cfg: f64,
    pub audio_cfg: f64,
}

impl Default for EchoMimicOptions {
    fn default() -> Self {
        Self {
            height: 768,
            width: 768,
            steps: 25,
            shift: 5.0,
            text_cfg: 4.0,
            audio_cfg: 2.9,
        }
    }
}

// Flow-matching Euler schedule with the Wan resolution shift. sigma runs 1 -> 0; the model
// predicts velocity v and each step integrates x += (sigma_next - sigma) * v.
pub struct FlowMatchScheduler {
    sigmas: Vec<f64>,
    timesteps: Vec<f64>,
}

impl FlowMatchScheduler {
    pub fn new(steps: usize, shift: f64) -> Self {
        let mut sigmas = Vec::with_capacity(steps + 1);
        for i in 0..steps {
            let s = 1.0 - i as f64 / steps as f64;
            sigmas.push(shift * s / (1.0 + (shift - 1.0) * s));
        }
        sigmas.push(0.0);
        let timesteps = sigmas[..steps].iter().map(|s| s * NUM_TRAIN_TIMESTEPS).collect();
        Self { sigmas, timesteps }
    }

    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    pub fn step(&self, model_out: &Tensor, i: usize, x: &Tensor) -> Result<Tensor> {
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        x + (model_out * dt)?
    }
}

pub struct EchoMimicGenerator {
    vae: AutoencoderKLWan,
    dit: EchoMimicDiT,
    opts: EchoMimicOptions,
    device: Device,
    dtype: DType,
}

impl EchoMimicGenerator {
    pub fn new(
        vae: AutoencoderKLWan,
        dit: EchoMimicDiT,
        opts: EchoMimicOptions,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            vae,
            dit,
            opts,
            device,
            dtype,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    // Flow-matching denoise of the 16ch latent with triple-batch CFG (uncond / drop-audio /
    // cond). text+clip context are zeros here (umT5/CLIP encoders not ported); identity rides
    // the VAE-concat masked_ref. Returns the denoised 16ch latent [1,16,F_lat,Hp,Wp].
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        masked_ref: &Tensor,
        mask4: &Tensor,
        audio: &Tensor,
        ip_mask: &Tensor,
        f_lat: usize,
        hp: usize,
        wp: usize,
    ) -> Result<Tensor> {
        let sched = FlowMatchScheduler::new(self.opts.steps, self.opts.shift);
        let clip_dim = self.dit.config().clip_dim;
        let mut latent = Tensor::randn(0f64, 1.0, (1, Z, f_lat, hp, wp), &self.device)?.to_dtype(self.dtype)?;

        // CFG context: text/clip zeros (no encoders); audio is [neg, neg, cond].
        let text3 = Tensor::zeros((3, 1, self.dit.config().text_dim), self.dtype, &self.device)?;
        let clip3 = Tensor::zeros((3, 257, clip_dim), self.dtype, &self.device)?;
        let neg_audio = audio.zeros_like()?;
        let audio3 = Tensor::cat(&[&neg_audio, &neg_audio, audio], 0)?;
        let ip3 = Tensor::cat(&[ip_mask, ip_mask, ip_mask], 0)?;

        for (i, &ts) in sched.timesteps().iter().enumerate() {
            let x = Tensor::cat(&[&latent, mask4, masked_ref], 1)?;
            let x3 = Tensor::cat(&[&x, &x, &x], 0)?;
            let t = Tensor::from_vec(vec![ts as f32; 3], 3, &self.device)?;
            let v = self
                .dit
                .forward(&x3, &t, &text3, &clip3, &audio3, &ip3, self.opts.audio_cfg)?;
            let v = self.cfg(&v)?;
            latent = sched.step(&v, i, &latent)?;
        }
        Ok(latent)
    }

    // v = uncond + text_cfg*(drop_audio - uncond) + audio_cfg*(cond - drop_audio).
    fn cfg(&self, v: &Tensor) -> Result<Tensor> {
        let uncond = v.narrow(0, 0, 1)?;
        let drop_audio = v.narrow(0, 1, 1)?;
        let cond = v.narrow(0, 2, 1)?;
        let text = ((&drop_audio - &uncond)? * self.opts.text_cfg)?;
        let aud = ((&cond - &drop_audio)? * self.opts.audio_cfg)?;
        (uncond + text)? + aud
    }

    // Build conditioning from a portrait and run the full pipeline -> RGB frames.
    pub fn generate_video(&self, portrait: &DynamicImage, audio: &Tensor, f_lat: usize) -> Result<Vec<DynamicImage>> {
        let (h, w) = (self.opts.height, self.opts.width);
        let (hp, wp) = (h / VAE_SPATIAL, w / VAE_SPATIAL);
        let f_pix = 4 * f_lat - 3;
        let portrait_t = portrait_to_tensor(portrait, h, w, &self.device)?.to_dtype(self.dtype)?;
        // masked reference video: portrait at frame 0, zeros elsewhere.
        let masked_video = if f_pix == 1 {
            portrait_t.clone()
        } else {
            let rest = Tensor::zeros((1, 3, f_pix - 1, h, w), self.dtype, &self.device)?;
            Tensor::cat(&[&portrait_t, &rest], 2)?
        };
        let masked_ref = self.vae.encode(&masked_video)?;
        let mask4 = Tensor::zeros((1, MASK_CH, f_lat, hp, wp), self.dtype, &self.device)?;
        // ip_mask gates the DiT's patched spatial tokens (patch stride 2 halves hp/wp).
        let ip_mask = Tensor::ones((1, (hp / 2) * (wp / 2)), self.dtype, &self.device)?;

        let latent = self.denoise(&masked_ref, &mask4, audio, &ip_mask, f_lat, hp, wp)?;
        let rgb = self.vae.decode(&latent)?; // [1,3,f_pix,h,w] in [-1,1]
        let n = rgb.dim(2)?;
        (0..n).map(|f| video_frame_to_image(&rgb.narrow(2, f, 1)?)).collect()
    }
}

impl Generator for EchoMimicGenerator {
    // Single-frame path for the per-frame Animator: portrait + this frame's audio -> 1 frame.
    fn generate(&self, visual: &DynamicImage, audio: &Tensor) -> Result<DynamicImage> {
        let audio = audio.to_dtype(self.dtype)?.contiguous()?;
        let frames = self.generate_video(visual, &audio, 1)?;
        frames
            .into_iter()
            .next()
            .ok_or_else(|| hanzo_ml::Error::msg("echomimic produced no frame"))
    }

    fn accepts(&self) -> VisualKind {
        VisualKind::Portrait
    }
}

// wav2vec2 is not ported; the pipeline takes precomputed audio_embeds [1,2F,768]. This encoder
// exists so EchoMimic slots into the Animator; raw-PCM encoding requires the wav2vec2 port.
pub struct Wav2Vec2Encoder;

impl DriveEncoder for Wav2Vec2Encoder {
    fn encode(&self, _audio: &DrivingAudio, _fps: f64) -> Result<Tensor> {
        hanzo_ml::bail!("EchoMimic wav2vec2 encoder not ported; supply precomputed audio_embeds [1,2F,768]")
    }
}

/// Portrait-driven full-frame I2V animator: `WholeFrame` locate + wav2vec2 `DriveEncoder` +
/// `EchoMimicGenerator` + `FullFrame` compose, composed via the generic `Animator`. Mirrors
/// MuseTalkAnimator; the backbone swap is the one `Generator` type, zero pipeline change.
pub struct EchoMimicAnimator(Animator<WholeFrame, Wav2Vec2Encoder, EchoMimicGenerator, FullFrame>);

impl EchoMimicAnimator {
    pub fn new(generator: EchoMimicGenerator) -> Self {
        let device = generator.device().clone();
        Self(Animator::new(
            WholeFrame,
            Wav2Vec2Encoder,
            generator,
            FullFrame,
            device,
            REDETECT_EVERY,
        ))
    }
}

impl FacialAnimator for EchoMimicAnimator {
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

// Portrait -> [1,3,1,H,W] in [-1,1] (the Wan VAE input range).
fn portrait_to_tensor(img: &DynamicImage, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let rgb = img.resize_exact(w as u32, h as u32, FilterType::Triangle).to_rgb8();
    let raw = rgb.into_raw();
    let plane = h * w;
    let mut data = vec![0f32; 3 * plane];
    for i in 0..plane {
        data[i] = raw[i * 3] as f32 / 127.5 - 1.0;
        data[plane + i] = raw[i * 3 + 1] as f32 / 127.5 - 1.0;
        data[2 * plane + i] = raw[i * 3 + 2] as f32 / 127.5 - 1.0;
    }
    Tensor::from_vec(data, (1, 3, 1, h, w), device)
}

// [1,3,1,H,W] in [-1,1] -> DynamicImage.
fn video_frame_to_image(t: &Tensor) -> Result<DynamicImage> {
    let t = t.squeeze(2)?; // [1,3,H,W]
    let (_, c, h, w) = t.dims4()?;
    debug_assert_eq!(c, 3);
    let v = ((t.to_dtype(DType::F32)? + 1.0)? * 127.5)?
        .clamp(0f32, 255f32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let plane = h * w;
    let mut buf = vec![0u8; 3 * plane];
    for i in 0..plane {
        buf[i * 3] = v[i] as u8;
        buf[i * 3 + 1] = v[plane + i] as u8;
        buf[i * 3 + 2] = v[2 * plane + i] as u8;
    }
    RgbImage::from_raw(w as u32, h as u32, buf)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| hanzo_ml::Error::msg("frame buffer size mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion_models::wan::echomimic_dit::EchoMimicConfig;
    use crate::diffusion_models::wan::vae::WanVaeConfig;
    use hanzo_ml::Shape;
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

    struct RandnBackend;
    impl SimpleBackend for RandnBackend {
        fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("gamma") || (name.ends_with("weight") && s.rank() == 1) {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, 0.02, s, dev)?.to_dtype(dtype)
            }
        }
        fn get_unchecked(&self, _n: &str, _d: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("needs shape")
        }
        fn contains_tensor(&self, _n: &str) -> bool {
            true
        }
    }

    fn vb(dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone())
    }

    fn tiny_dit_cfg() -> EchoMimicConfig {
        EchoMimicConfig {
            dim: 24,
            ffn_dim: 48,
            num_layers: 1,
            num_heads: 2,
            in_dim: 36,
            out_dim: 16,
            freq_dim: 16,
            text_dim: 32,
            clip_dim: 20,
            audio_dim: 18,
            eps: 1e-6,
        }
    }

    fn tiny_vae_cfg() -> WanVaeConfig {
        WanVaeConfig {
            base_dim: 8,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 1,
            temperal_downsample: vec![false, true, true],
        }
    }

    #[test]
    fn scheduler_sigmas_descend_to_zero() {
        let s = FlowMatchScheduler::new(25, 5.0);
        assert_eq!(s.sigmas.len(), 26);
        assert!((s.sigmas[0] - 1.0).abs() < 1e-9);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        for w in s.sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly decrease");
        }
    }

    #[test]
    fn generator_single_frame_shape() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let vae = AutoencoderKLWan::new(&tiny_vae_cfg(), vb(&dev), &dev)?;
        let dit = EchoMimicDiT::new(tiny_dit_cfg(), vb(&dev), dev.clone(), dtype)?;
        let opts = EchoMimicOptions {
            height: 16,
            width: 16,
            steps: 2,
            shift: 5.0,
            text_cfg: 4.0,
            audio_cfg: 2.9,
        };
        let gen = EchoMimicGenerator::new(vae, dit, opts, dev.clone(), dtype);

        let portrait = DynamicImage::new_rgb8(16, 16);
        // F_lat=1 -> F_pix=1 -> audio length 2.
        let audio = Tensor::randn(0f64, 1.0, (1, 2, 18), &dev)?.to_dtype(dtype)?;
        let frame = gen.generate(&portrait, &audio)?;
        assert_eq!((frame.width(), frame.height()), (16, 16));
        Ok(())
    }

    #[test]
    fn generator_multi_frame_video_shape() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let vae = AutoencoderKLWan::new(&tiny_vae_cfg(), vb(&dev), &dev)?;
        let dit = EchoMimicDiT::new(tiny_dit_cfg(), vb(&dev), dev.clone(), dtype)?;
        let opts = EchoMimicOptions {
            height: 16,
            width: 16,
            steps: 2,
            shift: 5.0,
            text_cfg: 4.0,
            audio_cfg: 2.9,
        };
        let gen = EchoMimicGenerator::new(vae, dit, opts, dev.clone(), dtype);

        let portrait = DynamicImage::new_rgb8(16, 16);
        // F_lat=2 -> F_pix=5 -> audio length 10.
        let audio = Tensor::randn(0f64, 1.0, (1, 10, 18), &dev)?.to_dtype(dtype)?;
        let frames = gen.generate_video(&portrait, &audio, 2)?;
        assert_eq!(frames.len(), 5, "F_lat=2 decodes to 5 pixel frames");
        assert_eq!((frames[0].width(), frames[0].height()), (16, 16));
        Ok(())
    }
}
