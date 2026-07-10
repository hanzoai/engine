#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Wan2.2 TI2V-5B video pipeline: umt5 text encode -> flow-match denoise (DiT velocity) -> Wan2.2
//! VAE decode. Composes the three ported backbones (`t5::T5EncoderModel` umt5, `t2v_dit`, `vae22`)
//! and the shared `FlowMatchScheduler`. Supports text-to-video and image/continuation-to-video:
//! the film primitive is `i2v` conditioning frame(s) at the head of the clip, so the tail of clip A
//! seeds clip B (Wan2.2 expand_timesteps: clean latent pinned at frame 0, per-token timestep 0).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{ensure, Context, Result};
use hanzo_ml::{DType, Device, Tensor};
use hanzo_quant::ShardedVarBuilder;
use image::DynamicImage;
use tokenizers::Tokenizer;

use super::echomimic::FlowMatchScheduler;
use super::t2v::{frames_from_pixels, WanT2vFrames, WanT2vParams};
use super::t2v_dit::{Wan2Config, Wan2TransformerDiT};
use super::vae22::{Wan22Vae, Wan22VaeConfig};
use crate::diffusion_models::t5::{Config as T5Config, T5EncoderModel};
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

const VAE_SPATIAL: usize = 16;
const VAE_TEMPORAL: usize = 4;
const PATCH_SPATIAL: usize = 2; // DiT patch_size [1,2,2]
const MAX_TEXT_TOKENS: usize = 512;
const DEFAULT_SHIFT: f64 = 5.0;
const DEFAULT_GUIDANCE: f64 = 5.0;
pub const DEFAULT_FPS: f64 = 24.0;

// Wan's default negative prompt (over-saturation, blur, watermarks, malformed limbs, ...).
const DEFAULT_NEGATIVE_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

#[derive(Debug, Clone)]
pub struct WanVideoConfig {
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub shift: f64,
    pub guidance: f64,
    pub fps: f64,
    pub negative_prompt: String,
}

impl WanVideoConfig {
    pub fn from_params(p: &WanT2vParams) -> Self {
        Self {
            num_frames: p.num_frames,
            height: p.height,
            width: p.width,
            steps: p.steps,
            shift: DEFAULT_SHIFT,
            guidance: DEFAULT_GUIDANCE,
            fps: DEFAULT_FPS,
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_string(),
        }
    }

    // Latent grid (T, H, W) and the DiT token count for expand_timesteps conditioning.
    fn latent_grid(&self) -> (usize, usize, usize) {
        let t = 1 + (self.num_frames - 1) / VAE_TEMPORAL;
        (t, self.height / VAE_SPATIAL, self.width / VAE_SPATIAL)
    }
}

// A head-of-clip conditioning: `k` clean latent frames pinned at frames 0..k, the film seam.
struct Condition {
    latent: Tensor, // [1, z, k, lat_h, lat_w] normalized clean latent
    k: usize,
    token_mask: Tensor, // [1, T*hp*wp]: 0 for conditioning-frame tokens, 1 elsewhere
}

impl Condition {
    fn new(latent: Tensor, grid: (usize, usize, usize), device: &Device) -> Result<Self> {
        let k = latent.dim(2)?;
        let (tl, lat_h, lat_w) = grid;
        let (hp, wp) = (lat_h / PATCH_SPATIAL, lat_w / PATCH_SPATIAL);
        let mut m = vec![1f32; tl * hp * wp];
        for f in 0..k.min(tl) {
            m[f * hp * wp..(f + 1) * hp * wp].fill(0.0);
        }
        let n = m.len();
        Ok(Self {
            latent,
            k,
            token_mask: Tensor::from_vec(m, (1, n), device)?,
        })
    }

    // Replace the leading k latent frames with the clean conditioning latent.
    fn pin(&self, latent: &Tensor) -> Result<Tensor> {
        let t = latent.dim(2)?;
        let tail = latent.narrow(2, self.k, t - self.k)?;
        Ok(Tensor::cat(&[&self.latent, &tail], 2)?)
    }

    // Per-token timestep [1, T*hp*wp]: 0 for conditioning tokens (mask 0), t elsewhere.
    fn timestep(&self, t: f64) -> Result<Tensor> {
        Ok((&self.token_mask * t)?)
    }
}

pub struct WanVideoPipeline {
    umt5: Mutex<T5EncoderModel>,
    dit: Wan2TransformerDiT,
    vae: Wan22Vae,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl WanVideoPipeline {
    pub fn from_dir(dir: &Path, device: Device, dtype: DType) -> Result<Self> {
        let umt5_vb = load_vb(&dir.join("text_encoder"), dtype, &device)?;
        let umt5 = T5EncoderModel::load(umt5_vb, &T5Config::umt5_xxl(), &device, false)?;

        let dit_vb = load_vb(&dir.join("transformer"), dtype, &device)?;
        let dit = Wan2TransformerDiT::new(Wan2Config::ti2v_5b(), dit_vb, device.clone())?;

        let vae_vb = load_vb(&dir.join("vae"), DType::F32, &device)?;
        let vae = Wan22Vae::new(&Wan22VaeConfig::ti2v_5b(), vae_vb, &device)?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        Ok(Self {
            umt5: Mutex::new(umt5),
            dit,
            vae,
            tokenizer,
            device,
            dtype,
        })
    }

    // Prompt -> umt5 last-hidden-state [1, L, text_dim] in the DiT dtype.
    fn encode_text(&self, prompt: &str) -> Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(MAX_TEXT_TOKENS);
        let len = ids.len();
        let ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let emb = self.umt5.lock().unwrap().forward(&ids)?;
        Ok(emb.to_dtype(self.dtype)?)
    }

    fn init_latent(&self, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, h, w) = grid;
        Ok(
            Tensor::randn(0f64, 1.0, (1, self.vae.z_dim(), t, h, w), &self.device)?
                .to_dtype(self.dtype)?,
        )
    }

    // Flow-match denoise loop with classifier-free guidance. `cond` pins the head frames (I2V).
    fn denoise(
        &self,
        cfg: &WanVideoConfig,
        mut latent: Tensor,
        text: &Tensor,
        neg: &Tensor,
        cond: Option<&Condition>,
    ) -> Result<Tensor> {
        let sched = FlowMatchScheduler::new(cfg.steps, cfg.shift);
        let scalar_ts: Vec<Tensor> = sched
            .timesteps()
            .iter()
            .map(|&t| Tensor::from_vec(vec![t as f32], 1, &self.device))
            .collect::<hanzo_ml::Result<_>>()?;

        for (i, &t) in sched.timesteps().iter().enumerate() {
            let (model_in, ts) = match cond {
                None => (latent.clone(), scalar_ts[i].clone()),
                Some(c) => (c.pin(&latent)?, c.timestep(t)?),
            };
            let v_cond = self.dit.forward(&model_in, &ts, text)?;
            let v = if (cfg.guidance - 1.0).abs() > f64::EPSILON {
                let v_uncond = self.dit.forward(&model_in, &ts, neg)?;
                (&v_uncond + ((&v_cond - &v_uncond)? * cfg.guidance)?)?
            } else {
                v_cond
            };
            latent = sched.step(&v, i, &latent)?;
        }
        if let Some(c) = cond {
            latent = c.pin(&latent)?;
        }
        Ok(latent)
    }

    /// Text-to-video: prompt -> a clip of `cfg.num_frames` RGB frames.
    pub fn t2v(&self, cfg: &WanVideoConfig, prompt: &str) -> Result<WanT2vFrames> {
        let text = self.encode_text(prompt)?;
        let neg = self.encode_text(&cfg.negative_prompt)?;
        let latent = self.init_latent(cfg.latent_grid())?;
        let latent = self.denoise(cfg, latent, &text, &neg, None)?;
        self.decode_frames(&latent, cfg.fps)
    }

    /// Image/continuation-to-video: `cond_frames` RGB `[1,3,Fc,H,W]` in `[-1,1]` seed the clip head.
    /// The film seam: pass the tail frame(s) of a prior clip to continue motion without a scene cut.
    pub fn i2v(
        &self,
        cfg: &WanVideoConfig,
        prompt: &str,
        cond_frames: &Tensor,
    ) -> Result<WanT2vFrames> {
        let text = self.encode_text(prompt)?;
        let neg = self.encode_text(&cfg.negative_prompt)?;
        let grid = cfg.latent_grid();
        let cond_latent = self.vae.encode(cond_frames)?.to_dtype(self.dtype)?;
        let cond = Condition::new(cond_latent, grid, &self.device)?;
        let latent = cond.pin(&self.init_latent(grid)?)?;
        let latent = self.denoise(cfg, latent, &text, &neg, Some(&cond))?;
        self.decode_frames(&latent, cfg.fps)
    }

    /// I2V from a single conditioning image (resized to the clip resolution). This is the HTTP entry
    /// for continuation: pass the last frame of a prior clip as `img`.
    pub fn i2v_image(
        &self,
        cfg: &WanVideoConfig,
        prompt: &str,
        img: &DynamicImage,
    ) -> Result<WanT2vFrames> {
        let cond = self.frame_to_tensor(img, cfg.height, cfg.width)?;
        self.i2v(cfg, prompt, &cond)
    }

    // RGB image -> f32 tensor [1,3,1,H,W] in [-1,1] (the VAE's input range), resized to (h, w).
    fn frame_to_tensor(&self, img: &DynamicImage, h: usize, w: usize) -> Result<Tensor> {
        let rgb = img
            .resize_exact(w as u32, h as u32, image::imageops::FilterType::Lanczos3)
            .to_rgb8();
        let raw: Vec<f32> = rgb
            .into_raw()
            .into_iter()
            .map(|b| b as f32 / 127.5 - 1.0)
            .collect();
        Ok(Tensor::from_vec(raw, (h, w, 3), &self.device)?
            .permute((2, 0, 1))?
            .contiguous()?
            .reshape((1, 3, 1, h, w))?)
    }

    // Normalized latent -> RGB frames. VAE runs in f32.
    fn decode_frames(&self, latent: &Tensor, fps: f64) -> Result<WanT2vFrames> {
        let video = self.vae.decode(&latent.to_dtype(DType::F32)?)?;
        let (_, _, f, h, w) = video.dims5()?;
        let pixels = ((video.squeeze(0)?.affine(0.5, 0.5)?).clamp(0f32, 1f32)?)
            .permute((1, 2, 3, 0))? // [3,F,H,W] -> [F,H,W,3]
            .contiguous()?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let frames = frames_from_pixels(&pixels, f, h, w)?;
        Ok(WanT2vFrames { frames, fps })
    }
}

fn load_vb(dir: &Path, dtype: DType, device: &Device) -> Result<ShardedVarBuilder> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read model dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    ensure!(!paths.is_empty(), "no safetensors in {}", dir.display());
    let n = paths.len();
    Ok(from_mmaped_safetensors(
        paths,
        Vec::new(),
        Some(dtype),
        device,
        vec![None; n],
        true,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?)
}

/// Process-global pipeline, lazily loaded from `WAN_MODEL_DIR` (a Wan2.2-TI2V-5B-Diffusers dir).
pub fn global() -> Result<&'static WanVideoPipeline> {
    static PIPE: OnceLock<WanVideoPipeline> = OnceLock::new();
    if let Some(p) = PIPE.get() {
        return Ok(p);
    }
    let dir = std::env::var("WAN_MODEL_DIR")
        .context("set WAN_MODEL_DIR to a Wan2.2-TI2V-5B-Diffusers directory")?;
    let device = Device::cuda_if_available(0)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let pipe = WanVideoPipeline::from_dir(Path::new(&dir), device, dtype)?;
    Ok(PIPE.get_or_init(|| pipe))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The film-continuation core: `pin` puts the clean conditioning latent at the head frames, and
    // `timestep` marks exactly those tokens with t=0 (expand_timesteps) while the rest carry t.
    #[test]
    fn condition_pins_head_and_zeros_timestep() -> Result<()> {
        let dev = Device::Cpu;
        let z = 2;
        // k=1 conditioning frame; latent grid T=3, 4x4 (hp=wp=2 -> 4 tokens/frame).
        let cond = Condition::new(
            Tensor::ones((1, z, 1, 4, 4), DType::F32, &dev)?,
            (3, 4, 4),
            &dev,
        )?;
        let latent = Tensor::zeros((1, z, 3, 4, 4), DType::F32, &dev)?;
        let pinned = cond.pin(&latent)?;
        assert_eq!(pinned.dims(), &[1, z, 3, 4, 4]);
        let f0 = pinned.narrow(2, 0, 1)?.mean_all()?.to_scalar::<f32>()?;
        let f1 = pinned.narrow(2, 1, 2)?.mean_all()?.to_scalar::<f32>()?;
        assert!(
            (f0 - 1.0).abs() < 1e-6,
            "head frame must be the clean cond latent"
        );
        assert!(f1.abs() < 1e-6, "tail frames must stay the noise latent");

        // N = 3*4 = 12 tokens; frame-0 tokens are 0, rest are t.
        let ts = cond.timestep(500.0)?;
        assert_eq!(ts.dims(), &[1, 12]);
        let v = ts.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v[0..4].iter().all(|&x| x == 0.0),
            "conditioning tokens must be t=0"
        );
        assert!(
            v[4..12].iter().all(|&x| x == 500.0),
            "generated tokens must carry t"
        );
        Ok(())
    }

    fn save_frames(frames: &[image::DynamicImage], dir: &str, tag: &str) -> Result<()> {
        for (i, f) in frames.iter().enumerate() {
            f.save(format!("{dir}/{tag}_{i:04}.png"))
                .map_err(|e| anyhow::anyhow!("save {tag} frame {i}: {e}"))?;
        }
        Ok(())
    }

    // Real generation on GPU: a T2V clip A, then clip B conditioned on A's last frame (the film
    // continuation seam). Env-gated: WAN_MODEL_DIR + WAN_GEN_OUT (dir for the PNG frames), plus
    // optional WAN_GEN_{FRAMES,W,H,STEPS} and WAN_PROMPT_{A,B}. Skips when unset.
    #[test]
    fn wan_generate_and_continue() -> Result<()> {
        let (Ok(_), Ok(out)) = (std::env::var("WAN_MODEL_DIR"), std::env::var("WAN_GEN_OUT"))
        else {
            eprintln!("skip wan_generate_and_continue: set WAN_MODEL_DIR + WAN_GEN_OUT");
            return Ok(());
        };
        let ev = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let cfg = WanVideoConfig {
            num_frames: ev("WAN_GEN_FRAMES", 25),
            height: ev("WAN_GEN_H", 512),
            width: ev("WAN_GEN_W", 512),
            steps: ev("WAN_GEN_STEPS", 20),
            shift: DEFAULT_SHIFT,
            guidance: DEFAULT_GUIDANCE,
            fps: DEFAULT_FPS,
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_string(),
        };
        let pipe = global()?;
        let prompt_a = std::env::var("WAN_PROMPT_A").unwrap_or_else(|_| {
            "a red fox trotting through fresh snow, cinematic, sunlight".into()
        });
        let a = pipe.t2v(&cfg, &prompt_a)?;
        save_frames(&a.frames, &out, "clipA")?;
        let last = a.frames.last().unwrap().clone();
        let prompt_b = std::env::var("WAN_PROMPT_B").unwrap_or_else(|_| {
            "the same red fox continues trotting then leaps over a log, cinematic".into()
        });
        let b = pipe.i2v_image(&cfg, &prompt_b, &last)?;
        save_frames(&b.frames, &out, "clipB")?;
        eprintln!(
            "wrote clipA={} clipB={} frames to {out}",
            a.frames.len(),
            b.frames.len()
        );
        Ok(())
    }
}
