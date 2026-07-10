//! Wan2.2 text-to-video generation entry point.
//!
//! The backbone pieces already live here: the 3D causal VAE (`vae::AutoencoderKLWan`) and the T2V
//! DiT (`t2v_dit::Wan2TransformerDiT`, a flow-matching velocity predictor). This module is the seam that
//! chains text-encode -> scheduler denoise loop -> VAE decode into RGB frames. The forward is the
//! one piece still pending model wiring (weights + umt5 text encoder + scheduler); everything that
//! consumes its output (the async `/v1/videos` job) is real and drives this function.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use anyhow::{bail, Result};
use image::{DynamicImage, RgbImage};

/// Parameters for a single text-to-video generation.
#[derive(Debug, Clone)]
pub struct WanT2vParams {
    pub prompt: String,
    pub num_frames: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
}

/// Rendered video: RGB frames plus the frame rate they were produced at. The handler layer muxes
/// these into a container; a generator never touches ffmpeg (frames out, container is not our job).
pub struct WanT2vFrames {
    pub frames: Vec<DynamicImage>,
    pub fps: f64,
}

/// Default frame rate for generated video when the model does not dictate one.
pub const DEFAULT_FPS: f64 = 16.0;

/// Run Wan2.2 text-to-video for `params`, returning decoded RGB frames.
///
/// Pending model wiring: this needs the Wan2.2-TI2V-5B weights loaded into `t2v_dit::Wan2TransformerDiT` + the
/// umt5 text encoder for the prompt embedding + a flow-match scheduler denoise loop feeding
/// `vae::AutoencoderKLWan::decode`. Until that is loaded, refuse rather than emit garbage.
pub fn wan_t2v_generate(params: &WanT2vParams) -> Result<WanT2vFrames> {
    validate(params)?;
    bail!(
        "WAN t2v forward pending model wiring: prompt {:?} ({}x{}, {} frames, {} steps). \
         The Wan2.2 DiT + VAE backbone exists (t2v_dit::Wan2, vae::AutoencoderKLWan) but the \
         weights/text-encoder/scheduler are not loaded yet.",
        params.prompt,
        params.width,
        params.height,
        params.num_frames,
        params.steps,
    )
}

fn validate(params: &WanT2vParams) -> Result<()> {
    if params.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    if params.num_frames == 0 {
        bail!("num_frames must be > 0");
    }
    if params.width == 0 || params.height == 0 {
        bail!("width and height must be > 0");
    }
    if params.steps == 0 {
        bail!("steps must be > 0");
    }
    Ok(())
}

/// Convert a decoded RGB video tensor `[3, F, H, W]` in `[0, 1]` into per-frame images. This is the
/// bridge from the VAE decode output to the container muxer, kept here so the forward path lands as
/// one function once weights are wired.
#[allow(dead_code)]
pub(crate) fn frames_from_pixels(
    pixels: &[f32],
    num_frames: usize,
    height: usize,
    width: usize,
) -> Result<Vec<DynamicImage>> {
    let per_frame = height * width * 3;
    if pixels.len() != num_frames * per_frame {
        bail!(
            "pixel buffer {} != {} frames * {}x{}x3",
            pixels.len(),
            num_frames,
            height,
            width
        );
    }
    let mut frames = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let mut img = RgbImage::new(width as u32, height as u32);
        let base = f * per_frame;
        for y in 0..height {
            for x in 0..width {
                let i = base + (y * width + x) * 3;
                let px = |c: usize| (pixels[i + c].clamp(0.0, 1.0) * 255.0).round() as u8;
                img.put_pixel(x as u32, y as u32, image::Rgb([px(0), px(1), px(2)]));
            }
        }
        frames.push(DynamicImage::ImageRgb8(img));
    }
    Ok(frames)
}
