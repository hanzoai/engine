//! Wan2.2 text-to-video generation entry point: params in, RGB frames out. The forward
//! (umt5 text-encode -> flow-match denoise -> Wan2.2 VAE decode) lives in `super::pipeline`;
//! this module is the thin request/frame boundary the async `/v1/videos` job drives.

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

/// Run Wan2.2 text-to-video for `params`, returning decoded RGB frames. The model (umt5 + DiT +
/// Wan2.2 VAE) is loaded once from `WAN_MODEL_DIR` and cached; see `super::pipeline`.
pub fn wan_t2v_generate(params: &WanT2vParams) -> Result<WanT2vFrames> {
    validate(params)?;
    let pipe = super::pipeline::global()?;
    let cfg = super::pipeline::WanVideoConfig::from_params(params);
    pipe.t2v(&cfg, &params.prompt)
}

/// Wan2.2 image/continuation-to-video: condition the clip on `image` (e.g. a prior clip's last
/// frame, for a seamless film continuation). Same model + config path as `wan_t2v_generate`.
pub fn wan_i2v_generate(params: &WanT2vParams, image: &DynamicImage) -> Result<WanT2vFrames> {
    validate(params)?;
    let pipe = super::pipeline::global()?;
    let cfg = super::pipeline::WanVideoConfig::from_params(params);
    pipe.i2v_image(&cfg, &params.prompt, image)
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
