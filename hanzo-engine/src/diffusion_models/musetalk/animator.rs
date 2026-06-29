#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};

use crate::diffusion_models::animation::{
    AnimationOutput, AnimationRequest, Animator, DriveEncoder, DrivingAudio, FaceLocator,
    FacialAnimator, Generator, InpaintPaste, Region, VisualKind,
};
use crate::diffusion_models::musetalk::MuseTalk;
use crate::speech_models::whisper::WhisperFeatureExtractor;
use crate::vision_models::s3fd::{FaceBox, S3fd, S3fdConfig};

// MuseTalk detects per frame; 1 == reference parity. Raise it to trade lip-sync
// accuracy (a stale box drifts as the head moves) for fewer S3FD forwards.
const DEFAULT_REDETECT_EVERY: usize = 1;

/// Animator knobs. Face-detector parameters are surfaced here so the detector
/// stays an internal composition detail.
#[derive(Clone, Copy, Debug)]
pub struct AnimatorOptions {
    /// Re-run face detection every N output frames; reuse the last box in between.
    /// 1 matches MuseTalk's per-frame detection; >1 is a speed/quality tradeoff.
    pub redetect_every: usize,
    pub face_score_threshold: f32,
    pub face_nms_iou: f32,
    pub face_max_side: u32,
}

impl Default for AnimatorOptions {
    fn default() -> Self {
        let s = S3fdConfig::default();
        Self {
            redetect_every: DEFAULT_REDETECT_EVERY,
            face_score_threshold: s.score_threshold,
            face_nms_iou: s.nms_iou,
            face_max_side: s.max_side,
        }
    }
}

/// S3FD face detection as a `FaceLocator`: the top box clamped to the frame, or `None`
/// when nothing is detected. Footage-generic, reusable by any inpainting backbone.
pub struct S3fdLocator {
    detector: S3fd,
}

impl S3fdLocator {
    pub fn new(vb: ShardedVarBuilder, device: &Device, opts: AnimatorOptions) -> Result<Self> {
        Ok(Self {
            detector: S3fd::new(
                vb,
                device,
                S3fdConfig {
                    score_threshold: opts.face_score_threshold,
                    nms_iou: opts.face_nms_iou,
                    max_side: opts.face_max_side,
                },
            )?,
        })
    }
}

impl FaceLocator for S3fdLocator {
    fn locate(&self, frame: &DynamicImage) -> Result<Option<Region>> {
        let (fw, fh) = frame.dimensions();
        Ok(self.detector.detect(frame)?.into_iter().next().map(|b| {
            let (x, y, w, h) = clamp_box(&b, fw, fh);
            Region { x, y, w, h }
        }))
    }
}

/// openai-whisper-tiny stacked encoder features as MuseTalk's lip-sync `DriveEncoder`.
impl DriveEncoder for WhisperFeatureExtractor {
    fn encode(&self, audio: &DrivingAudio, fps: f64) -> Result<Tensor> {
        self.features(&audio.pcm, audio.sample_rate, fps)
    }
}

/// MuseTalk VAE-encode + UNet cross-attn + VAE-decode + blend (keep upper half, generate
/// lower) as the `Generator`. Inpaints a face crop, so its `VisualKind` is `Footage`.
pub struct MuseTalkGenerator {
    musetalk: MuseTalk,
    size: usize,
    dtype: DType,
    device: Device,
}

impl MuseTalkGenerator {
    pub fn new(musetalk: MuseTalk) -> Self {
        Self {
            size: musetalk.resized_img(),
            dtype: musetalk.dtype(),
            device: musetalk.device().clone(),
            musetalk,
        }
    }
}

impl Generator for MuseTalkGenerator {
    fn generate(&self, visual: &DynamicImage, audio: &Tensor) -> Result<DynamicImage> {
        let face = image_to_face_tensor(visual, self.size, &self.device)?;
        // whisper features carry whisper's vb dtype; the UNet cross-attends in MuseTalk's.
        let audio = audio.to_dtype(self.dtype)?.contiguous()?;
        let generated = self.musetalk.forward(&face, &audio)?;
        let blended = self.musetalk.blend(&face, &generated)?;
        face_tensor_to_image(&blended)
    }

    fn accepts(&self) -> VisualKind {
        VisualKind::Footage
    }
}

/// Footage-driven lip-sync animator: `S3fdLocator` + whisper-tiny `DriveEncoder` +
/// `MuseTalkGenerator` + `InpaintPaste`. EchoMimic/LongCat/InfiniteTalk drop in as a new
/// `Generator` (+ `Compositor`), reusing locate/encode, with zero pipeline change.
pub struct MuseTalkAnimator(
    Animator<S3fdLocator, WhisperFeatureExtractor, MuseTalkGenerator, InpaintPaste>,
);

impl MuseTalkAnimator {
    pub fn new(
        musetalk: MuseTalk,
        whisper: WhisperFeatureExtractor,
        s3fd_vb: ShardedVarBuilder,
        opts: AnimatorOptions,
    ) -> Result<Self> {
        let want = musetalk.cross_attention_dim();
        let got = whisper.config().n_audio_state;
        if want != got {
            hanzo_ml::bail!(
                "whisper n_audio_state ({got}) must equal MuseTalk cross_attention_dim ({want})"
            );
        }
        let device = musetalk.device().clone();
        let locate = S3fdLocator::new(s3fd_vb, &device, opts)?;
        Ok(Self(Animator::new(
            locate,
            whisper,
            MuseTalkGenerator::new(musetalk),
            InpaintPaste,
            device,
            opts.redetect_every,
        )))
    }
}

impl FacialAnimator for MuseTalkAnimator {
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

fn clamp_box(b: &FaceBox, fw: u32, fh: u32) -> (u32, u32, u32, u32) {
    let x1 = (b.x1.floor().max(0.0) as u32).min(fw.saturating_sub(1));
    let y1 = (b.y1.floor().max(0.0) as u32).min(fh.saturating_sub(1));
    // `fw.max(x1 + 1)` keeps the clamp range valid even for a degenerate 0-size frame.
    let x2 = (b.x2.ceil().max(0.0) as u32).clamp(x1 + 1, fw.max(x1 + 1));
    let y2 = (b.y2.ceil().max(0.0) as u32).clamp(y1 + 1, fh.max(y1 + 1));
    (x1, y1, x2 - x1, y2 - y1)
}

/// `DynamicImage` -> MuseTalk face input `[1, 3, size, size]`, RGB, [0,1].
fn image_to_face_tensor(img: &DynamicImage, size: usize, device: &Device) -> Result<Tensor> {
    let rgb = img
        .resize_exact(size as u32, size as u32, FilterType::Triangle)
        .to_rgb8();
    let raw = rgb.into_raw();
    let plane = size * size;
    let mut data = vec![0f32; 3 * plane];
    for i in 0..plane {
        data[i] = raw[i * 3] as f32 / 255.0;
        data[plane + i] = raw[i * 3 + 1] as f32 / 255.0;
        data[2 * plane + i] = raw[i * 3 + 2] as f32 / 255.0;
    }
    Tensor::from_vec(data, (1, 3, size, size), device)
}

/// MuseTalk output `[1, 3, H, W]` (RGB, [0,1]) -> `DynamicImage`.
fn face_tensor_to_image(t: &Tensor) -> Result<DynamicImage> {
    let (_, c, h, w) = t.dims4()?;
    debug_assert_eq!(c, 3);
    let v = t.clamp(0f32, 1f32)?.flatten_all()?.to_vec1::<f32>()?;
    let plane = h * w;
    let mut buf = vec![0u8; 3 * plane];
    for i in 0..plane {
        buf[i * 3] = (v[i] * 255.0) as u8;
        buf[i * 3 + 1] = (v[plane + i] * 255.0) as u8;
        buf[i * 3 + 2] = (v[2 * plane + i] * 255.0) as u8;
    }
    let img = RgbImage::from_raw(w as u32, h as u32, buf)
        .ok_or_else(|| hanzo_ml::Error::msg("face buffer size mismatch"))?;
    Ok(DynamicImage::ImageRgb8(img))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_box_stays_in_bounds() {
        let b = FaceBox {
            x1: -5.0,
            y1: 2.0,
            x2: 200.0,
            y2: 50.0,
            score: 0.9,
        };
        let (x, y, w, h) = clamp_box(&b, 64, 64);
        assert_eq!((x, y), (0, 2));
        assert!(x + w <= 64 && y + h <= 64);
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn clamp_box_total_on_zero_dims() {
        // Must not panic on a degenerate 0-size frame (clamp range stays valid).
        let b = FaceBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
        };
        let (x, y, w, h) = clamp_box(&b, 0, 0);
        assert_eq!((x, y), (0, 0));
        assert!(w >= 1 && h >= 1);
    }
}
