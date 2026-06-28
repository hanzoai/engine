#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;
use hanzo_vision::{crop, paste_resized};
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};

use crate::diffusion_models::animation::{
    AnimationOutput, AnimationRequest, FacialAnimator, VisualKind, VisualSource,
};
use crate::diffusion_models::musetalk::MuseTalk;
use crate::speech_models::whisper::WhisperFeatureExtractor;
use crate::vision_models::s3fd::{FaceBox, S3fd, S3fdConfig};

const DEFAULT_REDETECT_EVERY: usize = 5;

/// Animator knobs. Face-detector parameters are surfaced here so the detector
/// stays an internal composition detail.
#[derive(Clone, Copy, Debug)]
pub struct AnimatorOptions {
    /// Re-run face detection every N output frames; reuse the last box in between.
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

/// Footage-driven (inpainting) facial animator: S3FD locates the face, the
/// openai-whisper-tiny features condition the MuseTalk UNet, and each frame is
/// crop -> forward -> blend (keep upper half, generate lower half) -> paste back.
pub struct MuseTalkAnimator {
    musetalk: MuseTalk,
    whisper: WhisperFeatureExtractor,
    detector: S3fd,
    device: Device,
    redetect_every: usize,
}

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
        let detector = S3fd::new(
            s3fd_vb,
            &device,
            S3fdConfig {
                score_threshold: opts.face_score_threshold,
                nms_iou: opts.face_nms_iou,
                max_side: opts.face_max_side,
            },
        )?;
        Ok(Self {
            musetalk,
            whisper,
            detector,
            device,
            redetect_every: opts.redetect_every.max(1),
        })
    }

    fn bbox_for(
        &self,
        frame: &DynamicImage,
        redetect: bool,
        last: &mut Option<FaceBox>,
    ) -> Result<(u32, u32, u32, u32)> {
        if redetect || last.is_none() {
            if let Some(top) = self.detector.detect(frame)?.into_iter().next() {
                *last = Some(top);
            }
        }
        let (fw, fh) = frame.dimensions();
        // No face ever found: fall back to the whole frame (headshot footage).
        Ok(last.map_or((0, 0, fw, fh), |b| clamp_box(&b, fw, fh)))
    }
}

impl FacialAnimator for MuseTalkAnimator {
    fn animate(&mut self, req: &AnimationRequest) -> Result<AnimationOutput> {
        let VisualSource::Footage { frames, .. } = &req.visual else {
            hanzo_ml::bail!("MuseTalkAnimator only animates footage");
        };
        if frames.is_empty() {
            hanzo_ml::bail!("MuseTalkAnimator: empty footage");
        }

        let feats = self
            .whisper
            .features(&req.driving.pcm, req.driving.sample_rate, req.fps)?;
        let n_frames = feats.dim(0)?;
        let size = self.musetalk.resized_img();

        let mut out = Vec::with_capacity(n_frames);
        let mut last_bbox: Option<FaceBox> = None;
        for t in 0..n_frames {
            let src = &frames[cycle_index(t, frames.len())];
            let redetect = t % self.redetect_every == 0;
            let (x, y, w, h) = self.bbox_for(src, redetect, &mut last_bbox)?;

            let face = image_to_face_tensor(&crop(src, x, y, w, h), size, &self.device)?;
            let audio = feats.narrow(0, t, 1)?.contiguous()?;
            let generated = self.musetalk.forward(&face, &audio)?;
            let blended = self.musetalk.blend(&face, &generated)?;

            let patch = face_tensor_to_image(&blended)?;
            let mut frame_out = src.clone();
            paste_resized(&mut frame_out, &patch, x, y, w, h);
            out.push(frame_out);
        }
        Ok(AnimationOutput {
            frames: out,
            fps: req.fps,
        })
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn accepts(&self) -> VisualKind {
        VisualKind::Footage
    }
}

/// Ping-pong (boomerang) index into `n` footage frames so audio longer than the
/// footage loops without a hard cut: 0,1,..,n-1,n-1,..,1,0,0,1,..
fn cycle_index(t: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * n;
    let m = t % period;
    if m < n {
        m
    } else {
        period - 1 - m
    }
}

fn clamp_box(b: &FaceBox, fw: u32, fh: u32) -> (u32, u32, u32, u32) {
    let x1 = (b.x1.floor().max(0.0) as u32).min(fw.saturating_sub(1));
    let y1 = (b.y1.floor().max(0.0) as u32).min(fh.saturating_sub(1));
    let x2 = (b.x2.ceil().max(0.0) as u32).clamp(x1 + 1, fw);
    let y2 = (b.y2.ceil().max(0.0) as u32).clamp(y1 + 1, fh);
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
    fn cycle_index_pingpongs() {
        assert_eq!((0..5).map(|t| cycle_index(t, 1)).collect::<Vec<_>>(), [0, 0, 0, 0, 0]);
        assert_eq!((0..5).map(|t| cycle_index(t, 2)).collect::<Vec<_>>(), [0, 1, 1, 0, 0]);
        assert_eq!(
            (0..8).map(|t| cycle_index(t, 3)).collect::<Vec<_>>(),
            [0, 1, 2, 2, 1, 0, 0, 1]
        );
    }

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
}
