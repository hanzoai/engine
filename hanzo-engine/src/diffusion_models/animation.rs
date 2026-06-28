use hanzo_ml::{Device, Result};
use image::DynamicImage;
use std::sync::Arc;

/// Sample rate the omni speak path emits; the only PCM rate a `DrivingAudio` carries.
pub const OMNI_SAMPLE_RATE: usize = 24_000;

pub struct DrivingAudio {
    pub pcm: Arc<Vec<f32>>,
    pub sample_rate: usize,
}

impl DrivingAudio {
    pub fn new(pcm: Arc<Vec<f32>>) -> Self {
        Self {
            pcm,
            sample_rate: OMNI_SAMPLE_RATE,
        }
    }
}

pub enum VisualSource {
    Footage { frames: Vec<DynamicImage> },
    Portrait { image: DynamicImage },
}

impl VisualSource {
    pub fn kind(&self) -> VisualKind {
        match self {
            Self::Footage { .. } => VisualKind::Footage,
            Self::Portrait { .. } => VisualKind::Portrait,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisualKind {
    Footage,
    Portrait,
    Either,
}

impl VisualKind {
    /// The one place the kind policy lives: an animator advertising `self` admits `source`.
    /// `Either` admits both; a concrete kind admits only its own source. The pipeline calls
    /// this once before dispatch so no animator impl ever re-checks its input shape.
    pub fn admits(self, source: &VisualSource) -> bool {
        match self {
            Self::Either => true,
            kind => kind == source.kind(),
        }
    }
}

pub struct AnimationRequest {
    pub driving: DrivingAudio,
    pub visual: VisualSource,
    pub fps: f64,
}

/// Per-request animation knobs carried on a `Sequence` (mirrors
/// `DiffusionGenerationParams`); the frames/PCM ride the existing multimodal
/// image/audio slots, so only the output rate + source kind live here.
#[derive(Clone, Copy, Debug)]
pub struct AnimationGenerationParams {
    pub fps: f64,
    pub kind: VisualKind,
}

pub struct AnimationOutput {
    pub frames: Vec<DynamicImage>,
    pub fps: f64,
}

pub trait FacialAnimator: Send + Sync {
    fn animate(&mut self, req: &AnimationRequest) -> Result<AnimationOutput>;
    fn device(&self) -> &Device;
    fn accepts(&self) -> VisualKind;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portrait() -> VisualSource {
        VisualSource::Portrait {
            image: DynamicImage::new_rgb8(8, 8),
        }
    }

    fn footage() -> VisualSource {
        VisualSource::Footage {
            frames: vec![DynamicImage::new_rgb8(8, 8)],
        }
    }

    #[test]
    fn visual_kind_gate() {
        assert_eq!(portrait().kind(), VisualKind::Portrait);
        assert_eq!(footage().kind(), VisualKind::Footage);

        assert!(VisualKind::Either.admits(&portrait()));
        assert!(VisualKind::Either.admits(&footage()));

        assert!(VisualKind::Footage.admits(&footage()));
        assert!(!VisualKind::Footage.admits(&portrait()));

        assert!(VisualKind::Portrait.admits(&portrait()));
        assert!(!VisualKind::Portrait.admits(&footage()));
    }
}
