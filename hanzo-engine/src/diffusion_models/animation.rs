use hanzo_ml::{Device, Result, Tensor};
use hanzo_vision::{crop, paste_resized};
use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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

/// A rectangle in frame pixel space; the unit of work threaded locate -> crop -> compose.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Region {
    fn whole(frame: &DynamicImage) -> Self {
        let (w, h) = frame.dimensions();
        Self { x: 0, y: 0, w, h }
    }
}

/// Where the generator operates in a frame; `None` == nothing located this frame.
pub trait FaceLocator: Send + Sync {
    fn locate(&self, frame: &DynamicImage) -> Result<Option<Region>>;
}

/// Locates the whole frame: portrait/I2V backbones generate over the entire image.
pub struct WholeFrame;

impl FaceLocator for WholeFrame {
    fn locate(&self, frame: &DynamicImage) -> Result<Option<Region>> {
        Ok(Some(Region::whole(frame)))
    }
}

/// Driving audio -> per-output-frame condition features `[T, _, D]`, `T = ceil(secs * fps)`.
pub trait DriveEncoder: Send + Sync {
    fn encode(&self, audio: &DrivingAudio, fps: f64) -> Result<Tensor>;
}

/// The neural core: one output frame from a visual input (footage crop or source portrait)
/// and that frame's audio features `[1, _, D]`. `accepts` is the source kind it consumes.
pub trait Generator: Send + Sync {
    fn generate(&self, visual: &DynamicImage, audio: &Tensor) -> Result<DynamicImage>;
    fn accepts(&self) -> VisualKind;
}

/// Places generated content back into the full frame.
pub trait Compositor: Send + Sync {
    fn compose(
        &self,
        frame: &DynamicImage,
        region: Region,
        generated: &DynamicImage,
    ) -> DynamicImage;
}

/// Footage inpainting: resize the patch to the located box and paste it over a frame copy.
pub struct InpaintPaste;

impl Compositor for InpaintPaste {
    fn compose(
        &self,
        frame: &DynamicImage,
        region: Region,
        generated: &DynamicImage,
    ) -> DynamicImage {
        let mut out = frame.clone();
        paste_resized(&mut out, generated, region.x, region.y, region.w, region.h);
        out
    }
}

/// I2V: the generator produced the whole frame, so the output is exactly that.
pub struct FullFrame;

impl Compositor for FullFrame {
    fn compose(
        &self,
        _frame: &DynamicImage,
        _region: Region,
        generated: &DynamicImage,
    ) -> DynamicImage {
        generated.clone()
    }
}

/// Composes the four orthogonal concerns into a `FacialAnimator`: per frame -> locate ->
/// crop -> generate(audio[t]) -> compose. A new backbone is one `Generator` (+ maybe a
/// `Compositor`), reusing locate/encode, with zero pipeline change.
pub struct Animator<L, E, G, C> {
    locate: L,
    encode: E,
    gen: G,
    compose: C,
    device: Device,
    redetect_every: usize,
}

impl<L: FaceLocator, E: DriveEncoder, G: Generator, C: Compositor> Animator<L, E, G, C> {
    pub fn new(
        locate: L,
        encode: E,
        gen: G,
        compose: C,
        device: Device,
        redetect_every: usize,
    ) -> Self {
        Self {
            locate,
            encode,
            gen,
            compose,
            device,
            redetect_every: redetect_every.max(1),
        }
    }
}

impl<L: FaceLocator, E: DriveEncoder, G: Generator, C: Compositor> FacialAnimator
    for Animator<L, E, G, C>
{
    fn animate(&mut self, req: &AnimationRequest) -> Result<AnimationOutput> {
        let frames: &[DynamicImage] = match &req.visual {
            VisualSource::Footage { frames } => frames,
            VisualSource::Portrait { image } => std::slice::from_ref(image),
        };
        if frames.is_empty() {
            hanzo_ml::bail!("animator: empty visual source");
        }
        if frames.iter().any(|f| f.width() == 0 || f.height() == 0) {
            hanzo_ml::bail!("animator: a frame has a zero dimension");
        }

        let feats = self.encode.encode(&req.driving, req.fps)?;
        let n_frames = feats.dim(0)?;

        let mut out = Vec::with_capacity(n_frames);
        let mut last: Option<Region> = None;
        for t in 0..n_frames {
            let src = &frames[cycle_index(t, frames.len())];
            if t % self.redetect_every == 0 || last.is_none() {
                if let Some(r) = self.locate.locate(src)? {
                    last = Some(r);
                }
            }
            let region = last.unwrap_or_else(|| Region::whole(src));
            let visual = crop(src, region.x, region.y, region.w, region.h);
            let generated = self.gen.generate(&visual, &feats.narrow(0, t, 1)?)?;
            out.push(self.compose.compose(src, region, &generated));
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
        self.gen.accepts()
    }
}

/// Ping-pong (boomerang) index into `n` frames so audio longer than the footage loops
/// without a hard cut: 0,1,..,n-1,n-1,..,1,0,0,1,.. (a single frame stays at 0).
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

    #[test]
    fn cycle_index_pingpongs() {
        assert_eq!(
            (0..5).map(|t| cycle_index(t, 1)).collect::<Vec<_>>(),
            [0, 0, 0, 0, 0]
        );
        assert_eq!(
            (0..5).map(|t| cycle_index(t, 2)).collect::<Vec<_>>(),
            [0, 1, 1, 0, 0]
        );
        assert_eq!(
            (0..8).map(|t| cycle_index(t, 3)).collect::<Vec<_>>(),
            [0, 1, 2, 2, 1, 0, 0, 1]
        );
    }
}
