//! Pluggable media backends. Each modality is an enum with an engine-backed
//! variant and a dependency-free variant, dispatched without vtables. The
//! engine variant is the real target; the procedural/silent variant proves the
//! whole pipeline end-to-end on a box with no image/video/TTS model loaded.

use crate::engine::Engine;
use crate::ffmpeg;
use anyhow::{bail, Result};
use std::path::Path;
use std::time::Duration;

/// Reference/keyframe images. `Engine` hits `/v1/images/generations`; `Procedural`
/// draws a stable captioned color card.
#[derive(Clone)]
pub enum Images {
    Engine { engine: Engine, model: String },
    Procedural,
}

impl Images {
    pub fn label(&self) -> &'static str {
        match self {
            Images::Engine { .. } => "engine",
            Images::Procedural => "procedural",
        }
    }

    /// Render `prompt` to a PNG at `out`. `label` seeds the procedural card's
    /// deterministic color (e.g. a character id).
    pub async fn make(&self, prompt: &str, label: &str, width: usize, height: usize, out: &Path) -> Result<()> {
        match self {
            Images::Engine { engine, model } => {
                let bytes = engine.image(model, prompt, width, height).await?;
                std::fs::write(out, bytes)?;
                Ok(())
            }
            Images::Procedural => ffmpeg::procedural_card(prompt, label, width, height, out).await,
        }
    }
}

/// Per-shot video clip. `Placeholder` Ken-Burns a still to duration (real mp4).
/// `Wan` drives the async `/v1/videos` endpoint (sibling; pending until it lands).
#[derive(Clone)]
pub enum Video {
    Placeholder,
    Wan { engine: Engine, steps: usize },
}

impl Video {
    pub fn kind(&self) -> &'static str {
        match self {
            Video::Placeholder => "placeholder",
            Video::Wan { .. } => "wan",
        }
    }

    /// Produce a silent clip of `dur_s` at `out`.
    /// `still` is the conditioning image (a fresh keyframe for `cut`, the prior
    /// shot's tail frame for `continue`); `prompt` drives the real video model.
    pub async fn clip(
        &self,
        prompt: &str,
        still: &Path,
        dur_s: f32,
        width: usize,
        height: usize,
        fps: usize,
        grade: Option<&str>,
        out: &Path,
    ) -> Result<()> {
        match self {
            Video::Placeholder => ffmpeg::kenburns_clip(still, dur_s, width, height, fps, grade, out).await,
            Video::Wan { engine, steps } => {
                let frames = ((dur_s * fps as f32).round() as usize).max(1);
                let id = engine.video_create(prompt, frames, width, height, *steps).await?;
                // Poll to completion. The endpoint is the durable contract; timeout guards a stuck job.
                let deadline = std::time::Instant::now() + Duration::from_secs(1800);
                loop {
                    let (status, _progress) = engine.video_status(&id).await?;
                    match status.as_str() {
                        "completed" => break,
                        "failed" => bail!("wan job {id} failed"),
                        _ if std::time::Instant::now() > deadline => bail!("wan job {id} timed out"),
                        _ => tokio::time::sleep(Duration::from_secs(2)).await,
                    }
                }
                let bytes = engine.video_content(&id).await?;
                let tmp = out.with_extension("wan.mp4");
                std::fs::write(&tmp, bytes)?;
                ffmpeg::normalize_clip(&tmp, width, height, fps, out).await?;
                let _ = std::fs::remove_file(&tmp);
                Ok(())
            }
        }
    }
}

/// Dialogue TTS. `Engine` hits `/v1/audio/speech`; `Placeholder` emits a quiet
/// sine sized to the estimated speaking time so mix placement is demonstrable.
#[derive(Clone)]
pub enum Speech {
    Engine { engine: Engine, model: String },
    Placeholder,
}

/// Rough speaking duration for a line: ~2.6 words/sec, floored at 0.8s.
pub fn estimate_speech_s(text: &str) -> f32 {
    let words = text.split_whitespace().count().max(1) as f32;
    (words / 2.6).max(0.8)
}

impl Speech {
    /// Synthesize `text` to a canonical WAV at `out`; returns its real duration.
    pub async fn say(&self, text: &str, out: &Path) -> Result<f32> {
        match self {
            Speech::Engine { engine, model } => {
                let bytes = engine.speech(model, text).await?;
                let tmp = out.with_extension("raw");
                std::fs::write(&tmp, bytes)?;
                ffmpeg::to_wav(&tmp, out).await?;
                let _ = std::fs::remove_file(&tmp);
                Ok(ffmpeg::probe(out).await?.duration_s())
            }
            Speech::Placeholder => {
                let dur = estimate_speech_s(text);
                ffmpeg::tone(dur, 220.0, out).await?;
                Ok(dur)
            }
        }
    }
}

/// Scene score. `Silent` is the default bed until the ACE-Step music endpoint
/// lands; `Engine` is wired to `/v1/audio/music` for when it does.
#[derive(Clone)]
pub enum Music {
    Silent,
    Engine { engine: Engine, model: String },
}

impl Music {
    pub async fn cue(&self, prompt: &str, dur_s: f32, out: &Path) -> Result<()> {
        match self {
            Music::Silent => ffmpeg::silence(dur_s, out).await,
            Music::Engine { engine, model } => {
                let bytes = engine.music(model, prompt, dur_s).await?;
                let tmp = out.with_extension("raw");
                std::fs::write(&tmp, bytes)?;
                ffmpeg::to_wav(&tmp, out).await?;
                let _ = std::fs::remove_file(&tmp);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_estimate_scales_with_words() {
        assert!(estimate_speech_s("one") >= 0.8);
        assert!(estimate_speech_s("a b c d e f g h i j") > estimate_speech_s("a b"));
    }

    #[test]
    fn backend_labels() {
        assert_eq!(Video::Placeholder.kind(), "placeholder");
        assert_eq!(Images::Procedural.label(), "procedural");
    }
}
