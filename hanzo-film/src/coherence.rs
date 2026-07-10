//! Coherence verification hook. After render, sample a frame from shot pairs that
//! share a character and score visual consistency.
//!
//! The shipped scorer is a dependency-free STRUCTURAL proxy: cosine similarity of
//! downsampled luma. It is honest about being a proxy, not identity-CLIP. The
//! `Scorer::Embedding` variant is wired to `/v1/embeddings` for the day a vision
//! embedding model is loaded, at which point it becomes true identity similarity
//! with no other change to the pipeline.

use crate::bible::Bible;
use crate::engine::Engine;
use crate::ffmpeg;
use crate::project::Project;
use anyhow::Result;
use base64::Engine as _;
use serde::Serialize;
use std::path::Path;

#[derive(Clone)]
pub enum Scorer {
    /// Downsampled-luma cosine. Real number, no network. Structural, not identity.
    Pixel,
    /// Embed each frame via `/v1/embeddings` and cosine the vectors (needs a
    /// vision-embedding model; text-only embed models will error and the caller
    /// falls back to `Pixel`).
    Embedding { engine: Engine, model: String },
}

#[derive(Debug, Serialize)]
pub struct PairScore {
    pub character: String,
    pub shot_a: String,
    pub shot_b: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub method: String,
    pub note: String,
    pub mean: f32,
    pub pairs: Vec<PairScore>,
}

impl Scorer {
    fn method(&self) -> &'static str {
        match self {
            Scorer::Pixel => "pixel-luma-cosine",
            Scorer::Embedding { .. } => "embedding-cosine",
        }
    }

    async fn vector(&self, frame_png: &Path) -> Result<Vec<f32>> {
        match self {
            Scorer::Pixel => luma_vector(frame_png),
            Scorer::Embedding { engine, model } => {
                let bytes = std::fs::read(frame_png)?;
                let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
                let data_uri = format!("data:image/png;base64,{b64}");
                engine.embed(model, &data_uri).await
            }
        }
    }
}

/// 32x32 grayscale, L2-normalized — a compact structural fingerprint of a frame.
fn luma_vector(path: &Path) -> Result<Vec<f32>> {
    let img = image::open(path)?.to_luma8();
    let small = image::imageops::resize(&img, 32, 32, image::imageops::FilterType::Triangle);
    let mut v: Vec<f32> = small.pixels().map(|p| p.0[0] as f32).collect();
    l2_normalize(&mut v);
    Ok(v)
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// For each character, sample a mid-frame from each shot they appear in, score
/// consecutive pairs, and aggregate. `max_pairs` caps work on long films.
pub async fn verify(project: &Project, bible: &Bible, scorer: Scorer, max_pairs: usize) -> Result<Report> {
    let mut pairs = Vec::new();

    for ch in &bible.characters {
        // Shots where this character is present (scene roster or a spoken line).
        let shots: Vec<(&str, f32)> = bible
            .shots()
            .filter(|(scene, shot)| {
                scene.characters.iter().any(|c| c == &ch.id)
                    || shot.dialogue.iter().any(|d| d.character_ref == ch.id)
            })
            .map(|(_, shot)| (shot.id.as_str(), shot.duration_s))
            .filter(|(id, _)| project.shot_clip(id).exists())
            .collect();

        for w in shots.windows(2) {
            if pairs.len() >= max_pairs {
                break;
            }
            let (a, a_dur) = w[0];
            let (b, b_dur) = w[1];
            let fa = project.root.join("shots").join(format!("{a}.mid.png"));
            let fb = project.root.join("shots").join(format!("{b}.mid.png"));
            ffmpeg::frame_at(&project.shot_clip(a), a_dur / 2.0, &fa).await?;
            ffmpeg::frame_at(&project.shot_clip(b), b_dur / 2.0, &fb).await?;
            let va = scorer.vector(&fa).await?;
            let vb = scorer.vector(&fb).await?;
            pairs.push(PairScore {
                character: ch.id.clone(),
                shot_a: a.to_string(),
                shot_b: b.to_string(),
                score: cosine(&va, &vb),
            });
        }
    }

    let mean = if pairs.is_empty() {
        1.0
    } else {
        pairs.iter().map(|p| p.score).sum::<f32>() / pairs.len() as f32
    };
    Ok(Report {
        method: scorer.method().to_string(),
        note: match scorer {
            Scorer::Pixel => "structural luma proxy; swap to a vision-embedding model for identity-grade CLIP scoring".into(),
            Scorer::Embedding { .. } => "vision-embedding cosine".into(),
        },
        mean,
        pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = a.clone();
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_length_mismatch_is_zero() {
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
    }
}
