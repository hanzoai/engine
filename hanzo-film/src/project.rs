//! On-disk project: a plain directory of JSON + media. No database. The directory
//! *is* the state, which is what makes every stage resumable and inspectable.
//!
//! Layout:
//! ```text
//! <proj>/
//!   project.json      brief + config (engine urls, model ids, geometry)
//!   bible.json        the validated Bible contract instance
//!   assets/           char_<id>.png, loc_<id>.png  (reference images)
//!   shots/            <shot>.mp4, <shot>.json (spec-hash sidecar), <shot>.tail.png
//!   audio/            line_<shot>_<n>.wav, music_<scene>.wav, mix_<shot>.wav
//!   timeline.json     EDL manifest (the durable assembly artifact)
//!   coherence.json    identity-consistency scores
//!   film.mp4          final render
//! ```

use crate::bible::Bible;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Rendering geometry + backend selection + engine wiring. Persisted so re-runs
/// are deterministic; CLI flags seed it once at `new` time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub engine_url: String,
    pub planner_model: String,
    pub image_model: String,
    pub tts_model: String,
    pub music_model: String,
    pub video: VideoBackend,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    pub concurrency: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoBackend {
    /// image keyframe + ffmpeg Ken-Burns to duration. Real output, no video model.
    Placeholder,
    /// the WAN `/v1/videos` async endpoint (sibling; pending until it lands).
    Wan,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            engine_url: "http://127.0.0.1:1234".into(),
            planner_model: "default".into(),
            image_model: "default".into(),
            tts_model: "default".into(),
            music_model: "default".into(),
            video: VideoBackend::Placeholder,
            width: 768,
            height: 432,
            fps: 24,
            concurrency: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub brief: String,
    pub config: Config,
}

/// Handle to a project directory. Cheap to clone; holds only the root path + manifest.
#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
    pub manifest: Manifest,
}

impl Project {
    /// Create a fresh project dir with a brief and config; fails if it already exists.
    pub fn create(root: impl AsRef<Path>, brief: String, config: Config) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if root.join("project.json").exists() {
            anyhow::bail!("project already exists at {}", root.display());
        }
        std::fs::create_dir_all(&root)?;
        let p = Self {
            root,
            manifest: Manifest { brief, config },
        };
        p.ensure_dirs()?;
        p.save_manifest()?;
        Ok(p)
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let raw = std::fs::read_to_string(root.join("project.json"))
            .with_context(|| format!("no project.json in {}", root.display()))?;
        let manifest: Manifest = serde_json::from_str(&raw)?;
        let p = Self { root, manifest };
        p.ensure_dirs()?;
        Ok(p)
    }

    fn ensure_dirs(&self) -> Result<()> {
        for d in ["assets", "shots", "audio"] {
            std::fs::create_dir_all(self.root.join(d))?;
        }
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.manifest.config
    }

    pub fn save_manifest(&self) -> Result<()> {
        write_json(&self.root.join("project.json"), &self.manifest)
    }

    // --- paths (single source of truth for the layout) ---------------------

    pub fn bible_path(&self) -> PathBuf {
        self.root.join("bible.json")
    }
    pub fn timeline_path(&self) -> PathBuf {
        self.root.join("timeline.json")
    }
    pub fn coherence_path(&self) -> PathBuf {
        self.root.join("coherence.json")
    }
    pub fn film_path(&self) -> PathBuf {
        self.root.join("film.mp4")
    }
    pub fn character_ref(&self, id: &str) -> PathBuf {
        self.root.join("assets").join(format!("char_{id}.png"))
    }
    pub fn location_ref(&self, id: &str) -> PathBuf {
        self.root.join("assets").join(format!("loc_{id}.png"))
    }
    pub fn shot_clip(&self, id: &str) -> PathBuf {
        self.root.join("shots").join(format!("{id}.mp4"))
    }
    pub fn shot_sidecar(&self, id: &str) -> PathBuf {
        self.root.join("shots").join(format!("{id}.json"))
    }
    pub fn shot_tail(&self, id: &str) -> PathBuf {
        self.root.join("shots").join(format!("{id}.tail.png"))
    }
    pub fn shot_keyframe(&self, id: &str) -> PathBuf {
        self.root.join("shots").join(format!("{id}.key.png"))
    }
    pub fn dialogue_wav(&self, shot: &str, n: usize) -> PathBuf {
        self.root.join("audio").join(format!("line_{shot}_{n}.wav"))
    }
    pub fn music_wav(&self, scene: &str) -> PathBuf {
        self.root.join("audio").join(format!("music_{scene}.wav"))
    }
    pub fn mix_wav(&self, shot: &str) -> PathBuf {
        self.root.join("audio").join(format!("mix_{shot}.wav"))
    }

    // --- bible ------------------------------------------------------------

    pub fn load_bible(&self) -> Result<Bible> {
        let raw = std::fs::read_to_string(self.bible_path())
            .with_context(|| "no bible.json; run `plan` first")?;
        let bible = Bible::from_json(&raw)?;
        bible.validate()?;
        Ok(bible)
    }

    pub fn save_bible(&self, bible: &Bible) -> Result<()> {
        bible.validate()?;
        std::fs::write(self.bible_path(), bible.to_json())?;
        Ok(())
    }
}

/// The spec-hash sidecar written next to each rendered shot. A shot re-renders
/// only when its clip is missing or the recorded hash no longer matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotRecord {
    pub shot_id: String,
    pub spec_hash: String,
    pub renderer: String,
    pub duration_s: f32,
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_open_roundtrip() {
        let dir = std::env::temp_dir().join(format!("film_proj_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = Config::default();
        let p = Project::create(&dir, "a brief".into(), cfg).unwrap();
        assert!(p.bible_path().starts_with(&dir));
        let re = Project::open(&dir).unwrap();
        assert_eq!(re.manifest.brief, "a brief");
        assert_eq!(re.config().width, 768);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn create_twice_fails() {
        let dir = std::env::temp_dir().join(format!("film_proj2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Project::create(&dir, "b".into(), Config::default()).unwrap();
        assert!(Project::create(&dir, "b".into(), Config::default()).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
