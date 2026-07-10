//! The render pipeline: assets -> shots -> audio -> assembly. Stages are
//! independent and resumable; each writes into the project dir and reads what the
//! prior stage left. Shots are idempotent jobs keyed by a content spec-hash and
//! dispatched in parallel across continuity runs.

use crate::backends::{estimate_speech_s, Images, Music, Speech, Video};
use crate::bible::{Bible, Continuity, Scene, Shot};
use crate::engine::Engine;
use crate::ffmpeg::{self, Placement};
use crate::planner::shot_prompt;
use crate::project::{read_json, write_json, Config, Project, ShotRecord, VideoBackend};
use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Renderer version — bump to invalidate all spec hashes when render semantics change.
const RENDERER_VERSION: u32 = 1;

pub struct Pipeline {
    pub project: Project,
    pub engine: Engine,
    pub images: Images,
    pub video: Video,
    pub speech: Speech,
    pub music: Music,
}

impl Pipeline {
    /// Build backends from the project config. Engine-backed by default; the
    /// procedural/silent fallbacks kick in per-item when an engine call fails.
    pub fn new(project: Project) -> Result<Self> {
        let cfg = project.config().clone();
        let engine = Engine::new(&cfg.engine_url)?;
        let images = Images::Engine { engine: engine.clone(), model: cfg.image_model.clone() };
        let video = match cfg.video {
            VideoBackend::Placeholder => Video::Placeholder,
            VideoBackend::Wan => Video::Wan { engine: engine.clone(), steps: 30 },
        };
        let speech = Speech::Engine { engine: engine.clone(), model: cfg.tts_model.clone() };
        let music = Music::Silent;
        Ok(Self { project, engine, images, video, speech, music })
    }

    fn cfg(&self) -> &Config {
        self.project.config()
    }

    // === asset pass =======================================================

    /// One canonical reference image per character and location. Idempotent: skips
    /// entities whose ref already exists. Fills `reference_image` on the bible.
    pub async fn assets(&self, bible: &mut Bible) -> Result<()> {
        let (w, h) = (self.cfg().width, self.cfg().height);
        for i in 0..bible.characters.len() {
            let c = &bible.characters[i];
            let out = self.project.character_ref(&c.id);
            if !out.exists() {
                let prompt = format!("character reference portrait. {}. {}", c.description, bible.style.prompt);
                self.still(&prompt, &c.id, w, h, &out).await.with_context(|| format!("asset for character {}", c.id))?;
            }
            bible.characters[i].reference_image = Some(rel(&self.project, &out));
        }
        for i in 0..bible.locations.len() {
            let l = &bible.locations[i];
            let out = self.project.location_ref(&l.id);
            if !out.exists() {
                let prompt = format!("establishing location. {}. {}", l.description, bible.style.prompt);
                self.still(&prompt, &l.id, w, h, &out).await.with_context(|| format!("asset for location {}", l.id))?;
            }
            bible.locations[i].reference_image = Some(rel(&self.project, &out));
        }
        self.project.save_bible(bible)?;
        Ok(())
    }

    /// Image with graceful fallback: try the configured backend, else a procedural card.
    async fn still(&self, prompt: &str, label: &str, w: usize, h: usize, out: &std::path::Path) -> Result<()> {
        if let Err(e) = self.images.make(prompt, label, w, h, out).await {
            tracing::warn!("image backend ({}) failed for {label}: {e}; using procedural card", self.images.label());
            Images::Procedural.make(prompt, label, w, h, out).await?;
        }
        Ok(())
    }

    // === shot render ======================================================

    /// Render every shot. Continuity runs (a `cut` followed by its `continue`
    /// shots) are the unit of parallelism: runs go concurrent up to the
    /// configured limit, shots within a run stay sequential because a `continue`
    /// conditions on its predecessor's tail frame. Idempotent per spec-hash.
    pub async fn render(&self, bible: &Bible) -> Result<usize> {
        let runs = continuity_runs(bible);
        let concurrency = self.cfg().concurrency.max(1);
        let results: Vec<Result<usize>> = stream::iter(runs.into_iter().map(|run| self.render_run(bible, run)))
            .buffer_unordered(concurrency)
            .collect()
            .await;
        let mut rendered = 0;
        for r in results {
            rendered += r?;
        }
        Ok(rendered)
    }

    async fn render_run(&self, bible: &Bible, run: Vec<ShotRef>) -> Result<usize> {
        let mut rendered = 0;
        let mut prior: Option<String> = None;
        for sr in run {
            let scene = &bible.scenes[sr.scene];
            let shot = &scene.shots[sr.shot];
            if self.render_shot(bible, scene, shot, prior.as_deref()).await? {
                rendered += 1;
            }
            // Ensure a tail frame exists for a possible following `continue`.
            let tail = self.project.shot_tail(&shot.id);
            if !tail.exists() {
                ffmpeg::tail_frame(&self.project.shot_clip(&shot.id), &tail).await?;
            }
            prior = Some(shot.id.clone());
        }
        Ok(rendered)
    }

    /// Returns true if the shot was (re)rendered, false if skipped as up-to-date.
    async fn render_shot(&self, bible: &Bible, scene: &Scene, shot: &Shot, prior: Option<&str>) -> Result<bool> {
        let cont = matches!(shot.continuity, Continuity::Continue) && prior.is_some();
        let hash = self.spec_hash(bible, scene, shot, prior, cont);
        let clip = self.project.shot_clip(&shot.id);
        let sidecar = self.project.shot_sidecar(&shot.id);

        if clip.exists() {
            if let Some(rec) = read_json::<ShotRecord>(&sidecar)? {
                if rec.spec_hash == hash {
                    tracing::info!("shot {} up-to-date, skipping", shot.id);
                    return Ok(false);
                }
            }
        }

        // Conditioning still: prior tail for `continue`, a fresh keyframe for `cut`.
        let still = if cont {
            self.project.shot_tail(prior.unwrap())
        } else {
            let key = self.project.shot_keyframe(&shot.id);
            let label = scene.characters.first().cloned().unwrap_or_else(|| scene.location_ref.clone());
            self.still(&shot_prompt(bible, scene, shot), &label, self.cfg().width, self.cfg().height, &key).await?;
            key
        };

        let grade = bible.style.grade.as_deref();
        self.video
            .clip(
                &shot_prompt(bible, scene, shot),
                &still,
                shot.duration_s,
                self.cfg().width,
                self.cfg().height,
                self.cfg().fps,
                grade,
                &clip,
            )
            .await
            .with_context(|| format!("rendering shot {}", shot.id))?;

        write_json(
            &sidecar,
            &ShotRecord {
                shot_id: shot.id.clone(),
                spec_hash: hash,
                renderer: self.video.kind().to_string(),
                duration_s: shot.duration_s,
            },
        )?;
        Ok(true)
    }

    /// Stable content hash: everything that determines a shot's pixels.
    fn spec_hash(&self, bible: &Bible, scene: &Scene, shot: &Shot, prior: Option<&str>, cont: bool) -> String {
        let cfg = self.cfg();
        let char_refs: Vec<(&str, &Option<String>)> = scene
            .characters
            .iter()
            .filter_map(|cr| bible.character(cr).map(|c| (c.id.as_str(), &c.reference_image)))
            .collect();
        let key = serde_json::json!({
            "renderer": self.video.kind(),
            "renderer_version": RENDERER_VERSION,
            "prompt": shot_prompt(bible, scene, shot),
            "shot_type": shot.shot_type,
            "duration_s": shot.duration_s,
            "continuity": if cont { "continue" } else { "cut" },
            "cond": if cont { prior.unwrap_or("") } else { "" },
            "style_prompt": bible.style.prompt,
            "style_grade": bible.style.grade,
            "style_lora": bible.style.lora,
            "geometry": [cfg.width, cfg.height, cfg.fps],
            "location_ref": scene.location_ref,
            "location_image": bible.location(&scene.location_ref).and_then(|l| l.reference_image.clone()),
            "characters": char_refs,
        });
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(&key).unwrap());
        format!("{:x}", hasher.finalize())
    }

    // === audio pass =======================================================

    /// Per scene: one music bed (silent by default). Per shot: TTS each dialogue
    /// line, place them on the shot timeline, and mix with the scene bed.
    pub async fn audio(&self, bible: &Bible) -> Result<()> {
        // Scene music beds (best-effort; silence on failure or when endpoint absent).
        for scene in &bible.scenes {
            let bed = self.project.music_wav(&scene.id);
            if bed.exists() {
                continue;
            }
            let dur: f32 = scene.shots.iter().map(|s| s.duration_s).sum();
            let prompt = format!("score for: {}. {}", scene.synopsis, bible.style.prompt);
            if let Err(e) = self.music.cue(&prompt, dur, &bed).await {
                tracing::warn!("music cue for {} failed ({e}); silent bed", scene.id);
                ffmpeg::silence(dur, &bed).await?;
            }
        }

        for scene in &bible.scenes {
            let bed = self.project.music_wav(&scene.id);
            for shot in &scene.shots {
                let mut placements: Vec<Placement> = Vec::new();
                let mut cursor = 0.30_f32;
                for (n, line) in shot.dialogue.iter().enumerate() {
                    let wav = self.project.dialogue_wav(&shot.id, n);
                    let dur = if wav.exists() {
                        ffmpeg::probe(&wav).await.map(|p| p.duration_s()).unwrap_or_else(|_| estimate_speech_s(&line.line))
                    } else {
                        self.say(&line.line, &wav).await?
                    };
                    placements.push(Placement { wav, start_s: cursor });
                    cursor += dur + 0.20;
                }
                let mix = self.project.mix_wav(&shot.id);
                ffmpeg::mix(Some(&bed), &placements, shot.duration_s, &mix)
                    .await
                    .with_context(|| format!("mixing audio for shot {}", shot.id))?;
            }
        }
        Ok(())
    }

    async fn say(&self, text: &str, out: &std::path::Path) -> Result<f32> {
        match self.speech.say(text, out).await {
            Ok(d) => Ok(d),
            Err(e) => {
                tracing::warn!("TTS backend failed ({e}); placeholder tone");
                Speech::Placeholder.say(text, out).await
            }
        }
    }

    // === assembly =========================================================

    /// Mux each shot's video with its audio mix, concat in screenplay order into
    /// the final mp4, and write the EDL timeline manifest.
    pub async fn assemble(&self, bible: &Bible) -> Result<Timeline> {
        let mut av_clips: Vec<PathBuf> = Vec::new();
        let mut entries = Vec::new();
        let mut cursor = 0.0_f32;

        for (scene, shot) in bible.shots() {
            let clip = self.project.shot_clip(&shot.id);
            let mix = self.project.mix_wav(&shot.id);
            let av = self.project.root.join("shots").join(format!("{}.av.mp4", shot.id));
            ffmpeg::mux(&clip, &mix, &av).await.with_context(|| format!("muxing shot {}", shot.id))?;

            let mut dcursor = cursor + 0.30;
            let dialogue: Vec<TlLine> = shot
                .dialogue
                .iter()
                .map(|l| {
                    let at = dcursor;
                    dcursor += estimate_speech_s(&l.line) + 0.20;
                    TlLine { character: l.character_ref.clone(), line: l.line.clone(), start_s: at }
                })
                .collect();

            entries.push(TlEntry {
                shot: shot.id.clone(),
                scene: scene.id.clone(),
                start_s: cursor,
                duration_s: shot.duration_s,
                continuity: format!("{:?}", shot.continuity).to_lowercase(),
                clip: rel(&self.project, &clip),
                dialogue,
            });
            cursor += shot.duration_s;
            av_clips.push(av);
        }

        ffmpeg::concat(&av_clips, &self.project.film_path()).await.context("final concat")?;

        let timeline = Timeline {
            title: bible.title.clone(),
            total_duration_s: cursor,
            width: self.cfg().width,
            height: self.cfg().height,
            fps: self.cfg().fps,
            entries,
        };
        write_json(&self.project.timeline_path(), &timeline)?;
        Ok(timeline)
    }
}

/// A shot's position in the bible (scene index, shot index).
#[derive(Clone, Copy)]
pub struct ShotRef {
    pub scene: usize,
    pub shot: usize,
}

/// Partition all shots into continuity runs: each run begins at a `cut` shot and
/// includes the following `continue` shots. Runs are mutually independent.
pub fn continuity_runs(bible: &Bible) -> Vec<Vec<ShotRef>> {
    let mut runs: Vec<Vec<ShotRef>> = Vec::new();
    for (si, scene) in bible.scenes.iter().enumerate() {
        for (ki, shot) in scene.shots.iter().enumerate() {
            let r = ShotRef { scene: si, shot: ki };
            let start_new = runs.is_empty() || matches!(shot.continuity, Continuity::Cut) || ki == 0;
            if start_new {
                runs.push(vec![r]);
            } else {
                runs.last_mut().unwrap().push(r);
            }
        }
    }
    runs
}

fn rel(project: &Project, p: &std::path::Path) -> String {
    p.strip_prefix(&project.root).unwrap_or(p).to_string_lossy().into_owned()
}

// --- EDL timeline manifest (the durable assembly artifact) ----------------

#[derive(Debug, Serialize)]
pub struct Timeline {
    pub title: String,
    pub total_duration_s: f32,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    pub entries: Vec<TlEntry>,
}

#[derive(Debug, Serialize)]
pub struct TlEntry {
    pub shot: String,
    pub scene: String,
    pub start_s: f32,
    pub duration_s: f32,
    pub continuity: String,
    pub clip: String,
    pub dialogue: Vec<TlLine>,
}

#[derive(Debug, Serialize)]
pub struct TlLine {
    pub character: String,
    pub line: String,
    pub start_s: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bible::*;

    fn bible_with(continuities: &[&[Continuity]]) -> Bible {
        let scenes = continuities
            .iter()
            .enumerate()
            .map(|(si, conts)| Scene {
                id: format!("sc{}", si + 1),
                location_ref: "l1".into(),
                characters: vec!["c1".into()],
                synopsis: "s".into(),
                shots: conts
                    .iter()
                    .enumerate()
                    .map(|(ki, c)| Shot {
                        id: format!("sc{}_sh{}", si + 1, ki + 1),
                        scene_ref: format!("sc{}", si + 1),
                        duration_s: 3.0,
                        shot_type: "wide".into(),
                        action_prompt: "a".into(),
                        dialogue: vec![],
                        continuity: *c,
                    })
                    .collect(),
            })
            .collect();
        Bible {
            version: BIBLE_VERSION,
            title: "T".into(),
            logline: "l".into(),
            style: Style::default(),
            characters: vec![Character { id: "c1".into(), name: "C".into(), description: "d".into(), reference_image: None, voice_id: None }],
            locations: vec![Location { id: "l1".into(), description: "d".into(), reference_image: None }],
            scenes,
        }
    }

    #[test]
    fn runs_split_on_cut() {
        // cut, continue, continue | cut, continue  => 2 runs of sizes 3 and 2
        let b = bible_with(&[&[Continuity::Cut, Continuity::Continue, Continuity::Continue], &[Continuity::Cut, Continuity::Continue]]);
        let runs = continuity_runs(&b);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].len(), 3);
        assert_eq!(runs[1].len(), 2);
    }

    #[test]
    fn all_cuts_are_all_independent() {
        let b = bible_with(&[&[Continuity::Cut, Continuity::Cut, Continuity::Cut]]);
        let runs = continuity_runs(&b);
        assert_eq!(runs.len(), 3);
        assert!(runs.iter().all(|r| r.len() == 1));
    }

    #[test]
    fn scene_boundary_forces_new_run_even_if_marked_continue() {
        // scene 2 shot 1 marked continue must still start a fresh run (ki==0).
        let b = bible_with(&[&[Continuity::Cut], &[Continuity::Continue]]);
        let runs = continuity_runs(&b);
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn spec_hash_changes_with_prompt_and_is_stable() {
        let dir = std::env::temp_dir().join(format!("film_hash_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let proj = Project::create(&dir, "b".into(), Config::default()).unwrap();
        let pipe = Pipeline { project: proj, engine: Engine::new("http://127.0.0.1:1").unwrap(), images: Images::Procedural, video: Video::Placeholder, speech: Speech::Placeholder, music: Music::Silent };
        let mut b = bible_with(&[&[Continuity::Cut]]);
        let s = &b.scenes[0].clone();
        let sh = &b.scenes[0].shots[0].clone();
        let h1 = pipe.spec_hash(&b, s, sh, None, false);
        let h1b = pipe.spec_hash(&b, s, sh, None, false);
        assert_eq!(h1, h1b, "hash must be stable across calls");
        b.scenes[0].shots[0].action_prompt = "different".into();
        let s2 = &b.scenes[0].clone();
        let sh2 = &b.scenes[0].shots[0].clone();
        let h2 = pipe.spec_hash(&b, s2, sh2, None, false);
        assert_ne!(h1, h2, "hash must change when the prompt changes");
        std::fs::remove_dir_all(&dir).ok();
    }
}
