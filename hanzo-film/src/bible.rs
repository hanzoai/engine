//! The Film Bible: the versioned JSON contract every stage reads and writes.
//!
//! Hierarchy mirrors production: bible -> scenes -> shots. Refs (`location_ref`,
//! `character_ref`, `scene_ref`) are string ids resolved against the top-level
//! `characters`/`locations` and the enclosing scene. `validate` is the single
//! gate that proves an instance is internally consistent before any render job runs.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Schema version. Bump only on a breaking shape change; readers reject mismatches.
pub const BIBLE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bible {
    #[serde(default = "default_version")]
    pub version: u32,
    pub title: String,
    pub logline: String,
    #[serde(default)]
    pub style: Style,
    #[serde(default)]
    pub characters: Vec<Character>,
    #[serde(default)]
    pub locations: Vec<Location>,
    #[serde(default)]
    pub scenes: Vec<Scene>,
}

fn default_version() -> u32 {
    BIBLE_VERSION
}

/// Global look. `prompt` is appended to every image/video prompt; `grade` is an
/// ffmpeg filter fragment (e.g. `eq=contrast=1.1:saturation=0.9`) applied to shots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Style {
    #[serde(default)]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub id: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub id: String,
    pub location_ref: String,
    #[serde(default)]
    pub characters: Vec<String>,
    #[serde(default)]
    pub synopsis: String,
    #[serde(default)]
    pub shots: Vec<Shot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shot {
    pub id: String,
    pub scene_ref: String,
    pub duration_s: f32,
    #[serde(default)]
    pub shot_type: String,
    pub action_prompt: String,
    #[serde(default)]
    pub dialogue: Vec<Line>,
    #[serde(default)]
    pub continuity: Continuity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub character_ref: String,
    pub line: String,
}

/// A `continue` shot conditions on the prior shot's tail frame (same camera run);
/// a `cut` starts fresh from its scene/location anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Continuity {
    #[default]
    Cut,
    Continue,
}

impl Bible {
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("bible serialization")
    }

    pub fn character(&self, id: &str) -> Option<&Character> {
        self.characters.iter().find(|c| c.id == id)
    }

    pub fn location(&self, id: &str) -> Option<&Location> {
        self.locations.iter().find(|l| l.id == id)
    }

    /// All shots in screenplay order, each paired with its scene.
    pub fn shots(&self) -> impl Iterator<Item = (&Scene, &Shot)> {
        self.scenes
            .iter()
            .flat_map(|sc| sc.shots.iter().map(move |sh| (sc, sh)))
    }

    pub fn total_duration_s(&self) -> f32 {
        self.scenes
            .iter()
            .flat_map(|s| &s.shots)
            .map(|s| s.duration_s)
            .sum()
    }

    /// The single consistency gate. Every id unique, every ref resolvable, durations positive.
    pub fn validate(&self) -> Result<()> {
        if self.version != BIBLE_VERSION {
            bail!(
                "bible version {} != supported {}",
                self.version,
                BIBLE_VERSION
            );
        }
        if self.title.trim().is_empty() {
            bail!("bible.title is empty");
        }

        let mut char_ids = HashSet::new();
        for c in &self.characters {
            if c.id.trim().is_empty() {
                bail!("character with empty id");
            }
            if !char_ids.insert(c.id.as_str()) {
                bail!("duplicate character id {}", c.id);
            }
        }
        let mut loc_ids = HashSet::new();
        for l in &self.locations {
            if l.id.trim().is_empty() {
                bail!("location with empty id");
            }
            if !loc_ids.insert(l.id.as_str()) {
                bail!("duplicate location id {}", l.id);
            }
        }

        if self.scenes.is_empty() {
            bail!("bible has no scenes");
        }

        let mut scene_ids = HashSet::new();
        let mut shot_ids = HashSet::new();
        for scene in &self.scenes {
            if !scene_ids.insert(scene.id.as_str()) {
                bail!("duplicate scene id {}", scene.id);
            }
            if !loc_ids.contains(scene.location_ref.as_str()) {
                bail!("scene {} references unknown location {}", scene.id, scene.location_ref);
            }
            for cr in &scene.characters {
                if !char_ids.contains(cr.as_str()) {
                    bail!("scene {} references unknown character {}", scene.id, cr);
                }
            }
            if scene.shots.is_empty() {
                bail!("scene {} has no shots", scene.id);
            }
            for shot in &scene.shots {
                if !shot_ids.insert(shot.id.as_str()) {
                    bail!("duplicate shot id {}", shot.id);
                }
                if shot.scene_ref != scene.id {
                    bail!(
                        "shot {} scene_ref {} != enclosing scene {}",
                        shot.id,
                        shot.scene_ref,
                        scene.id
                    );
                }
                if !(shot.duration_s > 0.0) {
                    bail!("shot {} has non-positive duration {}", shot.id, shot.duration_s);
                }
                if shot.action_prompt.trim().is_empty() {
                    bail!("shot {} has empty action_prompt", shot.id);
                }
                for line in &shot.dialogue {
                    if !char_ids.contains(line.character_ref.as_str()) {
                        bail!("shot {} dialogue references unknown character {}", shot.id, line.character_ref);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> Bible {
        Bible {
            version: BIBLE_VERSION,
            title: "T".into(),
            logline: "L".into(),
            style: Style::default(),
            characters: vec![Character {
                id: "hero".into(),
                name: "Hero".into(),
                description: "d".into(),
                reference_image: None,
                voice_id: None,
            }],
            locations: vec![Location {
                id: "room".into(),
                description: "d".into(),
                reference_image: None,
            }],
            scenes: vec![Scene {
                id: "s1".into(),
                location_ref: "room".into(),
                characters: vec!["hero".into()],
                synopsis: "syn".into(),
                shots: vec![Shot {
                    id: "s1_1".into(),
                    scene_ref: "s1".into(),
                    duration_s: 4.0,
                    shot_type: "wide".into(),
                    action_prompt: "hero enters".into(),
                    dialogue: vec![Line {
                        character_ref: "hero".into(),
                        line: "hello".into(),
                    }],
                    continuity: Continuity::Cut,
                }],
            }],
        }
    }

    #[test]
    fn valid_bible_passes() {
        minimal().validate().unwrap();
    }

    #[test]
    fn roundtrip_json() {
        let b = minimal();
        let s = b.to_json();
        let b2 = Bible::from_json(&s).unwrap();
        assert_eq!(b2.title, b.title);
        b2.validate().unwrap();
    }

    #[test]
    fn rejects_bad_location_ref() {
        let mut b = minimal();
        b.scenes[0].location_ref = "nope".into();
        assert!(b.validate().is_err());
    }

    #[test]
    fn rejects_bad_dialogue_ref() {
        let mut b = minimal();
        b.scenes[0].shots[0].dialogue[0].character_ref = "ghost".into();
        assert!(b.validate().is_err());
    }

    #[test]
    fn rejects_scene_ref_mismatch() {
        let mut b = minimal();
        b.scenes[0].shots[0].scene_ref = "wrong".into();
        assert!(b.validate().is_err());
    }

    #[test]
    fn rejects_duplicate_shot_id() {
        let mut b = minimal();
        let dup = b.scenes[0].shots[0].clone();
        b.scenes[0].shots.push(dup);
        assert!(b.validate().is_err());
    }

    #[test]
    fn rejects_nonpositive_duration() {
        let mut b = minimal();
        b.scenes[0].shots[0].duration_s = 0.0;
        assert!(b.validate().is_err());
    }

    #[test]
    fn continuity_defaults_to_cut() {
        let sh: Shot = serde_json::from_str(
            r#"{"id":"x","scene_ref":"s1","duration_s":2.0,"action_prompt":"a"}"#,
        )
        .unwrap();
        assert_eq!(sh.continuity, Continuity::Cut);
        assert!(sh.dialogue.is_empty());
    }

    #[test]
    fn total_duration_sums_shots() {
        let mut b = minimal();
        b.scenes[0].shots[0].duration_s = 3.5;
        assert!((b.total_duration_s() - 3.5).abs() < 1e-6);
    }
}
