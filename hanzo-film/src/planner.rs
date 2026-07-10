//! Brief -> Bible. Two structured LLM stages over `/v1/chat/completions`:
//!   A. meta   : title, logline, style, characters, locations, scene outlines
//!   B. shots  : per scene, the shot list (parallel calls)
//! The model supplies the *creative* content; `normalize` then assigns
//! deterministic ids and resolves every ref, so the output always validates even
//! when a small model is sloppy. Characters are global and reused across scenes,
//! which is what makes a character recur.

use crate::bible::*;
use crate::engine::Engine;
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

/// What the model returns for stage A (decoupled from the Bible contract).
#[derive(Debug, Deserialize)]
struct Meta {
    #[serde(default)]
    title: String,
    #[serde(default)]
    logline: String,
    #[serde(default)]
    style_prompt: String,
    #[serde(default)]
    characters: Vec<MetaChar>,
    #[serde(default)]
    locations: Vec<MetaLoc>,
    #[serde(default)]
    scenes: Vec<MetaScene>,
}

#[derive(Debug, Deserialize)]
struct MetaChar {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct MetaLoc {
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct MetaScene {
    #[serde(default)]
    synopsis: String,
    #[serde(default)]
    character_indices: Vec<usize>,
    #[serde(default)]
    location_index: usize,
}

#[derive(Debug, Deserialize)]
struct ShotList {
    #[serde(default)]
    shots: Vec<MetaShot>,
}

#[derive(Debug, Deserialize)]
struct MetaShot {
    #[serde(default)]
    shot_type: String,
    #[serde(default)]
    action_prompt: String,
    #[serde(default)]
    duration_s: f32,
    #[serde(default)]
    continuity: String,
    #[serde(default, alias = "dialogue_lines")]
    dialogue: Vec<MetaLine>,
}

#[derive(Debug, Deserialize)]
struct MetaLine {
    #[serde(default)]
    character_index: usize,
    #[serde(default)]
    line: String,
}

pub struct Planner<'a> {
    pub engine: &'a Engine,
    pub model: String,
    pub scenes: usize,
    pub shots_per_scene: usize,
}

impl<'a> Planner<'a> {
    /// Run both stages and return a validated Bible.
    pub async fn plan(&self, brief: &str) -> Result<Bible> {
        let meta = self
            .stage_meta(brief)
            .await
            .context("planner stage A (meta)")?;
        // Stage B in parallel across scenes.
        let names: Vec<String> = meta.characters.iter().map(|c| c.name.clone()).collect();
        let futs = meta.scenes.iter().enumerate().map(|(i, sc)| {
            let names = names.clone();
            async move { self.stage_shots(brief, i, sc, &names).await }
        });
        let shot_lists = futures::future::try_join_all(futs)
            .await
            .context("planner stage B (shots)")?;
        Ok(normalize(brief, meta, shot_lists))
    }

    async fn stage_meta(&self, brief: &str) -> Result<Meta> {
        // maxItems keeps the constrained decoder from running past the token budget:
        // the array is forced closed after N items, so the object always completes.
        let schema = json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "logline": {"type": "string"},
                "style_prompt": {"type": "string"},
                "characters": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "description": {"type": "string"}
                }, "required": ["name", "description"]}},
                "locations": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "object", "properties": {
                    "description": {"type": "string"}
                }, "required": ["description"]}},
                "scenes": {"type": "array", "minItems": self.scenes, "maxItems": self.scenes, "items": {"type": "object", "properties": {
                    "synopsis": {"type": "string"},
                    "character_indices": {"type": "array", "maxItems": 3, "items": {"type": "integer"}},
                    "location_index": {"type": "integer"}
                }, "required": ["synopsis", "character_indices", "location_index"]}}
            },
            "required": ["title", "logline", "characters", "locations", "scenes"]
        });
        let system = "You are a film development assistant. Turn a brief into a compact production bible. \
            Reuse the SAME characters across scenes. Be concise. Output strictly the requested JSON.";
        let user = format!(
            "Brief:\n{brief}\n\nProduce: a title; a one-sentence logline; a short 'style_prompt' visual style fragment; \
             2-3 main characters (name + one short visual-description sentence); {nloc} locations (one short visual sentence each); \
             and exactly {nsc} scenes. Each scene: a one-sentence synopsis, the character_indices (0-based, into the characters array) \
             present, and a location_index (0-based, into locations). Keep the same characters recurring across scenes. Keep every field terse.",
            nloc = 2usize,
            nsc = self.scenes,
        );
        let v = self
            .engine
            .chat_json(&self.model, system, &user, schema, 1200)
            .await?;
        Ok(serde_json::from_value(v)?)
    }

    async fn stage_shots(
        &self,
        brief: &str,
        idx: usize,
        scene: &MetaScene,
        names: &[String],
    ) -> Result<ShotList> {
        let schema = json!({
            "type": "object",
            "properties": {
                "shots": {"type": "array", "minItems": self.shots_per_scene, "maxItems": self.shots_per_scene, "items": {"type": "object", "properties": {
                    "shot_type": {"type": "string"},
                    "action_prompt": {"type": "string"},
                    "duration_s": {"type": "number"},
                    "continuity": {"type": "string", "enum": ["cut", "continue"]},
                    "dialogue": {"type": "array", "maxItems": 2, "items": {"type": "object", "properties": {
                        "character_index": {"type": "integer"},
                        "line": {"type": "string"}
                    }, "required": ["character_index", "line"]}}
                }, "required": ["shot_type", "action_prompt", "duration_s", "continuity", "dialogue"]}}
            },
            "required": ["shots"]
        });
        let roster = names
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{i}={n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let system = "You are a director breaking a scene into shots. Be concise. Output strictly the requested JSON.";
        let user = format!(
            "Film brief: {brief}\n\nScene {n} synopsis: {syn}\nCharacter roster (index=name): {roster}\n\n\
             Break this scene into exactly {k} shots. For each shot give: shot_type (wide/medium/close/etc), \
             action_prompt (ONE vivid visual sentence, under 15 words), duration_s (2-6), continuity ('cut' for a new setup, \
             'continue' to flow from the previous shot), and 0-2 dialogue lines (character_index into the roster + a short spoken line).",
            n = idx + 1,
            syn = scene.synopsis,
            roster = roster,
            k = self.shots_per_scene,
        );
        let v = self
            .engine
            .chat_json(&self.model, system, &user, schema, 1200)
            .await?;
        Ok(serde_json::from_value(v).unwrap_or(ShotList { shots: vec![] }))
    }
}

/// Map loose model output onto the strict Bible contract with deterministic ids.
fn normalize(brief: &str, meta: Meta, shot_lists: Vec<ShotList>) -> Bible {
    // Characters (guarantee >= 1).
    let mut characters: Vec<Character> = meta
        .characters
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.name.trim().is_empty())
        .enumerate()
        .map(|(id, (orig_i, c))| Character {
            id: format!("c{}", id + 1),
            name: c.name.trim().to_string(),
            description: nonempty(&c.description, "a character"),
            reference_image: None,
            voice_id: Some(format!("v{}", (orig_i % 4) + 1)),
        })
        .collect();
    if characters.is_empty() {
        characters.push(Character {
            id: "c1".into(),
            name: "Narrator".into(),
            description: "the narrator".into(),
            reference_image: None,
            voice_id: Some("v1".into()),
        });
    }
    // Original-index -> character id, for resolving scene/dialogue references.
    let char_id_by_orig: Vec<String> = {
        // Rebuild the same filter to recover original index alignment.
        let mut ids = Vec::new();
        let mut next = 1;
        for c in &meta.characters {
            if c.name.trim().is_empty() {
                ids.push(String::new());
            } else {
                ids.push(format!("c{next}"));
                next += 1;
            }
        }
        if ids.is_empty() {
            ids.push("c1".into());
        }
        ids
    };
    let resolve_char = |i: usize| -> Option<String> {
        char_id_by_orig
            .get(i)
            .filter(|s| !s.is_empty())
            .cloned()
            .or_else(|| characters.first().map(|c| c.id.clone()))
    };

    // Locations (guarantee >= 1).
    let mut locations: Vec<Location> = meta
        .locations
        .iter()
        .enumerate()
        .map(|(i, l)| Location {
            id: format!("l{}", i + 1),
            description: nonempty(&l.description, "a location"),
            reference_image: None,
        })
        .collect();
    if locations.is_empty() {
        locations.push(Location {
            id: "l1".into(),
            description: "a location".into(),
            reference_image: None,
        });
    }
    let loc_id = |i: usize| -> String {
        locations
            .get(i.min(locations.len().saturating_sub(1)))
            .map(|l| l.id.clone())
            .unwrap_or_else(|| locations[0].id.clone())
    };

    // Scenes + shots.
    let mut scenes = Vec::new();
    for (si, (ms, sl)) in meta.scenes.iter().zip(shot_lists.iter()).enumerate() {
        let scene_id = format!("sc{}", si + 1);
        let mut scene_chars: Vec<String> = ms
            .character_indices
            .iter()
            .filter_map(|&i| resolve_char(i))
            .collect();
        scene_chars.sort();
        scene_chars.dedup();
        if scene_chars.is_empty() {
            scene_chars = characters.iter().map(|c| c.id.clone()).collect();
        }

        let mut shots = Vec::new();
        for (ki, msh) in sl.shots.iter().enumerate() {
            if msh.action_prompt.trim().is_empty() {
                continue;
            }
            let dialogue: Vec<Line> = msh
                .dialogue
                .iter()
                .filter(|d| !d.line.trim().is_empty())
                .filter_map(|d| {
                    resolve_char(d.character_index).map(|character_ref| Line {
                        character_ref,
                        line: d.line.trim().to_string(),
                    })
                })
                .collect();
            let continuity = if ki == 0 || msh.continuity.trim() != "continue" {
                Continuity::Cut
            } else {
                Continuity::Continue
            };
            shots.push(Shot {
                id: format!("{scene_id}_sh{}", ki + 1),
                scene_ref: scene_id.clone(),
                duration_s: msh.duration_s.clamp(1.5, 8.0),
                shot_type: nonempty(&msh.shot_type, "medium"),
                action_prompt: msh.action_prompt.trim().to_string(),
                dialogue,
                continuity,
            });
        }
        // Floor: every scene needs at least one shot.
        if shots.is_empty() {
            shots.push(Shot {
                id: format!("{scene_id}_sh1"),
                scene_ref: scene_id.clone(),
                duration_s: 4.0,
                shot_type: "wide".into(),
                action_prompt: nonempty(&ms.synopsis, "establishing shot"),
                dialogue: vec![],
                continuity: Continuity::Cut,
            });
        }

        scenes.push(Scene {
            id: scene_id,
            location_ref: loc_id(ms.location_index),
            characters: scene_chars,
            synopsis: nonempty(&ms.synopsis, "a scene"),
            shots,
        });
    }
    // Floor: at least one scene.
    if scenes.is_empty() {
        scenes.push(Scene {
            id: "sc1".into(),
            location_ref: locations[0].id.clone(),
            characters: characters.iter().map(|c| c.id.clone()).collect(),
            synopsis: "opening scene".into(),
            shots: vec![Shot {
                id: "sc1_sh1".into(),
                scene_ref: "sc1".into(),
                duration_s: 4.0,
                shot_type: "wide".into(),
                action_prompt: "establishing shot".into(),
                dialogue: vec![],
                continuity: Continuity::Cut,
            }],
        });
    }

    Bible {
        version: BIBLE_VERSION,
        title: nonempty(&meta.title, "Untitled"),
        logline: nonempty(
            &meta.logline,
            brief.chars().take(120).collect::<String>().trim(),
        ),
        style: Style {
            prompt: nonempty(&meta.style_prompt, "cinematic, natural light"),
            lora: None,
            grade: None,
        },
        characters,
        locations,
        scenes,
    }
}

fn nonempty(s: &str, fallback: &str) -> String {
    let t = s.trim();
    if t.is_empty() {
        fallback.to_string()
    } else {
        t.to_string()
    }
}

/// Assemble the full prompt for a shot's image/video (style + location + characters + action).
pub fn shot_prompt(bible: &Bible, scene: &Scene, shot: &Shot) -> String {
    let mut parts = vec![bible.style.prompt.clone()];
    if let Some(loc) = bible.location(&scene.location_ref) {
        parts.push(loc.description.clone());
    }
    for cr in &scene.characters {
        if let Some(c) = bible.character(cr) {
            parts.push(format!("{}: {}", c.name, c.description));
        }
    }
    parts.push(shot.action_prompt.clone());
    parts.retain(|p| !p.trim().is_empty());
    parts.join(". ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_produces_valid_bible() {
        let meta = Meta {
            title: "The Signal".into(),
            logline: "".into(),
            style_prompt: "noir".into(),
            characters: vec![
                MetaChar {
                    name: "Ada".into(),
                    description: "engineer".into(),
                },
                MetaChar {
                    name: "".into(),
                    description: "".into(),
                }, // dropped
                MetaChar {
                    name: "Boro".into(),
                    description: "pilot".into(),
                },
            ],
            locations: vec![MetaLoc {
                description: "bridge".into(),
            }],
            scenes: vec![
                MetaScene {
                    synopsis: "they meet".into(),
                    character_indices: vec![0, 2],
                    location_index: 0,
                },
                MetaScene {
                    synopsis: "they argue".into(),
                    character_indices: vec![99],
                    location_index: 5,
                },
            ],
        };
        let shots = vec![
            ShotList {
                shots: vec![
                    MetaShot {
                        shot_type: "wide".into(),
                        action_prompt: "ada enters".into(),
                        duration_s: 4.0,
                        continuity: "cut".into(),
                        dialogue: vec![MetaLine {
                            character_index: 0,
                            line: "hi".into(),
                        }],
                    },
                    MetaShot {
                        shot_type: "close".into(),
                        action_prompt: "boro turns".into(),
                        duration_s: 20.0,
                        continuity: "continue".into(),
                        dialogue: vec![MetaLine {
                            character_index: 2,
                            line: "you".into(),
                        }],
                    },
                ],
            },
            ShotList { shots: vec![] }, // forces the floor
        ];
        let b = normalize("a brief about a signal", meta, shots);
        b.validate().expect("normalized bible must validate");
        assert_eq!(b.characters.len(), 2); // empty-name dropped
        assert_eq!(b.scenes.len(), 2);
        // scene 2 had a bogus location index -> clamped to a real one
        assert!(b.locations.iter().any(|l| l.id == b.scenes[1].location_ref));
        // duration clamped
        assert!(b.scenes[0].shots[1].duration_s <= 8.0);
        // dialogue char refs resolved to real ids
        assert!(b
            .character(&b.scenes[0].shots[0].dialogue[0].character_ref)
            .is_some());
        // recurring character: c1 appears in scene 1 roster
        assert!(b.scenes[0].characters.iter().any(|c| c == "c1"));
    }

    #[test]
    fn first_shot_is_always_cut() {
        let meta = Meta {
            title: "X".into(),
            logline: "l".into(),
            style_prompt: "s".into(),
            characters: vec![MetaChar {
                name: "A".into(),
                description: "d".into(),
            }],
            locations: vec![MetaLoc {
                description: "here".into(),
            }],
            scenes: vec![MetaScene {
                synopsis: "s".into(),
                character_indices: vec![0],
                location_index: 0,
            }],
        };
        let shots = vec![ShotList {
            shots: vec![MetaShot {
                shot_type: "".into(),
                action_prompt: "a".into(),
                duration_s: 3.0,
                continuity: "continue".into(),
                dialogue: vec![],
            }],
        }];
        let b = normalize("brief", meta, shots);
        assert_eq!(b.scenes[0].shots[0].continuity, Continuity::Cut);
    }
}
