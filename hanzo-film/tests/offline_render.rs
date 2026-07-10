//! End-to-end pipeline over the dependency-free backends (no engine): a hand-built
//! bible -> procedural assets -> placeholder shots -> tone dialogue + silent bed ->
//! assembled mp4. Asserts the final file really has a video and an audio stream of
//! the expected duration. This is the engine-free acceptance gate.

use hanzo_film::backends::{Images, Music, Speech, Video};
use hanzo_film::bible::*;
use hanzo_film::engine::Engine;
use hanzo_film::ffmpeg;
use hanzo_film::pipeline::Pipeline;
use hanzo_film::project::{Config, Project};

fn tiny_bible() -> Bible {
    let mk_shot = |scene: &str, k: usize, dur: f32, cont: Continuity, dlg: Vec<Line>| Shot {
        id: format!("{scene}_sh{k}"),
        scene_ref: scene.to_string(),
        duration_s: dur,
        shot_type: "wide".into(),
        action_prompt: format!("shot {k} action in {scene}"),
        dialogue: dlg,
        continuity: cont,
    };
    Bible {
        version: BIBLE_VERSION,
        title: "Offline Proof".into(),
        logline: "a tiny film to prove the pipeline".into(),
        style: Style {
            prompt: "cinematic, warm light".into(),
            lora: None,
            grade: Some("eq=contrast=1.05".into()),
        },
        characters: vec![
            Character {
                id: "c1".into(),
                name: "Ada".into(),
                description: "an engineer in a red coat".into(),
                reference_image: None,
                voice_id: Some("v1".into()),
            },
            Character {
                id: "c2".into(),
                name: "Boro".into(),
                description: "a pilot in grey".into(),
                reference_image: None,
                voice_id: Some("v2".into()),
            },
        ],
        locations: vec![Location {
            id: "l1".into(),
            description: "a ship bridge".into(),
            reference_image: None,
        }],
        scenes: vec![
            Scene {
                id: "sc1".into(),
                location_ref: "l1".into(),
                characters: vec!["c1".into(), "c2".into()],
                synopsis: "Ada and Boro meet".into(),
                shots: vec![
                    mk_shot(
                        "sc1",
                        1,
                        1.2,
                        Continuity::Cut,
                        vec![Line {
                            character_ref: "c1".into(),
                            line: "We have a signal".into(),
                        }],
                    ),
                    mk_shot("sc1", 2, 1.0, Continuity::Continue, vec![]),
                ],
            },
            Scene {
                id: "sc2".into(),
                location_ref: "l1".into(),
                characters: vec!["c1".into(), "c2".into()],
                synopsis: "they decide".into(),
                shots: vec![mk_shot(
                    "sc2",
                    1,
                    1.0,
                    Continuity::Cut,
                    vec![Line {
                        character_ref: "c2".into(),
                        line: "Then we go".into(),
                    }],
                )],
            },
        ],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn offline_pipeline_produces_av_mp4() {
    let dir = std::env::temp_dir().join(format!("film_offline_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let cfg = Config {
        width: 256,
        height: 144,
        fps: 12,
        concurrency: 3,
        ..Default::default()
    };
    let project = Project::create(&dir, "brief".into(), cfg).unwrap();

    let mut bible = tiny_bible();
    bible.validate().unwrap();
    project.save_bible(&bible).unwrap();

    // Force the engine-free backends; the URL is never dialed.
    let pipe = Pipeline {
        project: project.clone(),
        engine: Engine::new("http://127.0.0.1:9").unwrap(),
        images: Images::Procedural,
        video: Video::Placeholder,
        speech: Speech::Placeholder,
        music: Music::Silent,
    };

    pipe.assets(&mut bible).await.unwrap();
    assert!(
        project.character_ref("c1").exists(),
        "character ref generated"
    );
    assert!(
        project.location_ref("l1").exists(),
        "location ref generated"
    );

    let rendered = pipe.render(&bible).await.unwrap();
    assert_eq!(rendered, 3, "all 3 shots rendered");
    for id in ["sc1_sh1", "sc1_sh2", "sc2_sh1"] {
        assert!(project.shot_clip(id).exists(), "clip {id} exists");
        assert!(project.shot_sidecar(id).exists(), "sidecar {id} exists");
    }

    // Idempotent re-run: nothing re-renders.
    let again = pipe.render(&bible).await.unwrap();
    assert_eq!(again, 0, "second render skips everything (idempotent)");

    pipe.audio(&bible).await.unwrap();
    let tl = pipe.assemble(&bible).await.unwrap();
    assert_eq!(tl.entries.len(), 3);
    assert!(
        (tl.total_duration_s - 3.2).abs() < 0.05,
        "timeline duration = 1.2+1.0+1.0"
    );

    let film = project.film_path();
    assert!(film.exists(), "film.mp4 exists");
    let probe = ffmpeg::probe(&film).await.unwrap();
    let has_video = probe.streams.iter().any(|s| s.codec_type == "video");
    let has_audio = probe.streams.iter().any(|s| s.codec_type == "audio");
    assert!(has_video, "final mp4 has a video stream");
    assert!(has_audio, "final mp4 has an audio stream");
    assert!(
        (probe.duration_s() - 3.2).abs() < 0.4,
        "final mp4 duration ~= 3.2s, got {}",
        probe.duration_s()
    );

    assert!(project.timeline_path().exists(), "timeline.json written");
    std::fs::remove_dir_all(&dir).ok();
}
