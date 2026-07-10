//! `hanzo-film` CLI. Each subcommand is one resumable pipeline stage; `run` chains
//! them. Orchestration state is the project directory — nothing else is needed.

use anyhow::Result;
use clap::{Parser, Subcommand};
use hanzo_film::coherence::{self, Scorer};
use hanzo_film::engine::Engine;
use hanzo_film::pipeline::Pipeline;
use hanzo_film::planner::Planner;
use hanzo_film::project::{Config, Project, VideoBackend};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hanzo-film", about = "Long-form film orchestration over the Hanzo Engine.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a new project directory from a brief.
    New(NewArgs),
    /// brief -> validated bible.json (LLM planner).
    Plan(PlanArgs),
    /// Generate reference images for characters and locations.
    Assets(DirArg),
    /// Render all shots (parallel, idempotent, continuity-aware).
    Render(DirArg),
    /// Synthesize dialogue TTS + score and mix per shot.
    Audio(DirArg),
    /// Concat + mux into the final mp4 and write the timeline.
    Assemble(DirArg),
    /// Score identity coherence across shot pairs sharing a character.
    Verify(VerifyArgs),
    /// plan (if needed) -> assets -> render -> audio -> assemble -> verify.
    Run(PlanArgs),
}

#[derive(Parser)]
struct NewArgs {
    dir: PathBuf,
    #[arg(long)]
    brief: String,
    #[arg(long, default_value = "http://127.0.0.1:1234")]
    engine: String,
    #[arg(long, default_value = "default")]
    planner_model: String,
    #[arg(long, default_value = "default")]
    image_model: String,
    #[arg(long, default_value = "default")]
    tts_model: String,
    #[arg(long, default_value = "default")]
    music_model: String,
    #[arg(long, value_enum, default_value = "placeholder")]
    video: VideoArg,
    #[arg(long, default_value_t = 768)]
    width: usize,
    #[arg(long, default_value_t = 432)]
    height: usize,
    #[arg(long, default_value_t = 24)]
    fps: usize,
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum VideoArg {
    Placeholder,
    Wan,
}

#[derive(Parser)]
struct PlanArgs {
    dir: PathBuf,
    #[arg(long, default_value_t = 2)]
    scenes: usize,
    #[arg(long, default_value_t = 3)]
    shots_per_scene: usize,
}

#[derive(Parser)]
struct DirArg {
    dir: PathBuf,
}

#[derive(Parser)]
struct VerifyArgs {
    dir: PathBuf,
    #[arg(long, default_value_t = 16)]
    max_pairs: usize,
    /// Use a vision-embedding model via /v1/embeddings instead of the pixel proxy.
    #[arg(long)]
    embed_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("hanzo_film=info".parse().unwrap()))
        .with_target(false)
        .init();

    match Cli::parse().cmd {
        Cmd::New(a) => cmd_new(a),
        Cmd::Plan(a) => cmd_plan(a).await,
        Cmd::Assets(a) => cmd_assets(a.dir).await,
        Cmd::Render(a) => cmd_render(a.dir).await,
        Cmd::Audio(a) => cmd_audio(a.dir).await,
        Cmd::Assemble(a) => cmd_assemble(a.dir).await,
        Cmd::Verify(a) => cmd_verify(a).await,
        Cmd::Run(a) => cmd_run(a).await,
    }
}

fn cmd_new(a: NewArgs) -> Result<()> {
    let config = Config {
        engine_url: a.engine,
        planner_model: a.planner_model,
        image_model: a.image_model,
        tts_model: a.tts_model,
        music_model: a.music_model,
        video: match a.video {
            VideoArg::Placeholder => VideoBackend::Placeholder,
            VideoArg::Wan => VideoBackend::Wan,
        },
        width: a.width,
        height: a.height,
        fps: a.fps,
        concurrency: a.concurrency,
    };
    let p = Project::create(&a.dir, a.brief, config)?;
    println!("created project at {}", p.root.display());
    Ok(())
}

async fn plan_into(project: &Project, scenes: usize, shots_per_scene: usize) -> Result<()> {
    let cfg = project.config();
    let engine = Engine::new(&cfg.engine_url)?;
    if !engine.ready().await {
        anyhow::bail!("engine not reachable at {} — start it with `hanzo serve`", cfg.engine_url);
    }
    let planner = Planner { engine: &engine, model: cfg.planner_model.clone(), scenes, shots_per_scene };
    let bible = planner.plan(&project.manifest.brief).await?;
    project.save_bible(&bible)?;
    println!(
        "planned '{}': {} scenes, {} shots, {} characters, {:.0}s",
        bible.title,
        bible.scenes.len(),
        bible.shots().count(),
        bible.characters.len(),
        bible.total_duration_s()
    );
    Ok(())
}

async fn cmd_plan(a: PlanArgs) -> Result<()> {
    let project = Project::open(&a.dir)?;
    plan_into(&project, a.scenes, a.shots_per_scene).await
}

async fn cmd_assets(dir: PathBuf) -> Result<()> {
    let project = Project::open(&dir)?;
    let mut bible = project.load_bible()?;
    let pipe = Pipeline::new(project)?;
    pipe.assets(&mut bible).await?;
    println!("assets: {} characters, {} locations", bible.characters.len(), bible.locations.len());
    Ok(())
}

async fn cmd_render(dir: PathBuf) -> Result<()> {
    let project = Project::open(&dir)?;
    let bible = project.load_bible()?;
    let pipe = Pipeline::new(project)?;
    let n = pipe.render(&bible).await?;
    println!("render: {} shots (re)rendered of {}", n, bible.shots().count());
    Ok(())
}

async fn cmd_audio(dir: PathBuf) -> Result<()> {
    let project = Project::open(&dir)?;
    let bible = project.load_bible()?;
    let pipe = Pipeline::new(project)?;
    pipe.audio(&bible).await?;
    println!("audio: dialogue + score mixed for {} shots", bible.shots().count());
    Ok(())
}

async fn cmd_assemble(dir: PathBuf) -> Result<()> {
    let project = Project::open(&dir)?;
    let bible = project.load_bible()?;
    let film = project.film_path();
    let pipe = Pipeline::new(project)?;
    let tl = pipe.assemble(&bible).await?;
    println!("assembled {} ({:.1}s, {} shots) -> {}", tl.title, tl.total_duration_s, tl.entries.len(), film.display());
    Ok(())
}

async fn cmd_verify(a: VerifyArgs) -> Result<()> {
    let project = Project::open(&a.dir)?;
    let bible = project.load_bible()?;
    let scorer = match a.embed_model {
        Some(model) => Scorer::Embedding { engine: Engine::new(&project.config().engine_url)?, model },
        None => Scorer::Pixel,
    };
    let report = coherence::verify(&project, &bible, scorer, a.max_pairs).await?;
    hanzo_film::project::write_json(&project.coherence_path(), &report)?;
    println!("coherence [{}]: mean {:.3} over {} pairs — {}", report.method, report.mean, report.pairs.len(), report.note);
    Ok(())
}

async fn cmd_run(a: PlanArgs) -> Result<()> {
    let project = Project::open(&a.dir)?;
    if !project.bible_path().exists() {
        plan_into(&project, a.scenes, a.shots_per_scene).await?;
    } else {
        println!("bible.json exists — resuming (delete it to re-plan)");
    }
    let mut bible = project.load_bible()?;
    let pipe = Pipeline::new(project.clone())?;
    pipe.assets(&mut bible).await?;
    let n = pipe.render(&bible).await?;
    println!("render: {n} shots (re)rendered");
    pipe.audio(&bible).await?;
    let tl = pipe.assemble(&bible).await?;
    let report = coherence::verify(&project, &bible, Scorer::Pixel, 16).await?;
    hanzo_film::project::write_json(&project.coherence_path(), &report)?;
    println!(
        "DONE '{}': {:.1}s, {} shots -> {}\ncoherence [{}] mean {:.3}",
        tl.title,
        tl.total_duration_s,
        tl.entries.len(),
        project.film_path().display(),
        report.method,
        report.mean
    );
    Ok(())
}
