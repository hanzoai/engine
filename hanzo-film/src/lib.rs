//! hanzo-film: long-form film/episode orchestration over the Hanzo Engine `/v1`
//! endpoints. A one-paragraph brief becomes a validated Film Bible, thousands of
//! independent per-shot render jobs, dialogue + score, and a final assembled mp4 —
//! with all orchestration state as plain JSON files in a project directory.
//!
//! Stages (each resumable, each a CLI subcommand):
//!   plan     brief -> bible.json                (LLM, structured output)
//!   assets   reference images per entity        (image endpoint / procedural)
//!   render   bible -> per-shot clips            (parallel, idempotent, continuity)
//!   audio    dialogue TTS + score + per-shot mix
//!   assemble concat + mux -> film.mp4 + timeline.json
//!   verify   identity-coherence scoring hook

pub mod backends;
pub mod bible;
pub mod coherence;
pub mod engine;
pub mod ffmpeg;
pub mod pipeline;
pub mod planner;
pub mod project;
