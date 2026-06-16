//! Live Metal integration test for the `MistralEngine` facade.
//!
//! Proves the node -> engine -> model -> Metal path: loads a real model via
//! `MistralEngine::from_model_path`, registers it in the global registry, then
//! generates real text through the registered `InferenceEngine` (exactly what
//! precompile 0x0201 does at runtime).
//!
//! Gated on the `metal` feature and on a model being present. Set
//! `HANZO_TEST_MODEL_PATH` to override (defaults to the zen-nano checkout; the
//! facade auto-selects the bundled GGUF inside the dir). Run with:
//!
//!   cargo test -p hanzo-engine --features metal --test mistral_engine_metal -- --nocapture

#![cfg(feature = "metal")]

use std::sync::Arc;
use std::time::Instant;

use hanzo_engine::{inference_engine, register_inference_engine, InferenceEngine, MistralEngine};

const DEFAULT_MODEL_PATH: &str = "/Users/a/work/zen/models/zen-nano-0.6b";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zen_nano_generates_on_metal() {
    let path =
        std::env::var("HANZO_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());

    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: model path does not exist: {path}");
        return;
    }

    println!("loading MistralEngine from {path} on Metal...");
    let load_start = Instant::now();
    let engine = MistralEngine::from_model_path(&path)
        .await
        .expect("failed to load model via MistralEngine::from_model_path");
    println!("model loaded in {:.2?}", load_start.elapsed());

    // Register under the inference surface, then fetch it back the way the
    // precompile does, so we exercise the real global registry path.
    register_inference_engine(Arc::new(engine) as Arc<dyn InferenceEngine>)
        .expect("registration failed");
    let registered = inference_engine().expect("no inference engine registered");

    let prompt = "What is the Lux Network? Answer in one sentence.";
    println!("\nprompt: {prompt}");

    let gen_start = Instant::now();
    let output = registered
        .generate(prompt)
        .await
        .expect("generation failed");
    let elapsed = gen_start.elapsed();

    println!("\n--- generated completion ---\n{output}\n----------------------------");
    println!("generated in {:.2?}", elapsed);

    assert!(
        !output.trim().is_empty(),
        "engine produced empty output - inference did not really run"
    );
}
