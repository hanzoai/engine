//! End-to-end test that loads a real model and runs one inference call.
//!
//! Gated behind the `real-model` feature so CI doesn't pull a multi-GB
//! model down by default. To run:
//!
//! ```bash
//! HANZO_TEST_MODEL_PATH=/path/to/gguf cargo test -p hanzo-engine \
//!     --features real-model --test mistral -- --ignored
//! ```
//!
//! Or pass an HF repo via `HANZO_TEST_MODEL_REPO=Qwen/Qwen3-4B`.

#![cfg(feature = "real-model")]

use std::sync::Arc;

use hanzo_engine::{infer, register_inference_engine, MistralEngine};

#[test]
#[ignore = "requires HANZO_TEST_MODEL_PATH or HANZO_TEST_MODEL_REPO and downloads a model"]
fn real_model_round_trips() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let engine = rt.block_on(async {
        if let Ok(path) = std::env::var("HANZO_TEST_MODEL_PATH") {
            MistralEngine::from_model_path(&path)
                .await
                .expect("load model from path")
        } else if let Ok(repo) = std::env::var("HANZO_TEST_MODEL_REPO") {
            MistralEngine::from_hf_repo(&repo)
                .await
                .expect("load model from HF repo")
        } else {
            panic!("set HANZO_TEST_MODEL_PATH or HANZO_TEST_MODEL_REPO to run this test");
        }
    });

    let id = *engine.model_id();
    register_inference_engine(Arc::new(engine)).expect("register");

    let out = infer(&id, b"Reply with the single word 'hi'.")
        .expect("inference completes");
    assert!(!out.is_empty(), "model returned empty completion");

    // Decode to UTF-8 so we get a useful failure message if the model
    // returned binary data.
    let s = std::str::from_utf8(&out).expect("output is utf-8");
    assert!(s.len() < 1024, "completion suspiciously long: {s:?}");
}
