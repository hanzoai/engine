//! Registry behaviour tests for the canonical inference / embedding API.
//!
//! These tests run with no model loaded — they exercise only the registry
//! semantics. The real-model end-to-end test lives in `mistral.rs`.
//!
//! Note: each integration-test file in `tests/` compiles to its own binary,
//! so the global `OnceLock` registry in this binary is independent of the
//! one in `tests/mistral.rs`. All registry assertions can run together.

use std::sync::Arc;

use hanzo_engine::{
    embed, embedding_engine_registered, infer, inference_engine_registered,
    register_embedding_engine, register_inference_engine, EmbeddingEngine, EngineError,
    InferenceEngine,
};

struct MockInference;
impl InferenceEngine for MockInference {
    fn infer(&self, _model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
        // Round-trip the prompt with a prefix so the test can assert dispatch
        // actually reached us.
        let mut out = b"mock:".to_vec();
        out.extend_from_slice(prompt);
        Ok(out)
    }
}

struct MockEmbedding;
impl EmbeddingEngine for MockEmbedding {
    fn embed(&self, dim: usize, _text: &[u8]) -> Result<Vec<f32>, EngineError> {
        // Deterministic dummy vector; tests just check length + dispatch.
        Ok((0..dim).map(|i| i as f32).collect())
    }
}

/// Calling `infer` before registration returns `NoInferenceEngine`. Then
/// after registration a `MockInference` is dispatched and round-trips the
/// prompt.
#[test]
fn infer_dispatch_round_trips() {
    assert!(!inference_engine_registered());
    let id = [0u8; 32];

    let err = infer(&id, b"hello").expect_err("no engine registered yet");
    assert!(matches!(err, EngineError::NoInferenceEngine));

    register_inference_engine(Arc::new(MockInference)).expect("first register succeeds");
    assert!(inference_engine_registered());

    let out = infer(&id, b"hello").expect("dispatch ok");
    assert_eq!(out, b"mock:hello");

    // Second registration must fail — exactly one engine per process.
    let err = register_inference_engine(Arc::new(MockInference))
        .expect_err("second register must fail");
    assert!(matches!(err, EngineError::Other(_)));
}

/// Same as above but for the embedding registry.
#[test]
fn embed_dispatch_round_trips() {
    assert!(!embedding_engine_registered());

    let err = embed(8, b"hello").expect_err("no engine registered yet");
    assert!(matches!(err, EngineError::NoEmbeddingEngine));

    register_embedding_engine(Arc::new(MockEmbedding)).expect("first register succeeds");
    assert!(embedding_engine_registered());

    let v = embed(8, b"hello").expect("dispatch ok");
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[7], 7.0);

    let err = register_embedding_engine(Arc::new(MockEmbedding))
        .expect_err("second register must fail");
    assert!(matches!(err, EngineError::Other(_)));
}
