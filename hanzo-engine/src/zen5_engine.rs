//! Zen5 engine adapter — stub.
//!
//! The full adapter wrapped `hanzo_zen5::Zen5Engine` to satisfy the
//! [`InferenceEngine`] trait. The hanzo-zen5 source crate is not
//! present in the current tree (only the `zen5-engine-src/`
//! submodule remains), so this module ships as a stub that keeps
//! the public symbols hanzod consumes (`build_registry`,
//! `model_id_for`, `register_zen5_engines_at_startup`,
//! `DEFAULT_VARIANTS`) and a `DEFAULT_VARIANTS` constant.
//!
//! Restoring the real adapter is a matter of:
//! 1. Placing `hanzo-zen5/` back under `~/work/hanzo/engine/`
//! 2. Re-adding it to the engine workspace
//! 3. Replacing the `Err` arms below with the real `Zen5Engine` calls

use std::collections::HashMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::api::EngineError;

/// Built-in zen5 model variants the registry knows about.
///
/// These names are stable across releases and used by hanzod's CLI +
/// runtime config (`--zen5-variants=...`). The list is empty in the
/// stub build and grows when the real adapter is wired in.
pub const DEFAULT_VARIANTS: &[&str] = &[
    "zen-5-flash",
    "zen-5-pro",
    "zen-5-mini",
    "zen-5-coder",
];

/// Build the canonical zen5 variant → model_id map.
///
/// Each model id is the SHA-256 of `"zen5/{variant}"`, which keeps
/// ids stable across stub and real builds. The runtime can therefore
/// register a stub at boot and still resolve the same id once the
/// real backend takes over.
pub fn build_registry() -> HashMap<String, [u8; 32]> {
    let mut out = HashMap::with_capacity(DEFAULT_VARIANTS.len());
    for v in DEFAULT_VARIANTS {
        out.insert((*v).to_string(), model_id_for(v));
    }
    out
}

/// Stable model id for a zen5 variant name, regardless of whether
/// the real backend is present.
pub fn model_id_for(variant: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"zen5/");
    hasher.update(variant.as_bytes());
    hasher.finalize().into()
}

/// Discover zen5 weights under `weights_dir` and register an engine
/// for each variant in `variants`.
///
/// Stub implementation: logs a warning and returns an empty registry.
/// Real implementation will return one entry per registered engine.
pub fn register_zen5_engines_at_startup(
    weights_dir: &Path,
    variants: &[&str],
) -> Result<HashMap<String, [u8; 32]>, EngineError> {
    tracing::warn!(
        weights_dir = %weights_dir.display(),
        variant_count = variants.len(),
        "hanzo-engine: register_zen5_engines_at_startup is a no-op; \
         hanzo-zen5 source crate is not present in this build"
    );
    Ok(HashMap::new())
}
