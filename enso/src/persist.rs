//! safetensors load/flush of the offline-learned base policy `W` (the `DK`-length
//! `f64` vector at `Policy::w`). Phase 4d's `build_dataset.py --ledger` ridge fit
//! emits this; the engine loads it at startup so a refresh takes effect without a
//! recompile. One tensor, one concern.
//!
//! Per-user bandit state and the eval `ProfileTable` are deliberately not
//! persisted here: they only become meaningful once the observe/ingest pipelines
//! (Phase 4b/c) exist, so encoding them now would build functions no caller uses.
//! They attach at this same seam when those pipelines land.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::learner::Learner;
use crate::policy::DK;
use crate::Policy;

/// Tensor name for the base `W` inside the safetensors file.
const W_NAME: &str = "w";

/// Load a base `W` from a safetensors file and wrap it in a `Learner` centered
/// there (gamma/alpha match the cold-start bandit). A missing file is not an
/// error: the caller falls back to `Policy::zeros()` cold start.
pub fn load_learner(path: &Path, gamma: f64, alpha: f64) -> Result<Learner> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let w = read_f64(&st, W_NAME, DK)?;
    Ok(Learner::new(Policy::from_weights(w), gamma, alpha))
}

/// Flush a base `W` to a safetensors file. One `F64` tensor, length `DK`.
pub fn save_learner(path: &Path, learner: &Learner) -> Result<()> {
    let w = learner.base.w.clone();
    write_w(path, &w)
}

fn write_w(path: &Path, w: &[f64]) -> Result<()> {
    let data = bytemuck::cast_slice::<f64, u8>(w).to_vec();
    let view = TensorView::new(Dtype::F64, vec![w.len()], &data)?;
    safetensors::tensor::serialize_to_file(
        [(W_NAME, view)],
        Some(info()),
        path,
    )
    .map_err(|e| anyhow!(e))?;
    Ok(())
}

fn read_f64(st: &SafeTensors, name: &str, expect_len: usize) -> Result<Vec<f64>> {
    let tv = st.tensor(name)?;
    if tv.dtype() != Dtype::F64 {
        bail!("tensor {name}: expected F64, got {:?}", tv.dtype());
    }
    let bytes = tv.data();
    if bytes.len() != expect_len * std::mem::size_of::<f64>() {
        bail!(
            "tensor {name}: expected {expect_len} f64 ({} B), got {} B",
            expect_len * std::mem::size_of::<f64>(),
            bytes.len()
        );
    }
    let w: Vec<f64> = bytemuck::cast_slice::<u8, f64>(bytes).to_vec();
    Ok(w)
}

fn info() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("format".into(), "enso-base".into());
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Policy;

    #[test]
    fn w_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.safetensors");
        let mut w = vec![0.0f64; DK];
        for i in 0..DK {
            w[i] = (i as f64) * 0.5 - 3.0;
        }
        let learner = Learner::new(Policy::from_weights(w.clone()), 1.0, 0.5);
        save_learner(&path, &learner).unwrap();
        let back = load_learner(&path, 1.0, 0.5).unwrap();
        assert_eq!(back.base.w, w);
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let path = Path::new("/no/such/heads-base.safetensors");
        assert!(load_learner(path, 1.0, 0.5).is_err());
    }
}
