//! safetensors persistence of a scope's policy: the offline-learned base `W` (the
//! `DK`-length `f64` vector at `Policy::w`) plus the arms it was fit against
//! (`ProfileTable`) and — for a live flush — the per-user online bandits.
//!
//! ONE format across every scope (global `heads-base.safetensors`, per-org
//! `heads-<org>.safetensors`, and the live `state-<scope>.safetensors` the engine
//! flushes): a single `w` tensor carrying the base matrix, and the arms + bandits in
//! the safetensors `__metadata__` (JSON strings). A file with no `profiles` metadata
//! (a legacy W-only fit) loads with an empty table; a file with no `bandits` loads
//! cold online state. So `save` is the single writer for the fit job and the periodic
//! flush alike, and `load` reconstructs whatever a given file carries.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::learner::{BanditState, Learner};
use crate::policy::DK;
use crate::profile::ProfileTable;
use crate::Policy;

/// Tensor name for the base `W` inside the safetensors file.
const W_NAME: &str = "w";
/// Metadata keys for the non-tensor payload (arms + online bandits).
const META_FORMAT: &str = "format";
const META_PROFILES: &str = "profiles";
const META_BANDITS: &str = "bandits";
const FORMAT_TAG: &str = "enso-base-v2";

/// Load a scope artifact: the base `W` (wrapped in a `Learner` centered there,
/// gamma/alpha matching the cold-start bandit and any persisted per-user bandits
/// restored) and the arms `ProfileTable`. A missing file is an error; the caller
/// falls back to `Policy::zeros()` cold start. Absent `profiles`/`bandits` metadata
/// load as empty (forward/backward compatible with a W-only fit).
pub fn load(path: &Path, gamma: f64, alpha: f64) -> Result<(Learner, ProfileTable)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let w = read_f64(&st, W_NAME, DK)?;
    let mut learner = Learner::new(Policy::from_weights(w), gamma, alpha);

    let meta = metadata(&bytes)?;
    let table: ProfileTable = match meta.get(META_PROFILES) {
        Some(j) => serde_json::from_str(j).context("decode profiles metadata")?,
        None => ProfileTable::default(),
    };
    if let Some(j) = meta.get(META_BANDITS) {
        let states: Vec<(String, BanditState)> =
            serde_json::from_str(j).context("decode bandits metadata")?;
        learner.restore(&states);
    }
    Ok((learner, table))
}

/// Backward-compatible loader used where only the base `W` is wanted (kept so the
/// pre-v2 `load_learner` call sites don't need touching). Drops the table.
pub fn load_learner(path: &Path, gamma: f64, alpha: f64) -> Result<Learner> {
    Ok(load(path, gamma, alpha)?.0)
}

/// Persist a scope's policy: the base `W` tensor + the arms + any per-user online
/// bandits, all in the one format. The fit job calls this with a fresh learner (no
/// bandits → base file); the periodic flush calls it with the live learner (bandits
/// included → state file). One writer, one format, every scope.
pub fn save(path: &Path, learner: &Learner, table: &ProfileTable) -> Result<()> {
    let w = learner.base.w.clone();
    let data = bytemuck::cast_slice::<f64, u8>(&w).to_vec();
    let view = TensorView::new(Dtype::F64, vec![w.len()], &data)?;

    let mut meta = HashMap::new();
    meta.insert(META_FORMAT.to_string(), FORMAT_TAG.to_string());
    meta.insert(
        META_PROFILES.to_string(),
        serde_json::to_string(table).context("encode profiles metadata")?,
    );
    let states = learner.user_states();
    if !states.is_empty() {
        meta.insert(
            META_BANDITS.to_string(),
            serde_json::to_string(&states).context("encode bandits metadata")?,
        );
    }

    safetensors::tensor::serialize_to_file([(W_NAME, view)], Some(meta), path)
        .map_err(|e| anyhow!(e))?;
    Ok(())
}

/// Convenience for a base-only fit (no online bandits) — identical to `save` with a
/// learner that has no observations.
pub fn save_base(path: &Path, learner: &Learner, table: &ProfileTable) -> Result<()> {
    save(path, learner, table)
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
    Ok(bytemuck::cast_slice::<u8, f64>(bytes).to_vec())
}

/// Read the safetensors `__metadata__` map without pulling in a full deserialize of
/// the tensors (the header carries it). Empty when the file has none.
fn metadata(bytes: &[u8]) -> Result<HashMap<String, String>> {
    let (_, meta) = SafeTensors::read_metadata(bytes)?;
    Ok(meta.metadata().clone().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::DK;
    use crate::profile::{Profile, ProfileTable};
    use hanzo_router::registry::{Level, Modality};

    fn sample_table() -> ProfileTable {
        let mut q = [0.0; crate::featurize::NUM_TASKS];
        q[hanzo_router::registry::Task::General.index()] = 0.9;
        ProfileTable {
            profiles: vec![Profile {
                model: "gpt-5.5".into(),
                level: Level::Max,
                modality: Modality::Text,
                quality: q,
                latency_ms: 1000.0,
                cost: 3.0,
                vram_gb: 0.0,
                max_context: 200_000,
                samples: 10,
            }],
        }
    }

    #[test]
    fn w_and_table_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heads-base.safetensors");
        let mut w = vec![0.0f64; DK];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f64) * 0.5 - 3.0;
        }
        let learner = Learner::new(Policy::from_weights(w.clone()), 1.0, 0.5);
        let table = sample_table();
        save_base(&path, &learner, &table).unwrap();

        let (back, back_table) = load(&path, 1.0, 0.5).unwrap();
        assert_eq!(back.base.w, w);
        assert_eq!(back_table.profiles.len(), 1);
        assert_eq!(back_table.profiles[0].model, "gpt-5.5");
    }

    #[test]
    fn bandits_roundtrip_for_online_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state-base.safetensors");
        let mut learner = Learner::new(Policy::zeros(), 1.0, 0.5);
        let table = sample_table();
        // Observe once for a user so a bandit exists to persist.
        let x = vec![0.1; crate::policy::D];
        let p = table.profiles[0].quality_features();
        learner.observe("org/alice", &x, &p, 1.0);
        assert_eq!(learner.user_count(), 1);
        save(&path, &learner, &table).unwrap();

        let (restored, _) = load(&path, 1.0, 0.5).unwrap();
        assert_eq!(restored.user_count(), 1, "the online bandit must survive a save/load");
        // The restored theta matches the live one (online adaptation is not lost across
        // a restart) — to within the JSON round-trip's last-ULP tolerance.
        let orig = learner.effective_w("org/alice").to_vec();
        let got = restored.effective_w("org/alice");
        assert_eq!(got.len(), orig.len());
        for (a, b) in got.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-9, "theta drifted on reload: {a} vs {b}");
        }
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let path = Path::new("/no/such/heads-base.safetensors");
        assert!(load(path, 1.0, 0.5).is_err());
    }
}
