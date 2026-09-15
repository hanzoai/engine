//! Durable operator edits for a configured pool. Addresses and model aliases are
//! deployment-owned; the UI can only retune existing workers' scheduling knobs.
use crate::replica::{Balancer, BalancerConfig};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerUpdate {
    pub model: String,
    pub id: String,
    pub capacity: usize,
    pub weight: u32,
    pub roles: Vec<String>,
}

impl WorkerUpdate {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.model.is_empty()
            || self.id.is_empty()
            || self.model.len() > 512
            || self.id.len() > 512
        {
            return Err("model and worker ID are required (maximum 512 bytes)");
        }
        if !(1..=4096).contains(&self.capacity) || !(1..=1_000_000).contains(&self.weight) {
            return Err("capacity must be 1–4096; weight must be 1–1000000");
        }
        if self.roles.len() > 16
            || self.roles.iter().any(|r| {
                r.is_empty()
                    || r.len() > 64
                    || !r
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            })
        {
            return Err("roles must be up to 16 nonempty names using letters, digits, '_' or '-'");
        }
        Ok(())
    }
}

pub(crate) struct PoolFile {
    path: PathBuf,
    state: Mutex<(BalancerConfig, Vec<u8>)>,
}

impl PoolFile {
    pub fn new(path: PathBuf) -> anyhow::Result<Self> {
        let path = path.canonicalize()?;
        let bytes = std::fs::read(&path)?;
        let cfg = serde_yaml::from_slice(&bytes)?;
        Ok(Self {
            path,
            state: Mutex::new((cfg, bytes)),
        })
    }

    pub async fn update(&self, balancer: &Balancer, update: WorkerUpdate) -> Result<(), String> {
        update.validate().map_err(str::to_owned)?;
        let mut state = self.state.lock().await;
        let mut next = state.0.clone();
        let worker = next
            .models
            .get_mut(&update.model)
            .and_then(|pool| {
                pool.iter_mut()
                    .find(|r| if r.id.is_empty() { &r.url } else { &r.id } == &update.id)
            })
            .ok_or("worker is not in the configured model pool")?;
        // Dynamic registration is deliberately not a durable replacement for configuration.
        if !balancer
            .statuses()
            .get(&update.model)
            .is_some_and(|pool| pool.iter().any(|r| r.id == update.id))
        {
            return Err("configured worker is not in the running pool".into());
        }
        worker.capacity = update.capacity;
        worker.weight = update.weight;
        worker.roles = update.roles.clone();
        let bytes = serde_yaml::to_string(&next)
            .map_err(|_| "cannot encode pool configuration")?
            .into_bytes();
        let (path, previous, candidate) = (self.path.clone(), state.1.clone(), bytes.clone());
        tokio::task::spawn_blocking(move || replace_file(&path, &previous, &candidate))
            .await
            .map_err(|_| "pool write task failed")?
            .map_err(|_| {
                "pool file changed externally or could not be saved; no runtime settings changed"
            })?;
        // No removal API exists; the exact worker checked above still exists. The
        // disk commit precedes activation, so a crash/restart restores the change.
        let applied = balancer.update_worker(
            &update.model,
            &update.id,
            update.capacity,
            update.weight,
            update.roles,
        );
        debug_assert!(applied);
        *state = (next, bytes);
        Ok(())
    }
}

fn replace_file(path: &Path, previous: &[u8], candidate: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path)? != previous {
        return Err(std::io::Error::other("configuration changed externally"));
    }
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp = path.with_extension(format!("tmp-{}-{suffix}", std::process::id()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        file.set_permissions(std::fs::metadata(path)?.permissions())?;
        file.write_all(candidate)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replica::RoutingHints;
    use std::collections::HashSet;
    use std::time::Instant;

    #[tokio::test]
    async fn edits_survive_restart_preserve_pins_and_refuse_file_drift() {
        let path = std::env::temp_dir().join(format!(
            "hanzo-pool-{}-{}.yaml",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let yaml = "models:\n  zen-coder:\n    - id: spark\n      url: http://spark.local:30000\n      capacity: 4\n      weight: 300\n      roles: [main]\n      upstream_model: local-model\n      max_context: 1000000\n";
        std::fs::write(&path, yaml).unwrap();
        let balancer = Balancer::from_config(serde_yaml::from_str(yaml).unwrap());
        let pool = PoolFile::new(path.clone()).unwrap();
        let set = balancer.set_for(Some("zen-coder")).unwrap();
        let hints = RoutingHints {
            session: Some("acme/s1".into()),
            ..Default::default()
        };
        let lease = set
            .pick_agent(&hints, &HashSet::new(), Instant::now())
            .unwrap();
        let update = WorkerUpdate {
            model: "zen-coder".into(),
            id: "spark".into(),
            capacity: 8,
            weight: 400,
            roles: vec!["main".into(), "subagent".into()],
        };
        pool.update(&balancer, update.clone()).await.unwrap();
        assert_eq!(lease.inflight(), 1);
        assert_eq!(set.statuses()[0].capacity, 8);
        assert_eq!(set.statuses()[0].roles, update.roles);
        assert_eq!(
            set.pick_agent(&hints, &HashSet::new(), Instant::now())
                .unwrap()
                .id(),
            "spark"
        );
        let restored =
            Balancer::from_config(serde_yaml::from_slice(&std::fs::read(&path).unwrap()).unwrap());
        assert_eq!(restored.statuses()["zen-coder"][0].weight, 400);
        assert_eq!(restored.statuses()["zen-coder"][0].max_context, 1_000_000);
        assert_eq!(
            restored.statuses()["zen-coder"][0]
                .upstream_model
                .as_deref(),
            Some("local-model")
        );
        std::fs::write(&path, "models: {}\n").unwrap();
        let mut changed = update.clone();
        changed.capacity = 1;
        assert!(pool.update(&balancer, changed).await.is_err());
        assert_eq!(set.statuses()[0].capacity, 8);
        let mut invalid = update;
        invalid.weight = 0;
        assert!(pool.update(&balancer, invalid).await.is_err());
        std::fs::remove_file(path).unwrap();
    }
}
