//! Lab topology — declared inventory of every box that can train.
//!
//! A [`Node`] is a *value*: hostname, role, declared capacity. Probed runtime
//! capability is reported separately and never mixed in at this layer.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// What a node does in the federation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeRole {
    /// Runs DeltaSoup, hosts global router state.
    Coordinator,
    /// Trains and pushes deltas.
    Worker,
    /// Single-box dev mode — does both.
    Hybrid,
}

/// Declared properties of a lab node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Short id: "spark", "m4max", "m1", "strix".
    pub name: String,
    /// mDNS or IP — "spark.lan".
    pub host: String,
    pub role: NodeRole,
    /// "mlx" | "cuda" | "rocm" — what we expect the box to run.
    #[serde(rename = "backend", alias = "backend_hint")]
    pub backend_hint: String,
    pub memory_gb: u32,
    /// Peak NIC throughput (200 for ConnectX-7, 10 for GbE).
    #[serde(default = "default_nic")]
    pub nic_gbps: u32,
    /// Rough BF16 TFLOPS for the scheduler.
    #[serde(default = "default_tflops", rename = "tflops", alias = "tflops_hint")]
    pub tflops_hint: u32,

    /// zen5 expert IDs this node should host.
    #[serde(default)]
    pub pin_experts: Vec<String>,
    /// "text" | "vision" | "audio" | "video" | "3d".
    #[serde(default)]
    pub pin_modalities: Vec<String>,

    /// HMAC secret for transport. Skipped when None (dev mode).
    /// `null` after env expansion also yields None.
    #[serde(default, deserialize_with = "de_optional_null_string")]
    pub auth_token: Option<String>,
}

fn default_nic() -> u32 {
    10
}
fn default_tflops() -> u32 {
    10
}

/// Treat the literal string "null" (produced by `_expand_env` for unset vars)
/// as Python does: as if the field were absent.
fn de_optional_null_string<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(d)?;
    Ok(opt.filter(|s| s != "null" && !s.is_empty()))
}

impl Node {
    /// Geometric mean of memory and TFLOPS.
    ///
    /// Both bind in real training (memory → no OOM, TFLOPS → throughput);
    /// the geometric mean prevents one fast box getting all the work and
    /// OOM'ing on the bigger shards.
    pub fn capacity_score(&self) -> f64 {
        (self.memory_gb as f64).sqrt() * (self.tflops_hint as f64).sqrt()
    }

    pub fn is_coordinator(&self) -> bool {
        matches!(self.role, NodeRole::Coordinator | NodeRole::Hybrid)
    }

    pub fn is_worker(&self) -> bool {
        matches!(self.role, NodeRole::Worker | NodeRole::Hybrid)
    }
}

/// Top-level lab declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lab {
    pub nodes: Vec<Node>,
    #[serde(default = "default_job_dir")]
    pub job_dir: String,
    /// Push delta every N local steps.
    #[serde(default = "default_sync_interval")]
    pub sync_interval_steps: u32,
    /// DeltaSoup method: "byzantine_robust", "median", "mean".
    #[serde(default = "default_aggregation")]
    pub aggregation: String,
}

fn default_job_dir() -> String {
    ".zen-fed".into()
}
fn default_sync_interval() -> u32 {
    8
}
fn default_aggregation() -> String {
    "byzantine_robust".into()
}

impl Lab {
    pub fn coordinator(&self) -> Result<&Node> {
        self.nodes
            .iter()
            .find(|n| n.is_coordinator())
            .ok_or_else(|| anyhow!("lab has no coordinator"))
    }

    pub fn workers(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|n| n.is_worker())
    }

    pub fn find(&self, name: &str) -> Result<&Node> {
        self.nodes
            .iter()
            .find(|n| n.name == name)
            .ok_or_else(|| anyhow!("no node named {name:?} in lab"))
    }

    /// Build the map of worker name → HMAC secret, dropping workers without
    /// one (dev mode).
    pub fn secrets(&self) -> std::collections::HashMap<String, String> {
        self.workers()
            .filter_map(|n| n.auth_token.clone().map(|t| (n.name.clone(), t)))
            .collect()
    }

    /// Read YAML from disk, expanding `${VAR}` from the process environment.
    /// Unset variables expand to the literal string `null`, matching
    /// `_expand_env` in topology.py.
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read lab.yaml at {}", path.as_ref().display()))?;
        Self::from_yaml_str(&raw)
    }

    pub fn from_yaml_str(raw: &str) -> Result<Self> {
        let expanded = expand_env(raw);
        serde_yaml::from_str(&expanded).context("parse lab yaml")
    }
}

/// `${VAR}` → env value, or "null" if unset. Same behaviour as
/// `_expand_env` in the Python implementation.
fn expand_env(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'$' && bytes[i + 1] == b'{' {
            if let Some(end) = bytes[i + 2..].iter().position(|&b| b == b'}') {
                let var = &text[i + 2..i + 2 + end];
                // Only treat as a variable if name is all word chars (\w in regex).
                let valid = !var.is_empty()
                    && var.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
                if valid {
                    let val = std::env::var(var).unwrap_or_else(|_| "null".to_string());
                    let val = if val.is_empty() { "null".into() } else { val };
                    out.push_str(&val);
                    i += 2 + end + 1;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_score_is_geometric_mean() {
        let n = Node {
            name: "x".into(),
            host: "x".into(),
            role: NodeRole::Worker,
            backend_hint: "cpu".into(),
            memory_gb: 64,
            nic_gbps: 10,
            tflops_hint: 16,
            pin_experts: vec![],
            pin_modalities: vec![],
            auth_token: None,
        };
        // sqrt(64) * sqrt(16) = 8 * 4 = 32
        assert!((n.capacity_score() - 32.0).abs() < 1e-9);
    }

    #[test]
    fn env_expansion_unset_to_null() {
        std::env::remove_var("HANZO_FED_TEST_UNSET");
        assert_eq!(expand_env("${HANZO_FED_TEST_UNSET}"), "null");
    }

    #[test]
    fn env_expansion_set_passes_through() {
        std::env::set_var("HANZO_FED_TEST_SET", "hello");
        assert_eq!(expand_env("x${HANZO_FED_TEST_SET}y"), "xhelloy");
    }

    #[test]
    fn lab_yaml_minimal() {
        let yaml = r#"
nodes:
  - name: spark
    host: spark.local
    role: hybrid
    backend: cuda
    memory_gb: 128
    nic_gbps: 200
    tflops: 31
  - name: m1
    host: m1.local
    role: worker
    backend: mlx
    memory_gb: 64
"#;
        let lab = Lab::from_yaml_str(yaml).unwrap();
        assert_eq!(lab.nodes.len(), 2);
        assert_eq!(lab.coordinator().unwrap().name, "spark");
        assert_eq!(lab.workers().count(), 2); // hybrid counts as worker too
    }
}
