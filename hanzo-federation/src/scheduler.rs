//! Capacity-aware scheduler.
//!
//! Two kinds of work to assign:
//!   1. Data shards — capacity-weighted (sqrt(mem) × sqrt(tflops)).
//!   2. Expert pins — best-fit-decreasing on memory (zen5 MoDE pipeline).
//!
//! Both produce an [`Assignment`] value; nothing else is mutated.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::topology::{Lab, Node};

/// Plan for one federation round. Pure value, safe to send over the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    /// node name → fraction of data (sums to 1.0).
    pub data_weights: HashMap<String, f64>,
    /// node name → list of expert IDs hosted there.
    pub expert_pins: HashMap<String, Vec<String>>,
    /// node name → {"max_batch": N, "max_seq": N}.
    pub role_caps: HashMap<String, HashMap<String, u64>>,
}

#[derive(Debug)]
pub struct Scheduler<'a> {
    lab: &'a Lab,
}

impl<'a> Scheduler<'a> {
    pub fn new(lab: &'a Lab) -> Self {
        Self { lab }
    }

    /// Weight each worker's data slice by `capacity_score()`.
    pub fn shard_data(&self) -> HashMap<String, f64> {
        let scores: HashMap<String, f64> = self
            .lab
            .workers()
            .map(|n| (n.name.clone(), n.capacity_score()))
            .collect();
        let total: f64 = scores.values().sum();
        let total = if total == 0.0 { 1.0 } else { total };
        scores
            .into_iter()
            .map(|(k, s)| (k, s / total))
            .collect()
    }

    /// Best-fit-decreasing expert pinning. Hard-respects pre-declared pins.
    pub fn pin_experts(
        &self,
        experts: &[(String, u32)],
    ) -> Result<HashMap<String, Vec<String>>> {
        let sizes: HashMap<&str, u32> = experts.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let mut pinned: HashMap<String, Vec<String>> = HashMap::new();
        let mut remaining: HashMap<String, i64> = HashMap::new();

        for n in self.lab.workers() {
            let declared: u32 = n
                .pin_experts
                .iter()
                .map(|e| sizes.get(e.as_str()).copied().unwrap_or(0))
                .sum();
            if declared > n.memory_gb {
                return Err(anyhow!(
                    "node {} declared pins total {}GB but only has {}GB — \
                     split the model into smaller slabs or move some pins to a larger node",
                    n.name,
                    declared,
                    n.memory_gb
                ));
            }
            pinned.insert(n.name.clone(), n.pin_experts.clone());
            remaining.insert(n.name.clone(), (n.memory_gb as i64) - (declared as i64));
        }

        // Largest experts first.
        let mut by_size: Vec<&(String, u32)> = experts.iter().collect();
        by_size.sort_by(|a, b| b.1.cmp(&a.1));

        for (expert_id, gb) in by_size {
            if pinned.values().any(|v| v.iter().any(|e| e == expert_id)) {
                continue;
            }
            // Pick node with most remaining headroom that can fit it.
            let best = remaining
                .iter()
                .filter(|(_, mem)| **mem >= *gb as i64)
                .max_by_key(|(_, mem)| **mem)
                .map(|(k, _)| k.clone());
            let Some(best) = best else {
                return Err(anyhow!(
                    "expert {expert_id} ({gb}GB) does not fit on any node"
                ));
            };
            pinned.get_mut(&best).unwrap().push(expert_id.clone());
            *remaining.get_mut(&best).unwrap() -= *gb as i64;
        }
        Ok(pinned)
    }

    pub fn plan(&self, experts: &[(String, u32)]) -> Result<Assignment> {
        let data_weights = self.shard_data();
        let expert_pins = self.pin_experts(experts)?;
        let role_caps: HashMap<String, HashMap<String, u64>> = self
            .lab
            .workers()
            .map(|n| (n.name.clone(), caps_for(n)))
            .collect();
        Ok(Assignment {
            data_weights,
            expert_pins,
            role_caps,
        })
    }
}

/// Per-node training caps. Same formula as `_caps_for` in scheduler.py.
fn caps_for(n: &Node) -> HashMap<String, u64> {
    let headroom_bytes = (n.memory_gb as u64) * (1u64 << 30) * 20 / 100;
    let tok_budget = headroom_bytes / (8 * 1024); // bf16 ≈ 8KB/tok at h=4096
    let mut m = HashMap::new();
    if tok_budget < 4096 {
        m.insert("max_batch".into(), 1);
        m.insert("max_seq".into(), tok_budget.max(2048));
    } else if tok_budget < 65_536 {
        m.insert("max_batch".into(), 1);
        m.insert("max_seq".into(), tok_budget);
    } else {
        m.insert("max_batch".into(), tok_budget / 32768);
        m.insert("max_seq".into(), 32_768);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{Lab, Node, NodeRole};

    fn lab2() -> Lab {
        Lab {
            nodes: vec![
                Node {
                    name: "a".into(),
                    host: "a".into(),
                    role: NodeRole::Hybrid,
                    backend_hint: "cuda".into(),
                    memory_gb: 128,
                    nic_gbps: 200,
                    tflops_hint: 31,
                    pin_experts: vec![],
                    pin_modalities: vec![],
                    auth_token: None,
                },
                Node {
                    name: "b".into(),
                    host: "b".into(),
                    role: NodeRole::Worker,
                    backend_hint: "mlx".into(),
                    memory_gb: 64,
                    nic_gbps: 10,
                    tflops_hint: 21,
                    pin_experts: vec![],
                    pin_modalities: vec![],
                    auth_token: None,
                },
            ],
            job_dir: ".zen-fed".into(),
            sync_interval_steps: 8,
            aggregation: "byzantine_robust".into(),
        }
    }

    #[test]
    fn shards_sum_to_one() {
        let lab = lab2();
        let s = Scheduler::new(&lab);
        let w = s.shard_data();
        let total: f64 = w.values().sum();
        assert!((total - 1.0).abs() < 1e-9, "got {total}");
    }

    #[test]
    fn larger_node_gets_more_data() {
        let lab = lab2();
        let s = Scheduler::new(&lab);
        let w = s.shard_data();
        assert!(w["a"] > w["b"]);
    }

    #[test]
    fn pin_experts_best_fit() {
        let lab = lab2();
        let s = Scheduler::new(&lab);
        let pins = s
            .pin_experts(&[("big".into(), 100), ("small".into(), 10)])
            .unwrap();
        assert!(pins["a"].iter().any(|e| e == "big"));
    }

    #[test]
    fn pin_experts_rejects_oversize() {
        let lab = lab2();
        let s = Scheduler::new(&lab);
        let err = s
            .pin_experts(&[("huge".into(), 1000)])
            .unwrap_err();
        assert!(err.to_string().contains("does not fit"));
    }
}
