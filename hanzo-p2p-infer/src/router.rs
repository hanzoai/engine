// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Network-aware Mixture-of-Experts (MoE) routing and placement analysis.
//!
//! Compares **locality-aware expert placement** (co-locating co-activated experts inside
//! each pipeline stage's high-bandwidth local memory domain) against **naive cross-WAN
//! per-token expert all-to-all dispatch** (`2 * num_layers` WAN hops per token).

use crate::model::HybridModelSpec;
use crate::partition::NetworkTier;
use crate::transport::ActivationCodec;

/// MoE expert placement strategy across peer-to-peer nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPlacementPolicy {
    /// Experts for each layer live on the same peer / local LAN domain as the layer's
    /// attention/GDN trunk: zero cross-WAN expert hops per layer.
    LocalityAwareStageResident,
    /// Naive distributed expert parallelism across WAN peers: every token's top-K experts
    /// trigger an all-to-all dispatch + gather round-trip across the network per layer.
    NaiveCrossNetworkAllToAll,
}

impl ExpertPlacementPolicy {
    /// Canonical string identifier for telemetry and benchmarks.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LocalityAwareStageResident => "locality_aware_stage_resident",
            Self::NaiveCrossNetworkAllToAll => "naive_cross_network_all_to_all",
        }
    }
}

/// Quantified per-token communication overhead of MoE expert routing.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRoutingReport {
    /// Model geometry label.
    pub model_label: String,
    /// Placement strategy evaluated.
    pub policy: ExpertPlacementPolicy,
    /// Network tier traversed for remote expert hops.
    pub link_tier: NetworkTier,
    /// Activation encoding format used on the wire.
    pub codec: ActivationCodec,
    /// Number of remote expert round-trip hops per token across the full model.
    pub remote_expert_hops_per_token: usize,
    /// Total wire bytes transferred per token solely for MoE expert dispatch + gather.
    pub expert_wire_bytes_per_token: u64,
    /// Added per-token latency in milliseconds from MoE expert routing communication.
    pub added_latency_ms_per_token: f64,
    /// Effective decode throughput ceiling (tokens/sec) imposed by expert network transfer
    /// alone at batch size `batch_size`.
    pub network_ceiling_tok_per_sec: f64,
}

/// Evaluates network transfer volume and latency impact of MoE expert routing for `spec`.
///
/// For `LocalityAwareStageResident`, all experts for layer `l` reside on the stage peer
/// owning layer `l`, adding zero remote expert hops (only pipeline stage handoffs occur).
/// For `NaiveCrossNetworkAllToAll`, a fraction `(1 - 1/num_nodes)` of each token's `top_k`
/// routed experts sit on remote peers, incurring a dispatch + gather round trip on every layer.
#[must_use]
pub fn evaluate_expert_routing(
    spec: &HybridModelSpec,
    policy: ExpertPlacementPolicy,
    link_tier: NetworkTier,
    codec: ActivationCodec,
    num_nodes: usize,
    batch_size: usize,
) -> ExpertRoutingReport {
    let nodes = num_nodes.max(1);
    let b = batch_size.max(1);
    let act_bytes = match codec {
        ActivationCodec::Bf16Lossless => spec.activation_bytes_bf16(),
        ActivationCodec::Int8PerTensor => spec.activation_bytes_int8(),
    };

    match policy {
        ExpertPlacementPolicy::LocalityAwareStageResident => ExpertRoutingReport {
            model_label: spec.label.to_string(),
            policy,
            link_tier,
            codec,
            remote_expert_hops_per_token: 0,
            expert_wire_bytes_per_token: 0,
            added_latency_ms_per_token: 0.0,
            network_ceiling_tok_per_sec: f64::INFINITY,
        },
        ExpertPlacementPolicy::NaiveCrossNetworkAllToAll => {
            // Expected fraction of top_k experts residing on remote peers.
            let remote_frac = if nodes <= 1 {
                0.0
            } else {
                ((nodes - 1) as f64) / (nodes as f64)
            };
            let layers = spec.total_layers();
            // Dispatch + gather = 2 transfers per layer; RTT paid sequentially at each layer.
            let remote_hops = if nodes <= 1 { 0 } else { layers * 2 };
            let bytes_per_layer_per_tok = ((2.0 * (spec.top_k_experts as f64) * remote_frac)
                * (act_bytes as f64))
                .round() as u64;
            let total_wire_bytes_per_tok = bytes_per_layer_per_tok * (layers as u64);

            // At batch size B, each layer dispatches B tokens concurrently (amortizing RTT across B
            // for throughput, while per-token wall time still pays layers * RTT + serialization(B)).
            let total_batch_bytes_per_layer = bytes_per_layer_per_tok * (b as u64);
            let per_layer_ms = if nodes <= 1 {
                0.0
            } else {
                link_tier.rtt_ms()
                    + ((total_batch_bytes_per_layer as f64) / link_tier.bandwidth_bytes_per_sec())
                        * 1000.0
            };
            let step_latency_ms = per_layer_ms * (layers as f64);
            let ceiling_tps = if step_latency_ms > 0.0 {
                (b as f64) * 1000.0 / step_latency_ms
            } else {
                f64::INFINITY
            };

            ExpertRoutingReport {
                model_label: spec.label.to_string(),
                policy,
                link_tier,
                codec,
                remote_expert_hops_per_token: remote_hops,
                expert_wire_bytes_per_token: total_wire_bytes_per_tok,
                added_latency_ms_per_token: step_latency_ms,
                network_ceiling_tok_per_sec: ceiling_tps,
            }
        }
    }
}

/// Deterministic affinity table tracking observed expert activation frequencies per layer
/// so peers can co-locate hot expert pairs inside the same pipeline stage.
#[derive(Debug, Clone)]
pub struct ExpertAffinityTracker {
    num_layers: usize,
    num_experts: usize,
    counts: Vec<Vec<u64>>,
}

impl ExpertAffinityTracker {
    /// Creates an empty tracker for `num_layers` layers and `num_experts` experts per layer.
    #[must_use]
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            num_layers,
            num_experts,
            counts: vec![vec![0u64; num_experts]; num_layers],
        }
    }

    /// Records the activated expert indices for global layer `layer_idx`.
    pub fn record(&mut self, layer_idx: usize, experts: &[usize]) {
        if layer_idx < self.num_layers {
            for &e in experts {
                if e < self.num_experts {
                    self.counts[layer_idx][e] += 1;
                }
            }
        }
    }

    /// Returns the hottest expert index observed at `layer_idx`, if any tokens were recorded.
    #[must_use]
    pub fn hottest_expert(&self, layer_idx: usize) -> Option<(usize, u64)> {
        let row = self.counts.get(layer_idx)?;
        row.iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .filter(|&(_, c)| c > 0)
    }

    /// Total recorded expert dispatches across all layers.
    #[must_use]
    pub fn total_dispatches(&self) -> u64 {
        self.counts.iter().flat_map(|r| r.iter()).copied().sum()
    }
}
