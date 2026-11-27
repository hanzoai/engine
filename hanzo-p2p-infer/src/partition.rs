// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Capability-aware heterogeneous pipeline partitioning, link-tier modeling
//! (Local LAN/RDMA vs Metro Fiber vs Public Internet), disconnect-tolerant stage
//! replication/rebalancing, and versioned Hanzo Engine P2P manifest declarations.

use crate::model::HybridModelSpec;
use crate::transport::{ActivationCodec, TrustTier};
use std::fmt;

/// Physical interconnect class between peer stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkTier {
    /// Local datacenter Ethernet / RDMA fabric (`0.2 ms` RTT, `25 Gbps` nominal).
    LocalLanRdma,
    /// Regional metropolitan dark fiber (`5.0 ms` RTT, `10 Gbps` nominal).
    MetroFiber,
    /// Consumer / cross-region public internet (`50.0 ms` RTT, `200 Mbps` nominal).
    PublicInternet,
    /// Single-host PCIe Gen4 x16 CPU-to-GPU conventional layer offload baseline
    /// (`0.02 ms` RTT, `25 GB/s` effective host-to-device bandwidth).
    PcieGen4Offload,
}

impl NetworkTier {
    /// Representative round-trip latency in milliseconds.
    #[must_use]
    pub const fn rtt_ms(&self) -> f64 {
        match self {
            Self::LocalLanRdma => 0.2,
            Self::MetroFiber => 5.0,
            Self::PublicInternet => 50.0,
            Self::PcieGen4Offload => 0.02,
        }
    }

    /// One-way propagation latency (`rtt_ms / 2.0`) in milliseconds.
    #[must_use]
    pub const fn one_way_ms(&self) -> f64 {
        self.rtt_ms() * 0.5
    }

    /// Sustained effective link bandwidth in Gbps (bits per second * 1e-9).
    #[must_use]
    pub const fn bandwidth_gbps(&self) -> f64 {
        match self {
            Self::LocalLanRdma => 25.0,
            Self::MetroFiber => 10.0,
            Self::PublicInternet => 0.2,
            Self::PcieGen4Offload => 200.0, // 25 GB/s effective PCIe Gen4 x16
        }
    }

    /// Sustained effective link bandwidth in bytes per second.
    #[must_use]
    pub fn bandwidth_bytes_per_sec(&self) -> f64 {
        self.bandwidth_gbps() * 1e9 / 8.0
    }

    /// One-way transmission + propagation time in milliseconds for transferring `bytes`.
    #[must_use]
    pub fn transfer_ms(&self, bytes: u64) -> f64 {
        let serialization_ms = ((bytes as f64) / self.bandwidth_bytes_per_sec()) * 1000.0;
        self.one_way_ms() + serialization_ms
    }

    /// Canonical telemetry string tag.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LocalLanRdma => "local_lan_rdma",
            Self::MetroFiber => "metro_fiber",
            Self::PublicInternet => "public_internet",
            Self::PcieGen4Offload => "pcie_gen4_offload",
        }
    }
}

/// Independent hardware and network capability advertisement for a P2P worker peer.
#[derive(Debug, Clone, PartialEq)]
pub struct PeerCapability {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Independent operator / administrative domain identifier.
    pub operator_id: String,
    /// Geographic or network region label.
    pub region: String,
    /// Total accelerator VRAM in bytes.
    pub vram_bytes: u64,
    /// Fixed runtime workspace reserve in bytes (activations, CUDA/Vulkan graphs).
    pub workspace_bytes: u64,
    /// Peak BF16 compute throughput in TFLOP/s.
    pub peak_bf16_tflops: f64,
    /// Sustained HBM/unified memory bandwidth in GB/s.
    pub hbm_bandwidth_gbps: f64,
    /// Interconnect tier to adjacent pipeline stages.
    pub link_tier: NetworkTier,
    /// Operator advertised compute rate in USD per GPU-hour.
    pub hourly_cost_usd: f64,
    /// Maximum simultaneous inference sessions admitted before backpressure.
    pub max_concurrent_sessions: u32,
}

impl PeerCapability {
    /// Usable VRAM after reserving fixed runtime workspace bytes.
    #[must_use]
    pub const fn usable_vram_bytes(&self) -> u64 {
        self.vram_bytes.saturating_sub(self.workspace_bytes)
    }

    /// Reference single-node datacenter H100-SXM5 80 GiB accelerator capability.
    #[must_use]
    pub fn h100_sxm5_80gb(peer_id: &str, operator_id: &str, link_tier: NetworkTier) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            operator_id: operator_id.to_string(),
            region: "us-east-dc".to_string(),
            vram_bytes: 80 * 1024 * 1024 * 1024,
            workspace_bytes: 2 * 1024 * 1024 * 1024,
            peak_bf16_tflops: 989.0,
            hbm_bandwidth_gbps: 3350.0,
            link_tier,
            hourly_cost_usd: 2.50,
            max_concurrent_sessions: 64,
        }
    }

    /// Representative independent consumer/prosumer 24 GiB GPU peer (RTX 4090 / 3090 class).
    #[must_use]
    pub fn consumer_24gb(
        peer_id: &str,
        operator_id: &str,
        region: &str,
        link_tier: NetworkTier,
    ) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            operator_id: operator_id.to_string(),
            region: region.to_string(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            workspace_bytes: 1536 * 1024 * 1024, // 1.5 GiB workspace
            peak_bf16_tflops: 330.0,
            hbm_bandwidth_gbps: 1008.0,
            link_tier,
            hourly_cost_usd: 0.42,
            max_concurrent_sessions: 16,
        }
    }

    /// Representative mid-tier 32 GiB / unified-memory workstation peer (heterogeneous cluster).
    #[must_use]
    pub fn workstation_32gb(
        peer_id: &str,
        operator_id: &str,
        region: &str,
        link_tier: NetworkTier,
    ) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            operator_id: operator_id.to_string(),
            region: region.to_string(),
            vram_bytes: 32 * 1024 * 1024 * 1024,
            workspace_bytes: 2 * 1024 * 1024 * 1024,
            peak_bf16_tflops: 420.0,
            hbm_bandwidth_gbps: 1200.0,
            link_tier,
            hourly_cost_usd: 0.58,
            max_concurrent_sessions: 24,
        }
    }
}

/// Assignment of a contiguous layer range to a primary peer plus standby failover peers.
#[derive(Debug, Clone, PartialEq)]
pub struct StageAssignment {
    /// Pipeline stage ordinal (`0..num_stages`).
    pub stage_idx: usize,
    /// Primary peer identifier serving this stage.
    pub primary_peer_id: String,
    /// Standby/overlapping peer identifiers capable of promoting this stage on disconnect.
    pub standby_peer_ids: Vec<String>,
    /// Inclusive starting decoder layer index.
    pub layer_start: usize,
    /// Exclusive ending decoder layer index.
    pub layer_end: usize,
    /// Exact VRAM footprint of this stage's resident weights + KV + GDN state in bytes.
    pub vram_required_bytes: u64,
    /// Content-addressed VFS shard URI (`vfs://sha256/...`).
    pub vfs_shard_uri: String,
}

impl StageAssignment {
    /// Number of layers hosted in this stage.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_end - self.layer_start
    }
}

/// Complete capability-aware pipeline placement plan across independent peers.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelinePlan {
    /// Canonical model label.
    pub model_label: String,
    /// Total decoder layers partitioned.
    pub total_layers: usize,
    /// Ordered stage assignments covering `0..total_layers` with zero gaps or overlaps.
    pub stages: Vec<StageAssignment>,
    /// Minimum number of node disconnects tolerated via standby/rebalance coverage.
    pub fault_tolerance_k: usize,
    /// Aggregate cluster VRAM required across all stages in bytes.
    pub total_vram_required_bytes: u64,
}

/// Errors encountered while planning heterogeneous pipeline partitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionError {
    /// No peer nodes were provided.
    EmptyPeerSet,
    /// Aggregate usable peer VRAM cannot hold the model weights and active KV/GDN state.
    InsufficientClusterVram {
        /// Required aggregate bytes.
        required_bytes: u64,
        /// Available aggregate usable VRAM bytes.
        available_bytes: u64,
    },
    /// No single peer can host the mandatory minimum layer allocation.
    PeerVramExceeded {
        /// Peer identifier that overflowed.
        peer_id: String,
        /// Required bytes on that peer.
        required_bytes: u64,
        /// Usable VRAM on that peer.
        usable_bytes: u64,
    },
    /// Cannot tolerate peer disconnect because no surviving peer has capacity to absorb stage.
    NoFailoverCandidate {
        /// Failed stage ordinal.
        stage_idx: usize,
    },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPeerSet => write!(f, "cannot partition pipeline across zero peers"),
            Self::InsufficientClusterVram {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "insufficient aggregate cluster VRAM: required {required_bytes} B > available {available_bytes} B"
            ),
            Self::PeerVramExceeded {
                peer_id,
                required_bytes,
                usable_bytes,
            } => write!(
                f,
                "peer {peer_id} VRAM exceeded: required {required_bytes} B > usable {usable_bytes} B"
            ),
            Self::NoFailoverCandidate { stage_idx } => {
                write!(f, "no failover peer available for stage {stage_idx}")
            }
        }
    }
}

impl std::error::Error for PartitionError {}

/// Plans a heterogeneous, capability-aware contiguous pipeline partition across `peers`
/// for `spec`, accounting for exact resident embedding/LM-head weights, decoder layers,
/// FP16 attention KV cache, and unquantized FP32 GDN recurrent state.
pub fn plan_heterogeneous_pipeline(
    spec: &HybridModelSpec,
    peers: &[PeerCapability],
    seq_len: usize,
    active_seqs: usize,
) -> Result<PipelinePlan, PartitionError> {
    if peers.is_empty() {
        return Err(PartitionError::EmptyPeerSet);
    }

    let total_layers = spec.total_layers();
    let total_required = spec.shard_vram_bytes(total_layers, seq_len, active_seqs, true, true);
    let total_available: u64 = peers.iter().map(PeerCapability::usable_vram_bytes).sum();
    if total_required > total_available {
        return Err(PartitionError::InsufficientClusterVram {
            required_bytes: total_required,
            available_bytes: total_available,
        });
    }

    let n = peers.len().min(total_layers);
    let active_peers = &peers[..n];

    // Weight each peer's share by harmonic combination of usable VRAM and HBM bandwidth
    // so heterogeneous nodes receive balanced decode step latencies without OOMing.
    let weights: Vec<f64> = active_peers
        .iter()
        .map(|p| {
            let vram_gib = (p.usable_vram_bytes() as f64) / (1024.0 * 1024.0 * 1024.0);
            let bw_norm = p.hbm_bandwidth_gbps / 1000.0;
            (vram_gib * bw_norm).sqrt().max(0.1)
        })
        .collect();
    let sum_w: f64 = weights.iter().sum();

    let mut counts = vec![1usize; n];
    let remaining = total_layers - n;
    let mut assigned_extra = 0usize;
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        let exact = (remaining as f64) * (weights[i] / sum_w);
        let floor_val = exact.floor() as usize;
        counts[i] += floor_val;
        assigned_extra += floor_val;
        remainders.push((i, exact - (floor_val as f64)));
    }

    remainders.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for &(idx, _) in remainders.iter().take(remaining - assigned_extra) {
        counts[idx] += 1;
    }

    // Rebalance iteratively if any peer exceeds its usable VRAM budget (especially stage 0 /
    // last stage which also hold resident input embeddings or output LM head).
    for _ in 0..(total_layers * 2) {
        let mut overflow_idx = None;
        for i in 0..n {
            let req = spec.shard_vram_bytes(counts[i], seq_len, active_seqs, i == 0, i + 1 == n);
            if req > active_peers[i].usable_vram_bytes() && counts[i] > 1 {
                overflow_idx = Some(i);
                break;
            }
        }
        let Some(over) = overflow_idx else {
            break;
        };
        // Find peer with highest remaining VRAM headroom.
        let mut best_target = None;
        let mut best_headroom = 0u64;
        for j in 0..n {
            if j == over {
                continue;
            }
            let req_after =
                spec.shard_vram_bytes(counts[j] + 1, seq_len, active_seqs, j == 0, j + 1 == n);
            let usable = active_peers[j].usable_vram_bytes();
            if req_after <= usable {
                let headroom = usable - req_after;
                if headroom > best_headroom {
                    best_headroom = headroom;
                    best_target = Some(j);
                }
            }
        }
        if let Some(target) = best_target {
            counts[over] -= 1;
            counts[target] += 1;
        } else {
            break;
        }
    }

    let mut stages = Vec::with_capacity(n);
    let mut cursor = 0usize;
    let mut total_vram_required = 0u64;

    for i in 0..n {
        let layer_start = cursor;
        let layer_end = cursor + counts[i];
        cursor = layer_end;
        let holds_emb = i == 0;
        let holds_lm = i + 1 == n;
        let vram_req = spec.shard_vram_bytes(counts[i], seq_len, active_seqs, holds_emb, holds_lm);
        if vram_req > active_peers[i].usable_vram_bytes() {
            return Err(PartitionError::PeerVramExceeded {
                peer_id: active_peers[i].peer_id.clone(),
                required_bytes: vram_req,
                usable_bytes: active_peers[i].usable_vram_bytes(),
            });
        }
        total_vram_required += vram_req;

        // Assign standby/overlapping peers from other peers in ring order (different operator preferred).
        let mut standbys = Vec::new();
        for offset in 1..peers.len() {
            let cand = &peers[(i + offset) % peers.len()];
            if cand.peer_id != active_peers[i].peer_id {
                standbys.push(cand.peer_id.clone());
            }
        }

        let uri = format!(
            "vfs://hanzo-p2p/{}/layers-{:03}-{:03}.safetensors",
            spec.label.to_lowercase(),
            layer_start,
            layer_end
        );

        stages.push(StageAssignment {
            stage_idx: i,
            primary_peer_id: active_peers[i].peer_id.clone(),
            standby_peer_ids: standbys,
            layer_start,
            layer_end,
            vram_required_bytes: vram_req,
            vfs_shard_uri: uri,
        });
    }

    let fault_tolerance_k = if peers.len() >= 2 { 1 } else { 0 };

    Ok(PipelinePlan {
        model_label: spec.label.to_string(),
        total_layers,
        stages,
        fault_tolerance_k,
        total_vram_required_bytes: total_vram_required,
    })
}

/// Re-plans stage ownership when `failed_peer_id` disconnects, promoting the first healthy
/// standby peer to absorb the dropped stage without dropping the session.
pub fn replan_after_disconnect(
    current: &PipelinePlan,
    failed_peer_id: &str,
) -> Result<PipelinePlan, PartitionError> {
    let mut updated = current.clone();
    for stage in &mut updated.stages {
        stage.standby_peer_ids.retain(|id| id != failed_peer_id);
        if stage.primary_peer_id == failed_peer_id {
            let replacement = stage.standby_peer_ids.first().cloned().ok_or(
                PartitionError::NoFailoverCandidate {
                    stage_idx: stage.stage_idx,
                },
            )?;
            stage.primary_peer_id = replacement;
            stage.standby_peer_ids.remove(0);
        }
    }
    updated.fault_tolerance_k = updated
        .stages
        .iter()
        .map(|s| usize::from(!s.standby_peer_ids.is_empty()))
        .min()
        .unwrap_or(0);
    Ok(updated)
}

/// Versioned Hanzo Engine P2P interoperability manifest (`hanzo.p2p.infer/v1`).
///
/// Binds pipeline partitioning into Hanzo Engine, Enso decentralized routing,
/// Visor worker sandboxes, and content-addressed VFS shard URIs without requiring
/// any Hanzo-operated centralized coordinator (`hanzo_coordinator_required = false`).
#[derive(Debug, Clone, PartialEq)]
pub struct HanzoP2PManifest {
    /// Versioned manifest schema identifier (`"hanzo.p2p.infer/v1"`).
    pub schema_version: String,
    /// Minimum compatible Hanzo Engine version (`"hanzo-engine/1.7"`).
    pub engine_compat: String,
    /// Enso decentralized routing topic / route identifier.
    pub enso_route_id: String,
    /// Visor sandbox isolation profile applied on independent worker hosts.
    pub visor_sandbox_profile: String,
    /// Content-addressed VFS shard URIs for every pipeline stage.
    pub vfs_shard_uris: Vec<String>,
    /// Activation codec negotiated for inter-peer wire transfers.
    pub activation_codec: ActivationCodec,
    /// Explicitly declared trust tier (never conflating verification models).
    pub trust_tier: TrustTier,
    /// Whether any Hanzo-operated coordinator is required for discovery, routing,
    /// execution, failover, or receipt verification (MUST be `false`).
    pub hanzo_coordinator_required: bool,
}

impl HanzoP2PManifest {
    /// Builds a canonical `hanzo.p2p.infer/v1` manifest from a [`PipelinePlan`].
    #[must_use]
    pub fn from_plan(plan: &PipelinePlan, codec: ActivationCodec, enso_route_id: &str) -> Self {
        let vfs_shard_uris = plan
            .stages
            .iter()
            .map(|s| s.vfs_shard_uri.clone())
            .collect();
        Self {
            schema_version: "hanzo.p2p.infer/v1".to_string(),
            engine_compat: "hanzo-engine/1.7".to_string(),
            enso_route_id: enso_route_id.to_string(),
            visor_sandbox_profile: "visor-seccomp-net-egress-allowlist-v1".to_string(),
            vfs_shard_uris,
            activation_codec: codec,
            trust_tier: TrustTier::CommitmentReceiptAndSpotCheck,
            hanzo_coordinator_required: false,
        }
    }
}
