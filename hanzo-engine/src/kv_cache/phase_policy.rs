#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Phase-aware KV page precision policy and radix-indexed hybrid KV+GDN+Conv1D prefix reuse.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use hanzo_ml::{DType, Device, Result, Tensor};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::{
    CacheProvenance, HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType,
    HybridPrefixCheckpoint, HybridStateContract, RecurrentLayerConfig,
};
use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::models::gdn::{forward_pooled, GdnLayerCache, PoolSlots};
use crate::paged_attention::PagedCacheType;

pub const ENV_ENGINE_KV_PHASE_POLICY: &str = "ENGINE_KV_PHASE_POLICY";
pub const ENV_ENGINE_HYBRID_PREFIX_CACHE: &str = "ENGINE_HYBRID_PREFIX_CACHE";
pub const DEFAULT_MIN_FP8_DECODE_CTX: usize = 2048;

const CONV_KERNEL_WIDTH: usize = 4;
const DEFAULT_BLOCK_SIZE: usize = 16;
const REF_DECODER_KV_LAYERS: usize = 36;

/// KV cache precision policy across prompt prefill and long-context decode phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvPhasePolicy {
    Bf16Only,
    Fp8Always,
    PhaseAware { min_fp8_decode_ctx: usize },
}

impl Default for KvPhasePolicy {
    fn default() -> Self {
        Self::PhaseAware {
            min_fp8_decode_ctx: DEFAULT_MIN_FP8_DECODE_CTX,
        }
    }
}

impl KvPhasePolicy {
    pub fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if let Some(rest) = trimmed.strip_prefix("phase_aware:") {
            let threshold = rest.parse::<usize>().ok()?;
            return Some(Self::PhaseAware {
                min_fp8_decode_ctx: threshold,
            });
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "auto" | "phase_aware" | "prefill_bf16_decode_fp8" => Some(Self::default()),
            "bf16" | "bf16_only" => Some(Self::Bf16Only),
            "fp8" | "f8e4m3" | "fp8_always" => Some(Self::Fp8Always),
            _ => None,
        }
    }

    /// Prefill computes causal attention over full-precision activations under PhaseAware.
    pub fn prefill_compute_dtype(&self, act_dtype: DType) -> DType {
        match self {
            Self::Bf16Only | Self::PhaseAware { .. } => act_dtype,
            Self::Fp8Always => DType::F8E4M3,
        }
    }

    pub fn decode_page_dtype(&self, act_dtype: DType, context_len: usize) -> DType {
        match *self {
            Self::Bf16Only => act_dtype,
            Self::Fp8Always => DType::F8E4M3,
            Self::PhaseAware { min_fp8_decode_ctx } => {
                if context_len >= min_fp8_decode_ctx {
                    DType::F8E4M3
                } else {
                    act_dtype
                }
            }
        }
    }

    pub fn paged_cache_type_for_decode(&self, context_len: usize) -> PagedCacheType {
        match *self {
            Self::Bf16Only => PagedCacheType::Auto,
            Self::Fp8Always => PagedCacheType::F8E4M3,
            Self::PhaseAware { min_fp8_decode_ctx } => {
                if context_len >= min_fp8_decode_ctx {
                    PagedCacheType::F8E4M3
                } else {
                    PagedCacheType::Auto
                }
            }
        }
    }

    pub fn kv_bytes_per_token(
        &self,
        act_dtype: DType,
        context_len: usize,
        kv_layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> usize {
        let elem_bytes = self
            .decode_page_dtype(act_dtype, context_len)
            .size_in_bytes();
        2 * kv_layers * kv_heads * head_dim * elem_bytes
    }
}

static KV_PHASE_POLICY_OVERRIDE: OnceLock<Mutex<Option<KvPhasePolicy>>> = OnceLock::new();
static HYBRID_PREFIX_CACHE_OVERRIDE: OnceLock<Mutex<Option<bool>>> = OnceLock::new();

pub fn set_kv_phase_policy_override(policy: Option<KvPhasePolicy>) {
    let mtx = KV_PHASE_POLICY_OVERRIDE.get_or_init(|| Mutex::new(None));
    *mtx.lock().expect("kv phase policy lock") = policy;
}

pub fn active_kv_phase_policy() -> KvPhasePolicy {
    if let Some(mtx) = KV_PHASE_POLICY_OVERRIDE.get() {
        if let Some(policy) = *mtx.lock().expect("kv phase policy lock") {
            return policy;
        }
    }
    std::env::var(ENV_ENGINE_KV_PHASE_POLICY)
        .ok()
        .and_then(|v| KvPhasePolicy::parse(&v))
        .unwrap_or_default()
}

pub fn set_hybrid_prefix_cache_override(enabled: Option<bool>) {
    let mtx = HYBRID_PREFIX_CACHE_OVERRIDE.get_or_init(|| Mutex::new(None));
    *mtx.lock().expect("hybrid prefix cache lock") = enabled;
}

pub fn hybrid_prefix_cache_enabled() -> bool {
    if let Some(mtx) = HYBRID_PREFIX_CACHE_OVERRIDE.get() {
        if let Some(enabled) = *mtx.lock().expect("hybrid prefix cache lock") {
            return enabled;
        }
    }
    match std::env::var(ENV_ENGINE_HYBRID_PREFIX_CACHE)
        .ok()
        .as_deref()
    {
        Some("0" | "false" | "FALSE" | "no" | "off") => false,
        Some("1" | "true" | "TRUE" | "yes" | "on") => true,
        _ => true,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridPrefixStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub evictions: u64,
    pub reused_tokens: u64,
    pub resident_entries: usize,
    pub resident_bytes: usize,
}

#[derive(Default)]
struct RadixNode {
    edge: Vec<u32>,
    entry_id: Option<u64>,
    children: HashMap<u32, RadixNode>,
}

impl RadixNode {
    fn insert(&mut self, tokens: &[u32], entry_id: u64) {
        if tokens.is_empty() {
            self.entry_id = Some(entry_id);
            return;
        }
        let first = tokens[0];
        let Some(mut child) = self.children.remove(&first) else {
            self.children.insert(
                first,
                RadixNode {
                    edge: tokens.to_vec(),
                    entry_id: Some(entry_id),
                    children: HashMap::new(),
                },
            );
            return;
        };

        let common = child
            .edge
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if common < child.edge.len() {
            let tail_edge = child.edge.split_off(common);
            let tail_first = tail_edge[0];
            let split_child = RadixNode {
                edge: tail_edge,
                entry_id: child.entry_id.take(),
                children: std::mem::take(&mut child.children),
            };
            child.children.insert(tail_first, split_child);
        }

        if common == tokens.len() {
            child.entry_id = Some(entry_id);
        } else {
            child.insert(&tokens[common..], entry_id);
        }
        self.children.insert(first, child);
    }

    fn longest_prefix_entry(&self, query: &[u32], max_len: usize) -> Option<(usize, u64)> {
        let mut best = None;
        let mut node = self;
        let mut consumed = 0usize;
        if consumed <= max_len {
            if let Some(id) = node.entry_id {
                best = Some((consumed, id));
            }
        }
        while consumed < query.len() {
            let next_tok = query[consumed];
            let Some(child) = node.children.get(&next_tok) else {
                break;
            };
            let rem = &query[consumed..];
            if rem.len() < child.edge.len() || rem[..child.edge.len()] != child.edge[..] {
                break;
            }
            consumed += child.edge.len();
            if consumed > max_len {
                break;
            }
            node = child;
            if let Some(id) = node.entry_id {
                best = Some((consumed, id));
            }
        }
        best
    }

    fn remove_entry(&mut self, tokens: &[u32], target_id: u64) -> bool {
        if tokens.is_empty() {
            if self.entry_id == Some(target_id) {
                self.entry_id = None;
                return true;
            }
            return false;
        }
        let first = tokens[0];
        let Some(child) = self.children.get_mut(&first) else {
            return false;
        };
        if tokens.len() < child.edge.len() || tokens[..child.edge.len()] != child.edge[..] {
            return false;
        }
        let removed = child.remove_entry(&tokens[child.edge.len()..], target_id);
        if child.entry_id.is_none() && child.children.is_empty() {
            self.children.remove(&first);
        }
        removed
    }
}

/// Radix-indexed hybrid KV + GDN + Conv1D checkpoint cache isolated by tenant, model, and adapter.
pub struct HybridPrefixRadixCache {
    trees: HashMap<CacheProvenance, RadixNode>,
    lru: IndexMap<u64, HybridPrefixCheckpoint>,
    key_index: HashMap<(CacheProvenance, Vec<u32>), u64>,
    max_entries: usize,
    max_bytes: usize,
    resident_bytes: usize,
    next_id: u64,
    stats: HybridPrefixStats,
}

impl HybridPrefixRadixCache {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            trees: HashMap::new(),
            lru: IndexMap::new(),
            key_index: HashMap::new(),
            max_entries: max_entries.max(1),
            max_bytes: max_bytes.max(1),
            resident_bytes: 0,
            next_id: 1,
            stats: HybridPrefixStats::default(),
        }
    }

    pub fn stats(&self) -> HybridPrefixStats {
        let mut s = self.stats;
        s.resident_entries = self.lru.len();
        s.resident_bytes = self.resident_bytes;
        s
    }

    pub fn clear(&mut self) {
        let evicted = self.lru.len() as u64;
        self.trees.clear();
        self.lru.clear();
        self.key_index.clear();
        self.resident_bytes = 0;
        self.stats.evictions += evicted;
    }

    pub fn insert(&mut self, ckpt: HybridPrefixCheckpoint) -> bool {
        if !hybrid_prefix_cache_enabled()
            || ckpt.tokens.is_empty()
            || ckpt.token_len != ckpt.tokens.len()
        {
            return false;
        }
        let bytes = ckpt.total_bytes();
        if bytes > self.max_bytes {
            return false;
        }

        let key = (ckpt.provenance.clone(), ckpt.tokens.clone());
        if let Some(old_id) = self.key_index.remove(&key) {
            if let Some(old) = self.lru.shift_remove(&old_id) {
                self.resident_bytes = self.resident_bytes.saturating_sub(old.total_bytes());
                if let Some(tree) = self.trees.get_mut(&old.provenance) {
                    tree.remove_entry(&old.tokens, old_id);
                }
            }
        }

        while !self.lru.is_empty()
            && (self.lru.len() >= self.max_entries || self.resident_bytes + bytes > self.max_bytes)
        {
            let Some((evicted_id, evicted)) = self.lru.shift_remove_index(0) else {
                break;
            };
            self.resident_bytes = self.resident_bytes.saturating_sub(evicted.total_bytes());
            self.key_index
                .remove(&(evicted.provenance.clone(), evicted.tokens.clone()));
            if let Some(tree) = self.trees.get_mut(&evicted.provenance) {
                tree.remove_entry(&evicted.tokens, evicted_id);
                if tree.entry_id.is_none() && tree.children.is_empty() {
                    self.trees.remove(&evicted.provenance);
                }
            }
            self.stats.evictions += 1;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.trees
            .entry(ckpt.provenance.clone())
            .or_default()
            .insert(&ckpt.tokens, id);
        self.key_index.insert(key, id);
        self.resident_bytes += bytes;
        self.lru.insert(id, ckpt);
        self.stats.stores += 1;
        true
    }

    /// Match the longest cached checkpoint strictly shorter than `prompt_tokens` under `provenance`.
    pub fn match_longest_prefix(
        &mut self,
        provenance: &CacheProvenance,
        prompt_tokens: &[u32],
    ) -> Option<(HybridPrefixCheckpoint, Vec<u32>)> {
        self.stats.lookups += 1;
        if !hybrid_prefix_cache_enabled() || prompt_tokens.len() <= 1 {
            self.stats.misses += 1;
            return None;
        }
        let max_prefix = prompt_tokens.len() - 1;
        let hit = self
            .trees
            .get(provenance)
            .and_then(|tree| tree.longest_prefix_entry(prompt_tokens, max_prefix));
        let Some((matched_len, entry_id)) = hit else {
            self.stats.misses += 1;
            return None;
        };
        let Some(ckpt) = self.lru.shift_remove(&entry_id) else {
            self.stats.misses += 1;
            return None;
        };
        if ckpt.provenance != *provenance
            || ckpt.token_len != matched_len
            || ckpt.chain_digest != provenance.token_chain_digest(&prompt_tokens[..matched_len])
        {
            self.stats.misses += 1;
            return None;
        }
        let out = ckpt.clone();
        self.lru.insert(entry_id, ckpt);
        self.stats.hits += 1;
        self.stats.reused_tokens += matched_len as u64;
        Some((out, prompt_tokens[matched_len..].to_vec()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SummaryStats {
    pub n: usize,
    pub median: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
}

impl SummaryStats {
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        };
        let mean = sorted.iter().sum::<f64>() / (n as f64);
        if n == 1 {
            return Self {
                n,
                median,
                mean,
                std_dev: 0.0,
                ci95_low: mean,
                ci95_high: mean,
            };
        }
        let var = sorted
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / ((n - 1) as f64);
        let std_dev = var.sqrt();
        let se = std_dev / (n as f64).sqrt();
        let half_width = student_t_crit_975(n - 1) * se;
        Self {
            n,
            median,
            mean,
            std_dev,
            ci95_low: mean - half_width,
            ci95_high: mean + half_width,
        }
    }
}

fn student_t_crit_975(df: usize) -> f64 {
    const T_975: [f64; 30] = [
        12.706_205, 4.302_653, 3.182_446, 2.776_445, 2.570_582, 2.446_912, 2.364_624, 2.306_004,
        2.262_157, 2.228_139, 2.200_985, 2.178_813, 2.160_369, 2.144_787, 2.131_450, 2.119_905,
        2.109_816, 2.100_922, 2.093_024, 2.085_963, 2.079_614, 2.073_873, 2.068_658, 2.063_899,
        2.059_539, 2.055_529, 2.051_831, 2.048_407, 2.045_230, 2.042_272,
    ];
    if df == 0 {
        0.0
    } else if df <= T_975.len() {
        T_975[df - 1]
    } else {
        let d = df as f64;
        1.959_964 + 2.372_28 / d + 2.822_5 / (d * d)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridEvalConfig {
    pub num_gdn_layers: usize,
    pub num_attn_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub system_tokens: usize,
    pub repo_tokens: usize,
    pub tool_schema_tokens: usize,
    pub user_turn_tokens: usize,
    pub assistant_turn_tokens: usize,
    pub num_turns: usize,
    pub repetitions: usize,
}

impl Default for HybridEvalConfig {
    fn default() -> Self {
        Self {
            num_gdn_layers: 6,
            num_attn_layers: 2,
            num_heads: 24,
            num_kv_heads: 2,
            head_dim: 128,
            hidden_size: 3072,
            system_tokens: 512,
            repo_tokens: 2048,
            tool_schema_tokens: 512,
            user_turn_tokens: 256,
            assistant_turn_tokens: 128,
            num_turns: 5,
            repetitions: 10,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnReuseMetrics {
    pub turn: usize,
    pub mode: String,
    pub total_prompt_tokens: usize,
    pub reused_prefix_tokens: usize,
    pub prefilled_suffix_tokens: usize,
    pub cold_reprefill_ms: f64,
    pub warm_prefix_reuse_ms: f64,
    pub ttft_speedup: f64,
    pub cold_ms_samples: Vec<f64>,
    pub warm_ms_samples: Vec<f64>,
    pub cold_ms_stats: SummaryStats,
    pub warm_ms_stats: SummaryStats,
    pub ttft_speedup_stats: SummaryStats,
    pub saved_prefill_flops: f64,
    pub saved_kv_write_bytes: usize,
    pub checkpoint_bytes: usize,
    pub attn_out_max_rel: f32,
    pub attn_out_cosine: f64,
    pub gdn_out_max_rel: f32,
    pub gdn_state_max_rel: f32,
    pub conv_state_max_rel: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridMultiturnEvalReport {
    pub backend: String,
    pub dtype: String,
    pub is_synthetic_microbench: bool,
    pub warmup_runs: usize,
    pub repetitions: usize,
    pub config: HybridEvalConfig,
    pub shared_base_turns: Vec<TurnReuseMetrics>,
    pub cumulative_coding_turns: Vec<TurnReuseMetrics>,
    pub cumulative_cold_ms: f64,
    pub cumulative_warm_ms: f64,
    pub cumulative_ttft_speedup: f64,
    pub cumulative_cold_ms_samples: Vec<f64>,
    pub cumulative_warm_ms_samples: Vec<f64>,
    pub cumulative_speedup_samples: Vec<f64>,
    pub cumulative_cold_ms_stats: SummaryStats,
    pub cumulative_warm_ms_stats: SummaryStats,
    pub cumulative_ttft_speedup_stats: SummaryStats,
    pub total_saved_prefill_flops: f64,
    pub total_saved_kv_write_bytes: usize,
    pub cache_stats: HybridPrefixStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceIsolationReport {
    pub owner_hit_tokens: usize,
    pub cross_tenant_hits: usize,
    pub unsalted_hits_against_salted: usize,
    pub cross_model_hits: usize,
    pub cross_adapter_hits: usize,
    pub base_hits_against_adapter: usize,
    pub boundary_collision_hits: usize,
    pub disabled_env_hits: usize,
    pub post_mutation_checkpoint_rel_err: f32,
    pub lru_evictions_under_budget: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvPhasePointMetrics {
    pub head_dim: usize,
    pub context_len: usize,
    pub prefill_seq_len: usize,
    pub phase_aware_used_fp8: bool,
    /// True only when decode_speedup comes from a native FP8 paged attention kernel (CUDA);
    /// false on portable CPU where both BF16 and FP8 KV upcast to F32 before matmul.
    pub decode_speedup_is_native: bool,
    pub bf16_prefill_us: f64,
    pub fp8_always_prefill_us: f64,
    pub phase_aware_prefill_us: f64,
    pub fp8_always_prefill_max_rel: f32,
    pub phase_aware_prefill_max_rel: f32,
    pub bf16_decode_us: f64,
    pub fp8_decode_us: f64,
    pub decode_speedup: f64,
    pub bf16_decode_gb_s: f64,
    pub fp8_decode_gb_s: f64,
    pub bf16_kv_36l_mib: f64,
    pub fp8_kv_36l_mib: f64,
    /// Analytic tensor payload byte ratio (2.0x for BF16 vs F8E4M3).
    pub analytic_kv_bytes_ratio: f64,
    /// Resident 36-layer BF16 KV allocation bytes including page/NormalCache growth rounding.
    pub measured_bf16_kv_alloc_bytes: usize,
    /// Resident 36-layer FP8 KV + per-layer/per-head f32 scale metadata bytes including rounding.
    pub measured_fp8_kv_plus_scale_bytes: usize,
    /// Resident allocation ratio (measured_bf16_kv_alloc_bytes / measured_fp8_kv_plus_scale_bytes).
    pub measured_kv_alloc_bytes_ratio: f64,
    pub bf16_decode_max_rel_vs_f32: f32,
    pub fp8_decode_max_rel_vs_f32: f32,
    pub fp8_decode_rmse_vs_bf16: f32,
    pub fp8_decode_cosine_vs_f32: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvPhaseEvalReport {
    pub backend: String,
    pub active_policy: KvPhasePolicy,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub block_size: usize,
    pub points: Vec<KvPhasePointMetrics>,
}

struct HybridTokenStream {
    conv_in: Tensor,
    gdn_q: Tensor,
    gdn_k: Tensor,
    gdn_v: Tensor,
    gdn_g: Tensor,
    gdn_beta: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
}

impl HybridTokenStream {
    fn generate(
        cfg: &HybridEvalConfig,
        seq_len: usize,
        seed: u64,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let conv_dim = 2 * nkv * d + nh * d;
        let conv_in =
            seeded_tensor(&[1, seq_len, conv_dim], seed + 1, 0.35, dev)?.to_dtype(dtype)?;
        let gdn_q = seeded_tensor(&[1, seq_len, nh, d], seed + 2, 0.25, dev)?.to_dtype(dtype)?;
        let gdn_k = seeded_tensor(&[1, seq_len, nh, d], seed + 3, 0.25, dev)?.to_dtype(dtype)?;
        let gdn_v = seeded_tensor(&[1, seq_len, nh, d], seed + 4, 0.25, dev)?.to_dtype(dtype)?;
        let gdn_g =
            (seeded_tensor(&[1, seq_len, nh], seed + 5, 0.20, dev)? - 0.45)?.to_dtype(dtype)?;
        let gdn_beta = ((seeded_tensor(&[1, seq_len, nh], seed + 6, 0.35, dev)? + 1.0)? * 0.45)?
            .to_dtype(dtype)?;
        let attn_q = seeded_tensor(&[1, nh, seq_len, d], seed + 7, 0.30, dev)?.to_dtype(dtype)?;
        let attn_k = seeded_tensor(&[1, nkv, seq_len, d], seed + 8, 0.30, dev)?.to_dtype(dtype)?;
        let attn_v = seeded_tensor(&[1, nkv, seq_len, d], seed + 9, 0.30, dev)?.to_dtype(dtype)?;
        Ok(Self {
            conv_in,
            gdn_q,
            gdn_k,
            gdn_v,
            gdn_g,
            gdn_beta,
            attn_q,
            attn_k,
            attn_v,
        })
    }

    fn slice(&self, start: usize, len: usize) -> Result<Self> {
        Ok(Self {
            conv_in: self.conv_in.narrow(1, start, len)?.contiguous()?,
            gdn_q: self.gdn_q.narrow(1, start, len)?.contiguous()?,
            gdn_k: self.gdn_k.narrow(1, start, len)?.contiguous()?,
            gdn_v: self.gdn_v.narrow(1, start, len)?.contiguous()?,
            gdn_g: self.gdn_g.narrow(1, start, len)?.contiguous()?,
            gdn_beta: self.gdn_beta.narrow(1, start, len)?.contiguous()?,
            attn_q: self.attn_q.narrow(2, start, len)?.contiguous()?,
            attn_k: self.attn_k.narrow(2, start, len)?.contiguous()?,
            attn_v: self.attn_v.narrow(2, start, len)?.contiguous()?,
        })
    }
}

fn seeded_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as u32 as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn seeded_tensor(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    Tensor::from_vec(seeded_vec(n, seed, scale), shape, dev)
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let max_diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let denom = b.iter().map(|y| y.abs()).fold(0.0f32, f32::max).max(1e-6);
    max_diff / denom
}

fn rmse_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / (a.len() as f64);
    mse.sqrt() as f32
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 {
        1.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn tensor_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn build_hybrid_contract(
    cfg: &HybridEvalConfig,
    max_seq_len: usize,
    dtype: DType,
    dev: &Device,
) -> Result<HybridStateContract> {
    let max_seq_len = (max_seq_len + super::NormalCache::CACHE_GROW_SIZE * 2)
        .next_multiple_of(super::NormalCache::CACHE_GROW_SIZE);
    let conv_dim = 2 * cfg.num_kv_heads * cfg.head_dim + cfg.num_heads * cfg.head_dim;
    let rec_cfg = RecurrentLayerConfig {
        conv_dim,
        conv_width: CONV_KERNEL_WIDTH,
        state_dims: vec![cfg.num_heads, cfg.head_dim, cfg.head_dim],
        conv_dtype: dtype,
        state_dtype: DType::F32,
    };
    let mut layer_types = Vec::with_capacity(cfg.num_gdn_layers + cfg.num_attn_layers);
    let gdn_per_attn = (cfg.num_gdn_layers / cfg.num_attn_layers.max(1)).max(1);
    let mut gdn_placed = 0usize;
    let mut attn_placed = 0usize;
    while gdn_placed < cfg.num_gdn_layers || attn_placed < cfg.num_attn_layers {
        for _ in 0..gdn_per_attn {
            if gdn_placed < cfg.num_gdn_layers {
                layer_types.push(HybridLayerType::Recurrent);
                gdn_placed += 1;
            }
        }
        if attn_placed < cfg.num_attn_layers {
            layer_types.push(HybridLayerType::Attention);
            attn_placed += 1;
        }
    }
    let hcfg = HybridCacheConfig::uniform(layer_types, max_seq_len, rec_cfg);
    Ok(HybridStateContract::new(HybridCache::new(hcfg, dev)?))
}

fn build_causal_suffix_mask(
    q_len: usize,
    total_kv_len: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let q_offset = total_kv_len.saturating_sub(q_len);
    let mut mask_vals = vec![0.0f32; q_len * total_kv_len];
    for qi in 0..q_len {
        let max_k = q_offset + qi;
        let row = qi * total_kv_len;
        for ki in (max_k + 1)..total_kv_len {
            mask_vals[row + ki] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_vals, (1, 1, q_len, total_kv_len), dev)?.to_dtype(dtype)
}

fn forward_hybrid_chunk(
    contract: &mut HybridStateContract,
    request_id: usize,
    chunk: &HybridTokenStream,
) -> Result<(Tensor, Tensor)> {
    let slot = contract
        .binding(request_id)
        .ok_or_else(|| hanzo_ml::Error::Msg("unallocated hybrid request".into()))?
        .recurrent_slot;
    let dev = chunk.conv_in.device().clone();
    let seq_len = chunk.conv_in.dim(1)?;
    let mut last_gdn = None;
    let mut last_attn = None;
    let total_layers = contract.cache.caches.len();

    for layer_idx in 0..total_layers {
        let is_recurrent = contract
            .cache
            .get(layer_idx)
            .and_then(HybridLayerCache::as_recurrent_pool)
            .is_some();
        if is_recurrent {
            let start_off = contract
                .cache
                .get(layer_idx)
                .and_then(HybridLayerCache::as_recurrent_pool)
                .unwrap()
                .get_seqlen_offset(slot);
            let pool = contract
                .cache
                .get_mut(layer_idx)
                .and_then(HybridLayerCache::as_recurrent_pool_mut)
                .unwrap();
            let y = forward_pooled(
                pool,
                PoolSlots::One {
                    slot,
                    offset: start_off,
                },
                layer_idx,
                false,
                |cache: &mut GdnLayerCache| {
                    let window =
                        Tensor::cat(&[&cache.conv_state, &chunk.conv_in.transpose(1, 2)?], 2)?;
                    cache.conv_state =
                        window.narrow(2, seq_len, CONV_KERNEL_WIDTH)?.contiguous()?;
                    let out = cache.recurrence(
                        &chunk.gdn_q,
                        &chunk.gdn_k,
                        &chunk.gdn_v,
                        &chunk.gdn_g,
                        &chunk.gdn_beta,
                    )?;
                    cache.seqlen_offset += seq_len;
                    Ok(out)
                },
            )?;
            last_gdn = Some(y);
        } else {
            let kv = contract
                .cache
                .get_mut(layer_idx)
                .and_then(HybridLayerCache::as_kv_cache_mut)
                .unwrap();
            let (full_k, full_v) = kv.append(&chunk.attn_k, &chunk.attn_v)?;
            let kd = chunk.attn_q.dim(3)?;
            let total_kv = full_k.dim(2)?;
            let params = SdpaParams {
                n_kv_groups: chunk.attn_q.dim(1)? / chunk.attn_k.dim(1)?,
                softcap: None,
                softmax_scale: 1.0 / (kd as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            };
            let mask = if seq_len == 1 {
                AttentionMask::None
            } else {
                AttentionMask::Custom(build_causal_suffix_mask(
                    seq_len,
                    total_kv,
                    chunk.attn_q.dtype(),
                    &dev,
                )?)
            };
            let y = Sdpa.run_attention(&chunk.attn_q, &full_k, &full_v, &mask, None, &params)?;
            last_attn = Some(y);
        }
    }

    let new_len = contract.cache.recurrent_offset(slot).unwrap_or(seq_len);
    let n_blocks = new_len.div_ceil(DEFAULT_BLOCK_SIZE);
    let blocks: Vec<u32> = (0..n_blocks as u32).map(|b| 1000 + b).collect();
    contract.update_paged_kv(request_id, blocks, new_len)?;

    Ok((
        last_gdn.ok_or_else(|| hanzo_ml::Error::Msg("missing GDN layer".into()))?,
        last_attn.ok_or_else(|| hanzo_ml::Error::Msg("missing Attention layer".into()))?,
    ))
}

fn estimate_prefill_flops_saved(
    cfg: &HybridEvalConfig,
    reused_tokens: usize,
    total_tokens: usize,
) -> f64 {
    let r = reused_tokens as f64;
    let total = total_tokens as f64;
    let suffix = total.saturating_sub_f64(r);
    let total_layers = (cfg.num_gdn_layers + cfg.num_attn_layers) as f64;
    let d_model = cfg.hidden_size as f64;
    let linear_proj_per_tok = 24.0 * d_model * d_model * total_layers;
    let gdn_scan_per_tok =
        4.0 * (cfg.num_heads * cfg.head_dim * cfg.head_dim * cfg.num_gdn_layers) as f64;
    let attn_full =
        2.0 * (cfg.num_heads * cfg.head_dim * cfg.num_attn_layers) as f64 * total * total;
    let attn_suffix = 2.0
        * (cfg.num_heads * cfg.head_dim * cfg.num_attn_layers) as f64
        * suffix
        * (r + 0.5 * suffix);
    r * (linear_proj_per_tok + gdn_scan_per_tok) + (attn_full - attn_suffix).max(0.0)
}

trait SaturatingSubF64 {
    fn saturating_sub_f64(self, rhs: f64) -> f64;
}

impl SaturatingSubF64 for f64 {
    fn saturating_sub_f64(self, rhs: f64) -> f64 {
        (self - rhs).max(0.0)
    }
}

fn make_token_ids(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| seed.wrapping_mul(100_003).wrapping_add(i + 1))
        .collect()
}

fn sync_device(dev: &Device) -> Result<()> {
    dev.synchronize()
}

struct PreparedTurnStreams {
    shared_prompt: Vec<u32>,
    shared_full: HybridTokenStream,
    cum_prompt_toks: Vec<u32>,
    cum_prompt_stream: HybridTokenStream,
    cum_after_asst_toks: Vec<u32>,
    cum_asst_stream: Option<HybridTokenStream>,
}

/// Evaluate multi-turn hybrid KV + GDN + Conv1D prefix checkpoint reuse across shared and cumulative turns.
/// Runs 1 full untimed multi-turn warmup sequence followed by `cfg.repetitions` timed repetitions,
/// reporting per-repetition samples, median, mean, sample SD, and 95% Student-t CIs.
pub fn evaluate_multiturn_hybrid_prefix_reuse(
    dev: &Device,
    dtype: DType,
    cfg: &HybridEvalConfig,
) -> Result<HybridMultiturnEvalReport> {
    set_hybrid_prefix_cache_override(Some(true));
    let reps = cfg.repetitions.max(1);
    let warmup_runs = 1usize;
    let shared_len = cfg.system_tokens + cfg.repo_tokens + cfg.tool_schema_tokens;
    let max_cum_len =
        shared_len + cfg.num_turns * (cfg.user_turn_tokens + cfg.assistant_turn_tokens) + 64;
    let provenance = CacheProvenance::new(
        Some("tenant-hanzo-research"),
        "zen4-hybrid-32b",
        Some("lora-agent-coder-v1"),
    );

    let shared_toks = make_token_ids(1, shared_len);
    let shared_stream = HybridTokenStream::generate(cfg, shared_len, 10_000, dtype, dev)?;

    let mut prepared = Vec::with_capacity(cfg.num_turns);
    let mut history_toks = shared_toks.clone();
    let mut history_stream = HybridTokenStream::generate(cfg, shared_len, 10_000, dtype, dev)?;

    for turn in 1..=cfg.num_turns {
        let user_a_toks = make_token_ids(100 + turn as u32, cfg.user_turn_tokens);
        let mut shared_prompt = shared_toks.clone();
        shared_prompt.extend_from_slice(&user_a_toks);
        let user_a_stream = HybridTokenStream::generate(
            cfg,
            cfg.user_turn_tokens,
            20_000 + turn as u64 * 100,
            dtype,
            dev,
        )?;
        let shared_full = concat_hybrid_streams(&shared_stream, &user_a_stream)?;

        let user_b_toks = make_token_ids(500 + turn as u32, cfg.user_turn_tokens);
        let user_b_stream = HybridTokenStream::generate(
            cfg,
            cfg.user_turn_tokens,
            50_000 + turn as u64 * 100,
            dtype,
            dev,
        )?;
        history_toks.extend_from_slice(&user_b_toks);
        history_stream = concat_hybrid_streams(&history_stream, &user_b_stream)?;
        let cum_prompt_toks = history_toks.clone();
        let cum_prompt_stream = history_stream.slice(0, cum_prompt_toks.len())?;

        let cum_asst_stream = if cfg.assistant_turn_tokens > 0 {
            let asst_toks = make_token_ids(900 + turn as u32, cfg.assistant_turn_tokens);
            let asst_stream = HybridTokenStream::generate(
                cfg,
                cfg.assistant_turn_tokens,
                90_000 + turn as u64 * 100,
                dtype,
                dev,
            )?;
            history_toks.extend_from_slice(&asst_toks);
            history_stream = concat_hybrid_streams(&history_stream, &asst_stream)?;
            Some(asst_stream)
        } else {
            None
        };
        let cum_after_asst_toks = history_toks.clone();

        prepared.push(PreparedTurnStreams {
            shared_prompt,
            shared_full,
            cum_prompt_toks,
            cum_prompt_stream,
            cum_after_asst_toks,
            cum_asst_stream,
        });
    }

    let mut shared_cold_samples = vec![Vec::with_capacity(reps); cfg.num_turns];
    let mut shared_warm_samples = vec![Vec::with_capacity(reps); cfg.num_turns];
    let mut cum_cold_samples = vec![Vec::with_capacity(reps); cfg.num_turns];
    let mut cum_warm_samples = vec![Vec::with_capacity(reps); cfg.num_turns];
    let mut shared_parity: Vec<Option<TurnReuseMetrics>> = vec![None; cfg.num_turns];
    let mut cum_parity: Vec<Option<TurnReuseMetrics>> = vec![None; cfg.num_turns];
    let mut last_cache_stats = HybridPrefixStats::default();

    for pass in 0..(warmup_runs + reps) {
        let is_warmup = pass < warmup_runs;
        let check_parity = pass == warmup_runs;
        let mut radix_cache = HybridPrefixRadixCache::new(64, 4 << 30);

        for (idx, p) in prepared.iter().enumerate() {
            let turn = idx + 1;
            let total_len = p.shared_prompt.len();

            let mut cold_contract = build_hybrid_contract(cfg, max_cum_len, dtype, dev)?;
            cold_contract.allocate_request(1)?;
            sync_device(dev)?;
            let t_cold = Instant::now();
            let (cold_gdn, cold_attn) =
                forward_hybrid_chunk(&mut cold_contract, 1, &p.shared_full)?;
            sync_device(dev)?;
            let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;

            let mut warm_contract = build_hybrid_contract(cfg, max_cum_len, dtype, dev)?;
            warm_contract.allocate_request(1)?;
            sync_device(dev)?;
            let t_warm = Instant::now();
            let (reused_len, warm_gdn, warm_attn, ckpt_bytes) = if let Some((ckpt, suffix_toks)) =
                radix_cache.match_longest_prefix(&provenance, &p.shared_prompt)
            {
                let offset = warm_contract.import_prefix_checkpoint(1, &ckpt)?;
                debug_assert_eq!(offset + suffix_toks.len(), total_len);
                let suffix_stream = p.shared_full.slice(offset, suffix_toks.len())?;
                let (yg, ya) = forward_hybrid_chunk(&mut warm_contract, 1, &suffix_stream)?;
                (offset, yg, ya, ckpt.total_bytes())
            } else {
                let _ = forward_hybrid_chunk(&mut warm_contract, 1, &shared_stream)?;
                let shared_ckpt =
                    warm_contract.export_prefix_checkpoint(1, provenance.clone(), &shared_toks)?;
                let cbytes = shared_ckpt.total_bytes();
                assert!(radix_cache.insert(shared_ckpt));
                let suffix_stream = p.shared_full.slice(shared_len, cfg.user_turn_tokens)?;
                let (yg, ya) = forward_hybrid_chunk(&mut warm_contract, 1, &suffix_stream)?;
                (0usize, yg, ya, cbytes)
            };
            sync_device(dev)?;
            let warm_ms = t_warm.elapsed().as_secs_f64() * 1e3;

            if !is_warmup {
                shared_cold_samples[idx].push(cold_ms);
                shared_warm_samples[idx].push(warm_ms);
            }
            if check_parity {
                shared_parity[idx] = Some(compare_hybrid_turn(
                    turn,
                    "shared_agent_base",
                    cfg,
                    dtype,
                    total_len,
                    reused_len,
                    cold_ms,
                    warm_ms,
                    ckpt_bytes,
                    &cold_contract,
                    &warm_contract,
                    &cold_gdn,
                    &warm_gdn,
                    &cold_attn,
                    &warm_attn,
                )?);
            }
        }

        for (idx, p) in prepared.iter().enumerate() {
            let turn = idx + 1;
            let prompt_len = p.cum_prompt_toks.len();

            let mut cold_contract = build_hybrid_contract(cfg, max_cum_len, dtype, dev)?;
            cold_contract.allocate_request(1)?;
            sync_device(dev)?;
            let t_cold = Instant::now();
            let (cold_gdn, cold_attn) =
                forward_hybrid_chunk(&mut cold_contract, 1, &p.cum_prompt_stream)?;
            sync_device(dev)?;
            let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;

            let mut warm_contract = build_hybrid_contract(cfg, max_cum_len, dtype, dev)?;
            warm_contract.allocate_request(1)?;
            sync_device(dev)?;
            let t_warm = Instant::now();
            let (reused_len, warm_gdn, warm_attn, ckpt_bytes) = if let Some((ckpt, suffix_toks)) =
                radix_cache.match_longest_prefix(&provenance, &p.cum_prompt_toks)
            {
                let offset = warm_contract.import_prefix_checkpoint(1, &ckpt)?;
                let suffix_stream = p.cum_prompt_stream.slice(offset, suffix_toks.len())?;
                let (yg, ya) = forward_hybrid_chunk(&mut warm_contract, 1, &suffix_stream)?;
                (offset, yg, ya, ckpt.total_bytes())
            } else {
                let (yg, ya) = forward_hybrid_chunk(&mut warm_contract, 1, &p.cum_prompt_stream)?;
                (0usize, yg, ya, 0usize)
            };
            sync_device(dev)?;
            let warm_ms = t_warm.elapsed().as_secs_f64() * 1e3;

            if !is_warmup {
                cum_cold_samples[idx].push(cold_ms);
                cum_warm_samples[idx].push(warm_ms);
            }
            if check_parity {
                cum_parity[idx] = Some(compare_hybrid_turn(
                    turn,
                    "cumulative_coding_chat",
                    cfg,
                    dtype,
                    prompt_len,
                    reused_len,
                    cold_ms,
                    warm_ms,
                    ckpt_bytes,
                    &cold_contract,
                    &warm_contract,
                    &cold_gdn,
                    &warm_gdn,
                    &cold_attn,
                    &warm_attn,
                )?);
            }

            if let Some(ref asst_stream) = p.cum_asst_stream {
                let _ = forward_hybrid_chunk(&mut warm_contract, 1, asst_stream)?;
                let turn_ckpt = warm_contract.export_prefix_checkpoint(
                    1,
                    provenance.clone(),
                    &p.cum_after_asst_toks,
                )?;
                assert!(radix_cache.insert(turn_ckpt));
            }
        }
        last_cache_stats = radix_cache.stats();
    }

    let finalize_turn = |mut base: TurnReuseMetrics,
                         c_samples: Vec<f64>,
                         w_samples: Vec<f64>|
     -> TurnReuseMetrics {
        let sp_samples: Vec<f64> = c_samples
            .iter()
            .zip(&w_samples)
            .map(|(&c, &w)| c / w.max(1e-6))
            .collect();
        let c_stats = SummaryStats::from_samples(&c_samples);
        let w_stats = SummaryStats::from_samples(&w_samples);
        let sp_stats = SummaryStats::from_samples(&sp_samples);
        base.cold_reprefill_ms = c_stats.median;
        base.warm_prefix_reuse_ms = w_stats.median;
        base.ttft_speedup = sp_stats.median;
        base.cold_ms_samples = c_samples;
        base.warm_ms_samples = w_samples;
        base.cold_ms_stats = c_stats;
        base.warm_ms_stats = w_stats;
        base.ttft_speedup_stats = sp_stats;
        base
    };

    let shared_base_turns: Vec<TurnReuseMetrics> = shared_parity
        .into_iter()
        .zip(shared_cold_samples)
        .zip(shared_warm_samples)
        .map(|((p, cs), ws)| finalize_turn(p.expect("parity recorded"), cs, ws))
        .collect();

    let cumulative_coding_turns: Vec<TurnReuseMetrics> = cum_parity
        .into_iter()
        .zip(cum_cold_samples)
        .zip(cum_warm_samples)
        .map(|((p, cs), ws)| finalize_turn(p.expect("parity recorded"), cs, ws))
        .collect();

    let mut cumulative_cold_ms_samples = vec![0.0f64; reps];
    let mut cumulative_warm_ms_samples = vec![0.0f64; reps];
    for r in 0..reps {
        cumulative_cold_ms_samples[r] = cumulative_coding_turns
            .iter()
            .map(|t| t.cold_ms_samples[r])
            .sum();
        cumulative_warm_ms_samples[r] = cumulative_coding_turns
            .iter()
            .map(|t| t.warm_ms_samples[r])
            .sum();
    }
    let cumulative_speedup_samples: Vec<f64> = cumulative_cold_ms_samples
        .iter()
        .zip(&cumulative_warm_ms_samples)
        .map(|(&c, &w)| c / w.max(1e-6))
        .collect();

    let cumulative_cold_ms_stats = SummaryStats::from_samples(&cumulative_cold_ms_samples);
    let cumulative_warm_ms_stats = SummaryStats::from_samples(&cumulative_warm_ms_samples);
    let cumulative_ttft_speedup_stats = SummaryStats::from_samples(&cumulative_speedup_samples);

    let total_saved_prefill_flops: f64 = cumulative_coding_turns
        .iter()
        .map(|t| t.saved_prefill_flops)
        .sum();
    let total_saved_kv_write_bytes: usize = cumulative_coding_turns
        .iter()
        .map(|t| t.saved_kv_write_bytes)
        .sum();
    set_hybrid_prefix_cache_override(None);

    Ok(HybridMultiturnEvalReport {
        backend: backend_label(dev),
        dtype: format!("{dtype:?}"),
        is_synthetic_microbench: true,
        warmup_runs,
        repetitions: reps,
        config: cfg.clone(),
        shared_base_turns,
        cumulative_coding_turns,
        cumulative_cold_ms: cumulative_cold_ms_stats.median,
        cumulative_warm_ms: cumulative_warm_ms_stats.median,
        cumulative_ttft_speedup: cumulative_ttft_speedup_stats.median,
        cumulative_cold_ms_samples,
        cumulative_warm_ms_samples,
        cumulative_speedup_samples,
        cumulative_cold_ms_stats,
        cumulative_warm_ms_stats,
        cumulative_ttft_speedup_stats,
        total_saved_prefill_flops,
        total_saved_kv_write_bytes,
        cache_stats: last_cache_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn compare_hybrid_turn(
    turn: usize,
    mode: &str,
    cfg: &HybridEvalConfig,
    dtype: DType,
    total_len: usize,
    reused_len: usize,
    cold_ms: f64,
    warm_ms: f64,
    ckpt_bytes: usize,
    cold_contract: &HybridStateContract,
    warm_contract: &HybridStateContract,
    cold_gdn: &Tensor,
    warm_gdn: &Tensor,
    cold_attn: &Tensor,
    warm_attn: &Tensor,
) -> Result<TurnReuseMetrics> {
    let suffix_len = total_len - reused_len;
    let eval_len = warm_gdn.dim(1)?;
    let tail_start = total_len - eval_len;
    let cold_attn_tail = cold_attn.narrow(2, tail_start, eval_len)?.contiguous()?;
    let cold_gdn_tail = cold_gdn.narrow(1, tail_start, eval_len)?.contiguous()?;

    let ca_v = tensor_f32_vec(&cold_attn_tail)?;
    let wa_v = tensor_f32_vec(warm_attn)?;
    let cg_v = tensor_f32_vec(&cold_gdn_tail)?;
    let wg_v = tensor_f32_vec(warm_gdn)?;

    let (cold_conv, cold_rec, _) = cold_contract.inspect_recurrent_layer(1, 0)?;
    let (warm_conv, warm_rec, _) = warm_contract.inspect_recurrent_layer(1, 0)?;
    let cc_v = tensor_f32_vec(&cold_conv)?;
    let wc_v = tensor_f32_vec(&warm_conv)?;
    let cr_v = tensor_f32_vec(&cold_rec)?;
    let wr_v = tensor_f32_vec(&warm_rec)?;

    let kv_bytes_per_tok =
        2 * cfg.num_attn_layers * cfg.num_kv_heads * cfg.head_dim * dtype.size_in_bytes();

    Ok(TurnReuseMetrics {
        turn,
        mode: mode.to_owned(),
        total_prompt_tokens: total_len,
        reused_prefix_tokens: reused_len,
        prefilled_suffix_tokens: suffix_len,
        cold_reprefill_ms: cold_ms,
        warm_prefix_reuse_ms: warm_ms,
        ttft_speedup: cold_ms / warm_ms.max(1e-6),
        cold_ms_samples: vec![cold_ms],
        warm_ms_samples: vec![warm_ms],
        cold_ms_stats: SummaryStats::from_samples(&[cold_ms]),
        warm_ms_stats: SummaryStats::from_samples(&[warm_ms]),
        ttft_speedup_stats: SummaryStats::from_samples(&[cold_ms / warm_ms.max(1e-6)]),
        saved_prefill_flops: estimate_prefill_flops_saved(cfg, reused_len, total_len),
        saved_kv_write_bytes: reused_len * kv_bytes_per_tok,
        checkpoint_bytes: ckpt_bytes,
        attn_out_max_rel: max_rel_err(&wa_v, &ca_v),
        attn_out_cosine: cosine_similarity(&wa_v, &ca_v),
        gdn_out_max_rel: max_rel_err(&wg_v, &cg_v),
        gdn_state_max_rel: max_rel_err(&wr_v, &cr_v),
        conv_state_max_rel: max_rel_err(&wc_v, &cc_v),
    })
}

fn concat_hybrid_streams(
    a: &HybridTokenStream,
    b: &HybridTokenStream,
) -> Result<HybridTokenStream> {
    Ok(HybridTokenStream {
        conv_in: Tensor::cat(&[&a.conv_in, &b.conv_in], 1)?.contiguous()?,
        gdn_q: Tensor::cat(&[&a.gdn_q, &b.gdn_q], 1)?.contiguous()?,
        gdn_k: Tensor::cat(&[&a.gdn_k, &b.gdn_k], 1)?.contiguous()?,
        gdn_v: Tensor::cat(&[&a.gdn_v, &b.gdn_v], 1)?.contiguous()?,
        gdn_g: Tensor::cat(&[&a.gdn_g, &b.gdn_g], 1)?.contiguous()?,
        gdn_beta: Tensor::cat(&[&a.gdn_beta, &b.gdn_beta], 1)?.contiguous()?,
        attn_q: Tensor::cat(&[&a.attn_q, &b.attn_q], 2)?.contiguous()?,
        attn_k: Tensor::cat(&[&a.attn_k, &b.attn_k], 2)?.contiguous()?,
        attn_v: Tensor::cat(&[&a.attn_v, &b.attn_v], 2)?.contiguous()?,
    })
}

/// Verify strict provenance isolation (`cache_salt`, `model_id`, `adapter_id`), mutation safety, and LRU eviction.
pub fn verify_hybrid_prefix_provenance_isolation(
    dev: &Device,
) -> Result<ProvenanceIsolationReport> {
    set_hybrid_prefix_cache_override(Some(true));
    let small_cfg = HybridEvalConfig {
        num_gdn_layers: 2,
        num_attn_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        hidden_size: 256,
        system_tokens: 32,
        repo_tokens: 32,
        tool_schema_tokens: 16,
        user_turn_tokens: 16,
        assistant_turn_tokens: 8,
        num_turns: 2,
        repetitions: 1,
    };
    let prefix_len = 80usize;
    let query_len = 96usize;
    let prefix_toks = make_token_ids(77, prefix_len);
    let mut query_toks = prefix_toks.clone();
    query_toks.extend_from_slice(&make_token_ids(88, query_len - prefix_len));

    let owner = CacheProvenance::new(Some("org-acme"), "zen4-hybrid-32b", Some("lora-fin-v1"));
    let mut contract = build_hybrid_contract(&small_cfg, 256, DType::F32, dev)?;
    contract.allocate_request(1)?;
    let stream = HybridTokenStream::generate(&small_cfg, prefix_len, 4242, DType::F32, dev)?;
    let _ = forward_hybrid_chunk(&mut contract, 1, &stream)?;
    let ckpt = contract.export_prefix_checkpoint(1, owner.clone(), &prefix_toks)?;
    let orig_rec = tensor_f32_vec(&ckpt.recurrent_snapshots[0].recurrent_state)?;

    let mut cache = HybridPrefixRadixCache::new(16, 256 << 20);
    assert!(cache.insert(ckpt));

    let owner_hit_tokens = cache
        .match_longest_prefix(&owner, &query_toks)
        .map_or(0, |(c, _)| c.token_len);

    let other_tenant =
        CacheProvenance::new(Some("org-evil"), "zen4-hybrid-32b", Some("lora-fin-v1"));
    let cross_tenant_hits = usize::from(
        cache
            .match_longest_prefix(&other_tenant, &query_toks)
            .is_some(),
    );

    let unsalted = CacheProvenance::new(None, "zen4-hybrid-32b", Some("lora-fin-v1"));
    let unsalted_hits_against_salted =
        usize::from(cache.match_longest_prefix(&unsalted, &query_toks).is_some());

    let other_model = CacheProvenance::new(Some("org-acme"), "zen4-hybrid-8b", Some("lora-fin-v1"));
    let cross_model_hits = usize::from(
        cache
            .match_longest_prefix(&other_model, &query_toks)
            .is_some(),
    );

    let other_adapter =
        CacheProvenance::new(Some("org-acme"), "zen4-hybrid-32b", Some("lora-hr-v2"));
    let cross_adapter_hits = usize::from(
        cache
            .match_longest_prefix(&other_adapter, &query_toks)
            .is_some(),
    );

    let base_no_adapter = CacheProvenance::new(Some("org-acme"), "zen4-hybrid-32b", None);
    let base_hits_against_adapter = usize::from(
        cache
            .match_longest_prefix(&base_no_adapter, &query_toks)
            .is_some(),
    );

    // Field-boundary shift attack: ("org-acm", "ezen4-hybrid-32b", Some("lora-fin-v1"))
    let boundary_attack =
        CacheProvenance::new(Some("org-acm"), "ezen4-hybrid-32b", Some("lora-fin-v1"));
    let boundary_collision_hits = usize::from(
        cache
            .match_longest_prefix(&boundary_attack, &query_toks)
            .is_some(),
    );

    // Mutating a restored request in-place must leave the cached checkpoint bit-exact.
    let (restored_ckpt, _) = cache.match_longest_prefix(&owner, &query_toks).unwrap();
    let mut mut_contract = build_hybrid_contract(&small_cfg, 256, DType::F32, dev)?;
    mut_contract.import_prefix_checkpoint(7, &restored_ckpt)?;
    let noise = HybridTokenStream::generate(&small_cfg, 8, 9999, DType::F32, dev)?;
    let _ = forward_hybrid_chunk(&mut mut_contract, 7, &noise)?;
    let (after_ckpt, _) = cache.match_longest_prefix(&owner, &query_toks).unwrap();
    let after_rec = tensor_f32_vec(&after_ckpt.recurrent_snapshots[0].recurrent_state)?;
    let post_mutation_checkpoint_rel_err = max_rel_err(&after_rec, &orig_rec);

    // Disable toggle test (ENGINE_HYBRID_PREFIX_CACHE=0 equivalent).
    set_hybrid_prefix_cache_override(Some(false));
    let disabled_env_hits = usize::from(cache.match_longest_prefix(&owner, &query_toks).is_some());
    set_hybrid_prefix_cache_override(Some(true));

    // Byte/entry cap LRU eviction test.
    let mut tiny_cache = HybridPrefixRadixCache::new(2, after_ckpt.total_bytes() * 2);
    for i in 0..4u32 {
        let p = CacheProvenance::new(Some(&format!("t-{i}")), "zen4-hybrid-32b", None);
        let c = contract.export_prefix_checkpoint(1, p, &prefix_toks)?;
        assert!(tiny_cache.insert(c));
    }
    let lru_evictions_under_budget = tiny_cache.stats().evictions;
    set_hybrid_prefix_cache_override(None);

    Ok(ProvenanceIsolationReport {
        owner_hit_tokens,
        cross_tenant_hits,
        unsalted_hits_against_salted,
        cross_model_hits,
        cross_adapter_hits,
        base_hits_against_adapter,
        boundary_collision_hits,
        disabled_env_hits,
        post_mutation_checkpoint_rel_err,
        lru_evictions_under_budget,
    })
}

/// Evaluate phase-aware BF16 vs FP8 (`F8E4M3`) KV page precision across head dims and 4K..32K contexts.
pub fn evaluate_phase_aware_kv_precision(
    dev: &Device,
    head_dims: &[usize],
    context_lens: &[usize],
) -> Result<KvPhaseEvalReport> {
    let policy = active_kv_phase_policy();
    let num_heads = 24usize;
    let num_kv_heads = 2usize;
    let block_size = DEFAULT_BLOCK_SIZE;
    let prefill_seq_len = if dev.is_cuda() { 1024usize } else { 256usize };
    let mut points = Vec::with_capacity(head_dims.len() * context_lens.len());

    for &head_dim in head_dims {
        for &ctx_len in context_lens {
            let pt = evaluate_single_kv_phase_point(
                dev,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                prefill_seq_len.min(ctx_len),
                ctx_len,
            )?;
            points.push(pt);
        }
    }

    Ok(KvPhaseEvalReport {
        backend: backend_label(dev),
        active_policy: policy,
        num_heads,
        num_kv_heads,
        block_size,
        points,
    })
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn evaluate_single_kv_phase_point(
    dev: &Device,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    prefill_len: usize,
    ctx_len: usize,
) -> Result<KvPhasePointMetrics> {
    use hanzo_paged_attn::{gather_kv_cache, kv_scale_update, paged_attention, reshape_and_cache};

    if !dev.is_cuda() {
        return evaluate_single_kv_phase_point_portable(
            dev,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            prefill_len,
            ctx_len,
        );
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let num_blocks = ctx_len.div_ceil(block_size) + 1;
    let max_blocks = ctx_len.div_ceil(block_size);
    let slot_ids: Vec<i64> = (0..ctx_len as i64).collect();
    let slot_mapping = Tensor::from_vec(slot_ids, (ctx_len,), dev)?;
    let bt_ids: Vec<u32> = (0..max_blocks as u32).collect();
    let block_tables = Tensor::from_vec(bt_ids, (1, max_blocks), dev)?;
    let context_lens_t = Tensor::from_vec(vec![ctx_len as u32], (1,), dev)?;

    let k_f32 = seeded_tensor(
        &[ctx_len, num_kv_heads, head_dim],
        (ctx_len + head_dim) as u64 + 101,
        0.4,
        dev,
    )?;
    let v_f32 = seeded_tensor(
        &[ctx_len, num_kv_heads, head_dim],
        (ctx_len + head_dim) as u64 + 202,
        0.4,
        dev,
    )?;
    let q_dec_f32 = seeded_tensor(
        &[1, num_heads, head_dim],
        (ctx_len + head_dim) as u64 + 303,
        0.4,
        dev,
    )?;

    let k_bf16 = k_f32.to_dtype(DType::BF16)?;
    let v_bf16 = v_f32.to_dtype(DType::BF16)?;
    let q_dec_bf16 = q_dec_f32.to_dtype(DType::BF16)?;

    // Allocate BF16 paged KV cache (x = 16 / 2 = 8) and FP8 E4M3 paged KV cache (x = 16 / 1 = 16).
    let x_bf16 = 16 / DType::BF16.size_in_bytes();
    let kc_bf16 = Tensor::zeros(
        (
            num_blocks,
            num_kv_heads,
            head_dim / x_bf16,
            block_size,
            x_bf16,
        ),
        DType::BF16,
        dev,
    )?;
    let vc_bf16 = Tensor::zeros(
        (num_blocks, num_kv_heads, head_dim, block_size),
        DType::BF16,
        dev,
    )?;
    reshape_and_cache(
        &k_bf16,
        &v_bf16,
        None,
        None,
        &kc_bf16,
        &vc_bf16,
        &slot_mapping,
    )?;

    let x_fp8 = 16 / DType::F8E4M3.size_in_bytes();
    let kc_fp8 = Tensor::zeros(
        (
            num_blocks,
            num_kv_heads,
            head_dim / x_fp8,
            block_size,
            x_fp8,
        ),
        DType::F8E4M3,
        dev,
    )?;
    let vc_fp8 = Tensor::zeros(
        (num_blocks, num_kv_heads, head_dim, block_size),
        DType::F8E4M3,
        dev,
    )?;
    let k_scale = Tensor::new(1.0f32, dev)?;
    let v_scale = Tensor::new(1.0f32, dev)?;
    kv_scale_update(&k_bf16, &v_bf16, &k_scale, &v_scale)?;
    reshape_and_cache(
        &k_bf16,
        &v_bf16,
        Some(&k_scale),
        Some(&v_scale),
        &kc_fp8,
        &vc_fp8,
        &slot_mapping,
    )?;

    // F32 exact decode reference over [0..ctx_len].
    let q4_f32 = q_dec_f32.unsqueeze(2)?;
    let k4_f32 = k_f32.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let v4_f32 = v_f32.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let sdpa_params = SdpaParams {
        n_kv_groups: num_heads / num_kv_heads,
        softcap: None,
        softmax_scale: scale,
        sliding_window: None,
        sinks: None,
    };
    let ref_dec_f32 = Sdpa
        .run_attention(
            &q4_f32,
            &k4_f32,
            &v4_f32,
            &AttentionMask::None,
            None,
            &sdpa_params,
        )?
        .reshape((1, num_heads, head_dim))?;
    let ref_dec_v = tensor_f32_vec(&ref_dec_f32)?;

    let out_bf16 = paged_attention(
        &q_dec_bf16,
        None,
        None,
        &kc_bf16,
        &vc_bf16,
        &block_tables,
        &context_lens_t,
        None,
        ctx_len,
        scale,
        1.0,
        None,
    )?;
    let out_fp8 = paged_attention(
        &q_dec_bf16,
        Some(&k_scale),
        Some(&v_scale),
        &kc_fp8,
        &vc_fp8,
        &block_tables,
        &context_lens_t,
        None,
        ctx_len,
        scale,
        1.0,
        None,
    )?;
    let got_bf16_v = tensor_f32_vec(&out_bf16)?;
    let got_fp8_v = tensor_f32_vec(&out_fp8)?;

    let bf16_decode_us = bench_cuda_op_us(dev, 4, 5, 16, || {
        paged_attention(
            &q_dec_bf16,
            None,
            None,
            &kc_bf16,
            &vc_bf16,
            &block_tables,
            &context_lens_t,
            None,
            ctx_len,
            scale,
            1.0,
            None,
        )
    })?;
    let fp8_decode_us = bench_cuda_op_us(dev, 4, 5, 16, || {
        paged_attention(
            &q_dec_bf16,
            Some(&k_scale),
            Some(&v_scale),
            &kc_fp8,
            &vc_fp8,
            &block_tables,
            &context_lens_t,
            None,
            ctx_len,
            scale,
            1.0,
            None,
        )
    })?;

    // Prefill phase comparison on [0..prefill_len]: Bf16Only vs Fp8Always (gather FP8) vs PhaseAware.
    let q_pre_f32 = seeded_tensor(
        &[1, num_heads, prefill_len, head_dim],
        (prefill_len + head_dim) as u64 + 404,
        0.35,
        dev,
    )?;
    let q_pre_bf16 = q_pre_f32.to_dtype(DType::BF16)?;
    let k_pre_f32 = k_f32.narrow(0, 0, prefill_len)?.contiguous()?;
    let v_pre_f32 = v_f32.narrow(0, 0, prefill_len)?.contiguous()?;
    let k_pre_bf16 = k_pre_f32.to_dtype(DType::BF16)?;
    let v_pre_bf16 = v_pre_f32.to_dtype(DType::BF16)?;
    let slot_pre = slot_mapping.narrow(0, 0, prefill_len)?.contiguous()?;
    let pre_blocks = prefill_len.div_ceil(block_size);
    let bt_pre = block_tables.narrow(1, 0, pre_blocks)?.contiguous()?;
    let cu_pre = Tensor::from_slice(&[0u32, prefill_len as u32], (2,), dev)?;

    let k4_pre_f32 = k_pre_f32.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let v4_pre_f32 = v_pre_f32.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let mask_f32 = AttentionMask::Custom(build_causal_suffix_mask(
        prefill_len,
        prefill_len,
        DType::F32,
        dev,
    )?);
    let ref_pre_v = tensor_f32_vec(&Sdpa.run_attention(
        &q_pre_f32,
        &k4_pre_f32,
        &v4_pre_f32,
        &mask_f32,
        None,
        &sdpa_params,
    )?)?;

    let k4_pre_bf16 = k_pre_bf16.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let v4_pre_bf16 = v_pre_bf16.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let pre_mask = if crate::using_flash_attn() {
        AttentionMask::CausalFlash
    } else {
        AttentionMask::Custom(build_causal_suffix_mask(
            prefill_len,
            prefill_len,
            DType::BF16,
            dev,
        )?)
    };

    let phase_aware_out = Sdpa.run_attention(
        &q_pre_bf16,
        &k4_pre_bf16,
        &v4_pre_bf16,
        &pre_mask,
        None,
        &sdpa_params,
    )?;
    let phase_aware_pre_v = tensor_f32_vec(&phase_aware_out)?;

    let (kg_fp8, vg_fp8) = gather_kv_cache(
        &kc_fp8,
        &vc_fp8,
        Some(&k_scale),
        Some(&v_scale),
        &bt_pre,
        &cu_pre,
        DType::BF16,
    )?;
    let k4_fp8_rt = kg_fp8.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let v4_fp8_rt = vg_fp8.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let fp8_always_out = Sdpa.run_attention(
        &q_pre_bf16,
        &k4_fp8_rt,
        &v4_fp8_rt,
        &pre_mask,
        None,
        &sdpa_params,
    )?;
    let fp8_always_pre_v = tensor_f32_vec(&fp8_always_out)?;

    let bf16_prefill_us = bench_cuda_op_us(dev, 3, 5, 4, || {
        reshape_and_cache(
            &k_pre_bf16,
            &v_pre_bf16,
            None,
            None,
            &kc_bf16,
            &vc_bf16,
            &slot_pre,
        )?;
        Sdpa.run_attention(
            &q_pre_bf16,
            &k4_pre_bf16,
            &v4_pre_bf16,
            &pre_mask,
            None,
            &sdpa_params,
        )
    })?;
    let phase_aware_prefill_us = bench_cuda_op_us(dev, 3, 5, 4, || {
        reshape_and_cache(
            &k_pre_bf16,
            &v_pre_bf16,
            Some(&k_scale),
            Some(&v_scale),
            &kc_fp8,
            &vc_fp8,
            &slot_pre,
        )?;
        Sdpa.run_attention(
            &q_pre_bf16,
            &k4_pre_bf16,
            &v4_pre_bf16,
            &pre_mask,
            None,
            &sdpa_params,
        )
    })?;
    let fp8_always_prefill_us = bench_cuda_op_us(dev, 3, 5, 4, || {
        reshape_and_cache(
            &k_pre_bf16,
            &v_pre_bf16,
            Some(&k_scale),
            Some(&v_scale),
            &kc_fp8,
            &vc_fp8,
            &slot_pre,
        )?;
        let (kg, vg) = gather_kv_cache(
            &kc_fp8,
            &vc_fp8,
            Some(&k_scale),
            Some(&v_scale),
            &bt_pre,
            &cu_pre,
            DType::BF16,
        )?;
        let k4 = kg.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let v4 = vg.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        Sdpa.run_attention(&q_pre_bf16, &k4, &v4, &pre_mask, None, &sdpa_params)
    })?;

    let bf16_kv_bytes = (2 * num_kv_heads * ctx_len * head_dim * 2) as f64;
    let fp8_kv_bytes = (2 * num_kv_heads * ctx_len * head_dim) as f64;
    let mib = (1024 * 1024) as f64;
    let policy = active_kv_phase_policy();
    let phase_aware_used_fp8 = policy.decode_page_dtype(DType::BF16, ctx_len) == DType::F8E4M3;

    // Resident 36-layer allocation accounting for paged block capacity + per-layer/per-head f32 scales.
    let bf16_layer_alloc_bytes =
        (kc_bf16.elem_count() + vc_bf16.elem_count()) * DType::BF16.size_in_bytes();
    let fp8_scale_meta_bytes = 2 * num_kv_heads * std::mem::size_of::<f32>();
    let fp8_layer_alloc_bytes = (kc_fp8.elem_count() + vc_fp8.elem_count())
        * DType::F8E4M3.size_in_bytes()
        + fp8_scale_meta_bytes;
    let measured_bf16_kv_alloc_bytes = REF_DECODER_KV_LAYERS * bf16_layer_alloc_bytes;
    let measured_fp8_kv_plus_scale_bytes = REF_DECODER_KV_LAYERS * fp8_layer_alloc_bytes;

    Ok(KvPhasePointMetrics {
        head_dim,
        context_len: ctx_len,
        prefill_seq_len: prefill_len,
        phase_aware_used_fp8,
        decode_speedup_is_native: true,
        bf16_prefill_us,
        fp8_always_prefill_us,
        phase_aware_prefill_us,
        fp8_always_prefill_max_rel: max_rel_err(&fp8_always_pre_v, &ref_pre_v),
        phase_aware_prefill_max_rel: max_rel_err(&phase_aware_pre_v, &ref_pre_v),
        bf16_decode_us,
        fp8_decode_us,
        decode_speedup: bf16_decode_us / fp8_decode_us.max(1e-6),
        bf16_decode_gb_s: bf16_kv_bytes / (bf16_decode_us * 1e3),
        fp8_decode_gb_s: fp8_kv_bytes / (fp8_decode_us * 1e3),
        bf16_kv_36l_mib: (bf16_kv_bytes * REF_DECODER_KV_LAYERS as f64) / mib,
        fp8_kv_36l_mib: (fp8_kv_bytes * REF_DECODER_KV_LAYERS as f64) / mib,
        analytic_kv_bytes_ratio: bf16_kv_bytes / fp8_kv_bytes.max(1.0),
        measured_bf16_kv_alloc_bytes,
        measured_fp8_kv_plus_scale_bytes,
        measured_kv_alloc_bytes_ratio: (measured_bf16_kv_alloc_bytes as f64)
            / (measured_fp8_kv_plus_scale_bytes.max(1) as f64),
        bf16_decode_max_rel_vs_f32: max_rel_err(&got_bf16_v, &ref_dec_v),
        fp8_decode_max_rel_vs_f32: max_rel_err(&got_fp8_v, &ref_dec_v),
        fp8_decode_rmse_vs_bf16: rmse_err(&got_fp8_v, &got_bf16_v),
        fp8_decode_cosine_vs_f32: cosine_similarity(&got_fp8_v, &ref_dec_v),
    })
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
fn evaluate_single_kv_phase_point(
    dev: &Device,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    prefill_len: usize,
    ctx_len: usize,
) -> Result<KvPhasePointMetrics> {
    evaluate_single_kv_phase_point_portable(
        dev,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        prefill_len,
        ctx_len,
    )
}

fn evaluate_single_kv_phase_point_portable(
    dev: &Device,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    _block_size: usize,
    prefill_len: usize,
    ctx_len: usize,
) -> Result<KvPhasePointMetrics> {
    use float8::F8E4M3;

    let policy = active_kv_phase_policy();
    let phase_aware_used_fp8 = policy.decode_page_dtype(DType::BF16, ctx_len) == DType::F8E4M3;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let k_vec = seeded_vec(
        num_kv_heads * ctx_len * head_dim,
        (ctx_len + head_dim) as u64 + 101,
        0.4,
    );
    let v_vec = seeded_vec(
        num_kv_heads * ctx_len * head_dim,
        (ctx_len + head_dim) as u64 + 202,
        0.4,
    );
    let q_dec = seeded_tensor(
        &[1, num_heads, 1, head_dim],
        (ctx_len + head_dim) as u64 + 303,
        0.4,
        dev,
    )?;

    let quant_fp8 = |xs: &[f32]| -> Vec<f32> {
        let amax = xs.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
        let qscale = amax / 240.0;
        xs.iter()
            .map(|&x| f32::from(F8E4M3::from_f32(x / qscale)) * qscale)
            .collect()
    };
    let k_fp8_vec = quant_fp8(&k_vec);
    let v_fp8_vec = quant_fp8(&v_vec);

    let k_f32 = Tensor::from_vec(k_vec, (1, num_kv_heads, ctx_len, head_dim), dev)?;
    let v_f32 = Tensor::from_vec(v_vec, (1, num_kv_heads, ctx_len, head_dim), dev)?;
    let k_bf16 = k_f32.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let v_bf16 = v_f32.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let k_fp8 = Tensor::from_vec(k_fp8_vec, (1, num_kv_heads, ctx_len, head_dim), dev)?;
    let v_fp8 = Tensor::from_vec(v_fp8_vec, (1, num_kv_heads, ctx_len, head_dim), dev)?;

    let params = SdpaParams {
        n_kv_groups: num_heads / num_kv_heads,
        softcap: None,
        softmax_scale: scale,
        sliding_window: None,
        sinks: None,
    };
    let ref_dec_v = tensor_f32_vec(&Sdpa.run_attention(
        &q_dec,
        &k_f32,
        &v_f32,
        &AttentionMask::None,
        None,
        &params,
    )?)?;
    let bf16_dec_v = tensor_f32_vec(&Sdpa.run_attention(
        &q_dec,
        &k_bf16,
        &v_bf16,
        &AttentionMask::None,
        None,
        &params,
    )?)?;
    let fp8_dec_v = tensor_f32_vec(&Sdpa.run_attention(
        &q_dec,
        &k_fp8,
        &v_fp8,
        &AttentionMask::None,
        None,
        &params,
    )?)?;

    // Portable CPU fallback upcasts both BF16 and FP8 KV to F32 before matmul; measure with
    // warmup + median of 5 reps and mark decode_speedup_is_native = false.
    let bf16_decode_us = bench_portable_op_us(dev, 3, 5, || {
        Sdpa.run_attention(
            &q_dec,
            &k_bf16,
            &v_bf16,
            &AttentionMask::None,
            None,
            &params,
        )
    })?;
    let fp8_decode_us = bench_portable_op_us(dev, 3, 5, || {
        Sdpa.run_attention(&q_dec, &k_fp8, &v_fp8, &AttentionMask::None, None, &params)
    })?;

    // Separate causal prefill measurement on [0..prefill_len] (seq_q = prefill_len > 1).
    let q_pre_f32 = seeded_tensor(
        &[1, num_heads, prefill_len, head_dim],
        (prefill_len + head_dim) as u64 + 404,
        0.35,
        dev,
    )?;
    let q_pre_bf16 = q_pre_f32.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let k_pre_f32 = k_f32.narrow(2, 0, prefill_len)?.contiguous()?;
    let v_pre_f32 = v_f32.narrow(2, 0, prefill_len)?.contiguous()?;
    let k_pre_bf16 = k_bf16.narrow(2, 0, prefill_len)?.contiguous()?;
    let v_pre_bf16 = v_bf16.narrow(2, 0, prefill_len)?.contiguous()?;
    let k_pre_fp8 = k_fp8.narrow(2, 0, prefill_len)?.contiguous()?;
    let v_pre_fp8 = v_fp8.narrow(2, 0, prefill_len)?.contiguous()?;
    let pre_mask = AttentionMask::Custom(build_causal_suffix_mask(
        prefill_len,
        prefill_len,
        DType::F32,
        dev,
    )?);

    let ref_pre_v = tensor_f32_vec(
        &Sdpa.run_attention(&q_pre_f32, &k_pre_f32, &v_pre_f32, &pre_mask, None, &params)?,
    )?;
    let phase_aware_pre_v = tensor_f32_vec(&Sdpa.run_attention(
        &q_pre_bf16,
        &k_pre_bf16,
        &v_pre_bf16,
        &pre_mask,
        None,
        &params,
    )?)?;
    let fp8_always_pre_v = tensor_f32_vec(&Sdpa.run_attention(
        &q_pre_bf16,
        &k_pre_fp8,
        &v_pre_fp8,
        &pre_mask,
        None,
        &params,
    )?)?;

    let k_pre_raw_v = tensor_f32_vec(&k_pre_f32)?;
    let v_pre_raw_v = tensor_f32_vec(&v_pre_f32)?;

    let bf16_prefill_us = bench_portable_op_us(dev, 3, 5, || {
        let _kb = k_pre_f32.to_dtype(DType::BF16)?;
        let _vb = v_pre_f32.to_dtype(DType::BF16)?;
        Sdpa.run_attention(
            &q_pre_bf16,
            &k_pre_bf16,
            &v_pre_bf16,
            &pre_mask,
            None,
            &params,
        )
    })?;
    let phase_aware_prefill_us = bench_portable_op_us(dev, 3, 5, || {
        if phase_aware_used_fp8 {
            std::hint::black_box(quant_fp8(&k_pre_raw_v));
            std::hint::black_box(quant_fp8(&v_pre_raw_v));
        } else {
            let _kb = k_pre_f32.to_dtype(DType::BF16)?;
            let _vb = v_pre_f32.to_dtype(DType::BF16)?;
        }
        Sdpa.run_attention(
            &q_pre_bf16,
            &k_pre_bf16,
            &v_pre_bf16,
            &pre_mask,
            None,
            &params,
        )
    })?;
    let fp8_always_prefill_us = bench_portable_op_us(dev, 3, 5, || {
        let kf = Tensor::from_vec(
            quant_fp8(&k_pre_raw_v),
            (1, num_kv_heads, prefill_len, head_dim),
            dev,
        )?;
        let vf = Tensor::from_vec(
            quant_fp8(&v_pre_raw_v),
            (1, num_kv_heads, prefill_len, head_dim),
            dev,
        )?;
        Sdpa.run_attention(&q_pre_bf16, &kf, &vf, &pre_mask, None, &params)
    })?;

    let bf16_kv_bytes = (2 * num_kv_heads * ctx_len * head_dim * 2) as f64;
    let fp8_kv_bytes = (2 * num_kv_heads * ctx_len * head_dim) as f64;
    let mib = (1024 * 1024) as f64;

    // NormalCache rounds capacity to CACHE_GROW_SIZE (512 tokens) + per-layer/per-head f32 scales.
    let alloc_tokens = ctx_len
        .max(1)
        .next_multiple_of(super::NormalCache::CACHE_GROW_SIZE);
    let bf16_layer_alloc_bytes = 2 * num_kv_heads * alloc_tokens * head_dim * 2;
    let fp8_scale_meta_bytes = 2 * num_kv_heads * std::mem::size_of::<f32>();
    let fp8_layer_alloc_bytes = 2 * num_kv_heads * alloc_tokens * head_dim + fp8_scale_meta_bytes;
    let measured_bf16_kv_alloc_bytes = REF_DECODER_KV_LAYERS * bf16_layer_alloc_bytes;
    let measured_fp8_kv_plus_scale_bytes = REF_DECODER_KV_LAYERS * fp8_layer_alloc_bytes;

    Ok(KvPhasePointMetrics {
        head_dim,
        context_len: ctx_len,
        prefill_seq_len: prefill_len,
        phase_aware_used_fp8,
        decode_speedup_is_native: false,
        bf16_prefill_us,
        fp8_always_prefill_us,
        phase_aware_prefill_us,
        fp8_always_prefill_max_rel: max_rel_err(&fp8_always_pre_v, &ref_pre_v),
        phase_aware_prefill_max_rel: max_rel_err(&phase_aware_pre_v, &ref_pre_v),
        bf16_decode_us,
        fp8_decode_us,
        decode_speedup: bf16_decode_us / fp8_decode_us.max(1e-6),
        bf16_decode_gb_s: bf16_kv_bytes / (bf16_decode_us * 1e3),
        fp8_decode_gb_s: fp8_kv_bytes / (fp8_decode_us * 1e3),
        bf16_kv_36l_mib: (bf16_kv_bytes * REF_DECODER_KV_LAYERS as f64) / mib,
        fp8_kv_36l_mib: (fp8_kv_bytes * REF_DECODER_KV_LAYERS as f64) / mib,
        analytic_kv_bytes_ratio: bf16_kv_bytes / fp8_kv_bytes.max(1.0),
        measured_bf16_kv_alloc_bytes,
        measured_fp8_kv_plus_scale_bytes,
        measured_kv_alloc_bytes_ratio: (measured_bf16_kv_alloc_bytes as f64)
            / (measured_fp8_kv_plus_scale_bytes.max(1) as f64),
        bf16_decode_max_rel_vs_f32: max_rel_err(&bf16_dec_v, &ref_dec_v),
        fp8_decode_max_rel_vs_f32: max_rel_err(&fp8_dec_v, &ref_dec_v),
        fp8_decode_rmse_vs_bf16: rmse_err(&fp8_dec_v, &bf16_dec_v),
        fp8_decode_cosine_vs_f32: cosine_similarity(&fp8_dec_v, &ref_dec_v),
    })
}

fn bench_portable_op_us<F>(dev: &Device, warmup: usize, reps: usize, mut op: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..warmup {
        let out = op()?;
        std::hint::black_box(&out);
    }
    dev.synchronize()?;
    let n = reps.max(1);
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        let out = op()?;
        std::hint::black_box(&out);
        dev.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(samples[samples.len() / 2])
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn bench_cuda_op_us<F>(
    dev: &Device,
    warmup: usize,
    batches: usize,
    reps: usize,
    mut op: F,
) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..warmup {
        let _ = op()?;
    }
    dev.synchronize()?;
    let mut samples = Vec::with_capacity(batches);
    for _ in 0..batches {
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = op()?;
        }
        dev.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6 / (reps as f64));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

fn backend_label(dev: &Device) -> String {
    if dev.is_cuda() {
        "CUDA".to_owned()
    } else if dev.is_metal() {
        "Metal".to_owned()
    } else if dev.is_cpu() {
        "CPU".to_owned()
    } else {
        format!("{:?}", dev.location())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_phase_policy_parsing_and_phase_selection() {
        assert_eq!(
            KvPhasePolicy::parse("auto"),
            Some(KvPhasePolicy::PhaseAware {
                min_fp8_decode_ctx: DEFAULT_MIN_FP8_DECODE_CTX
            })
        );
        assert_eq!(
            KvPhasePolicy::parse("phase_aware:4096"),
            Some(KvPhasePolicy::PhaseAware {
                min_fp8_decode_ctx: 4096
            })
        );
        assert_eq!(KvPhasePolicy::parse("bf16"), Some(KvPhasePolicy::Bf16Only));
        assert_eq!(KvPhasePolicy::parse("fp8"), Some(KvPhasePolicy::Fp8Always));

        let p = KvPhasePolicy::PhaseAware {
            min_fp8_decode_ctx: 4096,
        };
        assert_eq!(p.prefill_compute_dtype(DType::BF16), DType::BF16);
        assert_eq!(p.decode_page_dtype(DType::BF16, 1024), DType::BF16);
        assert_eq!(p.decode_page_dtype(DType::BF16, 4096), DType::F8E4M3);
        assert_eq!(
            p.kv_bytes_per_token(DType::BF16, 1024, 36, 2, 128),
            2 * p.kv_bytes_per_token(DType::BF16, 4096, 36, 2, 128)
        );
    }

    #[test]
    fn cpu_hybrid_prefix_isolation_and_multiturn_parity() -> Result<()> {
        let dev = Device::Cpu;
        let iso = verify_hybrid_prefix_provenance_isolation(&dev)?;
        assert_eq!(iso.owner_hit_tokens, 80);
        assert_eq!(iso.cross_tenant_hits, 0);
        assert_eq!(iso.unsalted_hits_against_salted, 0);
        assert_eq!(iso.cross_model_hits, 0);
        assert_eq!(iso.cross_adapter_hits, 0);
        assert_eq!(iso.base_hits_against_adapter, 0);
        assert_eq!(iso.boundary_collision_hits, 0);
        assert_eq!(iso.disabled_env_hits, 0);
        assert_eq!(iso.post_mutation_checkpoint_rel_err, 0.0);
        assert_eq!(iso.lru_evictions_under_budget, 2);

        let cfg = HybridEvalConfig {
            num_gdn_layers: 2,
            num_attn_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            hidden_size: 256,
            system_tokens: 16,
            repo_tokens: 32,
            tool_schema_tokens: 16,
            user_turn_tokens: 16,
            assistant_turn_tokens: 8,
            num_turns: 5,
            repetitions: 3,
        };
        let rep = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::F32, &cfg)?;
        assert_eq!(rep.warmup_runs, 1);
        assert_eq!(rep.repetitions, 3);
        assert!(rep.is_synthetic_microbench);
        assert_eq!(rep.cumulative_cold_ms_samples.len(), 3);
        assert_eq!(rep.cumulative_warm_ms_samples.len(), 3);
        assert!(
            rep.cumulative_ttft_speedup_stats.ci95_low <= rep.cumulative_ttft_speedup_stats.mean
                && rep.cumulative_ttft_speedup_stats.mean
                    <= rep.cumulative_ttft_speedup_stats.ci95_high
        );
        for t in rep
            .shared_base_turns
            .iter()
            .chain(&rep.cumulative_coding_turns)
        {
            assert!(
                t.attn_out_max_rel < 1e-5,
                "turn {} attn rel {}",
                t.turn,
                t.attn_out_max_rel
            );
            assert!(
                t.gdn_out_max_rel < 1e-5,
                "turn {} gdn rel {}",
                t.turn,
                t.gdn_out_max_rel
            );
            assert!(
                t.gdn_state_max_rel < 1e-5,
                "turn {} gdn st rel {}",
                t.turn,
                t.gdn_state_max_rel
            );
            assert_eq!(t.conv_state_max_rel, 0.0);
        }
        assert_eq!(rep.shared_base_turns[4].reused_prefix_tokens, 64);
        assert_eq!(
            rep.cumulative_coding_turns[4].reused_prefix_tokens,
            64 + 4 * 24
        );
        Ok(())
    }
}
