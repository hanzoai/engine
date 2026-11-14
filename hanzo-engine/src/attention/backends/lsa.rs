#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Latent Sparse Attention (LSA) backend for `hanzo-engine`.
//!
//! Implements both **D-LSA** (distilled / training-free inference sparse attention) and
//! **N-LSA** (native sparse attention with two-path second-order Gaussian bucket summaries),
//! supporting:
//! - Position-free GQA head regrouping (`regroup`, `regroup_partial`)
//! - MLA Cholesky isometric latent metric (`LatentMetric`, `regroup_latent`)
//! - Distilled `[input_dim, index_dim]` projections loaded from `.safetensors` (`<layer>.W_Q_idx`,
//!   `<layer>.W_K_idx`) or training-free online PCA (`pca_projections`)
//! - Consecutive pool sums + mutual-cheapest Ward hierarchical clustering (`ward_merge`, `build_buckets`)
//! - Optional 1-bit sign-quantized covariance bucket shapes (`Shapes`, `build_shapes`, `shaped_scores`)
//! - Flat (`top_level == 0`) and Hierarchical Tree (`top_level >= 1`) beam-search bucket routing
//! - Subquadratic gather-based sparse SDPA (`lsa_sdpa`) + incremental per-layer bucket cache (`LsaSessionCache`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use hanzo_ml::{DType, Device, Error, Result, Tensor, D};
use hanzo_quant::MatMul;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::attention::SdpaParams;

use super::maybe_synchronize;

const NORM_EPS: f32 = 1e-12;
const CENTROID_EPS: f32 = 1e-30;
const MERGE_COST_EPS: f32 = 1e-12;

/// Where rotary position embedding sits inside the head dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RotaryLayout {
    #[serde(alias = "Prefix", alias = "PREFIX")]
    Prefix,
    #[serde(alias = "Suffix", alias = "SUFFIX")]
    Suffix,
    #[serde(alias = "None", alias = "NONE")]
    None,
}

/// Early budget override for sequence positions `< position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EarlyBudget {
    pub position: usize,
    pub top_p: usize,
    pub top_k: usize,
}

/// Configuration for Latent Sparse Attention in `hanzo-engine`.
///
/// Can be constructed programmatically in Rust, deserialized from TOML/YAML/JSON IaC manifests,
/// attached to [`crate::LoaderBuilder::with_lsa`], or installed globally via [`LsaConfig::install`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LsaConfig {
    /// Sliding window & cluster block size `n_w = n_cluster` (default 512).
    pub window_size: usize,
    /// Leading attention sink tokens `n_sink` (default 4).
    pub sink_size: usize,
    /// Maximum long-range keys unpacked per query `k` (default 4096).
    pub top_k: usize,
    /// Number of buckets selected per query `p` (default 1024).
    pub top_p: usize,
    /// Pool size `n_pool` before Ward clustering (default 4).
    pub pool_size: usize,
    /// Mean keys per bucket `\bar{n}_b` (default 8 -> 64 buckets per 512-key block).
    pub bucket_size: usize,
    /// Hard maximum keys per bucket `n_b^{max}` (default 16).
    pub bucket_cap: usize,
    /// Unrotated RoPE layout (`Prefix`, `Suffix`, or `None`).
    pub rotary_layout: RotaryLayout,
    /// Number of rotated dimensions in each head (`0` to use full head vector).
    pub rotary_dim: usize,
    /// Optional 1-bit covariance shape `(rank, deviations)`.
    pub bucket_shape: Option<(usize, f32)>,
    /// Tree depth (`0` for Flat design, `>= 1` for Tree/Hierarchical design).
    pub top_level: usize,
    /// Children per node bucket in Tree design (default 4).
    pub node_bucket_size: usize,
    /// Buckets per node cluster block in Tree design (default 4096).
    pub node_cluster_block_size: usize,
    /// Beam width in Tree design (default 256).
    pub beam_width: usize,
    /// Start node threshold in Tree design (default 1024).
    pub start_nodes: usize,
    /// Optional early budget for shorter prefix positions.
    pub early_budget: Option<EarlyBudget>,
    /// Optional training-free PCA index dimension (`None` uses full regrouped dim or loaded projections).
    pub pca_index_dim: Option<usize>,
    /// Minimum KV context length before routing to LSA; shorter contexts fall back to standard attention
    /// where dense flash/tiled SDPA has lower fixed overhead (default 8192).
    pub min_seq_len: usize,
    /// Optional path to a `.safetensors` file holding distilled `<layer>.W_Q_idx` and `<layer>.W_K_idx` maps.
    pub projections_path: Option<PathBuf>,
}

impl Default for LsaConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            sink_size: 4,
            top_k: 4096,
            top_p: 1024,
            pool_size: 4,
            bucket_size: 8,
            bucket_cap: 16,
            rotary_layout: RotaryLayout::Prefix,
            rotary_dim: 0,
            bucket_shape: None,
            top_level: 0,
            node_bucket_size: 4,
            node_cluster_block_size: 4096,
            beam_width: 256,
            start_nodes: 1024,
            early_budget: None,
            pca_index_dim: None,
            min_seq_len: 8192,
            projections_path: None,
        }
    }
}

fn global_lsa_slot() -> &'static RwLock<Option<Arc<LsaConfig>>> {
    static GLOBAL_LSA: OnceLock<RwLock<Option<Arc<LsaConfig>>>> = OnceLock::new();
    GLOBAL_LSA.get_or_init(|| {
        let env_on = std::env::var("HANZO_LSA")
            .ok()
            .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"));
        let initial = if env_on {
            let mut cfg = LsaConfig::zen_27b();
            if let Ok(ms) = std::env::var("HANZO_LSA_MIN_SEQ_LEN") {
                if let Ok(val) = ms.trim().parse::<usize>() {
                    cfg = cfg.with_min_seq_len(val);
                }
            }
            if let (Ok(tp), Ok(tk)) = (
                std::env::var("HANZO_LSA_TOP_P"),
                std::env::var("HANZO_LSA_TOP_K"),
            ) {
                if let (Ok(p), Ok(k)) = (tp.trim().parse::<usize>(), tk.trim().parse::<usize>()) {
                    cfg = cfg.with_budget(p, k);
                }
            }
            Some(Arc::new(cfg))
        } else {
            None
        };
        RwLock::new(initial)
    })
}

fn global_session_pool() -> &'static RwLock<HashMap<usize, Pinned<LsaSessionCache>>> {
    static GLOBAL_POOL: OnceLock<RwLock<HashMap<usize, Pinned<LsaSessionCache>>>> = OnceLock::new();
    GLOBAL_POOL.get_or_init(|| RwLock::new(HashMap::new()))
}

fn global_gpu_session_pool() -> &'static RwLock<HashMap<usize, Pinned<LsaGpuSessionCache>>> {
    static GLOBAL_GPU_POOL: OnceLock<RwLock<HashMap<usize, Pinned<LsaGpuSessionCache>>>> =
        OnceLock::new();
    GLOBAL_GPU_POOL.get_or_init(|| RwLock::new(HashMap::new()))
}

struct PagedLsaEntry {
    session: LsaGpuSessionCache,
    /// Physical block IDs covering KV tokens `[0 .. session.clustered_blocks * window_size)`.
    block_ids: Vec<usize>,
    /// Keep the KV cache storage alive while the entry references its storage pointer.
    _source: Tensor,
    generation: u64,
    used: u64,
}

#[derive(Clone, Copy)]
enum PendingPagedOp {
    Rollback { block_0: usize, keep_len: usize },
    Evict { block_0: usize },
}

fn global_paged_session_pool() -> &'static RwLock<HashMap<(usize, usize), PagedLsaEntry>> {
    static GLOBAL_PAGED_POOL: OnceLock<RwLock<HashMap<(usize, usize), PagedLsaEntry>>> =
        OnceLock::new();
    GLOBAL_PAGED_POOL.get_or_init(|| RwLock::new(HashMap::new()))
}

fn pending_paged_ops() -> &'static std::sync::Mutex<Vec<PendingPagedOp>> {
    static PENDING_PAGED_OPS: OnceLock<std::sync::Mutex<Vec<PendingPagedOp>>> = OnceLock::new();
    PENDING_PAGED_OPS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

fn global_gpu_mask_cache() -> &'static RwLock<HashMap<(usize, usize, usize), Tensor>> {
    static GLOBAL_MASK_CACHE: OnceLock<RwLock<HashMap<(usize, usize, usize), Tensor>>> =
        OnceLock::new();
    GLOBAL_MASK_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn get_cached_causal_mask(
    device: &Device,
    q_len: usize,
    kv_len: usize,
    q_offset: usize,
) -> Result<Tensor> {
    let key = (q_len, kv_len, q_offset);
    if let Ok(guard) = global_gpu_mask_cache().read() {
        if let Some(m) = guard.get(&key) {
            if m.device().same_device(device) {
                return Ok(m.clone());
            }
        }
    }
    let mut vals = vec![f32::NEG_INFINITY; q_len * kv_len];
    for qi in 0..q_len {
        let max_k = (q_offset + qi).min(kv_len.saturating_sub(1));
        let row = &mut vals[qi * kv_len..qi * kv_len + max_k + 1];
        row.fill(0.0);
    }
    let mask = Tensor::from_vec(vals, (1, 1, q_len, kv_len), device)?;
    if let Ok(mut guard) = global_gpu_mask_cache().write() {
        if guard.len() > 64 {
            guard.clear();
        }
        guard.insert(key, mask.clone());
    }
    Ok(mask)
}

/// Bumped by `install` and `clear`: every session older than this is stale.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// The generation the pools were last emptied for.
static SETTLED: AtomicU64 = AtomicU64::new(0);

/// Apply deferred paged-LSA rollback/eviction updates on the attention thread.
pub(crate) fn drain_paged_lsa_ops() {
    let ops = match pending_paged_ops().lock() {
        Ok(mut q) if !q.is_empty() => std::mem::take(&mut *q),
        _ => return,
    };
    if let Ok(mut pool) = global_paged_session_pool().write() {
        for op in ops {
            match op {
                PendingPagedOp::Evict { block_0 } => {
                    pool.retain(|&(_, b0), _| b0 != block_0);
                }
                PendingPagedOp::Rollback { block_0, keep_len } => {
                    if keep_len == 0 {
                        pool.retain(|&(_, b0), _| b0 != block_0);
                    } else {
                        for (&(_, b0), entry) in pool.iter_mut() {
                            if b0 == block_0 {
                                let _ = entry.session.truncate(keep_len);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `install` and `clear` only mark the sessions stale; this empties the pools, and it runs
/// on the thread that runs attention.
///
/// Session tensors live on the device. A CUDA buffer freed on a thread with no context
/// bound — the HTTP handler serving `POST`/`DELETE /v1/lsa` — fails inside its `Drop`,
/// cudarc records that error on the context, and the NEXT allocation returns it: the first
/// request after every `/v1/lsa` change answered 500 "Failed to allocate preallocated KV
/// cache". Emptying the pools here frees them where the context is bound.
pub(crate) fn settle() {
    let generation = GENERATION.load(Ordering::Acquire);
    if SETTLED.load(Ordering::Acquire) != generation {
        if let Ok(mut guard) = global_session_pool().write() {
            guard.clear();
        }
        if let Ok(mut guard) = global_gpu_session_pool().write() {
            guard.clear();
        }
        if let Ok(mut guard) = global_paged_session_pool().write() {
            guard.clear();
        }
        if let Ok(mut q) = pending_paged_ops().lock() {
            q.clear();
        }
        SETTLED.store(generation, Ordering::Release);
    }
    drain_paged_lsa_ops();
}

fn clear_global_session_pool() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

fn tensor_storage_key(t: &Tensor) -> usize {
    let (storage, _) = t.storage_and_layout();
    &*storage as *const _ as usize
}

/// A session and the K tensor it was built from.
///
/// Sessions are keyed by the address of K's storage, and an address names a buffer only
/// while that buffer is alive. A KV cache that grows frees its old buffer, and the
/// allocator hands the address to the next buffer it makes: another layer's, or the next
/// request's. A session keyed by a freed buffer then served that buffer's owner with
/// another layer's keys and values, so identical temperature-0 requests decoded garbage
/// once the cache grew. Holding `_source` keeps the buffer, and so its address, alive for
/// exactly as long as the session that is keyed by it.
struct Pinned<S> {
    session: S,
    _source: Tensor,
    used: u64,
}

/// Calls after which an untouched session is dropped. A live layer is touched every step,
/// so this drops sessions whose buffer the cache has replaced or whose sequence is gone;
/// dropping a live one costs a rebuild from K, never a wrong answer.
const SESSION_IDLE_CALLS: u64 = 256;

static SESSION_CALLS: AtomicU64 = AtomicU64::new(0);

/// The session for the storage `k` lives in.
///
/// A decode step (`t_q == 1`) keeps one, made by `new` when there is none, because its
/// layer names the same cache buffer step after step. A prompt only updates one that is
/// already there: a prompt's K is often a one-shot tensor (a gather from the paged cache)
/// that no later call names again, and pinning it would hold that copy for nothing.
fn session_for<'p, S>(
    pool: &'p mut HashMap<usize, Pinned<S>>,
    k: &Tensor,
    t_q: usize,
    new: impl FnOnce() -> S,
) -> Option<&'p mut S> {
    let now = SESSION_CALLS.fetch_add(1, Ordering::Relaxed);
    pool.retain(|_, p| now.saturating_sub(p.used) <= SESSION_IDLE_CALLS);
    let key = tensor_storage_key(k);
    let pinned = if t_q == 1 {
        pool.entry(key).or_insert_with(|| Pinned {
            session: new(),
            _source: k.clone(),
            used: now,
        })
    } else {
        pool.get_mut(&key)?
    };
    pinned._source = k.clone();
    pinned.used = now;
    Some(&mut pinned.session)
}

/// `x` with each `[T, D]` matrix in row-major order, which the matmuls need. K and V
/// gathered from the paged cache arrive as a transposed view (`[B, T, H, D]` storage read
/// as `[B, H, T, D]`); a cache view narrowed along T already qualifies and is not copied.
fn rows(x: &Tensor) -> Result<Tensor> {
    let stride = x.stride();
    if stride[3] == 1 && stride[2] == x.dim(3)? {
        Ok(x.clone())
    } else {
        x.contiguous()
    }
}

impl LsaConfig {
    /// Preset for Hanzo Zen 6 (27B) (`rotary_dim = 64` Prefix, `window_size = 512`,
    /// `sink_size = 4`, `top_k = 512`, `top_p = 128`, `min_seq_len = 16384`).
    pub fn zen_27b() -> Self {
        Self {
            rotary_layout: RotaryLayout::Prefix,
            rotary_dim: 64,
            min_seq_len: 16_384,
            ..Self::default()
        }
    }

    /// Alias for `zen_27b()` (Hanzo Zen 6 27B).
    pub fn zen6_27b() -> Self {
        Self::zen_27b()
    }

    /// Preset for Hanzo Zen 6 Nano (`2B` / `4B` / `9B`), sharing the 64-dim prefix RoPE layout.
    pub fn zen6_nano() -> Self {
        Self::zen_27b()
    }

    /// Effective minimum KV length above which LSA activates for a sequence.
    pub fn min_trigger(&self) -> usize {
        self.min_seq_len
            .max(self.window_size + self.sink_size + self.top_k)
    }

    /// Whether LSA should trigger for a sequence of `kv_len` tokens under this config.
    pub fn active_for_seq_len(&self, kv_len: usize) -> bool {
        kv_len > self.min_trigger()
    }

    pub fn with_min_seq_len(mut self, min_seq_len: usize) -> Self {
        self.min_seq_len = min_seq_len;
        self
    }

    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn with_sink_size(mut self, sink_size: usize) -> Self {
        self.sink_size = sink_size;
        self
    }

    pub fn with_budget(mut self, top_p: usize, top_k: usize) -> Self {
        self.top_p = top_p;
        self.top_k = top_k;
        self
    }

    pub fn with_early_budget(mut self, position: usize, top_p: usize, top_k: usize) -> Self {
        self.early_budget = Some(EarlyBudget {
            position,
            top_p,
            top_k,
        });
        self
    }

    pub fn with_rotary(mut self, rotary_dim: usize, rotary_layout: RotaryLayout) -> Self {
        self.rotary_dim = rotary_dim;
        self.rotary_layout = rotary_layout;
        self
    }

    pub fn with_tree(
        mut self,
        top_level: usize,
        node_bucket_size: usize,
        node_cluster_block_size: usize,
        beam_width: usize,
        start_nodes: usize,
    ) -> Self {
        self.top_level = top_level;
        self.node_bucket_size = node_bucket_size;
        self.node_cluster_block_size = node_cluster_block_size;
        self.beam_width = beam_width;
        self.start_nodes = start_nodes;
        self
    }

    pub fn with_bucket_shape(mut self, rank: usize, deviations: f32) -> Self {
        self.bucket_shape = Some((rank, deviations));
        self
    }

    pub fn with_pca_index_dim(mut self, dim: usize) -> Self {
        self.pca_index_dim = Some(dim);
        self
    }

    pub fn with_projections_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.projections_path = Some(path.into());
        self
    }

    /// Parse an `LsaConfig` from a TOML string.
    pub fn from_toml_str(toml_str: &str) -> std::result::Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }

    /// Parse an `LsaConfig` from a YAML string.
    pub fn from_yaml_str(yaml_str: &str) -> std::result::Result<Self, serde_saphyr::Error> {
        serde_saphyr::from_str(yaml_str)
    }

    /// Install this `LsaConfig` as the active engine-wide LSA routing configuration.
    pub fn install(self) {
        if let Ok(mut guard) = global_lsa_slot().write() {
            *guard = Some(Arc::new(self));
        }
        clear_global_session_pool();
    }

    /// Disable engine-wide LSA routing and return to standard attention.
    pub fn clear() {
        if let Ok(mut guard) = global_lsa_slot().write() {
            *guard = None;
        }
        clear_global_session_pool();
    }

    /// Read the currently installed engine-wide `LsaConfig`, if any.
    pub fn global() -> Option<Arc<Self>> {
        global_lsa_slot().read().ok()?.clone()
    }

    /// How many attention calls LSA has served since the process started: the
    /// evidence that a long context took the sparse path rather than dense SDPA.
    pub fn dispatches() -> u64 {
        DISPATCHES.load(Ordering::Relaxed)
    }
}

static DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Linear projections `W_Q^{(idx)}` and `W_K^{(idx)}` from `[input_dim]` to `[index_dim]`.
#[derive(Debug, Clone)]
pub struct LearnedProjection {
    pub query_proj: Tensor,
    pub key_proj: Tensor,
}

impl LearnedProjection {
    pub fn new(query_proj: Tensor, key_proj: Tensor) -> Result<Self> {
        if query_proj.dims() != key_proj.dims() || query_proj.rank() != 2 {
            hanzo_ml::bail!(
                "LearnedProjection requires matching 2D [input_dim, index_dim] tensors, got {:?} and {:?}",
                query_proj.dims(),
                key_proj.dims()
            );
        }
        Ok(Self {
            query_proj,
            key_proj,
        })
    }

    pub fn index_dim(&self) -> Result<usize> {
        self.query_proj.dim(1)
    }

    /// Load per-layer `<layer>.W_Q_idx` and `<layer>.W_K_idx` from a `.safetensors` file.
    pub fn load_safetensors<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> Result<HashMap<usize, Self>> {
        let tensors = hanzo_ml::safetensors::load(path, device)?;
        let mut by_layer = HashMap::new();
        for (name, w_q) in &tensors {
            if let Some(prefix) = name.strip_suffix(".W_Q_idx") {
                if let Ok(layer_idx) = prefix.parse::<usize>() {
                    let k_name = format!("{layer_idx}.W_K_idx");
                    let w_k = tensors.get(&k_name).ok_or_else(|| {
                        Error::msg(format!(
                            "missing {k_name} alongside {name} in LSA projections"
                        ))
                    })?;
                    by_layer.insert(layer_idx, Self::new(w_q.clone(), w_k.clone())?);
                }
            }
        }
        Ok(by_layer)
    }
}

/// Normalize each row (last axis) to unit Euclidean length; zero rows stay zero.
pub fn rownorm(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.sqr()?;
    let norm = sq
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(NORM_EPS as f64, f64::MAX)?;
    x_f32.broadcast_div(&norm)?.to_dtype(dtype)
}

/// Keep the unrotated slice of a `[..., head_dim]` tensor.
pub fn position_free(x: &Tensor, rotary_dim: usize, layout: RotaryLayout) -> Result<Tensor> {
    if rotary_dim == 0 || layout == RotaryLayout::None {
        return Ok(x.clone());
    }
    let d = x.dim(D::Minus1)?;
    if rotary_dim >= d {
        return Ok(x.clone());
    }
    match layout {
        RotaryLayout::Prefix => x.narrow(D::Minus1, rotary_dim, d - rotary_dim),
        RotaryLayout::Suffix => x.narrow(D::Minus1, 0, d - rotary_dim),
        RotaryLayout::None => Ok(x.clone()),
    }
}

/// Regroup GQA heads into a single unit-norm index vector per position.
///
/// - `queries`: `[B, Hq, Tq, D]`
/// - `keys`: `[B, Hkv, Tk, D]`
/// - Returns `(index_queries [B, Tq, Hkv * D], index_keys [B, Tk, Hkv * D])`.
pub fn regroup(queries: &Tensor, keys: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, h_q, t_q, d) = queries.dims4()?;
    let (b_k, h_kv, t_k, d_k) = keys.dims4()?;
    if b != b_k || d != d_k || h_kv == 0 || h_q % h_kv != 0 {
        hanzo_ml::bail!(
            "regroup shape mismatch: queries {:?} vs keys {:?}",
            queries.dims(),
            keys.dims()
        );
    }
    let group_size = h_q / h_kv;
    let grouped_q = queries
        .reshape((b, h_kv, group_size, t_q, d))?
        .sum(2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, t_q, h_kv * d))?;
    let flat_k = keys
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, t_k, h_kv * d))?;
    Ok((rownorm(&grouped_q)?, rownorm(&flat_k)?))
}

/// Apply `LearnedProjection` to regrouped queries and keys and L2 row-normalize.
pub fn project(
    queries: &Tensor,
    keys: &Tensor,
    projection: &LearnedProjection,
) -> Result<(Tensor, Tensor)> {
    let (b, h_q, t_q, d) = queries.dims4()?;
    let (_, h_kv, t_k, _) = keys.dims4()?;
    let group_size = h_q / h_kv;

    let q_in = queries
        .reshape((b, h_kv, group_size, t_q, d))?
        .sum(2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t_q, h_kv * d))?
        .to_dtype(projection.query_proj.dtype())?;
    let k_in = keys
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t_k, h_kv * d))?
        .to_dtype(projection.key_proj.dtype())?;

    let d_idx = projection.index_dim()?;
    let q_idx = q_in
        .matmul(&projection.query_proj)?
        .reshape((b, t_q, d_idx))?;
    let k_idx = k_in
        .matmul(&projection.key_proj)?
        .reshape((b, t_k, d_idx))?;
    Ok((rownorm(&q_idx)?, rownorm(&k_idx)?))
}

/// Top-`rank` eigenvalues and eigenvectors of a symmetric PSD matrix `a` (`[d, d]`).
fn top_k_symmetric_eigen(a: &[f32], d: usize, rank: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mat = a.to_vec();
    let mut vecs = vec![0.0f32; rank * d];
    let mut vals = vec![0.0f32; rank];

    for r in 0..rank {
        let mut v = vec![0.0f32; d];
        for i in 0..d {
            v[i] = ((i + r * 7 + 1) as f32).sin() + if i == r % d { 2.0 } else { 0.0 };
        }
        for prev in 0..r {
            let pv = &vecs[prev * d..(prev + 1) * d];
            let dot: f32 = v.iter().zip(pv.iter()).map(|(x, y)| x * y).sum();
            for i in 0..d {
                v[i] -= dot * pv[i];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }

        let mut eigval = 0.0f32;
        for _iter in 0..32 {
            let mut w = vec![0.0f32; d];
            for i in 0..d {
                let row = &mat[i * d..(i + 1) * d];
                w[i] = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            }
            for prev in 0..r {
                let pv = &vecs[prev * d..(prev + 1) * d];
                let dot: f32 = w.iter().zip(pv.iter()).map(|(x, y)| x * y).sum();
                for i in 0..d {
                    w[i] -= dot * pv[i];
                }
            }
            let w_norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if w_norm < 1e-12 {
                eigval = 0.0;
                break;
            }
            eigval = w_norm;
            let mut diff = 0.0f32;
            for i in 0..d {
                let next_vi = w[i] / w_norm;
                diff += (next_vi - v[i]).abs();
                v[i] = next_vi;
            }
            if diff < 1e-6 {
                break;
            }
        }

        let mut max_idx = 0usize;
        let mut max_abs = -1.0f32;
        for (i, &val) in v.iter().enumerate() {
            if val.abs() > max_abs {
                max_abs = val.abs();
                max_idx = i;
            }
        }
        if v[max_idx] < 0.0 {
            for val in &mut v {
                *val = -*val;
            }
        }

        for i in 0..d {
            for j in 0..d {
                mat[i * d + j] -= eigval * v[i] * v[j];
            }
        }

        vecs[r * d..(r + 1) * d].copy_from_slice(&v);
        vals[r] = eigval.max(0.0);
    }

    (vecs, vals)
}

/// Training-free PCA initialization of `LearnedProjection` (`W_Q = W_K` top `index_dim` principal directions).
pub fn pca_projections(keys_in: &Tensor, index_dim: usize) -> Result<LearnedProjection> {
    let dims = keys_in.dims();
    let d_in = *dims.last().ok_or_else(|| Error::msg("empty keys_in"))?;
    let n_rows = keys_in.elem_count() / d_in;
    if index_dim > d_in {
        hanzo_ml::bail!("index_dim ({index_dim}) cannot exceed input_dim ({d_in})");
    }
    let flat = keys_in
        .to_dtype(DType::F32)?
        .reshape((n_rows, d_in))?
        .to_vec2::<f32>()?;

    let mut cov = vec![0.0f32; d_in * d_in];
    let inv_n = 1.0 / (n_rows.max(1) as f32);
    for row in &flat {
        for i in 0..d_in {
            let ri = row[i] * inv_n;
            let base = i * d_in;
            for j in i..d_in {
                cov[base + j] += ri * row[j];
            }
        }
    }
    for i in 0..d_in {
        for j in 0..i {
            cov[i * d_in + j] = cov[j * d_in + i];
        }
    }

    let (eigvecs, _) = top_k_symmetric_eigen(&cov, d_in, index_dim);
    let mut w = vec![0.0f32; d_in * index_dim];
    for r in 0..index_dim {
        for i in 0..d_in {
            w[i * index_dim + r] = eigvecs[r * d_in + i];
        }
    }
    let proj =
        Tensor::from_vec(w, (d_in, index_dim), keys_in.device())?.to_dtype(keys_in.dtype())?;
    LearnedProjection::new(proj.clone(), proj)
}

/// Core mutual-cheapest Ward merge on one block of `P` pools.
pub fn ward_rounds_block(
    mut sums: Vec<f32>,
    mut sizes: Vec<i32>,
    p: usize,
    d: usize,
    num_buckets: usize,
    bucket_cap: i32,
) -> Result<(Vec<f32>, Vec<i32>, Vec<usize>)> {
    if num_buckets == 0 || num_buckets > p {
        hanzo_ml::bail!("num_buckets ({num_buckets}) must be in 1..={p}");
    }
    let mut alive = vec![true; p];
    let mut alive_count = p;
    let mut owner: Vec<usize> = (0..p).collect();

    let mut norms = vec![0.0f32; p];
    for b in 0..p {
        let row = &sums[b * d..(b + 1) * d];
        norms[b] = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    }

    let mut gram = vec![0.0f32; p * p];
    for b in 0..p {
        let rb = &sums[b * d..(b + 1) * d];
        for bp in b..p {
            let rbp = &sums[bp * d..(bp + 1) * d];
            let dot: f32 = rb.iter().zip(rbp.iter()).map(|(x, y)| x * y).sum();
            gram[b * p + bp] = dot;
            gram[bp * p + b] = dot;
        }
    }

    let mut best_partner = vec![usize::MAX; p];
    let mut best_cost = vec![f32::INFINITY; p];

    while alive_count > num_buckets {
        let needed = alive_count - num_buckets;

        for b in 0..p {
            best_partner[b] = usize::MAX;
            best_cost[b] = f32::INFINITY;
            if !alive[b] {
                continue;
            }
            let nb = norms[b];
            let sb = sizes[b];
            for bp in 0..p {
                if bp == b || !alive[bp] || sb + sizes[bp] > bucket_cap {
                    continue;
                }
                let nbp = norms[bp];
                let sq = (nb * nb + nbp * nbp + 2.0 * gram[b * p + bp]).max(MERGE_COST_EPS);
                let cost = nb + nbp - sq.sqrt();
                if cost < best_cost[b] {
                    best_cost[b] = cost;
                    best_partner[b] = bp;
                }
            }
        }

        let mut mutual: Vec<(f32, usize, usize)> = Vec::new();
        for b in 0..p {
            if !alive[b] {
                continue;
            }
            let bp = best_partner[b];
            if bp != usize::MAX && b < bp && alive[bp] && best_partner[bp] == b {
                mutual.push((best_cost[b], b, bp));
            }
        }

        if mutual.is_empty() {
            hanzo_ml::bail!(
                "ward_rounds_block: no allowed mutual pair while alive={alive_count} > target={num_buckets}"
            );
        }

        mutual.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let merges = mutual.len().min(needed);
        for &(_cost, b, bp) in &mutual[..merges] {
            for k in 0..d {
                sums[b * d + k] += sums[bp * d + k];
                sums[bp * d + k] = 0.0;
            }
            sizes[b] += sizes[bp];
            sizes[bp] = 0;
            alive[bp] = false;
            alive_count -= 1;

            for other in 0..p {
                if other != b && alive[other] {
                    let g = gram[b * p + other] + gram[bp * p + other];
                    gram[b * p + other] = g;
                    gram[other * p + b] = g;
                }
            }
            let rb = &sums[b * d..(b + 1) * d];
            let self_dot: f32 = rb.iter().map(|x| x * x).sum();
            gram[b * p + b] = self_dot;
            norms[b] = self_dot.sqrt();

            for pool_owner in &mut owner {
                if *pool_owner == bp {
                    *pool_owner = b;
                }
            }
        }
    }

    let mut survivor_rank = vec![usize::MAX; p];
    let mut surviving_sums = Vec::with_capacity(num_buckets * d);
    let mut surviving_sizes = Vec::with_capacity(num_buckets);
    let mut rank = 0usize;
    for b in 0..p {
        if alive[b] {
            survivor_rank[b] = rank;
            surviving_sums.extend_from_slice(&sums[b * d..(b + 1) * d]);
            surviving_sizes.push(sizes[b]);
            rank += 1;
        }
    }

    let labels: Vec<usize> = owner.into_iter().map(|o| survivor_rank[o]).collect();
    Ok((surviving_sums, surviving_sizes, labels))
}

/// Convert pool labels `[P]` into padded global key indices `[m][bucket_cap]`.
pub fn members_from_labels_block(
    labels: &[usize],
    num_buckets: usize,
    pool_size: usize,
    bucket_cap: usize,
    block_offset: i32,
) -> Vec<Vec<i32>> {
    let mut buckets = vec![Vec::with_capacity(bucket_cap); num_buckets];
    for (pool_idx, &bucket_id) in labels.iter().enumerate() {
        let base = block_offset + (pool_idx * pool_size) as i32;
        for offset in 0..(pool_size as i32) {
            if buckets[bucket_id].len() < bucket_cap {
                buckets[bucket_id].push(base + offset);
            }
        }
    }
    for member_list in &mut buckets {
        member_list.resize(bucket_cap, -1);
    }
    buckets
}

/// Low-rank 1-bit sign basis + residual floor for within-bucket key spread.
#[derive(Debug, Clone)]
pub struct Shapes {
    pub directions: Tensor,
    pub variances: Tensor,
    pub floor: Tensor,
}

/// Clustered buckets produced by `build_buckets` across a batch `[B, T, d_idx]`.
#[derive(Debug, Clone)]
pub struct Buckets {
    pub centroids: Tensor,
    pub bucket_members: Vec<Vec<Vec<i32>>>,
    pub bucket_sizes: Vec<Vec<i32>>,
    pub buckets_per_block: usize,
    pub num_closed_blocks: usize,
    pub shapes: Option<Vec<Shapes>>,
}

/// Build Ward-clustered buckets for closed blocks of `index_keys` `[B, T, d_idx]`.
pub fn build_buckets(
    index_keys: &Tensor,
    window_size: usize,
    pool_size: usize,
    bucket_size: usize,
    bucket_cap: usize,
) -> Result<Buckets> {
    let (b_sz, seq_len, d_idx) = index_keys.dims3()?;
    if window_size == 0 || window_size % pool_size != 0 || window_size % bucket_size != 0 {
        hanzo_ml::bail!(
            "window_size ({window_size}) must be divisible by pool_size ({pool_size}) and bucket_size ({bucket_size})"
        );
    }
    if bucket_cap < 2 * bucket_size {
        hanzo_ml::bail!(
            "bucket_cap ({bucket_cap}) must be >= 2 * bucket_size ({})",
            2 * bucket_size
        );
    }
    let num_closed_blocks = seq_len / window_size;
    let buckets_per_block = window_size / bucket_size;
    let total_buckets = num_closed_blocks * buckets_per_block;

    if num_closed_blocks == 0 {
        let centroids = Tensor::zeros((b_sz, 0, d_idx), index_keys.dtype(), index_keys.device())?;
        return Ok(Buckets {
            centroids,
            bucket_members: vec![Vec::new(); b_sz],
            bucket_sizes: vec![Vec::new(); b_sz],
            buckets_per_block,
            num_closed_blocks: 0,
            shapes: None,
        });
    }

    let closed_len = num_closed_blocks * window_size;
    let n_blocks = b_sz * num_closed_blocks;
    let p = window_size / pool_size;
    let m = buckets_per_block;

    let pool_sums = index_keys
        .narrow(1, 0, closed_len)?
        .contiguous()?
        .reshape((n_blocks, p, pool_size, d_idx))?
        .sum(2)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let per_block: Result<Vec<(Vec<f32>, Vec<i32>, Vec<Vec<i32>>)>> = (0..n_blocks)
        .into_par_iter()
        .map(|flat_blk| {
            let blk = flat_blk % num_closed_blocks;
            let start = flat_blk * p * d_idx;
            let block_sums = pool_sums[start..start + p * d_idx].to_vec();
            let block_sizes = vec![pool_size as i32; p];
            let (mut surv_sums, surv_sizes, labels) =
                ward_rounds_block(block_sums, block_sizes, p, d_idx, m, bucket_cap as i32)?;
            for b in 0..m {
                let slice = &mut surv_sums[b * d_idx..(b + 1) * d_idx];
                let norm = slice
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(CENTROID_EPS);
                for val in slice.iter_mut() {
                    *val /= norm;
                }
            }
            let members = members_from_labels_block(
                &labels,
                m,
                pool_size,
                bucket_cap,
                (blk * window_size) as i32,
            );
            Ok((surv_sums, surv_sizes, members))
        })
        .collect();

    let per_block = per_block?;
    let mut all_centroids = Vec::with_capacity(n_blocks * m * d_idx);
    let mut bucket_members = vec![Vec::with_capacity(total_buckets); b_sz];
    let mut bucket_sizes = vec![Vec::with_capacity(total_buckets); b_sz];

    for (flat_blk, (c, sz, mem)) in per_block.into_iter().enumerate() {
        let b = flat_blk / num_closed_blocks;
        all_centroids.extend_from_slice(&c);
        bucket_sizes[b].extend_from_slice(&sz);
        bucket_members[b].extend_from_slice(&mem);
    }

    let centroids = Tensor::from_vec(
        all_centroids,
        (b_sz, total_buckets, d_idx),
        index_keys.device(),
    )?
    .to_dtype(index_keys.dtype())?;

    Ok(Buckets {
        centroids,
        bucket_members,
        bucket_sizes,
        buckets_per_block,
        num_closed_blocks,
        shapes: None,
    })
}

/// Build 1-bit sign-quantized covariance shapes for a single batch element's buckets.
pub fn build_shapes(
    index_keys: &Tensor,
    centroids: &Tensor,
    bucket_members: &[Vec<i32>],
    bucket_sizes: &[i32],
    rank: usize,
) -> Result<Shapes> {
    let (_t, d) = index_keys.dims2()?;
    let (b_total, d_c) = centroids.dims2()?;
    if d != d_c || rank == 0 || rank >= d {
        hanzo_ml::bail!("build_shapes requires 0 < rank ({rank}) < d_idx ({d})");
    }

    let keys_flat = index_keys
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let centroids_flat = centroids
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scale = 1.0f32 / (d as f32).sqrt();

    let per_bucket: Vec<(Vec<f32>, Vec<f32>, f32)> = (0..b_total)
        .into_par_iter()
        .map(|b| {
            let count = bucket_sizes[b].max(1) as f32;
            let c = &centroids_flat[b * d..(b + 1) * d];
            let members = &bucket_members[b];

            let mut cov = vec![0.0f32; d * d];
            for &idx in members {
                if idx < 0 {
                    continue;
                }
                let k = &keys_flat[(idx as usize) * d..(idx as usize + 1) * d];
                for i in 0..d {
                    let di = k[i] - c[i];
                    let row_off = i * d;
                    for j in i..d {
                        cov[row_off + j] += di * (k[j] - c[j]);
                    }
                }
            }
            let inv_count = 1.0 / count;
            let mut trace = 0.0f32;
            for i in 0..d {
                cov[i * d + i] *= inv_count;
                trace += cov[i * d + i];
                for j in (i + 1)..d {
                    let v = cov[i * d + j] * inv_count;
                    cov[i * d + j] = v;
                    cov[j * d + i] = v;
                }
            }

            let (eigvecs, eigvals) = top_k_symmetric_eigen(&cov, d, rank);
            let mut dirs = vec![0.0f32; rank * d];
            for r in 0..rank {
                for i in 0..d {
                    let v = eigvecs[r * d + i];
                    dirs[r * d + i] = if v >= 0.0 { scale } else { -scale };
                }
            }
            let top_sum: f32 = eigvals.iter().sum();
            let floor = ((trace - top_sum) / ((d - rank) as f32)).max(0.0);
            (dirs, eigvals, floor)
        })
        .collect();

    let mut all_dirs = Vec::with_capacity(b_total * rank * d);
    let mut all_vars = Vec::with_capacity(b_total * rank);
    let mut all_floors = Vec::with_capacity(b_total);
    for (dirs, vars, fl) in per_bucket {
        all_dirs.extend_from_slice(&dirs);
        all_vars.extend_from_slice(&vars);
        all_floors.push(fl);
    }

    let device = index_keys.device();
    let dtype = index_keys.dtype();
    Ok(Shapes {
        directions: Tensor::from_vec(all_dirs, (b_total, rank, d), device)?.to_dtype(dtype)?,
        variances: Tensor::from_vec(all_vars, (b_total, rank), device)?.to_dtype(dtype)?,
        floor: Tensor::from_vec(all_floors, (b_total,), device)?.to_dtype(dtype)?,
    })
}

/// Attach `Shapes` to every batch element of `Buckets`.
pub fn shape_buckets(buckets: &mut Buckets, index_keys: &Tensor, rank: usize) -> Result<()> {
    let b_sz = index_keys.dim(0)?;
    let mut shapes_vec = Vec::with_capacity(b_sz);
    for b in 0..b_sz {
        let k_b = index_keys.narrow(0, b, 1)?.squeeze(0)?;
        let c_b = buckets.centroids.narrow(0, b, 1)?.squeeze(0)?;
        let s = build_shapes(
            &k_b,
            &c_b,
            &buckets.bucket_members[b],
            &buckets.bucket_sizes[b],
            rank,
        )?;
        shapes_vec.push(s);
    }
    buckets.shapes = Some(shapes_vec);
    Ok(())
}

/// Bucket score combining centroid alignment and directional spread: `q . c_b + deviations * spread(q, b)`.
pub fn shaped_scores(
    index_queries: &Tensor,
    centroids: &Tensor,
    shapes: &Shapes,
    deviations: f32,
) -> Result<Tensor> {
    let (t_q, d) = index_queries.dims2()?;
    let (b_total, rank, _) = shapes.directions.dims3()?;
    let q_f32 = index_queries.to_dtype(DType::F32)?;
    let c_f32 = centroids.to_dtype(DType::F32)?;
    let base = q_f32.matmul(&c_f32.t()?)?;
    if deviations == 0.0 || b_total == 0 {
        return Ok(base);
    }
    let dirs_flat = shapes
        .directions
        .to_dtype(DType::F32)?
        .reshape((b_total * rank, d))?;
    let proj_sq = q_f32
        .matmul(&dirs_flat.t()?)?
        .reshape((t_q, b_total, rank))?
        .sqr()?;
    let vars = shapes
        .variances
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .broadcast_as((t_q, b_total, rank))?;
    let captured = proj_sq.mul(&vars)?.sum(D::Minus1)?;
    let proj_norm_sq = proj_sq.sum(D::Minus1)?;
    let ones = Tensor::ones((t_q, b_total), DType::F32, index_queries.device())?;
    let residual_weight = ones.sub(&proj_norm_sq)?.clamp(0.0, 1.0)?;
    let floor = shapes
        .floor
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .broadcast_as((t_q, b_total))?;
    let spread = captured
        .add(&residual_weight.mul(&floor)?)?
        .clamp(0.0, f64::MAX)?
        .sqrt()?;
    base.add(&(spread * (deviations as f64))?)
}

/// Multi-level tree of node buckets built above level-0 buckets.
#[derive(Debug, Clone)]
pub struct Hierarchy {
    pub centroids: Vec<Tensor>,
    pub children: Vec<Vec<Vec<usize>>>,
    pub buckets_per_block: usize,
    pub node_cluster_block_size: usize,
    pub node_bucket_size: usize,
}

impl Hierarchy {
    pub fn top_level(&self) -> usize {
        self.centroids.len().saturating_sub(1)
    }
}

/// How much of each tree level exists when a query's cluster block opens.
#[derive(Debug, Clone)]
pub struct Growth {
    pub available_at_level: Vec<usize>,
    pub covered_by_parent: Vec<usize>,
    pub highest_grown_level: usize,
}

/// Pair `n_blocks` node blocks of `node_cluster_block_size` child sums into `node_bucket_size`-child node buckets.
pub fn pair_into_node_buckets(
    child_sums: &[f32],
    n_blocks: usize,
    node_cluster_block_size: usize,
    d: usize,
    node_bucket_size: usize,
) -> Result<(Vec<f32>, Vec<Vec<usize>>)> {
    if !node_bucket_size.is_power_of_two()
        || node_bucket_size < 2
        || node_cluster_block_size % node_bucket_size != 0
    {
        hanzo_ml::bail!(
            "invalid node_bucket_size ({node_bucket_size}) / node_cluster_block_size ({node_cluster_block_size})"
        );
    }
    let halvings = node_bucket_size.trailing_zeros() as usize;
    let out_per_block = node_cluster_block_size / node_bucket_size;

    let mut all_parent_sums = Vec::with_capacity(n_blocks * out_per_block * d);
    let mut all_children = Vec::with_capacity(n_blocks * out_per_block);

    for blk in 0..n_blocks {
        let base_child = blk * node_cluster_block_size;
        let mut cur_p = node_cluster_block_size;
        let mut cur_sums =
            child_sums[base_child * d..(base_child + node_cluster_block_size) * d].to_vec();
        let mut members: Vec<Vec<usize>> = (0..node_cluster_block_size)
            .map(|i| vec![base_child + i])
            .collect();

        for _ in 0..halvings {
            let next_p = cur_p / 2;
            let sizes = vec![1i32; cur_p];
            let (surv_sums, _surv_sizes, labels) =
                ward_rounds_block(cur_sums, sizes, cur_p, d, next_p, 2)?;
            let mut next_members = vec![Vec::new(); next_p];
            for (idx, &lbl) in labels.iter().enumerate() {
                next_members[lbl].extend_from_slice(&members[idx]);
            }
            cur_sums = surv_sums;
            members = next_members;
            cur_p = next_p;
        }

        all_parent_sums.extend_from_slice(&cur_sums);
        all_children.extend(members);
    }

    Ok((all_parent_sums, all_children))
}

/// Build a multi-level `Hierarchy` above level-0 buckets for a single sequence.
pub fn build_hierarchy(
    centroids_l0: &Tensor,
    bucket_sizes_l0: &[i32],
    buckets_per_block: usize,
    top_level: usize,
    node_bucket_size: usize,
    node_cluster_block_size: usize,
) -> Result<Hierarchy> {
    let (b_total, d) = centroids_l0.dims2()?;
    let device = centroids_l0.device();
    let dtype = centroids_l0.dtype();

    let c0_vec = centroids_l0
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut cur_sums = vec![0.0f32; b_total * d];
    for b in 0..b_total {
        let w = (bucket_sizes_l0[b].max(1)) as f32;
        for k in 0..d {
            cur_sums[b * d + k] = c0_vec[b * d + k] * w;
        }
    }
    let mut cur_count = b_total;

    let mut centroids = Vec::with_capacity(top_level + 1);
    let mut children = Vec::with_capacity(top_level + 1);
    centroids.push(centroids_l0.clone());
    children.push(Vec::new());

    for _level in 1..=top_level {
        let n_blocks = cur_count / node_cluster_block_size;
        if n_blocks == 0 {
            centroids.push(Tensor::zeros((0, d), dtype, device)?);
            children.push(Vec::new());
            cur_sums.clear();
            cur_count = 0;
            continue;
        }
        let (parent_sums, parent_children) = pair_into_node_buckets(
            &cur_sums,
            n_blocks,
            node_cluster_block_size,
            d,
            node_bucket_size,
        )?;
        let n_parents = parent_children.len();
        let mut parent_centroids = parent_sums.clone();
        for p in 0..n_parents {
            let slice = &mut parent_centroids[p * d..(p + 1) * d];
            let norm = slice
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(CENTROID_EPS);
            for v in slice.iter_mut() {
                *v /= norm;
            }
        }
        centroids
            .push(Tensor::from_vec(parent_centroids, (n_parents, d), device)?.to_dtype(dtype)?);
        children.push(parent_children);
        cur_sums = parent_sums;
        cur_count = n_parents;
    }

    Ok(Hierarchy {
        centroids,
        children,
        buckets_per_block,
        node_cluster_block_size,
        node_bucket_size,
    })
}

/// Compute `Growth` at sequence position `query_pos`.
pub fn growth_at(hierarchy: &Hierarchy, query_pos: usize, window_size: usize) -> Growth {
    let top_level = hierarchy.top_level();
    let mut available_at_level = vec![0usize; top_level + 1];
    let mut covered_by_parent = vec![0usize; top_level + 1];

    let closed_blocks = query_pos / window_size;
    available_at_level[0] = closed_blocks * hierarchy.buckets_per_block;

    let mut highest_grown_level = 0usize;
    for l in 1..=top_level {
        let child_avail = available_at_level[l - 1];
        let closed_node_blocks = child_avail / hierarchy.node_cluster_block_size;
        covered_by_parent[l - 1] = closed_node_blocks * hierarchy.node_cluster_block_size;
        available_at_level[l] =
            closed_node_blocks * (hierarchy.node_cluster_block_size / hierarchy.node_bucket_size);
        if available_at_level[l] > 0 {
            highest_grown_level = l;
        }
    }

    Growth {
        available_at_level,
        covered_by_parent,
        highest_grown_level,
    }
}

/// Top-down beam search over `Hierarchy` for a single query vector `q` (`[d_idx]`).
pub fn beam_search_single_query(
    q: &[f32],
    l0_centroids: &[f32],
    level_centroids: &[Vec<f32>],
    hierarchy: &Hierarchy,
    growth: &Growth,
    top_p: usize,
    beam_width: usize,
    start_nodes: usize,
) -> Vec<usize> {
    let d = q.len();
    let avail_l0 = growth.available_at_level[0];
    if avail_l0 == 0 || top_p == 0 {
        return Vec::new();
    }

    let mut start_level = 0usize;
    if growth.highest_grown_level >= 1 {
        start_level = growth.highest_grown_level;
        for l in 1..=growth.highest_grown_level {
            if growth.available_at_level[l] > 0 && growth.available_at_level[l] <= start_nodes {
                start_level = l;
                break;
            }
        }
    }

    let dot = |centroids_flat: &[f32], idx: usize| -> f32 {
        let row = &centroids_flat[idx * d..(idx + 1) * d];
        q.iter().zip(row.iter()).map(|(a, b)| a * b).sum()
    };

    if start_level == 0 {
        let mut scored: Vec<(f32, usize)> =
            (0..avail_l0).map(|b| (dot(l0_centroids, b), b)).collect();
        let keep = top_p.min(scored.len());
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        return scored[..keep].iter().map(|&(_, b)| b).collect();
    }

    let avail_start = growth.available_at_level[start_level];
    let mut beam: Vec<usize> = {
        let c_flat = &level_centroids[start_level];
        let mut scored: Vec<(f32, usize)> = (0..avail_start)
            .map(|idx| (dot(c_flat, idx), idx))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let keep = beam_width.min(scored.len());
        scored[..keep].iter().map(|&(_, idx)| idx).collect()
    };

    for level in (1..start_level).rev() {
        let mut candidates: Vec<usize> = Vec::with_capacity(
            beam.len() * hierarchy.node_bucket_size
                + (growth.available_at_level[level] - growth.covered_by_parent[level]),
        );
        for &parent_idx in &beam {
            candidates.extend_from_slice(&hierarchy.children[level + 1][parent_idx]);
        }
        for open_idx in growth.covered_by_parent[level]..growth.available_at_level[level] {
            candidates.push(open_idx);
        }
        let c_flat = &level_centroids[level];
        let mut scored: Vec<(f32, usize)> = candidates
            .into_iter()
            .map(|idx| (dot(c_flat, idx), idx))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let keep = beam_width.min(scored.len());
        beam = scored[..keep].iter().map(|&(_, idx)| idx).collect();
    }

    let mut l0_candidates: Vec<usize> = Vec::with_capacity(
        beam.len() * hierarchy.node_bucket_size
            + (growth.available_at_level[0] - growth.covered_by_parent[0]),
    );
    for &parent_idx in &beam {
        l0_candidates.extend_from_slice(&hierarchy.children[1][parent_idx]);
    }
    for open_idx in growth.covered_by_parent[0]..growth.available_at_level[0] {
        l0_candidates.push(open_idx);
    }

    let mut scored_l0: Vec<(f32, usize)> = l0_candidates
        .into_iter()
        .map(|b| (dot(l0_centroids, b), b))
        .collect();
    scored_l0.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let keep = top_p.min(scored_l0.len());
    scored_l0[..keep].iter().map(|&(_, b)| b).collect()
}

#[inline]
pub fn budget_at(
    pos: usize,
    top_p: usize,
    top_k: usize,
    early_budget: Option<EarlyBudget>,
) -> (usize, usize) {
    if let Some(eb) = early_budget {
        if pos < eb.position {
            return (eb.top_p, eb.top_k);
        }
    }
    (top_p, top_k)
}

/// GPU-resident descending top-`k` column indices along the last dimension of `scores` (`[rows, cols]`).
///
/// `hanzo_ml`'s single-block bitonic `arg_sort_last_dim` allocates `next_power_of_2(cols) * 4` bytes
/// of dynamic shared memory per block, which exceeds CUDA's 48 KiB default shared-memory limit when
/// `cols > 8192` (e.g., `64K+` / `128K` context under Paged-LSA). Tiling reductions into 2048-column
/// chunks keeps shared memory at 8 KiB while staying 100% on-device.
fn top_k_indices_last_dim(scores: &Tensor, k: usize) -> Result<Tensor> {
    const MAX_SINGLE_PASS: usize = 4096;
    const SAFE_SORT_TILE: usize = 2048;

    let (rows, mut cur_cols) = scores.dims2()?;
    let k = k.min(cur_cols);
    let dev = scores.device();
    if k == 0 {
        return Tensor::zeros((rows, 0), DType::U32, dev);
    }
    if cur_cols <= MAX_SINGLE_PASS {
        return scores
            .arg_sort_last_dim(false)?
            .narrow(1, 0, k)?
            .contiguous();
    }
    if k > SAFE_SORT_TILE / 2 {
        let rows_vec = scores.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut flat = Vec::with_capacity(rows * k);
        for row in &rows_vec {
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            for &idx in order.iter().take(k) {
                flat.push(idx as u32);
            }
        }
        return Tensor::from_vec(flat, (rows, k), dev);
    }

    let mut cur_scores = scores.contiguous()?;
    let mut cur_idx = Tensor::arange(0u32, cur_cols as u32, dev)?
        .unsqueeze(0)?
        .broadcast_as((rows, cur_cols))?
        .contiguous()?;

    while cur_cols > MAX_SINGLE_PASS {
        let n_full = cur_cols / SAFE_SORT_TILE;
        let rem = cur_cols % SAFE_SORT_TILE;
        let full_w = n_full * SAFE_SORT_TILE;
        let full_s = cur_scores
            .narrow(1, 0, full_w)?
            .contiguous()?
            .reshape((rows * n_full, SAFE_SORT_TILE))?;
        let full_i = cur_idx
            .narrow(1, 0, full_w)?
            .contiguous()?
            .reshape((rows * n_full, SAFE_SORT_TILE))?;
        let local_k = full_s
            .arg_sort_last_dim(false)?
            .narrow(1, 0, k)?
            .contiguous()?;
        let best_s = full_s.gather(&local_k, 1)?.reshape((rows, n_full * k))?;
        let best_i = full_i.gather(&local_k, 1)?.reshape((rows, n_full * k))?;

        if rem > 0 {
            let tail_s = cur_scores.narrow(1, full_w, rem)?.contiguous()?;
            let tail_i = cur_idx.narrow(1, full_w, rem)?.contiguous()?;
            let t_take = k.min(rem);
            let tail_k = tail_s
                .arg_sort_last_dim(false)?
                .narrow(1, 0, t_take)?
                .contiguous()?;
            let ts = tail_s.gather(&tail_k, 1)?;
            let ti = tail_i.gather(&tail_k, 1)?;
            cur_scores = Tensor::cat(&[&best_s, &ts], 1)?;
            cur_idx = Tensor::cat(&[&best_i, &ti], 1)?;
        } else {
            cur_scores = best_s;
            cur_idx = best_i;
        }
        cur_cols = cur_scores.dim(1)?;
    }

    let final_k = cur_scores
        .arg_sort_last_dim(false)?
        .narrow(1, 0, k)?
        .contiguous()?;
    cur_idx.gather(&final_k, 1)
}

/// Expand selected bucket IDs into up to `top_k` long-range key indices outside sinks and local window.
pub fn expand_buckets_single_query(
    selected_buckets: &[usize],
    bucket_members: &[Vec<i32>],
    query_pos: usize,
    window_size: usize,
    sink_size: usize,
    top_k: usize,
) -> Vec<i32> {
    if top_k == 0 || selected_buckets.is_empty() {
        return Vec::new();
    }
    let window_start = (query_pos + 1).saturating_sub(window_size) as i32;
    let sink_limit = sink_size as i32;
    let mut expanded = Vec::with_capacity(top_k);
    for &b in selected_buckets {
        if b >= bucket_members.len() {
            continue;
        }
        for &key_pos in &bucket_members[b] {
            if key_pos < 0 {
                continue;
            }
            if key_pos >= sink_limit && key_pos < window_start {
                expanded.push(key_pos);
                if expanded.len() >= top_k {
                    return expanded;
                }
            }
        }
    }
    expanded
}

/// Build the full causal keep-set `sinks U expanded U window` for a query at global position `query_pos`.
pub fn build_query_keep_set(
    expanded: &[i32],
    query_pos: usize,
    window_size: usize,
    sink_size: usize,
) -> Vec<u32> {
    let win_start = (query_pos + 1).saturating_sub(window_size);
    let sink_end = sink_size.min(win_start).min(query_pos + 1);
    let mut keep = Vec::with_capacity(sink_end + expanded.len() + (query_pos + 1 - win_start));
    for s in 0..sink_end {
        keep.push(s as u32);
    }
    for &k in expanded {
        if k >= 0 && (k as usize) <= query_pos {
            keep.push(k as u32);
        }
    }
    for w in win_start..=query_pos {
        keep.push(w as u32);
    }
    keep
}

/// Select `keep_indices` (`[B][Tq]`) for `q_idx` (`[B, Tq, d_idx]`) against `buckets`.
pub fn select_keep_indices(
    q_idx: &Tensor,
    buckets: &Buckets,
    lsa_cfg: &LsaConfig,
    query_offset: usize,
) -> Result<Vec<Vec<Vec<u32>>>> {
    let (b_sz, t_q, d_idx) = q_idx.dims3()?;
    let b_total = buckets.centroids.dim(1)?;
    let deviations = lsa_cfg.bucket_shape.map(|(_, z)| z).unwrap_or(0.0);

    if lsa_cfg.top_level == 0 {
        let mut all_keep = Vec::with_capacity(b_sz);
        for b in 0..b_sz {
            let q_b = q_idx.narrow(0, b, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
            let c_b = buckets
                .centroids
                .narrow(0, b, 1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;

            let scores_mat = if b_total == 0 {
                Vec::new()
            } else {
                let scores_t = if let Some(ref shapes_vec) = buckets.shapes {
                    shaped_scores(&q_b, &c_b, &shapes_vec[b], deviations)?
                } else {
                    q_b.matmul(&c_b.t()?)?
                };
                scores_t.flatten_all()?.to_vec1::<f32>()?
            };

            let members_b = &buckets.bucket_members[b];
            let b_per_blk = buckets.buckets_per_block;

            let keep_b: Vec<Vec<u32>> = (0..t_q)
                .into_par_iter()
                .map(|i| {
                    let pos = query_offset + i;
                    let visible = ((pos / lsa_cfg.window_size) * b_per_blk).min(b_total);
                    let (eff_p, eff_k) =
                        budget_at(pos, lsa_cfg.top_p, lsa_cfg.top_k, lsa_cfg.early_budget);

                    let selected: Vec<usize> = if visible == 0 || eff_p == 0 {
                        Vec::new()
                    } else {
                        let row = &scores_mat[i * b_total..i * b_total + visible];
                        let mut pairs: Vec<(f32, usize)> = row
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(idx, s)| (s, idx))
                            .collect();
                        pairs.sort_by(|a, b| {
                            b.0.partial_cmp(&a.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.1.cmp(&b.1))
                        });
                        let keep_p = eff_p.min(visible);
                        pairs[..keep_p].iter().map(|&(_, idx)| idx).collect()
                    };

                    let expanded = expand_buckets_single_query(
                        &selected,
                        members_b,
                        pos,
                        lsa_cfg.window_size,
                        lsa_cfg.sink_size,
                        eff_k,
                    );
                    build_query_keep_set(&expanded, pos, lsa_cfg.window_size, lsa_cfg.sink_size)
                })
                .collect();
            all_keep.push(keep_b);
        }
        Ok(all_keep)
    } else {
        let mut all_keep = Vec::with_capacity(b_sz);
        for b in 0..b_sz {
            let c0 = buckets.centroids.narrow(0, b, 1)?.squeeze(0)?;
            let hier = build_hierarchy(
                &c0,
                &buckets.bucket_sizes[b],
                buckets.buckets_per_block,
                lsa_cfg.top_level,
                lsa_cfg.node_bucket_size,
                lsa_cfg.node_cluster_block_size,
            )?;
            let q_flat = q_idx
                .narrow(0, b, 1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut level_vecs = Vec::with_capacity(hier.centroids.len());
            for c in &hier.centroids {
                if c.elem_count() == 0 {
                    level_vecs.push(Vec::new());
                } else {
                    level_vecs.push(c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
                }
            }
            let l0_flat = &level_vecs[0];
            let members_b = &buckets.bucket_members[b];

            let keep_b: Vec<Vec<u32>> = (0..t_q)
                .into_par_iter()
                .map(|i| {
                    let pos = query_offset + i;
                    let (eff_p, eff_k) =
                        budget_at(pos, lsa_cfg.top_p, lsa_cfg.top_k, lsa_cfg.early_budget);
                    let growth = growth_at(&hier, pos, lsa_cfg.window_size);
                    let q_row = &q_flat[i * d_idx..(i + 1) * d_idx];
                    let selected = beam_search_single_query(
                        q_row,
                        l0_flat,
                        &level_vecs,
                        &hier,
                        &growth,
                        eff_p,
                        lsa_cfg.beam_width,
                        lsa_cfg.start_nodes,
                    );
                    let expanded = expand_buckets_single_query(
                        &selected,
                        members_b,
                        pos,
                        lsa_cfg.window_size,
                        lsa_cfg.sink_size,
                        eff_k,
                    );
                    build_query_keep_set(&expanded, pos, lsa_cfg.window_size, lsa_cfg.sink_size)
                })
                .collect();
            all_keep.push(keep_b);
        }
        Ok(all_keep)
    }
}

/// Gather `idx_tensor` (`[M]`) along the sequence axis (dim 2) of `kv` (`[B, Hkv, Tk, D]`)
/// for batch `b`, returning `[Hkv, M, D]`.
/// Avoids copying the full KV cache even when `kv` is a `.narrow(2, 0, cur_len)` view.
fn gather_kv_positions(kv: &Tensor, b: usize, idx_tensor: &Tensor) -> Result<Tensor> {
    let kv_b = kv.narrow(0, b, 1)?.squeeze(0)?; // [Hkv, Tk, D]
    if kv_b.is_contiguous() {
        return kv_b.index_select(idx_tensor, 1);
    }
    let h_kv = kv_b.dim(0)?;
    let mut heads = Vec::with_capacity(h_kv);
    for h in 0..h_kv {
        // Squeeze to 2D [Tk, D] so stride [D, 1] is recognized as contiguous without copying the KV cache!
        let kv_bh = kv_b.narrow(0, h, 1)?.squeeze(0)?; // [Tk, D]
        let kv_bh = if kv_bh.is_contiguous() {
            kv_bh
        } else {
            kv_bh.contiguous()?
        };
        heads.push(kv_bh.index_select(idx_tensor, 0)?.unsqueeze(0)?);
    }
    Tensor::cat(&heads, 0)
}

/// Subquadratic sparse gather SDPA over selected `keep_indices` (`[B][Tq]`).
pub fn sparse_gather_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    keep_indices: &[Vec<Vec<u32>>],
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    let (b_sz, h_q, t_q, d_h) = q.dims4()?;
    let (_, h_kv, _t_k, _) = k.dims4()?;
    let d_v = v.dim(3)?;
    let group_size = h_q / h_kv;
    let dtype = q.dtype();
    let device = q.device();

    // Fast path for single-query decode (`t_q == 1`): gather `[m]` keys directly and run 1-row GQA SDPA!
    if t_q == 1 && b_sz == 1 {
        let indices = &keep_indices[0][0];
        let m = indices.len().max(1);
        let idx_tensor = Tensor::from_slice(indices, (m,), device)?;
        let k_sel = gather_kv_positions(k, 0, &idx_tensor)?.unsqueeze(0)?; // [1, Hkv, m, Dh]
        let v_sel = gather_kv_positions(v, 0, &idx_tensor)?.unsqueeze(0)?; // [1, Hkv, m, Dv]

        let q_g = q.reshape((1, h_kv, group_size, d_h))?;
        let mut att =
            MatMul.matmul_affine_mul(&q_g, &k_sel.t()?, sdpa_params.softmax_scale.into())?;
        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?.tanh()?;
            att = (att * softcap as f64)?;
        }
        let att_f32 = att.to_dtype(DType::F32)?;
        let probs = if let Some(sinks) = &sdpa_params.sinks {
            let att_heads = att_f32.reshape((1, h_q, 1, m))?;
            hanzo_quant::softmax_with_sinks(&att_heads, &sinks.to_dtype(DType::F32)?, None)?
                .to_dtype(dtype)?
                .reshape((1, h_kv, group_size, m))?
        } else {
            hanzo_nn::ops::softmax_last_dim(&att_f32)?.to_dtype(dtype)?
        };
        let out = MatMul.matmul(&probs, &v_sel)?;
        return out.reshape((1, h_q, 1, d_v));
    }

    const Q_CHUNK: usize = 256;
    let mut batch_outputs = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let q_b = q.narrow(0, b, 1)?.squeeze(0)?; // [Hq, Tq, Dh]
        let keep_b = &keep_indices[b];
        let mut q_chunks_out = Vec::with_capacity(t_q.div_ceil(Q_CHUNK));

        for q_start in (0..t_q).step_by(Q_CHUNK) {
            let q_len = Q_CHUNK.min(t_q - q_start);

            let mut union_keys: Vec<u32> = Vec::new();
            for qi in 0..q_len {
                union_keys.extend_from_slice(&keep_b[q_start + qi]);
            }
            union_keys.sort_unstable();
            union_keys.dedup();
            if union_keys.is_empty() {
                union_keys.push(0);
            }
            let u_len = union_keys.len();

            let mut mask_vals = vec![f32::NEG_INFINITY; q_len * u_len];
            for qi in 0..q_len {
                let row_off = qi * u_len;
                for &k_idx in &keep_b[q_start + qi] {
                    if let Ok(pos) = union_keys.binary_search(&k_idx) {
                        mask_vals[row_off + pos] = 0.0;
                    }
                }
            }

            let idx_tensor = Tensor::from_vec(union_keys, (u_len,), device)?;
            let k_sel = gather_kv_positions(k, b, &idx_tensor)?; // [Hkv, u_len, Dh]
            let v_sel = gather_kv_positions(v, b, &idx_tensor)?; // [Hkv, u_len, Dv]

            let q_chunk = q_b.narrow(1, q_start, q_len)?.contiguous()?.reshape((
                h_kv,
                group_size * q_len,
                d_h,
            ))?;

            let mut scores = MatMul
                .matmul_affine_mul(&q_chunk, &k_sel.t()?, sdpa_params.softmax_scale.into())?
                .reshape((h_kv, group_size, q_len, u_len))?;
            if let Some(softcap) = sdpa_params.softcap {
                scores = (scores / softcap as f64)?.tanh()?;
                scores = (scores * softcap as f64)?;
            }

            let mask_tensor = Tensor::from_vec(mask_vals, (1, 1, q_len, u_len), device)?
                .broadcast_as((h_kv, group_size, q_len, u_len))?;
            let scores_f32 = scores.to_dtype(DType::F32)?.add(&mask_tensor)?;
            let probs = hanzo_nn::ops::softmax_last_dim(&scores_f32)?
                .to_dtype(dtype)?
                .reshape((h_kv, group_size * q_len, u_len))?;

            let out_chunk = MatMul.matmul(&probs, &v_sel)?.reshape((h_q, q_len, d_v))?;
            q_chunks_out.push(out_chunk);
        }

        let b_out = Tensor::cat(&q_chunks_out, 1)?.unsqueeze(0)?;
        batch_outputs.push(b_out);
    }

    Tensor::cat(&batch_outputs, 0)
}

/// 100% GPU-resident LSA bucket cache & subquadratic GQA sparse attention for CUDA / Metal / ROCm.
///
/// Eliminates all GPU->CPU synchronizations (`to_vec1`), per-layer mask uploads (`from_vec`),
/// per-head `index_select` loops, and `repeat_kv` copies across prefill and decode.
#[derive(Debug, Clone)]
pub struct LsaGpuSessionCache {
    pub config: LsaConfig,
    clustered_blocks: usize,
    closed_k_buckets: Option<Tensor>,
    closed_v_buckets: Option<Tensor>,
    centroids: Option<Tensor>,
    sink_bias: Option<Tensor>,
}

impl LsaGpuSessionCache {
    pub fn new(config: LsaConfig) -> Self {
        Self {
            config,
            clustered_blocks: 0,
            closed_k_buckets: None,
            closed_v_buckets: None,
            centroids: None,
            sink_bias: None,
        }
    }

    pub fn reset(&mut self) {
        self.clustered_blocks = 0;
        self.closed_k_buckets = None;
        self.closed_v_buckets = None;
        self.centroids = None;
        self.sink_bias = None;
    }

    pub fn clustered_blocks(&self) -> usize {
        self.clustered_blocks
    }

    /// Trim cached closed buckets/centroids back so only blocks strictly within `[0 .. keep_len)` remain.
    pub fn truncate(&mut self, keep_len: usize) -> Result<()> {
        let w = self.config.window_size.max(16);
        let keep_blocks = keep_len / w;
        if keep_blocks >= self.clustered_blocks {
            return Ok(());
        }
        if keep_blocks == 0 {
            self.reset();
            return Ok(());
        }
        let bucket_size = self.config.bucket_size.max(1).min(w);
        let buckets_per_block = (w / bucket_size).max(1);
        let keep_m = keep_blocks * buckets_per_block;

        if let Some(t) = self.closed_k_buckets.take() {
            self.closed_k_buckets = Some(t.narrow(1, 0, keep_m)?.contiguous()?);
        }
        if let Some(t) = self.closed_v_buckets.take() {
            self.closed_v_buckets = Some(t.narrow(1, 0, keep_m)?.contiguous()?);
        }
        if let Some(t) = self.centroids.take() {
            self.centroids = Some(t.narrow(1, 0, keep_m)?.contiguous()?);
        }
        if let Some(t) = self.sink_bias.take() {
            self.sink_bias = Some(t.narrow(0, 0, keep_m)?.contiguous()?);
        }
        self.clustered_blocks = keep_blocks;
        Ok(())
    }

    /// Append one or more complete `window_size` blocks (`k_slice` / `v_slice` shaped
    /// `[1, h_kv, new_blocks * w, d]`) into the GPU bucket & centroid cache.
    fn append_closed_blocks_slice(&mut self, k_slice: &Tensor, v_slice: &Tensor) -> Result<()> {
        let (_, h_kv, new_len, d_h) = k_slice.dims4()?;
        let d_v = v_slice.dim(3)?;
        let device = k_slice.device();

        let w = self.config.window_size.max(16);
        let bucket_size = self.config.bucket_size.max(1).min(w);
        let buckets_per_block = (w / bucket_size).max(1);
        let new_blocks = new_len / w;
        if new_blocks == 0 {
            return Ok(());
        }
        let new_m = new_blocks * buckets_per_block;

        let k_new_buckets = k_slice
            .contiguous()?
            .reshape((h_kv, new_m, bucket_size * d_h))?;
        let v_new_buckets = v_slice
            .contiguous()?
            .reshape((h_kv, new_m, bucket_size * d_v))?;

        let k_free = position_free(k_slice, self.config.rotary_dim, self.config.rotary_layout)?;
        let d_free = k_free.dim(3)?;
        let c_raw = k_free
            .reshape((1, h_kv, new_m, bucket_size, d_free))?
            .sum(3)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, new_m, h_kv * d_free))?;
        let c_norm = rownorm(&c_raw)?.contiguous()?;

        self.closed_k_buckets = Some(match self.closed_k_buckets.take() {
            Some(prev) => Tensor::cat(&[&prev, &k_new_buckets], 1)?,
            None => k_new_buckets,
        });
        self.closed_v_buckets = Some(match self.closed_v_buckets.take() {
            Some(prev) => Tensor::cat(&[&prev, &v_new_buckets], 1)?,
            None => v_new_buckets,
        });
        self.centroids = Some(match self.centroids.take() {
            Some(prev) => Tensor::cat(&[&prev, &c_norm], 1)?,
            None => c_norm,
        });

        self.clustered_blocks += new_blocks;
        let m_total = self.clustered_blocks * buckets_per_block;
        let mut bias_vec = vec![0.0f32; m_total];
        if !bias_vec.is_empty() {
            bias_vec[0] = 1e4;
        }
        self.sink_bias = Some(Tensor::from_vec(bias_vec, (m_total,), device)?);
        Ok(())
    }

    /// Attend `q_chunk` (`[1, h_q, q_len, d_h]` at positions `[abs_q_start .. abs_q_start + q_len)`)
    /// using cached closed buckets covering `[0 .. win_start)` plus dense window `k_win`/`v_win`
    /// (`[1, h_kv, win_len, d]` covering `[win_start .. abs_q_start + q_len)`).
    ///
    /// Performs exact per-query centroid selection across arbitrary `q_len` (including `8`- and
    /// `16`-token speculative verification blocks) with causal masking over the dense local tail.
    #[allow(clippy::too_many_arguments)]
    fn attend_chunk(
        &self,
        q_chunk: &Tensor,
        abs_q_start: usize,
        k_win: &Tensor,
        v_win: &Tensor,
        win_start: usize,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (_, h_q, q_len, d_h) = q_chunk.dims4()?;
        let (_, h_kv, win_len, _) = k_win.dims4()?;
        let d_v = v_win.dim(3)?;
        let group_size = h_q / h_kv;
        let dtype = q_chunk.dtype();
        let device = q_chunk.device();

        let w = self.config.window_size.max(16);
        let bucket_size = self.config.bucket_size.max(1).min(w);
        let buckets_per_block = (w / bucket_size).max(1);
        let valid_blocks = (win_start / w).min(self.clustered_blocks);
        let valid_m = valid_blocks * buckets_per_block;
        let abs_q_end = abs_q_start + q_len;

        let (eff_p, eff_k) = budget_at(
            abs_q_end.saturating_sub(1),
            self.config.top_p,
            self.config.top_k,
            self.config.early_budget,
        );
        let max_buckets_by_k = (eff_k / bucket_size).max(1);
        let take = eff_p.min(max_buckets_by_k).min(valid_m);

        // No closed prefix blocks before `win_start` (or all buckets fit cleanly in full dense prefix).
        if valid_m == 0
            || take >= valid_m
            || (valid_m > 0 && take == 0 && self.config.sink_size == 0)
        {
            let (k_cat, v_cat, q_off) = if valid_m == 0 {
                (k_win.clone(), v_win.clone(), abs_q_start - win_start)
            } else {
                let prefix_len = valid_m * bucket_size;
                let k_pre = self
                    .closed_k_buckets
                    .as_ref()
                    .unwrap()
                    .narrow(1, 0, valid_m)?
                    .contiguous()?
                    .reshape((1, h_kv, prefix_len, d_h))?;
                let v_pre = self
                    .closed_v_buckets
                    .as_ref()
                    .unwrap()
                    .narrow(1, 0, valid_m)?
                    .contiguous()?
                    .reshape((1, h_kv, prefix_len, d_v))?;
                (
                    Tensor::cat(&[&k_pre, k_win], 2)?,
                    Tensor::cat(&[&v_pre, v_win], 2)?,
                    abs_q_start,
                )
            };
            let kv_len = k_cat.dim(2)?;
            let q_g = q_chunk
                .contiguous()?
                .reshape((1, h_kv, group_size * q_len, d_h))?;
            let mut att =
                MatMul.matmul_affine_mul(&q_g, &k_cat.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                att = ((att / softcap as f64)?.tanh()? * softcap as f64)?;
            }
            let att_f32 = att.reshape((1, h_q, q_len, kv_len))?.to_dtype(DType::F32)?;
            let att_masked = if q_len > 1 || kv_len > q_off + 1 {
                let mask = get_cached_causal_mask(device, q_len, kv_len, q_off)?;
                att_f32.broadcast_add(&mask)?
            } else {
                att_f32
            };
            let probs = hanzo_nn::ops::softmax_last_dim(&att_masked)?
                .to_dtype(dtype)?
                .reshape((1, h_kv, group_size * q_len, kv_len))?;
            return MatMul.matmul(&probs, &v_cat)?.reshape((1, h_q, q_len, d_v));
        }

        let take = take.max(1).min(valid_m);
        let m_long = take * bucket_size;
        let c_valid_t = self
            .centroids
            .as_ref()
            .unwrap()
            .narrow(1, 0, valid_m)?
            .squeeze(0)?
            .t()?;
        let b_valid = self.sink_bias.as_ref().unwrap().narrow(0, 0, valid_m)?;

        // Bound temporary `[h_kv * sub_q, m_long, d]` gather buffers (~64 MiB combined K+V target).
        let elem_per_tok = (h_kv * m_long.max(1) * d_h.max(d_v)).max(1);
        let sub_step = (16 * 1024 * 1024 / elem_per_tok).clamp(1, 64);

        let mut sub_outputs = Vec::with_capacity(q_len.div_ceil(sub_step));
        for sub_start in (0..q_len).step_by(sub_step) {
            let sub_q = sub_step.min(q_len - sub_start);
            let q_sub = q_chunk.narrow(2, sub_start, sub_q)?;
            let sub_abs_start = abs_q_start + sub_start;
            let sub_win_len = (sub_abs_start + sub_q - win_start).min(win_len);
            let k_win_sub = k_win.narrow(2, 0, sub_win_len)?;
            let v_win_sub = v_win.narrow(2, 0, sub_win_len)?;

            let q_free = position_free(&q_sub, self.config.rotary_dim, self.config.rotary_layout)?;
            let d_free = q_free.dim(3)?;
            let q_vecs = q_free
                .squeeze(0)?
                .reshape((h_kv, group_size, sub_q, d_free))?
                .sum(1)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((sub_q, h_kv * d_free))?;

            let bucket_scores = MatMul
                .matmul(&q_vecs, &c_valid_t)?
                .to_dtype(DType::F32)?
                .broadcast_add(&b_valid.unsqueeze(0)?)?;
            let top_buckets = top_k_indices_last_dim(&bucket_scores, take)?;

            let sub_out = if sub_q == 1 {
                let idx = top_buckets.squeeze(0)?;
                let k_long = self
                    .closed_k_buckets
                    .as_ref()
                    .unwrap()
                    .index_select(&idx, 1)?
                    .reshape((1, h_kv, m_long, d_h))?;
                let v_long = self
                    .closed_v_buckets
                    .as_ref()
                    .unwrap()
                    .index_select(&idx, 1)?
                    .reshape((1, h_kv, m_long, d_v))?;

                let q_g = q_sub.contiguous()?.reshape((1, h_kv, group_size, d_h))?;
                let att_long = MatMul.matmul_affine_mul(
                    &q_g,
                    &k_long.t()?,
                    sdpa_params.softmax_scale.into(),
                )?;
                let att_win = MatMul.matmul_affine_mul(
                    &q_g,
                    &k_win_sub.t()?,
                    sdpa_params.softmax_scale.into(),
                )?;
                let mut att = Tensor::cat(&[&att_long, &att_win], 3)?;
                if let Some(softcap) = sdpa_params.softcap {
                    att = ((att / softcap as f64)?.tanh()? * softcap as f64)?;
                }
                let att_f32 = att
                    .reshape((1, h_q, 1, m_long + sub_win_len))?
                    .to_dtype(DType::F32)?;
                let probs = hanzo_nn::ops::softmax_last_dim(&att_f32)?
                    .to_dtype(dtype)?
                    .reshape((1, h_kv, group_size, m_long + sub_win_len))?;
                let p_long = probs.narrow(3, 0, m_long)?.contiguous()?;
                let p_win = probs.narrow(3, m_long, sub_win_len)?.contiguous()?;
                (MatMul.matmul(&p_long, &v_long)? + MatMul.matmul(&p_win, &v_win_sub)?)?
                    .reshape((1, h_q, 1, d_v))?
            } else {
                let flat_idx = top_buckets.reshape((sub_q * take,))?;
                let k_long = self
                    .closed_k_buckets
                    .as_ref()
                    .unwrap()
                    .index_select(&flat_idx, 1)?
                    .reshape((h_kv, sub_q, m_long, d_h))?
                    .reshape((h_kv * sub_q, m_long, d_h))?;
                let v_long = self
                    .closed_v_buckets
                    .as_ref()
                    .unwrap()
                    .index_select(&flat_idx, 1)?
                    .reshape((h_kv, sub_q, m_long, d_v))?
                    .reshape((h_kv * sub_q, m_long, d_v))?;

                let q_g_tok = q_sub
                    .squeeze(0)?
                    .reshape((h_kv, group_size, sub_q, d_h))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((h_kv * sub_q, group_size, d_h))?;
                let att_long = MatMul
                    .matmul_affine_mul(&q_g_tok, &k_long.t()?, sdpa_params.softmax_scale.into())?
                    .reshape((h_kv, sub_q, group_size, m_long))?
                    .transpose(1, 2)?
                    .reshape((1, h_q, sub_q, m_long))?
                    .to_dtype(DType::F32)?;

                let q_g_win = q_sub
                    .contiguous()?
                    .reshape((1, h_kv, group_size * sub_q, d_h))?;
                let att_win = MatMul
                    .matmul_affine_mul(&q_g_win, &k_win_sub.t()?, sdpa_params.softmax_scale.into())?
                    .reshape((1, h_q, sub_q, sub_win_len))?
                    .to_dtype(DType::F32)?;
                let mask =
                    get_cached_causal_mask(device, sub_q, sub_win_len, sub_abs_start - win_start)?;
                let att_win_masked = att_win.broadcast_add(&mask)?;

                let mut att = Tensor::cat(&[&att_long, &att_win_masked], 3)?;
                if let Some(softcap) = sdpa_params.softcap {
                    att = ((att / softcap as f64)?.tanh()? * softcap as f64)?;
                }
                let probs = hanzo_nn::ops::softmax_last_dim(&att)?.to_dtype(dtype)?;
                let p_long = probs
                    .narrow(3, 0, m_long)?
                    .contiguous()?
                    .reshape((h_kv, group_size, sub_q, m_long))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((h_kv * sub_q, group_size, m_long))?;
                let p_win = probs
                    .narrow(3, m_long, sub_win_len)?
                    .contiguous()?
                    .reshape((1, h_kv, group_size * sub_q, sub_win_len))?;

                let out_long = MatMul
                    .matmul(&p_long, &v_long)?
                    .reshape((h_kv, sub_q, group_size, d_v))?
                    .transpose(1, 2)?
                    .reshape((1, h_q, sub_q, d_v))?;
                let out_win = MatMul
                    .matmul(&p_win, &v_win_sub)?
                    .reshape((1, h_q, sub_q, d_v))?;
                (out_long + out_win)?
            };
            sub_outputs.push(sub_out);
        }

        if sub_outputs.len() == 1 {
            Ok(sub_outputs.into_iter().next().unwrap())
        } else {
            let refs: Vec<&Tensor> = sub_outputs.iter().collect();
            Tensor::cat(&refs, 2)
        }
    }

    fn attend_range(
        &self,
        q: &Tensor,
        q_offset: usize,
        k_tail: &Tensor,
        v_tail: &Tensor,
        tail_start: usize,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (_, _, t_q, _) = q.dims4()?;
        let tail_len = k_tail.dim(2)?;
        let w = self.config.window_size.max(16);
        let early_pos = self.config.early_budget.map(|eb| eb.position);

        let mut chunks_out = Vec::with_capacity(t_q.div_ceil(w) + 1);
        let mut q_start = 0usize;
        while q_start < t_q {
            let abs_q_start = q_offset + q_start;
            let next_w = ((abs_q_start / w) + 1) * w;
            let mut abs_split = (q_offset + t_q).min(next_w);
            if let Some(ep) = early_pos {
                if abs_q_start < ep && abs_split > ep {
                    abs_split = ep;
                }
            }
            let q_len = abs_split - abs_q_start;
            let win_start = (abs_q_start.saturating_sub(w) / w) * w;
            let tail_off = win_start - tail_start;
            let win_len = (abs_split - win_start).min(tail_len - tail_off);
            let q_chunk = if q_start == 0 && q_len == t_q {
                q.clone()
            } else {
                q.narrow(2, q_start, q_len)?
            };
            let k_win = if tail_off == 0 && win_len == tail_len {
                k_tail.clone()
            } else {
                k_tail.narrow(2, tail_off, win_len)?
            };
            let v_win = if tail_off == 0 && win_len == tail_len {
                v_tail.clone()
            } else {
                v_tail.narrow(2, tail_off, win_len)?
            };
            chunks_out.push(self.attend_chunk(
                &q_chunk,
                abs_q_start,
                &k_win,
                &v_win,
                win_start,
                sdpa_params,
            )?);
            q_start += q_len;
        }

        if chunks_out.len() == 1 {
            Ok(chunks_out.into_iter().next().unwrap())
        } else {
            let refs: Vec<&Tensor> = chunks_out.iter().collect();
            Tensor::cat(&refs, 2)
        }
    }

    pub fn forward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (k, v) = (&rows(k)?, &rows(v)?);
        let (_b_sz, _h_q, t_q, _d_h) = q.dims4()?;
        let (_, _, t_k, _) = k.dims4()?;

        let w = self.config.window_size.max(16);
        let total_closed = t_k / w;

        if total_closed > self.clustered_blocks {
            let start_pos = self.clustered_blocks * w;
            let new_len = (total_closed - self.clustered_blocks) * w;
            let k_slice = k.narrow(2, start_pos, new_len)?;
            let v_slice = v.narrow(2, start_pos, new_len)?;
            self.append_closed_blocks_slice(&k_slice, &v_slice)?;
        }

        let query_offset = t_k.saturating_sub(t_q);
        let min_win_start = (query_offset.saturating_sub(w) / w) * w;
        let k_tail = k.narrow(2, min_win_start, t_k - min_win_start)?;
        let v_tail = v.narrow(2, min_win_start, t_k - min_win_start)?;
        self.attend_range(
            q,
            query_offset,
            &k_tail,
            &v_tail,
            min_win_start,
            sdpa_params,
        )
    }
}

/// Paged-KV descriptor for [`paged_lsa_sdpa`].
pub struct PagedKvGather<'a> {
    pub key_cache: &'a Tensor,
    pub block_size: usize,
    pub block_table: &'a [usize],
    pub kv_len: usize,
    pub q_offset: usize,
    /// When true, blocks in `[0 .. kv_len / w]` may be closed into buckets immediately;
    /// during speculative verification (`false`), only blocks wholly before `q_offset` close.
    pub allow_tail_close: bool,
}

/// Record a speculative rollback for all cached layers belonging to `block_0`.
///
/// The actual tensor truncation runs on the attention thread via [`drain_paged_lsa_ops`]
/// before the next read so device buffers are never released off-context.
pub(crate) fn rollback_paged_lsa(block_0: usize, keep_len: usize) {
    if let Ok(mut q) = pending_paged_ops().lock() {
        q.push(PendingPagedOp::Rollback { block_0, keep_len });
    }
}

/// Evict all cached Paged-LSA layer sessions anchored at physical block `block_0`.
pub(crate) fn evict_paged_lsa(block_0: usize) {
    if let Ok(mut q) = pending_paged_ops().lock() {
        q.push(PendingPagedOp::Evict { block_0 });
    }
}

/// Canonical Paged-LSA attention shared by paged decode, speculative block verification,
/// and paged prefill/continuation.
///
/// `gather_range(start_tok, end_tok)` returns contiguous `(k, v)` shaped `[1, h_kv, end_tok - start_tok, d]`,
/// touching ONLY newly closed `w`-blocks and the trailing active window rather than the full prefix.
pub fn paged_lsa_sdpa<F>(
    q: &Tensor,
    ctx: &PagedKvGather<'_>,
    mut gather_range: F,
    sdpa_params: &SdpaParams,
    lsa_cfg: &LsaConfig,
) -> Result<Tensor>
where
    F: FnMut(usize, usize) -> Result<(Tensor, Tensor)>,
{
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    settle();

    let (_, _, _t_q, _) = q.dims4()?;
    if ctx.block_table.is_empty() || ctx.kv_len == 0 || ctx.block_size == 0 {
        hanzo_ml::bail!("paged_lsa_sdpa called with empty block_table or kv_len");
    }

    let w = lsa_cfg.window_size.max(16);
    let cluster_upto = if ctx.allow_tail_close {
        ctx.kv_len
    } else {
        ctx.q_offset
    };
    let target_closed = cluster_upto / w;

    let layer_key = tensor_storage_key(ctx.key_cache);
    let block_0 = ctx.block_table[0];
    let generation = GENERATION.load(Ordering::Acquire);
    let now = SESSION_CALLS.fetch_add(1, Ordering::Relaxed);

    let mut pool = global_paged_session_pool()
        .write()
        .map_err(|_| Error::Msg("paged LSA session pool poisoned".into()))?;
    pool.retain(|_, e| now.saturating_sub(e.used) <= SESSION_IDLE_CALLS);

    let entry = pool
        .entry((layer_key, block_0))
        .or_insert_with(|| PagedLsaEntry {
            session: LsaGpuSessionCache::new(lsa_cfg.clone()),
            block_ids: Vec::new(),
            _source: ctx.key_cache.clone(),
            generation,
            used: now,
        });
    entry._source = ctx.key_cache.clone();
    entry.used = now;

    if entry.session.config != *lsa_cfg || entry.generation != generation {
        entry.session.config = lsa_cfg.clone();
        entry.session.reset();
        entry.block_ids.clear();
        entry.generation = generation;
    }

    // Reconcile prefix physical block identity in case prefix blocks were evicted/reallocated
    // or an earlier truncation trimmed `clustered_blocks` behind `entry.block_ids`.
    let cached_kv_blocks = (entry.session.clustered_blocks * w).div_ceil(ctx.block_size);
    entry.block_ids.truncate(cached_kv_blocks);
    let common_kv_blocks = entry
        .block_ids
        .iter()
        .zip(ctx.block_table.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let max_valid_tokens = (common_kv_blocks * ctx.block_size).min(cluster_upto);
    if max_valid_tokens / w < entry.session.clustered_blocks {
        entry.session.truncate(max_valid_tokens)?;
        let synced_kv_blocks = (entry.session.clustered_blocks * w).div_ceil(ctx.block_size);
        entry.block_ids.truncate(synced_kv_blocks);
    }

    // Incrementally gather & cluster only newly confirmed closed blocks.
    if target_closed > entry.session.clustered_blocks {
        let start_tok = entry.session.clustered_blocks * w;
        let end_tok = target_closed * w;
        let (k_new, v_new) = gather_range(start_tok, end_tok)?;
        let (k_new, v_new) = (rows(&k_new)?, rows(&v_new)?);
        entry.session.append_closed_blocks_slice(&k_new, &v_new)?;
        let needed_kv_blocks = end_tok.div_ceil(ctx.block_size).min(ctx.block_table.len());
        entry.block_ids.clear();
        entry
            .block_ids
            .extend_from_slice(&ctx.block_table[..needed_kv_blocks]);
    }

    // Gather `[min_win_start .. kv_len)` once and split on W-block boundaries so multi-query
    // verify or continuation crossing a W boundary uses each query block's exact closed prefix.
    let min_win_start = (ctx.q_offset.saturating_sub(w) / w) * w;
    let (k_tail, v_tail) = gather_range(min_win_start, ctx.kv_len)?;
    let (k_tail, v_tail) = (rows(&k_tail)?, rows(&v_tail)?);
    entry.session.attend_range(
        q,
        ctx.q_offset,
        &k_tail,
        &v_tail,
        min_win_start,
        sdpa_params,
    )
}

/// Run Latent Sparse Attention on `q` (`[B, Hq, Tq, Dh]`), `k` (`[B, Hkv, Tk, Dh]`), `v` (`[B, Hkv, Tk, Dv]`).
pub fn lsa_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sdpa_params: &SdpaParams,
    lsa_cfg: &LsaConfig,
    projection: Option<&LearnedProjection>,
) -> Result<Tensor> {
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    settle();
    let t_q = q.dim(2)?;
    let t_k = k.dim(2)?;

    // Fast 100% GPU-resident LSA path on CUDA / Metal / ROCm (`!is_cpu()`):
    if !q.device().is_cpu()
        && q.dim(0)? == 1
        && projection.is_none()
        && lsa_cfg.pca_index_dim.is_none()
        && sdpa_params.sinks.is_none()
    {
        if let Ok(mut pool) = global_gpu_session_pool().write() {
            let mut scratch;
            let session = match session_for(&mut pool, k, t_q, || {
                LsaGpuSessionCache::new(lsa_cfg.clone())
            }) {
                Some(session) => session,
                None => {
                    scratch = LsaGpuSessionCache::new(lsa_cfg.clone());
                    &mut scratch
                }
            };
            // Truncate back to blocks wholly before `[t_k - t_q .. t_k)` if keys were rewritten/rolled back.
            if session.config != *lsa_cfg {
                session.config = lsa_cfg.clone();
                session.reset();
            } else {
                session.truncate(t_k.saturating_sub(t_q))?;
            }
            return session.forward(q, k, v, sdpa_params);
        }
    }

    // Route standard GQA-regroup LSA calls through the stateful incremental session pool
    // keyed by K storage identity so decode steps (`t_q == 1`) reuse closed-block buckets & hierarchy!
    if projection.is_none() && lsa_cfg.pca_index_dim.is_none() {
        if let Ok(mut pool) = global_session_pool().write() {
            let mut scratch;
            let session = match session_for(&mut pool, k, t_q, || {
                LsaSessionCache::new(lsa_cfg.clone(), None)
            }) {
                Some(session) => session,
                None => {
                    scratch = LsaSessionCache::new(lsa_cfg.clone(), None);
                    &mut scratch
                }
            };
            // The incremental session cannot drop a suffix, so any call that rewrites a
            // position it already holds starts it over.
            if session.config != *lsa_cfg || t_k.saturating_sub(t_q) < session.total_len() {
                session.config = lsa_cfg.clone();
                session.reset();
            }
            let prev_len = session.total_len();
            let new_len = t_k.saturating_sub(prev_len);
            if new_len > 0 {
                let new_k = k.narrow(2, prev_len, new_len)?;
                return session.forward(q, k, v, &new_k, sdpa_params);
            }
        }
    }

    let query_offset = t_k.saturating_sub(t_q);

    let q_free = position_free(q, lsa_cfg.rotary_dim, lsa_cfg.rotary_layout)?;
    let k_free = position_free(k, lsa_cfg.rotary_dim, lsa_cfg.rotary_layout)?;

    let (q_idx, k_idx) = if let Some(proj) = projection {
        project(&q_free, &k_free, proj)?
    } else if let Some(pca_dim) = lsa_cfg.pca_index_dim {
        let (_, k_reg) = regroup(&q_free, &k_free)?;
        let pca_proj = pca_projections(&k_reg, pca_dim)?;
        project(&q_free, &k_free, &pca_proj)?
    } else {
        regroup(&q_free, &k_free)?
    };

    let mut buckets = build_buckets(
        &k_idx,
        lsa_cfg.window_size,
        lsa_cfg.pool_size,
        lsa_cfg.bucket_size,
        lsa_cfg.bucket_cap,
    )?;

    if let Some((rank, _)) = lsa_cfg.bucket_shape {
        if buckets.num_closed_blocks > 0 {
            shape_buckets(&mut buckets, &k_idx, rank)?;
        }
    }

    let keep_indices = select_keep_indices(&q_idx, &buckets, lsa_cfg, query_offset)?;
    sparse_gather_sdpa(q, k, v, &keep_indices, sdpa_params)
}

/// Stateful incremental LSA bucket cache for a layer across prefill + multi-step decode.
///
/// Only clusters newly closed `window_size` blocks as the KV cache grows, so decode steps
/// do zero Ward clustering and zero GPU->CPU centroid transfers unless a 512-token boundary is crossed!
#[derive(Debug, Clone)]
pub struct LsaSessionCache {
    pub config: LsaConfig,
    pub projection: Option<LearnedProjection>,
    index_keys: Option<Tensor>,
    unclustered_tail: Vec<Tensor>,
    total_len: usize,
    clustered_blocks: usize,
    buckets: Option<Buckets>,
    hierarchy: Option<Vec<Hierarchy>>,
    cached_level_vecs: Option<Vec<Vec<Vec<f32>>>>,
    cached_l0_vecs: Option<Vec<Vec<f32>>>,
}

impl LsaSessionCache {
    pub fn new(config: LsaConfig, projection: Option<LearnedProjection>) -> Self {
        Self {
            config,
            projection,
            index_keys: None,
            unclustered_tail: Vec::new(),
            total_len: 0,
            clustered_blocks: 0,
            buckets: None,
            hierarchy: None,
            cached_level_vecs: None,
            cached_l0_vecs: None,
        }
    }

    pub fn reset(&mut self) {
        self.index_keys = None;
        self.unclustered_tail.clear();
        self.total_len = 0;
        self.clustered_blocks = 0;
        self.buckets = None;
        self.hierarchy = None;
        self.cached_level_vecs = None;
        self.cached_l0_vecs = None;
    }

    pub fn total_len(&self) -> usize {
        self.total_len
    }

    pub fn clustered_blocks(&self) -> usize {
        self.clustered_blocks
    }

    pub fn buckets(&self) -> Option<&Buckets> {
        self.buckets.as_ref()
    }

    /// Run an incremental prefill or decode step given current `q` (`[B, Hq, Tq, Dh]`),
    /// full cached `k_full` (`[B, Hkv, Tk, Dh]`) and `v_full` (`[B, Hkv, Tk, Dv]`),
    /// and newly appended `new_k` (`[B, Hkv, Tk_new, Dh]`).
    pub fn forward(
        &mut self,
        q: &Tensor,
        k_full: &Tensor,
        v_full: &Tensor,
        new_k: &Tensor,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let t_q = q.dim(2)?;
        let t_k = k_full.dim(2)?;
        let query_offset = t_k.saturating_sub(t_q);

        let q_free = position_free(q, self.config.rotary_dim, self.config.rotary_layout)?;
        let k_new_free = position_free(new_k, self.config.rotary_dim, self.config.rotary_layout)?;

        let (q_idx, k_idx_new) = if let Some(ref proj) = self.projection {
            project(&q_free, &k_new_free, proj)?
        } else {
            regroup(&q_free, &k_new_free)?
        };

        let (b_sz, new_len, d_idx) = k_idx_new.dims3()?;
        self.total_len += new_len;
        self.unclustered_tail.push(k_idx_new);

        let total_closed = self.total_len / self.config.window_size;
        let new_blocks = total_closed.saturating_sub(self.clustered_blocks);

        if new_blocks > 0 {
            let mut pieces = Vec::with_capacity(1 + self.unclustered_tail.len());
            if let Some(prev) = self.index_keys.take() {
                pieces.push(prev);
            }
            pieces.append(&mut self.unclustered_tail);
            let piece_refs: Vec<&Tensor> = pieces.iter().collect();
            let all_keys = if piece_refs.len() == 1 {
                pieces.into_iter().next().unwrap()
            } else {
                Tensor::cat(&piece_refs, 1)?
            };

            let start_pos = self.clustered_blocks * self.config.window_size;
            let len_to_cluster = new_blocks * self.config.window_size;
            let slice_keys = all_keys.narrow(1, start_pos, len_to_cluster)?;
            let mut new_b = build_buckets(
                &slice_keys,
                self.config.window_size,
                self.config.pool_size,
                self.config.bucket_size,
                self.config.bucket_cap,
            )?;
            let offset_pos = start_pos as i32;
            if offset_pos > 0 {
                for b in 0..b_sz {
                    for mem_list in &mut new_b.bucket_members[b] {
                        for idx in mem_list.iter_mut() {
                            if *idx >= 0 {
                                *idx += offset_pos;
                            }
                        }
                    }
                }
            }

            match self.buckets.take() {
                Some(mut existing) => {
                    existing.centroids = Tensor::cat(&[&existing.centroids, &new_b.centroids], 1)?;
                    for b in 0..b_sz {
                        existing.bucket_members[b].extend_from_slice(&new_b.bucket_members[b]);
                        existing.bucket_sizes[b].extend_from_slice(&new_b.bucket_sizes[b]);
                    }
                    existing.num_closed_blocks = total_closed;
                    if let Some((rank, _)) = self.config.bucket_shape {
                        shape_buckets(&mut existing, &all_keys, rank)?;
                    }
                    self.buckets = Some(existing);
                }
                None => {
                    if let Some((rank, _)) = self.config.bucket_shape {
                        shape_buckets(&mut new_b, &all_keys, rank)?;
                    }
                    self.buckets = Some(new_b);
                }
            }
            self.clustered_blocks = total_closed;
            self.index_keys = Some(all_keys);

            let buckets_ref = self.buckets.as_ref().unwrap();
            if self.config.top_level >= 1 {
                let mut h_vec = Vec::with_capacity(b_sz);
                let mut all_level_vecs = Vec::with_capacity(b_sz);
                for b in 0..b_sz {
                    let c0 = buckets_ref.centroids.narrow(0, b, 1)?.squeeze(0)?;
                    let h = build_hierarchy(
                        &c0,
                        &buckets_ref.bucket_sizes[b],
                        buckets_ref.buckets_per_block,
                        self.config.top_level,
                        self.config.node_bucket_size,
                        self.config.node_cluster_block_size,
                    )?;
                    let mut level_vecs = Vec::with_capacity(h.centroids.len());
                    for c in &h.centroids {
                        if c.elem_count() == 0 {
                            level_vecs.push(Vec::new());
                        } else {
                            level_vecs
                                .push(c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
                        }
                    }
                    all_level_vecs.push(level_vecs);
                    h_vec.push(h);
                }
                self.hierarchy = Some(h_vec);
                self.cached_level_vecs = Some(all_level_vecs);
                self.cached_l0_vecs = None;
            } else {
                let mut l0_vecs = Vec::with_capacity(b_sz);
                for b in 0..b_sz {
                    let c0 = buckets_ref.centroids.narrow(0, b, 1)?.squeeze(0)?;
                    if c0.elem_count() == 0 {
                        l0_vecs.push(Vec::new());
                    } else {
                        l0_vecs.push(c0.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
                    }
                }
                self.cached_l0_vecs = Some(l0_vecs);
                self.cached_level_vecs = None;
            }
        } else if self.buckets.is_none() {
            let b_per_blk = self.config.window_size / self.config.bucket_size;
            self.buckets = Some(Buckets {
                centroids: Tensor::zeros((b_sz, 0, d_idx), q_idx.dtype(), q_idx.device())?,
                bucket_members: vec![Vec::new(); b_sz],
                bucket_sizes: vec![Vec::new(); b_sz],
                buckets_per_block: b_per_blk,
                num_closed_blocks: 0,
                shapes: None,
            });
            self.cached_l0_vecs = Some(vec![Vec::new(); b_sz]);
        }

        let buckets = self.buckets.as_ref().unwrap();

        // Fast cached host selection using pre-downloaded centroid vectors!
        let keep_indices = if let (Some(ref h_vec), Some(ref all_level_vecs)) =
            (&self.hierarchy, &self.cached_level_vecs)
        {
            let mut all_keep = Vec::with_capacity(b_sz);
            for b in 0..b_sz {
                let hier = &h_vec[b];
                let level_vecs = &all_level_vecs[b];
                let l0_flat = &level_vecs[0];
                let members_b = &buckets.bucket_members[b];
                let q_flat = q_idx
                    .narrow(0, b, 1)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;

                let keep_b: Vec<Vec<u32>> = if t_q == 1 {
                    let pos = query_offset;
                    let (eff_p, eff_k) = budget_at(
                        pos,
                        self.config.top_p,
                        self.config.top_k,
                        self.config.early_budget,
                    );
                    let growth = growth_at(hier, pos, self.config.window_size);
                    let q_row = &q_flat[0..d_idx];
                    let selected = beam_search_single_query(
                        q_row,
                        l0_flat,
                        level_vecs,
                        hier,
                        &growth,
                        eff_p,
                        self.config.beam_width,
                        self.config.start_nodes,
                    );
                    let expanded = expand_buckets_single_query(
                        &selected,
                        members_b,
                        pos,
                        self.config.window_size,
                        self.config.sink_size,
                        eff_k,
                    );
                    vec![build_query_keep_set(
                        &expanded,
                        pos,
                        self.config.window_size,
                        self.config.sink_size,
                    )]
                } else {
                    (0..t_q)
                        .into_par_iter()
                        .map(|i| {
                            let pos = query_offset + i;
                            let (eff_p, eff_k) = budget_at(
                                pos,
                                self.config.top_p,
                                self.config.top_k,
                                self.config.early_budget,
                            );
                            let growth = growth_at(hier, pos, self.config.window_size);
                            let q_row = &q_flat[i * d_idx..(i + 1) * d_idx];
                            let selected = beam_search_single_query(
                                q_row,
                                l0_flat,
                                level_vecs,
                                hier,
                                &growth,
                                eff_p,
                                self.config.beam_width,
                                self.config.start_nodes,
                            );
                            let expanded = expand_buckets_single_query(
                                &selected,
                                members_b,
                                pos,
                                self.config.window_size,
                                self.config.sink_size,
                                eff_k,
                            );
                            build_query_keep_set(
                                &expanded,
                                pos,
                                self.config.window_size,
                                self.config.sink_size,
                            )
                        })
                        .collect()
                };
                all_keep.push(keep_b);
            }
            all_keep
        } else if let (None, Some(ref l0_vecs)) = (&self.config.bucket_shape, &self.cached_l0_vecs)
        {
            let mut all_keep = Vec::with_capacity(b_sz);
            let b_per_blk = buckets.buckets_per_block;
            let num_buckets = buckets.num_closed_blocks * b_per_blk;
            for b in 0..b_sz {
                let c_flat = &l0_vecs[b];
                let members_b = &buckets.bucket_members[b];
                let q_flat = q_idx
                    .narrow(0, b, 1)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;

                let keep_b: Vec<Vec<u32>> = (0..t_q)
                    .map(|i| {
                        let pos = query_offset + i;
                        let (eff_p, eff_k) = budget_at(
                            pos,
                            self.config.top_p,
                            self.config.top_k,
                            self.config.early_budget,
                        );
                        let valid_blocks =
                            (pos / self.config.window_size).min(buckets.num_closed_blocks);
                        let valid_buckets = valid_blocks * b_per_blk;
                        let take = eff_p.min(valid_buckets);
                        let selected = if take == 0 || num_buckets == 0 {
                            Vec::new()
                        } else {
                            let q_row = &q_flat[i * d_idx..(i + 1) * d_idx];
                            let mut scored: Vec<(f32, usize)> = (0..valid_buckets)
                                .map(|m| {
                                    let c_row = &c_flat[m * d_idx..(m + 1) * d_idx];
                                    let s: f32 = q_row.iter().zip(c_row).map(|(x, y)| x * y).sum();
                                    (if s.is_finite() { s } else { f32::NEG_INFINITY }, m)
                                })
                                .collect();
                            scored.sort_by(|a, b| {
                                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            scored.into_iter().take(take).map(|(_, m)| m).collect()
                        };
                        let expanded = expand_buckets_single_query(
                            &selected,
                            members_b,
                            pos,
                            self.config.window_size,
                            self.config.sink_size,
                            eff_k,
                        );
                        build_query_keep_set(
                            &expanded,
                            pos,
                            self.config.window_size,
                            self.config.sink_size,
                        )
                    })
                    .collect();
                all_keep.push(keep_b);
            }
            all_keep
        } else {
            select_keep_indices(&q_idx, buckets, &self.config, query_offset)?
        };

        sparse_gather_sdpa(q, k_full, v_full, &keep_indices, sdpa_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{naive_sdpa, repeat_kv};

    fn det_tensor(
        shape: (usize, usize, usize, usize),
        seed: u64,
        device: &Device,
    ) -> Result<Tensor> {
        let (a, b, c, d) = shape;
        let n = a * b * c * d;
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut vals = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
            vals.push(u as f32);
        }
        Tensor::from_vec(vals, shape, device)
    }

    #[test]
    fn test_engine_lsa_sdpa_dense_parity_and_decode_session() -> Result<()> {
        let device = Device::Cpu;
        let (b, h_q, h_kv, t, d_h) = (1, 8, 2, 192, 32);
        let q = det_tensor((b, h_q, t, d_h), 11, &device)?;
        let k = det_tensor((b, h_kv, t, d_h), 22, &device)?;
        let v = det_tensor((b, h_kv, t, d_h), 33, &device)?;

        let sdpa_params = SdpaParams {
            n_kv_groups: h_q / h_kv,
            softcap: None,
            softmax_scale: 1.0 / (d_h as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };

        // 1. Full budget LSA must match eager causal naive_sdpa
        let full_cfg = LsaConfig {
            window_size: 32,
            sink_size: 4,
            top_k: 512,
            top_p: 128,
            pool_size: 2,
            bucket_size: 4,
            bucket_cap: 8,
            ..Default::default()
        };
        let lsa_out = lsa_sdpa(&q, &k, &v, &sdpa_params, &full_cfg, None)?;

        let mut causal_mask = vec![0.0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                causal_mask[i * t + j] = f32::NEG_INFINITY;
            }
        }
        let mask_t = Tensor::from_vec(causal_mask, (1, 1, t, t), &device)?;
        let k_rep = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v_rep = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;
        let eager_out = naive_sdpa(&q, &k_rep, &v_rep, Some(&mask_t), &sdpa_params)?;

        let a_vec: Vec<f32> = lsa_out.flatten_all()?.to_vec1()?;
        let b_vec: Vec<f32> = eager_out.flatten_all()?.to_vec1()?;
        let rms = (a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(x, y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            / (a_vec.len() as f64))
            .sqrt() as f32;
        assert!(rms < 1e-4, "LSA full budget vs naive_sdpa RMS diff: {rms}");

        // 2. Test incremental LsaSessionCache prefill (128 tokens) + 64 single-token decode steps
        let sparse_cfg = LsaConfig {
            window_size: 32,
            sink_size: 4,
            top_k: 16,
            top_p: 4,
            pool_size: 2,
            bucket_size: 4,
            bucket_cap: 8,
            top_level: 1,
            node_bucket_size: 2,
            node_cluster_block_size: 8,
            beam_width: 4,
            start_nodes: 8,
            bucket_shape: Some((2, 1.5)),
            ..Default::default()
        };
        let mut session = LsaSessionCache::new(sparse_cfg, None);
        let prefill_len = 128;
        let q_pre = q.narrow(2, 0, prefill_len)?;
        let k_pre = k.narrow(2, 0, prefill_len)?;
        let v_pre = v.narrow(2, 0, prefill_len)?;
        let out_pre = session.forward(&q_pre, &k_pre, &v_pre, &k_pre, &sdpa_params)?;
        assert_eq!(out_pre.dims(), &[b, h_q, prefill_len, d_h]);
        assert_eq!(session.clustered_blocks(), 4);

        for step in prefill_len..t {
            let cur_len = step + 1;
            let q_step = q.narrow(2, step, 1)?;
            let k_new = k.narrow(2, step, 1)?;
            let k_cur = k.narrow(2, 0, cur_len)?;
            let v_cur = v.narrow(2, 0, cur_len)?;
            let out_step = session.forward(&q_step, &k_cur, &v_cur, &k_new, &sdpa_params)?;
            assert_eq!(out_step.dims(), &[b, h_q, 1, d_h]);
        }
        assert_eq!(session.clustered_blocks(), t / 32);

        Ok(())
    }

    /// A request that takes over a KV buffer rewrites positions a decode session already
    /// clustered. The prompt over those positions must start the session over, so the next
    /// decode step reads the new keys: the same calls over a buffer that always held them
    /// give the same answer.
    #[test]
    fn test_lsa_session_restarts_when_a_prompt_rewrites_its_buffer() -> Result<()> {
        let device = Device::Cpu;
        let (h_q, h_kv, cap, d_h) = (8, 2, 192, 32);
        let cfg = LsaConfig {
            window_size: 32,
            sink_size: 4,
            top_k: 16,
            top_p: 4,
            pool_size: 2,
            bucket_size: 4,
            bucket_cap: 8,
            min_seq_len: 0,
            ..Default::default()
        };
        let sdpa_params = SdpaParams {
            n_kv_groups: h_q / h_kv,
            softcap: None,
            softmax_scale: 1.0 / (d_h as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        let q = det_tensor((1, h_q, cap, d_h), 7, &device)?;
        let (k_old, v_old) = (
            det_tensor((1, h_kv, cap, d_h), 8, &device)?,
            det_tensor((1, h_kv, cap, d_h), 9, &device)?,
        );
        let (k_new, v_new) = (
            det_tensor((1, h_kv, cap, d_h), 10, &device)?,
            det_tensor((1, h_kv, cap, d_h), 11, &device)?,
        );
        // Calls: a decode step at 160, a prompt over 64..161, a decode step at 161.
        let run = |k: &Tensor, v: &Tensor, rewrite: Option<(&Tensor, &Tensor)>| -> Result<Tensor> {
            lsa_sdpa(
                &q.narrow(2, 160, 1)?,
                &k.narrow(2, 0, 161)?,
                &v.narrow(2, 0, 161)?,
                &sdpa_params,
                &cfg,
                None,
            )?;
            if let Some((k_src, v_src)) = rewrite {
                k.slice_set(&k_src.narrow(2, 64, 97)?.contiguous()?, 2, 64)?;
                v.slice_set(&v_src.narrow(2, 64, 97)?.contiguous()?, 2, 64)?;
            }
            lsa_sdpa(
                &q.narrow(2, 64, 97)?,
                &k.narrow(2, 0, 161)?,
                &v.narrow(2, 0, 161)?,
                &sdpa_params,
                &cfg,
                None,
            )?;
            lsa_sdpa(
                &q.narrow(2, 161, 1)?,
                &k.narrow(2, 0, 162)?,
                &v.narrow(2, 0, 162)?,
                &sdpa_params,
                &cfg,
                None,
            )
        };

        // A buffer that held the old keys, then had 64.. rewritten to the new ones.
        let (k_buf, v_buf) = (k_old.copy()?, v_old.copy()?);
        let taken_over = run(&k_buf, &v_buf, Some((&k_new, &v_new)))?;
        // A buffer that always held exactly what that one holds now.
        let (k_ref, v_ref) = (k_buf.copy()?, v_buf.copy()?);
        let always = run(&k_ref, &v_ref, None)?;

        let diff = (taken_over - always)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "decode after a rewrite read stale keys: max diff {diff}"
        );
        Ok(())
    }

    #[test]
    fn test_lsa_config_rust_iac_and_toml_yaml() -> Result<()> {
        let zen_cfg = LsaConfig::zen_27b()
            .with_min_seq_len(32_768)
            .with_tree(1, 4, 4096, 256, 1024)
            .with_pca_index_dim(256);
        assert_eq!(zen_cfg.min_seq_len, 32_768);
        assert_eq!(zen_cfg.rotary_dim, 64);
        assert_eq!(zen_cfg.rotary_layout, RotaryLayout::Prefix);
        assert_eq!(zen_cfg.top_level, 1);
        assert_eq!(zen_cfg.pca_index_dim, Some(256));

        let toml_cfg = LsaConfig::from_toml_str(
            r#"
            min_seq_len = 16384
            window_size = 512
            sink_size = 4
            top_k = 4096
            top_p = 1024
            rotary_layout = "Prefix"
            rotary_dim = 64
            top_level = 1
            beam_width = 256
            start_nodes = 1024
            pca_index_dim = 256

            [early_budget]
            position = 262144
            top_k = 1024
            top_p = 256
            "#,
        )
        .unwrap();
        assert_eq!(toml_cfg.min_seq_len, 16_384);
        assert_eq!(toml_cfg.rotary_dim, 64);
        assert_eq!(
            toml_cfg.early_budget,
            Some(EarlyBudget {
                position: 262_144,
                top_k: 1024,
                top_p: 256
            })
        );

        let yaml_cfg = LsaConfig::from_yaml_str(
            r#"
            min_seq_len: 32768
            rotary_layout: Prefix
            rotary_dim: 64
            top_level: 1
            early_budget:
              position: 262144
              top_k: 1024
              top_p: 256
            "#,
        )
        .unwrap();
        assert_eq!(yaml_cfg.min_seq_len, 32_768);
        assert_eq!(yaml_cfg.top_level, 1);

        yaml_cfg.clone().install();
        assert!(LsaConfig::global().is_some());
        LsaConfig::clear();
        assert!(LsaConfig::global().is_none());

        Ok(())
    }

    #[test]
    fn test_paged_lsa_multi_query_verify_and_rollback() -> Result<()> {
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        if let Ok(cuda_dev) = Device::new_cuda(0) {
            devices.push(cuda_dev);
        }
        for device in devices {
            run_paged_lsa_multi_query_verify_and_rollback(&device)?;
        }
        Ok(())
    }

    fn run_paged_lsa_multi_query_verify_and_rollback(device: &Device) -> Result<()> {
        let device = device.clone();
        let (b, h_q, h_kv, d_h) = (1, 4, 2, 16);
        let sdpa_params = SdpaParams {
            n_kv_groups: h_q / h_kv,
            softcap: None,
            softmax_scale: 1.0 / (d_h as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };

        // Helper storing full [1, h_kv, t, d_h] sequence inside a mock page table (`block_size = 16`).
        let block_size = 16usize;
        let total_max = 192usize;
        let prefill_len = 136usize;
        let q_full = det_tensor((b, h_q, total_max, d_h), 101, &device)?;
        let k_a = det_tensor((b, h_kv, total_max, d_h), 202, &device)?;
        let v_a = det_tensor((b, h_kv, total_max, d_h), 303, &device)?;
        let k_b = det_tensor((b, h_kv, total_max, d_h), 404, &device)?;
        let v_b = det_tensor((b, h_kv, total_max, d_h), 505, &device)?;

        // Unique non-zero physical block IDs per run so the global paged session pool never collides.
        let run_paged_step = |tag_base: usize,
                              key_cache_id: &Tensor,
                              k_src: &Tensor,
                              v_src: &Tensor,
                              q_start: usize,
                              q_len: usize,
                              allow_tail_close: bool,
                              cfg: &LsaConfig|
         -> Result<Tensor> {
            let kv_len = q_start + q_len;
            let num_blks = kv_len.div_ceil(block_size);
            let block_table: Vec<usize> = (0..num_blks).map(|i| tag_base + i).collect();
            let q_chunk = q_full.narrow(2, q_start, q_len)?;
            paged_lsa_sdpa(
                &q_chunk,
                &PagedKvGather {
                    key_cache: key_cache_id,
                    block_size,
                    block_table: &block_table,
                    kv_len,
                    q_offset: q_start,
                    allow_tail_close,
                },
                |s, e| Ok((k_src.narrow(2, s, e - s)?, v_src.narrow(2, s, e - s)?)),
                &sdpa_params,
                cfg,
            )
        };

        // 1. Full-budget multi-query speculative verify (q_len = 16) must match exact causal eager SDPA.
        let full_cfg = LsaConfig {
            window_size: 32,
            sink_size: 4,
            top_k: 512,
            top_p: 128,
            pool_size: 2,
            bucket_size: 4,
            bucket_cap: 8,
            ..Default::default()
        };
        let layer_full = Tensor::zeros((1,), DType::F32, &device)?;
        let _ = run_paged_step(
            10_000,
            &layer_full,
            &k_a,
            &v_a,
            0,
            prefill_len,
            true,
            &full_cfg,
        )?;
        let paged_verify_16 = run_paged_step(
            10_000,
            &layer_full,
            &k_a,
            &v_a,
            prefill_len,
            16,
            false,
            &full_cfg,
        )?;

        let kv_end = prefill_len + 16;
        let mut mask = vec![0f32; 16 * kv_end];
        for i in 0..16 {
            for j in (prefill_len + i + 1)..kv_end {
                mask[i * kv_end + j] = f32::NEG_INFINITY;
            }
        }
        let mask_t = Tensor::from_vec(mask, (1, 1, 16, kv_end), &device)?;
        let eager_verify_16 = naive_sdpa(
            &q_full.narrow(2, prefill_len, 16)?,
            &repeat_kv(k_a.narrow(2, 0, kv_end)?, sdpa_params.n_kv_groups)?,
            &repeat_kv(v_a.narrow(2, 0, kv_end)?, sdpa_params.n_kv_groups)?,
            Some(&mask_t),
            &sdpa_params,
        )?;
        let verify_diff = paged_verify_16
            .sub(&eager_verify_16)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            verify_diff < 1e-4,
            "16-token Paged-LSA verify diverged from eager causal attention: max diff {verify_diff}"
        );
        evict_paged_lsa(10_000);

        // 2. Sparse speculative verify (q_len = 8 and q_len = 16) + partial-block / cross-W-block rollback
        // must discard rejected draft KV and match a clean session bit-identically (within f32 eps).
        let sparse_cfg = LsaConfig {
            window_size: 32,
            sink_size: 4,
            top_k: 16,
            top_p: 4,
            pool_size: 2,
            bucket_size: 4,
            bucket_cap: 8,
            top_level: 1,
            node_bucket_size: 2,
            node_cluster_block_size: 8,
            beam_width: 4,
            start_nodes: 8,
            bucket_shape: Some((2, 1.5)),
            ..Default::default()
        };

        // Speculative session (base block 20_000): prefill 136, verify 16 draft tokens (up to 152, partial W-block),
        // accept up to keep_len = 142, rollback, extend & close to 176, rollback across W-boundary to 150,
        // overwrite divergent tail [142..158] / [150..166] with B keys/values.
        let layer_spec = Tensor::zeros((1,), DType::F32, &device)?;
        let k_keep_142 = Tensor::cat(
            &[k_a.narrow(2, 0, 142)?, k_b.narrow(2, 142, total_max - 142)?],
            2,
        )?;
        let v_keep_142 = Tensor::cat(
            &[v_a.narrow(2, 0, 142)?, v_b.narrow(2, 142, total_max - 142)?],
            2,
        )?;

        // Pollute draft range 136..168 with k_a/v_a
        let _ = run_paged_step(
            20_000,
            &layer_spec,
            &k_a,
            &v_a,
            0,
            prefill_len,
            true,
            &sparse_cfg,
        )?;
        let _ = run_paged_step(
            20_000,
            &layer_spec,
            &k_a,
            &v_a,
            prefill_len,
            16,
            false,
            &sparse_cfg,
        )?;
        // Force closing block at 160 by verifying/decoding through 168
        let _ = run_paged_step(20_000, &layer_spec, &k_a, &v_a, 152, 8, false, &sparse_cfg)?;
        let _ = run_paged_step(20_000, &layer_spec, &k_a, &v_a, 160, 8, true, &sparse_cfg)?;

        // Rollback across closed W-block [128..160) back to keep_len = 142, then verify 16 fresh tokens with k_keep_142.
        rollback_paged_lsa(20_000, 142);
        let spec_after_rb = run_paged_step(
            20_000,
            &layer_spec,
            &k_keep_142,
            &v_keep_142,
            142,
            16,
            false,
            &sparse_cfg,
        )?;

        // Clean reference session (base block 30_000) that never saw rejected draft tokens.
        let layer_clean = Tensor::zeros((1,), DType::F32, &device)?;
        let _ = run_paged_step(
            30_000,
            &layer_clean,
            &k_keep_142,
            &v_keep_142,
            0,
            142,
            true,
            &sparse_cfg,
        )?;
        let clean_ref = run_paged_step(
            30_000,
            &layer_clean,
            &k_keep_142,
            &v_keep_142,
            142,
            16,
            false,
            &sparse_cfg,
        )?;

        let rb_diff = spec_after_rb
            .sub(&clean_ref)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            rb_diff < 1e-5,
            "Paged-LSA speculative rollback across W-block/page boundary leaked stale KV: max diff {rb_diff}"
        );
        evict_paged_lsa(20_000);
        evict_paged_lsa(30_000);

        // 3. Multi-query verify (`allow_tail_close=false`) crossing a W-block boundary (`128..144`, W=32)
        // must match token-by-token decode (`seq_len=1`, `allow_tail_close=true`) at every step.
        let boundary_base = 40_000;
        let boundary_ref = 50_000;
        let layer_vb = Tensor::zeros((1,), DType::F32, &device)?;
        let layer_db = Tensor::zeros((1,), DType::F32, &device)?;
        let _ = run_paged_step(
            boundary_base,
            &layer_vb,
            &k_a,
            &v_a,
            0,
            120,
            true,
            &sparse_cfg,
        )?;
        let verify_cross = run_paged_step(
            boundary_base,
            &layer_vb,
            &k_a,
            &v_a,
            120,
            16,
            false,
            &sparse_cfg,
        )?;

        let _ = run_paged_step(
            boundary_ref,
            &layer_db,
            &k_a,
            &v_a,
            0,
            120,
            true,
            &sparse_cfg,
        )?;
        let mut step_outs = Vec::with_capacity(16);
        for pos in 120..136 {
            step_outs.push(run_paged_step(
                boundary_ref,
                &layer_db,
                &k_a,
                &v_a,
                pos,
                1,
                true,
                &sparse_cfg,
            )?);
        }
        let step_refs: Vec<&Tensor> = step_outs.iter().collect();
        let decode_cat = Tensor::cat(&step_refs, 2)?;
        let cross_diff = verify_cross
            .sub(&decode_cat)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            cross_diff < 1e-5,
            "Paged-LSA verify crossing W-block boundary diverged from step-by-step decode: max diff {cross_diff}"
        );
        evict_paged_lsa(boundary_base);
        evict_paged_lsa(boundary_ref);

        Ok(())
    }

    #[test]
    fn test_top_k_indices_last_dim_large_cols_parity() -> Result<()> {
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        if let Ok(cuda_dev) = Device::new_cuda(0) {
            devices.push(cuda_dev);
        }

        let rows = 4usize;
        let k = 128usize;
        for dev in devices {
            for &cols in &[5000usize, 16_384, 16_517] {
                // Inject distinct irrational-stepped scores so top-k indices have a unique exact order.
                let mut host = Vec::with_capacity(rows * cols);
                for r in 0..rows {
                    for c in 0..cols {
                        let s = (((r * 131_071 + c * 265_443_5761) % 1_000_003) as f32) * 1e-3
                            + (c as f32) * 1e-7;
                        host.push(s);
                    }
                }
                let scores = Tensor::from_vec(host.clone(), (rows, cols), &dev)?;
                let got = top_k_indices_last_dim(&scores, k)?.to_vec2::<u32>()?;
                for r in 0..rows {
                    let row = &host[r * cols..(r + 1) * cols];
                    let mut ref_order: Vec<usize> = (0..cols).collect();
                    ref_order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    let expected: Vec<u32> =
                        ref_order.into_iter().take(k).map(|i| i as u32).collect();
                    assert_eq!(
                        got[r], expected,
                        "top_k_indices_last_dim mismatch on {dev:?} for cols={cols} row={r}"
                    );
                }
            }
        }
        Ok(())
    }
}
