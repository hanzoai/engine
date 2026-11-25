#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Checkpoint plus bounded delta-record replay cache and transactional rollback for Gated DeltaNet.

use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Storage, Tensor};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const ENV_ENGINE_REPLAY_GDN: &str = "ENGINE_REPLAY_GDN";
pub const ENV_ENGINE_REPLAY_GDN_WINDOW: &str = "ENGINE_REPLAY_GDN_WINDOW";
pub const DEFAULT_REPLAY_WINDOW: usize = 16;
pub const DEFAULT_MAX_SPECULATIVE_DRAFTS: usize = 32;
pub const SUPPORTED_REPLAY_WINDOWS: [usize; 4] = [4, 8, 16, 32];

const INITIAL_POOL_CAPACITY: usize = 4;
const MIN_PARALLEL_HEADS: usize = 4;
const F32_BYTES: usize = 4;
const EXTRA_LOG_HEADROOM: usize = 4;
const MAX_STACK_DIM: usize = 256;
const MAX_STACK_WINDOW: usize = 128;

fn parse_env_bool(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.trim() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Some(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Some(false),
        _ => None,
    }
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

/// Runtime configuration for opt-in ReplayGDN execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayGdnConfig {
    pub enabled: bool,
    pub window_size: usize,
    pub max_speculative_drafts: usize,
}

impl Default for ReplayGdnConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl ReplayGdnConfig {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            window_size: DEFAULT_REPLAY_WINDOW,
            max_speculative_drafts: DEFAULT_MAX_SPECULATIVE_DRAFTS,
        }
    }

    pub fn enabled(window_size: usize) -> Self {
        Self {
            enabled: true,
            window_size: window_size.max(1),
            max_speculative_drafts: DEFAULT_MAX_SPECULATIVE_DRAFTS,
        }
    }

    pub fn from_env() -> Self {
        let enabled = parse_env_bool(ENV_ENGINE_REPLAY_GDN).unwrap_or(false);
        let window_size = parse_env_usize(ENV_ENGINE_REPLAY_GDN_WINDOW)
            .unwrap_or(DEFAULT_REPLAY_WINDOW)
            .max(1);
        Self {
            enabled,
            window_size,
            max_speculative_drafts: DEFAULT_MAX_SPECULATIVE_DRAFTS,
        }
    }

    pub const fn with_max_speculative_drafts(mut self, drafts: usize) -> Self {
        self.max_speculative_drafts = drafts;
        self
    }
}

/// Dimensions of one GDN recurrent + causal Conv1D layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayLayerDims {
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
    pub conv_width: usize,
}

impl ReplayLayerDims {
    pub const fn state_elems_per_head(&self) -> usize {
        self.head_k_dim * self.head_v_dim
    }

    pub const fn state_bytes_per_head(&self) -> usize {
        self.state_elems_per_head() * F32_BYTES
    }

    pub const fn delta_record_floats_per_head(&self) -> usize {
        self.head_k_dim + self.head_v_dim + 1
    }

    pub const fn delta_record_bytes_per_head(&self) -> usize {
        self.delta_record_floats_per_head() * F32_BYTES
    }

    pub const fn raw_record_floats_per_head(&self) -> usize {
        2 * self.head_k_dim + self.head_v_dim + 2
    }

    pub const fn raw_record_bytes_per_head(&self) -> usize {
        self.raw_record_floats_per_head() * F32_BYTES
    }
}

/// Grouped per-step GDN recurrence tensors: q,k [B,S,H,K], v [B,S,H,V], g,beta [B,S,H].
pub struct GdnStepInputs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
}

/// Exact byte-level read/write accounting for ReplayGDN vs full-state recurrence.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayGdnMemoryStats {
    pub checkpoint_read_bytes: u64,
    pub checkpoint_write_bytes: u64,
    pub record_read_bytes: u64,
    pub record_write_bytes: u64,
    pub conv_read_bytes: u64,
    pub conv_write_bytes: u64,
    pub rollback_copied_bytes: u64,
    pub steps_executed: u64,
    pub staged_steps_executed: u64,
    pub flushes_executed: u64,
    pub rollbacks_executed: u64,
    pub rejected_tokens_discarded: u64,
}

impl ReplayGdnMemoryStats {
    pub const fn total_recurrent_read_bytes(&self) -> u64 {
        self.checkpoint_read_bytes + self.record_read_bytes
    }

    pub const fn total_recurrent_write_bytes(&self) -> u64 {
        self.checkpoint_write_bytes + self.record_write_bytes
    }

    pub const fn total_recurrent_rw_bytes(&self) -> u64 {
        self.total_recurrent_read_bytes() + self.total_recurrent_write_bytes()
    }

    pub const fn state_bytes_per_head(k_dim: usize, v_dim: usize) -> usize {
        k_dim * v_dim * F32_BYTES
    }

    pub const fn delta_record_bytes_per_head(k_dim: usize, v_dim: usize) -> usize {
        (k_dim + v_dim + 1) * F32_BYTES
    }

    pub const fn raw_record_bytes_per_head(k_dim: usize, v_dim: usize) -> usize {
        (2 * k_dim + v_dim + 2) * F32_BYTES
    }

    /// Conventional step-by-step recurrence R+W bytes for `steps` tokens across `total_heads` heads.
    pub const fn baseline_decode_rw_bytes(
        total_heads: usize,
        k_dim: usize,
        v_dim: usize,
        steps: u64,
    ) -> (u64, u64) {
        let per_step = (total_heads * Self::state_bytes_per_head(k_dim, v_dim)) as u64;
        (per_step * steps, per_step * steps)
    }

    /// Conventional trail-based speculative verify + rollback R+W bytes for `verify_len` tokens.
    pub const fn baseline_spec_verify_rw_bytes(
        total_heads: usize,
        k_dim: usize,
        v_dim: usize,
        verify_len: u64,
        rejected: u64,
        fused_trail_kernel: bool,
    ) -> (u64, u64) {
        let s_bytes = (total_heads * Self::state_bytes_per_head(k_dim, v_dim)) as u64;
        let (mut r_bytes, mut w_bytes) = if fused_trail_kernel {
            // Fused kernel reads S_0 once, writes final S_W plus W trail snapshots.
            (s_bytes, s_bytes * (1 + verify_len))
        } else {
            // Portable trail reads S_t W times, writes S_t W times, and copies W trail snapshots.
            (s_bytes * verify_len, s_bytes * 2 * verify_len)
        };
        if rejected > 0 {
            r_bytes += s_bytes;
            w_bytes += s_bytes;
        }
        (r_bytes, w_bytes)
    }
}

#[derive(Debug, Clone)]
struct HeadReplayState {
    s_checkpoint: Arc<Vec<f32>>,
    k_log: Vec<f32>,
    u_log: Vec<f32>,
    g_log: Vec<f32>,
}

impl HeadReplayState {
    fn new_zeroed(k_dim: usize, v_dim: usize, cap: usize) -> Self {
        Self {
            s_checkpoint: Arc::new(vec![0.0f32; k_dim * v_dim]),
            k_log: vec![0.0f32; cap * k_dim],
            u_log: vec![0.0f32; cap * v_dim],
            g_log: vec![0.0f32; cap],
        }
    }

    fn from_state_slice(state: &[f32], k_dim: usize, v_dim: usize, cap: usize) -> Self {
        debug_assert_eq!(state.len(), k_dim * v_dim);
        Self {
            s_checkpoint: Arc::new(state.to_vec()),
            k_log: vec![0.0f32; cap * k_dim],
            u_log: vec![0.0f32; cap * v_dim],
            g_log: vec![0.0f32; cap],
        }
    }

    fn reset(&mut self) {
        if let Some(s_mut) = Arc::get_mut(&mut self.s_checkpoint) {
            s_mut.fill(0.0);
        } else {
            self.s_checkpoint = Arc::new(vec![0.0f32; self.s_checkpoint.len()]);
        }
    }

    fn ensure_capacity(&mut self, k_dim: usize, v_dim: usize, needed: usize) {
        if needed > self.g_log.len() {
            let new_cap = needed.next_power_of_two();
            self.k_log.resize(new_cap * k_dim, 0.0);
            self.u_log.resize(new_cap * v_dim, 0.0);
            self.g_log.resize(new_cap, 0.0);
        }
    }

    fn fork_at(&self, committed_len: usize, k_dim: usize, v_dim: usize, cap: usize) -> Self {
        let real_cap = cap.max(committed_len + 1);
        let mut k_log = vec![0.0f32; real_cap * k_dim];
        let mut u_log = vec![0.0f32; real_cap * v_dim];
        let mut g_log = vec![0.0f32; real_cap];
        if committed_len > 0 {
            k_log[..committed_len * k_dim].copy_from_slice(&self.k_log[..committed_len * k_dim]);
            u_log[..committed_len * v_dim].copy_from_slice(&self.u_log[..committed_len * v_dim]);
            g_log[..committed_len].copy_from_slice(&self.g_log[..committed_len]);
        }
        Self {
            s_checkpoint: Arc::clone(&self.s_checkpoint),
            k_log,
            u_log,
            g_log,
        }
    }

    fn materialize_into(&self, active_len: usize, k_dim: usize, v_dim: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), k_dim * v_dim);
        if active_len == 0 {
            out.copy_from_slice(&self.s_checkpoint);
            return;
        }
        let mut alpha_after = vec![0.0f32; active_len];
        let mut acc_g = 0.0f32;
        for j in (0..active_len).rev() {
            alpha_after[j] = acc_g.exp();
            acc_g += self.g_log[j];
        }
        let alpha_0 = acc_g.exp();
        let s0 = &*self.s_checkpoint;
        for r in 0..k_dim {
            let s_row = &s0[r * v_dim..(r + 1) * v_dim];
            let o_row = &mut out[r * v_dim..(r + 1) * v_dim];
            for c in 0..v_dim {
                o_row[c] = alpha_0 * s_row[c];
            }
            for j in 0..active_len {
                let coeff = alpha_after[j] * self.k_log[j * k_dim + r];
                let uj = &self.u_log[j * v_dim..(j + 1) * v_dim];
                for c in 0..v_dim {
                    o_row[c] += coeff * uj[c];
                }
            }
        }
    }

    fn flush_prefix(&mut self, flush_len: usize, total_valid: usize, k_dim: usize, v_dim: usize) {
        if flush_len == 0 {
            return;
        }
        let mut alpha_after = vec![0.0f32; flush_len];
        let mut acc_g = 0.0f32;
        for j in (0..flush_len).rev() {
            alpha_after[j] = acc_g.exp();
            acc_g += self.g_log[j];
        }
        let alpha_0 = acc_g.exp();
        let s0_mut = Arc::make_mut(&mut self.s_checkpoint);
        for r in 0..k_dim {
            let s_row = &mut s0_mut[r * v_dim..(r + 1) * v_dim];
            for c in 0..v_dim {
                s_row[c] *= alpha_0;
            }
            for j in 0..flush_len {
                let coeff = alpha_after[j] * self.k_log[j * k_dim + r];
                let uj = &self.u_log[j * v_dim..(j + 1) * v_dim];
                for c in 0..v_dim {
                    s_row[c] += coeff * uj[c];
                }
            }
        }
        let rem = total_valid.saturating_sub(flush_len);
        if rem > 0 {
            self.k_log
                .copy_within(flush_len * k_dim..(flush_len + rem) * k_dim, 0);
            self.u_log
                .copy_within(flush_len * v_dim..(flush_len + rem) * v_dim, 0);
            self.g_log.copy_within(flush_len..flush_len + rem, 0);
        }
    }
}

struct StepOneCtx<'a> {
    q_in: &'a [f32],
    k_in: &'a [f32],
    v_in: &'a [f32],
    g_in: f32,
    beta_in: f32,
    q_scale: f32,
    m: usize,
    k_dim: usize,
    v_dim: usize,
    fuse_flush: bool,
}

fn step_one_head(head: &mut HeadReplayState, ctx: &StepOneCtx<'_>, y_out: &mut [f32]) {
    let (k_dim, v_dim, m) = (ctx.k_dim, ctx.v_dim, ctx.m);
    if k_dim <= MAX_STACK_DIM && v_dim <= MAX_STACK_DIM && m <= MAX_STACK_WINDOW {
        let mut q_scaled = [0.0f32; MAX_STACK_DIM];
        let mut pk = [0.0f32; MAX_STACK_DIM];
        let mut pq = [0.0f32; MAX_STACK_DIM];
        let mut kv_mem = [0.0f32; MAX_STACK_DIM];
        let mut u_cur = [0.0f32; MAX_STACK_DIM];
        let mut alpha_after = [0.0f32; MAX_STACK_WINDOW];
        step_one_head_bufs(
            head,
            ctx,
            &mut q_scaled[..k_dim],
            &mut pk[..v_dim],
            &mut pq[..v_dim],
            &mut kv_mem[..v_dim],
            &mut u_cur[..v_dim],
            &mut alpha_after[..m],
            y_out,
        );
    } else {
        let mut q_scaled = vec![0.0f32; k_dim];
        let mut pk = vec![0.0f32; v_dim];
        let mut pq = vec![0.0f32; v_dim];
        let mut kv_mem = vec![0.0f32; v_dim];
        let mut u_cur = vec![0.0f32; v_dim];
        let mut alpha_after = vec![0.0f32; m];
        step_one_head_bufs(
            head,
            ctx,
            &mut q_scaled,
            &mut pk,
            &mut pq,
            &mut kv_mem,
            &mut u_cur,
            &mut alpha_after,
            y_out,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn step_one_head_bufs(
    head: &mut HeadReplayState,
    ctx: &StepOneCtx<'_>,
    q_scaled: &mut [f32],
    pk: &mut [f32],
    pq: &mut [f32],
    kv_mem: &mut [f32],
    u_cur: &mut [f32],
    alpha_after: &mut [f32],
    y_out: &mut [f32],
) {
    let (k_dim, v_dim, m) = (ctx.k_dim, ctx.v_dim, ctx.m);
    assert_eq!(y_out.len(), v_dim);
    for r in 0..k_dim {
        q_scaled[r] = ctx.q_in[r] * ctx.q_scale;
    }

    let s0 = &*head.s_checkpoint;
    for r in 0..k_dim {
        let kr = ctx.k_in[r];
        let qr = q_scaled[r];
        let s_row = &s0[r * v_dim..(r + 1) * v_dim];
        for c in 0..v_dim {
            pk[c] += s_row[c] * kr;
            pq[c] += s_row[c] * qr;
        }
    }

    let mut acc_g = ctx.g_in;
    for j in (0..m).rev() {
        alpha_after[j] = acc_g.exp();
        acc_g += head.g_log[j];
    }
    let alpha_0 = acc_g.exp();

    for c in 0..v_dim {
        kv_mem[c] = alpha_0 * pk[c];
        y_out[c] = alpha_0 * pq[c];
    }

    for j in 0..m {
        let kj = &head.k_log[j * k_dim..(j + 1) * k_dim];
        let uj = &head.u_log[j * v_dim..(j + 1) * v_dim];
        let mut dot_kk = 0.0f32;
        let mut dot_kq = 0.0f32;
        for r in 0..k_dim {
            let kjr = kj[r];
            dot_kk += kjr * ctx.k_in[r];
            dot_kq += kjr * q_scaled[r];
        }
        let ck = alpha_after[j] * dot_kk;
        let cq = alpha_after[j] * dot_kq;
        for c in 0..v_dim {
            kv_mem[c] += ck * uj[c];
            y_out[c] += cq * uj[c];
        }
    }

    let mut dot_self_kq = 0.0f32;
    for r in 0..k_dim {
        dot_self_kq += ctx.k_in[r] * q_scaled[r];
    }
    for c in 0..v_dim {
        let uc = ctx.beta_in * (ctx.v_in[c] - kv_mem[c]);
        u_cur[c] = uc;
        y_out[c] += dot_self_kq * uc;
    }

    if ctx.fuse_flush {
        let s0_mut = Arc::make_mut(&mut head.s_checkpoint);
        for r in 0..k_dim {
            let s_row = &mut s0_mut[r * v_dim..(r + 1) * v_dim];
            let kr_cur = ctx.k_in[r];
            for c in 0..v_dim {
                s_row[c] = alpha_0 * s_row[c] + kr_cur * u_cur[c];
            }
            for j in 0..m {
                let coeff = alpha_after[j] * head.k_log[j * k_dim + r];
                let uj = &head.u_log[j * v_dim..(j + 1) * v_dim];
                for c in 0..v_dim {
                    s_row[c] += coeff * uj[c];
                }
            }
        }
    } else {
        head.k_log[m * k_dim..(m + 1) * k_dim].copy_from_slice(ctx.k_in);
        head.u_log[m * v_dim..(m + 1) * v_dim].copy_from_slice(u_cur);
        head.g_log[m] = ctx.g_in;
    }
}

struct MultiStepHeadCtx<'a> {
    q_data: &'a [f32],
    k_data: &'a [f32],
    v_data: &'a [f32],
    g_data: &'a [f32],
    beta_data: &'a [f32],
    batch_idx: usize,
    head_idx: usize,
    num_heads: usize,
    seq_total: usize,
    seq_start: usize,
    w_len: usize,
    base_m: usize,
    k_dim: usize,
    v_dim: usize,
    q_scale: f32,
    fuse_flush_end: bool,
}

/// Multi-token replay window over a single read of `s_checkpoint`, writing y into `y_head[w * v_dim..]`.
fn step_multi_head(head: &mut HeadReplayState, ctx: &MultiStepHeadCtx<'_>, y_head: &mut [f32]) {
    let (k_dim, v_dim, w_len, base_m) = (ctx.k_dim, ctx.v_dim, ctx.w_len, ctx.base_m);
    assert_eq!(y_head.len(), w_len * v_dim);

    let mut q_local = vec![0.0f32; w_len * k_dim];
    let mut k_local = vec![0.0f32; w_len * k_dim];
    for w in 0..w_len {
        let s_idx = ctx.seq_start + w;
        let base_k =
            ((ctx.batch_idx * ctx.seq_total + s_idx) * ctx.num_heads + ctx.head_idx) * k_dim;
        let k_src = &ctx.k_data[base_k..base_k + k_dim];
        let q_src = &ctx.q_data[base_k..base_k + k_dim];
        k_local[w * k_dim..(w + 1) * k_dim].copy_from_slice(k_src);
        let q_dst = &mut q_local[w * k_dim..(w + 1) * k_dim];
        for r in 0..k_dim {
            q_dst[r] = q_src[r] * ctx.q_scale;
        }
    }

    let mut pk_all = vec![0.0f32; w_len * v_dim];
    let mut pq_all = vec![0.0f32; w_len * v_dim];
    let s0 = &*head.s_checkpoint;
    for r in 0..k_dim {
        let s_row = &s0[r * v_dim..(r + 1) * v_dim];
        for w in 0..w_len {
            let kw = k_local[w * k_dim + r];
            let qw = q_local[w * k_dim + r];
            let pk_w = &mut pk_all[w * v_dim..(w + 1) * v_dim];
            let pq_w = &mut pq_all[w * v_dim..(w + 1) * v_dim];
            for c in 0..v_dim {
                pk_w[c] += s_row[c] * kw;
                pq_w[c] += s_row[c] * qw;
            }
        }
    }

    let mut alpha_after = vec![0.0f32; base_m + w_len];
    let mut kv_mem = vec![0.0f32; v_dim];
    for w in 0..w_len {
        let cur_m = base_m + w;
        let s_idx = ctx.seq_start + w;
        let scalar_idx = (ctx.batch_idx * ctx.seq_total + s_idx) * ctx.num_heads + ctx.head_idx;
        let v_base = scalar_idx * v_dim;
        let v_in = &ctx.v_data[v_base..v_base + v_dim];
        let g_in = ctx.g_data[scalar_idx];
        let beta_in = ctx.beta_data[scalar_idx];
        let k_in = &k_local[w * k_dim..(w + 1) * k_dim];
        let q_scaled = &q_local[w * k_dim..(w + 1) * k_dim];
        let pk = &pk_all[w * v_dim..(w + 1) * v_dim];
        let pq = &pq_all[w * v_dim..(w + 1) * v_dim];
        let y_out = &mut y_head[w * v_dim..(w + 1) * v_dim];

        let mut acc_g = g_in;
        for j in (0..cur_m).rev() {
            alpha_after[j] = acc_g.exp();
            acc_g += head.g_log[j];
        }
        let alpha_0 = acc_g.exp();

        for c in 0..v_dim {
            kv_mem[c] = alpha_0 * pk[c];
            y_out[c] = alpha_0 * pq[c];
        }

        for j in 0..cur_m {
            let kj = &head.k_log[j * k_dim..(j + 1) * k_dim];
            let uj = &head.u_log[j * v_dim..(j + 1) * v_dim];
            let mut dot_kk = 0.0f32;
            let mut dot_kq = 0.0f32;
            for r in 0..k_dim {
                let kjr = kj[r];
                dot_kk += kjr * k_in[r];
                dot_kq += kjr * q_scaled[r];
            }
            let ck = alpha_after[j] * dot_kk;
            let cq = alpha_after[j] * dot_kq;
            for c in 0..v_dim {
                kv_mem[c] += ck * uj[c];
                y_out[c] += cq * uj[c];
            }
        }

        let mut dot_self_kq = 0.0f32;
        for r in 0..k_dim {
            dot_self_kq += k_in[r] * q_scaled[r];
        }

        head.k_log[cur_m * k_dim..(cur_m + 1) * k_dim].copy_from_slice(k_in);
        let u_dst = &mut head.u_log[cur_m * v_dim..(cur_m + 1) * v_dim];
        for c in 0..v_dim {
            let uc = beta_in * (v_in[c] - kv_mem[c]);
            u_dst[c] = uc;
            y_out[c] += dot_self_kq * uc;
        }
        head.g_log[cur_m] = g_in;
    }

    if ctx.fuse_flush_end {
        let total = base_m + w_len;
        head.flush_prefix(total, total, k_dim, v_dim);
    }
}

fn to_cpu_f32_contiguous(t: &Tensor) -> Result<Tensor> {
    let t_cpu = if t.device().is_cpu() {
        t.clone()
    } else {
        t.to_device(&Device::Cpu)?
    };
    t_cpu.to_dtype(DType::F32)?.contiguous()
}

fn with_f32_slice<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> Result<R>) -> Result<R> {
    let (guard, layout) = t.storage_and_layout();
    match &*guard {
        Storage::Cpu(cpu) => {
            let slice = cpu.as_slice::<f32>()?;
            let start = layout.start_offset();
            f(&slice[start..start + t.elem_count()])
        }
        _ => hanzo_ml::bail!("expected CPU f32 storage"),
    }
}

/// Pure algebraic replay over raw input tuples `(q, k, v, g, beta)` without mutating `s_t0`.
/// Returns `(y_out [B, S, H, V], u_residuals [B, S, H, V])`.
pub fn replay_from_raw_inputs(
    s_t0: &Tensor,
    inputs: &GdnStepInputs<'_>,
) -> Result<(Tensor, Tensor)> {
    let out_dtype = inputs.q.dtype();
    let out_dev = inputs.q.device().clone();
    let (b_sz, s_len, nh, k_dim) = inputs.q.dims4()?;
    let v_dim = inputs.v.dim(3)?;

    let s_cpu = to_cpu_f32_contiguous(s_t0)?;
    let q_cpu = to_cpu_f32_contiguous(inputs.q)?;
    let k_cpu = to_cpu_f32_contiguous(inputs.k)?;
    let v_cpu = to_cpu_f32_contiguous(inputs.v)?;
    let g_cpu = to_cpu_f32_contiguous(inputs.g)?;
    let b_cpu = to_cpu_f32_contiguous(inputs.beta)?;
    let q_scale = (1.0 / (k_dim as f64).sqrt()) as f32;

    let total_heads = b_sz * nh;
    let mut y_bh = vec![0.0f32; total_heads * s_len * v_dim];
    let mut u_bh = vec![0.0f32; total_heads * s_len * v_dim];

    with_f32_slice(&s_cpu, |s_slice| {
        with_f32_slice(&q_cpu, |q_slice| {
            with_f32_slice(&k_cpu, |k_slice| {
                with_f32_slice(&v_cpu, |v_slice| {
                    with_f32_slice(&g_cpu, |g_slice| {
                        with_f32_slice(&b_cpu, |b_slice| {
                            let solve_one =
                                |(bh_idx, (y_dst, u_dst)): (usize, (&mut [f32], &mut [f32]))| {
                                    let b_idx = bh_idx / nh;
                                    let h_idx = bh_idx % nh;
                                    let s0 = &s_slice
                                        [bh_idx * k_dim * v_dim..(bh_idx + 1) * k_dim * v_dim];

                                    let mut q_loc = vec![0.0f32; s_len * k_dim];
                                    let mut k_loc = vec![0.0f32; s_len * k_dim];
                                    let mut g_loc = vec![0.0f32; s_len];
                                    for w in 0..s_len {
                                        let sc = (b_idx * s_len + w) * nh + h_idx;
                                        g_loc[w] = g_slice[sc];
                                        let kb = sc * k_dim;
                                        k_loc[w * k_dim..(w + 1) * k_dim]
                                            .copy_from_slice(&k_slice[kb..kb + k_dim]);
                                        for r in 0..k_dim {
                                            q_loc[w * k_dim + r] = q_slice[kb + r] * q_scale;
                                        }
                                    }

                                    let mut pk_all = vec![0.0f32; s_len * v_dim];
                                    let mut pq_all = vec![0.0f32; s_len * v_dim];
                                    for r in 0..k_dim {
                                        let s_row = &s0[r * v_dim..(r + 1) * v_dim];
                                        for w in 0..s_len {
                                            let kw = k_loc[w * k_dim + r];
                                            let qw = q_loc[w * k_dim + r];
                                            let pk_w = &mut pk_all[w * v_dim..(w + 1) * v_dim];
                                            let pq_w = &mut pq_all[w * v_dim..(w + 1) * v_dim];
                                            for c in 0..v_dim {
                                                pk_w[c] += s_row[c] * kw;
                                                pq_w[c] += s_row[c] * qw;
                                            }
                                        }
                                    }

                                    let mut alpha_after = vec![0.0f32; s_len];
                                    let mut kv_mem = vec![0.0f32; v_dim];
                                    for w in 0..s_len {
                                        let sc = (b_idx * s_len + w) * nh + h_idx;
                                        let v_in = &v_slice[sc * v_dim..(sc + 1) * v_dim];
                                        let beta_in = b_slice[sc];
                                        let k_in = &k_loc[w * k_dim..(w + 1) * k_dim];
                                        let q_in = &q_loc[w * k_dim..(w + 1) * k_dim];

                                        let mut acc_g = g_loc[w];
                                        for j in (0..w).rev() {
                                            alpha_after[j] = acc_g.exp();
                                            acc_g += g_loc[j];
                                        }
                                        let alpha_0 = acc_g.exp();

                                        let pk = &pk_all[w * v_dim..(w + 1) * v_dim];
                                        let pq = &pq_all[w * v_dim..(w + 1) * v_dim];
                                        let y_w = &mut y_dst[w * v_dim..(w + 1) * v_dim];
                                        for c in 0..v_dim {
                                            kv_mem[c] = alpha_0 * pk[c];
                                            y_w[c] = alpha_0 * pq[c];
                                        }
                                        for j in 0..w {
                                            let kj = &k_loc[j * k_dim..(j + 1) * k_dim];
                                            let uj = &u_dst[j * v_dim..(j + 1) * v_dim];
                                            let mut dot_kk = 0.0f32;
                                            let mut dot_kq = 0.0f32;
                                            for r in 0..k_dim {
                                                dot_kk += kj[r] * k_in[r];
                                                dot_kq += kj[r] * q_in[r];
                                            }
                                            let ck = alpha_after[j] * dot_kk;
                                            let cq = alpha_after[j] * dot_kq;
                                            for c in 0..v_dim {
                                                kv_mem[c] += ck * uj[c];
                                                y_w[c] += cq * uj[c];
                                            }
                                        }
                                        let mut dot_self_kq = 0.0f32;
                                        for r in 0..k_dim {
                                            dot_self_kq += k_in[r] * q_in[r];
                                        }
                                        let u_w = &mut u_dst[w * v_dim..(w + 1) * v_dim];
                                        for c in 0..v_dim {
                                            let uc = beta_in * (v_in[c] - kv_mem[c]);
                                            u_w[c] = uc;
                                            y_w[c] += dot_self_kq * uc;
                                        }
                                    }
                                };

                            if total_heads >= MIN_PARALLEL_HEADS {
                                y_bh.par_chunks_mut(s_len * v_dim)
                                    .zip(u_bh.par_chunks_mut(s_len * v_dim))
                                    .enumerate()
                                    .for_each(solve_one);
                            } else {
                                y_bh.chunks_mut(s_len * v_dim)
                                    .zip(u_bh.chunks_mut(s_len * v_dim))
                                    .enumerate()
                                    .for_each(solve_one);
                            }
                            Ok(())
                        })
                    })
                })
            })
        })
    })?;

    let unshuffle = |bh_buf: &[f32]| -> Result<Tensor> {
        let mut out = vec![0.0f32; b_sz * s_len * nh * v_dim];
        for b_idx in 0..b_sz {
            for w in 0..s_len {
                for h_idx in 0..nh {
                    let src = ((b_idx * nh + h_idx) * s_len + w) * v_dim;
                    let dst = ((b_idx * s_len + w) * nh + h_idx) * v_dim;
                    out[dst..dst + v_dim].copy_from_slice(&bh_buf[src..src + v_dim]);
                }
            }
        }
        Tensor::from_vec(out, (b_sz, s_len, nh, v_dim), &Device::Cpu)?
            .to_device(&out_dev)?
            .to_dtype(out_dtype)
    };
    Ok((unshuffle(&y_bh)?, unshuffle(&u_bh)?))
}

/// Batched ReplayGDN cache holding F32 checkpoint `S_{t0}`, bounded `(k_j, u_j, g_j)` log, and Conv1D ring.
#[derive(Debug, Clone)]
pub struct ReplayGdnCache {
    batch_size: usize,
    dims: ReplayLayerDims,
    window_size: usize,
    log_capacity: usize,
    checkpoint_offset: usize,
    committed_len: usize,
    staged_len: usize,
    heads: Vec<HeadReplayState>,
    conv_rings: Vec<Vec<f32>>,
    stats: ReplayGdnMemoryStats,
}

impl ReplayGdnCache {
    pub fn new(
        batch_size: usize,
        dims: ReplayLayerDims,
        window_size: usize,
        max_spec_drafts: usize,
    ) -> Self {
        let win = window_size.max(1);
        let cap = win + max_spec_drafts + EXTRA_LOG_HEADROOM;
        let total_heads = batch_size * dims.num_v_heads;
        let heads = (0..total_heads)
            .map(|_| HeadReplayState::new_zeroed(dims.head_k_dim, dims.head_v_dim, cap))
            .collect();
        let conv_cap_cols = dims.conv_width + cap;
        let conv_rings = (0..batch_size)
            .map(|_| vec![0.0f32; conv_cap_cols * dims.conv_dim])
            .collect();
        Self {
            batch_size,
            dims,
            window_size: win,
            log_capacity: cap,
            checkpoint_offset: 0,
            committed_len: 0,
            staged_len: 0,
            heads,
            conv_rings,
            stats: ReplayGdnMemoryStats::default(),
        }
    }

    pub fn from_initial_state(
        conv_state: &Tensor,
        recurrent_state: &Tensor,
        window_size: usize,
        max_spec_drafts: usize,
    ) -> Result<Self> {
        let (b_sz, conv_dim, conv_width) = conv_state.dims3()?;
        let (b2, nh, k_dim, v_dim) = recurrent_state.dims4()?;
        if b_sz != b2 {
            hanzo_ml::bail!("ReplayGdnCache batch mismatch: conv {b_sz} vs recurrent {b2}");
        }
        let dims = ReplayLayerDims {
            num_v_heads: nh,
            head_k_dim: k_dim,
            head_v_dim: v_dim,
            conv_dim,
            conv_width,
        };
        let win = window_size.max(1);
        let cap = win + max_spec_drafts + EXTRA_LOG_HEADROOM;
        let rec_cpu = to_cpu_f32_contiguous(recurrent_state)?;
        let conv_t = to_cpu_f32_contiguous(&conv_state.transpose(1, 2)?)?;

        let heads = with_f32_slice(&rec_cpu, |s_slice| {
            Ok((0..b_sz * nh)
                .map(|bh| {
                    let chunk = &s_slice[bh * k_dim * v_dim..(bh + 1) * k_dim * v_dim];
                    HeadReplayState::from_state_slice(chunk, k_dim, v_dim, cap)
                })
                .collect::<Vec<_>>())
        })?;

        let conv_cap_cols = conv_width + cap;
        let conv_rings = with_f32_slice(&conv_t, |c_slice| {
            Ok((0..b_sz)
                .map(|b_idx| {
                    let mut ring = vec![0.0f32; conv_cap_cols * conv_dim];
                    if conv_width * conv_dim > 0 {
                        let src = &c_slice
                            [b_idx * conv_width * conv_dim..(b_idx + 1) * conv_width * conv_dim];
                        ring[..conv_width * conv_dim].copy_from_slice(src);
                    }
                    ring
                })
                .collect::<Vec<_>>())
        })?;

        Ok(Self {
            batch_size: b_sz,
            dims,
            window_size: win,
            log_capacity: cap,
            checkpoint_offset: 0,
            committed_len: 0,
            staged_len: 0,
            heads,
            conv_rings,
            stats: ReplayGdnMemoryStats::default(),
        })
    }

    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub const fn dims(&self) -> ReplayLayerDims {
        self.dims
    }

    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    pub const fn checkpoint_offset(&self) -> usize {
        self.checkpoint_offset
    }

    pub const fn committed_len(&self) -> usize {
        self.committed_len
    }

    pub const fn staged_len(&self) -> usize {
        self.staged_len
    }

    pub const fn seqlen_offset(&self) -> usize {
        self.checkpoint_offset + self.staged_len
    }

    pub const fn committed_seqlen_offset(&self) -> usize {
        self.checkpoint_offset + self.committed_len
    }

    pub const fn stats(&self) -> ReplayGdnMemoryStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = ReplayGdnMemoryStats::default();
    }

    /// Zero-copy CoW prefix fork sharing immutable `Arc` checkpoints across all heads.
    pub fn fork(&self) -> Self {
        let heads = self
            .heads
            .iter()
            .map(|h| {
                h.fork_at(
                    self.committed_len,
                    self.dims.head_k_dim,
                    self.dims.head_v_dim,
                    self.log_capacity,
                )
            })
            .collect();
        let valid_cols = self.dims.conv_width + self.committed_len;
        let total_cols = self.dims.conv_width + self.log_capacity;
        let conv_rings = self
            .conv_rings
            .iter()
            .map(|r| {
                let mut dst = vec![0.0f32; total_cols * self.dims.conv_dim];
                let n = valid_cols * self.dims.conv_dim;
                dst[..n].copy_from_slice(&r[..n]);
                dst
            })
            .collect();
        Self {
            batch_size: self.batch_size,
            dims: self.dims,
            window_size: self.window_size,
            log_capacity: self.log_capacity,
            checkpoint_offset: self.checkpoint_offset,
            committed_len: self.committed_len,
            staged_len: self.committed_len,
            heads,
            conv_rings,
            stats: ReplayGdnMemoryStats::default(),
        }
    }

    fn ensure_log_capacity(&mut self, needed_steps: usize) {
        if needed_steps > self.log_capacity {
            let new_cap = needed_steps.next_power_of_two();
            for h in &mut self.heads {
                h.ensure_capacity(self.dims.head_k_dim, self.dims.head_v_dim, new_cap);
            }
            let new_cols = self.dims.conv_width + new_cap;
            for ring in &mut self.conv_rings {
                ring.resize(new_cols * self.dims.conv_dim, 0.0);
            }
            self.log_capacity = new_cap;
        }
    }

    /// Append raw Conv1D input `[B, S, conv_dim]` to the time-major ring history.
    pub fn advance_conv(&mut self, conv_in: &Tensor, stage: bool) -> Result<()> {
        let (b_sz, s_len, c_dim) = conv_in.dims3()?;
        if b_sz != self.batch_size || c_dim != self.dims.conv_dim {
            hanzo_ml::bail!(
                "advance_conv shape mismatch: got ({b_sz}, {s_len}, {c_dim}), expected ({}, *, {})",
                self.batch_size,
                self.dims.conv_dim
            );
        }
        if s_len == 0 {
            return Ok(());
        }
        let base_step = if stage {
            self.staged_len
        } else {
            self.committed_len
        };
        self.ensure_log_capacity(base_step + s_len);
        let c_cpu = to_cpu_f32_contiguous(conv_in)?;
        let start_col = self.dims.conv_width + base_step;
        with_f32_slice(&c_cpu, |src| {
            for (b_idx, ring) in self.conv_rings.iter_mut().enumerate() {
                let b_src = &src[b_idx * s_len * c_dim..(b_idx + 1) * s_len * c_dim];
                let dst = &mut ring[start_col * c_dim..(start_col + s_len) * c_dim];
                dst.copy_from_slice(b_src);
            }
            Ok(())
        })?;
        self.stats.conv_write_bytes += (b_sz * s_len * c_dim * F32_BYTES) as u64;
        Ok(())
    }

    fn compact_conv_prefix(&mut self, flushed_steps: usize, total_valid: usize) {
        if flushed_steps == 0 || self.dims.conv_dim == 0 {
            return;
        }
        let keep_cols = self.dims.conv_width + total_valid.saturating_sub(flushed_steps);
        let c_dim = self.dims.conv_dim;
        for ring in &mut self.conv_rings {
            ring.copy_within(
                flushed_steps * c_dim..(flushed_steps + keep_cols) * c_dim,
                0,
            );
        }
    }

    /// Execute committed decode or prefill steps, flushing only when committed_len reaches `window_size`.
    pub fn step_decode(&mut self, inputs: &GdnStepInputs<'_>) -> Result<Tensor> {
        if self.staged_len != self.committed_len {
            hanzo_ml::bail!(
                "ReplayGdnCache::step_decode called with {} uncommitted speculative steps",
                self.staged_len - self.committed_len
            );
        }
        let out_dtype = inputs.q.dtype();
        let out_dev = inputs.q.device().clone();
        let (b_sz, s_len, nh, k_dim) = inputs.q.dims4()?;
        let v_dim = inputs.v.dim(3)?;
        if b_sz != self.batch_size
            || nh != self.dims.num_v_heads
            || k_dim != self.dims.head_k_dim
            || v_dim != self.dims.head_v_dim
        {
            hanzo_ml::bail!("ReplayGdnCache::step_decode shape mismatch");
        }
        if s_len == 0 {
            return Tensor::zeros((b_sz, 0, nh, v_dim), out_dtype, &out_dev);
        }

        let q_cpu = to_cpu_f32_contiguous(inputs.q)?;
        let k_cpu = to_cpu_f32_contiguous(inputs.k)?;
        let v_cpu = to_cpu_f32_contiguous(inputs.v)?;
        let g_cpu = to_cpu_f32_contiguous(inputs.g)?;
        let b_cpu = to_cpu_f32_contiguous(inputs.beta)?;
        let q_scale = (1.0 / (k_dim as f64).sqrt()) as f32;
        let total_heads = b_sz * nh;

        // Fast path for single-token decode (seq_len == 1).
        if s_len == 1 {
            let m = self.committed_len;
            let fuse_flush = m + 1 == self.window_size;
            let mut y_out = vec![0.0f32; total_heads * v_dim];
            with_f32_slice(&q_cpu, |q_s| {
                with_f32_slice(&k_cpu, |k_s| {
                    with_f32_slice(&v_cpu, |v_s| {
                        with_f32_slice(&g_cpu, |g_s| {
                            with_f32_slice(&b_cpu, |b_s| {
                                let run_head = |(bh, (head, y_dst)): (
                                    usize,
                                    (&mut HeadReplayState, &mut [f32]),
                                )| {
                                    let ctx = StepOneCtx {
                                        q_in: &q_s[bh * k_dim..(bh + 1) * k_dim],
                                        k_in: &k_s[bh * k_dim..(bh + 1) * k_dim],
                                        v_in: &v_s[bh * v_dim..(bh + 1) * v_dim],
                                        g_in: g_s[bh],
                                        beta_in: b_s[bh],
                                        q_scale,
                                        m,
                                        k_dim,
                                        v_dim,
                                        fuse_flush,
                                    };
                                    step_one_head(head, &ctx, y_dst);
                                };
                                if total_heads >= MIN_PARALLEL_HEADS {
                                    self.heads
                                        .par_iter_mut()
                                        .zip(y_out.par_chunks_mut(v_dim))
                                        .enumerate()
                                        .for_each(run_head);
                                } else {
                                    self.heads
                                        .iter_mut()
                                        .zip(y_out.chunks_mut(v_dim))
                                        .enumerate()
                                        .for_each(run_head);
                                }
                                Ok(())
                            })
                        })
                    })
                })
            })?;

            let s_bytes = (total_heads * self.dims.state_bytes_per_head()) as u64;
            let rec_bytes = (total_heads * self.dims.delta_record_bytes_per_head()) as u64;
            self.stats.checkpoint_read_bytes += s_bytes;
            self.stats.record_read_bytes += (m as u64) * rec_bytes;
            self.stats.steps_executed += 1;
            if fuse_flush {
                self.stats.checkpoint_write_bytes += s_bytes;
                self.stats.flushes_executed += 1;
                self.compact_conv_prefix(self.window_size, self.window_size);
                self.checkpoint_offset += self.window_size;
                self.committed_len = 0;
                self.staged_len = 0;
            } else {
                self.stats.record_write_bytes += rec_bytes;
                self.committed_len = m + 1;
                self.staged_len = m + 1;
            }

            return Tensor::from_vec(y_out, (b_sz, 1, nh, v_dim), &Device::Cpu)?
                .to_device(&out_dev)?
                .to_dtype(out_dtype);
        }

        // Chunked multi-token path (processes <= window_size chunks per single S_0 read).
        let mut y_full = vec![0.0f32; b_sz * s_len * nh * v_dim];
        let mut cursor = 0usize;
        while cursor < s_len {
            let m = self.committed_len;
            let room = self.window_size - m;
            let w_len = (s_len - cursor).min(room);
            let fuse_flush_end = m + w_len == self.window_size;
            self.ensure_log_capacity(m + w_len);

            let mut y_chunk_bh = vec![0.0f32; total_heads * w_len * v_dim];
            with_f32_slice(&q_cpu, |q_s| {
                with_f32_slice(&k_cpu, |k_s| {
                    with_f32_slice(&v_cpu, |v_s| {
                        with_f32_slice(&g_cpu, |g_s| {
                            with_f32_slice(&b_cpu, |b_s| {
                                let run_chunk = |(bh, (head, y_dst)): (
                                    usize,
                                    (&mut HeadReplayState, &mut [f32]),
                                )| {
                                    let ctx = MultiStepHeadCtx {
                                        q_data: q_s,
                                        k_data: k_s,
                                        v_data: v_s,
                                        g_data: g_s,
                                        beta_data: b_s,
                                        batch_idx: bh / nh,
                                        head_idx: bh % nh,
                                        num_heads: nh,
                                        seq_total: s_len,
                                        seq_start: cursor,
                                        w_len,
                                        base_m: m,
                                        k_dim,
                                        v_dim,
                                        q_scale,
                                        fuse_flush_end,
                                    };
                                    step_multi_head(head, &ctx, y_dst);
                                };
                                if total_heads >= MIN_PARALLEL_HEADS {
                                    self.heads
                                        .par_iter_mut()
                                        .zip(y_chunk_bh.par_chunks_mut(w_len * v_dim))
                                        .enumerate()
                                        .for_each(run_chunk);
                                } else {
                                    self.heads
                                        .iter_mut()
                                        .zip(y_chunk_bh.chunks_mut(w_len * v_dim))
                                        .enumerate()
                                        .for_each(run_chunk);
                                }
                                Ok(())
                            })
                        })
                    })
                })
            })?;

            for b_idx in 0..b_sz {
                for w in 0..w_len {
                    for h_idx in 0..nh {
                        let src = ((b_idx * nh + h_idx) * w_len + w) * v_dim;
                        let dst = ((b_idx * s_len + (cursor + w)) * nh + h_idx) * v_dim;
                        y_full[dst..dst + v_dim].copy_from_slice(&y_chunk_bh[src..src + v_dim]);
                    }
                }
            }

            let s_bytes = (total_heads * self.dims.state_bytes_per_head()) as u64;
            let rec_bytes = (total_heads * self.dims.delta_record_bytes_per_head()) as u64;
            self.stats.checkpoint_read_bytes += s_bytes;
            let read_steps: usize = (0..w_len).map(|w| m + w).sum();
            self.stats.record_read_bytes += (read_steps as u64) * rec_bytes;
            self.stats.record_write_bytes += (w_len as u64) * rec_bytes;
            self.stats.steps_executed += w_len as u64;

            if fuse_flush_end {
                self.stats.checkpoint_write_bytes += s_bytes;
                self.stats.flushes_executed += 1;
                self.compact_conv_prefix(self.window_size, m + (s_len - cursor));
                self.checkpoint_offset += self.window_size;
                self.committed_len = 0;
                self.staged_len = 0;
            } else {
                self.committed_len = m + w_len;
                self.staged_len = m + w_len;
            }
            cursor += w_len;
        }

        Tensor::from_vec(y_full, (b_sz, s_len, nh, v_dim), &Device::Cpu)?
            .to_device(&out_dev)?
            .to_dtype(out_dtype)
    }

    /// Stage speculative verify tokens into `(k, u, g)` replay log in one `S_{t0}` read pass, deferring flush.
    pub fn stage_speculative(&mut self, inputs: &GdnStepInputs<'_>) -> Result<Tensor> {
        let out_dtype = inputs.q.dtype();
        let out_dev = inputs.q.device().clone();
        let (b_sz, w_len, nh, k_dim) = inputs.q.dims4()?;
        let v_dim = inputs.v.dim(3)?;
        if b_sz != self.batch_size
            || nh != self.dims.num_v_heads
            || k_dim != self.dims.head_k_dim
            || v_dim != self.dims.head_v_dim
        {
            hanzo_ml::bail!("ReplayGdnCache::stage_speculative shape mismatch");
        }
        if w_len == 0 {
            return Tensor::zeros((b_sz, 0, nh, v_dim), out_dtype, &out_dev);
        }

        let base_m = self.staged_len;
        self.ensure_log_capacity(base_m + w_len);

        let q_cpu = to_cpu_f32_contiguous(inputs.q)?;
        let k_cpu = to_cpu_f32_contiguous(inputs.k)?;
        let v_cpu = to_cpu_f32_contiguous(inputs.v)?;
        let g_cpu = to_cpu_f32_contiguous(inputs.g)?;
        let b_cpu = to_cpu_f32_contiguous(inputs.beta)?;
        let q_scale = (1.0 / (k_dim as f64).sqrt()) as f32;
        let total_heads = b_sz * nh;

        let mut y_bh = vec![0.0f32; total_heads * w_len * v_dim];
        with_f32_slice(&q_cpu, |q_s| {
            with_f32_slice(&k_cpu, |k_s| {
                with_f32_slice(&v_cpu, |v_s| {
                    with_f32_slice(&g_cpu, |g_s| {
                        with_f32_slice(&b_cpu, |b_s| {
                            let run_stage = |(bh, (head, y_dst)): (
                                usize,
                                (&mut HeadReplayState, &mut [f32]),
                            )| {
                                let ctx = MultiStepHeadCtx {
                                    q_data: q_s,
                                    k_data: k_s,
                                    v_data: v_s,
                                    g_data: g_s,
                                    beta_data: b_s,
                                    batch_idx: bh / nh,
                                    head_idx: bh % nh,
                                    num_heads: nh,
                                    seq_total: w_len,
                                    seq_start: 0,
                                    w_len,
                                    base_m,
                                    k_dim,
                                    v_dim,
                                    q_scale,
                                    fuse_flush_end: false,
                                };
                                step_multi_head(head, &ctx, y_dst);
                            };
                            if total_heads >= MIN_PARALLEL_HEADS {
                                self.heads
                                    .par_iter_mut()
                                    .zip(y_bh.par_chunks_mut(w_len * v_dim))
                                    .enumerate()
                                    .for_each(run_stage);
                            } else {
                                self.heads
                                    .iter_mut()
                                    .zip(y_bh.chunks_mut(w_len * v_dim))
                                    .enumerate()
                                    .for_each(run_stage);
                            }
                            Ok(())
                        })
                    })
                })
            })
        })?;

        self.staged_len = base_m + w_len;
        let s_bytes = (total_heads * self.dims.state_bytes_per_head()) as u64;
        let rec_bytes = (total_heads * self.dims.delta_record_bytes_per_head()) as u64;
        self.stats.checkpoint_read_bytes += s_bytes;
        let read_steps: usize = (0..w_len).map(|w| base_m + w).sum();
        self.stats.record_read_bytes += (read_steps as u64) * rec_bytes;
        self.stats.record_write_bytes += (w_len as u64) * rec_bytes;
        self.stats.steps_executed += w_len as u64;
        self.stats.staged_steps_executed += w_len as u64;

        let mut y_out = vec![0.0f32; b_sz * w_len * nh * v_dim];
        for b_idx in 0..b_sz {
            for w in 0..w_len {
                for h_idx in 0..nh {
                    let src = ((b_idx * nh + h_idx) * w_len + w) * v_dim;
                    let dst = ((b_idx * w_len + w) * nh + h_idx) * v_dim;
                    y_out[dst..dst + v_dim].copy_from_slice(&y_bh[src..src + v_dim]);
                }
            }
        }
        Tensor::from_vec(y_out, (b_sz, w_len, nh, v_dim), &Device::Cpu)?
            .to_device(&out_dev)?
            .to_dtype(out_dtype)
    }

    /// Commit `accepted` staged tokens and discard remaining staged tail in O(1) with 0 bytes copied.
    pub fn commit_and_rollback(&mut self, accepted: usize) -> Result<()> {
        let uncommitted = self.staged_len.saturating_sub(self.committed_len);
        if accepted > uncommitted {
            hanzo_ml::bail!(
                "ReplayGdnCache::commit_and_rollback accepted {accepted} > uncommitted {uncommitted}"
            );
        }
        let rejected = uncommitted - accepted;
        let new_committed = self.committed_len + accepted;
        self.committed_len = new_committed;
        self.staged_len = new_committed;
        if rejected > 0 {
            self.stats.rollbacks_executed += 1;
            self.stats.rejected_tokens_discarded += rejected as u64;
        }

        while self.committed_len >= self.window_size {
            self.flush_prefix_window(self.window_size);
        }
        Ok(())
    }

    fn flush_prefix_window(&mut self, flush_len: usize) {
        if flush_len == 0 {
            return;
        }
        let total_valid = self.committed_len;
        let (k_dim, v_dim) = (self.dims.head_k_dim, self.dims.head_v_dim);
        let total_heads = self.heads.len();
        if total_heads >= MIN_PARALLEL_HEADS {
            self.heads
                .par_iter_mut()
                .for_each(|h| h.flush_prefix(flush_len, total_valid, k_dim, v_dim));
        } else {
            self.heads
                .iter_mut()
                .for_each(|h| h.flush_prefix(flush_len, total_valid, k_dim, v_dim));
        }
        self.compact_conv_prefix(flush_len, total_valid);
        let s_bytes = (total_heads * self.dims.state_bytes_per_head()) as u64;
        let rec_bytes = (total_heads * self.dims.delta_record_bytes_per_head()) as u64;
        self.stats.checkpoint_read_bytes += s_bytes;
        self.stats.checkpoint_write_bytes += s_bytes;
        self.stats.record_read_bytes += (flush_len as u64) * rec_bytes;
        self.stats.flushes_executed += 1;

        let rem = total_valid - flush_len;
        self.checkpoint_offset += flush_len;
        self.committed_len = rem;
        self.staged_len = rem;
    }

    /// Explicitly fold all committed delta records into `S_{t0}`.
    pub fn flush(&mut self) -> Result<()> {
        if self.staged_len != self.committed_len {
            hanzo_ml::bail!("cannot flush while uncommitted speculative steps remain");
        }
        if self.committed_len > 0 {
            self.flush_prefix_window(self.committed_len);
        }
        Ok(())
    }

    /// Materialize exact `[B, H, K, V]` recurrent state at `seqlen_offset` without mutating cache.
    pub fn materialize_state(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let (k_dim, v_dim) = (self.dims.head_k_dim, self.dims.head_v_dim);
        let active = self.staged_len;
        let total_heads = self.heads.len();
        let mut out = vec![0.0f32; total_heads * k_dim * v_dim];
        if total_heads >= MIN_PARALLEL_HEADS {
            self.heads
                .par_iter()
                .zip(out.par_chunks_mut(k_dim * v_dim))
                .for_each(|(h, dst)| h.materialize_into(active, k_dim, v_dim, dst));
        } else {
            self.heads
                .iter()
                .zip(out.chunks_mut(k_dim * v_dim))
                .for_each(|(h, dst)| h.materialize_into(active, k_dim, v_dim, dst));
        }
        Tensor::from_vec(
            out,
            (self.batch_size, self.dims.num_v_heads, k_dim, v_dim),
            &Device::Cpu,
        )?
        .to_device(device)?
        .to_dtype(dtype)
    }

    /// Materialize active `[B, conv_dim, conv_width]` Conv1D state window at `seqlen_offset`.
    pub fn current_conv_state(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let (c_dim, c_width) = (self.dims.conv_dim, self.dims.conv_width);
        let active = self.staged_len;
        let mut out = vec![0.0f32; self.batch_size * c_dim * c_width];
        for (b_idx, ring) in self.conv_rings.iter().enumerate() {
            for w in 0..c_width {
                let col = &ring[(active + w) * c_dim..(active + w + 1) * c_dim];
                for c in 0..c_dim {
                    out[(b_idx * c_dim + c) * c_width + w] = col[c];
                }
            }
        }
        Tensor::from_vec(out, (self.batch_size, c_dim, c_width), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)
    }
}

/// CoW snapshot of a single pooled ReplayGDN sequence slot.
#[derive(Debug, Clone)]
pub struct ReplaySlotSnapshot {
    dims: ReplayLayerDims,
    #[allow(dead_code)]
    window_size: usize,
    log_capacity: usize,
    checkpoint_offset: usize,
    committed_len: usize,
    heads: Vec<HeadReplayState>,
    conv_ring: Vec<f32>,
}

impl ReplaySlotSnapshot {
    pub fn seqlen_offset(&self) -> usize {
        self.checkpoint_offset + self.committed_len
    }

    /// Bytes copied when creating this snapshot (0 checkpoint bytes due to CoW `Arc` sharing).
    pub fn delta_copied_bytes(&self) -> usize {
        let rec_bytes =
            self.dims.num_v_heads * self.committed_len * self.dims.delta_record_bytes_per_head();
        let conv_bytes =
            (self.dims.conv_width + self.committed_len) * self.dims.conv_dim * F32_BYTES;
        rec_bytes + conv_bytes
    }
}

#[derive(Debug, Clone)]
struct ReplaySlot {
    checkpoint_offset: usize,
    committed_len: usize,
    staged_len: usize,
    log_capacity: usize,
    heads: Vec<HeadReplayState>,
    conv_ring: Vec<f32>,
}

impl ReplaySlot {
    fn new(dims: ReplayLayerDims, cap: usize) -> Self {
        let heads = (0..dims.num_v_heads)
            .map(|_| HeadReplayState::new_zeroed(dims.head_k_dim, dims.head_v_dim, cap))
            .collect();
        let conv_ring = vec![0.0f32; (dims.conv_width + cap) * dims.conv_dim];
        Self {
            checkpoint_offset: 0,
            committed_len: 0,
            staged_len: 0,
            log_capacity: cap,
            heads,
            conv_ring,
        }
    }

    fn reset(&mut self) {
        self.checkpoint_offset = 0;
        self.committed_len = 0;
        self.staged_len = 0;
        self.conv_ring.fill(0.0);
        for h in &mut self.heads {
            h.reset();
        }
    }

    fn ensure_capacity(&mut self, dims: ReplayLayerDims, needed: usize) {
        if needed > self.log_capacity {
            let new_cap = needed.next_power_of_two();
            for h in &mut self.heads {
                h.ensure_capacity(dims.head_k_dim, dims.head_v_dim, new_cap);
            }
            self.conv_ring
                .resize((dims.conv_width + new_cap) * dims.conv_dim, 0.0);
            self.log_capacity = new_cap;
        }
    }

    fn compact_conv(&mut self, dims: ReplayLayerDims, flush_len: usize, total_valid: usize) {
        if flush_len == 0 || dims.conv_dim == 0 {
            return;
        }
        let keep_cols = dims.conv_width + total_valid.saturating_sub(flush_len);
        let c_dim = dims.conv_dim;
        self.conv_ring
            .copy_within(flush_len * c_dim..(flush_len + keep_cols) * c_dim, 0);
    }
}

/// Slot-indexed ReplayGDN pool backing `HybridStateContract` opt-in execution.
#[derive(Debug, Clone)]
pub struct ReplayGdnPool {
    dims: ReplayLayerDims,
    config: ReplayGdnConfig,
    slots: Vec<ReplaySlot>,
    free_slots: Vec<usize>,
    stats: ReplayGdnMemoryStats,
}

impl ReplayGdnPool {
    pub fn new(dims: ReplayLayerDims, config: ReplayGdnConfig) -> Self {
        let win = config.window_size.max(1);
        let cap = win + config.max_speculative_drafts + EXTRA_LOG_HEADROOM;
        let slots = (0..INITIAL_POOL_CAPACITY)
            .map(|_| ReplaySlot::new(dims, cap))
            .collect();
        let free_slots = (0..INITIAL_POOL_CAPACITY).rev().collect();
        Self {
            dims,
            config: ReplayGdnConfig {
                window_size: win,
                ..config
            },
            slots,
            free_slots,
            stats: ReplayGdnMemoryStats::default(),
        }
    }

    pub const fn dims(&self) -> ReplayLayerDims {
        self.dims
    }

    pub const fn config(&self) -> ReplayGdnConfig {
        self.config
    }

    pub const fn stats(&self) -> ReplayGdnMemoryStats {
        self.stats
    }

    pub fn ensure_slot(&mut self, slot_idx: usize) {
        let cap = self.config.window_size + self.config.max_speculative_drafts + EXTRA_LOG_HEADROOM;
        while self.slots.len() <= slot_idx {
            let new_idx = self.slots.len();
            self.slots.push(ReplaySlot::new(self.dims, cap));
            if new_idx != slot_idx {
                self.free_slots.push(new_idx);
            }
        }
        self.free_slots.retain(|&s| s != slot_idx);
    }

    pub fn allocate_at(&mut self, slot_idx: usize) {
        self.ensure_slot(slot_idx);
        self.slots[slot_idx].reset();
    }

    pub fn reset_slot(&mut self, slot_idx: usize) {
        if let Some(slot) = self.slots.get_mut(slot_idx) {
            slot.reset();
        }
    }

    pub fn free_slot(&mut self, slot_idx: usize) {
        if let Some(slot) = self.slots.get_mut(slot_idx) {
            slot.reset();
            if !self.free_slots.contains(&slot_idx) {
                self.free_slots.push(slot_idx);
            }
        }
    }

    pub fn seqlen_offset(&self, slot_idx: usize) -> usize {
        self.slots[slot_idx].checkpoint_offset + self.slots[slot_idx].staged_len
    }

    pub fn committed_offset(&self, slot_idx: usize) -> usize {
        self.slots[slot_idx].checkpoint_offset + self.slots[slot_idx].committed_len
    }

    pub fn snapshot_slot(&self, slot_idx: usize) -> Result<ReplaySlotSnapshot> {
        let slot = self
            .slots
            .get(slot_idx)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("invalid replay slot {slot_idx}")))?;
        let heads = slot
            .heads
            .iter()
            .map(|h| {
                h.fork_at(
                    slot.committed_len,
                    self.dims.head_k_dim,
                    self.dims.head_v_dim,
                    slot.log_capacity,
                )
            })
            .collect();
        let valid_cols = self.dims.conv_width + slot.committed_len;
        let total_cols = self.dims.conv_width + slot.log_capacity;
        let mut conv_ring = vec![0.0f32; total_cols * self.dims.conv_dim];
        let n = valid_cols * self.dims.conv_dim;
        conv_ring[..n].copy_from_slice(&slot.conv_ring[..n]);
        Ok(ReplaySlotSnapshot {
            dims: self.dims,
            window_size: self.config.window_size,
            log_capacity: slot.log_capacity,
            checkpoint_offset: slot.checkpoint_offset,
            committed_len: slot.committed_len,
            heads,
            conv_ring,
        })
    }

    pub fn restore_slot(&mut self, slot_idx: usize, snap: &ReplaySlotSnapshot) -> Result<()> {
        self.ensure_slot(slot_idx);
        let heads = snap
            .heads
            .iter()
            .map(|h| {
                h.fork_at(
                    snap.committed_len,
                    self.dims.head_k_dim,
                    self.dims.head_v_dim,
                    snap.log_capacity,
                )
            })
            .collect();
        self.slots[slot_idx] = ReplaySlot {
            checkpoint_offset: snap.checkpoint_offset,
            committed_len: snap.committed_len,
            staged_len: snap.committed_len,
            log_capacity: snap.log_capacity,
            heads,
            conv_ring: snap.conv_ring.clone(),
        };
        Ok(())
    }

    pub fn fork_slot(&mut self, src_slot: usize, dst_slot: usize) -> Result<ReplaySlotSnapshot> {
        let snap = self.snapshot_slot(src_slot)?;
        self.restore_slot(dst_slot, &snap)?;
        Ok(snap)
    }

    pub fn materialize_slot_tensors(
        &self,
        slot_idx: usize,
        device: &Device,
        conv_dtype: DType,
        state_dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let slot = self
            .slots
            .get(slot_idx)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("invalid replay slot {slot_idx}")))?;
        let (nh, k_dim, v_dim) = (
            self.dims.num_v_heads,
            self.dims.head_k_dim,
            self.dims.head_v_dim,
        );
        let active = slot.staged_len;
        let mut s_out = vec![0.0f32; nh * k_dim * v_dim];
        if nh >= MIN_PARALLEL_HEADS {
            slot.heads
                .par_iter()
                .zip(s_out.par_chunks_mut(k_dim * v_dim))
                .for_each(|(h, dst)| h.materialize_into(active, k_dim, v_dim, dst));
        } else {
            slot.heads
                .iter()
                .zip(s_out.chunks_mut(k_dim * v_dim))
                .for_each(|(h, dst)| h.materialize_into(active, k_dim, v_dim, dst));
        }
        let rec_tensor = Tensor::from_vec(s_out, (1, nh, k_dim, v_dim), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(state_dtype)?;

        let (c_dim, c_width) = (self.dims.conv_dim, self.dims.conv_width);
        let mut c_out = vec![0.0f32; c_dim * c_width];
        for w in 0..c_width {
            let col = &slot.conv_ring[(active + w) * c_dim..(active + w + 1) * c_dim];
            for c in 0..c_dim {
                c_out[c * c_width + w] = col[c];
            }
        }
        let conv_tensor = Tensor::from_vec(c_out, (1, c_dim, c_width), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(conv_dtype)?;
        Ok((conv_tensor, rec_tensor, slot.checkpoint_offset + active))
    }

    pub fn forward_slot(
        &mut self,
        slot_idx: usize,
        conv_in: Option<&Tensor>,
        inputs: &GdnStepInputs<'_>,
        stage_speculative: bool,
    ) -> Result<Tensor> {
        self.ensure_slot(slot_idx);
        let (b_sz, s_len, nh, k_dim) = inputs.q.dims4()?;
        let v_dim = inputs.v.dim(3)?;
        if b_sz != 1
            || nh != self.dims.num_v_heads
            || k_dim != self.dims.head_k_dim
            || v_dim != self.dims.head_v_dim
        {
            hanzo_ml::bail!("ReplayGdnPool::forward_slot expects batch=1 matching layer dims");
        }
        let out_dtype = inputs.q.dtype();
        let out_dev = inputs.q.device().clone();
        if s_len == 0 {
            return Tensor::zeros((1, 0, nh, v_dim), out_dtype, &out_dev);
        }

        if let Some(conv_x) = conv_in {
            let (_, cs_len, c_dim) = conv_x.dims3()?;
            if cs_len != s_len || c_dim != self.dims.conv_dim {
                hanzo_ml::bail!("ReplayGdnPool::forward_slot conv_in shape mismatch");
            }
            let base = if stage_speculative {
                self.slots[slot_idx].staged_len
            } else {
                self.slots[slot_idx].committed_len
            };
            self.slots[slot_idx].ensure_capacity(self.dims, base + s_len);
            let c_cpu = to_cpu_f32_contiguous(conv_x)?;
            let start_col = self.dims.conv_width + base;
            with_f32_slice(&c_cpu, |src| {
                let dst = &mut self.slots[slot_idx].conv_ring
                    [start_col * c_dim..(start_col + s_len) * c_dim];
                dst.copy_from_slice(src);
                Ok(())
            })?;
            self.stats.conv_write_bytes += (s_len * c_dim * F32_BYTES) as u64;
        }

        let q_cpu = to_cpu_f32_contiguous(inputs.q)?;
        let k_cpu = to_cpu_f32_contiguous(inputs.k)?;
        let v_cpu = to_cpu_f32_contiguous(inputs.v)?;
        let g_cpu = to_cpu_f32_contiguous(inputs.g)?;
        let b_cpu = to_cpu_f32_contiguous(inputs.beta)?;
        let q_scale = (1.0 / (k_dim as f64).sqrt()) as f32;
        let win = self.config.window_size;

        if stage_speculative {
            let base_m = self.slots[slot_idx].staged_len;
            self.slots[slot_idx].ensure_capacity(self.dims, base_m + s_len);
            let mut y_bh = vec![0.0f32; nh * s_len * v_dim];
            with_f32_slice(&q_cpu, |q_s| {
                with_f32_slice(&k_cpu, |k_s| {
                    with_f32_slice(&v_cpu, |v_s| {
                        with_f32_slice(&g_cpu, |g_s| {
                            with_f32_slice(&b_cpu, |b_s| {
                                let slot = &mut self.slots[slot_idx];
                                let run_stage = |(h_idx, (head, y_dst)): (
                                    usize,
                                    (&mut HeadReplayState, &mut [f32]),
                                )| {
                                    let ctx = MultiStepHeadCtx {
                                        q_data: q_s,
                                        k_data: k_s,
                                        v_data: v_s,
                                        g_data: g_s,
                                        beta_data: b_s,
                                        batch_idx: 0,
                                        head_idx: h_idx,
                                        num_heads: nh,
                                        seq_total: s_len,
                                        seq_start: 0,
                                        w_len: s_len,
                                        base_m,
                                        k_dim,
                                        v_dim,
                                        q_scale,
                                        fuse_flush_end: false,
                                    };
                                    step_multi_head(head, &ctx, y_dst);
                                };
                                if nh >= MIN_PARALLEL_HEADS {
                                    slot.heads
                                        .par_iter_mut()
                                        .zip(y_bh.par_chunks_mut(s_len * v_dim))
                                        .enumerate()
                                        .for_each(run_stage);
                                } else {
                                    slot.heads
                                        .iter_mut()
                                        .zip(y_bh.chunks_mut(s_len * v_dim))
                                        .enumerate()
                                        .for_each(run_stage);
                                }
                                Ok(())
                            })
                        })
                    })
                })
            })?;
            self.slots[slot_idx].staged_len = base_m + s_len;

            let s_bytes = (nh * self.dims.state_bytes_per_head()) as u64;
            let rec_bytes = (nh * self.dims.delta_record_bytes_per_head()) as u64;
            self.stats.checkpoint_read_bytes += s_bytes;
            let read_steps: usize = (0..s_len).map(|w| base_m + w).sum();
            self.stats.record_read_bytes += (read_steps as u64) * rec_bytes;
            self.stats.record_write_bytes += (s_len as u64) * rec_bytes;
            self.stats.steps_executed += s_len as u64;
            self.stats.staged_steps_executed += s_len as u64;

            let mut y_out = vec![0.0f32; s_len * nh * v_dim];
            for w in 0..s_len {
                for h_idx in 0..nh {
                    let src = (h_idx * s_len + w) * v_dim;
                    let dst = (w * nh + h_idx) * v_dim;
                    y_out[dst..dst + v_dim].copy_from_slice(&y_bh[src..src + v_dim]);
                }
            }
            return Tensor::from_vec(y_out, (1, s_len, nh, v_dim), &Device::Cpu)?
                .to_device(&out_dev)?
                .to_dtype(out_dtype);
        }

        // Committed path (single-token or chunked prefill).
        if self.slots[slot_idx].staged_len != self.slots[slot_idx].committed_len {
            hanzo_ml::bail!("ReplayGdnPool::forward_slot has uncommitted staged tokens");
        }
        let mut y_full = vec![0.0f32; s_len * nh * v_dim];
        let mut cursor = 0usize;
        let mut conv_flushed_total = 0usize;
        let initial_committed = self.slots[slot_idx].committed_len;

        while cursor < s_len {
            let m = self.slots[slot_idx].committed_len;
            let room = win - m;
            let w_len = (s_len - cursor).min(room);
            let fuse_flush_end = m + w_len == win;
            self.slots[slot_idx].ensure_capacity(self.dims, m + w_len);

            let mut y_chunk_bh = vec![0.0f32; nh * w_len * v_dim];
            with_f32_slice(&q_cpu, |q_s| {
                with_f32_slice(&k_cpu, |k_s| {
                    with_f32_slice(&v_cpu, |v_s| {
                        with_f32_slice(&g_cpu, |g_s| {
                            with_f32_slice(&b_cpu, |b_s| {
                                let slot = &mut self.slots[slot_idx];
                                if w_len == 1 {
                                    let run_one = |(h_idx, (head, y_dst)): (
                                        usize,
                                        (&mut HeadReplayState, &mut [f32]),
                                    )| {
                                        let sc = cursor * nh + h_idx;
                                        let ctx = StepOneCtx {
                                            q_in: &q_s[sc * k_dim..(sc + 1) * k_dim],
                                            k_in: &k_s[sc * k_dim..(sc + 1) * k_dim],
                                            v_in: &v_s[sc * v_dim..(sc + 1) * v_dim],
                                            g_in: g_s[sc],
                                            beta_in: b_s[sc],
                                            q_scale,
                                            m,
                                            k_dim,
                                            v_dim,
                                            fuse_flush: fuse_flush_end,
                                        };
                                        step_one_head(head, &ctx, y_dst);
                                    };
                                    if nh >= MIN_PARALLEL_HEADS {
                                        slot.heads
                                            .par_iter_mut()
                                            .zip(y_chunk_bh.par_chunks_mut(v_dim))
                                            .enumerate()
                                            .for_each(run_one);
                                    } else {
                                        slot.heads
                                            .iter_mut()
                                            .zip(y_chunk_bh.chunks_mut(v_dim))
                                            .enumerate()
                                            .for_each(run_one);
                                    }
                                } else {
                                    let run_chunk = |(h_idx, (head, y_dst)): (
                                        usize,
                                        (&mut HeadReplayState, &mut [f32]),
                                    )| {
                                        let ctx = MultiStepHeadCtx {
                                            q_data: q_s,
                                            k_data: k_s,
                                            v_data: v_s,
                                            g_data: g_s,
                                            beta_data: b_s,
                                            batch_idx: 0,
                                            head_idx: h_idx,
                                            num_heads: nh,
                                            seq_total: s_len,
                                            seq_start: cursor,
                                            w_len,
                                            base_m: m,
                                            k_dim,
                                            v_dim,
                                            q_scale,
                                            fuse_flush_end,
                                        };
                                        step_multi_head(head, &ctx, y_dst);
                                    };
                                    if nh >= MIN_PARALLEL_HEADS {
                                        slot.heads
                                            .par_iter_mut()
                                            .zip(y_chunk_bh.par_chunks_mut(w_len * v_dim))
                                            .enumerate()
                                            .for_each(run_chunk);
                                    } else {
                                        slot.heads
                                            .iter_mut()
                                            .zip(y_chunk_bh.chunks_mut(w_len * v_dim))
                                            .enumerate()
                                            .for_each(run_chunk);
                                    }
                                }
                                Ok(())
                            })
                        })
                    })
                })
            })?;

            for w in 0..w_len {
                for h_idx in 0..nh {
                    let src = (h_idx * w_len + w) * v_dim;
                    let dst = ((cursor + w) * nh + h_idx) * v_dim;
                    y_full[dst..dst + v_dim].copy_from_slice(&y_chunk_bh[src..src + v_dim]);
                }
            }

            let s_bytes = (nh * self.dims.state_bytes_per_head()) as u64;
            let rec_bytes = (nh * self.dims.delta_record_bytes_per_head()) as u64;
            self.stats.checkpoint_read_bytes += s_bytes;
            let read_steps: usize = (0..w_len).map(|w| m + w).sum();
            self.stats.record_read_bytes += (read_steps as u64) * rec_bytes;
            self.stats.steps_executed += w_len as u64;

            if fuse_flush_end {
                if w_len > 1 {
                    self.stats.record_write_bytes += (w_len as u64) * rec_bytes;
                }
                self.stats.checkpoint_write_bytes += s_bytes;
                self.stats.flushes_executed += 1;
                conv_flushed_total += win;
                self.slots[slot_idx].checkpoint_offset += win;
                self.slots[slot_idx].committed_len = 0;
                self.slots[slot_idx].staged_len = 0;
            } else {
                self.stats.record_write_bytes += (w_len as u64) * rec_bytes;
                self.slots[slot_idx].committed_len = m + w_len;
                self.slots[slot_idx].staged_len = m + w_len;
            }
            cursor += w_len;
        }

        if conv_flushed_total > 0 && conv_in.is_some() {
            self.slots[slot_idx].compact_conv(
                self.dims,
                conv_flushed_total,
                initial_committed + s_len,
            );
        }

        Tensor::from_vec(y_full, (1, s_len, nh, v_dim), &Device::Cpu)?
            .to_device(&out_dev)?
            .to_dtype(out_dtype)
    }

    /// Transactional O(1) commit + zero-copy rollback for speculative verification on `slot_idx`.
    pub fn commit_or_rollback_slot(
        &mut self,
        slot_idx: usize,
        keep_len: usize,
        rejected: usize,
    ) -> Result<()> {
        let slot = self
            .slots
            .get_mut(slot_idx)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("invalid replay slot {slot_idx}")))?;
        let current_total = slot.checkpoint_offset + slot.staged_len;
        if keep_len + rejected != current_total {
            hanzo_ml::bail!(
                "ReplayGdnPool rollback mismatch: keep_len {keep_len} + rejected {rejected} != {current_total}"
            );
        }
        if keep_len < slot.checkpoint_offset + slot.committed_len {
            hanzo_ml::bail!(
                "ReplayGdnPool cannot rollback below committed offset {}",
                slot.checkpoint_offset + slot.committed_len
            );
        }
        let target_len = keep_len - slot.checkpoint_offset;
        slot.committed_len = target_len;
        slot.staged_len = target_len;
        if rejected > 0 {
            self.stats.rollbacks_executed += 1;
            self.stats.rejected_tokens_discarded += rejected as u64;
        }

        let win = self.config.window_size;
        let (nh, k_dim, v_dim) = (
            self.dims.num_v_heads,
            self.dims.head_k_dim,
            self.dims.head_v_dim,
        );
        while slot.committed_len >= win {
            let total_valid = slot.committed_len;
            if nh >= MIN_PARALLEL_HEADS {
                slot.heads
                    .par_iter_mut()
                    .for_each(|h| h.flush_prefix(win, total_valid, k_dim, v_dim));
            } else {
                slot.heads
                    .iter_mut()
                    .for_each(|h| h.flush_prefix(win, total_valid, k_dim, v_dim));
            }
            slot.compact_conv(self.dims, win, total_valid);
            slot.checkpoint_offset += win;
            slot.committed_len = total_valid - win;
            slot.staged_len = slot.committed_len;

            let s_bytes = (nh * self.dims.state_bytes_per_head()) as u64;
            let rec_bytes = (nh * self.dims.delta_record_bytes_per_head()) as u64;
            self.stats.checkpoint_read_bytes += s_bytes;
            self.stats.checkpoint_write_bytes += s_bytes;
            self.stats.record_read_bytes += (win as u64) * rec_bytes;
            self.stats.flushes_executed += 1;
        }
        Ok(())
    }
}

/// Fair zero-allocation fused in-place F32 CPU baseline for apples-to-apples benchmarking.
pub fn fused_inplace_recurrence_cpu(
    inputs: &GdnStepInputs<'_>,
    state: &mut [f32],
    mut trail_out: Option<&mut Vec<Vec<f32>>>,
) -> Result<Tensor> {
    let out_dtype = inputs.q.dtype();
    let out_dev = inputs.q.device().clone();
    let (b_sz, s_len, nh, k_dim) = inputs.q.dims4()?;
    let v_dim = inputs.v.dim(3)?;
    let total_heads = b_sz * nh;
    if state.len() != total_heads * k_dim * v_dim {
        hanzo_ml::bail!("fused_inplace_recurrence_cpu state length mismatch");
    }

    let q_cpu = to_cpu_f32_contiguous(inputs.q)?;
    let k_cpu = to_cpu_f32_contiguous(inputs.k)?;
    let v_cpu = to_cpu_f32_contiguous(inputs.v)?;
    let g_cpu = to_cpu_f32_contiguous(inputs.g)?;
    let b_cpu = to_cpu_f32_contiguous(inputs.beta)?;
    let q_scale = (1.0 / (k_dim as f64).sqrt()) as f32;
    let mut y_out = vec![0.0f32; b_sz * s_len * nh * v_dim];

    with_f32_slice(&q_cpu, |q_s| {
        with_f32_slice(&k_cpu, |k_s| {
            with_f32_slice(&v_cpu, |v_s| {
                with_f32_slice(&g_cpu, |g_s| {
                    with_f32_slice(&b_cpu, |b_s| {
                        for step in 0..s_len {
                            let y_step = &mut y_out[0..b_sz * s_len * nh * v_dim];
                            let mut step_ys = vec![0.0f32; total_heads * v_dim];
                            let run_step =
                                |(bh, (s_head, y_dst)): (usize, (&mut [f32], &mut [f32]))| {
                                    let b_idx = bh / nh;
                                    let h_idx = bh % nh;
                                    let sc = (b_idx * s_len + step) * nh + h_idx;
                                    let q_in = &q_s[sc * k_dim..(sc + 1) * k_dim];
                                    let k_in = &k_s[sc * k_dim..(sc + 1) * k_dim];
                                    let v_in = &v_s[sc * v_dim..(sc + 1) * v_dim];
                                    let decay = g_s[sc].exp();
                                    let beta = b_s[sc];

                                    let mut kv_mem = vec![0.0f32; v_dim];
                                    for r in 0..k_dim {
                                        let kr = k_in[r];
                                        let s_row = &s_head[r * v_dim..(r + 1) * v_dim];
                                        for c in 0..v_dim {
                                            kv_mem[c] += (s_row[c] * decay) * kr;
                                        }
                                    }
                                    let mut u_vec = vec![0.0f32; v_dim];
                                    for c in 0..v_dim {
                                        u_vec[c] = beta * (v_in[c] - kv_mem[c]);
                                    }
                                    for r in 0..k_dim {
                                        let kr = k_in[r];
                                        let qr = q_in[r] * q_scale;
                                        let s_row = &mut s_head[r * v_dim..(r + 1) * v_dim];
                                        for c in 0..v_dim {
                                            let sv = s_row[c] * decay + kr * u_vec[c];
                                            s_row[c] = sv;
                                            y_dst[c] += sv * qr;
                                        }
                                    }
                                };
                            if total_heads >= MIN_PARALLEL_HEADS {
                                state
                                    .par_chunks_mut(k_dim * v_dim)
                                    .zip(step_ys.par_chunks_mut(v_dim))
                                    .enumerate()
                                    .for_each(run_step);
                            } else {
                                state
                                    .chunks_mut(k_dim * v_dim)
                                    .zip(step_ys.chunks_mut(v_dim))
                                    .enumerate()
                                    .for_each(run_step);
                            }
                            for b_idx in 0..b_sz {
                                for h_idx in 0..nh {
                                    let src = (b_idx * nh + h_idx) * v_dim;
                                    let dst = ((b_idx * s_len + step) * nh + h_idx) * v_dim;
                                    y_step[dst..dst + v_dim]
                                        .copy_from_slice(&step_ys[src..src + v_dim]);
                                }
                            }
                            if let Some(ref mut trail) = trail_out {
                                trail.push(state.to_vec());
                            }
                        }
                        Ok(())
                    })
                })
            })
        })
    })?;

    Tensor::from_vec(y_out, (b_sz, s_len, nh, v_dim), &Device::Cpu)?
        .to_device(&out_dev)?
        .to_dtype(out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gdn::recurrence_portable;
    use std::time::Instant;

    const TEST_TOL_REL: f32 = 1e-5;
    const LONG_SEQ_STEPS: usize = 520;
    const BENCH_DECODE_STEPS: usize = 64;
    const BENCH_ROLLBACK_ITERS: usize = 64;
    const REPORT_JSON_ENV: &str = "ENGINE_REPLAY_GDN_BENCH_OUT";

    fn det_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn l2_normalize_heads(mut v: Vec<f32>, dim: usize) -> Vec<f32> {
        for row in v.chunks_exact_mut(dim) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in row {
                *x /= norm;
            }
        }
        v
    }

    fn make_gdn_stream(
        batch: usize,
        seq: usize,
        heads: usize,
        k_dim: usize,
        v_dim: usize,
        seed: u64,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dev = Device::Cpu;
        let q = Tensor::from_vec(
            l2_normalize_heads(det_vec(batch * seq * heads * k_dim, seed + 1, 1.0), k_dim),
            (batch, seq, heads, k_dim),
            &dev,
        )?;
        let k = Tensor::from_vec(
            l2_normalize_heads(det_vec(batch * seq * heads * k_dim, seed + 2, 1.0), k_dim),
            (batch, seq, heads, k_dim),
            &dev,
        )?;
        let v = Tensor::from_vec(
            det_vec(batch * seq * heads * v_dim, seed + 3, 0.5),
            (batch, seq, heads, v_dim),
            &dev,
        )?;
        let g_raw = det_vec(batch * seq * heads, seed + 4, 0.35)
            .into_iter()
            .map(|x| x - 0.40)
            .collect();
        let g = Tensor::from_vec(g_raw, (batch, seq, heads), &dev)?;
        let b_raw = det_vec(batch * seq * heads, seed + 5, 0.45)
            .into_iter()
            .map(|x| x + 0.50)
            .collect();
        let beta = Tensor::from_vec(b_raw, (batch, seq, heads), &dev)?;
        Ok((q, k, v, g, beta))
    }

    fn max_rel_and_abs(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
        let av = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let bv = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(av.len(), bv.len());
        let max_mag = bv.iter().map(|y| y.abs()).fold(0.0f32, f32::max).max(1e-5);
        let mut max_abs = 0.0f32;
        let mut max_elem_rel = 0.0f32;
        for (&x, &y) in av.iter().zip(&bv) {
            assert!(x.is_finite() && y.is_finite());
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
            }
            let r = d / y.abs().max(1e-2);
            if r > max_elem_rel {
                max_elem_rel = r;
            }
        }
        let tensor_rel = max_abs / max_mag;
        Ok((tensor_rel.max(max_elem_rel), max_abs))
    }

    #[test]
    fn raw_input_algebraic_replay_matches_portable_and_residual_cache() -> Result<()> {
        let (b, nh, kd, vd) = (2usize, 8usize, 128usize, 128usize);
        let dims = ReplayLayerDims {
            num_v_heads: nh,
            head_k_dim: kd,
            head_v_dim: vd,
            conv_dim: 32,
            conv_width: 4,
        };
        assert_eq!(dims.state_bytes_per_head(), 65_536);
        assert_eq!(dims.delta_record_floats_per_head(), 257);
        assert_eq!(dims.delta_record_bytes_per_head(), 1_028);
        assert_eq!(dims.raw_record_floats_per_head(), 386);
        assert_eq!(dims.raw_record_bytes_per_head(), 1_544);

        let s0 = Tensor::from_vec(
            det_vec(b * nh * kd * vd, 77, 0.2),
            (b, nh, kd, vd),
            &Device::Cpu,
        )?;
        let conv0 = Tensor::zeros((b, 32, 4), DType::F32, &Device::Cpu)?;

        for &win in &SUPPORTED_REPLAY_WINDOWS {
            let (q, k, v, g, beta) = make_gdn_stream(b, win, nh, kd, vd, 200 + win as u64)?;
            let mut s_ref = s0.copy()?;
            let y_ref = recurrence_portable(&q, &k, &v, &g, &beta, &mut s_ref)?;

            let inputs = GdnStepInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };
            let (y_raw, _u_raw) = replay_from_raw_inputs(&s0, &inputs)?;
            let (rel_raw, _) = max_rel_and_abs(&y_raw, &y_ref)?;
            assert!(rel_raw < TEST_TOL_REL, "win={win} rel_raw={rel_raw:.2e}");

            let mut cache = ReplayGdnCache::from_initial_state(&conv0, &s0, win, 16)?;
            let mut ys = Vec::with_capacity(win);
            for t in 0..win {
                let qt = q.narrow(1, t, 1)?;
                let kt = k.narrow(1, t, 1)?;
                let vt = v.narrow(1, t, 1)?;
                let gt = g.narrow(1, t, 1)?;
                let bt = beta.narrow(1, t, 1)?;
                let step_in = GdnStepInputs {
                    q: &qt,
                    k: &kt,
                    v: &vt,
                    g: &gt,
                    beta: &bt,
                };
                ys.push(cache.step_decode(&step_in)?);
            }
            let y_step = Tensor::cat(&ys, 1)?;
            let s_mat = cache.materialize_state(&Device::Cpu, DType::F32)?;
            let (rel_y, _) = max_rel_and_abs(&y_step, &y_ref)?;
            let (rel_s, _) = max_rel_and_abs(&s_mat, &s_ref)?;
            assert!(
                rel_y < TEST_TOL_REL && rel_s < TEST_TOL_REL,
                "win={win} rel_y={rel_y:.2e} rel_s={rel_s:.2e}"
            );
            assert_eq!(cache.committed_len(), 0);
            assert_eq!(cache.stats().flushes_executed, 1);
            assert_eq!(
                cache.stats().checkpoint_write_bytes,
                (b * nh * 65_536) as u64
            );
        }
        Ok(())
    }

    #[test]
    fn long_sequence_512_plus_parity_across_windows_and_batches() -> Result<()> {
        let (kd, vd) = (128usize, 128usize);
        // Test 520+ steps across all L in {4, 8, 16, 32} and B in {1, 8, 32, 128}.
        for &(batch, nh, steps) in &[
            (1usize, 32usize, LONG_SEQ_STEPS),
            (8, 8, LONG_SEQ_STEPS),
            (32, 2, LONG_SEQ_STEPS),
            (128, 1, LONG_SEQ_STEPS),
        ] {
            let (q, k, v, g, beta) = make_gdn_stream(batch, steps, nh, kd, vd, 500 + batch as u64)?;
            let mut s_ref = Tensor::zeros((batch, nh, kd, vd), DType::F32, &Device::Cpu)?;
            let y_ref = recurrence_portable(&q, &k, &v, &g, &beta, &mut s_ref)?;

            let dims = ReplayLayerDims {
                num_v_heads: nh,
                head_k_dim: kd,
                head_v_dim: vd,
                conv_dim: 16,
                conv_width: 4,
            };

            for &win in &SUPPORTED_REPLAY_WINDOWS {
                let mut cache = ReplayGdnCache::new(batch, dims, win, 16);
                let mut ys = Vec::with_capacity(steps);
                for t in 0..steps {
                    let qt = q.narrow(1, t, 1)?;
                    let kt = k.narrow(1, t, 1)?;
                    let vt = v.narrow(1, t, 1)?;
                    let gt = g.narrow(1, t, 1)?;
                    let bt = beta.narrow(1, t, 1)?;
                    ys.push(cache.step_decode(&GdnStepInputs {
                        q: &qt,
                        k: &kt,
                        v: &vt,
                        g: &gt,
                        beta: &bt,
                    })?);
                }
                let y_got = Tensor::cat(&ys, 1)?;
                let s_got = cache.materialize_state(&Device::Cpu, DType::F32)?;
                let (rel_y, abs_y) = max_rel_and_abs(&y_got, &y_ref)?;
                let (rel_s, abs_s) = max_rel_and_abs(&s_got, &s_ref)?;
                assert!(
                    rel_y < TEST_TOL_REL && rel_s < TEST_TOL_REL,
                    "B={batch} H={nh} L={win} steps={steps}: rel_y={rel_y:.2e} (abs={abs_y:.2e}) rel_s={rel_s:.2e} (abs={abs_s:.2e})"
                );
                assert_eq!(cache.seqlen_offset(), steps);
                assert_eq!(cache.stats().flushes_executed, (steps / win) as u64);
            }
        }
        Ok(())
    }

    #[test]
    fn multi_round_speculative_rollback_and_cow_fork_parity() -> Result<()> {
        let (batch, nh, kd, vd, conv_dim, conv_k) =
            (4usize, 8usize, 128usize, 128usize, 32usize, 4usize);
        let total_true_tokens = 160usize;
        let (q_true, k_true, v_true, g_true, b_true) =
            make_gdn_stream(batch, total_true_tokens, nh, kd, vd, 901)?;
        let conv_true = Tensor::from_vec(
            det_vec(batch * total_true_tokens * conv_dim, 902, 0.5),
            (batch, total_true_tokens, conv_dim),
            &Device::Cpu,
        )?;

        let mut s_ref = Tensor::zeros((batch, nh, kd, vd), DType::F32, &Device::Cpu)?;
        let y_ref_all =
            recurrence_portable(&q_true, &k_true, &v_true, &g_true, &b_true, &mut s_ref)?;
        let expected_final_conv = conv_true
            .narrow(1, total_true_tokens - conv_k, conv_k)?
            .transpose(1, 2)?
            .contiguous()?;

        // Deterministic schedule covering 0% acceptance, partial acceptance, and 100% acceptance.
        let accept_pattern = [
            0usize, 3usize, 5usize, 1usize, 0usize, 4usize, 2usize, 5usize,
        ];
        let draft_width = 5usize;

        for &win in &SUPPORTED_REPLAY_WINDOWS {
            let dims = ReplayLayerDims {
                num_v_heads: nh,
                head_k_dim: kd,
                head_v_dim: vd,
                conv_dim,
                conv_width: conv_k,
            };
            let mut spec = ReplayGdnCache::new(batch, dims, win, 16);

            // Prefill first 13 tokens via chunked step_decode.
            let prefill = 13usize;
            spec.advance_conv(&conv_true.narrow(1, 0, prefill)?, false)?;
            let y_pre = spec.step_decode(&GdnStepInputs {
                q: &q_true.narrow(1, 0, prefill)?,
                k: &k_true.narrow(1, 0, prefill)?,
                v: &v_true.narrow(1, 0, prefill)?,
                g: &g_true.narrow(1, 0, prefill)?,
                beta: &b_true.narrow(1, 0, prefill)?,
            })?;
            let (rel_pre, _) = max_rel_and_abs(&y_pre, &y_ref_all.narrow(1, 0, prefill)?)?;
            assert!(rel_pre < TEST_TOL_REL);

            // CoW fork at unaligned offset 13 (m = 13 % win != 0) before speculative rounds.
            let mut forked = spec.fork();
            let fork_s_expected = spec.materialize_state(&Device::Cpu, DType::F32)?;

            let mut pos = prefill;
            let mut round = 0usize;
            let mut emitted_ys = vec![y_pre];

            while pos < total_true_tokens {
                let rem = total_true_tokens - pos;
                if rem <= draft_width {
                    spec.advance_conv(&conv_true.narrow(1, pos, rem)?, false)?;
                    let y_tail = spec.step_decode(&GdnStepInputs {
                        q: &q_true.narrow(1, pos, rem)?,
                        k: &k_true.narrow(1, pos, rem)?,
                        v: &v_true.narrow(1, pos, rem)?,
                        g: &g_true.narrow(1, pos, rem)?,
                        beta: &b_true.narrow(1, pos, rem)?,
                    })?;
                    emitted_ys.push(y_tail);
                    break;
                }

                let acc = accept_pattern[round % accept_pattern.len()].min(draft_width);
                let rej = draft_width - acc;
                let (q_bad, k_bad, v_bad, g_bad, b_bad) =
                    make_gdn_stream(batch, rej.max(1), nh, kd, vd, 2000 + round as u64)?;
                let conv_bad = Tensor::from_vec(
                    det_vec(batch * rej.max(1) * conv_dim, 3000 + round as u64, 0.9),
                    (batch, rej.max(1), conv_dim),
                    &Device::Cpu,
                )?;

                let cat_seq = |good: &Tensor, bad: &Tensor| -> Result<Tensor> {
                    if acc == 0 {
                        bad.narrow(1, 0, draft_width)?.contiguous()
                    } else if rej == 0 {
                        good.narrow(1, pos, draft_width)?.contiguous()
                    } else {
                        Tensor::cat(&[&good.narrow(1, pos, acc)?, &bad.narrow(1, 0, rej)?], 1)?
                            .contiguous()
                    }
                };

                let st_conv = cat_seq(&conv_true, &conv_bad)?;
                let st_q = cat_seq(&q_true, &q_bad)?;
                let st_k = cat_seq(&k_true, &k_bad)?;
                let st_v = cat_seq(&v_true, &v_bad)?;
                let st_g = cat_seq(&g_true, &g_bad)?;
                let st_b = cat_seq(&b_true, &b_bad)?;

                spec.advance_conv(&st_conv, true)?;
                let y_ver = spec.stage_speculative(&GdnStepInputs {
                    q: &st_q,
                    k: &st_k,
                    v: &st_v,
                    g: &st_g,
                    beta: &st_b,
                })?;
                spec.commit_and_rollback(acc)?;
                assert_eq!(spec.stats().rollback_copied_bytes, 0);

                if acc > 0 {
                    emitted_ys.push(y_ver.narrow(1, 0, acc)?);
                    pos += acc;
                } else {
                    // 0% draft acceptance: emit 1 authoritative bonus token at `pos`.
                    spec.advance_conv(&conv_true.narrow(1, pos, 1)?, false)?;
                    let y1 = spec.step_decode(&GdnStepInputs {
                        q: &q_true.narrow(1, pos, 1)?,
                        k: &k_true.narrow(1, pos, 1)?,
                        v: &v_true.narrow(1, pos, 1)?,
                        g: &g_true.narrow(1, pos, 1)?,
                        beta: &b_true.narrow(1, pos, 1)?,
                    })?;
                    emitted_ys.push(y1);
                    pos += 1;
                }
                round += 1;
            }

            let y_all_got = Tensor::cat(&emitted_ys, 1)?;
            let s_final_got = spec.materialize_state(&Device::Cpu, DType::F32)?;
            let conv_final_got = spec.current_conv_state(&Device::Cpu, DType::F32)?;
            let (rel_y, _) = max_rel_and_abs(&y_all_got, &y_ref_all)?;
            let (rel_s, _) = max_rel_and_abs(&s_final_got, &s_ref)?;
            let (rel_c, _) = max_rel_and_abs(&conv_final_got, &expected_final_conv)?;
            assert!(
                rel_y < TEST_TOL_REL && rel_s < TEST_TOL_REL && rel_c < TEST_TOL_REL,
                "win={win} speculative parity failed: rel_y={rel_y:.2e} rel_s={rel_s:.2e} rel_c={rel_c:.2e}"
            );

            // Verify forked session stayed completely isolated from spec's flushes and rollbacks.
            let fork_s_after = forked.materialize_state(&Device::Cpu, DType::F32)?;
            let (rel_fork, _) = max_rel_and_abs(&fork_s_after, &fork_s_expected)?;
            assert_eq!(rel_fork, 0.0, "CoW fork mutated by parent flush");
            let y_fork_next = forked.step_decode(&GdnStepInputs {
                q: &q_true.narrow(1, prefill, 4)?,
                k: &k_true.narrow(1, prefill, 4)?,
                v: &v_true.narrow(1, prefill, 4)?,
                g: &g_true.narrow(1, prefill, 4)?,
                beta: &b_true.narrow(1, prefill, 4)?,
            })?;
            let (rel_fn, _) = max_rel_and_abs(&y_fork_next, &y_ref_all.narrow(1, prefill, 4)?)?;
            assert!(rel_fn < TEST_TOL_REL);
        }
        Ok(())
    }

    #[derive(Serialize)]
    struct BenchDecodeRow {
        batch_size: usize,
        num_v_heads: usize,
        k_dim: usize,
        v_dim: usize,
        steps: usize,
        window_size: Option<usize>,
        engine: &'static str,
        wall_ms: f64,
        seq_tokens_per_sec: f64,
        step_latency_us: f64,
        read_bytes_per_seq_token: f64,
        write_bytes_per_seq_token: f64,
        total_rw_bytes_per_seq_token: f64,
        write_reduction_vs_baseline: f64,
        total_rw_reduction_vs_baseline: f64,
        max_rel_err_vs_portable: f32,
    }

    #[derive(Serialize)]
    struct BenchSpecVerifyRow {
        batch_size: usize,
        num_v_heads: usize,
        window_size: Option<usize>,
        draft_len: usize,
        accepted_len: usize,
        engine: &'static str,
        verify_plus_rollback_us: f64,
        rollback_only_ns: f64,
        rollback_copied_bytes: u64,
        total_rw_bytes_per_accepted_token: f64,
        rw_reduction_vs_portable_trail: f64,
        rw_reduction_vs_fused_trail: f64,
    }

    #[derive(Serialize)]
    struct QualificationRow {
        batch_size: usize,
        num_v_heads: usize,
        window_size: usize,
        steps: usize,
        scenario: &'static str,
        max_rel_output: f32,
        max_abs_output: f32,
        max_rel_state: f32,
        max_abs_state: f32,
        passed: bool,
    }

    #[derive(Serialize)]
    struct ReplayGdnBenchmarkReport {
        hardware: &'static str,
        head_geometry: &'static str,
        state_bytes_per_head: usize,
        delta_record_bytes_per_head: usize,
        raw_input_record_bytes_per_head: usize,
        per_step_write_compression_residual_x: f64,
        per_step_write_compression_raw_x: f64,
        analytical_optimal_single_decode_window: f64,
        qualification_512_plus: Vec<QualificationRow>,
        decode_benchmarks: Vec<BenchDecodeRow>,
        speculative_verify_rollback_benchmarks: Vec<BenchSpecVerifyRow>,
    }

    #[test]
    #[ignore = "CPU benchmark suite generating JSON report"]
    fn bench_replay_gdn_cpu_suite() -> Result<()> {
        let (nh, kd, vd) = (32usize, 128usize, 128usize);
        let dims = ReplayLayerDims {
            num_v_heads: nh,
            head_k_dim: kd,
            head_v_dim: vd,
            conv_dim: 256,
            conv_width: 4,
        };

        // 1. 512+ step numerical qualification table across B in {1, 8, 32, 128} and L in {4, 8, 16, 32}.
        let mut qualification = Vec::new();
        let qual_steps = 512usize;
        for &batch in &[1usize, 8, 32, 128] {
            let q_nh = if batch <= 8 {
                32
            } else if batch == 32 {
                8
            } else {
                4
            };
            let q_dims = ReplayLayerDims {
                num_v_heads: q_nh,
                ..dims
            };
            let (q, k, v, g, beta) =
                make_gdn_stream(batch, qual_steps, q_nh, kd, vd, 7000 + batch as u64)?;
            let mut s_port = Tensor::zeros((batch, q_nh, kd, vd), DType::F32, &Device::Cpu)?;
            let y_port = recurrence_portable(&q, &k, &v, &g, &beta, &mut s_port)?;

            for &win in &SUPPORTED_REPLAY_WINDOWS {
                let mut cache = ReplayGdnCache::new(batch, q_dims, win, 16);
                let mut ys = Vec::with_capacity(qual_steps);
                for t in 0..qual_steps {
                    ys.push(cache.step_decode(&GdnStepInputs {
                        q: &q.narrow(1, t, 1)?,
                        k: &k.narrow(1, t, 1)?,
                        v: &v.narrow(1, t, 1)?,
                        g: &g.narrow(1, t, 1)?,
                        beta: &beta.narrow(1, t, 1)?,
                    })?);
                }
                let y_got = Tensor::cat(&ys, 1)?;
                let s_got = cache.materialize_state(&Device::Cpu, DType::F32)?;
                let (ry, ay) = max_rel_and_abs(&y_got, &y_port)?;
                let (rs, as_) = max_rel_and_abs(&s_got, &s_port)?;
                assert!(ry < TEST_TOL_REL && rs < TEST_TOL_REL);
                qualification.push(QualificationRow {
                    batch_size: batch,
                    num_v_heads: q_nh,
                    window_size: win,
                    steps: qual_steps,
                    scenario: "sequential_decode_512",
                    max_rel_output: ry,
                    max_abs_output: ay,
                    max_rel_state: rs,
                    max_abs_state: as_,
                    passed: ry < TEST_TOL_REL && rs < TEST_TOL_REL,
                });
            }
        }

        // 2. Single-token decode throughput & memory traffic across B in {1, 8, 32, 128} and L in {4, 8, 16, 32}.
        let mut decode_rows = Vec::new();
        for &batch in &[1usize, 8, 32, 128] {
            let steps = if batch <= 32 { BENCH_DECODE_STEPS } else { 32 };
            let (q, k, v, g, beta) =
                make_gdn_stream(batch, steps, nh, kd, vd, 8000 + batch as u64)?;
            let step_slices: Vec<_> = (0..steps)
                .map(|t| {
                    Ok((
                        q.narrow(1, t, 1)?.contiguous()?,
                        k.narrow(1, t, 1)?.contiguous()?,
                        v.narrow(1, t, 1)?.contiguous()?,
                        g.narrow(1, t, 1)?.contiguous()?,
                        beta.narrow(1, t, 1)?.contiguous()?,
                    ))
                })
                .collect::<Result<_>>()?;

            let (base_r, base_w) =
                ReplayGdnMemoryStats::baseline_decode_rw_bytes(batch * nh, kd, vd, steps as u64);
            let seq_tokens = (batch * steps) as f64;
            let base_r_tok = base_r as f64 / seq_tokens;
            let base_w_tok = base_w as f64 / seq_tokens;
            let base_rw_tok = base_r_tok + base_w_tok;

            // Reference: recurrence_portable step-by-step.
            let mut s_port = Tensor::zeros((batch, nh, kd, vd), DType::F32, &Device::Cpu)?;
            let t0 = Instant::now();
            let mut y_port_last = Tensor::zeros((batch, 1, nh, vd), DType::F32, &Device::Cpu)?;
            for (qt, kt, vt, gt, bt) in &step_slices {
                y_port_last = recurrence_portable(qt, kt, vt, gt, bt, &mut s_port)?;
            }
            let port_ms = t0.elapsed().as_secs_f64() * 1e3;
            decode_rows.push(BenchDecodeRow {
                batch_size: batch,
                num_v_heads: nh,
                k_dim: kd,
                v_dim: vd,
                steps,
                window_size: None,
                engine: "recurrence_portable_f32",
                wall_ms: port_ms,
                seq_tokens_per_sec: seq_tokens / (port_ms * 1e-3),
                step_latency_us: (port_ms * 1e3) / (steps as f64),
                read_bytes_per_seq_token: base_r_tok,
                write_bytes_per_seq_token: base_w_tok,
                total_rw_bytes_per_seq_token: base_rw_tok,
                write_reduction_vs_baseline: 1.0,
                total_rw_reduction_vs_baseline: 1.0,
                max_rel_err_vs_portable: 0.0,
            });

            // Fair zero-alloc fused in-place F32 CPU baseline.
            let mut s_inplace = vec![0.0f32; batch * nh * kd * vd];
            // Warmup 2 steps
            for (qt, kt, vt, gt, bt) in step_slices.iter().take(2) {
                let _ = fused_inplace_recurrence_cpu(
                    &GdnStepInputs {
                        q: qt,
                        k: kt,
                        v: vt,
                        g: gt,
                        beta: bt,
                    },
                    &mut s_inplace,
                    None,
                )?;
            }
            s_inplace.fill(0.0);
            let t_in = Instant::now();
            let mut y_in_last = Tensor::zeros((batch, 1, nh, vd), DType::F32, &Device::Cpu)?;
            for (qt, kt, vt, gt, bt) in &step_slices {
                y_in_last = fused_inplace_recurrence_cpu(
                    &GdnStepInputs {
                        q: qt,
                        k: kt,
                        v: vt,
                        g: gt,
                        beta: bt,
                    },
                    &mut s_inplace,
                    None,
                )?;
            }
            let in_ms = t_in.elapsed().as_secs_f64() * 1e3;
            let (rel_in, _) = max_rel_and_abs(&y_in_last, &y_port_last)?;
            decode_rows.push(BenchDecodeRow {
                batch_size: batch,
                num_v_heads: nh,
                k_dim: kd,
                v_dim: vd,
                steps,
                window_size: None,
                engine: "fused_inplace_f32_cpu",
                wall_ms: in_ms,
                seq_tokens_per_sec: seq_tokens / (in_ms * 1e-3),
                step_latency_us: (in_ms * 1e3) / (steps as f64),
                read_bytes_per_seq_token: base_r_tok,
                write_bytes_per_seq_token: base_w_tok,
                total_rw_bytes_per_seq_token: base_rw_tok,
                write_reduction_vs_baseline: 1.0,
                total_rw_reduction_vs_baseline: 1.0,
                max_rel_err_vs_portable: rel_in,
            });

            for &win in &SUPPORTED_REPLAY_WINDOWS {
                let mut warm = ReplayGdnCache::new(batch, dims, win, 16);
                for (qt, kt, vt, gt, bt) in step_slices.iter().take(2) {
                    let _ = warm.step_decode(&GdnStepInputs {
                        q: qt,
                        k: kt,
                        v: vt,
                        g: gt,
                        beta: bt,
                    })?;
                }
                let mut cache = ReplayGdnCache::new(batch, dims, win, 16);
                let t_rep = Instant::now();
                let mut y_rep_last = Tensor::zeros((batch, 1, nh, vd), DType::F32, &Device::Cpu)?;
                for (qt, kt, vt, gt, bt) in &step_slices {
                    y_rep_last = cache.step_decode(&GdnStepInputs {
                        q: qt,
                        k: kt,
                        v: vt,
                        g: gt,
                        beta: bt,
                    })?;
                }
                let rep_ms = t_rep.elapsed().as_secs_f64() * 1e3;
                let (rel_rep, _) = max_rel_and_abs(&y_rep_last, &y_port_last)?;
                let st = cache.stats();
                let r_tok = st.total_recurrent_read_bytes() as f64 / seq_tokens;
                let w_tok = st.total_recurrent_write_bytes() as f64 / seq_tokens;
                let rw_tok = r_tok + w_tok;
                decode_rows.push(BenchDecodeRow {
                    batch_size: batch,
                    num_v_heads: nh,
                    k_dim: kd,
                    v_dim: vd,
                    steps,
                    window_size: Some(win),
                    engine: "replay_gdn",
                    wall_ms: rep_ms,
                    seq_tokens_per_sec: seq_tokens / (rep_ms * 1e-3),
                    step_latency_us: (rep_ms * 1e3) / (steps as f64),
                    read_bytes_per_seq_token: r_tok,
                    write_bytes_per_seq_token: w_tok,
                    total_rw_bytes_per_seq_token: rw_tok,
                    write_reduction_vs_baseline: base_w_tok / w_tok,
                    total_rw_reduction_vs_baseline: base_rw_tok / rw_tok,
                    max_rel_err_vs_portable: rel_rep,
                });
            }
        }

        // 3. Speculative verify (W=5: anchor + 4 drafts, accept 3, reject 2) + rollback latency.
        let mut spec_rows = Vec::new();
        let (verify_w, accepted_w) = (5usize, 3usize);
        let rejected_w = verify_w - accepted_w;
        for &batch in &[1usize, 8, 32, 128] {
            let iters = if batch <= 32 {
                BENCH_ROLLBACK_ITERS
            } else {
                16
            };
            let (q, k, v, g, beta) =
                make_gdn_stream(batch, verify_w, nh, kd, vd, 9500 + batch as u64)?;
            let inputs = GdnStepInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };

            let (p_r, p_w) = ReplayGdnMemoryStats::baseline_spec_verify_rw_bytes(
                batch * nh,
                kd,
                vd,
                verify_w as u64,
                rejected_w as u64,
                false,
            );
            let (f_r, f_w) = ReplayGdnMemoryStats::baseline_spec_verify_rw_bytes(
                batch * nh,
                kd,
                vd,
                verify_w as u64,
                rejected_w as u64,
                true,
            );
            let port_rw_per_acc = (p_r + p_w) as f64 / (batch * accepted_w) as f64;
            let fused_rw_per_acc = (f_r + f_w) as f64 / (batch * accepted_w) as f64;
            let state_copy_bytes = (batch * nh * kd * vd * F32_BYTES) as u64;

            // Baseline fused in-place with trail + memcpy rollback.
            let mut s_state = vec![0.0f32; batch * nh * kd * vd];
            let mut total_verify_ns = 0.0f64;
            let mut total_rb_ns = 0.0f64;
            for _ in 0..iters {
                let mut trail = Vec::with_capacity(verify_w);
                let tv = Instant::now();
                let _ = fused_inplace_recurrence_cpu(&inputs, &mut s_state, Some(&mut trail))?;
                let tr = Instant::now();
                s_state.copy_from_slice(&trail[accepted_w - 1]);
                let rb_ns = tr.elapsed().as_nanos() as f64;
                total_rb_ns += rb_ns;
                total_verify_ns += tv.elapsed().as_nanos() as f64;
            }
            spec_rows.push(BenchSpecVerifyRow {
                batch_size: batch,
                num_v_heads: nh,
                window_size: None,
                draft_len: verify_w,
                accepted_len: accepted_w,
                engine: "fused_inplace_trail_memcpy",
                verify_plus_rollback_us: (total_verify_ns / iters as f64) * 1e-3,
                rollback_only_ns: total_rb_ns / iters as f64,
                rollback_copied_bytes: state_copy_bytes,
                total_rw_bytes_per_accepted_token: port_rw_per_acc,
                rw_reduction_vs_portable_trail: 1.0,
                rw_reduction_vs_fused_trail: fused_rw_per_acc / port_rw_per_acc,
            });

            for &win in &SUPPORTED_REPLAY_WINDOWS {
                let mut cache = ReplayGdnCache::new(batch, dims, win, 16);
                let mut rep_verify_ns = 0.0f64;
                let mut rep_rb_ns = 0.0f64;
                let mut pure_rb_only_ns = 0.0f64;

                // Measure pure O(1) rollback latency (0 accepted -> never triggers window flush).
                let mut rb_probe = ReplayGdnCache::new(batch, dims, 32, 16);
                for _ in 0..iters {
                    let _ = rb_probe.stage_speculative(&inputs)?;
                    let tr = Instant::now();
                    rb_probe.commit_and_rollback(0)?;
                    pure_rb_only_ns += tr.elapsed().as_nanos() as f64;
                }

                for _ in 0..iters {
                    let tv = Instant::now();
                    let _ = cache.stage_speculative(&inputs)?;
                    let tr = Instant::now();
                    cache.commit_and_rollback(accepted_w)?;
                    rep_rb_ns += tr.elapsed().as_nanos() as f64;
                    rep_verify_ns += tv.elapsed().as_nanos() as f64;
                }
                let _ = rep_rb_ns;
                let st = cache.stats();
                let total_acc = (iters * batch * accepted_w) as f64;
                let rw_per_acc = st.total_recurrent_rw_bytes() as f64 / total_acc;
                spec_rows.push(BenchSpecVerifyRow {
                    batch_size: batch,
                    num_v_heads: nh,
                    window_size: Some(win),
                    draft_len: verify_w,
                    accepted_len: accepted_w,
                    engine: "replay_gdn_zero_copy",
                    verify_plus_rollback_us: (rep_verify_ns / iters as f64) * 1e-3,
                    rollback_only_ns: pure_rb_only_ns / iters as f64,
                    rollback_copied_bytes: st.rollback_copied_bytes,
                    total_rw_bytes_per_accepted_token: rw_per_acc,
                    rw_reduction_vs_portable_trail: port_rw_per_acc / rw_per_acc,
                    rw_reduction_vs_fused_trail: fused_rw_per_acc / rw_per_acc,
                });
            }
        }

        let s_b = dims.state_bytes_per_head();
        let d_b = dims.delta_record_bytes_per_head();
        let r_b = dims.raw_record_bytes_per_head();
        let report = ReplayGdnBenchmarkReport {
            hardware:
                "ARMv9.2 Cortex-X925 + Cortex-A725 (20 cores, LPDDR5X unified memory, CPU F32)",
            head_geometry: "H_v=32, K=128, V=128 (64 KiB F32 state/head)",
            state_bytes_per_head: s_b,
            delta_record_bytes_per_head: d_b,
            raw_input_record_bytes_per_head: r_b,
            per_step_write_compression_residual_x: s_b as f64 / d_b as f64,
            per_step_write_compression_raw_x: s_b as f64 / r_b as f64,
            analytical_optimal_single_decode_window: ((2.0 * s_b as f64) / (d_b as f64)).sqrt(),
            qualification_512_plus: qualification,
            decode_benchmarks: decode_rows,
            speculative_verify_rollback_benchmarks: spec_rows,
        };

        let json_str = serde_json::to_string_pretty(&report).expect("serialize benchmark report");
        if let Ok(path) = std::env::var(REPORT_JSON_ENV) {
            if !path.is_empty() {
                std::fs::write(&path, &json_str).expect("write benchmark JSON");
            }
        }
        println!("{json_str}");
        Ok(())
    }
}
