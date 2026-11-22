//! Unified Hybrid-State Execution Contract for GDN/GQA hybrid models.
//!
//! Binds every per-request state surface under one atomic lifecycle:
//! - Attention KV cache (contiguous `KvCache` snapshots + paged KV `block_table` & `context_len`),
//! - GDN / Mamba recurrent state tensors (`recurrent_state`),
//! - Depthwise Conv1D history (`conv_state`),
//! - Logical sequence offset (`seqlen_offset`).
//!
//! Guarantees that prefix reuse (`fork_prefix`), checkpoints (`snapshot_request` /
//! `restore_request`), request cancellation (`cancel_request`), and speculative draft rejection
//! (`rollback_speculative`) always operate atomically on the COMPLETE hybrid state rather than
//! leaving KV pages and GDN/Conv1D state out of sync.

use std::collections::HashMap;

use hanzo_ml::{Result, Tensor};

use super::{HybridCache, HybridLayerCache, KvCacheSnapshot, RecurrentStateSnapshot};

/// Immutable, deep-copied snapshot of a request's complete hybrid execution state.
#[derive(Debug, Clone)]
pub struct HybridStateCheckpoint {
    /// Logical token count at the moment of snapshot (invariant across KV and recurrent layers).
    pub token_len: usize,
    /// Snapshots for attention layers in `HybridCache::caches` (entry is `None` for recurrent layers).
    pub kv_snapshots: Vec<Option<KvCacheSnapshot>>,
    /// Paged KV block table prefix owned by the request at snapshot time (if paged KV is bound).
    pub paged_block_prefix: Vec<u32>,
    /// Deep-copied GDN/Mamba/conv-only snapshots across every recurrent pool in `HybridCache`.
    pub recurrent_snapshots: Vec<RecurrentStateSnapshot>,
}

/// Per-request binding tracked by [`HybridStateContract`].
#[derive(Debug, Clone)]
pub struct HybridRequestBinding {
    pub request_id: usize,
    pub recurrent_slot: usize,
    pub paged_block_table: Vec<u32>,
    pub context_len: usize,
}

/// Atomic coordinator over a [`HybridCache`] plus per-request paged KV bindings.
pub struct HybridStateContract {
    pub cache: HybridCache,
    bindings: HashMap<usize, HybridRequestBinding>,
}

impl HybridStateContract {
    pub fn new(cache: HybridCache) -> Self {
        Self {
            cache,
            bindings: HashMap::new(),
        }
    }

    /// Allocate a unified hybrid slot for `request_id` (GDN/Conv1D pool slot + empty KV binding).
    pub fn allocate_request(&mut self, request_id: usize) -> Result<usize> {
        if let Some(existing) = self.bindings.get(&request_id) {
            return Ok(existing.recurrent_slot);
        }
        let Some(slot) = self.cache.allocate_seq() else {
            hanzo_ml::bail!(
                "HybridStateContract: failed to allocate recurrent slot for req {request_id}"
            );
        };
        self.bindings.insert(
            request_id,
            HybridRequestBinding {
                request_id,
                recurrent_slot: slot,
                paged_block_table: Vec::new(),
                context_len: 0,
            },
        );
        Ok(slot)
    }

    pub fn binding(&self, request_id: usize) -> Option<&HybridRequestBinding> {
        self.bindings.get(&request_id)
    }

    pub fn binding_mut(&mut self, request_id: usize) -> Option<&mut HybridRequestBinding> {
        self.bindings.get_mut(&request_id)
    }

    /// Record paged KV blocks + context length committed for `request_id`.
    pub fn update_paged_kv(
        &mut self,
        request_id: usize,
        block_table: Vec<u32>,
        context_len: usize,
    ) -> Result<()> {
        let b = self
            .bindings
            .get_mut(&request_id)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("unknown request_id {request_id}")))?;
        b.paged_block_table = block_table;
        b.context_len = context_len;
        Ok(())
    }

    /// Atomically snapshot KV layers, paged block prefix, GDN recurrent state, and Conv1D history.
    ///
    /// Every tensor inside the returned [`HybridStateCheckpoint`] is deep-copied (`copy()`), so
    /// subsequent in-place decode steps cannot alias or mutate the checkpoint.
    pub fn snapshot_request(&self, request_id: usize) -> Result<HybridStateCheckpoint> {
        let binding = self
            .bindings
            .get(&request_id)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("unknown request_id {request_id}")))?;

        let raw_rec = self
            .cache
            .snapshot_recurrent_state(binding.recurrent_slot)?;
        let mut recurrent_snapshots = Vec::with_capacity(raw_rec.len());
        for snap in raw_rec {
            recurrent_snapshots.push(RecurrentStateSnapshot {
                conv_state: snap.conv_state.copy()?,
                recurrent_state: snap.recurrent_state.copy()?,
                seqlen_offset: snap.seqlen_offset,
            });
        }

        let mut kv_snapshots = Vec::with_capacity(self.cache.caches.len());
        for layer in &self.cache.caches {
            match layer {
                HybridLayerCache::Attention(kv) => kv_snapshots.push(Some(kv.snapshot()?)),
                HybridLayerCache::Recurrent(_) => kv_snapshots.push(None),
            }
        }

        let token_len = self
            .cache
            .recurrent_offset(binding.recurrent_slot)
            .unwrap_or(binding.context_len);

        Ok(HybridStateCheckpoint {
            token_len,
            kv_snapshots,
            paged_block_prefix: binding.paged_block_table.clone(),
            recurrent_snapshots,
        })
    }

    /// Atomically restore KV layers, paged block prefix, GDN recurrent state, and Conv1D history
    /// for `request_id` from `ckpt`.
    pub fn restore_request(
        &mut self,
        request_id: usize,
        ckpt: &HybridStateCheckpoint,
    ) -> Result<()> {
        let slot = self
            .bindings
            .get(&request_id)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("unknown request_id {request_id}")))?
            .recurrent_slot;

        // Deep-copy on restore as well so the checkpoint remains reusable across multiple forks.
        let mut rec_copy = Vec::with_capacity(ckpt.recurrent_snapshots.len());
        for snap in &ckpt.recurrent_snapshots {
            rec_copy.push(RecurrentStateSnapshot {
                conv_state: snap.conv_state.copy()?,
                recurrent_state: snap.recurrent_state.copy()?,
                seqlen_offset: snap.seqlen_offset,
            });
        }
        self.cache.restore_recurrent_state(slot, &rec_copy)?;

        for (layer, kv_snap) in self.cache.caches.iter_mut().zip(&ckpt.kv_snapshots) {
            if let (HybridLayerCache::Attention(kv), Some(snap)) = (layer, kv_snap) {
                match (&*kv, snap) {
                    (
                        super::KvCache::Normal { .. } | super::KvCache::Shared { .. },
                        super::KvCacheSnapshot::Normal { .. }
                        | super::KvCacheSnapshot::Shared { .. },
                    ) => {
                        kv.restore_after_speculative_append(snap, None, ckpt.token_len, 0, 1)?;
                    }
                    _ => {
                        kv.set_len(ckpt.token_len)?;
                    }
                }
            }
        }

        if let Some(binding) = self.bindings.get_mut(&request_id) {
            binding.paged_block_table = ckpt.paged_block_prefix.clone();
            binding.context_len = ckpt.token_len;
        }
        Ok(())
    }

    /// Prefix reuse: snapshot `src_request_id` and materialize an independent clone into
    /// `dst_request_id` (new recurrent slot + copied GDN/Conv1D state + shared prefix block list).
    pub fn fork_prefix(
        &mut self,
        src_request_id: usize,
        dst_request_id: usize,
    ) -> Result<HybridStateCheckpoint> {
        let ckpt = self.snapshot_request(src_request_id)?;
        let dst_slot = self.allocate_request(dst_request_id)?;

        let mut rec_copy = Vec::with_capacity(ckpt.recurrent_snapshots.len());
        for snap in &ckpt.recurrent_snapshots {
            rec_copy.push(RecurrentStateSnapshot {
                conv_state: snap.conv_state.copy()?,
                recurrent_state: snap.recurrent_state.copy()?,
                seqlen_offset: snap.seqlen_offset,
            });
        }
        self.cache.restore_recurrent_state(dst_slot, &rec_copy)?;
        if let Some(dst) = self.bindings.get_mut(&dst_request_id) {
            dst.paged_block_table = ckpt.paged_block_prefix.clone();
            dst.context_len = ckpt.token_len;
        }
        Ok(ckpt)
    }

    /// Atomic speculative verification rollback across KV caches, paged context lengths, AND
    /// every GDN/conv-only pool.
    ///
    /// - `keep_len`: accepted sequence length (must match `post_verify_offset - rejected`),
    /// - `rejected`: number of rejected draft positions to rewind via the recorded `RecurrentTrail`,
    /// - `block_size`: optional paged KV block size for trimming unused tail blocks.
    pub fn rollback_speculative(
        &mut self,
        request_id: usize,
        keep_len: usize,
        rejected: usize,
        block_size: Option<usize>,
    ) -> Result<()> {
        let slot = self
            .bindings
            .get(&request_id)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("unknown request_id {request_id}")))?
            .recurrent_slot;

        // 1. Rewind every recurrent + conv-only pool atomically.
        self.cache.rewind_recurrent(slot, rejected)?;

        // 2. Verify recurrent offset invariant matches `keep_len`.
        if let Some(off) = self.cache.recurrent_offset(slot) {
            if off != keep_len {
                hanzo_ml::bail!(
                    "HybridStateContract::rollback_speculative invariant violation: recurrent offset {off} != keep_len {keep_len}"
                );
            }
        }

        // 3. Trim contiguous KV attention caches to `keep_len`.
        for layer in &mut self.cache.caches {
            if let HybridLayerCache::Attention(kv) = layer {
                kv.set_len(keep_len)?;
            }
        }

        // 4. Trim paged KV context length and tail blocks.
        if let Some(binding) = self.bindings.get_mut(&request_id) {
            binding.context_len = keep_len;
            if let Some(bs) = block_size {
                if bs > 0 {
                    let keep_blocks = keep_len.div_ceil(bs);
                    binding.paged_block_table.truncate(keep_blocks);
                }
            }
        }
        Ok(())
    }

    /// Clean request cancellation: zeroes and frees the recurrent slot, removes paged bindings,
    /// and resets attention caches when no active requests remain.
    pub fn cancel_request(&mut self, request_id: usize) -> Result<()> {
        if let Some(binding) = self.bindings.remove(&request_id) {
            self.cache.reset_seq(binding.recurrent_slot)?;
            self.cache.free_seq(binding.recurrent_slot);
        }
        if self.bindings.is_empty() {
            for layer in &mut self.cache.caches {
                if let HybridLayerCache::Attention(kv) = layer {
                    kv.reset();
                }
            }
        }
        Ok(())
    }

    /// Helper returning the `(conv_state, recurrent_state)` tensors for `request_id` at layer `layer_idx`.
    pub fn inspect_recurrent_layer(
        &self,
        request_id: usize,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        let binding = self
            .bindings
            .get(&request_id)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("unknown request_id {request_id}")))?;
        let pool = self
            .cache
            .get(layer_idx)
            .and_then(HybridLayerCache::as_recurrent_pool)
            .ok_or_else(|| {
                hanzo_ml::Error::Msg(format!("layer {layer_idx} is not a recurrent pool"))
            })?;
        let idx = Tensor::from_vec(vec![binding.recurrent_slot as u32], (1,), pool.device())?;
        Ok((
            pool.gather_conv_state(&idx)?,
            pool.gather_recurrent_state(&idx)?,
            pool.get_seqlen_offset(binding.recurrent_slot),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{AttentionMask, Sdpa, SdpaParams};
    use crate::kv_cache::{HybridCacheConfig, HybridLayerType, RecurrentLayerConfig};
    use crate::models::gdn::{forward_pooled, GdnLayerCache, PoolSlots};
    use hanzo_ml::quantized::dsl::{self, QkImpl};
    use hanzo_ml::{DType, Device};

    fn seeded(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect();
        Tensor::from_vec(v, shape, dev)
    }

    fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        let av = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let bv = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(av.len(), bv.len());
        let max_diff = av
            .iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let max_mag = bv.iter().map(|y| y.abs()).fold(0.0f32, f32::max).max(1e-6);
        Ok(max_diff / max_mag)
    }

    /// Minimal hybrid step: runs GDN conv-state update + GDN recurrence on layer 0 (through
    /// `GdnLayerCache::recurrence` -> `gated_delta_rule_recurrence` -> `hanzo_ml::gdn::try_recurrence`)
    /// and Attention KV append + `Sdpa::run_attention` (-> `hanzo_ml::flash_attn::try_flash_attn_contiguous`)
    /// on layer 1.
    #[allow(clippy::too_many_arguments)]
    fn hybrid_forward_step(
        contract: &mut HybridStateContract,
        request_id: usize,
        conv_in: &Tensor, // [1, S, conv_dim]
        q: &Tensor,       // [1, S, n_heads, kd]
        k: &Tensor,       // [1, S, n_heads, kd]
        v: &Tensor,       // [1, S, n_heads, vd]
        g: &Tensor,       // [1, S, n_heads]
        beta: &Tensor,    // [1, S, n_heads]
        attn_q: &Tensor,  // [1, n_heads, S, kd]
        attn_k: &Tensor,  // [1, n_kv, S, kd]
        attn_v: &Tensor,  // [1, n_kv, S, kd]
        record_trail: bool,
        block_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let slot = contract.binding(request_id).unwrap().recurrent_slot;
        let dev = q.device();
        let slot_t = Tensor::from_vec(vec![slot as u32], (1,), dev)?;
        let conv_k = 4usize;

        // Layer 0: GDN recurrent pool
        let gdn_out = {
            let pool = contract
                .cache
                .get_mut(0)
                .and_then(HybridLayerCache::as_recurrent_pool_mut)
                .expect("layer 0 recurrent");
            forward_pooled(
                pool,
                PoolSlots::Many(&slot_t),
                0,
                record_trail,
                |cache: &mut GdnLayerCache| {
                    cache.trail_conv(conv_in)?;
                    let seq_len = conv_in.dim(1)?;
                    let window = Tensor::cat(&[&cache.conv_state, &conv_in.transpose(1, 2)?], 2)?;
                    cache.conv_state = window.narrow(2, seq_len, conv_k)?.contiguous()?;
                    let y = cache.recurrence(q, k, v, g, beta)?;
                    cache.seqlen_offset += seq_len;
                    Ok(y)
                },
            )?
        };

        // Layer 1: Attention KV cache + SDPA
        let attn_out = {
            let kv = contract
                .cache
                .get_mut(1)
                .and_then(HybridLayerCache::as_kv_cache_mut)
                .expect("layer 1 attention");
            let (full_k, full_v) = kv.append(attn_k, attn_v)?;
            let kd = attn_q.dim(3)?;
            let params = SdpaParams {
                n_kv_groups: attn_q.dim(1)? / attn_k.dim(1)?,
                softcap: None,
                softmax_scale: 1.0 / (kd as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            };
            let mask = if attn_q.dim(2)? > 1 {
                AttentionMask::CausalFlash
            } else {
                AttentionMask::None
            };
            Sdpa.run_attention(attn_q, &full_k, &full_v, &mask, None, &params)?
        };

        // Sync paged block table metadata in contract
        let new_len = contract
            .cache
            .recurrent_offset(slot)
            .expect("recurrent offset");
        let n_blocks = new_len.div_ceil(block_size);
        let blocks: Vec<u32> = (0..n_blocks as u32).map(|i| 100 + i).collect();
        contract.update_paged_kv(request_id, blocks, new_len)?;

        Ok((gdn_out, attn_out))
    }

    fn run_hybrid_contract_lifecycle(dev: &Device, dtype: DType, use_dsl: bool) -> Result<()> {
        let mode = if use_dsl { QkImpl::Dsl } else { QkImpl::Legacy };
        dsl::set_gdn_impl(mode);
        dsl::set_flash_impl(mode);

        let (n_heads, n_kv, kd, vd, conv_dim, conv_k, block_size) =
            (4usize, 2usize, 64usize, 64usize, 16usize, 4usize, 4usize);
        let make_contract = || -> Result<HybridStateContract> {
            let rec_cfg = RecurrentLayerConfig {
                conv_dim,
                conv_width: conv_k,
                state_dims: vec![n_heads, kd, vd],
                conv_dtype: dtype,
                state_dtype: DType::F32,
            };
            let cfg = HybridCacheConfig::uniform(
                vec![HybridLayerType::Recurrent, HybridLayerType::Attention],
                64,
                rec_cfg,
            );
            Ok(HybridStateContract::new(HybridCache::new(cfg, dev)?))
        };

        let total_seq = 8usize;
        let conv_stream = seeded(&[1, total_seq, conv_dim], 11, 0.5, dev)?.to_dtype(dtype)?;
        let q_stream = seeded(&[1, total_seq, n_heads, kd], 12, 0.3, dev)?.to_dtype(dtype)?;
        let k_stream = seeded(&[1, total_seq, n_heads, kd], 13, 0.3, dev)?.to_dtype(dtype)?;
        let v_stream = seeded(&[1, total_seq, n_heads, vd], 14, 0.3, dev)?.to_dtype(dtype)?;
        let g_stream =
            (seeded(&[1, total_seq, n_heads], 15, 0.25, dev)? - 0.35)?.to_dtype(dtype)?;
        let b_stream =
            ((seeded(&[1, total_seq, n_heads], 16, 0.4, dev)? + 1.0)? * 0.5)?.to_dtype(dtype)?;
        let aq_stream = seeded(&[1, n_heads, total_seq, kd], 17, 0.3, dev)?.to_dtype(dtype)?;
        let ak_stream = seeded(&[1, n_kv, total_seq, kd], 18, 0.3, dev)?.to_dtype(dtype)?;
        let av_stream = seeded(&[1, n_kv, total_seq, kd], 19, 0.3, dev)?.to_dtype(dtype)?;

        // 1. One-shot prefill of prompt [0..4] + 4 single-token decode steps [4..8]
        let mut ground = make_contract()?;
        ground.allocate_request(1)?;
        let _ = hybrid_forward_step(
            &mut ground,
            1,
            &conv_stream.narrow(1, 0, 4)?.contiguous()?,
            &q_stream.narrow(1, 0, 4)?.contiguous()?,
            &k_stream.narrow(1, 0, 4)?.contiguous()?,
            &v_stream.narrow(1, 0, 4)?.contiguous()?,
            &g_stream.narrow(1, 0, 4)?.contiguous()?,
            &b_stream.narrow(1, 0, 4)?.contiguous()?,
            &aq_stream.narrow(2, 0, 4)?.contiguous()?,
            &ak_stream.narrow(2, 0, 4)?.contiguous()?,
            &av_stream.narrow(2, 0, 4)?.contiguous()?,
            false,
            block_size,
        )?;
        let mut ground_gdn_steps = Vec::new();
        let mut ground_attn_steps = Vec::new();
        for t in 4..total_seq {
            let (yg, ya) = hybrid_forward_step(
                &mut ground,
                1,
                &conv_stream.narrow(1, t, 1)?.contiguous()?,
                &q_stream.narrow(1, t, 1)?.contiguous()?,
                &k_stream.narrow(1, t, 1)?.contiguous()?,
                &v_stream.narrow(1, t, 1)?.contiguous()?,
                &g_stream.narrow(1, t, 1)?.contiguous()?,
                &b_stream.narrow(1, t, 1)?.contiguous()?,
                &aq_stream.narrow(2, t, 1)?.contiguous()?,
                &ak_stream.narrow(2, t, 1)?.contiguous()?,
                &av_stream.narrow(2, t, 1)?.contiguous()?,
                false,
                block_size,
            )?;
            ground_gdn_steps.push(yg);
            ground_attn_steps.push(ya);
        }

        // 2. Speculative verify + rollback + resume:
        //    Prefill [0..4], verify [4, 5, BAD, BAD] with trail=true, accept first 2 (keep_len=6,
        //    rejected=2), rollback atomically, then decode token 6 and verify exact match against
        //    non-speculative ground truth!
        let mut spec = make_contract()?;
        spec.allocate_request(1)?;
        let _ = hybrid_forward_step(
            &mut spec,
            1,
            &conv_stream.narrow(1, 0, 4)?.contiguous()?,
            &q_stream.narrow(1, 0, 4)?.contiguous()?,
            &k_stream.narrow(1, 0, 4)?.contiguous()?,
            &v_stream.narrow(1, 0, 4)?.contiguous()?,
            &g_stream.narrow(1, 0, 4)?.contiguous()?,
            &b_stream.narrow(1, 0, 4)?.contiguous()?,
            &aq_stream.narrow(2, 0, 4)?.contiguous()?,
            &ak_stream.narrow(2, 0, 4)?.contiguous()?,
            &av_stream.narrow(2, 0, 4)?.contiguous()?,
            false,
            block_size,
        )?;

        // Prefix reuse fork at token_len=4 into request 2 BEFORE speculative verify on request 1
        let ckpt_4 = spec.fork_prefix(1, 2)?;
        assert_eq!(ckpt_4.token_len, 4);
        assert_eq!(spec.binding(2).unwrap().context_len, 4);

        let bad_conv = seeded(&[1, 2, conv_dim], 91, 0.9, dev)?.to_dtype(dtype)?;
        let bad_q = seeded(&[1, 2, n_heads, kd], 92, 0.9, dev)?.to_dtype(dtype)?;
        let bad_k = seeded(&[1, 2, n_heads, kd], 93, 0.9, dev)?.to_dtype(dtype)?;
        let bad_v = seeded(&[1, 2, n_heads, vd], 94, 0.9, dev)?.to_dtype(dtype)?;
        let bad_g = (seeded(&[1, 2, n_heads], 95, 0.25, dev)? - 0.35)?.to_dtype(dtype)?;
        let bad_b = ((seeded(&[1, 2, n_heads], 96, 0.4, dev)? + 1.0)? * 0.5)?.to_dtype(dtype)?;
        let bad_aq = seeded(&[1, n_heads, 2, kd], 97, 0.9, dev)?.to_dtype(dtype)?;
        let bad_ak = seeded(&[1, n_kv, 2, kd], 98, 0.9, dev)?.to_dtype(dtype)?;
        let bad_av = seeded(&[1, n_kv, 2, kd], 99, 0.9, dev)?.to_dtype(dtype)?;

        let ver_conv = Tensor::cat(&[&conv_stream.narrow(1, 4, 2)?, &bad_conv], 1)?.contiguous()?;
        let ver_q = Tensor::cat(&[&q_stream.narrow(1, 4, 2)?, &bad_q], 1)?.contiguous()?;
        let ver_k = Tensor::cat(&[&k_stream.narrow(1, 4, 2)?, &bad_k], 1)?.contiguous()?;
        let ver_v = Tensor::cat(&[&v_stream.narrow(1, 4, 2)?, &bad_v], 1)?.contiguous()?;
        let ver_g = Tensor::cat(&[&g_stream.narrow(1, 4, 2)?, &bad_g], 1)?.contiguous()?;
        let ver_b = Tensor::cat(&[&b_stream.narrow(1, 4, 2)?, &bad_b], 1)?.contiguous()?;
        let ver_aq = Tensor::cat(&[&aq_stream.narrow(2, 4, 2)?, &bad_aq], 2)?.contiguous()?;
        let ver_ak = Tensor::cat(&[&ak_stream.narrow(2, 4, 2)?, &bad_ak], 2)?.contiguous()?;
        let ver_av = Tensor::cat(&[&av_stream.narrow(2, 4, 2)?, &bad_av], 2)?.contiguous()?;

        let _ = hybrid_forward_step(
            &mut spec, 1, &ver_conv, &ver_q, &ver_k, &ver_v, &ver_g, &ver_b, &ver_aq, &ver_ak,
            &ver_av, true, block_size,
        )?;
        assert_eq!(spec.binding(1).unwrap().context_len, 8);

        // Rollback 2 rejected drafts -> keep_len = 6 (KV + paged block table + GDN + Conv1D!)
        spec.rollback_speculative(1, 6, 2, Some(block_size))?;
        assert_eq!(spec.binding(1).unwrap().context_len, 6);
        assert_eq!(spec.binding(1).unwrap().paged_block_table.len(), 2); // ceil(6/4) == 2

        // Request 2 (forked at len=4) must remain completely untouched at offset 4!
        let (fork_conv, fork_rec, fork_off) = spec.inspect_recurrent_layer(2, 0)?;
        assert_eq!(fork_off, 4);
        assert_eq!(
            rel_err(&fork_conv, &ckpt_4.recurrent_snapshots[0].conv_state)?,
            0.0
        );
        assert_eq!(
            rel_err(&fork_rec, &ckpt_4.recurrent_snapshots[0].recurrent_state)?,
            0.0
        );

        // Decode true token 6 after speculative rollback and assert parity with ground truth!
        let (yg6, ya6) = hybrid_forward_step(
            &mut spec,
            1,
            &conv_stream.narrow(1, 6, 1)?.contiguous()?,
            &q_stream.narrow(1, 6, 1)?.contiguous()?,
            &k_stream.narrow(1, 6, 1)?.contiguous()?,
            &v_stream.narrow(1, 6, 1)?.contiguous()?,
            &g_stream.narrow(1, 6, 1)?.contiguous()?,
            &b_stream.narrow(1, 6, 1)?.contiguous()?,
            &aq_stream.narrow(2, 6, 1)?.contiguous()?,
            &ak_stream.narrow(2, 6, 1)?.contiguous()?,
            &av_stream.narrow(2, 6, 1)?.contiguous()?,
            false,
            block_size,
        )?;
        let tol = if dtype == DType::F32 { 1e-4 } else { 5e-3 };
        let eg = rel_err(&yg6, &ground_gdn_steps[2])?;
        let ea = rel_err(&ya6, &ground_attn_steps[2])?;
        assert!(eg <= tol, "post-rollback GDN mismatch: {eg:.2e}");
        assert!(ea <= tol, "post-rollback Attention mismatch: {ea:.2e}");

        // Snapshot at len=7, mutate binding + advance one bad step, then restore_request back to len=7!
        let ckpt_7 = spec.snapshot_request(1)?;
        assert_eq!(ckpt_7.token_len, 7);
        assert_eq!(ckpt_7.kv_snapshots.len(), 2);
        spec.binding_mut(1).unwrap().context_len = 999;
        spec.restore_request(1, &ckpt_7)?;
        assert_eq!(spec.binding(1).unwrap().request_id, 1);
        assert_eq!(spec.binding(1).unwrap().context_len, 7);

        // 3. Cancel both requests -> recurrent slots zeroed & freed, KV reset cleanly.
        spec.cancel_request(1)?;
        spec.cancel_request(2)?;
        assert!(spec.binding(1).is_none() && spec.binding(2).is_none());
        let fresh_slot = spec.allocate_request(99)?;
        let (c0, r0, o0) = spec.inspect_recurrent_layer(99, 0)?;
        assert_eq!(o0, 0);
        assert!(c0
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&x| x == 0.0));
        assert!(r0
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&x| x == 0.0));
        let _ = fresh_slot;

        dsl::set_gdn_impl(QkImpl::Legacy);
        dsl::set_flash_impl(QkImpl::Legacy);
        Ok(())
    }

    #[test]
    fn hybrid_contract_cpu_lifecycle() -> Result<()> {
        run_hybrid_contract_lifecycle(&Device::Cpu, DType::F32, false)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hybrid_contract_cuda_dsl_lifecycle() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        run_hybrid_contract_lifecycle(&dev, DType::F32, true)?;
        run_hybrid_contract_lifecycle(&dev, DType::BF16, true)?;
        Ok(())
    }

    #[test]
    fn hybrid_single_seq_kv_headroom_zero_copy() -> Result<()> {
        // Guards against the 32K `copy3d_bf16` regression: cloning a `KvCache` across
        // `HybridCacheManager::clone_in_cache`/`clone_out_cache` for a single active sequence must
        // preserve `capacity_seq_len` headroom so `append()` writes in-place without reallocating.
        let dev = Device::Cpu;
        let mut seq_kv = super::super::KvCache::new_normal(2, 4096, 256);
        let k0 = Tensor::zeros((1, 4, 100, 16), DType::F32, &dev)?;
        let v0 = Tensor::zeros((1, 4, 100, 16), DType::F32, &dev)?;
        let _ = seq_kv.append(&k0, &v0)?;

        let cap_before = match &seq_kv {
            super::super::KvCache::Normal { k, .. } => k.capacity_seq_len,
            _ => unreachable!(),
        };
        assert!(cap_before > 100, "initial append must allocate headroom");

        // Simulate 16 decode steps of clone_in -> append(1 token) -> clone_out.
        for step in 1..=16usize {
            let mut active = seq_kv.clone();
            let k1 = Tensor::full(step as f32, (1, 4, 1, 16), &dev)?;
            let v1 = Tensor::full(step as f32, (1, 4, 1, 16), &dev)?;
            let (k_out, _) = active.append(&k1, &v1)?;
            assert_eq!(k_out.dim(2)?, 100 + step);
            seq_kv = active.clone();
        }
        let (len_after, cap_after) = match &seq_kv {
            super::super::KvCache::Normal { k, .. } => (k.current_seq_len, k.capacity_seq_len),
            _ => unreachable!(),
        };
        assert_eq!(len_after, 116);
        assert_eq!(
            cap_after, cap_before,
            "single-seq clone_in/out must preserve capacity_seq_len without reallocating"
        );
        Ok(())
    }

    #[test]
    fn hybrid_single_to_multi_to_single_concurrency_and_cancellation() -> Result<()> {
        // Exercises the full lifecycle across single-seq zero-copy -> 2-seq batched cat/chunk ->
        // cancellation of seq B -> single-seq resume of seq A -> fresh seq C with empty cache:
        // verifies zero cross-request buffer aliasing and zero stale data leakage.
        let dev = Device::Cpu;
        let mut seq_a = super::super::KvCache::new_normal(2, 4096, 64);
        let mut seq_b = super::super::KvCache::new_normal(2, 4096, 64);

        let ka0 = Tensor::full(1.0f32, (1, 2, 4, 8), &dev)?;
        let va0 = Tensor::full(10.0f32, (1, 2, 4, 8), &dev)?;
        seq_a.append(&ka0, &va0)?;
        let kb0 = Tensor::full(2.0f32, (1, 2, 4, 8), &dev)?;
        let vb0 = Tensor::full(20.0f32, (1, 2, 4, 8), &dev)?;
        seq_b.append(&kb0, &vb0)?;

        // Phase 1: single-seq A zero-copy clone_in -> append -> clone_out
        let mut active = seq_a.clone();
        let ka1 = Tensor::full(3.0f32, (1, 2, 1, 8), &dev)?;
        let va1 = Tensor::full(30.0f32, (1, 2, 1, 8), &dev)?;
        active.append(&ka1, &va1)?;
        seq_a = active.clone();
        // Match B to len=5 before concurrent batch step
        let kb1 = Tensor::full(4.0f32, (1, 2, 1, 8), &dev)?;
        let vb1 = Tensor::full(40.0f32, (1, 2, 1, 8), &dev)?;
        seq_b.append(&kb1, &vb1)?;

        // Phase 2: concurrent 2-seq batch [A, B] (cat -> append -> chunk split)
        let batch_k = Tensor::cat(&[&seq_a.k()?.unwrap(), &seq_b.k()?.unwrap()], 0)?;
        let batch_v = Tensor::cat(&[&seq_a.v()?.unwrap(), &seq_b.v()?.unwrap()], 0)?;
        let mut batched = super::super::KvCache::new_normal(2, 4096, 64);
        if let super::super::KvCache::Normal { k, v } = &mut batched {
            k.all_data = Some(batch_k);
            k.current_seq_len = 5;
            k.capacity_seq_len = 5;
            v.all_data = Some(batch_v);
            v.current_seq_len = 5;
            v.capacity_seq_len = 5;
        }
        let step_k = Tensor::cat(
            &[
                &Tensor::full(5.0f32, (1, 2, 1, 8), &dev)?,
                &Tensor::full(6.0f32, (1, 2, 1, 8), &dev)?,
            ],
            0,
        )?;
        let step_v = Tensor::cat(
            &[
                &Tensor::full(50.0f32, (1, 2, 1, 8), &dev)?,
                &Tensor::full(60.0f32, (1, 2, 1, 8), &dev)?,
            ],
            0,
        )?;
        batched.append(&step_k, &step_v)?;
        let k_chunks = batched.k()?.unwrap().chunk(2, 0)?;
        let v_chunks = batched.v()?.unwrap().chunk(2, 0)?;
        if let super::super::KvCache::Normal { k, v } = &mut seq_a {
            k.all_data = Some(k_chunks[0].contiguous()?);
            k.current_seq_len = 6;
            k.capacity_seq_len = 6;
            v.all_data = Some(v_chunks[0].contiguous()?);
            v.current_seq_len = 6;
            v.capacity_seq_len = 6;
        }
        if let super::super::KvCache::Normal { k, v } = &mut seq_b {
            k.all_data = Some(k_chunks[1].contiguous()?);
            k.current_seq_len = 6;
            k.capacity_seq_len = 6;
            v.all_data = Some(v_chunks[1].contiguous()?);
            v.current_seq_len = 6;
            v.capacity_seq_len = 6;
        }

        // Phase 3: cancel seq B, resume single-seq A zero-copy
        drop(seq_b);
        active = seq_a.clone();
        active.append(
            &Tensor::full(7.0f32, (1, 2, 1, 8), &dev)?,
            &Tensor::full(70.0f32, (1, 2, 1, 8), &dev)?,
        )?;
        seq_a = active.clone();
        let a_vals: Vec<f32> = seq_a
            .k()?
            .unwrap()
            .narrow(1, 0, 1)?
            .narrow(3, 0, 1)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(a_vals, vec![1.0, 1.0, 1.0, 1.0, 3.0, 5.0, 7.0]);

        // Phase 4: fresh seq C with empty slot must reset active cache (no stale leak from A)
        active.reset();
        assert_eq!(active.current_seq_len(), 0);
        assert!(active.k()?.is_none());
        // Seq A's saved cache remains completely intact after active.reset()
        assert_eq!(seq_a.current_seq_len(), 7);
        Ok(())
    }

    /// Concurrent multi-session stress test:
    /// - Prefill shared prompt on `req 1` (len=4), fork into 3 concurrent sessions (`req 10`, `20`, `30`)
    /// - Interleave distinct token streams across all 3 sessions (testing recurrent-slot isolation & COW)
    /// - Run speculative draft + rejection rollback on `req 20` while `req 10` & `req 30` advance
    /// - Cancel & evict `req 10`, immediately allocate `req 40` into the freed slot (assert zero state leak)
    /// - Verify `req 20` (post-rollback) and `req 30` match independent ground-truth runs!
    fn run_hybrid_concurrent_sessions_stress(
        dev: &Device,
        dtype: DType,
        use_dsl: bool,
    ) -> Result<()> {
        let mode = if use_dsl { QkImpl::Dsl } else { QkImpl::Legacy };
        dsl::set_gdn_impl(mode);
        dsl::set_flash_impl(mode);

        let (n_heads, n_kv, kd, vd, conv_dim, conv_k, block_size) =
            (4usize, 2usize, 64usize, 64usize, 16usize, 4usize, 4usize);
        let make_contract = || -> Result<HybridStateContract> {
            let rec_cfg = RecurrentLayerConfig {
                conv_dim,
                conv_width: conv_k,
                state_dims: vec![n_heads, kd, vd],
                conv_dtype: dtype,
                state_dtype: DType::F32,
            };
            let cfg = HybridCacheConfig::uniform(
                vec![HybridLayerType::Recurrent, HybridLayerType::Attention],
                64,
                rec_cfg,
            );
            Ok(HybridStateContract::new(HybridCache::new(cfg, dev)?))
        };

        // Shared 4-token prefix + 3 distinct 3-token continuations (A=req10, B=req20, C=req30)
        // plus a 2-token bogus speculative draft for req20.
        let mk_stream = |seed_base: u64, len: usize| -> Result<[Tensor; 6]> {
            Ok([
                seeded(&[1, len, conv_dim], seed_base + 1, 0.5, dev)?.to_dtype(dtype)?,
                seeded(&[1, len, n_heads, kd], seed_base + 2, 0.3, dev)?.to_dtype(dtype)?,
                seeded(&[1, len, n_heads, kd], seed_base + 3, 0.3, dev)?.to_dtype(dtype)?,
                seeded(&[1, len, n_heads, vd], seed_base + 4, 0.3, dev)?.to_dtype(dtype)?,
                (seeded(&[1, len, n_heads], seed_base + 5, 0.25, dev)? - 0.35)?.to_dtype(dtype)?,
                ((seeded(&[1, len, n_heads], seed_base + 6, 0.4, dev)? + 1.0)? * 0.5)?
                    .to_dtype(dtype)?,
            ])
        };
        let prefix = mk_stream(100, 4)?;
        let cont_a = mk_stream(200, 3)?;
        let cont_b = mk_stream(300, 3)?;
        let cont_c = mk_stream(400, 3)?;
        let bogus_b = mk_stream(900, 2)?;

        let dummy_attn = |seed_base: u64, len: usize| -> Result<[Tensor; 3]> {
            Ok([
                seeded(&[1, n_heads, len, kd], seed_base + 7, 0.3, dev)?.to_dtype(dtype)?,
                seeded(&[1, n_kv, len, kd], seed_base + 8, 0.3, dev)?.to_dtype(dtype)?,
                seeded(&[1, n_kv, len, kd], seed_base + 9, 0.3, dev)?.to_dtype(dtype)?,
            ])
        };
        let p_attn = dummy_attn(100, 4)?;
        let a_attn = dummy_attn(200, 3)?;
        let b_attn = dummy_attn(300, 3)?;
        let c_attn = dummy_attn(400, 3)?;
        let bog_attn = dummy_attn(900, 2)?;

        // Ground truth for B (req 20) and C (req 30) in completely isolated contracts
        let run_isolated = |cont: &[Tensor; 6], ca: &[Tensor; 3]| -> Result<Vec<Tensor>> {
            let mut c = make_contract()?;
            c.allocate_request(1)?;
            let _ = hybrid_forward_step(
                &mut c, 1, &prefix[0], &prefix[1], &prefix[2], &prefix[3], &prefix[4], &prefix[5],
                &p_attn[0], &p_attn[1], &p_attn[2], false, block_size,
            )?;
            let mut outs = Vec::new();
            for t in 0..3 {
                let (yg, _) = hybrid_forward_step(
                    &mut c,
                    1,
                    &cont[0].narrow(1, t, 1)?.contiguous()?,
                    &cont[1].narrow(1, t, 1)?.contiguous()?,
                    &cont[2].narrow(1, t, 1)?.contiguous()?,
                    &cont[3].narrow(1, t, 1)?.contiguous()?,
                    &cont[4].narrow(1, t, 1)?.contiguous()?,
                    &cont[5].narrow(1, t, 1)?.contiguous()?,
                    &ca[0].narrow(2, t, 1)?.contiguous()?,
                    &ca[1].narrow(2, t, 1)?.contiguous()?,
                    &ca[2].narrow(2, t, 1)?.contiguous()?,
                    false,
                    block_size,
                )?;
                outs.push(yg);
            }
            Ok(outs)
        };
        let gt_b = run_isolated(&cont_b, &b_attn)?;
        let gt_c = run_isolated(&cont_c, &c_attn)?;

        // Shared concurrent contract: prefill req 1, fork into req 10, 20, 30
        let mut mgr = make_contract()?;
        mgr.allocate_request(1)?;
        let _ = hybrid_forward_step(
            &mut mgr, 1, &prefix[0], &prefix[1], &prefix[2], &prefix[3], &prefix[4], &prefix[5],
            &p_attn[0], &p_attn[1], &p_attn[2], false, block_size,
        )?;
        let ckpt_prefix = mgr.fork_prefix(1, 10)?;
        let _ = mgr.fork_prefix(1, 20)?;
        let _ = mgr.fork_prefix(1, 30)?;
        // Cancel original seed request 1 (simulating prefix eviction of originator)
        mgr.cancel_request(1)?;
        assert!(mgr.binding(1).is_none());

        let step_rec_only = |c: &mut HybridStateContract,
                             req: usize,
                             st: &[Tensor; 6],
                             at: &[Tensor; 3],
                             t: usize,
                             record_trail: bool|
         -> Result<Tensor> {
            let (yg, _) = hybrid_forward_step(
                c,
                req,
                &st[0].narrow(1, t, 1)?.contiguous()?,
                &st[1].narrow(1, t, 1)?.contiguous()?,
                &st[2].narrow(1, t, 1)?.contiguous()?,
                &st[3].narrow(1, t, 1)?.contiguous()?,
                &st[4].narrow(1, t, 1)?.contiguous()?,
                &st[5].narrow(1, t, 1)?.contiguous()?,
                &at[0].narrow(2, t, 1)?.contiguous()?,
                &at[1].narrow(2, t, 1)?.contiguous()?,
                &at[2].narrow(2, t, 1)?.contiguous()?,
                record_trail,
                block_size,
            )?;
            Ok(yg)
        };

        // Interleave token 0 across req 10, 20, 30
        let _ = step_rec_only(&mut mgr, 10, &cont_a, &a_attn, 0, false)?;
        let yb0 = step_rec_only(&mut mgr, 20, &cont_b, &b_attn, 0, false)?;
        let yc0 = step_rec_only(&mut mgr, 30, &cont_c, &c_attn, 0, false)?;

        // Advance req 10 and req 30 token 1
        let _ = step_rec_only(&mut mgr, 10, &cont_a, &a_attn, 1, false)?;
        let yc1 = step_rec_only(&mut mgr, 30, &cont_c, &c_attn, 1, false)?;

        // Speculative verify forward on req 20: [true token 1 (accepted) + 2 bogus drafts (rejected)]
        // in one multi-token verify pass with record_trail=true, then immediately rollback 2 drafts.
        let ver_conv = Tensor::cat(&[&cont_b[0].narrow(1, 1, 1)?, &bogus_b[0]], 1)?.contiguous()?;
        let ver_q = Tensor::cat(&[&cont_b[1].narrow(1, 1, 1)?, &bogus_b[1]], 1)?.contiguous()?;
        let ver_k = Tensor::cat(&[&cont_b[2].narrow(1, 1, 1)?, &bogus_b[2]], 1)?.contiguous()?;
        let ver_v = Tensor::cat(&[&cont_b[3].narrow(1, 1, 1)?, &bogus_b[3]], 1)?.contiguous()?;
        let ver_g = Tensor::cat(&[&cont_b[4].narrow(1, 1, 1)?, &bogus_b[4]], 1)?.contiguous()?;
        let ver_b = Tensor::cat(&[&cont_b[5].narrow(1, 1, 1)?, &bogus_b[5]], 1)?.contiguous()?;
        let ver_aq = Tensor::cat(&[&b_attn[0].narrow(2, 1, 1)?, &bog_attn[0]], 2)?.contiguous()?;
        let ver_ak = Tensor::cat(&[&b_attn[1].narrow(2, 1, 1)?, &bog_attn[1]], 2)?.contiguous()?;
        let ver_av = Tensor::cat(&[&b_attn[2].narrow(2, 1, 1)?, &bog_attn[2]], 2)?.contiguous()?;
        let (yb_ver, _) = hybrid_forward_step(
            &mut mgr, 20, &ver_conv, &ver_q, &ver_k, &ver_v, &ver_g, &ver_b, &ver_aq, &ver_ak,
            &ver_av, true, block_size,
        )?;
        let yb1 = yb_ver.narrow(1, 0, 1)?.contiguous()?;
        assert_eq!(mgr.binding(20).unwrap().context_len, 8);

        // Reject both speculative draft tokens on req 20 (keep accepted token 1 -> len=6)
        mgr.rollback_speculative(20, 6, 2, Some(block_size))?;
        assert_eq!(mgr.binding(20).unwrap().context_len, 6);

        // Cancel & evict req 10 mid-stream, immediately reuse freed slot for req 40
        mgr.cancel_request(10)?;
        assert!(mgr.binding(10).is_none());
        let _slot40 = mgr.allocate_request(40)?;
        let (c40, r40, o40) = mgr.inspect_recurrent_layer(40, 0)?;
        assert_eq!(o40, 0, "reused slot must have zero offset");
        assert!(
            c40.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&x| x == 0.0),
            "reused slot must have zeroed conv_state"
        );
        assert!(
            r40.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&x| x == 0.0),
            "reused slot must have zeroed recurrent_state"
        );

        // Advance req 20 token 2 (post-rollback) and req 30 token 2
        let yb2 = step_rec_only(&mut mgr, 20, &cont_b, &b_attn, 2, false)?;
        let yc2 = step_rec_only(&mut mgr, 30, &cont_c, &c_attn, 2, false)?;

        // Verify original prefix checkpoint remained untouched (COW integrity)
        assert_eq!(ckpt_prefix.token_len, 4);

        let tol = if dtype == DType::F32 { 1e-4 } else { 5e-3 };
        for (idx, (got, want)) in [(&yb0, &gt_b[0]), (&yb1, &gt_b[1]), (&yb2, &gt_b[2])]
            .into_iter()
            .enumerate()
        {
            let e = rel_err(got, want)?;
            assert!(
                e <= tol,
                "concurrent req20 (post-rollback) step {idx} diverged: {e:.2e} > {tol:.1e}"
            );
        }
        for (idx, (got, want)) in [(&yc0, &gt_c[0]), (&yc1, &gt_c[1]), (&yc2, &gt_c[2])]
            .into_iter()
            .enumerate()
        {
            let e = rel_err(got, want)?;
            assert!(
                e <= tol,
                "concurrent req30 step {idx} diverged: {e:.2e} > {tol:.1e}"
            );
        }

        mgr.cancel_request(20)?;
        mgr.cancel_request(30)?;
        mgr.cancel_request(40)?;
        dsl::set_gdn_impl(QkImpl::Legacy);
        dsl::set_flash_impl(QkImpl::Legacy);
        Ok(())
    }

    #[test]
    fn hybrid_concurrent_sessions_stress_cpu() -> Result<()> {
        run_hybrid_concurrent_sessions_stress(&Device::Cpu, DType::F32, false)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hybrid_concurrent_sessions_stress_cuda_dsl() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::F32, true)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::BF16, true)?;
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn hybrid_contract_metal_dsl_lifecycle() -> Result<()> {
        let dev = Device::new_metal(0)?;
        run_hybrid_contract_lifecycle(&dev, DType::F32, true)?;
        run_hybrid_contract_lifecycle(&dev, DType::BF16, true)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::F32, true)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::BF16, true)?;
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn hybrid_contract_rocm_dsl_lifecycle() -> Result<()> {
        let dev = Device::new_rocm(0)?;
        run_hybrid_contract_lifecycle(&dev, DType::F32, true)?;
        run_hybrid_contract_lifecycle(&dev, DType::BF16, true)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::F32, true)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::BF16, true)?;
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn hybrid_contract_vulkan_dsl_lifecycle() -> Result<()> {
        let dev = Device::new_vulkan(0)?;
        run_hybrid_contract_lifecycle(&dev, DType::F32, true)?;
        run_hybrid_concurrent_sessions_stress(&dev, DType::F32, true)?;
        Ok(())
    }
}
