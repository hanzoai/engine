//! Hybrid cache for models that mix attention and recurrent layers (e.g., GraniteMoeHybrid, Qwen3 Next)
//!
//! This implements vLLM-style continuous batching for hybrid models:
//! - Attention layers use standard KV cache batching
//! - Recurrent layers (Mamba SSM or GDN) use a pool-based state with indexed access
//!
//! The key insight is that recurrent state is accessed via `state_indices` which map
//! each sequence in the current batch to its slot in the pool.

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor};

use super::KvCache;
use crate::layers_masker::PastKvLenCache;

/// The state a pool held after each position of one forward. A speculative verify runs the anchor
/// and every draft through the recurrent layers in that forward; when only some drafts are
/// accepted, the state after the last accepted one is the entry to restore.
#[derive(Debug, Clone)]
pub struct RecurrentTrail {
    /// Pool slots in batch order.
    pub slots: Vec<u32>,
    /// Offset of each slot before the forward, in batch order.
    pub start_offsets: Vec<usize>,
    /// `conv[t]`, `recurrent[t]`: batch-major state after position `t`. `recurrent` is empty
    /// for a conv-only pool, whose forward runs no recurrence.
    pub conv: Vec<Tensor>,
    pub recurrent: Vec<Tensor>,
}

/// Pool-based recurrent state cache for continuous batching.
///
/// Works for both Mamba SSM and GDN (Gated Delta Net) recurrent layers.
/// Instead of dynamically sized state tensors, we maintain a pool of
/// state slots that grows dynamically. Each sequence is assigned a slot index,
/// and the forward pass uses `index_select` (gather) and index assignment (scatter)
/// to access the correct states.
#[derive(Clone, Debug)]
pub struct RecurrentStatePool {
    /// Convolution state pool: (capacity, conv_dim, conv_width), in `conv_dtype`
    pub conv_state: Tensor,
    /// Recurrent state pool: (capacity, ...state_dims), in `state_dtype`
    /// For Mamba: (capacity, n_heads, head_dim, d_state)
    /// For GDN: (capacity, n_v_heads, key_dim, value_dim)
    pub recurrent_state: Tensor,
    /// Per-slot sequence length offsets (for tracking generation position)
    seqlen_offsets: Vec<usize>,
    /// Present only while it describes the forward that produced the current state.
    trail: Option<RecurrentTrail>,
    /// Stack of free slot indices (for allocation)
    free_slots: Vec<usize>,
    /// Current capacity (grows dynamically)
    capacity: usize,
    /// Shapes and dtypes for growing and resetting
    config: RecurrentLayerConfig,
    device: Device,
}

/// Initial pool capacity before dynamic growth.
const INITIAL_POOL_CAPACITY: usize = 4;

impl RecurrentStatePool {
    /// Create a new recurrent state pool of `config`'s shapes and dtypes.
    pub fn new(config: RecurrentLayerConfig, device: &Device) -> Result<Self> {
        let capacity = INITIAL_POOL_CAPACITY;
        let (conv_state, recurrent_state) = config.zeros(capacity, device)?;

        let free_slots: Vec<usize> = (0..capacity).rev().collect();
        let seqlen_offsets = vec![0; capacity];

        Ok(Self {
            conv_state,
            recurrent_state,
            seqlen_offsets,
            trail: None,
            free_slots,
            capacity,
            config,
            device: device.clone(),
        })
    }

    /// Grow the pool by doubling capacity.
    fn grow(&mut self) -> Result<()> {
        let new_capacity = self.capacity * 2;

        // Allocate larger pools and copy existing data
        let (new_conv, new_recurrent) = self.config.zeros(new_capacity, &self.device)?;
        new_conv.slice_set(&self.conv_state, 0, 0)?;
        new_recurrent.slice_set(&self.recurrent_state, 0, 0)?;

        // Add new slots to free list
        self.free_slots.extend((self.capacity..new_capacity).rev());
        self.seqlen_offsets.resize(new_capacity, 0);

        self.conv_state = new_conv;
        self.recurrent_state = new_recurrent;
        self.capacity = new_capacity;

        tracing::info!("Recurrent state pool grew to capacity {new_capacity}");
        Ok(())
    }

    /// Allocate a state slot for a new sequence. Returns the slot index.
    /// The pool grows dynamically if no free slots are available.
    /// The slot's state is reset to zeros to prevent state bleeding.
    pub fn allocate(&mut self) -> Option<usize> {
        if self.free_slots.is_empty() {
            if let Err(e) = self.grow() {
                tracing::error!("Failed to grow recurrent state pool: {e}");
                return None;
            }
        }
        let slot_idx = self.free_slots.pop()?;
        if self.reset_slot(slot_idx).is_err() {
            tracing::warn!("Failed to reset recurrent state slot {slot_idx}, state may be stale");
        }
        Some(slot_idx)
    }

    /// Free a state slot when a sequence completes.
    pub fn free(&mut self, slot_idx: usize) {
        debug_assert!(slot_idx < self.capacity);
        self.trail = None;
        self.seqlen_offsets[slot_idx] = 0;
        self.free_slots.push(slot_idx);
    }

    /// Get the seqlen offset for a slot
    pub fn get_seqlen_offset(&self, slot_idx: usize) -> usize {
        self.seqlen_offsets[slot_idx]
    }

    /// Set the seqlen offset for a slot
    pub fn set_seqlen_offset(&mut self, slot_idx: usize, offset: usize) {
        self.seqlen_offsets[slot_idx] = offset;
    }

    /// Increment seqlen offset for a slot
    pub fn increment_seqlen_offset(&mut self, slot_idx: usize, delta: usize) {
        self.seqlen_offsets[slot_idx] += delta;
    }

    /// Gather conv states for the given slot indices
    pub fn gather_conv_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.conv_state.index_select(state_indices, 0)
    }

    /// Gather recurrent states for the given slot indices
    pub fn gather_recurrent_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.recurrent_state.index_select(state_indices, 0)
    }

    /// Scatter conv states back to the pool for the given slot indices
    pub fn scatter_conv_state(&mut self, state_indices: &Tensor, values: &Tensor) -> Result<()> {
        let indices: Vec<u32> = state_indices.to_vec1()?;
        let dt = self.conv_state.dtype();
        for (batch_idx, &slot_idx) in indices.iter().enumerate() {
            let value = values
                .i(batch_idx)?
                .unsqueeze(0)?
                .contiguous()?
                .to_dtype(dt)?;
            self.conv_state.slice_set(&value, 0, slot_idx as usize)?;
        }
        Ok(())
    }

    /// Scatter recurrent states back to the pool for the given slot indices
    pub fn scatter_recurrent_state(
        &mut self,
        state_indices: &Tensor,
        values: &Tensor,
    ) -> Result<()> {
        let indices: Vec<u32> = state_indices.to_vec1()?;
        let dt = self.recurrent_state.dtype();
        for (batch_idx, &slot_idx) in indices.iter().enumerate() {
            let value = values
                .i(batch_idx)?
                .unsqueeze(0)?
                .contiguous()?
                .to_dtype(dt)?;
            self.recurrent_state
                .slice_set(&value, 0, slot_idx as usize)?;
        }
        Ok(())
    }

    /// Replace the trail. Every forward through the pool calls this, with `None` when it kept no
    /// trail, so a trail never outlives the state it describes.
    pub fn set_trail(&mut self, trail: Option<RecurrentTrail>) {
        self.trail = trail;
    }

    /// Undo the last `rejected` positions of the forward that produced `slot_idx`'s state.
    /// Fails, leaving the pool untouched, unless the trail covers exactly that forward. The conv
    /// trail sets the length, since every forward through a pool advances its conv state; a
    /// conv-only pool runs no recurrence and so trails none.
    pub fn rewind(&mut self, slot_idx: usize, rejected: usize) -> Result<()> {
        if rejected == 0 {
            return Ok(());
        }
        let Some(trail) = self.trail.as_ref() else {
            hanzo_ml::bail!(
                "recurrent rewind of {rejected} for slot {slot_idx}: the last forward kept no trail"
            );
        };
        let len = trail.conv.len();
        let Some(row) = trail.slots.iter().position(|&s| s as usize == slot_idx) else {
            hanzo_ml::bail!("recurrent rewind: slot {slot_idx} was not in the last forward");
        };
        let moved = if self.config.state_dims.is_empty() {
            0
        } else {
            len
        };
        if trail.recurrent.len() != moved || trail.start_offsets.len() != trail.slots.len() {
            hanzo_ml::bail!(
                "recurrent rewind: the trail has {len} conv and {} recurrent positions",
                trail.recurrent.len()
            );
        }
        if rejected >= len {
            hanzo_ml::bail!("recurrent rewind of {rejected} exceeds a trail of {len} positions");
        }
        let start = trail.start_offsets[row];
        if self.seqlen_offsets[slot_idx] != start + len {
            hanzo_ml::bail!(
                "recurrent rewind: slot {slot_idx} is at {}, the trail ends at {}",
                self.seqlen_offsets[slot_idx],
                start + len
            );
        }
        let keep = len - rejected;
        let conv = trail.conv[keep - 1]
            .narrow(0, row, 1)?
            .to_dtype(self.conv_state.dtype())?
            .contiguous()?;
        // A conv-only pool's recurrent state never moved, so it already is the kept one.
        let recurrent = trail
            .recurrent
            .get(keep - 1)
            .map(|state| {
                state
                    .narrow(0, row, 1)?
                    .to_dtype(self.recurrent_state.dtype())?
                    .contiguous()
            })
            .transpose()?;
        let offset = start + keep;
        self.conv_state.slice_set(&conv, 0, slot_idx)?;
        if let Some(recurrent) = recurrent {
            self.recurrent_state.slice_set(&recurrent, 0, slot_idx)?;
        }
        self.seqlen_offsets[slot_idx] = offset;
        Ok(())
    }

    /// Reset a specific slot's state to zeros
    pub fn reset_slot(&mut self, slot_idx: usize) -> Result<()> {
        let (zero_conv, zero_recurrent) = self.config.zeros(1, &self.device)?;
        self.conv_state.slice_set(&zero_conv, 0, slot_idx)?;
        self.recurrent_state
            .slice_set(&zero_recurrent, 0, slot_idx)?;
        self.seqlen_offsets[slot_idx] = 0;
        self.trail = None;
        Ok(())
    }

    /// Reset all slots
    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        self.seqlen_offsets.fill(0);
        self.trail = None;
        self.free_slots = (0..self.capacity).rev().collect();
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn num_free_slots(&self) -> usize {
        self.free_slots.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Per-layer cache that can be either attention (KV) or recurrent (state pool)
#[derive(Clone, Debug)]
pub enum HybridLayerCache {
    Attention(KvCache),
    Recurrent(RecurrentStatePool),
}

impl HybridLayerCache {
    pub fn reset(&mut self) {
        match self {
            Self::Attention(kv) => kv.reset(),
            Self::Recurrent(pool) => {
                let _ = pool.reset();
            }
        }
    }

    pub fn as_kv_cache(&self) -> Option<&KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Recurrent(_) => None,
        }
    }

    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Recurrent(_) => None,
        }
    }

    pub fn as_recurrent_pool(&self) -> Option<&RecurrentStatePool> {
        match self {
            Self::Attention(_) => None,
            Self::Recurrent(pool) => Some(pool),
        }
    }

    pub fn as_recurrent_pool_mut(&mut self) -> Option<&mut RecurrentStatePool> {
        match self {
            Self::Attention(_) => None,
            Self::Recurrent(pool) => Some(pool),
        }
    }
}

/// Layer type indicator for hybrid models
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridLayerType {
    Attention,
    Recurrent,
}

/// Configuration of one recurrent state pool
#[derive(Clone, Debug)]
pub struct RecurrentLayerConfig {
    /// Dimension of the convolution state
    pub conv_dim: usize,
    /// Kernel size for causal conv1d
    pub conv_width: usize,
    /// Shape of the recurrent state per slot.
    /// For Mamba: [n_heads, head_dim, d_state]
    /// For GDN: [n_v_heads, key_dim, value_dim]
    /// Empty for a conv-only pool, whose one scalar per slot no forward moves.
    pub state_dims: Vec<usize>,
    pub conv_dtype: DType,
    pub state_dtype: DType,
}

impl RecurrentLayerConfig {
    /// Zeroed conv and recurrent state for `slots` slots.
    fn zeros(&self, slots: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let conv = Tensor::zeros(
            (slots, self.conv_dim, self.conv_width),
            self.conv_dtype,
            device,
        )?;
        let mut shape = vec![slots];
        shape.extend_from_slice(&self.state_dims);
        let recurrent = Tensor::zeros(shape, self.state_dtype, device)?;
        Ok((conv, recurrent))
    }
}

/// Configuration for creating a hybrid cache
#[derive(Clone, Debug)]
pub struct HybridCacheConfig {
    pub layer_types: Vec<HybridLayerType>,
    pub max_seq_len: usize,
    /// One per recurrent pool, in cache order: one for each `Recurrent` layer, then side pools,
    /// which sit after the layers, from index `layer_types.len()` on.
    pub pools: Vec<RecurrentLayerConfig>,
}

impl HybridCacheConfig {
    /// `layer_types` with `pool` for every recurrent layer and no side pools.
    pub fn uniform(
        layer_types: Vec<HybridLayerType>,
        max_seq_len: usize,
        pool: RecurrentLayerConfig,
    ) -> Self {
        let recurrent = layer_types
            .iter()
            .filter(|&&t| t == HybridLayerType::Recurrent)
            .count();
        Self {
            pools: vec![pool; recurrent],
            layer_types,
            max_seq_len,
        }
    }
}

/// Hybrid cache that stores per-layer caches for mixed attention/recurrent models
///
/// For continuous batching:
/// - Attention layers use standard KV cache with batching support
/// - Recurrent layers use RecurrentStatePool with indexed access via state_indices
#[derive(Clone, Debug)]
pub struct HybridCache {
    pub caches: Vec<HybridLayerCache>,
    config: HybridCacheConfig,
    /// Current batch's state indices for recurrent pool access.
    /// Set by clone_in_cache before forward, used by model during forward.
    /// Shape: (batch_size,) containing pool slot indices.
    state_indices: Option<Tensor>,
    /// Host mirror of `state_indices`. The decode forward reads slots from here to
    /// gather/scatter recurrent state via constant-offset `narrow`/`slice_set` with NO
    /// device->host sync, so the GDN decode path stays capturable by a CUDA/HIP graph
    /// (a `state_indices.to_vec1()` would abort stream capture).
    state_indices_host: Option<Vec<u32>>,
    /// Tokens per sequence in the next forward when it verifies staged drafts, anchor included.
    verify_len: Option<usize>,
}

impl HybridCache {
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(config: HybridCacheConfig, device: &Device) -> Result<Self> {
        let mut pools = config.pools.iter();
        let mut caches = Vec::with_capacity(config.layer_types.len() + config.pools.len());

        for layer_type in &config.layer_types {
            let cache = match layer_type {
                HybridLayerType::Attention => HybridLayerCache::Attention(KvCache::new_normal(
                    2,
                    config.max_seq_len,
                    Self::CACHE_GROW_SIZE,
                )),
                HybridLayerType::Recurrent => {
                    let Some(pool) = pools.next() else {
                        hanzo_ml::bail!("hybrid cache: more recurrent layers than pools");
                    };
                    HybridLayerCache::Recurrent(RecurrentStatePool::new(pool.clone(), device)?)
                }
            };
            caches.push(cache);
        }
        // The pools left over are side pools, after the layers.
        for pool in pools {
            caches.push(HybridLayerCache::Recurrent(RecurrentStatePool::new(
                pool.clone(),
                device,
            )?));
        }

        Ok(Self {
            caches,
            config,
            state_indices: None,
            state_indices_host: None,
            verify_len: None,
        })
    }

    /// Announce the next forward: `Some(n)` when it verifies staged drafts, `n` tokens per
    /// sequence with the anchor. Set before every forward, so it never describes an older one.
    pub fn expect_verify(&mut self, verify_len: Option<usize>) {
        self.verify_len = verify_len;
    }

    /// Whether a forward of `seq_len` tokens is the announced verify, and so must keep a trail.
    pub fn records_trail(&self, seq_len: usize) -> bool {
        self.verify_len == Some(seq_len)
    }

    /// Tokens the recurrent pools have consumed for `slot_idx`. `None` without a recurrent
    /// pool, or when the pools disagree.
    pub fn recurrent_offset(&self, slot_idx: usize) -> Option<usize> {
        let mut offsets = self.caches.iter().filter_map(|cache| match cache {
            HybridLayerCache::Recurrent(pool) => Some(pool.get_seqlen_offset(slot_idx)),
            HybridLayerCache::Attention(_) => None,
        });
        let first = offsets.next()?;
        offsets.all(|offset| offset == first).then_some(first)
    }

    /// Undo the last `rejected` positions of the latest forward, in every recurrent pool.
    pub fn rewind_recurrent(&mut self, slot_idx: usize, rejected: usize) -> Result<()> {
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.rewind(slot_idx, rejected)?;
            }
        }
        Ok(())
    }

    /// Allocate state slots for a new sequence across all recurrent pools.
    /// Returns the slot index (same for all pools).
    pub fn allocate_seq(&mut self) -> Option<usize> {
        // Collect recurrent layer indices once so rollback can target only recurrent pools.
        let recurrent_layers: Vec<usize> = self
            .caches
            .iter()
            .enumerate()
            .filter_map(|(idx, cache)| match cache {
                HybridLayerCache::Recurrent(_) => Some(idx),
                HybridLayerCache::Attention(_) => None,
            })
            .collect();

        let mut expected_slot = None;
        let mut allocated_slots = Vec::new();

        for &layer_idx in &recurrent_layers {
            let slot_idx = {
                let HybridLayerCache::Recurrent(pool) = &mut self.caches[layer_idx] else {
                    unreachable!("recurrent_layers only contains recurrent entries");
                };
                match pool.allocate() {
                    Some(idx) => idx,
                    None => {
                        for (&rollback_layer_idx, &rollback_slot_idx) in
                            recurrent_layers.iter().zip(allocated_slots.iter())
                        {
                            if let HybridLayerCache::Recurrent(pool) =
                                &mut self.caches[rollback_layer_idx]
                            {
                                pool.free(rollback_slot_idx);
                            }
                        }
                        return None;
                    }
                }
            };

            if let Some(expected) = expected_slot {
                if slot_idx != expected {
                    tracing::warn!(
                        "Hybrid recurrent pool slot mismatch: expected {expected}, got {slot_idx}. Rolling back allocation."
                    );
                    if let HybridLayerCache::Recurrent(pool) = &mut self.caches[layer_idx] {
                        pool.free(slot_idx);
                    }
                    for (&rollback_layer_idx, &rollback_slot_idx) in
                        recurrent_layers.iter().zip(allocated_slots.iter())
                    {
                        if let HybridLayerCache::Recurrent(pool) =
                            &mut self.caches[rollback_layer_idx]
                        {
                            pool.free(rollback_slot_idx);
                        }
                    }
                    return None;
                }
            } else {
                expected_slot = Some(slot_idx);
            }

            allocated_slots.push(slot_idx);
        }

        expected_slot
    }

    /// Free state slots for a sequence across all recurrent pools.
    pub fn free_seq(&mut self, slot_idx: usize) {
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.free(slot_idx);
            }
        }
    }

    /// Reset a specific sequence's state in all recurrent pools.
    pub fn reset_seq(&mut self, slot_idx: usize) -> Result<()> {
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.reset_slot(slot_idx)?;
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Model layers. Side pools follow them in `caches`, so this is not `caches.len()`.
    pub fn num_layers(&self) -> usize {
        self.config.layer_types.len()
    }

    pub fn layer_types(&self) -> &[HybridLayerType] {
        &self.config.layer_types
    }

    pub fn config(&self) -> &HybridCacheConfig {
        &self.config
    }

    /// Get a mutable reference to a specific layer's cache; side pools follow from `num_layers()`
    pub fn get_mut(&mut self, layer: usize) -> Option<&mut HybridLayerCache> {
        self.caches.get_mut(layer)
    }

    /// Get a reference to a specific layer's cache; side pools follow from `num_layers()`
    pub fn get(&self, layer: usize) -> Option<&HybridLayerCache> {
        self.caches.get(layer)
    }

    /// Set the state indices for the current batch.
    /// Called by HybridCacheManager::clone_in_cache before forward.
    pub fn set_state_indices(&mut self, indices: Option<Tensor>) {
        self.state_indices = indices;
    }

    /// Set the host mirror of the state indices for the current batch. Kept in lockstep
    /// with `set_state_indices` so the model can read slots without a device sync.
    pub fn set_state_indices_host(&mut self, indices: Option<Vec<u32>>) {
        self.state_indices_host = indices;
    }

    /// Get the state indices for the current batch.
    /// Used by the model during forward to access recurrent state pool.
    pub fn state_indices(&self) -> Option<&Tensor> {
        self.state_indices.as_ref()
    }

    /// Get the host slot indices for the current batch (sync-free path).
    pub fn state_indices_host(&self) -> Option<&[u32]> {
        self.state_indices_host.as_deref()
    }
}

impl PastKvLenCache for HybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        for cache in &self.caches {
            if let HybridLayerCache::Attention(kv) = cache {
                return Ok(kv.current_seq_len());
            }
        }
        Ok(0)
    }
}

impl HybridCache {
    /// Truncate all attention layer KV caches to the given sequence length.
    /// Recurrent layers are unchanged: `rewind_recurrent` undoes a verify, snapshot/restore
    /// anything older.
    pub fn truncate_attention_to(&mut self, len: usize) -> Result<()> {
        for cache in &mut self.caches {
            if let HybridLayerCache::Attention(kv) = cache {
                kv.set_len(len)?;
            }
        }
        Ok(())
    }
}

/// Snapshot of a single recurrent pool's state for prefix caching. Recurrent state cannot be
/// rewound, so a snapshot serves exactly one prefix: the first `seqlen_offset` tokens.
#[derive(Clone, Debug)]
pub struct RecurrentStateSnapshot {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
}

impl RecurrentStateSnapshot {
    /// Device bytes this snapshot holds.
    pub fn bytes(&self) -> usize {
        [&self.conv_state, &self.recurrent_state]
            .into_iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

impl HybridCache {
    /// Snapshot the recurrent state for a sequence at the given slot index.
    /// Returns one snapshot per recurrent pool, in cache order.
    #[allow(clippy::cast_possible_truncation)]
    pub fn snapshot_recurrent_state(&self, slot_idx: usize) -> Result<Vec<RecurrentStateSnapshot>> {
        let mut snapshots = Vec::new();
        for cache in &self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let idx_tensor = Tensor::from_vec(vec![slot_idx as u32], (1,), pool.device())?;
                let conv = pool.gather_conv_state(&idx_tensor)?;
                let recurrent = pool.gather_recurrent_state(&idx_tensor)?;
                snapshots.push(RecurrentStateSnapshot {
                    conv_state: conv,
                    recurrent_state: recurrent,
                    seqlen_offset: pool.get_seqlen_offset(slot_idx),
                });
            }
        }
        Ok(snapshots)
    }

    /// Restore recurrent state snapshots into the pool at the given slot index.
    /// Snapshots must be in the same order as returned by `snapshot_recurrent_state`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn restore_recurrent_state(
        &mut self,
        slot_idx: usize,
        snapshots: &[RecurrentStateSnapshot],
    ) -> Result<()> {
        let mut snap_iter = snapshots.iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                if let Some(snap) = snap_iter.next() {
                    let conv = snap.conv_state.to_device(pool.device())?;
                    let recurrent = snap.recurrent_state.to_device(pool.device())?;
                    let idx_tensor = Tensor::from_vec(vec![slot_idx as u32], (1,), pool.device())?;
                    pool.scatter_conv_state(&idx_tensor, &conv)?;
                    pool.scatter_recurrent_state(&idx_tensor, &recurrent)?;
                    pool.set_seqlen_offset(slot_idx, snap.seqlen_offset);
                    pool.set_trail(None);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gdn::{forward_pooled, GdnLayerCache, PoolSlots};

    const CONV: (usize, usize) = (2, 3);
    const STATE: [usize; 3] = [1, 2, 2];
    const LEN: usize = 3;

    fn config(state_dims: &[usize], conv_dtype: DType, state_dtype: DType) -> RecurrentLayerConfig {
        RecurrentLayerConfig {
            conv_dim: CONV.0,
            conv_width: CONV.1,
            state_dims: state_dims.to_vec(),
            conv_dtype,
            state_dtype,
        }
    }

    fn pool() -> Result<RecurrentStatePool> {
        let mut pool =
            RecurrentStatePool::new(config(&STATE, DType::F32, DType::F32), &Device::Cpu)?;
        assert_eq!((pool.allocate(), pool.allocate()), (Some(0), Some(1)));
        Ok(pool)
    }

    /// Deterministic values in [-1, 1).
    fn seeded(shape: &[usize], seed: usize) -> Result<Tensor> {
        let n = shape.iter().product::<usize>();
        let v = (0..n)
            .map(|i| ((i * 2654435761 + seed * 40503) % 1009) as f32 / 504.5 - 1.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(v, shape, &Device::Cpu)
    }

    /// Entry `t`, batch row `row` holds the constant `10 t + row`, so a restored slot names the
    /// position and the row it came from.
    fn trail(slots: &[u32], start_offsets: &[usize]) -> Result<RecurrentTrail> {
        let at = |t: usize, tail: &[usize]| -> Result<Tensor> {
            let rows = (0..slots.len())
                .map(|row| {
                    let mut shape = vec![1];
                    shape.extend_from_slice(tail);
                    Tensor::full((10 * t + row) as f32, shape, &Device::Cpu)
                })
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&rows, 0)
        };
        Ok(RecurrentTrail {
            slots: slots.to_vec(),
            start_offsets: start_offsets.to_vec(),
            conv: (0..LEN)
                .map(|t| at(t, &[CONV.0, CONV.1]))
                .collect::<Result<_>>()?,
            recurrent: (0..LEN).map(|t| at(t, &STATE)).collect::<Result<_>>()?,
        })
    }

    /// Every value of `slot`, conv then recurrent, as f32.
    fn slot_values(pool: &RecurrentStatePool, slot: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let flat = |t: &Tensor| t.i(slot)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1();
        Ok((flat(&pool.conv_state)?, flat(&pool.recurrent_state)?))
    }

    /// Slots 1 and 0 ran a three-position forward as batch rows 0 and 1, from offsets 4 and 7.
    fn after_forward() -> Result<RecurrentStatePool> {
        let mut pool = pool()?;
        pool.set_seqlen_offset(1, 4 + LEN);
        pool.set_seqlen_offset(0, 7 + LEN);
        pool.set_trail(Some(trail(&[1, 0], &[4, 7])?));
        Ok(pool)
    }

    #[test]
    fn rewind_restores_the_kept_position_of_that_row_only() -> Result<()> {
        let mut pool = after_forward()?;
        let untouched = slot_values(&pool, 1)?;

        pool.rewind(0, 2)?;

        // Slot 0 is batch row 1, and keeping one of three positions is trail entry 0.
        let (conv, recurrent) = slot_values(&pool, 0)?;
        assert!(conv.iter().chain(&recurrent).all(|&v| v == 1.0));
        assert_eq!(pool.get_seqlen_offset(0), 7 + 1);
        assert_eq!(slot_values(&pool, 1)?, untouched);
        assert_eq!(pool.get_seqlen_offset(1), 4 + LEN);

        pool.rewind(1, 1)?;
        let (conv, recurrent) = slot_values(&pool, 1)?;
        assert!(conv.iter().chain(&recurrent).all(|&v| v == 10.0));
        assert_eq!(pool.get_seqlen_offset(1), 4 + 2);
        Ok(())
    }

    #[test]
    fn rewind_of_nothing_needs_no_trail() -> Result<()> {
        let mut pool = pool()?;
        pool.rewind(0, 0)
    }

    #[test]
    fn rewind_refuses_what_the_trail_does_not_cover() -> Result<()> {
        let mut bare = pool()?;
        assert!(bare.rewind(0, 1).is_err(), "no trail");

        let mut pool = after_forward()?;
        let before = (slot_values(&pool, 0)?, slot_values(&pool, 1)?);
        assert!(pool.rewind(0, LEN).is_err(), "every position rejected");
        assert!(pool.rewind(3, 1).is_err(), "slot outside the forward");

        pool.rewind(0, 1)?;
        let rewound = slot_values(&pool, 0)?;
        assert!(
            pool.rewind(0, 1).is_err(),
            "the trail no longer ends at the slot"
        );
        assert_eq!(slot_values(&pool, 0)?, rewound);
        assert_eq!(slot_values(&pool, 1)?, before.1);

        // A pool with recurrent state trails it at every position its conv trail covers.
        for cut in [1, LEN] {
            let mut pool = after_forward()?;
            let mut short = trail(&[1, 0], &[4, 7])?;
            short.recurrent.truncate(LEN - cut);
            pool.set_trail(Some(short));
            assert!(pool.rewind(0, 1).is_err(), "a recurrent trail {cut} short");
            assert_eq!(slot_values(&pool, 0)?, before.0);
        }
        Ok(())
    }

    #[test]
    fn state_replaced_outside_a_forward_drops_the_trail() -> Result<()> {
        let mut pool = after_forward()?;
        pool.reset_slot(1)?;
        assert!(pool.rewind(0, 1).is_err());

        let mut pool = after_forward()?;
        pool.free(1);
        assert!(pool.rewind(0, 1).is_err());
        Ok(())
    }

    #[test]
    fn only_the_announced_verify_keeps_a_trail() -> Result<()> {
        let mut cache = HybridCache::new(
            HybridCacheConfig::uniform(
                vec![HybridLayerType::Recurrent, HybridLayerType::Attention],
                16,
                config(&STATE, DType::F32, DType::F32),
            ),
            &Device::Cpu,
        )?;
        assert!(!cache.records_trail(4));
        cache.expect_verify(Some(4));
        assert!(cache.records_trail(4));
        assert!(!cache.records_trail(1) && !cache.records_trail(5));
        cache.expect_verify(None);
        assert!(!cache.records_trail(4));
        Ok(())
    }

    /// A conv-only layer: it notes its trail, keeps the last `CONV.1` raw inputs as its state
    /// (`GdnLayerCache::trail_conv`), and runs no recurrence.
    fn conv_only(cache: &mut GdnLayerCache, x: &Tensor) -> Result<Tensor> {
        cache.trail_conv(x)?;
        let seq_len = x.dim(1)?;
        let window = Tensor::cat(&[&cache.conv_state, &x.transpose(1, 2)?], 2)?;
        cache.conv_state = window.narrow(2, seq_len, CONV.1)?.contiguous()?;
        cache.seqlen_offset += seq_len;
        Ok(x.clone())
    }

    /// A conv-only pool keeps no recurrent trail. A verify through it, rejecting the tail, rewinds
    /// to the conv state that decoding the accepted tokens one at a time reaches, and the next true
    /// token lands where plain decoding put it.
    #[test]
    fn conv_only_pool_rewinds_on_its_conv_trail() -> Result<()> {
        const VERIFY: usize = 4;
        let (slot, prompt, kept) = (1usize, 2usize, 2usize);
        let stream = seeded(&[1, prompt + VERIFY, CONV.0], 1)?;
        let wrong = seeded(&[1, VERIFY - kept, CONV.0], 2)?;
        let indices = Tensor::from_vec(vec![slot as u32], 1, &Device::Cpu)?;
        let run = |pool: &mut RecurrentStatePool, x: &Tensor, trail: bool| {
            forward_pooled(pool, PoolSlots::Many(&indices), 0, trail, |cache| {
                conv_only(cache, x)
            })
        };
        let prefilled = || -> Result<RecurrentStatePool> {
            let mut pool =
                RecurrentStatePool::new(config(&[], DType::F32, DType::F32), &Device::Cpu)?;
            assert_eq!((pool.allocate(), pool.allocate()), (Some(0), Some(1)));
            run(&mut pool, &stream.narrow(1, 0, prompt)?, false)?;
            Ok(pool)
        };

        let mut plain = prefilled()?;
        let mut states = Vec::with_capacity(VERIFY);
        for t in 0..VERIFY {
            run(&mut plain, &stream.narrow(1, prompt + t, 1)?, false)?;
            states.push(slot_values(&plain, slot)?);
        }

        let mut spec = prefilled()?;
        let verify = Tensor::cat(&[stream.narrow(1, prompt, kept)?, wrong], 1)?;
        run(&mut spec, &verify, true)?;
        spec.rewind(slot, VERIFY - kept)?;
        assert_eq!(spec.get_seqlen_offset(slot), prompt + kept);
        assert_eq!(slot_values(&spec, slot)?, states[kept - 1]);
        assert_eq!(slot_values(&spec, slot)?.1, vec![0.0]);

        run(&mut spec, &stream.narrow(1, prompt + kept, 1)?, false)?;
        assert_eq!(slot_values(&spec, slot)?, states[kept]);
        Ok(())
    }

    /// A GDN layer, an attention layer, and a conv-only side pool after them. Both pools keep a
    /// bf16 conv state beside an f32 recurrent state.
    fn side_cache() -> Result<HybridCache> {
        let mut cfg = HybridCacheConfig::uniform(
            vec![HybridLayerType::Recurrent, HybridLayerType::Attention],
            16,
            config(&STATE, DType::BF16, DType::F32),
        );
        cfg.pools.push(config(&[], DType::BF16, DType::F32));
        HybridCache::new(cfg, &Device::Cpu)
    }

    fn pool_at(cache: &HybridCache, idx: usize) -> &RecurrentStatePool {
        cache
            .get(idx)
            .and_then(HybridLayerCache::as_recurrent_pool)
            .expect("a recurrent pool")
    }

    #[test]
    fn side_pools_follow_the_layers_in_their_own_dtypes() -> Result<()> {
        let mut cache = side_cache()?;
        assert_eq!((cache.num_layers(), cache.caches.len()), (2, 3));
        assert!(cache
            .get(1)
            .and_then(HybridLayerCache::as_kv_cache)
            .is_some());

        // Five sequences outgrow the first four slots; every pool hands out the same ones.
        for want in 0..5 {
            assert_eq!(cache.allocate_seq(), Some(want));
        }
        for idx in [0, 2] {
            let pool = pool_at(&cache, idx);
            assert_eq!(pool.capacity(), 8);
            assert_eq!(
                (pool.conv_state.dtype(), pool.recurrent_state.dtype()),
                (DType::BF16, DType::F32)
            );
        }
        assert_eq!(pool_at(&cache, 2).recurrent_state.dims(), &[8]);

        // Fill slot 3 from f32 values: the conv state keeps them rounded to bf16, the recurrent
        // state keeps them exactly. Then snapshot it, wipe it, and restore it there and into slot 1.
        let slot = Tensor::from_vec(vec![3u32], 1, &Device::Cpu)?;
        let mut filled = Vec::new();
        for (seed, idx) in [(3, 0), (5, 2)] {
            let pool = cache
                .get_mut(idx)
                .and_then(HybridLayerCache::as_recurrent_pool_mut)
                .expect("a recurrent pool");
            let conv = seeded(&[1, CONV.0, CONV.1], seed)?;
            let mut shape = vec![1];
            shape.extend_from_slice(&pool.recurrent_state.dims()[1..]);
            let recurrent = seeded(&shape, seed + 1)?;
            pool.scatter_conv_state(&slot, &conv)?;
            pool.scatter_recurrent_state(&slot, &recurrent)?;
            pool.set_seqlen_offset(3, 9);

            let flat = |t: &Tensor| t.flatten_all()?.to_vec1::<f32>();
            let rounded = conv.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
            assert_ne!(flat(&rounded)?, flat(&conv)?);
            let want = (flat(&rounded)?, flat(&recurrent)?);
            assert_eq!(slot_values(pool, 3)?, want);
            filled.push(want);
        }

        let snaps = cache.snapshot_recurrent_state(3)?;
        assert_eq!(snaps.len(), 2);
        for snap in &snaps {
            assert_eq!(
                (snap.conv_state.dtype(), snap.recurrent_state.dtype()),
                (DType::BF16, DType::F32)
            );
            assert_eq!(snap.seqlen_offset, 9);
        }
        cache.reset_seq(3)?;
        assert_eq!(
            slot_values(pool_at(&cache, 2), 3)?.0,
            vec![0.0; CONV.0 * CONV.1]
        );

        for target in [3, 1] {
            cache.restore_recurrent_state(target, &snaps)?;
            for (idx, want) in [0, 2].into_iter().zip(&filled) {
                assert_eq!(&slot_values(pool_at(&cache, idx), target)?, want);
            }
            assert_eq!(cache.recurrent_offset(target), Some(9));
        }

        let short = HybridCacheConfig {
            layer_types: vec![HybridLayerType::Recurrent; 2],
            max_seq_len: 16,
            pools: vec![config(&STATE, DType::BF16, DType::F32)],
        };
        assert!(HybridCache::new(short, &Device::Cpu).is_err());
        Ok(())
    }

    /// A verify rewinds every pool. The side pool's trail is conv alone, and its rewind must not
    /// fail the layers' rewind.
    #[test]
    fn rewind_reaches_a_conv_only_side_pool() -> Result<()> {
        let mut cache = side_cache()?;
        assert_eq!(
            (cache.allocate_seq(), cache.allocate_seq()),
            (Some(0), Some(1))
        );
        let mut conv_only = trail(&[1, 0], &[4, 7])?;
        conv_only.recurrent.clear();
        for (idx, trail) in [(0, trail(&[1, 0], &[4, 7])?), (2, conv_only)] {
            let pool = cache
                .get_mut(idx)
                .and_then(HybridLayerCache::as_recurrent_pool_mut)
                .expect("a recurrent pool");
            pool.set_seqlen_offset(1, 4 + LEN);
            pool.set_seqlen_offset(0, 7 + LEN);
            pool.set_trail(Some(trail));
        }

        cache.rewind_recurrent(0, 2)?;

        // Slot 0 is batch row 1, and keeping one of three positions is trail entry 0. The side
        // pool's scalar state was never moved, so it stays zero.
        assert_eq!(cache.recurrent_offset(0), Some(7 + 1));
        assert_eq!(cache.recurrent_offset(1), Some(4 + LEN));
        let (conv, recurrent) = slot_values(pool_at(&cache, 0), 0)?;
        assert!(conv.iter().chain(&recurrent).all(|&v| v == 1.0));
        let (conv, recurrent) = slot_values(pool_at(&cache, 2), 0)?;
        assert!(conv.iter().all(|&v| v == 1.0));
        assert_eq!(recurrent, vec![0.0]);
        Ok(())
    }
}
