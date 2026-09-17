//! Hybrid cache for models that mix attention and recurrent layers (e.g., GraniteMoeHybrid, Qwen3 Next)
//!
//! This implements vLLM-style continuous batching for hybrid models:
//! - Attention layers use standard KV cache batching
//! - Recurrent layers (Mamba SSM or GDN) use a pool-based state with indexed access
//!
//! The key insight is that recurrent state is accessed via `state_indices` which map
//! each sequence in the current batch to its slot in the pool.

use hanzo_ml::{Device, IndexOp, Result, Tensor};

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
    /// `conv[t]`, `recurrent[t]`: batch-major state after position `t`.
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
#[derive(Debug)]
pub struct RecurrentStatePool {
    /// Convolution state pool: (capacity, conv_dim, conv_width)
    pub conv_state: Tensor,
    /// Recurrent state pool: (capacity, ...state_dims)
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
    /// Shape parameters for growing
    conv_dim: usize,
    conv_width: usize,
    state_dims: Vec<usize>,
    dtype: hanzo_ml::DType,
    device: Device,
}

/// Initial pool capacity before dynamic growth.
const INITIAL_POOL_CAPACITY: usize = 4;

impl RecurrentStatePool {
    /// Create a new recurrent state pool.
    ///
    /// - `conv_dim`: dimension of the convolution state
    /// - `conv_width`: kernel size / d_conv for causal conv1d
    /// - `state_dims`: shape of the recurrent state per slot (e.g. `[n_heads, head_dim, d_state]`)
    pub fn new(
        conv_dim: usize,
        conv_width: usize,
        state_dims: Vec<usize>,
        dtype: hanzo_ml::DType,
        device: &Device,
    ) -> Result<Self> {
        let capacity = INITIAL_POOL_CAPACITY;

        let conv_state = Tensor::zeros((capacity, conv_dim, conv_width), dtype, device)?;

        let mut recurrent_shape = vec![capacity];
        recurrent_shape.extend_from_slice(&state_dims);
        let recurrent_state = Tensor::zeros(recurrent_shape, dtype, device)?;

        let free_slots: Vec<usize> = (0..capacity).rev().collect();
        let seqlen_offsets = vec![0; capacity];

        Ok(Self {
            conv_state,
            recurrent_state,
            seqlen_offsets,
            trail: None,
            free_slots,
            capacity,
            conv_dim,
            conv_width,
            state_dims,
            dtype,
            device: device.clone(),
        })
    }

    /// Grow the pool by doubling capacity.
    fn grow(&mut self) -> Result<()> {
        let new_capacity = self.capacity * 2;

        // Allocate new larger conv_state and copy existing data
        let new_conv = Tensor::zeros(
            (new_capacity, self.conv_dim, self.conv_width),
            self.dtype,
            &self.device,
        )?;
        new_conv.slice_set(&self.conv_state, 0, 0)?;

        // Allocate new larger recurrent_state and copy existing data
        let mut recurrent_shape = vec![new_capacity];
        recurrent_shape.extend_from_slice(&self.state_dims);
        let new_recurrent = Tensor::zeros(recurrent_shape, self.dtype, &self.device)?;
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
    /// Fails, leaving the pool untouched, unless the trail covers exactly that forward.
    pub fn rewind(&mut self, slot_idx: usize, rejected: usize) -> Result<()> {
        if rejected == 0 {
            return Ok(());
        }
        let Some(trail) = self.trail.as_ref() else {
            hanzo_ml::bail!(
                "recurrent rewind of {rejected} for slot {slot_idx}: the last forward kept no trail"
            );
        };
        let len = trail.recurrent.len();
        let Some(row) = trail.slots.iter().position(|&s| s as usize == slot_idx) else {
            hanzo_ml::bail!("recurrent rewind: slot {slot_idx} was not in the last forward");
        };
        if trail.conv.len() != len
            || trail.start_offsets.len() != trail.slots.len()
            || rejected >= len
        {
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
        let recurrent = trail.recurrent[keep - 1]
            .narrow(0, row, 1)?
            .to_dtype(self.recurrent_state.dtype())?
            .contiguous()?;
        let offset = start + keep;
        self.conv_state.slice_set(&conv, 0, slot_idx)?;
        self.recurrent_state.slice_set(&recurrent, 0, slot_idx)?;
        self.seqlen_offsets[slot_idx] = offset;
        Ok(())
    }

    /// Reset a specific slot's state to zeros
    pub fn reset_slot(&mut self, slot_idx: usize) -> Result<()> {
        let zero_conv = Tensor::zeros(
            (1, self.conv_dim, self.conv_width),
            self.dtype,
            &self.device,
        )?;

        let mut recurrent_shape = vec![1usize];
        recurrent_shape.extend_from_slice(&self.state_dims);
        let zero_recurrent = Tensor::zeros(recurrent_shape, self.dtype, &self.device)?;

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

    pub fn dtype(&self) -> hanzo_ml::DType {
        self.dtype
    }
}

impl Clone for RecurrentStatePool {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            seqlen_offsets: self.seqlen_offsets.clone(),
            trail: self.trail.clone(),
            free_slots: self.free_slots.clone(),
            capacity: self.capacity,
            conv_dim: self.conv_dim,
            conv_width: self.conv_width,
            state_dims: self.state_dims.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        }
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

/// Configuration for the recurrent layer state dimensions
#[derive(Clone, Debug)]
pub struct RecurrentLayerConfig {
    /// Dimension of the convolution state
    pub conv_dim: usize,
    /// Kernel size for causal conv1d
    pub conv_width: usize,
    /// Shape of the recurrent state per slot.
    /// For Mamba: [n_heads, head_dim, d_state]
    /// For GDN: [n_v_heads, key_dim, value_dim]
    pub state_dims: Vec<usize>,
}

/// Configuration for creating a hybrid cache
#[derive(Clone, Debug)]
pub struct HybridCacheConfig {
    pub layer_types: Vec<HybridLayerType>,
    pub max_seq_len: usize,
    pub recurrent: RecurrentLayerConfig,
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

    pub fn new(config: HybridCacheConfig, dtype: hanzo_ml::DType, device: &Device) -> Result<Self> {
        let mut caches = Vec::with_capacity(config.layer_types.len());

        for layer_type in &config.layer_types {
            let cache = match layer_type {
                HybridLayerType::Attention => HybridLayerCache::Attention(KvCache::new_normal(
                    2,
                    config.max_seq_len,
                    Self::CACHE_GROW_SIZE,
                )),
                HybridLayerType::Recurrent => HybridLayerCache::Recurrent(RecurrentStatePool::new(
                    config.recurrent.conv_dim,
                    config.recurrent.conv_width,
                    config.recurrent.state_dims.clone(),
                    dtype,
                    device,
                )?),
            };
            caches.push(cache);
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

    /// Tokens the recurrent layers have consumed for `slot_idx`. `None` without a recurrent
    /// layer, or when the layers disagree.
    pub fn recurrent_offset(&self, slot_idx: usize) -> Option<usize> {
        let mut offsets = self.caches.iter().filter_map(|cache| match cache {
            HybridLayerCache::Recurrent(pool) => Some(pool.get_seqlen_offset(slot_idx)),
            HybridLayerCache::Attention(_) => None,
        });
        let first = offsets.next()?;
        offsets.all(|offset| offset == first).then_some(first)
    }

    /// Undo the last `rejected` positions of the latest forward, in every recurrent layer.
    pub fn rewind_recurrent(&mut self, slot_idx: usize, rejected: usize) -> Result<()> {
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.rewind(slot_idx, rejected)?;
            }
        }
        Ok(())
    }

    /// Allocate state slots for a new sequence across all recurrent layers.
    /// Returns the slot index (same for all layers).
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

    /// Free state slots for a sequence across all recurrent layers.
    pub fn free_seq(&mut self, slot_idx: usize) {
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.free(slot_idx);
            }
        }
    }

    /// Reset a specific sequence's state in all recurrent layers.
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

    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    pub fn layer_types(&self) -> &[HybridLayerType] {
        &self.config.layer_types
    }

    pub fn config(&self) -> &HybridCacheConfig {
        &self.config
    }

    /// Get a mutable reference to a specific layer's cache
    pub fn get_mut(&mut self, layer: usize) -> Option<&mut HybridLayerCache> {
        self.caches.get_mut(layer)
    }

    /// Get a reference to a specific layer's cache
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

/// Snapshot of a single recurrent layer's state for prefix caching. Recurrent state cannot be
/// rewound, so a snapshot serves exactly one prefix: the first `seqlen_offset` tokens.
#[derive(Clone, Debug)]
pub struct RecurrentStateSnapshot {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
}

impl HybridCache {
    /// Snapshot the recurrent state for a sequence at the given slot index.
    /// Returns one snapshot per recurrent layer, in layer order.
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
    /// Snapshots must be in the same layer order as returned by `snapshot_recurrent_state`.
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
    use hanzo_ml::DType;

    const CONV: (usize, usize) = (2, 3);
    const STATE: [usize; 3] = [1, 2, 2];
    const LEN: usize = 3;

    fn pool() -> Result<RecurrentStatePool> {
        let mut pool =
            RecurrentStatePool::new(CONV.0, CONV.1, STATE.to_vec(), DType::F32, &Device::Cpu)?;
        assert_eq!((pool.allocate(), pool.allocate()), (Some(0), Some(1)));
        Ok(pool)
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

    fn slot_values(pool: &RecurrentStatePool, slot: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((
            pool.conv_state.i(slot)?.flatten_all()?.to_vec1()?,
            pool.recurrent_state.i(slot)?.flatten_all()?.to_vec1()?,
        ))
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
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent, HybridLayerType::Attention],
                max_seq_len: 16,
                recurrent: RecurrentLayerConfig {
                    conv_dim: CONV.0,
                    conv_width: CONV.1,
                    state_dims: STATE.to_vec(),
                },
            },
            DType::F32,
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
}
