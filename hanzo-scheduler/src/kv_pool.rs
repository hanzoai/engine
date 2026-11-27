// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact Paged Attention KV block allocator and fixed FP32 GDN recurrent state slot pool.
//!
//! Hybrid Transformer + Gated DeltaNet (GDN) models allocate two distinct per-sequence
//! GPU memory resources:
//! 1. **Paged Attention KV blocks** (`kv_bytes_per_block` bytes each, growing linearly with
//!    sequence length in units of `block_size` tokens).
//! 2. **Fixed FP32 GDN recurrent state slots** (`gdn_state_bytes_per_seq` bytes each:
//!    `27.0 MiB` on `4B-Hybrid`, `45.0 MiB` on `9B-Hybrid`, `192.0 MiB` on `27B-Hybrid`),
//!    held per active sequence regardless of context length.
//!
//! Both pools share the exact available VRAM envelope (`vram_bytes - weights - workspace`)
//! with strict fail-closed OOM protection (`SchedulerError::InsufficientKvBlocks` /
//! `SchedulerError::InsufficientGdnSlots`) and exact peak watermark tracking.

use std::collections::BTreeMap;
use std::fmt;

use crate::model::{HardwareSpec, HybridModelSpec};

/// Errors returned by the paged KV + GDN memory pool and scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// Not enough free physical KV blocks to satisfy the requested allocation.
    InsufficientKvBlocks {
        /// Sequence identifier requesting blocks.
        seq_id: usize,
        /// Additional blocks required.
        required: usize,
        /// Free blocks currently available in the pool.
        available: usize,
    },
    /// Not enough free GDN recurrent state slots (or VRAM budget) for a new active sequence.
    InsufficientGdnSlots {
        /// Sequence identifier requesting a GDN slot.
        seq_id: usize,
        /// Free GDN slots currently available.
        available: usize,
    },
    /// Sequence identifier is not currently resident in the pool.
    UnknownSequence {
        /// Missing sequence identifier.
        seq_id: usize,
    },
    /// Sequence identifier already has an active allocation in the pool.
    DuplicateSequence {
        /// Duplicate sequence identifier.
        seq_id: usize,
    },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Simulation hit its hard iteration cap while requests remained unfinished.
    SimulationStepLimitExceeded {
        /// Configured iteration cap.
        max_iterations: usize,
        /// Requests completed before the cap was hit.
        completed_requests: usize,
        /// Requests still unfinished when the cap was hit.
        unfinished_requests: usize,
    },
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientKvBlocks {
                seq_id,
                required,
                available,
            } => write!(
                f,
                "OOM guard: seq {seq_id} needs {required} KV blocks, only {available} free"
            ),
            Self::InsufficientGdnSlots { seq_id, available } => write!(
                f,
                "OOM guard: seq {seq_id} needs 1 GDN recurrent slot, only {available} free"
            ),
            Self::UnknownSequence { seq_id } => write!(f, "unknown sequence {seq_id} in KV pool"),
            Self::DuplicateSequence { seq_id } => {
                write!(f, "sequence {seq_id} is already allocated in KV pool")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid scheduler configuration: {msg}"),
            Self::SimulationStepLimitExceeded {
                max_iterations,
                completed_requests,
                unfinished_requests,
            } => write!(
                f,
                "simulation exceeded {max_iterations} steps with {unfinished_requests} unfinished requests ({completed_requests} completed)"
            ),
        }
    }
}

impl std::error::Error for SchedulerError {}

/// Per-sequence physical memory allocation table inside [`PagedKvPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceKvTable {
    /// Logical sequence identifier.
    pub seq_id: usize,
    /// Ordered list of physical KV block IDs currently owned by this sequence.
    pub block_ids: Vec<usize>,
    /// Physical GDN recurrent state slot ID owned by this sequence.
    pub gdn_slot_id: usize,
    /// Number of token positions currently materialized in KV blocks.
    pub materialized_tokens: usize,
}

/// Snapshot of a sequence swapped out to host memory (`PreemptionMode::SwapToHost`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwappedSequenceRecord {
    /// Logical sequence identifier.
    pub seq_id: usize,
    /// Number of KV blocks whose payloads were copied to host RAM.
    pub num_kv_blocks: usize,
    /// Number of materialized token positions preserved on host.
    pub materialized_tokens: usize,
    /// Total bytes transferred to host (KV blocks + FP32 GDN state).
    pub swapped_bytes: u64,
}

/// Exact paged KV block pool and fixed FP32 GDN recurrent state allocator.
///
/// Tracks physical block IDs, physical GDN slot IDs, exact byte occupancy,
/// strict VRAM OOM enforcement, preemption (recompute or host swap), and
/// peak memory watermarks without ever clamping infeasible demand.
#[derive(Debug, Clone)]
pub struct PagedKvPool {
    block_size: usize,
    kv_bytes_per_block: u64,
    gdn_bytes_per_slot: u64,
    total_vram_budget_bytes: u64,
    total_kv_blocks: usize,
    total_gdn_slots: usize,
    free_kv_blocks: Vec<usize>,
    free_gdn_slots: Vec<usize>,
    active_tables: BTreeMap<usize, SequenceKvTable>,
    swapped_tables: BTreeMap<usize, SwappedSequenceRecord>,
    peak_kv_blocks_used: usize,
    peak_gdn_slots_used: usize,
    peak_allocated_bytes: u64,
    peak_swapped_host_bytes: u64,
    total_blocks_allocated: u64,
    total_swap_out_bytes: u64,
    total_swap_in_bytes: u64,
}

impl PagedKvPool {
    /// Builds a [`PagedKvPool`] sized exactly from a [`HybridModelSpec`] and [`HardwareSpec`].
    ///
    /// Allocates GDN recurrent slots up to `max_gdn_slots` (bounded by available VRAM) and
    /// gives all remaining VRAM bytes in the KV+GDN pool to paged KV blocks.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if `block_size == 0` or the available VRAM
    /// cannot hold at least 1 GDN slot and 1 KV block.
    pub fn from_model_and_hardware(
        model: &HybridModelSpec,
        hardware: &HardwareSpec,
        block_size: usize,
        max_gdn_slots: usize,
    ) -> Result<Self, SchedulerError> {
        if block_size == 0 {
            return Err(SchedulerError::InvalidConfig(
                "block_size must be > 0".to_string(),
            ));
        }
        let avail_bytes = hardware.available_kv_and_gdn_bytes(model);
        let kv_bytes_per_block = model.kv_bytes_per_block(block_size);
        let gdn_bytes_per_slot = model.gdn_state_bytes_per_seq();
        if kv_bytes_per_block == 0 || gdn_bytes_per_slot == 0 {
            return Err(SchedulerError::InvalidConfig(
                "model KV/GDN byte sizes must be > 0".to_string(),
            ));
        }

        // Partition available VRAM cleanly: reserve GDN slots only up to at most 40% of pool
        // (or `max_gdn_slots`, whichever is smaller) while ensuring dynamic byte accounting
        // never exceeds `avail_bytes`.
        let max_gdn_by_vram = ((avail_bytes * 40 / 100) / gdn_bytes_per_slot).max(1) as usize;
        let total_gdn_slots = max_gdn_slots.max(1).min(max_gdn_by_vram);
        let gdn_reserved_bytes = (total_gdn_slots as u64) * gdn_bytes_per_slot;
        if avail_bytes <= gdn_reserved_bytes + kv_bytes_per_block {
            return Err(SchedulerError::InvalidConfig(format!(
                "insufficient VRAM ({avail_bytes} B) for model {}",
                model.label
            )));
        }
        let kv_budget_bytes = avail_bytes - gdn_reserved_bytes;
        let total_kv_blocks = (kv_budget_bytes / kv_bytes_per_block) as usize;
        Self::with_explicit_capacity(
            block_size,
            kv_bytes_per_block,
            gdn_bytes_per_slot,
            total_kv_blocks,
            total_gdn_slots,
        )
    }

    /// Creates a [`PagedKvPool`] with explicit block and GDN slot counts (useful for unit tests
    /// and controlled oversubscription scenarios).
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if any capacity parameter is zero.
    pub fn with_explicit_capacity(
        block_size: usize,
        kv_bytes_per_block: u64,
        gdn_bytes_per_slot: u64,
        total_kv_blocks: usize,
        total_gdn_slots: usize,
    ) -> Result<Self, SchedulerError> {
        if block_size == 0 || total_kv_blocks == 0 || total_gdn_slots == 0 {
            return Err(SchedulerError::InvalidConfig(
                "block_size, total_kv_blocks, and total_gdn_slots must be > 0".to_string(),
            ));
        }
        // Reverse order so `pop()` hands out block 0, 1, 2, ... deterministically.
        let free_kv_blocks: Vec<usize> = (0..total_kv_blocks).rev().collect();
        let free_gdn_slots: Vec<usize> = (0..total_gdn_slots).rev().collect();
        let total_vram_budget_bytes = (total_kv_blocks as u64) * kv_bytes_per_block
            + (total_gdn_slots as u64) * gdn_bytes_per_slot;
        Ok(Self {
            block_size,
            kv_bytes_per_block,
            gdn_bytes_per_slot,
            total_vram_budget_bytes,
            total_kv_blocks,
            total_gdn_slots,
            free_kv_blocks,
            free_gdn_slots,
            active_tables: BTreeMap::new(),
            swapped_tables: BTreeMap::new(),
            peak_kv_blocks_used: 0,
            peak_gdn_slots_used: 0,
            peak_allocated_bytes: 0,
            peak_swapped_host_bytes: 0,
            total_blocks_allocated: 0,
            total_swap_out_bytes: 0,
            total_swap_in_bytes: 0,
        })
    }

    /// Tokens per paged KV block.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Bytes per paged KV block.
    #[must_use]
    pub const fn kv_bytes_per_block(&self) -> u64 {
        self.kv_bytes_per_block
    }

    /// Fixed FP32 GDN recurrent state bytes per active sequence slot.
    #[must_use]
    pub const fn gdn_bytes_per_slot(&self) -> u64 {
        self.gdn_bytes_per_slot
    }

    /// Total physical KV blocks in the pool.
    #[must_use]
    pub const fn total_kv_blocks(&self) -> usize {
        self.total_kv_blocks
    }

    /// Total physical GDN recurrent state slots in the pool.
    #[must_use]
    pub const fn total_gdn_slots(&self) -> usize {
        self.total_gdn_slots
    }

    /// Total combined KV + GDN capacity in bytes.
    #[must_use]
    pub const fn total_capacity_bytes(&self) -> u64 {
        self.total_vram_budget_bytes
    }

    /// Currently free KV blocks.
    #[must_use]
    pub fn free_kv_blocks(&self) -> usize {
        self.free_kv_blocks.len()
    }

    /// Currently allocated KV blocks.
    #[must_use]
    pub fn used_kv_blocks(&self) -> usize {
        self.total_kv_blocks - self.free_kv_blocks.len()
    }

    /// Currently free GDN recurrent state slots.
    #[must_use]
    pub fn free_gdn_slots(&self) -> usize {
        self.free_gdn_slots.len()
    }

    /// Currently allocated GDN recurrent state slots.
    #[must_use]
    pub fn used_gdn_slots(&self) -> usize {
        self.total_gdn_slots - self.free_gdn_slots.len()
    }

    /// Currently allocated KV + GDN bytes on GPU.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        (self.used_kv_blocks() as u64) * self.kv_bytes_per_block
            + (self.used_gdn_slots() as u64) * self.gdn_bytes_per_slot
    }

    /// Peak KV blocks simultaneously allocated during the pool's lifetime.
    #[must_use]
    pub const fn peak_kv_blocks_used(&self) -> usize {
        self.peak_kv_blocks_used
    }

    /// Peak GDN slots simultaneously allocated during the pool's lifetime.
    #[must_use]
    pub const fn peak_gdn_slots_used(&self) -> usize {
        self.peak_gdn_slots_used
    }

    /// Peak simultaneous KV + GDN GPU bytes allocated during the pool's lifetime.
    #[must_use]
    pub const fn peak_allocated_bytes(&self) -> u64 {
        self.peak_allocated_bytes
    }

    /// Peak host RAM bytes occupied by swapped-out sequences.
    #[must_use]
    pub const fn peak_swapped_host_bytes(&self) -> u64 {
        self.peak_swapped_host_bytes
    }

    /// Cumulative host swap-out bytes transferred.
    #[must_use]
    pub const fn total_swap_out_bytes(&self) -> u64 {
        self.total_swap_out_bytes
    }

    /// Cumulative host swap-in bytes transferred.
    #[must_use]
    pub const fn total_swap_in_bytes(&self) -> u64 {
        self.total_swap_in_bytes
    }

    /// Cumulative KV block allocations performed.
    #[must_use]
    pub const fn total_blocks_allocated(&self) -> u64 {
        self.total_blocks_allocated
    }

    /// Current KV block utilization in `[0.0, 1.0]`.
    #[must_use]
    pub fn kv_utilization(&self) -> f64 {
        self.used_kv_blocks() as f64 / self.total_kv_blocks as f64
    }

    /// Number of KV blocks required to store `num_tokens` tokens.
    #[must_use]
    pub const fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Checks whether a new sequence needing `initial_tokens` can be admitted while preserving
    /// `headroom_blocks` free KV blocks for currently running sequences.
    #[must_use]
    pub fn can_admit(&self, initial_tokens: usize, headroom_blocks: usize) -> bool {
        let needed_blocks = self.blocks_for_tokens(initial_tokens.max(1));
        !self.free_gdn_slots.is_empty()
            && self.free_kv_blocks.len() >= needed_blocks.saturating_add(headroom_blocks)
    }

    /// Checks whether a sequence currently allocated can grow from its current materialized length
    /// to `new_total_tokens` without exceeding the KV pool.
    #[must_use]
    pub fn can_grow(&self, seq_id: usize, new_total_tokens: usize) -> bool {
        let Some(table) = self.active_tables.get(&seq_id) else {
            return false;
        };
        let target_blocks = self.blocks_for_tokens(new_total_tokens);
        let additional = target_blocks.saturating_sub(table.block_ids.len());
        self.free_kv_blocks.len() >= additional
    }

    /// Returns whether `seq_id` currently owns resident GPU KV + GDN allocations.
    #[must_use]
    pub fn is_resident(&self, seq_id: usize) -> bool {
        self.active_tables.contains_key(&seq_id)
    }

    /// Returns whether `seq_id` is currently swapped out to host memory.
    #[must_use]
    pub fn is_swapped(&self, seq_id: usize) -> bool {
        self.swapped_tables.contains_key(&seq_id)
    }

    /// Returns the resident [`SequenceKvTable`] for `seq_id`, if allocated.
    #[must_use]
    pub fn sequence_table(&self, seq_id: usize) -> Option<&SequenceKvTable> {
        self.active_tables.get(&seq_id)
    }

    /// Admits a new sequence, allocating 1 GDN recurrent state slot and enough KV blocks for
    /// `initial_tokens`. Atomic: on error, zero blocks or slots are leaked.
    ///
    /// # Errors
    /// Returns [`SchedulerError::DuplicateSequence`], [`SchedulerError::InsufficientGdnSlots`],
    /// or [`SchedulerError::InsufficientKvBlocks`].
    pub fn admit_sequence(
        &mut self,
        seq_id: usize,
        initial_tokens: usize,
    ) -> Result<&SequenceKvTable, SchedulerError> {
        if self.active_tables.contains_key(&seq_id) {
            return Err(SchedulerError::DuplicateSequence { seq_id });
        }
        if self.free_gdn_slots.is_empty() {
            return Err(SchedulerError::InsufficientGdnSlots {
                seq_id,
                available: 0,
            });
        }
        let needed_blocks = self.blocks_for_tokens(initial_tokens.max(1));
        if self.free_kv_blocks.len() < needed_blocks {
            return Err(SchedulerError::InsufficientKvBlocks {
                seq_id,
                required: needed_blocks,
                available: self.free_kv_blocks.len(),
            });
        }
        let gdn_slot_id = self.free_gdn_slots.pop().expect("checked non-empty");
        let mut block_ids = Vec::with_capacity(needed_blocks);
        for _ in 0..needed_blocks {
            block_ids.push(self.free_kv_blocks.pop().expect("checked len"));
        }
        self.total_blocks_allocated += needed_blocks as u64;
        self.active_tables.insert(
            seq_id,
            SequenceKvTable {
                seq_id,
                block_ids,
                gdn_slot_id,
                materialized_tokens: initial_tokens,
            },
        );
        self.update_peaks();
        Ok(self.active_tables.get(&seq_id).expect("just inserted"))
    }

    /// Grows an active sequence's KV block table so it holds at least `new_total_tokens` tokens.
    ///
    /// # Errors
    /// Returns [`SchedulerError::UnknownSequence`] or [`SchedulerError::InsufficientKvBlocks`].
    pub fn allocate_for_len(
        &mut self,
        seq_id: usize,
        new_total_tokens: usize,
    ) -> Result<usize, SchedulerError> {
        let target_blocks = self.blocks_for_tokens(new_total_tokens);
        let current_blocks = match self.active_tables.get(&seq_id) {
            Some(t) => t.block_ids.len(),
            None => return Err(SchedulerError::UnknownSequence { seq_id }),
        };
        let additional = target_blocks.saturating_sub(current_blocks);
        if self.free_kv_blocks.len() < additional {
            return Err(SchedulerError::InsufficientKvBlocks {
                seq_id,
                required: additional,
                available: self.free_kv_blocks.len(),
            });
        }
        let table = self.active_tables.get_mut(&seq_id).expect("checked exists");
        for _ in 0..additional {
            table
                .block_ids
                .push(self.free_kv_blocks.pop().expect("checked len"));
        }
        if new_total_tokens > table.materialized_tokens {
            table.materialized_tokens = new_total_tokens;
        }
        self.total_blocks_allocated += additional as u64;
        self.update_peaks();
        Ok(additional)
    }

    /// Preempts `seq_id` by recomputation: frees all GPU KV blocks and its GDN slot immediately,
    /// returning the number of materialized tokens that will need to be recomputed on re-admission.
    ///
    /// # Errors
    /// Returns [`SchedulerError::UnknownSequence`] if `seq_id` is not resident.
    pub fn preempt_recompute(&mut self, seq_id: usize) -> Result<usize, SchedulerError> {
        let table = self
            .active_tables
            .remove(&seq_id)
            .ok_or(SchedulerError::UnknownSequence { seq_id })?;
        self.free_gdn_slots.push(table.gdn_slot_id);
        for blk in table.block_ids.into_iter().rev() {
            self.free_kv_blocks.push(blk);
        }
        Ok(table.materialized_tokens)
    }

    /// Preempts `seq_id` by swapping its KV blocks + FP32 GDN state out to host RAM,
    /// freeing GPU capacity immediately and returning the swapped byte count.
    ///
    /// # Errors
    /// Returns [`SchedulerError::UnknownSequence`] if `seq_id` is not resident.
    pub fn swap_out(&mut self, seq_id: usize) -> Result<u64, SchedulerError> {
        let table = self
            .active_tables
            .remove(&seq_id)
            .ok_or(SchedulerError::UnknownSequence { seq_id })?;
        let num_kv_blocks = table.block_ids.len();
        let swapped_bytes =
            (num_kv_blocks as u64) * self.kv_bytes_per_block + self.gdn_bytes_per_slot;
        self.free_gdn_slots.push(table.gdn_slot_id);
        for blk in table.block_ids.into_iter().rev() {
            self.free_kv_blocks.push(blk);
        }
        self.total_swap_out_bytes += swapped_bytes;
        self.swapped_tables.insert(
            seq_id,
            SwappedSequenceRecord {
                seq_id,
                num_kv_blocks,
                materialized_tokens: table.materialized_tokens,
                swapped_bytes,
            },
        );
        let cur_host: u64 = self.swapped_tables.values().map(|r| r.swapped_bytes).sum();
        if cur_host > self.peak_swapped_host_bytes {
            self.peak_swapped_host_bytes = cur_host;
        }
        Ok(swapped_bytes)
    }

    /// Checks whether a swapped-out sequence can be swapped back into GPU memory while leaving
    /// `headroom_blocks` free blocks.
    #[must_use]
    pub fn can_swap_in(&self, seq_id: usize, headroom_blocks: usize) -> bool {
        let Some(rec) = self.swapped_tables.get(&seq_id) else {
            return false;
        };
        !self.free_gdn_slots.is_empty()
            && self.free_kv_blocks.len() >= rec.num_kv_blocks.saturating_add(headroom_blocks)
    }

    /// Restores a previously swapped-out sequence back into GPU memory, returning bytes transferred.
    ///
    /// # Errors
    /// Returns [`SchedulerError::UnknownSequence`], [`SchedulerError::InsufficientGdnSlots`],
    /// or [`SchedulerError::InsufficientKvBlocks`].
    pub fn swap_in(&mut self, seq_id: usize) -> Result<u64, SchedulerError> {
        let rec = self
            .swapped_tables
            .get(&seq_id)
            .cloned()
            .ok_or(SchedulerError::UnknownSequence { seq_id })?;
        if self.free_gdn_slots.is_empty() {
            return Err(SchedulerError::InsufficientGdnSlots {
                seq_id,
                available: 0,
            });
        }
        if self.free_kv_blocks.len() < rec.num_kv_blocks {
            return Err(SchedulerError::InsufficientKvBlocks {
                seq_id,
                required: rec.num_kv_blocks,
                available: self.free_kv_blocks.len(),
            });
        }
        self.swapped_tables.remove(&seq_id);
        let gdn_slot_id = self.free_gdn_slots.pop().expect("checked non-empty");
        let mut block_ids = Vec::with_capacity(rec.num_kv_blocks);
        for _ in 0..rec.num_kv_blocks {
            block_ids.push(self.free_kv_blocks.pop().expect("checked len"));
        }
        self.total_blocks_allocated += rec.num_kv_blocks as u64;
        self.total_swap_in_bytes += rec.swapped_bytes;
        self.active_tables.insert(
            seq_id,
            SequenceKvTable {
                seq_id,
                block_ids,
                gdn_slot_id,
                materialized_tokens: rec.materialized_tokens,
            },
        );
        self.update_peaks();
        Ok(rec.swapped_bytes)
    }

    /// Frees all GPU (or host-swapped) KV blocks and GDN slot held by a finished sequence.
    pub fn free_sequence(&mut self, seq_id: usize) {
        if let Some(table) = self.active_tables.remove(&seq_id) {
            self.free_gdn_slots.push(table.gdn_slot_id);
            for blk in table.block_ids.into_iter().rev() {
                self.free_kv_blocks.push(blk);
            }
        }
        self.swapped_tables.remove(&seq_id);
    }

    /// Verifies that zero blocks and zero GDN slots are leaked when no sequences remain active.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.active_tables.is_empty()
            && self.swapped_tables.is_empty()
            && self.free_kv_blocks.len() == self.total_kv_blocks
            && self.free_gdn_slots.len() == self.total_gdn_slots
    }

    fn update_peaks(&mut self) {
        let used_kv = self.used_kv_blocks();
        let used_gdn = self.used_gdn_slots();
        let bytes = self.allocated_bytes();
        if used_kv > self.peak_kv_blocks_used {
            self.peak_kv_blocks_used = used_kv;
        }
        if used_gdn > self.peak_gdn_slots_used {
            self.peak_gdn_slots_used = used_gdn;
        }
        if bytes > self.peak_allocated_bytes {
            self.peak_allocated_bytes = bytes;
        }
    }
}
