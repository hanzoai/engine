// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Graph memory-lifetime pool and sanitizer for CUDA decode & verification graphs.
//!
//! Enforces the hard memory-lifetime requirements of HIP-1346 / `#54` before every
//! graph replay, modeling `compute-sanitizer --tool memcheck/racecheck/initcheck`
//! checks directly in deterministic, `#![forbid(unsafe_code)]` Rust:
//!
//! 1. **Zero Use-After-Free / Poisoned Access**: Every captured node binds `(BufferId, BufferGeneration,
//!    byte_offset, byte_len)`. Freeing a referenced buffer marks it `Poisoned` (if still referenced by
//!    any live graph) or `Freed`; any subsequent replay fails closed immediately.
//! 2. **Zero Generation Drift**: Reallocating a buffer slot bumps `BufferGeneration` and moves the
//!    virtual address, causing any stale graph referencing the old generation to trigger
//!    `SanitizerViolation::GenerationMismatch`.
//! 3. **Zero Dynamic Allocation or Host-Sync Inside Capture**: Attempting an unpooled dynamic
//!    allocation or host synchronization (`cudaStreamSynchronize` / pageable HtoD) during active
//!    capture fails closed unless explicitly routed through a breakable graph boundary.
//! 4. **Zero Leaked Capture Scratch**: Any private-pool scratch allocated inside a capture epoch
//!    must be freed before `end_capture`, matching Hanzo Engine's `capture_cuda_graph_with_output`
//!    contract.
//! 5. **Zero Write-Alias Races**: Detects overlapping address ranges when a node writes to GDN
//!    recurrent state and concurrently aliases attention KV scratch buffers (`StateAliasRace`).
//! 6. **Persistent Static I/O & Metadata Buffers**: [`PersistentGraphBuffers`] mirrors Hanzo Engine's
//!    `CudaDecodeGraphMetadataBuffers` (including `block_table_signature` skip semantics) and extends
//!    it with fixed-address `gdn_recurrent_state_ptrs`, `spec_draft_tokens`, `lane_active_mask`,
//!    and `output_logits`.

use std::collections::BTreeMap;
use std::fmt;

/// Device placement mirror aligned with Hanzo Engine's `DeviceLocation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceLocation {
    /// Host CPU execution.
    Cpu,
    /// CUDA device identified by ordinal `gpu_id`.
    Cuda {
        /// Zero-based CUDA GPU ordinal.
        gpu_id: usize,
    },
    /// Metal device identified by ordinal `gpu_id`.
    Metal {
        /// Zero-based Metal GPU ordinal.
        gpu_id: usize,
    },
}

/// Unique opaque identifier for a virtual GPU pool allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(pub u64);

/// Monotonic generation counter for a buffer slot; increments on every reallocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferGeneration(pub u64);

/// Monotonic capture epoch identifying a single stream-capture session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CaptureEpoch(pub u64);

/// Lifecycle state of an allocation inside [`GraphMemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolBlockState {
    /// Pre-allocated fixed-address persistent buffer valid across graph replays.
    PersistentStatic,
    /// Temporary private-pool scratch allocated during capture; must be freed before `end_capture`.
    ScratchAllocated,
    /// Cleanly freed allocation with zero outstanding graph references.
    Freed,
    /// Freed while still referenced by one or more live captured graphs (use-after-free hazard).
    Poisoned,
}

/// Access mode for a captured node's reference to a GPU buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccessKind {
    /// Read-only access.
    Read,
    /// Write-only access.
    Write,
    /// In-place read-modify-write access (e.g., GDN recurrent state update).
    ReadWrite,
}

impl BufferAccessKind {
    /// Returns true if this access writes to the underlying buffer.
    #[must_use]
    pub const fn is_write(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }

    /// Returns true if this access reads from the underlying buffer.
    #[must_use]
    pub const fn is_read(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
}

/// Exact buffer binding recorded inside a captured CUDA graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedBufferRef {
    /// Target buffer handle.
    pub buffer_id: BufferId,
    /// Buffer generation observed at capture time.
    pub generation: BufferGeneration,
    /// Starting byte offset within the buffer.
    pub byte_offset: usize,
    /// Byte length accessed by the kernel node.
    pub byte_len: usize,
    /// Read/write access mode.
    pub access: BufferAccessKind,
    /// Human-readable tensor role label (e.g. `"gdn_recurrent_state_ptrs"`).
    pub label: String,
    /// Optional explicit virtual address override for testing deliberate alias corruption.
    pub virtual_addr_override: Option<u64>,
}

/// Hard memory-lifetime, capture-safety, or race violation detected by [`GraphSanitizer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SanitizerViolation {
    /// Graph node accessed a buffer that was freed after capture (`memcheck` UAF).
    UseAfterFree {
        /// Node index inside the captured graph.
        node_id: usize,
        /// Freed buffer identifier.
        buffer_id: BufferId,
        /// Buffer label.
        label: String,
    },
    /// Graph node accessed a buffer whose memory was freed while still pinned by a live graph.
    PoisonedBufferAccess {
        /// Node index inside the captured graph.
        node_id: usize,
        /// Poisoned buffer identifier.
        buffer_id: BufferId,
        /// Buffer label.
        label: String,
    },
    /// Graph node holds a stale generation because the backing slot was reallocated (`memcheck`).
    GenerationMismatch {
        /// Node index inside the captured graph.
        node_id: usize,
        /// Buffer identifier.
        buffer_id: BufferId,
        /// Generation pinned at capture time.
        captured_gen: BufferGeneration,
        /// Live generation currently in the pool.
        live_gen: BufferGeneration,
        /// Buffer label.
        label: String,
    },
    /// Graph node attempted to read or write past the allocated buffer capacity (`memcheck` OOB).
    OutOfBounds {
        /// Node index inside the captured graph.
        node_id: usize,
        /// Buffer identifier.
        buffer_id: BufferId,
        /// End byte offset requested (`byte_offset + byte_len`).
        access_end: usize,
        /// Allocated capacity in bytes.
        capacity_bytes: usize,
        /// Buffer label.
        label: String,
    },
    /// Graph node read from a buffer that was never initialized (`initcheck`).
    UninitializedRead {
        /// Node index inside the captured graph.
        node_id: usize,
        /// Buffer identifier.
        buffer_id: BufferId,
        /// Buffer label.
        label: String,
    },
    /// Capture-local scratch allocation was not freed before `end_capture`.
    LeakedCaptureScratch {
        /// Leaked scratch buffer identifier.
        buffer_id: BufferId,
        /// Leaked byte count.
        byte_len: usize,
        /// Buffer label.
        label: String,
    },
    /// Two buffer references in the same node alias overlapping virtual addresses with at least one write (`racecheck`).
    StateAliasRace {
        /// Node index where overlapping address ranges were observed.
        node_id: usize,
        /// First aliasing buffer.
        buf_a: BufferId,
        /// Second aliasing buffer.
        buf_b: BufferId,
        /// Number of overlapping bytes.
        overlap_bytes: u64,
        /// Diagnostic label describing the race.
        label: String,
    },
    /// Unpooled dynamic heap/device allocation attempted during active stream capture.
    DynamicAllocationDuringCapture {
        /// Requested allocation size in bytes.
        byte_len: usize,
        /// Requested tensor/buffer label.
        label: String,
    },
    /// Forbidden host synchronization or pageable HtoD copy attempted during active stream capture.
    HostSyncDuringCapture {
        /// Operation description that attempted to synchronize.
        reason: String,
    },
}

impl fmt::Display for SanitizerViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UseAfterFree {
                node_id,
                buffer_id,
                label,
            } => write!(
                f,
                "memcheck UseAfterFree at node {node_id}: buffer {:?} ({label}) was freed",
                buffer_id
            ),
            Self::PoisonedBufferAccess {
                node_id,
                buffer_id,
                label,
            } => write!(
                f,
                "memcheck PoisonedBufferAccess at node {node_id}: buffer {:?} ({label}) freed while referenced",
                buffer_id
            ),
            Self::GenerationMismatch {
                node_id,
                buffer_id,
                captured_gen,
                live_gen,
                label,
            } => write!(
                f,
                "memcheck GenerationMismatch at node {node_id}: buffer {:?} ({label}) captured={:?} live={:?}",
                buffer_id, captured_gen, live_gen
            ),
            Self::OutOfBounds {
                node_id,
                buffer_id,
                access_end,
                capacity_bytes,
                label,
            } => write!(
                f,
                "memcheck OutOfBounds at node {node_id}: buffer {:?} ({label}) end={access_end} > cap={capacity_bytes}",
                buffer_id
            ),
            Self::UninitializedRead {
                node_id,
                buffer_id,
                label,
            } => write!(
                f,
                "initcheck UninitializedRead at node {node_id}: buffer {:?} ({label}) never initialized",
                buffer_id
            ),
            Self::LeakedCaptureScratch {
                buffer_id,
                byte_len,
                label,
            } => write!(
                f,
                "memcheck LeakedCaptureScratch: buffer {:?} ({label}, {byte_len} bytes) survived end_capture",
                buffer_id
            ),
            Self::StateAliasRace {
                node_id,
                buf_a,
                buf_b,
                overlap_bytes,
                label,
            } => write!(
                f,
                "racecheck StateAliasRace at node {node_id}: {:?} aliases {:?} by {overlap_bytes} bytes ({label})",
                buf_a, buf_b
            ),
            Self::DynamicAllocationDuringCapture { byte_len, label } => write!(
                f,
                "capture DynamicAllocationForbidden: {byte_len} bytes ({label}) during active stream capture"
            ),
            Self::HostSyncDuringCapture { reason } => {
                write!(f, "capture HostSyncForbidden: {reason}")
            }
        }
    }
}

/// Summary report emitted after a clean [`GraphSanitizer`] replay verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizerReport {
    /// Total captured nodes checked.
    pub nodes_checked: usize,
    /// Total buffer bindings validated across all nodes.
    pub bindings_checked: usize,
    /// Number of violations detected (always `0` on `Ok`).
    pub violation_count: usize,
}

/// Metadata record for a single virtual allocation inside [`GraphMemoryPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolBlock {
    /// Buffer identifier.
    pub id: BufferId,
    /// Human-readable label.
    pub label: String,
    /// Current buffer generation (increments on reallocation).
    pub generation: BufferGeneration,
    /// Virtual GPU base address (256-byte aligned, monotonically allocated).
    pub virtual_addr: u64,
    /// Capacity in bytes.
    pub byte_len: usize,
    /// Current lifecycle state.
    pub state: PoolBlockState,
    /// Whether valid data has been written at least once (`initcheck`).
    pub initialized: bool,
    /// Number of live captured graphs pinning this buffer.
    pub graph_refcount: usize,
    /// Capture epoch in which this scratch block was allocated, if any.
    pub capture_epoch: Option<CaptureEpoch>,
}

/// Deterministic virtual GPU graph memory pool enforcing hard CUDA graph lifetime invariants.
#[derive(Debug, Clone)]
pub struct GraphMemoryPool {
    next_id: u64,
    next_virtual_addr: u64,
    blocks: BTreeMap<BufferId, PoolBlock>,
    in_capture: bool,
    current_epoch: CaptureEpoch,
    allow_private_pool_scratch: bool,
    total_persistent_copies: u64,
    total_persistent_bytes_copied: u64,
    live_private_graph_pool_bytes: usize,
    peak_private_graph_pool_bytes: usize,
}

impl Default for GraphMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphMemoryPool {
    /// Creates an empty virtual GPU graph memory pool starting at aligned base address `0x1000_0000`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            next_virtual_addr: 0x1000_0000,
            blocks: BTreeMap::new(),
            in_capture: false,
            current_epoch: CaptureEpoch(0),
            allow_private_pool_scratch: true,
            total_persistent_copies: 0,
            total_persistent_bytes_copied: 0,
            live_private_graph_pool_bytes: 0,
            peak_private_graph_pool_bytes: 0,
        }
    }

    /// Configures whether private-pool temporary scratch allocations are permitted during capture
    /// (must still be freed before `end_capture`). When `false`, ANY allocation during capture fails.
    pub fn set_allow_private_pool_scratch(&mut self, allow: bool) {
        self.allow_private_pool_scratch = allow;
    }

    /// Returns whether stream capture is currently active on this pool.
    #[must_use]
    pub const fn is_capturing(&self) -> bool {
        self.in_capture
    }

    /// Returns the current capture epoch.
    #[must_use]
    pub const fn current_epoch(&self) -> CaptureEpoch {
        self.current_epoch
    }

    /// Begins a stream capture session, incrementing the capture epoch.
    pub fn begin_capture(&mut self) -> Result<CaptureEpoch, SanitizerViolation> {
        if self.in_capture {
            return Err(SanitizerViolation::HostSyncDuringCapture {
                reason: "nested begin_capture while stream is already capturing".to_string(),
            });
        }
        self.current_epoch = CaptureEpoch(self.current_epoch.0 + 1);
        self.in_capture = true;
        Ok(self.current_epoch)
    }

    /// Ends the active stream capture session, failing closed if any capture-local scratch
    /// allocation from this epoch was not freed before `end_capture`.
    pub fn end_capture(&mut self) -> Result<CaptureEpoch, Vec<SanitizerViolation>> {
        let epoch = self.current_epoch;
        self.in_capture = false;
        let mut leaks = Vec::new();
        for block in self.blocks.values() {
            if block.state == PoolBlockState::ScratchAllocated && block.capture_epoch == Some(epoch)
            {
                leaks.push(SanitizerViolation::LeakedCaptureScratch {
                    buffer_id: block.id,
                    byte_len: block.byte_len,
                    label: block.label.clone(),
                });
            }
        }
        if leaks.is_empty() {
            Ok(epoch)
        } else {
            Err(leaks)
        }
    }

    /// Aborts an active capture and frees any capture-local scratch from the aborted epoch.
    pub fn abort_capture(&mut self) {
        let epoch = self.current_epoch;
        self.in_capture = false;
        for block in self.blocks.values_mut() {
            if block.state == PoolBlockState::ScratchAllocated && block.capture_epoch == Some(epoch)
            {
                block.state = PoolBlockState::Freed;
            }
        }
    }

    /// Checks whether a host synchronization or pageable HtoD copy is allowed right now.
    /// Always fails closed during active stream capture.
    pub fn check_host_sync(&self, reason: &str) -> Result<(), SanitizerViolation> {
        if self.in_capture {
            return Err(SanitizerViolation::HostSyncDuringCapture {
                reason: reason.to_string(),
            });
        }
        Ok(())
    }

    /// Allocates a fixed-address persistent static buffer outside of active stream capture.
    pub fn allocate_persistent(
        &mut self,
        label: impl Into<String>,
        byte_len: usize,
        zero_initialized: bool,
    ) -> Result<BufferId, SanitizerViolation> {
        let label = label.into();
        if self.in_capture {
            return Err(SanitizerViolation::DynamicAllocationDuringCapture { byte_len, label });
        }
        let id = BufferId(self.next_id);
        self.next_id += 1;
        let aligned = align_up_256(byte_len.max(1) as u64);
        let addr = self.next_virtual_addr;
        self.next_virtual_addr += aligned;

        self.blocks.insert(
            id,
            PoolBlock {
                id,
                label,
                generation: BufferGeneration(1),
                virtual_addr: addr,
                byte_len,
                state: PoolBlockState::PersistentStatic,
                initialized: zero_initialized,
                graph_refcount: 0,
                capture_epoch: None,
            },
        );
        Ok(id)
    }

    /// Allocates a capture-local private-pool scratch buffer. Fails closed if private-pool
    /// scratch is disabled or if called for an unpooled dynamic allocation during capture.
    pub fn allocate_capture_scratch(
        &mut self,
        label: impl Into<String>,
        byte_len: usize,
    ) -> Result<BufferId, SanitizerViolation> {
        let label = label.into();
        if self.in_capture && !self.allow_private_pool_scratch {
            return Err(SanitizerViolation::DynamicAllocationDuringCapture { byte_len, label });
        }
        let id = BufferId(self.next_id);
        self.next_id += 1;
        let aligned = align_up_256(byte_len.max(1) as u64);
        let addr = self.next_virtual_addr;
        self.next_virtual_addr += aligned;

        self.blocks.insert(
            id,
            PoolBlock {
                id,
                label,
                generation: BufferGeneration(1),
                virtual_addr: addr,
                byte_len,
                state: PoolBlockState::ScratchAllocated,
                initialized: true,
                graph_refcount: 0,
                capture_epoch: if self.in_capture {
                    Some(self.current_epoch)
                } else {
                    None
                },
            },
        );
        Ok(id)
    }

    /// Attempts an unpooled dynamic allocation — ALWAYS forbidden during active stream capture.
    pub fn allocate_unpooled_dynamic(
        &mut self,
        label: impl Into<String>,
        byte_len: usize,
    ) -> Result<BufferId, SanitizerViolation> {
        let label = label.into();
        if self.in_capture {
            return Err(SanitizerViolation::DynamicAllocationDuringCapture { byte_len, label });
        }
        self.allocate_persistent(label, byte_len, false)
    }

    /// Frees a buffer. If one or more live captured graphs still pin this buffer (`graph_refcount > 0`),
    /// the block transitions to [`PoolBlockState::Poisoned`] so any subsequent replay is caught as
    /// a hard use-after-free / poisoned-access sanitizer violation.
    pub fn free_buffer(&mut self, id: BufferId) -> bool {
        if let Some(block) = self.blocks.get_mut(&id) {
            if matches!(
                block.state,
                PoolBlockState::Freed | PoolBlockState::Poisoned
            ) {
                return false;
            }
            block.state = if block.graph_refcount > 0 {
                PoolBlockState::Poisoned
            } else {
                PoolBlockState::Freed
            };
            true
        } else {
            false
        }
    }

    /// Reallocates an existing buffer handle at a new virtual address and bumps `BufferGeneration`.
    /// Any previously captured graph referencing the old generation will fail sanitizer validation
    /// with [`SanitizerViolation::GenerationMismatch`].
    pub fn reallocate_buffer(
        &mut self,
        id: BufferId,
        new_byte_len: usize,
    ) -> Result<BufferGeneration, SanitizerViolation> {
        if self.in_capture {
            return Err(SanitizerViolation::DynamicAllocationDuringCapture {
                byte_len: new_byte_len,
                label: self
                    .blocks
                    .get(&id)
                    .map(|b| b.label.clone())
                    .unwrap_or_else(|| "unknown".to_string()),
            });
        }
        let aligned = align_up_256(new_byte_len.max(1) as u64);
        let new_addr = self.next_virtual_addr;
        self.next_virtual_addr += aligned;

        let block = self
            .blocks
            .get_mut(&id)
            .ok_or_else(|| SanitizerViolation::UseAfterFree {
                node_id: 0,
                buffer_id: id,
                label: "reallocate_missing".to_string(),
            })?;
        block.generation = BufferGeneration(block.generation.0 + 1);
        block.virtual_addr = new_addr;
        block.byte_len = new_byte_len;
        block.state = PoolBlockState::PersistentStatic;
        block.initialized = true;
        Ok(block.generation)
    }

    /// Copies `src_byte_len` bytes in-place into a pre-allocated [`PoolBlockState::PersistentStatic`]
    /// buffer without changing its virtual address or generation (mirroring `Var::set`).
    pub fn copy_into_persistent(
        &mut self,
        id: BufferId,
        src_byte_len: usize,
    ) -> Result<(), SanitizerViolation> {
        if self.in_capture {
            return Err(SanitizerViolation::HostSyncDuringCapture {
                reason: "host copy_into_persistent called during active stream capture".to_string(),
            });
        }
        let block = self
            .blocks
            .get_mut(&id)
            .ok_or_else(|| SanitizerViolation::UseAfterFree {
                node_id: 0,
                buffer_id: id,
                label: "copy_missing".to_string(),
            })?;
        match block.state {
            PoolBlockState::Freed => {
                return Err(SanitizerViolation::UseAfterFree {
                    node_id: 0,
                    buffer_id: id,
                    label: block.label.clone(),
                });
            }
            PoolBlockState::Poisoned => {
                return Err(SanitizerViolation::PoisonedBufferAccess {
                    node_id: 0,
                    buffer_id: id,
                    label: block.label.clone(),
                });
            }
            PoolBlockState::ScratchAllocated | PoolBlockState::PersistentStatic => {}
        }
        if src_byte_len > block.byte_len {
            return Err(SanitizerViolation::OutOfBounds {
                node_id: 0,
                buffer_id: id,
                access_end: src_byte_len,
                capacity_bytes: block.byte_len,
                label: block.label.clone(),
            });
        }
        block.initialized = true;
        self.total_persistent_copies += 1;
        self.total_persistent_bytes_copied += src_byte_len as u64;
        Ok(())
    }

    /// Creates a [`CapturedBufferRef`] pinned to the buffer's current live generation.
    pub fn make_ref(
        &self,
        id: BufferId,
        byte_offset: usize,
        byte_len: usize,
        access: BufferAccessKind,
    ) -> Result<CapturedBufferRef, SanitizerViolation> {
        let block = self
            .blocks
            .get(&id)
            .ok_or_else(|| SanitizerViolation::UseAfterFree {
                node_id: 0,
                buffer_id: id,
                label: "make_ref_missing".to_string(),
            })?;
        Ok(CapturedBufferRef {
            buffer_id: id,
            generation: block.generation,
            byte_offset,
            byte_len,
            access,
            label: block.label.clone(),
            virtual_addr_override: None,
        })
    }

    /// Increments the live graph reference count on `id`.
    pub fn retain_for_graph(&mut self, id: BufferId) {
        if let Some(block) = self.blocks.get_mut(&id) {
            block.graph_refcount += 1;
        }
    }

    /// Decrements the live graph reference count on `id`.
    pub fn release_from_graph(&mut self, id: BufferId) {
        if let Some(block) = self.blocks.get_mut(&id) {
            block.graph_refcount = block.graph_refcount.saturating_sub(1);
        }
    }

    /// Accounts for private graph pool bytes allocated when instantiating a captured graph.
    pub fn account_graph_pool_alloc(&mut self, bytes: usize) {
        self.live_private_graph_pool_bytes += bytes;
        if self.live_private_graph_pool_bytes > self.peak_private_graph_pool_bytes {
            self.peak_private_graph_pool_bytes = self.live_private_graph_pool_bytes;
        }
    }

    /// Releases private graph pool bytes when tearing down or evicting a captured graph.
    pub fn account_graph_pool_free(&mut self, bytes: usize) {
        self.live_private_graph_pool_bytes =
            self.live_private_graph_pool_bytes.saturating_sub(bytes);
    }

    /// Returns metadata for `id` if present.
    #[must_use]
    pub fn get(&self, id: BufferId) -> Option<&PoolBlock> {
        self.blocks.get(&id)
    }

    /// Total live persistent buffer bytes currently allocated.
    #[must_use]
    pub fn live_persistent_bytes(&self) -> usize {
        self.blocks
            .values()
            .filter(|b| b.state == PoolBlockState::PersistentStatic)
            .map(|b| b.byte_len)
            .sum()
    }

    /// Total live leaked bytes: any unfreed `ScratchAllocated`, `Poisoned`, or non-zero
    /// private graph pool bytes remaining after graph teardown.
    #[must_use]
    pub fn live_leaked_bytes(&self) -> usize {
        let block_leaks: usize = self
            .blocks
            .values()
            .filter(|b| {
                matches!(
                    b.state,
                    PoolBlockState::ScratchAllocated | PoolBlockState::Poisoned
                )
            })
            .map(|b| b.byte_len)
            .sum();
        block_leaks + self.live_private_graph_pool_bytes
    }

    /// Total live private graph pool bytes across currently cached graphs.
    #[must_use]
    pub const fn live_private_graph_pool_bytes(&self) -> usize {
        self.live_private_graph_pool_bytes
    }

    /// Peak private graph pool bytes observed across the lifetime of this pool.
    #[must_use]
    pub const fn peak_private_graph_pool_bytes(&self) -> usize {
        self.peak_private_graph_pool_bytes
    }

    /// Total in-place persistent buffer copies issued.
    #[must_use]
    pub const fn total_persistent_copies(&self) -> u64 {
        self.total_persistent_copies
    }

    /// Total bytes copied into persistent buffers.
    #[must_use]
    pub const fn total_persistent_bytes_copied(&self) -> u64 {
        self.total_persistent_bytes_copied
    }
}

/// Pre-allocated fixed-address I/O and metadata buffers for CUDA graph replay.
///
/// Directly mirrors Hanzo Engine's `CudaDecodeGraphMetadataBuffers` (including
/// `block_table_signature` skip semantics so unchanged block tables avoid redundant
/// device copies) and adds fixed-address buffers for linear-attention (`gdn_recurrent_state_ptrs`),
/// speculative verification (`spec_draft_tokens`), padded bucket masking (`lane_active_mask`),
/// and output logits (`output_logits`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentGraphBuffers {
    /// Padded batch bucket capacity (`B`).
    pub batch_bucket: usize,
    /// Padded speculative verification width capacity (`K`).
    pub verify_width_bucket: usize,
    /// Padded paged KV context bucket capacity in tokens.
    pub context_bucket: usize,
    /// Fixed buffer holding `[B * K]` input token IDs (`u32`).
    pub input_ids: BufferId,
    /// Fixed buffer holding `[B * K]` RoPE / sequence position IDs (`u32`).
    pub position_ids: BufferId,
    /// Fixed buffer holding `[B * K]` KV cache physical slot mappings (`i64`).
    pub slot_mapping: BufferId,
    /// Fixed buffer holding `[B, max_blocks]` paged KV block table indices (`i32`).
    pub block_tables: BufferId,
    /// Fixed buffer holding `[B]` live sequence context lengths (`i32`).
    pub context_lens: BufferId,
    /// Fixed buffer holding `[B * num_gdn_layers]` recurrent state base addresses (`u64`).
    pub gdn_recurrent_state_ptrs: BufferId,
    /// Fixed buffer holding `[B * K]` speculative draft token IDs (`u32`).
    pub spec_draft_tokens: BufferId,
    /// Fixed buffer holding `[B]` active-lane masks (`u8`: `1` = active, `0` = padded).
    pub lane_active_mask: BufferId,
    /// Fixed buffer holding `[B * K * vocab_size]` output logits (`f32`).
    pub output_logits: BufferId,
    /// Cached block-table signature used to skip redundant block-table copies on replay.
    pub block_table_signature: Option<Vec<u64>>,
    /// Number of steps where block-table copy was skipped due to matching signature.
    pub block_table_copy_skips: u64,
}

/// Result of an in-place persistent buffer update prior to graph replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentUpdateStats {
    /// Total bytes copied into fixed-address persistent buffers for this step.
    pub bytes_copied: usize,
    /// True if `block_tables` copy was skipped because `block_table_signature` matched.
    pub block_table_skipped: bool,
}

impl PersistentGraphBuffers {
    /// Allocates all persistent static buffers for a `(batch_bucket, verify_width_bucket, context_bucket)`
    /// graph entry inside `pool`.
    pub fn allocate(
        pool: &mut GraphMemoryPool,
        batch_bucket: usize,
        verify_width_bucket: usize,
        context_bucket: usize,
        block_size: usize,
        num_gdn_layers: usize,
        vocab_size: usize,
    ) -> Result<Self, SanitizerViolation> {
        let tokens = batch_bucket.max(1) * verify_width_bucket.max(1);
        let max_blocks = context_bucket.div_ceil(block_size.max(1)).max(1);

        let input_ids = pool.allocate_persistent("input_ids", tokens * 4, true)?;
        let position_ids = pool.allocate_persistent("position_ids", tokens * 4, true)?;
        let slot_mapping = pool.allocate_persistent("slot_mapping", tokens * 8, true)?;
        let block_tables =
            pool.allocate_persistent("block_tables", batch_bucket.max(1) * max_blocks * 4, true)?;
        let context_lens =
            pool.allocate_persistent("context_lens", batch_bucket.max(1) * 4, true)?;
        let gdn_recurrent_state_ptrs = pool.allocate_persistent(
            "gdn_recurrent_state_ptrs",
            batch_bucket.max(1) * num_gdn_layers.max(1) * 8,
            true,
        )?;
        let spec_draft_tokens = pool.allocate_persistent("spec_draft_tokens", tokens * 4, true)?;
        let lane_active_mask =
            pool.allocate_persistent("lane_active_mask", batch_bucket.max(1), true)?;
        let output_logits =
            pool.allocate_persistent("output_logits", tokens * vocab_size.max(1) * 4, true)?;

        Ok(Self {
            batch_bucket,
            verify_width_bucket,
            context_bucket,
            input_ids,
            position_ids,
            slot_mapping,
            block_tables,
            context_lens,
            gdn_recurrent_state_ptrs,
            spec_draft_tokens,
            lane_active_mask,
            output_logits,
            block_table_signature: None,
            block_table_copy_skips: 0,
        })
    }

    /// Returns all persistent buffer handles owned by this metadata bundle.
    #[must_use]
    pub fn all_buffer_ids(&self) -> [BufferId; 9] {
        [
            self.input_ids,
            self.position_ids,
            self.slot_mapping,
            self.block_tables,
            self.context_lens,
            self.gdn_recurrent_state_ptrs,
            self.spec_draft_tokens,
            self.lane_active_mask,
            self.output_logits,
        ]
    }

    /// Updates all per-step persistent buffers strictly in-place via `pool.copy_into_persistent`,
    /// skipping `block_tables` when `new_block_table_signature` matches the cached signature.
    pub fn update_in_place(
        &mut self,
        pool: &mut GraphMemoryPool,
        active_batch: usize,
        active_verify_width: usize,
        block_table_words: usize,
        num_gdn_layers: usize,
        new_block_table_signature: Option<&[u64]>,
    ) -> Result<PersistentUpdateStats, SanitizerViolation> {
        let tokens = active_batch.max(1) * active_verify_width.max(1);
        let mut bytes_copied = 0usize;

        let input_bytes = tokens * 4;
        pool.copy_into_persistent(self.input_ids, input_bytes)?;
        bytes_copied += input_bytes;

        let pos_bytes = tokens * 4;
        pool.copy_into_persistent(self.position_ids, pos_bytes)?;
        bytes_copied += pos_bytes;

        let slot_bytes = tokens * 8;
        pool.copy_into_persistent(self.slot_mapping, slot_bytes)?;
        bytes_copied += slot_bytes;

        let ctx_bytes = active_batch.max(1) * 4;
        pool.copy_into_persistent(self.context_lens, ctx_bytes)?;
        bytes_copied += ctx_bytes;

        let gdn_ptr_bytes = active_batch.max(1) * num_gdn_layers.max(1) * 8;
        pool.copy_into_persistent(self.gdn_recurrent_state_ptrs, gdn_ptr_bytes)?;
        bytes_copied += gdn_ptr_bytes;

        let draft_bytes = tokens * 4;
        pool.copy_into_persistent(self.spec_draft_tokens, draft_bytes)?;
        bytes_copied += draft_bytes;

        let mask_bytes = self.batch_bucket.max(1);
        pool.copy_into_persistent(self.lane_active_mask, mask_bytes)?;
        bytes_copied += mask_bytes;

        let signature_matches = match (&self.block_table_signature, new_block_table_signature) {
            (Some(prev), Some(curr)) => prev.as_slice() == curr,
            _ => false,
        };
        let block_table_skipped = if signature_matches {
            self.block_table_copy_skips += 1;
            true
        } else {
            let bt_bytes = block_table_words * 4;
            pool.copy_into_persistent(self.block_tables, bt_bytes)?;
            bytes_copied += bt_bytes;
            self.block_table_signature = new_block_table_signature.map(|s| s.to_vec());
            false
        };

        Ok(PersistentUpdateStats {
            bytes_copied,
            block_table_skipped,
        })
    }

    /// Releases all persistent static buffers owned by this bundle from `pool`.
    pub fn release(&self, pool: &mut GraphMemoryPool) {
        for id in self.all_buffer_ids() {
            pool.free_buffer(id);
        }
    }

    /// Total persistent buffer bytes owned by this bundle inside `pool`.
    #[must_use]
    pub fn total_bytes(&self, pool: &GraphMemoryPool) -> usize {
        self.all_buffer_ids()
            .iter()
            .filter_map(|id| pool.get(*id).map(|b| b.byte_len))
            .sum()
    }
}

/// Fail-closed sanitizer verifying every captured node's buffer references before replay.
#[derive(Debug, Default, Clone, Copy)]
pub struct GraphSanitizer;

impl GraphSanitizer {
    /// Validates every node's buffer bindings against `pool`.
    ///
    /// Checks:
    /// 1. Buffer exists and is neither `Freed` (`UseAfterFree`) nor `Poisoned` (`PoisonedBufferAccess`).
    /// 2. Captured `BufferGeneration` matches live generation (`GenerationMismatch`).
    /// 3. `byte_offset + byte_len <= capacity_bytes` (`OutOfBounds`).
    /// 4. Reads target initialized memory (`UninitializedRead`).
    /// 5. Distinct bindings inside the same node do not alias overlapping virtual address ranges
    ///    when at least one binding writes (`StateAliasRace`).
    pub fn validate_replay(
        pool: &GraphMemoryPool,
        node_bindings: &[(usize, &[CapturedBufferRef])],
    ) -> Result<SanitizerReport, Vec<SanitizerViolation>> {
        let mut violations = Vec::new();
        let mut total_bindings = 0usize;

        for &(node_id, refs) in node_bindings {
            // Resolved `(start, end, buffer_id, is_write, label)` for alias race detection.
            let mut ranges: Vec<(u64, u64, BufferId, bool, &str)> = Vec::with_capacity(refs.len());

            for binding in refs {
                total_bindings += 1;
                let Some(block) = pool.get(binding.buffer_id) else {
                    violations.push(SanitizerViolation::UseAfterFree {
                        node_id,
                        buffer_id: binding.buffer_id,
                        label: binding.label.clone(),
                    });
                    continue;
                };

                match block.state {
                    PoolBlockState::Freed => {
                        violations.push(SanitizerViolation::UseAfterFree {
                            node_id,
                            buffer_id: binding.buffer_id,
                            label: binding.label.clone(),
                        });
                        continue;
                    }
                    PoolBlockState::Poisoned => {
                        violations.push(SanitizerViolation::PoisonedBufferAccess {
                            node_id,
                            buffer_id: binding.buffer_id,
                            label: binding.label.clone(),
                        });
                        continue;
                    }
                    PoolBlockState::ScratchAllocated | PoolBlockState::PersistentStatic => {}
                }

                if binding.generation != block.generation {
                    violations.push(SanitizerViolation::GenerationMismatch {
                        node_id,
                        buffer_id: binding.buffer_id,
                        captured_gen: binding.generation,
                        live_gen: block.generation,
                        label: binding.label.clone(),
                    });
                }

                let access_end = binding.byte_offset.saturating_add(binding.byte_len);
                if access_end > block.byte_len {
                    violations.push(SanitizerViolation::OutOfBounds {
                        node_id,
                        buffer_id: binding.buffer_id,
                        access_end,
                        capacity_bytes: block.byte_len,
                        label: binding.label.clone(),
                    });
                }

                if binding.access.is_read() && !block.initialized {
                    violations.push(SanitizerViolation::UninitializedRead {
                        node_id,
                        buffer_id: binding.buffer_id,
                        label: binding.label.clone(),
                    });
                }

                let base = binding.virtual_addr_override.unwrap_or(block.virtual_addr);
                let start = base.saturating_add(binding.byte_offset as u64);
                let end = start.saturating_add(binding.byte_len as u64);
                ranges.push((
                    start,
                    end,
                    binding.buffer_id,
                    binding.access.is_write(),
                    binding.label.as_str(),
                ));
            }

            // Pairwise address-range alias check (racecheck) across distinct buffer IDs.
            for i in 0..ranges.len() {
                for j in (i + 1)..ranges.len() {
                    let (s_a, e_a, id_a, w_a, lbl_a) = ranges[i];
                    let (s_b, e_b, id_b, w_b, lbl_b) = ranges[j];
                    if id_a == id_b || !(w_a || w_b) {
                        continue;
                    }
                    let ov_start = s_a.max(s_b);
                    let ov_end = e_a.min(e_b);
                    if ov_start < ov_end {
                        violations.push(SanitizerViolation::StateAliasRace {
                            node_id,
                            buf_a: id_a,
                            buf_b: id_b,
                            overlap_bytes: ov_end - ov_start,
                            label: format!("{lbl_a} <-> {lbl_b}"),
                        });
                    }
                }
            }
        }

        if violations.is_empty() {
            Ok(SanitizerReport {
                nodes_checked: node_bindings.len(),
                bindings_checked: total_bindings,
                violation_count: 0,
            })
        } else {
            Err(violations)
        }
    }
}

const fn align_up_256(bytes: u64) -> u64 {
    (bytes + 255) & !255
}
