// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Incremental GPU checkpoint/restore engine (`hanzo-gcr`):
//! - 3 matched checkpointing modes in one unified engine:
//!   1. `FullNaive` (`full_naive_checkpoint`)
//!   2. `FullMutableOnly` (`full_mutable_only_checkpoint`)
//!   3. `GcrIncremental` (`gcr_incremental_delta`)
//! - Bounded delta chain compaction (`max_chain_depth`, `max_chain_delta_ratio`) + chunk GC
//! - Crash-safe atomic commit with dirty-bitmap rollback on I/O failure
//! - Parallel/pipelined verified restore via `std::thread::scope` (zero external crates)
//! - Explicit rejection of missing parents, cyclic chains, corrupted chunks/manifests, and cross-tenant restores.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::crypto::Digest;
use crate::error::{GcrError, GcrResult};
use crate::store::{
    decode_manifest_envelope, encode_manifest_envelope, validate_checkpoint_id, CheckpointStore,
    IoStats,
};
use crate::vram::{GpuMemorySpace, Namespace, PageRegion};

/// The three matched checkpoint strategies evaluated in a common harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CheckpointMode {
    /// Dumps the entire VRAM image (immutable weights + all clean & dirty mutable pages) on every pause.
    FullNaive = 0,
    /// Skips immutable read-only weights; dumps all mutable pages (clean + dirty) on every pause.
    FullMutableOnly = 1,
    /// Skips immutable weights and clean mutable pages; writes only dirty-page deltas with bounded compaction.
    GcrIncremental = 2,
}

impl CheckpointMode {
    /// Canonical benchmark name for this mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullNaive => "full_naive_checkpoint",
            Self::FullMutableOnly => "full_mutable_only_checkpoint",
            Self::GcrIncremental => "gcr_incremental_delta",
        }
    }

    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::FullNaive),
            1 => Some(Self::FullMutableOnly),
            2 => Some(Self::GcrIncremental),
            _ => None,
        }
    }
}

/// Configuration for [`CheckpointEngine`].
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Active checkpoint strategy.
    pub mode: CheckpointMode,
    /// Maximum allowed delta chain depth before automatic compaction (`0` compacts every pause).
    pub max_chain_depth: u32,
    /// Maximum ratio of cumulative historical delta entries in the active chain to `mutable_pages`
    /// before triggering automatic compaction (e.g., `2.0` compacts once cumulative deltas reach 2x mutable state).
    pub max_chain_delta_ratio: f64,
    /// Hard safety cap on parent-chain traversal during restore (prevents resource exhaustion).
    pub hard_max_restore_chain: u32,
    /// Allow lossless zero-RLE compression on sparse pages (only used when strictly smaller).
    pub allow_zero_rle: bool,
    /// Trigger unreferenced chunk garbage collection inside the namespace after compaction.
    pub gc_on_compaction: bool,
    /// Number of parallel worker threads used during restore chunk verification/loading.
    pub restore_workers: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            mode: CheckpointMode::GcrIncremental,
            max_chain_depth: 4,
            max_chain_delta_ratio: 2.0,
            hard_max_restore_chain: 256,
            allow_zero_rle: false,
            gc_on_compaction: true,
            restore_workers: 4,
        }
    }
}

/// Metadata entry for a single page chunk referenced by a [`CheckpointManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageChunkRef {
    /// Logical VRAM page index.
    pub page_id: u32,
    /// Region classification of this page.
    pub region: PageRegion,
    /// SHA-256 digest of the raw page bytes (content address in the namespace chunk store).
    pub chunk_digest: Digest,
    /// Raw uncompressed page length in bytes.
    pub raw_len: u32,
    /// Encoded wire length in bytes (header + payload).
    pub stored_len: u32,
    /// CRC32C of the encoded wire payload.
    pub crc32c: u32,
}

/// Self-verifying checkpoint manifest linking incremental delta chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointManifest {
    /// Unique checkpoint identifier within `(tenant_id, job_id)`.
    pub checkpoint_id: String,
    /// Owning tenant identifier.
    pub tenant_id: String,
    /// Owning job identifier.
    pub job_id: String,
    /// Cryptographic namespace tag binding this manifest to `(tenant_id, job_id)`.
    pub namespace_tag: Digest,
    /// Checkpoint strategy that produced this manifest.
    pub mode: CheckpointMode,
    /// Parent checkpoint ID for incremental delta chains (`None` for root/compacted/full snapshots).
    pub parent_checkpoint_id: Option<String>,
    /// Depth in the delta chain (`0` for root/compacted/full snapshot, `parent.chain_depth + 1` for deltas).
    pub chain_depth: u32,
    /// `true` if this manifest was created by automatic or explicit chain compaction.
    pub is_compacted: bool,
    /// Content-addressed digest of the shared immutable model weight base.
    pub model_base_digest: Digest,
    /// Logical page size in bytes.
    pub page_size: u32,
    /// Total pages in the job's VRAM address space.
    pub total_pages: u32,
    /// Number of immutable weight pages (`0..immutable_pages`).
    pub immutable_pages: u32,
    /// Number of mutable state pages (`immutable_pages..total_pages`).
    pub mutable_pages: u32,
    /// Total logical VRAM size of the job in bytes (`total_pages * page_size`).
    pub total_job_bytes: u64,
    /// Cumulative number of delta entries across the active chain rooted at this manifest.
    pub cumulative_chain_entries: u64,
    /// Full VRAM SHA-256 state digest at pause time for 100.0% bit-exact post-resume verification.
    pub state_digest: Digest,
    /// Page chunk references included in this manifest (dirty delta or full snapshot).
    pub entries: Vec<PageChunkRef>,
    /// SHA-256 digest of the canonical manifest body.
    pub manifest_digest: Digest,
}

impl CheckpointManifest {
    /// Serializes the canonical body and wraps it in a `GCRM` authenticated envelope.
    pub fn to_envelope(&self) -> (Vec<u8>, Digest) {
        let body = self.encode_body();
        encode_manifest_envelope(&body)
    }

    /// Parses and cryptographically verifies a `GCRM` manifest envelope.
    pub fn from_envelope(expected_id: &str, envelope: &[u8]) -> GcrResult<Self> {
        let (body, body_digest) = decode_manifest_envelope(expected_id, envelope)?;
        let mut cursor = ByteCursor::new(expected_id, body);

        let checkpoint_id = cursor.read_string()?;
        if checkpoint_id != expected_id {
            return Err(GcrError::CorruptedManifest {
                checkpoint_id: expected_id.to_string(),
                reason: format!(
                    "manifest inner id '{checkpoint_id}' does not match requested '{expected_id}'"
                ),
            });
        }
        let tenant_id = cursor.read_string()?;
        let job_id = cursor.read_string()?;
        let namespace_tag = cursor.read_digest()?;
        let mode_byte = cursor.read_u8()?;
        let mode =
            CheckpointMode::from_u8(mode_byte).ok_or_else(|| GcrError::CorruptedManifest {
                checkpoint_id: expected_id.to_string(),
                reason: format!("unknown checkpoint mode {mode_byte}"),
            })?;
        let has_parent = cursor.read_u8()?;
        let parent_checkpoint_id = match has_parent {
            0 => None,
            1 => Some(cursor.read_string()?),
            other => {
                return Err(GcrError::CorruptedManifest {
                    checkpoint_id: expected_id.to_string(),
                    reason: format!("invalid parent flag {other}"),
                })
            }
        };
        let chain_depth = cursor.read_u32()?;
        let is_compacted = match cursor.read_u8()? {
            0 => false,
            1 => true,
            other => {
                return Err(GcrError::CorruptedManifest {
                    checkpoint_id: expected_id.to_string(),
                    reason: format!("invalid compacted flag {other}"),
                })
            }
        };
        let model_base_digest = cursor.read_digest()?;
        let page_size = cursor.read_u32()?;
        let total_pages = cursor.read_u32()?;
        let immutable_pages = cursor.read_u32()?;
        let mutable_pages = cursor.read_u32()?;
        let total_job_bytes = cursor.read_u64()?;
        let cumulative_chain_entries = cursor.read_u64()?;
        let state_digest = cursor.read_digest()?;

        let entry_count = cursor.read_u32()? as usize;
        if entry_count > total_pages as usize {
            return Err(GcrError::CorruptedManifest {
                checkpoint_id: expected_id.to_string(),
                reason: format!("entry count {entry_count} exceeds total_pages {total_pages}"),
            });
        }
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let page_id = cursor.read_u32()?;
            let region_byte = cursor.read_u8()?;
            let region =
                PageRegion::from_u8(region_byte).ok_or_else(|| GcrError::CorruptedManifest {
                    checkpoint_id: expected_id.to_string(),
                    reason: format!("invalid page region {region_byte}"),
                })?;
            let chunk_digest = cursor.read_digest()?;
            let raw_len = cursor.read_u32()?;
            let stored_len = cursor.read_u32()?;
            let crc32c = cursor.read_u32()?;
            entries.push(PageChunkRef {
                page_id,
                region,
                chunk_digest,
                raw_len,
                stored_len,
                crc32c,
            });
        }
        cursor.finish()?;

        Ok(Self {
            checkpoint_id,
            tenant_id,
            job_id,
            namespace_tag,
            mode,
            parent_checkpoint_id,
            chain_depth,
            is_compacted,
            model_base_digest,
            page_size,
            total_pages,
            immutable_pages,
            mutable_pages,
            total_job_bytes,
            cumulative_chain_entries,
            state_digest,
            entries,
            manifest_digest: body_digest,
        })
    }

    fn encode_body(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256 + self.entries.len() * 52);
        write_string(&mut buf, &self.checkpoint_id);
        write_string(&mut buf, &self.tenant_id);
        write_string(&mut buf, &self.job_id);
        buf.extend_from_slice(self.namespace_tag.as_bytes());
        buf.push(self.mode as u8);
        match &self.parent_checkpoint_id {
            None => buf.push(0),
            Some(parent) => {
                buf.push(1);
                write_string(&mut buf, parent);
            }
        }
        buf.extend_from_slice(&self.chain_depth.to_le_bytes());
        buf.push(u8::from(self.is_compacted));
        buf.extend_from_slice(self.model_base_digest.as_bytes());
        buf.extend_from_slice(&self.page_size.to_le_bytes());
        buf.extend_from_slice(&self.total_pages.to_le_bytes());
        buf.extend_from_slice(&self.immutable_pages.to_le_bytes());
        buf.extend_from_slice(&self.mutable_pages.to_le_bytes());
        buf.extend_from_slice(&self.total_job_bytes.to_le_bytes());
        buf.extend_from_slice(&self.cumulative_chain_entries.to_le_bytes());
        buf.extend_from_slice(self.state_digest.as_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.page_id.to_le_bytes());
            buf.push(entry.region as u8);
            buf.extend_from_slice(entry.chunk_digest.as_bytes());
            buf.extend_from_slice(&entry.raw_len.to_le_bytes());
            buf.extend_from_slice(&entry.stored_len.to_le_bytes());
            buf.extend_from_slice(&entry.crc32c.to_le_bytes());
        }
        buf
    }
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

struct ByteCursor<'a> {
    checkpoint_id: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(checkpoint_id: &'a str, bytes: &'a [u8]) -> Self {
        Self {
            checkpoint_id,
            bytes,
            pos: 0,
        }
    }

    fn take(&mut self, n: usize) -> GcrResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GcrError::CorruptedManifest {
                checkpoint_id: self.checkpoint_id.to_string(),
                reason: "integer overflow while decoding manifest".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(GcrError::CorruptedManifest {
                checkpoint_id: self.checkpoint_id.to_string(),
                reason: "unexpected EOF while decoding manifest".to_string(),
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> GcrResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> GcrResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> GcrResult<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_digest(&mut self) -> GcrResult<Digest> {
        let b = self.take(32)?;
        let mut arr = [0u8; 32];
        arr.copy_from_slice(b);
        Ok(Digest(arr))
    }

    fn read_string(&mut self) -> GcrResult<String> {
        let len = self.read_u32()? as usize;
        if len > 1024 {
            return Err(GcrError::CorruptedManifest {
                checkpoint_id: self.checkpoint_id.to_string(),
                reason: format!("manifest string length {len} exceeds 1024"),
            });
        }
        let b = self.take(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| GcrError::CorruptedManifest {
            checkpoint_id: self.checkpoint_id.to_string(),
            reason: "invalid UTF-8 in manifest string".to_string(),
        })
    }

    fn finish(&self) -> GcrResult<()> {
        if self.pos != self.bytes.len() {
            return Err(GcrError::CorruptedManifest {
                checkpoint_id: self.checkpoint_id.to_string(),
                reason: format!(
                    "trailing unparsed bytes in manifest: {} remaining",
                    self.bytes.len() - self.pos
                ),
            });
        }
        Ok(())
    }
}

/// Metrics returned for a single pause/checkpoint operation.
#[derive(Debug, Clone)]
pub struct CheckpointReceipt {
    /// Committed manifest metadata.
    pub manifest: CheckpointManifest,
    /// Wall-clock pause time in milliseconds (measured via monotonic `Instant`).
    pub pause_latency_ms: f64,
    /// Exact I/O delta incurred by this pause.
    pub io_delta: IoStats,
    /// Number of dirty pages drained during this pause.
    pub dirty_pages_drained: usize,
    /// Number of page entries recorded in this manifest.
    pub manifest_entries: usize,
    /// Whether this pause triggered bounded chain compaction.
    pub compacted: bool,
    /// Number of unreferenced chunks deleted by post-compaction GC.
    pub gc_chunks_deleted: u64,
    /// Bytes reclaimed by post-compaction GC.
    pub gc_bytes_reclaimed: u64,
}

/// Metrics returned for a single resume/restore operation.
#[derive(Debug, Clone)]
pub struct RestoreReceipt {
    /// Restored checkpoint identifier.
    pub checkpoint_id: String,
    /// Wall-clock resume time in milliseconds (measured via monotonic `Instant`).
    pub resume_latency_ms: f64,
    /// Exact I/O delta incurred by this restore.
    pub io_delta: IoStats,
    /// Number of manifests traversed along the delta chain (`1` for full/compacted, `chain_depth + 1` for deltas).
    pub manifests_traversed: u32,
    /// Number of unique page chunks fetched and verified.
    pub chunks_restored: usize,
    /// Chain read-amplification ratio (`cumulative_chain_entries / mutable_pages` for naive replay,
    /// or `manifests_traversed` metadata hops).
    pub chain_entry_amplification: f64,
    /// `true` iff the restored VRAM SHA-256 digest matches the manifest digest bit-for-bit.
    pub bit_exact_verified: bool,
}

/// Unified GPU checkpoint/restore engine supporting all 3 matched strategies.
#[derive(Debug)]
pub struct CheckpointEngine<'a, S: CheckpointStore> {
    store: &'a S,
    config: CheckpointConfig,
}

impl<'a, S: CheckpointStore> CheckpointEngine<'a, S> {
    /// Creates a new checkpoint engine bound to `store` and `config`.
    pub const fn new(store: &'a S, config: CheckpointConfig) -> Self {
        Self { store, config }
    }

    /// Returns the engine configuration.
    pub const fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Pauses and checkpoints `vram` under `checkpoint_id`, optionally chaining onto `parent_id`.
    ///
    /// Atomically drains the dirty-page bitmap; if any storage operation fails before manifest commit,
    /// the drained dirty bits are restored so no mutations are silently lost.
    pub fn checkpoint(
        &self,
        vram: &GpuMemorySpace,
        checkpoint_id: impl Into<String>,
        parent_id: Option<&str>,
    ) -> GcrResult<CheckpointReceipt> {
        let checkpoint_id = checkpoint_id.into();
        validate_checkpoint_id(&checkpoint_id)?;
        if let Some(pid) = parent_id {
            validate_checkpoint_id(pid)?;
        }

        let start = Instant::now();
        let io_before = self.store.io_stats();
        let ns = vram.namespace();

        // Atomically drain dirty bits upfront; restore them on any error path.
        let drained_dirty = vram.dirty_bitmap().take_and_clear_all();
        let result = self.checkpoint_inner(vram, &checkpoint_id, parent_id, &drained_dirty);
        if result.is_err() {
            vram.dirty_bitmap().restore_bits(&drained_dirty);
        }
        let (manifest, compacted, gc_deleted, gc_reclaimed) = result?;

        let io_after = self.store.io_stats();
        let io_delta = io_after.delta_since(&io_before);
        let pause_latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Sanity-check namespace isolation on returned manifest.
        debug_assert_eq!(manifest.namespace_tag, ns.tag());

        Ok(CheckpointReceipt {
            manifest_entries: manifest.entries.len(),
            manifest,
            pause_latency_ms,
            io_delta,
            dirty_pages_drained: drained_dirty.len(),
            compacted,
            gc_chunks_deleted: gc_deleted,
            gc_bytes_reclaimed: gc_reclaimed,
        })
    }

    fn checkpoint_inner(
        &self,
        vram: &GpuMemorySpace,
        checkpoint_id: &str,
        parent_id: Option<&str>,
        drained_dirty: &[usize],
    ) -> GcrResult<(CheckpointManifest, bool, u64, u64)> {
        let ns = vram.namespace();
        let w = vram.immutable_page_count();
        let total = vram.total_page_count();
        let mutable_count = vram.mutable_page_count();

        match self.config.mode {
            CheckpointMode::FullNaive => {
                let mut entries = Vec::with_capacity(total);
                for page_id in 0..total {
                    let raw = vram.read_page(page_id)?;
                    let region = vram.page_region(page_id)?;
                    // Naive full checkpoint writes every page fresh without CAS dedup.
                    let info = self.store.put_chunk(ns, raw, false, false)?;
                    if page_id >= w {
                        vram.cache_mutable_leaf_digest(page_id, info.digest);
                    }
                    entries.push(PageChunkRef {
                        page_id: page_id as u32,
                        region,
                        chunk_digest: info.digest,
                        raw_len: info.raw_len,
                        stored_len: info.wire_len,
                        crc32c: info.payload_crc32c,
                    });
                }
                let manifest = self.commit_manifest(
                    vram,
                    checkpoint_id,
                    None,
                    0,
                    false,
                    entries.len() as u64,
                    entries,
                )?;
                Ok((manifest, false, 0, 0))
            }
            CheckpointMode::FullMutableOnly => {
                let mut entries = Vec::with_capacity(mutable_count);
                for page_id in w..total {
                    let raw = vram.read_page(page_id)?;
                    let region = vram.page_region(page_id)?;
                    // Full mutable snapshot skips weights but writes every mutable page fresh.
                    let info = self
                        .store
                        .put_chunk(ns, raw, self.config.allow_zero_rle, false)?;
                    vram.cache_mutable_leaf_digest(page_id, info.digest);
                    entries.push(PageChunkRef {
                        page_id: page_id as u32,
                        region,
                        chunk_digest: info.digest,
                        raw_len: info.raw_len,
                        stored_len: info.wire_len,
                        crc32c: info.payload_crc32c,
                    });
                }
                let manifest = self.commit_manifest(
                    vram,
                    checkpoint_id,
                    None,
                    0,
                    false,
                    entries.len() as u64,
                    entries,
                )?;
                Ok((manifest, false, 0, 0))
            }
            CheckpointMode::GcrIncremental => {
                let parent_manifest = match parent_id {
                    None => None,
                    Some(pid) => Some(self.load_and_validate_manifest(ns, pid, vram)?),
                };

                let Some(parent) = parent_manifest else {
                    // Base mutable checkpoint (depth 0): persist all mutable pages once.
                    let mut entries = Vec::with_capacity(mutable_count);
                    for page_id in w..total {
                        let raw = vram.read_page(page_id)?;
                        let region = vram.page_region(page_id)?;
                        let info =
                            self.store
                                .put_chunk(ns, raw, self.config.allow_zero_rle, true)?;
                        vram.cache_mutable_leaf_digest(page_id, info.digest);
                        entries.push(PageChunkRef {
                            page_id: page_id as u32,
                            region,
                            chunk_digest: info.digest,
                            raw_len: info.raw_len,
                            stored_len: info.wire_len,
                            crc32c: info.payload_crc32c,
                        });
                    }
                    let manifest = self.commit_manifest(
                        vram,
                        checkpoint_id,
                        None,
                        0,
                        false,
                        entries.len() as u64,
                        entries,
                    )?;
                    return Ok((manifest, false, 0, 0));
                };

                // Filter drained dirty pages to mutable pages only (weights are immutable).
                let dirty_mutable: Vec<usize> = drained_dirty
                    .iter()
                    .copied()
                    .filter(|&pid| pid >= w && pid < total)
                    .collect();

                let next_depth = parent.chain_depth.saturating_add(1);
                let next_cumulative_entries = parent
                    .cumulative_chain_entries
                    .saturating_add(dirty_mutable.len() as u64);
                let next_delta_ratio = if mutable_count > 0 {
                    next_cumulative_entries as f64 / mutable_count as f64
                } else {
                    1.0
                };

                let should_compact = next_depth > self.config.max_chain_depth
                    || (self.config.max_chain_delta_ratio.is_finite()
                        && next_delta_ratio > self.config.max_chain_delta_ratio);

                if should_compact {
                    // Compact into a depth-0 self-contained mutable manifest (`parent = None`).
                    // Unchanged mutable pages already exist in the namespace CAS store, so we
                    // only physically write newly dirtied pages while building a complete flat
                    // manifest — bounding restore to 1 manifest hop and enabling chunk GC!
                    let parent_chain = self.walk_manifest_chain(ns, &parent.checkpoint_id, vram)?;
                    let mut resolved = resolve_newest_mutable_entries(&parent_chain, w, total)?;

                    for &page_id in &dirty_mutable {
                        let raw = vram.read_page(page_id)?;
                        let region = vram.page_region(page_id)?;
                        let info =
                            self.store
                                .put_chunk(ns, raw, self.config.allow_zero_rle, true)?;
                        vram.cache_mutable_leaf_digest(page_id, info.digest);
                        resolved.insert(
                            page_id,
                            PageChunkRef {
                                page_id: page_id as u32,
                                region,
                                chunk_digest: info.digest,
                                raw_len: info.raw_len,
                                stored_len: info.wire_len,
                                crc32c: info.payload_crc32c,
                            },
                        );
                    }

                    let mut entries = Vec::with_capacity(mutable_count);
                    let mut live_digests = HashSet::with_capacity(mutable_count);
                    for page_id in w..total {
                        let entry = match resolved.remove(&page_id) {
                            Some(e) => e,
                            None => {
                                let raw = vram.read_page(page_id)?;
                                let region = vram.page_region(page_id)?;
                                let info = self.store.put_chunk(
                                    ns,
                                    raw,
                                    self.config.allow_zero_rle,
                                    true,
                                )?;
                                vram.cache_mutable_leaf_digest(page_id, info.digest);
                                PageChunkRef {
                                    page_id: page_id as u32,
                                    region,
                                    chunk_digest: info.digest,
                                    raw_len: info.raw_len,
                                    stored_len: info.wire_len,
                                    crc32c: info.payload_crc32c,
                                }
                            }
                        };
                        live_digests.insert(entry.chunk_digest);
                        entries.push(entry);
                    }

                    let manifest = self.commit_manifest(
                        vram,
                        checkpoint_id,
                        None,
                        0,
                        true,
                        entries.len() as u64,
                        entries,
                    )?;

                    let (gc_del, gc_bytes) = if self.config.gc_on_compaction {
                        self.store.gc_unreferenced_chunks(ns, &live_digests)?
                    } else {
                        (0, 0)
                    };

                    Ok((manifest, true, gc_del, gc_bytes))
                } else {
                    // Write ONLY dirty mutable page deltas and link to parent_checkpoint_id.
                    let mut entries = Vec::with_capacity(dirty_mutable.len());
                    for &page_id in &dirty_mutable {
                        let raw = vram.read_page(page_id)?;
                        let region = vram.page_region(page_id)?;
                        let info =
                            self.store
                                .put_chunk(ns, raw, self.config.allow_zero_rle, true)?;
                        vram.cache_mutable_leaf_digest(page_id, info.digest);
                        entries.push(PageChunkRef {
                            page_id: page_id as u32,
                            region,
                            chunk_digest: info.digest,
                            raw_len: info.raw_len,
                            stored_len: info.wire_len,
                            crc32c: info.payload_crc32c,
                        });
                    }
                    let manifest = self.commit_manifest(
                        vram,
                        checkpoint_id,
                        Some(parent.checkpoint_id),
                        next_depth,
                        false,
                        next_cumulative_entries,
                        entries,
                    )?;
                    Ok((manifest, false, 0, 0))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_manifest(
        &self,
        vram: &GpuMemorySpace,
        checkpoint_id: &str,
        parent_checkpoint_id: Option<String>,
        chain_depth: u32,
        is_compacted: bool,
        cumulative_chain_entries: u64,
        entries: Vec<PageChunkRef>,
    ) -> GcrResult<CheckpointManifest> {
        let ns = vram.namespace();
        let mut manifest = CheckpointManifest {
            checkpoint_id: checkpoint_id.to_string(),
            tenant_id: ns.tenant_id().to_string(),
            job_id: ns.job_id().to_string(),
            namespace_tag: ns.tag(),
            mode: self.config.mode,
            parent_checkpoint_id,
            chain_depth,
            is_compacted,
            model_base_digest: vram.model_base_digest(),
            page_size: vram.page_size() as u32,
            total_pages: vram.total_page_count() as u32,
            immutable_pages: vram.immutable_page_count() as u32,
            mutable_pages: vram.mutable_page_count() as u32,
            total_job_bytes: vram.total_bytes(),
            cumulative_chain_entries,
            state_digest: vram.state_digest(),
            entries,
            manifest_digest: Digest::ZERO,
        };
        let (envelope, body_digest) = manifest.to_envelope();
        manifest.manifest_digest = body_digest;
        self.store.put_manifest(ns, checkpoint_id, &envelope)?;
        Ok(manifest)
    }

    /// Restores `checkpoint_id` into `target` with strict tenant/job isolation, cycle detection,
    /// CRC32C + SHA-256 verification on every page, parallel chunk loading, and transactional commit.
    ///
    /// If any check fails, `target` is left completely unmodified.
    pub fn restore(
        &self,
        target: &mut GpuMemorySpace,
        checkpoint_id: &str,
    ) -> GcrResult<RestoreReceipt> {
        validate_checkpoint_id(checkpoint_id)?;
        let start = Instant::now();
        let io_before = self.store.io_stats();
        let ns = target.namespace().clone();
        let w = target.immutable_page_count();
        let total = target.total_page_count();
        let mutable_count = target.mutable_page_count();
        let page_size = target.page_size();

        let chain = self.walk_manifest_chain(&ns, checkpoint_id, target)?;
        let tip = &chain[0];
        let manifests_traversed = chain.len() as u32;

        // If FullNaive, verify all immutable weight chunks are also readable & match the mounted weights.
        let mut pages_to_fetch: Vec<(usize, PageChunkRef)> = Vec::new();
        if tip.mode == CheckpointMode::FullNaive {
            let mut by_page: HashMap<usize, PageChunkRef> =
                HashMap::with_capacity(tip.entries.len());
            for entry in &tip.entries {
                by_page.insert(entry.page_id as usize, entry.clone());
            }
            for page_id in 0..total {
                let entry =
                    by_page
                        .remove(&page_id)
                        .ok_or_else(|| GcrError::CorruptedManifest {
                            checkpoint_id: tip.checkpoint_id.clone(),
                            reason: format!("full_naive manifest missing page {page_id}"),
                        })?;
                pages_to_fetch.push((page_id, entry));
            }
        } else {
            let mut resolved = resolve_newest_mutable_entries(&chain, w, total)?;
            for page_id in w..total {
                let entry =
                    resolved
                        .remove(&page_id)
                        .ok_or_else(|| GcrError::CorruptedManifest {
                            checkpoint_id: tip.checkpoint_id.clone(),
                            reason: format!(
                                "delta chain rooted at {} missing mutable page {page_id}",
                                tip.checkpoint_id
                            ),
                        })?;
                pages_to_fetch.push((page_id, entry));
            }
        }

        let chunks_restored = pages_to_fetch.len();
        let fetched = self.fetch_pages_parallel(&ns, &pages_to_fetch, page_size)?;

        // Stage mutable pages & their verified SHA-256 digests transactionally.
        let mut staged_mutable = vec![Vec::new(); mutable_count];
        let mut staged_digests = vec![Digest::ZERO; mutable_count];
        for (page_id, digest, bytes) in fetched {
            if page_id < w {
                let expected_weight = target.read_page(page_id)?;
                if bytes.as_slice() != expected_weight {
                    return Err(GcrError::LayoutMismatch {
                        reason: format!(
                            "restored weight page {page_id} diverges from mounted model weights"
                        ),
                    });
                }
            } else {
                staged_mutable[page_id - w] = bytes;
                staged_digests[page_id - w] = digest;
            }
        }

        // Verify full post-restore VRAM Merkle root BEFORE mutating `target`.
        let staged_digest = target.compute_merkle_root_from_leaves(&staged_digests);
        if !staged_digest.ct_eq(&tip.state_digest) {
            return Err(GcrError::StateDigestMismatch {
                expected: tip.state_digest.to_hex(),
                actual: staged_digest.to_hex(),
            });
        }

        target.commit_restored_mutable_pages(staged_mutable, staged_digests)?;
        let post_digest = target.state_digest();
        let bit_exact_verified = post_digest.ct_eq(&tip.state_digest);

        let io_after = self.store.io_stats();
        let io_delta = io_after.delta_since(&io_before);
        let resume_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let chain_entry_amplification = if mutable_count > 0 {
            tip.cumulative_chain_entries as f64 / mutable_count as f64
        } else {
            1.0
        };

        Ok(RestoreReceipt {
            checkpoint_id: checkpoint_id.to_string(),
            resume_latency_ms,
            io_delta,
            manifests_traversed,
            chunks_restored,
            chain_entry_amplification,
            bit_exact_verified,
        })
    }

    fn fetch_pages_parallel(
        &self,
        ns: &Namespace,
        tasks: &[(usize, PageChunkRef)],
        page_size: usize,
    ) -> GcrResult<Vec<(usize, Digest, Vec<u8>)>> {
        let workers = self.config.restore_workers.max(1);
        if workers == 1 || tasks.len() < 8 {
            let mut out = Vec::with_capacity(tasks.len());
            for (page_id, entry) in tasks {
                let bytes = self.store.get_chunk(ns, &entry.chunk_digest, page_size)?;
                out.push((*page_id, entry.chunk_digest, bytes));
            }
            return Ok(out);
        }

        let chunk_size = tasks.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for slice in tasks.chunks(chunk_size) {
                let store = self.store;
                handles.push(
                    scope.spawn(move || -> GcrResult<Vec<(usize, Digest, Vec<u8>)>> {
                        let mut local = Vec::with_capacity(slice.len());
                        for (page_id, entry) in slice {
                            let bytes = store.get_chunk(ns, &entry.chunk_digest, page_size)?;
                            local.push((*page_id, entry.chunk_digest, bytes));
                        }
                        Ok(local)
                    }),
                );
            }

            let mut combined = Vec::with_capacity(tasks.len());
            for handle in handles {
                let part = handle.join().map_err(|_| {
                    GcrError::Io("restore worker thread panicked unexpectedly".to_string())
                })??;
                combined.extend(part);
            }
            Ok(combined)
        })
    }

    /// Loads and validates a manifest chain from tip (`checkpoint_id`) to root (`depth == 0`),
    /// explicitly detecting missing parents, cyclic references, depth mismatches, and tenant leaks.
    fn walk_manifest_chain(
        &self,
        ns: &Namespace,
        tip_id: &str,
        vram: &GpuMemorySpace,
    ) -> GcrResult<Vec<CheckpointManifest>> {
        let mut chain: Vec<CheckpointManifest> = Vec::new();
        let mut visited = HashSet::new();
        let mut current_id = tip_id.to_string();
        let mut expected_depth: Option<u32> = None;

        loop {
            if !visited.insert(current_id.clone()) {
                return Err(GcrError::CyclicCheckpointChain {
                    checkpoint_id: current_id,
                });
            }
            if chain.len() as u32 > self.config.hard_max_restore_chain {
                return Err(GcrError::ChainDepthExceeded {
                    depth: chain.len() as u32,
                    max_depth: self.config.hard_max_restore_chain,
                });
            }

            let manifest = match self.load_and_validate_manifest(ns, &current_id, vram) {
                Ok(m) => m,
                Err(GcrError::CheckpointNotFound(_)) if !chain.is_empty() => {
                    let child_id = chain.last().expect("non-empty").checkpoint_id.clone();
                    return Err(GcrError::MissingParentCheckpoint {
                        checkpoint_id: child_id,
                        parent_id: current_id,
                    });
                }
                Err(e) => return Err(e),
            };

            if let Some(exp) = expected_depth {
                if manifest.chain_depth != exp {
                    return Err(GcrError::CorruptedManifest {
                        checkpoint_id: manifest.checkpoint_id,
                        reason: format!(
                            "chain depth discontinuity: expected {exp}, got {}",
                            manifest.chain_depth
                        ),
                    });
                }
            }

            let next_parent = manifest.parent_checkpoint_id.clone();
            let depth = manifest.chain_depth;
            chain.push(manifest);

            match next_parent {
                None => {
                    if depth != 0 {
                        let cid = chain.last().expect("non-empty").checkpoint_id.clone();
                        return Err(GcrError::CorruptedManifest {
                            checkpoint_id: cid,
                            reason: format!(
                                "root manifest has non-zero chain_depth {depth} without parent"
                            ),
                        });
                    }
                    break;
                }
                Some(pid) => {
                    if depth == 0 {
                        let cid = chain.last().expect("non-empty").checkpoint_id.clone();
                        return Err(GcrError::CorruptedManifest {
                            checkpoint_id: cid,
                            reason: "chain_depth 0 manifest must not declare a parent".to_string(),
                        });
                    }
                    expected_depth = Some(depth - 1);
                    current_id = pid;
                }
            }
        }

        Ok(chain)
    }

    fn load_and_validate_manifest(
        &self,
        ns: &Namespace,
        checkpoint_id: &str,
        vram: &GpuMemorySpace,
    ) -> GcrResult<CheckpointManifest> {
        let raw = self.store.get_manifest(ns, checkpoint_id)?;
        let manifest = CheckpointManifest::from_envelope(checkpoint_id, &raw)?;

        // Strict tenant/job isolation check (0.0% cross-tenant or cross-job restore).
        if manifest.tenant_id != ns.tenant_id()
            || manifest.job_id != ns.job_id()
            || !manifest.namespace_tag.ct_eq(&ns.tag())
        {
            return Err(GcrError::TenantMismatch {
                expected_tenant: ns.tenant_id().to_string(),
                expected_job: ns.job_id().to_string(),
                actual_tenant: manifest.tenant_id,
                actual_job: manifest.job_id,
            });
        }

        // Strict VRAM layout & shared weight base compatibility check.
        if manifest.page_size as usize != vram.page_size()
            || manifest.total_pages as usize != vram.total_page_count()
            || manifest.immutable_pages as usize != vram.immutable_page_count()
            || manifest.mutable_pages as usize != vram.mutable_page_count()
        {
            return Err(GcrError::LayoutMismatch {
                reason: format!(
                    "manifest geometry (page_size={}, total={}, imm={}, mut={}) != target ({}, {}, {}, {})",
                    manifest.page_size,
                    manifest.total_pages,
                    manifest.immutable_pages,
                    manifest.mutable_pages,
                    vram.page_size(),
                    vram.total_page_count(),
                    vram.immutable_page_count(),
                    vram.mutable_page_count(),
                ),
            });
        }
        if !manifest.model_base_digest.ct_eq(&vram.model_base_digest()) {
            return Err(GcrError::LayoutMismatch {
                reason: "manifest model_base_digest does not match mounted immutable weights"
                    .to_string(),
            });
        }

        Ok(manifest)
    }
}

fn resolve_newest_mutable_entries(
    chain_tip_to_root: &[CheckpointManifest],
    immutable_pages: usize,
    total_pages: usize,
) -> GcrResult<HashMap<usize, PageChunkRef>> {
    let mut resolved = HashMap::with_capacity(total_pages.saturating_sub(immutable_pages));
    for manifest in chain_tip_to_root {
        for entry in &manifest.entries {
            let pid = entry.page_id as usize;
            if pid < immutable_pages || pid >= total_pages {
                return Err(GcrError::CorruptedManifest {
                    checkpoint_id: manifest.checkpoint_id.clone(),
                    reason: format!(
                        "mutable delta references out-of-range page {pid} (mutable range {immutable_pages}..{total_pages})"
                    ),
                });
            }
            resolved.entry(pid).or_insert_with(|| entry.clone());
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{FsCheckpointStore, MemoryCheckpointStore};
    use crate::vram::{PageSize, SharedWeightBase};

    fn make_weights(page_size: usize, count: usize) -> SharedWeightBase {
        let mut pages = Vec::with_capacity(count);
        for i in 0..count {
            let mut p = vec![0u8; page_size];
            for (j, b) in p.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(31) ^ j.wrapping_mul(17)) & 0xff) as u8;
            }
            pages.push(p);
        }
        SharedWeightBase::new(PageSize::Custom(page_size), pages).unwrap()
    }

    fn make_vram(
        tenant: &str,
        job: &str,
        weights: &SharedWeightBase,
        mutable_pages: usize,
    ) -> GpuMemorySpace {
        let ns = Namespace::new(tenant, job).unwrap();
        let regions = vec![PageRegion::KvCache; mutable_pages];
        GpuMemorySpace::new(ns, weights.clone(), regions).unwrap()
    }

    #[test]
    fn multi_hop_incremental_chain_and_compaction_are_bit_exact() {
        let store = MemoryCheckpointStore::new();
        let weights = make_weights(512, 8);
        let mut vram = make_vram("tenant-alpha", "agent-01", &weights, 16);

        let engine = CheckpointEngine::new(
            &store,
            CheckpointConfig {
                mode: CheckpointMode::GcrIncremental,
                max_chain_depth: 3,
                max_chain_delta_ratio: 10.0,
                restore_workers: 4,
                ..CheckpointConfig::default()
            },
        );

        let mut last_id: Option<String> = None;
        let mut expected_digests = Vec::new();

        for step in 0..8usize {
            // Dirty 2 pages per step.
            let p1 = 8 + (step * 2) % 16;
            let p2 = 8 + (step * 2 + 1) % 16;
            vram.overwrite_page(p1, &vec![(step as u8).wrapping_add(11); 512])
                .unwrap();
            vram.write_page_range(p2, 64, &[0xA5, step as u8, 0x5A, 0xCC])
                .unwrap();

            let cid = format!("ckpt-{step}");
            let receipt = engine.checkpoint(&vram, &cid, last_id.as_deref()).unwrap();
            if step == 0 {
                assert_eq!(receipt.manifest.chain_depth, 0);
                assert!(!receipt.compacted);
            } else if step == 4 {
                // Depth would have been 4 > max_chain_depth (3), so compaction triggers!
                assert_eq!(receipt.manifest.chain_depth, 0);
                assert!(receipt.compacted);
            } else {
                assert!(!receipt.compacted);
                assert_eq!(receipt.dirty_pages_drained, 2);
                assert_eq!(receipt.manifest_entries, 2);
            }
            expected_digests.push((cid.clone(), vram.state_digest()));
            last_id = Some(cid);
        }

        // Restore the latest compacted chain tip into a cold-boot GPU memory space.
        let (final_id, final_digest) = expected_digests.last().unwrap();
        let mut cold = vram.cold_boot_clone_layout().unwrap();
        assert_ne!(cold.state_digest(), *final_digest);
        let restored = engine.restore(&mut cold, final_id).unwrap();
        assert!(restored.bit_exact_verified);
        assert_eq!(cold.state_digest(), *final_digest);
        assert_eq!(cold.dirty_bitmap().dirty_count(), 0);
    }

    #[test]
    fn corruption_truncation_missing_parent_cycle_and_cross_tenant_are_rejected() {
        let store = MemoryCheckpointStore::new();
        let weights = make_weights(512, 4);
        let mut vram = make_vram("tenant-a", "job-1", &weights, 8);
        let engine = CheckpointEngine::new(&store, CheckpointConfig::default());

        vram.overwrite_page(4, &vec![0x11; 512]).unwrap();
        let r0 = engine.checkpoint(&vram, "c0", None).unwrap();
        vram.overwrite_page(5, &vec![0x22; 512]).unwrap();
        let r1 = engine.checkpoint(&vram, "c1", Some("c0")).unwrap();

        // 1. Cross-tenant restore MUST fail (0% leak).
        let mut other_tenant = make_vram("tenant-b", "job-1", &weights, 8);
        assert!(engine.restore(&mut other_tenant, "c1").is_err());

        // 2. Forged tenant manifest placed in tenant-b's namespace MUST fail TenantMismatch.
        let raw_c1 = store.get_manifest(vram.namespace(), "c1").unwrap();
        store.inject_raw_manifest(other_tenant.namespace(), "c1", raw_c1);
        assert!(matches!(
            engine.restore(&mut other_tenant, "c1"),
            Err(GcrError::TenantMismatch { .. })
        ));

        // 3. Cross-job (same tenant, different job) MUST fail.
        let mut other_job = make_vram("tenant-a", "job-2", &weights, 8);
        assert!(engine.restore(&mut other_job, "c1").is_err());

        // 4. Bit-flipped chunk MUST fail CorruptedChunk and leave target untouched.
        let bad_digest = r1.manifest.entries[0].chunk_digest;
        let orig_chunk = {
            let mut cold = vram.cold_boot_clone_layout().unwrap();
            assert!(engine.restore(&mut cold, "c1").is_ok());
            // Re-encode to grab valid bytes for backup.
            crate::store::encode_chunk_frame(vram.read_page(5).unwrap(), false).0
        };
        let mut corrupted = orig_chunk.clone();
        let last_idx = corrupted.len() - 1;
        corrupted[last_idx] ^= 0x80;
        store.inject_raw_chunk(vram.namespace(), bad_digest, corrupted);
        let mut target = vram.cold_boot_clone_layout().unwrap();
        let before_digest = target.state_digest();
        assert!(matches!(
            engine.restore(&mut target, "c1"),
            Err(GcrError::CorruptedChunk { .. })
        ));
        assert_eq!(target.state_digest(), before_digest);

        // 5. Truncated chunk MUST fail CorruptedChunk.
        store.inject_raw_chunk(vram.namespace(), bad_digest, orig_chunk[..20].to_vec());
        assert!(matches!(
            engine.restore(&mut target, "c1"),
            Err(GcrError::CorruptedChunk { .. })
        ));
        // Restore good chunk.
        store.inject_raw_chunk(vram.namespace(), bad_digest, orig_chunk);

        // 6. Cyclic parent chain MUST fail CyclicCheckpointChain.
        let mut cyc0 = r0.manifest.clone();
        cyc0.parent_checkpoint_id = Some("c1".to_string());
        cyc0.chain_depth = 2;
        let (cyc0_env, _) = cyc0.to_envelope();
        store.inject_raw_manifest(vram.namespace(), "c0", cyc0_env);
        let mut cyc1 = r1.manifest.clone();
        cyc1.parent_checkpoint_id = Some("c0".to_string());
        cyc1.chain_depth = 3;
        let (cyc1_env, _) = cyc1.to_envelope();
        store.inject_raw_manifest(vram.namespace(), "c1", cyc1_env);
        assert!(matches!(
            engine.restore(&mut target, "c1"),
            Err(GcrError::CyclicCheckpointChain { .. })
        ));

        // 7. Missing parent MUST fail MissingParentCheckpoint.
        let (orig_c1_env, _) = r1.manifest.to_envelope();
        store.inject_raw_manifest(vram.namespace(), "c1", orig_c1_env);
        store.remove_manifest(vram.namespace(), "c0");
        assert!(matches!(
            engine.restore(&mut target, "c1"),
            Err(GcrError::MissingParentCheckpoint { .. })
        ));
    }

    #[test]
    fn fs_store_crash_safe_atomic_commit_and_all_three_modes_match() {
        let dir = std::env::temp_dir().join(format!(
            "hanzo_gcr_unit_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = FsCheckpointStore::new(&dir, true).unwrap();
        let weights = make_weights(4096, 8);

        for (idx, mode) in [
            CheckpointMode::FullNaive,
            CheckpointMode::FullMutableOnly,
            CheckpointMode::GcrIncremental,
        ]
        .iter()
        .enumerate()
        {
            let mut vram = make_vram("org-hanzo", &format!("mode-{idx}"), &weights, 8);
            let engine = CheckpointEngine::new(
                &store,
                CheckpointConfig {
                    mode: *mode,
                    max_chain_depth: 4,
                    restore_workers: 2,
                    ..CheckpointConfig::default()
                },
            );

            vram.overwrite_page(8, &vec![0x33; 4096]).unwrap();
            let r0 = engine.checkpoint(&vram, "p0", None).unwrap();
            vram.overwrite_page(9, &vec![0x44; 4096]).unwrap();
            let r1 = engine.checkpoint(&vram, "p1", Some("p0")).unwrap();

            assert!(r0.io_delta.fsync_calls >= 2);
            if *mode == CheckpointMode::GcrIncremental {
                assert_eq!(r1.io_delta.chunks_written, 1);
            } else if *mode == CheckpointMode::FullMutableOnly {
                assert_eq!(r1.io_delta.chunks_written, 8);
            } else {
                assert_eq!(r1.io_delta.chunks_written, 16);
            }

            let mut cold = vram.cold_boot_clone_layout().unwrap();
            let restored = engine.restore(&mut cold, "p1").unwrap();
            assert!(restored.bit_exact_verified);
            assert_eq!(cold.state_digest(), vram.state_digest());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
