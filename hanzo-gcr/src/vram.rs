// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Simulated GPU VRAM page-table allocator with exact atomic dirty-page tracking,
//! immutable shared model weights, and strict `(tenant_id, job_id)` isolation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::crypto::{sha256, Digest, Sha256};
use crate::error::{GcrError, GcrResult};

/// Supported logical GPU VRAM page sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageSize {
    /// 4 KiB standard OS / fine-grained KV page.
    KiB4,
    /// 64 KiB paged-attention / GPU sub-block page.
    KiB64,
    /// 2 MiB GPU huge-page / CUDA VMM allocation granule.
    MiB2,
    /// Custom validated power-of-two page size in `[256, 16 MiB]`.
    Custom(usize),
}

impl PageSize {
    /// Minimum allowed page size (256 bytes, useful for compact unit tests).
    pub const MIN_BYTES: usize = 256;
    /// Maximum allowed page size (16 MiB).
    pub const MAX_BYTES: usize = 16 * 1024 * 1024;

    /// Validates and returns the page size in bytes.
    pub fn bytes(self) -> GcrResult<usize> {
        let size = match self {
            Self::KiB4 => 4 * 1024,
            Self::KiB64 => 64 * 1024,
            Self::MiB2 => 2 * 1024 * 1024,
            Self::Custom(n) => n,
        };
        if !(Self::MIN_BYTES..=Self::MAX_BYTES).contains(&size) || !size.is_power_of_two() {
            return Err(GcrError::InvalidPageSize(size));
        }
        Ok(size)
    }

    /// Human-readable label for benchmark reports.
    pub fn label(self) -> String {
        match self {
            Self::KiB4 => "4KiB".to_string(),
            Self::KiB64 => "64KiB".to_string(),
            Self::MiB2 => "2MiB".to_string(),
            Self::Custom(n) => format!("{n}B"),
        }
    }
}

/// Logical memory region tag for each VRAM page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PageRegion {
    /// Read-only shared model weights (content-addressed base; never duplicated in deltas).
    ImmutableWeights = 0,
    /// Mutable KV-cache pages for autoregressive decoding / agent state.
    KvCache = 1,
    /// Mutable LoRA adapter weight pages (A/B matrices).
    LoRaAdapter = 2,
    /// Mutable optimizer moments (AdamW m/v, master weights, gradient accumulators).
    OptimizerState = 3,
    /// Mutable activation checkpoints / scratchpad pages.
    Activations = 4,
}

impl PageRegion {
    /// Returns `true` if the page is read-only after initialization.
    #[inline]
    pub const fn is_immutable(self) -> bool {
        matches!(self, Self::ImmutableWeights)
    }

    /// Converts a wire byte back into a `PageRegion`.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ImmutableWeights),
            1 => Some(Self::KvCache),
            2 => Some(Self::LoRaAdapter),
            3 => Some(Self::OptimizerState),
            4 => Some(Self::Activations),
            _ => None,
        }
    }
}

/// Strict per-`(tenant_id, job_id)` isolation namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Namespace {
    tenant_id: String,
    job_id: String,
    tag: Digest,
}

impl Namespace {
    /// Constructs a validated namespace, rejecting unsafe characters and path traversal.
    pub fn new(tenant_id: impl Into<String>, job_id: impl Into<String>) -> GcrResult<Self> {
        let tenant_id = tenant_id.into();
        let job_id = job_id.into();
        validate_identifier("tenant_id", &tenant_id)?;
        validate_identifier("job_id", &job_id)?;

        let mut h = Sha256::new();
        h.update(b"hanzo-gcr:namespace:v1\0");
        h.update((tenant_id.len() as u64).to_le_bytes().as_slice());
        h.update(tenant_id.as_bytes());
        h.update(b"\0");
        h.update((job_id.len() as u64).to_le_bytes().as_slice());
        h.update(job_id.as_bytes());
        let tag = h.finalize();

        Ok(Self {
            tenant_id,
            job_id,
            tag,
        })
    }

    /// Tenant identifier.
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Job identifier.
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Domain-separated SHA-256 namespace binding tag.
    pub const fn tag(&self) -> Digest {
        self.tag
    }

    /// Safe deterministic directory key for tenant isolation on disk.
    pub fn tenant_dir_key(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"hanzo-gcr:tenant-dir:v1\0");
        h.update(self.tenant_id.as_bytes());
        h.finalize().to_hex()
    }

    /// Safe deterministic directory key for job isolation within a tenant directory.
    pub fn job_dir_key(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"hanzo-gcr:job-dir:v1\0");
        h.update(self.job_id.as_bytes());
        h.finalize().to_hex()
    }
}

fn validate_identifier(field: &str, value: &str) -> GcrResult<()> {
    if value.is_empty() || value.len() > 128 {
        return Err(GcrError::InvalidNamespace(format!(
            "{field} length must be in 1..=128 bytes"
        )));
    }
    if value == "." || value == ".." || value.contains("..") {
        return Err(GcrError::InvalidNamespace(format!(
            "{field} contains forbidden traversal sequence"
        )));
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err(GcrError::InvalidNamespace(format!(
            "{field} contains forbidden characters (allowed: [A-Za-z0-9._-])"
        )));
    }
    Ok(())
}

/// Lock-free exact atomic dirty-page bitmap backed by `AtomicU64` words.
#[derive(Debug)]
pub struct AtomicDirtyBitmap {
    num_pages: usize,
    words: Vec<AtomicU64>,
}

impl AtomicDirtyBitmap {
    /// Creates an all-clean bitmap covering `num_pages` pages.
    pub fn new(num_pages: usize) -> Self {
        let word_count = num_pages.div_ceil(64);
        let mut words = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            words.push(AtomicU64::new(0));
        }
        Self { num_pages, words }
    }

    /// Number of tracked pages.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_pages
    }

    /// Returns `true` if zero pages are tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_pages == 0
    }

    /// Atomically marks `page_id` as dirty.
    #[inline]
    pub fn mark_dirty(&self, page_id: usize) -> GcrResult<()> {
        if page_id >= self.num_pages {
            return Err(GcrError::PageOutOfBounds {
                page_id,
                total_pages: self.num_pages,
            });
        }
        let word_idx = page_id / 64;
        let bit = 1u64 << (page_id % 64);
        self.words[word_idx].fetch_or(bit, Ordering::AcqRel);
        Ok(())
    }

    /// Returns `true` if `page_id` is currently marked dirty.
    #[inline]
    pub fn is_dirty(&self, page_id: usize) -> bool {
        if page_id >= self.num_pages {
            return false;
        }
        let word_idx = page_id / 64;
        let bit = 1u64 << (page_id % 64);
        (self.words[word_idx].load(Ordering::Acquire) & bit) != 0
    }

    /// Returns the current number of dirty pages.
    pub fn dirty_count(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }

    /// Returns a sorted snapshot of currently dirty page indices without clearing them.
    pub fn dirty_indices(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (word_idx, word) in self.words.iter().enumerate() {
            let mut bits = word.load(Ordering::Acquire);
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                let page_id = word_idx * 64 + tz;
                if page_id < self.num_pages {
                    out.push(page_id);
                }
                bits &= bits - 1;
            }
        }
        out
    }

    /// Atomically swaps every word to `0` and returns all previously dirty page indices in ascending order.
    ///
    /// Because each word is exchanged with `swap(0, AcqRel)`, any concurrent write arriving
    /// after the swap sets a fresh dirty bit for the next checkpoint rather than being lost.
    pub fn take_and_clear_all(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (word_idx, word) in self.words.iter().enumerate() {
            let mut bits = word.swap(0, Ordering::AcqRel);
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                let page_id = word_idx * 64 + tz;
                if page_id < self.num_pages {
                    out.push(page_id);
                }
                bits &= bits - 1;
            }
        }
        out
    }

    /// Re-marks a set of dirty page indices (used for crash/error rollback if a commit fails).
    pub fn restore_bits(&self, page_ids: &[usize]) {
        for &page_id in page_ids {
            if page_id < self.num_pages {
                let word_idx = page_id / 64;
                let bit = 1u64 << (page_id % 64);
                self.words[word_idx].fetch_or(bit, Ordering::AcqRel);
            }
        }
    }

    /// Clears all dirty bits unconditionally.
    pub fn clear_all(&self) {
        for word in &self.words {
            word.store(0, Ordering::Release);
        }
    }
}

/// Content-addressed immutable read-only model weights shared across jobs and pause/resume cycles.
#[derive(Debug, Clone)]
pub struct SharedWeightBase {
    page_size: usize,
    pages: Arc<Vec<Vec<u8>>>,
    page_digests: Arc<Vec<Digest>>,
    base_digest: Digest,
}

impl SharedWeightBase {
    /// Builds an immutable shared weight base from fixed-size pages.
    pub fn new(page_size: PageSize, pages: Vec<Vec<u8>>) -> GcrResult<Self> {
        let ps = page_size.bytes()?;
        let mut page_digests = Vec::with_capacity(pages.len());
        let mut agg = Sha256::new();
        agg.update(b"hanzo-gcr:weight-base:v1\0");
        agg.update((ps as u64).to_le_bytes().as_slice());
        agg.update((pages.len() as u64).to_le_bytes().as_slice());

        for page in &pages {
            if page.len() != ps {
                return Err(GcrError::PageLengthMismatch {
                    expected: ps,
                    actual: page.len(),
                });
            }
            let d = sha256(page);
            agg.update(d.as_bytes());
            page_digests.push(d);
        }

        Ok(Self {
            page_size: ps,
            pages: Arc::new(pages),
            page_digests: Arc::new(page_digests),
            base_digest: agg.finalize(),
        })
    }

    /// Configured page size in bytes.
    #[inline]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of immutable weight pages.
    #[inline]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Total weight bytes.
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        (self.pages.len() as u64) * (self.page_size as u64)
    }

    /// Content-addressed digest uniquely identifying this immutable weight base.
    #[inline]
    pub const fn base_digest(&self) -> Digest {
        self.base_digest
    }

    /// Borrows an immutable weight page by index.
    #[inline]
    pub fn page(&self, idx: usize) -> Option<&[u8]> {
        self.pages.get(idx).map(Vec::as_slice)
    }

    /// Returns the precomputed SHA-256 digest of weight page `idx`.
    #[inline]
    pub fn page_digest(&self, idx: usize) -> Option<Digest> {
        self.page_digests.get(idx).copied()
    }
}

/// Simulated GPU VRAM page table for a single isolated `(tenant_id, job_id)` execution.
///
/// Pages `0..immutable_pages` map read-only to the shared `SharedWeightBase` (zero copy!).
/// Pages `immutable_pages..total_pages` are mutable job state (KV cache, LoRA adapters,
/// optimizer states, activations) tracked via `AtomicDirtyBitmap`.
#[derive(Debug)]
pub struct GpuMemorySpace {
    namespace: Namespace,
    page_size: usize,
    weights: SharedWeightBase,
    mutable_regions: Vec<PageRegion>,
    mutable_pages: Vec<Vec<u8>>,
    mutable_leaf_digests: Mutex<Vec<Option<Digest>>>,
    dirty_bitmap: AtomicDirtyBitmap,
    write_ops: u64,
}

impl GpuMemorySpace {
    /// Allocates a VRAM address space mounting `weights` read-only at pages `0..W` and
    /// zero-initialized mutable pages at `W..(W + mutable_regions.len())`.
    pub fn new(
        namespace: Namespace,
        weights: SharedWeightBase,
        mutable_regions: Vec<PageRegion>,
    ) -> GcrResult<Self> {
        for &region in &mutable_regions {
            if region.is_immutable() {
                return Err(GcrError::LayoutMismatch {
                    reason: "mutable_regions must not contain ImmutableWeights".to_string(),
                });
            }
        }
        let page_size = weights.page_size();
        let total_pages = weights.page_count() + mutable_regions.len();
        let zero_digest = sha256(&vec![0u8; page_size]);
        let mut mutable_pages = Vec::with_capacity(mutable_regions.len());
        let mut mutable_leaf_digests = Vec::with_capacity(mutable_regions.len());
        for _ in 0..mutable_regions.len() {
            mutable_pages.push(vec![0u8; page_size]);
            mutable_leaf_digests.push(Some(zero_digest));
        }
        let dirty_bitmap = AtomicDirtyBitmap::new(total_pages);

        Ok(Self {
            namespace,
            page_size,
            weights,
            mutable_regions,
            mutable_pages,
            mutable_leaf_digests: Mutex::new(mutable_leaf_digests),
            dirty_bitmap,
            write_ops: 0,
        })
    }

    /// Owning `(tenant_id, job_id)` namespace.
    #[inline]
    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Logical page size in bytes.
    #[inline]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of immutable shared weight pages (`0..immutable_page_count()`).
    #[inline]
    pub fn immutable_page_count(&self) -> usize {
        self.weights.page_count()
    }

    /// Number of mutable state pages.
    #[inline]
    pub fn mutable_page_count(&self) -> usize {
        self.mutable_pages.len()
    }

    /// Total logical pages (`immutable + mutable`).
    #[inline]
    pub fn total_page_count(&self) -> usize {
        self.weights.page_count() + self.mutable_pages.len()
    }

    /// Total VRAM footprint in bytes (`total_page_count * page_size`).
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        (self.total_page_count() as u64) * (self.page_size as u64)
    }

    /// Mutable VRAM footprint in bytes (`mutable_page_count * page_size`).
    #[inline]
    pub fn mutable_bytes(&self) -> u64 {
        (self.mutable_page_count() as u64) * (self.page_size as u64)
    }

    /// Content-addressed digest of the shared immutable weight base.
    #[inline]
    pub const fn model_base_digest(&self) -> Digest {
        self.weights.base_digest()
    }

    /// Reference to the shared immutable weight base.
    #[inline]
    pub fn shared_weights(&self) -> &SharedWeightBase {
        &self.weights
    }

    /// Reference to the atomic dirty-page bitmap.
    #[inline]
    pub fn dirty_bitmap(&self) -> &AtomicDirtyBitmap {
        &self.dirty_bitmap
    }

    /// Cumulative write operations observed by this memory space.
    #[inline]
    pub const fn write_ops(&self) -> u64 {
        self.write_ops
    }

    /// Returns the `PageRegion` of `page_id`.
    pub fn page_region(&self, page_id: usize) -> GcrResult<PageRegion> {
        let w = self.weights.page_count();
        if page_id < w {
            Ok(PageRegion::ImmutableWeights)
        } else if page_id < w + self.mutable_regions.len() {
            Ok(self.mutable_regions[page_id - w])
        } else {
            Err(GcrError::PageOutOfBounds {
                page_id,
                total_pages: self.total_page_count(),
            })
        }
    }

    /// Returns a slice of all mutable page regions (for recreating a cold-boot target).
    pub fn mutable_regions(&self) -> &[PageRegion] {
        &self.mutable_regions
    }

    /// Reads page `page_id` bytes (works for both immutable weights and mutable state).
    pub fn read_page(&self, page_id: usize) -> GcrResult<&[u8]> {
        let w = self.weights.page_count();
        if page_id < w {
            Ok(self.weights.page(page_id).expect("checked bounds"))
        } else if page_id < w + self.mutable_pages.len() {
            Ok(self.mutable_pages[page_id - w].as_slice())
        } else {
            Err(GcrError::PageOutOfBounds {
                page_id,
                total_pages: self.total_page_count(),
            })
        }
    }

    /// Overwrites an entire mutable page, invalidates its cached leaf digest, and atomically sets its dirty bit.
    ///
    /// Returns `Err(GcrError::ImmutablePageWrite)` if `page_id` belongs to read-only weights.
    pub fn overwrite_page(&mut self, page_id: usize, data: &[u8]) -> GcrResult<()> {
        if data.len() != self.page_size {
            return Err(GcrError::PageLengthMismatch {
                expected: self.page_size,
                actual: data.len(),
            });
        }
        let w = self.weights.page_count();
        if page_id < w {
            return Err(GcrError::ImmutablePageWrite { page_id });
        }
        let idx = page_id - w;
        if idx >= self.mutable_pages.len() {
            return Err(GcrError::PageOutOfBounds {
                page_id,
                total_pages: self.total_page_count(),
            });
        }
        self.mutable_pages[idx].copy_from_slice(data);
        self.mutable_leaf_digests.get_mut().expect("leaf lock")[idx] = None;
        self.dirty_bitmap.mark_dirty(page_id)?;
        self.write_ops += 1;
        Ok(())
    }

    /// Writes a byte range `[offset..offset+data.len()]` inside a mutable page and sets its dirty bit.
    pub fn write_page_range(
        &mut self,
        page_id: usize,
        offset: usize,
        data: &[u8],
    ) -> GcrResult<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or(GcrError::PageLengthMismatch {
                expected: self.page_size,
                actual: usize::MAX,
            })?;
        if end > self.page_size {
            return Err(GcrError::PageLengthMismatch {
                expected: self.page_size,
                actual: end,
            });
        }
        let w = self.weights.page_count();
        if page_id < w {
            return Err(GcrError::ImmutablePageWrite { page_id });
        }
        let idx = page_id - w;
        if idx >= self.mutable_pages.len() {
            return Err(GcrError::PageOutOfBounds {
                page_id,
                total_pages: self.total_page_count(),
            });
        }
        self.mutable_pages[idx][offset..end].copy_from_slice(data);
        self.mutable_leaf_digests.get_mut().expect("leaf lock")[idx] = None;
        self.dirty_bitmap.mark_dirty(page_id)?;
        self.write_ops += 1;
        Ok(())
    }

    /// Updates a mutable page's cached SHA-256 leaf digest (after chunk encoding already hashed it).
    pub(crate) fn cache_mutable_leaf_digest(&self, page_id: usize, digest: Digest) {
        let w = self.weights.page_count();
        if page_id >= w {
            let idx = page_id - w;
            let mut guard = self.mutable_leaf_digests.lock().expect("leaf lock");
            if idx < guard.len() {
                guard[idx] = Some(digest);
            }
        }
    }

    /// Marks every mutable page dirty (e.g., on initial load before first checkpoint).
    pub fn mark_all_mutable_dirty(&self) {
        let w = self.weights.page_count();
        for page_id in w..self.total_page_count() {
            let _ = self.dirty_bitmap.mark_dirty(page_id);
        }
    }

    /// Returns the page IDs of all mutable pages matching `region`.
    pub fn page_ids_in_region(&self, region: PageRegion) -> Vec<usize> {
        let w = self.weights.page_count();
        if region.is_immutable() {
            return (0..w).collect();
        }
        self.mutable_regions
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| if r == region { Some(w + i) } else { None })
            .collect()
    }

    /// Transactionally replaces all mutable page buffers and their verified leaf digests.
    ///
    /// Clears the dirty bitmap upon commit so the resumed job starts clean.
    pub(crate) fn commit_restored_mutable_pages(
        &mut self,
        restored: Vec<Vec<u8>>,
        verified_digests: Vec<Digest>,
    ) -> GcrResult<()> {
        if restored.len() != self.mutable_pages.len()
            || verified_digests.len() != self.mutable_pages.len()
        {
            return Err(GcrError::LayoutMismatch {
                reason: format!(
                    "expected {} mutable pages, got {}",
                    self.mutable_pages.len(),
                    restored.len()
                ),
            });
        }
        for page in &restored {
            if page.len() != self.page_size {
                return Err(GcrError::PageLengthMismatch {
                    expected: self.page_size,
                    actual: page.len(),
                });
            }
        }
        self.mutable_pages = restored;
        *self.mutable_leaf_digests.get_mut().expect("leaf lock") =
            verified_digests.into_iter().map(Some).collect();
        self.dirty_bitmap.clear_all();
        Ok(())
    }

    /// Computes a deterministic SHA-256 Merkle-root digest over the complete VRAM image
    /// (immutable weight base + every mutable page's SHA-256 leaf digest) for 100.0% bit-exact
    /// verification in `O(dirty_pages)` incremental cost.
    pub fn state_digest(&self) -> Digest {
        let mut guard = self.mutable_leaf_digests.lock().expect("leaf lock");
        let mut h = Sha256::new();
        h.update(b"hanzo-gcr:vram-merkle:v1\0");
        h.update((self.page_size as u64).to_le_bytes().as_slice());
        h.update((self.weights.page_count() as u64).to_le_bytes().as_slice());
        h.update((self.mutable_pages.len() as u64).to_le_bytes().as_slice());
        h.update(self.weights.base_digest().as_bytes());
        for (idx, (region, page)) in self
            .mutable_regions
            .iter()
            .zip(self.mutable_pages.iter())
            .enumerate()
        {
            let leaf = match guard[idx] {
                Some(d) => d,
                None => {
                    let d = sha256(page);
                    guard[idx] = Some(d);
                    d
                }
            };
            h.update(&[*region as u8]);
            h.update(leaf.as_bytes());
        }
        h.finalize()
    }

    /// Computes the exact VRAM Merkle-root digest directly from already-verified mutable leaf digests.
    pub(crate) fn compute_merkle_root_from_leaves(&self, leaf_digests: &[Digest]) -> Digest {
        let mut h = Sha256::new();
        h.update(b"hanzo-gcr:vram-merkle:v1\0");
        h.update((self.page_size as u64).to_le_bytes().as_slice());
        h.update((self.weights.page_count() as u64).to_le_bytes().as_slice());
        h.update((leaf_digests.len() as u64).to_le_bytes().as_slice());
        h.update(self.weights.base_digest().as_bytes());
        for (region, leaf) in self.mutable_regions.iter().zip(leaf_digests.iter()) {
            h.update(&[*region as u8]);
            h.update(leaf.as_bytes());
        }
        h.finalize()
    }

    /// Creates a fresh cold-boot GPU memory space sharing the exact same immutable weights and
    /// mutable layout (zero-initialized mutable pages), ready to accept a checkpoint restore.
    pub fn cold_boot_clone_layout(&self) -> GcrResult<Self> {
        Self::new(
            self.namespace.clone(),
            self.weights.clone(),
            self.mutable_regions.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_rejects_traversal_and_unsafe_chars() {
        assert!(Namespace::new("tenant-a", "job_01.prod").is_ok());
        assert!(Namespace::new("", "job-1").is_err());
        assert!(Namespace::new("tenant-a", "").is_err());
        assert!(Namespace::new("../evil", "job-1").is_err());
        assert!(Namespace::new("tenant/a", "job-1").is_err());
        assert!(Namespace::new("tenant\\a", "job-1").is_err());
        assert!(Namespace::new("tenant a", "job-1").is_err());
        assert!(Namespace::new("tenant-a", "..").is_err());
    }

    #[test]
    fn immutable_weight_write_is_rejected_and_dirty_bitmap_is_exact() {
        let weights = SharedWeightBase::new(
            PageSize::Custom(512),
            vec![vec![0xAA; 512], vec![0xBB; 512]],
        )
        .unwrap();
        let ns = Namespace::new("acme", "infer-1").unwrap();
        let mut vram = GpuMemorySpace::new(
            ns,
            weights,
            vec![
                PageRegion::KvCache,
                PageRegion::KvCache,
                PageRegion::LoRaAdapter,
            ],
        )
        .unwrap();

        assert_eq!(
            vram.overwrite_page(0, &vec![0xFF; 512]),
            Err(GcrError::ImmutablePageWrite { page_id: 0 })
        );
        assert_eq!(vram.dirty_bitmap().dirty_count(), 0);

        vram.write_page_range(2, 16, b"hello-kv").unwrap();
        vram.overwrite_page(4, &vec![0x42; 512]).unwrap();
        assert_eq!(vram.dirty_bitmap().dirty_indices(), vec![2, 4]);

        let drained = vram.dirty_bitmap().take_and_clear_all();
        assert_eq!(drained, vec![2, 4]);
        assert_eq!(vram.dirty_bitmap().dirty_count(), 0);

        vram.dirty_bitmap().restore_bits(&drained);
        assert_eq!(vram.dirty_bitmap().dirty_indices(), vec![2, 4]);
    }

    #[test]
    fn page_size_validation() {
        assert_eq!(PageSize::KiB4.bytes().unwrap(), 4096);
        assert_eq!(PageSize::KiB64.bytes().unwrap(), 65536);
        assert_eq!(PageSize::MiB2.bytes().unwrap(), 2_097_152);
        assert!(PageSize::Custom(128).bytes().is_err());
        assert!(PageSize::Custom(3000).bytes().is_err());
    }
}
