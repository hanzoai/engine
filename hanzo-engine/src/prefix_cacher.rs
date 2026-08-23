#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use hanzo_ml::{Device, Result};
use indexmap::IndexMap;
use itertools::Itertools;
use tracing::info;

use crate::{
    disk_kv_cache::{key_for_bytes, DiskKvCache, KvcHeader, SaveReason},
    kv_cache::{ByteReader, RecurrentStateSnapshot, RestoreLimits},
    paged_attention::block_hash::BlockHash,
    pipeline::KvCache,
    sequence::Sequence,
};

#[derive(PartialEq, Eq, Debug, Hash)]
struct Tokens(Vec<u32>);

impl Tokens {
    /// Returns the length of the common prefix shared with `other`.
    fn shared_prefix_len(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(other.0.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

#[derive(Clone)]
pub(crate) struct CacheElement {
    cache: Vec<Option<KvCache>>,
    /// Recurrent state snapshots for hybrid models (one per recurrent layer)
    recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    audio_hashes: Option<Vec<u64>>,
    image_hashes: Option<Vec<u64>>,
    video_hashes: Option<Vec<u64>>,
}

impl CacheElement {
    fn can_rewind_to(&self, len: usize) -> bool {
        self.cache
            .iter()
            .flatten()
            .all(|layer| layer.try_set_len(len).is_ok())
    }

    /// Serialize a whole prefix — every layer in order, recurrent-state snapshots,
    /// and modality hashes — into a self-describing byte buffer for the disk spill
    /// store. Each layer reuses [`KvCache::to_payload`]; GPU tensors are copied to
    /// host as part of that per-layer safetensors encode. This is the inverse of
    /// [`Self::from_bytes`].
    fn to_bytes(&self, fingerprint: u64) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.push(PREFIX_PAYLOAD_VERSION);
        out.extend_from_slice(&fingerprint.to_le_bytes());

        out.extend_from_slice(&(self.cache.len() as u32).to_le_bytes());
        for layer in &self.cache {
            match layer {
                Some(kv) => {
                    out.push(1u8);
                    let payload = kv.to_payload()?;
                    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
                    out.extend_from_slice(&payload);
                }
                None => out.push(0u8),
            }
        }

        match &self.recurrent_snapshots {
            Some(snaps) => {
                out.push(1u8);
                out.extend_from_slice(&(snaps.len() as u32).to_le_bytes());
                for snap in snaps {
                    write_recurrent_snapshot(&mut out, snap)?;
                }
            }
            None => out.push(0u8),
        }

        write_hashes(&mut out, self.image_hashes.as_deref());
        write_hashes(&mut out, self.audio_hashes.as_deref());
        write_hashes(&mut out, self.video_hashes.as_deref());

        Ok(out)
    }

    /// Reconstruct a prefix from [`Self::to_bytes`], materializing every tensor on
    /// `device`. A trust boundary: the model fingerprint must match `limits`, every
    /// count/dimension is bounded by `limits` before any allocation, and
    /// malformed/truncated/foreign input returns an error rather than panicking or
    /// over-allocating.
    fn from_bytes(bytes: &[u8], device: &Device, limits: &RestoreLimits) -> Result<Self> {
        let mut r = ByteReader::new(bytes);
        let version = r.read_u8()?;
        if version != PREFIX_PAYLOAD_VERSION {
            hanzo_ml::bail!("prefix payload: unsupported version {version}");
        }
        let fingerprint = r.read_u64()?;
        if fingerprint != limits.fingerprint {
            hanzo_ml::bail!(
                "prefix payload: model fingerprint mismatch (got {fingerprint:#018x}, \
                 expected {:#018x}) — refusing cross-model KV",
                limits.fingerprint
            );
        }

        let n_layers = usize_from_u64(u64::from(r.read_u32()?))?;
        if n_layers > limits.max_layers {
            hanzo_ml::bail!(
                "prefix payload: n_layers {n_layers} exceeds model layers {}",
                limits.max_layers
            );
        }
        let mut cache = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            match r.read_u8()? {
                0 => cache.push(None),
                1 => {
                    let len = usize_from_u64(r.read_u64()?)?;
                    if len as u64 > limits.max_payload_bytes {
                        hanzo_ml::bail!(
                            "prefix payload: layer length {len} exceeds budget {}",
                            limits.max_payload_bytes
                        );
                    }
                    let payload = r.read_bytes(len)?;
                    cache.push(Some(KvCache::from_payload(payload, device, limits)?));
                }
                other => hanzo_ml::bail!("prefix payload: bad layer flag {other}"),
            }
        }

        let recurrent_snapshots = match r.read_u8()? {
            0 => None,
            1 => {
                let n = usize_from_u64(u64::from(r.read_u32()?))?;
                // At most one recurrent snapshot per layer.
                if n > limits.max_layers {
                    hanzo_ml::bail!(
                        "prefix payload: recurrent count {n} exceeds model layers {}",
                        limits.max_layers
                    );
                }
                let mut snaps = Vec::with_capacity(n);
                for _ in 0..n {
                    snaps.push(read_recurrent_snapshot(&mut r, device, limits)?);
                }
                Some(snaps)
            }
            other => hanzo_ml::bail!("prefix payload: bad recurrent flag {other}"),
        };

        let image_hashes = read_hashes(&mut r, limits)?;
        let audio_hashes = read_hashes(&mut r, limits)?;
        let video_hashes = read_hashes(&mut r, limits)?;

        Ok(Self {
            cache,
            recurrent_snapshots,
            image_hashes,
            audio_hashes,
            video_hashes,
        })
    }
}

pub struct PrefixCacheManagerV2 {
    caches: IndexMap<Tokens, CacheElement>,
    paged_recurrent_caches: IndexMap<Vec<BlockHash>, Vec<RecurrentStateSnapshot>>,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
    /// Optional disk-backed spill store. When present, on-device prefixes are
    /// written here on eviction instead of being dropped, and can be restored
    /// later (including across process restarts).
    disk: Option<DiskSpill>,
}

/// A disk-backed KV spill store bundled with the live-model limits/fingerprint
/// used to validate untrusted `.kv` payloads on restore and tag them on spill.
struct DiskSpill {
    store: DiskKvCache,
    limits: RestoreLimits,
}

#[derive(Clone)]
pub enum MatchingCache {
    Normal {
        normal: Vec<Option<KvCache>>,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
        images_to_keep: usize,
        audios_to_keep: usize,
        videos_to_keep: usize,
        toks: Vec<u32>,
        offset: usize,
    },
}

impl PrefixCacheManagerV2 {
    pub fn new(n_on_device: usize, no_prefix_cache: bool, has_paged_attention: bool) -> Self {
        if !no_prefix_cache && !has_paged_attention {
            info!("Prefix caching enabled (sequence-level, non-paged attention). Expect higher multi-turn throughput for both text and multimodal.");
        }
        PrefixCacheManagerV2 {
            caches: IndexMap::new(),
            paged_recurrent_caches: IndexMap::new(),
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
            disk: None,
        }
    }

    /// Attach a disk-backed KV spill store with the validation `limits` for the
    /// live model.
    ///
    /// With a disk cache set, [`Self::evict_caches`] spills each evicted on-device
    /// prefix to disk (instead of dropping it), and [`Self::restore_from_disk`] can
    /// repopulate exact token prefixes on a later — even cross-process — cold start.
    /// `limits` carries the model fingerprint and the bounds (max layers/context,
    /// disk budget) that every untrusted `.kv` payload is validated against before
    /// any allocation. This is what makes 1M-context KV cheap to keep and agent
    /// prefixes reusable.
    #[must_use]
    pub(crate) fn with_disk_cache(mut self, store: DiskKvCache, limits: RestoreLimits) -> Self {
        self.disk = Some(DiskSpill { store, limits });
        self
    }

    fn paged_recurrent_capacity(&self) -> usize {
        self.n_on_device.max(1).saturating_mul(8)
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(
        &mut self,
        seq: &mut Sequence,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    ) {
        // Do not cache if prefix caching disabled
        if self.no_prefix_cache {
            return;
        }

        // For paged attention, prefix caching is handled by the KVCacheManager.
        // PrefixCacheManagerV2 only handles non-paged attention caching.
        if !self.has_paged_attention {
            let cache = seq.normal_cache().to_vec();

            self.caches.insert(
                seq.get_toks().to_vec().into(),
                CacheElement {
                    cache,
                    recurrent_snapshots,
                    image_hashes: seq.image_hashes().map(|x| x.to_vec()),
                    audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
                    video_hashes: seq.video_hashes().map(|x| x.to_vec()),
                },
            );
        }
    }

    /// Evict caches down to the on-device budget. Returns the number of prefixes
    /// evicted. Without a disk store this drops the oldest device-resident
    /// prefixes (freeing device memory); with a disk store attached each evicted
    /// prefix is spilled to disk instead, so it survives both memory pressure and
    /// process restarts.
    pub fn evict_caches(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        if self.disk.is_some() {
            self.evict_caches_spilling()
        } else {
            self.evict_caches_dropping()
        }
    }

    /// No-disk eviction: drop the oldest device-resident prefixes. This is the
    /// original behavior, kept byte-identical for the common (no spill) path.
    fn evict_caches_dropping(&mut self) -> Result<usize> {
        let mut n_on_device = 0;
        for cache in self.caches.values() {
            let first_non_none = cache
                .cache
                .iter()
                .find_or_first(|x| x.as_ref().is_some_and(|kv| kv.k().ok().flatten().is_some()));
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Shared { .. } => continue,
            };

            if !matches!(cache_device, Device::Cpu) {
                n_on_device += 1;
            }
        }
        let mut n_evicted = 0;
        // Intentionally evict the first ones first, as they are the oldest
        for cache in self.caches.values_mut() {
            if n_on_device - n_evicted <= self.n_on_device {
                break;
            }
            let first_non_none = cache
                .cache
                .iter()
                .find_or_first(|x| x.as_ref().is_some_and(|kv| kv.k().ok().flatten().is_some()));
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Shared { .. } => continue,
            };

            if !matches!(cache_device, Device::Cpu) {
                cache.cache.clear();
                n_evicted += 1;
            }
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());

        Ok(n_evicted)
    }

    /// Disk-backed eviction: bound the in-memory map to the on-device budget and
    /// spill the oldest overflow prefixes to disk (oldest first — the map
    /// preserves insertion order). A spill failure logs and still drops the
    /// prefix, so eviction never blocks generation under memory pressure.
    fn evict_caches_spilling(&mut self) -> Result<usize> {
        let target = self.n_on_device;
        let total = self.caches.len();
        if total <= target {
            return Ok(0);
        }
        let n_to_evict = total - target;
        // Disjoint field borrows: read `disk` while mutating `caches`.
        let spill = self.disk.as_ref().expect("disk store present");
        let store = &spill.store;
        let fingerprint = spill.limits.fingerprint;
        let mut n_evicted = 0;
        for (tokens, cache) in self.caches.iter_mut() {
            if n_evicted >= n_to_evict {
                break;
            }
            if let Err(e) = spill_prefix(store, fingerprint, &tokens.0, cache) {
                tracing::warn!("KV prefix spill failed (dropping instead): {e}");
            }
            cache.cache.clear();
            n_evicted += 1;
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());

        // Enforce the on-disk budget (LRU ceiling) after writing the freshly
        // spilled prefixes, so distinct prefixes can't accumulate unbounded.
        if let Err(e) = store.evict_to_budget() {
            tracing::warn!("KV disk budget enforcement failed: {e}");
        }

        Ok(n_evicted)
    }

    /// Load and deserialize a single spilled prefix by its content key,
    /// materializing tensors on `device`. Returns the prefix's tokens (recovered
    /// from the stored key bytes) alongside the cache element, or `None` when no
    /// disk store is attached or the key is absent. This is the exact (single-key)
    /// restore primitive; [`Self::restore_from_disk`] uses it to repopulate the map.
    pub(crate) fn load_from_disk(
        &self,
        key: &str,
        device: &Device,
    ) -> Result<Option<(Vec<u32>, CacheElement)>> {
        let Some(spill) = self.disk.as_ref() else {
            return Ok(None);
        };
        let Some(hit) = spill
            .store
            .load(key)
            .map_err(|e| hanzo_ml::Error::Msg(format!("kv disk load failed: {e}")))?
        else {
            return Ok(None);
        };
        let Some(toks) = tokens_from_le_bytes(&hit.rendered_text) else {
            hanzo_ml::bail!("kv disk entry {key}: malformed token bytes");
        };
        // Integrity: the header records the token count at spill time; a mismatch
        // means a corrupt or truncated entry, so refuse it rather than trust it.
        if hit.header.token_count as usize != toks.len() {
            hanzo_ml::bail!(
                "kv disk entry {key}: header token_count {} != {} recovered tokens",
                hit.header.token_count,
                toks.len()
            );
        }
        let element = CacheElement::from_bytes(&hit.payload, device, &spill.limits)?;
        Ok(Some((toks, element)))
    }

    /// Repopulate the in-memory prefix map from the disk spill store on a cold
    /// start. Restores the most-recently-used prefixes first, materializing
    /// tensors on `device`, bounded to the on-device budget so startup never
    /// exceeds the memory the running engine maintains. Returns the number of
    /// prefixes restored. A corrupt or undecodable entry is logged and skipped.
    pub(crate) fn restore_from_disk(&mut self, device: &Device) -> Result<usize> {
        if self.no_prefix_cache || self.has_paged_attention {
            return Ok(0);
        }
        let budget = self.n_on_device.max(1);
        // Scope the `disk` borrow to the key fetch so the loop can reborrow `self`.
        let keys = {
            let Some(spill) = self.disk.as_ref() else {
                return Ok(0);
            };
            // Cold start: sweep orphaned partial writes from a prior crash first.
            if let Err(e) = spill.store.sweep_partial() {
                tracing::warn!("KV disk tmp sweep failed: {e}");
            }
            let mut keys = spill
                .store
                .keys_by_recency()
                .map_err(|e| hanzo_ml::Error::Msg(format!("kv disk scan failed: {e}")))?;
            // Keep the `budget` hottest, then insert oldest-first below so the most
            // recently used land last in the map (evicted/re-spilled last).
            keys.truncate(budget);
            keys.reverse();
            keys
        };
        let mut restored = 0;
        for key in keys {
            match self.load_from_disk(&key, device) {
                Ok(Some((toks, element))) => {
                    self.caches.insert(toks.into(), element);
                    restored += 1;
                }
                Ok(None) => continue,
                Err(e) => tracing::warn!("KV disk entry {key} skipped on restore: {e}"),
            }
        }
        Ok(restored)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let len = self.caches.len();
        self.caches.clear();
        self.paged_recurrent_caches.clear();
        Ok(len)
    }

    /// Add a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// This is used by hybrid models to restore recurrent states alongside paged KV prefix hits.
    pub fn add_paged_recurrent_prefix(
        &mut self,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
    ) {
        if self.no_prefix_cache
            || !self.has_paged_attention
            || key.is_empty()
            || snapshots.is_empty()
        {
            return;
        }

        // Maintain LRU order by reinserting on update.
        let _ = self.paged_recurrent_caches.shift_remove(&key);
        self.paged_recurrent_caches.insert(key, snapshots);

        while self.paged_recurrent_caches.len() > self.paged_recurrent_capacity() {
            let _ = self.paged_recurrent_caches.shift_remove_index(0);
        }
    }

    /// Lookup a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// Returns a cloned snapshot and updates LRU order.
    pub fn get_paged_recurrent_prefix(
        &mut self,
        key: &[BlockHash],
    ) -> Option<Vec<RecurrentStateSnapshot>> {
        if self.no_prefix_cache || !self.has_paged_attention || key.is_empty() {
            return None;
        }

        let key = key.to_vec();
        let snapshots = self.paged_recurrent_caches.shift_remove(&key)?;
        let out = snapshots.clone();
        self.paged_recurrent_caches.insert(key, snapshots);
        Some(out)
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        audio_hashes: Option<&[u64]>,
        video_hashes: Option<&[u64]>,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        if self.has_paged_attention {
            // For paged attention, prefix caching is handled by the KVCacheManager.
            // PrefixCacheManagerV2 only handles non-paged attention caching.
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());

        let mut best_match: Option<(usize, &CacheElement, usize, usize, usize)> = None;
        for (k, v) in &self.caches {
            let match_len = toks.shared_prefix_len(k);
            if match_len == 0 {
                continue;
            }

            let images_match_until = match image_hashes {
                Some(input_hashes) => match &v.image_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            let audios_match_until = match audio_hashes {
                Some(input_hashes) => match &v.audio_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            // Vision/audio models can use repeated placeholder tokens (e.g. <image_soft_token>)
            // that are identical regardless of the actual image/audio content. This means
            // token-level matching can extend through image/audio regions even when the
            // underlying content differs, leaving stale vision/audio encoder outputs in the
            // KV cache. Skip entries where any overlapping images or audios diverge.
            let cached_image_count = v.image_hashes.as_ref().map_or(0, |h| h.len());
            let input_image_count = image_hashes.map_or(0, |h| h.len());
            if images_match_until < input_image_count.min(cached_image_count) {
                continue;
            }

            let cached_audio_count = v.audio_hashes.as_ref().map_or(0, |h| h.len());
            let input_audio_count = audio_hashes.map_or(0, |h| h.len());
            if audios_match_until < input_audio_count.min(cached_audio_count) {
                continue;
            }

            let videos_match_until = match video_hashes {
                Some(input_hashes) => match &v.video_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            let cached_video_count = v.video_hashes.as_ref().map_or(0, |h| h.len());
            let input_video_count = video_hashes.map_or(0, |h| h.len());
            if videos_match_until < input_video_count.min(cached_video_count) {
                continue;
            }

            // Sliding/rotating caches only retain a fixed tail. If a cache has already
            // truncated older tokens, it can still safely serve an exact extension of the
            // cached prefix, but it cannot be rewound to an earlier logical length. Skip such
            // candidates here so a rolled-over cache does not block a shorter valid prefix hit.
            if !v.can_rewind_to(match_len) {
                continue;
            }

            if best_match
                .as_ref()
                .is_none_or(|(len, _, _, _, _)| match_len > *len)
            {
                best_match = Some((
                    match_len,
                    v,
                    images_match_until,
                    audios_match_until,
                    videos_match_until,
                ));
            }
        }

        if let Some((
            match_len,
            cache_element,
            images_match_until,
            audios_match_until,
            videos_match_until,
        )) = best_match
        {
            let new_toks = toks.0[match_len..].to_vec();
            if new_toks.is_empty() {
                return Ok(None);
            }

            let mut cache = cache_element.clone();
            // Count how many input images are not already cached
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(images_match_until)
            } else {
                0
            };
            let audios_to_keep = if let Some(input_hashes) = audio_hashes {
                input_hashes.len().saturating_sub(audios_match_until)
            } else {
                0
            };
            let videos_to_keep = if let Some(input_hashes) = video_hashes {
                input_hashes.len().saturating_sub(videos_match_until)
            } else {
                0
            };
            for layer in cache.cache.iter_mut().flatten() {
                if layer.try_set_len(match_len).is_err() {
                    return Ok(None);
                }
            }
            for layer in cache.cache.iter_mut().flatten() {
                layer.set_len(match_len)?;
            }
            return Ok(Some(MatchingCache::Normal {
                normal: cache.cache,
                recurrent_snapshots: cache.recurrent_snapshots,
                images_to_keep,
                audios_to_keep,
                videos_to_keep,
                toks: new_toks,
                offset: match_len,
            }));
        }

        Ok(None)
    }
}

// ===========================================================================
// Disk-spill prefix framing: CacheElement <-> bytes, plus spill/restore glue.
//
// A prefix is the layer-ordered `Vec<Option<KvCache>>` (Normal/Rotating attention
// KV), optional recurrent-state snapshots (hybrid models), and the modality hash
// vectors used during matching. The per-layer tensor encoding is owned by
// `KvCache::to_payload`; this layer adds the prefix-level framing on top, reusing
// the same bounds-checked `ByteReader`. PagedAttention KV lives in the separate
// GPU block pool (`paged_attention::KVCacheManager`) and is not covered here.
//
// `.kv` files are an untrusted-input trust boundary read at cold start, so the
// payload carries a model fingerprint (rejected on mismatch) and every count and
// dimension is bounded against `RestoreLimits` before any allocation.
//
// Layout (little-endian):
//   [u8 version]
//   [u64 fingerprint]      model identity; reject if != live model
//   [u32 n_layers]         bounded by model layers
//     per layer: [u8 present] (0/1); if 1: [u64 len][KvCache::to_payload bytes]
//   [u8 has_recurrent]
//     if 1: [u32 n] per snapshot: [u64 blob_len][safetensors {conv,recurrent}][u64 seqlen_offset]
//   image, audio, video hashes (in order):
//     [u8 present] (0/1); if 1: [u32 count][count * u64]
// ===========================================================================

const PREFIX_PAYLOAD_VERSION: u8 = 1;

fn write_recurrent_snapshot(out: &mut Vec<u8>, snap: &RecurrentStateSnapshot) -> Result<()> {
    // narrow()/gather views may be non-contiguous; safetensors needs a flat copy.
    let conv = snap.conv_state.contiguous()?;
    let recurrent = snap.recurrent_state.contiguous()?;
    let blob = safetensors::serialize(
        [
            ("conv".to_string(), &conv),
            ("recurrent".to_string(), &recurrent),
        ],
        None,
    )
    .map_err(|e| hanzo_ml::Error::Msg(format!("prefix payload: recurrent encode failed: {e}")))?;
    out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    out.extend_from_slice(&blob);
    out.extend_from_slice(&(snap.seqlen_offset as u64).to_le_bytes());
    Ok(())
}

fn read_recurrent_snapshot(
    r: &mut ByteReader<'_>,
    device: &Device,
    limits: &RestoreLimits,
) -> Result<RecurrentStateSnapshot> {
    let blob_len = usize_from_u64(r.read_u64()?)?;
    if blob_len as u64 > limits.max_payload_bytes {
        hanzo_ml::bail!(
            "prefix payload: recurrent blob_len {blob_len} exceeds budget {}",
            limits.max_payload_bytes
        );
    }
    let blob = r.read_bytes(blob_len)?;
    let mut map = hanzo_ml::safetensors::load_buffer(blob, device)?;
    let conv_state = map
        .remove("conv")
        .ok_or_else(|| hanzo_ml::Error::Msg("prefix payload: missing conv state".into()))?;
    let recurrent_state = map
        .remove("recurrent")
        .ok_or_else(|| hanzo_ml::Error::Msg("prefix payload: missing recurrent state".into()))?;
    let seqlen_offset = usize_from_u64(r.read_u64()?)?;
    Ok(RecurrentStateSnapshot {
        conv_state,
        recurrent_state,
        seqlen_offset,
    })
}

fn write_hashes(out: &mut Vec<u8>, hashes: Option<&[u64]>) {
    match hashes {
        Some(h) => {
            out.push(1u8);
            out.extend_from_slice(&(h.len() as u32).to_le_bytes());
            for &x in h {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        None => out.push(0u8),
    }
}

fn read_hashes(r: &mut ByteReader<'_>, limits: &RestoreLimits) -> Result<Option<Vec<u64>>> {
    match r.read_u8()? {
        0 => Ok(None),
        1 => {
            let n = usize_from_u64(u64::from(r.read_u32()?))?;
            // A prefix can't carry more modality items than it has tokens.
            if n > limits.max_seq_len {
                hanzo_ml::bail!(
                    "prefix payload: hash count {n} exceeds model max context {}",
                    limits.max_seq_len
                );
            }
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.read_u64()?);
            }
            Ok(Some(v))
        }
        other => hanzo_ml::bail!("prefix payload: bad hash flag {other}"),
    }
}

fn usize_from_u64(v: u64) -> Result<usize> {
    usize::try_from(v)
        .map_err(|_| hanzo_ml::Error::Msg("prefix payload: length exceeds usize".into()))
}

/// Encode tokens as little-endian `u32` bytes — the content the disk key hashes
/// and the `rendered_text` field carries so a cold start can recover the prefix.
fn tokens_to_le_bytes(toks: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(toks.len() * 4);
    for &t in toks {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

/// Inverse of [`tokens_to_le_bytes`]; `None` if the byte length is not a multiple
/// of 4 (a corrupt or foreign entry).
fn tokens_from_le_bytes(bytes: &[u8]) -> Option<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
    )
}

/// Serialize a prefix (tagged with the live-model `fingerprint`) and persist it
/// under the content key of its token bytes.
fn spill_prefix(
    disk: &DiskKvCache,
    fingerprint: u64,
    toks: &[u32],
    element: &CacheElement,
) -> Result<()> {
    let payload = element.to_bytes(fingerprint)?;
    let token_bytes = tokens_to_le_bytes(toks);
    let key = key_for_bytes(&token_bytes);
    let header = KvcHeader::new(
        0,
        SaveReason::Evict,
        u32::try_from(toks.len()).unwrap_or(u32::MAX),
        0,
    );
    disk.save(&key, header, &token_bytes, &payload)
        .map_err(|e| hanzo_ml::Error::Msg(format!("kv spill save failed: {e}")))
}

#[cfg(test)]
mod tests {
    use hanzo_ml::{DType, Device, Tensor};

    use super::{
        key_for_bytes, tokens_to_le_bytes, CacheElement, DiskKvCache, MatchingCache,
        PrefixCacheManagerV2,
    };
    use crate::kv_cache::{KvCache, RestoreLimits, RotatingCache, SingleCache};

    /// Generous, fingerprint-0 limits for the happy-path disk tests.
    fn test_limits() -> RestoreLimits {
        RestoreLimits {
            fingerprint: 0,
            max_layers: 4096,
            max_seq_len: 1 << 20,
            max_payload_bytes: 1 << 32,
        }
    }

    fn make_cache_tensor(len: usize) -> hanzo_ml::Result<Tensor> {
        Tensor::zeros((1, 1, len, 1), DType::F32, &Device::Cpu)
    }

    fn make_rotating_kv_cache(
        logical_len: usize,
        sliding_window: usize,
    ) -> hanzo_ml::Result<KvCache> {
        let src = make_cache_tensor(logical_len)?;
        let mut k = RotatingCache::new(2, sliding_window, sliding_window);
        let mut v = RotatingCache::new(2, sliding_window, sliding_window);
        let _ = k.append(&src)?;
        let _ = v.append(&src)?;
        Ok(KvCache::Rotating { k, v })
    }

    fn make_normal_kv_cache(logical_len: usize) -> hanzo_ml::Result<KvCache> {
        let src = make_cache_tensor(logical_len)?;
        let mut k = SingleCache::new(2, logical_len, logical_len);
        let mut v = SingleCache::new(2, logical_len, logical_len);
        k.append(&src)?;
        v.append(&src)?;
        Ok(KvCache::Normal { k, v })
    }

    #[test]
    fn skips_rolled_over_rotating_candidate_that_cannot_rewind() -> hanzo_ml::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);

        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into(),
            CacheElement {
                cache: vec![Some(make_rotating_kv_cache(10, 4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 9].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(6)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 99],
            None,
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 5);
                assert_eq!(toks, vec![6, 7, 99]);
            }
            None => panic!("expected a shorter valid prefix-cache hit"),
        }

        Ok(())
    }

    #[test]
    fn allows_exact_extension_from_rolled_over_rotating_cache() -> hanzo_ml::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);

        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into(),
            CacheElement {
                cache: vec![Some(make_rotating_kv_cache(10, 4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            None,
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 10);
                assert_eq!(toks, vec![11]);
            }
            None => panic!("expected exact-extension prefix-cache hit"),
        }

        Ok(())
    }

    fn element_with(layers: Vec<Option<KvCache>>) -> CacheElement {
        CacheElement {
            cache: layers,
            recurrent_snapshots: None,
            audio_hashes: None,
            image_hashes: None,
            video_hashes: None,
        }
    }

    fn vals(kv: &KvCache, get_v: bool) -> Vec<f32> {
        let t = if get_v { kv.v() } else { kv.k() }.unwrap().unwrap();
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn cache_element_round_trips_bytes() -> hanzo_ml::Result<()> {
        let element = CacheElement {
            cache: vec![
                Some(make_normal_kv_cache(6)?),
                None,
                Some(make_normal_kv_cache(3)?),
            ],
            recurrent_snapshots: None,
            audio_hashes: Some(vec![7, 8]),
            image_hashes: Some(vec![42]),
            video_hashes: None,
        };

        let bytes = element.to_bytes(0)?;
        let back = CacheElement::from_bytes(&bytes, &Device::Cpu, &test_limits())?;

        assert_eq!(back.cache.len(), 3);
        assert!(back.cache[1].is_none(), "None layer preserved");
        assert_eq!(back.image_hashes, Some(vec![42]));
        assert_eq!(back.audio_hashes, Some(vec![7, 8]));
        assert_eq!(back.video_hashes, None);

        for (a, b) in element.cache.iter().zip(&back.cache) {
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.current_seq_len(), b.current_seq_len());
                    assert_eq!(vals(a, false), vals(b, false));
                    assert_eq!(vals(a, true), vals(b, true));
                }
                (None, None) => {}
                _ => panic!("layer presence mismatch across round-trip"),
            }
        }
        Ok(())
    }

    #[test]
    fn evict_spills_to_disk_and_load_restores() -> hanzo_ml::Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let disk = DiskKvCache::new(tmp.path(), 64).unwrap();
        // n_on_device = 0 => the single prefix overflows and must spill.
        let mut mgr =
            PrefixCacheManagerV2::new(0, false, false).with_disk_cache(disk, test_limits());

        let toks = vec![10u32, 20, 30, 40];
        let original = element_with(vec![Some(make_normal_kv_cache(4)?)]);
        mgr.caches.insert(toks.clone().into(), original.clone());

        assert_eq!(mgr.evict_caches()?, 1, "the overflow prefix is evicted");
        assert!(mgr.caches.is_empty(), "evicted prefix left memory");

        // The spill file exists on disk under the token content key.
        let key = key_for_bytes(&tokens_to_le_bytes(&toks));
        assert!(
            tmp.path().join(format!("{key}.kv")).exists(),
            "spill file present on disk"
        );

        // load_from_disk restores a cache whose tensors equal the original.
        let (restored_toks, restored) = mgr
            .load_from_disk(&key, &Device::Cpu)?
            .expect("restore hit");
        assert_eq!(restored_toks, toks, "tokens recovered from disk key");
        let orig_kv = original.cache[0].as_ref().unwrap();
        let rest_kv = restored.cache[0].as_ref().unwrap();
        assert_eq!(orig_kv.current_seq_len(), rest_kv.current_seq_len());
        assert_eq!(vals(orig_kv, false), vals(rest_kv, false));
        assert_eq!(vals(orig_kv, true), vals(rest_kv, true));
        Ok(())
    }

    #[test]
    fn evict_without_disk_keeps_path_unchanged() -> hanzo_ml::Result<()> {
        // No disk store: the device-gated drop path runs. CPU-resident test caches
        // are never device-resident, so nothing is evicted — byte-identical to the
        // original behavior.
        let mut mgr = PrefixCacheManagerV2::new(0, false, false);
        mgr.caches.insert(
            vec![1u32, 2, 3].into(),
            element_with(vec![Some(make_normal_kv_cache(3)?)]),
        );
        assert_eq!(mgr.evict_caches()?, 0);
        assert_eq!(
            mgr.caches.len(),
            1,
            "CPU prefix retained without a disk store"
        );
        Ok(())
    }

    #[test]
    fn restore_from_disk_repopulates_memory() -> hanzo_ml::Result<()> {
        let tmp = tempfile::tempdir().unwrap();

        // First manager spills two prefixes to disk on eviction.
        {
            let disk = DiskKvCache::new(tmp.path(), 64).unwrap();
            let mut mgr =
                PrefixCacheManagerV2::new(0, false, false).with_disk_cache(disk, test_limits());
            mgr.caches.insert(
                vec![1u32, 2, 3].into(),
                element_with(vec![Some(make_normal_kv_cache(3)?)]),
            );
            mgr.caches.insert(
                vec![9u32, 8].into(),
                element_with(vec![Some(make_normal_kv_cache(2)?)]),
            );
            assert_eq!(mgr.evict_caches()?, 2);
        }

        // A fresh manager (cold start) restores them from disk and serves matches.
        let disk = DiskKvCache::new(tmp.path(), 64).unwrap();
        let mut mgr2 =
            PrefixCacheManagerV2::new(8, false, false).with_disk_cache(disk, test_limits());
        assert_eq!(mgr2.restore_from_disk(&Device::Cpu)?, 2);

        let hit = mgr2.search_for_matching_cache(&[1, 2, 3, 99], None, None, None)?;
        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 3);
                assert_eq!(toks, vec![99]);
            }
            None => panic!("restored prefix should match"),
        }
        Ok(())
    }

    // --- Negative / fuzz: crafted payloads must Err or skip, never OOM/panic. ---

    #[test]
    fn prefix_oversized_n_layers_is_rejected_not_allocated() {
        // version + matching fingerprint(0) + absurd n_layers. Must error at the
        // bound check before any `Vec::with_capacity`/allocation.
        let mut bytes = vec![super::PREFIX_PAYLOAD_VERSION];
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(CacheElement::from_bytes(&bytes, &Device::Cpu, &test_limits()).is_err());
    }

    #[test]
    fn prefix_foreign_fingerprint_is_rejected() -> hanzo_ml::Result<()> {
        let element = element_with(vec![Some(make_normal_kv_cache(3)?)]);
        let bytes = element.to_bytes(0xDEAD_BEEF)?;
        // Live model fingerprint 0 != payload's 0xDEADBEEF -> reject.
        assert!(CacheElement::from_bytes(&bytes, &Device::Cpu, &test_limits()).is_err());
        // Same bytes with the matching fingerprint deserialize fine.
        let matching = RestoreLimits {
            fingerprint: 0xDEAD_BEEF,
            ..test_limits()
        };
        assert!(CacheElement::from_bytes(&bytes, &Device::Cpu, &matching).is_ok());
        Ok(())
    }

    #[test]
    fn restore_skips_foreign_model_spill() -> hanzo_ml::Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        // Spill under "model A" (fingerprint 111).
        {
            let disk = DiskKvCache::new(tmp.path(), 64).unwrap();
            let limits_a = RestoreLimits {
                fingerprint: 111,
                ..test_limits()
            };
            let mut mgr =
                PrefixCacheManagerV2::new(0, false, false).with_disk_cache(disk, limits_a);
            mgr.caches.insert(
                vec![1u32, 2, 3].into(),
                element_with(vec![Some(make_normal_kv_cache(3)?)]),
            );
            assert_eq!(mgr.evict_caches()?, 1);
        }
        // Cold start as "model B" (fingerprint 222): the entry must be skipped,
        // never restored, and never panic/OOM.
        let disk = DiskKvCache::new(tmp.path(), 64).unwrap();
        let limits_b = RestoreLimits {
            fingerprint: 222,
            ..test_limits()
        };
        let mut mgr_b = PrefixCacheManagerV2::new(8, false, false).with_disk_cache(disk, limits_b);
        assert_eq!(
            mgr_b.restore_from_disk(&Device::Cpu)?,
            0,
            "foreign-model KV must not restore"
        );
        assert!(mgr_b
            .search_for_matching_cache(&[1, 2, 3, 9], None, None, None)?
            .is_none());
        Ok(())
    }
}
