// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Crash-safe content-addressed checkpoint storage backends:
//! - [`FsCheckpointStore`]: real OS file-backed store with temp-file + `sync_all()` + atomic rename,
//!   physical per-`(tenant_id, job_id)` directory partitioning, CRC32C + SHA-256 verification, and GC.
//! - [`MemoryCheckpointStore`]: deterministic in-memory store sharing the identical wire format and
//!   verification pipeline for high-speed unit/concurrency tests.
//! - [`StorageTierModel`]: analytical NVMe & Hanzo S3-VFS transfer latency projections computed from
//!   exact measured I/O counters.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::crypto::{crc32c, sha256, Digest};
use crate::error::{GcrError, GcrResult};
use crate::vram::Namespace;

/// Magic header for encoded page chunk blobs (`"HGCR"`).
pub const CHUNK_MAGIC: [u8; 4] = *b"HGCR";
/// Current chunk wire format version.
pub const CHUNK_VERSION: u8 = 1;
/// Chunk header size in bytes: magic(4) + ver(1) + codec(1) + reserved(2) + raw_len(4) + stored_len(4) + crc32c(4) + sha256(32) = 52.
pub const CHUNK_HEADER_LEN: usize = 52;

/// Magic header for encoded checkpoint manifests (`"GCRM"`).
pub const MANIFEST_MAGIC: [u8; 4] = *b"GCRM";
/// Current manifest wire format version.
pub const MANIFEST_VERSION: u8 = 1;
/// Manifest envelope header size: magic(4) + ver(1) + reserved(3) + body_len(4) + crc32c(4) + sha256(32) = 48.
pub const MANIFEST_HEADER_LEN: usize = 48;

/// Lossless chunk payload encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChunkCodec {
    /// Raw uncompressed page bytes.
    Raw = 0,
    /// Zero-run compressed encoding (used only when strictly smaller than raw).
    ZeroRle = 1,
}

impl ChunkCodec {
    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Raw),
            1 => Some(Self::ZeroRle),
            _ => None,
        }
    }
}

/// Encoded chunk metadata returned on write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredChunkInfo {
    /// SHA-256 digest of the raw uncompressed page bytes.
    pub digest: Digest,
    /// Raw page byte length.
    pub raw_len: u32,
    /// Total wire bytes of the framed chunk (header + payload).
    pub wire_len: u32,
    /// CRC32C of the wire payload.
    pub payload_crc32c: u32,
    /// `true` if the chunk already existed in this namespace and was deduplicated.
    pub deduplicated: bool,
}

/// Exact I/O counters accumulated by a checkpoint store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoStats {
    /// Total physical bytes written to chunk and manifest files.
    pub bytes_written: u64,
    /// Total physical bytes read from chunk and manifest files.
    pub bytes_read: u64,
    /// Number of physical chunk files written (excluding dedup hits).
    pub chunks_written: u64,
    /// Number of chunk writes skipped because the content-addressed chunk already existed in-namespace.
    pub chunks_dedup_skipped: u64,
    /// Number of chunk files read.
    pub chunks_read: u64,
    /// Number of manifest files written.
    pub manifests_written: u64,
    /// Number of manifest files read.
    pub manifests_read: u64,
    /// Number of `File::sync_all()` durability flushes executed.
    pub fsync_calls: u64,
    /// Number of unreferenced chunks deleted by garbage collection.
    pub chunks_gc_deleted: u64,
    /// Bytes reclaimed by garbage collection.
    pub bytes_gc_reclaimed: u64,
}

impl IoStats {
    /// Computes the delta `self - earlier` between two I/O snapshots.
    pub fn delta_since(&self, earlier: &Self) -> Self {
        Self {
            bytes_written: self.bytes_written.saturating_sub(earlier.bytes_written),
            bytes_read: self.bytes_read.saturating_sub(earlier.bytes_read),
            chunks_written: self.chunks_written.saturating_sub(earlier.chunks_written),
            chunks_dedup_skipped: self
                .chunks_dedup_skipped
                .saturating_sub(earlier.chunks_dedup_skipped),
            chunks_read: self.chunks_read.saturating_sub(earlier.chunks_read),
            manifests_written: self
                .manifests_written
                .saturating_sub(earlier.manifests_written),
            manifests_read: self.manifests_read.saturating_sub(earlier.manifests_read),
            fsync_calls: self.fsync_calls.saturating_sub(earlier.fsync_calls),
            chunks_gc_deleted: self
                .chunks_gc_deleted
                .saturating_sub(earlier.chunks_gc_deleted),
            bytes_gc_reclaimed: self
                .bytes_gc_reclaimed
                .saturating_sub(earlier.bytes_gc_reclaimed),
        }
    }
}

/// Analytical storage-tier transfer model (NVMe local scratch vs Hanzo S3-VFS object tier).
#[derive(Debug, Clone, Copy)]
pub struct StorageTierModel {
    /// Tier name (e.g., `"nvme_gen4"` or `"hanzo_s3_vfs"`).
    pub name: &'static str,
    /// Sustained bandwidth in bytes/sec.
    pub bandwidth_bytes_per_sec: f64,
    /// Per-object / per-manifest RPC/syscall latency in milliseconds.
    pub per_op_latency_ms: f64,
}

impl StorageTierModel {
    /// Local PCIe Gen4 NVMe scratch tier (3.5 GB/s sustained, 0.08 ms/op).
    pub const NVME_GEN4: Self = Self {
        name: "nvme_gen4",
        bandwidth_bytes_per_sec: 3.5e9,
        per_op_latency_ms: 0.08,
    };

    /// Canonical Hanzo S3-VFS object tier (800 MB/s aggregate stream, 4.0 ms manifest/batch RTT).
    pub const HANZO_S3_VFS: Self = Self {
        name: "hanzo_s3_vfs",
        bandwidth_bytes_per_sec: 0.8e9,
        per_op_latency_ms: 4.0,
    };

    /// Projects write time in milliseconds given measured bytes and sequential manifest/object ops.
    /// Chunks are assumed pipelined in batches of `concurrency`, while manifests are serial.
    pub fn projected_write_ms(&self, delta: &IoStats, concurrency: usize) -> f64 {
        let conc = concurrency.max(1) as f64;
        let xfer_ms = (delta.bytes_written as f64 / self.bandwidth_bytes_per_sec) * 1000.0;
        let chunk_waves = (delta.chunks_written as f64 / conc).ceil();
        let op_ms = (chunk_waves + delta.manifests_written as f64) * self.per_op_latency_ms;
        xfer_ms + op_ms
    }

    /// Projects restore read time in milliseconds given measured bytes, manifest chain hops, and workers.
    pub fn projected_read_ms(&self, delta: &IoStats, concurrency: usize) -> f64 {
        let conc = concurrency.max(1) as f64;
        let xfer_ms = (delta.bytes_read as f64 / self.bandwidth_bytes_per_sec) * 1000.0;
        // Manifests in a delta chain must be walked sequentially (parent link discovery),
        // while chunk fetches are parallelized across `concurrency` workers.
        let chunk_waves = (delta.chunks_read as f64 / conc).ceil();
        let op_ms = (delta.manifests_read as f64 + chunk_waves) * self.per_op_latency_ms;
        xfer_ms + op_ms
    }
}

/// Encodes raw page bytes into a self-verifying `HGCR` wire frame with CRC32C and SHA-256.
pub fn encode_chunk_frame(raw_page: &[u8], allow_zero_rle: bool) -> (Vec<u8>, Digest, u32) {
    let digest = sha256(raw_page);
    let (codec, payload): (ChunkCodec, Vec<u8>) = if allow_zero_rle {
        let rle = encode_zero_rle(raw_page);
        if rle.len() < raw_page.len() {
            (ChunkCodec::ZeroRle, rle)
        } else {
            (ChunkCodec::Raw, raw_page.to_vec())
        }
    } else {
        (ChunkCodec::Raw, raw_page.to_vec())
    };

    let payload_crc = crc32c(&payload);
    let mut frame = Vec::with_capacity(CHUNK_HEADER_LEN + payload.len());
    frame.extend_from_slice(&CHUNK_MAGIC);
    frame.push(CHUNK_VERSION);
    frame.push(codec as u8);
    frame.extend_from_slice(&[0u8; 2]);
    frame.extend_from_slice(&(raw_page.len() as u32).to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload_crc.to_le_bytes());
    frame.extend_from_slice(digest.as_bytes());
    frame.extend_from_slice(&payload);
    (frame, digest, payload_crc)
}

/// Decodes and cryptographically verifies an `HGCR` wire frame against `expected_digest`.
pub fn decode_and_verify_chunk_frame(
    frame: &[u8],
    expected_digest: &Digest,
    expected_page_size: usize,
) -> GcrResult<Vec<u8>> {
    let chunk_id = expected_digest.short_hex();
    if frame.len() < CHUNK_HEADER_LEN {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!(
                "truncated chunk header: {} bytes < {CHUNK_HEADER_LEN}",
                frame.len()
            ),
        });
    }
    if frame[0..4] != CHUNK_MAGIC {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: "invalid chunk magic header".to_string(),
        });
    }
    if frame[4] != CHUNK_VERSION {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!("unsupported chunk version {}", frame[4]),
        });
    }
    let codec = ChunkCodec::from_u8(frame[5]).ok_or_else(|| GcrError::CorruptedChunk {
        chunk_id: chunk_id.clone(),
        reason: format!("unknown chunk codec {}", frame[5]),
    })?;
    let raw_len = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]) as usize;
    let stored_len = u32::from_le_bytes([frame[12], frame[13], frame[14], frame[15]]) as usize;
    let header_crc = u32::from_le_bytes([frame[16], frame[17], frame[18], frame[19]]);
    let mut header_digest_bytes = [0u8; 32];
    header_digest_bytes.copy_from_slice(&frame[20..52]);
    let header_digest = Digest(header_digest_bytes);

    if raw_len != expected_page_size {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!("raw length mismatch: header={raw_len}, expected={expected_page_size}"),
        });
    }
    if frame.len() != CHUNK_HEADER_LEN + stored_len {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!(
                "truncated or trailing chunk bytes: file={}, expected={}",
                frame.len(),
                CHUNK_HEADER_LEN + stored_len
            ),
        });
    }
    let payload = &frame[CHUNK_HEADER_LEN..];
    let actual_crc = crc32c(payload);
    if actual_crc != header_crc {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!(
                "CRC32C mismatch: header={header_crc:#010x}, actual={actual_crc:#010x}"
            ),
        });
    }
    if !header_digest.ct_eq(expected_digest) {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: "chunk header SHA-256 does not match manifest reference".to_string(),
        });
    }

    let decoded = match codec {
        ChunkCodec::Raw => payload.to_vec(),
        ChunkCodec::ZeroRle => {
            decode_zero_rle(payload, raw_len).map_err(|reason| GcrError::CorruptedChunk {
                chunk_id: chunk_id.clone(),
                reason,
            })?
        }
    };
    if decoded.len() != expected_page_size {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!(
                "decoded page length {} != expected {expected_page_size}",
                decoded.len()
            ),
        });
    }

    let actual_digest = sha256(&decoded);
    if !actual_digest.ct_eq(expected_digest) {
        return Err(GcrError::CorruptedChunk {
            chunk_id,
            reason: format!(
                "SHA-256 mismatch: expected {}, actual {}",
                expected_digest.short_hex(),
                actual_digest.short_hex()
            ),
        });
    }
    Ok(decoded)
}

/// Simple deterministic lossless zero-run encoding (`0x00` followed by `u8` run length 1..=255).
fn encode_zero_rle(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2);
    let mut i = 0;
    while i < input.len() {
        if input[i] == 0 {
            let mut run = 1usize;
            while i + run < input.len() && input[i + run] == 0 && run < 255 {
                run += 1;
            }
            out.push(0);
            out.push(run as u8);
            i += run;
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
    out
}

fn decode_zero_rle(input: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(expected_len);
    let mut i = 0;
    while i < input.len() {
        if input[i] == 0 {
            if i + 1 >= input.len() {
                return Err("truncated zero-RLE escape".to_string());
            }
            let run = input[i + 1] as usize;
            if run == 0 {
                return Err("invalid zero-length RLE run".to_string());
            }
            if out.len() + run > expected_len {
                return Err("zero-RLE overflow past page size".to_string());
            }
            out.resize(out.len() + run, 0);
            i += 2;
        } else {
            if out.len() >= expected_len {
                return Err("zero-RLE literal overflow past page size".to_string());
            }
            out.push(input[i]);
            i += 1;
        }
    }
    if out.len() != expected_len {
        return Err(format!(
            "zero-RLE underflow: decoded {} < expected {expected_len}",
            out.len()
        ));
    }
    Ok(out)
}

/// Wraps a canonical serialized manifest body in a self-verifying `GCRM` envelope.
pub fn encode_manifest_envelope(body: &[u8]) -> (Vec<u8>, Digest) {
    let body_digest = sha256(body);
    let body_crc = crc32c(body);
    let mut out = Vec::with_capacity(MANIFEST_HEADER_LEN + body.len());
    out.extend_from_slice(&MANIFEST_MAGIC);
    out.push(MANIFEST_VERSION);
    out.extend_from_slice(&[0u8; 3]);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body_crc.to_le_bytes());
    out.extend_from_slice(body_digest.as_bytes());
    out.extend_from_slice(body);
    (out, body_digest)
}

/// Unwraps and cryptographically verifies a `GCRM` manifest envelope.
pub fn decode_manifest_envelope<'a>(
    checkpoint_id: &str,
    envelope: &'a [u8],
) -> GcrResult<(&'a [u8], Digest)> {
    if envelope.len() < MANIFEST_HEADER_LEN {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: checkpoint_id.to_string(),
            reason: format!(
                "truncated manifest header: {} < {MANIFEST_HEADER_LEN}",
                envelope.len()
            ),
        });
    }
    if envelope[0..4] != MANIFEST_MAGIC {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: checkpoint_id.to_string(),
            reason: "invalid manifest magic header".to_string(),
        });
    }
    if envelope[4] != MANIFEST_VERSION {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: checkpoint_id.to_string(),
            reason: format!("unsupported manifest version {}", envelope[4]),
        });
    }
    let body_len =
        u32::from_le_bytes([envelope[8], envelope[9], envelope[10], envelope[11]]) as usize;
    let header_crc = u32::from_le_bytes([envelope[12], envelope[13], envelope[14], envelope[15]]);
    let mut digest_bytes = [0u8; 32];
    digest_bytes.copy_from_slice(&envelope[16..48]);
    let header_digest = Digest(digest_bytes);

    if envelope.len() != MANIFEST_HEADER_LEN + body_len {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: checkpoint_id.to_string(),
            reason: format!(
                "manifest byte length mismatch: file={}, expected={}",
                envelope.len(),
                MANIFEST_HEADER_LEN + body_len
            ),
        });
    }
    let body = &envelope[MANIFEST_HEADER_LEN..];
    let actual_crc = crc32c(body);
    if actual_crc != header_crc {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: checkpoint_id.to_string(),
            reason: format!(
                "manifest CRC32C mismatch: header={header_crc:#010x}, actual={actual_crc:#010x}"
            ),
        });
    }
    let actual_digest = sha256(body);
    if !actual_digest.ct_eq(&header_digest) {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: checkpoint_id.to_string(),
            reason: "manifest SHA-256 integrity mismatch".to_string(),
        });
    }
    Ok((body, actual_digest))
}

/// Abstract crash-safe tenant-isolated checkpoint store.
pub trait CheckpointStore: Send + Sync {
    /// Persists a content-addressed page chunk within `namespace`.
    ///
    /// If `dedup` is `true` and a chunk with the exact same digest already exists in `namespace`,
    /// skips writing and increments `chunks_dedup_skipped`.
    fn put_chunk(
        &self,
        namespace: &Namespace,
        raw_page: &[u8],
        allow_zero_rle: bool,
        dedup: bool,
    ) -> GcrResult<StoredChunkInfo>;

    /// Reads, decodes, and SHA-256 verifies a chunk within `namespace`.
    fn get_chunk(
        &self,
        namespace: &Namespace,
        digest: &Digest,
        expected_page_size: usize,
    ) -> GcrResult<Vec<u8>>;

    /// Atomically persists a framed manifest blob for `checkpoint_id` within `namespace`.
    fn put_manifest(
        &self,
        namespace: &Namespace,
        checkpoint_id: &str,
        envelope: &[u8],
    ) -> GcrResult<()>;

    /// Reads a raw framed manifest blob for `checkpoint_id` within `namespace`.
    fn get_manifest(&self, namespace: &Namespace, checkpoint_id: &str) -> GcrResult<Vec<u8>>;

    /// Deletes any chunks in `namespace` whose digest is not in `live_digests`, returning `(count, bytes)`.
    fn gc_unreferenced_chunks(
        &self,
        namespace: &Namespace,
        live_digests: &HashSet<Digest>,
    ) -> GcrResult<(u64, u64)>;

    /// Returns total on-disk/stored bytes currently occupied by `namespace` (chunks + manifests).
    fn namespace_storage_bytes(&self, namespace: &Namespace) -> GcrResult<u64>;

    /// Returns a snapshot of cumulative I/O statistics.
    fn io_stats(&self) -> IoStats;
}

/// Real OS file-backed crash-safe checkpoint store.
///
/// Layout:
/// `<root>/tenants/<tenant_hash>/<job_hash>/chunks/<chunk_sha256>.chunk`
/// `<root>/tenants/<tenant_hash>/<job_hash>/manifests/<checkpoint_id>.manifest`
///
/// Every write uses temp-file creation -> `write_all()` -> `File::sync_all()` -> atomic `fs::rename()`.
#[derive(Debug)]
pub struct FsCheckpointStore {
    root: PathBuf,
    fsync_enabled: bool,
    tmp_seq: AtomicU64,
    stats: Mutex<IoStats>,
}

impl FsCheckpointStore {
    /// Opens or creates an OS file-backed checkpoint store rooted at `root`.
    pub fn new(root: impl Into<PathBuf>, fsync_enabled: bool) -> GcrResult<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("tenants"))?;
        Ok(Self {
            root,
            fsync_enabled,
            tmp_seq: AtomicU64::new(1),
            stats: Mutex::new(IoStats::default()),
        })
    }

    /// Returns the store root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Computes the isolated directory for `namespace`.
    pub fn namespace_dir(&self, namespace: &Namespace) -> PathBuf {
        self.root
            .join("tenants")
            .join(namespace.tenant_dir_key())
            .join(namespace.job_dir_key())
    }

    /// Returns the exact on-disk path of a chunk inside `namespace` (useful for corruption testing).
    pub fn chunk_path(&self, namespace: &Namespace, digest: &Digest) -> PathBuf {
        self.namespace_dir(namespace)
            .join("chunks")
            .join(format!("{}.chunk", digest.to_hex()))
    }

    /// Returns the exact on-disk path of a manifest inside `namespace` (useful for fault injection testing).
    pub fn manifest_path(&self, namespace: &Namespace, checkpoint_id: &str) -> GcrResult<PathBuf> {
        validate_checkpoint_id(checkpoint_id)?;
        Ok(self
            .namespace_dir(namespace)
            .join("manifests")
            .join(format!("{checkpoint_id}.manifest")))
    }

    fn ensure_namespace_dirs(&self, namespace: &Namespace) -> GcrResult<(PathBuf, PathBuf)> {
        let ns_dir = self.namespace_dir(namespace);
        let chunks_dir = ns_dir.join("chunks");
        let manifests_dir = ns_dir.join("manifests");
        fs::create_dir_all(&chunks_dir)?;
        fs::create_dir_all(&manifests_dir)?;
        Ok((chunks_dir, manifests_dir))
    }

    fn atomic_write_file(&self, final_path: &Path, data: &[u8]) -> GcrResult<u64> {
        let parent = final_path
            .parent()
            .ok_or_else(|| GcrError::Io("path has no parent directory".to_string()))?;
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let file_name = final_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("blob");
        let tmp_path = parent.join(format!(".{file_name}.tmp.{pid}.{seq}"));

        let write_result = (|| -> GcrResult<u64> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&tmp_path)?;
            file.write_all(data)?;
            let mut fsyncs = 0u64;
            if self.fsync_enabled {
                file.sync_all()?;
                fsyncs += 1;
            }
            drop(file);
            fs::rename(&tmp_path, final_path)?;
            if self.fsync_enabled {
                if let Ok(dir_handle) = File::open(parent) {
                    let _ = dir_handle.sync_all();
                    fsyncs += 1;
                }
            }
            Ok(fsyncs)
        })();

        if write_result.is_err() {
            let _ = fs::remove_file(&tmp_path);
        }
        write_result
    }
}

impl CheckpointStore for FsCheckpointStore {
    fn put_chunk(
        &self,
        namespace: &Namespace,
        raw_page: &[u8],
        allow_zero_rle: bool,
        dedup: bool,
    ) -> GcrResult<StoredChunkInfo> {
        let (_chunks_dir, _manifests_dir) = self.ensure_namespace_dirs(namespace)?;
        let (frame, digest, payload_crc) = encode_chunk_frame(raw_page, allow_zero_rle);
        let path = self.chunk_path(namespace, &digest);

        if dedup && path.exists() {
            let mut stats = self.stats.lock().expect("stats lock");
            stats.chunks_dedup_skipped += 1;
            return Ok(StoredChunkInfo {
                digest,
                raw_len: raw_page.len() as u32,
                wire_len: frame.len() as u32,
                payload_crc32c: payload_crc,
                deduplicated: true,
            });
        }

        let fsyncs = self.atomic_write_file(&path, &frame)?;
        {
            let mut stats = self.stats.lock().expect("stats lock");
            stats.bytes_written += frame.len() as u64;
            stats.chunks_written += 1;
            stats.fsync_calls += fsyncs;
        }
        Ok(StoredChunkInfo {
            digest,
            raw_len: raw_page.len() as u32,
            wire_len: frame.len() as u32,
            payload_crc32c: payload_crc,
            deduplicated: false,
        })
    }

    fn get_chunk(
        &self,
        namespace: &Namespace,
        digest: &Digest,
        expected_page_size: usize,
    ) -> GcrResult<Vec<u8>> {
        let path = self.chunk_path(namespace, digest);
        let bytes = fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GcrError::CorruptedChunk {
                    chunk_id: digest.short_hex(),
                    reason: "chunk file missing in namespace".to_string(),
                }
            } else {
                GcrError::Io(e.to_string())
            }
        })?;
        {
            let mut stats = self.stats.lock().expect("stats lock");
            stats.bytes_read += bytes.len() as u64;
            stats.chunks_read += 1;
        }
        decode_and_verify_chunk_frame(&bytes, digest, expected_page_size)
    }

    fn put_manifest(
        &self,
        namespace: &Namespace,
        checkpoint_id: &str,
        envelope: &[u8],
    ) -> GcrResult<()> {
        let (_chunks_dir, _manifests_dir) = self.ensure_namespace_dirs(namespace)?;
        let path = self.manifest_path(namespace, checkpoint_id)?;
        let fsyncs = self.atomic_write_file(&path, envelope)?;
        {
            let mut stats = self.stats.lock().expect("stats lock");
            stats.bytes_written += envelope.len() as u64;
            stats.manifests_written += 1;
            stats.fsync_calls += fsyncs;
        }
        Ok(())
    }

    fn get_manifest(&self, namespace: &Namespace, checkpoint_id: &str) -> GcrResult<Vec<u8>> {
        let path = self.manifest_path(namespace, checkpoint_id)?;
        let bytes = fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GcrError::CheckpointNotFound(checkpoint_id.to_string())
            } else {
                GcrError::Io(e.to_string())
            }
        })?;
        {
            let mut stats = self.stats.lock().expect("stats lock");
            stats.bytes_read += bytes.len() as u64;
            stats.manifests_read += 1;
        }
        Ok(bytes)
    }

    fn gc_unreferenced_chunks(
        &self,
        namespace: &Namespace,
        live_digests: &HashSet<Digest>,
    ) -> GcrResult<(u64, u64)> {
        let chunks_dir = self.namespace_dir(namespace).join("chunks");
        if !chunks_dir.exists() {
            return Ok((0, 0));
        }
        let mut deleted = 0u64;
        let mut reclaimed = 0u64;
        for entry in fs::read_dir(&chunks_dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(hex) = name.strip_suffix(".chunk") else {
                continue;
            };
            let Some(digest) = Digest::from_hex(hex) else {
                continue;
            };
            if !live_digests.contains(&digest) {
                let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
                if fs::remove_file(&path).is_ok() {
                    deleted += 1;
                    reclaimed += len;
                }
            }
        }
        if deleted > 0 {
            let mut stats = self.stats.lock().expect("stats lock");
            stats.chunks_gc_deleted += deleted;
            stats.bytes_gc_reclaimed += reclaimed;
        }
        Ok((deleted, reclaimed))
    }

    fn namespace_storage_bytes(&self, namespace: &Namespace) -> GcrResult<u64> {
        let ns_dir = self.namespace_dir(namespace);
        if !ns_dir.exists() {
            return Ok(0);
        }
        let mut total = 0u64;
        for sub in ["chunks", "manifests"] {
            let dir = ns_dir.join(sub);
            if !dir.exists() {
                continue;
            }
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        total += meta.len();
                    }
                }
            }
        }
        Ok(total)
    }

    fn io_stats(&self) -> IoStats {
        *self.stats.lock().expect("stats lock")
    }
}

/// Deterministic in-memory checkpoint store using the identical framing and verification pipeline.
#[derive(Debug, Default)]
pub struct MemoryCheckpointStore {
    inner: Mutex<MemInner>,
}

#[derive(Debug, Default)]
struct MemInner {
    chunks: HashMap<(Digest, Digest), Vec<u8>>,
    manifests: HashMap<(Digest, String), Vec<u8>>,
    stats: IoStats,
}

impl MemoryCheckpointStore {
    /// Creates an empty in-memory checkpoint store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fault-injection helper for unit tests: replaces raw chunk bytes directly.
    pub fn inject_raw_chunk(&self, namespace: &Namespace, digest: Digest, raw_frame: Vec<u8>) {
        let mut guard = self.inner.lock().expect("mem store lock");
        guard.chunks.insert((namespace.tag(), digest), raw_frame);
    }

    /// Fault-injection helper for unit tests: replaces raw manifest bytes directly.
    pub fn inject_raw_manifest(
        &self,
        namespace: &Namespace,
        checkpoint_id: &str,
        raw_envelope: Vec<u8>,
    ) {
        let mut guard = self.inner.lock().expect("mem store lock");
        guard
            .manifests
            .insert((namespace.tag(), checkpoint_id.to_string()), raw_envelope);
    }

    /// Fault-injection helper for unit tests: removes a manifest directly.
    pub fn remove_manifest(&self, namespace: &Namespace, checkpoint_id: &str) {
        let mut guard = self.inner.lock().expect("mem store lock");
        guard
            .manifests
            .remove(&(namespace.tag(), checkpoint_id.to_string()));
    }
}

impl CheckpointStore for MemoryCheckpointStore {
    fn put_chunk(
        &self,
        namespace: &Namespace,
        raw_page: &[u8],
        allow_zero_rle: bool,
        dedup: bool,
    ) -> GcrResult<StoredChunkInfo> {
        let (frame, digest, payload_crc) = encode_chunk_frame(raw_page, allow_zero_rle);
        let key = (namespace.tag(), digest);
        let mut guard = self.inner.lock().expect("mem store lock");
        if dedup && guard.chunks.contains_key(&key) {
            guard.stats.chunks_dedup_skipped += 1;
            return Ok(StoredChunkInfo {
                digest,
                raw_len: raw_page.len() as u32,
                wire_len: frame.len() as u32,
                payload_crc32c: payload_crc,
                deduplicated: true,
            });
        }
        guard.stats.bytes_written += frame.len() as u64;
        guard.stats.chunks_written += 1;
        guard.chunks.insert(key, frame.clone());
        Ok(StoredChunkInfo {
            digest,
            raw_len: raw_page.len() as u32,
            wire_len: frame.len() as u32,
            payload_crc32c: payload_crc,
            deduplicated: false,
        })
    }

    fn get_chunk(
        &self,
        namespace: &Namespace,
        digest: &Digest,
        expected_page_size: usize,
    ) -> GcrResult<Vec<u8>> {
        let key = (namespace.tag(), *digest);
        let bytes = {
            let mut guard = self.inner.lock().expect("mem store lock");
            let data = guard
                .chunks
                .get(&key)
                .ok_or_else(|| GcrError::CorruptedChunk {
                    chunk_id: digest.short_hex(),
                    reason: "chunk missing in namespace".to_string(),
                })?
                .clone();
            guard.stats.bytes_read += data.len() as u64;
            guard.stats.chunks_read += 1;
            data
        };
        decode_and_verify_chunk_frame(&bytes, digest, expected_page_size)
    }

    fn put_manifest(
        &self,
        namespace: &Namespace,
        checkpoint_id: &str,
        envelope: &[u8],
    ) -> GcrResult<()> {
        validate_checkpoint_id(checkpoint_id)?;
        let mut guard = self.inner.lock().expect("mem store lock");
        guard.stats.bytes_written += envelope.len() as u64;
        guard.stats.manifests_written += 1;
        guard.manifests.insert(
            (namespace.tag(), checkpoint_id.to_string()),
            envelope.to_vec(),
        );
        Ok(())
    }

    fn get_manifest(&self, namespace: &Namespace, checkpoint_id: &str) -> GcrResult<Vec<u8>> {
        validate_checkpoint_id(checkpoint_id)?;
        let mut guard = self.inner.lock().expect("mem store lock");
        let data = guard
            .manifests
            .get(&(namespace.tag(), checkpoint_id.to_string()))
            .ok_or_else(|| GcrError::CheckpointNotFound(checkpoint_id.to_string()))?
            .clone();
        guard.stats.bytes_read += data.len() as u64;
        guard.stats.manifests_read += 1;
        Ok(data)
    }

    fn gc_unreferenced_chunks(
        &self,
        namespace: &Namespace,
        live_digests: &HashSet<Digest>,
    ) -> GcrResult<(u64, u64)> {
        let ns_tag = namespace.tag();
        let mut guard = self.inner.lock().expect("mem store lock");
        let mut deleted = 0u64;
        let mut reclaimed = 0u64;
        guard.chunks.retain(|(tag, digest), bytes| {
            if *tag == ns_tag && !live_digests.contains(digest) {
                deleted += 1;
                reclaimed += bytes.len() as u64;
                false
            } else {
                true
            }
        });
        guard.stats.chunks_gc_deleted += deleted;
        guard.stats.bytes_gc_reclaimed += reclaimed;
        Ok((deleted, reclaimed))
    }

    fn namespace_storage_bytes(&self, namespace: &Namespace) -> GcrResult<u64> {
        let ns_tag = namespace.tag();
        let guard = self.inner.lock().expect("mem store lock");
        let chunk_bytes: u64 = guard
            .chunks
            .iter()
            .filter(|((tag, _), _)| *tag == ns_tag)
            .map(|(_, v)| v.len() as u64)
            .sum();
        let manifest_bytes: u64 = guard
            .manifests
            .iter()
            .filter(|((tag, _), _)| *tag == ns_tag)
            .map(|(_, v)| v.len() as u64)
            .sum();
        Ok(chunk_bytes + manifest_bytes)
    }

    fn io_stats(&self) -> IoStats {
        self.inner.lock().expect("mem store lock").stats
    }
}

/// Validates a checkpoint identifier to prevent path traversal in filesystem stores.
pub fn validate_checkpoint_id(id: &str) -> GcrResult<()> {
    if id.is_empty() || id.len() > 128 {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: id.to_string(),
            reason: "checkpoint_id length must be in 1..=128".to_string(),
        });
    }
    if id == "." || id == ".." || id.contains("..") {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: id.to_string(),
            reason: "checkpoint_id contains forbidden traversal sequence".to_string(),
        });
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err(GcrError::CorruptedManifest {
            checkpoint_id: id.to_string(),
            reason: "checkpoint_id contains forbidden characters".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_framing_detects_truncation_and_bit_flips() {
        let page = vec![0x5Au8; 4096];
        let (frame, digest, _) = encode_chunk_frame(&page, false);
        let decoded = decode_and_verify_chunk_frame(&frame, &digest, 4096).unwrap();
        assert_eq!(decoded, page);

        // Truncated frame must fail
        assert!(matches!(
            decode_and_verify_chunk_frame(&frame[..frame.len() - 1], &digest, 4096),
            Err(GcrError::CorruptedChunk { .. })
        ));

        // Flipped payload bit must fail CRC32C / SHA-256
        let mut flipped = frame.clone();
        flipped[CHUNK_HEADER_LEN + 10] ^= 0x01;
        assert!(matches!(
            decode_and_verify_chunk_frame(&flipped, &digest, 4096),
            Err(GcrError::CorruptedChunk { .. })
        ));
    }

    #[test]
    fn zero_rle_roundtrips_sparse_and_dense_pages() {
        let mut sparse = vec![0u8; 4096];
        sparse[100] = 0xAB;
        sparse[2000] = 0xCD;
        let (frame, digest, _) = encode_chunk_frame(&sparse, true);
        assert!(frame.len() < 100);
        let decoded = decode_and_verify_chunk_frame(&frame, &digest, 4096).unwrap();
        assert_eq!(decoded, sparse);
    }
}
