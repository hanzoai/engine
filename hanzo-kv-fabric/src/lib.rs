// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Epoch-fenced KV block directory and framed async TCP KV transport (`HKV1`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

pub mod segment;
pub use segment::{salted_segment_hash, SegmentDirectory, SegmentLocation, SegmentMeta};

const WIRE_MAGIC: [u8; 4] = *b"HKV1";
const OP_PULL: u8 = 1;
const OP_PUSH: u8 = 2;

const STATUS_OK: u8 = 0;
const STATUS_NOT_FOUND: u8 = 1;
const STATUS_STALE_EPOCH: u8 = 2;
const STATUS_CHECKSUM_MISMATCH: u8 = 3;
const STATUS_ERR: u8 = 4;

/// Maximum single KV block frame size accepted on the wire (64 MiB).
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvFabricError {
    #[error("stale epoch for worker {worker_id}: got {got}, current {current}")]
    StaleEpoch {
        worker_id: String,
        got: u64,
        current: u64,
    },
    #[error("block {block_hash:#018x} not found")]
    BlockNotFound { block_hash: u64 },
    #[error("checksum mismatch on block {block_hash:#018x}")]
    ChecksumMismatch { block_hash: u64 },
    #[error("segment {segment_hash:#018x} covers no tokens")]
    EmptySegment { segment_hash: u64 },
    #[error("segment {segment_hash:#018x} incomplete: {num_computed_tokens} of {num_tokens} tokens computed")]
    IncompleteSegment {
        segment_hash: u64,
        num_computed_tokens: u64,
        num_tokens: u64,
    },
    #[error("frame size {len} exceeds maximum {MAX_FRAME_BYTES}")]
    FrameTooLarge { len: usize },
    #[error("invalid wire header or status")]
    InvalidFrame,
    #[error("io error: {0}")]
    Io(String),
}

impl From<std::io::Error> for KvFabricError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

/// Compute a 32-byte SHA-256 integrity digest for a serialized KV cache block.
pub fn compute_block_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    hasher.finalize().into()
}

/// Metadata for one KV cache block published by a worker replica.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBlockMeta {
    pub block_hash: u64,
    pub num_bytes: u64,
    pub sha256: [u8; 32],
}

/// Fresh replica owner record stored in [`EpochFencedKvDirectory`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockLocation {
    pub worker_id: String,
    pub peer_addr: SocketAddr,
    pub epoch: u64,
    pub block_hash: u64,
    pub num_bytes: u64,
    pub sha256: [u8; 32],
}

/// Epoch-fenced directory mapping block hashes to active replica owners.
#[derive(Debug, Default)]
pub struct EpochFencedKvDirectory {
    worker_epochs: RwLock<FxHashMap<String, u64>>,
    blocks: RwLock<FxHashMap<u64, HashMap<String, BlockLocation>>>,
}

impl EpochFencedKvDirectory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or advance a worker's active epoch. Rejects regressive epochs.
    pub fn register_worker_epoch(&self, worker_id: &str, epoch: u64) -> Result<u64, KvFabricError> {
        let mut epochs = self.worker_epochs.write();
        let entry = epochs.entry(worker_id.to_string()).or_insert(epoch);
        if epoch < *entry {
            return Err(KvFabricError::StaleEpoch {
                worker_id: worker_id.to_string(),
                got: epoch,
                current: *entry,
            });
        }
        *entry = epoch;
        Ok(epoch)
    }

    /// Fence `worker_id` at least up to `min_new_epoch` and purge its older blocks.
    pub fn fence_epoch(&self, worker_id: &str, min_new_epoch: u64) -> u64 {
        let next = {
            let mut epochs = self.worker_epochs.write();
            let entry = epochs.entry(worker_id.to_string()).or_insert(0);
            let next = (*entry + 1).max(min_new_epoch);
            *entry = next;
            next
        };
        self.purge_worker_blocks_below_epoch(worker_id, next);
        next
    }

    /// Return the worker's current fenced epoch, if registered.
    pub fn current_epoch(&self, worker_id: &str) -> Option<u64> {
        self.worker_epochs.read().get(worker_id).copied()
    }

    /// Publish block ownership for `worker_id` under `epoch`.
    /// Rejects stale epochs if the worker is fenced at a higher epoch.
    pub fn publish_blocks(
        &self,
        worker_id: &str,
        epoch: u64,
        peer_addr: SocketAddr,
        blocks: &[KvBlockMeta],
    ) -> Result<(), KvFabricError> {
        let current = self.register_worker_epoch(worker_id, epoch)?;
        debug_assert!(epoch >= current);

        let mut dir_blocks = self.blocks.write();
        for meta in blocks {
            let loc = BlockLocation {
                worker_id: worker_id.to_string(),
                peer_addr,
                epoch,
                block_hash: meta.block_hash,
                num_bytes: meta.num_bytes,
                sha256: meta.sha256,
            };
            dir_blocks
                .entry(meta.block_hash)
                .or_default()
                .insert(worker_id.to_string(), loc);
        }
        Ok(())
    }

    /// Look up active replica owners for `block_hash`, pruning stale-epoch entries.
    pub fn lookup_fresh_owners(&self, block_hash: u64) -> Vec<BlockLocation> {
        let epochs = self.worker_epochs.read();
        let mut dir_blocks = self.blocks.write();
        let Some(owners) = dir_blocks.get_mut(&block_hash) else {
            return Vec::new();
        };
        owners.retain(|worker_id, loc| {
            epochs
                .get(worker_id)
                .is_some_and(|epoch| loc.epoch >= *epoch)
        });
        let fresh: Vec<BlockLocation> = owners.values().cloned().collect();
        if owners.is_empty() {
            dir_blocks.remove(&block_hash);
        }
        fresh
    }

    /// Evict all block ownership entries belonging to `worker_id`.
    pub fn evict_worker(&self, worker_id: &str) {
        self.purge_worker_blocks_below_epoch(worker_id, u64::MAX);
    }

    fn purge_worker_blocks_below_epoch(&self, worker_id: &str, min_epoch: u64) {
        let mut dir_blocks = self.blocks.write();
        dir_blocks.retain(|_, owners| {
            if let Some(loc) = owners.get(worker_id) {
                if loc.epoch < min_epoch {
                    owners.remove(worker_id);
                }
            }
            !owners.is_empty()
        });
    }
}

/// Pluggable local KV backing store (host RAM, pinned DMA arena, or RDMA buffer).
pub trait KvStorageBackend: Send + Sync + 'static {
    fn get_block(&self, block_hash: u64) -> Result<Option<(u64, Bytes)>, KvFabricError>;
    fn put_block(&self, block_hash: u64, epoch: u64, payload: Bytes) -> Result<(), KvFabricError>;
    fn current_epoch(&self) -> u64;
}

/// Simple in-memory KV storage backend for local testing and CPU tier staging.
#[derive(Debug, Default)]
pub struct InMemoryKvBackend {
    epoch: RwLock<u64>,
    blocks: RwLock<FxHashMap<u64, (u64, Bytes)>>,
}

impl InMemoryKvBackend {
    pub fn new(initial_epoch: u64) -> Self {
        Self {
            epoch: RwLock::new(initial_epoch),
            blocks: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn set_epoch(&self, epoch: u64) {
        let mut guard = self.epoch.write();
        if epoch > *guard {
            *guard = epoch;
        }
    }
}

impl KvStorageBackend for InMemoryKvBackend {
    fn get_block(&self, block_hash: u64) -> Result<Option<(u64, Bytes)>, KvFabricError> {
        Ok(self.blocks.read().get(&block_hash).cloned())
    }

    fn put_block(&self, block_hash: u64, epoch: u64, payload: Bytes) -> Result<(), KvFabricError> {
        let current = *self.epoch.read();
        if epoch < current {
            return Err(KvFabricError::StaleEpoch {
                worker_id: "local".to_string(),
                got: epoch,
                current,
            });
        }
        self.blocks.write().insert(block_hash, (epoch, payload));
        Ok(())
    }

    fn current_epoch(&self) -> u64 {
        *self.epoch.read()
    }
}

/// Async framed TCP transport for pulling and pushing verified KV blocks.
pub struct KvBlockTransport<B: KvStorageBackend> {
    local_addr: SocketAddr,
    backend: Arc<B>,
    server_task: JoinHandle<()>,
}

impl<B: KvStorageBackend> KvBlockTransport<B> {
    /// Start listening on `bind_addr` (pass `127.0.0.1:0` for an ephemeral port).
    pub async fn bind(bind_addr: SocketAddr, backend: Arc<B>) -> Result<Self, KvFabricError> {
        let listener = TcpListener::bind(bind_addr).await?;
        let local_addr = listener.local_addr()?;
        let backend_clone = Arc::clone(&backend);

        let server_task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let b = Arc::clone(&backend_clone);
                tokio::spawn(async move {
                    let _ = handle_connection(stream, b).await;
                });
            }
        });

        Ok(Self {
            local_addr,
            backend,
            server_task,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    /// Pull `block_hash` from `peer`, enforcing `epoch >= min_epoch` and verifying SHA-256.
    pub async fn pull_block(
        peer: SocketAddr,
        block_hash: u64,
        min_epoch: u64,
    ) -> Result<(u64, Bytes), KvFabricError> {
        let mut stream = TcpStream::connect(peer).await?;
        let _ = stream.set_nodelay(true);
        let mut req = [0u8; 21];
        req[..4].copy_from_slice(&WIRE_MAGIC);
        req[4] = OP_PULL;
        req[5..13].copy_from_slice(&block_hash.to_le_bytes());
        req[13..21].copy_from_slice(&min_epoch.to_le_bytes());
        stream.write_all(&req).await?;
        stream.flush().await?;

        let mut status = [0u8; 1];
        stream.read_exact(&mut status).await?;
        match status[0] {
            STATUS_OK => {
                let mut hdr = [0u8; 44]; // epoch (8) + len (4) + sha256 (32)
                stream.read_exact(&mut hdr).await?;
                let epoch = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
                let len = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
                if len > MAX_FRAME_BYTES {
                    return Err(KvFabricError::FrameTooLarge { len });
                }
                let expected_hash: [u8; 32] = hdr[12..44].try_into().unwrap();
                let mut payload = vec![0u8; len];
                stream.read_exact(&mut payload).await?;
                if compute_block_digest(&payload) != expected_hash {
                    return Err(KvFabricError::ChecksumMismatch { block_hash });
                }
                if epoch < min_epoch {
                    return Err(KvFabricError::StaleEpoch {
                        worker_id: peer.to_string(),
                        got: epoch,
                        current: min_epoch,
                    });
                }
                Ok((epoch, Bytes::from(payload)))
            }
            STATUS_NOT_FOUND => Err(KvFabricError::BlockNotFound { block_hash }),
            STATUS_STALE_EPOCH => Err(KvFabricError::StaleEpoch {
                worker_id: peer.to_string(),
                got: 0,
                current: min_epoch,
            }),
            STATUS_CHECKSUM_MISMATCH => Err(KvFabricError::ChecksumMismatch { block_hash }),
            _ => Err(KvFabricError::InvalidFrame),
        }
    }

    /// Push a KV block payload with SHA-256 checksum and epoch fence to `peer`.
    pub async fn push_block(
        peer: SocketAddr,
        block_hash: u64,
        epoch: u64,
        payload: &[u8],
    ) -> Result<(), KvFabricError> {
        if payload.len() > MAX_FRAME_BYTES {
            return Err(KvFabricError::FrameTooLarge { len: payload.len() });
        }
        let digest = compute_block_digest(payload);
        let mut stream = TcpStream::connect(peer).await?;
        let _ = stream.set_nodelay(true);
        let mut hdr = [0u8; 57]; // magic (4) + op (1) + hash (8) + epoch (8) + len (4) + sha256 (32)
        hdr[0..4].copy_from_slice(&WIRE_MAGIC);
        hdr[4] = OP_PUSH;
        hdr[5..13].copy_from_slice(&block_hash.to_le_bytes());
        hdr[13..21].copy_from_slice(&epoch.to_le_bytes());
        hdr[21..25].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        hdr[25..57].copy_from_slice(&digest);
        stream.write_all(&hdr).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;

        let mut status = [0u8; 1];
        stream.read_exact(&mut status).await?;
        match status[0] {
            STATUS_OK => Ok(()),
            STATUS_STALE_EPOCH => Err(KvFabricError::StaleEpoch {
                worker_id: peer.to_string(),
                got: epoch,
                current: epoch + 1,
            }),
            STATUS_CHECKSUM_MISMATCH => Err(KvFabricError::ChecksumMismatch { block_hash }),
            _ => Err(KvFabricError::InvalidFrame),
        }
    }
}

impl<B: KvStorageBackend> Drop for KvBlockTransport<B> {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

async fn handle_connection<B: KvStorageBackend>(
    mut stream: TcpStream,
    backend: Arc<B>,
) -> Result<(), KvFabricError> {
    let _ = stream.set_nodelay(true);
    let mut magic_op = [0u8; 5];
    stream.read_exact(&mut magic_op).await?;
    if magic_op[..4] != WIRE_MAGIC {
        let _ = stream.write_all(&[STATUS_ERR]).await;
        return Err(KvFabricError::InvalidFrame);
    }

    match magic_op[4] {
        OP_PULL => {
            let mut rest = [0u8; 16];
            stream.read_exact(&mut rest).await?;
            let block_hash = u64::from_le_bytes(rest[0..8].try_into().unwrap());
            let min_epoch = u64::from_le_bytes(rest[8..16].try_into().unwrap());
            match backend.get_block(block_hash)? {
                None => {
                    stream.write_all(&[STATUS_NOT_FOUND]).await?;
                }
                Some((epoch, _)) if epoch < min_epoch || epoch < backend.current_epoch() => {
                    stream.write_all(&[STATUS_STALE_EPOCH]).await?;
                }
                Some((epoch, payload)) => {
                    let digest = compute_block_digest(&payload);
                    let mut resp_hdr = [0u8; 45];
                    resp_hdr[0] = STATUS_OK;
                    resp_hdr[1..9].copy_from_slice(&epoch.to_le_bytes());
                    resp_hdr[9..13].copy_from_slice(&(payload.len() as u32).to_le_bytes());
                    resp_hdr[13..45].copy_from_slice(&digest);
                    stream.write_all(&resp_hdr).await?;
                    stream.write_all(&payload).await?;
                }
            }
            stream.flush().await?;
            Ok(())
        }
        OP_PUSH => {
            let mut rest = [0u8; 52];
            stream.read_exact(&mut rest).await?;
            let block_hash = u64::from_le_bytes(rest[0..8].try_into().unwrap());
            let epoch = u64::from_le_bytes(rest[8..16].try_into().unwrap());
            let len = u32::from_le_bytes(rest[16..20].try_into().unwrap()) as usize;
            if len > MAX_FRAME_BYTES {
                let _ = stream.write_all(&[STATUS_ERR]).await;
                return Err(KvFabricError::FrameTooLarge { len });
            }
            let expected_hash: [u8; 32] = rest[20..52].try_into().unwrap();
            let mut payload = vec![0u8; len];
            stream.read_exact(&mut payload).await?;
            if compute_block_digest(&payload) != expected_hash {
                stream.write_all(&[STATUS_CHECKSUM_MISMATCH]).await?;
                stream.flush().await?;
                return Err(KvFabricError::ChecksumMismatch { block_hash });
            }
            match backend.put_block(block_hash, epoch, Bytes::from(payload)) {
                Ok(()) => stream.write_all(&[STATUS_OK]).await?,
                Err(KvFabricError::StaleEpoch { .. }) => {
                    stream.write_all(&[STATUS_STALE_EPOCH]).await?;
                }
                Err(_) => stream.write_all(&[STATUS_ERR]).await?,
            }
            stream.flush().await?;
            Ok(())
        }
        _ => {
            let _ = stream.write_all(&[STATUS_ERR]).await;
            Err(KvFabricError::InvalidFrame)
        }
    }
}

/// Compute a deterministic 64-bit XXH3 block hash for KV token chunks.
pub fn kv_block_hash(tokens: &[u32], layer_group: u16) -> u64 {
    let mut buf = Vec::with_capacity(2 + std::mem::size_of_val(tokens));
    buf.extend_from_slice(&layer_group.to_le_bytes());
    for &t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    xxhash_rust::xxh3::xxh3_64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_enforces_monotonic_epochs_and_prunes_stale_owners() {
        let dir = EpochFencedKvDirectory::new();
        let addr_a: SocketAddr = "127.0.0.1:9001".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:9002".parse().unwrap();

        let hash = kv_block_hash(&[101, 202, 303], 0);
        let payload = b"kv-tensor-page-0";
        let meta = KvBlockMeta {
            block_hash: hash,
            num_bytes: payload.len() as u64,
            sha256: compute_block_digest(payload),
        };

        dir.publish_blocks("worker-a", 1, addr_a, std::slice::from_ref(&meta))
            .unwrap();
        dir.publish_blocks("worker-b", 1, addr_b, std::slice::from_ref(&meta))
            .unwrap();
        assert_eq!(dir.lookup_fresh_owners(hash).len(), 2);

        // Fences worker-a to epoch 2 and invalidates epoch-1 blocks.
        let fenced = dir.fence_epoch("worker-a", 2);
        assert_eq!(fenced, 2);

        // Stale publish under epoch 1 must be rejected.
        let err = dir
            .publish_blocks("worker-a", 1, addr_a, std::slice::from_ref(&meta))
            .unwrap_err();
        assert!(matches!(err, KvFabricError::StaleEpoch { .. }));

        // Only worker-b remains as a fresh owner until worker-a republishes at epoch 2.
        let owners = dir.lookup_fresh_owners(hash);
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].worker_id, "worker-b");

        dir.publish_blocks("worker-a", 2, addr_a, std::slice::from_ref(&meta))
            .unwrap();
        assert_eq!(dir.lookup_fresh_owners(hash).len(), 2);

        dir.evict_worker("worker-b");
        let owners = dir.lookup_fresh_owners(hash);
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].worker_id, "worker-a");
    }

    #[tokio::test]
    async fn transport_roundtrips_and_enforces_epoch_and_sha256() {
        let dir = EpochFencedKvDirectory::new();
        let peer_backend = Arc::new(InMemoryKvBackend::new(2));
        let transport =
            KvBlockTransport::bind("127.0.0.1:0".parse().unwrap(), Arc::clone(&peer_backend))
                .await
                .unwrap();

        let hash = kv_block_hash(&[1, 2, 3, 4], 1);
        let payload = Bytes::from_static(b"kv-cache-attention-page-data-0123456789");

        // Stale epoch push rejected cleanly.
        let err = KvBlockTransport::<InMemoryKvBackend>::push_block(
            transport.local_addr(),
            hash,
            1,
            &payload,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, KvFabricError::StaleEpoch { .. }));

        // Valid epoch push + directory publish + verified pull.
        KvBlockTransport::<InMemoryKvBackend>::push_block(
            transport.local_addr(),
            hash,
            2,
            &payload,
        )
        .await
        .unwrap();
        let meta = KvBlockMeta {
            block_hash: hash,
            num_bytes: payload.len() as u64,
            sha256: compute_block_digest(&payload),
        };
        dir.publish_blocks("replica-1", 2, transport.local_addr(), &[meta])
            .unwrap();

        let owners = dir.lookup_fresh_owners(hash);
        assert_eq!(owners.len(), 1);

        let (epoch, got) =
            KvBlockTransport::<InMemoryKvBackend>::pull_block(owners[0].peer_addr, hash, 2)
                .await
                .unwrap();
        assert_eq!(epoch, 2);
        assert_eq!(got, payload);

        // Requesting min_epoch > stored epoch is fenced as stale.
        let err = KvBlockTransport::<InMemoryKvBackend>::pull_block(owners[0].peer_addr, hash, 3)
            .await
            .unwrap_err();
        assert!(matches!(err, KvFabricError::StaleEpoch { .. }));
    }

    #[tokio::test]
    async fn transport_detects_corrupted_push_checksum() {
        let peer_backend = Arc::new(InMemoryKvBackend::new(1));
        let transport =
            KvBlockTransport::bind("127.0.0.1:0".parse().unwrap(), Arc::clone(&peer_backend))
                .await
                .unwrap();

        let mut stream = TcpStream::connect(transport.local_addr()).await.unwrap();
        let hash = 42u64;
        let payload = b"corrupted-payload";
        let bad_digest = [0xAAu8; 32];
        let mut hdr = [0u8; 57];
        hdr[0..4].copy_from_slice(&WIRE_MAGIC);
        hdr[4] = OP_PUSH;
        hdr[5..13].copy_from_slice(&hash.to_le_bytes());
        hdr[13..21].copy_from_slice(&1u64.to_le_bytes());
        hdr[21..25].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        hdr[25..57].copy_from_slice(&bad_digest);
        stream.write_all(&hdr).await.unwrap();
        stream.write_all(payload).await.unwrap();
        stream.flush().await.unwrap();

        let mut status = [0u8; 1];
        stream.read_exact(&mut status).await.unwrap();
        assert_eq!(status[0], STATUS_CHECKSUM_MISMATCH);
    }
}
