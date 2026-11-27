// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Versioned binary KV + GDN (Gated DeltaNet) recurrent state snapshots with
//! fail-closed CRC32 and SHA-256 verification.
//!
//! Hybrid attention + GDN models (4B/9B/27B) maintain both:
//! - Attention KV pages (per block of `block_size` tokens, in FP16/BF16/INT8/KIVI-INT4/KIVI-INT2).
//! - Recurrent GDN state checkpoints (`S_t` + rolling conv1d buffers) captured strictly at
//!   block boundaries in full `FP32` precision (never low-bit quantized, per HIP-1346).

use crate::crypto::{crc32_ieee, Digest256, Sha256};
use crate::error::PrefixCacheError;
use crate::key::{CompatibilityProfile, PrefixCacheKey};

/// Magic 4-byte file header identifying Hanzo Prefix Cache Snapshots.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"HPCS";

/// Current binary snapshot format version.
pub const SNAPSHOT_FORMAT_VERSION: u16 = 1;

/// Header flag indicating quantized or full-precision KV attention pages are present.
pub const FLAG_HAS_KV: u16 = 1 << 0;

/// Header flag indicating an unquantized block-boundary GDN recurrent state checkpoint is present.
pub const FLAG_HAS_GDN: u16 = 1 << 1;

/// Fixed byte length of the serialized [`SnapshotHeader`].
pub const SNAPSHOT_HEADER_BYTES: usize = 172;

/// Fixed-layout binary header protecting a persisted KV+GDN block snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotHeader {
    /// Format version (`SNAPSHOT_FORMAT_VERSION`).
    pub format_version: u16,
    /// Bitmask of `FLAG_HAS_KV` / `FLAG_HAS_GDN`.
    pub flags: u16,
    /// Block size in tokens.
    pub block_size: u32,
    /// Zero-based starting token position of this block.
    pub token_start: u32,
    /// Number of tokens covered by this block (must equal `block_size`).
    pub token_len: u32,
    /// Number of attention layers covered by `kv_payload`.
    pub attn_layer_count: u16,
    /// Number of recurrent GDN layers covered by `gdn_payload`.
    pub gdn_layer_count: u16,
    /// Exact 7-field [`PrefixCacheKey`] digest.
    pub key_digest: Digest256,
    /// Exact 5-field [`CompatibilityProfile`] digest.
    pub compat_digest: Digest256,
    /// Parent block's `key_digest` (`Digest256::ZERO` for block 0).
    pub parent_block_digest: Digest256,
    /// Byte length of the KV payload slice.
    pub kv_payload_len: u32,
    /// Byte length of the GDN recurrent state checkpoint slice.
    pub gdn_payload_len: u32,
    /// IEEE CRC32 checksum of the KV payload bytes.
    pub kv_crc32: u32,
    /// IEEE CRC32 checksum of the GDN payload bytes.
    pub gdn_crc32: u32,
    /// SHA-256 digest over `kv_payload || gdn_payload`.
    pub payload_sha256: Digest256,
    /// IEEE CRC32 checksum over all header bytes preceding this field.
    pub header_crc32: u32,
}

impl SnapshotHeader {
    /// Serialize header into a fixed 172-byte little-endian buffer with computed `header_crc32`.
    #[must_use]
    pub fn encode(&self) -> [u8; SNAPSHOT_HEADER_BYTES] {
        let mut buf = [0u8; SNAPSHOT_HEADER_BYTES];
        buf[0..4].copy_from_slice(&SNAPSHOT_MAGIC);
        buf[4..6].copy_from_slice(&self.format_version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.block_size.to_le_bytes());
        buf[12..16].copy_from_slice(&self.token_start.to_le_bytes());
        buf[16..20].copy_from_slice(&self.token_len.to_le_bytes());
        buf[20..22].copy_from_slice(&self.attn_layer_count.to_le_bytes());
        buf[22..24].copy_from_slice(&self.gdn_layer_count.to_le_bytes());
        buf[24..56].copy_from_slice(self.key_digest.as_bytes());
        buf[56..88].copy_from_slice(self.compat_digest.as_bytes());
        buf[88..120].copy_from_slice(self.parent_block_digest.as_bytes());
        buf[120..124].copy_from_slice(&self.kv_payload_len.to_le_bytes());
        buf[124..128].copy_from_slice(&self.gdn_payload_len.to_le_bytes());
        buf[128..132].copy_from_slice(&self.kv_crc32.to_le_bytes());
        buf[132..136].copy_from_slice(&self.gdn_crc32.to_le_bytes());
        buf[136..168].copy_from_slice(self.payload_sha256.as_bytes());
        let hcrc = crc32_ieee(&buf[0..168]);
        buf[168..172].copy_from_slice(&hcrc.to_le_bytes());
        buf
    }

    /// Decode and fail-closed verify magic, format version, and `header_crc32`.
    pub fn decode_verified(buf: &[u8]) -> Result<Self, PrefixCacheError> {
        if buf.len() < SNAPSHOT_HEADER_BYTES {
            return Err(PrefixCacheError::MalformedSnapshotLength {
                found: buf.len(),
                expected: SNAPSHOT_HEADER_BYTES,
            });
        }
        let magic = [buf[0], buf[1], buf[2], buf[3]];
        if magic != SNAPSHOT_MAGIC {
            return Err(PrefixCacheError::BadMagic(magic));
        }
        let format_version = u16::from_le_bytes([buf[4], buf[5]]);
        if format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(PrefixCacheError::UnsupportedVersion {
                found: format_version,
                expected: SNAPSHOT_FORMAT_VERSION,
            });
        }
        let stored_crc = u32::from_le_bytes([buf[168], buf[169], buf[170], buf[171]]);
        let computed_crc = crc32_ieee(&buf[0..168]);
        if stored_crc != computed_crc {
            return Err(PrefixCacheError::HeaderCrcMismatch {
                expected: stored_crc,
                computed: computed_crc,
            });
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&buf[24..56]);
        let mut compat_bytes = [0u8; 32];
        compat_bytes.copy_from_slice(&buf[56..88]);
        let mut parent_bytes = [0u8; 32];
        parent_bytes.copy_from_slice(&buf[88..120]);
        let mut payload_hash_bytes = [0u8; 32];
        payload_hash_bytes.copy_from_slice(&buf[136..168]);

        Ok(Self {
            format_version,
            flags: u16::from_le_bytes([buf[6], buf[7]]),
            block_size: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            token_start: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            token_len: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            attn_layer_count: u16::from_le_bytes([buf[20], buf[21]]),
            gdn_layer_count: u16::from_le_bytes([buf[22], buf[23]]),
            key_digest: Digest256::from_bytes(key_bytes),
            compat_digest: Digest256::from_bytes(compat_bytes),
            parent_block_digest: Digest256::from_bytes(parent_bytes),
            kv_payload_len: u32::from_le_bytes([buf[120], buf[121], buf[122], buf[123]]),
            gdn_payload_len: u32::from_le_bytes([buf[124], buf[125], buf[126], buf[127]]),
            kv_crc32: u32::from_le_bytes([buf[128], buf[129], buf[130], buf[131]]),
            gdn_crc32: u32::from_le_bytes([buf[132], buf[133], buf[134], buf[135]]),
            payload_sha256: Digest256::from_bytes(payload_hash_bytes),
            header_crc32: stored_crc,
        })
    }
}

/// Self-contained versioned KV + GDN snapshot for one block-aligned prefix node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixBlockSnapshot {
    /// Verified binary header with integrity checksums and key bindings.
    pub header: SnapshotHeader,
    /// Packed attention KV page bytes for this block.
    pub kv_payload: Vec<u8>,
    /// Unquantized block-boundary recurrent GDN state checkpoint bytes.
    pub gdn_payload: Vec<u8>,
}

impl PrefixBlockSnapshot {
    /// Construct a new versioned snapshot with freshly computed CRC32 and SHA-256 digests.
    pub fn build(
        key: &PrefixCacheKey,
        compat: &CompatibilityProfile,
        parent_block_digest: Digest256,
        token_start: u32,
        token_len: u32,
        kv_payload: Vec<u8>,
        gdn_payload: Vec<u8>,
    ) -> Result<Self, PrefixCacheError> {
        if token_len != compat.runtime_config.block_size {
            return Err(PrefixCacheError::InvalidBlockAlignment {
                reason: format!(
                    "snapshot token_len {token_len} != block_size {}",
                    compat.runtime_config.block_size
                ),
            });
        }
        if !token_start.is_multiple_of(compat.runtime_config.block_size) {
            return Err(PrefixCacheError::InvalidBlockAlignment {
                reason: format!(
                    "snapshot token_start {token_start} is not aligned to block_size {}",
                    compat.runtime_config.block_size
                ),
            });
        }

        let mut flags = 0u16;
        if !kv_payload.is_empty() {
            flags |= FLAG_HAS_KV;
        }
        if !gdn_payload.is_empty() {
            flags |= FLAG_HAS_GDN;
        }

        let kv_crc32 = crc32_ieee(&kv_payload);
        let gdn_crc32 = crc32_ieee(&gdn_payload);
        let mut h = Sha256::new();
        h.update(&kv_payload);
        h.update(&gdn_payload);
        let payload_sha256 = h.finalize();

        let mut header = SnapshotHeader {
            format_version: SNAPSHOT_FORMAT_VERSION,
            flags,
            block_size: compat.runtime_config.block_size,
            token_start,
            token_len,
            attn_layer_count: compat.runtime_config.attn_layers,
            gdn_layer_count: compat.runtime_config.gdn_layers,
            key_digest: key.key_digest,
            compat_digest: compat.canonical_digest(),
            parent_block_digest,
            kv_payload_len: kv_payload.len() as u32,
            gdn_payload_len: gdn_payload.len() as u32,
            kv_crc32,
            gdn_crc32,
            payload_sha256,
            header_crc32: 0,
        };
        let encoded_hdr = header.encode();
        header.header_crc32 = u32::from_le_bytes([
            encoded_hdr[168],
            encoded_hdr[169],
            encoded_hdr[170],
            encoded_hdr[171],
        ]);

        Ok(Self {
            header,
            kv_payload,
            gdn_payload,
        })
    }

    /// Total serialized byte footprint of this snapshot (header + KV + GDN payloads).
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        SNAPSHOT_HEADER_BYTES + self.kv_payload.len() + self.gdn_payload.len()
    }

    /// Encode header + KV payload + GDN checkpoint into a contiguous binary buffer.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_len());
        out.extend_from_slice(&self.header.encode());
        out.extend_from_slice(&self.kv_payload);
        out.extend_from_slice(&self.gdn_payload);
        out
    }

    /// Decode and fail-closed verify every integrity check, key binding, and compatibility digest.
    pub fn decode_verified(
        bytes: &[u8],
        expected_key: &PrefixCacheKey,
        expected_compat: &CompatibilityProfile,
    ) -> Result<Self, PrefixCacheError> {
        let snap = Self::decode_self_consistent(bytes)?;
        if !snap.header.key_digest.ct_eq(&expected_key.key_digest) {
            return Err(PrefixCacheError::KeyMismatch);
        }
        if !snap
            .header
            .compat_digest
            .ct_eq(&expected_compat.canonical_digest())
        {
            return Err(PrefixCacheError::CompatibilityMismatch);
        }
        Ok(snap)
    }

    /// Decode and verify internal structural, version, CRC32, and SHA-256 integrity
    /// (used during cold-restart directory scans before key-specific lookups).
    pub fn decode_self_consistent(bytes: &[u8]) -> Result<Self, PrefixCacheError> {
        let header = SnapshotHeader::decode_verified(bytes)?;
        let kv_len = header.kv_payload_len as usize;
        let gdn_len = header.gdn_payload_len as usize;
        let expected_total = SNAPSHOT_HEADER_BYTES
            .saturating_add(kv_len)
            .saturating_add(gdn_len);
        if bytes.len() != expected_total {
            return Err(PrefixCacheError::MalformedSnapshotLength {
                found: bytes.len(),
                expected: expected_total,
            });
        }

        let kv_start = SNAPSHOT_HEADER_BYTES;
        let kv_end = kv_start + kv_len;
        let gdn_end = kv_end + gdn_len;
        let kv_slice = &bytes[kv_start..kv_end];
        let gdn_slice = &bytes[kv_end..gdn_end];

        let kv_crc = crc32_ieee(kv_slice);
        if kv_crc != header.kv_crc32 {
            return Err(PrefixCacheError::KvCrcMismatch {
                expected: header.kv_crc32,
                computed: kv_crc,
            });
        }

        let gdn_crc = crc32_ieee(gdn_slice);
        if gdn_crc != header.gdn_crc32 {
            return Err(PrefixCacheError::GdnCrcMismatch {
                expected: header.gdn_crc32,
                computed: gdn_crc,
            });
        }

        let mut h = Sha256::new();
        h.update(kv_slice);
        h.update(gdn_slice);
        let computed_sha = h.finalize();
        if !computed_sha.ct_eq(&header.payload_sha256) {
            return Err(PrefixCacheError::PayloadHashMismatch);
        }

        Ok(Self {
            header,
            kv_payload: kv_slice.to_vec(),
            gdn_payload: gdn_slice.to_vec(),
        })
    }
}

/// Deterministic KV page + recurrent GDN state transition kernel for bit-exact validation.
///
/// Evolves an unquantized `FP32` GDN recurrent state across `block_tokens` via a
/// deterministic DeltaNet-style recurrence and synthesizes the matching quantized KV page
/// bytes so restored snapshots can be verified bit-for-bit against fresh prefill execution.
pub fn synthesize_block_kv_gdn_state(
    prev_gdn_state: &[f32],
    block_tokens: &[u32],
    block_index: u32,
    kv_bytes_len: usize,
    gdn_state_elems: usize,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let mut state = if prev_gdn_state.len() == gdn_state_elems {
        prev_gdn_state.to_vec()
    } else {
        vec![0.0f32; gdn_state_elems]
    };

    // Deterministic GDN recurrence: S_t = alpha * S_{t-1} + beta * (v_t - S_{t-1} * k_t)
    let alpha = 0.968_75f32;
    let beta = 0.031_25f32;
    for (pos_in_block, &tok) in block_tokens.iter().enumerate() {
        let abs_pos = (block_index as usize) * block_tokens.len() + pos_in_block;
        let mix = tok.wrapping_mul(0x9e37_79b9) ^ ((abs_pos as u32).wrapping_mul(0x85eb_ca6b));
        for (idx, s) in state.iter_mut().enumerate() {
            let lane = mix.wrapping_add((idx as u32).wrapping_mul(0xc2b2_ae35));
            let k = ((lane & 0x3ff) as f32 / 1024.0) - 0.5;
            let v = (((lane >> 10) & 0x3ff) as f32 / 1024.0) - 0.5;
            *s = (alpha * *s) + beta * (v - (*s * k));
        }
    }

    // Serialize unquantized FP32 GDN checkpoint bytes.
    let mut gdn_bytes = Vec::with_capacity(gdn_state_elems * 4);
    for &val in &state {
        gdn_bytes.extend_from_slice(&val.to_le_bytes());
    }

    // Synthesize deterministic KV page bytes bound to block_index, tokens, and state digest.
    let mut seed_hasher = Sha256::new();
    seed_hasher.update_u32(block_index);
    for &tok in block_tokens {
        seed_hasher.update_u32(tok);
    }
    seed_hasher.update(&gdn_bytes);
    let mut block_digest = seed_hasher.finalize().0;

    let mut kv_bytes = Vec::with_capacity(kv_bytes_len);
    let mut chunk_ctr = 0u32;
    while kv_bytes.len() < kv_bytes_len {
        let remain = kv_bytes_len - kv_bytes.len();
        let take = remain.min(32);
        kv_bytes.extend_from_slice(&block_digest[..take]);
        if kv_bytes.len() < kv_bytes_len {
            chunk_ctr += 1;
            let mut nh = Sha256::new();
            nh.update(&block_digest);
            nh.update_u32(chunk_ctr);
            block_digest = nh.finalize().0;
        }
    }

    (kv_bytes, state, gdn_bytes)
}

/// Deserialize an unquantized `FP32` GDN state vector from snapshot bytes.
pub fn decode_gdn_f32_state(gdn_bytes: &[u8]) -> Result<Vec<f32>, PrefixCacheError> {
    if !gdn_bytes.len().is_multiple_of(4) {
        return Err(PrefixCacheError::MalformedSnapshotLength {
            found: gdn_bytes.len(),
            expected: gdn_bytes.len() + (4 - (gdn_bytes.len() % 4)),
        });
    }
    let mut out = Vec::with_capacity(gdn_bytes.len() / 4);
    for &chunk in gdn_bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(chunk));
    }
    Ok(out)
}
