// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Structured error types for `hanzo-gcr`.

use std::fmt;

/// All failure modes returned by `hanzo-gcr` memory, store, and checkpoint operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GcrError {
    /// Requested page size is not a supported power of two in `[256, 16 MiB]`.
    InvalidPageSize(usize),
    /// A page index was outside the allocated VRAM page table.
    PageOutOfBounds {
        /// Requested logical page index.
        page_id: usize,
        /// Total number of allocated pages.
        total_pages: usize,
    },
    /// Attempted to mutate a frozen read-only weight page.
    ImmutablePageWrite {
        /// Logical page index of the immutable weight page.
        page_id: usize,
    },
    /// Provided byte slice length did not match the configured page size or range.
    PageLengthMismatch {
        /// Expected byte length.
        expected: usize,
        /// Actual provided byte length.
        actual: usize,
    },
    /// Invalid tenant or job identifier (empty, too long, or unsafe characters).
    InvalidNamespace(String),
    /// Cross-tenant or cross-job restore / manifest mismatch.
    TenantMismatch {
        /// Expected owning tenant identifier.
        expected_tenant: String,
        /// Expected owning job identifier.
        expected_job: String,
        /// Actual tenant identifier recorded in the manifest.
        actual_tenant: String,
        /// Actual job identifier recorded in the manifest.
        actual_job: String,
    },
    /// Requested checkpoint manifest does not exist in the namespace store.
    CheckpointNotFound(String),
    /// A delta manifest references a parent checkpoint that does not exist.
    MissingParentCheckpoint {
        /// Child checkpoint identifier whose parent is missing.
        checkpoint_id: String,
        /// Missing parent checkpoint identifier.
        parent_id: String,
    },
    /// A manifest chain contains a cycle.
    CyclicCheckpointChain {
        /// Checkpoint identifier where the cycle was detected.
        checkpoint_id: String,
    },
    /// A manifest chain exceeded the configured hard safety traversal limit.
    ChainDepthExceeded {
        /// Observed traversal depth.
        depth: u32,
        /// Configured hard maximum traversal depth.
        max_depth: u32,
    },
    /// Chunk payload failed framing, truncation, CRC32C, or SHA-256 verification.
    CorruptedChunk {
        /// Short hex digest identifying the corrupted chunk.
        chunk_id: String,
        /// Human-readable verification failure reason.
        reason: String,
    },
    /// Manifest failed framing, truncation, CRC32C, or SHA-256 verification.
    CorruptedManifest {
        /// Identifier of the corrupted manifest.
        checkpoint_id: String,
        /// Human-readable verification failure reason.
        reason: String,
    },
    /// Restored checkpoint geometry or model base digest does not match target VRAM.
    LayoutMismatch {
        /// Human-readable geometry or weight-base mismatch explanation.
        reason: String,
    },
    /// Post-restore VRAM SHA-256 state digest did not match the manifest digest.
    StateDigestMismatch {
        /// Expected SHA-256 digest from manifest.
        expected: String,
        /// Actual SHA-256 digest computed from staged VRAM pages.
        actual: String,
    },
    /// Underlying filesystem or OS I/O error.
    Io(String),
}

impl fmt::Display for GcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPageSize(size) => {
                write!(
                    f,
                    "invalid page size {size}: must be power of two in [256, 16777216]"
                )
            }
            Self::PageOutOfBounds {
                page_id,
                total_pages,
            } => write!(
                f,
                "page index {page_id} out of bounds for {total_pages} allocated pages"
            ),
            Self::ImmutablePageWrite { page_id } => {
                write!(f, "cannot mutate immutable read-only weight page {page_id}")
            }
            Self::PageLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "page byte length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidNamespace(reason) => write!(f, "invalid namespace: {reason}"),
            Self::TenantMismatch {
                expected_tenant,
                expected_job,
                actual_tenant,
                actual_job,
            } => write!(
                f,
                "tenant/job isolation violation: expected ({expected_tenant}, {expected_job}), got ({actual_tenant}, {actual_job})"
            ),
            Self::CheckpointNotFound(id) => write!(f, "checkpoint not found: {id}"),
            Self::MissingParentCheckpoint {
                checkpoint_id,
                parent_id,
            } => write!(
                f,
                "checkpoint {checkpoint_id} references missing parent {parent_id}"
            ),
            Self::CyclicCheckpointChain { checkpoint_id } => {
                write!(
                    f,
                    "cyclic parent reference detected at checkpoint {checkpoint_id}"
                )
            }
            Self::ChainDepthExceeded { depth, max_depth } => {
                write!(
                    f,
                    "checkpoint chain depth {depth} exceeds hard limit {max_depth}"
                )
            }
            Self::CorruptedChunk { chunk_id, reason } => {
                write!(f, "corrupted chunk {chunk_id}: {reason}")
            }
            Self::CorruptedManifest {
                checkpoint_id,
                reason,
            } => write!(f, "corrupted manifest {checkpoint_id}: {reason}"),
            Self::LayoutMismatch { reason } => write!(f, "VRAM layout mismatch: {reason}"),
            Self::StateDigestMismatch { expected, actual } => write!(
                f,
                "post-restore VRAM state digest mismatch: expected {expected}, got {actual}"
            ),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for GcrError {}

impl From<std::io::Error> for GcrError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

/// Convenience result alias for `hanzo-gcr`.
pub type GcrResult<T> = Result<T, GcrError>;
