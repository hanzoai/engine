// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed error definitions for `hanzo-prefixcache`.

use std::fmt;
use std::io;

/// All typed errors produced by prefix cache keying, tenant ACL enforcement,
/// versioned KV+GDN snapshot decoding, radix indexing, tiered storage, and scheduling.
#[derive(Debug)]
pub enum PrefixCacheError {
    /// Snapshot magic header bytes did not match `HPCS`.
    BadMagic([u8; 4]),
    /// Snapshot format version is incompatible with this build.
    UnsupportedVersion {
        /// Version found in the binary snapshot header.
        found: u16,
        /// Version supported by this reader.
        expected: u16,
    },
    /// Snapshot binary buffer was shorter than required or had trailing garbage.
    MalformedSnapshotLength {
        /// Observed byte length.
        found: usize,
        /// Expected byte length.
        expected: usize,
    },
    /// Snapshot header CRC32 checksum failed verification.
    HeaderCrcMismatch {
        /// Expected CRC32 recorded in header.
        expected: u32,
        /// Recomputed CRC32 from header bytes.
        computed: u32,
    },
    /// KV payload CRC32 checksum failed verification.
    KvCrcMismatch {
        /// Expected CRC32 recorded in header.
        expected: u32,
        /// Recomputed CRC32 from KV bytes.
        computed: u32,
    },
    /// GDN recurrent state payload CRC32 checksum failed verification.
    GdnCrcMismatch {
        /// Expected CRC32 recorded in header.
        expected: u32,
        /// Recomputed CRC32 from GDN bytes.
        computed: u32,
    },
    /// Combined payload SHA-256 digest failed verification.
    PayloadHashMismatch,
    /// Snapshot key digest did not match the requested 7-field `PrefixCacheKey`.
    KeyMismatch,
    /// Snapshot compatibility digest did not match the active model/runtime profile.
    CompatibilityMismatch,
    /// Caller lacks read or write authorization for the requested tenant scope.
    UnauthorizedTenantAccess {
        /// Calling tenant identifier.
        caller_tenant: String,
        /// Reason authorization was denied.
        reason: String,
    },
    /// Attempted to blend or attach a private tenant turn onto a shared node scope.
    ScopeBoundaryViolation {
        /// Human-readable explanation of the scope violation.
        reason: String,
    },
    /// Token sequence or block boundary argument violated invariants.
    InvalidBlockAlignment {
        /// Reason alignment failed.
        reason: String,
    },
    /// Attempted to unpin or reference a radix block that is not present or not pinned.
    InvalidPinOperation {
        /// Reason the pin operation failed.
        reason: String,
    },
    /// No compatible or under-capacity worker was available to schedule the request.
    NoEligibleWorker {
        /// Reason no worker could be chosen.
        reason: String,
    },
    /// Underlying filesystem I/O failure on Tier 1 or Tier 2 backing storage.
    Io(io::Error),
}

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic(magic) => write!(f, "invalid snapshot magic bytes: {magic:?}"),
            Self::UnsupportedVersion { found, expected } => {
                write!(
                    f,
                    "unsupported snapshot format version {found} (expected {expected})"
                )
            }
            Self::MalformedSnapshotLength { found, expected } => {
                write!(
                    f,
                    "malformed snapshot length: found {found} bytes, expected {expected}"
                )
            }
            Self::HeaderCrcMismatch { expected, computed } => {
                write!(
                    f,
                    "snapshot header CRC32 mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::KvCrcMismatch { expected, computed } => {
                write!(
                    f,
                    "snapshot KV payload CRC32 mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::GdnCrcMismatch { expected, computed } => {
                write!(
                    f,
                    "snapshot GDN payload CRC32 mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::PayloadHashMismatch => {
                write!(f, "snapshot combined payload SHA-256 mismatch")
            }
            Self::KeyMismatch => {
                write!(f, "snapshot 7-field PrefixCacheKey digest mismatch")
            }
            Self::CompatibilityMismatch => {
                write!(f, "snapshot model/runtime compatibility digest mismatch")
            }
            Self::UnauthorizedTenantAccess {
                caller_tenant,
                reason,
            } => {
                write!(
                    f,
                    "unauthorized tenant access for '{caller_tenant}': {reason}"
                )
            }
            Self::ScopeBoundaryViolation { reason } => {
                write!(f, "tenant scope boundary violation: {reason}")
            }
            Self::InvalidBlockAlignment { reason } => {
                write!(f, "invalid block alignment: {reason}")
            }
            Self::InvalidPinOperation { reason } => {
                write!(f, "invalid pin operation: {reason}")
            }
            Self::NoEligibleWorker { reason } => {
                write!(f, "no eligible worker available: {reason}")
            }
            Self::Io(err) => write!(f, "tiered storage I/O error: {err}"),
        }
    }
}

impl std::error::Error for PrefixCacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for PrefixCacheError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
