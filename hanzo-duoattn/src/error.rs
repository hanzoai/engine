// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed errors for DuoAttention calibration, profile verification, and dual-cache execution.

use std::fmt;

/// Errors returned by `hanzo-duoattn` operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DuoAttnError {
    /// Fail-closed rejection when a calibration profile does not match the target model checkpoint.
    ///
    /// DuoAttention head assignments are model-specific and NOT a safe universal transformation.
    ModelMismatch {
        /// Model checkpoint ID bound inside the calibration profile.
        profile_model: String,
        /// Target runtime model checkpoint ID.
        target_model: String,
    },
    /// Fail-closed rejection when architectural dimensions disagree between profile and runtime.
    ArchitectureMismatch {
        /// Name of the mismatched field (e.g., `num_attn_layers`, `num_kv_heads`, `head_dim`).
        field: &'static str,
        /// Value recorded in the profile.
        profile_value: usize,
        /// Value required by the runtime model.
        target_value: usize,
    },
    /// Fail-closed rejection when a profile's SHA-256 checksum fails verification.
    ProfileChecksumMismatch {
        /// Checksum recorded in the profile.
        recorded: String,
        /// Recomputed canonical SHA-256 checksum.
        computed: String,
    },
    /// Profile schema version is unsupported by this crate.
    UnsupportedProfileVersion {
        /// Encountered schema version.
        version: u32,
    },
    /// Attempted to apply DuoAttention KV head partitioning to a recurrent GDN layer.
    ///
    /// In hybrid Attention+GDN architectures (`4B/9B/27B`), recurrent GDN state matrices
    /// must remain untouched and exempt from attention KV streaming eviction.
    RecurrentLayerExempt {
        /// Zero-based global layer index of the recurrent GDN layer.
        layer_idx: usize,
    },
    /// Invalid configuration parameter supplied by caller.
    InvalidConfig(String),
    /// Tensor or vector shape mismatch at a kernel boundary.
    DimensionMismatch {
        /// Call site context.
        context: &'static str,
        /// Expected element count or dimension.
        expected: usize,
        /// Actual element count or dimension.
        actual: usize,
    },
    /// Failed to serialize or parse a calibration profile artifact.
    SerializationError(String),
    /// Filesystem I/O failure.
    IoError(String),
}

impl fmt::Display for DuoAttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelMismatch {
                profile_model,
                target_model,
            } => write!(
                f,
                "fail-closed cross-model calibration rejection: profile bound to '{profile_model}' cannot be applied to '{target_model}' (DuoAttention requires per-model offline calibration)"
            ),
            Self::ArchitectureMismatch {
                field,
                profile_value,
                target_value,
            } => write!(
                f,
                "architecture mismatch for '{field}': profile has {profile_value}, target model requires {target_value}"
            ),
            Self::ProfileChecksumMismatch { recorded, computed } => write!(
                f,
                "calibration profile SHA-256 checksum mismatch: recorded {recorded}, computed {computed}"
            ),
            Self::UnsupportedProfileVersion { version } => {
                write!(f, "unsupported calibration profile version: {version}")
            }
            Self::RecurrentLayerExempt { layer_idx } => write!(
                f,
                "layer {layer_idx} is a recurrent GDN layer and is exempt from DuoAttention KV streaming eviction"
            ),
            Self::InvalidConfig(msg) => write!(f, "invalid DuoAttention config: {msg}"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in {context}: expected {expected}, got {actual}"
            ),
            Self::SerializationError(msg) => write!(f, "profile serialization error: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for DuoAttnError {}

impl From<std::io::Error> for DuoAttnError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}
