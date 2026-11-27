//! Grounded repository corpus loading, SHA-256 verification, byte-level causal
//! sequence tokenization, and deterministic worker data sharding.
//!
//! Every training and validation sequence is constructed directly from the three
//! repository calibration and metadata files concatenated in canonical order.
//! No synthetic or fabricated loss trajectories are used anywhere in the crate.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::sha256::{sha256_hex, Sha256Hasher};

/// Canonical relative paths of the three grounded repository corpus files.
pub const CORPUS_RELATIVE_PATHS: [&str; 3] = [
    "calibration_data/calibration_datav3_small.txt",
    "chat_templates/llama3.json",
    "orderings/xlora-paper-ordering.json",
];

/// Verified SHA-256 digest of the concatenated three corpus files (140,742 bytes).
pub const EXPECTED_CORPUS_SHA256: &str =
    "84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856";

/// Vocabulary size for compact lossless printable-ASCII + control byte mapping:
/// `0 = '\n'`, `1 = '\t'`, `2..=96 = ASCII 32..=126`, `97 = other bytes`.
pub const CORPUS_VOCAB_SIZE: usize = 98;

/// Errors raised while loading or verifying the grounded repository corpus.
#[derive(Debug, Clone)]
pub enum CorpusError {
    /// One of the expected repository files could not be read.
    IoError {
        /// File path that failed to read.
        path: String,
        /// Underlying OS error string.
        message: String,
    },
    /// Concatenated corpus SHA-256 did not match [`EXPECTED_CORPUS_SHA256`].
    DigestMismatch {
        /// Expected SHA-256 hex digest.
        expected: String,
        /// Actual SHA-256 hex digest computed from disk.
        actual: String,
    },
    /// Sequence length was invalid for the loaded byte stream.
    InvalidSequenceLength(usize),
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError { path, message } => {
                write!(f, "failed reading corpus file {path}: {message}")
            }
            Self::DigestMismatch { expected, actual } => {
                write!(
                    f,
                    "corpus SHA-256 mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidSequenceLength(n) => write!(f, "invalid sequence length: {n}"),
        }
    }
}

impl std::error::Error for CorpusError {}

/// Maps a raw UTF-8 byte into a compact token index in `0..CORPUS_VOCAB_SIZE`.
#[must_use]
pub fn byte_to_token(byte: u8) -> u8 {
    match byte {
        b'\n' => 0,
        b'\t' => 1,
        32..=126 => byte - 30,
        _ => (CORPUS_VOCAB_SIZE - 1) as u8,
    }
}

/// Verified grounded training and validation corpus split into fixed-length
/// causal token sequences of length `seq_len + 1` (input prefix + shifted target).
#[derive(Clone, Debug)]
pub struct GroundedCorpus {
    /// Verified SHA-256 hex digest of the concatenated raw corpus bytes.
    pub corpus_sha256: String,
    /// Total raw byte length of the concatenated corpus files.
    pub raw_bytes_len: usize,
    /// Context length `seq_len` (each sequence stores `seq_len + 1` tokens).
    pub seq_len: usize,
    /// Training sequences (`seq_len + 1` tokens each).
    pub train_sequences: Vec<Vec<u8>>,
    /// Held-out validation sequences (`seq_len + 1` tokens each).
    pub val_sequences: Vec<Vec<u8>>,
    /// Verification spot-check micro-batch carved from held-out validation data.
    pub spot_check_sequences: Vec<Vec<u8>>,
}

impl GroundedCorpus {
    /// Discovers the repository root starting from `CARGO_MANIFEST_DIR` or CWD.
    #[must_use]
    pub fn discover_repo_root() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(parent) = manifest_dir.parent() {
            if parent.join(CORPUS_RELATIVE_PATHS[0]).is_file() {
                return parent.to_path_buf();
            }
        }
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if cwd.join(CORPUS_RELATIVE_PATHS[0]).is_file() {
            return cwd;
        }
        manifest_dir
    }

    /// Loads the three repository files from `repo_root`, verifies their exact
    /// concatenated SHA-256 against [`EXPECTED_CORPUS_SHA256`], and partitions
    /// token windows into disjoint train, validation, and spot-check sets.
    pub fn load_from_repo(repo_root: &Path, seq_len: usize) -> Result<Self, CorpusError> {
        if seq_len < 4 {
            return Err(CorpusError::InvalidSequenceLength(seq_len));
        }

        let mut raw = Vec::with_capacity(145_000);
        let mut hasher = Sha256Hasher::new();

        for rel in CORPUS_RELATIVE_PATHS {
            let full = repo_root.join(rel);
            let bytes = fs::read(&full).map_err(|e| CorpusError::IoError {
                path: rel.to_string(),
                message: e.to_string(),
            })?;
            hasher.update(&bytes);
            raw.extend_from_slice(&bytes);
        }

        let digest = crate::sha256::to_hex(&hasher.finalize());
        if digest != EXPECTED_CORPUS_SHA256 {
            return Err(CorpusError::DigestMismatch {
                expected: EXPECTED_CORPUS_SHA256.to_string(),
                actual: digest,
            });
        }

        let tokens: Vec<u8> = raw.iter().copied().map(byte_to_token).collect();
        let window = seq_len + 1;
        let stride = seq_len;

        let mut all_windows = Vec::new();
        let mut pos = 0usize;
        while pos + window <= tokens.len() {
            all_windows.push(tokens[pos..pos + window].to_vec());
            pos += stride;
        }

        // Strict chronological hold-out: first 82% train, next 12% validation,
        // final 6% dedicated spot-check verification micro-set.
        let total = all_windows.len();
        let train_end = (total * 82) / 100;
        let val_end = (total * 94) / 100;

        let train_sequences = all_windows[..train_end].to_vec();
        let val_sequences = all_windows[train_end..val_end].to_vec();
        let spot_check_sequences = all_windows[val_end..].to_vec();

        Ok(Self {
            corpus_sha256: digest,
            raw_bytes_len: raw.len(),
            seq_len,
            train_sequences,
            val_sequences,
            spot_check_sequences,
        })
    }

    /// Returns a deterministic worker shard of training sequences using
    /// round-robin striping across `num_workers` so each worker sees an
    /// independent, non-overlapping slice of the grounded corpus.
    #[must_use]
    pub fn shard_for_worker(&self, worker_idx: usize, num_workers: usize) -> Vec<Vec<u8>> {
        let count = num_workers.max(1);
        let idx = worker_idx % count;
        self.train_sequences
            .iter()
            .enumerate()
            .filter_map(|(i, seq)| {
                if i % count == idx {
                    Some(seq.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns a deterministic contiguous slice of validation sequences capped
    /// at `max_sequences` for reproducible validation loss measurement.
    #[must_use]
    pub fn eval_subset(&self, max_sequences: usize) -> Vec<&[u8]> {
        self.val_sequences
            .iter()
            .take(max_sequences)
            .map(Vec::as_slice)
            .collect()
    }

    /// Returns a deterministic spot-check subset for Byzantine validation gating.
    #[must_use]
    pub fn spot_check_subset(&self, max_sequences: usize) -> Vec<&[u8]> {
        self.spot_check_sequences
            .iter()
            .take(max_sequences)
            .map(Vec::as_slice)
            .collect()
    }

    /// Computes a deterministic SHA-256 fingerprint of a shard's token layout.
    #[must_use]
    pub fn shard_digest(sequences: &[Vec<u8>]) -> String {
        let mut flat = Vec::new();
        for seq in sequences {
            flat.extend_from_slice(seq);
        }
        sha256_hex(&flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_verifies_grounded_corpus_sha256() {
        let root = GroundedCorpus::discover_repo_root();
        let corpus = GroundedCorpus::load_from_repo(&root, 16).expect("grounded corpus loads");
        assert_eq!(corpus.corpus_sha256, EXPECTED_CORPUS_SHA256);
        assert_eq!(corpus.raw_bytes_len, 140_742);
        assert!(corpus.train_sequences.len() > 5_000);
        assert!(corpus.val_sequences.len() > 500);
        assert!(corpus.spot_check_sequences.len() > 200);
    }

    #[test]
    fn shards_are_disjoint_and_cover_training_set() {
        let root = GroundedCorpus::discover_repo_root();
        let corpus = GroundedCorpus::load_from_repo(&root, 16).expect("grounded corpus loads");
        let s0 = corpus.shard_for_worker(0, 3);
        let s1 = corpus.shard_for_worker(1, 3);
        let s2 = corpus.shard_for_worker(2, 3);
        assert_eq!(s0.len() + s1.len() + s2.len(), corpus.train_sequences.len());
        assert_ne!(
            GroundedCorpus::shard_digest(&s0),
            GroundedCorpus::shard_digest(&s1)
        );
    }
}
