//! Grounded repository corpus loader and deterministic SplitMix64 PRNG.
//!
//! All offline PowerInfer neuron/expert activation traces, synthetic quantized weights,
//! and verification token inputs are deterministically derived from the three versioned
//! repository corpus files:
//! - `calibration_data/calibration_datav3_small.txt`
//! - `chat_templates/llama3.json`
//! - `orderings/xlora-paper-ordering.json`

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::sha256::{to_hex, Sha256};

/// Relative paths of the three versioned repository corpus files from workspace root.
pub const CORPUS_RELATIVE_PATHS: [&str; 3] = [
    "calibration_data/calibration_datav3_small.txt",
    "chat_templates/llama3.json",
    "orderings/xlora-paper-ordering.json",
];

/// Deterministic 64-bit SplitMix64 PRNG (zero external dependencies).
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Initialises a new generator from a 64-bit seed.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next 64-bit pseudorandom word.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Generates a uniform `f32` in `[-1.0, 1.0]`.
    pub fn next_signed_f32(&mut self) -> f32 {
        (self.next_f64() as f32) * 2.0 - 1.0
    }
}

/// Attestation record for a single versioned corpus file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusFileRecord {
    /// Repository-relative path.
    pub relative_path: String,
    /// Exact file size in bytes.
    pub byte_len: usize,
    /// Lowercase hexadecimal SHA-256 digest of file contents.
    pub sha256_hex: String,
}

/// Loaded grounded repository corpus with verified SHA-256 attestation.
#[derive(Clone, Debug)]
pub struct GroundedCorpus {
    /// Per-file attestation records.
    pub files: Vec<CorpusFileRecord>,
    /// Total bytes across all corpus files.
    pub total_bytes: usize,
    /// Combined SHA-256 over all three files concatenated in canonical order.
    pub combined_sha256_hex: String,
    /// Raw concatenated corpus bytes used for deterministic profiling & seeding.
    pub raw_bytes: Vec<u8>,
}

impl GroundedCorpus {
    /// Resolves the repository root from either the workspace root or crate subdirectory
    /// and loads all three versioned corpus files.
    pub fn load_from_workspace() -> io::Result<Self> {
        let root = find_repo_root()?;
        Self::load_from_root(&root)
    }

    /// Loads all three versioned corpus files relative to `repo_root`.
    pub fn load_from_root(repo_root: &Path) -> io::Result<Self> {
        let mut files = Vec::with_capacity(CORPUS_RELATIVE_PATHS.len());
        let mut combined = Sha256::new();
        let mut raw_bytes = Vec::new();

        for rel in CORPUS_RELATIVE_PATHS {
            let full = repo_root.join(rel);
            let data = fs::read(&full)?;
            let digest = Sha256::digest(&data);
            combined.update(&data);
            raw_bytes.extend_from_slice(&data);
            files.push(CorpusFileRecord {
                relative_path: rel.to_string(),
                byte_len: data.len(),
                sha256_hex: to_hex(&digest),
            });
        }

        let total_bytes = raw_bytes.len();
        let combined_sha256_hex = to_hex(&combined.finalize());
        Ok(Self {
            files,
            total_bytes,
            combined_sha256_hex,
            raw_bytes,
        })
    }

    /// Derives a deterministic 64-bit seed mixing `user_seed` with the grounded corpus digest.
    pub fn derived_seed(&self, user_seed: u64) -> u64 {
        let mut h = Sha256::new();
        h.update(&user_seed.to_le_bytes());
        h.update(self.combined_sha256_hex.as_bytes());
        let digest = h.finalize();
        u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ])
    }

    /// Extracts empirical token-bigram frequency counts from the grounded corpus
    /// (`num_buckets` buckets), naturally exhibiting heavy-tailed power-law skew
    /// characteristic of real text/code/JSON workloads.
    pub fn empirical_bucket_frequencies(&self, num_buckets: usize) -> Vec<u64> {
        let n = num_buckets.max(1);
        let mut counts = vec![1u64; n];
        if self.raw_bytes.len() < 2 {
            return counts;
        }
        // Build frequency table of observed byte bigrams, then fold into `n` ranked buckets
        // preserving the empirical power-law tail of the grounded corpus.
        let mut bigram_counts = vec![0u64; 65_536];
        for win in self.raw_bytes.windows(2) {
            let key = ((win[0] as usize) << 8) | (win[1] as usize);
            bigram_counts[key] += 1;
        }
        let mut nonzero: Vec<u64> = bigram_counts.into_iter().filter(|&c| c > 0).collect();
        nonzero.sort_unstable_by(|a, b| b.cmp(a));
        if nonzero.is_empty() {
            return counts;
        }
        for (i, count) in counts.iter_mut().enumerate() {
            let start = (i * nonzero.len()) / n;
            let end = (((i + 1) * nonzero.len()) / n)
                .max(start + 1)
                .min(nonzero.len());
            let sum: u64 = nonzero[start..end].iter().sum();
            *count = sum.max(1);
        }
        counts
    }
}

/// Locates the repository root containing `calibration_data/calibration_datav3_small.txt`.
pub fn find_repo_root() -> io::Result<PathBuf> {
    let candidates = [
        PathBuf::from("."),
        PathBuf::from(".."),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    ];
    for cand in candidates {
        if cand.join(CORPUS_RELATIVE_PATHS[0]).is_file() {
            return Ok(cand);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "could not locate grounded HIP-1346 corpus files in workspace",
    ))
}
