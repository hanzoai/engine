// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Grounded repository corpus loader (SHA-256 verified) and deterministic workload builders.
//!
//! All evaluation workloads and token-level verification sequences are deterministically derived
//! from 3 versioned files in the `hanzoai/engine` repository:
//! - `calibration_data/calibration_datav3_small.txt`
//! - `chat_templates/llama3.json`
//! - `orderings/xlora-paper-ordering.json`

use std::fs;
use std::path::{Path, PathBuf};

use crate::config::SloTargets;
use crate::request::{InferenceRequest, RequestClass};
use crate::sha256::{hex_lower, Sha256, SplitMix64};
use crate::verify::VerifySequenceSpec;

/// Relative paths of the 3 canonical grounded corpus files inside the repository root.
pub const CORPUS_FILES: [&str; 3] = [
    "calibration_data/calibration_datav3_small.txt",
    "chat_templates/llama3.json",
    "orderings/xlora-paper-ordering.json",
];

/// Provenance record for a single repository corpus file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusFileProvenance {
    /// Repository-relative file path (never a local machine path).
    pub relative_path: String,
    /// Exact file byte length.
    pub byte_len: usize,
    /// Lowercase hex FIPS 180-4 SHA-256 digest.
    pub sha256_hex: String,
}

/// Loaded and SHA-256-verified grounded repository corpus.
#[derive(Debug, Clone)]
pub struct GroundedCorpus {
    /// Per-file provenance entries in canonical order.
    pub files: Vec<CorpusFileProvenance>,
    /// Combined byte length across all corpus files.
    pub total_bytes: usize,
    /// Combined SHA-256 hex digest over all files in canonical order.
    pub combined_sha256_hex: String,
    /// Concatenated corpus bytes used to derive deterministic token streams.
    bytes: Vec<u8>,
}

impl GroundedCorpus {
    /// Locates the workspace root containing `calibration_data/calibration_datav3_small.txt`
    /// starting from `CARGO_MANIFEST_DIR` or the current working directory.
    #[must_use]
    pub fn find_repo_root() -> PathBuf {
        let candidates = [
            PathBuf::from("."),
            PathBuf::from(".."),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
        ];
        for cand in candidates {
            if cand.join(CORPUS_FILES[0]).is_file() {
                return cand;
            }
        }
        PathBuf::from(".")
    }

    /// Loads and SHA-256-hashes the 3 canonical repository corpus files relative to `repo_root`.
    ///
    /// # Errors
    /// Returns an `io::Error` if any required corpus file cannot be read.
    pub fn load(repo_root: &Path) -> std::io::Result<Self> {
        let mut files = Vec::with_capacity(CORPUS_FILES.len());
        let mut combined_hasher = Sha256::new();
        let mut all_bytes = Vec::new();
        let mut total_bytes = 0usize;

        for rel in CORPUS_FILES {
            let full = repo_root.join(rel);
            let data = fs::read(&full)?;
            let digest = Sha256::digest_hex(&data);
            combined_hasher.update(rel.as_bytes());
            combined_hasher.update(b"\0");
            combined_hasher.update(&data);
            total_bytes += data.len();
            all_bytes.extend_from_slice(&data);
            files.push(CorpusFileProvenance {
                relative_path: rel.to_string(),
                byte_len: data.len(),
                sha256_hex: digest,
            });
        }

        let combined_sha256_hex = hex_lower(&combined_hasher.finalize());
        Ok(Self {
            files,
            total_bytes,
            combined_sha256_hex,
            bytes: all_bytes,
        })
    }

    /// Derives a deterministic token-ID sequence of length `len` in `0..vocab_size` from the
    /// grounded corpus bytes at a seeded offset.
    #[must_use]
    pub fn derive_tokens(&self, seed: u64, len: usize, vocab_size: usize) -> Vec<u32> {
        let mut rng = SplitMix64::new(seed);
        let n = self.bytes.len().max(1);
        let start = (rng.next_u64() as usize) % n;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let b0 = self.bytes[(start + i * 3) % n] as u64;
            let b1 = self.bytes[(start + i * 3 + 1) % n] as u64;
            let mix = (b0 << 8) ^ b1 ^ (rng.next_u64() & 0xff);
            out.push((mix % (vocab_size as u64)) as u32);
        }
        out
    }

    /// Builds a mixed-length batch of [`VerifySequenceSpec`] derived from the grounded corpus
    /// for bit-exact token-level equivalence verification.
    #[must_use]
    pub fn build_verify_specs(&self, seed: u64, vocab_size: usize) -> Vec<VerifySequenceSpec> {
        let lengths = [(19usize, 12usize), (67, 16), (113, 14), (35, 18), (91, 10)];
        lengths
            .iter()
            .enumerate()
            .map(|(idx, &(prompt_len, decode_steps))| VerifySequenceSpec {
                seq_id: idx,
                prompt_tokens: self.derive_tokens(
                    seed.wrapping_add((idx as u64 + 1) * 0x9e37),
                    prompt_len,
                    vocab_size,
                ),
                decode_steps,
            })
            .collect()
    }

    /// Builds a single-user (`C = 1`) interactive workload: one cold `32,768`-token repo prefill
    /// with `128` generated tokens.
    #[must_use]
    pub fn build_single_user_workload(&self, slo: SloTargets) -> Vec<InferenceRequest> {
        vec![InferenceRequest {
            id: 0,
            class: RequestClass::LongRepoPrefill,
            arrival_ms: 0.0,
            prompt_tokens: 32_768,
            warm_prefix_tokens: 0,
            output_tokens: 128,
            slo_ttft_ms: slo.long_ttft_p95_ms,
            slo_itl_ms: slo.itl_p95_ms,
            session_id: Some(0),
            turn_index: 0,
            depends_on: None,
        }]
    }

    /// Builds the canonical HIP-1346 20-turn coding-agent loop workload:
    /// - Turn 0 (cold): `32,768` repo prefill + `256` output tokens (`33,024` context)
    /// - Turns 1..=20 (warm): prefix reused + `4,800` new tool/diff tokens + `256` output tokens
    ///   reaching `129,088` peak context (`5,120` warm generated tokens + `256` cold = `5,376` total).
    ///
    /// When `bg_interactive_count > 0`, interleaves short `256`-token interactive requests
    /// arriving throughout the session to stress head-of-line isolation.
    #[must_use]
    pub fn build_coding_agent_20_turn_workload(
        &self,
        seed: u64,
        slo: SloTargets,
        bg_interactive_count: usize,
    ) -> Vec<InferenceRequest> {
        let mut rng = SplitMix64::new(seed ^ 0xA6E7_0020);
        let mut reqs = Vec::with_capacity(21 + bg_interactive_count);
        let mut context_len = 0usize;

        // Turn 0: Cold 32,768-token repository prefill.
        reqs.push(InferenceRequest {
            id: 0,
            class: RequestClass::CodingAgentTurn,
            arrival_ms: 0.0,
            prompt_tokens: 32_768,
            warm_prefix_tokens: 0,
            output_tokens: 256,
            slo_ttft_ms: slo.long_ttft_p95_ms,
            slo_itl_ms: slo.itl_p95_ms,
            session_id: Some(0),
            turn_index: 0,
            depends_on: None,
        });
        context_len += 32_768 + 256;

        // Turns 1..=20: 4,800-token tool/diff delta prefill + 256 output tokens.
        for turn in 1..=20 {
            let prompt_tokens = context_len + 4_800;
            reqs.push(InferenceRequest {
                id: turn,
                class: RequestClass::CodingAgentTurn,
                arrival_ms: 0.0,
                prompt_tokens,
                warm_prefix_tokens: context_len,
                output_tokens: 256,
                slo_ttft_ms: slo.long_ttft_p95_ms * 0.5,
                slo_itl_ms: slo.itl_p95_ms,
                session_id: Some(0),
                turn_index: turn,
                depends_on: Some(turn - 1),
            });
            context_len = prompt_tokens + 256;
        }

        // Optional concurrent short interactive background requests (starting at t=0.0 so
        // multi-tenant mode is genuinely active alongside Turn 0 and across all 20 warm turns).
        for i in 0..bg_interactive_count {
            let jitter = if i == 0 {
                0.0
            } else {
                (self.bytes[i % self.bytes.len().max(1)] as f64) * 0.25
            };
            let arrival_ms = if i == 0 {
                0.0
            } else {
                (i as f64) * 900.0 + jitter
            };
            let prompt = 200 + rng.next_range_usize(0, 112);
            let output = 48 + rng.next_range_usize(0, 48);
            reqs.push(InferenceRequest {
                id: 21 + i,
                class: RequestClass::ShortInteractive,
                arrival_ms,
                prompt_tokens: prompt,
                warm_prefix_tokens: 0,
                output_tokens: output,
                slo_ttft_ms: slo.short_ttft_p95_ms,
                slo_itl_ms: slo.itl_p95_ms,
                session_id: None,
                turn_index: 0,
                depends_on: None,
            });
        }
        reqs
    }

    /// Builds a mixed long/short workload at concurrency `C ∈ {1, 4, 16, 64, 256}`.
    ///
    /// Mix proportions (for `C >= 4`):
    /// - ~30% long repository prefills (`16K..32K` tokens, `48..80` output tokens)
    /// - ~70% short interactive chats (`192..384` tokens, `32..64` output tokens)
    ///
    /// Long prefills arrive staggered across the active decoding window so unchunked prefill-first
    /// scheduling authentically exposes decode ITL stalls while decode-priority exposes TTFT starvation.
    #[must_use]
    pub fn build_mixed_concurrency_workload(
        &self,
        concurrency: usize,
        seed: u64,
        slo: SloTargets,
    ) -> Vec<InferenceRequest> {
        if concurrency <= 1 {
            return self.build_single_user_workload(slo);
        }
        let mut rng = SplitMix64::new(seed ^ ((concurrency as u64) * 0x517c_c1b7));
        let mut reqs = Vec::with_capacity(concurrency);
        let n_bytes = self.bytes.len().max(1);

        let long_count = ((concurrency * 3) / 10).max(1);
        let mut longs_assigned = 0usize;
        for i in 0..concurrency {
            let corpus_byte = self.bytes[(i * 97 + (seed as usize)) % n_bytes] as usize;
            let stride = (concurrency / long_count).max(2);
            let is_long = i >= 2 && (i - 2) % stride == 0 && longs_assigned < long_count;
            let (class, prompt_tokens, output_tokens, arrival_ms, slo_ttft) = if is_long {
                let long_idx = longs_assigned;
                longs_assigned += 1;
                let p = if long_idx.is_multiple_of(2) {
                    32_768
                } else {
                    16_384 + (corpus_byte * 32)
                };
                let o = 48 + (corpus_byte % 33);
                // Stagger long arrivals so they arrive while short chats are actively decoding.
                let arr = 40.0 + (long_idx as f64) * 65.0 + rng.next_f64() * 12.0;
                (
                    RequestClass::LongRepoPrefill,
                    p,
                    o,
                    arr,
                    slo.long_ttft_p95_ms,
                )
            } else {
                let p = 192 + (corpus_byte % 192);
                let o = 32 + rng.next_range_usize(0, 32);
                let arr = if i < 2 {
                    (i as f64) * 3.0
                } else {
                    8.0 + ((i as f64) * 16.0).min(1200.0) + rng.next_f64() * 10.0
                };
                (
                    RequestClass::ShortInteractive,
                    p,
                    o,
                    arr,
                    slo.short_ttft_p95_ms,
                )
            };

            reqs.push(InferenceRequest {
                id: i,
                class,
                arrival_ms,
                prompt_tokens,
                warm_prefix_tokens: 0,
                output_tokens,
                slo_ttft_ms: slo_ttft,
                slo_itl_ms: slo.itl_p95_ms,
                session_id: None,
                turn_index: 0,
                depends_on: None,
            });
        }
        reqs
    }

    /// Builds the `C = 256` KV-memory oversubscription stress workload (Negative Regime 2):
    /// 144 long `32,768`-token repo prefills + 112 short `256`-token chats arriving in a tight
    /// burst, creating exact unbounded KV+GDN demand (`> 100 GiB` on `9B-Hybrid`) that exceeds
    /// the H100-SXM5 80 GiB KV+GDN pool.
    #[must_use]
    pub fn build_oversubscription_256_workload(
        &self,
        seed: u64,
        slo: SloTargets,
    ) -> Vec<InferenceRequest> {
        let mut rng = SplitMix64::new(seed ^ 0x0256_00FF);
        let mut reqs = Vec::with_capacity(256);
        for i in 0..256 {
            let is_long = i % 16 < 9; // 144 long 32K prompts + 112 short interactive chats.
            if is_long {
                reqs.push(InferenceRequest {
                    id: i,
                    class: RequestClass::LongRepoPrefill,
                    arrival_ms: (i as f64) * 1.5,
                    prompt_tokens: 32_768,
                    warm_prefix_tokens: 0,
                    output_tokens: 32,
                    slo_ttft_ms: slo.long_ttft_p95_ms,
                    slo_itl_ms: slo.itl_p95_ms,
                    session_id: None,
                    turn_index: 0,
                    depends_on: None,
                });
            } else {
                reqs.push(InferenceRequest {
                    id: i,
                    class: RequestClass::ShortInteractive,
                    arrival_ms: (i as f64) * 1.5 + rng.next_f64() * 2.0,
                    prompt_tokens: 256,
                    warm_prefix_tokens: 0,
                    output_tokens: 48,
                    slo_ttft_ms: slo.short_ttft_p95_ms,
                    slo_itl_ms: slo.itl_p95_ms,
                    session_id: None,
                    turn_index: 0,
                    depends_on: None,
                });
            }
        }
        reqs
    }
}
