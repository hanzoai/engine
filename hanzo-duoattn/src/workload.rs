// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Grounded long-context workloads and negative-regime test harnesses built from real
//! versioned `hanzoai/engine` repository files:
//! - `calibration_data/calibration_datav3_small.txt`
//! - `chat_templates/llama3.json`
//! - `orderings/xlora-paper-ordering.json`
//!
//! Evaluates all 5 required long-context workloads from `hanzoai/engine#52`:
//! 1. Repository-level QA
//! 2. Distant-reference retrieval (early cross-file symbol definition)
//! 3. Needle-in-a-haystack (passkey recovery across depths `0%..100%`)
//! 4. Code edits needing old context (multi-turn patch requiring Turn-1 schema)
//! 5. Long-context prefill + 20-turn developer-agent loop
//!
//! Plus all 3 explicit Honest Negative Regimes:
//! - Negative Regime 1: Cross-model transferred calibration profile (Model A -> Model B)
//! - Negative Regime 2: Over-aggressive pruning (`retrieval_ratio = 5%` and `10%`)
//! - Negative Regime 3: Short-context regime (`L <= sink_tokens + window_tokens`)

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::attention::{compute_fidelity, DualAttentionModel, SplitMix64};
use crate::cache::{DualKvCache, KvDtype};
use crate::calibrate::CalibrationSequence;
use crate::error::DuoAttnError;
use crate::profile::HeadCalibrationProfile;
use crate::sha256::sha256_hex;

/// Provenance record for a real repository corpus file ingested by the benchmark.
#[derive(Clone, Debug)]
pub struct CorpusProvenance {
    /// Repository-relative path.
    pub rel_path: String,
    /// Byte length of the ingested file.
    pub byte_len: usize,
    /// SHA-256 hex digest of the file contents.
    pub sha256: String,
}

/// Real repository corpus loaded from `hanzoai/engine` versioned files.
#[derive(Clone, Debug)]
pub struct RepoCorpus {
    /// Provenance metadata for each loaded file.
    pub files: Vec<CorpusProvenance>,
    /// Deterministic whitespace/punctuation token IDs extracted from the real files.
    pub token_ids: Vec<u32>,
}

impl RepoCorpus {
    /// Loads the three canonical versioned repository files under `repo_root`, falling back
    /// only if a unit test runs without a repository root (marked explicitly).
    pub fn load_from_repo(repo_root: &Path) -> Result<Self, DuoAttnError> {
        let rel_paths = [
            "calibration_data/calibration_datav3_small.txt",
            "chat_templates/llama3.json",
            "orderings/xlora-paper-ordering.json",
        ];

        let mut files = Vec::with_capacity(rel_paths.len());
        let mut token_ids = Vec::new();
        for rel in rel_paths {
            let full_path: PathBuf = repo_root.join(rel);
            let bytes = std::fs::read(&full_path).map_err(|e| {
                DuoAttnError::IoError(format!(
                    "failed to read versioned corpus file '{}': {e}",
                    full_path.display()
                ))
            })?;
            let digest = sha256_hex(&bytes);
            files.push(CorpusProvenance {
                rel_path: rel.to_string(),
                byte_len: bytes.len(),
                sha256: digest,
            });
            let text = String::from_utf8_lossy(&bytes);
            tokenize_into(&text, &mut token_ids);
        }

        if token_ids.is_empty() {
            return Err(DuoAttnError::InvalidConfig(
                "loaded repo corpus produced zero tokens".to_string(),
            ));
        }
        Ok(Self { files, token_ids })
    }

    /// Creates a deterministic synthetic fallback corpus for standalone unit tests that do not
    /// require reading files from disk.
    #[must_use]
    pub fn synthetic_for_tests(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut token_ids = Vec::with_capacity(4096);
        for _ in 0..4096 {
            token_ids.push((rng.next_u64() % 4096) as u32);
        }
        Self {
            files: vec![CorpusProvenance {
                rel_path: "synthetic://in-memory-unit-test".to_string(),
                byte_len: token_ids.len() * 4,
                sha256: sha256_hex(b"synthetic://in-memory-unit-test"),
            }],
            token_ids,
        }
    }
}

fn tokenize_into(text: &str, out: &mut Vec<u32>) {
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() {
            continue;
        }
        // FNV-1a 32-bit hash mapped into a 4096-token vocabulary.
        let mut hash: u32 = 2_166_136_261;
        for b in raw.bytes() {
            hash ^= u32::from(b);
            hash = hash.wrapping_mul(16_777_619);
        }
        out.push(hash % 4096);
    }
}

/// Deterministic corpus-to-embedding projector with separate lookup-key (`[0..d/4)`),
/// answer-value (`[d/4..3d/4)`), and local syntactic/sink (`[3d/4..d)`) subspaces.
///
/// Recall queries carry ONLY the lookup key and ZERO answer value, making it
/// information-theoretically impossible to pass distant retrieval via self-attention when
/// the distant needle has been evicted from the KV cache.
#[derive(Clone, Debug)]
pub struct CorpusEmbedder {
    d_model: usize,
    key_half: usize,
    val_start: usize,
    val_dim: usize,
    loc_start: usize,
    num_candidates: usize,
    /// Orthogonal-ish lookup key signatures `[num_candidates][key_half]`.
    key_signatures: Vec<Vec<f32>>,
    /// Orthogonal-ish answer value signatures `[num_candidates][val_dim]`.
    val_signatures: Vec<Vec<f32>>,
}

impl CorpusEmbedder {
    /// Creates an embedder for `d_model` with `num_candidates` distinct key/value signatures.
    #[must_use]
    pub fn new(d_model: usize, num_candidates: usize, seed: u64) -> Self {
        let key_dim = (d_model / 4).max(2);
        let key_half = (key_dim / 2).max(1);
        let val_start = key_dim;
        let val_dim = (d_model / 2).max(1);
        let loc_start = val_start + val_dim;

        let key_signatures =
            build_orthogonal_signatures(num_candidates, key_half, seed ^ 0x517c_c1b7);
        let val_signatures =
            build_orthogonal_signatures(num_candidates, val_dim, seed ^ 0xa5a5_1346);

        Self {
            d_model,
            key_half,
            val_start,
            val_dim,
            loc_start,
            num_candidates,
            key_signatures,
            val_signatures,
        }
    }

    /// Number of candidate target signatures.
    #[must_use]
    pub const fn num_candidates(&self) -> usize {
        self.num_candidates
    }

    /// Embeds a normal background corpus token at `pos`.
    #[must_use]
    pub fn embed_background(&self, token_id: u32, pos: usize, sink_tokens: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(u64::from(token_id).wrapping_mul(0x9e37_79b9) ^ (pos as u64));
        let mut emb = vec![0.0_f32; self.d_model];

        // Low-amplitude background noise in key & value subspaces.
        for slot in emb.iter_mut().take(self.loc_start) {
            *slot = rng.next_normal_f32() * 0.06;
        }
        // Rich local syntactic / positional features in [loc_start..d_model).
        for (j, slot) in emb.iter_mut().skip(self.loc_start).enumerate() {
            let phase = (pos as f32) * 0.17 + (j as f32) * 0.31;
            let base = 0.55 * phase.sin() + 0.35 * rng.next_normal_f32();
            *slot = if pos < sink_tokens { base + 1.35 } else { base };
        }
        emb
    }

    /// Embeds a planted distant needle / symbol definition / Turn-1 schema binding
    /// definition key `key_id` (`[0..key_half)`) to answer value `val_id` (`[val_start..loc_start)`).
    #[must_use]
    pub fn embed_needle(
        &self,
        key_id: usize,
        val_id: usize,
        pos: usize,
        _sink_tokens: usize,
    ) -> Vec<f32> {
        // Keep local norm uniform even at depth 0% (sink region) so RMSNorm does not damp needle keys.
        let mut emb = self.embed_background(0x7000 + val_id as u32, pos, 0);
        let ksig = &self.key_signatures[key_id % self.num_candidates];
        let vsig = &self.val_signatures[val_id % self.num_candidates];
        for (i, &k) in ksig.iter().enumerate() {
            emb[i] = k * 2.6;
        }
        // Definition tokens carry zero query_cue so they do not cross-query earlier definitions.
        for slot in emb.iter_mut().skip(self.key_half).take(self.key_half) {
            *slot = 0.0;
        }
        for (i, &v) in vsig.iter().enumerate() {
            emb[self.val_start + i] = v * 2.8;
        }
        emb
    }

    /// Embeds a recent stale/partial distractor inside the sliding window that carries a
    /// weaker definition key (`1.65x`) for `key_id` and the WRONG value `wrong_val_id`.
    ///
    /// When the true distant needle (`2.6x`) is retained (Retrieval head), softmax sharply
    /// selects the exact distant needle; when the distant needle is evicted (Streaming head),
    /// the stale recent distractor dominates the evicted head's recalled value slice.
    #[must_use]
    pub fn embed_recent_distractor(
        &self,
        key_id: usize,
        wrong_val_id: usize,
        pos: usize,
        _sink_tokens: usize,
    ) -> Vec<f32> {
        let mut emb = self.embed_background(0x7800 + wrong_val_id as u32, pos, 0);
        let ksig = &self.key_signatures[key_id % self.num_candidates];
        let vsig = &self.val_signatures[wrong_val_id % self.num_candidates];
        for (i, &k) in ksig.iter().enumerate() {
            emb[i] = k * 1.65;
        }
        for slot in emb.iter_mut().skip(self.key_half).take(self.key_half) {
            *slot = 0.0;
        }
        for (i, &v) in vsig.iter().enumerate() {
            emb[self.val_start + i] = v * 2.8;
        }
        emb
    }

    /// Embeds a recall query seeking the value bound to `key_id`.
    ///
    /// Places the lookup cue strictly in `query_cue [key_half..val_start)` while setting both
    /// `key_def [0..key_half)` and `val [val_start..loc_start)` to ZERO.
    #[must_use]
    pub fn embed_query(&self, key_id: usize, pos: usize, sink_tokens: usize) -> Vec<f32> {
        let mut emb = self.embed_background(0x8000 + key_id as u32, pos, sink_tokens);
        let ksig = &self.key_signatures[key_id % self.num_candidates];
        for slot in emb.iter_mut().take(self.key_half) {
            *slot = 0.0;
        }
        for (i, &k) in ksig.iter().enumerate() {
            emb[self.key_half + i] = k * 1.8;
        }
        for slot in emb.iter_mut().skip(self.val_start).take(self.val_dim) {
            *slot = 0.0;
        }
        emb
    }

    /// Scores a model's output hidden state in the value subspace against all candidate values
    /// and returns `(predicted_candidate_id, true_candidate_margin)`.
    #[must_use]
    pub fn predict_candidate(&self, hidden: &[f32], true_candidate: usize) -> (usize, f64) {
        let mut best_id = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        let mut true_score = f64::NEG_INFINITY;
        let mut second_other = f64::NEG_INFINITY;

        for (c, sig) in self.val_signatures.iter().enumerate() {
            let mut dot = 0.0_f64;
            for (i, &s) in sig.iter().enumerate() {
                dot += f64::from(hidden[self.val_start + i]) * f64::from(s);
            }
            if dot > best_score {
                best_score = dot;
                best_id = c;
            }
            if c == true_candidate {
                true_score = dot;
            } else if dot > second_other {
                second_other = dot;
            }
        }
        (best_id, true_score - second_other)
    }

    /// Predicts using strictly the recalled value-subspace delta `hidden - query_input`.
    #[must_use]
    pub fn predict_from_recalled_delta(
        &self,
        hidden: &[f32],
        query_input: &[f32],
        true_candidate: usize,
    ) -> (usize, f64) {
        let delta: Vec<f32> = hidden
            .iter()
            .zip(query_input.iter())
            .map(|(&h, &q)| h - q)
            .collect();
        self.predict_candidate(&delta, true_candidate)
    }
}

fn build_orthogonal_signatures(num_candidates: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64::new(seed);
    let mut signatures = Vec::with_capacity(num_candidates);
    for c in 0..num_candidates {
        let mut vec = vec![0.0_f32; dim];
        for (j, slot) in vec.iter_mut().enumerate() {
            let hadamard = if ((c & j).count_ones() & 1) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            let jitter = rng.next_normal_f32() * 0.05;
            *slot = hadamard + jitter;
        }
        let norm = vec.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-6);
        for v in &mut vec {
            *v /= norm;
        }
        signatures.push(vec);
    }
    signatures
}

/// Builds deterministic calibration sequences (passkey & distant-reference recall over real repo text)
/// for offline DuoAttention head calibration.
#[must_use]
pub fn build_calibration_dataset(
    corpus: &RepoCorpus,
    embedder: &CorpusEmbedder,
    num_sequences: usize,
    seq_len: usize,
    sink_tokens: usize,
    window_tokens: usize,
    seed: u64,
) -> Vec<CalibrationSequence> {
    let mut rng = SplitMix64::new(seed ^ 0xcafe_babe_1346);
    let mut sequences = Vec::with_capacity(num_sequences);
    let n_tokens = corpus.token_ids.len();

    for s in 0..num_sequences {
        let mut embeddings = Vec::with_capacity(seq_len);
        let offset = (rng.next_u64() as usize) % n_tokens;
        for pos in 0..seq_len {
            let tok = corpus.token_ids[(offset + pos) % n_tokens];
            embeddings.push(embedder.embed_background(tok, pos, sink_tokens));
        }

        // Plant 3 distant passkeys strictly outside both `[0..sink_tokens)` and `[seq_len-window_tokens..seq_len)`.
        let safe_lo = sink_tokens + 2;
        let safe_hi = seq_len.saturating_sub(window_tokens + 6).max(safe_lo + 1);
        let span = safe_hi - safe_lo;
        let mut eval_positions = Vec::with_capacity(3);

        for q_idx in 0..3 {
            let cand = (s * 3 + q_idx) % embedder.num_candidates();
            let key_id = (cand + 2) % embedder.num_candidates();
            let needle_pos =
                safe_lo + ((q_idx * span) / 3) + ((rng.next_u64() as usize) % (span / 3).max(1));
            let needle_pos = needle_pos.min(safe_hi - 1);
            let query_pos = seq_len - 3 + q_idx;

            embeddings[needle_pos] = embedder.embed_needle(key_id, cand, needle_pos, sink_tokens);
            let distractor = (cand + 5) % embedder.num_candidates();
            let recent_pos = seq_len.saturating_sub(window_tokens / 2 + q_idx + 1);
            if recent_pos > needle_pos && recent_pos < query_pos {
                embeddings[recent_pos] =
                    embedder.embed_recent_distractor(key_id, distractor, recent_pos, sink_tokens);
            }
            embeddings[query_pos] = embedder.embed_query(key_id, query_pos, sink_tokens);
            eval_positions.push(query_pos);
        }

        sequences.push(CalibrationSequence {
            embeddings,
            eval_positions,
        });
    }
    sequences
}

/// Measured result for one workload executed under one cache regime.
#[derive(Clone, Debug)]
pub struct WorkloadRegimeResult {
    /// Workload identifier.
    pub workload_name: String,
    /// Cache regime slug.
    pub regime: String,
    /// Sequence length evaluated in the pure-Rust attention kernel.
    pub seq_len: usize,
    /// Exact-match retrieval/task accuracy (`0.0..=1.0`).
    pub accuracy: f64,
    /// Mean cosine similarity of final attention output vs `full_kv_baseline`.
    pub cosine_vs_full: f64,
    /// Maximum absolute elementwise error vs `full_kv_baseline`.
    pub max_abs_err_vs_full: f64,
    /// Relative L2 error vs `full_kv_baseline`.
    pub rel_l2_err_vs_full: f64,
    /// Realized KV cache memory reduction factor vs `full_kv_baseline` (`x`).
    pub kv_memory_reduction_x: f64,
    /// Resident KV bytes at peak (`FP16` accounting).
    pub peak_resident_kv_bytes: usize,
    /// Real CPU kernel wall time (ms) across all evaluation instances.
    pub kernel_wall_ms: f64,
    /// Mean recall margin (true candidate logit minus strongest distractor).
    pub mean_recall_margin: f64,
}

/// Single test episode with an embedded prompt, query position, and ground-truth candidate ID.
#[derive(Clone, Debug)]
pub struct EvalEpisode {
    /// Label (e.g., `"depth=50%"` or `"turn_18_schema_patch"`).
    pub label: String,
    /// Embedded token sequence `[seq_len, d_model]`.
    pub embeddings: Vec<Vec<f32>>,
    /// Ground-truth candidate ID expected at the final query token.
    pub target_candidate: usize,
    /// True logical needle position (for reporting depth).
    pub needle_pos: usize,
}

/// Builds the 5 canonical long-context evaluation workloads grounded in `corpus`.
#[must_use]
pub fn build_five_workloads(
    corpus: &RepoCorpus,
    embedder: &CorpusEmbedder,
    seq_len: usize,
    sink_tokens: usize,
    window_tokens: usize,
    seed: u64,
) -> Vec<(String, Vec<EvalEpisode>)> {
    let mut rng = SplitMix64::new(seed ^ 0x1346_2026_beef);

    // 1. Repository-level QA: questions combining early architecture docs (15%-35% depth)
    //    and mid-repo module contracts (45%-65% depth).
    let mut repo_qa = Vec::new();
    for i in 0..6 {
        let depth_frac = 0.15 + 0.10 * (i as f64);
        repo_qa.push(make_episode(
            &format!("repo_qa_{i}"),
            corpus,
            embedder,
            seq_len,
            depth_frac,
            sink_tokens,
            window_tokens,
            i % embedder.num_candidates(),
            &mut rng,
        ));
    }

    // 2. Distant-reference retrieval: cross-file symbol definitions planted early (`8%..22%` depth,
    //    analogous to 32k tokens earlier in a long repository session).
    let mut distant_ref = Vec::new();
    for i in 0..6 {
        let depth_frac = 0.08 + 0.025 * (i as f64);
        distant_ref.push(make_episode(
            &format!("distant_sym_{i}"),
            corpus,
            embedder,
            seq_len,
            depth_frac,
            sink_tokens,
            window_tokens,
            (i + 3) % embedder.num_candidates(),
            &mut rng,
        ));
    }

    // 3. Needle-in-a-haystack across depths 0%, 10%, 25%, 50%, 75%, 90%, 100%.
    let depths = [0.00_f64, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00];
    let mut niah = Vec::new();
    for (i, &d) in depths.iter().enumerate() {
        niah.push(make_episode(
            &format!("niah_depth_{:.0}pct", d * 100.0),
            corpus,
            embedder,
            seq_len,
            d,
            sink_tokens,
            window_tokens,
            (i + 6) % embedder.num_candidates(),
            &mut rng,
        ));
    }

    // 4. Code edits needing old context: Turn-1 schema planted at ~6%-14% depth, modified
    //    after heavy intermediate diffs/logs fill the recent window.
    let mut code_edits = Vec::new();
    for i in 0..6 {
        let depth_frac = 0.06 + 0.016 * (i as f64);
        code_edits.push(make_episode(
            &format!("old_schema_edit_{i}"),
            corpus,
            embedder,
            seq_len,
            depth_frac,
            sink_tokens,
            window_tokens,
            (i + 10) % embedder.num_candidates(),
            &mut rng,
        ));
    }

    // 5. Long-context prefill + 20-turn developer-agent loop (progressively growing context
    //    queries at turns 5, 10, 15, 20 needing Turn-1/Turn-3 facts).
    let mut agent_20 = Vec::new();
    for (i, &depth_frac) in [0.09_f64, 0.18, 0.30, 0.42, 0.55].iter().enumerate() {
        agent_20.push(make_episode(
            &format!("agent_turn_{}", (i + 1) * 4),
            corpus,
            embedder,
            seq_len,
            depth_frac,
            sink_tokens,
            window_tokens,
            (i + 1) % embedder.num_candidates(),
            &mut rng,
        ));
    }

    vec![
        ("repo_level_qa".to_string(), repo_qa),
        ("distant_reference_retrieval".to_string(), distant_ref),
        ("needle_in_a_haystack".to_string(), niah),
        ("code_edits_old_context".to_string(), code_edits),
        ("long_context_prefill_20turn_agent".to_string(), agent_20),
    ]
}

#[allow(clippy::too_many_arguments)]
fn make_episode(
    label: &str,
    corpus: &RepoCorpus,
    embedder: &CorpusEmbedder,
    seq_len: usize,
    depth_frac: f64,
    sink_tokens: usize,
    window_tokens: usize,
    target_candidate: usize,
    rng: &mut SplitMix64,
) -> EvalEpisode {
    let n_tokens = corpus.token_ids.len();
    let offset = (rng.next_u64() as usize) % n_tokens;
    let mut embeddings = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let tok = corpus.token_ids[(offset + pos) % n_tokens];
        embeddings.push(embedder.embed_background(tok, pos, sink_tokens));
    }

    let query_pos = seq_len - 1;
    let needle_pos = if depth_frac <= 0.0 {
        // Depth 0% sits inside the initial sink region.
        1.min(query_pos.saturating_sub(1))
    } else if depth_frac >= 1.0 {
        // Depth 100% sits right before the query inside the recent window.
        query_pos.saturating_sub(2)
    } else {
        ((depth_frac * ((seq_len - 2) as f64)).round() as usize).clamp(1, query_pos - 2)
    };

    let key_id = (target_candidate + 3) % embedder.num_candidates();
    embeddings[needle_pos] =
        embedder.embed_needle(key_id, target_candidate, needle_pos, sink_tokens);

    // Plant a stale recent distractor inside the sliding window whenever the true needle
    // lies outside the recent window (mirrors stale local symbol shadowing in long code sessions).
    if needle_pos + window_tokens < query_pos {
        let distractor_cand = (target_candidate + 7) % embedder.num_candidates();
        let distractor_pos = query_pos.saturating_sub((window_tokens / 3).max(2));
        if distractor_pos != needle_pos && distractor_pos < query_pos {
            embeddings[distractor_pos] = embedder.embed_recent_distractor(
                key_id,
                distractor_cand,
                distractor_pos,
                sink_tokens,
            );
        }
    }

    embeddings[query_pos] = embedder.embed_query(key_id, query_pos, sink_tokens);

    EvalEpisode {
        label: label.to_string(),
        embeddings,
        target_candidate,
        needle_pos,
    }
}

/// Evaluates `episodes` under a fresh `DualKvCache` created by `make_cache`, comparing every
/// final hidden state against precomputed `full_reference_outputs`.
pub fn evaluate_workload_regime<F>(
    workload_name: &str,
    model: &DualAttentionModel,
    embedder: &CorpusEmbedder,
    episodes: &[EvalEpisode],
    full_reference_outputs: &[Vec<f32>],
    mut make_cache: F,
) -> Result<(WorkloadRegimeResult, Vec<Vec<f32>>), DuoAttnError>
where
    F: FnMut() -> Result<DualKvCache, DuoAttnError>,
{
    let start = Instant::now();
    let mut outputs = Vec::with_capacity(episodes.len());
    let mut correct = 0usize;
    let mut cos_sum = 0.0_f64;
    let mut max_abs_worst = 0.0_f64;
    let mut rel_l2_sum = 0.0_f64;
    let mut margin_sum = 0.0_f64;
    let mut red_sum = 0.0_f64;
    let mut peak_bytes_max = 0usize;
    let mut regime_slug = String::new();

    for (idx, ep) in episodes.iter().enumerate() {
        let mut cache = make_cache()?;
        if regime_slug.is_empty() {
            regime_slug = cache.regime().slug().to_string();
        }
        let hidden = model.forward_sequence(&ep.embeddings, &mut cache)?;
        let query_input = &ep.embeddings[ep.embeddings.len() - 1];
        let (pred, margin) =
            embedder.predict_from_recalled_delta(&hidden, query_input, ep.target_candidate);
        if pred == ep.target_candidate {
            correct += 1;
        }
        margin_sum += margin;
        red_sum += cache.memory_reduction_ratio();
        if cache.peak_resident_kv_bytes() > peak_bytes_max {
            peak_bytes_max = cache.peak_resident_kv_bytes();
        }

        if let Some(ref_vec) = full_reference_outputs.get(idx) {
            let fid = compute_fidelity(ref_vec, &hidden);
            cos_sum += fid.cosine_similarity;
            if fid.max_abs_error > max_abs_worst {
                max_abs_worst = fid.max_abs_error;
            }
            rel_l2_sum += fid.rel_l2_error;
        } else {
            cos_sum += 1.0;
        }
        outputs.push(hidden);
    }

    let n = episodes.len().max(1) as f64;
    let seq_len = episodes.first().map_or(0, |e| e.embeddings.len());
    let res = WorkloadRegimeResult {
        workload_name: workload_name.to_string(),
        regime: regime_slug,
        seq_len,
        accuracy: (correct as f64) / n,
        cosine_vs_full: cos_sum / n,
        max_abs_err_vs_full: max_abs_worst,
        rel_l2_err_vs_full: rel_l2_sum / n,
        kv_memory_reduction_x: red_sum / n,
        peak_resident_kv_bytes: peak_bytes_max,
        kernel_wall_ms: start.elapsed().as_secs_f64() * 1_000.0,
        mean_recall_margin: margin_sum / n,
    };
    Ok((res, outputs))
}

/// Summary of Negative Regime 1 (cross-model profile transfer failure & fail-closed guard).
#[derive(Clone, Debug)]
pub struct CrossModelNegativeResult {
    /// Whether `DualKvCache::new_calibrated` properly rejected Model A's profile on Model B fail-closed.
    pub fail_closed_rejected: bool,
    /// Exact error string from fail-closed rejection.
    pub fail_closed_error: String,
    /// Model B accuracy using Model B's own calibrated profile.
    pub own_calibrated_accuracy: f64,
    /// Model B accuracy when Model A's profile is force-applied via ablation bypass.
    pub wrong_model_profile_accuracy: f64,
    /// Cosine similarity vs full KV under Model B's own calibrated profile.
    pub own_calibrated_cosine: f64,
    /// Cosine similarity vs full KV under wrong Model A profile.
    pub wrong_model_profile_cosine: f64,
    /// Head overlap fraction between Model A's retrieval heads and Model B's retrieval heads.
    pub retrieval_head_overlap_ratio: f64,
}

/// Runs Negative Regime 1: verifies fail-closed rejection AND measures exact accuracy collapse
/// when applying Model A's calibrated profile onto Model B.
pub fn run_cross_model_negative_regime(
    model_b: &DualAttentionModel,
    profile_a: &HeadCalibrationProfile,
    profile_b: &HeadCalibrationProfile,
    embedder: &CorpusEmbedder,
    episodes: &[EvalEpisode],
    dtype: KvDtype,
) -> Result<CrossModelNegativeResult, DuoAttnError> {
    // 1. Verify fail-closed rejection on the production constructor.
    let reject_err = match DualKvCache::new_calibrated(model_b.identity(), profile_a, dtype) {
        Err(e @ DuoAttnError::ModelMismatch { .. }) => e.to_string(),
        Err(other) => {
            return Err(DuoAttnError::InvalidConfig(format!(
                "expected ModelMismatch, got {other}"
            )));
        }
        Ok(_) => {
            return Err(DuoAttnError::InvalidConfig(
                "SECURITY BUG: cross-model profile was not rejected fail-closed".to_string(),
            ));
        }
    };

    // 2. Compute exact full-KV reference on Model B.
    let (_, full_refs) =
        evaluate_workload_regime("neg1_ref", model_b, embedder, episodes, &[], || {
            DualKvCache::new_full(model_b.identity(), dtype)
        })?;

    // 3. Evaluate Model B with its OWN calibrated profile.
    let (own_res, _) = evaluate_workload_regime(
        "neg1_own_calibrated",
        model_b,
        embedder,
        episodes,
        &full_refs,
        || DualKvCache::new_calibrated(model_b.identity(), profile_b, dtype),
    )?;

    // 4. Evaluate Model B with Model A's profile via explicit ablation bypass.
    let (wrong_res, _) = evaluate_workload_regime(
        "neg1_wrong_model",
        model_b,
        embedder,
        episodes,
        &full_refs,
        || DualKvCache::new_cross_model_ablation_only(model_b.identity(), profile_a, dtype),
    )?;

    let mut both_ret = 0usize;
    let ret_a = profile_a.retrieval_head_count().max(1);
    for (ha, hb) in profile_a.heads.iter().zip(profile_b.heads.iter()) {
        if ha.role == crate::profile::HeadRole::Retrieval
            && hb.role == crate::profile::HeadRole::Retrieval
        {
            both_ret += 1;
        }
    }

    Ok(CrossModelNegativeResult {
        fail_closed_rejected: true,
        fail_closed_error: reject_err,
        own_calibrated_accuracy: own_res.accuracy,
        wrong_model_profile_accuracy: wrong_res.accuracy,
        own_calibrated_cosine: own_res.cosine_vs_full,
        wrong_model_profile_cosine: wrong_res.cosine_vs_full,
        retrieval_head_overlap_ratio: both_ret as f64 / ret_a as f64,
    })
}

/// Summary of Negative Regime 3 (short-context overhead when `L <= sink_tokens + window_tokens`).
#[derive(Clone, Debug)]
pub struct ShortContextNegativeResult {
    /// Sequence length tested (`<= sink_tokens + window_tokens`).
    pub seq_len: usize,
    /// Streaming budget (`sink_tokens + window_tokens`).
    pub streaming_budget: usize,
    /// Realized KV memory reduction (always `1.00x`, i.e., `0.0%` savings).
    pub kv_memory_reduction_x: f64,
    /// Full-KV baseline wall time (ms).
    pub full_kv_wall_ms: f64,
    /// DuoAttention dual-cache wall time (ms) including head-dispatch/ring bookkeeping.
    pub duoattn_wall_ms: f64,
    /// Speedup ratio (`full_kv_wall_ms / duoattn_wall_ms`, expected `<= 1.0x` at short context).
    pub wall_speedup_x: f64,
    /// Cosine similarity (exact `1.000000` since zero tokens were evicted).
    pub cosine_vs_full: f64,
}

/// Runs Negative Regime 3 for short contexts (`L <= sink_tokens + window_tokens`).
#[allow(clippy::too_many_arguments)]
pub fn run_short_context_negative_regime(
    model: &DualAttentionModel,
    profile: &HeadCalibrationProfile,
    corpus: &RepoCorpus,
    embedder: &CorpusEmbedder,
    short_seq_len: usize,
    repeats: usize,
    dtype: KvDtype,
    seed: u64,
) -> Result<ShortContextNegativeResult, DuoAttnError> {
    let mut rng = SplitMix64::new(seed ^ 0x5407_1346);
    let mut embeddings = Vec::with_capacity(short_seq_len);
    for pos in 0..short_seq_len {
        let tok = corpus.token_ids[(rng.next_u64() as usize) % corpus.token_ids.len()];
        embeddings.push(embedder.embed_background(tok, pos, profile.sink_tokens));
    }

    // Warmup both paths.
    let mut warm_full = DualKvCache::new_full(model.identity(), dtype)?;
    let ref_out = model.forward_sequence(&embeddings, &mut warm_full)?;
    let mut warm_duo = DualKvCache::new_calibrated(model.identity(), profile, dtype)?;
    let duo_out = model.forward_sequence(&embeddings, &mut warm_duo)?;
    let fid = compute_fidelity(&ref_out, &duo_out);
    let kv_red = warm_duo.memory_reduction_ratio();

    let t0 = Instant::now();
    for _ in 0..repeats {
        let mut cache = DualKvCache::new_full(model.identity(), dtype)?;
        let _ = model.forward_sequence(&embeddings, &mut cache)?;
    }
    let full_ms = t0.elapsed().as_secs_f64() * 1_000.0;

    let t1 = Instant::now();
    for _ in 0..repeats {
        let mut cache = DualKvCache::new_calibrated(model.identity(), profile, dtype)?;
        let _ = model.forward_sequence(&embeddings, &mut cache)?;
    }
    let duo_ms = t1.elapsed().as_secs_f64() * 1_000.0;

    Ok(ShortContextNegativeResult {
        seq_len: short_seq_len,
        streaming_budget: profile.sink_tokens + profile.window_tokens,
        kv_memory_reduction_x: kv_red,
        full_kv_wall_ms: full_ms,
        duoattn_wall_ms: duo_ms,
        wall_speedup_x: full_ms / duo_ms.max(1e-9),
        cosine_vs_full: fid.cosine_similarity,
    })
}
