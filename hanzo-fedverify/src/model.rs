// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic language-model feature extraction, softmax/LoRA model, local SGD,
//! and independent held-out quality evaluation pinned to versioned `hanzoai/engine`
//! repository corpus files.

use crate::crypto::{sha256, Digest32, SplitMix64};
use crate::identity::hash_f64_slice;
use std::fs;
use std::path::{Path, PathBuf};

/// Number of lexical/structural token classes predicted by the federated model.
pub const NUM_CLASSES: usize = 12;

/// Input context feature dimension per training/validation example.
pub const FEATURE_DIM: usize = 32;

/// Total dense model weight dimension (`NUM_CLASSES * FEATURE_DIM`).
pub const MODEL_DIM: usize = NUM_CLASSES * FEATURE_DIM;

/// Number of real corpus tokens represented per sliding-window example (`4 context + 1 target`).
pub const TOKENS_PER_SAMPLE: u64 = 5;

/// Default versioned repository corpus files relative to workspace root.
pub const PINNED_CORPUS_PATHS: [&str; 3] = [
    "calibration_data/calibration_datav3_small.txt",
    "chat_templates/llama3.json",
    "orderings/xlora-paper-ordering.json",
];

/// Cryptographic pin of a real versioned repository corpus file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusPin {
    /// Repository-relative file path.
    pub relative_path: String,
    /// File length in bytes.
    pub byte_len: u64,
    /// FIPS 180-4 SHA-256 digest of the file contents.
    pub sha256_digest: Digest32,
}

/// Single next-token-class training or evaluation sample.
#[derive(Debug, Clone)]
pub struct TokenSample {
    /// Context feature vector of length `FEATURE_DIM`.
    pub features: [f64; FEATURE_DIM],
    /// Ground-truth next-token class in `0..NUM_CLASSES`.
    pub target_class: usize,
}

/// Partitioning mode across federated participants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Stratified interleaved shards across the corpus (mild natural heterogeneity).
    StratifiedIid,
    /// Extreme Non-IID skew: each participant primarily holds 2 dominant token classes.
    ExtremeNonIid,
}

/// Complete corpus bundle with strictly separated training, independent validation, and test splits.
#[derive(Debug, Clone)]
pub struct CorpusDataset {
    /// Cryptographic pins of all source files loaded.
    pub pins: Vec<CorpusPin>,
    /// Total tokens parsed from the pinned files.
    pub total_tokens: usize,
    /// Pool of training samples (partitioned across participants).
    pub train_pool: Vec<TokenSample>,
    /// Strictly held-out independent validation split (used by `hanzo_fedverify_dso` quality gate).
    pub independent_val: Vec<TokenSample>,
    /// Strictly held-out final test split (never seen by participants or quality gate).
    pub held_out_test: Vec<TokenSample>,
}

impl CorpusDataset {
    /// Loads and cryptographically pins the real versioned `hanzoai/engine` repository files.
    pub fn load_from_repo_root(repo_root: &Path, seed: u64) -> Result<Self, String> {
        let mut pins = Vec::with_capacity(PINNED_CORPUS_PATHS.len());
        let mut combined_tokens: Vec<String> = Vec::new();

        for rel in PINNED_CORPUS_PATHS {
            let full = repo_root.join(rel);
            let bytes = fs::read(&full).map_err(|e| {
                format!("failed to read pinned corpus file {}: {e}", full.display())
            })?;
            let digest = sha256(&bytes);
            pins.push(CorpusPin {
                relative_path: rel.to_string(),
                byte_len: bytes.len() as u64,
                sha256_digest: digest,
            });
            let text = String::from_utf8_lossy(&bytes);
            tokenize_into(&text, &mut combined_tokens);
        }

        Self::from_tokens_and_pins(combined_tokens, pins, seed)
    }

    /// Locates the workspace root automatically (supporting running from workspace root or crate dir).
    #[must_use]
    pub fn find_repo_root() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if manifest_dir
            .parent()
            .is_some_and(|p| p.join(PINNED_CORPUS_PATHS[0]).exists())
        {
            return manifest_dir.parent().expect("parent exists").to_path_buf();
        }
        PathBuf::from(".")
    }

    /// Builds a deterministic corpus dataset directly from raw text (useful for self-contained tests).
    pub fn from_raw_text(text: &str, seed: u64) -> Result<Self, String> {
        let mut tokens = Vec::new();
        tokenize_into(text, &mut tokens);
        let pin = CorpusPin {
            relative_path: "inline_corpus".to_string(),
            byte_len: text.len() as u64,
            sha256_digest: sha256(text.as_bytes()),
        };
        Self::from_tokens_and_pins(tokens, vec![pin], seed)
    }

    fn from_tokens_and_pins(
        tokens: Vec<String>,
        pins: Vec<CorpusPin>,
        seed: u64,
    ) -> Result<Self, String> {
        if tokens.len() < 128 {
            return Err(format!(
                "corpus too small: need at least 128 tokens, found {}",
                tokens.len()
            ));
        }

        let classes: Vec<usize> = tokens.iter().map(|t| classify_token(t)).collect();
        let mut samples = Vec::with_capacity(tokens.len().saturating_sub(4));

        for i in 4..tokens.len() {
            let mut features = [0.0f64; FEATURE_DIM];
            let c1 = classes[i - 1];
            let c2 = classes[i - 2];
            let c3 = classes[i - 3];
            let c4 = classes[i - 4];

            // 0..12: one-hot of immediate predecessor token class
            features[c1] = 1.0;

            // 12..24: decayed context-window token class histogram
            features[12 + c2] += 0.65;
            features[12 + c3] += 0.35;
            features[12 + c4] += 0.20;

            // 24..28: lexical morphology indicators of preceding two tokens
            let prev = &tokens[i - 1];
            let prev2 = &tokens[i - 2];
            features[24] = ((prev.len().min(16)) as f64) / 16.0;
            features[25] = ((prev2.len().min(16)) as f64) / 16.0;
            features[26] = if prev.bytes().any(|b| b.is_ascii_digit()) {
                1.0
            } else {
                0.0
            };
            features[27] = if prev.bytes().next().is_some_and(|b| b.is_ascii_uppercase()) {
                1.0
            } else {
                0.0
            };

            // 28..31: deterministic bigram interaction buckets
            let bigram_hash = (c2 * 31 + c1 * 17 + (c3 * 7)) % 3;
            features[28 + bigram_hash] = 1.0;

            // 31: bias unit
            features[31] = 1.0;

            samples.push(TokenSample {
                features,
                target_class: classes[i],
            });
        }

        // Cap sample count to a clean, fast deterministic subset while preserving full corpus coverage.
        let total = samples.len();
        let mut indices: Vec<usize> = (0..total).collect();
        let mut rng = SplitMix64::new(seed ^ 0x4841_4e5a_4f5f_3733);
        for i in (1..indices.len()).rev() {
            let j = rng.next_usize(i + 1);
            indices.swap(i, j);
        }

        // Use up to 12,000 samples (9,000 train pool, 1,500 independent val, 1,500 held-out test).
        let usable = total.min(12_000);
        let val_count = (usable * 125) / 1000;
        let test_count = (usable * 125) / 1000;
        let train_count = usable - val_count - test_count;

        let mut train_pool = Vec::with_capacity(train_count);
        let mut independent_val = Vec::with_capacity(val_count);
        let mut held_out_test = Vec::with_capacity(test_count);

        for (pos, &idx) in indices[..usable].iter().enumerate() {
            let sample = samples[idx].clone();
            if pos < train_count {
                train_pool.push(sample);
            } else if pos < train_count + val_count {
                independent_val.push(sample);
            } else {
                held_out_test.push(sample);
            }
        }

        Ok(Self {
            pins,
            total_tokens: tokens.len(),
            train_pool,
            independent_val,
            held_out_test,
        })
    }

    /// Partitions `train_pool` across `num_participants` according to `strategy`.
    #[must_use]
    pub fn partition_for_participants(
        &self,
        num_participants: usize,
        strategy: PartitionStrategy,
    ) -> Vec<Vec<TokenSample>> {
        assert!(num_participants > 0, "num_participants must be positive");
        let mut shards: Vec<Vec<TokenSample>> = (0..num_participants).map(|_| Vec::new()).collect();

        match strategy {
            PartitionStrategy::StratifiedIid => {
                for (i, sample) in self.train_pool.iter().enumerate() {
                    shards[i % num_participants].push(sample.clone());
                }
            }
            PartitionStrategy::ExtremeNonIid => {
                // Realistic severe Non-IID label skew: each participant receives ~80% of its
                // samples from 2 assigned dominant lexical classes and ~20% background samples,
                // creating substantial gradient divergence without degenerate zero-class support.
                let mut by_class: Vec<Vec<&TokenSample>> =
                    (0..NUM_CLASSES).map(|_| Vec::new()).collect();
                for s in &self.train_pool {
                    by_class[s.target_class].push(s);
                }
                let mut cursors = [0usize; NUM_CLASSES];
                let target_per_node = (self.train_pool.len() / num_participants).max(16);
                let dom_count = (target_per_node * 80) / 100;
                let bg_count = target_per_node.saturating_sub(dom_count);

                for (p, shard) in shards.iter_mut().enumerate() {
                    let c1 = p % NUM_CLASSES;
                    let c2 = (p + 1) % NUM_CLASSES;
                    for k in 0..dom_count {
                        let cls = if k % 2 == 0 { c1 } else { c2 };
                        let bucket = &by_class[cls];
                        if !bucket.is_empty() {
                            let idx = cursors[cls] % bucket.len();
                            cursors[cls] += 1;
                            shard.push(bucket[idx].clone());
                        }
                    }
                    for k in 0..bg_count {
                        let cls = (p + 2 + k) % NUM_CLASSES;
                        let bucket = &by_class[cls];
                        if !bucket.is_empty() {
                            let idx = cursors[cls] % bucket.len();
                            cursors[cls] += 1;
                            shard.push(bucket[idx].clone());
                        }
                    }
                }
            }
        }
        shards
    }
}

fn tokenize_into(text: &str, out: &mut Vec<String>) {
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else if matches!(
            ch,
            '{' | '}' | '[' | ']' | ':' | ',' | '.' | ';' | '?' | '!' | '(' | ')' | '"' | '<' | '>'
        ) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
}

fn classify_token(tok: &str) -> usize {
    if tok.len() == 1 {
        let b = tok.as_bytes()[0];
        if matches!(
            b,
            b'{' | b'}' | b'[' | b']' | b':' | b',' | b'"' | b'<' | b'>' | b'%'
        ) {
            return 0;
        }
        if matches!(b, b'.' | b';' | b'?' | b'!' | b'(' | b')' | b'-') {
            return 1;
        }
        if matches!(b, b'#' | b'*' | b'=' | b'|' | b'_') {
            return 11;
        }
    }

    if tok.contains("model.")
        || tok.contains("adapter_")
        || tok.contains("http")
        || tok.contains("{#")
        || tok.starts_with('#')
        || tok.starts_with('*')
        || tok.contains('_')
    {
        return 11;
    }

    if tok.bytes().any(|b| b.is_ascii_digit()) {
        return 2;
    }

    let lower = tok.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "the" | "a" | "an" | "this" | "that" | "these" | "those" | "both" | "each" | "all" | "some"
    ) {
        return 3;
    }
    if matches!(
        lower.as_str(),
        "in" | "of"
            | "to"
            | "for"
            | "with"
            | "on"
            | "at"
            | "by"
            | "from"
            | "as"
            | "into"
            | "between"
            | "through"
            | "after"
            | "before"
            | "under"
            | "over"
            | "during"
            | "than"
    ) {
        return 4;
    }
    if matches!(
        lower.as_str(),
        "and"
            | "or"
            | "but"
            | "however"
            | "therefore"
            | "moreover"
            | "whereas"
            | "while"
            | "although"
            | "because"
            | "which"
            | "also"
            | "not"
            | "if"
            | "else"
            | "endif"
    ) {
        return 5;
    }
    if matches!(
        lower.as_str(),
        "is" | "was"
            | "were"
            | "are"
            | "be"
            | "been"
            | "had"
            | "have"
            | "has"
            | "can"
            | "may"
            | "did"
            | "set"
            | "show"
            | "found"
            | "fed"
    ) || lower.ends_with("ed")
        || lower.ends_with("ing")
    {
        return 6;
    }

    if tok.bytes().next().is_some_and(|b| b.is_ascii_uppercase()) {
        return 10;
    }
    if tok.len() <= 4 {
        return 7;
    }
    if tok.len() <= 7 {
        return 8;
    }
    9
}

/// Evaluation metrics returned on any validation or test split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalMetrics {
    /// Mean multi-class cross-entropy loss (nats).
    pub cross_entropy_loss: f64,
    /// Perplexity (`exp(cross_entropy_loss)`).
    pub perplexity: f64,
    /// Top-1 next-token-class prediction accuracy in `[0.0, 1.0]`.
    pub top1_accuracy: f64,
    /// Fraction of predictions collapsing to a specific target poison class (`class 0`).
    pub poison_class_rate: f64,
}

/// Deterministic linear softmax language model (`W [NUM_CLASSES × FEATURE_DIM]`).
#[derive(Debug, Clone, PartialEq)]
pub struct FederatedModel {
    /// Flattened row-major parameter matrix of length `MODEL_DIM`.
    pub weights: Vec<f64>,
}

impl FederatedModel {
    /// Deterministically initializes weights with small zero-mean variance from `seed`.
    #[must_use]
    pub fn init(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed ^ 0x1350_0073_0001);
        let mut weights = Vec::with_capacity(MODEL_DIM);
        for _ in 0..MODEL_DIM {
            weights.push(rng.next_normal() * 0.02);
        }
        Self { weights }
    }

    /// Computes canonical SHA-256 hash of the current global model parameters.
    #[must_use]
    pub fn model_hash(&self) -> Digest32 {
        hash_f64_slice(&self.weights)
    }

    /// Evaluates cross-entropy loss, perplexity, top-1 accuracy, and class-0 rate on `samples`.
    #[must_use]
    pub fn evaluate(&self, samples: &[TokenSample]) -> EvalMetrics {
        Self::evaluate_weights(&self.weights, samples)
    }

    /// Evaluates `weights + scale * delta` without cloning the entire model permanently.
    #[must_use]
    pub fn evaluate_with_delta(
        &self,
        delta: &[f64],
        scale: f64,
        samples: &[TokenSample],
    ) -> EvalMetrics {
        let mut probe = self.weights.clone();
        for (w, &d) in probe.iter_mut().zip(delta.iter()) {
            *w += scale * d;
        }
        Self::evaluate_weights(&probe, samples)
    }

    /// Evaluates cross-entropy loss and accuracy for an arbitrary weight vector.
    #[must_use]
    pub fn evaluate_weights(weights: &[f64], samples: &[TokenSample]) -> EvalMetrics {
        if samples.is_empty() {
            return EvalMetrics {
                cross_entropy_loss: 0.0,
                perplexity: 1.0,
                top1_accuracy: 0.0,
                poison_class_rate: 0.0,
            };
        }
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let mut poison_hits = 0usize;

        for s in samples {
            let mut logits = [0.0f64; NUM_CLASSES];
            let mut max_logit = f64::NEG_INFINITY;
            for (c, logit) in logits.iter_mut().enumerate() {
                let row = &weights[c * FEATURE_DIM..(c + 1) * FEATURE_DIM];
                let mut dot = 0.0f64;
                for (w, &x) in row.iter().zip(s.features.iter()) {
                    dot += w * x;
                }
                *logit = dot;
                if dot > max_logit {
                    max_logit = dot;
                }
            }
            if !max_logit.is_finite() {
                return EvalMetrics {
                    cross_entropy_loss: 50.0,
                    perplexity: 50.0f64.exp(),
                    top1_accuracy: 0.0,
                    poison_class_rate: 1.0,
                };
            }
            let mut sum_exp = 0.0f64;
            let mut best_c = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for (c, &l) in logits.iter().enumerate() {
                if l > best_v {
                    best_v = l;
                    best_c = c;
                }
                sum_exp += (l - max_logit).exp();
            }
            let log_prob = (logits[s.target_class] - max_logit) - sum_exp.ln();
            let sample_loss = (-log_prob).min(50.0);
            total_loss += sample_loss;
            if best_c == s.target_class {
                correct += 1;
            }
            if best_c == 0 {
                poison_hits += 1;
            }
        }

        let n = samples.len() as f64;
        let mean_loss = (total_loss / n).min(50.0);
        EvalMetrics {
            cross_entropy_loss: mean_loss,
            perplexity: mean_loss.exp(),
            top1_accuracy: (correct as f64) / n,
            poison_class_rate: (poison_hits as f64) / n,
        }
    }

    /// Runs `local_steps` deterministic mini-batch/full-batch SGD epochs on `local_shard`
    /// and returns `(delta, loss_before, loss_after)`.
    #[must_use]
    pub fn local_train_delta(
        &self,
        local_shard: &[TokenSample],
        learning_rate: f64,
        l2_reg: f64,
        local_epochs: usize,
    ) -> (Vec<f64>, f64, f64) {
        if local_shard.is_empty() {
            return (vec![0.0; MODEL_DIM], 0.0, 0.0);
        }
        let loss_before = self.evaluate(local_shard).cross_entropy_loss;
        let mut local_w = self.weights.clone();
        const BATCH_SIZE: usize = 32;

        for _ in 0..local_epochs {
            for batch in local_shard.chunks(BATCH_SIZE) {
                let inv_b = 1.0 / (batch.len() as f64);
                let mut grad = [0.0f64; MODEL_DIM];
                for s in batch {
                    let mut logits = [0.0f64; NUM_CLASSES];
                    let mut max_logit = f64::NEG_INFINITY;
                    for (c, logit) in logits.iter_mut().enumerate() {
                        let row = &local_w[c * FEATURE_DIM..(c + 1) * FEATURE_DIM];
                        let mut dot = 0.0f64;
                        for (w, &x) in row.iter().zip(s.features.iter()) {
                            dot += w * x;
                        }
                        *logit = dot;
                        if dot > max_logit {
                            max_logit = dot;
                        }
                    }
                    let mut sum_exp = 0.0f64;
                    let mut probs = [0.0f64; NUM_CLASSES];
                    for c in 0..NUM_CLASSES {
                        let e = (logits[c] - max_logit).exp();
                        probs[c] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for (c, p) in probs.iter_mut().enumerate() {
                        let diff = *p * inv_sum - if c == s.target_class { 1.0 } else { 0.0 };
                        let g_row = &mut grad[c * FEATURE_DIM..(c + 1) * FEATURE_DIM];
                        for (g, &x) in g_row.iter_mut().zip(s.features.iter()) {
                            *g += diff * x;
                        }
                    }
                }
                for i in 0..MODEL_DIM {
                    let g = grad[i] * inv_b + l2_reg * local_w[i];
                    local_w[i] -= learning_rate * g;
                }
            }
        }

        let loss_after = Self::evaluate_weights(&local_w, local_shard).cross_entropy_loss;
        let mut delta = Vec::with_capacity(MODEL_DIM);
        for i in 0..MODEL_DIM {
            delta.push(local_w[i] - self.weights[i]);
        }
        (delta, loss_before, loss_after)
    }

    /// Factorizes a dense delta into a low-rank LoRA adapter `(A [rank*cols] || B [rows*rank])`
    /// via deterministic rank-`r` randomized/power-iteration SVD approximation.
    #[must_use]
    pub fn compress_to_lora_adapter(dense_delta: &[f64], rank: usize) -> Vec<f64> {
        let rows = NUM_CLASSES;
        let cols = FEATURE_DIM;
        let r = rank.min(rows).min(cols).max(1);
        let mut residual = dense_delta.to_vec();
        let mut a = vec![0.0f64; r * cols];
        let mut b = vec![0.0f64; rows * r];

        for k in 0..r {
            // Deterministic power iteration for top singular vector of residual matrix.
            let mut v = vec![0.0f64; cols];
            for (j, vj) in v.iter_mut().enumerate() {
                *vj = if (j + k) % 2 == 0 { 1.0 } else { -0.5 };
            }
            normalize_in_place(&mut v);
            let mut u = vec![0.0f64; rows];

            for _ in 0..8 {
                for (i, ui) in u.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for j in 0..cols {
                        s += residual[i * cols + j] * v[j];
                    }
                    *ui = s;
                }
                let sigma = l2_norm(&u);
                if sigma < 1e-12 {
                    break;
                }
                for ui in &mut u {
                    *ui /= sigma;
                }
                for (j, vj) in v.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for i in 0..rows {
                        s += residual[i * cols + j] * u[i];
                    }
                    *vj = s;
                }
                let vn = l2_norm(&v);
                if vn < 1e-12 {
                    break;
                }
                for vj in &mut v {
                    *vj /= vn;
                }
            }

            // Compute exact Rayleigh quotient sigma = u^T * R * v
            let mut sigma = 0.0;
            for i in 0..rows {
                for j in 0..cols {
                    sigma += u[i] * residual[i * cols + j] * v[j];
                }
            }
            let sqrt_s = sigma.abs().sqrt();
            let sign = if sigma >= 0.0 { 1.0 } else { -1.0 };
            for j in 0..cols {
                a[k * cols + j] = sqrt_s * v[j];
            }
            for i in 0..rows {
                b[i * r + k] = sign * sqrt_s * u[i];
            }
            for i in 0..rows {
                for j in 0..cols {
                    residual[i * cols + j] -= b[i * r + k] * a[k * cols + j];
                }
            }
        }

        let mut payload = Vec::with_capacity(a.len() + b.len());
        payload.extend_from_slice(&a);
        payload.extend_from_slice(&b);
        payload
    }

    /// Applies an aggregated delta vector directly to the global model weights.
    pub fn apply_delta(&mut self, delta: &[f64]) {
        for (w, &d) in self.weights.iter_mut().zip(delta.iter()) {
            if d.is_finite() {
                *w += d;
            }
        }
    }
}

/// Computes the Euclidean L2 norm of a vector.
#[must_use]
pub fn l2_norm(v: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for &x in v {
        sum += x * x;
    }
    sum.sqrt()
}

/// Computes cosine similarity between two equal-length vectors (`0.0` if either norm is zero).
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 1e-15 || !denom.is_finite() {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Computes squared Euclidean distance between two equal-length vectors.
#[must_use]
pub fn l2_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

fn normalize_in_place(v: &mut [f64]) {
    let n = l2_norm(v);
    if n > 1e-15 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}
