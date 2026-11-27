//! Pure-Rust executable small Zen causal language model with frozen base
//! weights and trainable low-rank (LoRA) adapters across attention, MLP, and
//! output-projection layers.
//!
//! Forward passes and analytical backpropagation are implemented explicitly in
//! safe Rust without external tensor dependencies, and verified against central
//! finite-difference gradients in unit tests.

use crate::corpus::CORPUS_VOCAB_SIZE;
use crate::lora::LoraLinear;
use crate::sha256::{sha256, sha256_hex, SplitMix64};

/// RMSNorm epsilon constant for numerical stability.
const RMS_EPS: f32 = 1e-5;

/// Hyperparameters for [`SmallZenLm`].
#[derive(Clone, Debug, PartialEq)]
pub struct SmallZenConfig {
    /// Vocabulary size (number of distinct token IDs).
    pub vocab_size: usize,
    /// Model hidden width `d_model`.
    pub d_model: usize,
    /// Feed-forward intermediate width `d_ff`.
    pub d_ff: usize,
    /// Maximum causal sequence length `seq_len`.
    pub seq_len: usize,
    /// LoRA bottleneck rank `r`.
    pub lora_rank: usize,
    /// LoRA scaling numerator `alpha` (effective scale `alpha / r`).
    pub lora_alpha: f32,
    /// Deterministic initialization seed for base and LoRA weights.
    pub init_seed: u64,
}

impl Default for SmallZenConfig {
    fn default() -> Self {
        Self {
            vocab_size: CORPUS_VOCAB_SIZE,
            d_model: 48,
            d_ff: 96,
            seq_len: 16,
            lora_rank: 8,
            lora_alpha: 16.0,
            init_seed: 1350,
        }
    }
}

impl SmallZenConfig {
    /// Returns a canonical SHA-256 fingerprint of this model architecture.
    #[must_use]
    pub fn config_sha256(&self) -> String {
        let canonical = format!(
            "zen-small-v1:v={}:d={}:ff={}:seq={}:r={}:a={:.6}:seed={}",
            self.vocab_size,
            self.d_model,
            self.d_ff,
            self.seq_len,
            self.lora_rank,
            self.lora_alpha,
            self.init_seed
        );
        sha256_hex(canonical.as_bytes())
    }

    /// Computes explicit memory residency vs per-token streamed lookup accounting.
    ///
    /// Separates the full resident input-embedding table
    /// (`vocab_size * d_model * 2` bytes in BF16) from a single-token row lookup
    /// (`d_model * 2` bytes) so bandwidth projections never conflate table
    /// residency with per-token memory traffic.
    #[must_use]
    pub fn residency_accounting_bf16(&self) -> ModelResidencyAccounting {
        let bpw = 2u64;
        let resident_input_embedding_bytes = (self.vocab_size as u64) * (self.d_model as u64) * bpw;
        let single_row_embedding_lookup_bytes = (self.d_model as u64) * bpw;
        let pos_embed_bytes = (self.seq_len as u64) * (self.d_model as u64) * bpw;
        let attention_base_bytes = 4 * (self.d_model as u64) * (self.d_model as u64) * bpw;
        let mlp_base_bytes = 2 * (self.d_model as u64) * (self.d_ff as u64) * bpw;
        let lm_head_base_bytes = (self.vocab_size as u64) * (self.d_model as u64) * bpw;
        let total_base_resident_bytes = resident_input_embedding_bytes
            + pos_embed_bytes
            + attention_base_bytes
            + mlp_base_bytes
            + lm_head_base_bytes;
        let streamed_per_token_base_bytes = single_row_embedding_lookup_bytes
            + (self.d_model as u64) * bpw
            + attention_base_bytes
            + mlp_base_bytes
            + lm_head_base_bytes;
        ModelResidencyAccounting {
            resident_input_embedding_bytes,
            single_row_embedding_lookup_bytes,
            total_base_resident_bytes,
            streamed_per_token_base_bytes,
        }
    }
}

/// Separation of full resident weight tables from per-token streamed lookups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelResidencyAccounting {
    /// Full resident input-embedding table size (`vocab_size * d_model * 2` bytes).
    pub resident_input_embedding_bytes: u64,
    /// Single-token input-embedding lookup size (`d_model * 2` bytes).
    pub single_row_embedding_lookup_bytes: u64,
    /// Total resident base weights in BF16 bytes.
    pub total_base_resident_bytes: u64,
    /// Bytes touched by a single-token decode forward pass (single row lookup
    /// rather than scanning the full embedding table).
    pub streamed_per_token_base_bytes: u64,
}

/// Held-out evaluation summary for [`SmallZenLm`].
#[derive(Clone, Debug)]
pub struct EvalMetrics {
    /// Mean token-level cross-entropy loss (nats).
    pub loss: f64,
    /// Perplexity `exp(loss)`.
    pub perplexity: f64,
    /// Top-1 next-token prediction accuracy in `[0.0, 1.0]`.
    pub top1_accuracy: f64,
    /// Number of target token positions evaluated.
    pub tokens_evaluated: usize,
}

/// Pure-Rust single-block causal Transformer with frozen base projections and
/// trainable LoRA adapters on Q, V, O, MLP-Up, MLP-Down, and LM-Head.
#[derive(Clone, Debug)]
pub struct SmallZenLm {
    /// Architectural configuration.
    pub config: SmallZenConfig,
    /// Frozen token embedding table `[vocab_size, d_model]`.
    token_embed: Vec<f32>,
    /// Frozen positional embedding table `[seq_len, d_model]`.
    pos_embed: Vec<f32>,
    /// Frozen base projections.
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    w_lm: Vec<f32>,
    /// Trainable LoRA adapters.
    pub lora_q: LoraLinear,
    /// Trainable LoRA value adapter.
    pub lora_v: LoraLinear,
    /// Trainable LoRA attention-output adapter.
    pub lora_o: LoraLinear,
    /// Trainable LoRA MLP up-projection adapter.
    pub lora_up: LoraLinear,
    /// Trainable LoRA MLP down-projection adapter.
    pub lora_down: LoraLinear,
    /// Trainable LoRA LM-head projection adapter.
    pub lora_lm: LoraLinear,
}

impl SmallZenLm {
    /// Constructs a deterministically initialized small Zen model with frozen
    /// base weights and zero-delta LoRA adapters (`B = 0`).
    #[must_use]
    pub fn new(config: SmallZenConfig) -> Self {
        let mut rng = SplitMix64::new(config.init_seed);
        let d = config.d_model;
        let ff = config.d_ff;
        let v = config.vocab_size;
        let t = config.seq_len;

        let init_mat = |rows: usize, cols: usize, scale: f32, r: &mut SplitMix64| -> Vec<f32> {
            let mut m = Vec::with_capacity(rows * cols);
            for _ in 0..(rows * cols) {
                m.push(r.next_gaussian_f32() * scale);
            }
            m
        };

        let emb_std = 0.25f32;
        let d_std = (1.0 / (d as f32)).sqrt() * 0.5;
        let ff_std = (1.0 / (ff as f32)).sqrt() * 0.5;

        let token_embed = init_mat(v, d, emb_std, &mut rng);
        let pos_embed = init_mat(t, d, 0.08, &mut rng);
        let w_q = init_mat(d, d, d_std, &mut rng);
        let w_k = init_mat(d, d, d_std, &mut rng);
        let w_v = init_mat(d, d, d_std, &mut rng);
        let w_o = init_mat(d, d, d_std, &mut rng);
        let w_up = init_mat(ff, d, d_std, &mut rng);
        let w_down = init_mat(d, ff, ff_std, &mut rng);
        let w_lm = init_mat(v, d, d_std, &mut rng);

        let r = config.lora_rank;
        let a = config.lora_alpha;
        let lora_q = LoraLinear::new(d, d, r, a, &mut rng);
        let lora_v = LoraLinear::new(d, d, r, a, &mut rng);
        let lora_o = LoraLinear::new(d, d, r, a, &mut rng);
        let lora_up = LoraLinear::new(d, ff, r, a, &mut rng);
        let lora_down = LoraLinear::new(ff, d, r, a, &mut rng);
        let lora_lm = LoraLinear::new(d, v, r, a, &mut rng);

        Self {
            config,
            token_embed,
            pos_embed,
            w_q,
            w_k,
            w_v,
            w_o,
            w_up,
            w_down,
            w_lm,
            lora_q,
            lora_v,
            lora_o,
            lora_up,
            lora_down,
            lora_lm,
        }
    }

    /// Total number of frozen base scalar parameters.
    #[must_use]
    pub fn base_param_count(&self) -> usize {
        self.token_embed.len()
            + self.pos_embed.len()
            + self.w_q.len()
            + self.w_k.len()
            + self.w_v.len()
            + self.w_o.len()
            + self.w_up.len()
            + self.w_down.len()
            + self.w_lm.len()
    }

    /// Total number of trainable LoRA scalar parameters across all 6 adapters.
    #[must_use]
    pub fn lora_param_count(&self) -> usize {
        self.lora_q.param_count()
            + self.lora_v.param_count()
            + self.lora_o.param_count()
            + self.lora_up.param_count()
            + self.lora_down.param_count()
            + self.lora_lm.param_count()
    }

    /// Exports all trainable LoRA weights into a flat canonical `Vec<f32>`.
    #[must_use]
    pub fn export_lora_params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.lora_param_count());
        self.lora_q.write_params(&mut out);
        self.lora_v.write_params(&mut out);
        self.lora_o.write_params(&mut out);
        self.lora_up.write_params(&mut out);
        self.lora_down.write_params(&mut out);
        self.lora_lm.write_params(&mut out);
        out
    }

    /// Loads a flat canonical LoRA parameter slice into all 6 adapters.
    pub fn load_lora_params(&mut self, params: &[f32]) -> Result<(), String> {
        if params.len() != self.lora_param_count() {
            return Err(format!(
                "expected {} LoRA params, got {}",
                self.lora_param_count(),
                params.len()
            ));
        }
        let mut off = 0usize;
        off = self.lora_q.read_params(params, off)?;
        off = self.lora_v.read_params(params, off)?;
        off = self.lora_o.read_params(params, off)?;
        off = self.lora_up.read_params(params, off)?;
        off = self.lora_down.read_params(params, off)?;
        let _ = self.lora_lm.read_params(params, off)?;
        Ok(())
    }

    /// Computes a deterministic SHA-256 digest of the current LoRA parameters.
    #[must_use]
    pub fn lora_params_sha256(&self) -> [u8; 32] {
        let params = self.export_lora_params();
        let mut bytes = Vec::with_capacity(params.len() * 4);
        for &p in &params {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        sha256(&bytes)
    }

    /// Evaluates held-out cross-entropy loss, perplexity, and top-1 next-token
    /// accuracy without mutating weights.
    #[must_use]
    pub fn eval_metrics(&self, sequences: &[&[u8]]) -> EvalMetrics {
        let mut total_loss = 0.0f64;
        let mut total_correct = 0usize;
        let mut total_tokens = 0usize;

        for seq in sequences {
            let t_len = seq.len().saturating_sub(1).min(self.config.seq_len);
            if t_len == 0 {
                continue;
            }
            let cache = self.forward_single(seq, t_len);
            for t in 0..t_len {
                let target = (seq[t + 1] as usize).min(self.config.vocab_size - 1);
                let probs = &cache.probs[t];
                let p = probs[target].max(1e-12) as f64;
                total_loss -= p.ln();
                let mut best_idx = 0usize;
                let mut best_p = probs[0];
                for (v_idx, &pv) in probs.iter().enumerate().skip(1) {
                    if pv > best_p {
                        best_p = pv;
                        best_idx = v_idx;
                    }
                }
                if best_idx == target {
                    total_correct += 1;
                }
                total_tokens += 1;
            }
        }

        let loss = if total_tokens > 0 {
            total_loss / (total_tokens as f64)
        } else {
            0.0
        };
        let perplexity = loss.min(30.0).exp();
        let top1_accuracy = if total_tokens > 0 {
            (total_correct as f64) / (total_tokens as f64)
        } else {
            0.0
        };
        EvalMetrics {
            loss,
            perplexity,
            top1_accuracy,
            tokens_evaluated: total_tokens,
        }
    }

    /// Runs an exact forward + analytical backward pass over a mini-batch of
    /// causal sequences, returning `(mean_cross_entropy_loss, flat_lora_grads, tokens)`.
    #[must_use]
    pub fn forward_and_backward_batch(&self, sequences: &[&[u8]]) -> (f64, Vec<f32>, usize) {
        let total_params = self.lora_param_count();
        let mut grads = vec![0.0f32; total_params];

        let mut total_tokens = 0usize;
        for seq in sequences {
            total_tokens += seq.len().saturating_sub(1).min(self.config.seq_len);
        }
        if total_tokens == 0 {
            return (0.0, grads, 0);
        }

        let inv_tokens = 1.0 / (total_tokens as f32);
        let mut total_loss = 0.0f64;

        let off_q = 0usize;
        let off_v = off_q + self.lora_q.param_count();
        let off_o = off_v + self.lora_v.param_count();
        let off_up = off_o + self.lora_o.param_count();
        let off_down = off_up + self.lora_up.param_count();
        let off_lm = off_down + self.lora_down.param_count();

        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        for seq in sequences {
            let t_len = seq.len().saturating_sub(1).min(self.config.seq_len);
            if t_len == 0 {
                continue;
            }
            let cache = self.forward_single(seq, t_len);

            let mut grad_h1 = vec![vec![0.0f32; d]; t_len];

            // Backward through LM head, RMSNorm 3, and MLP per timestep
            for t in 0..t_len {
                let target = (seq[t + 1] as usize).min(v - 1);
                let probs = &cache.probs[t];
                total_loss -= (probs[target].max(1e-12) as f64).ln();

                // dL/dlogits[t] = (probs - one_hot(target)) / total_tokens
                let mut grad_logits = vec![0.0f32; v];
                for i in 0..v {
                    grad_logits[i] = probs[i] * inv_tokens;
                }
                grad_logits[target] -= inv_tokens;

                // Backprop through W_lm + lora_lm
                let mut grad_n3 = vec![0.0f32; d];
                matvec_t_add(&self.w_lm, v, d, &grad_logits, &mut grad_n3);
                {
                    let a_len = self.lora_lm.a.len();
                    let (ga, gb) =
                        grads[off_lm..off_lm + self.lora_lm.param_count()].split_at_mut(a_len);
                    self.lora_lm.backward_accumulate(
                        &cache.n3[t],
                        &cache.h_lm[t],
                        &grad_logits,
                        ga,
                        gb,
                        &mut grad_n3,
                    );
                }

                // Backprop through RMSNorm 3 -> grad_h2
                let grad_h2 = rms_norm_backward(&cache.n3[t], cache.rms3[t], &grad_n3);

                // Residual split: h2 = h1 + mlp_out
                // Backprop through W_down + lora_down
                let mut grad_act = vec![0.0f32; ff];
                matvec_t_add(&self.w_down, d, ff, &grad_h2, &mut grad_act);
                {
                    let a_len = self.lora_down.a.len();
                    let (ga, gb) = grads[off_down..off_down + self.lora_down.param_count()]
                        .split_at_mut(a_len);
                    self.lora_down.backward_accumulate(
                        &cache.mlp_act[t],
                        &cache.h_down[t],
                        &grad_h2,
                        ga,
                        gb,
                        &mut grad_act,
                    );
                }

                // Backprop through SiLU
                let mut grad_pre = vec![0.0f32; ff];
                for i in 0..ff {
                    let z = cache.mlp_pre[t][i];
                    let sig = sigmoid(z);
                    let dsilu = sig * (1.0 + z * (1.0 - sig));
                    grad_pre[i] = grad_act[i] * dsilu;
                }

                // Backprop through W_up + lora_up
                let mut grad_n2 = vec![0.0f32; d];
                matvec_t_add(&self.w_up, ff, d, &grad_pre, &mut grad_n2);
                {
                    let a_len = self.lora_up.a.len();
                    let (ga, gb) =
                        grads[off_up..off_up + self.lora_up.param_count()].split_at_mut(a_len);
                    self.lora_up.backward_accumulate(
                        &cache.n2[t],
                        &cache.h_up[t],
                        &grad_pre,
                        ga,
                        gb,
                        &mut grad_n2,
                    );
                }

                // Backprop through RMSNorm 2 + residual from h2
                let grad_h1_from_mlp = rms_norm_backward(&cache.n2[t], cache.rms2[t], &grad_n2);
                for i in 0..d {
                    grad_h1[t][i] = grad_h2[i] + grad_h1_from_mlp[i];
                }
            }

            // Backprop through attention output projection W_o + lora_o
            let mut grad_ctx = vec![vec![0.0f32; d]; t_len];
            for t in 0..t_len {
                matvec_t_add(&self.w_o, d, d, &grad_h1[t], &mut grad_ctx[t]);
                let a_len = self.lora_o.a.len();
                let (ga, gb) = grads[off_o..off_o + self.lora_o.param_count()].split_at_mut(a_len);
                self.lora_o.backward_accumulate(
                    &cache.ctx[t],
                    &cache.h_o[t],
                    &grad_h1[t],
                    ga,
                    gb,
                    &mut grad_ctx[t],
                );
            }

            // Backprop through causal scaled dot-product attention
            let mut grad_q = vec![vec![0.0f32; d]; t_len];
            let mut grad_k = vec![vec![0.0f32; d]; t_len];
            let mut grad_v = vec![vec![0.0f32; d]; t_len];

            for t in 0..t_len {
                let p_row = &cache.attn_weights[t];
                let mut dp = vec![0.0f32; t + 1];
                let mut dot_dp_p = 0.0f32;
                for j in 0..=t {
                    let mut g = 0.0f32;
                    for i in 0..d {
                        g += grad_ctx[t][i] * cache.v[j][i];
                        grad_v[j][i] += p_row[j] * grad_ctx[t][i];
                    }
                    dp[j] = g;
                    dot_dp_p += g * p_row[j];
                }
                for j in 0..=t {
                    let ds = p_row[j] * (dp[j] - dot_dp_p) * inv_sqrt_d;
                    for i in 0..d {
                        grad_q[t][i] += ds * cache.k[j][i];
                        grad_k[j][i] += ds * cache.q[t][i];
                    }
                }
            }

            // Backprop through Q and V LoRA adapters (K and embeddings are frozen)
            let mut dummy_grad_n1 = vec![0.0f32; d];
            for t in 0..t_len {
                dummy_grad_n1.fill(0.0);
                {
                    let a_len = self.lora_q.a.len();
                    let (ga, gb) =
                        grads[off_q..off_q + self.lora_q.param_count()].split_at_mut(a_len);
                    self.lora_q.backward_accumulate(
                        &cache.n1[t],
                        &cache.h_q[t],
                        &grad_q[t],
                        ga,
                        gb,
                        &mut dummy_grad_n1,
                    );
                }
                dummy_grad_n1.fill(0.0);
                {
                    let a_len = self.lora_v.a.len();
                    let (ga, gb) =
                        grads[off_v..off_v + self.lora_v.param_count()].split_at_mut(a_len);
                    self.lora_v.backward_accumulate(
                        &cache.n1[t],
                        &cache.h_v[t],
                        &grad_v[t],
                        ga,
                        gb,
                        &mut dummy_grad_n1,
                    );
                }
                // Unused grad_k is consumed by frozen W_k -> frozen embeddings.
                let _ = &grad_k[t];
            }
        }

        (total_loss / (total_tokens as f64), grads, total_tokens)
    }

    fn forward_single(&self, seq: &[u8], t_len: usize) -> ForwardCache {
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        let mut x0 = Vec::with_capacity(t_len);
        let mut n1 = Vec::with_capacity(t_len);
        let mut q = Vec::with_capacity(t_len);
        let mut k = Vec::with_capacity(t_len);
        let mut v_proj = Vec::with_capacity(t_len);
        let mut h_q = Vec::with_capacity(t_len);
        let mut h_v = Vec::with_capacity(t_len);

        for t in 0..t_len {
            let tok = (seq[t] as usize).min(v - 1);
            let mut x = vec![0.0f32; d];
            let emb_row = &self.token_embed[tok * d..(tok + 1) * d];
            let pos_row = &self.pos_embed[t * d..(t + 1) * d];
            for i in 0..d {
                x[i] = emb_row[i] + pos_row[i];
            }
            let (normed, _rms) = rms_norm(&x);

            let mut qt = matvec(&self.w_q, d, d, &normed);
            let hq = self.lora_q.forward_add(&normed, &mut qt);
            let kt = matvec(&self.w_k, d, d, &normed);
            let mut vt = matvec(&self.w_v, d, d, &normed);
            let hv = self.lora_v.forward_add(&normed, &mut vt);

            x0.push(x);
            n1.push(normed);
            q.push(qt);
            k.push(kt);
            v_proj.push(vt);
            h_q.push(hq);
            h_v.push(hv);
        }

        // Causal self-attention
        let mut attn_weights = Vec::with_capacity(t_len);
        let mut ctx = Vec::with_capacity(t_len);
        let mut h_o = Vec::with_capacity(t_len);
        let mut n2 = Vec::with_capacity(t_len);
        let mut rms2 = Vec::with_capacity(t_len);
        let mut h_up = Vec::with_capacity(t_len);
        let mut mlp_pre = Vec::with_capacity(t_len);
        let mut mlp_act = Vec::with_capacity(t_len);
        let mut h_down = Vec::with_capacity(t_len);
        let mut n3 = Vec::with_capacity(t_len);
        let mut rms3 = Vec::with_capacity(t_len);
        let mut h_lm = Vec::with_capacity(t_len);
        let mut probs = Vec::with_capacity(t_len);

        for t in 0..t_len {
            let mut scores = vec![0.0f32; t + 1];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..=t {
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += q[t][i] * k[j][i];
                }
                let s = dot * inv_sqrt_d;
                scores[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            for s in &mut scores {
                *s /= sum_exp;
            }

            let mut c = vec![0.0f32; d];
            for (j, &w) in scores.iter().enumerate() {
                for i in 0..d {
                    c[i] += w * v_proj[j][i];
                }
            }

            let mut attn_out = matvec(&self.w_o, d, d, &c);
            let ho = self.lora_o.forward_add(&c, &mut attn_out);

            let mut h1_t = vec![0.0f32; d];
            for i in 0..d {
                h1_t[i] = x0[t][i] + attn_out[i];
            }

            let (n2_t, r2_t) = rms_norm(&h1_t);
            let mut pre = matvec(&self.w_up, ff, d, &n2_t);
            let hup = self.lora_up.forward_add(&n2_t, &mut pre);
            let act: Vec<f32> = pre.iter().map(|&z| z * sigmoid(z)).collect();

            let mut mlp_out = matvec(&self.w_down, d, ff, &act);
            let hdn = self.lora_down.forward_add(&act, &mut mlp_out);

            let mut h2_t = vec![0.0f32; d];
            for i in 0..d {
                h2_t[i] = h1_t[i] + mlp_out[i];
            }

            let (n3_t, r3_t) = rms_norm(&h2_t);
            let mut logits = matvec(&self.w_lm, v, d, &n3_t);
            let hlm = self.lora_lm.forward_add(&n3_t, &mut logits);
            let p = softmax(&logits);

            attn_weights.push(scores);
            ctx.push(c);
            h_o.push(ho);
            n2.push(n2_t);
            rms2.push(r2_t);
            h_up.push(hup);
            mlp_pre.push(pre);
            mlp_act.push(act);
            h_down.push(hdn);
            n3.push(n3_t);
            rms3.push(r3_t);
            h_lm.push(hlm);
            probs.push(p);
        }

        ForwardCache {
            n1,
            q,
            k,
            v: v_proj,
            h_q,
            h_v,
            attn_weights,
            ctx,
            h_o,
            n2,
            rms2,
            h_up,
            mlp_pre,
            mlp_act,
            h_down,
            n3,
            rms3,
            h_lm,
            probs,
        }
    }
}

struct ForwardCache {
    n1: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    h_q: Vec<Vec<f32>>,
    h_v: Vec<Vec<f32>>,
    attn_weights: Vec<Vec<f32>>,
    ctx: Vec<Vec<f32>>,
    h_o: Vec<Vec<f32>>,
    n2: Vec<Vec<f32>>,
    rms2: Vec<f32>,
    h_up: Vec<Vec<f32>>,
    mlp_pre: Vec<Vec<f32>>,
    mlp_act: Vec<Vec<f32>>,
    h_down: Vec<Vec<f32>>,
    n3: Vec<Vec<f32>>,
    rms3: Vec<f32>,
    h_lm: Vec<Vec<f32>>,
    probs: Vec<Vec<f32>>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rms_norm(x: &[f32]) -> (Vec<f32>, f32) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let rms = (mean_sq + RMS_EPS).sqrt();
    let inv = 1.0 / rms;
    let out = x.iter().map(|&v| v * inv).collect();
    (out, rms)
}

fn rms_norm_backward(y: &[f32], rms: f32, grad_y: &[f32]) -> Vec<f32> {
    let d = y.len().max(1) as f32;
    let mean_gy: f32 = grad_y
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| g * yi)
        .sum::<f32>()
        / d;
    let inv = 1.0 / rms;
    grad_y
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi * mean_gy) * inv)
        .collect()
}

fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for (r, out_val) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for (&wij, &xj) in row.iter().zip(x.iter()) {
            acc += wij * xj;
        }
        *out_val = acc;
    }
    out
}

fn matvec_t_add(w: &[f32], rows: usize, cols: usize, y: &[f32], out_x: &mut [f32]) {
    for (r, &yr) in y.iter().enumerate().take(rows) {
        let row = &w[r * cols..(r + 1) * cols];
        for (c, &wrc) in row.iter().enumerate() {
            out_x[c] += wrc * yr;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max_v = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_v {
            max_v = v;
        }
    }
    let mut out = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(logits.iter()) {
        let e = (v - max_v).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-30);
    for o in &mut out {
        *o *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_lora_gradients_match_central_finite_differences() {
        let cfg = SmallZenConfig {
            vocab_size: 16,
            d_model: 8,
            d_ff: 12,
            seq_len: 5,
            lora_rank: 2,
            lora_alpha: 4.0,
            init_seed: 99,
        };
        let mut model = SmallZenLm::new(cfg);
        let mut rng = SplitMix64::new(12345);
        // Perturb B slightly away from zero so upstream paths through all 6 LoRA
        // adapters carry non-zero gradient signal simultaneously.
        let mut params = model.export_lora_params();
        for p in &mut params {
            *p += rng.next_gaussian_f32() * 0.15;
        }
        model
            .load_lora_params(&params)
            .expect("load perturbed params");

        let seq1: [u8; 6] = [1, 4, 2, 7, 3, 5];
        let seq2: [u8; 6] = [0, 9, 11, 2, 6, 8];
        let batch: [&[u8]; 2] = [&seq1, &seq2];

        let (_loss, ana_grads, tokens) = model.forward_and_backward_batch(&batch);
        assert_eq!(tokens, 10);

        let eps = 1e-3f32;
        let mut max_err = 0.0f32;
        // Sample indices evenly across all 6 adapters (A and B matrices).
        for idx in (0..params.len()).step_by(5) {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[idx] += eps;
            p_minus[idx] -= eps;

            model.load_lora_params(&p_plus).unwrap();
            let l_plus = model.eval_metrics(&batch).loss as f32;
            model.load_lora_params(&p_minus).unwrap();
            let l_minus = model.eval_metrics(&batch).loss as f32;

            let num_grad = (l_plus - l_minus) / (2.0 * eps);
            let err = (num_grad - ana_grads[idx]).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 2e-3,
                "gradient mismatch at idx {idx}: numerical={num_grad}, analytical={}, err={err}",
                ana_grads[idx]
            );
        }
        assert!(max_err < 2e-3, "max gradient error {max_err}");
    }

    #[test]
    fn residency_accounting_separates_full_table_from_single_row() {
        let cfg = SmallZenConfig::default();
        let acc = cfg.residency_accounting_bf16();
        assert_eq!(
            acc.resident_input_embedding_bytes,
            (cfg.vocab_size * cfg.d_model * 2) as u64
        );
        assert_eq!(
            acc.single_row_embedding_lookup_bytes,
            (cfg.d_model * 2) as u64
        );
        assert!(acc.resident_input_embedding_bytes > acc.single_row_embedding_lookup_bytes);
    }
}
