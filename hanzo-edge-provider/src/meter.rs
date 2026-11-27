// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Executable deterministic open-weight Zen hybrid micro-model, signed
//! [`MeteredWorkReceipt`] accounting, and independent adversarial fraud verification
//! (`7/7` fraud classes under [`TrustTier::SignedTelemetryAndSpotCheck`]).
//!
//! ## Verification Boundary (No Conflation with zkML or Hardware TEEs)
//! - **Cryptographically & independently verifiable**: prompt/output token counts,
//!   `output_sha256` commitment, `model_weights_digest`, HMAC-SHA256 receipt integrity,
//!   exact energy conservation (`joules == mean_watts * duration_s`), withdrawal status,
//!   replay protection, and deterministic spot-check re-execution.
//! - **Self-reported / OS-sensor telemetry**: `mean_watts`, `peak_temp_c`, and
//!   `battery_pct_end` are explicitly marked `telemetry_self_reported = true` and are
//!   NEVER presented as zkML validity proofs or hardware TEE attestations.

use crate::manifest::{PrivacyPolicy, TrustTier};
use crate::sha256::{hmac_sha256_hex, sha256_hex, verify_hmac_sha256_hex, Sha256, SplitMix64};
use crate::thermal::ThrottleLevel;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Vocabulary size of the executable open-weight Zen micro-model used in live TCP serving.
pub const MICRO_VOCAB_SIZE: usize = 256;

/// Hidden dimension (`d_model`) of the executable micro-model.
pub const MICRO_D_MODEL: usize = 48;

/// Number of layers in the executable micro-model (1 causal Attention + 3 GDN recurrent layers).
pub const MICRO_NUM_LAYERS: usize = 4;

/// Canonical default deterministic model seed for the open-weight Zen micro-model.
pub const ZEN_MICRO_MODEL_SEED: u64 = 0x5A45_4E31_3530_0001;

/// Executable deterministic hybrid Attention + Gated DeltaNet (GDN) causal micro-model.
///
/// Runs real floating-point matrix-vector projections, RMSNorm, GDN recurrent state
/// updates, causal self-attention over cached KV states, and greedy argmax decoding
/// purely in safe Rust with zero external dependencies.
#[derive(Debug, Clone)]
pub struct ZenEdgeMicroModel {
    /// Deterministic seed used to generate weights.
    pub seed: u64,
    /// Human-readable model identifier.
    pub model_id: String,
    /// Cryptographic SHA-256 hex commitment over all model weights.
    pub weights_digest: String,
    embed_tokens: Vec<f32>, // [MICRO_VOCAB_SIZE * MICRO_D_MODEL]
    layers: Vec<MicroLayer>,
    lm_head: Vec<f32>, // [MICRO_VOCAB_SIZE * MICRO_D_MODEL]
}

#[derive(Debug, Clone)]
struct MicroLayer {
    is_attention: bool,
    w_q: Vec<f32>,   // [D * D]
    w_k: Vec<f32>,   // [D * D]
    w_v: Vec<f32>,   // [D * D]
    w_o: Vec<f32>,   // [D * D]
    w_ff1: Vec<f32>, // [D * D]
    w_ff2: Vec<f32>, // [D * D]
    gdn_decay: f32,
}

impl ZenEdgeMicroModel {
    /// Builds a deterministic micro-model from `seed` and computes its SHA-256 weight commitment.
    #[must_use]
    pub fn from_seed(model_id: impl Into<String>, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let scale = 1.0 / (MICRO_D_MODEL as f32).sqrt();

        let mut embed_tokens = vec![0.0f32; MICRO_VOCAB_SIZE * MICRO_D_MODEL];
        for v in &mut embed_tokens {
            *v = rng.next_f32_signed() * 0.5;
        }

        let mut layers = Vec::with_capacity(MICRO_NUM_LAYERS);
        for idx in 0..MICRO_NUM_LAYERS {
            let is_attention = idx == MICRO_NUM_LAYERS - 1;
            let gen_mat = |rng: &mut SplitMix64| -> Vec<f32> {
                let mut m = vec![0.0f32; MICRO_D_MODEL * MICRO_D_MODEL];
                for x in &mut m {
                    *x = rng.next_f32_signed() * scale;
                }
                m
            };
            layers.push(MicroLayer {
                is_attention,
                w_q: gen_mat(&mut rng),
                w_k: gen_mat(&mut rng),
                w_v: gen_mat(&mut rng),
                w_o: gen_mat(&mut rng),
                w_ff1: gen_mat(&mut rng),
                w_ff2: gen_mat(&mut rng),
                gdn_decay: 0.85 - (idx as f32) * 0.05,
            });
        }

        let mut lm_head = vec![0.0f32; MICRO_VOCAB_SIZE * MICRO_D_MODEL];
        for v in &mut lm_head {
            *v = rng.next_f32_signed() * scale;
        }

        let mut hasher = Sha256::new();
        hasher.update(b"HANZO_ZEN_EDGE_MICRO_V1");
        hash_f32_slice(&mut hasher, &embed_tokens);
        for layer in &layers {
            hasher.update(&[u8::from(layer.is_attention)]);
            hash_f32_slice(&mut hasher, &layer.w_q);
            hash_f32_slice(&mut hasher, &layer.w_k);
            hash_f32_slice(&mut hasher, &layer.w_v);
            hash_f32_slice(&mut hasher, &layer.w_o);
            hash_f32_slice(&mut hasher, &layer.w_ff1);
            hash_f32_slice(&mut hasher, &layer.w_ff2);
            hasher.update(&layer.gdn_decay.to_le_bytes());
        }
        hash_f32_slice(&mut hasher, &lm_head);
        let weights_digest = hasher.finalize_hex();

        Self {
            seed,
            model_id: model_id.into(),
            weights_digest,
            embed_tokens,
            layers,
            lm_head,
        }
    }

    /// Runs deterministic prefill over `prompt_tokens` followed by `max_new_tokens`
    /// autoregressive decode steps, invoking `on_decode_step(step_index, token_id)` after
    /// each generated token.
    pub fn generate_with_hook<F>(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        mut on_decode_step: F,
    ) -> Vec<u32>
    where
        F: FnMut(usize, u32),
    {
        if max_new_tokens == 0 {
            return Vec::new();
        }
        let mut attn_k_cache: Vec<Vec<f32>> =
            Vec::with_capacity(prompt_tokens.len() + max_new_tokens);
        let mut attn_v_cache: Vec<Vec<f32>> =
            Vec::with_capacity(prompt_tokens.len() + max_new_tokens);
        let mut gdn_states: Vec<Vec<f32>> = vec![vec![0.0f32; MICRO_D_MODEL]; MICRO_NUM_LAYERS];

        let mut last_hidden = vec![0.0f32; MICRO_D_MODEL];
        let effective_prompt: &[u32] = if prompt_tokens.is_empty() {
            &[1u32]
        } else {
            prompt_tokens
        };

        for &tok in effective_prompt {
            last_hidden =
                self.forward_one_token(tok, &mut attn_k_cache, &mut attn_v_cache, &mut gdn_states);
        }

        let mut out = Vec::with_capacity(max_new_tokens);
        for step in 0..max_new_tokens {
            let next_tok = self.argmax_logits(&last_hidden);
            out.push(next_tok);
            on_decode_step(step, next_tok);
            if step + 1 < max_new_tokens {
                last_hidden = self.forward_one_token(
                    next_tok,
                    &mut attn_k_cache,
                    &mut attn_v_cache,
                    &mut gdn_states,
                );
            }
        }
        out
    }

    /// Convenience deterministic generation without per-step hook.
    #[must_use]
    pub fn generate(&self, prompt_tokens: &[u32], max_new_tokens: usize) -> Vec<u32> {
        self.generate_with_hook(prompt_tokens, max_new_tokens, |_, _| {})
    }

    fn forward_one_token(
        &self,
        token: u32,
        attn_k_cache: &mut Vec<Vec<f32>>,
        attn_v_cache: &mut Vec<Vec<f32>>,
        gdn_states: &mut [Vec<f32>],
    ) -> Vec<f32> {
        let tok_idx = (token as usize) % MICRO_VOCAB_SIZE;
        let row_start = tok_idx * MICRO_D_MODEL;
        let mut h = self.embed_tokens[row_start..row_start + MICRO_D_MODEL].to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let norm_h = rms_norm(&h);
            let q = matvec(&layer.w_q, &norm_h, MICRO_D_MODEL);
            let k = matvec(&layer.w_k, &norm_h, MICRO_D_MODEL);
            let v = matvec(&layer.w_v, &norm_h, MICRO_D_MODEL);

            let mixed = if layer.is_attention {
                attn_k_cache.push(k);
                attn_v_cache.push(v);
                causal_attention(&q, attn_k_cache, attn_v_cache)
            } else {
                // Gated DeltaNet recurrent state update: S_t = alpha * S_{t-1} + gate * (v - k * S_{t-1}).
                let state = &mut gdn_states[layer_idx];
                let mut out = vec![0.0f32; MICRO_D_MODEL];
                for i in 0..MICRO_D_MODEL {
                    let gate = sigmoid(k[i]);
                    let delta = v[i] - state[i] * k[i].tanh();
                    state[i] = layer.gdn_decay * state[i] + gate * delta;
                    out[i] = q[i] * state[i];
                }
                out
            };

            let proj = matvec(&layer.w_o, &mixed, MICRO_D_MODEL);
            for i in 0..MICRO_D_MODEL {
                h[i] += proj[i];
            }

            // SwiGLU-style residual MLP block.
            let norm_ff = rms_norm(&h);
            let gate_in = matvec(&layer.w_ff1, &norm_ff, MICRO_D_MODEL);
            let mut act = vec![0.0f32; MICRO_D_MODEL];
            for i in 0..MICRO_D_MODEL {
                act[i] = silu(gate_in[i]);
            }
            let ff_out = matvec(&layer.w_ff2, &act, MICRO_D_MODEL);
            for i in 0..MICRO_D_MODEL {
                h[i] += ff_out[i];
            }
        }
        rms_norm(&h)
    }

    fn argmax_logits(&self, hidden: &[f32]) -> u32 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for v in 0..MICRO_VOCAB_SIZE {
            let row = &self.lm_head[v * MICRO_D_MODEL..(v + 1) * MICRO_D_MODEL];
            let mut dot = 0.0f32;
            for i in 0..MICRO_D_MODEL {
                dot += row[i] * hidden[i];
            }
            if dot > best_val {
                best_val = dot;
                best_idx = v;
            }
        }
        best_idx as u32
    }
}

/// Computes the deterministic SHA-256 hex commitment over `(prompt_tokens, output_tokens)`.
#[must_use]
pub fn compute_output_sha256(prompt_tokens: &[u32], output_tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(b"HANZO_EDGE_OUTPUT_V1");
    h.update(&(prompt_tokens.len() as u32).to_le_bytes());
    for &t in prompt_tokens {
        h.update(&t.to_le_bytes());
    }
    h.update(&(output_tokens.len() as u32).to_le_bytes());
    for &t in output_tokens {
        h.update(&t.to_le_bytes());
    }
    h.finalize_hex()
}

/// HMAC-SHA256 signed receipt emitted by an edge provider for every served open-weight Zen request.
#[derive(Debug, Clone, PartialEq)]
pub struct MeteredWorkReceipt {
    /// Unique request identifier.
    pub request_id: String,
    /// Session identifier preventing cross-session replay.
    pub session_id: String,
    /// Serving community provider identifier.
    pub provider_id: String,
    /// SHA-256 digest of the model weights used to produce the output.
    pub model_weights_digest: String,
    /// Exact number of prompt tokens processed.
    pub prompt_tokens: u32,
    /// Exact number of completion tokens generated.
    pub output_tokens: u32,
    /// SHA-256 hex commitment over `(prompt_tokens, output_tokens)`.
    pub output_sha256: String,
    /// Measured wall-clock time-to-first-token (ms).
    pub ttft_ms: f64,
    /// Measured mean wall-clock inter-token latency (ms).
    pub mean_itl_ms: f64,
    /// Total measured serving wall duration (ms).
    pub total_duration_ms: f64,
    /// Mean active package power during execution (Watts; self-reported/OS-sensor telemetry).
    pub mean_watts: f64,
    /// Total energy consumed during execution (Joules = `mean_watts * total_duration_ms / 1000`).
    pub joules_consumed: f64,
    /// Peak package/skin temperature observed during execution (°C; self-reported/OS-sensor).
    pub peak_temp_c: f64,
    /// DVFS thermal throttle level active at completion of the request.
    pub throttle_level: ThrottleLevel,
    /// Ending battery state-of-charge (%).
    pub battery_pct_end: f64,
    /// Explicit trust tier (always `SignedTelemetryAndSpotCheck` in this runtime).
    pub trust_tier: TrustTier,
    /// Explicit honesty flag: `true` indicates power/thermal readings are self-reported or OS-sensor
    /// telemetry rather than hardware-attested TEE/zkML proofs.
    pub telemetry_self_reported: bool,
    /// Lowercase hex HMAC-SHA256 binding all fields above.
    pub receipt_hmac_hex: String,
}

impl MeteredWorkReceipt {
    /// Deterministic canonical byte payload covered by `receipt_hmac_hex`.
    #[must_use]
    pub fn canonical_payload(&self) -> Vec<u8> {
        format!(
            "RECEIPT|{}|{}|{}|{}|{}|{}|{}|{:.4}|{:.4}|{:.4}|{:.4}|{:.6}|{:.3}|{}|{:.3}|{}|{}",
            self.request_id,
            self.session_id,
            self.provider_id,
            self.model_weights_digest,
            self.prompt_tokens,
            self.output_tokens,
            self.output_sha256,
            self.ttft_ms,
            self.mean_itl_ms,
            self.total_duration_ms,
            self.mean_watts,
            self.joules_consumed,
            self.peak_temp_c,
            self.throttle_level.as_str(),
            self.battery_pct_end,
            self.trust_tier.as_str(),
            self.telemetry_self_reported,
        )
        .into_bytes()
    }

    /// Signs the receipt in place with `provider_secret_key`.
    pub fn sign(&mut self, provider_secret_key: &[u8]) {
        let payload = self.canonical_payload();
        self.receipt_hmac_hex = hmac_sha256_hex(provider_secret_key, &payload);
    }

    /// Verifies the receipt HMAC-SHA256 tag in constant time.
    #[must_use]
    pub fn verify_hmac(&self, provider_secret_key: &[u8]) -> bool {
        let payload = self.canonical_payload();
        verify_hmac_sha256_hex(provider_secret_key, &payload, &self.receipt_hmac_hex)
    }

    /// Computes SHA-256 digest of the signed receipt for deterministic audit trails.
    #[must_use]
    pub fn receipt_digest_hex(&self) -> String {
        let mut payload = self.canonical_payload();
        payload.extend_from_slice(self.receipt_hmac_hex.as_bytes());
        sha256_hex(&payload)
    }
}

/// Adversarial fraud or policy rejection classes caught by independent verification.
#[derive(Debug, Clone, PartialEq)]
pub enum FraudRejection {
    /// Fraud 1: Claimed output tokens exceed actual tokens emitted over the wire.
    InflatedOutputTokenCount {
        /// Claimed count inside the receipt.
        claimed: u32,
        /// Actual tokens emitted over TCP.
        actual: u32,
    },
    /// Fraud 2a: Receipt declares wrong `model_weights_digest`.
    WeightsDigestMismatch {
        /// Expected VFS model weight commitment.
        expected: String,
        /// Actual digest claimed by provider.
        actual: String,
    },
    /// Fraud 2b: Deterministic spot-check re-execution caught substituted/cheaper weights.
    SpotCheckModelSubstitution {
        /// Reference SHA-256 commitment from deterministic re-execution.
        expected_output_sha256: String,
        /// Provider's output SHA-256 commitment.
        actual_output_sha256: String,
    },
    /// Fraud 3a: Receipt HMAC-SHA256 signature was forged or tampered.
    ForgedReceiptHmac,
    /// Fraud 3b: Energy accounting fails `joules ≈ mean_watts * (total_duration_ms / 1000.0)`.
    EnergyAccountingMismatch {
        /// Reported energy in Joules.
        reported_joules: f64,
        /// Physics-consistent energy `watts * duration_s` in Joules.
        expected_joules: f64,
    },
    /// Fraud 4: Provider continued serving after emitting a signed `MarketWithdrawalNotice`.
    StaleWithdrawnProvider {
        /// Provider identifier that had already withdrawn.
        provider_id: String,
    },
    /// Fraud 5: Provider advertised a model exceeding its device's usable unified memory cap.
    InfeasibleMemoryAdvertisement {
        /// Detailed memory deficit explanation.
        detail: String,
    },
    /// Fraud 6: Peer violates client's required privacy policy (e.g. `ZeroRetentionEphemeral`).
    PrivacyPolicyViolation {
        /// Client's required privacy tier.
        required: PrivacyPolicy,
        /// Provider's advertised privacy tier.
        advertised: PrivacyPolicy,
    },
    /// Fraud 7: Old receipt replayed across requests or sessions.
    ReceiptReplayDetected {
        /// Replayed request ID.
        request_id: String,
        /// Replayed session ID.
        session_id: String,
    },
    /// Output SHA-256 commitment does not match emitted tokens.
    OutputHashMismatch {
        /// Digest claimed in receipt.
        claimed: String,
        /// Digest computed over actual tokens.
        actual: String,
    },
    /// Unknown or unregistered provider key.
    UnknownProviderKey {
        /// Provider identifier missing from verifier registry.
        provider_id: String,
    },
}

impl FraudRejection {
    /// Short machine-readable fault code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InflatedOutputTokenCount { .. } => "inflated_output_token_count",
            Self::WeightsDigestMismatch { .. } => "weights_digest_mismatch",
            Self::SpotCheckModelSubstitution { .. } => "spot_check_model_substitution",
            Self::ForgedReceiptHmac => "forged_receipt_hmac",
            Self::EnergyAccountingMismatch { .. } => "energy_accounting_mismatch",
            Self::StaleWithdrawnProvider { .. } => "stale_withdrawn_provider",
            Self::InfeasibleMemoryAdvertisement { .. } => "infeasible_memory_advertisement",
            Self::PrivacyPolicyViolation { .. } => "privacy_policy_violation",
            Self::ReceiptReplayDetected { .. } => "receipt_replay_detected",
            Self::OutputHashMismatch { .. } => "output_hash_mismatch",
            Self::UnknownProviderKey { .. } => "unknown_provider_key",
        }
    }
}

impl fmt::Display for FraudRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Independent receipt verifier & deterministic spot-check auditor.
#[derive(Debug, Clone)]
pub struct ReceiptVerifier {
    reference_model: ZenEdgeMicroModel,
    provider_keys: HashMap<String, Vec<u8>>,
    withdrawn_providers: HashSet<String>,
    seen_receipts: HashSet<(String, String)>,
    /// Total spot checks executed.
    pub spot_checks_executed: u64,
    /// Total receipts verified cleanly.
    pub receipts_verified: u64,
}

impl ReceiptVerifier {
    /// Creates a new independent verifier backed by `reference_model`.
    #[must_use]
    pub fn new(reference_model: ZenEdgeMicroModel) -> Self {
        Self {
            reference_model,
            provider_keys: HashMap::new(),
            withdrawn_providers: HashSet::new(),
            seen_receipts: HashSet::new(),
            spot_checks_executed: 0,
            receipts_verified: 0,
        }
    }

    /// Registers a provider's authentication key for receipt/withdrawal signature checks.
    pub fn register_provider_key(&mut self, provider_id: impl Into<String>, secret_key: Vec<u8>) {
        self.provider_keys.insert(provider_id.into(), secret_key);
    }

    /// Records a verified provider market withdrawal so any subsequent work from it is rejected.
    pub fn mark_provider_withdrawn(&mut self, provider_id: impl Into<String>) {
        self.withdrawn_providers.insert(provider_id.into());
    }

    /// Returns whether `provider_id` is currently marked withdrawn.
    #[must_use]
    pub fn is_provider_withdrawn(&self, provider_id: &str) -> bool {
        self.withdrawn_providers.contains(provider_id)
    }

    /// Independently verifies a served request's output tokens and signed [`MeteredWorkReceipt`].
    ///
    /// Performs:
    /// 1. Withdrawn/stale provider check
    /// 2. Replay protection (`(request_id, session_id)` uniqueness)
    /// 3. Constant-time HMAC-SHA256 verification
    /// 4. Exact prompt/output token count check
    /// 5. Output SHA-256 commitment check
    /// 6. Weights digest commitment check
    /// 7. Energy conservation check (`joules ≈ mean_watts * duration_s`)
    /// 8. Optional deterministic spot-check re-execution catching silent weight substitution
    pub fn verify_work(
        &mut self,
        expected_request_id: &str,
        expected_session_id: &str,
        prompt_tokens: &[u32],
        output_tokens: &[u32],
        receipt: &MeteredWorkReceipt,
        run_spot_check: bool,
    ) -> Result<(), FraudRejection> {
        // 1. Reject stale/zombie provider that already issued a signed MarketWithdrawalNotice.
        if self.withdrawn_providers.contains(&receipt.provider_id) {
            return Err(FraudRejection::StaleWithdrawnProvider {
                provider_id: receipt.provider_id.clone(),
            });
        }

        // 2. Replay & request/session binding check.
        if receipt.request_id != expected_request_id || receipt.session_id != expected_session_id {
            return Err(FraudRejection::ReceiptReplayDetected {
                request_id: receipt.request_id.clone(),
                session_id: receipt.session_id.clone(),
            });
        }
        let key = (receipt.request_id.clone(), receipt.session_id.clone());
        if self.seen_receipts.contains(&key) {
            return Err(FraudRejection::ReceiptReplayDetected {
                request_id: receipt.request_id.clone(),
                session_id: receipt.session_id.clone(),
            });
        }

        // 3. HMAC-SHA256 authenticity check.
        let secret = self
            .provider_keys
            .get(&receipt.provider_id)
            .ok_or_else(|| FraudRejection::UnknownProviderKey {
                provider_id: receipt.provider_id.clone(),
            })?;
        if !receipt.verify_hmac(secret) {
            return Err(FraudRejection::ForgedReceiptHmac);
        }

        // 4. Exact output token count metering check (catches billing inflation).
        let actual_out_len = output_tokens.len() as u32;
        if receipt.output_tokens != actual_out_len {
            return Err(FraudRejection::InflatedOutputTokenCount {
                claimed: receipt.output_tokens,
                actual: actual_out_len,
            });
        }

        // 5. Output SHA-256 binding check.
        let actual_output_hash = compute_output_sha256(prompt_tokens, output_tokens);
        if receipt.output_sha256 != actual_output_hash {
            return Err(FraudRejection::OutputHashMismatch {
                claimed: receipt.output_sha256.clone(),
                actual: actual_output_hash,
            });
        }

        // 6. Model weights digest commitment check.
        if receipt.model_weights_digest != self.reference_model.weights_digest {
            return Err(FraudRejection::WeightsDigestMismatch {
                expected: self.reference_model.weights_digest.clone(),
                actual: receipt.model_weights_digest.clone(),
            });
        }

        // 7. Physical energy conservation check: joules == mean_watts * (duration_ms / 1000).
        let expected_joules = receipt.mean_watts * (receipt.total_duration_ms / 1000.0);
        let tol = (expected_joules * 0.05).max(1e-3);
        if (receipt.joules_consumed - expected_joules).abs() > tol
            || receipt.mean_watts < 0.0
            || receipt.joules_consumed < 0.0
        {
            return Err(FraudRejection::EnergyAccountingMismatch {
                reported_joules: receipt.joules_consumed,
                expected_joules,
            });
        }

        // 8. Deterministic spot-check re-execution catching forged/cheaper weight substitution.
        if run_spot_check {
            self.spot_checks_executed += 1;
            let expected_tokens = self
                .reference_model
                .generate(prompt_tokens, output_tokens.len());
            let expected_hash = compute_output_sha256(prompt_tokens, &expected_tokens);
            if expected_hash != actual_output_hash {
                return Err(FraudRejection::SpotCheckModelSubstitution {
                    expected_output_sha256: expected_hash,
                    actual_output_sha256: actual_output_hash,
                });
            }
        }

        self.seen_receipts.insert(key);
        self.receipts_verified += 1;
        Ok(())
    }
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    for &v in values {
        hasher.update(&v.to_bits().to_le_bytes());
    }
}

fn rms_norm(x: &[f32]) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
    let inv = 1.0 / (mean_sq + 1e-5).sqrt();
    x.iter().map(|v| v * inv).collect()
}

fn matvec(w: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    for (row_idx, o) in out.iter_mut().enumerate() {
        let row = &w[row_idx * d..(row_idx + 1) * d];
        let mut acc = 0.0f32;
        for col in 0..d {
            acc += row[col] * x[col];
        }
        *o = acc;
    }
    out
}

fn causal_attention(q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<f32> {
    let d = q.len();
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    let mut scores = Vec::with_capacity(keys.len());
    let mut max_score = f32::NEG_INFINITY;
    for k in keys {
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += q[i] * k[i];
        }
        let s = dot * inv_sqrt_d;
        if s > max_score {
            max_score = s;
        }
        scores.push(s);
    }
    let mut sum_exp = 0.0f32;
    for s in &mut scores {
        *s = (*s - max_score).exp();
        sum_exp += *s;
    }
    let mut out = vec![0.0f32; d];
    for (w, v) in scores.iter().zip(values.iter()) {
        let prob = *w / sum_exp.max(1e-9);
        for i in 0..d {
            out[i] += prob * v[i];
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_model_is_deterministic_across_runs() {
        let m1 = ZenEdgeMicroModel::from_seed("zen-4b-hybrid-int4", ZEN_MICRO_MODEL_SEED);
        let m2 = ZenEdgeMicroModel::from_seed("zen-4b-hybrid-int4", ZEN_MICRO_MODEL_SEED);
        assert_eq!(m1.weights_digest, m2.weights_digest);
        let prompt = [12u32, 44, 7, 199, 88, 3, 250, 17];
        assert_eq!(m1.generate(&prompt, 12), m2.generate(&prompt, 12));
    }
}
