// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic fixed-point step-trace ML execution VM and canonical state roots.
//!
//! Each inference executes a deterministic sequence of atomic transformer/MLP
//! operations (`EmbedToken`, `RmsNorm`, `LinearProj`, `ResidualAdd`, `SiLU`,
//! `FinalRmsNorm`, `LmHeadLogits`, `ArgMaxEmit`) over Q12 activations and int8
//! weights with `i128` accumulators. Every intermediate step `0..=T` produces a
//! canonical domain-separated SHA-256 `StateRoot` and an optional Merkle tree
//! commitment, enabling $O(\log_2 T)$ bisection and $O(1)$ single-step judging
//! via [`verify_single_step`].

use crate::crypto::{
    ct_eq_digest, Digest, Sha256, SplitMix64, TraceMerkleTree, DOMAIN_MODEL, DOMAIN_STATE,
};
use crate::error::OpAiError;
use crate::profile::{
    add_i32, fixed_point_silu_q12, isqrt_u128, narrow_i32, rescale_i128, round_div_i128,
    DyadicScale, ExecutionProfile, InputCommitment,
};

/// Number of atomic steps per transformer layer:
/// 1. Pre-proj RMSNorm
/// 2. Attention-mix int8 projection + bias
/// 3. Post-proj residual add
/// 4. Pre-FFN RMSNorm
/// 5. FFN up int8 projection + bias
/// 6. FFN fixed-point SiLU LUT activation
/// 7. FFN down int8 projection + bias
/// 8. Post-FFN residual add
pub const STEPS_PER_LAYER: u64 = 8;

/// Non-layer atomic steps per generated token:
/// `EmbedToken` + `FinalRmsNorm` + `LmHeadLogits` + `ArgMaxEmit`.
pub const BASE_STEPS_PER_TOKEN: u64 = 4;

/// A single quantized transformer/MLP block with int8 weights and Q12 norms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedLayer {
    /// Pre-projection RMSNorm weights in Q12 (`len == hidden_dim`).
    pub norm1_gamma: Vec<i32>,
    /// Square mixing projection weights in row-major `hidden_dim x hidden_dim`.
    pub mix_weights: Vec<i8>,
    /// Exact dyadic scale for `mix_weights`.
    pub mix_scale: DyadicScale,
    /// Bias vector for mixing projection (`len == hidden_dim`).
    pub mix_bias: Vec<i32>,
    /// Pre-FFN RMSNorm weights in Q12 (`len == hidden_dim`).
    pub norm2_gamma: Vec<i32>,
    /// FFN up-projection weights in row-major `ffn_dim x hidden_dim`.
    pub ffn_up_weights: Vec<i8>,
    /// Exact dyadic scale for `ffn_up_weights`.
    pub ffn_up_scale: DyadicScale,
    /// Bias vector for FFN up-projection (`len == ffn_dim`).
    pub ffn_up_bias: Vec<i32>,
    /// FFN down-projection weights in row-major `hidden_dim x ffn_dim`.
    pub ffn_down_weights: Vec<i8>,
    /// Exact dyadic scale for `ffn_down_weights`.
    pub ffn_down_scale: DyadicScale,
    /// Bias vector for FFN down-projection (`len == hidden_dim`).
    pub ffn_down_bias: Vec<i32>,
}

/// Deterministic quantized transformer/MLP model executed by the step-trace VM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedModel {
    /// Vocabulary size (`>= 2`).
    pub vocab_size: usize,
    /// Hidden representation width `D` (`>= 2`).
    pub hidden_dim: usize,
    /// Feed-forward intermediate width `F` (`>= 2`).
    pub ffn_dim: usize,
    /// Token embedding table in Q12, row-major `vocab_size x hidden_dim`.
    pub embeddings: Vec<i32>,
    /// Ordered transformer/MLP layers (`0` layers permitted for shallow workloads).
    pub layers: Vec<QuantizedLayer>,
    /// Final RMSNorm weights in Q12 (`len == hidden_dim`).
    pub final_norm_gamma: Vec<i32>,
    /// Output LM head weights in row-major `vocab_size x hidden_dim`.
    pub lm_head_weights: Vec<i8>,
    /// Exact dyadic scale for `lm_head_weights`.
    pub lm_head_scale: DyadicScale,
    /// Output LM head bias (`len == vocab_size`).
    pub lm_head_bias: Vec<i32>,
}

impl QuantizedModel {
    /// Deterministically synthesize a well-conditioned quantized model from `seed`.
    pub fn synthesize(
        seed: u64,
        vocab_size: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        num_layers: usize,
    ) -> Result<Self, OpAiError> {
        if vocab_size < 2 || hidden_dim < 2 || ffn_dim < 2 {
            return Err(OpAiError::InvalidModel(
                "vocab_size, hidden_dim, and ffn_dim must all be >= 2".to_string(),
            ));
        }
        let mut rng = SplitMix64::new(seed);
        let embeddings: Vec<i32> = (0..vocab_size * hidden_dim)
            .map(|_| rng.next_i32_symmetric(2048))
            .collect();

        // Dyadic scales chosen to keep activations stably inside Q12 [-16384, 16384].
        let mix_scale = DyadicScale::new(1, 9)?;
        let ffn_up_scale = DyadicScale::new(1, 9)?;
        let ffn_down_scale = DyadicScale::new(1, 9)?;
        let lm_head_scale = DyadicScale::new(1, 8)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let norm1_gamma: Vec<i32> = (0..hidden_dim)
                .map(|_| 4096 + rng.next_i32_symmetric(512))
                .collect();
            let mix_weights: Vec<i8> = (0..hidden_dim * hidden_dim)
                .map(|_| rng.next_i8_symmetric(64))
                .collect();
            let mix_bias: Vec<i32> = (0..hidden_dim)
                .map(|_| rng.next_i32_symmetric(256))
                .collect();
            let norm2_gamma: Vec<i32> = (0..hidden_dim)
                .map(|_| 4096 + rng.next_i32_symmetric(512))
                .collect();
            let ffn_up_weights: Vec<i8> = (0..ffn_dim * hidden_dim)
                .map(|_| rng.next_i8_symmetric(64))
                .collect();
            let ffn_up_bias: Vec<i32> = (0..ffn_dim).map(|_| rng.next_i32_symmetric(256)).collect();
            let ffn_down_weights: Vec<i8> = (0..hidden_dim * ffn_dim)
                .map(|_| rng.next_i8_symmetric(64))
                .collect();
            let ffn_down_bias: Vec<i32> = (0..hidden_dim)
                .map(|_| rng.next_i32_symmetric(256))
                .collect();
            layers.push(QuantizedLayer {
                norm1_gamma,
                mix_weights,
                mix_scale,
                mix_bias,
                norm2_gamma,
                ffn_up_weights,
                ffn_up_scale,
                ffn_up_bias,
                ffn_down_weights,
                ffn_down_scale,
                ffn_down_bias,
            });
        }

        let final_norm_gamma: Vec<i32> = (0..hidden_dim)
            .map(|_| 4096 + rng.next_i32_symmetric(512))
            .collect();
        let lm_head_weights: Vec<i8> = (0..vocab_size * hidden_dim)
            .map(|_| rng.next_i8_symmetric(64))
            .collect();
        let lm_head_bias: Vec<i32> = (0..vocab_size)
            .map(|_| rng.next_i32_symmetric(256))
            .collect();

        let model = Self {
            vocab_size,
            hidden_dim,
            ffn_dim,
            embeddings,
            layers,
            final_norm_gamma,
            lm_head_weights,
            lm_head_scale,
            lm_head_bias,
        };
        model.validate()?;
        Ok(model)
    }

    /// Validate all tensor shapes and scale exponents.
    pub fn validate(&self) -> Result<(), OpAiError> {
        if self.vocab_size < 2 || self.hidden_dim < 2 || self.ffn_dim < 2 {
            return Err(OpAiError::InvalidModel(
                "model dimensions must be >= 2".to_string(),
            ));
        }
        if self.embeddings.len() != self.vocab_size * self.hidden_dim {
            return Err(OpAiError::InvalidModel(
                "embeddings length mismatch".to_string(),
            ));
        }
        if self.final_norm_gamma.len() != self.hidden_dim {
            return Err(OpAiError::InvalidModel(
                "final_norm_gamma length mismatch".to_string(),
            ));
        }
        if self.lm_head_weights.len() != self.vocab_size * self.hidden_dim {
            return Err(OpAiError::InvalidModel(
                "lm_head_weights length mismatch".to_string(),
            ));
        }
        if self.lm_head_bias.len() != self.vocab_size {
            return Err(OpAiError::InvalidModel(
                "lm_head_bias length mismatch".to_string(),
            ));
        }
        if self.lm_head_scale.shift > 62 {
            return Err(OpAiError::InvalidModel(
                "lm_head_scale shift > 62".to_string(),
            ));
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            if layer.norm1_gamma.len() != self.hidden_dim
                || layer.mix_weights.len() != self.hidden_dim * self.hidden_dim
                || layer.mix_bias.len() != self.hidden_dim
                || layer.norm2_gamma.len() != self.hidden_dim
                || layer.ffn_up_weights.len() != self.ffn_dim * self.hidden_dim
                || layer.ffn_up_bias.len() != self.ffn_dim
                || layer.ffn_down_weights.len() != self.hidden_dim * self.ffn_dim
                || layer.ffn_down_bias.len() != self.hidden_dim
            {
                return Err(OpAiError::InvalidModel(format!(
                    "layer {idx} tensor shape mismatch"
                )));
            }
            if layer.mix_scale.shift > 62
                || layer.ffn_up_scale.shift > 62
                || layer.ffn_down_scale.shift > 62
            {
                return Err(OpAiError::InvalidModel(format!(
                    "layer {idx} dyadic scale shift > 62"
                )));
            }
        }
        Ok(())
    }

    /// Number of atomic steps per generated token for this model.
    pub fn steps_per_token(&self) -> u64 {
        BASE_STEPS_PER_TOKEN + STEPS_PER_LAYER * (self.layers.len() as u64)
    }

    /// Total atomic steps required to generate `max_new_tokens`.
    pub fn total_steps(&self, max_new_tokens: u32) -> u64 {
        self.steps_per_token() * u64::from(max_new_tokens)
    }

    /// Canonical domain-separated SHA-256 commitment over all model weights & scales.
    pub fn digest(&self) -> Digest {
        let mut h = Sha256::with_domain(DOMAIN_MODEL);
        h.update_u64_le(self.vocab_size as u64);
        h.update_u64_le(self.hidden_dim as u64);
        h.update_u64_le(self.ffn_dim as u64);
        h.update_u64_le(self.layers.len() as u64);
        for &v in &self.embeddings {
            h.update_i32_le(v);
        }
        for layer in &self.layers {
            for &v in &layer.norm1_gamma {
                h.update_i32_le(v);
            }
            for &w in &layer.mix_weights {
                h.update_u8(w as u8);
            }
            layer.mix_scale.hash_into(&mut h);
            for &b in &layer.mix_bias {
                h.update_i32_le(b);
            }
            for &v in &layer.norm2_gamma {
                h.update_i32_le(v);
            }
            for &w in &layer.ffn_up_weights {
                h.update_u8(w as u8);
            }
            layer.ffn_up_scale.hash_into(&mut h);
            for &b in &layer.ffn_up_bias {
                h.update_i32_le(b);
            }
            for &w in &layer.ffn_down_weights {
                h.update_u8(w as u8);
            }
            layer.ffn_down_scale.hash_into(&mut h);
            for &b in &layer.ffn_down_bias {
                h.update_i32_le(b);
            }
        }
        for &v in &self.final_norm_gamma {
            h.update_i32_le(v);
        }
        for &w in &self.lm_head_weights {
            h.update_u8(w as u8);
        }
        self.lm_head_scale.hash_into(&mut h);
        for &b in &self.lm_head_bias {
            h.update_i32_le(b);
        }
        h.finalize()
    }
}

/// Atomic verifiable step operation executed at step index `k -> k + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOp {
    /// Deterministic token embedding lookup & positional accumulation.
    EmbedToken {
        /// Output token index being generated (`0..max_new_tokens`).
        gen_idx: u32,
    },
    /// Pre-projection fixed-point RMSNorm (`isqrt_u128`).
    RmsNormPreProj {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// Attention-mixing int8 matrix-vector multiply + bias.
    AttentionMixProj {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// Residual connection after mixing projection.
    ResidualPostProj {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// Pre-FFN fixed-point RMSNorm (`isqrt_u128`).
    RmsNormPreFfn {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// FFN up-projection int8 matrix-vector multiply + bias.
    FfnUpProj {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// FFN fixed-point SiLU LUT activation (`SIGMOID_LUT_Q15`).
    FfnSiLuActivation {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// FFN down-projection int8 matrix-vector multiply + bias.
    FfnDownProj {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// Residual connection after FFN down-projection.
    ResidualPostFfn {
        /// Output token index.
        gen_idx: u32,
        /// Transformer layer index.
        layer: u32,
    },
    /// Final fixed-point RMSNorm prior to LM head.
    FinalRmsNorm {
        /// Output token index.
        gen_idx: u32,
    },
    /// LM head int8 matrix-vector multiply + bias producing vocabulary logits.
    LmHeadLogits {
        /// Output token index.
        gen_idx: u32,
    },
    /// Deterministic tie-breaking ArgMax `(value desc, index asc)` emitting one token.
    ArgMaxEmit {
        /// Output token index.
        gen_idx: u32,
    },
}

impl StepOp {
    /// Decode the canonical [`StepOp`] for step index `step` in a model with `num_layers`.
    pub fn for_step(step: u64, num_layers: usize, total_steps: u64) -> Result<Self, OpAiError> {
        if step >= total_steps {
            return Err(OpAiError::InvalidStepIndex { step, total_steps });
        }
        let spt = BASE_STEPS_PER_TOKEN + STEPS_PER_LAYER * (num_layers as u64);
        let gen_idx = (step / spt) as u32;
        let offset = step % spt;
        if offset == 0 {
            return Ok(Self::EmbedToken { gen_idx });
        }
        let layer_span = STEPS_PER_LAYER * (num_layers as u64);
        if offset <= layer_span {
            let rel = offset - 1;
            let layer = (rel / STEPS_PER_LAYER) as u32;
            let sub = rel % STEPS_PER_LAYER;
            return Ok(match sub {
                0 => Self::RmsNormPreProj { gen_idx, layer },
                1 => Self::AttentionMixProj { gen_idx, layer },
                2 => Self::ResidualPostProj { gen_idx, layer },
                3 => Self::RmsNormPreFfn { gen_idx, layer },
                4 => Self::FfnUpProj { gen_idx, layer },
                5 => Self::FfnSiLuActivation { gen_idx, layer },
                6 => Self::FfnDownProj { gen_idx, layer },
                7 => Self::ResidualPostFfn { gen_idx, layer },
                _ => unreachable!("sub < 8"),
            });
        }
        let tail = offset - 1 - layer_span;
        Ok(match tail {
            0 => Self::FinalRmsNorm { gen_idx },
            1 => Self::LmHeadLogits { gen_idx },
            2 => Self::ArgMaxEmit { gen_idx },
            _ => unreachable!("tail < 3"),
        })
    }

    /// Stable short tag describing the operation category for transcripts and telemetry.
    pub fn op_tag(self) -> &'static str {
        match self {
            Self::EmbedToken { .. } => "embed_token",
            Self::RmsNormPreProj { .. } => "rms_norm_pre_proj",
            Self::AttentionMixProj { .. } => "attention_mix_proj",
            Self::ResidualPostProj { .. } => "residual_post_proj",
            Self::RmsNormPreFfn { .. } => "rms_norm_pre_ffn",
            Self::FfnUpProj { .. } => "ffn_up_proj",
            Self::FfnSiLuActivation { .. } => "ffn_silu_activation",
            Self::FfnDownProj { .. } => "ffn_down_proj",
            Self::ResidualPostFfn { .. } => "residual_post_ffn",
            Self::FinalRmsNorm { .. } => "final_rms_norm",
            Self::LmHeadLogits { .. } => "lm_head_logits",
            Self::ArgMaxEmit { .. } => "argmax_emit",
        }
    }
}

/// Complete deterministic VM architectural state at step boundary `step_index` (`0..=T`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmState {
    /// Completed step counter (`0` before the first step, `T` after the final step).
    pub step_index: u64,
    /// Current active token ID (initialized from the final prompt token, updated by `ArgMaxEmit`).
    pub active_token: u32,
    /// Main hidden activation vector (`len == hidden_dim`, Q12).
    pub hidden: Vec<i32>,
    /// Residual stream vector (`len == hidden_dim`, Q12).
    pub residual: Vec<i32>,
    /// Scratch register of width `hidden_dim`.
    pub scratch_d: Vec<i32>,
    /// Scratch register of width `ffn_dim`.
    pub scratch_f: Vec<i32>,
    /// Vocabulary logits register (`len == vocab_size`).
    pub logits: Vec<i32>,
    /// Autoregressively emitted output token IDs so far.
    pub output_tokens: Vec<u32>,
}

impl VmState {
    /// Construct the canonical initial VM state at step `0` before any ML step executes.
    pub fn initial(model: &QuantizedModel, input: &InputCommitment) -> Result<Self, OpAiError> {
        model.validate()?;
        if input.prompt_tokens.is_empty() {
            return Err(OpAiError::InvalidInput(
                "prompt_tokens must be non-empty".to_string(),
            ));
        }
        let active_token = *input.prompt_tokens.last().unwrap_or(&0) % (model.vocab_size as u32);
        Ok(Self {
            step_index: 0,
            active_token,
            hidden: vec![0i32; model.hidden_dim],
            residual: vec![0i32; model.hidden_dim],
            scratch_d: vec![0i32; model.hidden_dim],
            scratch_f: vec![0i32; model.ffn_dim],
            logits: vec![0i32; model.vocab_size],
            output_tokens: Vec::new(),
        })
    }

    /// Validate that register dimensions match the quantized model.
    pub fn validate_for_model(&self, model: &QuantizedModel) -> Result<(), OpAiError> {
        if self.hidden.len() != model.hidden_dim
            || self.residual.len() != model.hidden_dim
            || self.scratch_d.len() != model.hidden_dim
            || self.scratch_f.len() != model.ffn_dim
            || self.logits.len() != model.vocab_size
        {
            return Err(OpAiError::InvalidModel(
                "VmState register dimensions do not match QuantizedModel".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical domain-separated SHA-256 `StateRoot` for this state.
    ///
    /// Binds `(profile_digest, model_digest, input_hash)` alongside `step_index`
    /// and every architectural register so no state from another profile, model,
    /// or input can ever collide or be replayed across contexts.
    pub fn state_root(
        &self,
        profile_digest: &Digest,
        model_digest: &Digest,
        input_hash: &Digest,
    ) -> Digest {
        let mut h = Sha256::with_domain(DOMAIN_STATE);
        h.update(profile_digest);
        h.update(model_digest);
        h.update(input_hash);
        h.update_u64_le(self.step_index);
        h.update_u32_le(self.active_token);

        h.update_u64_le(self.hidden.len() as u64);
        for &v in &self.hidden {
            h.update_i32_le(v);
        }
        h.update_u64_le(self.residual.len() as u64);
        for &v in &self.residual {
            h.update_i32_le(v);
        }
        h.update_u64_le(self.scratch_d.len() as u64);
        for &v in &self.scratch_d {
            h.update_i32_le(v);
        }
        h.update_u64_le(self.scratch_f.len() as u64);
        for &v in &self.scratch_f {
            h.update_i32_le(v);
        }
        h.update_u64_le(self.logits.len() as u64);
        for &v in &self.logits {
            h.update_i32_le(v);
        }
        h.update_u64_le(self.output_tokens.len() as u64);
        for &tok in &self.output_tokens {
            h.update_u32_le(tok);
        }
        h.finalize()
    }
}

/// Fault injection specification used to model dishonest/corrupted providers
/// in tests and benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultSpec {
    /// Perturb `hidden[index]` by `delta` immediately after executing `step` (`step -> step + 1`).
    PerturbHidden {
        /// Target step index `0..T`.
        step: u64,
        /// Element index inside `hidden`.
        index: usize,
        /// Additive signed perturbation.
        delta: i32,
    },
    /// Perturb `scratch_d[index]` or `scratch_f[index]` after executing `step`.
    PerturbScratch {
        /// Target step index `0..T`.
        step: u64,
        /// Element index inside the scratch buffer.
        index: usize,
        /// Additive signed perturbation.
        delta: i32,
    },
    /// Perturb `logits[index]` after executing `step`.
    PerturbLogit {
        /// Target step index `0..T`.
        step: u64,
        /// Target vocabulary index.
        index: usize,
        /// Additive signed perturbation.
        delta: i32,
    },
    /// Force wrong emitted token at an `ArgMaxEmit` step.
    CorruptOutputToken {
        /// Target step index `0..T`.
        step: u64,
        /// XOR mask applied to emitted token (modulo `vocab_size`).
        xor_mask: u32,
    },
    /// Corrupt the active output register produced by `StepOp::for_step(step)` so the
    /// fault propagates realistically through subsequent steps to `end_root`.
    CorruptStepOutput {
        /// Target step index `0..T`.
        step: u64,
        /// Non-zero signed perturbation magnitude.
        delta: i32,
    },
    /// Forge the committed `state_roots` suffix from `step + 1..=T` without modifying
    /// VM registers (tests commitment-only cheating that reaches `end_root`).
    ForgePostRootOnly {
        /// Target step index `0..T` where commitment forgery begins.
        step: u64,
    },
    /// Forge the initial `start_root` (`state_roots[0]`) to test 0-round start-root fraud proofs.
    ForgeStartRoot,
}

/// Complete execution trace produced by a provider or independent challenger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionTrace {
    /// Canonical execution profile digest.
    pub profile_digest: Digest,
    /// Canonical quantized model digest.
    pub model_digest: Digest,
    /// Canonical input commitment digest.
    pub input_hash: Digest,
    /// Total atomic steps `T`.
    pub total_steps: u64,
    /// Canonical `StateRoot` at every step boundary `0..=T` (`len == T + 1`).
    pub state_roots: Vec<Digest>,
    /// Binary SHA-256 Merkle root over `state_roots[0..=T]`.
    pub trace_merkle_root: Digest,
    /// Snapshotted VM states at every step boundary `0..=T` (`len == T + 1`).
    pub states: Vec<VmState>,
    /// Final emitted output token IDs.
    pub output_tokens: Vec<u32>,
}

impl ExecutionTrace {
    /// Return the initial state root at step `0`.
    pub fn start_root(&self) -> Digest {
        self.state_roots[0]
    }

    /// Return the final state root at step `T`.
    pub fn end_root(&self) -> Digest {
        self.state_roots[self.total_steps as usize]
    }

    /// Return the `StateRoot` at step `k` (`0..=T`).
    pub fn root_at(&self, step: u64) -> Result<Digest, OpAiError> {
        let idx = step as usize;
        self.state_roots
            .get(idx)
            .copied()
            .ok_or(OpAiError::InvalidStepIndex {
                step,
                total_steps: self.total_steps,
            })
    }

    /// Return a reference to the `VmState` at step `k` (`0..=T`).
    pub fn state_at(&self, step: u64) -> Result<&VmState, OpAiError> {
        let idx = step as usize;
        self.states.get(idx).ok_or(OpAiError::InvalidStepIndex {
            step,
            total_steps: self.total_steps,
        })
    }

    /// Build a [`TraceMerkleTree`] over `state_roots` for inclusion proofs.
    pub fn merkle_tree(&self) -> Result<TraceMerkleTree, OpAiError> {
        TraceMerkleTree::build(&self.state_roots)
    }
}

/// Result returned by the on-chain-equivalent deterministic single-step judge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleStepVerdict {
    /// The single step index `lo` that was judged (`lo -> lo + 1`).
    pub step_index: u64,
    /// Short operation tag for step `lo`.
    pub op_tag: String,
    /// Whether the provided `pre_state` matched `expected_pre_root` and `step_index`.
    pub pre_root_valid: bool,
    /// Recomputed `StateRoot` of `pre_state`.
    pub recomputed_pre_root: Digest,
    /// Recomputed `StateRoot` after executing step `lo -> lo + 1` (or zero if `pre_root_valid == false`).
    pub actual_post_root: Digest,
    /// Defender's claimed `StateRoot` at `lo + 1`.
    pub claimed_post_root: Digest,
    /// True iff `pre_root_valid && actual_post_root == claimed_post_root`.
    pub defender_valid: bool,
}

/// Execute ONLY the single deterministic step `step_index -> step_index + 1`
/// starting from `pre_state`.
pub fn execute_single_step(
    profile: &ExecutionProfile,
    model: &QuantizedModel,
    input: &InputCommitment,
    pre_state: &VmState,
    step_index: u64,
    total_steps: u64,
) -> Result<VmState, OpAiError> {
    profile.validate()?;
    model.validate()?;
    pre_state.validate_for_model(model)?;
    let max_allowed = model.total_steps(input.max_new_tokens);
    if total_steps == 0 || total_steps > max_allowed {
        return Err(OpAiError::InvalidClaim(format!(
            "total_steps {total_steps} out of range 1..={max_allowed}"
        )));
    }
    if pre_state.step_index != step_index {
        return Err(OpAiError::InvalidStepIndex {
            step: pre_state.step_index,
            total_steps,
        });
    }

    let op = StepOp::for_step(step_index, model.layers.len(), total_steps)?;
    let mut next = pre_state.clone();
    next.step_index = step_index + 1;

    match op {
        StepOp::EmbedToken { gen_idx } => {
            let d = model.hidden_dim;
            if gen_idx == 0 {
                // Fold all prompt tokens deterministically into the initial hidden state.
                let count = input.prompt_tokens.len() as i128;
                let mut acc = vec![0i128; d];
                for (pos, &raw_tok) in input.prompt_tokens.iter().enumerate() {
                    let tok = (raw_tok as usize) % model.vocab_size;
                    let row = &model.embeddings[tok * d..(tok + 1) * d];
                    let pos_bias = ((pos as i128) + 1) * 13;
                    for (j, &emb) in row.iter().enumerate() {
                        acc[j] += i128::from(emb) + ((pos_bias + j as i128) & 0x7f) - 64;
                    }
                }
                for j in 0..d {
                    let avg = round_div_i128(acc[j], count, profile.rounding_mode)?;
                    next.hidden[j] = narrow_i32(avg, profile.overflow_mode)?;
                }
            } else {
                let tok = (pre_state.active_token as usize) % model.vocab_size;
                let row = &model.embeddings[tok * d..(tok + 1) * d];
                let pos_bias =
                    ((input.prompt_tokens.len() + gen_idx as usize) as i32).wrapping_mul(13);
                for j in 0..d {
                    let blended = (i128::from(pre_state.hidden[j]) >> 1)
                        + i128::from(row[j])
                        + i128::from((pos_bias.wrapping_add(j as i32)) & 0x7f)
                        - 64;
                    next.hidden[j] = narrow_i32(blended, profile.overflow_mode)?;
                }
            }
            next.residual.copy_from_slice(&next.hidden);
        }
        StepOp::RmsNormPreProj { layer, .. } => {
            let gamma = &model.layers[layer as usize].norm1_gamma;
            rmsnorm_q12(&pre_state.hidden, gamma, profile, &mut next.scratch_d)?;
        }
        StepOp::AttentionMixProj { layer, .. } => {
            let blk = &model.layers[layer as usize];
            int8_matvec_bias(
                &pre_state.scratch_d,
                &blk.mix_weights,
                blk.mix_scale,
                &blk.mix_bias,
                model.hidden_dim,
                model.hidden_dim,
                profile,
                &mut next.scratch_d,
            )?;
        }
        StepOp::ResidualPostProj { .. } => {
            for j in 0..model.hidden_dim {
                next.hidden[j] = add_i32(
                    pre_state.residual[j],
                    pre_state.scratch_d[j],
                    profile.overflow_mode,
                )?;
            }
            next.residual.copy_from_slice(&next.hidden);
        }
        StepOp::RmsNormPreFfn { layer, .. } => {
            let gamma = &model.layers[layer as usize].norm2_gamma;
            rmsnorm_q12(&pre_state.hidden, gamma, profile, &mut next.scratch_d)?;
        }
        StepOp::FfnUpProj { layer, .. } => {
            let blk = &model.layers[layer as usize];
            int8_matvec_bias(
                &pre_state.scratch_d,
                &blk.ffn_up_weights,
                blk.ffn_up_scale,
                &blk.ffn_up_bias,
                model.ffn_dim,
                model.hidden_dim,
                profile,
                &mut next.scratch_f,
            )?;
        }
        StepOp::FfnSiLuActivation { .. } => {
            for j in 0..model.ffn_dim {
                next.scratch_f[j] = fixed_point_silu_q12(
                    pre_state.scratch_f[j],
                    profile.rounding_mode,
                    profile.overflow_mode,
                )?;
            }
        }
        StepOp::FfnDownProj { layer, .. } => {
            let blk = &model.layers[layer as usize];
            int8_matvec_bias(
                &pre_state.scratch_f,
                &blk.ffn_down_weights,
                blk.ffn_down_scale,
                &blk.ffn_down_bias,
                model.hidden_dim,
                model.ffn_dim,
                profile,
                &mut next.scratch_d,
            )?;
        }
        StepOp::ResidualPostFfn { .. } => {
            for j in 0..model.hidden_dim {
                next.hidden[j] = add_i32(
                    pre_state.residual[j],
                    pre_state.scratch_d[j],
                    profile.overflow_mode,
                )?;
            }
            next.residual.copy_from_slice(&next.hidden);
        }
        StepOp::FinalRmsNorm { .. } => {
            rmsnorm_q12(
                &pre_state.hidden,
                &model.final_norm_gamma,
                profile,
                &mut next.scratch_d,
            )?;
        }
        StepOp::LmHeadLogits { .. } => {
            int8_matvec_bias(
                &pre_state.scratch_d,
                &model.lm_head_weights,
                model.lm_head_scale,
                &model.lm_head_bias,
                model.vocab_size,
                model.hidden_dim,
                profile,
                &mut next.logits,
            )?;
        }
        StepOp::ArgMaxEmit { .. } => {
            // Deterministic total-order ArgMax: higher value wins, lower index breaks ties.
            let mut best_idx = 0usize;
            let mut best_val = pre_state.logits[0];
            for (idx, &val) in pre_state.logits.iter().enumerate().skip(1) {
                if val > best_val {
                    best_val = val;
                    best_idx = idx;
                }
            }
            let emitted = best_idx as u32;
            next.active_token = emitted;
            next.output_tokens.push(emitted);
        }
    }

    Ok(next)
}

/// Execute the full deterministic step-trace for `total_steps` (defaulting to
/// `model.total_steps(input.max_new_tokens)` when `step_cap` is `None`).
pub fn execute_trace(
    profile: &ExecutionProfile,
    model: &QuantizedModel,
    input: &InputCommitment,
    step_cap: Option<u64>,
) -> Result<ExecutionTrace, OpAiError> {
    execute_trace_with_fault(profile, model, input, step_cap, None)
}

/// Execute the step-trace with an optional adversarial [`FaultSpec`] injected at a single step.
///
/// When a register-level fault is injected at step `k`, steps `k+1..T` continue
/// executing from the corrupted state (creating a realistic divergent suffix
/// `state_roots[k+1..=T]`) so interactive bisection must locate the exact first
/// corrupted transition `k -> k + 1`.
pub fn execute_trace_with_fault(
    profile: &ExecutionProfile,
    model: &QuantizedModel,
    input: &InputCommitment,
    step_cap: Option<u64>,
    fault: Option<FaultSpec>,
) -> Result<ExecutionTrace, OpAiError> {
    profile.validate()?;
    model.validate()?;

    let profile_digest = profile.digest();
    let model_digest = model.digest();
    if !ct_eq_digest(&input.profile_digest, &profile_digest) {
        return Err(OpAiError::InvalidInput(
            "InputCommitment.profile_digest does not match ExecutionProfile".to_string(),
        ));
    }
    if !ct_eq_digest(&input.model_digest, &model_digest) {
        return Err(OpAiError::InvalidInput(
            "InputCommitment.model_digest does not match QuantizedModel".to_string(),
        ));
    }
    let input_hash = input.input_hash();

    let full_steps = model.total_steps(input.max_new_tokens);
    let total_steps = match step_cap {
        Some(cap) => {
            if cap == 0 || cap > full_steps {
                return Err(OpAiError::InvalidInput(format!(
                    "step_cap {cap} must be in 1..={full_steps}"
                )));
            }
            cap
        }
        None => full_steps,
    };

    let init = VmState::initial(model, input)?;
    let mut init_root = init.state_root(&profile_digest, &model_digest, &input_hash);
    if matches!(fault, Some(FaultSpec::ForgeStartRoot)) {
        init_root[0] ^= 0xff;
    }

    let cap_len = (total_steps as usize) + 1;
    let mut states = Vec::with_capacity(cap_len);
    let mut state_roots = Vec::with_capacity(cap_len);
    states.push(init);
    state_roots.push(init_root);

    for step in 0..total_steps {
        let mut next = execute_single_step(
            profile,
            model,
            input,
            &states[step as usize],
            step,
            total_steps,
        )?;
        let mut forge_root = false;
        if let Some(f) = fault {
            match f {
                FaultSpec::PerturbHidden {
                    step: target,
                    index,
                    delta,
                } if target == step => {
                    let i = index % next.hidden.len();
                    let d = if delta == 0 { 1 } else { delta };
                    next.hidden[i] = next.hidden[i].wrapping_add(d);
                    next.residual[i] = next.residual[i].wrapping_add(d);
                }
                FaultSpec::PerturbScratch {
                    step: target,
                    index,
                    delta,
                } if target == step => {
                    let i = index % next.scratch_d.len();
                    let d = if delta == 0 { 1 } else { delta };
                    next.scratch_d[i] = next.scratch_d[i].wrapping_add(d);
                }
                FaultSpec::PerturbLogit {
                    step: target,
                    index,
                    delta,
                } if target == step => {
                    let i = index % next.logits.len();
                    let d = if delta == 0 { 1 } else { delta };
                    next.logits[i] = next.logits[i].wrapping_add(d);
                }
                FaultSpec::CorruptOutputToken {
                    step: target,
                    xor_mask,
                } if target == step => {
                    let mask = if xor_mask == 0 { 1 } else { xor_mask };
                    if let Some(last) = next.output_tokens.last_mut() {
                        let forged = (*last ^ mask) % (model.vocab_size as u32);
                        let forged = if forged == *last {
                            (*last + 1) % (model.vocab_size as u32)
                        } else {
                            forged
                        };
                        *last = forged;
                        next.active_token = forged;
                    } else {
                        next.active_token = (next.active_token + 1) % (model.vocab_size as u32);
                    }
                }
                FaultSpec::CorruptStepOutput {
                    step: target,
                    delta,
                } if target == step => {
                    let d = if delta == 0 { 512 } else { delta };
                    let op = StepOp::for_step(step, model.layers.len(), total_steps)?;
                    match op {
                        StepOp::EmbedToken { .. }
                        | StepOp::ResidualPostProj { .. }
                        | StepOp::ResidualPostFfn { .. } => {
                            for j in 0..next.hidden.len() {
                                let signed_d = if j % 2 == 0 { d } else { -d };
                                next.hidden[j] = next.hidden[j].wrapping_add(signed_d);
                                next.residual[j] = next.residual[j].wrapping_add(signed_d);
                            }
                        }
                        StepOp::RmsNormPreProj { .. }
                        | StepOp::AttentionMixProj { .. }
                        | StepOp::RmsNormPreFfn { .. }
                        | StepOp::FfnDownProj { .. }
                        | StepOp::FinalRmsNorm { .. } => {
                            for j in 0..next.scratch_d.len() {
                                let signed_d = if j % 2 == 0 { d } else { -d };
                                next.scratch_d[j] = next.scratch_d[j].wrapping_add(signed_d);
                            }
                        }
                        StepOp::FfnUpProj { .. } | StepOp::FfnSiLuActivation { .. } => {
                            for j in 0..next.scratch_f.len() {
                                let signed_d = if j % 2 == 0 { d } else { -d };
                                next.scratch_f[j] = next.scratch_f[j].wrapping_add(signed_d);
                            }
                        }
                        StepOp::LmHeadLogits { .. } => {
                            // Boost a non-argmax vocabulary entry so the emitted token flips
                            // deterministically and propagates through subsequent autoregressive steps.
                            let mut best_idx = 0usize;
                            let mut best_val = next.logits[0];
                            for (idx, &val) in next.logits.iter().enumerate().skip(1) {
                                if val > best_val {
                                    best_val = val;
                                    best_idx = idx;
                                }
                            }
                            let alt = (best_idx + 1) % next.logits.len();
                            next.logits[alt] = best_val.saturating_add(d.abs().max(1_024));
                        }
                        StepOp::ArgMaxEmit { .. } => {
                            if let Some(last) = next.output_tokens.last_mut() {
                                let forged = (*last + 1) % (model.vocab_size as u32);
                                *last = forged;
                                next.active_token = forged;
                            }
                        }
                    }
                }
                FaultSpec::ForgePostRootOnly { step: target } if step >= target => {
                    forge_root = true;
                }
                _ => {}
            }
        }
        let mut root = next.state_root(&profile_digest, &model_digest, &input_hash);
        if forge_root {
            root[0] ^= 0x5a;
        }
        states.push(next);
        state_roots.push(root);
    }

    let merkle = TraceMerkleTree::build(&state_roots)?;
    let output_tokens = states
        .last()
        .map(|s| s.output_tokens.clone())
        .unwrap_or_default();

    Ok(ExecutionTrace {
        profile_digest,
        model_digest,
        input_hash,
        total_steps,
        state_roots,
        trace_merkle_root: merkle.root(),
        states,
        output_tokens,
    })
}

/// Pre-validated immutable judge context binding `(profile, model, input)` and
/// their canonical SHA-256 digests once per deployed model/claim for true `O(1)`
/// single-step judging without re-hashing model weights per step.
#[derive(Debug, Clone)]
pub struct JudgeContext<'a> {
    /// Bound deterministic execution profile.
    pub profile: &'a ExecutionProfile,
    /// Bound quantized model (immutably borrowed, preventing digest staleness).
    pub model: &'a QuantizedModel,
    /// Bound input commitment.
    pub input: &'a InputCommitment,
    /// Pre-computed canonical profile digest.
    pub profile_digest: Digest,
    /// Pre-computed canonical model digest.
    pub model_digest: Digest,
    /// Pre-computed canonical input hash.
    pub input_hash: Digest,
    /// Maximum allowed atomic steps for `(model, input)`.
    pub max_steps: u64,
}

impl<'a> JudgeContext<'a> {
    /// Validate `(profile, model, input)` once and pin their canonical digests.
    pub fn new(
        profile: &'a ExecutionProfile,
        model: &'a QuantizedModel,
        input: &'a InputCommitment,
    ) -> Result<Self, OpAiError> {
        profile.validate()?;
        model.validate()?;
        let profile_digest = profile.digest();
        let model_digest = model.digest();
        let input_hash = input.input_hash();
        let max_steps = model.total_steps(input.max_new_tokens);
        Ok(Self {
            profile,
            model,
            input,
            profile_digest,
            model_digest,
            input_hash,
            max_steps,
        })
    }
}

/// On-chain-equivalent deterministic single-step judge (`O(1)` execution work
/// independent of layer count `L` and trace length `T`) using a pre-validated [`JudgeContext`].
pub fn verify_single_step_with_ctx(
    ctx: &JudgeContext<'_>,
    total_steps: u64,
    step_index: u64,
    pre_state: &VmState,
    expected_pre_root: &Digest,
    claimed_post_root: &Digest,
) -> Result<SingleStepVerdict, OpAiError> {
    if total_steps == 0 || total_steps > ctx.max_steps {
        return Err(OpAiError::InvalidClaim(format!(
            "total_steps {total_steps} out of range 1..={}",
            ctx.max_steps
        )));
    }

    let op = StepOp::for_step(step_index, ctx.model.layers.len(), total_steps)?;

    if pre_state.validate_for_model(ctx.model).is_err() || pre_state.step_index != step_index {
        return Ok(SingleStepVerdict {
            step_index,
            op_tag: op.op_tag().to_string(),
            pre_root_valid: false,
            recomputed_pre_root: [0u8; 32],
            actual_post_root: [0u8; 32],
            claimed_post_root: *claimed_post_root,
            defender_valid: false,
        });
    }

    let recomputed_pre_root =
        pre_state.state_root(&ctx.profile_digest, &ctx.model_digest, &ctx.input_hash);
    if !ct_eq_digest(&recomputed_pre_root, expected_pre_root) {
        return Ok(SingleStepVerdict {
            step_index,
            op_tag: op.op_tag().to_string(),
            pre_root_valid: false,
            recomputed_pre_root,
            actual_post_root: [0u8; 32],
            claimed_post_root: *claimed_post_root,
            defender_valid: false,
        });
    }

    let post_state = execute_single_step(
        ctx.profile,
        ctx.model,
        ctx.input,
        pre_state,
        step_index,
        total_steps,
    )?;
    let actual_post_root =
        post_state.state_root(&ctx.profile_digest, &ctx.model_digest, &ctx.input_hash);
    let defender_valid = ct_eq_digest(&actual_post_root, claimed_post_root);

    Ok(SingleStepVerdict {
        step_index,
        op_tag: op.op_tag().to_string(),
        pre_root_valid: true,
        recomputed_pre_root,
        actual_post_root,
        claimed_post_root: *claimed_post_root,
        defender_valid,
    })
}

/// On-chain-equivalent deterministic single-step judge (`O(1)` execution work).
///
/// Re-executes **ONLY** single step `step_index -> step_index + 1` from the
/// agreed `pre_state` (verifying that `pre_state.state_root(...) == expected_pre_root`)
/// and checks whether the deterministic post-state hashes to `claimed_post_root`.
#[allow(clippy::too_many_arguments)]
pub fn verify_single_step(
    profile: &ExecutionProfile,
    model: &QuantizedModel,
    input: &InputCommitment,
    total_steps: u64,
    step_index: u64,
    pre_state: &VmState,
    expected_pre_root: &Digest,
    claimed_post_root: &Digest,
) -> Result<SingleStepVerdict, OpAiError> {
    let ctx = JudgeContext::new(profile, model, input)?;
    verify_single_step_with_ctx(
        &ctx,
        total_steps,
        step_index,
        pre_state,
        expected_pre_root,
        claimed_post_root,
    )
}

fn rmsnorm_q12(
    x: &[i32],
    gamma: &[i32],
    profile: &ExecutionProfile,
    out: &mut [i32],
) -> Result<(), OpAiError> {
    let d = x.len() as u128;
    let mut sum_sq = 0u128;
    for &v in x {
        let a = u128::from(v.unsigned_abs());
        sum_sq = sum_sq
            .checked_add(a * a)
            .ok_or(OpAiError::ArithmeticOverflow("rmsnorm_q12::sum_sq"))?;
    }
    // Inputs x are Q12, so sum_sq / d is Q24; adding eps_q24 >= 1 yields rms_q12 >= 1.
    let mean_sq_q24 = (sum_sq / d)
        .checked_add(u128::from(profile.rmsnorm_eps_q24))
        .ok_or(OpAiError::ArithmeticOverflow("rmsnorm_q12::mean_sq"))?;
    let rms_q12 = isqrt_u128(mean_sq_q24).max(1) as i128;

    for j in 0..x.len() {
        let num = i128::from(x[j]) * i128::from(gamma[j]);
        let scaled = round_div_i128(num, rms_q12, profile.rounding_mode)?;
        out[j] = narrow_i32(scaled, profile.overflow_mode)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn int8_matvec_bias(
    input: &[i32],
    weights: &[i8],
    scale: DyadicScale,
    bias: &[i32],
    rows: usize,
    cols: usize,
    profile: &ExecutionProfile,
    out: &mut [i32],
) -> Result<(), OpAiError> {
    for r in 0..rows {
        let row_offset = r * cols;
        let mut acc = 0i128;
        for c in 0..cols {
            acc += i128::from(input[c]) * i128::from(weights[row_offset + c]);
        }
        let rescaled = rescale_i128(acc, scale, profile.rounding_mode)?;
        let narrowed = narrow_i32(rescaled, profile.overflow_mode)?;
        out[r] = add_i32(narrowed, bias[r], profile.overflow_mode)?;
    }
    Ok(())
}
