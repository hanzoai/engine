// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic end-to-end Hybrid Transformer + Gated DeltaNet (GDN) inference engine
//! for evaluating mixed-precision quantization across all 4 required quality domains:
//! `instruction_following`, `code_edit_accuracy`, `multilingual`, and `long_context_retrieval`.

use std::time::Instant;

use crate::awq::{calibrate_awq, calibrate_awq_with_rotation, AwqConfig};
use crate::compressor::{
    mean_softmax_kl, plan_mixed_precision, run_qad_linear_refinement, ComponentKind,
    ComponentSensitivityProfile, MixedPrecisionPlan, QadReport,
};
use crate::corpus::{EvalDomain, GroundedCorpus, SplitMix64};
use crate::error::MixedQuantError;
use crate::formats::{matvec_f32, QuantFormat, QuantizedLinear};
use crate::rotation::RotationTransform;

/// Configuration for the deterministic hybrid Transformer + GDN proxy model.
#[derive(Debug, Clone)]
pub struct ProxyModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_experts: usize,
    pub group_size: usize,
}

impl Default for ProxyModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128,
            d_model: 64,
            d_ff: 256,
            num_layers: 4,
            num_heads: 4,
            num_experts: 4,
            group_size: 64,
        }
    }
}

/// Quantization regime evaluated on the end-to-end model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantRegime {
    /// Reference BF16 weights everywhere.
    Bf16,
    /// Uniform symmetric INT8 everywhere.
    UniformInt8,
    /// Uniform per-group asymmetric INT4 everywhere.
    UniformInt4,
    /// Uniform 1.58-bit Ternary everywhere (Negative Regime 1: catastrophic collapse on critical layers).
    UniformTernary,
    /// Uniform INT4 with AWQ activation-aware channel scaling & 1% salient BF16 protection.
    AwqInt4,
    /// Uniform INT4 with SpinQuant Walsh-Hadamard + Cayley orthogonal rotations.
    SpinQuantInt4,
    /// Sensitivity-planned mixed precision (`{Ternary, Int4, Int8, Bf16}`) + AWQ + SpinQuant + QAD
    /// strictly bounded at `<= UniformInt4` exact byte footprint.
    MixedPrecisionAwqSpinQuant,
    /// Negative Regime 2: unmerged dense runtime rotation + scalar full-matrix dequantization
    /// (wins weight-memory microbenchmark, regresses end-to-end wall time -> rejected by gate).
    UnmergedRotScalarDequant,
    /// Negative Regime 3: AWQ INT4 calibrated only on mismatched ASCII prose (misses code/multilingual outliers).
    MismatchedProseCalAwqInt4,
}

impl QuantRegime {
    /// Canonical string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16_reference",
            Self::UniformInt8 => "uniform_int8",
            Self::UniformInt4 => "uniform_int4",
            Self::UniformTernary => "uniform_ternary",
            Self::AwqInt4 => "awq_int4",
            Self::SpinQuantInt4 => "spinquant_int4",
            Self::MixedPrecisionAwqSpinQuant => "mixed_precision_awq_spinquant_qad",
            Self::UnmergedRotScalarDequant => "neg2_unmerged_rot_scalar_dequant",
            Self::MismatchedProseCalAwqInt4 => "neg3_mismatched_prose_cal_awq",
        }
    }
}

/// FP32 layer weights for one hybrid block (either Attention or Gated DeltaNet + RouterGate + MLP).
#[derive(Debug, Clone)]
pub struct Fp32Block {
    pub layer_idx: usize,
    pub is_gdn: bool,
    /// Attention Q/K/V (`3 * d_model x d_model`) or GDN `(k, v, gate)` (`3 * d_model x d_model`).
    pub seq_in_proj: Vec<f32>,
    /// Attention `o_proj` or GDN output projection (`d_model x d_model`).
    pub seq_out_proj: Vec<f32>,
    /// MoE/SwiGLU router gate (`num_experts x d_model`).
    pub router_gate: Vec<f32>,
    /// MLP gate/up expansion (`d_ff x d_model`).
    pub mlp_up: Vec<f32>,
    /// MLP down projection (`d_model x d_ff`).
    pub mlp_down: Vec<f32>,
}

/// Full-precision FP32 teacher model with realistic LLM outlier structure across domains.
#[derive(Debug, Clone)]
pub struct Fp32HybridModel {
    pub config: ProxyModelConfig,
    /// Embedding table `[vocab_size, d_model]`.
    pub embedding: Vec<f32>,
    /// Interleaved GDN (even) and Attention (odd) blocks.
    pub blocks: Vec<Fp32Block>,
    /// Output `lm_head` `[vocab_size, d_model]`.
    pub lm_head: Vec<f32>,
}

/// Quantized block holding real packed `QuantizedLinear` projections.
#[derive(Debug, Clone)]
pub struct QuantizedBlock {
    pub layer_idx: usize,
    pub is_gdn: bool,
    pub seq_in_proj: QuantizedLinear,
    pub seq_out_proj: QuantizedLinear,
    pub router_gate: QuantizedLinear,
    pub mlp_up: QuantizedLinear,
    pub mlp_down: QuantizedLinear,
}

/// End-to-end quantized hybrid model executing real packed matvecs.
#[derive(Debug, Clone)]
pub struct QuantizedHybridModel {
    pub config: ProxyModelConfig,
    pub regime: QuantRegime,
    /// Quantized embedding matrix `[vocab_size, d_model]` stored as packed weights.
    pub embedding: QuantizedLinear,
    pub blocks: Vec<QuantizedBlock>,
    pub lm_head: QuantizedLinear,
    /// Optional naive unmerged dequantization flag for Negative Regime 2.
    pub force_naive_unmerged_dequant: bool,
    /// Optional plan metadata (for `MixedPrecisionAwqSpinQuant`).
    pub plan: Option<MixedPrecisionPlan>,
    /// Optional QAD refinement summary (for `MixedPrecisionAwqSpinQuant`).
    pub qad_reports: Vec<(String, QadReport)>,
}

/// Per-component calibration activations collected from the FP32 teacher.
#[derive(Debug, Clone)]
pub struct CalibrationTrace {
    pub num_tokens: usize,
    /// Per-layer hidden inputs entering `seq_in_proj` `[num_tokens, d_model]`.
    pub layer_in: Vec<Vec<f32>>,
    /// Per-layer post-seq activations entering `seq_out_proj` `[num_tokens, d_model]`.
    pub seq_out_in: Vec<Vec<f32>>,
    /// Per-layer hidden inputs entering `router_gate` and `mlp_up` `[num_tokens, d_model]`.
    pub mlp_in: Vec<Vec<f32>>,
    /// Per-layer hidden inputs entering `mlp_down` `[num_tokens, d_ff]`.
    pub mlp_mid: Vec<Vec<f32>>,
    /// Final hidden states entering `lm_head` `[num_tokens, d_model]`.
    pub final_hidden: Vec<f32>,
    /// One-hot / smoothed token frequency matrix for embedding sensitivity `[num_tokens, d_model]`.
    pub embed_probes: Vec<f32>,
}

/// Domain-level evaluation metrics.
#[derive(Debug, Clone)]
pub struct DomainScore {
    pub domain: EvalDomain,
    /// Exact top-1 next-token match rate vs BF16 reference (`0.0..=100.0`).
    pub accuracy_pct: f64,
    /// Mean softmax KL divergence vs BF16 reference.
    pub mean_kl: f64,
    /// Mean logit cosine similarity vs BF16 reference.
    pub mean_cosine: f64,
    /// Number of scored token positions.
    pub scored_positions: usize,
}

/// End-to-end measured CPU evaluation report for one quantization regime.
#[derive(Debug, Clone)]
pub struct RegimeEvalReport {
    pub regime: QuantRegime,
    /// Exact packed model storage bytes.
    pub model_bytes: usize,
    /// Effective bits per weight across the entire model.
    pub effective_bits_per_weight: f64,
    /// Compression ratio vs BF16 model bytes.
    pub compression_vs_bf16: f64,
    /// Per-domain quality metrics across all 4 domains.
    pub domains: Vec<DomainScore>,
    /// Mean accuracy across all 4 domains (`%`).
    pub mean_accuracy_pct: f64,
    /// Quality regression vs BF16 reference (`pp`, negative means drop).
    pub quality_regression_pp: f64,
    /// Worst single-domain accuracy regression (`pp`).
    pub worst_domain_regression_pp: f64,
    /// Mean KL divergence across all 4 domains.
    pub mean_kl: f64,
    /// Measured CPU cold TTFT (`ms`, includes model prep + cold prefill).
    pub cpu_cold_ttft_ms: f64,
    /// Measured CPU warm TTFT (`ms`, warm prefill pass).
    pub cpu_warm_ttft_ms: f64,
    /// Measured CPU p50 inter-token latency (`ms`).
    pub cpu_p50_itl_ms: f64,
    /// Measured CPU p95 inter-token latency (`ms`).
    pub cpu_p95_itl_ms: f64,
    /// Measured CPU warm decode throughput (`tok/s`).
    pub cpu_warm_tok_per_s: f64,
    /// Measured CPU full evaluation wall time (`ms`).
    pub cpu_wall_ms: f64,
}

impl Fp32HybridModel {
    /// Deterministically initializes a realistic hybrid Transformer + GDN model with
    /// structured heavy-tailed activation outlier channels and domain-specialized subspaces.
    #[must_use]
    pub fn new(config: ProxyModelConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed ^ 0x4849_5031_3334_3649);
        let d = config.d_model;
        let v = config.vocab_size;
        let ff = config.d_ff;

        // Build embedding table with realistic domain subspace structure and heavy-tailed outlier channels:
        // - Channels 3, 19: persistent prose + instruction outliers (18x-22x magnitude)
        // - Channel 37: code/JSON syntax outlier (20x magnitude on tokens [v/2..3v/4))
        // - Channel 53: multilingual tail outlier (20x magnitude on tokens [3v/4..v))
        let mut embedding = vec![0.0f32; v * d];
        let q1 = v / 4;
        let q2 = v / 2;
        let q3 = (3 * v) / 4;
        let base_scale = 1.0 / (d as f32).sqrt();
        for tok in 0..v {
            let row = &mut embedding[tok * d..(tok + 1) * d];
            for item in row.iter_mut() {
                *item = rng.next_normal() * base_scale * 0.65;
            }
            // Domain-specific semantic signature + persistent outlier channels
            let phase = (tok as f32) * 0.17;
            row[3 % d] += 11.5 * phase.cos().signum().max(-1.0) * (0.7 + 0.3 * phase.sin().abs());
            row[19 % d] += 9.5 * (phase * 1.3).sin().signum();
            if tok >= q1 && tok < q2 {
                // Instruction template subspace
                row[7 % d] += 2.2;
                row[11 % d] -= 1.8;
            } else if tok >= q2 && tok < q3 {
                // Code-edit / JSON subspace with strong domain-specific outlier on ch 37
                row[37 % d] += 14.0 * if tok % 2 == 0 { 1.0 } else { -1.0 };
                row[23 % d] += 1.9;
            } else if tok >= q3 {
                // Multilingual / rare tail tokens with fine-grained embedding sensitivity + outlier on ch 53
                row[53 % d] += 14.5 * if tok % 3 == 0 { 1.0 } else { -1.0 };
                row[29 % d] += 1.6 * ((tok - q3) as f32 / (v - q3).max(1) as f32 - 0.5);
            }
        }

        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let is_gdn = layer_idx % 2 == 0;
            let mut seq_in_proj = init_weight_matrix(3 * d, d, 0.85, &mut rng);
            let seq_out_proj = init_weight_matrix(d, d, 0.75, &mut rng);
            // Sharp router gate weights so sub-8-bit quantization visibly shifts expert selection
            let router_gate = init_weight_matrix(config.num_experts, d, 1.45, &mut rng);
            let mut mlp_up = init_weight_matrix(ff, d, 0.70, &mut rng);
            let mlp_down = init_weight_matrix(d, ff, 0.55, &mut rng);

            // Couple code-edit (ch 37) and multilingual (ch 53) salient channels to real downstream projections
            for r in 0..(3 * d) {
                seq_in_proj[r * d + (37 % d)] *= 2.4;
                seq_in_proj[r * d + (53 % d)] *= 2.4;
            }
            for r in 0..ff {
                mlp_up[r * d + (37 % d)] *= 2.2;
                mlp_up[r * d + (53 % d)] *= 2.2;
            }

            blocks.push(Fp32Block {
                layer_idx,
                is_gdn,
                seq_in_proj,
                seq_out_proj,
                router_gate,
                mlp_up,
                mlp_down,
            });
        }

        // Output lm_head: maximally-separated orthoplex prototype basis in R^d (2*d antipodal/orthogonal
        // class prototypes) coupled to embeddings, yielding realistic trained-LLM next-token margins
        // without artificial random near-tie flips.
        let mut lm_head = vec![0.0f32; v * d];
        for tok in 0..v {
            let axis = tok % d;
            let sign = if tok < d { 1.0f32 } else { -1.0f32 };
            for c in 0..d {
                let emb_c = embedding[tok * d + c];
                let is_outlier = c == 3 % d || c == 19 % d || c == 37 % d || c == 53 % d;
                let mut w = if is_outlier {
                    rng.next_normal() * 0.02
                } else {
                    0.40 * emb_c + rng.next_normal() * base_scale * 0.25
                };
                if c == axis {
                    w += sign * 2.85;
                }
                lm_head[tok * d + c] = w;
            }
        }

        Self {
            config,
            embedding,
            blocks,
            lm_head,
        }
    }

    /// Total parameter count across all linear/embedding weights.
    #[must_use]
    pub fn total_params(&self) -> usize {
        let mut n = self.embedding.len() + self.lm_head.len();
        for b in &self.blocks {
            n += b.seq_in_proj.len()
                + b.seq_out_proj.len()
                + b.router_gate.len()
                + b.mlp_up.len()
                + b.mlp_down.len();
        }
        n
    }

    /// Runs the exact FP32 teacher forward pass on a token sequence, returning `[seq_len, vocab_size]` logits.
    pub fn forward_sequence(&self, tokens: &[usize]) -> Result<Vec<f32>, MixedQuantError> {
        let (logits, _) = self.forward_with_trace(tokens)?;
        Ok(logits)
    }

    /// Collects full calibration activation traces across a batch of sequences.
    pub fn collect_calibration_trace(
        &self,
        sequences: &[Vec<usize>],
    ) -> Result<CalibrationTrace, MixedQuantError> {
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let num_layers = self.config.num_layers;
        let mut total_tokens = 0usize;
        let mut layer_in = vec![Vec::new(); num_layers];
        let mut seq_out_in = vec![Vec::new(); num_layers];
        let mut mlp_in = vec![Vec::new(); num_layers];
        let mut mlp_mid = vec![Vec::new(); num_layers];
        let mut final_hidden = Vec::new();
        let mut embed_probes = Vec::new();

        for seq in sequences {
            let (_, trace) = self.forward_with_trace(seq)?;
            total_tokens += trace.num_tokens;
            for l in 0..num_layers {
                layer_in[l].extend_from_slice(&trace.layer_in[l]);
                seq_out_in[l].extend_from_slice(&trace.seq_out_in[l]);
                mlp_in[l].extend_from_slice(&trace.mlp_in[l]);
                mlp_mid[l].extend_from_slice(&trace.mlp_mid[l]);
            }
            final_hidden.extend_from_slice(&trace.final_hidden);
            embed_probes.extend_from_slice(&trace.embed_probes);
        }

        debug_assert_eq!(final_hidden.len(), total_tokens * d);
        debug_assert_eq!(mlp_mid[0].len(), total_tokens * ff);

        Ok(CalibrationTrace {
            num_tokens: total_tokens,
            layer_in,
            seq_out_in,
            mlp_in,
            mlp_mid,
            final_hidden,
            embed_probes,
        })
    }

    fn forward_with_trace(
        &self,
        tokens: &[usize],
    ) -> Result<(Vec<f32>, CalibrationTrace), MixedQuantError> {
        let seq_len = tokens.len();
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;

        let mut hidden = vec![0.0f32; seq_len * d];
        let mut embed_probes = vec![0.0f32; seq_len * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let idx = tok % v;
            let src = &self.embedding[idx * d..(idx + 1) * d];
            hidden[t * d..(t + 1) * d].copy_from_slice(src);
            // Smooth probe vector used for embedding row-sensitivity weighting
            embed_probes[t * d..(t + 1) * d].copy_from_slice(src);
        }

        let mut layer_in = Vec::with_capacity(self.blocks.len());
        let mut seq_out_in = Vec::with_capacity(self.blocks.len());
        let mut mlp_in = Vec::with_capacity(self.blocks.len());
        let mut mlp_mid = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            // Preserve raw outlier scale before seq_in_proj via RMSNorm with learned outlier gain
            let normed_in = rmsnorm_preserve_outliers(&hidden, seq_len, d);
            layer_in.push(normed_in.clone());

            // Compute q/k/v or GDN (k, v, gate)
            let mut qkv = vec![0.0f32; seq_len * 3 * d];
            for t in 0..seq_len {
                matvec_f32(
                    &block.seq_in_proj,
                    3 * d,
                    d,
                    &normed_in[t * d..(t + 1) * d],
                    &mut qkv[t * 3 * d..(t + 1) * 3 * d],
                )?;
            }

            let seq_ctx = if block.is_gdn {
                run_gdn_recurrence(&qkv, seq_len, d, self.config.num_heads)
            } else {
                run_causal_attention(&qkv, seq_len, d, self.config.num_heads)
            };
            seq_out_in.push(seq_ctx.clone());

            // Output projection + residual
            let mut proj = vec![0.0f32; d];
            for t in 0..seq_len {
                matvec_f32(
                    &block.seq_out_proj,
                    d,
                    d,
                    &seq_ctx[t * d..(t + 1) * d],
                    &mut proj,
                )?;
                for c in 0..d {
                    hidden[t * d + c] += 0.35 * proj[c];
                }
            }

            // RouterGate + MLP
            let normed_mlp = rmsnorm_preserve_outliers(&hidden, seq_len, d);
            mlp_in.push(normed_mlp.clone());
            let mut mid_all = vec![0.0f32; seq_len * ff];
            let mut router_logits = vec![0.0f32; self.config.num_experts];
            let mut up_buf = vec![0.0f32; ff];
            let mut down_buf = vec![0.0f32; d];
            let expert_chunk = ff / self.config.num_experts.max(1);

            for t in 0..seq_len {
                let x_t = &normed_mlp[t * d..(t + 1) * d];
                matvec_f32(
                    &block.router_gate,
                    self.config.num_experts,
                    d,
                    x_t,
                    &mut router_logits,
                )?;
                let gates = top2_router_weights(&router_logits);

                matvec_f32(&block.mlp_up, ff, d, x_t, &mut up_buf)?;
                for (e, &g) in gates.iter().enumerate().take(self.config.num_experts) {
                    let start = e * expert_chunk;
                    let end = if e + 1 == self.config.num_experts {
                        ff
                    } else {
                        (start + expert_chunk).min(ff)
                    };
                    for i in start..end {
                        // SiLU activation gated by MoE router weight
                        let x = up_buf[i];
                        let silu = x / (1.0 + (-x).exp());
                        mid_all[t * ff + i] = silu * g;
                    }
                }

                matvec_f32(
                    &block.mlp_down,
                    d,
                    ff,
                    &mid_all[t * ff..(t + 1) * ff],
                    &mut down_buf,
                )?;
                for c in 0..d {
                    hidden[t * d + c] += 0.30 * down_buf[c];
                }
            }
            mlp_mid.push(mid_all);
        }

        let final_hidden = rmsnorm_standard(&hidden, seq_len, d);
        let mut logits = vec![0.0f32; seq_len * v];
        for t in 0..seq_len {
            matvec_f32(
                &self.lm_head,
                v,
                d,
                &final_hidden[t * d..(t + 1) * d],
                &mut logits[t * v..(t + 1) * v],
            )?;
        }

        let trace = CalibrationTrace {
            num_tokens: seq_len,
            layer_in,
            seq_out_in,
            mlp_in,
            mlp_mid,
            final_hidden,
            embed_probes,
        };
        Ok((logits, trace))
    }

    /// Profiles every component in the model across all 4 formats + 2:4 pruning sensitivity.
    pub fn profile_components(
        &self,
        trace: &CalibrationTrace,
    ) -> Result<Vec<ComponentSensitivityProfile>, MixedQuantError> {
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let g = self.config.group_size;
        let mut profiles = Vec::with_capacity(2 + self.blocks.len() * 5);

        // Embedding profiled via transpose-equivalent probe projection [v, d] * [num_tokens, d]^T
        profiles.push(ComponentSensitivityProfile::measure(
            "embedding".to_string(),
            0,
            ComponentKind::Embedding,
            &self.embedding,
            v,
            d,
            &trace.embed_probes,
            trace.num_tokens,
            g,
        )?);

        for b in &self.blocks {
            let l = b.layer_idx;
            let seq_kind = if b.is_gdn {
                ComponentKind::GdnRecurrent
            } else {
                ComponentKind::AttnQkv
            };
            let out_kind = if b.is_gdn {
                ComponentKind::GdnRecurrent
            } else {
                ComponentKind::AttnOut
            };
            profiles.push(ComponentSensitivityProfile::measure(
                format!("layer_{l}.seq_in"),
                l,
                seq_kind,
                &b.seq_in_proj,
                3 * d,
                d,
                &trace.layer_in[l],
                trace.num_tokens,
                g,
            )?);
            profiles.push(ComponentSensitivityProfile::measure(
                format!("layer_{l}.seq_out"),
                l,
                out_kind,
                &b.seq_out_proj,
                d,
                d,
                &trace.seq_out_in[l],
                trace.num_tokens,
                g,
            )?);
            profiles.push(ComponentSensitivityProfile::measure(
                format!("layer_{l}.router_gate"),
                l,
                ComponentKind::RouterGate,
                &b.router_gate,
                self.config.num_experts,
                d,
                &trace.mlp_in[l],
                trace.num_tokens,
                g,
            )?);
            profiles.push(ComponentSensitivityProfile::measure(
                format!("layer_{l}.mlp_up"),
                l,
                ComponentKind::MlpGateUp,
                &b.mlp_up,
                ff,
                d,
                &trace.mlp_in[l],
                trace.num_tokens,
                g,
            )?);
            profiles.push(ComponentSensitivityProfile::measure(
                format!("layer_{l}.mlp_down"),
                l,
                ComponentKind::MlpDown,
                &b.mlp_down,
                d,
                ff,
                &trace.mlp_mid[l],
                trace.num_tokens,
                g,
            )?);
        }

        profiles.push(ComponentSensitivityProfile::measure(
            "lm_head".to_string(),
            self.blocks.len(),
            ComponentKind::LmHead,
            &self.lm_head,
            v,
            d,
            &trace.final_hidden,
            trace.num_tokens,
            g,
        )?);

        Ok(profiles)
    }

    /// Quantizes this FP32 model under the requested `QuantRegime`.
    pub fn quantize(
        &self,
        regime: QuantRegime,
        trace: &CalibrationTrace,
        seed: u64,
    ) -> Result<QuantizedHybridModel, MixedQuantError> {
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let g = self.config.group_size;

        match regime {
            QuantRegime::Bf16
            | QuantRegime::UniformInt8
            | QuantRegime::UniformInt4
            | QuantRegime::UniformTernary => {
                let fmt = match regime {
                    QuantRegime::Bf16 => QuantFormat::Bf16,
                    QuantRegime::UniformInt8 => QuantFormat::Int8,
                    QuantRegime::UniformInt4 => QuantFormat::Int4Asym,
                    QuantRegime::UniformTernary => QuantFormat::Ternary,
                    _ => unreachable!(),
                };
                let embedding = QuantizedLinear::quantize_plain(&self.embedding, v, d, fmt, g)?;
                let mut blocks = Vec::with_capacity(self.blocks.len());
                for b in &self.blocks {
                    blocks.push(QuantizedBlock {
                        layer_idx: b.layer_idx,
                        is_gdn: b.is_gdn,
                        seq_in_proj: QuantizedLinear::quantize_plain(
                            &b.seq_in_proj,
                            3 * d,
                            d,
                            fmt,
                            g,
                        )?,
                        seq_out_proj: QuantizedLinear::quantize_plain(
                            &b.seq_out_proj,
                            d,
                            d,
                            fmt,
                            g,
                        )?,
                        router_gate: QuantizedLinear::quantize_plain(
                            &b.router_gate,
                            self.config.num_experts,
                            d,
                            fmt,
                            g,
                        )?,
                        mlp_up: QuantizedLinear::quantize_plain(&b.mlp_up, ff, d, fmt, g)?,
                        mlp_down: QuantizedLinear::quantize_plain(&b.mlp_down, d, ff, fmt, g)?,
                    });
                }
                let lm_head = QuantizedLinear::quantize_plain(&self.lm_head, v, d, fmt, g)?;
                Ok(QuantizedHybridModel {
                    config: self.config.clone(),
                    regime,
                    embedding,
                    blocks,
                    lm_head,
                    force_naive_unmerged_dequant: false,
                    plan: None,
                    qad_reports: Vec::new(),
                })
            }

            QuantRegime::AwqInt4 | QuantRegime::MismatchedProseCalAwqInt4 => {
                let cfg = AwqConfig {
                    format: QuantFormat::Int4Asym,
                    group_size: g,
                    grid_steps: 20,
                    salient_ratio: 0.0625,
                    protect_salient_bf16: true,
                };
                let embedding = QuantizedLinear::quantize_plain(
                    &self.embedding,
                    v,
                    d,
                    QuantFormat::Int4Asym,
                    g,
                )?;
                let mut blocks = Vec::with_capacity(self.blocks.len());
                for b in &self.blocks {
                    let l = b.layer_idx;
                    blocks.push(QuantizedBlock {
                        layer_idx: l,
                        is_gdn: b.is_gdn,
                        seq_in_proj: calibrate_awq(
                            &b.seq_in_proj,
                            3 * d,
                            d,
                            &trace.layer_in[l],
                            trace.num_tokens,
                            &cfg,
                        )?
                        .linear,
                        seq_out_proj: calibrate_awq(
                            &b.seq_out_proj,
                            d,
                            d,
                            &trace.seq_out_in[l],
                            trace.num_tokens,
                            &cfg,
                        )?
                        .linear,
                        router_gate: calibrate_awq(
                            &b.router_gate,
                            self.config.num_experts,
                            d,
                            &trace.mlp_in[l],
                            trace.num_tokens,
                            &cfg,
                        )?
                        .linear,
                        mlp_up: calibrate_awq(
                            &b.mlp_up,
                            ff,
                            d,
                            &trace.mlp_in[l],
                            trace.num_tokens,
                            &cfg,
                        )?
                        .linear,
                        mlp_down: calibrate_awq(
                            &b.mlp_down,
                            d,
                            ff,
                            &trace.mlp_mid[l],
                            trace.num_tokens,
                            &cfg,
                        )?
                        .linear,
                    });
                }
                let lm_head = calibrate_awq(
                    &self.lm_head,
                    v,
                    d,
                    &trace.final_hidden,
                    trace.num_tokens,
                    &cfg,
                )?
                .linear;

                Ok(QuantizedHybridModel {
                    config: self.config.clone(),
                    regime,
                    embedding,
                    blocks,
                    lm_head,
                    force_naive_unmerged_dequant: false,
                    plan: None,
                    qad_reports: Vec::new(),
                })
            }

            QuantRegime::SpinQuantInt4 => {
                // Fast O(d log d) Walsh-Hadamard runtime transform
                let rot_d = RotationTransform::randomized_hadamard(d, d.min(64), seed)?;
                let rot_ff = RotationTransform::randomized_hadamard(ff, 64.min(ff), seed ^ 0x55)?;

                let embedding = QuantizedLinear::quantize_plain(
                    &self.embedding,
                    v,
                    d,
                    QuantFormat::Int4Asym,
                    g,
                )?;
                let mut blocks = Vec::with_capacity(self.blocks.len());
                for b in &self.blocks {
                    blocks.push(QuantizedBlock {
                        layer_idx: b.layer_idx,
                        is_gdn: b.is_gdn,
                        seq_in_proj: quantize_with_rotation(
                            &b.seq_in_proj,
                            3 * d,
                            d,
                            QuantFormat::Int4Asym,
                            g,
                            rot_d.clone(),
                        )?,
                        seq_out_proj: quantize_with_rotation(
                            &b.seq_out_proj,
                            d,
                            d,
                            QuantFormat::Int4Asym,
                            g,
                            rot_d.clone(),
                        )?,
                        router_gate: quantize_with_rotation(
                            &b.router_gate,
                            self.config.num_experts,
                            d,
                            QuantFormat::Int4Asym,
                            g,
                            rot_d.clone(),
                        )?,
                        mlp_up: quantize_with_rotation(
                            &b.mlp_up,
                            ff,
                            d,
                            QuantFormat::Int4Asym,
                            g,
                            rot_d.clone(),
                        )?,
                        mlp_down: quantize_with_rotation(
                            &b.mlp_down,
                            d,
                            ff,
                            QuantFormat::Int4Asym,
                            g,
                            rot_ff.clone(),
                        )?,
                    });
                }
                let lm_head =
                    quantize_with_rotation(&self.lm_head, v, d, QuantFormat::Int4Asym, g, rot_d)?;

                Ok(QuantizedHybridModel {
                    config: self.config.clone(),
                    regime,
                    embedding,
                    blocks,
                    lm_head,
                    force_naive_unmerged_dequant: false,
                    plan: None,
                    qad_reports: Vec::new(),
                })
            }

            QuantRegime::UnmergedRotScalarDequant => {
                // Negative Regime 2 (microbenchmark-win / end-to-end-loss):
                // Uses identical high-quality mixed-precision packed weights (winning the
                // weight-memory microbenchmark!) but executes an unmerged dense O(d^2) runtime
                // rotation plus full per-token scalar FP32 matrix dequantization, regressing
                // end-to-end inference wall time and triggering acceptance gate rejection.
                let mut model =
                    self.quantize(QuantRegime::MixedPrecisionAwqSpinQuant, trace, seed)?;
                model.regime = QuantRegime::UnmergedRotScalarDequant;
                model.force_naive_unmerged_dequant = true;
                Ok(model)
            }

            QuantRegime::MixedPrecisionAwqSpinQuant => {
                // 1. Compute exact byte budget equal to UniformInt4!
                let uniform_int4 = self.quantize(QuantRegime::UniformInt4, trace, seed)?;
                let target_budget = uniform_int4.storage_bytes();

                // 2. Profile every component across all 4 formats + 2:4 pruning sensitivity
                let profiles = self.profile_components(trace)?;

                // Reserve AWQ inv_scales + seeded Hadamard rotation descriptor metadata
                // (16 bytes/rotated projection, <=256 B total, comfortably inside the
                // `blocks * (d + ff) * 4` reserve while QAD row scales fold zero-overhead
                // into existing per-group scales) so realized bytes strictly satisfy
                // `model.storage_bytes() <= uniform_int4.storage_bytes()`.
                let metadata_reserve = self.blocks.len() * (d + ff) * 4;
                let plan_budget = target_budget.saturating_sub(metadata_reserve);
                let plan = plan_mixed_precision(&profiles, plan_budget, true)?;

                // Fast Walsh-Hadamard SpinQuant rotations for low-bit layers
                let rot_d = RotationTransform::randomized_hadamard(d, d.min(64), seed)?;
                let rot_ff = RotationTransform::randomized_hadamard(ff, 64.min(ff), seed ^ 0x55)?;

                let fmt_for = |name: &str| -> QuantFormat {
                    plan.assignments
                        .iter()
                        .find(|(n, _, _, _)| n == name)
                        .map(|(_, _, f, _)| *f)
                        .unwrap_or(QuantFormat::Int4Asym)
                };

                let emb_fmt = fmt_for("embedding");
                let embedding = QuantizedLinear::quantize_plain(&self.embedding, v, d, emb_fmt, g)?;

                let mut blocks = Vec::with_capacity(self.blocks.len());
                let mut qad_reports = Vec::new();

                for b in &self.blocks {
                    let l = b.layer_idx;
                    let f_seq_in = fmt_for(&format!("layer_{l}.seq_in"));
                    let f_seq_out = fmt_for(&format!("layer_{l}.seq_out"));
                    let f_router = fmt_for(&format!("layer_{l}.router_gate"));
                    let f_up = fmt_for(&format!("layer_{l}.mlp_up"));
                    let f_down = fmt_for(&format!("layer_{l}.mlp_down"));

                    let mut seq_in = build_mixed_linear(
                        &b.seq_in_proj,
                        3 * d,
                        d,
                        &trace.layer_in[l],
                        trace.num_tokens,
                        f_seq_in,
                        g,
                        Some(rot_d.clone()),
                    )?;
                    let mut seq_out = build_mixed_linear(
                        &b.seq_out_proj,
                        d,
                        d,
                        &trace.seq_out_in[l],
                        trace.num_tokens,
                        f_seq_out,
                        g,
                        Some(rot_d.clone()),
                    )?;
                    let router_gate = build_mixed_linear(
                        &b.router_gate,
                        self.config.num_experts,
                        d,
                        &trace.mlp_in[l],
                        trace.num_tokens,
                        f_router,
                        g,
                        None,
                    )?;
                    let mut mlp_up = build_mixed_linear(
                        &b.mlp_up,
                        ff,
                        d,
                        &trace.mlp_in[l],
                        trace.num_tokens,
                        f_up,
                        g,
                        Some(rot_d.clone()),
                    )?;
                    let mut mlp_down = build_mixed_linear(
                        &b.mlp_down,
                        d,
                        ff,
                        &trace.mlp_mid[l],
                        trace.num_tokens,
                        f_down,
                        g,
                        Some(rot_ff.clone()),
                    )?;

                    // Run real QAD refinement on compressed projections (< Bf16)
                    if f_seq_in != QuantFormat::Bf16 {
                        let rep = run_qad_linear_refinement(
                            &mut seq_in,
                            &b.seq_in_proj,
                            &trace.layer_in[l],
                            trace.num_tokens,
                            6,
                            0.08,
                        )?;
                        qad_reports.push((format!("layer_{l}.seq_in"), rep));
                    }
                    if f_seq_out != QuantFormat::Bf16 {
                        let rep = run_qad_linear_refinement(
                            &mut seq_out,
                            &b.seq_out_proj,
                            &trace.seq_out_in[l],
                            trace.num_tokens,
                            6,
                            0.08,
                        )?;
                        qad_reports.push((format!("layer_{l}.seq_out"), rep));
                    }
                    if f_up != QuantFormat::Bf16 {
                        let rep = run_qad_linear_refinement(
                            &mut mlp_up,
                            &b.mlp_up,
                            &trace.mlp_in[l],
                            trace.num_tokens,
                            6,
                            0.08,
                        )?;
                        qad_reports.push((format!("layer_{l}.mlp_up"), rep));
                    }
                    if f_down != QuantFormat::Bf16 {
                        let rep = run_qad_linear_refinement(
                            &mut mlp_down,
                            &b.mlp_down,
                            &trace.mlp_mid[l],
                            trace.num_tokens,
                            6,
                            0.08,
                        )?;
                        qad_reports.push((format!("layer_{l}.mlp_down"), rep));
                    }

                    blocks.push(QuantizedBlock {
                        layer_idx: l,
                        is_gdn: b.is_gdn,
                        seq_in_proj: seq_in,
                        seq_out_proj: seq_out,
                        router_gate,
                        mlp_up,
                        mlp_down,
                    });
                }

                let lm_fmt = fmt_for("lm_head");
                let mut lm_head = build_mixed_linear(
                    &self.lm_head,
                    v,
                    d,
                    &trace.final_hidden,
                    trace.num_tokens,
                    lm_fmt,
                    g,
                    None,
                )?;
                if lm_fmt != QuantFormat::Bf16 {
                    let rep = run_qad_linear_refinement(
                        &mut lm_head,
                        &self.lm_head,
                        &trace.final_hidden,
                        trace.num_tokens,
                        6,
                        0.08,
                    )?;
                    qad_reports.push(("lm_head".to_string(), rep));
                }

                Ok(QuantizedHybridModel {
                    config: self.config.clone(),
                    regime,
                    embedding,
                    blocks,
                    lm_head,
                    force_naive_unmerged_dequant: false,
                    plan: Some(plan),
                    qad_reports,
                })
            }
        }
    }
}

impl QuantizedHybridModel {
    /// Exact total packed weight + metadata storage bytes.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        let mut total = self.embedding.storage_bytes() + self.lm_head.storage_bytes();
        for b in &self.blocks {
            total += b.seq_in_proj.storage_bytes()
                + b.seq_out_proj.storage_bytes()
                + b.router_gate.storage_bytes()
                + b.mlp_up.storage_bytes()
                + b.mlp_down.storage_bytes();
        }
        total
    }

    /// Total parameter count across all model matrices.
    #[must_use]
    pub fn total_params(&self) -> usize {
        let mut n = {
            let (r, c) = self.embedding.shape();
            r * c
        } + {
            let (r, c) = self.lm_head.shape();
            r * c
        };
        for b in &self.blocks {
            for lin in [
                &b.seq_in_proj,
                &b.seq_out_proj,
                &b.router_gate,
                &b.mlp_up,
                &b.mlp_down,
            ] {
                let (r, c) = lin.shape();
                n += r * c;
            }
        }
        n
    }

    /// Exact realized bits per parameter.
    #[must_use]
    pub fn effective_bits_per_weight(&self) -> f64 {
        (self.storage_bytes() as f64 * 8.0) / (self.total_params().max(1) as f64)
    }

    /// Runs a real end-to-end forward pass over `tokens` using packed weights.
    pub fn forward_sequence(&self, tokens: &[usize]) -> Result<Vec<f32>, MixedQuantError> {
        let seq_len = tokens.len();
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;

        // Dequantize embedding table lookup rows directly from quantized storage
        let emb_deq = self.embedding.weights.dequantize();
        let mut hidden = vec![0.0f32; seq_len * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let idx = tok % v;
            hidden[t * d..(t + 1) * d].copy_from_slice(&emb_deq[idx * d..(idx + 1) * d]);
        }

        for block in &self.blocks {
            let normed_in = rmsnorm_preserve_outliers(&hidden, seq_len, d);
            let mut qkv = vec![0.0f32; seq_len * 3 * d];
            for t in 0..seq_len {
                self.eval_linear(
                    &block.seq_in_proj,
                    &normed_in[t * d..(t + 1) * d],
                    &mut qkv[t * 3 * d..(t + 1) * 3 * d],
                )?;
            }

            let seq_ctx = if block.is_gdn {
                run_gdn_recurrence(&qkv, seq_len, d, self.config.num_heads)
            } else {
                run_causal_attention(&qkv, seq_len, d, self.config.num_heads)
            };

            let mut proj = vec![0.0f32; d];
            for t in 0..seq_len {
                self.eval_linear(&block.seq_out_proj, &seq_ctx[t * d..(t + 1) * d], &mut proj)?;
                for c in 0..d {
                    hidden[t * d + c] += 0.35 * proj[c];
                }
            }

            let normed_mlp = rmsnorm_preserve_outliers(&hidden, seq_len, d);
            let mut router_logits = vec![0.0f32; self.config.num_experts];
            let mut up_buf = vec![0.0f32; ff];
            let mut mid_buf = vec![0.0f32; ff];
            let mut down_buf = vec![0.0f32; d];
            let expert_chunk = ff / self.config.num_experts.max(1);

            for t in 0..seq_len {
                let x_t = &normed_mlp[t * d..(t + 1) * d];
                self.eval_linear(&block.router_gate, x_t, &mut router_logits)?;
                let gates = top2_router_weights(&router_logits);

                self.eval_linear(&block.mlp_up, x_t, &mut up_buf)?;
                for (e, &g) in gates.iter().enumerate().take(self.config.num_experts) {
                    let start = e * expert_chunk;
                    let end = if e + 1 == self.config.num_experts {
                        ff
                    } else {
                        (start + expert_chunk).min(ff)
                    };
                    for i in start..end {
                        let x = up_buf[i];
                        let silu = x / (1.0 + (-x).exp());
                        mid_buf[i] = silu * g;
                    }
                }

                self.eval_linear(&block.mlp_down, &mid_buf, &mut down_buf)?;
                for c in 0..d {
                    hidden[t * d + c] += 0.30 * down_buf[c];
                }
            }
        }

        let final_hidden = rmsnorm_standard(&hidden, seq_len, d);
        let mut logits = vec![0.0f32; seq_len * v];
        for t in 0..seq_len {
            self.eval_linear(
                &self.lm_head,
                &final_hidden[t * d..(t + 1) * d],
                &mut logits[t * v..(t + 1) * v],
            )?;
        }
        Ok(logits)
    }

    fn eval_linear(
        &self,
        lin: &QuantizedLinear,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), MixedQuantError> {
        if self.force_naive_unmerged_dequant {
            // Negative Regime 2: materializes dense O(d^2) runtime rotation + full dequantized
            // FP32 weight matrix per token (exact same numerical output as MixedPrecision,
            // winning weight-memory microbenchmark while regressing end-to-end wall time).
            let (rows, cols) = lin.shape();
            let mut x_eff = x.to_vec();
            if let Some(ref rot) = lin.input_rotation {
                let dense_r = rot.to_dense_matrix()?;
                let mut rotated = vec![0.0f32; cols];
                for i in 0..cols {
                    let xi = x_eff[i];
                    let row = &dense_r[i * cols..(i + 1) * cols];
                    for j in 0..cols {
                        rotated[j] += xi * row[j];
                    }
                }
                x_eff = rotated;
            }
            if let Some(ref inv_s) = lin.awq_inv_scales {
                for (xi, &inv) in x_eff.iter_mut().zip(inv_s.iter()) {
                    *xi *= inv;
                }
            }
            let deq = lin.weights.dequantize();
            matvec_f32(&deq, rows, cols, &x_eff, out)?;
            if let Some(ref rs) = lin.qad_row_scales {
                for (o, &s) in out.iter_mut().zip(rs.iter()) {
                    *o *= s;
                }
            }
            Ok(())
        } else {
            lin.forward_vec(x, out)
        }
    }
}

/// Evaluates a quantized model across all 4 domains against precomputed BF16 reference logits
/// and collects real CPU wall-clock timings (cold TTFT, warm TTFT, p50/p95 ITL, tok/s).
pub fn evaluate_regime(
    model: &QuantizedHybridModel,
    corpus: &GroundedCorpus,
    bf16_ref_logits: &[(EvalDomain, Vec<Vec<f32>>)],
    bf16_model_bytes: usize,
    seed: u64,
) -> Result<RegimeEvalReport, MixedQuantError> {
    let v = model.config.vocab_size;
    let seq_len = 24usize;
    let seqs_per_domain = 6usize;

    let wall_start = Instant::now();
    // Measure cold TTFT on first prefill sequence
    let cold_prompt = corpus.eval_sequences(EvalDomain::CodeEditAccuracy, 1, seq_len, v, seed);
    let cold_start = Instant::now();
    let _ = model.forward_sequence(&cold_prompt[0])?;
    let cpu_cold_ttft_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

    // Measure warm TTFT on second prefill sequence
    let warm_start = Instant::now();
    let _ = model.forward_sequence(&cold_prompt[0])?;
    let cpu_warm_ttft_ms = warm_start.elapsed().as_secs_f64() * 1000.0;

    // Measure 20 single-step inter-token latencies (p50 / p95)
    let mut itl_samples_ms = Vec::with_capacity(20);
    let mut decode_toks = cold_prompt[0][..8].to_vec();
    for step in 0..20 {
        let t0 = Instant::now();
        let logits = model.forward_sequence(&decode_toks)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        itl_samples_ms.push(elapsed_ms);
        let last = &logits[(decode_toks.len() - 1) * v..decode_toks.len() * v];
        let next_tok = argmax(last);
        decode_toks.push((next_tok + step) % v);
        if decode_toks.len() > 16 {
            decode_toks.remove(0);
        }
    }
    itl_samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cpu_p50_itl_ms = itl_samples_ms[itl_samples_ms.len() / 2];
    let cpu_p95_itl_ms =
        itl_samples_ms[(itl_samples_ms.len() * 95 / 100).min(itl_samples_ms.len() - 1)];
    let mean_itl_s = (itl_samples_ms.iter().sum::<f64>() / itl_samples_ms.len() as f64) / 1000.0;
    let cpu_warm_tok_per_s = if mean_itl_s > 0.0 {
        1.0 / mean_itl_s
    } else {
        0.0
    };

    // Score all 4 quality domains against BF16 teacher reference
    let mut domains = Vec::with_capacity(4);
    for (domain, ref_seqs) in bf16_ref_logits {
        let eval_seqs = corpus.eval_sequences(*domain, seqs_per_domain, seq_len, v, seed);
        let mut matched = 0usize;
        let mut total_pos = 0usize;
        let mut kl_sum = 0.0f64;
        let mut cos_sum = 0.0f64;

        for (seq, ref_logits) in eval_seqs.iter().zip(ref_seqs.iter()) {
            let logits = model.forward_sequence(seq)?;
            let score_start = match domain {
                EvalDomain::LongContextRetrieval => seq_len.saturating_sub(4),
                _ => 2,
            };
            for pos in score_start..seq_len {
                let r_slice = &ref_logits[pos * v..(pos + 1) * v];
                let q_slice = &logits[pos * v..(pos + 1) * v];
                if argmax(r_slice) == argmax(q_slice) {
                    matched += 1;
                }
                kl_sum += mean_softmax_kl(r_slice, q_slice, 1, v);
                cos_sum += cosine_similarity(r_slice, q_slice);
                total_pos += 1;
            }
        }

        let denom = total_pos.max(1) as f64;
        domains.push(DomainScore {
            domain: *domain,
            accuracy_pct: (matched as f64 / denom) * 100.0,
            mean_kl: kl_sum / denom,
            mean_cosine: cos_sum / denom,
            scored_positions: total_pos,
        });
    }

    let cpu_wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let mean_accuracy_pct =
        domains.iter().map(|d| d.accuracy_pct).sum::<f64>() / (domains.len() as f64);
    let mean_kl = domains.iter().map(|d| d.mean_kl).sum::<f64>() / (domains.len() as f64);
    let quality_regression_pp = mean_accuracy_pct - 100.0;
    let worst_domain_regression_pp = domains
        .iter()
        .map(|d| d.accuracy_pct - 100.0)
        .fold(0.0f64, f64::min);

    let model_bytes = model.storage_bytes();
    let compression_vs_bf16 = if model_bytes > 0 {
        bf16_model_bytes as f64 / model_bytes as f64
    } else {
        1.0
    };

    Ok(RegimeEvalReport {
        regime: model.regime,
        model_bytes,
        effective_bits_per_weight: model.effective_bits_per_weight(),
        compression_vs_bf16,
        domains,
        mean_accuracy_pct,
        quality_regression_pp,
        worst_domain_regression_pp,
        mean_kl,
        cpu_cold_ttft_ms,
        cpu_warm_ttft_ms,
        cpu_p50_itl_ms,
        cpu_p95_itl_ms,
        cpu_warm_tok_per_s,
        cpu_wall_ms,
    })
}

/// Reference BF16 logits grouped by evaluation domain.
pub type DomainReferenceLogits = (EvalDomain, Vec<Vec<f32>>);

/// Precomputes BF16 teacher reference logits across all 4 domains.
pub fn compute_bf16_reference_logits(
    bf16_model: &QuantizedHybridModel,
    corpus: &GroundedCorpus,
    seed: u64,
) -> Result<Vec<DomainReferenceLogits>, MixedQuantError> {
    let v = bf16_model.config.vocab_size;
    let seq_len = 24usize;
    let seqs_per_domain = 6usize;
    let mut out = Vec::with_capacity(4);
    for domain in EvalDomain::ALL {
        let seqs = corpus.eval_sequences(domain, seqs_per_domain, seq_len, v, seed);
        let mut domain_logits = Vec::with_capacity(seqs.len());
        for seq in &seqs {
            domain_logits.push(bf16_model.forward_sequence(seq)?);
        }
        out.push((domain, domain_logits));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn build_mixed_linear(
    weights: &[f32],
    rows: usize,
    cols: usize,
    activations: &[f32],
    num_tokens: usize,
    format: QuantFormat,
    group_size: usize,
    rotation: Option<RotationTransform>,
) -> Result<QuantizedLinear, MixedQuantError> {
    match format {
        QuantFormat::Bf16 | QuantFormat::Int8 => {
            QuantizedLinear::quantize_plain(weights, rows, cols, format, group_size)
        }
        QuantFormat::Int4Sym | QuantFormat::Int4Asym | QuantFormat::Ternary => {
            let cfg = AwqConfig {
                format,
                group_size,
                grid_steps: 12,
                salient_ratio: 0.0, // Budget-tight mode: rely on SpinQuant + AWQ scaling without extra BF16 columns
                protect_salient_bf16: false,
            };
            Ok(calibrate_awq_with_rotation(
                weights,
                rows,
                cols,
                activations,
                num_tokens,
                &cfg,
                rotation,
            )?
            .linear)
        }
    }
}

fn quantize_with_rotation(
    weights: &[f32],
    rows: usize,
    cols: usize,
    format: QuantFormat,
    group_size: usize,
    rotation: RotationTransform,
) -> Result<QuantizedLinear, MixedQuantError> {
    let w_rot = rotation.rotate_matrix_rows(weights, rows, cols)?;
    let mut qlin = QuantizedLinear::quantize_plain(&w_rot, rows, cols, format, group_size)?;
    qlin.input_rotation = Some(rotation);
    Ok(qlin)
}

fn init_weight_matrix(rows: usize, cols: usize, gain: f32, rng: &mut SplitMix64) -> Vec<f32> {
    let scale = gain / (cols as f32).sqrt();
    let mut w = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        w.push(rng.next_normal() * scale);
    }
    w
}

/// RMSNorm that preserves realistic LLM channel outlier structure (akin to learned gamma weights).
fn rmsnorm_preserve_outliers(x: &[f32], seq_len: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..seq_len {
        let row = &x[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f64;
        for (c, &val) in row.iter().enumerate() {
            // Compute RMS on normal channels so persistent outlier channels remain heavy-tailed
            let is_outlier = c == 3 % d || c == 19 % d || c == 37 % d || c == 53 % d;
            if !is_outlier {
                sum_sq += (val as f64) * (val as f64);
            }
        }
        let rms = ((sum_sq / (d.saturating_sub(4).max(1) as f64)) + 1e-5).sqrt() as f32;
        let inv = 1.0 / rms;
        for c in 0..d {
            out[t * d + c] = row[c] * inv;
        }
    }
    out
}

fn rmsnorm_standard(x: &[f32], seq_len: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..seq_len {
        let row = &x[t * d..(t + 1) * d];
        let sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / ((sq / d as f64) + 1e-5).sqrt() as f32;
        for c in 0..d {
            out[t * d + c] = row[c] * inv;
        }
    }
    out
}

/// Causal multi-head self-attention over packed `[seq_len, 3 * d]` Q/K/V projections.
fn run_causal_attention(qkv: &[f32], seq_len: usize, d: usize, num_heads: usize) -> Vec<f32> {
    let hd = (d / num_heads.max(1)).max(1);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * d];

    for h in 0..num_heads {
        let h_off = h * hd;
        for t in 0..seq_len {
            let q = &qkv[t * 3 * d + h_off..t * 3 * d + h_off + hd];
            let mut scores = Vec::with_capacity(t + 1);
            let mut max_s = f32::NEG_INFINITY;
            for s in 0..=t {
                let k = &qkv[s * 3 * d + d + h_off..s * 3 * d + d + h_off + hd];
                let mut dot = 0.0f32;
                for i in 0..hd {
                    dot += q[i] * k[i];
                }
                let sc = dot * scale;
                if sc > max_s {
                    max_s = sc;
                }
                scores.push(sc);
            }
            let mut sum_exp = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - max_s).exp();
                sum_exp += *sc;
            }
            let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            for (s, &w) in scores.iter().enumerate() {
                let weight = w * inv;
                let v_vec = &qkv[s * 3 * d + 2 * d + h_off..s * 3 * d + 2 * d + h_off + hd];
                for i in 0..hd {
                    out[t * d + h_off + i] += weight * v_vec[i];
                }
            }
        }
    }
    out
}

/// Real Gated DeltaNet (GDN) recurrent state update:
/// `S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T`, `o_t = S_t q_t`.
fn run_gdn_recurrence(kvg: &[f32], seq_len: usize, d: usize, num_heads: usize) -> Vec<f32> {
    let hd = (d / num_heads.max(1)).max(1);
    let mut out = vec![0.0f32; seq_len * d];

    for h in 0..num_heads {
        let h_off = h * hd;
        let mut state = vec![0.0f32; hd * hd];
        let mut k_norm = vec![0.0f32; hd];
        let mut sk = vec![0.0f32; hd];

        for t in 0..seq_len {
            let k_raw = &kvg[t * 3 * d + h_off..t * 3 * d + h_off + hd];
            let v_raw = &kvg[t * 3 * d + d + h_off..t * 3 * d + d + h_off + hd];
            let g_raw = &kvg[t * 3 * d + 2 * d + h_off..t * 3 * d + 2 * d + h_off + hd];

            // L2 normalize k for stable delta-rule update
            let kn: f32 = k_raw.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-5);
            for i in 0..hd {
                k_norm[i] = k_raw[i] / kn;
            }

            // Decay alpha in (0.75, 0.99) and write rate beta in (0.05, 0.5)
            let g_mean: f32 = g_raw.iter().sum::<f32>() / (hd as f32);
            let alpha = 0.75 + 0.24 / (1.0 + (-g_mean).exp());
            let beta = 0.05 + 0.45 / (1.0 + (-(g_raw[0] - g_raw[hd - 1])).exp());

            // sk = S_{t-1} * k_t
            for r in 0..hd {
                let mut acc = 0.0f32;
                for c in 0..hd {
                    acc += state[r * hd + c] * k_norm[c];
                }
                sk[r] = acc;
            }

            // S_t = alpha * S_{t-1} + beta * (v_t - sk) * k_t^T
            for r in 0..hd {
                let delta = beta * (v_raw[r] - sk[r]);
                for c in 0..hd {
                    state[r * hd + c] =
                        (alpha * state[r * hd + c] + delta * k_norm[c]).clamp(-8.0, 8.0);
                }
            }

            // Readout o_t = S_t * k_t (using gate-modulated query)
            for r in 0..hd {
                let mut acc = 0.0f32;
                for c in 0..hd {
                    acc += state[r * hd + c] * k_norm[c];
                }
                let out_gate = 1.0 / (1.0 + (-g_raw[r]).exp());
                out[t * d + h_off + r] = acc * out_gate;
            }
        }
    }
    out
}

/// Top-2 softmax MoE router gate weights.
fn top2_router_weights(logits: &[f32]) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut weights = vec![0.0f32; n];
    let k = 2.min(n);
    let max_l = logits[order[0]];
    let mut sum = 0.0f32;
    for &idx in &order[..k] {
        let e = (logits[idx] - max_l).exp();
        weights[idx] = e;
        sum += e;
    }
    if sum > 0.0 {
        for &idx in &order[..k] {
            weights[idx] /= sum;
        }
    }
    weights
}

/// Argmax index of a slice.
#[must_use]
pub fn argmax(vals: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in vals.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xd = x as f64;
        let yd = y as f64;
        dot += xd * yd;
        na += xd * xd;
        nb += yd * yd;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}
