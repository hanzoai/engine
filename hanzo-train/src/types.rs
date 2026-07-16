//! Tinker-shaped value types for the training API.
//!
//! These mirror the four Tinker primitives' data contracts:
//! `Datum`, `LoraConfig`, `AdamParams`, `SamplingParams`. They are plain values
//! (no engine coupling) so the pure batching / masking / shape logic is testable
//! in isolation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A tokenized model input. Mirrors Tinker's `ModelInput` (the token half of a `Datum`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct ModelInput {
    pub tokens: Vec<u32>,
}

impl ModelInput {
    pub fn from_ints(tokens: impl Into<Vec<u32>>) -> Self {
        Self {
            tokens: tokens.into(),
        }
    }

    pub fn to_ints(&self) -> &[u32] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn append_int(&mut self, token: u32) {
        self.tokens.push(token);
    }
}

/// One supervised training example. Mirrors Tinker's `Datum` with the two loss
/// inputs it needs for supervised fine-tuning folded in:
/// - `target_tokens[i]` is the next-token label for position `i`
/// - `weights[i]` is the loss mask (0.0 = ignore, e.g. prompt; 1.0 = train, e.g. completion)
///
/// Invariant: `model_input.len() == target_tokens.len() == weights.len()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct Datum {
    pub model_input: ModelInput,
    pub target_tokens: Vec<u32>,
    pub weights: Vec<f32>,
}

impl Datum {
    /// Validate the length invariant; returns the shared sequence length.
    pub fn validate(&self) -> Result<usize, DatumError> {
        let n = self.model_input.len();
        if self.target_tokens.len() != n || self.weights.len() != n {
            return Err(DatumError::LengthMismatch {
                input: n,
                targets: self.target_tokens.len(),
                weights: self.weights.len(),
            });
        }
        Ok(n)
    }

    /// Sum of the loss mask — the number of supervised positions in this datum.
    pub fn trained_tokens(&self) -> f32 {
        self.weights.iter().sum()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DatumError {
    #[error("datum length mismatch: input={input}, targets={targets}, weights={weights}")]
    LengthMismatch {
        input: usize,
        targets: usize,
        weights: usize,
    },
}

/// LoRA adapter configuration. Field semantics and shapes match the engine's
/// inference-side `hanzo_quant::lora::LoraConfig` so a saved adapter round-trips
/// back into the engine for sampling.
///
/// Deserializes with per-field defaults (rank 16, alpha 32, the seven
/// llama-family projections), so `{}` is a valid config on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(default)]
pub struct LoraConfig {
    /// Low-rank dimension `r`.
    pub rank: usize,
    /// LoRA scaling numerator `alpha`; effective scale is `alpha / rank`.
    pub alpha: f64,
    /// Projection module leaf names to adapt, e.g. `q_proj`, `gate_proj`.
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    /// The seven attention + MLP projections adapted by default (Llama / Qwen2 family).
    pub fn default_target_modules() -> Vec<String> {
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Effective LoRA scale applied to the `B·A` delta.
    pub fn scale(&self) -> f64 {
        if self.rank > 0 {
            self.alpha / self.rank as f64
        } else {
            1.0
        }
    }

    pub fn adapts(&self, leaf: &str) -> bool {
        self.target_modules.iter().any(|m| m == leaf)
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target_modules: Self::default_target_modules(),
        }
    }
}

/// On-disk PEFT adapter config (`adapter_config.json`). Written on save so the
/// engine's loader (`serde_json::from_str::<hanzo_quant::lora::LoraConfig>`) accepts it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeftAdapterConfig {
    pub peft_type: String,
    pub task_type: String,
    pub r: usize,
    pub lora_alpha: f64,
    pub target_modules: Vec<String>,
    pub bias: String,
}

impl From<&LoraConfig> for PeftAdapterConfig {
    fn from(c: &LoraConfig) -> Self {
        Self {
            peft_type: "LORA".to_string(),
            task_type: "CAUSAL_LM".to_string(),
            r: c.rank,
            lora_alpha: c.alpha,
            target_modules: c.target_modules.clone(),
            bias: "none".to_string(),
        }
    }
}

/// AdamW hyperparameters for `optim_step`. Mirrors Tinker's `AdamParams`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(default)]
pub struct AdamParams {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl AdamParams {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            ..Self::default()
        }
    }

    pub fn to_hanzo(self) -> hanzo_nn::ParamsAdamW {
        hanzo_nn::ParamsAdamW {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
}

impl Default for AdamParams {
    fn default() -> Self {
        // LLM fine-tuning defaults: beta2=0.95, no weight decay on LoRA params.
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Sampling controls for `sample`. Mirrors Tinker's `SamplingParams`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(default)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub seed: u64,
    /// Token ids that stop generation (in addition to the model EOS).
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            seed: 0,
            stop_tokens: Vec::new(),
        }
    }
}

/// Result of a `forward_backward` call. Mirrors Tinker's `{loss, metrics}` output.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct ForwardBackwardOutput {
    /// Mean next-token cross-entropy over supervised positions in the batch.
    pub loss: f32,
    /// Number of supervised (mask=1) positions the loss averaged over.
    pub num_tokens: usize,
    pub metrics: BTreeMap<String, f32>,
}
