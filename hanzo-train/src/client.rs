//! The Tinker-shaped training client.
//!
//! Four primitives, mirroring Thinking Machines' Tinker:
//!   - [`create_lora_training_client`] — load a base model frozen, inject LoRA `Var`s
//!   - [`TrainingClient::forward_backward`] — forward with grad → masked next-token CE → backward
//!   - [`TrainingClient::optim_step`] — apply AdamW to the accumulated gradients
//!   - [`TrainingClient::sample`] — decode from the current (base+LoRA) weights
//!   - [`TrainingClient::save_weights_and_get_sampling_client`] — write a PEFT adapter the engine loads
//!
//! The training loop mirrors the engine's proven AnyMoE gate trainer
//! (`hanzo-engine/src/pipeline/amoe.rs`): build `AdamW`, forward, `cross_entropy`,
//! `loss.backward()`, `optimizer.step(&gradstore)`.

use std::path::{Path, PathBuf};

use hanzo_ml::{backprop::GradStore, DType, Device, Tensor, D};
use hanzo_nn::{AdamW, Optimizer, VarMap};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokenizers::Tokenizer;

use crate::loader::{load_base_model, BaseModel};
use crate::model::LoraModel;
use crate::types::{
    AdamParams, Datum, ForwardBackwardOutput, LoraConfig, ModelInput, SamplingParams,
};

pub struct TrainingClient {
    model: LoraModel,
    varmap: VarMap,
    lora: LoraConfig,
    tokenizer: Tokenizer,
    eos_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    device: Device,
    dtype: DType,
    optimizer: Option<AdamW>,
    grads: Option<GradStore>,
}

/// Load `base_model` frozen and attach trainable LoRA adapters — Tinker's
/// `create_lora_training_client(base_model, LoraConfig)`. Resolves the base with
/// hanzo-train's standalone loader ([`load_base_model`]).
pub fn create_lora_training_client(
    base_model: &str,
    lora: LoraConfig,
    device: Device,
    dtype: DType,
) -> anyhow::Result<TrainingClient> {
    from_base(
        load_base_model(base_model, dtype, &device)?,
        lora,
        device,
        dtype,
    )
}

/// Attach trainable LoRA adapters to a base resolved by the *engine* — its config,
/// tokenizer, and frozen weights extracted from an already-loaded engine model
/// (`hanzo_engine::models::llama_train::base_weights_for_training`), rather than
/// hanzo-train's standalone loader. Same [`TrainingClient`], different base source:
/// LoRA fine-tuning runs against the engine's own loaded weights.
pub fn create_lora_training_client_from_engine(
    base: BaseModel,
    lora: LoraConfig,
    device: Device,
    dtype: DType,
) -> anyhow::Result<TrainingClient> {
    from_base(base, lora, device, dtype)
}

/// The single construction path shared by both constructors: freeze the base, inject the
/// LoRA `Var`s, and wrap it in a [`TrainingClient`]. The training logic lives only here.
fn from_base(
    base: BaseModel,
    lora: LoraConfig,
    device: Device,
    dtype: DType,
) -> anyhow::Result<TrainingClient> {
    let varmap = VarMap::new();
    let model = LoraModel::new(&base.config, &base.weights, &lora, &varmap)?;
    Ok(TrainingClient {
        model,
        varmap,
        lora,
        tokenizer: base.tokenizer,
        eos_token_id: base.eos_token_id,
        bos_token_id: base.bos_token_id,
        device,
        dtype,
        optimizer: None,
        grads: None,
    })
}

impl TrainingClient {
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }
    pub fn num_trainable_params(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// BOS (if the model has one) + tokenized `text` — the prompt encoding
    /// used by `sample` callers everywhere.
    pub fn encode_prompt(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let mut ids = Vec::new();
        if let Some(b) = self.bos_token_id {
            ids.push(b);
        }
        ids.extend(
            self.tokenizer
                .encode(text, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .iter()
                .copied(),
        );
        Ok(ids)
    }

    /// Decode sampled token ids to text, skipping special tokens.
    pub fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(anyhow::Error::msg)
    }

    /// Forward the batch with gradient tracking, compute mask-weighted next-token
    /// cross-entropy, and backward. Gradients accumulate until [`Self::optim_step`].
    pub fn forward_backward(&mut self, data: &[Datum]) -> anyhow::Result<ForwardBackwardOutput> {
        if data.is_empty() {
            anyhow::bail!("forward_backward called with an empty batch");
        }

        let mut nll_sums: Vec<Tensor> = Vec::with_capacity(data.len());
        let mut total_weight = 0f32;

        for datum in data {
            datum.validate()?;
            let n = datum.model_input.len();
            let input = Tensor::from_slice(datum.model_input.to_ints(), (1, n), &self.device)?;
            let targets = Tensor::from_slice(datum.target_tokens.as_slice(), (n,), &self.device)?;
            let weights = Tensor::from_slice(datum.weights.as_slice(), (n,), &self.device)?
                .to_dtype(self.dtype)?;

            // (1, n, vocab) -> (n, vocab)
            let logits = self.model.forward(&input)?;
            let vocab = logits.dim(D::Minus1)?;
            let logits = logits.reshape((n, vocab))?;

            nll_sums.push(crate::loss::masked_nll_sum(&logits, &targets, &weights)?);
            total_weight += datum.trained_tokens();
        }

        if total_weight <= 0.0 {
            anyhow::bail!("batch has no supervised tokens (all loss weights are zero)");
        }

        let mut total_nll = nll_sums[0].clone();
        for t in &nll_sums[1..] {
            total_nll = (total_nll + t)?;
        }
        let loss = (total_nll * (1.0 / total_weight as f64))?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        let grads = loss.backward()?;
        match self.grads.take() {
            None => self.grads = Some(grads),
            Some(mut acc) => {
                acc.extend(grads)?;
                self.grads = Some(acc);
            }
        }

        let mut out = ForwardBackwardOutput {
            loss: loss_val,
            num_tokens: total_weight as usize,
            ..Default::default()
        };
        out.metrics.insert("loss".into(), loss_val);
        Ok(out)
    }

    /// Apply AdamW to the accumulated gradients and clear them. The optimizer (and
    /// its momentum state) persists across calls; hyperparameters are refreshed each step.
    pub fn optim_step(&mut self, params: AdamParams) -> anyhow::Result<()> {
        let grads = self
            .grads
            .take()
            .ok_or_else(|| anyhow::anyhow!("optim_step called before forward_backward"))?;

        match &mut self.optimizer {
            None => {
                self.optimizer = Some(AdamW::new(self.varmap.all_vars(), params.to_hanzo())?);
            }
            Some(opt) => opt.set_params(params.to_hanzo()),
        }
        self.optimizer.as_mut().unwrap().step(&grads)?;
        Ok(())
    }

    /// Decode up to `params.max_tokens` new tokens from the current weights.
    /// Returns only the newly generated tokens (Tinker's `sample`).
    pub fn sample(&self, input: &ModelInput, params: &SamplingParams) -> anyhow::Result<Vec<u32>> {
        let mut tokens = input.tokens.clone();
        let mut generated = Vec::new();
        let mut rng = StdRng::seed_from_u64(params.seed);

        for _ in 0..params.max_tokens {
            let n = tokens.len();
            let ids = Tensor::from_slice(tokens.as_slice(), (1, n), &self.device)?;
            let logits = self.model.forward(&ids)?; // (1, n, vocab)
            let last = logits
                .narrow(1, n - 1, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?;
            let next = pick_token(&last.to_vec1::<f32>()?, params, &mut rng);
            tokens.push(next);
            generated.push(next);
            if Some(next) == self.eos_token_id || params.stop_tokens.contains(&next) {
                break;
            }
        }
        Ok(generated)
    }

    /// Write the LoRA adapter (safetensors + `adapter_config.json`) in PEFT layout so
    /// the engine's inference loader can pick it up. Returns the adapter directory.
    pub fn save_weights_and_get_sampling_client(
        &self,
        dir: impl AsRef<Path>,
    ) -> anyhow::Result<PathBuf> {
        crate::save::write_adapter(dir, self.model.adapters(), &self.lora)
    }
}

/// Sample one token id from raw logits under `params`.
fn pick_token(logits: &[f32], params: &SamplingParams, rng: &mut StdRng) -> u32 {
    if params.temperature <= 0.0 {
        return argmax(logits);
    }

    // temperature scale
    let inv_t = 1.0 / params.temperature as f32;
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v * inv_t))
        .collect();

    // top-k filter
    if params.top_k > 0 && (params.top_k as usize) < scaled.len() {
        scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scaled.truncate(params.top_k as usize);
    }

    // softmax over the (possibly filtered) candidates
    let max = scaled.iter().fold(f32::NEG_INFINITY, |m, &(_, v)| m.max(v));
    let mut probs: Vec<(usize, f32)> = scaled.iter().map(|&(i, v)| (i, (v - max).exp())).collect();
    let sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    for p in probs.iter_mut() {
        p.1 /= sum;
    }

    // top-p (nucleus) filter
    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0f32;
        let mut keep = probs.len();
        for (idx, &(_, p)) in probs.iter().enumerate() {
            cum += p;
            if cum >= params.top_p as f32 {
                keep = idx + 1;
                break;
            }
        }
        probs.truncate(keep);
        let renorm: f32 = probs.iter().map(|&(_, p)| p).sum();
        for p in probs.iter_mut() {
            p.1 /= renorm;
        }
    }

    // inverse-CDF sample
    let u: f32 = rng.random();
    let mut cum = 0f32;
    for &(i, p) in &probs {
        cum += p;
        if u <= cum {
            return i as u32;
        }
    }
    probs.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}
