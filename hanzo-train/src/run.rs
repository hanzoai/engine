//! One-shot LoRA supervised fine-tuning run — the loop behind `hanzo train`.
//!
//! Drives the four Tinker-shaped primitives end to end: load + inject LoRA,
//! epoch over a `{prompt, completion}` JSONL dataset with shuffled batches,
//! `forward_backward` → `optim_step` per step, save the PEFT adapter, and
//! optionally sample a smoke-check completion from the trained weights.

use std::path::PathBuf;

use hanzo_ml::{DType, Device};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::client::create_lora_training_client;
use crate::data::load_jsonl;
use crate::types::{AdamParams, LoraConfig, ModelInput, SamplingParams};

/// Configuration for a supervised fine-tuning run.
#[derive(Clone, Debug)]
pub struct SftConfig {
    /// Base model: a Hugging Face repo id or a local directory.
    pub model: String,
    /// Training data: JSONL of `{"prompt", "completion"}`.
    pub data: PathBuf,
    pub lora: LoraConfig,
    pub lr: f64,
    /// Number of optimizer steps.
    pub steps: usize,
    /// Examples per forward_backward / optim_step.
    pub batch_size: usize,
    /// Output directory for the saved adapter.
    pub out: PathBuf,
    /// Shuffle seed.
    pub seed: u64,
    /// After training, greedy-sample a completion for this prompt (smoke check).
    pub sample_prompt: Option<String>,
}

/// What a finished run produced.
#[derive(Clone, Debug)]
pub struct SftReport {
    /// Where the PEFT adapter was written.
    pub adapter: PathBuf,
    pub trainable_params: usize,
    pub steps: usize,
    pub final_loss: f32,
    /// The smoke-check completion, if `sample_prompt` was set.
    pub sample: Option<String>,
}

/// Run supervised fine-tuning to completion. Blocking; call from a thread that
/// may compute for a while.
pub fn run_sft(cfg: &SftConfig) -> anyhow::Result<SftReport> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    tracing::info!(model = %cfg.model, "loading base model + injecting LoRA");
    let mut client = create_lora_training_client(&cfg.model, cfg.lora.clone(), device, dtype)?;
    let trainable_params = client.num_trainable_params();
    tracing::info!(trainable_params, "base loaded, adapters injected");

    let dataset = load_jsonl(
        &cfg.data,
        client.tokenizer(),
        client.bos_token_id(),
        client.eos_token_id(),
    )?;
    tracing::info!(examples = dataset.len(), "dataset tokenized");

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut order: Vec<usize> = (0..dataset.len()).collect();
    order.shuffle(&mut rng);
    let mut cursor = 0usize;

    let adam = AdamParams {
        lr: cfg.lr,
        ..AdamParams::default()
    };

    let mut final_loss = f32::NAN;
    for step in 1..=cfg.steps {
        // Build the next batch, cycling and reshuffling through the dataset.
        let mut batch = Vec::with_capacity(cfg.batch_size);
        while batch.len() < cfg.batch_size {
            if cursor >= order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            batch.push(dataset[order[cursor]].clone());
            cursor += 1;
        }

        let out = client.forward_backward(&batch)?;
        client.optim_step(adam)?;
        final_loss = out.loss;
        tracing::info!(step, loss = out.loss, tokens = out.num_tokens, "step");
    }

    let adapter = client.save_weights_and_get_sampling_client(&cfg.out)?;
    tracing::info!(adapter = %adapter.display(), "saved LoRA adapter");

    let sample = match &cfg.sample_prompt {
        None => None,
        Some(prompt) => {
            let sampled = client.sample(
                &ModelInput::from_ints(client.encode_prompt(prompt)?),
                &SamplingParams {
                    max_tokens: 64,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
            )?;
            let text = client.decode(&sampled)?;
            tracing::info!(completion = %text, "sample");
            Some(text)
        }
    };

    Ok(SftReport {
        adapter,
        trainable_params,
        steps: cfg.steps,
        final_loss,
        sample,
    })
}
