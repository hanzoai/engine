//! `hanzo train` — native LoRA supervised fine-tuning (the Tinker-shaped
//! `hanzo-train` loop: load frozen base + LoRA, forward_backward/optim_step
//! over a `{prompt, completion}` JSONL dataset, save a PEFT adapter).

use std::path::PathBuf;

use anyhow::Result;
use hanzo_train::{run_sft, LoraConfig, SftConfig};

/// Arguments for `hanzo train`, as parsed by the CLI.
pub struct TrainRunConfig {
    pub model: String,
    pub data: PathBuf,
    pub lora_rank: usize,
    pub lora_alpha: Option<f64>,
    pub lr: f64,
    pub steps: usize,
    pub batch_size: usize,
    pub out: PathBuf,
    pub seed: u64,
    pub sample_prompt: Option<String>,
}

pub fn run_train(cfg: TrainRunConfig) -> Result<()> {
    let lora = LoraConfig {
        rank: cfg.lora_rank,
        alpha: cfg.lora_alpha.unwrap_or(2.0 * cfg.lora_rank as f64),
        target_modules: LoraConfig::default_target_modules(),
    };
    let report = run_sft(&SftConfig {
        model: cfg.model,
        data: cfg.data,
        lora,
        lr: cfg.lr,
        steps: cfg.steps,
        batch_size: cfg.batch_size,
        out: cfg.out,
        seed: cfg.seed,
        sample_prompt: cfg.sample_prompt,
    })?;

    println!("adapter: {}", report.adapter.display());
    println!(
        "trainable params: {}  steps: {}  final loss: {:.4}",
        report.trainable_params, report.steps, report.final_loss
    );
    if let Some(sample) = report.sample {
        println!("sample: {sample}");
    }
    Ok(())
}
