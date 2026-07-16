//! `hanzo distill` — sequence-level distillation: a teacher samples
//! completions for a prompt set, the student LoRA-fine-tunes on them
//! (`hanzo-train`'s run_distill: generate → distill.jsonl → run_sft).

use std::path::PathBuf;

use anyhow::Result;
use hanzo_train::{run_distill, DistillConfig, LoraConfig};

/// Arguments for `hanzo distill`, as parsed by the CLI.
pub struct DistillRunConfig {
    pub teacher: String,
    pub student: String,
    pub prompts: PathBuf,
    pub max_tokens: usize,
    pub temperature: f64,
    pub lora_rank: usize,
    pub lora_alpha: Option<f64>,
    pub lr: f64,
    pub steps: usize,
    pub batch_size: usize,
    pub out: PathBuf,
    pub seed: u64,
    pub sample_prompt: Option<String>,
}

pub fn run_distill_cmd(cfg: DistillRunConfig) -> Result<()> {
    let lora = LoraConfig {
        rank: cfg.lora_rank,
        alpha: cfg.lora_alpha.unwrap_or(2.0 * cfg.lora_rank as f64),
        target_modules: LoraConfig::default_target_modules(),
    };
    let report = run_distill(&DistillConfig {
        teacher: cfg.teacher,
        student: cfg.student,
        prompts: cfg.prompts,
        max_tokens: cfg.max_tokens,
        temperature: cfg.temperature,
        lora,
        lr: cfg.lr,
        steps: cfg.steps,
        batch_size: cfg.batch_size,
        out: cfg.out,
        seed: cfg.seed,
        sample_prompt: cfg.sample_prompt,
    })?;

    println!(
        "dataset: {} ({} examples)",
        report.dataset.display(),
        report.examples
    );
    println!("adapter: {}", report.sft.adapter.display());
    println!(
        "trainable params: {}  steps: {}  final loss: {:.4}",
        report.sft.trainable_params, report.sft.steps, report.sft.final_loss
    );
    if let Some(sample) = report.sft.sample {
        println!("sample: {sample}");
    }
    Ok(())
}
