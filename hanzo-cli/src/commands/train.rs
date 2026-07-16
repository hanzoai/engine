//! `hanzo train` — native LoRA supervised fine-tuning (the Tinker-shaped
//! `hanzo-train` loop: load frozen base + LoRA, forward_backward/optim_step
//! over a `{prompt, completion}` JSONL dataset, save a PEFT adapter).

use std::path::PathBuf;

use anyhow::Result;
use hanzo_ml::{DType, Device};
use hanzo_train::{
    create_lora_training_client_from_engine, run_sft, run_sft_with_client, LoraConfig, SftConfig,
};

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
    let sft = SftConfig {
        model: cfg.model,
        data: cfg.data,
        lora: lora.clone(),
        lr: cfg.lr,
        steps: cfg.steps,
        batch_size: cfg.batch_size,
        out: cfg.out,
        seed: cfg.seed,
        sample_prompt: cfg.sample_prompt,
    };

    // Llama-family models train against the engine's own loaded (dequantized) weights; every
    // other architecture falls back to hanzo-train's standalone loader. Same loop either way.
    let (device, dtype) = (Device::Cpu, DType::F32);
    let report = match hanzo_engine::load_llama_base_for_training(&sft.model, dtype, &device)? {
        Some(base) => {
            tracing::info!(model = %sft.model, "training via engine-loaded llama base");
            let client = create_lora_training_client_from_engine(base, lora, device, dtype)?;
            run_sft_with_client(client, &sft)?
        }
        None => run_sft(&sft)?,
    };

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
