//! `hanzo-train` CLI — LoRA supervised fine-tuning of a small decoder LLM.
//!
//!   hanzo-train --model HuggingFaceTB/SmolLM2-135M --data sft.jsonl \
//!               --lora-rank 16 --lr 1e-4 --steps 100 --out ./adapter
//!
//! Dataset is JSONL with `{"prompt": "...", "completion": "..."}` per line.

use clap::Parser;
use hanzo_ml::{DType, Device};
use hanzo_train::{
    create_lora_training_client, data::load_jsonl, AdamParams, LoraConfig, ModelInput,
    SamplingParams,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

#[derive(Parser, Debug)]
#[command(name = "hanzo-train", about = "Native Rust LoRA fine-tuning (Tinker-shaped)")]
struct Args {
    /// Base model: a Hugging Face repo id or a local directory.
    #[arg(long)]
    model: String,

    /// Training data: JSONL of {"prompt", "completion"}.
    #[arg(long)]
    data: String,

    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,

    /// LoRA alpha. Defaults to 2 * rank.
    #[arg(long)]
    lora_alpha: Option<f64>,

    /// Learning rate.
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    /// Number of optimizer steps.
    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Examples per forward_backward / optim_step.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Output directory for the saved adapter.
    #[arg(long, default_value = "./hanzo-train-adapter")]
    out: String,

    /// Shuffle seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// After training, sample a completion for this prompt (optional smoke check).
    #[arg(long)]
    sample_prompt: Option<String>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hanzo_train=info".into()),
        )
        .init();

    let args = Args::parse();
    let device = Device::Cpu;
    let dtype = DType::F32;

    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(2.0 * args.lora_rank as f64),
        target_modules: LoraConfig::default_target_modules(),
    };

    tracing::info!(model = %args.model, "loading base model + injecting LoRA");
    let mut client = create_lora_training_client(&args.model, lora, device, dtype)?;
    tracing::info!(
        trainable_params = client.num_trainable_params(),
        "base loaded, adapters injected"
    );

    let dataset = load_jsonl(
        &args.data,
        client.tokenizer(),
        client.bos_token_id(),
        client.eos_token_id(),
    )?;
    tracing::info!(examples = dataset.len(), "dataset tokenized");

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut order: Vec<usize> = (0..dataset.len()).collect();
    order.shuffle(&mut rng);
    let mut cursor = 0usize;

    let adam = AdamParams {
        lr: args.lr,
        ..AdamParams::default()
    };

    for step in 1..=args.steps {
        // Build the next batch, cycling and reshuffling through the dataset.
        let mut batch = Vec::with_capacity(args.batch_size);
        while batch.len() < args.batch_size {
            if cursor >= order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            batch.push(dataset[order[cursor]].clone());
            cursor += 1;
        }

        let out = client.forward_backward(&batch)?;
        client.optim_step(adam)?;
        tracing::info!(step, loss = out.loss, tokens = out.num_tokens, "step");
    }

    let path = client.save_weights_and_get_sampling_client(&args.out)?;
    tracing::info!(adapter = %path.display(), "saved LoRA adapter");

    if let Some(prompt) = args.sample_prompt {
        let mut ids = Vec::new();
        if let Some(b) = client.bos_token_id() {
            ids.push(b);
        }
        ids.extend(
            client
                .tokenizer()
                .encode(prompt.as_str(), false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .iter()
                .copied(),
        );
        let sampled = client.sample(
            &ModelInput::from_ints(ids),
            &SamplingParams {
                max_tokens: 64,
                temperature: 0.0,
                ..SamplingParams::default()
            },
        )?;
        let text = client
            .tokenizer()
            .decode(&sampled, true)
            .map_err(anyhow::Error::msg)?;
        tracing::info!(completion = %text, "sample");
    }

    Ok(())
}
