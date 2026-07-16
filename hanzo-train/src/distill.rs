//! Sequence-level distillation: a teacher model samples completions for a
//! prompt set, and the student LoRA-fine-tunes on the teacher's outputs.
//!
//! Pure composition of the existing primitives — teacher `sample` →
//! `{prompt, completion}` JSONL artifact (kept for inspection) → [`run_sft`].
//! The teacher loads with no adapters (empty `target_modules`), i.e. exactly
//! the base model.

use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use hanzo_ml::{DType, Device};
use serde::Deserialize;

use crate::client::create_lora_training_client;
use crate::data::Example;
use crate::run::{run_sft, SftConfig, SftReport};
use crate::types::{LoraConfig, ModelInput, SamplingParams};

/// Configuration for a distillation run.
#[derive(Clone, Debug)]
pub struct DistillConfig {
    /// Teacher model: a Hugging Face repo id or a local directory.
    pub teacher: String,
    /// Student model to fine-tune.
    pub student: String,
    /// Prompt set: JSONL of `{"prompt": ...}`.
    pub prompts: PathBuf,
    /// Teacher generation length per prompt.
    pub max_tokens: usize,
    /// Teacher sampling temperature (0 = greedy).
    pub temperature: f64,
    /// Student LoRA shape.
    pub lora: LoraConfig,
    pub lr: f64,
    pub steps: usize,
    pub batch_size: usize,
    /// Output directory: gets `distill.jsonl` (the teacher dataset) and
    /// `adapter/` (the trained student adapter).
    pub out: PathBuf,
    pub seed: u64,
    /// After training, greedy-sample the student for this prompt (smoke check).
    pub sample_prompt: Option<String>,
}

/// What a finished distillation produced.
#[derive(Clone, Debug)]
pub struct DistillReport {
    /// The teacher-generated `{prompt, completion}` dataset.
    pub dataset: PathBuf,
    pub examples: usize,
    /// The student fine-tune that ran on it.
    pub sft: SftReport,
}

#[derive(Deserialize)]
struct PromptLine {
    prompt: String,
}

/// Load a `{"prompt": ...}` JSONL prompt set.
pub fn load_prompts(path: impl AsRef<Path>) -> anyhow::Result<Vec<String>> {
    let reader = BufReader::new(File::open(path.as_ref())?);
    let mut prompts = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: PromptLine = serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("line {}: {e}", lineno + 1))?;
        prompts.push(parsed.prompt);
    }
    anyhow::ensure!(!prompts.is_empty(), "no prompts parsed from prompt set");
    Ok(prompts)
}

/// Write examples as `{prompt, completion}` JSONL.
pub fn write_examples(path: impl AsRef<Path>, examples: &[Example]) -> anyhow::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path.as_ref())?;
    for ex in examples {
        serde_json::to_writer(&mut file, ex)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

/// Run distillation to completion. Blocking; call from a thread that may
/// compute for a while.
pub fn run_distill(cfg: &DistillConfig) -> anyhow::Result<DistillReport> {
    let prompts = load_prompts(&cfg.prompts)?;

    // The teacher is the pure base model: no target modules => no adapters.
    tracing::info!(teacher = %cfg.teacher, "loading teacher");
    let teacher = create_lora_training_client(
        &cfg.teacher,
        LoraConfig {
            rank: 1,
            alpha: 1.0,
            target_modules: Vec::new(),
        },
        Device::Cpu,
        DType::F32,
    )?;

    let mut examples = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let params = SamplingParams {
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            seed: cfg.seed.wrapping_add(i as u64),
            ..SamplingParams::default()
        };
        let tokens = teacher.sample(
            &ModelInput::from_ints(teacher.encode_prompt(prompt)?),
            &params,
        )?;
        let completion = teacher.decode(&tokens)?;
        if completion.trim().is_empty() {
            tracing::warn!(
                prompt_index = i,
                "teacher produced an empty completion; skipping"
            );
            continue;
        }
        tracing::info!(
            prompt_index = i,
            generated_tokens = tokens.len(),
            "teacher completion"
        );
        examples.push(Example {
            prompt: prompt.clone(),
            completion,
        });
    }
    anyhow::ensure!(
        !examples.is_empty(),
        "teacher produced no non-empty completions"
    );
    drop(teacher); // free the teacher before the student loads

    let dataset = cfg.out.join("distill.jsonl");
    write_examples(&dataset, &examples)?;
    tracing::info!(dataset = %dataset.display(), examples = examples.len(), "teacher dataset written");

    let sft = run_sft(&SftConfig {
        model: cfg.student.clone(),
        data: dataset.clone(),
        lora: cfg.lora.clone(),
        lr: cfg.lr,
        steps: cfg.steps,
        batch_size: cfg.batch_size,
        out: cfg.out.join("adapter"),
        seed: cfg.seed,
        sample_prompt: cfg.sample_prompt.clone(),
    })?;

    Ok(DistillReport {
        dataset,
        examples: examples.len(),
        sft,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_parse_and_reject_empty_sets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");

        fs::write(&path, "{\"prompt\": \"a\"}\n\n{\"prompt\": \"b\"}\n").unwrap();
        assert_eq!(load_prompts(&path).unwrap(), vec!["a", "b"]);

        fs::write(&path, "\n\n").unwrap();
        assert!(load_prompts(&path).is_err());

        fs::write(&path, "{\"nope\": 1}\n").unwrap();
        let err = load_prompts(&path).unwrap_err().to_string();
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn examples_round_trip_through_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/distill.jsonl");
        let examples = vec![
            Example {
                prompt: "2+2=".into(),
                completion: "4".into(),
            },
            Example {
                prompt: "with \"quotes\"\nand newline".into(),
                completion: " spaced ".into(),
            },
        ];
        write_examples(&path, &examples).unwrap();

        let reader = BufReader::new(File::open(&path).unwrap());
        let back: Vec<Example> = reader
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].prompt, "2+2=");
        assert_eq!(back[1].prompt, "with \"quotes\"\nand newline");
        assert_eq!(back[1].completion, " spaced ");
    }
}
