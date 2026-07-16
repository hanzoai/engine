//! Resolve a base model (Hugging Face repo id or local directory) into the pieces
//! training needs: config, tokenizer, and frozen weight tensors.
//!
//! This deliberately reuses the same building blocks the engine loader uses under
//! the hood — `hf_hub` for fetching, `tokenizers` for the tokenizer, and
//! `hanzo_ml::safetensors` for weights — rather than reimplementing any of them.

use std::{collections::HashMap, path::PathBuf};

use hanzo_ml::{DType, Device, Tensor};
use hf_hub::api::sync::ApiBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::model::Weights;

/// Everything needed to construct a [`crate::model::LoraModel`] and tokenize data.
pub struct BaseModel {
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub weights: Weights,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Deserialize)]
struct SpecialTokens {
    #[serde(default)]
    bos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

/// Fetch a file from a local directory or a Hugging Face repo, returning its path.
fn resolve(model: &str, file: &str) -> anyhow::Result<Option<PathBuf>> {
    let local = std::path::Path::new(model);
    if local.is_dir() {
        let p = local.join(file);
        return Ok(p.exists().then_some(p));
    }
    let token = std::env::var("HF_TOKEN").ok();
    let api = ApiBuilder::new().with_token(token).build()?;
    match api.model(model.to_string()).get(file) {
        Ok(p) => Ok(Some(p)),
        // A missing optional file (e.g. the shard index) is not an error.
        Err(_) => Ok(None),
    }
}

fn require(model: &str, file: &str) -> anyhow::Result<PathBuf> {
    resolve(model, file)?.ok_or_else(|| anyhow::anyhow!("`{file}` not found for model `{model}`"))
}

fn first_int(v: &serde_json::Value) -> Option<u32> {
    match v {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32),
        serde_json::Value::Array(a) => a.first().and_then(first_int),
        _ => None,
    }
}

/// Load and merge the model's safetensors shards into one frozen tensor map (cast to `dtype`).
fn load_weights(model: &str, dtype: DType, device: &Device) -> anyhow::Result<Weights> {
    let mut map: HashMap<String, Tensor> = HashMap::new();

    if let Some(index_path) = resolve(model, "model.safetensors.index.json")? {
        let index: SafetensorsIndex =
            serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
        let mut shards: Vec<String> = index.weight_map.into_values().collect();
        shards.sort();
        shards.dedup();
        for shard in shards {
            let path = require(model, &shard)?;
            for (k, v) in hanzo_ml::safetensors::load(path, device)? {
                map.insert(k, v);
            }
        }
    } else {
        let path = require(model, "model.safetensors")?;
        for (k, v) in hanzo_ml::safetensors::load(path, device)? {
            map.insert(k, v);
        }
    }

    if map.is_empty() {
        anyhow::bail!("no tensors loaded for model `{model}`");
    }
    Ok(Weights::new(map, dtype, device.clone()))
}

/// Resolve a base model. `dtype` is the working dtype for training (F32 recommended on CPU).
pub fn load_base_model(model: &str, dtype: DType, device: &Device) -> anyhow::Result<BaseModel> {
    let config_path = require(model, "config.json")?;
    let config_text = std::fs::read_to_string(&config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_text)?;
    let special: SpecialTokens = serde_json::from_str(&config_text)?;

    let tokenizer_path = require(model, "tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;

    let weights = load_weights(model, dtype, device)?;

    Ok(BaseModel {
        config,
        tokenizer,
        weights,
        bos_token_id: special.bos_token_id.as_ref().and_then(first_int),
        eos_token_id: special.eos_token_id.as_ref().and_then(first_int),
    })
}
