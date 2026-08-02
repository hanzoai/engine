//! Grad-enabled training forward for the llama family.
//!
//! The engine's inference [`Llama`](super::llama::Llama) forward is fused: paged/SDPA
//! attention, fused RoPE/softmax/RmsNorm CustomOps — none of which have a backward, so
//! they sever the autograd graph. LoRA fine-tuning needs gradients to flow through the
//! whole decoder, so a training forward must route through primitive (differentiable)
//! ops instead.
//!
//! That differentiable decoder already exists, proven, in [`hanzo_train::model::LoraModel`]
//! (softmax over scaled QKᵀ, `rotary_emb::rope_slow`, `RmsNorm::forward_diff`, no kv-cache,
//! LoRA factors as trainable [`Var`](hanzo_ml::Var)s from a [`VarMap`]). Rather than
//! duplicate it, this module bridges the engine to it:
//!
//!   * [`base_weights_for_training`] extracts the frozen base tensors out of a *loaded*
//!     engine [`Llama`] (dequantizing the projections), keyed by Hugging Face name — so
//!     LoRA runs against the engine's own loaded weights, not a re-load.
//!   * [`LlamaTrainForward`] builds the differentiable, LoRA-injected forward over the
//!     engine's [`Config`], with the seven llama projections adapted from a [`VarMap`].
//!
//! CPU + F32 is the supported training config.

// The `hanzo train` entrypoint drives this seam: [`load_llama_base_for_training`] assembles a
// [`BaseModel`] from an engine-loaded [`Llama`] and hands it to
// `hanzo_train::create_lora_training_client_from_engine`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_train::config::ModelConfig;
use hanzo_train::model::Weights;
use hanzo_train::BaseModel;

use super::llama::{Config, Llama};
use crate::pipeline::IsqModel;

#[cfg(test)]
use hanzo_nn::VarMap;
#[cfg(test)]
use hanzo_train::lora::LoraDelta;
#[cfg(test)]
use hanzo_train::model::LoraModel;
#[cfg(test)]
use hanzo_train::LoraConfig;

/// Map the engine's llama [`Config`] onto the training decoder's config. The engine
/// llama derives `head_dim` as `hidden_size / num_attention_heads`, which is exactly the
/// training config's default, so `head_dim` stays `None`.
pub(crate) fn training_config(cfg: &Config) -> ModelConfig {
    ModelConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: Some(cfg.num_key_value_heads),
        head_dim: None,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: f64::from(cfg.rope_theta),
        max_position_embeddings: cfg.max_position_embeddings,
        tie_word_embeddings: cfg.tie_word_embeddings,
    }
}

/// Frozen base weights of a *loaded* engine [`Llama`], keyed by Hugging Face tensor name,
/// ready to build a [`LlamaTrainForward`]. Projections are dequantized to plain tensors;
/// norms and embeddings come from the model's own residual set.
///
/// The projection order follows `Llama`'s [`IsqModel::get_layers`]: index 0 is `lm_head`,
/// then each layer contributes `q, k, v, o, gate, up, down` — the same enumeration
/// `Llama::imatrix_names` documents.
pub(crate) fn base_weights_for_training(model: &mut Llama) -> Result<HashMap<String, Tensor>> {
    const PER_LAYER: usize = 7;
    let mut map = HashMap::new();

    {
        let (layers, _mapper) = model.get_layers();
        if layers.is_empty() || (layers.len() - 1) % PER_LAYER != 0 {
            return Err(hanzo_ml::Error::msg(format!(
                "unexpected llama projection count {} (want 1 + 7·layers)",
                layers.len()
            )));
        }
        map.insert("lm_head.weight".to_string(), layers[0].0.dequantize_w()?);

        let n_layers = (layers.len() - 1) / PER_LAYER;
        for i in 0..n_layers {
            let base = 1 + i * PER_LAYER;
            let attn = format!("model.layers.{i}.self_attn");
            let mlp = format!("model.layers.{i}.mlp");
            for (off, name) in [
                format!("{attn}.q_proj.weight"),
                format!("{attn}.k_proj.weight"),
                format!("{attn}.v_proj.weight"),
                format!("{attn}.o_proj.weight"),
                format!("{mlp}.gate_proj.weight"),
                format!("{mlp}.up_proj.weight"),
                format!("{mlp}.down_proj.weight"),
            ]
            .into_iter()
            .enumerate()
            {
                map.insert(name, layers[base + off].0.dequantize_w()?);
            }
        }
    }

    // Norms + embeddings, HF-named ("model.embed_tokens.weight", "model.norm.weight",
    // "model.layers.{i}.{input,post_attention}_layernorm.weight").
    for (name, tensor) in model.residual_tensors() {
        map.insert(name, tensor);
    }

    Ok(map)
}

/// Load a llama-family base model by Hugging Face id (or local directory) into an engine
/// [`Llama`], then hand its config, tokenizer, and frozen (dequantized) base weights to
/// `hanzo-train` as a [`BaseModel`].
///
/// Returns `Ok(None)` when the model's architecture is not llama-family — the caller then
/// falls back to hanzo-train's standalone loader. Otherwise this is the engine-backed source
/// for `hanzo train`: LoRA fine-tunes against the engine's own loaded weights via
/// [`hanzo_train::create_lora_training_client_from_engine`]. CPU + F32 is the supported
/// training config.
pub fn load_llama_base_for_training(
    model_id: &str,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<Option<BaseModel>> {
    use crate::device_map::DeviceMapSetting;
    use crate::paged_attention::AttentionImplementation;
    use crate::pipeline::NormalLoadingMetadata;
    use indicatif::MultiProgress;
    use std::sync::Arc;

    let config_path = require(model_id, "config.json")?;
    let config_text = std::fs::read_to_string(&config_path)?;
    // Only the llama family routes through the engine extractor (the projection layout is
    // llama-specific); everything else defers to the standalone loader.
    if !is_llama_family(&config_text) {
        return Ok(None);
    }
    let cfg: Config = serde_json::from_str(&config_text)?;

    let tokenizer_path = require(model_id, "tokenizer.json")?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;

    let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
        weight_files(model_id)?,
        Vec::new(),
        Some(dtype),
        device,
        vec![None],
        true,
        None,
        |_| true,
        Arc::new(|_| crate::utils::varbuilder_utils::DeviceForLoadTensor::Base),
    )?;

    let mapper = DeviceMapSetting::dummy().into_mapper(
        cfg.num_hidden_layers,
        device,
        None,
        std::slice::from_ref(device),
    )?;
    let meta = NormalLoadingMetadata {
        mapper,
        loading_isq: false,
        real_device: device.clone(),
        multi_progress: Arc::new(MultiProgress::new()),
        matformer_slicing_config: None,
    };
    let mut model = Llama::new(&cfg, vb, false, meta, AttentionImplementation::Eager)?;

    let weights = base_weights_for_training(&mut model)?;
    let (bos_token_id, eos_token_id) = special_token_ids(&config_text);

    Ok(Some(BaseModel {
        config: training_config(&cfg),
        tokenizer,
        weights: Weights::new(weights, dtype, device.clone()),
        bos_token_id,
        eos_token_id,
    }))
}

/// Whether the config's first architecture resolves to the llama family.
fn is_llama_family(config_text: &str) -> bool {
    #[derive(serde::Deserialize, Default)]
    struct Arch {
        #[serde(default)]
        architectures: Vec<String>,
    }
    let arch: Arch = serde_json::from_str(config_text).unwrap_or_default();
    arch.architectures.first().is_some_and(|name| {
        matches!(
            crate::pipeline::NormalLoaderType::from_causal_lm_name(name),
            Ok(crate::pipeline::NormalLoaderType::Llama)
        )
    })
}

/// Resolve a file from a local directory or a Hugging Face repo.
fn resolve(model: &str, file: &str) -> anyhow::Result<Option<PathBuf>> {
    let local = Path::new(model);
    if local.is_dir() {
        let p = local.join(file);
        return Ok(p.exists().then_some(p));
    }
    let token = std::env::var("HF_TOKEN").ok();
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_token(token)
        .build()?;
    match api.model(model.to_string()).get(file) {
        Ok(p) => Ok(Some(p)),
        // A missing optional file (e.g. the shard index) is not an error.
        Err(_) => Ok(None),
    }
}

fn require(model: &str, file: &str) -> anyhow::Result<PathBuf> {
    resolve(model, file)?.ok_or_else(|| anyhow::anyhow!("`{file}` not found for model `{model}`"))
}

/// The model's safetensors shard paths — the single file, or every shard in the index.
fn weight_files(model: &str) -> anyhow::Result<Vec<PathBuf>> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: HashMap<String, String>,
    }
    if let Some(index_path) = resolve(model, "model.safetensors.index.json")? {
        let index: Index = serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
        let mut shards: Vec<String> = index.weight_map.into_values().collect();
        shards.sort();
        shards.dedup();
        shards.into_iter().map(|s| require(model, &s)).collect()
    } else {
        Ok(vec![require(model, "model.safetensors")?])
    }
}

/// Parse bos/eos ids from the model config (scalar or first-of-array).
fn special_token_ids(config_text: &str) -> (Option<u32>, Option<u32>) {
    #[derive(serde::Deserialize, Default)]
    struct Special {
        #[serde(default)]
        bos_token_id: Option<serde_json::Value>,
        #[serde(default)]
        eos_token_id: Option<serde_json::Value>,
    }
    fn first_int(v: &serde_json::Value) -> Option<u32> {
        match v {
            serde_json::Value::Number(n) => n.as_u64().and_then(|x| u32::try_from(x).ok()),
            serde_json::Value::Array(a) => a.first().and_then(first_int),
            _ => None,
        }
    }
    let s: Special = serde_json::from_str(config_text).unwrap_or_default();
    (
        s.bos_token_id.as_ref().and_then(first_int),
        s.eos_token_id.as_ref().and_then(first_int),
    )
}

/// A differentiable, LoRA-injected forward for the llama family, used by this module's tests
/// to prove the extracted base weights train. The production `hanzo train` path drives the
/// same [`LoraModel`] through [`hanzo_train::create_lora_training_client_from_engine`].
#[cfg(test)]
pub(crate) struct LlamaTrainForward {
    model: LoraModel,
}

#[cfg(test)]
impl LlamaTrainForward {
    /// Build the training forward. `weights` are the frozen base tensors (HF-named);
    /// `varmap` receives the trainable LoRA factors.
    pub(crate) fn new(
        cfg: &Config,
        weights: HashMap<String, Tensor>,
        lora: &LoraConfig,
        varmap: &VarMap,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let train_cfg = training_config(cfg);
        let weights = Weights::new(weights, dtype, device.clone());
        let model = LoraModel::new(&train_cfg, &weights, lora, varmap)?;
        Ok(Self { model })
    }

    /// Grad-tracked logits over the vocabulary for every position: `(b, l) -> (b, l, vocab)`.
    pub(crate) fn forward_train(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids)
    }

    /// The trainable LoRA factors, in a stable order (for saving an adapter).
    pub(crate) fn adapters(&self) -> &[LoraDelta] {
        self.model.adapters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::D;
    use hanzo_nn::{AdamW, Optimizer, ParamsAdamW};
    use hanzo_train::loss::masked_nll_sum;

    /// Tiny 2-layer llama with GQA, small dims, tied embeddings.
    fn tiny_config() -> Config {
        Config {
            hidden_size: 16,
            intermediate_size: 32,
            vocab_size: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    /// Deterministic N(0,1) sample stream, the same shape gguf_moe.rs uses and for
    /// the same reason: the CPU backend's `Device::set_seed` is a NO-OP (candle's
    /// CPU rng is not seedable), so an unseeded `Tensor::randn` gives this test a
    /// different starting point on every run. The loss assertion below has an
    /// absolute term, and a run that reduced the loss 9.09 -> 0.53 — a 17x drop —
    /// still failed it by 0.03. That is the guard flaking for reasons unrelated to
    /// what it guards.
    fn normals(seed: u64, n: usize) -> Vec<f32> {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect()
    }

    /// Random-init base weights, HF-named exactly as the engine loads them. Embeddings
    /// are tied to `lm_head`, so there is no separate `lm_head.weight` (the loaded model
    /// derives it from `model.embed_tokens.weight`).
    fn tiny_weights(cfg: &Config, device: &Device) -> HashMap<String, Tensor> {
        let hd = cfg.hidden_size / cfg.num_attention_heads;
        let n_q = cfg.num_attention_heads * hd;
        let n_kv = cfg.num_key_value_heads * hd;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        // Realistic embedding scale so the tied head can grow confident logits.
        m.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(
                normals(0xE3BD, cfg.vocab_size * cfg.hidden_size)
                    .into_iter()
                    .map(|v| v * 0.5)
                    .collect::<Vec<f32>>(),
                (cfg.vocab_size, cfg.hidden_size),
                device,
            )
            .unwrap(),
        );
        // Each projection gets its own stream, keyed by shape+index, so adding a
        // layer cannot shift the values of the ones before it.
        let mut nth = 0u64;
        let mut put = |name: String, out: usize, inp: usize| {
            nth += 1;
            m.insert(
                name,
                Tensor::from_vec(
                    normals(0x5EED + nth, out * inp)
                        .into_iter()
                        .map(|v| v * 0.05)
                        .collect::<Vec<f32>>(),
                    (out, inp),
                    device,
                )
                .unwrap(),
            );
        };
        for l in 0..cfg.num_hidden_layers {
            let a = format!("model.layers.{l}.self_attn");
            let f = format!("model.layers.{l}.mlp");
            put(format!("{a}.q_proj.weight"), n_q, cfg.hidden_size);
            put(format!("{a}.k_proj.weight"), n_kv, cfg.hidden_size);
            put(format!("{a}.v_proj.weight"), n_kv, cfg.hidden_size);
            put(format!("{a}.o_proj.weight"), cfg.hidden_size, n_q);
            put(
                format!("{f}.gate_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            );
            put(
                format!("{f}.up_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            );
            put(
                format!("{f}.down_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
            );
        }
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                m.insert(
                    format!("{p}.{norm}.weight"),
                    Tensor::ones((cfg.hidden_size,), DType::F32, device).unwrap(),
                );
            }
        }
        m.insert(
            "model.norm.weight".to_string(),
            Tensor::ones((cfg.hidden_size,), DType::F32, device).unwrap(),
        );
        m
    }

    fn tiny_lora() -> LoraConfig {
        LoraConfig {
            rank: 8,
            alpha: 16.0,
            target_modules: LoraConfig::default_target_modules(),
        }
    }

    /// Run the toy memorization loop, returning `(first_loss, last_loss)`. On the first
    /// step it also asserts every LoRA factor — and only those — receives a gradient.
    fn train_toy(fwd: &LlamaTrainForward, varmap: &VarMap) -> (f32, f32) {
        // Only the LoRA factors are trainable: 7 projections × 2 factors × 2 layers = 28.
        assert_eq!(varmap.all_vars().len(), 28);
        let device = Device::Cpu;
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 2e-2,
                beta1: 0.9,
                beta2: 0.95,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )
        .unwrap();

        // Memorize a fixed next-token mapping: [1,2,3,4] -> [2,3,4,5].
        let n = 4;
        let input = Tensor::from_slice(&[1u32, 2, 3, 4], (1, n), &device).unwrap();
        let targets = Tensor::from_slice(&[2u32, 3, 4, 5], (n,), &device).unwrap();
        let mask = Tensor::from_slice(&[1f32; 4], (n,), &device).unwrap();
        let total_w = 4.0f64;

        let (mut first, mut last) = (f32::NAN, f32::NAN);
        for step in 0..150 {
            let logits = fwd.forward_train(&input).unwrap();
            let vocab = logits.dim(D::Minus1).unwrap();
            let logits = logits.reshape((n, vocab)).unwrap();
            let loss =
                (masked_nll_sum(&logits, &targets, &mask).unwrap() * (1.0 / total_w)).unwrap();
            let lv = loss.to_scalar::<f32>().unwrap();
            let grads = loss.backward().unwrap();
            if step == 0 {
                // Every LoRA factor gets a gradient — the graph is intact through the whole
                // decoder. The base weights are plain frozen tensors (not `Var`s), so they
                // are neither in `varmap` nor updated: `28` trainable, `28` with gradients.
                let with_grad = varmap
                    .all_vars()
                    .iter()
                    .filter(|v| grads.get(v.as_tensor()).is_some())
                    .count();
                assert_eq!(
                    with_grad, 28,
                    "all (and only) LoRA factors must receive gradients"
                );
                first = lv;
            }
            opt.step(&grads).unwrap();
            last = lv;
        }
        (first, last)
    }

    #[test]
    fn forward_train_reduces_loss_on_a_toy_sequence() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg, &device);
        let varmap = VarMap::new();
        let fwd = LlamaTrainForward::new(&cfg, weights, &tiny_lora(), &varmap, DType::F32, &device)
            .unwrap();

        // Seven adapted projections registered for saving.
        assert_eq!(fwd.adapters().len(), 7 * cfg.num_hidden_layers);

        let (first, last) = train_toy(&fwd, &varmap);
        assert!(first.is_finite() && last.is_finite());
        assert!(
            last < first * 0.1 && last < 0.5,
            "LoRA training should drive the loss down: first={first}, last={last}"
        );
    }

    /// End-to-end: load a real engine [`Llama`] from an in-memory checkpoint, extract its
    /// base weights, and train LoRA against them. Proves the training forward runs against
    /// the engine's *own loaded model*, and that [`base_weights_for_training`] round-trips.
    #[test]
    fn trains_against_engine_loaded_llama() {
        use crate::device_map::DeviceMapSetting;
        use crate::paged_attention::AttentionImplementation;
        use crate::pipeline::NormalLoadingMetadata;
        use hanzo_quant::ShardedSafeTensors;
        use indicatif::MultiProgress;
        use std::sync::Arc;

        let device = Device::Cpu;
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg, &device);

        // Wrap the tensor map as a safetensors backend and load the engine's own model.
        let vb = ShardedSafeTensors::wrap(Box::new(weights.clone()), DType::F32, device.clone());
        let mapper = DeviceMapSetting::dummy()
            .into_mapper(
                cfg.num_hidden_layers,
                &device,
                None,
                std::slice::from_ref(&device),
            )
            .unwrap();
        let meta = NormalLoadingMetadata {
            mapper,
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        };
        let mut model = Llama::new(&cfg, vb, false, meta, AttentionImplementation::Eager).unwrap();

        // Extract the loaded model's frozen base weights.
        let extracted = base_weights_for_training(&mut model).unwrap();

        // Every base tensor the training forward needs round-trips out of the loaded model
        // with identical values (dequantizing an unquantized weight is a copy).
        for (name, want) in &weights {
            let got = extracted
                .get(name)
                .unwrap_or_else(|| panic!("missing extracted tensor `{name}`"));
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = got
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(want.len(), got.len(), "shape mismatch for `{name}`");
            assert!(
                want.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-5),
                "value mismatch for `{name}`"
            );
        }
        // Tied embeddings surface as the derived head too.
        assert!(extracted.contains_key("lm_head.weight"));

        // Those extracted weights drive a LoRA training forward that reduces the loss.
        let varmap = VarMap::new();
        let fwd =
            LlamaTrainForward::new(&cfg, extracted, &tiny_lora(), &varmap, DType::F32, &device)
                .unwrap();
        let (first, last) = train_toy(&fwd, &varmap);
        assert!(first.is_finite() && last.is_finite());
        assert!(
            last < first * 0.1 && last < 0.5,
            "LoRA training over the engine-loaded model should drive the loss down: \
             first={first}, last={last}"
        );
    }
}
