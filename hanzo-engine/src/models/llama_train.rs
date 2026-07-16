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

// The in-crate consumer that drives this seam — the pipeline/CLI `hanzo train` entrypoint
// that assembles a `hanzo_train::BaseModel` from a loaded model and calls
// `hanzo_train::create_lora_training_client_from_engine` — lands in a separate change (the
// `hanzo train` subcommand). Until then the seam is exercised by this module's tests.
#![allow(dead_code)]

use std::collections::HashMap;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::VarMap;
use hanzo_train::config::ModelConfig;
use hanzo_train::lora::LoraDelta;
use hanzo_train::model::{LoraModel, Weights};
use hanzo_train::LoraConfig;

use super::llama::{Config, Llama};
use crate::pipeline::IsqModel;

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

/// A differentiable, LoRA-injected forward for the llama family, built over the engine's
/// own [`Config`] and frozen base weights. Base weights stay frozen; only the LoRA `A`/`B`
/// factors — registered in the caller's [`VarMap`] on the seven llama projections — train.
///
/// Reachable from outside the inference path so a trainer (e.g. `hanzo-train`) can drive
/// LoRA fine-tuning over an engine-loaded model: build the base map with
/// [`base_weights_for_training`], then [`LlamaTrainForward::new`].
pub(crate) struct LlamaTrainForward {
    model: LoraModel,
}

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
            Tensor::randn(0f32, 0.5f32, (cfg.vocab_size, cfg.hidden_size), device).unwrap(),
        );
        let mut put = |name: String, out: usize, inp: usize| {
            m.insert(
                name,
                Tensor::randn(0f32, 0.05f32, (out, inp), device).unwrap(),
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
