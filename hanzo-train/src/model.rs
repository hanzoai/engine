//! LoRA-injected decoder forward for the Llama / Qwen2 family.
//!
//! Structurally identical to `hanzo_transformers::models::qwen2` (RMSNorm, RoPE,
//! GQA, SwiGLU MLP), with two differences required for training:
//!   1. base weights are frozen plain tensors loaded from safetensors,
//!   2. the q/k/v/o/gate/up/down projections are [`LoraLinear`] with trainable
//!      `A`/`B` factors from a [`VarMap`].
//!
//! No KV cache: the forward returns logits for every position, which is what
//! next-token supervised training needs; sampling recomputes the grown prefix.

use std::collections::HashMap;

use hanzo_ml::{DType, Device, Module, Result, Tensor, D};
use hanzo_nn::{rotary_emb, Embedding, Init, Linear, RmsNorm, VarMap};

use crate::config::ModelConfig;
use crate::lora::{LoraDelta, LoraLinear};
use crate::types::LoraConfig;

/// Frozen base weights keyed by Hugging Face tensor name.
pub struct Weights {
    map: HashMap<String, Tensor>,
    dtype: DType,
    device: Device,
}

impl Weights {
    pub fn new(map: HashMap<String, Tensor>, dtype: DType, device: Device) -> Self {
        Self { map, dtype, device }
    }

    fn get(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("missing base tensor `{name}`")))?
            .to_dtype(self.dtype)
    }

    fn get_opt(&self, name: &str) -> Result<Option<Tensor>> {
        match self.map.get(name) {
            Some(t) => Ok(Some(t.to_dtype(self.dtype)?)),
            None => Ok(None),
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}

struct Attention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // `rope_slow` and `ops::softmax` are composed of primitive (differentiable)
        // ops; their fused CustomOp counterparts (`rope`, `softmax_last_dim`) are
        // inference-only kernels with no backward, which would sever the graph.
        let q = rotary_emb::rope_slow(&q, cos, sin)?;
        let k = rotary_emb::rope_slow(&k, cos, sin)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = hanzo_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v)?; // (b, heads, l, hd)

        let out = out.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate_proj: LoraLinear,
    up_proj: LoraLinear,
    down_proj: LoraLinear,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = hanzo_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_ln: RmsNorm,
}

impl Layer {
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // `forward_diff` is RmsNorm's differentiable path (primitive ops); the plain
        // `forward` dispatches to a fused CustomOp with no backward.
        let residual = xs;
        let hs = self.input_ln.forward_diff(xs)?;
        let hs = self.attn.forward(&hs, cos, sin, mask)?;
        let xs = (residual + hs)?;
        let residual = &xs;
        let hs = self.post_ln.forward_diff(&xs)?;
        let hs = self.mlp.forward(&hs)?;
        residual + hs
    }
}

/// A trainable LoRA decoder-LM. Base frozen; only the LoRA factors are [`Var`](hanzo_ml::Var)s
/// in `varmap`.
pub struct LoraModel {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    dtype: DType,
    /// The LoRA factors, in a stable order, for saving.
    adapters: Vec<LoraDelta>,
}

/// Grouped-query attention key/value expansion: `(b, n_kv, l, hd) -> (b, n_kv*n_rep, l, hd)`.
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, l, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, l, hd))?
        .reshape((b, n_kv * n_rep, l, hd))
}

fn rope_tables(cfg: &ModelConfig, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = cfg.head_dim();
    let max_seq = cfg.max_position_embeddings;
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / (cfg.rope_theta.powf(i as f64 / dim as f64)) as f32)
        .collect();
    let inv_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_len), device)?;
    let t = Tensor::arange(0u32, max_seq as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq, 1))?;
    let freqs = t.matmul(&inv_freq)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

impl LoraModel {
    /// Build the model: frozen base from `weights`, trainable LoRA factors registered in `varmap`.
    pub fn new(
        cfg: &ModelConfig,
        weights: &Weights,
        lora: &LoraConfig,
        varmap: &VarMap,
    ) -> Result<Self> {
        let device = weights.device.clone();
        let dtype = weights.dtype;
        let mut adapters = Vec::new();

        let embed_w = weights.get("model.embed_tokens.weight")?;
        let embed = Embedding::new(embed_w.clone(), cfg.hidden_size);

        let (cos, sin) = rope_tables(cfg, dtype, &device)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let hd = cfg.head_dim();
            let n_heads = cfg.num_attention_heads;
            let n_kv = cfg.num_kv_heads();

            let attn = Attention {
                q_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.self_attn.q_proj"),
                    "q_proj",
                    cfg.hidden_size,
                    n_heads * hd,
                    &mut adapters,
                )?,
                k_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.self_attn.k_proj"),
                    "k_proj",
                    cfg.hidden_size,
                    n_kv * hd,
                    &mut adapters,
                )?,
                v_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.self_attn.v_proj"),
                    "v_proj",
                    cfg.hidden_size,
                    n_kv * hd,
                    &mut adapters,
                )?,
                o_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.self_attn.o_proj"),
                    "o_proj",
                    n_heads * hd,
                    cfg.hidden_size,
                    &mut adapters,
                )?,
                num_heads: n_heads,
                num_kv_heads: n_kv,
                num_kv_groups: cfg.num_kv_groups(),
                head_dim: hd,
                hidden_size: cfg.hidden_size,
            };
            let mlp = Mlp {
                gate_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.mlp.gate_proj"),
                    "gate_proj",
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    &mut adapters,
                )?,
                up_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.mlp.up_proj"),
                    "up_proj",
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    &mut adapters,
                )?,
                down_proj: make_linear(
                    weights,
                    varmap,
                    lora,
                    &format!("{p}.mlp.down_proj"),
                    "down_proj",
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    &mut adapters,
                )?,
            };
            let input_ln = RmsNorm::new(
                weights.get(&format!("{p}.input_layernorm.weight"))?,
                cfg.rms_norm_eps,
            );
            let post_ln = RmsNorm::new(
                weights.get(&format!("{p}.post_attention_layernorm.weight"))?,
                cfg.rms_norm_eps,
            );
            layers.push(Layer {
                attn,
                mlp,
                input_ln,
                post_ln,
            });
        }

        let norm = RmsNorm::new(weights.get("model.norm.weight")?, cfg.rms_norm_eps);

        // Tied embeddings when there is no separate lm_head in the checkpoint.
        let lm_head_w = if weights.contains("lm_head.weight") {
            weights.get("lm_head.weight")?
        } else {
            embed_w
        };
        let lm_head = Linear::new(lm_head_w, None);

        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
            cos,
            sin,
            device,
            dtype,
            adapters,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn adapters(&self) -> &[LoraDelta] {
        &self.adapters
    }

    fn causal_mask(&self, l: usize) -> Result<Option<Tensor>> {
        if l <= 1 {
            return Ok(None);
        }
        let mut m = vec![0f32; l * l];
        for i in 0..l {
            for j in (i + 1)..l {
                m[i * l + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(m, (l, l), &self.device)?
            .reshape((1, 1, l, l))?
            .to_dtype(self.dtype)?;
        Ok(Some(mask))
    }

    /// Hidden states for `input_ids` `(b, l)` -> `(b, l, hidden)`.
    fn hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, l) = input_ids.dims2()?;
        let cos = self.cos.narrow(0, 0, l)?;
        let sin = self.sin.narrow(0, 0, l)?;
        let mask = self.causal_mask(l)?;
        let mut xs = self.embed.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin, mask.as_ref())?;
        }
        self.norm.forward_diff(&xs)
    }

    /// Logits over the vocabulary for every position: `(b, l) -> (b, l, vocab)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hs = self.hidden(input_ids)?;
        self.lm_head.forward(&hs)
    }
}

#[allow(clippy::too_many_arguments)]
fn make_linear(
    weights: &Weights,
    varmap: &VarMap,
    lora: &LoraConfig,
    prefix: &str,
    leaf: &str,
    in_dim: usize,
    out_dim: usize,
    adapters: &mut Vec<LoraDelta>,
) -> Result<LoraLinear> {
    let weight = weights.get(&format!("{prefix}.weight"))?;
    let bias = weights.get_opt(&format!("{prefix}.bias"))?;
    let base = Linear::new(weight, bias);

    if !lora.adapts(leaf) || lora.rank == 0 {
        return Ok(LoraLinear::frozen(base));
    }

    let a = varmap.get(
        (lora.rank, in_dim),
        &format!("{prefix}.lora_A.weight"),
        Init::Randn {
            mean: 0.0,
            stdev: 1.0 / lora.rank as f64,
        },
        weights.dtype,
        &weights.device,
    )?;
    let b = varmap.get(
        (out_dim, lora.rank),
        &format!("{prefix}.lora_B.weight"),
        Init::Const(0.0),
        weights.dtype,
        &weights.device,
    )?;
    let delta = LoraDelta {
        name: prefix.to_string(),
        a,
        b,
        scale: lora.scale(),
    };
    adapters.push(delta.clone());
    Ok(LoraLinear::with_delta(base, delta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::masked_nll_sum;
    use hanzo_nn::{AdamW, Optimizer, ParamsAdamW};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 48,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
        }
    }

    fn tiny_weights(cfg: &ModelConfig, device: &Device) -> Weights {
        let hd = cfg.head_dim();
        let mut m: HashMap<String, Tensor> = HashMap::new();
        // Embedding is tied to lm_head here; give it a realistic scale so logits can
        // become confident (a 0.02-scale tied head caps loss near ln(vocab) regardless
        // of how well the adapters fit).
        m.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::randn(0f32, 0.5f32, (cfg.vocab_size, cfg.hidden_size), device).unwrap(),
        );
        let mut put = |name: &str, out: usize, inp: usize| {
            m.insert(
                name.to_string(),
                Tensor::randn(0f32, 0.05f32, (out, inp), device).unwrap(),
            );
        };
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            put(
                &format!("{p}.self_attn.q_proj.weight"),
                cfg.num_attention_heads * hd,
                cfg.hidden_size,
            );
            put(
                &format!("{p}.self_attn.k_proj.weight"),
                cfg.num_kv_heads() * hd,
                cfg.hidden_size,
            );
            put(
                &format!("{p}.self_attn.v_proj.weight"),
                cfg.num_kv_heads() * hd,
                cfg.hidden_size,
            );
            put(
                &format!("{p}.self_attn.o_proj.weight"),
                cfg.hidden_size,
                cfg.num_attention_heads * hd,
            );
            put(
                &format!("{p}.mlp.gate_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            );
            put(
                &format!("{p}.mlp.up_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            );
            put(
                &format!("{p}.mlp.down_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
            );
        }
        // RMSNorm weights are all-ones (identity scale) — build them separately.
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
        Weights::new(m, DType::F32, device.clone())
    }

    #[test]
    fn micro_var_through_linear_gets_grad() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let w = vm
            .get(
                (3, 4),
                "w",
                Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
                DType::F32,
                &dev,
            )
            .unwrap();
        let x = Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap();
        let y = Linear::new(w.clone(), None).forward(&x).unwrap(); // (2,3)
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let vars = vm.all_vars();
        // The tensor handed to the layer shares the Var's id, so its gradient is found.
        assert_eq!(vars[0].as_tensor().id(), w.id());
        assert!(grads.get(vars[0].as_tensor()).is_some());
    }

    #[test]
    fn lora_training_reduces_loss_on_a_toy_sequence() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg, &device);
        let lora = LoraConfig {
            rank: 8,
            alpha: 16.0,
            target_modules: LoraConfig::default_target_modules(),
        };
        let varmap = VarMap::new();
        let model = LoraModel::new(&cfg, &weights, &lora, &varmap).unwrap();

        // Only LoRA factors are trainable: 7 projections x 2 factors x 2 layers = 28 vars.
        assert_eq!(varmap.all_vars().len(), 28);

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

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..150 {
            let logits = model.forward(&input).unwrap();
            let vocab = logits.dim(D::Minus1).unwrap();
            let logits = logits.reshape((n, vocab)).unwrap();
            let loss =
                (masked_nll_sum(&logits, &targets, &mask).unwrap() * (1.0 / total_w)).unwrap();
            let lv = loss.to_scalar::<f32>().unwrap();
            if step == 0 {
                // Every LoRA factor must receive a gradient — otherwise the graph is
                // severed by a fused inference kernel and nothing trains.
                let grads = loss.backward().unwrap();
                let with_grad = varmap
                    .all_vars()
                    .iter()
                    .filter(|v| grads.get(v.as_tensor()).is_some())
                    .count();
                assert_eq!(with_grad, 28, "all LoRA factors must receive gradients");
                opt.step(&grads).unwrap();
                first = lv;
            } else {
                let grads = loss.backward().unwrap();
                opt.step(&grads).unwrap();
            }
            last = lv;
        }

        // The adapters memorize the toy mapping: loss collapses from ~ln(vocab) toward 0.
        assert!(first.is_finite() && last.is_finite());
        assert!(
            last < first * 0.1 && last < 0.5,
            "LoRA training should drive the loss down: first={first}, last={last}"
        );
    }
}
