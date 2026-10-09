#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! EmbeddingGemma 2 (`EmbeddingGemma2Model`, google/embeddinggemma-2): its text backbone, as
//! transformers 5.19 `models/embedding_gemma2/modeling_embedding_gemma2.py` computes it.
//!
//! Every layer attends both ways. A sliding layer sees the keys within `sliding_window` of a query
//! (inclusive), a full layer sees every key with its own head dim and key-value head count
//! (`per_layer_config`). Queries and keys are RMS-normed with a scale, values without one, and the
//! attention scale is 1. After the MLP each layer gates in its slice of a per-layer embedding (PLE)
//! projected from the token embeddings alone, then multiplies its output by `layer_scalar`. The
//! final norm is followed by `embedding_projection`, hidden -> `embedding_dim`, on every token;
//! sentence-transformers' mean pooling and normalization follow in the pipeline.
//!
//! Its norms scale by the stored weight (Gemma 4), not by 1 + weight (Gemma 3). The vision and
//! audio towers are not loaded: `/v1/embeddings` takes text.

use std::{collections::HashMap, sync::Arc};

use hanzo_ml::{Device, Module, Result, Tensor};
use hanzo_nn::Linear;
use hanzo_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};
use serde::Deserialize;

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{embedding, Activation, Mlp, RmsNorm, RotaryEmbedding, ScaledEmbedding, Sdpa},
    layers_masker::BidirectionalMasker,
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::FlashParams, EmbeddingModel, IsqModel, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

/// Tokens one input may hold: the model card's context window. `max_position_embeddings`
/// (262144) is the range its rotary tables were defined over, not what it was trained to read.
pub const CONTEXT: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Rope {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ropes {
    pub sliding_attention: Rope,
    pub full_attention: Rope,
}

impl Default for Ropes {
    fn default() -> Self {
        Self {
            sliding_attention: Rope {
                rope_theta: 10_000.,
            },
            full_attention: Rope {
                rope_theta: 1_000_000.,
            },
        }
    }
}

/// One layer's attention shape where it differs from the text config's own.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LayerShape {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    /// Keyed by layer index as written ("05"); absent, every full layer takes 512 and 1.
    #[serde(default)]
    pub per_layer_config: Option<HashMap<String, LayerShape>>,
    #[serde(default)]
    pub rope_parameters: Ropes,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    #[serde(default)]
    pub attention_bias: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingGemma2Config {
    pub text_config: TextConfig,
}

impl TextConfig {
    /// `layer_types`, or transformers' default: every sixth layer and the last are full.
    pub fn layer_type(&self, layer: usize) -> LayerType {
        match &self.layer_types {
            Some(types) => types[layer],
            None if (layer + 1).is_multiple_of(6) || layer + 1 == self.num_hidden_layers => {
                LayerType::FullAttention
            }
            None => LayerType::SlidingAttention,
        }
    }

    /// The layer's (head_dim, key-value heads).
    pub fn layer_shape(&self, layer: usize) -> Result<LayerShape> {
        let own = LayerShape {
            head_dim: self.head_dim,
            num_key_value_heads: self.num_key_value_heads,
        };
        match &self.per_layer_config {
            Some(shapes) => {
                for (key, shape) in shapes {
                    let index: usize = key.parse().map_err(|_| {
                        hanzo_ml::Error::Msg(format!("per_layer_config key `{key}` is no layer"))
                    })?;
                    if index == layer {
                        return Ok(*shape);
                    }
                }
                Ok(own)
            }
            None if self.layer_type(layer) == LayerType::FullAttention => Ok(LayerShape {
                head_dim: 512,
                num_key_value_heads: 1,
            }),
            None => Ok(own),
        }
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding: bool,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        cfg: &TextConfig,
        layer: usize,
        rope: Arc<RotaryEmbedding>,
        mapper: &dyn DeviceMapper,
        vb: ShardedVarBuilder,
        loading_isq: bool,
    ) -> Result<Self> {
        let LayerShape {
            head_dim,
            num_key_value_heads,
        } = cfg.layer_shape(layer)?;
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let quant = &cfg.quantization_config;
        let linear = |i, o, name| {
            hanzo_quant::linear_b(
                i,
                o,
                bias,
                quant,
                mapper.set_device(layer, vb.pp(name), loading_isq),
            )
        };
        let norm_vb = |name| mapper.set_device(layer, vb.pp(name), false);
        Ok(Self {
            q_proj: linear(hidden, heads * head_dim, "q_proj")?,
            k_proj: linear(hidden, num_key_value_heads * head_dim, "k_proj")?,
            v_proj: linear(hidden, num_key_value_heads * head_dim, "v_proj")?,
            o_proj: linear(heads * head_dim, hidden, "o_proj")?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, norm_vb("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, norm_vb("k_norm"))?,
            v_norm: RmsNorm::new_gemma_3n(head_dim, cfg.rms_norm_eps, false, norm_vb("v_norm"))?,
            rope,
            num_heads: heads,
            num_kv_heads: num_key_value_heads,
            head_dim,
            sliding: cfg.layer_type(layer) == LayerType::SlidingAttention,
            sdpa_params: SdpaParams {
                n_kv_groups: heads / num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0,
                sliding_window: None,
                sinks: None,
            },
        })
    }

    /// `sliding_mask` is `None` when no two tokens are farther apart than the window.
    fn forward(&self, xs: &Tensor, sliding_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let heads = |x: Tensor, n: usize| -> Result<Tensor> {
            x.reshape((b, l, n, self.head_dim))?.transpose(1, 2)
        };
        let q = heads(self.q_proj.forward(xs)?, self.num_heads)?.apply(&self.q_norm)?;
        let k = heads(self.k_proj.forward(xs)?, self.num_kv_heads)?.apply(&self.k_norm)?;
        let v = heads(self.v_proj.forward(xs)?, self.num_kv_heads)?.apply(&self.v_norm)?;
        let (q, k) = self.rope.forward(&q, &k, &vec![0; b])?;
        let mask = if self.sliding { sliding_mask } else { None };
        let out = Sdpa.run_attention_noflash(&q, &k, &v, mask, &self.sdpa_params, false)?;
        self.o_proj
            .forward(&out.transpose(1, 2)?.reshape((b, l, ()))?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    /// This layer's rows of `ple.per_layer_model_projection`: token embedding -> its PLE input,
    /// then `ple.per_layer_projection_norm`, which every layer shares.
    ple_input: Linear,
    ple_input_norm: RmsNorm,
    ple_gate: Arc<dyn QuantMethod>,
    ple_projection: Arc<dyn QuantMethod>,
    ple_norm: RmsNorm,
    scalar: Tensor,
    act: Activation,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &TextConfig,
        layer: usize,
        rope: Arc<RotaryEmbedding>,
        ple_input: Linear,
        ple_input_norm: RmsNorm,
        mapper: &dyn DeviceMapper,
        vb: ShardedVarBuilder,
        loading_isq: bool,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let norm = |name, size| {
            RmsNorm::new(
                size,
                cfg.rms_norm_eps,
                mapper.set_device(layer, vb.pp(name), false),
            )
        };
        let ple = vb.pp("ple_block");
        let ple_dim = cfg.hidden_size_per_layer_input;
        Ok(Self {
            attn: Attention::new(cfg, layer, rope, mapper, vb.pp("self_attn"), loading_isq)?,
            mlp: Box::new(Mlp::new(
                mapper.set_device(layer, vb.pp("mlp"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_activation,
                comm,
            )?),
            input_layernorm: norm("input_layernorm", cfg.hidden_size)?,
            post_attention_layernorm: norm("post_attention_layernorm", cfg.hidden_size)?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm", cfg.hidden_size)?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm", cfg.hidden_size)?,
            ple_input,
            ple_input_norm,
            ple_gate: hanzo_quant::linear_no_bias(
                cfg.hidden_size,
                ple_dim,
                &cfg.quantization_config,
                mapper.set_device(layer, ple.pp("per_layer_input_gate"), loading_isq),
            )?,
            ple_projection: hanzo_quant::linear_no_bias(
                ple_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                mapper.set_device(layer, ple.pp("per_layer_projection"), loading_isq),
            )?,
            ple_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer, ple.pp("post_per_layer_input_norm"), false),
            )?,
            scalar: mapper
                .set_device(layer, vb.clone(), false)
                .get((1,), "layer_scalar")?,
            act: cfg.hidden_activation,
        })
    }

    /// `embeds` are the scaled token embeddings the model began from.
    fn forward(
        &self,
        xs: &Tensor,
        embeds: &Tensor,
        ple_scale: f64,
        sliding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ple = (self.ple_input.forward(embeds)? * ple_scale)?.apply(&self.ple_input_norm)?;
        let h = self
            .attn
            .forward(&xs.apply(&self.input_layernorm)?, sliding_mask)?;
        let xs = (xs + h.apply(&self.post_attention_layernorm)?)?;
        let h = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?;
        let xs = (&xs + h.apply(&self.post_feedforward_layernorm)?)?;
        let h = crate::ops::mul_and_act(&self.ple_gate.forward(&xs)?, &ple, self.act)?;
        let h = self.ple_projection.forward(&h)?.apply(&self.ple_norm)?;
        (xs + h)?.broadcast_mul(&self.scalar)
    }
}

pub struct EmbeddingGemma2 {
    embed_tokens: ScaledEmbedding,
    layers: Vec<Layer>,
    /// `ple.per_layer_model_projection`, whose rows the layers read in slices.
    ple_weight: Tensor,
    /// `ple.per_layer_projection_norm`.
    ple_norm: RmsNorm,
    /// `ple.per_layer_model_projection_scale`, hidden_size^-1/2.
    ple_scale: f64,
    norm: RmsNorm,
    projection: Arc<dyn QuantMethod>,
    sliding_window: usize,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl EmbeddingGemma2 {
    pub fn new(
        cfg: &EmbeddingGemma2Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            hanzo_ml::bail!("Expected AttentionImplementation::Eager");
        }
        let cfg = &cfg.text_config;
        let vb = vb.pp("language_model");
        let mapper = normal_loading_metadata.mapper;
        let quant = &cfg.quantization_config;
        let hidden = cfg.hidden_size;
        let ple_dim = cfg.hidden_size_per_layer_input;

        let embed_tokens = ScaledEmbedding::new(
            (hidden as f64).sqrt(),
            embedding(
                cfg.vocab_size,
                hidden,
                mapper.set_nm_device(vb.pp("embed_tokens"), false),
                quant,
            )?,
        );
        let ple_vb = vb.pp("ple");
        let ple_weight = mapper.set_nm_device(ple_vb.clone(), false).get(
            (cfg.num_hidden_layers * ple_dim, hidden),
            "per_layer_model_projection.weight",
        )?;
        let ple_norm = RmsNorm::new(
            ple_dim,
            cfg.rms_norm_eps,
            mapper.set_nm_device(ple_vb.pp("per_layer_projection_norm"), false),
        )?;

        // One rotary table per (device, layer type): sliding and full layers differ in theta and
        // head dim.
        let mut ropes: HashMap<_, Arc<RotaryEmbedding>> = HashMap::new();
        for layer in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let kind = cfg.layer_type(layer);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry((device.location(), kind == LayerType::SlidingAttention))
            {
                let theta = match kind {
                    LayerType::SlidingAttention => cfg.rope_parameters.sliding_attention.rope_theta,
                    LayerType::FullAttention => cfg.rope_parameters.full_attention.rope_theta,
                };
                e.insert(Arc::new(RotaryEmbedding::new(
                    theta as f32,
                    cfg.layer_shape(layer)?.head_dim,
                    CONTEXT,
                    device,
                    is_gptx,
                    vb.dtype(),
                )?));
            }
        }

        let vb_l = vb.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer| {
            let device = mapper
                .device_for(layer, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rope = ropes[&(
                device.location(),
                cfg.layer_type(layer) == LayerType::SlidingAttention,
            )]
                .clone();
            let ple_input = Linear::new(
                ple_weight
                    .narrow(0, layer * ple_dim, ple_dim)?
                    .to_device(device)?,
                None,
            );
            let ple_input_norm = RmsNorm::new(
                ple_dim,
                cfg.rms_norm_eps,
                mapper.set_device(layer, ple_vb.pp("per_layer_projection_norm"), false),
            )?;
            let comm = mapper.get_comm_for(layer)?;
            Layer::new(
                cfg,
                layer,
                rope,
                ple_input,
                ple_input_norm,
                &*mapper,
                vb_l.pp(layer),
                normal_loading_metadata.loading_isq,
                &comm,
            )
        })?;

        Ok(Self {
            embed_tokens,
            layers,
            ple_weight,
            ple_norm,
            ple_scale: (hidden as f64).powf(-0.5),
            norm: RmsNorm::new(
                hidden,
                cfg.rms_norm_eps,
                mapper.set_nm_device(vb.pp("norm"), false),
            )?,
            projection: hanzo_quant::linear_no_bias(
                hidden,
                cfg.embedding_dim,
                quant,
                mapper.set_nm_device(vb.pp("embedding_projection"), false),
            )?,
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            mapper,
        })
    }

    /// Token states `[batch, seq, embedding_dim]` for unpadded `input_ids` `[batch, seq]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeds = self.embed_tokens.forward(input_ids)?;
        let (_, l) = input_ids.dims2()?;
        // A sliding layer sees |i - j| <= sliding_window; the masker masks |i - j| >= its width.
        let sliding_mask = if l > self.sliding_window + 1 {
            Some(BidirectionalMasker.make_sliding_mask(
                input_ids,
                embeds.dtype(),
                self.sliding_window + 1,
            )?)
        } else {
            None
        };
        let mut xs = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let mask = match &sliding_mask {
                Some(m) => Some(m.to_device(xs.device())?),
                None => None,
            };
            xs = layer.forward(
                &xs,
                &embeds.to_device(xs.device())?,
                self.ple_scale,
                mask.as_ref(),
            )?;
        }
        let xs = xs.to_device(&self.device)?.apply(&self.norm)?;
        self.projection.forward(&xs)
    }
}

impl IsqModel for EmbeddingGemma2 {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_proj, Some(i)));
            tensors.push((&mut layer.attn.k_proj, Some(i)));
            tensors.push((&mut layer.attn.v_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
            tensors.extend(layer.mlp.get_isq_layers().into_iter().map(|m| (m, Some(i))));
            tensors.push((&mut layer.ple_gate, Some(i)));
            tensors.push((&mut layer.ple_projection, Some(i)));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let lm = uvb.pp("language_model");
        lm.pp("embed_tokens").add(&self.embed_tokens);
        lm.pp("norm").add(&self.norm);
        lm.pp("embedding_projection").add(&self.projection);
        lm.pp("ple")
            .pp("per_layer_projection_norm")
            .add(&self.ple_norm);
        lm.pp("ple")
            .add_tensor("per_layer_model_projection.weight", self.ple_weight.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            let l = lm.pp("layers").pp(i);
            l.add_tensor("layer_scalar", layer.scalar.clone());
            l.pp("input_layernorm").add(&layer.input_layernorm);
            l.pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            l.pp("pre_feedforward_layernorm")
                .add(&layer.pre_feedforward_layernorm);
            l.pp("post_feedforward_layernorm")
                .add(&layer.post_feedforward_layernorm);
            l.pp("self_attn").pp("q_norm").add(&layer.attn.q_norm);
            l.pp("self_attn").pp("k_norm").add(&layer.attn.k_norm);
            l.pp("ple_block")
                .pp("post_per_layer_input_norm")
                .add(&layer.ple_norm);
        }
        uvb.to_safetensors()
    }
}

impl EmbeddingModel for EmbeddingGemma2 {
    fn forward(&self, input_ids: &Tensor, _flash_params: &FlashParams) -> Result<Tensor> {
        self.forward(input_ids)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for EmbeddingGemma2 {}

/// Weights the loader reads, in elements: (per-layer, the rest).
pub fn sizes(cfg: &TextConfig, weight_pack_factor: usize) -> Result<(Vec<usize>, usize)> {
    let h = cfg.hidden_size;
    let p = cfg.hidden_size_per_layer_input;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for layer in 0..cfg.num_hidden_layers {
        let LayerShape {
            head_dim,
            num_key_value_heads,
        } = cfg.layer_shape(layer)?;
        let q = cfg.num_attention_heads * head_dim;
        let kv = num_key_value_heads * head_dim;
        let packed = (h * q + 2 * h * kv + q * h + 3 * h * cfg.intermediate_size + 2 * h * p)
            / weight_pack_factor;
        let norms = 5 * h + 2 * head_dim + p * h + 1;
        layers.push(packed + norms);
    }
    let rest = cfg.vocab_size * h / weight_pack_factor + p + h + h * cfg.embedding_dim;
    Ok((layers, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TextConfig {
        serde_json::from_str::<EmbeddingGemma2Config>(
            r#"{"architectures": ["EmbeddingGemma2Model"], "text_config": {
                "attention_bias": false, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 512, "hidden_size_per_layer_input": 512, "embedding_dim": 768,
                "intermediate_size": 2048, "num_attention_heads": 4, "num_hidden_layers": 24,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "sliding_window": 512,
                "vocab_size": 262144,
                "per_layer_config": {"05": {"head_dim": 512, "num_key_value_heads": 1},
                                     "11": {"head_dim": 512, "num_key_value_heads": 1},
                                     "17": {"head_dim": 512, "num_key_value_heads": 1},
                                     "23": {"head_dim": 512, "num_key_value_heads": 1}},
                "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}}"#,
        )
        .unwrap()
        .text_config
    }

    #[test]
    fn full_layers_take_their_own_shape() -> Result<()> {
        let cfg = config();
        assert_eq!(cfg.layer_type(5), LayerType::FullAttention);
        assert_eq!(cfg.layer_type(4), LayerType::SlidingAttention);
        assert_eq!(cfg.layer_type(23), LayerType::FullAttention);
        assert_eq!(cfg.layer_shape(5)?.head_dim, 512);
        assert_eq!(cfg.layer_shape(5)?.num_key_value_heads, 1);
        assert_eq!(cfg.layer_shape(4)?.head_dim, 256);
        assert_eq!(cfg.layer_shape(4)?.num_key_value_heads, 2);
        Ok(())
    }

    #[test]
    fn the_text_weights_are_the_checkpoints() -> Result<()> {
        // google/embeddinggemma-2's safetensors header (914f7f89): language_model.* holds
        // 271,002,648 values; the per-layer sizes carry the PLE projection's rows.
        let (layers, rest) = sizes(&config(), 1)?;
        assert_eq!(layers.iter().sum::<usize>(), 130_099_224 + 6_291_456);
        assert_eq!(rest, 134_217_728 + 393_216 + 512 + 512);
        Ok(())
    }
}
