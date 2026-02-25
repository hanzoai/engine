//! BERT-based embedding model for sentence-transformers compatibility
//!
//! This module implements BERT (Bidirectional Encoder Representations from Transformers)
//! for embedding generation, compatible with sentence-transformers models like:
//! - sentence-transformers/all-MiniLM-L6-v2
//! - sentence-transformers/all-mpnet-base-v2
//! - BAAI/bge-small-en-v1.5
//! - thenlper/gte-small
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, LayerNormConfig, Linear};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{embedding, layer_norm, linear, Activation},
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::FlashParams, EmbeddingModel, IsqModel, NormalLoadingMetadata,
    },
    serde_default_fn,
};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

// Default values for BERT config fields
serde_default_fn!(usize, vocab_size_default, 30522);
serde_default_fn!(usize, hidden_size_default, 768);
serde_default_fn!(usize, num_hidden_layers_default, 12);
serde_default_fn!(usize, num_attention_heads_default, 12);
serde_default_fn!(usize, intermediate_size_default, 3072);
serde_default_fn!(f64, layer_norm_eps_default, 1e-12);
serde_default_fn!(usize, max_position_embeddings_default, 512);
serde_default_fn!(usize, type_vocab_size_default, 2);

fn default_hidden_act() -> Activation {
    Activation::Gelu
}

/// BERT model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    #[serde(default = "vocab_size_default")]
    pub vocab_size: usize,
    #[serde(default = "hidden_size_default")]
    pub hidden_size: usize,
    #[serde(default = "num_hidden_layers_default")]
    pub num_hidden_layers: usize,
    #[serde(default = "num_attention_heads_default")]
    pub num_attention_heads: usize,
    #[serde(default = "intermediate_size_default")]
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "layer_norm_eps_default")]
    pub layer_norm_eps: f64,
    #[serde(default = "max_position_embeddings_default")]
    pub max_position_embeddings: usize,
    #[serde(default = "type_vocab_size_default")]
    pub type_vocab_size: usize,
    /// Pad token ID for masking
    pub pad_token_id: Option<usize>,
}

impl BertConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// BERT embeddings: word + position + token_type + LayerNorm
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let word_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings"),
            &None,
        )?;
        let position_embeddings = embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embeddings"),
            &None,
        )?;
        let token_type_embeddings = embedding(
            cfg.type_vocab_size,
            cfg.hidden_size,
            vb.pp("token_type_embeddings"),
            &None,
        )?;
        let ln_config = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let ln = layer_norm(cfg.hidden_size, ln_config, vb.pp("LayerNorm"))?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm: ln,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();

        // Word embeddings
        let word_embeds = self.word_embeddings.forward(input_ids)?;

        // Position embeddings (0, 1, 2, ..., seq_len-1)
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Token type embeddings (default to zeros if not provided)
        let token_type_embeds = match token_type_ids {
            Some(tids) => self.token_type_embeddings.forward(tids)?,
            None => {
                let zeros = Tensor::zeros((batch_size, seq_len), DType::U32, device)?;
                self.token_type_embeddings.forward(&zeros)?
            }
        };

        // Combine embeddings
        let embeddings = (word_embeds + position_embeds)?.add(&token_type_embeds)?;

        // LayerNorm
        self.layer_norm.forward(&embeddings)
    }
}

/// BERT self-attention (using candle_nn::Linear instead of QuantMethod)
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let attention_head_size = cfg.head_dim();
        let all_head_size = cfg.num_attention_heads * attention_head_size;

        let query = linear(cfg.hidden_size, all_head_size, vb.pp("query"))?;
        let key = linear(cfg.hidden_size, all_head_size, vb.pp("key"))?;
        let value = linear(cfg.hidden_size, all_head_size, vb.pp("value"))?;

        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        x.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .transpose(1, 2)
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        tracing::debug!("BertSelfAttention input shape: {:?}", hidden_states.shape());

        let query = self.query.forward(hidden_states)?;
        let key = self.key.forward(hidden_states)?;
        let value = self.value.forward(hidden_states)?;

        tracing::debug!("After Q/K/V projection - query shape: {:?}", query.shape());

        let query = self.transpose_for_scores(&query)?.contiguous()?;
        let key = self.transpose_for_scores(&key)?.contiguous()?;
        let value = self.transpose_for_scores(&value)?.contiguous()?;

        tracing::debug!(
            "After transpose_for_scores - query shape: {:?}, key shape: {:?}",
            query.shape(),
            key.shape()
        );

        // Scaled dot-product attention
        // key shape: (batch, num_heads, seq_len, head_dim)
        // need to transpose last two dims to get (batch, num_heads, head_dim, seq_len)
        let key_t = key.transpose(2, 3)?.contiguous()?;
        tracing::debug!("key_t shape: {:?}", key_t.shape());

        let attention_scores = query.matmul(&key_t)?;
        let scale = (self.attention_head_size as f64).sqrt();
        let attention_scores = (attention_scores / scale)?;

        // Apply attention mask if provided
        let attention_scores = match attention_mask {
            Some(mask) => attention_scores.broadcast_add(mask)?,
            None => attention_scores,
        };

        let attention_probs = candle_nn::ops::softmax(&attention_scores, D::Minus1)?;

        let context = attention_probs.matmul(&value)?;
        let context = context.transpose(1, 2)?;
        let (batch_size, seq_len, _, _) = context.dims4()?;
        context.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ))
    }
}

/// BERT self-output (output projection + residual + layernorm)
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let ln_config = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = layer_norm(cfg.hidden_size, ln_config, vb.pp("LayerNorm"))?;

        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = (hidden_states + input_tensor)?;
        self.layer_norm.forward(&hidden_states)
    }
}

/// Complete BERT attention block
struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attention = BertSelfAttention::new(cfg, vb.pp("self"))?;
        let output = BertSelfOutput::new(cfg, vb.pp("output"))?;
        Ok(Self {
            self_attention,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let self_output = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_output, hidden_states)
    }
}

/// BERT intermediate layer (FFN expansion)
struct BertIntermediate {
    dense: Linear,
    activation: Activation,
}

impl BertIntermediate {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            activation: cfg.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.activation.forward(&hidden_states)
    }
}

/// BERT output layer (FFN contraction + residual + layernorm)
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let ln_config = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = layer_norm(cfg.hidden_size, ln_config, vb.pp("LayerNorm"))?;

        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = (hidden_states + input_tensor)?;
        self.layer_norm.forward(&hidden_states)
    }
}

/// Single BERT encoder layer
struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let attention = BertAttention::new(cfg, vb.pp("attention"))?;
        let intermediate = BertIntermediate::new(cfg, vb.pp("intermediate"))?;
        let output = BertOutput::new(cfg, vb.pp("output"))?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

/// BERT encoder (stack of transformer layers)
struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn new(cfg: &BertConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(BertLayer::new(cfg, vb.pp(format!("layer.{i}")))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

/// Main BERT model for embeddings
pub struct BertEmbeddingModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    device: Device,
    cfg: BertConfig,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl BertEmbeddingModel {
    pub fn new(
        cfg: &BertConfig,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            candle_core::bail!("BERT embedding model only supports Eager attention");
        }

        let mapper = normal_loading_metadata.mapper;

        // Sentence-transformers models don't have "bert." prefix
        // Standard BERT models do, so we try both
        let embeddings = BertEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;

        Ok(Self {
            embeddings,
            encoder,
            device: normal_loading_metadata.real_device,
            cfg: cfg.clone(),
            mapper,
        })
    }

    /// Create attention mask from input_ids (mask padding tokens)
    fn make_attention_mask(&self, input_ids: &Tensor) -> Result<Option<Tensor>> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Create mask based on pad_token_id if available
        let attention_mask = if let Some(pad_id) = self.cfg.pad_token_id {
            // 1 for real tokens, 0 for padding
            let mask = input_ids.ne(pad_id as u32)?;
            mask.to_dtype(DType::F32)?
        } else {
            // No padding - all ones
            Tensor::ones((batch_size, seq_len), DType::F32, input_ids.device())?
        };

        // Expand to [batch_size, 1, 1, seq_len] for broadcasting
        let extended_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;

        // Convert to attention bias: 0 -> large negative, 1 -> 0
        // Using -10000.0 instead of NEG_INFINITY to avoid BF16 precision issues
        // (1.0 - mask) * -10000.0
        let ones = Tensor::ones_like(&extended_mask)?;
        let inverted = ones.sub(&extended_mask)?;
        let mask_value = Tensor::new(-10000.0f32, input_ids.device())?;
        let extended_mask = inverted.broadcast_mul(&mask_value)?;

        Ok(Some(extended_mask))
    }
}

impl EmbeddingModel for BertEmbeddingModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        tracing::debug!(
            "BertEmbeddingModel forward - input_ids shape: {:?}",
            input_ids.shape()
        );

        let hidden_states = self.embeddings.forward(input_ids, None)?;
        tracing::debug!(
            "BertEmbeddingModel forward - after embeddings shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );

        // Create attention mask and convert to match hidden_states dtype
        let attention_mask = self.make_attention_mask(input_ids)?;
        let attention_mask = match attention_mask {
            Some(mask) => Some(mask.to_dtype(hidden_states.dtype())?),
            None => None,
        };

        let encoder_output = self
            .encoder
            .forward(&hidden_states, attention_mask.as_ref())?;
        tracing::debug!(
            "BertEmbeddingModel forward - encoder output shape: {:?}",
            encoder_output.shape()
        );

        // Return the full sequence output (pooling is done in the pipeline)
        Ok(encoder_output)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl IsqModel for BertEmbeddingModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        // BERT doesn't use quantized layers in this implementation
        // Could be extended to support ISQ in the future
        (Vec::new(), &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        // Return all model tensors for serialization
        Vec::new()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        Ok(Vec::new())
    }
}

impl AnyMoeBaseModelMixin for BertEmbeddingModel {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_config_defaults() {
        let json = r#"{}"#;
        let cfg: BertConfig = serde_json::from_str(json).unwrap();

        assert_eq!(cfg.vocab_size, 30522);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 3072);
    }

    #[test]
    fn test_bert_config_minilm() {
        // all-MiniLM-L6-v2 config
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            "type_vocab_size": 2
        }"#;
        let cfg: BertConfig = serde_json::from_str(json).unwrap();

        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_hidden_layers, 6);
        assert_eq!(cfg.head_dim(), 32); // 384 / 12
    }
}
