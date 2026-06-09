#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::Linear;
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{
    embedding, linear_no_bias, Activation, CausalMaskConfig, CausalMasker, RmsNorm, RotaryEmbedding,
};
use crate::layers_masker::PastKvLenCache;

use super::config::TextDecoderConfig;

/// Past-length source for the prefill scaffold: no cached tokens yet.
struct NoPastKv;
impl PastKvLenCache for NoPastKv {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(0)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(cfg: &TextDecoderConfig, rotary_emb: Arc<RotaryEmbedding>, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_proj = linear_no_bias(hidden, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(&self, xs: &Tensor, mask: &AttentionMask, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm on Q and K, then NeoX-style RoPE. `seqlen_offsets` is one start
        // offset per batch row; the cache is walked `offset..offset+seq_len` internally.
        let (q, k) = self.rotary_emb.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            seqlen_offsets,
        )?;

        let attn = Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?;
        let attn = if !matches!(mask, AttentionMask::None) {
            attn.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &TextDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(hidden, inter, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, inter, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(inter, hidden, vb.pp("down_proj"))?,
            act: Activation::Silu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextDecoderConfig, rotary_emb: Arc<RotaryEmbedding>, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &AttentionMask, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, seqlen_offsets)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// Qwen3 text decoder (`thinker.model`) + tied `lm_head`. Prefill-only forward:
/// streaming/incremental decode (KV cache reuse) is future work.
pub struct Qwen3AsrTextDecoder {
    embed_tokens: hanzo_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: Option<usize>,
}

impl Qwen3AsrTextDecoder {
    pub fn new(cfg: &TextDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"), &None)?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            &device,
            true, // Qwen3 uses NeoX-style (rotate_half) RoPE.
            dtype,
        )?);

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary_emb.clone(), vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
            dtype,
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Run the decoder over precomputed input embeddings (audio already merged in).
    /// `positions` is a 1-D u32 tensor of per-batch-row RoPE start offsets; perf's
    /// `forward_qk_norm` walks `offset..offset+seq_len` internally, so a prefill from
    /// position 0 is expressed as `[0]`.
    pub fn forward_embeds(&self, input_embeds: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = input_embeds.dims3()?;
        let seqlen_offsets = positions
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|p| p as usize)
            .collect::<Vec<_>>();
        let dummy = Tensor::zeros((b_sz, seq_len), DType::U32, input_embeds.device())?;
        let mask = CausalMasker.make_causal_mask(
            &dummy,
            &NoPastKv,
            input_embeds.dtype(),
            &CausalMaskConfig {
                sliding_window: self.sliding_window,
                ..Default::default()
            },
        )?;

        let mut xs = input_embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask, &seqlen_offsets)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}
