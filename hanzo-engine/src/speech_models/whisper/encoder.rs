#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Module, Result, Tensor};
use hanzo_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{conv1d, layer_norm, linear, linear_no_bias, Activation};

use super::WhisperConfig;

const CONV_KERNEL: usize = 3;
const LN_EPS: f64 = 1e-5;

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    num_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(cfg: &WhisperConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.n_audio_state;
        let head_dim = dim / cfg.n_audio_head;
        // openai-whisper MultiHeadAttention: query/value/out carry bias, key does not.
        let query = linear(dim, dim, vb.pp("query"))?;
        let key = linear_no_bias(dim, dim, vb.pp("key"))?;
        let value = linear(dim, dim, vb.pp("value"))?;
        let out = linear(dim, dim, vb.pp("out"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            num_heads: cfg.n_audio_head,
            head_dim,
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let shape = (b, t, self.num_heads, self.head_dim);
        let q = self.query.forward(xs)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = self.key.forward(xs)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = self.value.forward(xs)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        // Encoder self-attention is bidirectional: AttentionMask::None.
        let attn = Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, None, &self.sdpa_params)?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.out.forward(&attn)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &WhisperConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.n_audio_state;
        // openai-whisper MLP is Sequential(Linear, GELU, Linear) -> indices 0 and 2.
        let fc1 = linear(dim, dim * 4, vb.pp("0"))?;
        let fc2 = linear(dim * 4, dim, vb.pp("2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: Activation::Gelu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.apply(&self.act)?)
    }
}

struct ResidualBlock {
    attn: Attention,
    attn_ln: LayerNorm,
    mlp: Mlp,
    mlp_ln: LayerNorm,
}

impl ResidualBlock {
    fn new(cfg: &WhisperConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("attn"))?,
            attn_ln: layer_norm(cfg.n_audio_state, LN_EPS, vb.pp("attn_ln"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            mlp_ln: layer_norm(cfg.n_audio_state, LN_EPS, vb.pp("mlp_ln"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.attn_ln.forward(xs)?)?)?;
        &xs + self.mlp.forward(&self.mlp_ln.forward(&xs)?)?
    }
}

/// openai-whisper audio encoder: 2-layer Conv1d stem (2x time downsample) +
/// learned positional embedding + N residual attention blocks. `encode_window`
/// returns the MuseTalk feature stack: the post-conv embedding plus every block
/// output, so the per-frame slicer can reshape `window x stack` into 50 rows.
pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualBlock>,
    dtype: DType,
}

impl WhisperEncoder {
    pub fn new(cfg: &WhisperConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let conv1 = conv1d(
            cfg.n_mels,
            cfg.n_audio_state,
            CONV_KERNEL,
            Conv1dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = conv1d(
            cfg.n_audio_state,
            cfg.n_audio_state,
            CONV_KERNEL,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let positional_embedding =
            vb.get((cfg.n_audio_ctx, cfg.n_audio_state), "positional_embedding")?;
        let vb_blocks = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.n_audio_layer);
        for i in 0..cfg.n_audio_layer {
            blocks.push(ResidualBlock::new(cfg, vb_blocks.pp(i))?);
        }
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            dtype,
        })
    }

    /// `mel` is one padded window `[1, n_mels, 2*n_audio_ctx]`. Output is the
    /// stacked maps `[n_audio_layer + 1, n_audio_ctx, n_audio_state]`.
    pub fn encode_window(&self, mel: &Tensor) -> Result<Tensor> {
        let mel = mel.to_dtype(self.dtype)?;
        let xs = self.conv1.forward(&mel)?.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let pos = self
            .positional_embedding
            .narrow(0, 0, xs.dim(1)?)?
            .unsqueeze(0)?;
        let mut x = xs.broadcast_add(&pos)?;
        let mut maps = Vec::with_capacity(self.blocks.len() + 1);
        maps.push(x.squeeze(0)?);
        for block in &self.blocks {
            x = block.forward(&x)?;
            maps.push(x.squeeze(0)?);
        }
        Tensor::stack(&maps, 0)
    }
}
