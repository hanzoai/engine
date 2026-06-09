#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{conv2d, layer_norm, linear, linear_no_bias, Activation};

use super::config::AudioEncoderConfig;

const CONV_KERNEL: usize = 3;
const CONV_STRIDE: usize = 2;
const CONV_PADDING: usize = 1;
const SINUSOID_BASE: f64 = 10_000.0;

/// Standard sinusoidal absolute position table `[max_pos, dim]`.
fn sinusoidal_embedding(max_pos: usize, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1f64 / SINUSOID_BASE.powf(2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let pos = Tensor::arange(0u32, max_pos as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_pos, 1))?;
    let args = pos.matmul(&inv_freq)?;
    let sin = args.sin()?;
    let cos = args.cos()?;
    Tensor::cat(&[&sin, &cos], 1)?.to_dtype(dtype)
}

struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl EncoderAttention {
    fn new(cfg: &AudioEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.d_model;
        let head_dim = cfg.head_dim();
        // AuT self-attention projections all carry bias.
        let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_heads,
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
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Bidirectional (non-causal): AttentionMask::None on the eager path.
        let attn = Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, None, &self.sdpa_params)?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.out_proj.forward(&attn)
    }
}

struct EncoderMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl EncoderMlp {
    fn new(cfg: &AudioEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.d_model, cfg.ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: Activation::Gelu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.apply(&self.act)?;
        self.fc2.forward(&xs)
    }
}

struct EncoderLayer {
    self_attn: EncoderAttention,
    mlp: EncoderMlp,
    attn_norm: LayerNorm,
    final_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &AudioEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = EncoderAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = EncoderMlp::new(cfg, vb.clone())?;
        let attn_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let final_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            mlp,
            attn_norm,
            final_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attn_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let normed = self.final_norm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&normed)?;
        residual + mlp_out
    }
}

/// Qwen3-ASR AuT audio encoder: Conv2d stem (8x downsample) + sinusoidal pos +
/// bidirectional transformer + 2-layer output projection to the LM hidden size.
pub struct Qwen3AsrAudioEncoder {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv_out: Linear,
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    pos_embed: Tensor,
    dtype: DType,
}

impl Qwen3AsrAudioEncoder {
    pub fn new(cfg: &AudioEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let conv_cfg = Conv2dConfig {
            padding: CONV_PADDING,
            stride: CONV_STRIDE,
            ..Default::default()
        };
        let conv1 = conv2d(1, cfg.conv_channels, CONV_KERNEL, conv_cfg, vb.pp("conv2d1"))?;
        let conv2 = conv2d(
            cfg.conv_channels,
            cfg.conv_channels,
            CONV_KERNEL,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv3 = conv2d(
            cfg.conv_channels,
            cfg.conv_channels,
            CONV_KERNEL,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        // Flattened conv-stem features -> d_model. `conv_out` ships no bias.
        let conv_out = linear_no_bias(cfg.conv_feature_dim(), cfg.d_model, vb.pp("conv_out"))?;

        let ln_post = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1 = linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        let pos_embed =
            sinusoidal_embedding(cfg.max_audio_positions(), cfg.d_model, &device, dtype)?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            pos_embed,
            dtype,
        })
    }

    /// Input: log-mel features `[B, n_mels, T]`. Output: `[B, T/8, output_dim]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (b, _mels, _t) = mel.dims3()?;
        // [B, n_mels, T] -> [B, 1, n_mels, T] channel dim for Conv2d.
        let xs = mel.to_dtype(self.dtype)?.unsqueeze(1)?;
        let xs = self.conv1.forward(&xs)?.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;
        let xs = self.conv3.forward(&xs)?.gelu_erf()?;

        // [B, C, F', T'] -> [B, T', C*F'] flatten freq into channels, then project to d_model.
        let (_b, c, f, t) = xs.dims4()?;
        let xs = xs
            .permute((0, 3, 1, 2))?
            .reshape((b, t, c * f))?
            .contiguous()?;
        let xs = self.conv_out.forward(&xs)?;

        let seq_len = xs.dim(1)?;
        let pos = self.pos_embed.narrow(0, 0, seq_len)?.unsqueeze(0)?;
        let mut hidden = xs.broadcast_add(&pos)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        let hidden = self.ln_post.forward(&hidden)?;
        let hidden = self.proj1.forward(&hidden)?.gelu_erf()?;
        self.proj2.forward(&hidden)
    }
}
