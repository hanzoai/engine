#![allow(dead_code)]

//! Qwen3-Omni-MoE **Code2Wav** (`code2wav.*`): the neural-codec vocoder that renders the Talker's
//! discrete codec codes into a 24 kHz waveform.
//!
//! Structurally this is the Qwen3-TTS codec decoder (see `speech_models::qwen3_tts::model`) with the
//! quantizer stage removed: instead of an RVQ codebook decode, omni embeds the codes **directly**.
//! All 16 quantizer streams index one shared `code_embedding` table of
//! `[num_quantizers * codebook_size, hidden_size]` rows; quantizer `i`'s code `c` maps to row
//! `i * codebook_size + c`, and the per-quantizer embeddings are **averaged** (not summed) to a
//! single `[B, T, hidden]` stream (matching HF `code_embedding(codes + code_offset).mean(1)`).
//!
//! The decode pipeline (mirrors `Qwen3OmniMoeCode2Wav.forward`):
//! ```text
//! codes[B,Q,T] --embed+mean--> [B,T,H]
//!   --pre_transformer (8 Qwen3 layers, RoPE + LayerScale, sliding-window-72 causal)--> [B,T,H]
//!   --transpose--> [B,H,T]
//!   --upsample x2 (CausalTransConv + ConvNeXt)--> [B,H,2*2*T]
//!   --decoder_in conv--> [B,decoder_dim,·]
//!   --4 DecoderBlocks (SnakeBeta + CausalTransConv upsample[8,5,4,3] + 3 residual units)-->
//!   --SnakeBeta--> --decoder_out conv--> [B,1,samples]  (clamped to [-1,1])
//! ```
//!
//! The whole vocoder runs in f32 (`vb.set_dtype(DType::F32)`); codec precision matters and the
//! upstream weights are otherwise model-dtype.

use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::{
    attention::{naive_sdpa, SdpaParams},
    layers::{self, repeat_kv, RmsNorm, RotaryEmbedding},
};

use super::config::Code2WavConfig;

const SNAKE_NO_DIV_BY_ZERO: f64 = 1e-9;
const CONVNEXT_LN_EPS: f64 = 1e-6;
const SAMPLE_RATE: usize = 24000;

/// `SnakeBeta`: `x + (1/exp(beta)) * sin(x*exp(alpha))^2` over channels of a `(b, c, t)` tensor.
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn new(vb: ShardedVarBuilder, dim: usize) -> Result<Self> {
        let alpha = vb.get(dim, "alpha")?.to_dtype(DType::F32)?;
        let beta = vb.get(dim, "beta")?.to_dtype(DType::F32)?;
        Ok(Self { alpha, beta })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims3()?;
        let alpha = self.alpha.reshape((1, dims.1, 1))?.exp()?;
        let beta = self.beta.reshape((1, dims.1, 1))?.exp()?;
        // 1 / (exp(beta) + 1e-9); affine keeps f32 (vs a scalar add that promotes to f64).
        let inv_beta = beta.affine(1.0, SNAKE_NO_DIV_BY_ZERO)?.recip()?;
        let sa = xs.broadcast_mul(&alpha)?.sin()?;
        let term = sa.sqr()?.broadcast_mul(&inv_beta)?;
        xs + term
    }
}

/// Causal Conv1d (left-pad-only) matching `Qwen3OmniMoeCausalConvNet`: left pad
/// `(kernel-1)*dilation + 1 - stride`, plus a right `extra_padding` so the conv covers the input.
struct CausalConv1d {
    conv: Conv1d,
    stride: usize,
    eff_kernel: usize,
    pad_left: usize,
}

impl CausalConv1d {
    fn new(
        vb: ShardedVarBuilder,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv = layers::conv1d(in_c, out_c, kernel, cfg, vb.pp("conv"))?;
        let eff_kernel = (kernel - 1) * dilation + 1;
        Ok(Self {
            conv,
            stride,
            eff_kernel,
            pad_left: eff_kernel - stride,
        })
    }

    fn extra_padding(&self, len: usize) -> usize {
        let n_frames = (len + self.pad_left - self.eff_kernel) as f64 / self.stride as f64 + 1.0;
        let ideal =
            (n_frames.ceil() as usize - 1) * self.stride + (self.eff_kernel - self.pad_left);
        ideal.saturating_sub(len)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let len = xs.dim(D::Minus1)?;
        let extra = self.extra_padding(len);
        let padded = xs.pad_with_zeros(D::Minus1, self.pad_left, extra)?;
        self.conv.forward(&padded)
    }
}

/// Causal transposed Conv1d matching `Qwen3OmniMoeCausalTransConvNet`: standard ConvTranspose, then
/// trim `kernel - stride` from **both** ends (omni differs from qwen3_tts, which trims only right).
struct CausalTransConv1d {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    trim: usize,
}

impl CausalTransConv1d {
    fn new(
        vb: ShardedVarBuilder,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let weight = vb.pp("conv").get((in_c, out_c, kernel), "weight")?;
        let bias = vb.pp("conv").get(out_c, "bias")?;
        Ok(Self {
            weight,
            bias,
            stride,
            trim: kernel - stride,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = xs.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let out = out.broadcast_add(&self.bias.reshape((1, (), 1))?)?;
        if self.trim > 0 {
            let len = out.dim(D::Minus1)?;
            out.narrow(D::Minus1, self.trim, len - 2 * self.trim)
        } else {
            Ok(out)
        }
    }
}

/// ConvNeXt block: depthwise causal conv (k=7, groups=dim), LayerNorm, pwconv1 (->4*dim), GELU,
/// pwconv2 (->dim), gamma scale, residual. Operates on `(b, dim, t)`.
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(vb: ShardedVarBuilder, dim: usize) -> Result<Self> {
        let dwconv = CausalConv1d::new(vb.pp("dwconv"), dim, dim, 7, 1, 1, dim)?;
        let norm = layers::layer_norm(dim, CONVNEXT_LN_EPS, vb.pp("norm"))?;
        let pwconv1 = layers::linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = layers::linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let input = xs;
        let h = self.dwconv.forward(xs)?;
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma)?;
        let h = h.transpose(1, 2)?.contiguous()?;
        input + h
    }
}

/// LayerScale: diagonal learnt scale on a sublayer's residual output (`scale * x`).
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(vb: ShardedVarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            scale: vb.get(dim, "scale")?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`, all linears bias-free.
struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluMlp {
    fn new(vb: ShardedVarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: layers::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: layers::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: layers::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// One `pre_transformer` layer: RMSNorm + GQA(+RoPE) + LayerScale, then RMSNorm + SwiGLU +
/// LayerScale. No q/k norm and no attention bias (`q_norm = k_norm = Identity`). Sliding-window-72
/// causal attention on every layer.
struct CodecTransformerLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: SwiGluMlp,
    self_attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
    rope: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl CodecTransformerLayer {
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<RotaryEmbedding>,
        cfg: &Code2WavConfig,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = h / nh;
        let vb_a = vb.pp("self_attn");
        let q_proj = layers::linear_no_bias(h, nh * hd, vb_a.pp("q_proj"))?;
        let k_proj = layers::linear_no_bias(h, nkv * hd, vb_a.pp("k_proj"))?;
        let v_proj = layers::linear_no_bias(h, nkv * hd, vb_a.pp("v_proj"))?;
        let o_proj = layers::linear_no_bias(nh * hd, h, vb_a.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            input_layernorm: RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                h,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: SwiGluMlp::new(vb.pp("mlp"), h, cfg.intermediate_size)?,
            self_attn_layer_scale: LayerScale::new(vb.pp("self_attn_layer_scale"), h)?,
            mlp_layer_scale: LayerScale::new(vb.pp("mlp_layer_scale"), h)?,
            rope,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            sdpa_params: SdpaParams {
                n_kv_groups: nh / nkv,
                softcap: None,
                softmax_scale: 1.0 / (hd as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    fn attn(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (q, k) = self.rope.forward(&q, &k, &[0])?;
        let k = repeat_kv(k, self.sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v, self.sdpa_params.n_kv_groups)?;
        let attn = naive_sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            mask,
            &self.sdpa_params,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.o_proj.forward(&attn)
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.attn(&h, mask)?;
        let xs = (residual + self.self_attn_layer_scale.forward(&h)?)?;
        let residual = &xs;
        let h = self.post_attention_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        residual + self.mlp_layer_scale.forward(&h)?
    }
}

/// The `pre_transformer` refinement stack: `num_hidden_layers` transformer layers + final RMSNorm.
/// Operates directly on `(b, t, hidden_size)` — omni has no input/output projection.
struct CodecPreTransformer {
    layers: Vec<CodecTransformerLayer>,
    norm: RmsNorm,
    sliding_window: Option<usize>,
}

impl CodecPreTransformer {
    fn new(vb: ShardedVarBuilder, cfg: &Code2WavConfig) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings.max(8192),
            vb.device(),
            true,
            DType::F32,
        )?);
        let vb_l = vb.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| CodecTransformerLayer::new(vb_l.pp(i), rope.clone(), cfg))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            layers,
            norm,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }
        self.norm.forward(&h)
    }

    fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }
}

/// `SnakeBeta -> CausalConv(k=7, dilation) -> SnakeBeta -> CausalConv(k=1)`, residual.
struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl DecoderResidualUnit {
    fn new(vb: ShardedVarBuilder, dim: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(vb.pp("act1"), dim)?,
            conv1: CausalConv1d::new(vb.pp("conv1"), dim, dim, 7, dilation, 1, 1)?,
            act2: SnakeBeta::new(vb.pp("act2"), dim)?,
            conv2: CausalConv1d::new(vb.pp("conv2"), dim, dim, 1, 1, 1, 1)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.act1.forward(xs)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        h + residual
    }
}

/// One decoder block: `SnakeBeta -> CausalTransConv (upsample by rate) -> 3 residual units`
/// (dilations 1, 3, 9). Operates on `(b, c, t)`.
struct DecoderBlock {
    pre_act: SnakeBeta,
    upconv: CausalTransConv1d,
    res_units: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    fn new(
        vb: ShardedVarBuilder,
        in_dim: usize,
        out_dim: usize,
        upsample_rate: usize,
    ) -> Result<Self> {
        let vb_b = vb.pp("block");
        let pre_act = SnakeBeta::new(vb_b.pp(0), in_dim)?;
        let upconv =
            CausalTransConv1d::new(vb_b.pp(1), in_dim, out_dim, 2 * upsample_rate, upsample_rate)?;
        let mut res_units = Vec::with_capacity(3);
        for (j, dilation) in [1usize, 3, 9].into_iter().enumerate() {
            res_units.push(DecoderResidualUnit::new(vb_b.pp(2 + j), out_dim, dilation)?);
        }
        Ok(Self {
            pre_act,
            upconv,
            res_units,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.pre_act.forward(xs)?;
        h = self.upconv.forward(&h)?;
        for ru in &self.res_units {
            h = ru.forward(&h)?;
        }
        Ok(h)
    }
}

/// Additive sliding-window causal mask of shape `(1, 1, t, t)`. Query `i` attends to key `j` iff
/// `i - window < j <= i` (window `None` = full causal). Matches HF `create_sliding_window_causal_mask`
/// (`kv_idx > q_idx - sliding_window`).
fn causal_mask_sliding(
    t: usize,
    window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let masked = j > i || window.is_some_and(|w| i >= w && j <= i - w);
            if masked {
                data[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device)?.to_dtype(dtype)
}

/// Qwen3-Omni Code2Wav neural-codec vocoder: discrete codec codes -> 24 kHz waveform.
pub struct OmniCode2Wav {
    code_embedding: Embedding,
    code_offset: Tensor,
    num_quantizers: usize,
    pre_transformer: CodecPreTransformer,
    upsample: Vec<(CausalTransConv1d, ConvNeXtBlock)>,
    decoder_in: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_act: SnakeBeta,
    decoder_out: CausalConv1d,
}

impl OmniCode2Wav {
    pub fn new(cfg: &Code2WavConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        // The vocoder runs entirely in f32 regardless of the model's load dtype.
        let vb = vb.set_dtype(DType::F32);

        let hidden = cfg.hidden_size;
        let num_quantizers = cfg.num_quantizers;

        // Direct code embedding: one shared table over all quantizers, averaged per frame.
        let code_embedding = layers::embedding(
            num_quantizers * cfg.codebook_size,
            hidden,
            vb.pp("code_embedding"),
            &None,
        )?;
        let offsets: Vec<u32> = (0..num_quantizers)
            .map(|i| (i * cfg.codebook_size) as u32)
            .collect();
        let code_offset = Tensor::from_vec(offsets, (1, num_quantizers, 1), device)?;

        let pre_transformer = CodecPreTransformer::new(vb.pp("pre_transformer"), cfg)?;

        let vb_up = vb.pp("upsample");
        let mut upsample = Vec::with_capacity(cfg.upsampling_ratios.len());
        for (j, &factor) in cfg.upsampling_ratios.iter().enumerate() {
            let vb_j = vb_up.pp(j);
            let tconv = CausalTransConv1d::new(vb_j.pp(0), hidden, hidden, factor, factor)?;
            let cnx = ConvNeXtBlock::new(vb_j.pp(1), hidden)?;
            upsample.push((tconv, cnx));
        }

        let vb_dec = vb.pp("decoder");
        let decoder_dim = cfg.decoder_dim;
        let decoder_in = CausalConv1d::new(vb_dec.pp(0), hidden, decoder_dim, 7, 1, 1, 1)?;
        let mut decoder_blocks = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
            let in_dim = decoder_dim >> i;
            let out_dim = decoder_dim >> (i + 1);
            decoder_blocks.push(DecoderBlock::new(vb_dec.pp(i + 1), in_dim, out_dim, rate)?);
        }
        let out_dim = decoder_dim >> cfg.upsample_rates.len();
        let tail = cfg.upsample_rates.len() + 1;
        let decoder_act = SnakeBeta::new(vb_dec.pp(tail), out_dim)?;
        let decoder_out = CausalConv1d::new(vb_dec.pp(tail + 1), out_dim, 1, 7, 1, 1, 1)?;

        Ok(Self {
            code_embedding,
            code_offset,
            num_quantizers,
            pre_transformer,
            upsample,
            decoder_in,
            decoder_blocks,
            decoder_act,
            decoder_out,
        })
    }

    /// codes: `(b, num_quantizers, t)` int codes -> waveform `(b, 1, samples)`, clamped to `[-1, 1]`.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let q = codes.dim(1)?;
        if q != self.num_quantizers {
            hanzo_ml::bail!(
                "Code2Wav expected {} quantizer streams, got {q}",
                self.num_quantizers
            );
        }

        // Embed each quantizer stream into the shared table (offset by quantizer index), average.
        let idx = codes.to_dtype(DType::U32)?.broadcast_add(&self.code_offset)?;
        let emb = self.code_embedding.forward(&idx)?; // (b, q, t, hidden)
        let mut h = emb.mean(1)?; // (b, t, hidden)

        // Pre-transformer refinement under a sliding-window-72 causal mask.
        let t = h.dim(1)?;
        let mask =
            causal_mask_sliding(t, self.pre_transformer.sliding_window(), DType::F32, h.device())?;
        h = self.pre_transformer.forward(&h, Some(&mask))?;

        // Convolutional upsampling decoder operates channels-first.
        h = h.transpose(1, 2)?.contiguous()?; // (b, hidden, t)
        for (tconv, cnx) in &self.upsample {
            h = tconv.forward(&h)?;
            h = cnx.forward(&h)?;
        }
        h = self.decoder_in.forward(&h)?;
        for block in &self.decoder_blocks {
            h = block.forward(&h)?;
        }
        h = self.decoder_act.forward(&h)?;
        h = self.decoder_out.forward(&h)?;
        h.clamp(-1f32, 1f32)
    }

    pub fn sample_rate(&self) -> usize {
        SAMPLE_RATE
    }
}

#[cfg(test)]
mod code2wav_tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{Device, Tensor};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn read_f32_le(p: &str) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn read_i64_le(p: &str) -> Vec<i64> {
        std::fs::read(p).unwrap().chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) { d += (*x as f64)*(*y as f64); na += (*x as f64).powi(2); nb += (*y as f64).powi(2); }
        (d / (na.sqrt()*nb.sqrt())) as f32
    }

    /// Decode the FIXED codes fixture with the real `code2wav.*` weights and assert the waveform
    /// matches the HF Qwen3OmniMoeCode2Wav reference (cosine > 0.99). Env-gated on the checkpoint.
    #[test]
    fn omni_code2wav_matches_reference() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let fix = std::env::var("OMNI_FIX_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/omni_fixtures".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        if !index.is_file() || !PathBuf::from(&fix).join("c2w_wav.f32").is_file() {
            eprintln!("zen-omni weights/fixtures absent; skipping code2wav validation");
            return;
        }
        let device = Device::Cpu;
        let cfg: super::super::config::Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap()).unwrap();

        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shards = std::collections::BTreeSet::new();
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shards.insert(v.as_str().unwrap().to_string());
        }
        let paths: Vec<PathBuf> = shards.iter().map(|s| dirp.join(s)).collect();
        let vb = from_mmaped_safetensors(
            paths, Vec::new(), Some(DType::F32), &device, vec![None], true, None,
            |n: String| n.starts_with("code2wav."),
            Arc::new(|_| DeviceForLoadTensor::Base),
        ).unwrap();

        let model = OmniCode2Wav::new(&cfg.code2wav_config, vb.pp("code2wav"), &device).unwrap();

        // Fixed codes [1, 16, 32] from the oracle.
        let codes_i64 = read_i64_le(&format!("{fix}/c2w_codes.i64"));
        let q = cfg.code2wav_config.num_quantizers;
        let t = codes_i64.len() / q;
        let codes_u32: Vec<u32> = codes_i64.iter().map(|&v| v as u32).collect();
        let codes = Tensor::from_vec(codes_u32, (1, q, t), &device).unwrap();

        let wav = model.decode(&codes).unwrap();
        let got: Vec<f32> = wav.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let refv = read_f32_le(&format!("{fix}/c2w_wav.f32"));
        eprintln!("[code2wav] got {} samples, ref {} samples", got.len(), refv.len());
        let n = got.len().min(refv.len());
        let cos = cosine(&got[..n], &refv[..n]);
        eprintln!("[code2wav] wav cosine = {cos:.6} (len got={} ref={})", got.len(), refv.len());
        assert!((got.len() as i64 - refv.len() as i64).abs() <= 8, "wav length mismatch");
        assert!(cos > 0.99, "code2wav wav cosine {cos} < 0.99");
    }
}
