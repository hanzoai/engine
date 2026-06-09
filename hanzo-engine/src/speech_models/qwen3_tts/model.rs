use std::sync::Arc;

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::{
    attention::{naive_sdpa, SdpaParams},
    layers::{self, repeat_kv, RmsNorm, RotaryEmbedding},
    utils::progress::{new_multi_progress, NiceProgressBar},
};

use super::config::{
    CodePredictorConfig, CodecConfig, CodecDecoderConfig, Qwen3TtsConfig, TalkerConfig,
};

const SNAKE_NO_DIV_BY_ZERO: f64 = 1e-9;

/// Qwen3-TTS attention (GQA + per-head q/k RMSNorm + RoPE). Mirrors `Qwen3TTSAttention`/
/// `Qwen3TTSTalkerAttention`: for the text-only TTS path the interleaved 3D mrope collapses to a
/// standard 1D RoPE (all three position rows are identical), so a plain `RotaryEmbedding` is exact.
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl Qwen3Attention {
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<RotaryEmbedding>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let q_proj = layers::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = layers::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = layers::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = layers::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;
        let mut q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?;
        let mut k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        q = q.transpose(1, 2)?.contiguous()?;
        k = k.transpose(1, 2)?.contiguous()?;

        let (q, k) = self.rope.forward(&q, &k, seqlen_offsets)?;

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
}

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

struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: SwiGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<RotaryEmbedding>,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(
            vb.pp("self_attn"),
            rope,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
        )?;
        let mlp = SwiGluMlp::new(vb.pp("mlp"), hidden_size, intermediate_size)?;
        let input_layernorm = RmsNorm::new(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::new(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, seqlen_offsets, mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.mlp.forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

/// `Qwen3TTSTalkerResizeMLP`: text-token embedding (text_hidden_size) -> talker hidden_size.
/// `fc2(act(fc1(x)))`, both linears have bias; act is silu.
struct ResizeMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ResizeMlp {
    fn new(
        vb: ShardedVarBuilder,
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            fc1: layers::linear(input_size, intermediate_size, vb.pp("linear_fc1"))?,
            fc2: layers::linear(intermediate_size, output_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.silu()?)
    }
}

/// The autoregressive Qwen3-TTS talker backbone. Consumes pre-summed input embeddings and emits
/// hidden states + codebook-0 logits per frame.
pub struct Talker {
    text_embed: Embedding,
    text_projection: ResizeMlp,
    codec_embed: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    codec_head: Linear,
}

impl Talker {
    pub fn new(cfg: &TalkerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings.max(32768),
            vb.device(),
            true,
            vb.dtype(),
        )?);

        let vb_model = vb.pp("model");
        let text_embed = layers::embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            vb_model.pp("text_embedding"),
            &None,
        )?;
        let codec_embed = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_model.pp("codec_embedding"),
            &None,
        )?;
        let text_projection = ResizeMlp::new(
            vb.pp("text_projection"),
            cfg.text_hidden_size,
            cfg.text_hidden_size,
            cfg.hidden_size,
        )?;

        let vb_l = vb_model.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading talker layers",
            &new_multi_progress(),
        )
        .run(false, |i| {
            Qwen3DecoderLayer::new(
                vb_l.pp(i),
                rope.clone(),
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
                cfg.rms_norm_eps,
            )
        })?;

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;
        let codec_head = layers::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("codec_head"))?;

        Ok(Self {
            text_embed,
            text_projection,
            codec_embed,
            layers,
            norm,
            codec_head,
        })
    }

    /// Text-token ids -> talker hidden space: `text_projection(text_embedding(ids))`.
    pub fn embed_text(&self, ids: &Tensor) -> Result<Tensor> {
        let raw = self.text_embed.forward(ids)?;
        self.text_projection.forward(&raw)
    }

    pub fn embed_codec(&self, ids: &Tensor) -> Result<Tensor> {
        self.codec_embed.forward(ids)
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut xs = inputs_embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, seqlen_offsets, mask)?;
        }
        let hidden = self.norm.forward(&xs)?;
        let logits = self.codec_head.forward(&hidden)?;
        Ok((hidden, logits))
    }
}

/// Multi-token-prediction sub-talker. Given the talker hidden state for the current frame (group 0)
/// plus the embedding of the just-predicted code, autoregressively predicts codebooks 1..num_code_groups
/// as a single attention sequence. Per-group input embeddings and per-group output heads.
pub struct CodePredictor {
    codec_embed: Vec<Embedding>,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    heads: Vec<Linear>,
}

impl CodePredictor {
    pub fn new(cfg: &CodePredictorConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings.max(8192),
            vb.device(),
            true,
            vb.dtype(),
        )?);

        let vb_model = vb.pp("model");
        let vb_ce = vb_model.pp("codec_embedding");
        let codec_embed = (0..cfg.num_code_groups.saturating_sub(1))
            .map(|i| layers::embedding(cfg.vocab_size, cfg.hidden_size, vb_ce.pp(i), &None))
            .collect::<Result<Vec<_>>>()?;

        let vb_l = vb_model.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                Qwen3DecoderLayer::new(
                    vb_l.pp(i),
                    rope.clone(),
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    head_dim,
                    cfg.rms_norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let vb_h = vb.pp("lm_head");
        let heads = (0..cfg.num_code_groups.saturating_sub(1))
            .map(|i| layers::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_h.pp(i)))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            codec_embed,
            layers,
            norm,
            heads,
        })
    }

    /// Embedding of group-`group` code (group 1..num_code_groups, embedding index = group-1).
    pub fn embed_group(&self, group: usize, ids: &Tensor) -> Result<Tensor> {
        self.codec_embed[group - 1].forward(ids)
    }

    /// Run the sub-talker over `inputs_embeds` and return logits for group `group` from its
    /// last position (head index = group-1, matching `lm_head[generation_steps]`).
    pub fn step(
        &self,
        inputs_embeds: &Tensor,
        mask: Option<&Tensor>,
        group: usize,
    ) -> Result<Tensor> {
        let mut xs = inputs_embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &[0], mask)?;
        }
        let hidden = self.norm.forward(&xs)?;
        self.heads[group - 1].forward(&hidden)
    }
}

/// `SnakeBeta`: x + (1/exp(beta)) * sin(x*exp(alpha))^2 over channels of an (b, c, t) tensor.
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
        // 1 / (exp(beta) + 1e-9); affine keeps the f32 dtype (vs a scalar add that promotes to f64).
        let inv_beta = beta.affine(1.0, SNAKE_NO_DIV_BY_ZERO)?.recip()?;
        let sa = xs.broadcast_mul(&alpha)?.sin()?;
        let term = sa.sqr()?.broadcast_mul(&inv_beta)?;
        xs + term
    }
}

/// Causal Conv1d (left-pad-only) matching `Qwen3TTSTokenizerV2CausalConvNet`.
/// padding = (kernel_size-1)*dilation + 1 - stride; plus a right "extra_padding" so the conv covers
/// the full input. Streaming/look-ahead is not modeled (non-streaming decode).
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
        let ideal = (n_frames.ceil() as usize - 1) * self.stride + (self.eff_kernel - self.pad_left);
        ideal.saturating_sub(len)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let len = xs.dim(D::Minus1)?;
        let extra = self.extra_padding(len);
        let padded = xs.pad_with_zeros(D::Minus1, self.pad_left, extra)?;
        self.conv.forward(&padded)
    }
}

/// Causal transposed Conv1d matching `Qwen3TTSTokenizerV2CausalTransConvNet`: standard ConvTranspose,
/// then trim `kernel - stride` from the right.
struct CausalTransConv1d {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    right_pad: usize,
}

impl CausalTransConv1d {
    fn new(
        vb: ShardedVarBuilder,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        dtype: DType,
    ) -> Result<Self> {
        let weight = vb.pp("conv").get((in_c, out_c, kernel), "weight")?.to_dtype(dtype)?;
        let bias = vb.pp("conv").get(out_c, "bias")?.to_dtype(dtype)?;
        Ok(Self {
            weight,
            bias,
            stride,
            right_pad: kernel - stride,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = xs.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let out = out.broadcast_add(&self.bias.reshape((1, (), 1))?)?;
        let len = out.dim(D::Minus1)?;
        if self.right_pad > 0 {
            out.narrow(D::Minus1, 0, len - self.right_pad)
        } else {
            Ok(out)
        }
    }
}

/// EMA-codebook decode: `embedding = embedding_sum / cluster_usage.clamp(min=eps)`, then gather.
struct EuclideanCodebook {
    embedding: Tensor,
}

impl EuclideanCodebook {
    fn new(vb: ShardedVarBuilder, codebook_size: usize, dim: usize, eps: f64) -> Result<Self> {
        let usage = vb
            .get(codebook_size, "cluster_usage")?
            .to_dtype(DType::F32)?
            .clamp(eps, f64::INFINITY)?;
        let sum = vb
            .get((codebook_size, dim), "embedding_sum")?
            .to_dtype(DType::F32)?;
        let embedding = sum.broadcast_div(&usage.unsqueeze(1)?)?;
        Ok(Self { embedding })
    }

    /// codes: (b, t) int -> (b, t, dim)
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (b, t) = codes.dims2()?;
        let flat = codes.reshape((b * t,))?;
        let out = self.embedding.index_select(&flat, 0)?;
        out.reshape((b, t, self.embedding.dim(1)?))
    }
}

/// A single RVQ stage (one codebook + 1x1 input/output conv projections). project_out is Identity
/// in this checkpoint (codebook_dim == inner dim), so only the codebook + output_proj are used here.
struct ResidualVectorQuantizer {
    codebooks: Vec<EuclideanCodebook>,
    output_proj: Conv1d,
}

impl ResidualVectorQuantizer {
    fn new(
        vb: ShardedVarBuilder,
        n_q: usize,
        codebook_size: usize,
        inner_dim: usize,
        output_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        let vb_vq = vb.pp("vq").pp("layers");
        let codebooks = (0..n_q)
            .map(|i| {
                EuclideanCodebook::new(vb_vq.pp(i).pp("_codebook"), codebook_size, inner_dim, eps)
            })
            .collect::<Result<Vec<_>>>()?;
        let output_proj = layers::conv1d_no_bias(
            inner_dim,
            output_dim,
            1,
            Conv1dConfig::default(),
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            codebooks,
            output_proj,
        })
    }

    /// codes: (b, n_q, t) -> (b, output_dim, t)
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (i, cb) in self.codebooks.iter().enumerate() {
            let idx = codes.i((.., i, ..))?.contiguous()?;
            let q = cb.decode(&idx)?.transpose(1, 2)?;
            sum = Some(match sum {
                Some(s) => (s + q)?,
                None => q,
            });
        }
        self.output_proj.forward(&sum.unwrap())
    }
}

/// SplitResidualVectorQuantizer: semantic (rvq_first, n_q_semantic codebooks) + acoustic (rvq_rest).
struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    fn new(vb: ShardedVarBuilder, cfg: &CodecDecoderConfig) -> Result<Self> {
        let inner = cfg.codebook_dim / 2;
        let n_q_semantic = cfg.num_semantic_quantizers;
        let n_q_acoustic = cfg.num_quantizers - n_q_semantic;
        let rvq_first = ResidualVectorQuantizer::new(
            vb.pp("rvq_first"),
            n_q_semantic,
            cfg.codebook_size,
            inner,
            cfg.codebook_dim,
            cfg.rms_norm_eps,
        )?;
        let rvq_rest = ResidualVectorQuantizer::new(
            vb.pp("rvq_rest"),
            n_q_acoustic,
            cfg.codebook_size,
            inner,
            cfg.codebook_dim,
            cfg.rms_norm_eps,
        )?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q_semantic,
        })
    }

    /// codes: (b, num_quantizers, t) -> (b, codebook_dim, t)
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let total = codes.dim(1)?;
        let first = codes.narrow(1, 0, self.n_q_semantic)?;
        let mut out = self.rvq_first.decode(&first)?;
        if total > self.n_q_semantic {
            let rest = codes.narrow(1, self.n_q_semantic, total - self.n_q_semantic)?;
            out = (out + self.rvq_rest.decode(&rest)?)?;
        }
        Ok(out)
    }
}

/// LayerScale: diagonal learnt scale on residual outputs.
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

/// One pre_transformer layer (RMSNorm + GQA + RoPE + layer-scale, then RMSNorm + SwiGLU + layer-scale).
/// Self-attention has no q/k norm and no bias (`q_norm = Identity`).
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
        cfg: &CodecDecoderConfig,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
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

/// The pre_transformer refinement stack: input_proj (latent->hidden), N transformer layers, norm,
/// output_proj (hidden->latent). Operates on (b, t, latent).
struct CodecPreTransformer {
    input_proj: Linear,
    layers: Vec<CodecTransformerLayer>,
    norm: RmsNorm,
    output_proj: Linear,
    sliding_window: Option<usize>,
}

impl CodecPreTransformer {
    fn new(vb: ShardedVarBuilder, cfg: &CodecDecoderConfig) -> Result<Self> {
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            cfg.head_dim,
            cfg.max_position_embeddings.max(8192),
            vb.device(),
            true,
            vb.dtype(),
        )?);
        let input_proj = layers::linear(cfg.latent_dim, cfg.hidden_size, vb.pp("input_proj"))?;
        let vb_l = vb.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| CodecTransformerLayer::new(vb_l.pp(i), rope.clone(), cfg))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let output_proj = layers::linear(cfg.hidden_size, cfg.latent_dim, vb.pp("output_proj"))?;
        Ok(Self {
            input_proj,
            layers,
            norm,
            output_proj,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.input_proj.forward(xs)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }
        let h = self.norm.forward(&h)?;
        self.output_proj.forward(&h)
    }

    fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }
}

/// ConvNeXt block: depthwise causal conv (k=7, groups=dim), LayerNorm, pwconv1 (->4*dim), GELU,
/// pwconv2 (->dim), gamma scale, residual. Operates on (b, dim, t).
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
        let norm = layers::layer_norm(dim, 1e-6, vb.pp("norm"))?;
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

/// One decoder.{1..} block: SnakeBeta -> CausalTransConv (upsample by rate) -> 3 residual units
/// (dilations 1,3,9). Operates on (b, c, t).
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
        dtype: DType,
    ) -> Result<Self> {
        let vb_b = vb.pp("block");
        let pre_act = SnakeBeta::new(vb_b.pp(0), in_dim)?;
        let upconv =
            CausalTransConv1d::new(vb_b.pp(1), in_dim, out_dim, 2 * upsample_rate, upsample_rate, dtype)?;
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

/// SnakeBeta -> CausalConv(k=7, dilation) -> SnakeBeta -> CausalConv(k=1), residual.
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

/// The full 12.5 Hz neural codec decoder: SplitRVQ -> pre_conv -> pre_transformer -> upsample ->
/// conv decoder stack -> 24 kHz waveform.
pub struct CodecDecoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer: CodecPreTransformer,
    upsample: Vec<(CausalTransConv1d, ConvNeXtBlock)>,
    decoder_in: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_act: SnakeBeta,
    decoder_out: CausalConv1d,
    cfg: CodecConfig,
}

impl CodecDecoder {
    pub fn new(cfg: &CodecConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dc = &cfg.decoder_config;
        let dtype = vb.dtype();
        let vb_d = vb.pp("decoder");

        let quantizer = SplitResidualVectorQuantizer::new(vb_d.pp("quantizer"), dc)?;
        let pre_conv = CausalConv1d::new(vb_d.pp("pre_conv"), dc.codebook_dim, dc.latent_dim, 3, 1, 1, 1)?;
        let pre_transformer = CodecPreTransformer::new(vb_d.pp("pre_transformer"), dc)?;

        let vb_up = vb_d.pp("upsample");
        let mut upsample = Vec::with_capacity(dc.upsampling_ratios.len());
        for (i, &factor) in dc.upsampling_ratios.iter().enumerate() {
            let vb_i = vb_up.pp(i);
            let tconv =
                CausalTransConv1d::new(vb_i.pp(0), dc.latent_dim, dc.latent_dim, factor, factor, dtype)?;
            let cnx = ConvNeXtBlock::new(vb_i.pp(1), dc.latent_dim)?;
            upsample.push((tconv, cnx));
        }

        let vb_dec = vb_d.pp("decoder");
        let decoder_in = CausalConv1d::new(vb_dec.pp(0), dc.latent_dim, dc.decoder_dim, 7, 1, 1, 1)?;
        let mut decoder_blocks = Vec::with_capacity(dc.upsample_rates.len());
        for (i, &rate) in dc.upsample_rates.iter().enumerate() {
            let in_dim = dc.decoder_dim >> i;
            let out_dim = dc.decoder_dim >> (i + 1);
            decoder_blocks.push(DecoderBlock::new(vb_dec.pp(i + 1), in_dim, out_dim, rate, dtype)?);
        }
        let out_dim = dc.decoder_dim >> dc.upsample_rates.len();
        let tail_idx = dc.upsample_rates.len() + 1;
        let decoder_act = SnakeBeta::new(vb_dec.pp(tail_idx), out_dim)?;
        let decoder_out = CausalConv1d::new(vb_dec.pp(tail_idx + 1), out_dim, 1, 7, 1, 1, 1)?;

        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            decoder_in,
            decoder_blocks,
            decoder_act,
            decoder_out,
            cfg: cfg.clone(),
        })
    }

    /// codes: (b, num_quantizers, t) -> pcm (b, 1, samples). The whole codec decode runs in f32.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut h = self.quantizer.decode(codes)?;
        h = self.pre_conv.forward(&h)?;
        h = h.transpose(1, 2)?.contiguous()?;

        let t = h.dim(1)?;
        let mask = causal_mask_sliding(t, self.pre_transformer.sliding_window(), DType::F32, h.device())?;
        h = self.pre_transformer.forward(&h, Some(&mask))?;

        h = h.transpose(1, 2)?.contiguous()?;
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

    /// Decode that also returns intermediate stages for numerical validation against the reference:
    /// (quant, pre_conv-transposed, pre_transformer-out, post-upsample, final wav).
    #[cfg(test)]
    pub fn decode_debug(
        &self,
        codes: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let quant = self.quantizer.decode(codes)?;
        let pc = self.pre_conv.forward(&quant)?;
        let pc_t = pc.transpose(1, 2)?.contiguous()?;
        let t = pc_t.dim(1)?;
        let mask =
            causal_mask_sliding(t, self.pre_transformer.sliding_window(), DType::F32, pc_t.device())?;
        let pt = self.pre_transformer.forward(&pc_t, Some(&mask))?;
        let mut h = pt.transpose(1, 2)?.contiguous()?;
        for (tconv, cnx) in &self.upsample {
            h = tconv.forward(&h)?;
            h = cnx.forward(&h)?;
        }
        let up = h.clone();
        h = self.decoder_in.forward(&h)?;
        for block in &self.decoder_blocks {
            h = block.forward(&h)?;
        }
        h = self.decoder_act.forward(&h)?;
        h = self.decoder_out.forward(&h)?;
        let wav = h.clamp(-1f32, 1f32)?;
        Ok((quant, pc_t, pt, up, wav))
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.output_sample_rate
    }

    pub fn total_upsample(&self) -> usize {
        let dc = &self.cfg.decoder_config;
        dc.upsample_rates.iter().product::<usize>() * dc.upsampling_ratios.iter().product::<usize>()
    }
}

/// Top-level model: talker, sub-talker code predictor, codec decoder.
pub struct Qwen3TtsModel {
    pub talker: Talker,
    pub code_predictor: CodePredictor,
    pub codec: CodecDecoder,
}

impl Qwen3TtsModel {
    pub fn new(
        cfg: &Qwen3TtsConfig,
        codec_cfg: &CodecConfig,
        vb: ShardedVarBuilder,
        codec_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let talker = Talker::new(&cfg.talker_config, vb.pp("talker"))?;
        let code_predictor = CodePredictor::new(
            &cfg.talker_config.code_predictor_config,
            vb.pp("talker").pp("code_predictor"),
        )?;
        let codec = CodecDecoder::new(codec_cfg, codec_vb)?;
        Ok(Self {
            talker,
            code_predictor,
            codec,
        })
    }
}

/// Additive causal mask of shape (1, 1, t, t).
pub fn causal_mask(t: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    causal_mask_sliding(t, None, dtype, device)
}

/// Additive causal mask with optional sliding-window: position j attends to i iff
/// `i - window < j <= i` (window None = full causal).
pub fn causal_mask_sliding(
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
