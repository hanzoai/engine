use std::sync::Arc;

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor};
use hanzo_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Module};
use hanzo_quant::{
    apply_immediate_isq, QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear,
};

use crate::{
    attention::{naive_sdpa, SdpaParams},
    layers::{self, repeat_kv, RmsNorm, RotaryEmbedding},
    utils::progress::{new_multi_progress, NiceProgressBar},
};

use super::config::{
    CodePredictorConfig, CodecConfig, CodecDecoderConfig, Qwen3TtsConfig, TalkerConfig,
};

fn unquant_linear(linear: hanzo_nn::Linear, vb: &ShardedVarBuilder) -> Result<Arc<dyn QuantMethod>> {
    let mut q: Arc<dyn QuantMethod> =
        Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(linear))?);
    q = apply_immediate_isq(q, vb.clone())?;
    Ok(q)
}

/// Qwen3-style attention block (GQA + per-head q/k RMSNorm + RoPE).
/// Mirrors `models::qwen3::Attention` but is self-contained for the speech path,
/// using the same `naive_sdpa` helper as the Dia speech model.
struct Qwen3Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
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

        let q_proj = unquant_linear(q_proj, &vb)?;
        let k_proj = unquant_linear(k_proj, &vb)?;
        let v_proj = unquant_linear(v_proj, &vb)?;
        let o_proj = unquant_linear(o_proj, &vb)?;

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
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
}

impl SwiGluMlp {
    fn new(vb: ShardedVarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let gate_proj = layers::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = layers::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = layers::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: unquant_linear(gate_proj, &vb)?,
            up_proj: unquant_linear(up_proj, &vb)?,
            down_proj: unquant_linear(down_proj, &vb)?,
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

/// The autoregressive Qwen3 backbone that consumes interleaved text + codec embeddings
/// and emits hidden states + codebook-0 logits per frame.
pub struct Talker {
    text_embed: Embedding,
    codec_embed: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    codec_head: Arc<dyn QuantMethod>,
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

        let text_embed = layers::embedding(
            cfg.text_vocab_size,
            cfg.hidden_size,
            vb.pp("text_embedding"),
            &None,
        )?;
        let codec_embed =
            layers::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("codec_embedding"), &None)?;

        let vb_l = vb.pp("model").pp("layers");
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

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model").pp("norm"))?;

        let codec_head =
            layers::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("codec_head"))?;
        let codec_head = unquant_linear(codec_head, &vb)?;

        Ok(Self {
            text_embed,
            codec_embed,
            layers,
            norm,
            codec_head,
        })
    }

    pub fn embed_text(&self, ids: &Tensor) -> Result<Tensor> {
        self.text_embed.forward(ids)
    }

    pub fn embed_codec(&self, ids: &Tensor) -> Result<Tensor> {
        self.codec_embed.forward(ids)
    }

    /// Runs the backbone over `inputs_embeds` and returns (last_hidden_state, codebook0_logits).
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

/// Multi-token-prediction head: a small transformer that, given the talker hidden state for a frame,
/// autoregressively predicts codebooks 1..num_code_groups.
pub struct CodePredictor {
    small_to_mtp_projection: Arc<dyn QuantMethod>,
    codec_embed: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    heads: Vec<Arc<dyn QuantMethod>>,
}

impl CodePredictor {
    pub fn new(
        cfg: &CodePredictorConfig,
        talker_hidden_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings.max(8192),
            vb.device(),
            true,
            vb.dtype(),
        )?);

        let small_to_mtp_projection = layers::linear_no_bias(
            talker_hidden_size,
            cfg.hidden_size,
            vb.pp("small_to_mtp_projection"),
        )?;
        let small_to_mtp_projection = unquant_linear(small_to_mtp_projection, &vb)?;

        let codec_embed =
            layers::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("codec_embedding"), &None)?;

        let vb_l = vb.pp("model").pp("layers");
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

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model").pp("norm"))?;

        let vb_h = vb.pp("heads");
        let heads = (0..cfg.num_code_groups.saturating_sub(1))
            .map(|i| {
                let l = layers::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_h.pp(i))?;
                unquant_linear(l, &vb)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            small_to_mtp_projection,
            codec_embed,
            layers,
            norm,
            heads,
        })
    }

    pub fn project(&self, talker_hidden: &Tensor) -> Result<Tensor> {
        self.small_to_mtp_projection.forward(talker_hidden)
    }

    /// One MTP step: embed previously-predicted code, run the transformer, return logits for group `group_idx`.
    pub fn step(
        &self,
        inputs_embeds: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
        group_idx: usize,
    ) -> Result<Tensor> {
        let mut xs = inputs_embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, seqlen_offsets, mask)?;
        }
        let hidden = self.norm.forward(&xs)?;
        let head = &self.heads[group_idx.min(self.heads.len() - 1)];
        head.forward(&hidden)
    }

    pub fn embed_codec(&self, ids: &Tensor) -> Result<Tensor> {
        self.codec_embed.forward(ids)
    }
}

/// Residual vector quantizer that maps codebook indices back to continuous latents.
struct ResidualVectorQuantizer {
    codebooks: Vec<Embedding>,
    out_proj: Conv1d,
}

impl ResidualVectorQuantizer {
    fn new(cfg: &CodecDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let vb_q = vb.pp("quantizers");
        let mut codebooks = Vec::with_capacity(cfg.num_quantizers);
        for i in 0..cfg.num_quantizers {
            let size = if i < cfg.num_semantic_quantizers {
                cfg.semantic_codebook_size
            } else {
                cfg.codebook_size
            };
            codebooks.push(layers::embedding(
                size,
                cfg.codebook_dim,
                vb_q.pp(i).pp("codebook"),
                &None,
            )?);
        }
        let out_proj = layers::conv1d_no_bias(
            cfg.codebook_dim,
            cfg.latent_dim,
            1,
            Conv1dConfig::default(),
            vb.pp("out_proj"),
        )?;
        Ok(Self {
            codebooks,
            out_proj,
        })
    }

    /// codes: (b, num_quantizers, t) -> latents (b, latent_dim, t)
    fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (i, cb) in self.codebooks.iter().enumerate() {
            let idx = codes.i((.., i, ..))?;
            let emb = cb.forward(&idx)?;
            sum = Some(match sum {
                Some(s) => (s + emb)?,
                None => emb,
            });
        }
        let latents = sum.unwrap().transpose(1, 2)?;
        self.out_proj.forward(&latents)
    }
}

/// Streaming conv upsampler that renders latents to a 24 kHz PCM waveform.
struct CodecUpsampler {
    conv_in: Conv1d,
    blocks: Vec<ConvTranspose1d>,
    conv_out: Conv1d,
}

impl CodecUpsampler {
    fn new(cfg: &CodecDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut channels = cfg.decoder_dim;
        let conv_in = layers::conv1d_no_bias(
            cfg.latent_dim,
            channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_in"),
        )?;

        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, stride) in cfg.upsample_rates.iter().enumerate() {
            let out_c = (channels / 2).max(1);
            let cfg_t = ConvTranspose1dConfig {
                stride: *stride,
                padding: stride.div_ceil(2),
                ..Default::default()
            };
            let weight = vb_b
                .pp(i)
                .get((channels, out_c, 2 * stride), "weight")?;
            let bias = vb_b.pp(i).get(out_c, "bias")?;
            blocks.push(ConvTranspose1d::new(weight, Some(bias), cfg_t));
            channels = out_c;
        }

        let conv_out = layers::conv1d(
            channels,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

    fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv_in.forward(latents)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?.gelu_erf()?;
        }
        self.conv_out.forward(&xs)
    }
}

/// The 12.5 Hz neural codec decoder: RVQ codes -> latents -> waveform.
/// Note: the upstream decoder also has a `num_hidden_layers`-deep transformer over the
/// latents before upsampling; that refinement stack is not yet wired here (see module notes).
pub struct CodecDecoder {
    quantizer: ResidualVectorQuantizer,
    upsampler: CodecUpsampler,
    cfg: CodecConfig,
}

impl CodecDecoder {
    pub fn new(cfg: &CodecConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let vb_d = vb.pp("decoder");
        let quantizer = ResidualVectorQuantizer::new(&cfg.decoder_config, vb_d.pp("quantizer"))?;
        let upsampler = CodecUpsampler::new(&cfg.decoder_config, vb_d.pp("upsampler"))?;
        Ok(Self {
            quantizer,
            upsampler,
            cfg: cfg.clone(),
        })
    }

    /// codes: (b, num_quantizers, t) -> pcm (b, 1, samples)
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let latents = self.quantizer.from_codes(codes)?;
        self.upsampler.forward(&latents)
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.output_sample_rate
    }
}

/// Top-level model bundling the talker, the MTP code predictor, and the codec decoder.
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
            cfg.talker_config.hidden_size,
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

/// Builds an additive causal mask of shape (1, 1, t, t) for the speech backbones.
pub fn causal_mask(t: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device)?.to_dtype(dtype)
}
