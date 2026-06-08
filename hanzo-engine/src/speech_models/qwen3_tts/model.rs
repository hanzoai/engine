#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// Qwen3-TTS (zen3-tts) model. A Qwen3 "talker" backbone autoregressively emits primary codec
// tokens (codebook 0); a small 5-layer "code predictor" emits the 15 residual codebooks per frame;
// the WavTokenizer-style codec decoder turns the 16-codebook frames into a 24kHz waveform.
//
// Milestone-1 scope: greedy/sampled single-stream synthesis on one device, naive eager attention,
// standard interleaved RoPE in place of full 3D MRoPE (text + audio are 1D temporal streams, so the
// three MRoPE position sections collapse to identical 1D positions when no vision is present).

use std::sync::Arc;

use hanzo_ml::{DType, Device, IndexOp, Module, Result, Tensor, D};
use hanzo_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{naive_sdpa, SdpaParams};
use crate::layers::{embedding, linear, linear_no_bias, repeat_kv, RmsNorm};

use super::config::{
    CodePredictorConfig, CodecDecoderConfig, Qwen3TtsCodecConfig, Qwen3TtsConfig, TalkerConfig,
};

// ---- interleaved RoPE -------------------------------------------------------

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(
        head_dim: usize,
        max_pos: usize,
        theta: f64,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let t = Tensor::arange(0u32, max_pos as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?; // [max_pos, half]
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    // x: [b, h, t, d], positions start at `offset`. Interleaved layout (even/odd pairs).
    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, _h, t, d) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, t)?; // [t, d/2]
        let sin = self.sin.narrow(0, offset, t)?;
        let cos = Tensor::stack(&[&cos, &cos], D::Minus1)?.reshape((t, d))?;
        let sin = Tensor::stack(&[&sin, &sin], D::Minus1)?.reshape((t, d))?;
        let cos = cos.reshape((1, 1, t, d))?;
        let sin = sin.reshape((1, 1, t, d))?;
        let xr = Self::rotate_half_interleaved(x)?;
        x.broadcast_mul(&cos)? + xr.broadcast_mul(&sin)?
    }

    fn rotate_half_interleaved(x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let d = dims[dims.len() - 1];
        let x = x.reshape((dims[0], dims[1], dims[2], d / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let rot = Tensor::cat(&[&x1.neg()?, &x0], D::Minus1)?;
        rot.reshape((dims[0], dims[1], dims[2], d))
    }
}

// ---- shared Qwen3 transformer block (used by talker + code predictor) ------

struct Qwen3Attention {
    q_proj: hanzo_nn::Linear,
    k_proj: hanzo_nn::Linear,
    v_proj: hanzo_nn::Linear,
    o_proj: hanzo_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Arc<Rope>,
    sdpa: SdpaParams,
}

impl Qwen3Attention {
    fn new(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f64,
        rope: Arc<Rope>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let q_proj = linear_no_bias(hidden, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, eps, vb.pp("k_norm"))?;
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
            rope,
            sdpa: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    // Full-sequence (prefill-style) forward with causal mask. No KV cache: milestone-1 recomputes
    // the whole prefix each step. Correct but O(n^2) per token.
    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;

        let q = self.rope.apply(&q.contiguous()?, offset)?;
        let k = self.rope.apply(&k.contiguous()?, offset)?;

        let k = repeat_kv(k, self.sdpa.n_kv_groups)?;
        let v = repeat_kv(v, self.sdpa.n_kv_groups)?;

        let attn = naive_sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            mask,
            &self.sdpa,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.o_proj.forward(&attn)
    }
}

struct Qwen3Mlp {
    gate: hanzo_nn::Linear,
    up: hanzo_nn::Linear,
    down: hanzo_nn::Linear,
}

impl Qwen3Mlp {
    fn new(hidden: usize, inter: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(hidden, inter, vb.pp("gate_proj"))?,
            up: linear_no_bias(hidden, inter, vb.pp("up_proj"))?,
            down: linear_no_bias(inter, hidden, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Qwen3Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(xs)?.silu()?;
        let u = self.up.forward(xs)?;
        self.down.forward(&(g * u)?)
    }
}

struct Qwen3Layer {
    attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_ln: RmsNorm,
    post_ln: RmsNorm,
}

impl Qwen3Layer {
    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_ln.forward(xs)?;
        let h = self.attn.forward(&h, mask, offset)?;
        let xs = (residual + h)?;
        let residual = &xs;
        let h = self.mlp.forward(&self.post_ln.forward(&xs)?)?;
        residual + h
    }
}

struct LayerSpec {
    n: usize,
    hidden: usize,
    inter: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

fn build_layers(spec: &LayerSpec, rope: &Arc<Rope>, vb: ShardedVarBuilder) -> Result<Vec<Qwen3Layer>> {
    let mut layers = Vec::with_capacity(spec.n);
    for i in 0..spec.n {
        let vb_l = vb.pp(i);
        let attn = Qwen3Attention::new(
            spec.hidden,
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.eps,
            rope.clone(),
            vb_l.pp("self_attn"),
        )?;
        let mlp = Qwen3Mlp::new(spec.hidden, spec.inter, vb_l.pp("mlp"))?;
        let input_ln = RmsNorm::new(spec.hidden, spec.eps, vb_l.pp("input_layernorm"))?;
        let post_ln = RmsNorm::new(spec.hidden, spec.eps, vb_l.pp("post_attention_layernorm"))?;
        layers.push(Qwen3Layer {
            attn,
            mlp,
            input_ln,
            post_ln,
        });
    }
    Ok(layers)
}

fn causal_mask(t: usize, offset: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let total = t + offset;
    let mut data = vec![0f32; t * total];
    for i in 0..t {
        for j in 0..total {
            if j > i + offset {
                data[i * total + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, t, total), dev)?.to_dtype(dtype)
}

// ---- code predictor (5-layer Qwen3 over the 16 codebook groups) ------------

struct CodePredictor {
    codec_embeddings: Vec<hanzo_nn::Embedding>,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    lm_heads: Vec<hanzo_nn::Linear>,
    num_code_groups: usize,
    eps: f64,
}

impl CodePredictor {
    fn new(
        cfg: &CodePredictorConfig,
        dtype: DType,
        dev: &Device,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let rope = Arc::new(Rope::new(cfg.head_dim, 64, cfg.rope_theta, dtype, dev)?);
        let vb_m = vb.pp("model");
        // Residual predictor: 15 codebook tables / lm_heads (one per residual group 1..=15), each
        // over `vocab_size` (2048). Group 0 is handled by the talker's own codec_head.
        let n_residual = cfg.num_code_groups - 1;
        let mut codec_embeddings = Vec::with_capacity(n_residual);
        let vb_ce = vb_m.pp("codec_embedding");
        for i in 0..n_residual {
            codec_embeddings.push(embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb_ce.pp(i),
                &None,
            )?);
        }
        let layers = build_layers(
            &LayerSpec {
                n: cfg.num_hidden_layers,
                hidden: cfg.hidden_size,
                inter: cfg.intermediate_size,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                eps: cfg.rms_norm_eps,
            },
            &rope,
            vb_m.pp("layers"),
        )?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let mut lm_heads = Vec::with_capacity(n_residual);
        let vb_h = vb.pp("lm_head");
        for i in 0..n_residual {
            lm_heads.push(linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_h.pp(i))?);
        }
        Ok(Self {
            codec_embeddings,
            layers,
            norm,
            lm_heads,
            num_code_groups: cfg.num_code_groups,
            eps: cfg.rms_norm_eps,
        })
    }

    // Given the talker hidden state for one frame (`talker_hidden`: [1,1,H]), autoregressively emit
    // residual codebooks 1..=15. Returns all 16 codebook ids for the frame (group 0 = `cb0`).
    fn predict_frame(&self, talker_hidden: &Tensor, cb0: u32, dev: &Device) -> Result<Vec<u32>> {
        let dtype = talker_hidden.dtype();
        let mut codes = vec![cb0];
        // Sequence starts from the talker hidden state; each step appends the embedding of the
        // just-sampled residual code via that residual group's table, and predicts the next group.
        let mut seq = talker_hidden.clone(); // [1,1,H]
        let mut last = cb0;
        for g in 0..self.codec_embeddings.len() {
            let emb = self.codec_embeddings[g]
                .forward(&Tensor::from_vec(vec![last], (1, 1), dev)?)?
                .to_dtype(dtype)?;
            seq = Tensor::cat(&[&seq, &emb], 1)?;
            let t = seq.dim(1)?;
            let mask = causal_mask(t, 0, dtype, dev)?;
            let mut h = seq.clone();
            for layer in &self.layers {
                h = layer.forward(&h, Some(&mask), 0)?;
            }
            h = self.norm.forward(&h)?;
            let last_h = h.i((.., t - 1, ..))?;
            let logits = self.lm_heads[g].forward(&last_h)?;
            let id = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
            codes.push(id);
            last = id;
        }
        let _ = (self.num_code_groups, self.eps);
        Ok(codes)
    }
}

// ---- talker (28-layer Qwen3) -----------------------------------------------

pub struct Talker {
    text_embedding: hanzo_nn::Embedding,
    text_proj_fc1: hanzo_nn::Linear,
    text_proj_fc2: hanzo_nn::Linear,
    codec_embedding: hanzo_nn::Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    codec_head: hanzo_nn::Linear,
    code_predictor: CodePredictor,
    cfg: TalkerConfig,
    dtype: DType,
    device: Device,
}

impl Talker {
    pub fn new(
        cfg: &Qwen3TtsConfig,
        dtype: DType,
        dev: &Device,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let tc = &cfg.talker_config;
        let rope = Arc::new(Rope::new(
            tc.head_dim,
            tc.max_position_embeddings.min(8192),
            tc.rope_theta,
            dtype,
            dev,
        )?);
        let vb_t = vb.pp("talker");
        let vb_m = vb_t.pp("model");

        let text_embedding = embedding(
            tc.text_vocab_size,
            tc.text_hidden_size,
            vb_m.pp("text_embedding"),
            &None,
        )?;
        let vb_tp = vb_t.pp("text_projection");
        let text_proj_fc1 = linear(
            tc.text_hidden_size,
            tc.text_hidden_size,
            vb_tp.pp("linear_fc1"),
        )?;
        let text_proj_fc2 = linear(tc.text_hidden_size, tc.hidden_size, vb_tp.pp("linear_fc2"))?;
        let codec_embedding = embedding(
            tc.vocab_size,
            tc.hidden_size,
            vb_m.pp("codec_embedding"),
            &None,
        )?;

        let layers = build_layers(
            &LayerSpec {
                n: tc.num_hidden_layers,
                hidden: tc.hidden_size,
                inter: tc.intermediate_size,
                num_heads: tc.num_attention_heads,
                num_kv_heads: tc.num_key_value_heads,
                head_dim: tc.head_dim,
                eps: tc.rms_norm_eps,
            },
            &rope,
            vb_m.pp("layers"),
        )?;
        let norm = RmsNorm::new(tc.hidden_size, tc.rms_norm_eps, vb_m.pp("norm"))?;
        let codec_head = linear_no_bias(tc.hidden_size, tc.vocab_size, vb_t.pp("codec_head"))?;
        let code_predictor = CodePredictor::new(
            &tc.code_predictor_config,
            dtype,
            dev,
            vb_t.pp("code_predictor"),
        )?;

        Ok(Self {
            text_embedding,
            text_proj_fc1,
            text_proj_fc2,
            codec_embedding,
            layers,
            norm,
            codec_head,
            code_predictor,
            cfg: tc.clone(),
            dtype,
            device: dev.clone(),
        })
    }

    fn project_text(&self, ids: &Tensor) -> Result<Tensor> {
        let e = self.text_embedding.forward(ids)?.to_dtype(self.dtype)?;
        let h = self.text_proj_fc1.forward(&e)?.silu()?;
        self.text_proj_fc2.forward(&h)
    }

    fn embed_codec(&self, id: u32) -> Result<Tensor> {
        self.codec_embedding
            .forward(&Tensor::from_vec(vec![id], (1, 1), &self.device)?)?
            .to_dtype(self.dtype)
    }

    // Build the prefill input embedding sequence: projected text tokens, then a tts_bos codec
    // embedding to kick off generation. (Speaker conditioning omitted in milestone-1.)
    fn build_prefill(&self, text_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(text_ids.to_vec(), (1, text_ids.len()), &self.device)?;
        let text_embeds = self.project_text(&ids)?; // [1, T, H]
        let bos = self.embed_codec(self.cfg.codec_bos_id)?;
        Tensor::cat(&[&text_embeds, &bos], 1)
    }

    // Autoregressively decode codec frames. Returns `[num_codebooks, num_frames]` codes (u32).
    pub fn generate_codes(&self, text_ids: &[u32], max_frames: usize) -> Result<Tensor> {
        let mut seq = self.build_prefill(text_ids)?; // [1, P, H]
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let eos = self.cfg.codec_eos_token_id;

        for _ in 0..max_frames {
            let t = seq.dim(1)?;
            let mask = causal_mask(t, 0, self.dtype, &self.device)?;
            let mut h = seq.clone();
            for layer in &self.layers {
                h = layer.forward(&h, Some(&mask), 0)?;
            }
            h = self.norm.forward(&h)?;
            let last_h = h.i((.., t - 1, ..))?; // [1, H]
            let logits = self.codec_head.forward(&last_h)?;
            let cb0 = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
            if cb0 == eos {
                break;
            }
            let frame_hidden = last_h.unsqueeze(1)?; // [1,1,H]
            let codes = self
                .code_predictor
                .predict_frame(&frame_hidden, cb0, &self.device)?;
            frames.push(codes);
            // Feed the chosen primary codec token embedding back as the next input position.
            let next = self.embed_codec(cb0)?;
            seq = Tensor::cat(&[&seq, &next], 1)?;
        }

        if frames.is_empty() {
            return Tensor::zeros((self.cfg.num_code_groups, 0), DType::U32, &self.device);
        }
        let num_cb = frames[0].len();
        let num_frames = frames.len();
        let mut flat = vec![0u32; num_cb * num_frames];
        for (fi, f) in frames.iter().enumerate() {
            for (ci, c) in f.iter().enumerate() {
                flat[ci * num_frames + fi] = *c;
            }
        }
        Tensor::from_vec(flat, (num_cb, num_frames), &self.device)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ---- codec / vocoder decoder (codes -> 24kHz waveform) ---------------------

fn conv1d_b(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_c, in_c, k), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

fn conv_transpose1d_b(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: ConvTranspose1dConfig,
    vb: ShardedVarBuilder,
) -> Result<ConvTranspose1d> {
    let weight = vb.get((in_c, out_c, k), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}

// SnakeBeta (BigVGAN/Vocos): alpha and beta are stored in log space, so they are exponentiated
// before use: y = x + (1/(exp(beta)+eps)) * sin(exp(alpha)*x)^2.
struct SnakeBeta {
    alpha: Tensor, // [C], log-space
    beta: Tensor,  // [C], log-space
}

impl SnakeBeta {
    fn load(c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get(c, "alpha")?,
            beta: vb.get(c, "beta")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        let c = self.alpha.dim(0)?;
        let alpha = self.alpha.reshape((1, c, 1))?.exp()?;
        let beta = self.beta.reshape((1, c, 1))?.exp()?;
        let sin = alpha.broadcast_mul(x)?.sin()?;
        let sin2 = (&sin * &sin)?;
        let inv_beta = (beta + 1e-9)?.recip()?;
        x.broadcast_add(&inv_beta.broadcast_mul(&sin2)?)
    }
}

// AMP residual block: act1 -> conv1(k7,dilation) -> act2 -> conv2(k1), plus residual.
struct AmpBlock {
    act1: SnakeBeta,
    conv1: Conv1d,
    act2: SnakeBeta,
    conv2: Conv1d,
}

impl AmpBlock {
    fn load(c: usize, dilation: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let act1 = SnakeBeta::load(c, vb.pp("act1"))?;
        let pad = ((7 - 1) * dilation) / 2;
        let conv1 = conv1d_b(
            c,
            c,
            7,
            Conv1dConfig {
                padding: pad,
                dilation,
                ..Default::default()
            },
            vb.pp("conv1").pp("conv"),
        )?;
        let act2 = SnakeBeta::load(c, vb.pp("act2"))?;
        let conv2 = conv1d_b(c, c, 1, Conv1dConfig::default(), vb.pp("conv2").pp("conv"))?;
        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        x + h
    }
}

// One decoder upsampling stage: SnakeBeta -> ConvTranspose1d(k=2*stride, stride) -> 3 AmpBlocks.
struct DecoderStage {
    pre_act: SnakeBeta,
    up: ConvTranspose1d,
    blocks: Vec<AmpBlock>,
}

impl DecoderStage {
    fn load(in_c: usize, out_c: usize, stride: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_act = SnakeBeta::load(in_c, vb.pp("block").pp(0))?;
        let k = 2 * stride;
        let up = conv_transpose1d_b(
            in_c,
            out_c,
            k,
            ConvTranspose1dConfig {
                stride,
                padding: stride.div_ceil(2),
                ..Default::default()
            },
            vb.pp("block").pp(1).pp("conv"),
        )?;
        let dilations = [1usize, 3, 9];
        let mut blocks = Vec::new();
        for (bi, d) in dilations.iter().enumerate() {
            blocks.push(AmpBlock::load(out_c, *d, vb.pp("block").pp(2 + bi))?);
        }
        Ok(Self {
            pre_act,
            up,
            blocks,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.pre_act.forward(x)?;
        h = self.up.forward(&h)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        Ok(h)
    }
}

// ConvNeXt block used in the `upsample.*.1` stages.
struct ConvNeXt {
    dwconv: Conv1d,
    norm_w: Tensor,
    norm_b: Tensor,
    pw1: hanzo_nn::Linear,
    pw2: hanzo_nn::Linear,
    gamma: Tensor,
}

impl ConvNeXt {
    fn load(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // Depthwise conv: groups == dim, so the weight is [dim, 1, k].
        let dwconv = {
            let vb_c = vb.pp("dwconv").pp("conv");
            let w = vb_c.get((dim, 1, 7), "weight")?;
            let b = vb_c.get(dim, "bias")?;
            Conv1d::new(
                w,
                Some(b),
                Conv1dConfig {
                    padding: 3,
                    groups: dim,
                    ..Default::default()
                },
            )
        };
        let norm_w = vb.get(dim, "norm.weight")?;
        let norm_b = vb.get(dim, "norm.bias")?;
        let pw1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pw2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm_w,
            norm_b,
            pw1,
            pw2,
            gamma,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        let residual = x;
        let h = self.dwconv.forward(x)?;
        // LayerNorm over channel dim -> operate in [B, T, C].
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = layer_norm(&h, &self.norm_w, &self.norm_b)?;
        let h = self.pw1.forward(&h)?.gelu_erf()?;
        let h = self.pw2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma)?;
        let h = h.transpose(1, 2)?.contiguous()?;
        residual + h
    }
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + 1e-6)?.sqrt()?)?;
    let out = normed.broadcast_mul(w)?.broadcast_add(b)?;
    out.to_dtype(dtype)
}

struct UpsampleStage {
    up: ConvTranspose1d,
    convnext: ConvNeXt,
}

impl UpsampleStage {
    fn load(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let up = conv_transpose1d_b(
            dim,
            dim,
            2,
            ConvTranspose1dConfig {
                stride: 2,
                ..Default::default()
            },
            vb.pp(0).pp("conv"),
        )?;
        let convnext = ConvNeXt::load(dim, vb.pp(1))?;
        Ok(Self { up, convnext })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.up.forward(x)?;
        self.convnext.forward(&h)
    }
}

// Residual VQ dequantizer: maps integer codes to continuous embeddings via codebook centroids.
struct Rvq {
    input_proj: Conv1d,
    output_proj: Conv1d,
    codebooks: Vec<Tensor>, // each [codebook_size, codebook_dim]
}

impl Rvq {
    fn load(
        n_layers: usize,
        in_dim: usize,
        code_dim: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let input_proj = {
            let w = vb.pp("input_proj").get((code_dim, in_dim, 1), "weight")?;
            Conv1d::new(w, None, Conv1dConfig::default())
        };
        let output_proj = {
            let w = vb.pp("output_proj").get((in_dim, code_dim, 1), "weight")?;
            Conv1d::new(w, None, Conv1dConfig::default())
        };
        let mut codebooks = Vec::with_capacity(n_layers);
        let vb_l = vb.pp("vq").pp("layers");
        let size = 2048usize;
        let dim = 256usize;
        for i in 0..n_layers {
            let vb_cb = vb_l.pp(i).pp("_codebook");
            let embed_sum = vb_cb.get((size, dim), "embedding_sum")?;
            let usage = vb_cb.get(size, "cluster_usage")?.reshape((size, 1))?;
            let centroid = embed_sum
                .to_dtype(DType::F32)?
                .broadcast_div(&(usage.to_dtype(DType::F32)? + 1e-9)?)?;
            codebooks.push(centroid);
        }
        Ok(Self {
            input_proj,
            output_proj,
            codebooks,
        })
    }

    // codes: [n_layers, T] u32. Returns [1, in_dim, T] embedding sum across layers.
    fn dequantize(&self, codes: &Tensor) -> Result<Tensor> {
        let n = self.codebooks.len();
        let t = codes.dim(1)?;
        let dtype = self.input_proj.weight().dtype();
        let dev = codes.device();
        let code_dim = self.codebooks[0].dim(1)?;
        let mut acc = Tensor::zeros((t, code_dim), DType::F32, dev)?;
        for i in 0..n {
            let idx = codes.i((i, ..))?; // [T] u32
            let emb = self.codebooks[i]
                .to_dtype(DType::F32)?
                .index_select(&idx, 0)?; // [T, dim]
            acc = (acc + emb)?;
        }
        let acc = acc.t()?.unsqueeze(0)?.to_dtype(dtype)?; // [1, dim, T]
        self.output_proj.forward(&acc)
    }
}

struct CodecQuantizer {
    rvq_first: Rvq,
    rvq_rest: Rvq,
}

impl CodecQuantizer {
    fn load(cfg: &CodecDecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let in_dim = cfg.vector_quantization_hidden_dimension; // 512
        let code_dim = cfg.codebook_dim; // not the 256 proj dim; proj is 256
        let _ = code_dim;
        let proj_dim = 256;
        let rvq_first = Rvq::load(1, in_dim, proj_dim, vb.pp("rvq_first"))?;
        let rvq_rest = Rvq::load(cfg.num_quantizers - 1, in_dim, proj_dim, vb.pp("rvq_rest"))?;
        Ok(Self {
            rvq_first,
            rvq_rest,
        })
    }

    // codes: [16, T]. First codebook -> rvq_first, remaining 15 -> rvq_rest. Sum the two latents.
    fn dequantize(&self, codes: &Tensor) -> Result<Tensor> {
        let first = codes.i((0..1, ..))?;
        let rest = codes.i((1.., ..))?;
        let a = self.rvq_first.dequantize(&first)?;
        let b = self.rvq_rest.dequantize(&rest)?;
        a + b
    }
}

// pre_transformer: 8 Qwen3-ish layers with input/output projection and layer-scale, no q/k norm.
struct PreTransformerLayer {
    q_proj: hanzo_nn::Linear,
    k_proj: hanzo_nn::Linear,
    v_proj: hanzo_nn::Linear,
    o_proj: hanzo_nn::Linear,
    input_ln: RmsNorm,
    post_ln: RmsNorm,
    mlp: Qwen3Mlp,
    attn_scale: Tensor,
    mlp_scale: Tensor,
    rope: Arc<Rope>,
    num_heads: usize,
    head_dim: usize,
    n_kv_groups: usize,
    sdpa: SdpaParams,
}

impl PreTransformerLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let residual = xs;
        let h = self.input_ln.forward(xs)?;
        let q = self
            .q_proj
            .forward(&h)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&h)?
            .reshape((b, t, self.num_heads / self.n_kv_groups, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&h)?
            .reshape((b, t, self.num_heads / self.n_kv_groups, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.rope.apply(&q.contiguous()?, 0)?;
        let k = self.rope.apply(&k.contiguous()?, 0)?;
        let k = repeat_kv(k, self.n_kv_groups)?;
        let v = repeat_kv(v, self.n_kv_groups)?;
        let attn = naive_sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            None,
            &self.sdpa,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        let attn = self.o_proj.forward(&attn)?;
        let attn = attn.broadcast_mul(&self.attn_scale)?;
        let xs = (residual + attn)?;
        let residual = &xs;
        let h = self.mlp.forward(&self.post_ln.forward(&xs)?)?;
        let h = h.broadcast_mul(&self.mlp_scale)?;
        residual + h
    }
}

struct PreTransformer {
    input_proj: hanzo_nn::Linear,
    layers: Vec<PreTransformerLayer>,
    norm: RmsNorm,
    output_proj: hanzo_nn::Linear,
}

impl PreTransformer {
    fn load(
        cfg: &CodecDecoderConfig,
        dtype: DType,
        dev: &Device,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size; // 512
                                      // The pre_transformer sits between pre_conv (512->1024) and the upsample stack, operating on
                                      // the `latent_dim` (1024) features projected down to `hidden` (512) and back.
        let io_dim = cfg.latent_dim; // 1024
        let rope = Arc::new(Rope::new(
            cfg.head_dim,
            cfg.max_position_embeddings.min(8192),
            cfg.rope_theta,
            dtype,
            dev,
        )?);
        let input_proj = linear(io_dim, hidden, vb.pp("input_proj"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        let n_kv_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        for i in 0..cfg.num_hidden_layers {
            let vbl = vb_l.pp(i);
            let q_proj = linear_no_bias(
                hidden,
                cfg.num_attention_heads * cfg.head_dim,
                vbl.pp("self_attn").pp("q_proj"),
            )?;
            let k_proj = linear_no_bias(
                hidden,
                cfg.num_key_value_heads * cfg.head_dim,
                vbl.pp("self_attn").pp("k_proj"),
            )?;
            let v_proj = linear_no_bias(
                hidden,
                cfg.num_key_value_heads * cfg.head_dim,
                vbl.pp("self_attn").pp("v_proj"),
            )?;
            let o_proj = linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                hidden,
                vbl.pp("self_attn").pp("o_proj"),
            )?;
            let input_ln = RmsNorm::new(hidden, cfg.rms_norm_eps, vbl.pp("input_layernorm"))?;
            let post_ln =
                RmsNorm::new(hidden, cfg.rms_norm_eps, vbl.pp("post_attention_layernorm"))?;
            let mlp = Qwen3Mlp::new(hidden, cfg.intermediate_size, vbl.pp("mlp"))?;
            let attn_scale = vbl.pp("self_attn_layer_scale").get(hidden, "scale")?;
            let mlp_scale = vbl.pp("mlp_layer_scale").get(hidden, "scale")?;
            layers.push(PreTransformerLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                input_ln,
                post_ln,
                mlp,
                attn_scale,
                mlp_scale,
                rope: rope.clone(),
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim,
                n_kv_groups,
                sdpa: SdpaParams {
                    n_kv_groups,
                    softcap: None,
                    softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
            });
        }
        let norm = RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("norm"))?;
        let output_proj = linear(hidden, io_dim, vb.pp("output_proj"))?;
        Ok(Self {
            input_proj,
            layers,
            norm,
            output_proj,
        })
    }

    // x: [B, C(=1024), T] -> [B, C(=1024), T]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.transpose(1, 2)?.contiguous()?; // [B, T, C]
        let mut h = self.input_proj.forward(&h)?;
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        h = self.norm.forward(&h)?;
        h = self.output_proj.forward(&h)?;
        h.transpose(1, 2)?.contiguous()
    }
}

pub struct CodecDecoder {
    quantizer: CodecQuantizer,
    pre_conv: Conv1d,
    pre_transformer: PreTransformer,
    upsample: Vec<UpsampleStage>,
    decoder_in: Conv1d,
    stages: Vec<DecoderStage>,
    final_act: SnakeBeta,
    decoder_out: Conv1d,
}

impl CodecDecoder {
    pub fn new(
        cfg: &Qwen3TtsCodecConfig,
        dtype: DType,
        dev: &Device,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let dc = &cfg.decoder_config;
        let vb_d = vb.pp("decoder");

        let quantizer = CodecQuantizer::load(dc, vb_d.pp("quantizer"))?;
        // pre_conv: [512 -> 1024, k3, pad1]
        let pre_conv = conv1d_b(
            512,
            1024,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb_d.pp("pre_conv").pp("conv"),
        )?;
        let pre_transformer = PreTransformer::load(dc, dtype, dev, vb_d.pp("pre_transformer"))?;

        let mut upsample = Vec::new();
        let vb_up = vb_d.pp("upsample");
        for i in 0..dc.upsampling_ratios.len() {
            upsample.push(UpsampleStage::load(1024, vb_up.pp(i))?);
        }

        // Main decoder conv stack. Channel/stride schedule inferred from weight shapes:
        // decoder.0: 1024 -> 1536 (k7). stages 1..=4 halve channels and upsample by upsample_rates.
        let vb_dec = vb_d.pp("decoder");
        let decoder_in = conv1d_b(
            1024,
            1536,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb_dec.pp(0).pp("conv"),
        )?;
        let chans = [1536usize, 768, 384, 192, 96];
        let mut stages = Vec::new();
        for (si, stride) in dc.upsample_rates.iter().enumerate() {
            stages.push(DecoderStage::load(
                chans[si],
                chans[si + 1],
                *stride,
                vb_dec.pp(1 + si),
            )?);
        }
        let final_act = SnakeBeta::load(96, vb_dec.pp(1 + dc.upsample_rates.len()))?;
        let decoder_out = conv1d_b(
            96,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb_dec.pp(2 + dc.upsample_rates.len()).pp("conv"),
        )?;

        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            decoder_in,
            stages,
            final_act,
            decoder_out,
        })
    }

    // codes: [16, T] u32 -> waveform Vec<f32> at 24kHz.
    pub fn decode(&self, codes: &Tensor) -> Result<Vec<f32>> {
        let latent = self.quantizer.dequantize(codes)?; // [1, 512, T]
        let mut h = self.pre_conv.forward(&latent)?; // [1, 1024, T]
        h = self.pre_transformer.forward(&h)?; // [1, 1024, T]
        for u in &self.upsample {
            h = u.forward(&h)?; // 2x each
        }
        h = self.decoder_in.forward(&h)?; // [1, 1536, T']
        for s in &self.stages {
            h = s.forward(&h)?;
        }
        h = self.final_act.forward(&h)?;
        h = self.decoder_out.forward(&h)?; // [1, 1, T'']
        let wav = h.i((0, 0))?.to_dtype(DType::F32)?;
        wav.to_vec1::<f32>()
    }
}
