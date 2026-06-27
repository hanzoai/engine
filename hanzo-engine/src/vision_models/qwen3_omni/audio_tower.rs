#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3-Omni Thinker audio tower (`thinker.audio_tower`).
//!
//! A Whisper-style log-mel -> hidden encoder, structurally identical to the
//! Qwen3-ASR AuT encoder (HF `Qwen3OmniMoeAudioEncoder`): a 3-layer stride-2
//! Conv2d stem (8x downsample) + per-chunk sinusoidal positions + a
//! block-diagonal (windowed) bidirectional transformer with biased MHA, then
//! `ln_post` + a 2-layer projection (`proj1` -> gelu -> `proj2`) to `output_dim`.
//!
//! Layout verified against `zen-omni-30b-instruct` shard 1 (`thinker.audio_tower.*`):
//!   - `conv2d{1,2,3}`: k=3 s=2 p=1, channels 1->480->480->480 (`downsample_hidden_size`),
//!     every conv biased.
//!   - `conv_out`: `[1280, 7680]`, NO bias (480 channels * 16 freq bins -> d_model 1280;
//!     16 = `num_mel_bins` 128 halved three times by the stride-2 stem).
//!   - `layers.{i}.self_attn.{q,k,v,out}_proj`: `[1280, 1280]`, ALL four biased; full MHA
//!     (20 heads, head_dim 64, no GQA), bidirectional.
//!   - `layers.{i}.fc1`: `[5120, 1280]` + bias, `fc2`: `[1280, 5120]` + bias; gelu.
//!   - `layers.{i}.self_attn_layer_norm` (pre-attn) + `final_layer_norm` (pre-FFN),
//!     both weight+bias.
//!   - `ln_post`, `proj1` `[1280, 1280]`, `proj2` `[2048, 1280]`: all weight+bias.
//!   - No stored positional table (sinusoidal positions are computed, like Whisper).

use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{conv2d, layer_norm, linear, linear_no_bias, Activation};

use super::config::OmniAudioConfig;

const CONV_KERNEL: usize = 3;
const CONV_STRIDE: usize = 2;
const CONV_PADDING: usize = 1;
// HF `nn.LayerNorm` default eps.
const LN_EPS: f64 = 1e-5;
const SINUSOID_BASE: f64 = 10_000.0;

/// Length after one `k=3 s=2 p=1` Conv2d: `floor((L - 1) / 2) + 1`.
fn conv_down(l: usize) -> usize {
    (l - 1) / 2 + 1
}

/// Length after the 3-layer stride-2 conv stem (8x downsample, applied to either
/// the time or the frequency axis).
fn conv_len3(l: usize) -> usize {
    conv_down(conv_down(conv_down(l)))
}

/// Whisper-style sinusoidal absolute position table `[max_pos, dim]`.
///
/// Matches HF `SinusoidsPositionEmbedding`: the log-timescale increment is
/// divided by `(channels / 2 - 1)` (NOT `channels`), and the layout is
/// `[sin(pos * inv), cos(pos * inv)]` concatenated along the channel axis.
fn sinusoidal_embedding(
    max_pos: usize,
    dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let half = dim / 2;
    let increment = SINUSOID_BASE.ln() / (half as f64 - 1.0);
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (-increment * i as f64).exp() as f32)
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
    fn new(cfg: &OmniAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.d_model;
        let head_dim = cfg.head_dim();
        // Verified against the checkpoint: q/k/v/out_proj all carry bias.
        let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.encoder_attention_heads,
            head_dim,
            sdpa_params: SdpaParams {
                // Full MHA: heads == kv heads (HF `num_key_value_groups = 1`).
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    /// Bidirectional self-attention over `xs` `[B, T, d]`. Attention is global
    /// over the whole `T` axis; block-diagonal windowing is handled by the
    /// caller slicing `xs` into per-window batches before calling this.
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
    fn new(cfg: &OmniAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: cfg.activation_function,
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
    fn new(cfg: &OmniAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = EncoderAttention::new(cfg, vb.pp("self_attn"))?;
        // `fc1`/`fc2`/`*_layer_norm` live directly on the layer vb (no sub-prefix).
        let mlp = EncoderMlp::new(cfg, vb.clone())?;
        let attn_norm = layer_norm(cfg.d_model, LN_EPS, vb.pp("self_attn_layer_norm"))?;
        let final_norm = layer_norm(cfg.d_model, LN_EPS, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            mlp,
            attn_norm,
            final_norm,
        })
    }

    /// Pre-norm Whisper block: `x + attn(ln1(x))` then `x + mlp(ln2(x))`.
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

/// Qwen3-Omni Thinker audio tower: chunked Conv2d stem (8x downsample) +
/// per-chunk sinusoidal positions + block-diagonal (windowed) bidirectional
/// transformer + `ln_post` + 2-layer output projection to `output_dim`.
pub struct OmniAudioTower {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv_out: Linear,
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    pos_embed: Tensor,
    act: Activation,
    dtype: DType,
    /// Raw mel frames per conv chunk (`n_window * 2`).
    chunk_size: usize,
    /// Inference attention-window size in raw mel frames (`n_window_infer`).
    n_window_infer: usize,
}

impl OmniAudioTower {
    pub fn new(cfg: &OmniAudioConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let dtype = vb.dtype();
        let conv_cfg = Conv2dConfig {
            padding: CONV_PADDING,
            stride: CONV_STRIDE,
            ..Default::default()
        };
        let ch = cfg.downsample_hidden_size;
        let conv1 = conv2d(1, ch, CONV_KERNEL, conv_cfg, vb.pp("conv2d1"))?;
        let conv2 = conv2d(ch, ch, CONV_KERNEL, conv_cfg, vb.pp("conv2d2"))?;
        let conv3 = conv2d(ch, ch, CONV_KERNEL, conv_cfg, vb.pp("conv2d3"))?;

        // Flattened conv-stem features (`channels * freq_after_stem`) -> d_model.
        // `conv_out` ships no bias.
        let conv_feature_dim = ch * conv_len3(cfg.num_mel_bins);
        let conv_out = linear_no_bias(conv_feature_dim, cfg.d_model, vb.pp("conv_out"))?;

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let ln_post = layer_norm(cfg.d_model, LN_EPS, vb.pp("ln_post"))?;
        let proj1 = linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        let pos_embed = sinusoidal_embedding(cfg.max_source_positions, cfg.d_model, device, dtype)?;

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
            act: cfg.activation_function,
            dtype,
            chunk_size: cfg.n_window * 2,
            n_window_infer: cfg.n_window_infer,
        })
    }

    /// Input: log-mel features `[1, num_mel_bins, T]`. Output: `[1, T', output_dim]`.
    ///
    /// Faithful to HF `Qwen3OmniMoeAudioEncoder.forward`:
    ///   1. split the mel into `chunk_size`-frame chunks (right-pad the tail);
    ///   2. conv-stem each chunk independently (so per-chunk conv boundaries and
    ///      the post-CNN frame count match the reference);
    ///   3. add *per-chunk* sinusoidal positions (reset to 0 each chunk);
    ///   4. drop conv outputs that came from the padded tail of each chunk;
    ///   5. run the transformer with block-diagonal attention over inference
    ///      windows of `window_aftercnn` post-CNN frames; then ln_post + proj.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (b, n_mels, t) = mel.dims3()?;
        debug_assert_eq!(b, 1, "OmniAudioTower expects a single clip per call");
        let mel = mel.to_dtype(self.dtype)?;

        // --- 1. chunk the mel along time into `chunk_size`-frame windows ---
        let chunk_size = self.chunk_size;
        let num_chunks = t.div_ceil(chunk_size);
        // Per-chunk post-CNN time length (for a full `chunk_size`-frame chunk).
        let full_chunk_tlen = conv_len3(chunk_size);

        // --- 2. conv-stem each chunk (padded to chunk_size for a uniform conv) ---
        let mut chunk_feats = Vec::with_capacity(num_chunks);
        let mut valid_tlens = Vec::with_capacity(num_chunks);
        for ci in 0..num_chunks {
            let start = ci * chunk_size;
            let len = (t - start).min(chunk_size);
            valid_tlens.push(conv_len3(len));
            // [1, n_mels, len] -> right-pad to chunk_size -> [1, 1, n_mels, chunk_size].
            let chunk = mel.narrow(2, start, len)?;
            let chunk = if len < chunk_size {
                chunk.pad_with_zeros(2, 0, chunk_size - len)?
            } else {
                chunk
            };
            let xs = chunk.reshape((1, 1, n_mels, chunk_size))?;
            let xs = self.conv1.forward(&xs)?.apply(&self.act)?;
            let xs = self.conv2.forward(&xs)?.apply(&self.act)?;
            let xs = self.conv3.forward(&xs)?.apply(&self.act)?;
            // [1, C, F, t_chunk] -> [1, t_chunk, C*F] -> conv_out -> [1, t_chunk, d_model].
            let (_, c, f, tc) = xs.dims4()?;
            debug_assert_eq!(tc, full_chunk_tlen);
            let xs = xs
                .permute((0, 3, 1, 2))?
                .reshape((1, tc, c * f))?
                .contiguous()?;
            let xs = self.conv_out.forward(&xs)?;
            // 3. per-chunk sinusoidal positions, reset to 0..t_chunk each chunk.
            let pos = self.pos_embed.narrow(0, 0, tc)?.unsqueeze(0)?;
            chunk_feats.push(xs.broadcast_add(&pos)?);
        }

        // --- 4. concat chunks and drop padded-tail conv outputs (valid gather) ---
        let mut valid = Vec::with_capacity(num_chunks);
        for (ci, feat) in chunk_feats.iter().enumerate() {
            valid.push(feat.narrow(1, 0, valid_tlens[ci])?);
        }
        let hidden = Tensor::cat(&valid, 1)?; // [1, S, d_model]
        let seq_len = hidden.dim(1)?;

        // --- 5. block-diagonal (windowed) transformer ---
        // window_aftercnn = (post-CNN frames per full chunk) * (n_window_infer / chunk_size).
        let n_window_ratio = (self.n_window_infer / chunk_size).max(1);
        let window_aftercnn = (full_chunk_tlen * n_window_ratio).max(1);

        let hidden = if seq_len <= window_aftercnn {
            // Single window: global bidirectional attention (the common short-clip case).
            let mut h = hidden;
            for layer in &self.layers {
                h = layer.forward(&h)?;
            }
            h
        } else {
            // Multiple windows: run the full layer stack per contiguous window so
            // attention never crosses a window boundary (matches HF cu_seqlens).
            let mut outs = Vec::new();
            let mut off = 0usize;
            while off < seq_len {
                let wlen = window_aftercnn.min(seq_len - off);
                let mut w = hidden.narrow(1, off, wlen)?.contiguous()?;
                for layer in &self.layers {
                    w = layer.forward(&w)?;
                }
                outs.push(w);
                off += wlen;
            }
            Tensor::cat(&outs, 1)?
        };

        let hidden = self.ln_post.forward(&hidden)?;
        let hidden = self.proj1.forward(&hidden)?.apply(&self.act)?;
        self.proj2.forward(&hidden)
    }
}
