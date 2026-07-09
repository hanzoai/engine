#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! InfiniteTalk / MultiTalk audio conditioning (arXiv:2505.22647).
//!
//! Two independent pieces, both `wav2vec2`-feature-in (the encoder is precomputed offline, never
//! ported here):
//! - [`AudioProjModel`]: 12-layer-stacked wav2vec2 features in a 5-frame window -> 32 context tokens.
//! - [`SingleStreamAudioAttn`]: the per-DiT-block injectable cross-attn (video latents query, audio
//!   K/V), with L-RoPE for multi-person routing. The shared Wan DiT block calls `forward` once.

use hanzo_ml::{DType, Result, Tensor, D};
use hanzo_nn::{LayerNorm, Linear, RmsNorm};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::layers::{self, MatMul};
use crate::pipeline::text_models_inputs_processor::FlashParams;

pub const WAV2VEC_LAYERS: usize = 12;
pub const WAV2VEC_DIM: usize = 768;
pub const AUDIO_WINDOW: usize = 5; // offsets [-2,-1,0,1,2]
pub const AUDIO_WINDOW_HALF: i64 = 2;
pub const CONTEXT_TOKENS: usize = 32;
pub const PROJ_INTERMEDIATE: usize = 512;
pub const AUDIO_PROJ_INPUT: usize = AUDIO_WINDOW * WAV2VEC_LAYERS * WAV2VEC_DIM; // 46080
const RMS_EPS: f64 = 1e-6;
const LN_EPS: f64 = 1e-5;

// L-RoPE multi-person label scheme (MultiTalk sec.3.3).
pub const CLASS_RANGE: f64 = 24.0;
pub const CLASS_INTERVAL: f64 = 4.0;
pub const BG_LABEL: f64 = CLASS_RANGE / 2.0; // 12
pub const AUDIO_LABEL_P1: f64 = 2.0;
pub const AUDIO_LABEL_P2: f64 = CLASS_RANGE - 2.0; // 22
pub const ROPE_1D_BASE: f64 = 10000.0;

fn layer_norm(dim: usize, vb: ShardedVarBuilder) -> Result<LayerNorm> {
    let w = vb.get(dim, "weight")?;
    let b = vb.get(dim, "bias")?;
    Ok(LayerNorm::new(w, b, LN_EPS))
}

/// `[B,T,LAYERS,DIM]` stacked wav2vec2 features -> `[B,T,WINDOW,LAYERS,DIM]` by gathering each
/// frame's `[-2..2]` neighbours (edge-clamped). `T` is the audio token axis the streaming sampler
/// indexes 1:1 with latent frames, so the caller groups the 25fps wav2vec frames to latent frames.
pub fn build_audio_windows(feats: &Tensor) -> Result<Tensor> {
    let (_b, t, _l, _c) = feats.dims4()?;
    let dev = feats.device();
    let mut cols = Vec::with_capacity(AUDIO_WINDOW);
    for off in -AUDIO_WINDOW_HALF..=AUDIO_WINDOW_HALF {
        let idx: Vec<u32> = (0..t)
            .map(|i| (i as i64 + off).clamp(0, t as i64 - 1) as u32)
            .collect();
        let idx = Tensor::from_vec(idx, t, dev)?;
        cols.push(feats.index_select(&idx, 1)?);
    }
    Tensor::stack(&cols, 2)
}

/// MultiTalk `AudioProjModel`: window -> 32 context tokens of `WAV2VEC_DIM`. `proj1_vf` handles the
/// wider first-frame group (`seq_len_vf` sub-frames); the streaming path uses `forward` (5-frame).
pub struct AudioProjModel {
    proj1: Linear,
    proj1_vf: Linear,
    proj2: Linear,
    proj3: Linear,
    norm: LayerNorm,
    seq_len_vf: usize,
}

impl AudioProjModel {
    pub fn new(seq_len_vf: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let input_dim_vf = seq_len_vf * WAV2VEC_LAYERS * WAV2VEC_DIM;
        Ok(Self {
            proj1: layers::linear(AUDIO_PROJ_INPUT, PROJ_INTERMEDIATE, vb.pp("proj1"))?,
            proj1_vf: layers::linear(input_dim_vf, PROJ_INTERMEDIATE, vb.pp("proj1_vf"))?,
            proj2: layers::linear(PROJ_INTERMEDIATE, PROJ_INTERMEDIATE, vb.pp("proj2"))?,
            proj3: layers::linear(
                PROJ_INTERMEDIATE,
                CONTEXT_TOKENS * WAV2VEC_DIM,
                vb.pp("proj3"),
            )?,
            norm: layer_norm(WAV2VEC_DIM, vb.pp("norm"))?,
            seq_len_vf,
        })
    }

    pub fn seq_len_vf(&self) -> usize {
        self.seq_len_vf
    }

    /// `[B,F,WINDOW,LAYERS,DIM]` -> `[B,F,CONTEXT_TOKENS,DIM]`.
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let (b, f, w, l, c) = audio.dims5()?;
        let x = audio.reshape((b * f, w * l * c))?;
        self.project_flat(&x, &self.proj1)?
            .reshape((b, f, CONTEXT_TOKENS, c))
    }

    /// First-frame group via `proj1_vf` (`[B,F,seq_len_vf,LAYERS,DIM]` -> `[B,F,CONTEXT_TOKENS,DIM]`).
    pub fn forward_vf(&self, audio: &Tensor) -> Result<Tensor> {
        let (b, f, w, l, c) = audio.dims5()?;
        let x = audio.reshape((b * f, w * l * c))?;
        self.project_flat(&x, &self.proj1_vf)?
            .reshape((b, f, CONTEXT_TOKENS, c))
    }

    fn project_flat(&self, x: &Tensor, proj1: &Linear) -> Result<Tensor> {
        let rows = x.dim(0)?;
        let x = x.apply(proj1)?.relu()?;
        let x = x.apply(&self.proj2)?.relu()?;
        let x = x
            .apply(&self.proj3)?
            .reshape((rows, CONTEXT_TOKENS, WAV2VEC_DIM))?;
        x.apply(&self.norm)
    }
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let w = (MatMul.matmul(&q, &k.t()?)? * scale)?;
    let o = MatMul.matmul(&hanzo_nn::ops::softmax_last_dim(&w)?, &v)?;
    batch_dims.push(o.dim(D::Minus2)?);
    batch_dims.push(o.dim(D::Minus1)?);
    o.reshape(batch_dims)
}

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let params = SdpaParams {
        n_kv_groups: 1,
        sliding_window: None,
        softcap: None,
        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        sinks: None,
    };
    let flash = FlashParams::empty(false);
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    match Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, Some(&flash), &params) {
        Ok(o) => Ok(o),
        Err(_) => scaled_dot_product_attention(&q, &k, &v),
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// 1D RoPE on `[B,H,L,D]` with a continuous per-token position (the L-RoPE label), base 10000.
pub fn apply_rope_1d(x: &Tensor, positions: &[f64], base: f64) -> Result<Tensor> {
    let (_b, _h, l, d) = x.dims4()?;
    if positions.len() != l {
        hanzo_ml::bail!("rope_1d: {} positions for {l} tokens", positions.len());
    }
    let dev = x.device();
    let half = d / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / base.powf(2.0 * i as f64 / d as f64)) as f32)
        .collect();
    let inv = Tensor::from_vec(inv, (1, half), dev)?;
    let pos = Tensor::from_vec(
        positions.iter().map(|p| *p as f32).collect::<Vec<_>>(),
        (l, 1),
        dev,
    )?;
    let ang = pos.broadcast_mul(&inv)?;
    let emb = Tensor::cat(&[&ang, &ang], D::Minus1)?;
    let cos = emb.cos()?.reshape((1, 1, l, d))?.to_dtype(x.dtype())?;
    let sin = emb.sin()?.reshape((1, 1, l, d))?.to_dtype(x.dtype())?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold(0, |acc, (i, &v)| if v > row[acc] { i } else { acc })
}

fn norm_into(w: f32, lo: f32, hi: f32, a: f64, b: f64) -> f64 {
    if hi <= lo {
        return a;
    }
    a + ((w - lo) / (hi - lo)) as f64 * (b - a)
}

/// Per-video-token L-RoPE label from the ref->video attention similarity `[L, K]` (K = persons +
/// background, person columns first). Winning category min-max-normalized into its rope sub-range;
/// background tokens take the static `BG_LABEL`. (`human_num==1` skips L-RoPE entirely upstream.)
pub fn video_token_labels(ref_attn: &Tensor) -> Result<Vec<f64>> {
    let s = ref_attn.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let k = ref_attn.dim(1)?;
    let bg = k - 1;
    let cats: Vec<usize> = s.iter().map(|r| argmax(r)).collect();
    let win: Vec<f32> = s.iter().zip(&cats).map(|(r, &c)| r[c]).collect();
    let range = |person: usize| {
        let vals: Vec<f32> = win
            .iter()
            .zip(&cats)
            .filter(|(_, &c)| c == person)
            .map(|(&w, _)| w)
            .collect();
        let lo = vals.iter().cloned().fold(f32::MAX, f32::min);
        let hi = vals.iter().cloned().fold(f32::MIN, f32::max);
        (lo, hi)
    };
    let (p1_lo, p1_hi) = range(0);
    let (p2_lo, p2_hi) = range(1);
    Ok(cats
        .iter()
        .zip(&win)
        .map(|(&c, &w)| match c {
            _ if c == bg => BG_LABEL,
            0 => norm_into(w, p1_lo, p1_hi, 0.0, CLASS_INTERVAL),
            1 => norm_into(w, p2_lo, p2_hi, CLASS_RANGE - CLASS_INTERVAL, CLASS_RANGE),
            _ => BG_LABEL,
        })
        .collect())
}

/// Static audio-stream label for person `p` (0-based): P1 -> 2, P2 -> 22.
pub fn audio_stream_label(p: usize) -> f64 {
    if p == 0 {
        AUDIO_LABEL_P1
    } else {
        AUDIO_LABEL_P2
    }
}

/// MultiTalk `SingleStreamMutiAttention`: the per-block audio cross-attn injected into the Wan DiT.
/// Video latents form Q (`q_linear`), audio context tokens form K/V (`kv_linear`, 768 -> 2*dim).
/// Single-person `forward` is plain cross-attn; `forward_multi` adds L-RoPE label routing.
pub struct SingleStreamAudioAttn {
    q_linear: Linear,
    kv_linear: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SingleStreamAudioAttn {
    // `add_q_norm`/`add_k_norm` (the second-person stream norms) land with the multi-person
    // two-stream concat path, not loaded here so we don't carry unused params.
    pub fn new(dim: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            q_linear: layers::linear(dim, dim, vb.pp("q_linear"))?,
            kv_linear: layers::linear(WAV2VEC_DIM, dim * 2, vb.pp("kv_linear"))?,
            q_norm: RmsNorm::new(vb.get(head_dim, "q_norm.weight")?, RMS_EPS),
            k_norm: RmsNorm::new(vb.get(head_dim, "k_norm.weight")?, RMS_EPS),
            proj: layers::linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim,
        })
    }

    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        x.reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)
    }

    /// Project the audio context tokens `[N,M,768]` to per-head K,V `[N,H,M,D]`. Separated so a
    /// cross-chunk KV-cache can wrap it (the pinned context reuses prior audio K/V across chunks).
    pub fn project_kv(&self, audio: &Tensor) -> Result<(Tensor, Tensor)> {
        let kv = audio.apply(&self.kv_linear)?;
        let dim = self.num_heads * self.head_dim;
        let k = self
            .heads(&kv.narrow(D::Minus1, 0, dim)?)?
            .apply(&self.k_norm)?;
        let v = self.heads(&kv.narrow(D::Minus1, dim, dim)?)?;
        Ok((k, v))
    }

    fn finish(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (n, _h, l, _d) = q.dims4()?;
        let o = sdpa(q, k, v)?
            .transpose(1, 2)?
            .reshape((n, l, self.num_heads * self.head_dim))?;
        o.apply(&self.proj)
    }

    /// Single-person cross-attn. `x` [N,L,dim] (N = frames flattened into batch -> frame-local audio
    /// attention, i.e. the `ip_mask` realized by batching), `audio` [N,M,768]. Returns [N,L,dim].
    pub fn forward(&self, x: &Tensor, audio: &Tensor) -> Result<Tensor> {
        let q = self.heads(&x.apply(&self.q_linear)?)?.apply(&self.q_norm)?;
        let (k, v) = self.project_kv(audio)?;
        self.finish(&q, &k, &v)
    }

    /// Multi-person path: L-RoPE rotates Q by per-video-token labels and K by the audio-stream label,
    /// decoupling persons. `human_num == 1` upstream should call [`forward`] instead.
    pub fn forward_multi(
        &self,
        x: &Tensor,
        audio: &Tensor,
        video_labels: &[f64],
        audio_label: f64,
    ) -> Result<Tensor> {
        let q = self.heads(&x.apply(&self.q_linear)?)?.apply(&self.q_norm)?;
        let q = apply_rope_1d(&q, video_labels, ROPE_1D_BASE)?;
        let (k, v) = self.project_kv(audio)?;
        let m = k.dim(2)?;
        let k = apply_rope_1d(&k, &vec![audio_label; m], ROPE_1D_BASE)?;
        self.finish(&q, &k, &v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{Device, Shape};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

    struct RandnBackend;
    impl SimpleBackend for RandnBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("weight") && s.rank() == 1 {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, 0.02, s, dev)?.to_dtype(dtype)
            }
        }
        fn get_unchecked(&self, _n: &str, _d: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("needs explicit shape")
        }
        fn contains_tensor(&self, _n: &str) -> bool {
            true
        }
    }

    fn vb(dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone())
    }

    fn finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    }

    #[test]
    fn windows_have_five_frames() -> Result<()> {
        let dev = Device::Cpu;
        let feats = Tensor::randn(0f32, 1.0, (1usize, 7, WAV2VEC_LAYERS, WAV2VEC_DIM), &dev)?;
        let w = build_audio_windows(&feats)?;
        assert_eq!(w.dims(), &[1, 7, AUDIO_WINDOW, WAV2VEC_LAYERS, WAV2VEC_DIM]);
        // frame 0 window: offsets [-2,-1,0,1,2] clamp to [0,0,0,1,2]
        let center = w.narrow(1, 0, 1)?.narrow(2, 2, 1)?; // offset 0
        let left = w.narrow(1, 0, 1)?.narrow(2, 0, 1)?; // offset -2 clamped to 0
        let diff = (center - left)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn audio_proj_emits_context_tokens() -> Result<()> {
        let dev = Device::Cpu;
        let proj = AudioProjModel::new(WAV2VEC_LAYERS, vb(&dev))?;
        let feats = Tensor::randn(0f32, 1.0, (1usize, 6, WAV2VEC_LAYERS, WAV2VEC_DIM), &dev)?;
        let out = proj.forward(&build_audio_windows(&feats)?)?;
        assert_eq!(out.dims(), &[1, 6, CONTEXT_TOKENS, WAV2VEC_DIM]);
        assert!(finite(&out));
        assert_eq!(AUDIO_PROJ_INPUT, 46080);
        Ok(())
    }

    #[test]
    fn audio_attn_single_person_shape() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, heads) = (640usize, 5usize); // head_dim 128, like the 14B block
        let attn = SingleStreamAudioAttn::new(dim, heads, vb(&dev))?;
        let x = Tensor::randn(0f32, 1.0, (2usize, 64, dim), &dev)?; // 2 frames, 64 latent tokens
        let audio = Tensor::randn(0f32, 1.0, (2usize, CONTEXT_TOKENS, WAV2VEC_DIM), &dev)?;
        let out = attn.forward(&x, &audio)?;
        assert_eq!(out.dims(), &[2, 64, dim]);
        assert!(finite(&out));
        Ok(())
    }

    #[test]
    fn audio_attn_multi_person_lrope_shape() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, heads) = (640usize, 5usize);
        let attn = SingleStreamAudioAttn::new(dim, heads, vb(&dev))?;
        let x = Tensor::randn(0f32, 1.0, (1usize, 16, dim), &dev)?;
        let audio = Tensor::randn(0f32, 1.0, (1usize, CONTEXT_TOKENS, WAV2VEC_DIM), &dev)?;
        let labels = vec![1.0f64; 16];
        let out = attn.forward_multi(&x, &audio, &labels, AUDIO_LABEL_P1)?;
        assert_eq!(out.dims(), &[1, 16, dim]);
        assert!(finite(&out));
        Ok(())
    }

    #[test]
    fn lrope_labels_land_in_person_ranges() -> Result<()> {
        let dev = Device::Cpu;
        let s = Tensor::from_vec(
            vec![5f32, 0., 0., 4., 0., 0., 0., 5., 0., 0., 0., 5.],
            (4usize, 3),
            &dev,
        )?;
        let labels = video_token_labels(&s)?;
        assert!((0.0..=CLASS_INTERVAL).contains(&labels[0]));
        assert!((0.0..=CLASS_INTERVAL).contains(&labels[1]));
        assert!((CLASS_RANGE - CLASS_INTERVAL..=CLASS_RANGE).contains(&labels[2]));
        assert_eq!(labels[3], BG_LABEL);
        assert_eq!(audio_stream_label(0), AUDIO_LABEL_P1);
        assert_eq!(audio_stream_label(1), AUDIO_LABEL_P2);
        Ok(())
    }

    #[test]
    fn rope_1d_preserves_shape_and_norm() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1usize, 2, 6, 8), &dev)?;
        let pos: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let y = apply_rope_1d(&x, &pos, ROPE_1D_BASE)?;
        assert_eq!(y.dims(), x.dims());
        // rotation is norm-preserving
        let nx = x.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let ny = y.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            (nx - ny).abs() / nx < 1e-4,
            "rope changed norm: {nx} vs {ny}"
        );
        Ok(())
    }
}
