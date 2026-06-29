#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! InfiniteTalk streaming sampler: infinite-length video via overlapped 81-frame chunks.
//!
//! Pure latent-space orchestration over a [`ChunkDenoiser`] (the Wan2.1-14B DiT, owned by the
//! shared `wan` module). Per arXiv:2508.14033 sec.3.3-3.4: a chunk is 81 frames -> 21 latents;
//! `motion_frame=9` -> 3 latent slots of context pinned from the previous chunk tail and re-noised
//! at every denoise step (motion momentum); 72 new frames/chunk; the per-chunk reference is sampled
//! from the adjacent chunk (<=1s, "M3" soft-conditioning). No pixels, no VAE, no DiT internals here.

use hanzo_ml::{DType, Device, Result, Tensor};

pub const FPS: f64 = 25.0;
pub const TEMPORAL_COMPRESSION: usize = 4;
pub const CHUNK_FRAMES: usize = 81;
pub const MOTION_FRAMES: usize = 9;
pub const NEW_FRAMES: usize = CHUNK_FRAMES - MOTION_FRAMES; // 72
pub const CHUNK_LATENTS: usize = frames_to_latents(CHUNK_FRAMES); // 21
pub const CONTEXT_LATENTS: usize = frames_to_latents(MOTION_FRAMES); // 3
pub const NEW_LATENTS: usize = CHUNK_LATENTS - CONTEXT_LATENTS; // 18
pub const DEFAULT_MAX_FRAMES: usize = 1000;
pub const DEFAULT_STEPS: usize = 40;
pub const DEFAULT_SHIFT: f64 = 5.0;
pub const REF_MAX_DIST_S: f64 = 1.0;

const LATENT_T_AXIS: usize = 2;

/// Causal-VAE frame->latent map: `(F-1)//4 + 1`.
pub const fn frames_to_latents(f: usize) -> usize {
    (f - 1) / TEMPORAL_COMPRESSION + 1
}

/// Inverse map: `(L-1)*4 + 1`.
pub const fn latents_to_frames(l: usize) -> usize {
    (l - 1) * TEMPORAL_COMPRESSION + 1
}

/// The single integration seam with the shared Wan DiT @ 14B (built by the EchoMimic port at 1.3B,
/// instantiated here at 14B). Returns the flow-matching velocity for a noisy latent chunk; the
/// audio cross-attn + L-RoPE (see `audio.rs`) are injected inside the DiT block, fed `audio`.
pub trait ChunkDenoiser {
    /// `latent` [B,C,T,H,W], `audio` per-latent-frame tokens [B,T,M,D], `t` in (0,1]. Same-shape out.
    fn denoise_chunk(
        &self,
        latent: &Tensor,
        audio: &Tensor,
        cond: &ChunkCond,
        t: f64,
    ) -> Result<Tensor>;
}

/// Per-chunk conditioning. `reference` is the M3 sparse-frame ref latent the sampler picks each
/// chunk; `text`/`image_embed` are precomputed (umT5 / CLIP-ViT-H) and constant across chunks.
pub struct ChunkCond {
    pub reference: Tensor,
    pub text: Option<Tensor>,
    pub image_embed: Option<Tensor>,
}

/// Conditioning that does not vary per chunk; the sampler clones it and fills `reference`.
#[derive(Clone)]
pub struct CondTemplate {
    pub text: Option<Tensor>,
    pub image_embed: Option<Tensor>,
}

impl CondTemplate {
    pub fn none() -> Self {
        Self {
            text: None,
            image_embed: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    pub steps: usize,
    pub shift: f64,
    pub max_frames: usize,
    pub fps: f64,
    pub ref_max_dist_s: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEPS,
            shift: DEFAULT_SHIFT,
            max_frames: DEFAULT_MAX_FRAMES,
            fps: FPS,
            ref_max_dist_s: REF_MAX_DIST_S,
        }
    }
}

/// How many chunks + latent frames a target frame count expands to. Pure arithmetic; the headline
/// 81->21 / 9->3 / 72-new invariants live here so they are testable without a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    pub chunks: usize,
    pub total_latents: usize,
}

pub fn plan_for_frames(frames: usize) -> ChunkPlan {
    let need = frames_to_latents(frames.max(1));
    if need <= CHUNK_LATENTS {
        return ChunkPlan {
            chunks: 1,
            total_latents: CHUNK_LATENTS,
        };
    }
    let extra = need - CHUNK_LATENTS;
    let chunks = 1 + extra.div_ceil(NEW_LATENTS);
    ChunkPlan {
        chunks,
        total_latents: CHUNK_LATENTS + (chunks - 1) * NEW_LATENTS,
    }
}

/// M3 reference frame index: ~1s back into the adjacent (just-produced) chunk, clamped to >=0.
pub fn m3_ref_index(produced: usize, ref_dist: usize) -> usize {
    produced.saturating_sub(ref_dist.max(1))
}

fn ref_dist_latents(cfg: &StreamConfig) -> usize {
    ((cfg.ref_max_dist_s * cfg.fps / TEMPORAL_COMPRESSION as f64).round() as usize).max(1)
}

/// FlowMatchEuler timesteps 1.0 -> 0.0 with Wan resolution shift (SD3/Wan form).
pub fn flow_sigmas(steps: usize, shift: f64) -> Vec<f64> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps as f64;
            shift * t / (1.0 + (shift - 1.0) * t)
        })
        .collect()
}

/// Re-pin the first [`CONTEXT_LATENTS`] slots to the known context re-noised to level `t`:
/// `x_t = (1-t)*clean + t*noise`. This is the cross-boundary motion injection.
fn pin_context(x: &Tensor, clean: &Tensor, noise_ctx: &Tensor, t: f64) -> Result<Tensor> {
    let ctx_t = ((clean * (1.0 - t))? + (noise_ctx * t)?)?;
    let rest = x.narrow(LATENT_T_AXIS, CONTEXT_LATENTS, x.dim(LATENT_T_AXIS)? - CONTEXT_LATENTS)?;
    Tensor::cat(&[&ctx_t, &rest], LATENT_T_AXIS)
}

/// Denoise one [`CHUNK_LATENTS`]-frame chunk. `context` (clean, [B,C,CONTEXT_LATENTS,H,W]) is `None`
/// for the first chunk; otherwise it is pinned + re-noised every step so motion carries across.
#[allow(clippy::too_many_arguments)]
pub fn denoise_one_chunk(
    denoiser: &dyn ChunkDenoiser,
    context: Option<&Tensor>,
    audio: &Tensor,
    cond: &ChunkCond,
    shape: (usize, usize, usize, usize, usize),
    cfg: &StreamConfig,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let noise = Tensor::randn(0f32, 1.0, shape, device)?.to_dtype(dtype)?;
    let noise_ctx = match context {
        Some(_) => Some(noise.narrow(LATENT_T_AXIS, 0, CONTEXT_LATENTS)?),
        None => None,
    };
    let sigmas = flow_sigmas(cfg.steps, cfg.shift);
    let mut x = noise;
    for w in sigmas.windows(2) {
        let (t, t_next) = (w[0], w[1]);
        if let (Some(clean), Some(nz)) = (context, noise_ctx.as_ref()) {
            x = pin_context(&x, clean, nz, t)?;
        }
        let v = denoiser.denoise_chunk(&x, audio, cond, t)?;
        x = (x + (v * (t_next - t))?)?;
    }
    if let Some(clean) = context {
        let rest = x.narrow(LATENT_T_AXIS, CONTEXT_LATENTS, x.dim(LATENT_T_AXIS)? - CONTEXT_LATENTS)?;
        x = Tensor::cat(&[clean, &rest], LATENT_T_AXIS)?;
    }
    Ok(x)
}

/// Audio window [start, start+len) over the per-latent-frame token axis, padding past the end by
/// repeating the last frame so a chunk overhanging the clip still aligns.
fn slice_audio(audio: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let t = audio.dim(1)?;
    let s = start.min(t.saturating_sub(1));
    let avail = (t - s).min(len);
    let head = audio.narrow(1, s, avail)?;
    if avail == len {
        return Ok(head);
    }
    let last = audio.narrow(1, t - 1, 1)?;
    let mut pad_dims = last.dims().to_vec();
    pad_dims[1] = len - avail;
    let pad = last.broadcast_as(pad_dims)?.contiguous()?;
    Tensor::cat(&[&head, &pad], 1)
}

/// Generate the full latent video by streaming overlapped chunks. `init_ref` [B,C,1,H,W] is the
/// chunk-0 reference (portrait latent); `audio` [B,T,M,D] is per-latent-frame audio tokens. Output
/// [B,C,L,H,W] with `L = min(frames_to_latents(max_frames), audio T)`.
pub fn stream(
    denoiser: &dyn ChunkDenoiser,
    init_ref: &Tensor,
    audio: &Tensor,
    cond_tmpl: &CondTemplate,
    cfg: &StreamConfig,
) -> Result<Tensor> {
    let (b, c, _r, h, w) = init_ref.dims5()?;
    let dtype = init_ref.dtype();
    let device = init_ref.device().clone();

    let t_audio = audio.dim(1)?;
    let target = frames_to_latents(cfg.max_frames).min(t_audio);
    if target == 0 {
        hanzo_ml::bail!("infinitetalk stream: empty audio / zero target frames");
    }
    let ref_dist = ref_dist_latents(cfg);

    let mut frames: Vec<Tensor> = Vec::with_capacity(target);
    let mut produced = 0usize;
    let mut chunk_idx = 0usize;
    while produced < target {
        let first = chunk_idx == 0;
        let context = if first {
            None
        } else {
            let tail: Vec<Tensor> = (produced - CONTEXT_LATENTS..produced)
                .map(|i| frames[i].clone())
                .collect();
            Some(Tensor::cat(&tail, LATENT_T_AXIS)?)
        };
        let start = if first { 0 } else { produced - CONTEXT_LATENTS };
        let audio_chunk = slice_audio(audio, start, CHUNK_LATENTS)?;
        let reference = if first {
            init_ref.clone()
        } else {
            frames[m3_ref_index(produced, ref_dist)].clone()
        };
        let cond = ChunkCond {
            reference,
            text: cond_tmpl.text.clone(),
            image_embed: cond_tmpl.image_embed.clone(),
        };
        let chunk = denoise_one_chunk(
            denoiser,
            context.as_ref(),
            &audio_chunk,
            &cond,
            (b, c, CHUNK_LATENTS, h, w),
            cfg,
            dtype,
            &device,
        )?;
        let new = if first {
            chunk
        } else {
            chunk.narrow(LATENT_T_AXIS, CONTEXT_LATENTS, NEW_LATENTS)?
        };
        let n_new = new.dim(LATENT_T_AXIS)?;
        for i in 0..n_new {
            frames.push(new.narrow(LATENT_T_AXIS, i, 1)?.contiguous()?);
        }
        produced += n_new;
        chunk_idx += 1;
    }
    let out: Vec<&Tensor> = frames.iter().take(target).collect();
    Tensor::cat(&out, LATENT_T_AXIS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn latent(t: usize, dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1.0, (1usize, 4, t, 8, 8), dev).unwrap()
    }

    fn audio_tokens(t: usize, dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1.0, (1usize, t, 32, 768), dev).unwrap()
    }

    // Predicts zero velocity: the integrator leaves the noise untouched, so the only thing that can
    // move a latent slot is context re-pinning -> lets us assert pinning exactly.
    struct ZeroDenoiser;
    impl ChunkDenoiser for ZeroDenoiser {
        fn denoise_chunk(&self, l: &Tensor, _a: &Tensor, _c: &ChunkCond, _t: f64) -> Result<Tensor> {
            l.zeros_like()
        }
    }

    #[derive(Default)]
    struct Recorder {
        latent_t: Vec<usize>,
        audio_t: Vec<usize>,
    }
    struct RecordingDenoiser(Mutex<Recorder>);
    impl ChunkDenoiser for RecordingDenoiser {
        fn denoise_chunk(&self, l: &Tensor, a: &Tensor, _c: &ChunkCond, _t: f64) -> Result<Tensor> {
            let mut r = self.0.lock().unwrap();
            r.latent_t.push(l.dim(LATENT_T_AXIS).unwrap());
            r.audio_t.push(a.dim(1).unwrap());
            l.zeros_like()
        }
    }

    #[test]
    fn chunk_invariants() {
        assert_eq!(CHUNK_LATENTS, 21);
        assert_eq!(CONTEXT_LATENTS, 3);
        assert_eq!(NEW_LATENTS, 18);
        assert_eq!(NEW_FRAMES, 72);
        assert_eq!(NEW_LATENTS * TEMPORAL_COMPRESSION, NEW_FRAMES);
        assert_eq!(CONTEXT_LATENTS + NEW_LATENTS, CHUNK_LATENTS);
        assert_eq!(frames_to_latents(81), 21);
        assert_eq!(frames_to_latents(9), 3);
        assert_eq!(latents_to_frames(21), 81);
    }

    #[test]
    fn plan_expands_correctly() {
        assert_eq!(plan_for_frames(81), ChunkPlan { chunks: 1, total_latents: 21 });
        // 153 frames -> 39 latents -> 1 + ceil((39-21)/18) = 2 chunks, 21 + 18 = 39 latents
        assert_eq!(plan_for_frames(153), ChunkPlan { chunks: 2, total_latents: 39 });
        // 154 frames -> 39 latents (same)
        assert_eq!(plan_for_frames(154), ChunkPlan { chunks: 2, total_latents: 39 });
        // one latent past a chunk forces a new chunk
        assert_eq!(plan_for_frames(latents_to_frames(22)), ChunkPlan { chunks: 2, total_latents: 39 });
    }

    #[test]
    fn m3_reference_within_one_second() {
        let cfg = StreamConfig::default();
        let d = ref_dist_latents(&cfg); // 25fps / 4 ~= 6 latent frames ~ 1s
        assert_eq!(d, 6);
        assert_eq!(m3_ref_index(21, d), 15);
        assert!(21 - m3_ref_index(21, d) <= d);
        assert_eq!(m3_ref_index(3, d), 0); // clamps at start
    }

    #[test]
    fn pin_context_recovers_clean_context() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = StreamConfig { steps: 4, ..Default::default() };
        let ctx = latent(CONTEXT_LATENTS, &dev);
        let cond = ChunkCond { reference: latent(1, &dev), text: None, image_embed: None };
        let out = denoise_one_chunk(
            &ZeroDenoiser,
            Some(&ctx),
            &audio_tokens(CHUNK_LATENTS, &dev),
            &cond,
            (1, 4, CHUNK_LATENTS, 8, 8),
            &cfg,
            DType::F32,
            &dev,
        )?;
        assert_eq!(out.dims(), &[1, 4, CHUNK_LATENTS, 8, 8]);
        let got = out.narrow(LATENT_T_AXIS, 0, CONTEXT_LATENTS)?;
        let diff = (got - &ctx)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "context not preserved exactly, max abs diff {diff}");
        Ok(())
    }

    #[test]
    fn stream_shapes_and_chunking() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = StreamConfig { steps: 2, max_frames: 153, ..Default::default() };
        let plan = plan_for_frames(cfg.max_frames);
        let rec = RecordingDenoiser(Mutex::new(Recorder::default()));
        let out = stream(
            &rec,
            &latent(1, &dev),
            &audio_tokens(plan.total_latents, &dev),
            &CondTemplate::none(),
            &cfg,
        )?;
        assert_eq!(out.dim(LATENT_T_AXIS)?, plan.total_latents);
        let r = rec.0.lock().unwrap();
        assert_eq!(r.latent_t.len(), plan.chunks * cfg.steps);
        assert!(r.latent_t.iter().all(|&t| t == CHUNK_LATENTS));
        assert!(r.audio_t.iter().all(|&t| t == CHUNK_LATENTS));
        Ok(())
    }

    #[test]
    fn first_chunk_has_no_context_rest_do() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = StreamConfig { steps: 1, max_frames: 153, ..Default::default() };
        let out = stream(
            &ZeroDenoiser,
            &latent(1, &dev),
            &audio_tokens(plan_for_frames(153).total_latents, &dev),
            &CondTemplate::none(),
            &cfg,
        )?;
        // 2 chunks: 21 (all new) + 18 (new only) = 39
        assert_eq!(out.dim(LATENT_T_AXIS)?, 39);
        Ok(())
    }

    #[test]
    fn audio_slice_pads_past_end() -> Result<()> {
        let dev = Device::Cpu;
        let a = audio_tokens(5, &dev);
        let s = slice_audio(&a, 3, CHUNK_LATENTS)?;
        assert_eq!(s.dim(1)?, CHUNK_LATENTS);
        // tail rows equal the last real row (frame 4)
        let last = a.narrow(1, 4, 1)?;
        let padded = s.narrow(1, CHUNK_LATENTS - 1, 1)?;
        let diff = (padded - last)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }
}
