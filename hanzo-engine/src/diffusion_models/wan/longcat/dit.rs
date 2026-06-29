#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Assembles the LongCat-Video-Avatar-1.5 DiT (depth 48, hidden 4096, 32 heads) and its inference
//! machinery: flow-matching velocity forward, block-causal KV-cache, the DMD few-step
//! FlowMatchEuler schedule, and LoRA merge. bf16 is the working dtype (15.85B params); dtype is a
//! plain parameter so an fp8/fp4 quant path can slot in (the matmuls never hard-code a dtype).

use hanzo_ml::{Context, DType, Device, Result, Tensor};
use hanzo_nn::Linear;
use hanzo_quant::ShardedVarBuilder;

use crate::attention::AttentionMask;
use crate::diffusion_models::wan::longcat::audio::AudioProjModel;
use crate::diffusion_models::wan::longcat::blocks::{Block, BlockCtx, KvSlot};
use crate::diffusion_models::wan::longcat::common::{block_causal_mask, Rope3D, ROPE_THETA};
use crate::diffusion_models::wan::longcat::embed::{
    CaptionEmbedder, FinalLayer, PatchEmbed3D, TimestepEmbedder,
};

// DMD2 schedule shift; >1 front-loads high-noise steps. Exact sigmas come from scheduler_config.json
// + the dmd_lora; this is the structurally-correct schedule pending that file.
const DMD_SHIFT: f64 = 3.0;
const TIMESTEP_SCALE: f64 = 1000.0;

/// LongCat avatar DiT configuration. `avatar()` is the shipped 15.85B model; `tiny()` is for tests.
#[derive(Debug, Clone)]
pub struct LongCatConfig {
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: [usize; 3],
    pub caption_channels: usize,
    pub adaln_tembed_dim: usize,
    pub freq_embed_dim: usize,
    pub rope_axes_dim: [usize; 3],
    pub audio_channel: usize,
    pub audio_window: usize,
    pub audio_block: usize,
    pub audio_intermediate: usize,
    pub audio_output_dim: usize,
    pub audio_context_tokens: usize,
    pub vae_scale: usize,
}

impl LongCatConfig {
    pub fn avatar() -> Self {
        Self {
            hidden_size: 4096,
            depth: 48,
            num_heads: 32,
            head_dim: 128,
            ffn_dim: 16384,
            in_channels: 16,
            out_channels: 16,
            patch_size: [1, 2, 2],
            caption_channels: 4096,
            adaln_tembed_dim: 512,
            freq_embed_dim: 256,
            rope_axes_dim: [44, 42, 42],
            audio_channel: 1280,
            audio_window: 5,
            audio_block: 5,
            audio_intermediate: 512,
            audio_output_dim: 768,
            audio_context_tokens: 32,
            vae_scale: 4,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hidden_size: 256,
            depth: 2,
            num_heads: 4,
            head_dim: 64,
            ffn_dim: 512,
            in_channels: 16,
            out_channels: 16,
            patch_size: [1, 2, 2],
            caption_channels: 64,
            adaln_tembed_dim: 64,
            freq_embed_dim: 64,
            rope_axes_dim: [24, 20, 20],
            audio_channel: 32,
            audio_window: 5,
            audio_block: 5,
            audio_intermediate: 48,
            audio_output_dim: 24,
            audio_context_tokens: 4,
            vae_scale: 4,
        }
    }

    pub fn patch_out(&self) -> usize {
        self.patch_size.iter().product::<usize>() * self.out_channels
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_size != self.num_heads * self.head_dim {
            hanzo_ml::bail!(
                "hidden_size {} != num_heads {} * head_dim {}",
                self.hidden_size,
                self.num_heads,
                self.head_dim
            );
        }
        if self.rope_axes_dim.iter().sum::<usize>() != self.head_dim {
            hanzo_ml::bail!(
                "rope_axes_dim {:?} must sum to head_dim {}",
                self.rope_axes_dim,
                self.head_dim
            );
        }
        Ok(())
    }
}

/// Per-block condition-token K/V cached across denoise steps (one slot per block).
pub struct KvCache {
    slots: Vec<Option<KvSlot>>,
}

impl KvCache {
    pub fn new(depth: usize) -> Self {
        Self {
            slots: vec![None; depth],
        }
    }
}

/// Flow-matching Euler-discrete schedule. `new` is the 50-step default; `distilled` is the DMD
/// few-step lever. sigmas run 1 -> 0; velocity steps integrate `x += v * dsigma`.
pub struct FlowMatchEuler {
    sigmas: Vec<f64>,
}

impl FlowMatchEuler {
    pub fn new(num_steps: usize) -> Self {
        let sigmas = (0..=num_steps)
            .map(|i| 1.0 - i as f64 / num_steps as f64)
            .collect();
        Self { sigmas }
    }

    pub fn distilled(num_steps: usize) -> Self {
        let sigmas = (0..=num_steps)
            .map(|i| {
                let t = 1.0 - i as f64 / num_steps as f64;
                DMD_SHIFT * t / (1.0 + (DMD_SHIFT - 1.0) * t)
            })
            .collect();
        Self { sigmas }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn sigma(&self, i: usize) -> f64 {
        self.sigmas[i]
    }

    /// Diffusion timestep for step `i` (flow-match t in [0, 1000]).
    pub fn timestep(&self, i: usize) -> f64 {
        self.sigmas[i] * TIMESTEP_SCALE
    }

    /// Euler update `x_{i+1} = x_i + v * (sigma_{i+1} - sigma_i)`.
    pub fn step(&self, x: &Tensor, v: &Tensor, i: usize) -> Result<Tensor> {
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        x + (v * dt)?
    }
}

/// LoRA delta merged into a base weight: `W' = W + alpha * (up @ down)` (down `[r,in]`, up `[out,r]`).
pub fn merge_lora(weight: &Tensor, down: &Tensor, up: &Tensor, alpha: f64) -> Result<Tensor> {
    let delta = up.matmul(down)?;
    weight + (delta * alpha)?
}

/// Rebuild a `Linear` with a LoRA delta merged into its weight (bias unchanged).
pub fn merged_linear(
    base: &Linear,
    down: &Tensor,
    up: &Tensor,
    alpha: f64,
) -> Result<Linear> {
    Ok(Linear::new(
        merge_lora(base.weight(), down, up, alpha)?,
        base.bias().cloned(),
    ))
}

/// The assembled LongCat avatar DiT.
pub struct LongCatAvatarDiT {
    x_embedder: PatchEmbed3D,
    t_embedder: TimestepEmbedder,
    y_embedder: CaptionEmbedder,
    audio_proj: AudioProjModel,
    blocks: Vec<Block>,
    final_layer: FinalLayer,
    rope: Rope3D,
    cfg: LongCatConfig,
    device: Device,
    dtype: DType,
}

impl LongCatAvatarDiT {
    pub fn new(cfg: LongCatConfig, vb: ShardedVarBuilder, device: Device) -> Result<Self> {
        cfg.validate()?;
        let dtype = vb.dtype();
        let x_embedder = PatchEmbed3D::new(&cfg, vb.pp("x_embedder").set_device(device.clone()))?;
        let t_embedder = TimestepEmbedder::new(&cfg, vb.pp("t_embedder").set_device(device.clone()))?;
        let y_embedder = CaptionEmbedder::new(&cfg, vb.pp("y_embedder").set_device(device.clone()))?;
        let audio_proj = AudioProjModel::new(&cfg, vb.pp("audio_proj").set_device(device.clone()))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::new(&cfg, vb_b.pp(i).set_device(device.clone()))?);
        }
        let final_layer = FinalLayer::new(&cfg, vb.pp("final_layer").set_device(device.clone()))?;
        let rope = Rope3D::new(cfg.rope_axes_dim, ROPE_THETA, device.clone())?;
        Ok(Self {
            x_embedder,
            t_embedder,
            y_embedder,
            audio_proj,
            blocks,
            final_layer,
            rope,
            cfg,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &LongCatConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    // Project text + audio context once per forward; both feed every block's cross-attn K/V.
    fn context(&self, text: &Tensor, audio_emb: &Tensor) -> Result<(Tensor, Tensor)> {
        let text_kv = self.y_embedder.forward(text)?;
        let proj = self.audio_proj.forward(audio_emb)?;
        let (b, t, ctx, d) = proj.dims4()?;
        let audio_kv = proj.reshape((b, t * ctx, d))?;
        Ok((text_kv, audio_kv))
    }

    fn unpatchify(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, hp, wp) = grid;
        let [pt, ph, pw] = self.cfg.patch_size;
        let oc = self.cfg.out_channels;
        let b = x.dim(0)?;
        x.reshape(&[b, t, hp, wp, pt, ph, pw, oc])?
            .permute([0, 7, 1, 4, 2, 5, 3, 6])?
            .contiguous()?
            .reshape((b, oc, t * pt, hp * ph, wp * pw))
    }

    /// Dense flow-matching velocity for the full latent `[B, C, T, H, W]`. `n_cond_frames` leading
    /// latent frames are block-causal condition tokens (`0` = pure T2V/I2V over the whole clip).
    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        text: &Tensor,
        audio_emb: &Tensor,
        n_cond_frames: usize,
    ) -> Result<Tensor> {
        let cond = self.t_embedder.forward(timestep, self.dtype)?;
        let (text_kv, audio_kv) = self.context(text, audio_emb)?;
        let (mut x, grid) = self.x_embedder.forward(latent)?;
        let (t, hp, wp) = grid;
        let (cos, sin) = self.rope.table(t, hp, wp)?;
        let n_tokens = t * hp * wp;
        let n_cond_tokens = n_cond_frames * hp * wp;
        let mask = block_causal_mask(n_cond_tokens, n_tokens, self.dtype, &self.device)?;
        let mask = mask.as_ref().unwrap_or(&AttentionMask::None);
        let ctx = BlockCtx {
            cond: &cond,
            cos: &cos,
            sin: &sin,
            text_kv: &text_kv,
            audio_kv: &audio_kv,
        };
        for block in &self.blocks {
            x = block.forward(&x, &ctx, mask)?;
        }
        self.unpatchify(&self.final_layer.forward(&x, &cond)?, grid)
    }

    /// Step-0 forward: dense over the full latent, filling `cache` with each block's condition K/V.
    pub fn forward_prime(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        text: &Tensor,
        audio_emb: &Tensor,
        n_cond_frames: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let cond = self.t_embedder.forward(timestep, self.dtype)?;
        let (text_kv, audio_kv) = self.context(text, audio_emb)?;
        let (mut x, grid) = self.x_embedder.forward(latent)?;
        let (t, hp, wp) = grid;
        let (cos, sin) = self.rope.table(t, hp, wp)?;
        let n_tokens = t * hp * wp;
        let n_cond_tokens = n_cond_frames * hp * wp;
        let mask = block_causal_mask(n_cond_tokens, n_tokens, self.dtype, &self.device)?;
        let mask = mask.as_ref().unwrap_or(&AttentionMask::None);
        let ctx = BlockCtx {
            cond: &cond,
            cos: &cos,
            sin: &sin,
            text_kv: &text_kv,
            audio_kv: &audio_kv,
        };
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, slot) = block.prime(&x, &ctx, mask, n_cond_tokens)?;
            x = out;
            cache.slots[i] = Some(slot);
        }
        self.unpatchify(&self.final_layer.forward(&x, &cond)?, grid)
    }

    /// Steps 1..N: velocity for the noisy latent `[B, C, T_noisy, H, W]` only, reusing cached
    /// condition K/V. `n_cond_frames` positions the noisy tokens in the full RoPE grid.
    pub fn forward_cached(
        &self,
        latent_noisy: &Tensor,
        timestep: &Tensor,
        text: &Tensor,
        audio_emb: &Tensor,
        n_cond_frames: usize,
        cache: &KvCache,
    ) -> Result<Tensor> {
        let cond = self.t_embedder.forward(timestep, self.dtype)?;
        let (text_kv, audio_kv) = self.context(text, audio_emb)?;
        let (mut x, grid) = self.x_embedder.forward(latent_noisy)?;
        let (t_noisy, hp, wp) = grid;
        let t_full = n_cond_frames + t_noisy;
        let (cos_full, sin_full) = self.rope.table(t_full, hp, wp)?;
        let n_cond_tokens = n_cond_frames * hp * wp;
        let n_noisy_tokens = t_noisy * hp * wp;
        let cos = cos_full.narrow(0, n_cond_tokens, n_noisy_tokens)?;
        let sin = sin_full.narrow(0, n_cond_tokens, n_noisy_tokens)?;
        let ctx = BlockCtx {
            cond: &cond,
            cos: &cos,
            sin: &sin,
            text_kv: &text_kv,
            audio_kv: &audio_kv,
        };
        for (i, block) in self.blocks.iter().enumerate() {
            let slot = cache.slots[i]
                .as_ref()
                .context("forward_cached called before forward_prime filled the KV-cache")?;
            x = block.forward_noisy(&x, &ctx, slot)?;
        }
        self.unpatchify(&self.final_layer.forward(&x, &cond)?, grid)
    }
}
