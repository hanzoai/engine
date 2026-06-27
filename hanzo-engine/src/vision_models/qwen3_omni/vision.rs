//! Qwen3-Omni Thinker vision tower (`thinker.visual.*`) — image+video understanding.
//!
//! HF `Qwen3OmniMoeVisionEncoder`: the same Qwen3-VL-family ViT (absolute pos embed + 2-D vision
//! RoPE + cu_seqlens-windowed SDPA + spatial-merge PatchMerger + deepstack mergers) as
//! [`crate::vision_models::qwen3_vl::vision::Qwen3VLVisionModel`]. It is mirrored here rather than
//! reused directly because the Omni checkpoint names the merger weights differently — `merger.ln_q`,
//! `merger.mlp.{0,2}` and `merger_list.{k}` (vs qwen3_vl's `merger.norm`, `merger.linear_fc1/2` and
//! `deepstack_merger_list.{k}`) — and its mergers apply the **exact (erf) GELU** of `nn.GELU()`, so a
//! clean reuse is impossible without editing qwen3_vl. Everything else (patch embed, pos-embed
//! bilinear interpolation, vision RoPE, windowed attention, block MLP with `gelu_pytorch_tanh`) is
//! byte-for-byte the same family and is mirrored faithfully; validated to cosine > 0.99 against the
//! HF tower in [`omni_vision_matches_reference`].
//!
//! [`OmniVisionTower::forward`] returns the merged embeds `[n_merged, out_hidden_size=2048]`
//! (HF `pooler_output`) — exactly the Thinker-space token rows that [`super::modality::fuse_modalities`]
//! scatters into image/video placeholder positions. [`VisionModality`] adapts the tower to the
//! [`super::modality::ModalityEncoder`] contract so it plugs into the existing fusion with zero
//! changes there.

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_nn::{Embedding, LayerNorm, LayerNormConfig, Linear, Module};
use hanzo_quant::{QuantizedConfig, ShardedVarBuilder};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{self, Activation, Conv3dConfig, Conv3dNoBias, Sdpa},
    pipeline::text_models_inputs_processor::FlashParams,
};

use super::config::VisionConfig;
use super::modality::{ModalityEncoder, ModalityInput};

struct PatchEmbed {
    proj: Conv3dNoBias,
    bias: Tensor,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let proj_vb = vb.pp("proj");
        let proj = Conv3dNoBias::new(
            cfg.in_chans,
            cfg.hidden_size,
            [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size],
            Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            proj_vb.clone(),
        )?;
        let bias = proj_vb.get(cfg.hidden_size, "bias")?;
        Ok(Self {
            proj,
            bias,
            in_channels: cfg.in_chans,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.reshape((
            (),
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        let xs = self.proj.forward(&xs)?;
        let xs = xs.reshape(((), self.hidden_size))?;
        let bias = self.bias.unsqueeze(0)?;
        xs.broadcast_add(&bias)
    }
}

struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(dim: usize, hidden_dim: usize, act: Activation, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: layers::linear(dim, hidden_dim, vb.pp("linear_fc1"))?,
            fc2: layers::linear(hidden_dim, dim, vb.pp("linear_fc2"))?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.apply(&self.act)?;
        self.fc2.forward(&xs)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: layers::linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: layers::linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;
        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let mut q = qkv.i(0)?.squeeze(0)?;
        let mut k = qkv.i(1)?.squeeze(0)?;
        let v = qkv.i(2)?.squeeze(0)?;

        let orig_dtype = q.dtype();
        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;
        q = q.to_dtype(orig_dtype)?;
        k = k.to_dtype(orig_dtype)?;

        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let flash_params = FlashParams::empty(false);

            let mut chunk_out = Sdpa
                .run_attention(
                    &q_chunk.unsqueeze(0)?,
                    &k_chunk.unsqueeze(0)?,
                    &v_chunk.unsqueeze(0)?,
                    &AttentionMask::None,
                    Some(&flash_params),
                    &SdpaParams {
                        n_kv_groups: 1,
                        sliding_window: None,
                        softcap: None,
                        softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                        sinks: None,
                    },
                )?
                .squeeze(0)?
                .transpose(0, 1)?;
            chunk_out = chunk_out.reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out.to_dtype(xs.dtype())?);
        }
        let attn_output = Tensor::cat(&outputs, 0)?;
        self.proj.forward(&attn_output)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm1 = layers::layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm1"))?;
        let norm2 = layers::layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm2"))?;
        let attn = VisionAttention::new(cfg.hidden_size, cfg.num_heads, vb.pp("attn"))?;
        let mlp = VisionMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_act,
            vb.pp("mlp"),
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.norm1.forward(xs)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;
        let xs_att = xs.add(&attn_out)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs_att)?)?;
        xs_att.add(&mlp_out)
    }
}

/// Omni `Qwen3OmniMoeVisionPatchMerger`: `ln_q` -> spatial-merge reshape -> `mlp.0` (Linear) ->
/// **exact (erf) GELU** -> `mlp.2` (Linear). `use_postshuffle_norm` selects whether `ln_q` normalizes
/// over the pre-shuffle `hidden_size` (the main merger) or the post-shuffle `merged_hidden_size` (the
/// deepstack mergers). Mirrors HF `nn.GELU()` exactly via `gelu_erf`.
struct PatchMerger {
    ln_q: LayerNorm,
    use_postshuffle_norm: bool,
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
    fc1: Linear,
    fc2: Linear,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, use_postshuffle_norm: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let norm_dim = if use_postshuffle_norm {
            merged_hidden_size
        } else {
            cfg.hidden_size
        };
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            ln_q: layers::layer_norm(norm_dim, norm_cfg, vb.pp("ln_q"))?,
            use_postshuffle_norm,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            merged_hidden_size,
            fc1: layers::linear(merged_hidden_size, merged_hidden_size, vb.pp("mlp").pp("0"))?,
            fc2: layers::linear(
                merged_hidden_size,
                cfg.out_hidden_size,
                vb.pp("mlp").pp("2"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            hanzo_ml::bail!(
                "Sequence length {} is not divisible by spatial merge unit {}",
                seq_len,
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;
        let norm_input = if self.use_postshuffle_norm {
            xs.reshape((grouped, self.merged_hidden_size))?
        } else {
            xs.clone()
        };
        let normed = self.ln_q.forward(&norm_input)?;
        let reshaped = if self.use_postshuffle_norm {
            normed
        } else {
            normed.reshape((grouped, self.merged_hidden_size))?
        };
        let xs = self.fc1.forward(&reshaped)?;
        let xs = xs.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

/// The Omni Thinker vision tower. `forward` returns the merged Thinker-space embeds
/// `[n_merged, out_hidden_size]`; `forward_features` additionally returns the deepstack features
/// (HF `deepstack_features`) for the follow-up Thinker visual-position injection.
pub struct OmniVisionTower {
    patch_embed: PatchEmbed,
    pos_embed: Embedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
    deepstack_lookup: Vec<Option<usize>>,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    num_grid_per_side: usize,
    hidden_size: usize,
}

impl OmniVisionTower {
    pub fn new(cfg: &VisionConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let quant: Option<QuantizedConfig> = None;
        let pos_embed = layers::embedding(
            cfg.num_position_embeddings,
            cfg.hidden_size,
            vb.pp("pos_embed"),
            &quant,
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("blocks.{i}")))?);
        }

        let merger = PatchMerger::new(cfg, false, vb.pp("merger"))?;
        let deepstack_mergers = cfg
            .deepstack_visual_indexes
            .iter()
            .enumerate()
            .map(|(i, _)| PatchMerger::new(cfg, true, vb.pp(format!("merger_list.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let mut deepstack_lookup = vec![None; cfg.depth];
        for (idx, &layer_idx) in cfg.deepstack_visual_indexes.iter().enumerate() {
            if layer_idx < cfg.depth {
                deepstack_lookup[layer_idx] = Some(idx);
            }
        }

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, device)?;

        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt().round() as usize;
        if num_grid_per_side * num_grid_per_side != cfg.num_position_embeddings {
            hanzo_ml::bail!(
                "num_position_embeddings {} is not a perfect square",
                cfg.num_position_embeddings
            );
        }

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            deepstack_mergers,
            deepstack_lookup,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
            num_grid_per_side,
            hidden_size: cfg.hidden_size,
        })
    }

    fn linspace_points(&self, steps: usize) -> Vec<f32> {
        if steps == 1 {
            return vec![0.0];
        }
        let max_val = (self.num_grid_per_side - 1) as f32;
        let step = max_val / (steps.saturating_sub(1)) as f32;
        (0..steps).map(|i| i as f32 * step).collect()
    }

    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.pos_embed.embeddings().device();
        let dtype = self.pos_embed.embeddings().dtype();
        let grid = grid_thw.to_vec2::<u32>()?;

        let mut idx_lists: [Vec<i64>; 4] = Default::default();
        let mut weight_lists: [Vec<f32>; 4] = Default::default();
        let mut hw_lengths = Vec::with_capacity(grid.len());

        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            hw_lengths.push(h * w);

            let h_vals = self.linspace_points(h);
            let w_vals = self.linspace_points(w);

            let h_floor: Vec<usize> = h_vals.iter().map(|v| v.floor() as usize).collect();
            let w_floor: Vec<usize> = w_vals.iter().map(|v| v.floor() as usize).collect();
            let h_ceil: Vec<usize> = h_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(self.num_grid_per_side - 1))
                .collect();
            let w_ceil: Vec<usize> = w_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(self.num_grid_per_side - 1))
                .collect();
            let dh: Vec<f32> = h_vals
                .iter()
                .zip(&h_floor)
                .map(|(v, f)| v - *f as f32)
                .collect();
            let dw: Vec<f32> = w_vals
                .iter()
                .zip(&w_floor)
                .map(|(v, f)| v - *f as f32)
                .collect();

            for ((&hf, &hc), &dh_val) in h_floor.iter().zip(&h_ceil).zip(&dh) {
                for ((&wf, &wc), &dw_val) in w_floor.iter().zip(&w_ceil).zip(&dw) {
                    let base00 = (hf * self.num_grid_per_side + wf) as i64;
                    let base01 = (hf * self.num_grid_per_side + wc) as i64;
                    let base10 = (hc * self.num_grid_per_side + wf) as i64;
                    let base11 = (hc * self.num_grid_per_side + wc) as i64;

                    let w00 = (1.0 - dh_val) * (1.0 - dw_val);
                    let w01 = (1.0 - dh_val) * dw_val;
                    let w10 = dh_val * (1.0 - dw_val);
                    let w11 = dh_val * dw_val;

                    idx_lists[0].push(base00);
                    idx_lists[1].push(base01);
                    idx_lists[2].push(base10);
                    idx_lists[3].push(base11);

                    weight_lists[0].push(w00);
                    weight_lists[1].push(w01);
                    weight_lists[2].push(w10);
                    weight_lists[3].push(w11);
                }
            }
        }

        let idx_tensors = idx_lists
            .iter()
            .map(|idxs| Tensor::from_vec(idxs.clone(), (idxs.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let idx_tensor = Tensor::stack(&idx_tensors, 0)?;

        let weight_tensors = weight_lists
            .iter()
            .map(|weights| Tensor::from_vec(weights.clone(), (weights.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let weight_tensor = Tensor::stack(&weight_tensors, 0)?.to_dtype(dtype)?;

        let pos_embeds = self.pos_embed.forward(&idx_tensor)?;
        let pos_embeds = pos_embeds.broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let pos_embeds = pos_embeds.sum(0)?;

        let mut splits = Vec::with_capacity(hw_lengths.len());
        let mut start = 0;
        for len in hw_lengths {
            splits.push(pos_embeds.narrow(0, start, len)?);
            start += len;
        }

        let mut permuted = Vec::with_capacity(grid.len());
        for (pos_embed, g) in splits.into_iter().zip(&grid) {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let pos_embed = pos_embed.repeat((t, 1))?;
            let pos_embed = pos_embed.reshape((
                t,
                h / self.spatial_merge_size,
                self.spatial_merge_size,
                w / self.spatial_merge_size,
                self.spatial_merge_size,
                self.hidden_size,
            ))?;
            let pos_embed = pos_embed
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((t * h * w, self.hidden_size))?;
            permuted.push(pos_embed);
        }

        Tensor::cat(&permuted, 0)
    }

    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.rotary_pos_emb.inv_freq.device();
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_hw = grid
            .iter()
            .flat_map(|v| v[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let freq_table = self.rotary_pos_emb.make_embeds(max_hw)?;

        let mut coords: Vec<(i64, i64)> = Vec::new();
        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / self.spatial_merge_size;
            let merged_w = w / self.spatial_merge_size;

            let mut base_coords: Vec<(i64, i64)> = Vec::with_capacity(h * w);
            for br in 0..merged_h {
                for bc in 0..merged_w {
                    for ir in 0..self.spatial_merge_size {
                        for ic in 0..self.spatial_merge_size {
                            base_coords.push((
                                (br * self.spatial_merge_size + ir) as i64,
                                (bc * self.spatial_merge_size + ic) as i64,
                            ));
                        }
                    }
                }
            }

            for _ in 0..(g[0] as usize) {
                coords.extend(base_coords.iter().cloned());
            }
        }

        let total_tokens = coords.len();
        let mut rows = Vec::with_capacity(total_tokens);
        let mut cols = Vec::with_capacity(total_tokens);
        for &(r, c) in &coords {
            rows.push(r);
            cols.push(c);
        }
        let rows = Tensor::from_vec(rows, (total_tokens,), device)?;
        let cols = Tensor::from_vec(cols, (total_tokens,), device)?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
    }

    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|v| v[0] as usize).sum::<usize>() + 1);
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    /// Full HF `Qwen3OmniMoeVisionEncoder.forward`: returns `(merged, deepstack_features)` where
    /// `merged` is `pooler_output` `[n_merged, out_hidden_size]` and `deepstack_features` are the
    /// three `merger_list` outputs (each `[n_merged, out_hidden_size]`).
    pub fn forward_features(
        &self,
        pixels: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let dtype = self.pos_embed.embeddings().dtype();
        let xs = self.patch_embed.forward(&pixels.to_dtype(dtype)?)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let mut hidden_states = xs.add(&pos_embeds)?;

        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        let mut deepstack_features = Vec::new();
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
            if let Some(merger_idx) = self.deepstack_lookup[layer_idx] {
                let feat = self.deepstack_mergers[merger_idx].forward(&hidden_states)?;
                deepstack_features.push(feat);
            }
        }

        let merged = self.merger.forward(&hidden_states)?;
        Ok((merged, deepstack_features))
    }

    /// The merged Thinker-space embeds `[n_merged, out_hidden_size]` — the rows scattered into the
    /// image/video placeholder positions by [`super::modality::fuse_modalities`].
    pub fn forward(&self, pixels: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        Ok(self.forward_features(pixels, grid_thw)?.0)
    }
}

/// The Omni vision tower wrapped as a [`ModalityEncoder`]: pre-patchified pixels
/// `[num_patches, in_chans*temporal_patch_size*patch_size^2]` + the explicit `[n, 3]` (t, h, w) patch
/// grid -> Thinker-space `[n_merged, 2048]` token embeddings, scattered at the image (or video)
/// placeholder token id. One tower instance is shared (via `Arc`) by the image and video encoders —
/// they are the same `thinker.visual.*` weights.
///
/// The grid travels with the payload ([`ModalityInput::Image`] / [`ModalityInput::Video`]), so
/// `encode` hands it straight to [`OmniVisionTower::forward`]; non-square and multi-frame inputs are
/// exact (no square-grid derivation).
pub struct VisionModality {
    tower: std::sync::Arc<OmniVisionTower>,
    token: u32,
}

impl VisionModality {
    pub fn new(tower: std::sync::Arc<OmniVisionTower>, token: u32) -> Self {
        Self { tower, token }
    }
}

impl ModalityEncoder for VisionModality {
    fn placeholder_token(&self) -> u32 {
        self.token
    }

    fn encode(&self, input: &ModalityInput, _device: &Device) -> Result<Tensor> {
        let (pixels, grid_thw) = match input {
            ModalityInput::Image { pixels, grid_thw }
            | ModalityInput::Video { pixels, grid_thw } => (pixels, grid_thw),
            ModalityInput::Audio(_) => {
                hanzo_ml::bail!("VisionModality encodes ModalityInput::Image / ::Video only")
            }
        };
        self.tower.forward(pixels, grid_thw)
    }
}

#[cfg(test)]
mod vision_tests {
    use super::OmniVisionTower;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use crate::vision_models::qwen3_omni::config::Qwen3OmniConfig;
    use hanzo_ml::{DType, Device, Tensor};
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn read_f32_le(path: &PathBuf) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i64_le(path: &PathBuf) -> Vec<i64> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// Loads the REAL `thinker.visual.*` weights, runs [`OmniVisionTower::forward`] on the fixed
    /// deterministic image dumped by `dump_vision_ref.py`, and asserts the merged embeds match the HF
    /// `Qwen3OmniMoeVisionEncoder` reference to cosine > 0.99. Env-gated on `ZEN_OMNI_DIR` + the
    /// fixtures so CI without the checkpoint skips cleanly. F32 on CPU both sides.
    #[test]
    fn omni_vision_matches_reference() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        let fx = PathBuf::from(std::env::var("ZEN_OMNI_FIXTURES").unwrap_or_else(|_| {
            "/tmp/claude-1000/-home-z-work-lux/\
                 95715740-b8bb-4d96-8a9c-010a600ec9a6/scratchpad"
                .to_string()
        }));
        let hidden_path = fx.join("vis_hidden.f32");
        let grid_path = fx.join("vis_grid.i64");
        let emb_path = fx.join("vis_emb.f32");
        if !index.is_file() || !hidden_path.is_file() || !grid_path.is_file() || !emb_path.is_file()
        {
            eprintln!("[vision] weights/fixtures absent; skipping");
            return;
        }

        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg: Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap())
                .unwrap();
        let vcfg = &cfg.thinker_config.vision_config;

        // Minimal shard set: only the shards actually holding thinker.visual.* (all in shard 1).
        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shard_set = BTreeSet::new();
        for (k, v) in index_json["weight_map"].as_object().unwrap() {
            if k.starts_with("thinker.visual.") {
                shard_set.insert(v.as_str().unwrap().to_string());
            }
        }
        let paths: Vec<PathBuf> = shard_set.iter().map(|s| dirp.join(s)).collect();
        eprintln!("[vision] loading {} shard(s): {:?}", paths.len(), shard_set);

        let vb = from_mmaped_safetensors(
            paths,
            Vec::new(),
            Some(dtype),
            &device,
            vec![None],
            true,
            None,
            |name: String| name.starts_with("thinker.visual."),
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();

        let tower = OmniVisionTower::new(vcfg, vb.pp("thinker").pp("visual"), &device).unwrap();

        // Fixed deterministic input: [256, 1536] patches, grid [[1,16,16]].
        let hidden_v = read_f32_le(&hidden_path);
        let feat = vcfg.in_chans * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size;
        let n_patches = hidden_v.len() / feat;
        let hidden = Tensor::from_vec(hidden_v, (n_patches, feat), &device).unwrap();
        let grid_i64 = read_i64_le(&grid_path);
        let grid_u32: Vec<u32> = grid_i64.iter().map(|&x| x as u32).collect();
        let grid = Tensor::from_vec(grid_u32, (grid_i64.len() / 3, 3), &device).unwrap();
        eprintln!("[vision] input {:?} grid {:?}", hidden.dims(), grid.dims());

        let merged = tower.forward(&hidden, &grid).unwrap();
        eprintln!("[vision] rust merged {:?}", merged.dims());

        let got: Vec<f32> = merged
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let refv = read_f32_le(&emb_path);
        assert_eq!(
            got.len(),
            refv.len(),
            "emb len rust {} ref {}",
            got.len(),
            refv.len()
        );
        let cos = cosine(&got, &refv);
        let max_abs = got
            .iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[vision] cosine = {cos:.6}  max_abs_diff = {max_abs:.6}");
        assert!(cos > 0.99, "vision merged cosine {cos:.6} <= 0.99");
        eprintln!("[vision] PASS: cosine={cos:.6}");
    }
}
