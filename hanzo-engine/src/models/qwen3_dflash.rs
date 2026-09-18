#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! **DFlash 2** block-diffusion speculative draft model for Qwen3.
//!
//! A DFlash draft is NOT a normal decoder. It consumes several *target*
//! decoder-layer hidden states, fuses them into a fixed "context memory", seeds a
//! block of `block_size` slots (the last confirmed token at slot 0, `mask_token_id`
//! at the rest), and — in a single non-causal forward through `num_hidden_layers`
//! Qwen3 layers — drafts the whole block at once. Slot 0 is the anchor, never a
//! prediction, so one forward proposes `block_size - 1` tokens.
//!
//! DFlash **2** adds the two pieces that make a one-shot block worth drafting:
//!
//! * A **two-tap dynamic depthwise convolution** around each sublayer,
//!   `Conv(x)_t = k_{t,0}·x_t + k_{t,1}·x_{t-1}`, where each tap is a learned base
//!   kernel plus a per-slot correction predicted from the hidden state and shared
//!   across `conv_group_size` channels. The taps never reach past slot 0, so the
//!   block stays self-contained. This carries the short-range within-block work
//!   that attention cannot do while it is busy reading the target context, and is
//!   what stops draft quality decaying toward the end of the block.
//! * A **pairwise path selector**. Per-slot argmaxes disagree with each other — a
//!   repeated word, a broken phrase — and the block is then cut short at verify
//!   even though the right token was already in the slot's candidate list. The
//!   selector keeps each slot's `selector_top_k` candidates and scores every
//!   adjacent pair as `S_t(a, b) = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩`: the draft's own
//!   logit plus a low-rank bilinear match between the predecessor's and the
//!   candidate's codebook rows, gated by the slot's hidden state. Scoring is
//!   parallel; only the left-to-right walk over the precomputed scores is
//!   sequential, and it touches neither the backbone nor the LM head.
//!
//! Reference: `nemo_automodel/components/speculative/dflash/{draft_qwen3.py,
//! draft_qwen3_dflash2.py}`; checkpoint `incoai/Qwen3.8-27B-DFlash2`
//! (`architectures: ["DFlash2DraftModel"]`, 5 layers, `is_causal: false`,
//! `sliding_window: 2048`).
//!
//! Notes on reuse:
//! * The checkpoint carries **no** `embed_tokens` and **no** `lm_head` — a DFlash
//!   draft shares the target's. Both arrive as callbacks: the block seed is
//!   embedded through [`TargetTokenEmbedder`] and the block hidden states are
//!   decoded through [`TargetLmHead`], which is also what makes the draft's logits
//!   directly comparable with the verifier's.
//! * The Qwen3 decoder math (per-head q/k RMSNorm, GQA 32/8, head_dim 128, neox
//!   RoPE θ=1e7) is replicated here rather than through `models::qwen3::Attention`
//!   because that struct does kv-cache *self*-attention and cannot express
//!   attention over a fixed fused memory concatenated with a bidirectional block.

use hanzo_ml::{DType, Device, Module, Result, Shape, Tensor, D};
use hanzo_nn::Linear;
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
use serde::Deserialize;
use std::sync::Arc;

use crate::layers::{linear_no_bias, RmsNorm, RotaryEmbedding};
use crate::speculative::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    SpeculativeSharedHeads, TargetTokenEmbedder,
};

/// The target's `lm_head`, borrowed for one draft: `[slots, hidden] -> [slots, vocab]`.
/// The callback owns the cast into the target's dtype; the draft only reads the shape.
pub type TargetLmHead<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

/// The `dflash_config` block of the checkpoint `config.json` — everything that is
/// DFlash's own, as opposed to the Qwen3 decoder shape it inherits from the target.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlashParameters {
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub mask_token_id: u32,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layer_ids: Vec<i64>,
}

/// Draft config, parsed from the DFlash 2 checkpoint `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlash2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeParameters,
    pub dflash_config: DFlashParameters,
    /// Window width, in positions. Inert unless `use_sliding_window` is set —
    /// `Qwen3Config` zeroes it otherwise, and the draft reads the same rule.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Per-layer attention kind; only `sliding_attention` layers take the window.
    /// Absent ⇒ every layer is full attention.
    #[serde(default)]
    pub layer_types: Vec<String>,
}

impl DFlash2Config {
    pub fn from_json_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        serde_json::from_str(&s).map_err(|e| hanzo_ml::Error::msg(e.to_string()))
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    /// Number of fused target-layer hidden states (== `fc` input factor).
    pub fn num_fused_layers(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    /// Slots in one draft block, anchor included.
    pub fn block_size(&self) -> usize {
        self.dflash_config.block_size
    }

    /// Tokens one block actually proposes. Slot 0 holds the last confirmed token —
    /// the selector's starting predecessor — so the draft predicts the other slots.
    pub fn draft_len(&self) -> usize {
        self.dflash_config.block_size.saturating_sub(1)
    }

    /// Window for one layer, or `None` for full attention.
    pub fn layer_window(&self, layer: usize) -> Option<usize> {
        let window = if self.use_sliding_window {
            self.sliding_window
        } else {
            None
        };
        match self.layer_types.get(layer) {
            Some(kind) if kind == "sliding_attention" => window,
            _ => None,
        }
    }
}

/// The two-tap dynamic depthwise convolution wrapped around one sublayer.
///
/// One instance covers both the convolution *before* the sublayer ([`prepare`]) and
/// the one *after* it ([`finish`]); both sets of per-slot taps are predicted from
/// the sublayer's input, so the projection runs once per sublayer.
///
/// [`prepare`]: BlockConv::prepare
/// [`finish`]: BlockConv::finish
struct BlockConv {
    /// `[2, kernel_size, hidden]` — row 0 the pre-sublayer taps, row 1 the post-.
    base_kernel: Tensor,
    /// `hidden -> 2 * kernel_size * groups`, read back as `[2, kernel_size, groups]`.
    kernel_projection: Linear,
    kernel_size: usize,
    group_size: usize,
    groups: usize,
}

impl BlockConv {
    fn load(cfg: &DFlash2Config, vb: ShardedVarBuilder, name: &str) -> Result<Self> {
        let h = cfg.hidden_size;
        let kernel_size = cfg.dflash_config.conv_kernel_size;
        let group_size = cfg.dflash_config.conv_group_size;
        if kernel_size == 0 {
            return Err(hanzo_ml::Error::msg(
                "dflash_config.conv_kernel_size must be >= 1",
            ));
        }
        if group_size == 0 || !h.is_multiple_of(group_size) {
            return Err(hanzo_ml::Error::msg(format!(
                "dflash_config.conv_group_size {group_size} must divide hidden_size {h}"
            )));
        }
        let groups = h / group_size;
        Ok(Self {
            base_kernel: named(
                &vb,
                "base_kernel",
                (2, kernel_size, h),
                &format!("{name}.base_kernel: [pre/post, kernel, hidden]"),
            )?,
            kernel_projection: linear_no_bias(h, 2 * kernel_size * groups, vb.pp("kernel_projection"))
                .map_err(|e| {
                    hanzo_ml::Error::msg(format!(
                        "{name}.kernel_projection.weight: expected \
                         [2 * kernel {kernel_size} * groups {groups}, hidden {h}] — {e}"
                    ))
                })?,
            kernel_size,
            group_size,
            groups,
        })
    }

    /// Convolve the sublayer input and hand back the taps [`finish`] needs.
    ///
    /// [`finish`]: BlockConv::finish
    fn prepare(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let slots = hidden.dim(0)?;
        let dynamic = self
            .kernel_projection
            .forward(hidden)?
            .reshape((slots, 2, self.kernel_size, self.groups))?;
        let pre = dynamic.narrow(1, 0, 1)?.squeeze(1)?;
        let post = dynamic.narrow(1, 1, 1)?.squeeze(1)?;
        Ok((self.convolve(hidden, &pre, 0)?, post))
    }

    /// Convolve the sublayer output with the taps [`prepare`] produced.
    ///
    /// [`prepare`]: BlockConv::prepare
    fn finish(&self, hidden: &Tensor, dynamic: &Tensor) -> Result<Tensor> {
        self.convolve(hidden, dynamic, 1)
    }

    /// `out[t] = Σ_offset (base_kernel[half, offset] + dynamic[t, offset]) · x[t - offset]`.
    ///
    /// The base tap is per-channel, the correction per group of `group_size`
    /// channels. Slots before the block's start read zero rather than the previous
    /// block's tail, which is what keeps one block's draft independent of the last.
    fn convolve(&self, hidden: &Tensor, dynamic: &Tensor, half: usize) -> Result<Tensor> {
        let (slots, hidden_size) = hidden.dims2()?;
        let x = hidden.reshape((slots, self.groups, self.group_size))?;
        let base = self
            .base_kernel
            .narrow(0, half, 1)?
            .reshape((self.kernel_size, self.groups, self.group_size))?;
        let mut out: Option<Tensor> = None;
        for offset in 0..self.kernel_size {
            // A tap that reaches past the block start reads only zeros.
            if offset >= slots {
                break;
            }
            let shifted = if offset == 0 {
                x.clone()
            } else {
                x.narrow(0, 0, slots - offset)?.pad_with_zeros(0, offset, 0)?
            };
            let taps = base.narrow(0, offset, 1)?.broadcast_add(
                &dynamic
                    .narrow(1, offset, 1)?
                    .reshape((slots, self.groups, 1))?,
            )?;
            let term = taps.mul(&shifted)?;
            out = Some(match out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        match out {
            Some(acc) => acc.reshape((slots, hidden_size)),
            None => Tensor::zeros((slots, hidden_size), hidden.dtype(), hidden.device()),
        }
    }
}

/// Pairwise path selector over each slot's top-`k` candidates.
///
/// The codebooks are bare parameters, not `nn.Embedding` modules, so the
/// checkpoint keys are `candidate_selector.{predecessor,successor}_codebook` with
/// no trailing `.weight`.
struct CandidateSelector {
    /// `hidden -> rank`: the context gate `H(h_t)`.
    hidden_projection: Linear,
    /// `[vocab, rank]` — indexed by the *predecessor* token, the `A` factor.
    predecessor_codebook: Tensor,
    /// `[vocab, rank]` — indexed by the *candidate* token, the `B` factor.
    successor_codebook: Tensor,
    top_k: usize,
}

impl CandidateSelector {
    fn load(cfg: &DFlash2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let rank = cfg.dflash_config.selector_rank;
        let top_k = cfg.dflash_config.selector_top_k;
        let vocab = cfg.vocab_size;
        if rank == 0 {
            return Err(hanzo_ml::Error::msg("dflash_config.selector_rank must be >= 1"));
        }
        if top_k == 0 || top_k > vocab {
            return Err(hanzo_ml::Error::msg(format!(
                "dflash_config.selector_top_k {top_k} must be in 1..={vocab}"
            )));
        }
        Ok(Self {
            hidden_projection: linear_no_bias(
                cfg.hidden_size,
                rank,
                vb.pp("hidden_projection"),
            )
            .map_err(|e| {
                hanzo_ml::Error::msg(format!(
                    "candidate_selector.hidden_projection.weight: expected \
                     [rank {rank}, hidden {}] — {e}",
                    cfg.hidden_size
                ))
            })?,
            predecessor_codebook: named(
                &vb,
                "predecessor_codebook",
                (vocab, rank),
                "candidate_selector.predecessor_codebook: [vocab, rank]",
            )?,
            successor_codebook: named(
                &vb,
                "successor_codebook",
                (vocab, rank),
                "candidate_selector.successor_codebook: [vocab, rank]",
            )?,
            top_k,
        })
    }

    /// Trace one coherent path left to right through the per-slot candidate lists.
    ///
    /// Slot `t`'s score is `U_t(b) + ⟨A(prev) ⊙ H(h_t), B(b)⟩`, where `prev` is the
    /// token the walk committed at `t-1` (the anchor at `t = 0`). Returns the path
    /// plus, per slot, a full-vocabulary logit row carrying those scores on the
    /// candidates and `-inf` elsewhere: `softmax` of a row is exactly the
    /// distribution the walk drew from, which is the `q` the verifier's rejection
    /// step needs. A draft supported only on the candidate set is the point — the
    /// walk can never emit anything outside it.
    fn walk(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchor_token: u32,
        temperature: f64,
    ) -> Result<(Vec<u32>, Tensor)> {
        let device = hidden.device().clone();
        let (slots, vocab) = logits.dims2()?;
        let top_k = self.top_k.min(vocab);

        // The walk is host code over each slot's candidates, so the candidates come from a host
        // partial select over the logit rows; nothing on the device sorts a vocabulary-wide row.
        let rows_h = logits.contiguous()?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let (candidates, unary): (Vec<Vec<u32>>, Vec<Vec<f32>>) =
            rows_h.iter().map(|row| top_k_of(row, top_k)).unzip();
        let gate_hidden = self.hidden_projection.forward(hidden)?; // [slots, rank]

        let mut tokens = Vec::with_capacity(slots);
        let mut rows = vec![f32::NEG_INFINITY; slots * vocab];
        let mut previous = anchor_token;
        for slot in 0..slots {
            let previous_t = Tensor::from_vec(vec![previous], (1,), &device)?;
            let gate = self
                .predecessor_codebook
                .index_select(&previous_t, 0)?
                .mul(&gate_hidden.narrow(0, slot, 1)?)?; // [1, rank]
            let ids_h = &candidates[slot];
            if ids_h.is_empty() {
                hanzo_ml::bail!("slot {slot} has no finite logit");
            }
            let ids = Tensor::from_vec(ids_h.clone(), (ids_h.len(),), &device)?;
            let successors = self.successor_codebook.index_select(&ids, 0)?; // [top_k, rank]
            let pairwise = gate
                .matmul(&successors.t()?.contiguous()?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let scores: Vec<f32> = unary[slot]
                .iter()
                .zip(pairwise[0].iter())
                .map(|(u, p)| u + p)
                .collect();
            for (i, &id) in candidates[slot].iter().enumerate() {
                rows[slot * vocab + id as usize] = scores[i];
            }
            previous = candidates[slot][select_candidate(&scores, temperature)];
            tokens.push(previous);
        }
        Ok((tokens, Tensor::from_vec(rows, (slots, vocab), &device)?))
    }
}

/// One DFlash 2 decoder layer: a Qwen3 layer whose attention keys/values come from
/// `[fused_context ‖ block]`, with a two-tap conv wrapped around each sublayer.
struct DFlash2Layer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    attention_conv: BlockConv,
    mlp_conv: BlockConv,
}

impl DFlash2Layer {
    fn load(cfg: &DFlash2Config, vb: ShardedVarBuilder, index: usize) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        let eps = cfg.rms_norm_eps;

        let va = vb.pp("self_attn");
        Ok(Self {
            q_proj: linear_no_bias(h, q_dim, va.pp("q_proj"))?,
            k_proj: linear_no_bias(h, kv_dim, va.pp("k_proj"))?,
            v_proj: linear_no_bias(h, kv_dim, va.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, h, va.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, eps, va.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, eps, va.pp("k_norm"))?,
            input_layernorm: RmsNorm::new(h, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(h, eps, vb.pp("post_attention_layernorm"))?,
            gate_proj: linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp").pp("gate_proj"))?,
            up_proj: linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp").pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, h, vb.pp("mlp").pp("down_proj"))?,
            attention_conv: BlockConv::load(
                cfg,
                vb.pp("attention_conv"),
                &format!("layers.{index}.attention_conv"),
            )?,
            mlp_conv: BlockConv::load(
                cfg,
                vb.pp("mlp_conv"),
                &format!("layers.{index}.mlp_conv"),
            )?,
        })
    }
}

/// Native Qwen3 DFlash 2 draft model.
pub struct Qwen3DFlash2 {
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DFlash2Layer>,
    norm: RmsNorm,
    selector: CandidateSelector,
    rotary: RotaryEmbedding,
    cfg: DFlash2Config,
    device: Device,
    dtype: DType,
}

impl Qwen3DFlash2 {
    /// The dtype the draft computes in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn config(&self) -> &DFlash2Config {
        &self.cfg
    }

    /// Load every DFlash 2 tensor from a root VarBuilder (no `model.` prefix).
    pub fn load(cfg: DFlash2Config, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let device = vb.device().clone();
        let dtype = vb.dtype();

        if cfg.block_size() < 2 {
            return Err(hanzo_ml::Error::msg(format!(
                "dflash_config.block_size {} leaves no slot to predict — slot 0 is the anchor",
                cfg.block_size()
            )));
        }
        if cfg.num_fused_layers() == 0 {
            return Err(hanzo_ml::Error::msg(
                "dflash_config.target_layer_ids is empty — there is no context to fuse",
            ));
        }

        let fc = linear_no_bias(cfg.num_fused_layers() * h, h, vb.pp("fc")).map_err(|e| {
            hanzo_ml::Error::msg(format!(
                "fc.weight: expected [hidden {h}, {} fused layers * hidden {h}] — {e}",
                cfg.num_fused_layers()
            ))
        })?;
        let hidden_norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("hidden_norm"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DFlash2Layer::load(&cfg, vb.pp("layers").pp(i), i)?);
        }

        let norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("norm"))?;
        let selector = CandidateSelector::load(&cfg, vb.pp("candidate_selector"))?;

        let rotary = RotaryEmbedding::new(
            cfg.rope_theta() as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            &device,
            true, // gpt-neox / rotate_half convention (Qwen3)
            dtype,
        )?;

        Ok(Self {
            fc,
            hidden_norm,
            layers,
            norm,
            selector,
            rotary,
            cfg,
            device,
            dtype,
        })
    }

    /// How far before the anchor any layer attends: the widest window, or `None` when a
    /// full-attention layer sees the whole context.
    pub fn reach(&self) -> Option<usize> {
        (0..self.layers.len())
            .map(|layer| self.cfg.layer_window(layer))
            .try_fold(0usize, |widest, window| window.map(|w| widest.max(w)))
    }

    /// Draft one block: `block_size - 1` tokens from one forward.
    ///
    /// * `target_hiddens` — the `num_fused_layers` target decoder-layer hidden
    ///   states (`target_layer_ids`) for the positions the window holds. The draft
    ///   reads what falls inside its reach before the anchor; a window that starts
    ///   later than that only shortens its context.
    /// * `anchor_token` — the last confirmed token, seeding block slot 0.
    /// * `anchor_pos` — RoPE position of the anchor; block slot `i` sits at
    ///   `anchor_pos + i`.
    /// * `embed` / `lm_head` — the target's embedding table and head; DFlash has
    ///   neither of its own.
    /// * `temperature` — selector walk temperature (`< 1e-5` ⇒ argmax).
    ///
    /// Returns `(tokens [block_size - 1] u32, logits [block_size - 1, vocab] f32)`,
    /// where each logit row is the selector's proposal distribution for that slot.
    pub fn draft_block(
        &self,
        target_hiddens: &crate::speculative::HiddenWindow,
        anchor_token: u32,
        anchor_pos: usize,
        embed: &TargetTokenEmbedder<'_>,
        lm_head: &TargetLmHead<'_>,
        temperature: f64,
    ) -> Result<(Tensor, Tensor)> {
        let dev = &self.device;
        let bs = self.cfg.block_size();

        if target_hiddens.layers.len() != self.cfg.num_fused_layers() {
            return Err(hanzo_ml::Error::msg(format!(
                "draft_block expected {} target hidden states, got {}",
                self.cfg.num_fused_layers(),
                target_hiddens.layers.len()
            )));
        }
        if anchor_token as usize >= self.cfg.vocab_size {
            return Err(hanzo_ml::Error::msg(format!(
                "anchor token {anchor_token} is outside the draft vocab {}",
                self.cfg.vocab_size
            )));
        }

        // 1. Fuse: concat the selected target hiddens along the feature dim, project
        //    through `fc`, normalise. This is the whole context the draft attends to.
        //
        //    Each layer's window bounds how far back a slot can see, so a context row older
        //    than the widest window is masked for every query in every layer and contributes
        //    exactly nothing. Dropping those rows before the fuse makes a draft cost
        //    O(window) rather than O(context) — at a 100K-token prefix, the difference
        //    between a draft and a second prefill. A full-attention layer sees everything,
        //    and then nothing is dropped.
        //
        //    Rows at or past the anchor are never attended either, so the usable context is
        //    `[ctx_start, ctx_end)` with `ctx_end` clamped to the anchor. Every fused layer
        //    covers the same positions, so one range slices them all.
        let held = target_hiddens.start..target_hiddens.end()?;
        let std::ops::Range {
            start: ctx_start,
            end: ctx_end,
        } = context_range(self.reach(), held, anchor_pos);
        let ctx_parts = target_hiddens
            .rows(ctx_start, ctx_end)?
            .iter()
            .map(|t| t.to_device(dev)?.to_dtype(self.dtype))
            .collect::<Result<Vec<_>>>()?;
        let ctx_refs: Vec<&Tensor> = ctx_parts.iter().collect();
        let ctx_cat = Tensor::cat(&ctx_refs, D::Minus1)?; // [ctx_len, num_fused * hidden]
        let ctx_len = ctx_cat.dim(0)?;
        let tctx = self.hidden_norm.forward(&self.fc.forward(&ctx_cat)?)?; // [ctx_len, hidden]

        // 2. Seed the block: [anchor, MASK, MASK, ...], embedded through the target.
        let mut ids = Vec::with_capacity(bs);
        ids.push(anchor_token);
        ids.extend(std::iter::repeat_n(self.cfg.dflash_config.mask_token_id, bs - 1));
        let ids_t = Tensor::from_vec(ids, (1, bs), dev)?;
        let mut hstate = as_2d(&embed(&ids_t)?)?.to_dtype(self.dtype)?; // [bs, hidden]

        // 3. RoPE tables: draft positions for q, context+draft positions for k.
        let draft_pos: Vec<u32> = (0..bs as u32).map(|i| anchor_pos as u32 + i).collect();
        let mut full_pos: Vec<u32> = (ctx_start as u32..(ctx_start + ctx_len) as u32).collect();
        full_pos.extend_from_slice(&draft_pos);
        let draft_pos_t = Tensor::from_vec(draft_pos, (bs,), dev)?;
        let full_pos_t = Tensor::from_vec(full_pos, (ctx_len + bs,), dev)?;
        let (cos_draft, sin_draft) = self.rope_cos_sin(&draft_pos_t)?;
        let (cos_full, sin_full) = self.rope_cos_sin(&full_pos_t)?;

        // 4. One mask per distinct window. Every layer of the published checkpoint is
        //    `sliding_attention` with the same width, so this builds exactly one and
        //    the rest are refcount bumps.
        let mut masks: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        for layer in 0..self.layers.len() {
            let window = self.cfg.layer_window(layer);
            let mask = match (0..layer).find(|&earlier| self.cfg.layer_window(earlier) == window) {
                Some(earlier) => masks[earlier].clone(),
                None => block_mask(
                    ctx_start,
                    ctx_len,
                    bs,
                    anchor_pos,
                    window,
                    &self.device,
                    self.dtype,
                )?,
            };
            masks.push(mask);
        }

        // 5. Backbone. Each sublayer is convolved on the way in and on the way out,
        //    with the taps predicted once from the sublayer's input.
        for (layer, mask) in self.layers.iter().zip(masks.iter()) {
            let residual = hstate;
            let normed = layer.input_layernorm.forward(&residual)?;
            let (normed, attention_taps) = layer.attention_conv.prepare(&normed)?;
            let attn = self.attention(
                layer, &normed, &tctx, &cos_draft, &sin_draft, &cos_full, &sin_full, mask,
            )?;
            let attn = layer.attention_conv.finish(&attn, &attention_taps)?;
            hstate = residual.add(&attn)?;

            let residual = hstate;
            let normed = layer.post_attention_layernorm.forward(&residual)?;
            let (normed, mlp_taps) = layer.mlp_conv.prepare(&normed)?;
            let gate = hanzo_nn::ops::silu(&layer.gate_proj.forward(&normed)?)?;
            let up = layer.up_proj.forward(&normed)?;
            let mlp = layer.down_proj.forward(&gate.mul(&up)?)?;
            let mlp = layer.mlp_conv.finish(&mlp, &mlp_taps)?;
            hstate = residual.add(&mlp)?;
        }
        let hstate = self.norm.forward(&hstate)?; // [bs, hidden]

        // 6. Slot 0 carries the anchor and is never a prediction; decode the rest
        //    through the target's head.
        let predicted = hstate.narrow(0, 1, bs - 1)?.contiguous()?; // [bs - 1, hidden]
        let logits = lm_head(&predicted)?;
        let logits = as_2d(&logits)?.to_dtype(DType::F32)?; // [bs - 1, vocab]
        let vocab = logits.dim(D::Minus1)?;
        if vocab > self.cfg.vocab_size {
            return Err(hanzo_ml::Error::msg(format!(
                "target head emitted {vocab} logits but the selector codebooks cover {}",
                self.cfg.vocab_size
            )));
        }

        // 7. Walk the selector for a block whose slots agree with each other.
        let (tokens, proposal_logits) =
            self.selector
                .walk(&predicted, &logits, anchor_token, temperature)?;

        let tokens_t = Tensor::from_vec(tokens, (bs - 1,), dev)?;
        Ok((tokens_t, proposal_logits))
    }

    // --- internals -------------------------------------------------------

    /// DFlash attention for one layer. `x`: block hidden `[bs, hidden]`;
    /// `tctx`: fused context `[ctx_len, hidden]`. Queries are the block only;
    /// keys/values span `[context ‖ block]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &DFlash2Layer,
        x: &Tensor,
        tctx: &Tensor,
        cos_draft: &Tensor,
        sin_draft: &Tensor,
        cos_full: &Tensor,
        sin_full: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let bs = x.dim(0)?;
        let ctx_len = tctx.dim(0)?;
        let kl = ctx_len + bs;
        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let nrep = nh / nkv;
        let eps = self.cfg.rms_norm_eps as f32;

        // Query: block only.
        let q = layer.q_proj.forward(x)?.reshape((bs, nh, hd))?;
        let q = hanzo_nn::ops::rms_norm(&q.contiguous()?, layer.q_norm.weight(), eps)?;
        let q = q.transpose(0, 1)?.contiguous()?; // [nh, bs, hd]
        let q = apply_rope(&q, cos_draft, sin_draft)?;

        // Keys/values: [fused context ‖ block].
        let k_ctx = layer.k_proj.forward(tctx)?;
        let k_noise = layer.k_proj.forward(x)?;
        let k = Tensor::cat(&[&k_ctx, &k_noise], 0)?.reshape((kl, nkv, hd))?;
        let k = hanzo_nn::ops::rms_norm(&k.contiguous()?, layer.k_norm.weight(), eps)?;
        let k = k.transpose(0, 1)?.contiguous()?; // [nkv, kl, hd]
        let k = apply_rope(&k, cos_full, sin_full)?;
        let k = repeat_kv(&k, nrep)?; // [nh, kl, hd]

        let v_ctx = layer.v_proj.forward(tctx)?;
        let v_noise = layer.v_proj.forward(x)?;
        let v = Tensor::cat(&[&v_ctx, &v_noise], 0)?
            .reshape((kl, nkv, hd))?
            .transpose(0, 1)?
            .contiguous()?; // [nkv, kl, hd]
        let v = repeat_kv(&v, nrep)?; // [nh, kl, hd]

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)? // [nh, bs, kl]
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;
        let probs = hanzo_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v.contiguous()?)?; // [nh, bs, hd]
        let out = out.transpose(0, 1)?.contiguous()?.reshape((bs, nh * hd))?;
        layer.o_proj.forward(&out)
    }

    /// Gather + expand the neox RoPE cos/sin rows for the given positions.
    fn rope_cos_sin(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rotary.get_cos_sin()?; // [max_pos, hd/2]
        let cos = cos.index_select(positions, 0)?; // [L, hd/2]
        let sin = sin.index_select(positions, 0)?;
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?; // [L, hd]
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        Ok((cos, sin))
    }

}

/// The context positions a draft at `anchor_pos` reads, given the positions `held`. Rows at or
/// past the anchor are never attended, and nothing older than `reach` is visible to any layer.
fn context_range(
    reach: Option<usize>,
    held: std::ops::Range<usize>,
    anchor_pos: usize,
) -> std::ops::Range<usize> {
    let end = anchor_pos.min(held.end);
    let start = reach
        .map_or(0, |w| end.saturating_sub(w))
        .max(held.start)
        .min(end);
    start..end
}

/// Additive attention mask `[bs, ctx_len + bs]`: `0` where attended, `-inf`
/// where masked. Context column `j` holds absolute position `ctx_start + j`.
/// Context columns are attended iff they precede the anchor;
/// every block column is attended (the block is bidirectional — a slot reads
/// the mask tokens after it, which is what lets one forward fill the block).
/// A `window` additionally keeps only keys within it, symmetrically and
/// exclusively, matching the reference's decode-time band.
fn block_mask(
    ctx_start: usize,
    ctx_len: usize,
    bs: usize,
    anchor_pos: usize,
    window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let kl = ctx_len + bs;
    let neg = f32::NEG_INFINITY;
    let mut data = Vec::with_capacity(bs * kl);
    for i in 0..bs {
        let query_pos = anchor_pos + i;
        for j in 0..kl {
            let in_block = j >= ctx_len;
            let key_pos = if in_block {
                anchor_pos + (j - ctx_len)
            } else {
                ctx_start + j
            };
            let mut keep = in_block || key_pos < anchor_pos;
            if let Some(w) = window {
                keep &= query_pos.abs_diff(key_pos) < w;
            }
            data.push(if keep { 0f32 } else { neg });
        }
    }
    Tensor::from_vec(data, (bs, kl), device)?.to_dtype(dtype)
}

/// `vb.get`, but the error says what the tensor is FOR. The shard reports a bare
/// path and a shape mismatch; a draft whose `dflash_config` disagrees with its own
/// weights is otherwise a puzzle to read.
fn named<S: Into<Shape>>(
    vb: &ShardedVarBuilder,
    name: &str,
    shape: S,
    layout: &str,
) -> Result<Tensor> {
    let shape: Shape = shape.into();
    let dims = shape.dims().to_vec();
    vb.get(shape, name)
        .map_err(|e| hanzo_ml::Error::msg(format!("{layout} = {dims:?} — {e}")))
}

/// Squeeze an optional leading batch dim: `[1, n, h]` or `[n, h]` → `[n, h]`.
fn as_2d(t: &Tensor) -> Result<Tensor> {
    match t.rank() {
        2 => Ok(t.clone()),
        3 => {
            let (b, _, _) = t.dims3()?;
            if b != 1 {
                return Err(hanzo_ml::Error::msg(format!(
                    "draft tensor batch dim must be 1, got {b}"
                )));
            }
            t.squeeze(0)
        }
        r => Err(hanzo_ml::Error::msg(format!(
            "draft tensor must be rank 2 or 3, got rank {r}"
        ))),
    }
}

/// neox `rotate_half`: split the last dim in half → `[-x2, x1]`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply neox RoPE. `x`: `[heads, L, head_dim]`; `cos`/`sin`: `[L, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(0)?; // [1, L, head_dim]
    let sin = sin.unsqueeze(0)?;
    let rot = rotate_half(x)?;
    x.broadcast_mul(&cos)?.add(&rot.broadcast_mul(&sin)?)
}

/// GQA key/value expansion (repeat_interleave): `[nkv, L, d]` → `[nkv*nrep, L, d]`.
fn repeat_kv(x: &Tensor, nrep: usize) -> Result<Tensor> {
    if nrep == 1 {
        return Ok(x.clone());
    }
    let (nkv, l, d) = x.dims3()?;
    x.unsqueeze(1)?
        .broadcast_as((nkv, nrep, l, d))?
        .contiguous()?
        .reshape((nkv * nrep, l, d))
}

/// Pick one candidate slot from its scores: `temperature < 1e-5` ⇒ first argmax,
/// else a multinomial draw over `softmax(scores / temperature)`. The draw is over
/// the selector's `top_k` candidates, not the vocabulary — the walk cannot leave
/// the candidate set.
/// The `k` largest entries of `row`, largest first: their indices and values. NaN is never
/// chosen; ties go to the lower index.
fn top_k_of(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut idx: Vec<u32> = (0..row.len() as u32)
        .filter(|&i| !row[i as usize].is_nan())
        .collect();
    let k = k.min(idx.len());
    let larger_first = |&a: &u32, &b: &u32| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .expect("NaN was filtered out")
            .then(a.cmp(&b))
    };
    if k > 0 {
        idx.select_nth_unstable_by(k - 1, larger_first);
    }
    idx.truncate(k);
    idx.sort_unstable_by(larger_first);
    let vals = idx.iter().map(|&i| row[i as usize]).collect();
    (idx, vals)
}

fn select_candidate(scores: &[f32], temperature: f64) -> usize {
    if scores.is_empty() {
        return 0;
    }
    if temperature < 1e-5 {
        let mut best = 0usize;
        for (i, s) in scores.iter().enumerate() {
            if *s > scores[best] {
                best = i;
            }
        }
        return best;
    }
    let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores
        .iter()
        .map(|v| (f64::from(*v - mx) / temperature).exp() as f32)
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    use rand::Rng;
    let r: f32 = rand::rng().random::<f32>();
    let mut acc = 0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Build a root VarBuilder over one DFlash 2 safetensors shard.
pub fn dflash_varbuilder(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<ShardedVarBuilder> {
    let predicate: Arc<dyn Fn(String) -> bool + Send + Sync> = Arc::new(|_| true);
    unsafe { ShardedSafeTensors::sharded(&[path], dtype, device, None, predicate) }
}

/// The `SpeculativeProposer` adapter plugging DFlash 2 into the generic speculative driver.
///
/// One forward through [`Qwen3DFlash2`] drafts a whole block from the target's
/// multi-layer hidden prefix (`ctx.target_hidden_layers`, one `[prefix_len, hidden]`
/// tensor per fused target layer) plus the just-sampled anchor. Correctness comes
/// from the target verify forward, never from the draft.
///
/// Holds no per-sequence state — every block is drafted from the target's hidden
/// prefix — so the default `retain_seqs` no-op is already correct.
pub struct DFlash2Proposer {
    draft: Qwen3DFlash2,
    /// The target's embedding and output head. DFlash carries neither, and decoding
    /// through the target's own head is what puts the draft's logits on the verifier's
    /// scale. Held here because the pipelines call `propose` without an embedder.
    heads: SpeculativeSharedHeads,
}

impl DFlash2Proposer {
    pub fn new(draft: Qwen3DFlash2, heads: SpeculativeSharedHeads) -> Self {
        Self { draft, heads }
    }

    /// Load the draft at `dir` (`config.json` + `model.safetensors`) beside a target that
    /// lends `heads`. A non-zero `block_size` shortens the drafted block: fewer masked slots
    /// is a valid draft, while a block longer than the trained one asks for positions the
    /// draft never learned to fill. The draft runs F32 on CPU and BF16 on accelerators — its
    /// dtype moves the accept rate only, since the target verify decides every emitted token.
    pub fn from_checkpoint(
        dir: &std::path::Path,
        block_size: usize,
        device: &Device,
        heads: SpeculativeSharedHeads,
    ) -> Result<Self> {
        let mut cfg = DFlash2Config::from_json_file(dir.join("config.json"))?;
        if block_size != 0 {
            if !(2..=cfg.block_size()).contains(&block_size) {
                hanzo_ml::bail!(
                    "dflash block_size {block_size} is outside 2..={}, the block this checkpoint was trained on",
                    cfg.block_size()
                );
            }
            cfg.dflash_config.block_size = block_size;
        }
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let vb = dflash_varbuilder(&dir.join("model.safetensors"), dtype, device)?;
        Ok(Self::new(Qwen3DFlash2::load(cfg, vb)?, heads))
    }

    /// What the target must capture for this draft: its fused layers, in its dtype, and as
    /// many positions as it reads. That is its reach before the anchor plus one block, since a
    /// verify leaves up to a block of rows past the anchor before rejection trims them. A
    /// full-attention layer reads the whole prefix.
    pub fn capture_request(&self) -> crate::speculative::CaptureRequest {
        let cfg = self.draft.config();
        crate::speculative::CaptureRequest {
            layers: cfg
                .dflash_config
                .target_layer_ids
                .iter()
                .map(|&id| id as usize)
                .collect(),
            retain: self.draft.reach().map(|reach| reach + cfg.block_size()),
            dtype: Some(self.draft.dtype()),
        }
    }

    pub fn block_size(&self) -> usize {
        self.draft.config().block_size()
    }
}

impl SpeculativeProposer for DFlash2Proposer {
    fn proposal_len(&self) -> usize {
        self.draft.config().draft_len()
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let batch = ctx.sampled_tokens.len();
        // The target captures hiddens on single-sequence forwards only, so a batched step —
        // or a sequence whose prefix was never captured — has nothing to draft from. That is
        // an ordinary state under concurrency, not a failure: propose nothing and the target
        // decodes those sequences one token at a time.
        let stand_down = || {
            Ok(SpeculativeProposalBatch::new(vec![
                SpeculativeProposal::new(Vec::new());
                batch
            ]))
        };
        if batch != 1 {
            return stand_down();
        }
        let Some(hiddens) = ctx.target_hidden_layers.as_ref() else {
            return stand_down();
        };
        let anchor_token = ctx.sampled_tokens[0];
        let anchor_pos = ctx.base_lens[0];

        // A window that ends before the anchor, or starts at or after it, holds nothing the
        // draft can read (right after a discontinuity): propose nothing and the target decodes
        // one token.
        if hiddens.end()? < anchor_pos || hiddens.start >= anchor_pos {
            return stand_down();
        }

        let embed_fn = Arc::clone(&self.heads.embed);
        let embed = move |ids: &Tensor| embed_fn(ids);
        let head = Arc::clone(&self.heads.lm_head);
        let lm_head = move |hidden: &Tensor| head(hidden);
        // Deterministic walk (argmax): draft quality only moves the accept rate, and
        // the target verify decides every emitted token.
        let (tokens_t, logits) =
            self.draft
                .draft_block(hiddens, anchor_token, anchor_pos, &embed, &lm_head, 0.0)?;

        let tokens: Vec<u32> = tokens_t.to_vec1::<u32>()?;
        // The verifier indexes logit rows by draft position and expects [1, n, vocab].
        let logits = logits.unsqueeze(0)?;

        Ok(SpeculativeProposalBatch::new(vec![
            SpeculativeProposal::with_logits(tokens, logits),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CKPT_DIR: &str = "/home/z/work/zen/hf/dflash/Qwen3.8-27B-DFlash2";

    /// Checkpoint override (env `DFLASH_CKPT`) so the load tests can point at ANY
    /// DFlash 2 checkpoint.
    fn ckpt_dir() -> String {
        std::env::var("DFLASH_CKPT").unwrap_or_else(|_| CKPT_DIR.to_string())
    }

    /// Build a `BlockConv` straight from tensors: the conv is pure arithmetic, so it
    /// is testable without a checkpoint.
    fn conv(base: Vec<f32>, hidden: usize, kernel: usize, group_size: usize) -> Result<BlockConv> {
        let dev = Device::Cpu;
        let groups = hidden / group_size;
        Ok(BlockConv {
            base_kernel: Tensor::from_vec(base, (2, kernel, hidden), &dev)?,
            // Zero projection ⇒ no per-slot correction, so only `base_kernel` acts.
            kernel_projection: Linear::new(
                Tensor::zeros((2 * kernel * groups, hidden), DType::F32, &dev)?,
                None,
            ),
            kernel_size: kernel,
            group_size,
            groups,
        })
    }

    /// The published init — unit self-tap, zero predecessor tap, zero correction —
    /// makes the conv the identity, which is what lets a fresh DFlash 2 start out
    /// numerically equal to plain DFlash.
    #[test]
    fn dflash_conv_identity_is_passthrough() -> Result<()> {
        let dev = Device::Cpu;
        let (hidden, kernel, group_size, slots) = (4usize, 2usize, 2usize, 3usize);
        let mut base = vec![0f32; 2 * kernel * hidden];
        for half in 0..2 {
            for c in 0..hidden {
                base[half * kernel * hidden + c] = 1.0; // tap 0 == self
            }
        }
        let c = conv(base, hidden, kernel, group_size)?;
        let x = Tensor::from_vec(
            (0..(slots * hidden)).map(|v| v as f32).collect::<Vec<_>>(),
            (slots, hidden),
            &dev,
        )?;
        let (pre, taps) = c.prepare(&x)?;
        assert_eq!(pre.to_vec2::<f32>()?, x.to_vec2::<f32>()?);
        assert_eq!(c.finish(&x, &taps)?.to_vec2::<f32>()?, x.to_vec2::<f32>()?);
        Ok(())
    }

    /// The predecessor tap reads slot `t-1` of the SAME block; slot 0 reads zero.
    /// That boundary is what keeps one block's draft independent of the last.
    #[test]
    fn dflash_conv_predecessor_tap_stops_at_the_block_start() -> Result<()> {
        let dev = Device::Cpu;
        let (hidden, kernel, group_size, slots) = (4usize, 2usize, 2usize, 3usize);
        let mut base = vec![0f32; 2 * kernel * hidden];
        for half in 0..2 {
            for c in 0..hidden {
                base[half * kernel * hidden + hidden + c] = 1.0; // tap 1 == predecessor
            }
        }
        let c = conv(base, hidden, kernel, group_size)?;
        let x = Tensor::from_vec(
            (0..(slots * hidden)).map(|v| (v + 1) as f32).collect::<Vec<_>>(),
            (slots, hidden),
            &dev,
        )?;
        let (out, _) = c.prepare(&x)?;
        let out = out.to_vec2::<f32>()?;
        let src = x.to_vec2::<f32>()?;
        assert_eq!(out[0], vec![0f32; hidden], "slot 0 must read zero padding");
        assert_eq!(out[1], src[0]);
        assert_eq!(out[2], src[1]);
        Ok(())
    }

    /// The selector overrides a per-slot argmax when the pairwise term says the
    /// neighbours disagree — the whole point of walking a path instead of taking
    /// each slot's top-1 independently.
    #[test]
    fn dflash_selector_walk_follows_the_pairwise_score() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, rank, hidden, slots) = (6usize, 2usize, 2usize, 1usize);

        // Gate = predecessor_codebook[prev] * hidden_projection(h). Identity
        // projection and a unit predecessor row make the gate the anchor's row.
        let mut predecessor = vec![0f32; vocab * rank];
        predecessor[3 * rank] = 1.0; // anchor token 3 -> [1, 0]
        let mut successor = vec![0f32; vocab * rank];
        successor[5 * rank] = 10.0; // token 5 scores +10 after token 3
        let selector = CandidateSelector {
            // 0.5 over a 2-wide all-ones hidden makes H(h) = [1, 1], so the gate is
            // exactly the predecessor's codebook row and the arithmetic stays legible.
            hidden_projection: Linear::new(Tensor::full(0.5f32, (rank, hidden), &dev)?, None),
            predecessor_codebook: Tensor::from_vec(predecessor, (vocab, rank), &dev)?,
            successor_codebook: Tensor::from_vec(successor, (vocab, rank), &dev)?,
            top_k: 3,
        };

        let hidden_t = Tensor::ones((slots, hidden), DType::F32, &dev)?;
        // Token 4 is the unary argmax; token 5 trails it but wins on the pair score.
        let mut logits = vec![0f32; slots * vocab];
        logits[4] = 2.0;
        logits[5] = 1.0;
        logits[1] = 0.5;
        let logits = Tensor::from_vec(logits, (slots, vocab), &dev)?;

        let (tokens, rows) = selector.walk(&hidden_t, &logits, 3, 0.0)?;
        assert_eq!(tokens, vec![5]);

        // The returned row is the proposal distribution: scores on the candidates,
        // -inf everywhere else, so softmax is exactly what the walk drew from.
        let row = rows.to_vec2::<f32>()?[0].clone();
        assert!((row[5] - 11.0).abs() < 1e-5, "row[5] = {}", row[5]);
        assert!((row[4] - 2.0).abs() < 1e-5, "row[4] = {}", row[4]);
        assert!(row[0].is_infinite() && row[0].is_sign_negative());
        assert!(row[2].is_infinite() && row[2].is_sign_negative());
        Ok(())
    }

    /// With a zero successor codebook every score collapses to the draft's own
    /// logit, so the walk degenerates to per-slot argmax — the published init.
    #[test]
    fn dflash_selector_zero_codebook_is_plain_argmax() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, rank, hidden, slots) = (5usize, 2usize, 2usize, 2usize);
        let selector = CandidateSelector {
            hidden_projection: Linear::new(Tensor::ones((rank, hidden), DType::F32, &dev)?, None),
            predecessor_codebook: Tensor::ones((vocab, rank), DType::F32, &dev)?,
            successor_codebook: Tensor::zeros((vocab, rank), DType::F32, &dev)?,
            top_k: 2,
        };
        let hidden_t = Tensor::ones((slots, hidden), DType::F32, &dev)?;
        let mut logits = vec![0f32; slots * vocab];
        logits[2] = 5.0; // slot 0 -> token 2
        logits[vocab + 4] = 5.0; // slot 1 -> token 4
        let logits = Tensor::from_vec(logits, (slots, vocab), &dev)?;
        let (tokens, _) = selector.walk(&hidden_t, &logits, 0, 0.0)?;
        assert_eq!(tokens, vec![2, 4]);
        Ok(())
    }

    /// `layer_window` must honour BOTH switches: a window only applies to a
    /// `sliding_attention` layer of a config that turned sliding on.
    #[test]
    fn dflash_layer_window_needs_both_switches() -> Result<()> {
        let mut cfg: DFlash2Config = serde_json::from_str(
            r#"{
              "vocab_size": 8, "hidden_size": 4, "intermediate_size": 8,
              "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
              "head_dim": 2, "rms_norm_eps": 1e-6, "max_position_embeddings": 16,
              "rope_parameters": {"rope_theta": 10000.0},
              "dflash_config": {"block_size": 4, "conv_group_size": 2, "conv_kernel_size": 2,
                "mask_token_id": 7, "selector_rank": 2, "selector_top_k": 2,
                "target_layer_ids": [0, 1]},
              "sliding_window": 32, "use_sliding_window": true,
              "layer_types": ["sliding_attention", "full_attention"]
            }"#,
        )
        .map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        assert_eq!(cfg.layer_window(0), Some(32));
        assert_eq!(cfg.layer_window(1), None);
        assert_eq!(cfg.layer_window(9), None);
        cfg.use_sliding_window = false;
        assert_eq!(cfg.layer_window(0), None);
        assert_eq!(cfg.draft_len(), 3);
        assert_eq!(cfg.num_fused_layers(), 2);
        Ok(())
    }

    /// Dropping context rows older than the window must be lossless, and no more than
    /// lossless: every dropped column is `-inf` for every slot, the kept columns read exactly
    /// as they do in the unsliced mask, and the oldest kept region still holds a visible key —
    /// so a bound off by one in either direction fails here.
    #[test]
    fn dflash_window_slice_drops_only_invisible_context() -> Result<()> {
        let dev = Device::Cpu;
        let (bs, w) = (8usize, 16usize);
        for anchor_pos in [w - 1, w, w + 1, 100, 1000] {
            let ctx_len = anchor_pos;
            let ctx_start = ctx_len.saturating_sub(w);
            let full = block_mask(0, ctx_len, bs, anchor_pos, Some(w), &dev, DType::F32)?
                .to_vec2::<f32>()?;
            let sliced = block_mask(
                ctx_start,
                ctx_len - ctx_start,
                bs,
                anchor_pos,
                Some(w),
                &dev,
                DType::F32,
            )?
            .to_vec2::<f32>()?;
            for (slot, (full_row, sliced_row)) in full.iter().zip(&sliced).enumerate() {
                assert!(
                    full_row[..ctx_start].iter().all(|v| *v == f32::NEG_INFINITY),
                    "anchor {anchor_pos} slot {slot}: a dropped row was visible"
                );
                assert_eq!(
                    &full_row[ctx_start..],
                    &sliced_row[..],
                    "anchor {anchor_pos} slot {slot}: slicing changed what is visible"
                );
            }
            if ctx_start > 0 {
                // Slot 0 reaches furthest back; its oldest visible key must sit in the kept
                // region's first two columns, or the slice keeps more than the window needs.
                assert!(
                    sliced[0][..2].iter().any(|v| *v == 0.0),
                    "anchor {anchor_pos}: the slice keeps rows no slot can see"
                );
            }
        }
        Ok(())
    }

    /// A capture that keeps only `reach + block` positions must hand the draft the very rows the
    /// whole prefix would, at every anchor a verify can land on. A capture that starts later
    /// than the reach shortens the context and nothing else.
    #[test]
    fn dflash_context_range_is_the_same_from_a_bounded_window() {
        let (reach, block) = (16usize, 8usize);
        let retain = reach + block;
        for verified_to in [retain, retain + 1, 100, 1000] {
            // After a verify the capture holds the newest `retain` rows ending at `verified_to`;
            // the next anchor sits anywhere in the block that verify covered.
            let bounded = verified_to - retain..verified_to;
            for anchor_pos in verified_to + 1 - block..=verified_to {
                assert_eq!(
                    context_range(Some(reach), bounded.clone(), anchor_pos),
                    context_range(Some(reach), 0..verified_to, anchor_pos),
                    "verified_to {verified_to} anchor {anchor_pos}"
                );
            }
        }
        assert_eq!(context_range(Some(reach), 0..10, 10), 0..10);
        assert_eq!(context_range(Some(reach), 95..100, 100), 95..100);
        assert_eq!(context_range(None, 0..100, 60), 0..60);
        assert_eq!(context_range(None, 40..100, 60), 40..60);
    }

    /// End-to-end over the real checkpoint IF present (skips gracefully otherwise):
    /// fused target hiddens + an anchor -> `block_size - 1` in-vocab tokens whose
    /// proposal rows are `[block_size - 1, vocab]`. No GPU, no model download — the
    /// target's embedding and head are stubbed with deterministic tensors.
    #[test]
    fn dflash_load_and_draft_block() -> Result<()> {
        let dir_s = ckpt_dir();
        let dir = std::path::Path::new(&dir_s);
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            eprintln!("skipping: checkpoint not found at {}", weights.display());
            return Ok(());
        }

        let dev = Device::Cpu;
        let cfg = DFlash2Config::from_json_file(dir.join("config.json"))?;
        let block = cfg.block_size();
        let draft_len = cfg.draft_len();
        let n_fused = cfg.num_fused_layers();
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let vb = dflash_varbuilder(&weights, DType::F32, &dev)?;
        let model = Qwen3DFlash2::load(cfg, vb)?;

        // Stand in for the target's embedding table and head.
        let table = Tensor::randn(0f32, 0.02f32, (vocab, h), &dev)?;
        let head_w = Tensor::randn(0f32, 0.02f32, (vocab, h), &dev)?;
        let embed = |ids: &Tensor| -> Result<Tensor> {
            let flat = ids.flatten_all()?;
            let rows = table.index_select(&flat, 0)?;
            rows.reshape((ids.dim(0)?, ids.dim(1)?, h))
        };
        let lm_head = |hidden: &Tensor| -> Result<Tensor> { hidden.matmul(&head_w.t()?.contiguous()?) };

        let ctx_len = 12usize;
        let mut hiddens = Vec::with_capacity(n_fused);
        for _ in 0..n_fused {
            hiddens.push(Tensor::randn(0f32, 1f32, (ctx_len, h), &dev)?);
        }

        let hiddens = crate::speculative::HiddenWindow::new(0, hiddens)?;
        let (tokens, logits) =
            model.draft_block(&hiddens, 12345, ctx_len, &embed, &lm_head, 0.0)?;
        assert_eq!(tokens.dims(), &[draft_len]);
        assert_eq!(logits.dims(), &[draft_len, vocab]);
        assert_eq!(draft_len, block - 1);

        let toks = tokens.to_vec1::<u32>()?;
        assert!(
            toks.iter().all(|&t| (t as usize) < vocab),
            "draft token out of vocab range"
        );
        // Every row must carry at least one finite score — an all-`-inf` row would
        // mean the walk proposed from an empty candidate set.
        for row in logits.to_vec2::<f32>()? {
            assert!(row.iter().any(|v| v.is_finite()));
        }
        eprintln!("dflash draft tokens = {toks:?}");
        Ok(())
    }
}
