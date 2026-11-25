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

use hanzo_ml::quantized::gguf_file;
use hanzo_ml::{DType, Device, Module, Result, Shape, Tensor, D};
use hanzo_nn::Linear;
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::layers::{linear_no_bias, RmsNorm, RotaryEmbedding, Yarn};
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
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
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

    /// Parse DFlash 2 configuration from a GGUF file's metadata and tensor headers.
    ///
    /// Note that DFlash GGUFs store `dflash.target_layers` as 1-based layer indices
    /// (e.g. `[6, 20, 34, 48, 62]` for 27B, `[2, 6, ...]` for 4B/9B), whereas
    /// HuggingFace `config.json` and [`HiddenPrefixCapture`] use 0-based indices;
    /// this converts them to 0-based on load.
    pub fn from_gguf_content(content: &gguf_file::Content) -> Result<Self> {
        let arch = gguf_meta_string(content, "general.architecture")?;
        if arch != "dflash" {
            hanzo_ml::bail!("expected GGUF general.architecture = \"dflash\", got \"{arch}\"");
        }

        let selector_rank = gguf_meta_usize(content, "dflash.selector_rank")?;
        let pred_info = content
            .tensor_infos
            .get("selector_predecessor.weight")
            .ok_or_else(|| {
                hanzo_ml::Error::msg("GGUF missing tensor `selector_predecessor.weight`")
            })?;
        let pred_dims = pred_info.shape.dims();
        if pred_dims.len() != 2 {
            hanzo_ml::bail!(
                "GGUF `selector_predecessor.weight` expected 2D shape, got {:?}",
                pred_dims
            );
        }
        let vocab_size = if pred_dims[1] == selector_rank {
            pred_dims[0]
        } else if pred_dims[0] == selector_rank {
            pred_dims[1]
        } else {
            hanzo_ml::bail!(
                "GGUF `selector_predecessor.weight` shape {:?} incompatible with selector_rank {selector_rank}",
                pred_dims
            );
        };

        let target_layers_1based = gguf_meta_usize_array(content, "dflash.target_layers")?;
        let mut target_layer_ids = Vec::with_capacity(target_layers_1based.len());
        for raw in target_layers_1based {
            if raw == 0 {
                hanzo_ml::bail!(
                    "GGUF `dflash.target_layers` contains 0; expected 1-based layer IDs"
                );
            }
            target_layer_ids.push((raw - 1) as i64);
        }

        let mask_token_id = gguf_meta_usize_opt(content, "dflash.mask_token_id")?
            .or(gguf_meta_usize_opt(
                content,
                "tokenizer.ggml.mask_token_id",
            )?)
            .ok_or_else(|| {
                hanzo_ml::Error::msg(
                    "GGUF missing `dflash.mask_token_id` and `tokenizer.ggml.mask_token_id`",
                )
            })? as u32;

        let sliding_window = gguf_meta_usize_opt(content, "dflash.attention.sliding_window")?;
        let pattern = gguf_meta_bool_array_opt(content, "dflash.attention.sliding_window_pattern")?;
        let num_hidden_layers = gguf_meta_usize(content, "dflash.block_count")?;
        let (use_sliding_window, layer_types) = if let Some(pat) = pattern {
            let any_sliding = pat.iter().any(|&b| b) && sliding_window.is_some();
            let types = pat
                .into_iter()
                .map(|is_sliding| {
                    if is_sliding {
                        "sliding_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect();
            (any_sliding, types)
        } else if sliding_window.is_some() {
            (
                true,
                vec!["sliding_attention".to_string(); num_hidden_layers],
            )
        } else {
            (false, Vec::new())
        };

        let rope_type = gguf_meta_string_opt(content, "dflash.rope.scaling.type")?;
        let factor = gguf_meta_f64_opt(content, "dflash.rope.scaling.factor")?;
        let original_max_position_embeddings =
            gguf_meta_usize_opt(content, "dflash.rope.scaling.original_context_length")?;
        let beta_fast = gguf_meta_f64_opt(content, "dflash.rope.scaling.beta_fast")?;
        let beta_slow = gguf_meta_f64_opt(content, "dflash.rope.scaling.beta_slow")?;

        Ok(Self {
            vocab_size,
            hidden_size: gguf_meta_usize(content, "dflash.embedding_length")?,
            intermediate_size: gguf_meta_usize(content, "dflash.feed_forward_length")?,
            num_hidden_layers,
            num_attention_heads: gguf_meta_usize(content, "dflash.attention.head_count")?,
            num_key_value_heads: gguf_meta_usize(content, "dflash.attention.head_count_kv")?,
            head_dim: gguf_meta_usize(content, "dflash.attention.key_length")?,
            rms_norm_eps: gguf_meta_f64_opt(content, "dflash.attention.layer_norm_rms_epsilon")?
                .unwrap_or(1e-6),
            max_position_embeddings: gguf_meta_usize(content, "dflash.context_length")?,
            rope_parameters: RopeParameters {
                rope_theta: gguf_meta_f64_opt(content, "dflash.rope.freq_base")?.unwrap_or(1e7),
                rope_type,
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
            },
            dflash_config: DFlashParameters {
                block_size: gguf_meta_usize(content, "dflash.block_size")?,
                conv_group_size: gguf_meta_usize(content, "dflash.conv_group_size")?,
                conv_kernel_size: gguf_meta_usize(content, "dflash.conv_kernel_size")?,
                mask_token_id,
                selector_rank,
                selector_top_k: gguf_meta_usize(content, "dflash.selector_top_k")?,
                target_layer_ids,
            },
            sliding_window,
            use_sliding_window,
            layer_types,
        })
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn yarn(&self) -> Result<Option<Yarn>> {
        if self.rope_parameters.rope_type.as_deref() != Some("yarn") {
            return Ok(None);
        }
        let factor = self.rope_parameters.factor.unwrap_or(1.0);
        if factor <= 1.0 {
            return Ok(None);
        }
        let original = self
            .rope_parameters
            .original_max_position_embeddings
            .unwrap_or(self.max_position_embeddings);
        Ok(Some(Yarn {
            factor,
            length: original.max(1),
            beta_fast: self.rope_parameters.beta_fast.unwrap_or(32.0),
            beta_slow: self.rope_parameters.beta_slow.unwrap_or(1.0),
        }))
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
            kernel_projection: linear_no_bias(
                h,
                2 * kernel_size * groups,
                vb.pp("kernel_projection"),
            )
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

    fn prepare(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let slots = hidden.dim(0)?;
        let dynamic = self.kernel_projection.forward(hidden)?.reshape((
            slots,
            2,
            self.kernel_size,
            self.groups,
        ))?;
        Ok((self.convolve(hidden, &dynamic, 0, None)?, dynamic))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn finish(&self, hidden: &Tensor, dynamic: &Tensor) -> Result<Tensor> {
        self.convolve(hidden, dynamic, 1, None)
    }

    fn finish_residual(
        &self,
        hidden: &Tensor,
        dynamic: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        self.convolve(hidden, dynamic, 1, Some(residual))
    }

    fn convolve(
        &self,
        hidden: &Tensor,
        dynamic: &Tensor,
        half: usize,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if hidden.device().is_cuda()
            && matches!(hidden.dtype(), DType::F16 | DType::BF16 | DType::F32)
        {
            return crate::cuda::dflash2::dflash2_block_conv_cuda(
                crate::cuda::dflash2::DFlash2BlockConvParams {
                    x: hidden,
                    dynamic,
                    base_kernel: &self.base_kernel,
                    residual,
                    kernel_size: self.kernel_size,
                    group_size: self.group_size,
                    half_idx: half,
                },
            );
        }

        let (slots, hidden_size) = hidden.dims2()?;
        let x = hidden.reshape((slots, self.groups, self.group_size))?;
        let base = self.base_kernel.narrow(0, half, 1)?.reshape((
            self.kernel_size,
            self.groups,
            self.group_size,
        ))?;
        let dyn_half = dynamic.narrow(1, half, 1)?.squeeze(1)?;
        let mut out: Option<Tensor> = None;
        for offset in 0..self.kernel_size {
            if offset >= slots {
                break;
            }
            let shifted = if offset == 0 {
                x.clone()
            } else {
                x.narrow(0, 0, slots - offset)?
                    .pad_with_zeros(0, offset, 0)?
            };
            let taps = base.narrow(0, offset, 1)?.broadcast_add(
                &dyn_half
                    .narrow(1, offset, 1)?
                    .reshape((slots, self.groups, 1))?,
            )?;
            let term = taps.mul(&shifted)?;
            out = Some(match out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let conv_out = match out {
            Some(acc) => acc.reshape((slots, hidden_size))?,
            None => Tensor::zeros((slots, hidden_size), hidden.dtype(), hidden.device())?,
        };
        match residual {
            Some(res) => conv_out.add(res),
            None => Ok(conv_out),
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
            return Err(hanzo_ml::Error::msg(
                "dflash_config.selector_rank must be >= 1",
            ));
        }
        if top_k == 0 || top_k > vocab {
            return Err(hanzo_ml::Error::msg(format!(
                "dflash_config.selector_top_k {top_k} must be in 1..={vocab}"
            )));
        }
        Ok(Self {
            hidden_projection: linear_no_bias(cfg.hidden_size, rank, vb.pp("hidden_projection"))
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
        let (tokens, maybe_rows) =
            self.walk_inner(hidden, logits, anchor_token, temperature, true)?;
        let rows = maybe_rows.ok_or_else(|| {
            hanzo_ml::Error::msg("selector.walk internal error: missing proposal logits")
        })?;
        Ok((tokens, rows))
    }

    fn walk_inner(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchor_token: u32,
        temperature: f64,
        need_logits: bool,
    ) -> Result<(Vec<u32>, Option<Tensor>)> {
        let device = hidden.device().clone();
        let (slots, vocab) = logits.dims2()?;
        let top_k = self.top_k.min(vocab);
        if slots == 0 {
            let empty = if need_logits {
                Some(Tensor::zeros((0, vocab), DType::F32, &device)?)
            } else {
                None
            };
            return Ok((Vec::new(), empty));
        }

        let mut cuda_flat_ids: Option<Tensor> = None;
        let (candidates, unary): (Vec<Vec<u32>>, Vec<Vec<f32>>) = if logits.device().is_cuda()
            && matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16)
            && top_k > 0
        {
            let (values, indices) = if top_k <= crate::cuda::dflash2::DFLASH2_MAX_TOPK {
                let (vals, ids) = crate::cuda::dflash2::dflash2_topk_cuda(logits, top_k)?;
                if temperature < 1e-5
                    && !need_logits
                    && self.predecessor_codebook.dim(1)?
                        <= crate::cuda::dflash2::DFLASH2_MAX_WALK_RANK
                    && matches!(hidden.dtype(), DType::F32 | DType::BF16 | DType::F16)
                {
                    let gate_hidden = self.hidden_projection.forward(hidden)?;
                    let out_tokens = crate::cuda::dflash2::dflash2_greedy_walk_cuda(
                        crate::cuda::dflash2::DFlash2GreedyWalkParams {
                            topk_values: &vals,
                            topk_indices: &ids,
                            gate_hidden: &gate_hidden,
                            pred_codebook: &self.predecessor_codebook,
                            succ_codebook: &self.successor_codebook,
                            anchor_token,
                        },
                    )?;
                    let toks = out_tokens.to_vec1::<u32>()?;
                    if let Some(bad_slot) = toks.iter().position(|&t| t == u32::MAX) {
                        hanzo_ml::bail!("slot {bad_slot} has no finite logit");
                    }
                    return Ok((toks, None));
                }
                (vals, ids)
            } else {
                let crate::ops::TopKOutput { values, indices } =
                    crate::ops::TopKLastDimOp::topk(&logits.contiguous()?, top_k)?;
                (values.to_dtype(DType::F32)?, indices)
            };
            let vals_h = values.to_vec2::<f32>()?;
            let ids_h = indices.to_vec2::<u32>()?;
            let mut cands = Vec::with_capacity(slots);
            let mut uns = Vec::with_capacity(slots);
            let mut all_finite = true;
            for s in 0..slots {
                let mut sc = Vec::with_capacity(top_k);
                let mut su = Vec::with_capacity(top_k);
                for (&id, &v) in ids_h[s].iter().zip(vals_h[s].iter()) {
                    if id != u32::MAX && v.is_finite() {
                        sc.push(id);
                        su.push(v);
                    } else {
                        all_finite = false;
                    }
                }
                cands.push(sc);
                uns.push(su);
            }
            if all_finite {
                cuda_flat_ids = Some(indices.flatten_all()?);
            }
            (cands, uns)
        } else {
            let rows_h = logits
                .contiguous()?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            rows_h.iter().map(|row| top_k_of(row, top_k)).unzip()
        };

        let gate_hidden = self.hidden_projection.forward(hidden)?; // [slots, rank]
        let rank = gate_hidden.dim(1)?;
        let all_full = top_k > 0 && candidates.iter().all(|c| c.len() == top_k);
        let weighted_succ = if all_full {
            let flat_ids = match cuda_flat_ids {
                Some(t) => t,
                None => {
                    let flat: Vec<u32> =
                        candidates.iter().flat_map(|c| c.iter().copied()).collect();
                    Tensor::from_vec(flat, (slots * top_k,), &device)?
                }
            };
            let succ = self.successor_codebook.index_select(&flat_ids, 0)?; // [slots * top_k, rank]
            Some(
                succ.reshape((slots, top_k, rank))?
                    .broadcast_mul(&gate_hidden.unsqueeze(1)?)?
                    .transpose(1, 2)?
                    .contiguous()?, // [slots, rank, top_k]
            )
        } else {
            None
        };

        let mut tokens = Vec::with_capacity(slots);
        let mut rows = if need_logits {
            vec![f32::NEG_INFINITY; slots * vocab]
        } else {
            Vec::new()
        };

        let total_ws = slots.saturating_mul(rank).saturating_mul(top_k);
        if let Some(ws) = weighted_succ.as_ref().filter(|_| total_ws <= 2_000_000) {
            let mut pred_ids = Vec::with_capacity(1 + (slots - 1) * top_k);
            pred_ids.push(anchor_token);
            for slot in 0..slots - 1 {
                pred_ids.extend_from_slice(&candidates[slot]);
            }
            let pred_t = Tensor::from_vec(pred_ids, (1 + (slots - 1) * top_k,), &device)?;
            let pred_vecs = self
                .predecessor_codebook
                .index_select(&pred_t, 0)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let ws_flat = ws.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

            let mut prev_cand_idx = 0usize;
            for slot in 0..slots {
                let ids_h = &candidates[slot];
                let prev_vec = if slot == 0 {
                    &pred_vecs[0]
                } else {
                    &pred_vecs[1 + (slot - 1) * top_k + prev_cand_idx]
                };
                let ws_slot = &ws_flat[slot * rank * top_k..(slot + 1) * rank * top_k];
                let mut scores = unary[slot].clone();
                for (r, &pv) in prev_vec.iter().enumerate() {
                    let row_w = &ws_slot[r * top_k..(r + 1) * top_k];
                    for (s, &w) in scores.iter_mut().zip(row_w.iter()) {
                        *s += pv * w;
                    }
                }
                if need_logits {
                    for (i, &id) in ids_h.iter().enumerate() {
                        rows[slot * vocab + id as usize] = scores[i];
                    }
                }
                prev_cand_idx = select_candidate(&scores, temperature);
                tokens.push(ids_h[prev_cand_idx]);
            }
        } else {
            let mut previous = anchor_token;
            for slot in 0..slots {
                let ids_h = &candidates[slot];
                if ids_h.is_empty() {
                    hanzo_ml::bail!("slot {slot} has no finite logit");
                }
                let prev_row = self
                    .predecessor_codebook
                    .narrow(0, previous as usize, 1)?
                    .contiguous()?; // [1, rank]
                let pairwise: Vec<f32> = if let Some(ref ws) = weighted_succ {
                    let slot_ws = ws.narrow(0, slot, 1)?.squeeze(0)?; // [rank, top_k]
                    prev_row
                        .matmul(&slot_ws)?
                        .to_dtype(DType::F32)?
                        .squeeze(0)?
                        .to_vec1::<f32>()?
                } else {
                    let gate = prev_row.mul(&gate_hidden.narrow(0, slot, 1)?)?;
                    let ids = Tensor::from_vec(ids_h.clone(), (ids_h.len(),), &device)?;
                    let successors = self.successor_codebook.index_select(&ids, 0)?;
                    gate.matmul(&successors.t()?.contiguous()?)?
                        .to_dtype(DType::F32)?
                        .squeeze(0)?
                        .to_vec1::<f32>()?
                };
                let scores: Vec<f32> = unary[slot]
                    .iter()
                    .zip(pairwise.iter())
                    .map(|(u, p)| u + p)
                    .collect();
                if need_logits {
                    for (i, &id) in ids_h.iter().enumerate() {
                        rows[slot * vocab + id as usize] = scores[i];
                    }
                }
                previous = ids_h[select_candidate(&scores, temperature)];
                tokens.push(previous);
            }
        }
        let proposal_logits = if need_logits {
            Some(Tensor::from_vec(rows, (slots, vocab), &device)?)
        } else {
            None
        };
        Ok((tokens, proposal_logits))
    }
}

/// One DFlash 2 decoder layer: a Qwen3 layer whose attention keys/values come from
/// `[fused_context ‖ block]`, with a two-tap conv wrapped around each sublayer.
struct DFlash2Layer {
    qkv_proj: Linear,
    kv_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_up_proj: Linear,
    down_proj: Linear,
    attention_conv: BlockConv,
    mlp_conv: BlockConv,
}

impl DFlash2Layer {
    fn from_separate_projs(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        gate_proj: Linear,
        up_proj: Linear,
    ) -> Result<(Linear, Linear, Linear)> {
        let q_dim = q_proj.weight().dim(0)?;
        let kv_dim = k_proj.weight().dim(0)?;
        let qkv_w =
            Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?.contiguous()?;
        let kv_w = qkv_w.narrow(0, q_dim, 2 * kv_dim)?;
        let gate_up_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?.contiguous()?;
        Ok((
            Linear::new(qkv_w, None),
            Linear::new(kv_w, None),
            Linear::new(gate_up_w, None),
        ))
    }

    fn load(cfg: &DFlash2Config, vb: ShardedVarBuilder, index: usize) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        let eps = cfg.rms_norm_eps;

        let va = vb.pp("self_attn");
        let q_proj = linear_no_bias(h, q_dim, va.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, kv_dim, va.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, kv_dim, va.pp("v_proj"))?;
        let gate_proj = linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp").pp("gate_proj"))?;
        let up_proj = linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp").pp("up_proj"))?;
        let (qkv_proj, kv_proj, gate_up_proj) =
            Self::from_separate_projs(q_proj, k_proj, v_proj, gate_proj, up_proj)?;

        Ok(Self {
            qkv_proj,
            kv_proj,
            o_proj: linear_no_bias(q_dim, h, va.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, eps, va.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, eps, va.pp("k_norm"))?,
            input_layernorm: RmsNorm::new(h, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(h, eps, vb.pp("post_attention_layernorm"))?,
            gate_up_proj,
            down_proj: linear_no_bias(cfg.intermediate_size, h, vb.pp("mlp").pp("down_proj"))?,
            attention_conv: BlockConv::load(
                cfg,
                vb.pp("attention_conv"),
                &format!("layers.{index}.attention_conv"),
            )?,
            mlp_conv: BlockConv::load(cfg, vb.pp("mlp_conv"), &format!("layers.{index}.mlp_conv"))?,
        })
    }
}

const DFLASH_CTX_TAIL_FLUSH: usize = 512;

struct DFlashContextSegment {
    start_pos: usize,
    end_pos: usize,
    layer_kv: Vec<(Tensor, Tensor)>,
}

struct DFlashContextCache {
    trunk: Option<DFlashContextSegment>,
    tail: Option<DFlashContextSegment>,
    last_row_probe: Vec<f32>,
}

impl DFlashContextCache {
    fn start_pos(&self) -> usize {
        if let Some(ref trunk) = self.trunk {
            trunk.start_pos
        } else if let Some(ref tail) = self.tail {
            tail.start_pos
        } else {
            0
        }
    }

    fn end_pos(&self) -> usize {
        if let Some(ref tail) = self.tail {
            tail.end_pos
        } else if let Some(ref trunk) = self.trunk {
            trunk.end_pos
        } else {
            0
        }
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
    ctx_cache: std::sync::Mutex<Option<DFlashContextCache>>,
    mask_cache: std::sync::Mutex<HashMap<(usize, usize, usize, Option<usize>), Tensor>>,
    mask_embed_cache: std::sync::Mutex<Option<Tensor>>,
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

        let rotary = RotaryEmbedding::new_yarn(
            cfg.rope_theta() as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.yarn()?,
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
            ctx_cache: std::sync::Mutex::new(None),
            mask_cache: std::sync::Mutex::new(HashMap::new()),
            mask_embed_cache: std::sync::Mutex::new(None),
        })
    }

    /// Load a DFlash 2 draft model from a GGUF stream (`general.architecture = "dflash"`).
    ///
    /// If `block_size != 0`, overrides the block size (validated in `2..=trained_block_size`).
    pub fn load_gguf<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        block_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let mut cfg = DFlash2Config::from_gguf_content(&content)?;
        if block_size != 0 {
            if !(2..=cfg.block_size()).contains(&block_size) {
                hanzo_ml::bail!(
                    "dflash block_size {block_size} is outside 2..={}, the block this checkpoint was trained on",
                    cfg.block_size()
                );
            }
            cfg.dflash_config.block_size = block_size;
        }
        if cfg.block_size() < 2 {
            hanzo_ml::bail!(
                "dflash_config.block_size {} leaves no slot to predict — slot 0 is the anchor",
                cfg.block_size()
            );
        }
        if cfg.num_fused_layers() == 0 {
            hanzo_ml::bail!(
                "dflash_config.target_layer_ids is empty — there is no context to fuse"
            );
        }

        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        let kernel_size = cfg.dflash_config.conv_kernel_size;
        let group_size = cfg.dflash_config.conv_group_size;
        if kernel_size == 0 {
            hanzo_ml::bail!("dflash_config.conv_kernel_size must be >= 1");
        }
        if group_size == 0 || !h.is_multiple_of(group_size) {
            hanzo_ml::bail!(
                "dflash_config.conv_group_size {group_size} must divide hidden_size {h}"
            );
        }
        let groups = h / group_size;

        let rank = cfg.dflash_config.selector_rank;
        let top_k = cfg.dflash_config.selector_top_k;
        if rank == 0 {
            hanzo_ml::bail!("dflash_config.selector_rank must be >= 1");
        }
        if top_k == 0 || top_k > cfg.vocab_size {
            hanzo_ml::bail!(
                "dflash_config.selector_top_k {top_k} must be in 1..={}",
                cfg.vocab_size
            );
        }

        let fc = gguf_linear(
            &content,
            reader,
            "fc.weight",
            &[h, cfg.num_fused_layers() * h],
            device,
            dtype,
        )?;
        let hidden_norm = gguf_rms_norm(
            &content,
            reader,
            "enc.output_norm.weight",
            h,
            cfg.rms_norm_eps,
            device,
            dtype,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("blk.{i}");
            let eps = cfg.rms_norm_eps;
            let q_proj = gguf_linear(
                &content,
                reader,
                &format!("{p}.attn_q.weight"),
                &[q_dim, h],
                device,
                dtype,
            )?;
            let k_proj = gguf_linear(
                &content,
                reader,
                &format!("{p}.attn_k.weight"),
                &[kv_dim, h],
                device,
                dtype,
            )?;
            let v_proj = gguf_linear(
                &content,
                reader,
                &format!("{p}.attn_v.weight"),
                &[kv_dim, h],
                device,
                dtype,
            )?;
            let gate_proj = gguf_linear(
                &content,
                reader,
                &format!("{p}.ffn_gate.weight"),
                &[cfg.intermediate_size, h],
                device,
                dtype,
            )?;
            let up_proj = gguf_linear(
                &content,
                reader,
                &format!("{p}.ffn_up.weight"),
                &[cfg.intermediate_size, h],
                device,
                dtype,
            )?;
            let (qkv_proj, kv_proj, gate_up_proj) =
                DFlash2Layer::from_separate_projs(q_proj, k_proj, v_proj, gate_proj, up_proj)?;
            layers.push(DFlash2Layer {
                qkv_proj,
                kv_proj,
                o_proj: gguf_linear(
                    &content,
                    reader,
                    &format!("{p}.attn_output.weight"),
                    &[h, q_dim],
                    device,
                    dtype,
                )?,
                q_norm: gguf_rms_norm(
                    &content,
                    reader,
                    &format!("{p}.attn_q_norm.weight"),
                    hd,
                    eps,
                    device,
                    dtype,
                )?,
                k_norm: gguf_rms_norm(
                    &content,
                    reader,
                    &format!("{p}.attn_k_norm.weight"),
                    hd,
                    eps,
                    device,
                    dtype,
                )?,
                input_layernorm: gguf_rms_norm(
                    &content,
                    reader,
                    &format!("{p}.attn_norm.weight"),
                    h,
                    eps,
                    device,
                    dtype,
                )?,
                post_attention_layernorm: gguf_rms_norm(
                    &content,
                    reader,
                    &format!("{p}.ffn_norm.weight"),
                    h,
                    eps,
                    device,
                    dtype,
                )?,
                gate_up_proj,
                down_proj: gguf_linear(
                    &content,
                    reader,
                    &format!("{p}.ffn_down.weight"),
                    &[h, cfg.intermediate_size],
                    device,
                    dtype,
                )?,
                attention_conv: BlockConv {
                    base_kernel: gguf_tensor(
                        &content,
                        reader,
                        &format!("{p}.attn_conv_base"),
                        &[2, kernel_size, h],
                        device,
                        dtype,
                    )?,
                    kernel_projection: gguf_linear(
                        &content,
                        reader,
                        &format!("{p}.attn_conv_proj.weight"),
                        &[2 * kernel_size * groups, h],
                        device,
                        dtype,
                    )?,
                    kernel_size,
                    group_size,
                    groups,
                },
                mlp_conv: BlockConv {
                    base_kernel: gguf_tensor(
                        &content,
                        reader,
                        &format!("{p}.ffn_conv_base"),
                        &[2, kernel_size, h],
                        device,
                        dtype,
                    )?,
                    kernel_projection: gguf_linear(
                        &content,
                        reader,
                        &format!("{p}.ffn_conv_proj.weight"),
                        &[2 * kernel_size * groups, h],
                        device,
                        dtype,
                    )?,
                    kernel_size,
                    group_size,
                    groups,
                },
            });
        }

        let norm = gguf_rms_norm(
            &content,
            reader,
            "output_norm.weight",
            h,
            cfg.rms_norm_eps,
            device,
            dtype,
        )?;
        let selector = CandidateSelector {
            hidden_projection: gguf_linear(
                &content,
                reader,
                "selector_hidden.weight",
                &[rank, h],
                device,
                dtype,
            )?,
            predecessor_codebook: gguf_tensor(
                &content,
                reader,
                "selector_predecessor.weight",
                &[cfg.vocab_size, rank],
                device,
                dtype,
            )?,
            successor_codebook: gguf_tensor(
                &content,
                reader,
                "selector_successor.weight",
                &[cfg.vocab_size, rank],
                device,
                dtype,
            )?,
            top_k,
        };

        let rotary = RotaryEmbedding::new_yarn(
            cfg.rope_theta() as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.yarn()?,
            device,
            true,
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
            device: device.clone(),
            dtype,
            ctx_cache: std::sync::Mutex::new(None),
            mask_cache: std::sync::Mutex::new(HashMap::new()),
            mask_embed_cache: std::sync::Mutex::new(None),
        })
    }

    /// How far before the anchor any layer attends: the widest window, or `None` when a
    /// full-attention layer sees the whole context.
    pub fn reach(&self) -> Option<usize> {
        (0..self.layers.len())
            .map(|layer| self.cfg.layer_window(layer))
            .try_fold(0usize, |widest, window| window.map(|w| widest.max(w)))
    }

    fn probe_row(
        &self,
        target_hiddens: &crate::speculative::HiddenWindow,
        abs_pos: usize,
    ) -> Result<Vec<f32>> {
        let Some(first_layer) = target_hiddens.layers.first() else {
            return Ok(Vec::new());
        };
        if abs_pos < target_hiddens.start {
            return Ok(Vec::new());
        }
        let rel = abs_pos - target_hiddens.start;
        let (rows, cols) = first_layer.dims2()?;
        if rel >= rows {
            return Ok(Vec::new());
        }
        first_layer
            .narrow(0, rel, 1)?
            .narrow(1, 0, cols.min(8))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn project_context_slice(
        &self,
        target_hiddens: &crate::speculative::HiddenWindow,
        slice_start: usize,
        slice_end: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        const CHUNK: usize = 2048;
        let dev = &self.device;
        let hd = self.cfg.head_dim;
        let nkv = self.cfg.num_key_value_heads;
        let eps = self.cfg.rms_norm_eps as f32;
        let n_layers = self.layers.len();
        let mut layer_ks: Vec<Vec<Tensor>> = vec![Vec::new(); n_layers];
        let mut layer_vs: Vec<Vec<Tensor>> = vec![Vec::new(); n_layers];

        let mut c_start = slice_start;
        while c_start < slice_end {
            let c_end = (c_start + CHUNK).min(slice_end);
            let c_len = c_end - c_start;
            let ctx_parts = target_hiddens
                .rows(c_start, c_end)?
                .iter()
                .map(|t| t.to_device(dev)?.to_dtype(self.dtype))
                .collect::<Result<Vec<_>>>()?;
            let ctx_refs: Vec<&Tensor> = ctx_parts.iter().collect();
            let ctx_cat = Tensor::cat(&ctx_refs, D::Minus1)?; // [c_len, num_fused * hidden]
            let tctx = self.hidden_norm.forward(&self.fc.forward(&ctx_cat)?)?; // [c_len, hidden]
            drop(ctx_cat);
            drop(ctx_parts);

            let pos: Vec<u32> = (c_start as u32..c_end as u32).collect();
            let pos_t = Tensor::from_vec(pos, (c_len,), dev)?;
            let (cos_ctx, sin_ctx) = self.rope_cos_sin(&pos_t)?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let kv_dim = nkv * hd;
                let kv_ctx = layer.kv_proj.forward(&tctx)?;
                let k_ctx = kv_ctx.narrow(1, 0, kv_dim)?.reshape((c_len, nkv, hd))?;
                let k_ctx =
                    hanzo_nn::ops::rms_norm(&k_ctx.contiguous()?, layer.k_norm.weight(), eps)?;
                let k_ctx = k_ctx.transpose(0, 1)?.contiguous()?;
                let k_ctx = apply_rope(&k_ctx, &cos_ctx, &sin_ctx)?;

                let v_ctx = kv_ctx
                    .narrow(1, kv_dim, kv_dim)?
                    .reshape((c_len, nkv, hd))?
                    .transpose(0, 1)?
                    .contiguous()?;
                layer_ks[layer_idx].push(k_ctx);
                layer_vs[layer_idx].push(v_ctx);
            }
            c_start = c_end;
        }

        let mut out = Vec::with_capacity(n_layers);
        for (ks, vs) in layer_ks.into_iter().zip(layer_vs.into_iter()) {
            let k_full = if ks.len() == 1 {
                ks.into_iter().next().unwrap()
            } else {
                let refs: Vec<&Tensor> = ks.iter().collect();
                Tensor::cat(&refs, 1)?
            };
            let v_full = if vs.len() == 1 {
                vs.into_iter().next().unwrap()
            } else {
                let refs: Vec<&Tensor> = vs.iter().collect();
                Tensor::cat(&refs, 1)?
            };
            out.push((k_full, v_full));
        }
        Ok(out)
    }

    fn resolve_context_kv(
        &self,
        target_hiddens: &crate::speculative::HiddenWindow,
        ctx_start: usize,
        ctx_end: usize,
    ) -> Result<Vec<DFlashContextSegment>> {
        let mut guard = self
            .ctx_cache
            .lock()
            .map_err(|_| hanzo_ml::Error::msg("dflash ctx_cache lock poisoned"))?;
        if ctx_end <= ctx_start {
            *guard = None;
            return Ok(Vec::new());
        }

        let clone_segments = |cache: &DFlashContextCache| -> Vec<DFlashContextSegment> {
            let mut segs = Vec::with_capacity(2);
            if let Some(ref trunk) = cache.trunk {
                segs.push(DFlashContextSegment {
                    start_pos: trunk.start_pos,
                    end_pos: trunk.end_pos,
                    layer_kv: trunk.layer_kv.clone(),
                });
            }
            if let Some(ref tail) = cache.tail {
                segs.push(DFlashContextSegment {
                    start_pos: tail.start_pos,
                    end_pos: tail.end_pos,
                    layer_kv: tail.layer_kv.clone(),
                });
            }
            segs
        };

        if let Some(cache) = guard.as_mut() {
            let cached_start = cache.start_pos();
            let cached_end = cache.end_pos();
            let can_extend = cached_start <= ctx_start
                && ctx_start <= cached_end
                && cached_end <= ctx_end
                && cached_end > 0
                && self
                    .probe_row(target_hiddens, cached_end - 1)
                    .map(|p| p == cache.last_row_probe)
                    .unwrap_or(false);
            if can_extend {
                if ctx_start > cached_start {
                    if let Some(ref mut trunk) = cache.trunk {
                        if ctx_start < trunk.end_pos {
                            let trim = ctx_start - trunk.start_pos;
                            let kept = trunk.end_pos - ctx_start;
                            for (k_c, v_c) in &mut trunk.layer_kv {
                                *k_c = k_c.narrow(1, trim, kept)?;
                                *v_c = v_c.narrow(1, trim, kept)?;
                            }
                            trunk.start_pos = ctx_start;
                        } else {
                            cache.trunk = None;
                        }
                    }
                    if let Some(ref mut tail) = cache.tail {
                        if ctx_start > tail.start_pos && ctx_start < tail.end_pos {
                            let trim = ctx_start - tail.start_pos;
                            let kept = tail.end_pos - ctx_start;
                            for (k_c, v_c) in &mut tail.layer_kv {
                                *k_c = k_c.narrow(1, trim, kept)?;
                                *v_c = v_c.narrow(1, trim, kept)?;
                            }
                            tail.start_pos = ctx_start;
                        } else if ctx_start >= tail.end_pos {
                            cache.tail = None;
                        }
                    }
                }
                if ctx_end > cached_end {
                    let delta_kv =
                        self.project_context_slice(target_hiddens, cached_end, ctx_end)?;
                    match cache.tail.as_mut() {
                        Some(tail) => {
                            for ((k_c, v_c), (k_d, v_d)) in tail.layer_kv.iter_mut().zip(delta_kv) {
                                *k_c = Tensor::cat(&[&*k_c, &k_d], 1)?;
                                *v_c = Tensor::cat(&[&*v_c, &v_d], 1)?;
                            }
                            tail.end_pos = ctx_end;
                        }
                        None => {
                            cache.tail = Some(DFlashContextSegment {
                                start_pos: cached_end,
                                end_pos: ctx_end,
                                layer_kv: delta_kv,
                            });
                        }
                    }
                    cache.last_row_probe = self.probe_row(target_hiddens, ctx_end - 1)?;
                    if cache
                        .tail
                        .as_ref()
                        .is_some_and(|tail| tail.end_pos - tail.start_pos >= DFLASH_CTX_TAIL_FLUSH)
                    {
                        let tail = cache.tail.take().unwrap();
                        if let Some(trunk) = cache.trunk.as_mut() {
                            for ((k_c, v_c), (k_d, v_d)) in
                                trunk.layer_kv.iter_mut().zip(tail.layer_kv)
                            {
                                *k_c = Tensor::cat(&[&*k_c, &k_d], 1)?;
                                *v_c = Tensor::cat(&[&*v_c, &v_d], 1)?;
                            }
                            trunk.end_pos = tail.end_pos;
                        } else {
                            cache.trunk = Some(tail);
                        }
                    }
                }
                return Ok(clone_segments(cache));
            }
        }

        let layer_kv = self.project_context_slice(target_hiddens, ctx_start, ctx_end)?;
        let last_row_probe = self.probe_row(target_hiddens, ctx_end - 1)?;
        *guard = Some(DFlashContextCache {
            trunk: Some(DFlashContextSegment {
                start_pos: ctx_start,
                end_pos: ctx_end,
                layer_kv,
            }),
            tail: None,
            last_row_probe,
        });
        Ok(clone_segments(guard.as_ref().unwrap()))
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
        let (tokens, maybe_logits) = self.draft_block_inner(
            target_hiddens,
            anchor_token,
            anchor_pos,
            embed,
            lm_head,
            temperature,
            true,
            None,
        )?;
        let logits = maybe_logits.ok_or_else(|| {
            hanzo_ml::Error::msg("draft_block internal error: missing proposal logits")
        })?;
        let tokens_t = Tensor::from_vec(tokens, (self.cfg.block_size() - 1,), &self.device)?;
        Ok((tokens_t, logits))
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_block_inner(
        &self,
        target_hiddens: &crate::speculative::HiddenWindow,
        anchor_token: u32,
        anchor_pos: usize,
        embed: &TargetTokenEmbedder<'_>,
        lm_head: &TargetLmHead<'_>,
        temperature: f64,
        need_logits: bool,
        block_size_override: Option<usize>,
    ) -> Result<(Vec<u32>, Option<Tensor>)> {
        let prof = crate::engine_env_bool("ENGINE_DFLASH_PROFILE", "HANZO_DFLASH_PROFILE", false)
            || (std::env::var_os("ENGINE_DFLASH_PROFILE").is_none()
                && std::env::var_os("HANZO_DFLASH_PROFILE")
                    .is_some_and(|v| v != "0" && v != "false" && !v.is_empty()));
        let t_total = Instant::now();
        let dev = &self.device;
        let max_bs = self.cfg.block_size();
        let bs = block_size_override.map_or(max_bs, |v| v.clamp(2, max_bs));

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

        // 1. Resolve cached context K/V across draft rounds (only projecting newly accepted
        //    delta tokens on rounds 2..N).
        let t0 = Instant::now();
        let held = target_hiddens.start..target_hiddens.end()?;
        let std::ops::Range {
            start: ctx_start,
            end: ctx_end,
        } = context_range(self.reach(), held, anchor_pos);
        let ctx_kv = self.resolve_context_kv(target_hiddens, ctx_start, ctx_end)?;
        let ms_ctx = t0.elapsed().as_secs_f64() * 1e3;

        let mask_row = {
            let mut guard = self
                .mask_embed_cache
                .lock()
                .map_err(|_| hanzo_ml::Error::msg("dflash mask_embed_cache lock poisoned"))?;
            if let Some(ref row) = *guard {
                row.clone()
            } else {
                let mask_id =
                    Tensor::new(&[self.cfg.dflash_config.mask_token_id], dev)?.reshape((1, 1))?;
                let row = as_2d(&embed(&mask_id)?)?
                    .to_dtype(self.dtype)?
                    .contiguous()?;
                *guard = Some(row.clone());
                row
            }
        };
        let anchor_id = Tensor::new(&[anchor_token], dev)?.reshape((1, 1))?;
        let anchor_row = as_2d(&embed(&anchor_id)?)?.to_dtype(self.dtype)?;
        let mask_rows = mask_row.broadcast_as((bs - 1, self.cfg.hidden_size))?;
        let mut hstate = Tensor::cat(&[&anchor_row, &mask_rows], 0)?;

        // 3. RoPE tables for the draft block positions.
        let draft_pos: Vec<u32> = (0..bs as u32).map(|i| anchor_pos as u32 + i).collect();
        let draft_pos_t = Tensor::from_vec(draft_pos, (bs,), dev)?;
        let (cos_draft, sin_draft) = self.rope_cos_sin(&draft_pos_t)?;

        // 4. Per-layer window slicing and mask construction (memoized by window + cached).
        let t_mask = Instant::now();
        let mut masks: Vec<Option<Tensor>> = Vec::with_capacity(self.layers.len());
        let mut layer_starts: Vec<usize> = Vec::with_capacity(self.layers.len());
        for layer_idx in 0..self.layers.len() {
            let window = self.cfg.layer_window(layer_idx);
            let layer_ctx_start = match window {
                Some(w) => anchor_pos
                    .saturating_sub(w.saturating_sub(1))
                    .max(ctx_start)
                    .min(ctx_end),
                None => ctx_start,
            };
            let layer_ctx_len = ctx_end.saturating_sub(layer_ctx_start);
            layer_starts.push(layer_ctx_start);

            let mask = if window.is_none() && ctx_end <= anchor_pos {
                None
            } else if let Some(earlier) = (0..layer_idx)
                .find(|&e| self.cfg.layer_window(e) == window && layer_starts[e] == layer_ctx_start)
            {
                masks[earlier].clone()
            } else if ctx_end <= anchor_pos {
                let key = (anchor_pos - layer_ctx_start, layer_ctx_len, bs, window);
                let mut guard = self
                    .mask_cache
                    .lock()
                    .map_err(|_| hanzo_ml::Error::msg("dflash mask_cache lock poisoned"))?;
                if let Some(cached) = guard.get(&key) {
                    Some(cached.clone())
                } else {
                    let m = block_mask(
                        layer_ctx_start,
                        layer_ctx_len,
                        bs,
                        anchor_pos,
                        window,
                        &self.device,
                        self.dtype,
                    )?;
                    if guard.len() >= 16 {
                        guard.clear();
                    }
                    guard.insert(key, m.clone());
                    Some(m)
                }
            } else {
                Some(block_mask(
                    layer_ctx_start,
                    layer_ctx_len,
                    bs,
                    anchor_pos,
                    window,
                    &self.device,
                    self.dtype,
                )?)
            };
            masks.push(mask);
        }
        let ms_mask = t_mask.elapsed().as_secs_f64() * 1e3;

        // 5. Backbone. Each sublayer is convolved on the way in and on the way out,
        //    with the taps predicted once from the sublayer's input.
        let t_layers = Instant::now();
        for (layer_idx, (layer, mask)) in self.layers.iter().zip(masks.iter()).enumerate() {
            let residual = hstate;
            let normed = layer.input_layernorm.forward(&residual)?;
            let (normed, attention_taps) = layer.attention_conv.prepare(&normed)?;
            let need_start = layer_starts[layer_idx];
            let mut layer_ctx_segs: Vec<(Tensor, Tensor)> = Vec::with_capacity(ctx_kv.len());
            for seg in &ctx_kv {
                let s0 = need_start.max(seg.start_pos);
                let s1 = ctx_end.min(seg.end_pos);
                if s1 > s0 {
                    let (ref k_c, ref v_c) = seg.layer_kv[layer_idx];
                    let off = s0 - seg.start_pos;
                    let len = s1 - s0;
                    layer_ctx_segs.push((k_c.narrow(1, off, len)?, v_c.narrow(1, off, len)?));
                }
            }
            let attn = self.attention(
                layer,
                &normed,
                &layer_ctx_segs,
                &cos_draft,
                &sin_draft,
                mask.as_ref(),
            )?;
            hstate = layer
                .attention_conv
                .finish_residual(&attn, &attention_taps, &residual)?;

            let residual = hstate;
            let normed = layer.post_attention_layernorm.forward(&residual)?;
            let (normed, mlp_taps) = layer.mlp_conv.prepare(&normed)?;
            let inter = self.cfg.intermediate_size;
            let gate_up = layer.gate_up_proj.forward(&normed)?;
            let gate = gate_up.narrow(1, 0, inter)?;
            let up = gate_up.narrow(1, inter, inter)?;
            let act = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
            let mlp = layer.down_proj.forward(&act)?;
            hstate = layer.mlp_conv.finish_residual(&mlp, &mlp_taps, &residual)?;
        }
        let hstate = self.norm.forward(&hstate)?; // [bs, hidden]
        let ms_layers = t_layers.elapsed().as_secs_f64() * 1e3;

        // 6. Slot 0 carries the anchor and is never a prediction; decode the rest
        //    through the target's head.
        let t_head = Instant::now();
        let predicted = hstate.narrow(0, 1, bs - 1)?.contiguous()?; // [bs - 1, hidden]
        let logits = lm_head(&predicted)?;
        let logits = as_2d(&logits)?; // [bs - 1, vocab]
        let vocab = logits.dim(D::Minus1)?;
        if vocab > self.cfg.vocab_size {
            return Err(hanzo_ml::Error::msg(format!(
                "target head emitted {vocab} logits but the selector codebooks cover {}",
                self.cfg.vocab_size
            )));
        }
        let ms_head = t_head.elapsed().as_secs_f64() * 1e3;

        // 7. Walk the selector for a block whose slots agree with each other.
        let t_walk = Instant::now();
        let (tokens, proposal_logits) = self.selector.walk_inner(
            &predicted,
            &logits,
            anchor_token,
            temperature,
            need_logits,
        )?;
        let ms_walk = t_walk.elapsed().as_secs_f64() * 1e3;
        if prof {
            let ms_tot = t_total.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[dflash-prof] ctx={} bs={} tot={:.2}ms (ctx_kv={:.2} mask={:.2} layers={:.2} head={:.2} walk={:.2})",
                ctx_end - ctx_start,
                bs,
                ms_tot,
                ms_ctx,
                ms_mask,
                ms_layers,
                ms_head,
                ms_walk
            );
        }

        Ok((tokens, proposal_logits))
    }

    // --- internals -------------------------------------------------------

    /// DFlash attention for one layer. `x`: block hidden `[bs, hidden]`;
    /// `ctx_kv`: cached `(k_ctx_roped, v_ctx)` segments of shape `[nkv, seg_len, hd]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &DFlash2Layer,
        x: &Tensor,
        ctx_kv: &[(Tensor, Tensor)],
        cos_draft: &Tensor,
        sin_draft: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let bs = x.dim(0)?;
        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let nrep = nh / nkv;
        let eps = self.cfg.rms_norm_eps as f32;

        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv = layer.qkv_proj.forward(x)?;

        let q = qkv.narrow(1, 0, q_dim)?.reshape((bs, nh, hd))?;
        let q = hanzo_nn::ops::rms_norm(&q.contiguous()?, layer.q_norm.weight(), eps)?;
        let q = q.transpose(0, 1)?.contiguous()?;
        let q = apply_rope(&q, cos_draft, sin_draft)?;

        let k_noise = qkv.narrow(1, q_dim, kv_dim)?.reshape((bs, nkv, hd))?;
        let k_noise = hanzo_nn::ops::rms_norm(&k_noise.contiguous()?, layer.k_norm.weight(), eps)?;
        let k_noise = k_noise.transpose(0, 1)?.contiguous()?;
        let k_noise = apply_rope(&k_noise, cos_draft, sin_draft)?;

        let v_noise = qkv
            .narrow(1, q_dim + kv_dim, kv_dim)?
            .reshape((bs, nkv, hd))?
            .transpose(0, 1)?
            .contiguous()?;

        // GQA-reshaped attention avoids concatenating full K/V context caches and avoids
        // expanding KV heads `nrep`x with `repeat_kv`.
        let scale = 1.0 / (hd as f64).sqrt();
        let q_gqa = q.reshape((nkv, nrep * bs, hd))?;
        let s_noise = q_gqa.matmul(&k_noise.transpose(1, 2)?)?; // [nkv, nrep * bs, bs]
        let mut s_parts: Vec<Tensor> = Vec::with_capacity(ctx_kv.len() + 1);
        let mut ctx_len = 0usize;
        for (k_ctx, _v_ctx) in ctx_kv {
            let seg_len = k_ctx.dim(1)?;
            if seg_len > 0 {
                s_parts.push(q_gqa.matmul(&k_ctx.transpose(1, 2)?)?); // [nkv, nrep * bs, seg_len]
                ctx_len += seg_len;
            }
        }
        let scores = if ctx_len > 0 {
            s_parts.push(s_noise);
            let s_refs: Vec<&Tensor> = s_parts.iter().collect();
            let s = Tensor::cat(&s_refs, 2)?;
            s.reshape((nh, bs, ctx_len + bs))?.affine(scale, 0.0)?
        } else {
            s_noise.reshape((nh, bs, bs))?.affine(scale, 0.0)?
        };
        let scores = match mask {
            Some(m) => scores.broadcast_add(&m.unsqueeze(0)?)?,
            None => scores,
        };
        let probs = hanzo_nn::ops::softmax_last_dim(&scores)?;
        let probs_gqa = probs.reshape((nkv, nrep * bs, ctx_len + bs))?;
        let out_gqa = if ctx_len > 0 {
            let mut acc: Option<Tensor> = None;
            let mut offset = 0usize;
            for (_k_ctx, v_ctx) in ctx_kv {
                let seg_len = v_ctx.dim(1)?;
                if seg_len == 0 {
                    continue;
                }
                let p_seg = probs_gqa.narrow(2, offset, seg_len)?.contiguous()?;
                let term = p_seg.matmul(v_ctx)?;
                acc = Some(match acc {
                    Some(prev) => prev.add(&term)?,
                    None => term,
                });
                offset += seg_len;
            }
            let p_noise = probs_gqa.narrow(2, ctx_len, bs)?.contiguous()?;
            let noise_term = p_noise.matmul(&v_noise)?;
            match acc {
                Some(prev) => prev.add(&noise_term)?,
                None => noise_term,
            }
        } else {
            probs_gqa.matmul(&v_noise)?
        };
        let out = out_gqa
            .reshape((nh, bs, hd))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((bs, nh * hd))?;
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
#[allow(dead_code)]
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

fn gguf_value_to_usize(key: &str, v: &gguf_file::Value) -> Result<usize> {
    match *v {
        gguf_file::Value::U8(x) => Ok(x as usize),
        gguf_file::Value::I8(x) if x >= 0 => Ok(x as usize),
        gguf_file::Value::U16(x) => Ok(x as usize),
        gguf_file::Value::I16(x) if x >= 0 => Ok(x as usize),
        gguf_file::Value::U32(x) => Ok(x as usize),
        gguf_file::Value::I32(x) if x >= 0 => Ok(x as usize),
        gguf_file::Value::U64(x) => Ok(x as usize),
        gguf_file::Value::I64(x) if x >= 0 => Ok(x as usize),
        _ => hanzo_ml::bail!("GGUF metadata key `{key}` expected non-negative int, got {v:?}"),
    }
}

fn gguf_meta_usize_opt(content: &gguf_file::Content, key: &str) -> Result<Option<usize>> {
    content
        .metadata
        .get(key)
        .map(|v| gguf_value_to_usize(key, v))
        .transpose()
}

fn gguf_meta_usize(content: &gguf_file::Content, key: &str) -> Result<usize> {
    gguf_meta_usize_opt(content, key)?
        .ok_or_else(|| hanzo_ml::Error::msg(format!("GGUF missing metadata key `{key}`")))
}

fn gguf_meta_f64_opt(content: &gguf_file::Content, key: &str) -> Result<Option<f64>> {
    match content.metadata.get(key) {
        None => Ok(None),
        Some(gguf_file::Value::F32(x)) => Ok(Some(f64::from(*x))),
        Some(gguf_file::Value::F64(x)) => Ok(Some(*x)),
        Some(v) => hanzo_ml::bail!("GGUF metadata key `{key}` expected float, got {v:?}"),
    }
}

fn gguf_meta_string_opt(content: &gguf_file::Content, key: &str) -> Result<Option<String>> {
    match content.metadata.get(key) {
        None => Ok(None),
        Some(gguf_file::Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => hanzo_ml::bail!("GGUF metadata key `{key}` expected string, got {v:?}"),
    }
}

fn gguf_meta_string(content: &gguf_file::Content, key: &str) -> Result<String> {
    gguf_meta_string_opt(content, key)?
        .ok_or_else(|| hanzo_ml::Error::msg(format!("GGUF missing metadata key `{key}`")))
}

fn gguf_meta_usize_array(content: &gguf_file::Content, key: &str) -> Result<Vec<usize>> {
    let Some(gguf_file::Value::Array(items)) = content.metadata.get(key) else {
        hanzo_ml::bail!("GGUF missing or non-array metadata key `{key}`");
    };
    items.iter().map(|v| gguf_value_to_usize(key, v)).collect()
}

fn gguf_meta_bool_array_opt(content: &gguf_file::Content, key: &str) -> Result<Option<Vec<bool>>> {
    let Some(v) = content.metadata.get(key) else {
        return Ok(None);
    };
    let gguf_file::Value::Array(items) = v else {
        hanzo_ml::bail!("GGUF metadata key `{key}` expected array, got {v:?}");
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match *item {
            gguf_file::Value::Bool(b) => out.push(b),
            _ => hanzo_ml::bail!("GGUF metadata array `{key}` expected bool elements"),
        }
    }
    Ok(Some(out))
}

fn gguf_tensor<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    expected_shape: &[usize],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let qt = content.tensor(reader, name, device)?;
    let t = qt.dequantize(device)?.to_dtype(dtype)?;
    if t.dims() != expected_shape {
        hanzo_ml::bail!(
            "GGUF tensor `{name}` shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            t.dims()
        );
    }
    Ok(t)
}

fn gguf_linear<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    expected_shape: &[usize],
    device: &Device,
    dtype: DType,
) -> Result<Linear> {
    Ok(Linear::new(
        gguf_tensor(content, reader, name, expected_shape, device, dtype)?,
        None,
    ))
}

fn gguf_rms_norm<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    size: usize,
    eps: f64,
    device: &Device,
    dtype: DType,
) -> Result<RmsNorm> {
    RmsNorm::from_w(
        gguf_tensor(content, reader, name, &[size], device, dtype)?,
        eps,
    )
}

fn resolve_gguf_checkpoint(path: &std::path::Path) -> Result<Option<std::path::PathBuf>> {
    if path.is_file() {
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        if !is_gguf {
            hanzo_ml::bail!(
                "dflash checkpoint `{}` is a file but not a .gguf archive",
                path.display()
            );
        }
        return Ok(Some(path.to_path_buf()));
    }
    if path.is_dir() && !path.join("config.json").exists() {
        let mut ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .map_err(|e| hanzo_ml::Error::msg(e.to_string()))?
            .filter_map(|e| e.ok().map(|ent| ent.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            })
            .collect();
        ggufs.sort();
        if ggufs.len() == 1 {
            return Ok(Some(ggufs.remove(0)));
        }
        if ggufs.len() > 1 {
            hanzo_ml::bail!(
                "dflash directory `{}` has multiple .gguf files and no config.json; pass the .gguf path directly",
                path.display()
            );
        }
    }
    Ok(None)
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
const ADAPTIVE_DEFAULT_INITIAL_BS: usize = 8;
const ADAPTIVE_WARMUP_ROUNDS: usize = 4;
const ADAPTIVE_PROBE_STEPS: usize = 3;
const ADAPTIVE_CUDA_GRAPH_DISCOUNT: f64 = 0.85;

fn dflash_adaptive_enabled() -> bool {
    crate::engine_env_bool("ENGINE_DFLASH_ADAPTIVE", "HANZO_DFLASH_ADAPTIVE", true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdaptiveDecision {
    Draft(usize),
    ProbePlain,
    Paused,
}

#[derive(Debug, Clone, Copy)]
enum LastAdaptiveStep {
    Speculative {
        anchor_pos: usize,
        draft_len: usize,
        started_at: Instant,
    },
    PlainProbe {
        anchor_pos: usize,
        started_at: Instant,
        warmup: bool,
    },
}

#[derive(Debug, Clone)]
enum AdaptivePhase {
    Warmup {
        rounds_ms: Vec<f64>,
        commits: Vec<usize>,
    },
    PlainProbe {
        remaining: usize,
        spec_ms_per_tok: f64,
        mean_commit: f64,
        probe_min_ms: f64,
    },
    Active {
        ema_commit: f64,
        total_rounds: usize,
    },
    StoodDown,
}

#[derive(Debug, Clone)]
struct SeqAdaptiveState {
    active_bs: usize,
    phase: AdaptivePhase,
    last_step: Option<LastAdaptiveStep>,
    stand_down_reason: Option<String>,
}

impl SeqAdaptiveState {
    fn new(max_bs: usize) -> Self {
        Self {
            active_bs: max_bs.min(ADAPTIVE_DEFAULT_INITIAL_BS).max(2.min(max_bs)),
            phase: AdaptivePhase::Warmup {
                rounds_ms: Vec::with_capacity(ADAPTIVE_WARMUP_ROUNDS),
                commits: Vec::with_capacity(ADAPTIVE_WARMUP_ROUNDS),
            },
            last_step: None,
            stand_down_reason: None,
        }
    }
}

#[derive(Debug)]
struct AdaptiveDFlashGate {
    enabled: bool,
    max_bs: usize,
    seqs: HashMap<usize, SeqAdaptiveState>,
}

impl AdaptiveDFlashGate {
    fn new(max_bs: usize, enabled: bool) -> Self {
        Self {
            enabled,
            max_bs,
            seqs: HashMap::new(),
        }
    }

    fn is_paused_for_seq(&self, seq_id: Option<usize>) -> bool {
        if !self.enabled {
            return false;
        }
        seq_id
            .and_then(|id| self.seqs.get(&id))
            .is_some_and(|st| matches!(st.phase, AdaptivePhase::StoodDown))
    }

    fn stand_down_reason(&self, seq_id: usize) -> Option<&str> {
        self.seqs
            .get(&seq_id)
            .and_then(|st| st.stand_down_reason.as_deref())
    }

    fn retain_seqs(&mut self, live: &[usize]) {
        self.seqs.retain(|id, _| live.contains(id));
    }

    fn on_step(&mut self, seq_id: usize, anchor_pos: usize, now: Instant) -> AdaptiveDecision {
        if !self.enabled || self.max_bs < 2 {
            return AdaptiveDecision::Draft(self.max_bs);
        }
        let max_bs = self.max_bs;
        let state = self
            .seqs
            .entry(seq_id)
            .or_insert_with(|| SeqAdaptiveState::new(max_bs));

        if let Some(prev) = state.last_step.take() {
            match prev {
                LastAdaptiveStep::Speculative {
                    anchor_pos: prev_pos,
                    draft_len,
                    started_at,
                } if anchor_pos > prev_pos => {
                    let commit = (anchor_pos - prev_pos).min(draft_len + 1);
                    let elapsed_ms =
                        now.saturating_duration_since(started_at).as_secs_f64() * 1000.0;
                    Self::record_spec_observation(seq_id, state, max_bs, commit, elapsed_ms);
                }
                LastAdaptiveStep::PlainProbe {
                    anchor_pos: prev_pos,
                    started_at,
                    warmup,
                } if !warmup && anchor_pos > prev_pos => {
                    let step_ms = now.saturating_duration_since(started_at).as_secs_f64() * 1000.0
                        / ((anchor_pos - prev_pos) as f64);
                    Self::record_probe_observation(seq_id, state, step_ms);
                }
                _ => {}
            }
        }

        match &mut state.phase {
            AdaptivePhase::Warmup { .. } | AdaptivePhase::Active { .. } => {
                let bs = state.active_bs;
                state.last_step = Some(LastAdaptiveStep::Speculative {
                    anchor_pos,
                    draft_len: bs.saturating_sub(1),
                    started_at: now,
                });
                AdaptiveDecision::Draft(bs)
            }
            AdaptivePhase::PlainProbe { remaining, .. } => {
                let is_warmup = *remaining == ADAPTIVE_PROBE_STEPS;
                *remaining = remaining.saturating_sub(1);
                state.last_step = Some(LastAdaptiveStep::PlainProbe {
                    anchor_pos,
                    started_at: now,
                    warmup: is_warmup,
                });
                AdaptiveDecision::ProbePlain
            }
            AdaptivePhase::StoodDown => AdaptiveDecision::Paused,
        }
    }

    fn record_spec_observation(
        seq_id: usize,
        state: &mut SeqAdaptiveState,
        max_bs: usize,
        commit: usize,
        elapsed_ms: f64,
    ) {
        match &mut state.phase {
            AdaptivePhase::Warmup { rounds_ms, commits } => {
                rounds_ms.push(elapsed_ms);
                commits.push(commit);
                if commits.len() < ADAPTIVE_WARMUP_ROUNDS {
                    return;
                }
                let total_commit: usize = commits.iter().sum();
                let mean_commit = (total_commit as f64) / (commits.len() as f64);

                // Drop the single slowest warmup round to stay resilient against transient GPU contention spikes.
                let mut pairs: Vec<(f64, usize)> = rounds_ms
                    .iter()
                    .copied()
                    .zip(commits.iter().copied())
                    .collect();
                pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let keep = (pairs.len() - 1).max(1);
                let kept_ms: f64 = pairs[..keep].iter().map(|(ms, _)| *ms).sum();
                let kept_tok: usize = pairs[..keep].iter().map(|(_, c)| *c).sum();
                let spec_ms_per_tok = kept_ms / (kept_tok.max(1) as f64);

                if mean_commit < 2.2 {
                    let reason = format!(
                        "warmup_low_commit(mean_commit={mean_commit:.2},spec={spec_ms_per_tok:.2}ms/tok)"
                    );
                    tracing::info!(
                        "dflash2 adaptive: seq {seq_id} standing down after warmup (mean_commit={mean_commit:.2}, spec={spec_ms_per_tok:.2}ms/tok)"
                    );
                    state.stand_down_reason = Some(reason);
                    state.phase = AdaptivePhase::StoodDown;
                } else if mean_commit >= 4.2 {
                    if state.active_bs < max_bs && mean_commit >= (state.active_bs as f64 - 1.5) {
                        state.active_bs = max_bs;
                    }
                    tracing::info!(
                        "dflash2 adaptive: seq {seq_id} locked active at bs={} (mean_commit={mean_commit:.2}, spec={spec_ms_per_tok:.2}ms/tok)",
                        state.active_bs
                    );
                    state.phase = AdaptivePhase::Active {
                        ema_commit: mean_commit,
                        total_rounds: commits.len(),
                    };
                } else {
                    state.phase = AdaptivePhase::PlainProbe {
                        remaining: ADAPTIVE_PROBE_STEPS,
                        spec_ms_per_tok,
                        mean_commit,
                        probe_min_ms: f64::INFINITY,
                    };
                }
            }
            AdaptivePhase::Active {
                ema_commit,
                total_rounds,
            } => {
                *total_rounds += 1;
                *ema_commit = 0.25 * (commit as f64) + 0.75 * *ema_commit;
                if state.active_bs < max_bs
                    && *total_rounds >= 3
                    && *ema_commit >= (state.active_bs as f64 - 0.8)
                {
                    state.active_bs = max_bs;
                } else if *total_rounds >= 10 && *ema_commit < 2.0 {
                    let reason = format!(
                        "low_ema_commit(ema_commit={ema_commit:.2},rounds={})",
                        *total_rounds
                    );
                    tracing::info!(
                        "dflash2 adaptive: seq {seq_id} standing down (ema_commit={ema_commit:.2} < 2.00)"
                    );
                    state.stand_down_reason = Some(reason);
                    state.phase = AdaptivePhase::StoodDown;
                }
            }
            AdaptivePhase::PlainProbe { .. } | AdaptivePhase::StoodDown => {}
        }
    }

    fn record_probe_observation(seq_id: usize, state: &mut SeqAdaptiveState, step_ms: f64) {
        if let AdaptivePhase::PlainProbe {
            remaining,
            spec_ms_per_tok,
            mean_commit,
            probe_min_ms,
        } = &mut state.phase
        {
            *probe_min_ms = probe_min_ms.min(step_ms);
            if *remaining == 0 {
                let est_plain_ms = *probe_min_ms * ADAPTIVE_CUDA_GRAPH_DISCOUNT;
                if *spec_ms_per_tok < est_plain_ms {
                    let ema = *mean_commit;
                    tracing::info!(
                        "dflash2 adaptive: seq {seq_id} staying active at bs={} (spec={:.2}ms/tok < est_plain={:.2}ms/tok, mean_commit={ema:.2})",
                        state.active_bs,
                        *spec_ms_per_tok,
                        est_plain_ms
                    );
                    state.phase = AdaptivePhase::Active {
                        ema_commit: ema,
                        total_rounds: ADAPTIVE_WARMUP_ROUNDS,
                    };
                } else {
                    let reason = format!(
                        "slower_than_plain_probe(spec={:.2}ms/tok>=est_plain={:.2}ms/tok,mean_commit={:.2})",
                        *spec_ms_per_tok, est_plain_ms, *mean_commit
                    );
                    tracing::info!(
                        "dflash2 adaptive: seq {seq_id} standing down for CUDA-graph decode (spec={:.2}ms/tok >= est_plain={:.2}ms/tok, mean_commit={:.2})",
                        *spec_ms_per_tok,
                        est_plain_ms,
                        *mean_commit
                    );
                    state.stand_down_reason = Some(reason);
                    state.phase = AdaptivePhase::StoodDown;
                }
            }
        }
    }
}

pub struct DFlash2Proposer {
    draft: Qwen3DFlash2,
    /// The target's embedding and output head. DFlash carries neither, and decoding
    /// through the target's own head is what puts the draft's logits on the verifier's
    /// scale. Held here because the pipelines call `propose` without an embedder.
    heads: SpeculativeSharedHeads,
    adaptive: AdaptiveDFlashGate,
}

impl DFlash2Proposer {
    pub fn new(draft: Qwen3DFlash2, heads: SpeculativeSharedHeads) -> Self {
        let max_bs = draft.config().block_size();
        Self {
            draft,
            heads,
            adaptive: AdaptiveDFlashGate::new(max_bs, dflash_adaptive_enabled()),
        }
    }

    /// Load the draft at `path` — either a standalone `.gguf` archive (`general.architecture = "dflash"`)
    /// or a directory holding `config.json` + `model.safetensors` — beside a target that lends
    /// `heads`. A non-zero `block_size` shortens the drafted block: fewer masked slots
    /// is a valid draft, while a block longer than the trained one asks for positions the
    /// draft never learned to fill. The draft runs F32 on CPU and BF16 on accelerators — its
    /// dtype moves the accept rate only, since the target verify decides every emitted token.
    pub fn from_checkpoint(
        path: &std::path::Path,
        block_size: usize,
        device: &Device,
        heads: SpeculativeSharedHeads,
    ) -> Result<Self> {
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        if let Some(gguf_path) = resolve_gguf_checkpoint(path)? {
            let mut file = std::fs::File::open(&gguf_path)
                .map_err(|e| hanzo_ml::Error::msg(format!("{}: {e}", gguf_path.display())))?;
            return Ok(Self::new(
                Qwen3DFlash2::load_gguf(&mut file, block_size, device, dtype)?,
                heads,
            ));
        }

        let mut cfg = DFlash2Config::from_json_file(path.join("config.json"))?;
        if block_size != 0 {
            if !(2..=cfg.block_size()).contains(&block_size) {
                hanzo_ml::bail!(
                    "dflash block_size {block_size} is outside 2..={}, the block this checkpoint was trained on",
                    cfg.block_size()
                );
            }
            cfg.dflash_config.block_size = block_size;
        }
        let vb = dflash_varbuilder(&path.join("model.safetensors"), dtype, device)?;
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

    fn retain_seqs(&mut self, live: &[usize]) {
        self.adaptive.retain_seqs(live);
    }

    fn is_paused_for_seq(&self, seq_id: Option<usize>) -> bool {
        self.adaptive.is_paused_for_seq(seq_id)
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
                SpeculativeProposal::new(
                    Vec::new()
                );
                batch
            ]))
        };
        if batch != 1 {
            return stand_down();
        }
        let seq_id = ctx.seq_ids.first().copied().unwrap_or(0);
        if self.adaptive.is_paused_for_seq(Some(seq_id)) {
            if let (Some(seq), Some(reason)) = (
                ctx.sequences.first(),
                self.adaptive.stand_down_reason(seq_id),
            ) {
                seq.record_speculative_stand_down(reason);
            }
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

        let active_bs = match self.adaptive.on_step(seq_id, anchor_pos, Instant::now()) {
            AdaptiveDecision::Draft(bs) => {
                if let Some(seq) = ctx.sequences.first() {
                    seq.record_draft_block_size(bs);
                }
                bs
            }
            AdaptiveDecision::ProbePlain => return stand_down(),
            AdaptiveDecision::Paused => {
                if let (Some(seq), Some(reason)) = (
                    ctx.sequences.first(),
                    self.adaptive.stand_down_reason(seq_id),
                ) {
                    seq.record_speculative_stand_down(reason);
                }
                return stand_down();
            }
        };

        let embed_fn = Arc::clone(&self.heads.embed);
        let embed = move |ids: &Tensor| embed_fn(ids);
        let head = Arc::clone(&self.heads.lm_head);
        let lm_head = move |hidden: &Tensor| head(hidden);
        // Argmax walk. The proposal carries no distribution, so the verifier accepts a draft only
        // when its own sample matches it, which is exact at any temperature; the p/q ratio test
        // would need the drafts drawn from the logits.
        let (tokens, _) = self.draft.draft_block_inner(
            hiddens,
            anchor_token,
            anchor_pos,
            &embed,
            &lm_head,
            0.0,
            false,
            Some(active_bs),
        )?;
        Ok(SpeculativeProposalBatch::new(vec![
            SpeculativeProposal::new(tokens),
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
            (0..(slots * hidden))
                .map(|v| (v + 1) as f32)
                .collect::<Vec<_>>(),
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
                    full_row[..ctx_start]
                        .iter()
                        .all(|v| *v == f32::NEG_INFINITY),
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
        let lm_head =
            |hidden: &Tensor| -> Result<Tensor> { hidden.matmul(&head_w.t()?.contiguous()?) };

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

    #[test]
    fn dflash_gguf_metadata_and_synthetic_load() -> Result<()> {
        use hanzo_ml::quantized::{GgmlDType, QTensor};
        use std::io::Cursor;

        let dev = Device::Cpu;
        let (vocab, h, inter, heads, kv_heads, hd) =
            (16usize, 32usize, 64usize, 4usize, 2usize, 8usize);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let (layers_n, block_size, kernel, group_size, rank, top_k) =
            (2usize, 4usize, 2usize, 2usize, 32usize, 4usize);
        let groups = h / group_size;
        let target_layers_1based = [2u32, 6u32, 10u32];
        let fused = target_layers_1based.len();

        let kv_owned: Vec<(&str, gguf_file::Value)> = vec![
            (
                "general.architecture",
                gguf_file::Value::String("dflash".into()),
            ),
            ("dflash.block_count", gguf_file::Value::U32(layers_n as u32)),
            ("dflash.context_length", gguf_file::Value::U32(64)),
            ("dflash.embedding_length", gguf_file::Value::U32(h as u32)),
            (
                "dflash.feed_forward_length",
                gguf_file::Value::U32(inter as u32),
            ),
            (
                "dflash.attention.head_count",
                gguf_file::Value::U32(heads as u32),
            ),
            (
                "dflash.attention.head_count_kv",
                gguf_file::Value::U32(kv_heads as u32),
            ),
            (
                "dflash.attention.key_length",
                gguf_file::Value::U32(hd as u32),
            ),
            (
                "dflash.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F32(1e-6),
            ),
            ("dflash.rope.freq_base", gguf_file::Value::F32(10_000_000.0)),
            (
                "dflash.rope.scaling.type",
                gguf_file::Value::String("yarn".into()),
            ),
            ("dflash.rope.scaling.factor", gguf_file::Value::F32(4.0)),
            (
                "dflash.rope.scaling.original_context_length",
                gguf_file::Value::U32(16),
            ),
            (
                "dflash.block_size",
                gguf_file::Value::U32(block_size as u32),
            ),
            (
                "dflash.conv_kernel_size",
                gguf_file::Value::U32(kernel as u32),
            ),
            (
                "dflash.conv_group_size",
                gguf_file::Value::U32(group_size as u32),
            ),
            ("dflash.selector_rank", gguf_file::Value::U32(rank as u32)),
            ("dflash.selector_top_k", gguf_file::Value::U32(top_k as u32)),
            (
                "dflash.target_layers",
                gguf_file::Value::Array(
                    target_layers_1based
                        .iter()
                        .map(|&v| gguf_file::Value::U32(v))
                        .collect(),
                ),
            ),
            ("dflash.attention.sliding_window", gguf_file::Value::U32(32)),
            (
                "dflash.attention.sliding_window_pattern",
                gguf_file::Value::Array(vec![
                    gguf_file::Value::Bool(true),
                    gguf_file::Value::Bool(false),
                ]),
            ),
            ("tokenizer.ggml.mask_token_id", gguf_file::Value::U32(7)),
        ];

        let mut tensors_owned: Vec<(String, QTensor)> = Vec::new();
        let mut push_t = |name: &str, shape: &[usize], quant: bool| -> Result<()> {
            let len: usize = shape.iter().product();
            let t = Tensor::full(0.05f32, len, &dev)?.reshape(shape)?;
            let dt = if quant {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            tensors_owned.push((name.to_string(), QTensor::quantize(&t, dt)?));
            Ok(())
        };
        push_t("fc.weight", &[h, fused * h], true)?;
        push_t("enc.output_norm.weight", &[h], false)?;
        push_t("output_norm.weight", &[h], false)?;
        push_t("selector_hidden.weight", &[rank, h], true)?;
        push_t("selector_predecessor.weight", &[vocab, rank], true)?;
        push_t("selector_successor.weight", &[vocab, rank], true)?;
        for i in 0..layers_n {
            let p = format!("blk.{i}");
            push_t(&format!("{p}.attn_norm.weight"), &[h], false)?;
            push_t(&format!("{p}.ffn_norm.weight"), &[h], false)?;
            push_t(&format!("{p}.attn_q.weight"), &[q_dim, h], true)?;
            push_t(&format!("{p}.attn_k.weight"), &[kv_dim, h], true)?;
            push_t(&format!("{p}.attn_v.weight"), &[kv_dim, h], true)?;
            push_t(&format!("{p}.attn_output.weight"), &[h, q_dim], true)?;
            push_t(&format!("{p}.attn_q_norm.weight"), &[hd], false)?;
            push_t(&format!("{p}.attn_k_norm.weight"), &[hd], false)?;
            push_t(&format!("{p}.attn_conv_base"), &[2, kernel, h], false)?;
            push_t(
                &format!("{p}.attn_conv_proj.weight"),
                &[2 * kernel * groups, h],
                true,
            )?;
            push_t(&format!("{p}.ffn_conv_base"), &[2, kernel, h], false)?;
            push_t(
                &format!("{p}.ffn_conv_proj.weight"),
                &[2 * kernel * groups, h],
                true,
            )?;
            push_t(&format!("{p}.ffn_gate.weight"), &[inter, h], true)?;
            push_t(&format!("{p}.ffn_up.weight"), &[inter, h], true)?;
            push_t(&format!("{p}.ffn_down.weight"), &[h, inter], true)?;
        }

        let kv_refs: Vec<(&str, &gguf_file::Value)> =
            kv_owned.iter().map(|(k, v)| (*k, v)).collect();
        let tensor_refs: Vec<(&str, &QTensor)> =
            tensors_owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut buf = Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &kv_refs, &tensor_refs)?;

        let mut reader = Cursor::new(buf.into_inner());
        let model = Qwen3DFlash2::load_gguf(&mut reader, 3, &dev, DType::F32)?;
        assert_eq!(model.config().dflash_config.target_layer_ids, vec![1, 5, 9]);
        assert_eq!(model.config().dflash_config.mask_token_id, 7);
        assert_eq!(model.config().block_size(), 3);
        assert_eq!(model.config().draft_len(), 2);
        assert_eq!(model.config().layer_window(0), Some(32));
        assert_eq!(model.config().layer_window(1), None);
        assert!(model.config().yarn()?.is_some());

        let embed_w = Tensor::randn(0f32, 0.1f32, (vocab, h), &dev)?;
        let head_w = Tensor::randn(0f32, 0.1f32, (vocab, h), &dev)?;
        let embed = |ids: &Tensor| -> Result<Tensor> {
            let flat = ids.flatten_all()?;
            embed_w
                .index_select(&flat, 0)?
                .reshape((ids.dim(0)?, ids.dim(1)?, h))
        };
        let lm_head =
            |hidden: &Tensor| -> Result<Tensor> { hidden.matmul(&head_w.t()?.contiguous()?) };
        let hiddens = (0..fused)
            .map(|_| Tensor::randn(0f32, 1f32, (6, h), &dev))
            .collect::<Result<Vec<_>>>()?;
        let window = crate::speculative::HiddenWindow::new(0, hiddens)?;
        let (tokens, logits) = model.draft_block(&window, 2, 6, &embed, &lm_head, 0.0)?;
        assert_eq!(tokens.dims(), &[2]);
        assert_eq!(logits.dims(), &[2, vocab]);
        for row in logits.to_vec2::<f32>()? {
            assert!(row.iter().any(|v| v.is_finite()));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dflash_selector_and_gqa_attention_parity() -> Result<()> {
        let cuda_dev = Device::new_cuda(0)?;
        let cpu_dev = Device::Cpu;
        let (vocab, rank, hidden, slots, top_k) = (128usize, 8usize, 16usize, 4usize, 6usize);

        let proj_w = Tensor::randn(0f32, 0.2f32, (rank, hidden), &cpu_dev)?;
        let pred_cb = Tensor::randn(0f32, 0.2f32, (vocab, rank), &cpu_dev)?;
        let succ_cb = Tensor::randn(0f32, 0.2f32, (vocab, rank), &cpu_dev)?;
        let hidden_cpu = Tensor::randn(0f32, 0.5f32, (slots, hidden), &cpu_dev)?;
        let logits_cpu = Tensor::randn(0f32, 2.0f32, (slots, vocab), &cpu_dev)?;

        let sel_cpu = CandidateSelector {
            hidden_projection: Linear::new(proj_w.clone(), None),
            predecessor_codebook: pred_cb.clone(),
            successor_codebook: succ_cb.clone(),
            top_k,
        };
        let (cpu_tokens, cpu_rows) = sel_cpu.walk(&hidden_cpu, &logits_cpu, 7, 0.0)?;

        let sel_cuda = CandidateSelector {
            hidden_projection: Linear::new(proj_w.to_device(&cuda_dev)?, None),
            predecessor_codebook: pred_cb.to_device(&cuda_dev)?,
            successor_codebook: succ_cb.to_device(&cuda_dev)?,
            top_k,
        };
        let hidden_cuda = hidden_cpu.to_device(&cuda_dev)?;
        let logits_cuda = logits_cpu.to_device(&cuda_dev)?;
        let (cuda_tokens, cuda_rows) = sel_cuda.walk(&hidden_cuda, &logits_cuda, 7, 0.0)?;
        assert_eq!(cuda_tokens, cpu_tokens);

        let (no_logit_tokens, no_logits) =
            sel_cuda.walk_inner(&hidden_cuda, &logits_cuda, 7, 0.0, false)?;
        assert_eq!(no_logit_tokens, cpu_tokens);
        assert!(no_logits.is_none());

        let cpu_v = cpu_rows.flatten_all()?.to_vec1::<f32>()?;
        let cuda_v = cuda_rows
            .to_device(&cpu_dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for i in 0..cpu_v.len() {
            if cpu_v[i].is_infinite() {
                assert!(cuda_v[i].is_infinite() && cuda_v[i].is_sign_negative());
            } else {
                assert!(
                    (cpu_v[i] - cuda_v[i]).abs() < 1e-4,
                    "selector walk logit mismatch at {i}: {} vs {}",
                    cpu_v[i],
                    cuda_v[i]
                );
            }
        }

        // Compare GQA-reshaped DFlash2 attention (with narrowed ctx_kv) against repeat_kv reference.
        let cfg: DFlash2Config = serde_json::from_str(
            r#"{
              "vocab_size": 128, "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 1,
              "head_dim": 4, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
              "rope_parameters": {"rope_theta": 10000.0},
              "dflash_config": {
                "target_layer_ids": [0], "mask_token_id": 1, "block_size": 4,
                "conv_group_size": 16, "conv_kernel_size": 2,
                "selector_rank": 8, "selector_top_k": 6
              }
            }"#,
        )
        .map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        let make_lin = |out_d: usize, in_d: usize| -> Result<Linear> {
            Ok(Linear::new(
                Tensor::randn(0f32, 0.2f32, (out_d, in_d), &cuda_dev)?,
                None,
            ))
        };
        let make_conv = || -> Result<BlockConv> {
            Ok(BlockConv {
                base_kernel: Tensor::zeros((2, 2, 16), DType::F32, &cuda_dev)?,
                kernel_projection: make_lin(4, 16)?,
                kernel_size: 2,
                group_size: 16,
                groups: 1,
            })
        };
        let q_proj = make_lin(16, 16)?;
        let k_proj = make_lin(4, 16)?;
        let v_proj = make_lin(4, 16)?;
        let gate_proj = make_lin(32, 16)?;
        let up_proj = make_lin(32, 16)?;
        let (qkv_proj, kv_proj, gate_up_proj) = DFlash2Layer::from_separate_projs(
            q_proj.clone(),
            k_proj.clone(),
            v_proj.clone(),
            gate_proj,
            up_proj,
        )?;
        let layer = DFlash2Layer {
            qkv_proj,
            kv_proj,
            o_proj: make_lin(16, 16)?,
            q_norm: RmsNorm::from_w(Tensor::ones(4, DType::F32, &cuda_dev)?, 1e-6)?,
            k_norm: RmsNorm::from_w(Tensor::ones(4, DType::F32, &cuda_dev)?, 1e-6)?,
            input_layernorm: RmsNorm::from_w(Tensor::ones(16, DType::F32, &cuda_dev)?, 1e-6)?,
            post_attention_layernorm: RmsNorm::from_w(
                Tensor::ones(16, DType::F32, &cuda_dev)?,
                1e-6,
            )?,
            gate_up_proj,
            down_proj: make_lin(16, 32)?,
            attention_conv: make_conv()?,
            mlp_conv: make_conv()?,
        };
        let model = Qwen3DFlash2 {
            fc: make_lin(16, 16)?,
            hidden_norm: RmsNorm::from_w(Tensor::ones(16, DType::F32, &cuda_dev)?, 1e-6)?,
            layers: vec![layer],
            norm: RmsNorm::from_w(Tensor::ones(16, DType::F32, &cuda_dev)?, 1e-6)?,
            selector: sel_cuda,
            rotary: RotaryEmbedding::new_yarn(1e4, 4, 64, None, &cuda_dev, true, DType::F32)?,
            cfg,
            device: cuda_dev.clone(),
            dtype: DType::F32,
            ctx_cache: std::sync::Mutex::new(None),
            mask_cache: std::sync::Mutex::new(HashMap::new()),
            mask_embed_cache: std::sync::Mutex::new(None),
        };
        let x = Tensor::randn(0f32, 0.3f32, (4, 16), &cuda_dev)?;
        let k_big = Tensor::randn(0f32, 0.3f32, (1, 16, 4), &cuda_dev)?;
        let v_big = Tensor::randn(0f32, 0.3f32, (1, 16, 4), &cuda_dev)?;
        // Narrowed context slice (off = 6, len = 10) to verify strided cuBLAS handling.
        let k_ctx = k_big.narrow(1, 6, 10)?;
        let v_ctx = v_big.narrow(1, 6, 10)?;
        let pos = Tensor::from_vec(vec![16u32, 17, 18, 19], (4,), &cuda_dev)?;
        let (cos_d, sin_d) = model.rope_cos_sin(&pos)?;
        let mask = block_mask(6, 10, 4, 16, Some(12), &cuda_dev, DType::F32)?;
        let got = model.attention(
            &model.layers[0],
            &x,
            &[(k_ctx.clone(), v_ctx.clone())],
            &cos_d,
            &sin_d,
            Some(&mask),
        )?;

        // Explicit repeat_kv + Tensor::cat reference.
        let l = &model.layers[0];
        let q = apply_rope(
            &hanzo_nn::ops::rms_norm(
                &q_proj.forward(&x)?.reshape((4, 4, 4))?.contiguous()?,
                l.q_norm.weight(),
                1e-6,
            )?
            .transpose(0, 1)?
            .contiguous()?,
            &cos_d,
            &sin_d,
        )?;
        let kn = apply_rope(
            &hanzo_nn::ops::rms_norm(
                &k_proj.forward(&x)?.reshape((4, 1, 4))?.contiguous()?,
                l.k_norm.weight(),
                1e-6,
            )?
            .transpose(0, 1)?
            .contiguous()?,
            &cos_d,
            &sin_d,
        )?;
        let vn = v_proj
            .forward(&x)?
            .reshape((4, 1, 4))?
            .transpose(0, 1)?
            .contiguous()?;
        let k_cat = repeat_kv(&Tensor::cat(&[&k_ctx, &kn], 1)?, 4)?;
        let v_cat = repeat_kv(&Tensor::cat(&[&v_ctx, &vn], 1)?, 4)?;
        let ref_scores = q
            .matmul(&k_cat.transpose(1, 2)?.contiguous()?)?
            .affine(0.5, 0.0)?
            .broadcast_add(&mask.unsqueeze(0)?)?;
        let ref_probs = hanzo_nn::ops::softmax_last_dim(&ref_scores)?;
        let ref_out = l.o_proj.forward(
            &ref_probs
                .matmul(&v_cat.contiguous()?)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((4, 16))?,
        )?;

        let got_v = got.flatten_all()?.to_vec1::<f32>()?;
        let ref_v = ref_out.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..got_v.len() {
            assert!(
                (got_v[i] - ref_v[i]).abs() < 1e-4,
                "gqa attention mismatch at {i}: {} vs {}",
                got_v[i],
                ref_v[i]
            );
        }
        Ok(())
    }

    #[test]
    fn dflash_adaptive_gate_expands_on_high_acceptance_and_retains_seqs() {
        use std::time::Duration;
        let mut gate = AdaptiveDFlashGate::new(16, true);
        let t0 = Instant::now();
        let mut pos = 100usize;

        // First 4 warmup rounds start at bs=8; committing 7 tokens/round (out of 8 max)
        // triggers expansion back to checkpoint max_bs=16.
        for r in 0..ADAPTIVE_WARMUP_ROUNDS {
            let dec = gate.on_step(1, pos, t0 + Duration::from_millis((r as u64) * 20));
            assert_eq!(dec, AdaptiveDecision::Draft(8));
            pos += 7;
        }
        let dec_after = gate.on_step(1, pos, t0 + Duration::from_millis(80));
        assert_eq!(dec_after, AdaptiveDecision::Draft(16));
        assert!(!gate.is_paused_for_seq(Some(1)));

        // retain_seqs drops finished sequences cleanly.
        gate.retain_seqs(&[2]);
        assert!(!gate.seqs.contains_key(&1));
    }

    #[test]
    fn dflash_adaptive_gate_probes_borderline_and_stands_down_when_slower() {
        use std::time::Duration;
        let mut gate = AdaptiveDFlashGate::new(16, true);
        let t0 = Instant::now();
        let mut now = t0;
        let mut pos = 100usize;

        // Seq 1: borderline 3.25 commit/round at 110ms/round (~33.8 ms/tok),
        // plain m=1 probe takes 34ms (est CUDA-graph = 28.9 ms/tok) -> stands down.
        let commits = [3, 4, 3, 3];
        for c in commits {
            assert_eq!(gate.on_step(1, pos, now), AdaptiveDecision::Draft(8));
            pos += c;
            now += Duration::from_millis(110);
        }
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            assert_eq!(gate.on_step(1, pos, now), AdaptiveDecision::ProbePlain);
            assert!(!gate.is_paused_for_seq(Some(1)));
            pos += 1;
            now += Duration::from_millis(34);
        }
        assert_eq!(gate.on_step(1, pos, now), AdaptiveDecision::Paused);
        assert!(gate.is_paused_for_seq(Some(1)));
        assert!(gate
            .stand_down_reason(1)
            .is_some_and(|r| r.starts_with("slower_than_plain_probe(")));

        // Seq 2: same 3.25 commit/round at 90ms/round (~27.7 ms/tok),
        // plain m=1 probe takes 125ms (est = 106.25 ms/tok) -> stays active.
        pos = 500;
        for c in commits {
            assert_eq!(gate.on_step(2, pos, now), AdaptiveDecision::Draft(8));
            pos += c;
            now += Duration::from_millis(90);
        }
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            assert_eq!(gate.on_step(2, pos, now), AdaptiveDecision::ProbePlain);
            pos += 1;
            now += Duration::from_millis(125);
        }
        assert_eq!(gate.on_step(2, pos, now), AdaptiveDecision::Draft(8));
        assert!(!gate.is_paused_for_seq(Some(2)));
        assert_eq!(gate.stand_down_reason(2), None);
        assert!(gate.is_paused_for_seq(Some(1)));
    }

    #[test]
    fn dflash_adaptive_gate_immediately_stands_down_on_collapsing_acceptance() {
        use std::time::Duration;
        let mut gate = AdaptiveDFlashGate::new(16, true);
        let mut now = Instant::now();
        let mut pos = 50usize;

        for _ in 0..ADAPTIVE_WARMUP_ROUNDS {
            assert_eq!(gate.on_step(7, pos, now), AdaptiveDecision::Draft(8));
            pos += 1; // 0 drafts accepted
            now += Duration::from_millis(40);
        }
        assert_eq!(gate.on_step(7, pos, now), AdaptiveDecision::Paused);
        assert!(gate.is_paused_for_seq(Some(7)));
        assert!(gate
            .stand_down_reason(7)
            .is_some_and(|r| r.starts_with("warmup_low_commit(")));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dflash2_cuda_block_conv_and_residual_match_cpu() -> Result<()> {
        let Ok(cuda_dev) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let (slots, hidden, kernel, group_size) = (16usize, 256usize, 2usize, 16usize);
        let groups = hidden / group_size;

        let base_v: Vec<f32> = (0..(2 * kernel * hidden))
            .map(|i| ((i as f32) * 0.031 - 0.5).sin() * 0.5)
            .collect();
        let proj_v: Vec<f32> = (0..(2 * kernel * groups * hidden))
            .map(|i| ((i as f32) * 0.017 + 0.2).cos() * 0.005)
            .collect();
        let x_v: Vec<f32> = (0..(slots * hidden))
            .map(|i| ((i as f32) * 0.023 - 0.8).sin())
            .collect();
        let res_v: Vec<f32> = (0..(slots * hidden))
            .map(|i| ((i as f32) * 0.011 + 0.4).cos() * 0.7)
            .collect();

        let cpu_f32 = BlockConv {
            base_kernel: Tensor::from_vec(base_v.clone(), (2, kernel, hidden), &cpu)?,
            kernel_projection: Linear::new(
                Tensor::from_vec(proj_v.clone(), (2 * kernel * groups, hidden), &cpu)?,
                None,
            ),
            kernel_size: kernel,
            group_size,
            groups,
        };
        let x_cpu_f32 = Tensor::from_vec(x_v.clone(), (slots, hidden), &cpu)?;
        let res_cpu_f32 = Tensor::from_vec(res_v.clone(), (slots, hidden), &cpu)?;
        let (pre_ref_f32, taps_ref_f32) = cpu_f32.prepare(&x_cpu_f32)?;
        let fin_ref_f32 = cpu_f32.finish_residual(&pre_ref_f32, &taps_ref_f32, &res_cpu_f32)?;

        for dt in [DType::F32, DType::F16, DType::BF16] {
            let kernel_tol = match dt {
                DType::F32 => 0.0,
                DType::F16 => 2e-3,
                DType::BF16 => 2e-2,
                _ => unreachable!(),
            };
            let e2e_tol = match dt {
                DType::F32 => 1e-4,
                DType::F16 => 5e-3,
                DType::BF16 => 4e-2,
                _ => unreachable!(),
            };
            let cuda_conv = BlockConv {
                base_kernel: Tensor::from_vec(base_v.clone(), (2, kernel, hidden), &cuda_dev)?
                    .to_dtype(dt)?,
                kernel_projection: Linear::new(
                    Tensor::from_vec(proj_v.clone(), (2 * kernel * groups, hidden), &cuda_dev)?
                        .to_dtype(dt)?,
                    None,
                ),
                kernel_size: kernel,
                group_size,
                groups,
            };

            let x_cuda = Tensor::from_vec(x_v.clone(), (slots, hidden), &cuda_dev)?.to_dtype(dt)?;
            let res_cuda =
                Tensor::from_vec(res_v.clone(), (slots, hidden), &cuda_dev)?.to_dtype(dt)?;

            let taps_exact_cuda = taps_ref_f32.to_dtype(dt)?.to_device(&cuda_dev)?;
            let pre_exact_cuda = cuda_conv.convolve(&x_cuda, &taps_exact_cuda, 0, None)?;
            let fin_exact_cuda =
                cuda_conv.convolve(&pre_exact_cuda, &taps_exact_cuda, 1, Some(&res_cuda))?;
            let pre_exact_diff = (&pre_exact_cuda.to_device(&cpu)?.to_dtype(DType::F32)?
                - &pre_ref_f32)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let fin_exact_diff = (&fin_exact_cuda.to_device(&cpu)?.to_dtype(DType::F32)?
                - &fin_ref_f32)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                pre_exact_diff <= kernel_tol && fin_exact_diff <= kernel_tol,
                "dflash2_block_conv kernel mismatch dt={dt:?}: pre={pre_exact_diff} fin={fin_exact_diff}"
            );

            let (pre_cuda, taps_cuda) = cuda_conv.prepare(&x_cuda)?;
            let fin_cuda = cuda_conv.finish_residual(&pre_cuda, &taps_cuda, &res_cuda)?;

            let pre_diff = (&pre_cuda.to_device(&cpu)?.to_dtype(DType::F32)? - &pre_ref_f32)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let fin_diff = (&fin_cuda.to_device(&cpu)?.to_dtype(DType::F32)? - &fin_ref_f32)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                pre_diff <= e2e_tol && fin_diff <= e2e_tol,
                "dflash2_block_conv e2e mismatch dt={dt:?}: pre={pre_diff} fin={fin_diff}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dflash2_cuda_topk_and_greedy_walk_match_cpu() -> Result<()> {
        let Ok(cuda_dev) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let (rows, vocab, k) = (8usize, 4096usize, 64usize);
        let mut logits_v: Vec<f32> = (0..(rows * vocab))
            .map(|i| ((i as f32) * 0.137 - 2.3).sin() * 5.0)
            .collect();
        logits_v[5] = 100.0;
        logits_v[19] = 100.0;
        logits_v[777] = f32::NAN;
        logits_v[vocab + 12] = f32::NEG_INFINITY;

        let logits_cuda = Tensor::from_vec(logits_v.clone(), (rows, vocab), &cuda_dev)?;
        let (vals_cuda, ids_cuda) = crate::cuda::dflash2::dflash2_topk_cuda(&logits_cuda, k)?;
        let got_vals = vals_cuda.to_vec2::<f32>()?;
        let got_ids = ids_cuda.to_vec2::<u32>()?;
        for r in 0..rows {
            let (want_ids, want_vals) = top_k_of(&logits_v[r * vocab..(r + 1) * vocab], k);
            assert_eq!(got_ids[r], want_ids, "topk ids mismatch at row {r}");
            for j in 0..k {
                assert!(
                    (got_vals[r][j] - want_vals[j]).abs() < 1e-6,
                    "topk val mismatch at ({r}, {j})"
                );
            }
        }

        let (slots, hidden, rank, top_k) = (15usize, 64usize, 128usize, 32usize);
        let w_h: Vec<f32> = (0..(rank * hidden))
            .map(|i| ((i as f32) * 0.029).sin() * 0.2)
            .collect();
        let pred_v: Vec<f32> = (0..(vocab * rank))
            .map(|i| ((i as f32) * 0.017 - 0.3).cos() * 0.3)
            .collect();
        let succ_v: Vec<f32> = (0..(vocab * rank))
            .map(|i| ((i as f32) * 0.023 + 0.1).sin() * 0.3)
            .collect();
        let hid_v: Vec<f32> = (0..(slots * hidden))
            .map(|i| ((i as f32) * 0.041).cos())
            .collect();
        let walk_logits_v: Vec<f32> = (0..(slots * vocab))
            .map(|i| ((i as f32) * 0.071 - 1.1).sin() * 2.0)
            .collect();

        let sel_cpu = CandidateSelector {
            hidden_projection: Linear::new(
                Tensor::from_vec(w_h.clone(), (rank, hidden), &cpu)?,
                None,
            ),
            predecessor_codebook: Tensor::from_vec(pred_v.clone(), (vocab, rank), &cpu)?,
            successor_codebook: Tensor::from_vec(succ_v.clone(), (vocab, rank), &cpu)?,
            top_k,
        };
        let sel_cuda = CandidateSelector {
            hidden_projection: Linear::new(Tensor::from_vec(w_h, (rank, hidden), &cuda_dev)?, None),
            predecessor_codebook: Tensor::from_vec(pred_v, (vocab, rank), &cuda_dev)?,
            successor_codebook: Tensor::from_vec(succ_v, (vocab, rank), &cuda_dev)?,
            top_k,
        };

        let (want_toks, _) = sel_cpu.walk_inner(
            &Tensor::from_vec(hid_v.clone(), (slots, hidden), &cpu)?,
            &Tensor::from_vec(walk_logits_v.clone(), (slots, vocab), &cpu)?,
            42,
            0.0,
            false,
        )?;
        let (got_toks, got_logits) = sel_cuda.walk_inner(
            &Tensor::from_vec(hid_v, (slots, hidden), &cuda_dev)?,
            &Tensor::from_vec(walk_logits_v, (slots, vocab), &cuda_dev)?,
            42,
            0.0,
            false,
        )?;
        assert!(got_logits.is_none());
        assert_eq!(
            got_toks, want_toks,
            "CUDA greedy walk tokens mismatch vs CPU"
        );
        Ok(())
    }
}
