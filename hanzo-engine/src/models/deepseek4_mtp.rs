//! DeepSeek-V4 MTP (Multi-Token-Prediction) speculative draft head —
//! `general.architecture = "deepseek4_mtp_support"` (the separate ~3.6 GB GGUF
//! `DeepSeek-V4-Flash-MTP-…`, `nextn_predict_layers = 1`).
//!
//! The MTP head is structurally ONE V4 decoder block (`mtp.0.*`: attn + MoE + per-
//! sublayer Hyper-Connections — loaded via the shared [`DecoderLayer::load`]) wrapped
//! with a NextN entry/exit: it predicts token `t+1` from the base model's hidden state
//! at `t` plus the embedding of token `t`:
//!
//! ```text
//!   x = e_proj · enorm(embed(token_t))  +  h_proj · hnorm(hidden_t)   // [b,s,e]
//!   carrier = HC.expand(x); carrier = block(carrier); x = head_hc.reduce(carrier)
//!   logits  = base_output( norm(x) )                                   // [b,s,vocab]
//! ```
//!
//! `e_proj`/`h_proj` are two `[e,e]` projections summed (≡ a `[2e,e]` concat-proj),
//! matching DeepSeek's NextN module + llama.cpp's MTP (PR #22673, NextN tensors +
//! `n_layer_nextn`). The output head + token embeddings are SHARED with the base model.
//!
//! Speculative use: research (antirez/ds4 + SGLang) shows naive **depth-1 MTP is
//! net-negative single-stream** (accept ~1.8-2.3, verify 2× → −21%). The value is
//! **EAGLE-style** (accept ~2.5) with **batched verify** (stream weights once for all
//! K draft tokens inside a fixed-shape graph). So [`MtpHead::draft`] is built to be
//! looped for multi-step drafting; the accept/verify loop is the engine-side follow-on.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::sync::Arc;

use crate::gguf::Content;
use crate::layers::{
    CausalMaskConfig, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RotaryEmbedding, RmsNorm,
};
use crate::layers_masker::PastKvLenCache;
use crate::pipeline::KvCache;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::QuantMethod;

use crate::speculative::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    TargetTokenEmbedder,
};

use super::deepseek4::HyperConnections;
use super::quantized_deepseek4::{deq, gguf_linear, rms_from, DecoderLayer, PropsGGUF};

/// The DeepSeek-V4 MTP draft head: a single V4 block + NextN entry/exit. Shares the
/// base model's token embeddings + output projection (passed to [`Self::draft`]).
pub struct MtpHead {
    enorm: RmsNorm,
    hnorm: RmsNorm,
    e_proj: Arc<dyn QuantMethod>,
    h_proj: Arc<dyn QuantMethod>,
    block: DecoderLayer,
    head_hc: HyperConnections,
    norm: RmsNorm,
    n_hc: usize,
    #[allow(dead_code)]
    head_dim: usize,
    device: Device,
    dtype: DType,
    /// Single-block KV cache, held as a 1-element Vec so it satisfies `PastKvLenCache`
    /// (mirrors the base model's `NormalCache`).
    cache: Vec<KvCache>,
}

impl MtpHead {
    /// Load the `mtp.0.*` head from an opened MTP-GGUF [`Content`]. `props` is the base
    /// model's config (the MTP block shares all V4 hyperparams). The MTP block is a
    /// Full-mode V4 block (no compressor, bias-routed MoE).
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        props: &PropsGGUF,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let eps = props.rms_norm_eps as f64;
        let softmax_scale = 1.0 / (props.head_dim as f32).sqrt();
        let group_in = props.head_count * props.head_dim / props.o_groups;

        // Full-mode RoPE (θ=rope_theta, no YaRN) — the MTP block is not compressed.
        let rope = Arc::new(DeepSeekV2RotaryEmbedding::new(
            &DeepSeekV2RopeConfig {
                rope_scaling: None,
                max_position_embeddings: props.max_seq_len,
                rope_theta: props.rope_theta,
                qk_rope_head_dim: props.rope_head_dim,
            },
            DType::F32,
            device,
        )?);

        // The V4 block at `mtp.0` (Full mode, bias-routed, swiglu clamp 10 like V4).
        let block = DecoderLayer::load(
            ct,
            "mtp.0",
            props,
            device,
            rope,
            /*compress_ratio*/ 0,
            group_in,
            eps,
            softmax_scale,
            dtype,
            /*is_hash*/ false,
            /*swiglu_clamp*/ 10.0,
        )?;

        // NextN entry: norm + project the prev hidden and the next-token embedding.
        let enorm = rms_from(deq(ct, "mtp.0.enorm.weight", device)?, eps)?;
        let hnorm = rms_from(deq(ct, "mtp.0.hnorm.weight", device)?, eps)?;
        let e_proj = gguf_linear(ct.tensor("mtp.0.e_proj.weight", device)?)?;
        let h_proj = gguf_linear(ct.tensor("mtp.0.h_proj.weight", device)?)?;
        let norm = rms_from(deq(ct, "mtp.0.norm.weight", device)?, eps)?;

        // Head Hyper-Connection (reduce-only) — `hc_head_*`.
        let head_hc = HyperConnections::from_parts(
            gguf_linear(ct.tensor("mtp.0.hc_head_fn.weight", device)?)?,
            deq(ct, "mtp.0.hc_head_scale.weight", device)?.to_vec1::<f32>()?,
            deq(ct, "mtp.0.hc_head_base.weight", device)?,
            props.hc_count,
            props.hc_sinkhorn_iters,
            props.hc_eps,
            true,
        )?;

        Ok(Self {
            enorm,
            hnorm,
            e_proj,
            h_proj,
            block,
            head_hc,
            norm,
            n_hc: props.hc_count,
            head_dim: props.head_dim,
            device: device.clone(),
            dtype,
            cache: vec![KvCache::new_normal(2, props.max_seq_len, 512)],
        })
    }

    /// Reset the draft head's KV cache (call before each fresh speculative round).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        for c in &mut self.cache {
            c.reset();
        }
    }

    /// Draft the next-token logits from the base model's `hidden` state `[b,s,e]` and
    /// the corresponding `token_ids` `[b,s]`, sharing the base `embed` + `output` head.
    /// `start_offsets` are the per-sequence positions (for RoPE/causality).
    ///
    /// Returns `(logits [b,s,vocab], hidden [b,s,e])`. The returned post-norm hidden is
    /// the EAGLE chain feature: feed it back (with the drafted token) as `hidden` for the
    /// next chained draft step, extending the draft beyond depth-1 off a single head.
    pub fn draft(
        &mut self,
        hidden: &Tensor,
        token_ids: &Tensor,
        embed: &Embedding,
        output: &Arc<dyn QuantMethod>,
        start_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let e = self.e_proj.forward(
            &self
                .enorm
                .forward(&embed.forward(token_ids)?.to_dtype(self.dtype)?)?,
        )?;
        let h = self
            .h_proj
            .forward(&self.hnorm.forward(&hidden.to_dtype(self.dtype)?)?)?;
        let x = (e + h)?; // [b, s, e]

        let mask = CausalMasker.make_causal_mask(
            token_ids,
            &self.cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let positions = Tensor::from_vec(
            start_offsets.iter().map(|&o| o as u32).collect::<Vec<_>>(),
            start_offsets.len(),
            &self.device,
        )?;

        // One V4 block over the HC carrier (no compressor — Full mode, so the
        // `is_prefill` gate is moot here; pass false).
        let mut hc = HyperConnections::expand(&x, self.n_hc)?;
        hc = self.block.forward(
            &hc,
            token_ids,
            &mask,
            &positions,
            &mut self.cache[0],
            None,
            false,
        )?;
        let x = self.head_hc.reduce_output(&hc)?;
        let x = self.norm.forward(&x)?;
        let logits = output.forward(&x.contiguous()?)?;
        Ok((logits, x))
    }

    /// Head dim (for callers wiring the draft into a speculative loop).
    #[allow(dead_code)]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Drop the draft block's KV history. The proposer resets before each draft so the
    /// MTP attends only to the current token over the target hidden (which already
    /// encodes the full context) — a stateless draft. This trades some accept rate for
    /// a drift-free cache; it never affects correctness (the target verify does).
    pub fn reset_cache(&mut self) {
        for c in self.cache.iter_mut() {
            c.reset();
        }
    }
}

/// The `SpeculativeProposer` adapter that plugs the V4 MTP head into the generic
/// speculative driver. Holds the head plus clones of the base model's token
/// embeddings + output projection (the head shares both). Correctness comes from the
/// target verify forward, NOT from the draft — so this uses a stateless single-token
/// draft (see `MtpHead::reset_cache`); accept rate is what it trades, and that's what
/// the validation measures.
pub struct Deepseek4MtpRuntime {
    mtp: MtpHead,
    embed: Embedding,
    output: Arc<dyn QuantMethod>,
    /// The MAXIMUM chained-draft depth. The chain stops early (adaptive draft
    /// length) when the draft head's confidence drops below `confidence_threshold`.
    n_predict: usize,
    /// Adaptive draft-length gate (DSpark's load-aware idea, minus the trained
    /// predictor): keep extending the EAGLE chain only while the just-drafted
    /// token's softmax top-probability stays `>= confidence_threshold` on every
    /// active row. `0.0` never gates (the chain always runs the full `n_predict` —
    /// exactly the prior fixed-depth behavior). A high-confidence draft token is a
    /// strong proxy for "the target will accept it", so gating spends draft depth
    /// on the easy spans (long confident runs) and stays shallow on hard spans
    /// (where deeper drafts would just be rejected, wasting verify width). It is
    /// correctness-neutral — the target verify still decides every emitted token.
    confidence_threshold: f32,
}

impl Deepseek4MtpRuntime {
    pub fn new(
        mtp: MtpHead,
        embed: Embedding,
        output: Arc<dyn QuantMethod>,
        n_predict: usize,
    ) -> Self {
        // Precedent: infra/experiment tuning knobs (e.g. IGPU_MEMORY_FRACTION) are
        // plain env vars. Default 0.0 = fixed-depth = byte-identical to before.
        let confidence_threshold = std::env::var("MTP_CONF_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.0);
        Self {
            mtp,
            embed,
            output,
            n_predict: n_predict.max(1),
            confidence_threshold,
        }
    }
}

/// DeepSeek-V4 is self-speculative: its MTP head ships as a companion GGUF (the
/// path carried by `cfg`), sharing the base model's token embeddings + output
/// projection. This impl is the ONE place that knowledge lives — the pipeline
/// asks for the capability and never names DeepSeek. Other MTP models (GLM-5.2's
/// in-band `nextn` head, …) implement the same trait next to their own weights.
impl crate::speculative::SelfSpeculative for super::quantized_deepseek4::ModelWeights {
    fn attach_mtp(
        &self,
        cfg: &crate::speculative::MtpConfig,
    ) -> Result<Box<dyn SpeculativeProposer + Send + Sync>> {
        let path = cfg.resolve_path()?;
        let mut readers = [std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)?];
        let mut readers_ref: Vec<&mut std::fs::File> = readers.iter_mut().collect();
        let mut ct = crate::gguf::Content::from_readers(&mut readers_ref)?;
        let head = MtpHead::load(
            &mut ct,
            self.base_props(),
            &self.device,
            self.compute_dtype(),
        )?;
        let runtime = Deepseek4MtpRuntime::new(
            head,
            self.embeddings().clone(),
            self.output_head(),
            cfg.n_predict.unwrap_or(1),
        );
        // The proposer needs the pre-output hidden every step; the clone is cheap
        // versus the forward, so stash it once speculation is attached.
        self.set_store_spec_hidden(true);
        Ok(Box::new(runtime))
    }
}

impl SpeculativeProposer for Deepseek4MtpRuntime {
    fn proposal_len(&self) -> usize {
        self.n_predict
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let hiddens = ctx.target_hiddens.ok_or_else(|| {
            hanzo_ml::Error::Msg("V4 MTP requires the target hidden state for proposal".into())
        })?;
        let batch = ctx.sampled_tokens.len();
        if batch == 0 {
            return Ok(SpeculativeProposalBatch::new(Vec::new()));
        }
        let device = self.mtp.device().clone();

        // EAGLE-style chained draft off a single head: step 0 drafts from the target
        // hidden + anchor; each next step feeds the MTP's OWN post-norm hidden + the
        // token it just drafted, extending the draft to `n_predict` tokens. The KV cache
        // accumulates the chain (reset once per proposal); the target verify decides
        // acceptance, so draft quality only affects accept rate, never correctness.
        let mut hidden = if hiddens.dims().len() == 3 {
            hiddens
        } else {
            hiddens.unsqueeze(1)?
        };
        let mut cur_tokens: Vec<u32> = ctx.sampled_tokens.to_vec();
        let base: Vec<usize> = ctx.base_lens.to_vec();

        self.mtp.reset_cache();
        let mut step_tokens: Vec<Vec<u32>> = Vec::with_capacity(self.n_predict); // [step][row]
        let mut step_logits: Vec<Tensor> = Vec::with_capacity(self.n_predict); // [step] = [batch,1,vocab]
        for k in 0..self.n_predict {
            let token_ids = Tensor::from_vec(cur_tokens.clone(), (batch, 1), &device)?;
            let start_offsets: Vec<usize> = base.iter().map(|b| b + k).collect();
            let (logits, next_hidden) = self.mtp.draft(
                &hidden,
                &token_ids,
                &self.embed,
                &self.output,
                &start_offsets,
            )?;
            let draft = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;
            let draft_ids: Vec<u32> = draft.flatten_all()?.to_vec1::<u32>()?;
            if draft_ids.len() != batch {
                hanzo_ml::bail!(
                    "V4 MTP draft produced {} tokens for {batch} rows",
                    draft_ids.len()
                );
            }
            cur_tokens = draft_ids.clone();
            hidden = next_hidden;
            step_tokens.push(draft_ids);
            step_logits.push(logits.clone());

            // Adaptive draft length: after emitting step k, decide whether to extend
            // the chain. Stop once the WEAKEST active row's draft confidence drops
            // below the gate — a batch-uniform length keeps the staged proposals
            // homogeneous (one target verify shape), so batched verify is unaffected;
            // only the depth varies per proposal round. `0.0` never triggers (the
            // min top-prob is always >= 0), preserving fixed-depth behavior exactly.
            if self.confidence_threshold > 0.0 && k + 1 < self.n_predict {
                // Confidence in F32: logits are the model compute dtype (bf16);
                // softmax + the to_vec1::<f32> readback both need F32.
                let probs = hanzo_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?; // [batch,1,vocab]
                let top = probs.max(D::Minus1)?; // [batch,1]
                let top_probs: Vec<f32> = top.flatten_all()?.to_vec1::<f32>()?;
                let min_conf = top_probs.iter().cloned().fold(f32::INFINITY, f32::min);
                if min_conf < self.confidence_threshold {
                    break;
                }
            }
        }

        // Per row: the n_predict drafted tokens + their logits stacked as [1, n_predict, vocab]
        // (the verifier indexes logit rows by draft position).
        let mut proposals = Vec::with_capacity(batch);
        for i in 0..batch {
            let toks: Vec<u32> = step_tokens.iter().map(|s| s[i]).collect();
            let row_logits = step_logits
                .iter()
                .map(|l| l.narrow(0, i, 1))
                .collect::<Result<Vec<_>>>()?;
            let row_logits = Tensor::cat(&row_logits, 1)?; // [1, n_predict, vocab]
            proposals.push(SpeculativeProposal::with_logits(toks, row_logits));
        }
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}
