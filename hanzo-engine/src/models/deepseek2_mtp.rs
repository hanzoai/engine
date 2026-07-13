//! GLM-5.2 (`glm-dsa`) in-band `nextn` MTP (Multi-Token-Prediction) draft head.
//!
//! Unlike DeepSeek-V4 — whose MTP head ships as a SEPARATE companion GGUF (see
//! [`super::deepseek4_mtp`]) — GLM-5.2 carries its draft head IN-BAND: `nextn_predict_layers`
//! extra blocks trail the main blocks inside `block_count`. The main model drops them
//! (`quantized_deepseek2::PropsGGUF`); this module loads the (depth-1) head at
//! `blk.{block_count}` on demand for self-speculative decoding.
//!
//! Structurally the head is ONE deepseek2 decoder block (`blk.{N}.*`: MLA attention +
//! bias-routed MoE — the SAME [`LayerWeights::load`] the base model uses) wrapped with a
//! NextN entry/exit that predicts token `t+1` from the base hidden at `t` plus the
//! embedding of token `t` (DeepSeek-V3 chain; colibrì `glm.c` `mtp_draft`, README line 29):
//!
//! ```text
//!   x       = eh_proj( [ enorm(embed(token_t)) ; hnorm(hidden_t) ] )   // [b,s,e]
//!   x       = block(x)                                                 // one deepseek2 block
//!   logits  = base_output( shared_head_norm(x) )                       // [b,s,vocab]
//! ```
//!
//! `eh_proj` is a SINGLE `[2e,e]` concat projection (GLM's form; DeepSeek-V4 instead sums
//! two `[e,e]` projections — equivalent). The output head + token embeddings are SHARED
//! with the base model. The head must be **int8** (colibrì [#8]): at int4 draft acceptance
//! collapses to 0–4% and speculation never engages (vs 39–59% at int8) — [`enforce_int8_head`]
//! rejects a sub-int8 `eh_proj` at load rather than attaching a dead draft head.
//!
//! [#8]: https://github.com/JustVugg/colibri/issues/8

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::sync::Arc;

use crate::gguf::Content;
use crate::layers::{
    CausalMaskConfig, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
    DeepSeekV2RotaryEmbedding, QRmsNorm, ScaledRopeType,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::gguf_moe::gguf_linear;
use crate::pipeline::KvCache;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::QuantMethod;

use crate::speculative::{
    MtpConfig, SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
    SpeculativeProposer, TargetTokenEmbedder,
};

use super::quantized_deepseek2::{LayerWeights, ModelWeights, PropsGGUF};

/// colibrì [#8]: the GLM-5.2 MTP head MUST be int8. At int4 the draft-acceptance rate
/// collapses to 0–4% and speculation never engages (vs 39–59% at int8). Enforced at load:
/// reject an `eh_proj` weight quantized below 8 effective bits/weight — naming the offending
/// dtype — rather than silently attaching a draft head that never proposes.
///
/// [#8]: https://github.com/JustVugg/colibri/issues/8
fn enforce_int8_head(w: &hanzo_ml::quantized::QTensor) -> Result<()> {
    let dt = w.dtype();
    // Effective bits/weight = block byte-size × 8 ÷ elements-per-block. Q8_0 ≈ 8.5, Q4_K ≈ 4.5.
    let bits = (dt.type_size() * 8) as f64 / dt.block_size() as f64;
    if bits < 8.0 {
        hanzo_ml::bail!(
            "GLM-5.2 MTP head `eh_proj` is {dt:?} (~{bits:.1} bit/weight); colibrì #8 requires an \
             int8 head (>= 8 bit/weight) — at int4 draft acceptance collapses to 0–4% and \
             speculation never engages. Re-convert the nextn head at int8 (the converter default)."
        );
    }
    Ok(())
}

/// The GLM-5.2 in-band `nextn` draft head: NextN entry/exit around one deepseek2 block.
/// Shares the base model's token embeddings + output projection (passed to [`Self::draft`]).
pub struct GlmMtpHead {
    enorm: QRmsNorm,
    hnorm: QRmsNorm,
    eh_proj: Arc<dyn QuantMethod>,
    block: LayerWeights,
    norm: QRmsNorm,
    device: Device,
    dtype: DType,
    /// Single-block draft KV cache, a 1-element Vec so it satisfies `PastKvLenCache`.
    cache: Vec<KvCache>,
}

impl GlmMtpHead {
    /// Load the in-band `nextn` head from the (re-opened) base GGUF `ct`. `props` is the
    /// base model's config (the head shares all deepseek2 hyperparams); the head lives at
    /// `blk.{props.block_count}` (main blocks already excluded).
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        props: &PropsGGUF,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if props.nextn_predict_layers == 0 {
            hanzo_ml::bail!(
                "this deepseek2/glm-dsa GGUF has no in-band `nextn` MTP head (nextn_predict_layers = 0)"
            );
        }
        // Depth-1 MTP: the first (and, for GLM-5.2, only) nextn block sits at blk.{block_count}.
        let nextn_idx = props.block_count;
        let prefix = format!("blk.{nextn_idx}");
        let eps = props.rms_norm_eps;

        let q_head_dim = props.qk_nope_head_dim + props.qk_rope_head_dim;
        // softmax_scale = 1/sqrt(q_head_dim), YaRN-mscaled to match the base layers (see from_gguf).
        let mut softmax_scale = 1.0 / (q_head_dim as f32).sqrt();
        if let (Some(factor), Some(_orig)) = (props.rope_scaling_factor, props.rope_yarn_orig_ctx) {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, 1.0);
            softmax_scale = softmax_scale * mscale * mscale;
        }

        // Same RoPE config as the base layers (θ=rope_freq_base, optional YaRN).
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: props.rope_scaling_factor.and_then(|factor| {
                props
                    .rope_yarn_orig_ctx
                    .map(|orig| DeepSeekV2RopeScaling::Yarn {
                        original_max_position_embeddings: orig,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                        factor,
                        mscale: 1.0,
                        mscale_all_dim: 1.0,
                        scaling_type: ScaledRopeType::Yarn,
                    })
            }),
            max_position_embeddings: props.max_seq_len,
            rope_theta: props.rope_freq_base,
            qk_rope_head_dim: props.qk_rope_head_dim,
        };
        let rotary = Arc::new(DeepSeekV2RotaryEmbedding::new(&rope_cfg, DType::F32, device)?);

        // The nextn block is a full deepseek2 decoder block — SAME loader as the main model,
        // Eager (no paged attn: the draft runs over its own small KvCache).
        let block = LayerWeights::load(
            ct, nextn_idx, props, device, rotary, softmax_scale, q_head_dim, None, dtype,
        )?;

        // NextN entry/exit. enorm/hnorm normalize the token embedding / base hidden; eh_proj is
        // the single [2e -> e] concat projection; shared_head_norm is the MTP final norm (logits
        // share the base output head). See colibrì glm.c mtp_draft (lines ~1583+).
        let enorm = QRmsNorm::new_dtype(
            ct.tensor(&format!("{prefix}.nextn.enorm.weight"), device)?,
            eps,
            dtype,
        )?;
        let hnorm = QRmsNorm::new_dtype(
            ct.tensor(&format!("{prefix}.nextn.hnorm.weight"), device)?,
            eps,
            dtype,
        )?;
        let eh_q = ct.tensor(&format!("{prefix}.nextn.eh_proj.weight"), device)?;
        enforce_int8_head(&eh_q)?;
        let eh_proj = gguf_linear(eh_q)?;
        let norm = QRmsNorm::new_dtype(
            ct.tensor(&format!("{prefix}.nextn.shared_head_norm.weight"), device)?,
            eps,
            dtype,
        )?;

        Ok(Self {
            enorm,
            hnorm,
            eh_proj,
            block,
            norm,
            device: device.clone(),
            dtype,
            cache: vec![KvCache::new_normal(2, props.max_seq_len, 512)],
        })
    }

    /// Draft the next-token logits from the base model's post-norm `hidden` `[b,s,e]` and the
    /// corresponding `token_ids` `[b,s]`, sharing the base `embed` + `output` head.
    /// `start_offsets` are the per-sequence positions (for RoPE/causality).
    ///
    /// Returns `(logits [b,s,vocab], hidden [b,s,e])`. The returned post-norm hidden is the
    /// EAGLE chain feature: feed it back (with the drafted token) as `hidden` for the next
    /// chained step, extending the draft beyond depth-1 off the single head.
    pub fn draft(
        &mut self,
        hidden: &Tensor,
        token_ids: &Tensor,
        embed: &Embedding,
        output: &Arc<dyn QuantMethod>,
        start_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // NextN entry: e = enorm(embed(tok)), h = hnorm(hidden); x = eh_proj([e ; h]).
        let e = self
            .enorm
            .forward(&embed.forward(token_ids)?.to_dtype(self.dtype)?)?;
        let h = self.hnorm.forward(&hidden.to_dtype(self.dtype)?)?;
        let cat = Tensor::cat(&[&e, &h], D::Minus1)?.contiguous()?; // [b, s, 2e]
        let x = self.eh_proj.forward(&cat)?; // [b, s, e]

        let mask = CausalMasker.make_causal_mask(
            token_ids,
            &self.cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;

        // One deepseek2 decoder block over the fused carrier (Eager over the draft KvCache).
        let x = self
            .block
            .forward_block(x, &mask, start_offsets, &mut self.cache[0], None)?;
        let x = self.norm.forward(&x)?;
        let logits = output.forward(&x.contiguous()?)?;
        Ok((logits, x))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Drop the draft block's KV history. The proposer resets before each draft so the head
    /// attends only to the current token over the target hidden (which already encodes the
    /// full context) — a stateless draft. Correctness comes from the target verify, so this
    /// only trades accept rate; it never affects the emitted stream.
    pub fn reset_cache(&mut self) {
        for c in self.cache.iter_mut() {
            c.reset();
        }
    }
}

/// The `SpeculativeProposer` adapter that plugs the GLM-5.2 in-band `nextn` head into the
/// generic speculative driver. Holds the head plus clones of the base model's token
/// embeddings + output projection (the head shares both). Mirrors
/// [`super::deepseek4_mtp::Deepseek4MtpRuntime`]: correctness comes from the target verify,
/// so the draft is a stateless single-token step chained EAGLE-style.
pub struct Deepseek2MtpRuntime {
    mtp: GlmMtpHead,
    embed: Embedding,
    output: Arc<dyn QuantMethod>,
    /// Maximum chained-draft depth; the chain stops early when the drafted token's
    /// confidence drops below `confidence_threshold`.
    n_predict: usize,
    /// Adaptive draft-length gate: extend the EAGLE chain only while the just-drafted
    /// token's softmax top-probability stays `>= confidence_threshold` on every active row.
    /// `0.0` never gates (fixed depth `n_predict`, byte-identical to before). Correctness-
    /// neutral — the target verify still decides every emitted token.
    confidence_threshold: f32,
}

impl Deepseek2MtpRuntime {
    pub fn new(
        mtp: GlmMtpHead,
        embed: Embedding,
        output: Arc<dyn QuantMethod>,
        n_predict: usize,
    ) -> Self {
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

/// GLM-5.2 is self-speculative: its `nextn` MTP head is IN-BAND (same GGUF the pipeline
/// already opened), sharing the base model's token embeddings + output projection. This impl
/// is the ONE place that knowledge lives — the pipeline asks for the capability and never
/// names GLM. (DeepSeek-V4 implements the same trait against a companion GGUF.)
impl crate::speculative::SelfSpeculative for ModelWeights {
    fn attach_mtp(
        &self,
        cfg: &MtpConfig,
    ) -> Result<Box<dyn SpeculativeProposer + Send + Sync>> {
        if self.base_props().nextn_predict_layers == 0 {
            hanzo_ml::bail!(
                "this model has no in-band `nextn` MTP head for self-speculative decoding"
            );
        }
        // In-band head: re-open the SAME base GGUF `cfg` points at and read `blk.{main}.nextn.*`.
        let path = cfg.resolve_path()?;
        let mut readers = [std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)?];
        let mut readers_ref: Vec<&mut std::fs::File> = readers.iter_mut().collect();
        let mut ct = crate::gguf::Content::from_readers(&mut readers_ref)?;
        let head = GlmMtpHead::load(
            &mut ct,
            self.base_props(),
            &self.device,
            self.compute_dtype(),
        )?;
        let runtime = Deepseek2MtpRuntime::new(
            head,
            self.embeddings()?,
            self.output_head()?,
            cfg.n_predict.unwrap_or(1),
        );
        // The proposer needs the pre-output hidden every step; enable the forward stash.
        self.set_store_spec_hidden(true);
        Ok(Box::new(runtime))
    }
}

impl SpeculativeProposer for Deepseek2MtpRuntime {
    fn proposal_len(&self) -> usize {
        self.n_predict
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let hiddens = ctx.target_hiddens.ok_or_else(|| {
            hanzo_ml::Error::Msg("GLM MTP requires the target hidden state for proposal".into())
        })?;
        let batch = ctx.sampled_tokens.len();
        if batch == 0 {
            return Ok(SpeculativeProposalBatch::new(Vec::new()));
        }
        let device = self.mtp.device().clone();

        // EAGLE-style chained draft off the single in-band head: step 0 drafts from the target
        // hidden + anchor; each next step feeds the head's OWN post-norm hidden + the token it
        // just drafted, extending the draft to `n_predict` tokens. The KV cache accumulates the
        // chain (reset once per proposal); the target verify decides acceptance, so draft
        // quality only affects accept rate, never correctness.
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
                    "GLM MTP draft produced {} tokens for {batch} rows",
                    draft_ids.len()
                );
            }
            cur_tokens = draft_ids.clone();
            hidden = next_hidden;
            step_tokens.push(draft_ids);
            step_logits.push(logits.clone());

            // Adaptive draft length: stop once the WEAKEST active row's draft confidence drops
            // below the gate. A batch-uniform length keeps the staged proposals homogeneous (one
            // target verify shape). `0.0` never triggers (min top-prob is always >= 0).
            if self.confidence_threshold > 0.0 && k + 1 < self.n_predict {
                let probs = hanzo_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?; // [batch,1,vocab]
                let top = probs.max(D::Minus1)?; // [batch,1]
                let top_probs: Vec<f32> = top.flatten_all()?.to_vec1::<f32>()?;
                let min_conf = top_probs.iter().cloned().fold(f32::INFINITY, f32::min);
                if min_conf < self.confidence_threshold {
                    break;
                }
            }
        }

        // Per row: the drafted tokens + their logits stacked as [1, n_predict, vocab]
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
