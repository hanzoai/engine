//! Native MTP (multi-token prediction) speculative proposer for the quantized (GGUF) pipeline.
//!
//! Mirrors `vision_models::gemma4::mtp::Gemma4MtpRuntime`, but the nextn weights live on the
//! quantized `ModelWeights` itself (loaded by `models::quantized_qwen3_5_moe`), so the proposer
//! drives the model's `mtp_step` rather than owning a separate assistant model. Each step embeds
//! the last token, runs the nextn block over the target's donor KV cache (read-only), samples a
//! draft token, and carries the nextn hidden state forward `n_predict` times. The target KV cache
//! is never mutated by the proposer.

use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Result, Tensor, D};
use rand_isaac::Isaac64Rng;

use crate::models::quantized_qwen3_5_moe::ModelWeights as QQwen35;
use crate::sequence::Sequence;

use super::proposer::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, TargetTokenEmbedder,
};

const DEFAULT_N_PREDICT: usize = 1;

pub struct QuantizedMtpRuntime {
    n_predict: usize,
}

impl QuantizedMtpRuntime {
    /// `n_predict` defaults to the number of trailing nextn blocks the model loaded.
    pub fn new(model_n_predict: usize, requested: Option<usize>) -> Self {
        let n_predict = requested
            .filter(|n| *n > 0)
            .unwrap_or(model_n_predict.max(DEFAULT_N_PREDICT));
        Self { n_predict }
    }

    pub fn proposal_len(&self) -> usize {
        self.n_predict
    }

    /// Drive the nextn AR loop against `model`. Mirrors the Gemma4 propose loop.
    pub fn propose(
        &self,
        model: &QQwen35,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: &TargetTokenEmbedder<'_>,
    ) -> Result<SpeculativeProposalBatch> {
        let batch = ctx.sampled_tokens.len();
        if batch == 0 {
            return Ok(SpeculativeProposalBatch::new(Vec::new()));
        }
        // The quantized hybrid pipeline runs one sequence at a time; batched donor reads would need
        // per-row block tables, which this non-paged path does not carry.
        if batch != 1 {
            return Ok(SpeculativeProposalBatch::new(
                (0..batch)
                    .map(|_| SpeculativeProposal::new(Vec::new()))
                    .collect(),
            ));
        }

        let target_hiddens = ctx.target_hiddens.ok_or_else(|| {
            hanzo_ml::Error::Msg(
                "MTP requires target hidden state for speculative proposal.".into(),
            )
        })?;
        if target_hiddens.dim(0)? != batch {
            hanzo_ml::bail!(
                "MTP hidden batch mismatch: hidden={}, sampled={batch}",
                target_hiddens.dim(0)?
            );
        }

        // Donor KV is read straight from the model's own (non-paged) hybrid cache: the last main
        // full-attention layer's K/V over the verified context.
        let (donor_k, donor_v) = model.mtp_donor_kv()?;

        let seq = ctx.sequences[0];
        let base_len = ctx.base_lens[0];
        let mut context = seq.get_toks().to_vec();
        if !ctx.sampled_tokens_emitted {
            context.push(ctx.sampled_tokens[0]);
        }

        let mut last_token =
            Tensor::from_vec(ctx.sampled_tokens.to_vec(), (batch, 1), &model.device)?;
        let mut hidden = target_hiddens;
        let mut tokens = Vec::with_capacity(self.n_predict);
        let mut logits_rows = Vec::with_capacity(self.n_predict);

        for step in 0..self.n_predict {
            let input_embed = target_embedder(&last_token)?;
            // The nextn token at draft step `step` sits at absolute position base_len + step; the
            // donor cache already holds the first base_len keys, so it attends over [0..=base_len+step].
            let position = base_len + step;
            let (draft_logits, next_hidden) =
                model.mtp_step(&input_embed, &hidden, &donor_k, &donor_v, position)?;
            let row_logits = draft_logits.reshape(((), draft_logits.dim(D::Minus1)?))?;
            let sampled = sample_draft_token(&row_logits, seq, &mut context, &ctx.rng)?;
            last_token = Tensor::from_vec(vec![sampled], (batch, 1), &model.device)?;
            tokens.push(sampled);
            logits_rows.push(row_logits.narrow(0, 0, 1)?);
            hidden = next_hidden;
        }

        let logits = Tensor::cat(&logits_rows, 0)?;
        Ok(SpeculativeProposalBatch::new(vec![
            SpeculativeProposal::with_logits(tokens, logits),
        ]))
    }
}

fn sample_draft_token(
    logits: &Tensor,
    seq: &Sequence,
    context: &mut Vec<u32>,
    rng: &Arc<Mutex<Isaac64Rng>>,
) -> Result<u32> {
    let row = logits.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
    let sampled = seq
        .sampler()
        .sample(row, context, false, rng.clone(), false, false)?;
    context.push(sampled.token);
    Ok(sampled.token)
}
