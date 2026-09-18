//! The speculative draft Qwen3.5 runs off its own multi-token-prediction head.
//!
//! A row fed token `t + 1` over the target's final-norm hidden state at `t` predicts token
//! `t + 2`. Chaining that — the head's own hidden state and its argmax at the next position —
//! drafts `n_predict` tokens per target step, decoded through the target's `lm_head` so every
//! draft lands on the verifier's own scale.
//!
//! The head itself is [`MtpStep`]: one step, and the chain. Qwen3.5 ships one in its safetensors
//! checkpoint (`mtp.*`) and one in its GGUF (`blk.N.nextn.*`); both drive this proposer, which is
//! where the chain, the KV and the anchor positions live.

use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Device, Result, Tensor, D};

use crate::kv_cache::KvCache;
use crate::speculative::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    SpeculativeSharedHeads, TargetTokenEmbedder,
};

/// Drafts per target step. A wider model amortizes the verify forward over more drafts, so it
/// drafts deeper; past that, acceptance falls faster than the saved forwards pay for.
const DEFAULT_N_PREDICT: usize = 2;
const DEEP_N_PREDICT: usize = 3;
const DEEP_HIDDEN_SIZE: usize = 4096;

/// The three MRoPE planes of one position.
pub type Mrope = [u32; 3];

/// The MRoPE positions of the target rows the next draft starts from, one per sequence. The
/// model fills it when it selects those rows; the proposer takes it when it drafts.
pub type AnchorPositions = Arc<Mutex<Option<Vec<Mrope>>>>;

/// How deep to draft when the caller names no depth.
pub(crate) fn default_n_predict(hidden_size: usize) -> usize {
    if hidden_size >= DEEP_HIDDEN_SIZE {
        DEEP_N_PREDICT
    } else {
        DEFAULT_N_PREDICT
    }
}

/// The head's KV, holding one row per chained draft and nothing else. It is the head's own, so
/// the head takes no slot in the target's paged cache and the target's cache stays sized by the
/// model's own layers; a paged head would instead take the slot after them, and this is the only
/// place that decides.
pub(crate) fn chain_cache(max_draft: usize) -> KvCache {
    KvCache::new_normal(2, max_draft, max_draft)
}

/// One multi-token-prediction head, as the chain drives it.
pub trait MtpStep {
    /// One step over `[batch, 1, hidden]` inputs at `[3, batch, 1]` MRoPE positions. Returns the
    /// normed hidden state, which is both the `lm_head` input and the next step's.
    fn step(
        &mut self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor>;

    /// Drop the chain's KV, so the next chain attends only to itself.
    fn reset(&mut self);

    fn device(&self) -> &Device;

    fn dtype(&self) -> DType;
}

/// Drives an [`MtpStep`] head as the target's own speculative draft.
pub struct Qwen3_5MtpProposer {
    head: Box<dyn MtpStep + Send + Sync>,
    heads: SpeculativeSharedHeads,
    n_predict: usize,
    anchors: AnchorPositions,
}

impl Qwen3_5MtpProposer {
    pub(crate) fn new(
        head: Box<dyn MtpStep + Send + Sync>,
        heads: SpeculativeSharedHeads,
        n_predict: usize,
        anchors: AnchorPositions,
    ) -> Self {
        Self {
            head,
            heads,
            n_predict,
            anchors,
        }
    }
}

impl SpeculativeProposer for Qwen3_5MtpProposer {
    fn proposal_len(&self) -> usize {
        self.n_predict
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let batch = ctx.sampled_tokens.len();
        if batch == 0 {
            return Ok(SpeculativeProposalBatch::new(Vec::new()));
        }
        let hidden = ctx.target_hiddens.ok_or_else(|| {
            hanzo_ml::Error::msg("Qwen3.5 MTP needs the target hidden state to draft")
        })?;
        let mut mrope = self
            .anchors
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .ok_or_else(|| {
                hanzo_ml::Error::msg("Qwen3.5 MTP needs the target row positions to draft")
            })?;
        if mrope.len() != batch || hidden.dim(0)? != batch {
            hanzo_ml::bail!(
                "Qwen3.5 MTP batch mismatch: {batch} anchors sampled, {} positions, {} hidden rows",
                mrope.len(),
                hidden.dim(0)?
            );
        }

        let device = self.head.device().clone();
        let mut hidden = match hidden.dims() {
            [_, _, _] => hidden.to_device(&device)?.to_dtype(self.head.dtype())?,
            [_, _] => hidden
                .unsqueeze(1)?
                .to_device(&device)?
                .to_dtype(self.head.dtype())?,
            other => hanzo_ml::bail!("Qwen3.5 MTP target hidden has shape {other:?}"),
        };
        let mut tokens = ctx.sampled_tokens.to_vec();
        self.head.reset();

        let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(self.n_predict);
        let mut logits: Vec<Tensor> = Vec::with_capacity(self.n_predict);
        for step in 0..self.n_predict {
            let ids = Tensor::from_vec(tokens.clone(), (batch, 1), &device)?;
            let embeds = (self.heads.embed)(&ids)?.to_dtype(self.head.dtype())?;
            let mut planes = Vec::with_capacity(3 * batch);
            for plane in 0..3 {
                planes.extend(mrope.iter().map(|pos| pos[plane]));
            }
            let positions = Tensor::from_vec(planes, (3, batch, 1), &device)?;
            let normed = self.head.step(&embeds, &hidden, &positions)?;
            let step_logits = (self.heads.lm_head)(&normed)?;
            let step_drafts: Vec<u32> = step_logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            if step_drafts.len() != batch {
                hanzo_ml::bail!(
                    "Qwen3.5 MTP drafted {} tokens for {batch} sequences",
                    step_drafts.len()
                );
            }
            logits.push(step_logits);
            if step + 1 < self.n_predict {
                // The draft becomes the next step's input, one position on, off the head's own
                // hidden state.
                tokens.clone_from(&step_drafts);
                hidden = normed;
                for pos in mrope.iter_mut() {
                    for plane in pos.iter_mut() {
                        *plane += 1;
                    }
                }
            }
            drafts.push(step_drafts);
        }

        // Per sequence: its drafted tokens and their logits as `[1, n_predict, vocab]`, the rows
        // the verifier indexes by draft position.
        let mut proposals = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_logits = logits
                .iter()
                .map(|step| step.narrow(0, row, 1))
                .collect::<Result<Vec<_>>>()?;
            proposals.push(SpeculativeProposal::with_logits(
                drafts.iter().map(|step| step[row]).collect(),
                Tensor::cat(&row_logits, 1)?,
            ));
        }
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}

/// The tiny weights both heads' tests are built from.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::Mrope;
    use crate::speculative::SpeculativeSharedHeads;
    use hanzo_ml::{Device, Result, Tensor};
    use std::sync::Arc;

    /// Small, distinct, reproducible weights: enough spread that a zero output means a broken
    /// forward rather than a symmetric one.
    pub(crate) fn synthetic(
        rows: usize,
        cols: usize,
        seed: u32,
        device: &Device,
    ) -> Result<Tensor> {
        let count = u32::try_from(rows * cols).expect("test tensors are small");
        let data: Vec<f32> = (0..count)
            .map(|i| {
                let x = i.wrapping_mul(2654435761).wrapping_add(seed);
                let bounded = u16::try_from((x >> 8) % 1000).expect("bounded by 1000");
                f32::from(bounded) / 4000.0 - 0.125
            })
            .collect();
        Tensor::from_vec(data, (rows, cols), device)
    }

    /// Token embeddings and an output head, as the target lends them.
    pub(crate) fn shared_heads(
        vocab: usize,
        hidden: usize,
        device: &Device,
    ) -> Result<SpeculativeSharedHeads> {
        let embed = synthetic(vocab, hidden, 21, device)?;
        let out = synthetic(vocab, hidden, 22, device)?;
        Ok(SpeculativeSharedHeads {
            embed: Arc::new(move |ids: &Tensor| {
                let (batch, seq) = ids.dims2()?;
                embed
                    .index_select(&ids.flatten_all()?, 0)?
                    .reshape((batch, seq, hidden))
            }),
            lm_head: Arc::new(move |hidden: &Tensor| hidden.broadcast_matmul(&out.t()?)),
        })
    }

    /// `[3, rows, 1]` MRoPE position ids, one anchor per row.
    pub(crate) fn positions(anchors: &[Mrope], device: &Device) -> Result<Tensor> {
        let mut planes = Vec::with_capacity(3 * anchors.len());
        for plane in 0..3 {
            planes.extend(anchors.iter().map(|pos| pos[plane]));
        }
        Tensor::from_vec(planes, (3, anchors.len(), 1), device)
    }
}
