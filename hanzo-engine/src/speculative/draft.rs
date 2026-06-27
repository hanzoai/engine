use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Device, Result, Tensor};
use rand_isaac::Isaac64Rng;
use tokio::sync::Mutex as AsyncMutex;

use crate::pipeline::text_models_inputs_processor::{FlashParams, ModelInputs};
use crate::pipeline::{EitherCache, ForwardInputsResult, NormalCache, Pipeline};
use crate::sequence::Sequence;

use super::proposer::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    TargetTokenEmbedder,
};

/// Loaded draft pipeline shared from the builder into the target's `attach_speculative`.
pub type DraftPipeline = Arc<AsyncMutex<dyn Pipeline + Send + Sync>>;

/// Classic draft+target speculative decoding proposer (Leviathan et al. / Chen et al.).
///
/// Runs a standalone small draft model `gamma` autoregressive steps per target verification step.
/// The draft keeps its own per-sequence KV cache, reconciled against the accepted context before
/// each round: rollback on rejection, short prefill when the target accepted everything. The draft
/// is untrusted; the verifier corrects every token against the target distribution, so draft cache
/// drift can only lower acceptance, never break exactness.
pub struct DraftModelProposer {
    draft: DraftPipeline,
    gamma: usize,
    device: Device,
    max_seq_len: usize,
    empty_cache: NormalCache,
    caches: HashMap<usize, NormalCache>,
}

impl DraftModelProposer {
    pub fn new(draft: DraftPipeline, gamma: usize) -> Result<Self> {
        if gamma == 0 {
            hanzo_ml::bail!("draft-model speculative decoding requires gamma >= 1");
        }
        let guard = draft
            .try_lock()
            .map_err(|_| hanzo_ml::Error::msg("draft pipeline is not exclusively owned"))?;
        let device = guard.device();
        let max_seq_len = guard.get_metadata().max_seq_len;
        let EitherCache::Normal(_) = guard.cache() else {
            hanzo_ml::bail!("draft model must use a normal KV cache for speculative decoding");
        };
        let empty_cache = guard.cache().normal().clone();
        drop(guard);
        Ok(Self {
            draft,
            gamma,
            device,
            max_seq_len,
            empty_cache,
            caches: HashMap::new(),
        })
    }

    fn propose_one(
        &mut self,
        draft: &mut (dyn Pipeline + Send + Sync),
        seq: &Sequence,
        seq_id: usize,
        base_len: usize,
        anchor: u32,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<SpeculativeProposal> {
        let toks = seq.get_toks();
        debug_assert_eq!(toks.len(), base_len + 1);
        debug_assert_eq!(toks[base_len], anchor);

        // No room for the round in the draft cache: fall back to plain target decode this step.
        if base_len + self.gamma + 1 > self.max_seq_len {
            self.caches.remove(&seq_id);
            return Ok(SpeculativeProposal::new(Vec::new()));
        }

        let seq_cache = self
            .caches
            .remove(&seq_id)
            .unwrap_or_else(|| self.empty_cache.clone());
        install_cache(draft, seq_cache);

        // Reconcile the draft cache to exactly the accepted prefix [0, base_len).
        let have = cache_len(draft);
        if have > base_len {
            rollback_cache(draft, base_len)?;
        } else if have < base_len {
            self.forward(draft, &toks[have..base_len], have)?;
        }

        let sampler = seq.sampler();
        let mut context = toks.to_vec();
        let mut tokens = Vec::with_capacity(self.gamma);
        let mut logit_rows = Vec::with_capacity(self.gamma);
        let mut next_input = anchor;
        for step in 0..self.gamma {
            let logits = self.forward(draft, &[next_input], base_len + step)?;
            let row = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let sampled =
                sampler.sample(row.clone(), &context, false, rng.clone(), false, false)?;
            context.push(sampled.token);
            tokens.push(sampled.token);
            logit_rows.push(row.unsqueeze(0)?);
            next_input = sampled.token;
        }

        let seq_cache = uninstall_cache(draft, self.empty_cache.clone());
        self.caches.insert(seq_id, seq_cache);

        let logits = Tensor::cat(&logit_rows, 0)?;
        Ok(SpeculativeProposal::with_logits(tokens, logits))
    }

    fn forward(
        &self,
        draft: &mut (dyn Pipeline + Send + Sync),
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Tensor> {
        let seq_len = tokens.len();
        let input_ids = Tensor::from_vec(tokens.to_vec(), (1, seq_len), &self.device)?;
        let inputs = ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets: vec![start_pos],
            seqlen_offsets_full: None,
            context_lens: vec![(seq_len - 1, 1)],
            position_ids: vec![start_pos + seq_len],
            paged_attn_meta: None,
            flash_meta: FlashParams::empty(true),
            flash_meta_full: None,
        };
        match draft.forward_inputs(Box::new(inputs), false)? {
            ForwardInputsResult::CausalGeneration { logits } => Ok(logits),
            _ => hanzo_ml::bail!("draft model forward did not produce causal-generation logits"),
        }
    }
}

impl SpeculativeProposer for DraftModelProposer {
    fn proposal_len(&self) -> usize {
        self.gamma
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        static ACTIVE_ONCE: std::sync::Once = std::sync::Once::new();
        ACTIVE_ONCE.call_once(|| {
            eprintln!(
                "[hanzo] draft-model speculative proposer ACTIVE (gamma={})",
                self.gamma
            );
        });

        let batch = ctx.sampled_tokens.len();
        let draft_arc = self.draft.clone();
        let mut draft = draft_arc
            .try_lock()
            .map_err(|_| hanzo_ml::Error::msg("draft pipeline is not exclusively owned"))?;
        let mut proposals = Vec::with_capacity(batch);
        for i in 0..batch {
            proposals.push(self.propose_one(
                &mut *draft,
                ctx.sequences[i],
                ctx.seq_ids[i],
                ctx.base_lens[i],
                ctx.sampled_tokens[i],
                ctx.rng.clone(),
            )?);
        }
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}

fn cache_len(draft: &(dyn Pipeline + Send + Sync)) -> usize {
    draft
        .cache()
        .normal()
        .0
        .first()
        .map(|c| c.current_seq_len())
        .unwrap_or(0)
}

fn install_cache(draft: &(dyn Pipeline + Send + Sync), cache: NormalCache) {
    *draft.cache().normal() = cache;
}

fn uninstall_cache(draft: &(dyn Pipeline + Send + Sync), empty: NormalCache) -> NormalCache {
    std::mem::replace(&mut *draft.cache().normal(), empty)
}

fn rollback_cache(draft: &(dyn Pipeline + Send + Sync), keep_len: usize) -> Result<()> {
    for layer in &mut draft.cache().normal().0 {
        layer.set_len(keep_len)?;
    }
    Ok(())
}
