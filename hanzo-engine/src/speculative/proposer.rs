use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Result, Tensor, D};
use rand_isaac::Isaac64Rng;

use crate::pipeline::text_models_inputs_processor::PagedAttentionMeta;
use crate::sequence::Sequence;

pub type TargetTokenEmbedder<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

pub enum SpeculativeKvCache<'a> {
    Paged {
        metadata: &'a PagedAttentionMeta,
        kv_cache: &'a [(Tensor, Tensor)],
    },
    /// Normal (non-paged) KV cache. Self-speculative proposers that keep their own
    /// draft KV (e.g. DeepSeek-V4 MTP) don't read the target tensors — the confirmed
    /// tokens arrive via `sampled_tokens`/`sequences` — so this variant carries none.
    Normal,
}

pub struct SpeculativeProposeBatchCtx<'a> {
    pub sampled_tokens: &'a [u32],
    pub sampled_tokens_emitted: bool,
    pub seq_ids: &'a [usize],
    pub base_lens: &'a [usize],
    pub sequences: &'a [&'a Sequence],
    pub cache: SpeculativeKvCache<'a>,
    pub target_hiddens: Option<Tensor>,
    /// Multi-layer target hidden prefix for DSpark-style parallel-block proposers: one
    /// tensor per fused target decoder layer, each `[prefix_len, hidden]`. `None` for
    /// single-hidden proposers (V4 MTP) and standalone draft models — additive, so
    /// existing proposers are unaffected.
    pub target_hidden_layers: Option<super::HiddenWindow>,
    pub rng: Arc<Mutex<Isaac64Rng>>,
}

/// Each sequence's drafting context: its tokens, then the anchor unless it is already emitted.
/// The verifier scores draft i against the same context grown by drafts 0..i.
pub(crate) fn draft_contexts(ctx: &SpeculativeProposeBatchCtx<'_>) -> Vec<Vec<u32>> {
    ctx.sequences
        .iter()
        .zip(ctx.sampled_tokens)
        .map(|(seq, &anchor)| {
            let mut context = seq.get_toks().to_vec();
            if !ctx.sampled_tokens_emitted {
                context.push(anchor);
            }
            context
        })
        .collect()
}

/// One draft step's tokens from its logits (`[batch, .., vocab]`), each drawn by its sequence's own
/// sampler over `contexts[row]`, which then grows by the draft. The verifier accepts a draft with
/// probability min(1, p/q) where q is that same sampler's distribution, which is exact only when
/// the draft was drawn from q. Greedy sequences are verified by matching, so an all-greedy batch
/// takes one device argmax. `rng` is the engine's shared RNG; a seeded sequence draws from its own.
pub(crate) fn sample_drafts(
    logits: &Tensor,
    sequences: &[&Sequence],
    contexts: &mut [Vec<u32>],
    rng: &Arc<Mutex<Isaac64Rng>>,
) -> Result<Vec<u32>> {
    let batch = sequences.len();
    if contexts.len() != batch {
        hanzo_ml::bail!(
            "draft sampling context batch mismatch: contexts={}, sequences={batch}",
            contexts.len()
        );
    }
    let rows = logits.reshape((batch, ()))?;
    let tokens = if sequences.iter().all(|seq| seq.sampler().is_argmax()) {
        rows.argmax(D::Minus1)?.to_dtype(DType::U32)?.to_vec1()?
    } else {
        let mut tokens = Vec::with_capacity(batch);
        for (row, seq) in sequences.iter().enumerate() {
            let row_logits = rows.get(row)?.to_dtype(DType::F32)?;
            let sampled = seq.sampler().sample(
                row_logits,
                &contexts[row],
                false,
                seq.rng(rng),
                false,
                batch > 1,
            )?;
            tokens.push(sampled.token);
        }
        tokens
    };
    for (context, &token) in contexts.iter_mut().zip(&tokens) {
        context.push(token);
    }
    Ok(tokens)
}

#[derive(Clone, Debug)]
pub struct SpeculativeProposal {
    pub tokens: Vec<u32>,
    pub logits: Option<Tensor>,
}

impl SpeculativeProposal {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            logits: None,
        }
    }

    pub fn with_logits(tokens: Vec<u32>, logits: Tensor) -> Self {
        Self {
            tokens,
            logits: Some(logits),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct SpeculativeProposalBatch {
    pub proposals: Vec<SpeculativeProposal>,
}

impl SpeculativeProposalBatch {
    pub fn new(proposals: Vec<SpeculativeProposal>) -> Self {
        Self { proposals }
    }
}

pub trait SpeculativeProposer {
    fn proposal_len(&self) -> usize;

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch>;

    /// Drop per-sequence proposer state for sequences no longer live. `live` is the set of
    /// still-running sequence ids; anything keyed off a finished sequence must be released so a
    /// long-running server doesn't leak. Default no-op: proposers that hold no per-seq state
    /// (e.g. MTP) need not implement it.
    fn retain_seqs(&mut self, _live: &[usize]) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::test_sequence;
    use hanzo_ml::Device;
    use rand::SeedableRng;

    fn rng() -> Arc<Mutex<Isaac64Rng>> {
        Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(7)))
    }

    #[test]
    fn greedy_batch_drafts_the_argmax_and_grows_contexts() -> Result<()> {
        let (a, b) = (
            test_sequence(vec![1, 2], None),
            test_sequence(vec![3], None),
        );
        let seqs = [&a, &b];
        let logits = Tensor::new(&[[[0f32, 3.0, 1.0]], [[2.0, 0.0, 1.0]]], &Device::Cpu)?;
        let mut contexts = vec![vec![1, 2], vec![3]];
        let drafts = sample_drafts(&logits, &seqs, &mut contexts, &rng())?;
        assert_eq!(drafts, vec![1, 0]);
        assert_eq!(contexts, vec![vec![1, 2, 1], vec![3, 0]]);
        Ok(())
    }

    // A seeded sequence drafts from its own RNG, so its drafts replay whatever the shared RNG did.
    #[test]
    fn a_seeded_sequence_drafts_from_its_own_rng() -> Result<()> {
        let logits = Tensor::new(&[[[0f32, 1.0, 2.0, 0.5]]], &Device::Cpu)?;
        let drafts = |seed: Option<u64>, shared: u64| -> Result<Vec<u32>> {
            let mut seq = test_sequence(vec![5], Some(1.0));
            if let Some(seed) = seed {
                seq.set_seed(seed);
            }
            let shared = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(shared)));
            (0..32)
                .map(|_| {
                    let mut contexts = vec![vec![5]];
                    Ok(sample_drafts(&logits, &[&seq], &mut contexts, &shared)?[0])
                })
                .collect()
        };
        assert_eq!(drafts(Some(11), 1)?, drafts(Some(11), 2)?);
        assert_ne!(drafts(None, 1)?, drafts(None, 2)?);
        Ok(())
    }

    // The verifier accepts draft x with probability min(1, p(x)/q(x)) and otherwise samples the
    // residual max(0, p - q); that emits exactly p only if x ~ q. So at T > 0 drafts must follow the
    // sequence's own candidate distribution, not its argmax.
    #[test]
    fn sampled_drafts_follow_the_candidate_distribution() -> Result<()> {
        let seq = test_sequence(vec![5], Some(1.0));
        let row = [0f32, 1.0, 2.0, 0.5];
        let q = seq
            .sampler()
            .speculative_candidate_probs(Tensor::new(&row, &Device::Cpu)?, &[5])?;
        let logits = Tensor::new(&[[row]], &Device::Cpu)?;
        let rng = rng();
        let draws = 40_000usize;
        let mut counts = [0usize; 4];
        for _ in 0..draws {
            let mut contexts = vec![vec![5]];
            counts[sample_drafts(&logits, &[&seq], &mut contexts, &rng)?[0] as usize] += 1;
        }
        for (token, (&count, &want)) in counts.iter().zip(&q).enumerate() {
            let got = count as f32 / draws as f32;
            // 4 standard errors of a binomial frequency at 40k draws is under 0.01.
            assert!(
                (got - want).abs() < 0.01,
                "token {token}: drafted {got}, q {want}"
            );
        }
        Ok(())
    }
}
