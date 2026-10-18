//! Prompt-lookup (n-gram) speculative proposer.
//!
//! The draft is read straight from the sequence's own token history: the tail
//! n-gram is matched against earlier context, and the tokens that followed the
//! match become the proposal. No draft model and no extra forward — the target
//! verifier corrects every token, so a wrong or absent guess is simply rejected
//! and the output is byte-identical to plain decoding. It excels on grounded,
//! repetitive decoding — code edits, RAG, "repeat this back" — where long spans
//! recur verbatim, and is quantization-agnostic.
//!
//! Reference: apoorvumang/prompt-lookup-decoding; the same primitive ships as
//! Hugging Face `prompt_lookup_num_tokens`.

use hanzo_ml::Result;

use super::proposer::{
    draft_contexts, SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
    SpeculativeProposer, TargetTokenEmbedder,
};

/// Batches at or below this size run in latency mode, where a rejected draft costs
/// little and an accepted one saves a whole decode step. Above it the target forward
/// is already compute-bound and speculation is net-negative (measured), so the
/// proposer stands down and the step is a plain decode.
const LATENCY_BATCH_MAX: usize = 4;

/// The most recent continuation of the tail n-gram of `toks`, up to `gamma` tokens.
///
/// Tries the longest needle first (`ngram_max` down to `ngram_min`): the last `n`
/// tokens are the needle, and the most recent earlier occurrence of that needle in
/// `toks` predicts what comes next. Returns the ≤`gamma` tokens that followed that
/// occurrence (bounded by the end of history). Empty when no needle matches.
fn lookup(toks: &[u32], ngram_min: usize, ngram_max: usize, gamma: usize) -> Vec<u32> {
    let len = toks.len();
    // A needle needs at least one token before it to have an earlier occurrence.
    let longest = ngram_max.min(len.saturating_sub(1));
    for n in (ngram_min..=longest).rev() {
        let needle = &toks[len - n..];
        // Scan candidate starts from just before the tail backwards: the first hit is
        // the most recent earlier occurrence.
        for start in (0..len - n).rev() {
            if &toks[start..start + n] == needle {
                let from = start + n;
                let to = (from + gamma).min(len);
                return toks[from..to].to_vec();
            }
        }
    }
    Vec::new()
}

/// Where a lookup draft comes from: the tail needle of `ngram_max` down to `ngram_min` tokens, and
/// up to `gamma` tokens after its most recent earlier occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lookup {
    pub ngram_min: usize,
    pub ngram_max: usize,
    pub gamma: usize,
}

impl Lookup {
    pub fn new(ngram_min: usize, ngram_max: usize, gamma: usize) -> Result<Self> {
        if ngram_min == 0 {
            hanzo_ml::bail!("prompt-lookup speculative decoding requires ngram_min >= 1");
        }
        if ngram_max < ngram_min {
            hanzo_ml::bail!(
                "prompt-lookup speculative decoding requires ngram_max >= ngram_min (got ngram_max={ngram_max}, ngram_min={ngram_min})"
            );
        }
        if gamma == 0 {
            hanzo_ml::bail!("prompt-lookup speculative decoding requires gamma >= 1");
        }
        Ok(Self {
            ngram_min,
            ngram_max,
            gamma,
        })
    }

    fn draft(&self, toks: &[u32]) -> Vec<u32> {
        lookup(toks, self.ngram_min, self.ngram_max, self.gamma)
    }
}

/// Prompt-lookup / n-gram speculative proposer. See the module docs.
pub struct PromptLookupProposer {
    lookup: Lookup,
}

impl PromptLookupProposer {
    pub fn new(ngram_min: usize, ngram_max: usize, gamma: usize) -> Result<Self> {
        Ok(Self {
            lookup: Lookup::new(ngram_min, ngram_max, gamma)?,
        })
    }
}

impl SpeculativeProposer for PromptLookupProposer {
    fn proposal_len(&self) -> usize {
        self.lookup.gamma
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        // Auto-gate: speculate only in latency mode. At high batch the target forward is
        // compute-bound, so draft nothing and let the driver do a plain decode this step.
        let gated = ctx.sampled_tokens.len() > LATENCY_BATCH_MAX;
        // The just-sampled anchor is already the last token of each sequence, so the tail
        // n-gram is exactly `get_toks()`'s suffix. `logits: None` routes the draft to the
        // verifier's exact-match (lossless) acceptance path.
        let proposals = ctx
            .sequences
            .iter()
            .map(|seq| {
                let tokens = if gated {
                    Vec::new()
                } else {
                    self.lookup.draft(seq.get_toks())
                };
                SpeculativeProposal::new(tokens)
            })
            .collect();
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}

/// Another proposer's draft, continued by prompt lookup.
///
/// The draft runs first. Each sequence's tokens grown by its draft are then looked up, and what
/// followed the match continues the draft by up to `gamma` tokens, so a model's own guess at the
/// next token can open a span copied from the context. A looked-up token has no draft
/// distribution for the verifier to divide by, so only an all-greedy batch continues: greedy
/// verification matches tokens and reads no draft logits. Every sequence continues by the
/// shortest continuation any of them found, so the batch keeps the one width a verify forward
/// takes.
pub struct Extended {
    draft: Box<dyn SpeculativeProposer + Send + Sync>,
    lookup: Lookup,
}

impl Extended {
    pub fn new(draft: Box<dyn SpeculativeProposer + Send + Sync>, lookup: Lookup) -> Self {
        Self { draft, lookup }
    }
}

impl SpeculativeProposer for Extended {
    fn proposal_len(&self) -> usize {
        self.draft.proposal_len() + self.lookup.gamma
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let greedy = ctx.sequences.iter().all(|seq| seq.sampler().is_argmax());
        let contexts = draft_contexts(&ctx);
        let mut batch = self.draft.propose(ctx, target_embedder)?;
        if !greedy || batch.proposals.iter().any(SpeculativeProposal::is_empty) {
            return Ok(batch);
        }
        let found: Vec<Vec<u32>> = contexts
            .into_iter()
            .zip(&batch.proposals)
            .map(|(mut context, proposal)| {
                context.extend_from_slice(&proposal.tokens);
                self.lookup.draft(&context)
            })
            .collect();
        let width = found.iter().map(Vec::len).min().unwrap_or(0);
        if width == 0 {
            return Ok(batch);
        }
        for (proposal, more) in batch.proposals.iter_mut().zip(&found) {
            proposal.tokens.extend_from_slice(&more[..width]);
            proposal.logits = None;
        }
        Ok(batch)
    }

    fn retain_seqs(&mut self, live: &[usize]) {
        self.draft.retain_seqs(live);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::test_sequence;
    use crate::speculative::SpeculativeKvCache;
    use hanzo_ml::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::{Arc, Mutex};

    /// Drafts a fixed token per sequence, with a logit row each.
    struct Fixed(Vec<u32>);

    impl SpeculativeProposer for Fixed {
        fn proposal_len(&self) -> usize {
            1
        }

        fn propose(
            &mut self,
            _ctx: SpeculativeProposeBatchCtx<'_>,
            _target_embedder: Option<&TargetTokenEmbedder<'_>>,
        ) -> Result<SpeculativeProposalBatch> {
            self.0
                .iter()
                .map(|&t| {
                    Ok(SpeculativeProposal::with_logits(
                        vec![t],
                        Tensor::zeros((1, 1, 16), hanzo_ml::DType::F32, &Device::Cpu)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()
                .map(SpeculativeProposalBatch::new)
        }
    }

    fn extend(drafts: Vec<u32>, sequences: &[&crate::sequence::Sequence]) -> Result<Vec<Vec<u32>>> {
        let mut chain = Extended::new(Box::new(Fixed(drafts)), Lookup::new(2, 2, 3)?);
        assert_eq!(chain.proposal_len(), 4);
        let anchors: Vec<u32> = sequences
            .iter()
            .map(|s| *s.get_toks().last().unwrap())
            .collect();
        let seq_ids: Vec<usize> = (0..sequences.len()).collect();
        let base_lens = vec![0; sequences.len()];
        let batch = chain.propose(
            SpeculativeProposeBatchCtx {
                sampled_tokens: &anchors,
                sampled_tokens_emitted: true,
                seq_ids: &seq_ids,
                base_lens: &base_lens,
                sequences,
                cache: SpeculativeKvCache::Normal,
                target_hiddens: None,
                target_hidden_layers: None,
                rng: Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0))),
            },
            None,
        )?;
        for proposal in &batch.proposals {
            // A continued draft verifies by matching; a bare one keeps its logits.
            assert_eq!(proposal.logits.is_some(), proposal.tokens.len() == 1);
        }
        Ok(batch.proposals.into_iter().map(|p| p.tokens).collect())
    }

    /// The draft token completes a needle the context holds, and what followed it there continues
    /// the draft.
    #[test]
    fn a_draft_opens_a_copied_span() -> Result<()> {
        let seq = test_sequence(vec![5, 6, 7, 8, 9, 1, 5], None);
        // [5, 6] recurs at the start, followed by 7, 8, 9.
        assert_eq!(extend(vec![6], &[&seq])?, vec![vec![6, 7, 8, 9]]);
        // Nothing earlier ends in [5, 4]: the draft stands alone.
        assert_eq!(extend(vec![4], &[&seq])?, vec![vec![4]]);
        Ok(())
    }

    #[test]
    fn a_batch_continues_by_its_shortest_find() -> Result<()> {
        let long = test_sequence(vec![5, 6, 7, 8, 9, 1, 5], None);
        // [2, 3] recurs one token back, so only [2, 3] follows it before the context ends.
        let short = test_sequence(vec![9, 2, 3, 2], None);
        assert_eq!(
            extend(vec![6, 3], &[&long, &short])?,
            vec![vec![6, 7, 8], vec![3, 2, 3]]
        );
        Ok(())
    }

    /// A sampled sequence keeps the bare draft: a looked-up token has no distribution to verify.
    #[test]
    fn a_sampled_batch_is_not_continued() -> Result<()> {
        let seq = test_sequence(vec![5, 6, 7, 8, 9, 1, 5], Some(0.8));
        assert_eq!(extend(vec![6], &[&seq])?, vec![vec![6]]);
        Ok(())
    }

    #[test]
    fn empty_and_tiny_history_never_panic() {
        assert!(lookup(&[], 1, 3, 4).is_empty());
        assert!(lookup(&[1], 1, 3, 4).is_empty());
        // A single repeat is a valid 1-gram match: after the first `1` came another `1`.
        assert_eq!(lookup(&[1, 1], 1, 3, 4), vec![1]);
    }

    #[test]
    fn strictly_unique_tail_has_no_match() {
        assert!(lookup(&[1, 2, 3, 4, 5], 1, 3, 4).is_empty());
    }

    #[test]
    fn returns_continuation_after_earlier_occurrence() {
        // Tail 2-gram [1,2] recurs; the earlier [1,2] at index 1 is followed by [3,1].
        let toks = [5, 1, 2, 3, 1, 2];
        assert_eq!(lookup(&toks, 1, 2, 2), vec![3, 1]);
    }

    #[test]
    fn prefers_most_recent_occurrence() {
        // Tail [1,2] occurs earlier at index 0 (→[9,..]) and index 3 (→[8,..]).
        // Most-recent wins: continuation starts with 8, not 9.
        let toks = [1, 2, 9, 1, 2, 8, 1, 2];
        assert_eq!(lookup(&toks, 1, 2, 2), vec![8, 1]);
        assert_eq!(lookup(&toks, 1, 2, 1), vec![8]);
    }

    #[test]
    fn prefers_longest_needle_then_falls_back() {
        // No 3-gram [7,2,3] earlier, but 2-gram [2,3] recurs at index 1 → [7,2].
        let toks = [1, 2, 3, 7, 2, 3];
        assert_eq!(lookup(&toks, 1, 3, 2), vec![7, 2]);
    }

    #[test]
    fn continuation_is_bounded_by_gamma_and_history_end() {
        // gamma exceeds what history holds after the match: cap at history end.
        let toks = [1, 2, 3, 1];
        assert_eq!(lookup(&toks, 1, 3, 5), vec![2, 3, 1]);
        // gamma caps a longer available run.
        let toks2 = [1, 2, 3, 4, 5, 1];
        assert_eq!(lookup(&toks2, 1, 3, 2), vec![2, 3]);
    }

    #[test]
    fn ngram_min_floor_is_respected() {
        // Only two tokens: a 2-gram needle has no earlier occurrence, and min=2
        // forbids falling back to a 1-gram, so nothing is proposed.
        assert!(lookup(&[9, 1], 2, 3, 2).is_empty());
    }
}
