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
//! Hugging Face `prompt_lookup_num_tokens` and vLLM's `[ngram]` method.

use hanzo_ml::Result;

use super::proposer::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    TargetTokenEmbedder,
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

/// Prompt-lookup / n-gram speculative proposer. See the module docs.
pub struct PromptLookupProposer {
    ngram_min: usize,
    ngram_max: usize,
    gamma: usize,
}

impl PromptLookupProposer {
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
}

impl SpeculativeProposer for PromptLookupProposer {
    fn proposal_len(&self) -> usize {
        self.gamma
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
                    lookup(seq.get_toks(), self.ngram_min, self.ngram_max, self.gamma)
                };
                SpeculativeProposal::new(tokens)
            })
            .collect();
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}

#[cfg(test)]
mod tests {
    use super::lookup;

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
