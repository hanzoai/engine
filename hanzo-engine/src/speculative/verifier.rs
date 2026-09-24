use std::sync::Arc;

use hanzo_ml::{DType, Result, Tensor};
use rand::Rng;
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sampler::Logprobs;
use crate::sequence::{Sequence, SequenceRecognizer, SequenceState};

pub struct VerificationOutcome {
    pub accepted_drafts: usize,
    pub proposed_drafts: usize,
    pub keep_len: usize,
    pub continuation_token: Option<u32>,
}

/// Verifies one sequence's staged drafts against the target's logits for them, emitting every
/// accepted draft and then the target's own next token. `rng` is the engine's shared RNG; a seeded
/// sequence draws from its own. The drafts count toward the request's usage as they are judged, so
/// the response a finishing token sends carries them.
#[allow(clippy::too_many_arguments)]
pub async fn finish_verified_step<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
    proposal_logits: Option<Tensor>,
    base_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    anchor_to_emit: Option<Logprobs>,
) -> Result<VerificationOutcome> {
    let general_metadata = pipeline.get_metadata();
    let eos_tok = if disable_eos_stop {
        None
    } else {
        Some(&general_metadata.eos_tok[..])
    };
    let return_logprobs = seq.return_logprobs();
    let rng = seq.rng(&rng);
    let outcome = |accepted_drafts, keep_len, continuation_token| VerificationOutcome {
        accepted_drafts,
        proposed_drafts: proposal.len(),
        keep_len,
        continuation_token,
    };

    if let Some(anchor) = anchor_to_emit {
        finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, anchor, eos_tok, true).await?;
        if matches!(seq.getstate(), SequenceState::Done(_)) {
            seq.clear_staged_speculative_tokens();
            return Ok(outcome(0, base_len + 1, None));
        }
    }
    seq.record_drafts(proposal.len(), 0);

    // Speculative sampling needs the drafts' distribution and a free sampler. A greedy or
    // grammar-constrained sequence, and any token the engine writes itself (a think close), is
    // verified by matching the target's own choice.
    let candidates = proposal_logits.filter(|_| {
        !seq.sampler().is_argmax() && matches!(seq.recognizer, SequenceRecognizer::None)
    });

    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let row = logit_row(&verify_logits, idx)?;
        let (emitted, is_draft) = match &candidates {
            Some(candidates) if seq.forced_token().is_none() => {
                let candidate_row = logit_row(candidates, idx)?;
                judge_draft(seq, row, candidate_row, draft, return_logprobs, &rng)?
            }
            _ => {
                let sampled =
                    sample_sequence(row, seq, return_logprobs, rng.clone(), false, false, false)
                        .await?;
                let is_draft = sampled.token == draft;
                (sampled, is_draft)
            }
        };
        if is_draft {
            accepted += 1;
            seq.record_drafts(0, 1);
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, emitted, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                seq.clear_staged_speculative_tokens();
                return Ok(outcome(accepted, base_len + 1 + accepted, None));
            }
            continue;
        }
        let token = emitted.token;
        finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, emitted, eos_tok, true).await?;
        let continuation = if matches!(seq.getstate(), SequenceState::Done(_)) {
            seq.clear_staged_speculative_tokens();
            None
        } else {
            Some(token)
        };
        return Ok(outcome(accepted, base_len + 1 + accepted, continuation));
    }

    // Every draft held: the row after the last one gives the next token.
    let row = logit_row(&verify_logits, accepted)?;
    let continuation = match &candidates {
        Some(_) if seq.forced_token().is_none() => {
            let sampler = seq.sampler();
            let target_probs =
                sampler.speculative_target_probs(flat_logits(row)?, seq.get_toks())?;
            sampler.sample_from_probs(&target_probs, return_logprobs, rng.clone())?
        }
        _ => sample_sequence(row, seq, return_logprobs, rng.clone(), false, false, false).await?,
    };
    let token = continuation.token;
    finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, continuation, eos_tok, true).await?;
    let continuation = if matches!(seq.getstate(), SequenceState::Done(_)) {
        seq.clear_staged_speculative_tokens();
        None
    } else {
        Some(token)
    };
    Ok(outcome(accepted, base_len + 1 + accepted, continuation))
}

/// Judges one draft by speculative sampling: the draft is kept with probability min(1, p/q), and
/// otherwise the token comes from the residual max(0, p - q), so the emitted token follows the
/// target's distribution p exactly. Returns the emitted token and whether it is the draft.
fn judge_draft(
    seq: &Sequence,
    target_row: Tensor,
    candidate_row: Tensor,
    draft: u32,
    return_logprobs: bool,
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<(Logprobs, bool)> {
    let sampler = seq.sampler();
    let target_probs =
        sampler.speculative_target_probs(flat_logits(target_row)?, seq.get_toks())?;
    let candidate_probs =
        sampler.speculative_candidate_probs(flat_logits(candidate_row)?, seq.get_toks())?;
    if target_probs.len() != candidate_probs.len() {
        hanzo_ml::bail!(
            "speculative target/candidate vocab mismatch: target={}, candidate={}",
            target_probs.len(),
            candidate_probs.len()
        );
    }
    let draft_idx = draft as usize;
    let p_i = target_probs.get(draft_idx).copied().unwrap_or(0.0);
    let q_i = candidate_probs.get(draft_idx).copied().unwrap_or(0.0);
    let accept_prob = if q_i <= 0.0 {
        if p_i > 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (p_i / q_i).min(1.0)
    };
    let draw = {
        let mut rng = rng.lock().expect("could not lock rng mutex");
        rng.random::<f32>()
    };
    if draw <= accept_prob {
        return Ok((
            sampler.logprobs_from_probs(draft, &target_probs, return_logprobs)?,
            true,
        ));
    }

    let mut adjusted_probs = target_probs
        .iter()
        .zip(candidate_probs.iter())
        .map(|(p, q)| (p - q).max(0.0))
        .collect::<Vec<_>>();
    if normalize_probs(&mut adjusted_probs).is_err() {
        adjusted_probs = target_probs;
    }
    let sampled = sampler.sample_from_probs(&adjusted_probs, return_logprobs, rng.clone())?;
    Ok((sampled, false))
}

fn logit_row(logits: &Tensor, row: usize) -> Result<Tensor> {
    match logits.dims() {
        [_, rows, _] => {
            if row >= *rows {
                hanzo_ml::bail!("speculative logit row {row} is out of range for {rows} rows");
            }
            logits.narrow(1, row, 1)
        }
        [rows, _] => {
            if row >= *rows {
                hanzo_ml::bail!("speculative logit row {row} is out of range for {rows} rows");
            }
            logits.narrow(0, row, 1)
        }
        shape => hanzo_ml::bail!("speculative logits have unsupported shape {shape:?}"),
    }
}

fn flat_logits(logits: Tensor) -> Result<Tensor> {
    match logits.dims() {
        [1, 1, _] => logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32),
        [1, _] => logits.squeeze(0)?.to_dtype(DType::F32),
        [_] => logits.to_dtype(DType::F32),
        dims => hanzo_ml::bail!("speculative logit row must flatten to rank 1, got {dims:?}"),
    }
}

fn normalize_probs(probs: &mut [f32]) -> Result<()> {
    let sum: f32 = probs
        .iter()
        .copied()
        .filter(|prob| prob.is_finite() && *prob > 0.0)
        .sum();
    if sum <= 0.0 {
        hanzo_ml::bail!("all probabilities are zero in speculative adjusted distribution");
    }
    for prob in probs.iter_mut() {
        if prob.is_finite() && *prob > 0.0 {
            *prob /= sum;
        } else {
            *prob = 0.0;
        }
    }
    Ok(())
}
