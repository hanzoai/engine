//! Dataset loading and supervised loss-mask construction.
//!
//! The mask logic ([`build_datum`]) is pure and unit-tested: given already-tokenized
//! prompt and completion ids it produces a next-token-prediction [`Datum`] whose loss
//! mask covers exactly the completion (and the prompt→completion transition).

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::types::{Datum, ModelInput};

/// One line of a `{ "prompt": ..., "completion": ... }` JSONL file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Example {
    pub prompt: String,
    pub completion: String,
}

/// Build a supervised next-token [`Datum`] from tokenized prompt + completion.
///
/// `full = prompt ++ completion`; the model sees `full[..n-1]` and must predict
/// `full[1..]`. Position `i` is supervised (weight 1.0) exactly when its target
/// `full[i+1]` lies in the completion, i.e. `i + 1 >= prompt.len()`.
///
/// Returns `None` if there is nothing to train (needs >= 1 completion token and a
/// combined length >= 2).
pub fn build_datum(prompt: &[u32], completion: &[u32]) -> Option<Datum> {
    if completion.is_empty() {
        return None;
    }
    let prompt_len = prompt.len();
    let mut full = Vec::with_capacity(prompt_len + completion.len());
    full.extend_from_slice(prompt);
    full.extend_from_slice(completion);
    if full.len() < 2 {
        return None;
    }

    let input_tokens = full[..full.len() - 1].to_vec();
    let target_tokens = full[1..].to_vec();
    let weights = (0..full.len() - 1)
        .map(|i| if i + 1 >= prompt_len { 1.0 } else { 0.0 })
        .collect();

    Some(Datum {
        model_input: ModelInput::from_ints(input_tokens),
        target_tokens,
        weights,
    })
}

/// Tokenize an [`Example`] into a supervised [`Datum`].
///
/// BOS (if `bos`) prefixes the prompt; EOS (if `eos`) terminates the completion so
/// the model learns to stop. Special tokens are not auto-added by the tokenizer —
/// we place BOS/EOS explicitly to keep the loss-mask boundary exact.
pub fn tokenize_example(
    tokenizer: &Tokenizer,
    ex: &Example,
    bos: Option<u32>,
    eos: Option<u32>,
) -> anyhow::Result<Option<Datum>> {
    let mut prompt = Vec::new();
    if let Some(b) = bos {
        prompt.push(b);
    }
    prompt.extend(
        tokenizer
            .encode(ex.prompt.as_str(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .iter()
            .copied(),
    );

    let mut completion: Vec<u32> = tokenizer
        .encode(ex.completion.as_str(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    if let Some(e) = eos {
        completion.push(e);
    }

    Ok(build_datum(&prompt, &completion))
}

/// Load and tokenize a `{prompt, completion}` JSONL file into a dataset of [`Datum`].
pub fn load_jsonl(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
    bos: Option<u32>,
    eos: Option<u32>,
) -> anyhow::Result<Vec<Datum>> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ex: Example = serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("line {}: {e}", lineno + 1))?;
        if let Some(datum) = tokenize_example(tokenizer, &ex, bos, eos)? {
            data.push(datum);
        }
    }
    if data.is_empty() {
        anyhow::bail!("no trainable examples parsed from dataset");
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_covers_completion_and_transition() {
        // prompt = [10, 11], completion = [20, 21]
        // full = [10, 11, 20, 21]; input = [10,11,20]; target = [11,20,21]
        // weights: i=0 target=11 (prompt) -> 0; i=1 target=20 (completion) -> 1; i=2 target=21 -> 1
        let d = build_datum(&[10, 11], &[20, 21]).unwrap();
        assert_eq!(d.model_input.tokens, vec![10, 11, 20]);
        assert_eq!(d.target_tokens, vec![11, 20, 21]);
        assert_eq!(d.weights, vec![0.0, 1.0, 1.0]);
        assert_eq!(d.trained_tokens(), 2.0);
        d.validate().unwrap();
    }

    #[test]
    fn empty_prompt_trains_every_position() {
        let d = build_datum(&[], &[5, 6, 7]).unwrap();
        assert_eq!(d.model_input.tokens, vec![5, 6]);
        assert_eq!(d.target_tokens, vec![6, 7]);
        assert_eq!(d.weights, vec![1.0, 1.0]);
    }

    #[test]
    fn single_completion_token_after_prompt() {
        // prompt=[1,2,3], completion=[9]; full=[1,2,3,9]
        // input=[1,2,3]; target=[2,3,9]; only last target is completion.
        let d = build_datum(&[1, 2, 3], &[9]).unwrap();
        assert_eq!(d.weights, vec![0.0, 0.0, 1.0]);
        assert_eq!(d.trained_tokens(), 1.0);
    }

    #[test]
    fn no_completion_is_rejected() {
        assert!(build_datum(&[1, 2], &[]).is_none());
    }

    #[test]
    fn too_short_is_rejected() {
        assert!(build_datum(&[], &[]).is_none());
        // single total token: nothing to predict.
        assert!(build_datum(&[], &[1]).is_none());
    }
}
