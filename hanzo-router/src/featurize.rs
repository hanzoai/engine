//! Request -> feature vector `x`. The smallest, fastest thing that works: task
//! one-hot, length, media flag, bias, and a hashed n-gram block for domain
//! signal. No neural net, sub-microsecond.
//!
//! This is a router primitive: it turns a [`Request`] into the `x` the learned
//! scorer ranks arms with, and it is the SINGLE source of that vector so the
//! offline fit and the serving path featurize identically (no train/serve skew).
//! The [`Featurizer`] trait is the seam: a finetunable encoder can later produce
//! the same `x` (same `dim`) without any other piece changing.

use crate::classify::{Classifier, Heuristic, Request};
use crate::registry::Task;

pub const NUM_TASKS: usize = 8;
pub const NUM_HASH: usize = 5;
/// `x` layout: [task one-hot (8) | hashed n-grams (5) | log-len, media, bias (3)].
pub const FEAT_DIM: usize = NUM_TASKS + NUM_HASH + 3;

const LEN_NORM: f64 = 14.0; // ~ln(1e6); keeps log-length in ~[0,1]

pub trait Featurizer {
    fn dim(&self) -> usize;
    fn featurize(&self, req: &Request) -> Vec<f64>;
    /// The task the policy treats this request as (drives the quality column and
    /// the `min_quality` gate). Kept on the featurizer so it stays the single
    /// source of request->task truth.
    fn task_of(&self, req: &Request) -> Task;
}

/// Hashing-trick featurizer. Reuses the router's [`Heuristic`] classifier so the
/// task signal matches the rest of `hanzo-router`.
#[derive(Debug, Clone, Default)]
pub struct HashFeaturizer {
    classifier: Heuristic,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl Featurizer for HashFeaturizer {
    fn dim(&self) -> usize {
        FEAT_DIM
    }

    fn task_of(&self, req: &Request) -> Task {
        req.task_hint
            .unwrap_or_else(|| self.classifier.classify(req))
    }

    fn featurize(&self, req: &Request) -> Vec<f64> {
        let mut x = vec![0.0; FEAT_DIM];
        x[self.task_of(req).index()] = 1.0;

        let hash_base = NUM_TASKS;
        let toks: Vec<&str> = req
            .text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        let bump = |s: &str, x: &mut [f64]| {
            let b = (fnv1a(s.as_bytes()) as usize) % NUM_HASH;
            x[hash_base + b] += 1.0;
        };
        for w in &toks {
            bump(&w.to_lowercase(), &mut x);
        }
        for pair in toks.windows(2) {
            bump(&format!("{} {}", pair[0], pair[1]).to_lowercase(), &mut x);
        }
        let norm: f64 = x[hash_base..hash_base + NUM_HASH]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for v in &mut x[hash_base..hash_base + NUM_HASH] {
                *v /= norm;
            }
        }

        let tail = NUM_TASKS + NUM_HASH;
        x[tail] = (1.0 + req.approx_tokens as f64).ln() / LEN_NORM;
        x[tail + 1] = if req.has_media { 1.0 } else { 0.0 };
        x[tail + 2] = 1.0;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_and_one_hot_and_determinism() {
        let f = HashFeaturizer::default();
        let req = Request {
            text: "fix this ```rust``` bug".into(),
            approx_tokens: 100,
            ..Default::default()
        };
        let x = f.featurize(&req);
        assert_eq!(x.len(), FEAT_DIM);
        assert_eq!(f.task_of(&req), Task::Code);
        assert_eq!(x[Task::Code.index()], 1.0);
        assert_eq!(x[FEAT_DIM - 1], 1.0); // bias
        assert_eq!(x, f.featurize(&req)); // deterministic
    }

    #[test]
    fn media_flag_and_hint_override() {
        let f = HashFeaturizer::default();
        let req = Request {
            text: "describe".into(),
            has_media: true,
            task_hint: Some(Task::Vision),
            ..Default::default()
        };
        let x = f.featurize(&req);
        assert_eq!(x[Task::Vision.index()], 1.0);
        assert_eq!(x[NUM_TASKS + NUM_HASH + 1], 1.0); // media flag
    }
}
