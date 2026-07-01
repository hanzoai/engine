//! Piece 3/6: the learnable utility. `utility(x, p) = x^T W p` -- a bilinear
//! form that scores how well profile `p` serves request `x`. `W` is the only
//! offline-learned object; per-user adaptation is a delta on `W` produced by the
//! [`crate::learner`] and applied here through [`Policy::utility_with`].
//!
//! The bilinear form is deliberately the whole model: with the task one-hot in
//! `x` and quality-by-task in `p`, `W` learns the task->quality alignment that
//! routing needs, and stays a single matrix the bandit can update online. A
//! small MLP head would attach at this same seam (score `phi = vec(x p^T)`
//! through an MLP instead of a dot) without touching any other piece; the linear
//! head is what the proof needs and what the online learner stays convex over.

use crate::featurize::FEAT_DIM;
use crate::linalg;
use crate::profile::PROFILE_DIM;

pub const D: usize = FEAT_DIM;
pub const K: usize = PROFILE_DIM;
pub const DK: usize = D * K;

/// The base policy: a `D x K` row-major weight matrix.
#[derive(Debug, Clone)]
pub struct Policy {
    pub w: Vec<f64>,
}

impl Policy {
    pub fn zeros() -> Self {
        Self { w: vec![0.0; DK] }
    }

    pub fn from_weights(w: Vec<f64>) -> Self {
        assert_eq!(w.len(), DK, "policy weight matrix must be D*K");
        Self { w }
    }

    /// Score with the base weights.
    pub fn utility(&self, x: &[f64], p: &[f64]) -> f64 {
        linalg::bilinear(x, &self.w, p, D, K)
    }

    /// Score with a caller-supplied weight matrix (the per-user effective `W`).
    pub fn utility_with(w: &[f64], x: &[f64], p: &[f64]) -> f64 {
        linalg::bilinear(x, w, p, D, K)
    }
}
