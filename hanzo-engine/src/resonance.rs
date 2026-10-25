//! Phase-aware semantic similarity for embedding-backed agent memory.
//!
//! An independent Rust implementation of the published resonance equation in
//! A. Listopad, "Wave-Based Semantic Memory with Resonance-Based Retrieval",
//! arXiv:2509.09691v1 (2025). No ResonanceDB source code is incorporated.
//!
//! IMPORTANT: Wrapping an existing real embedding as a complex waveform does
//! NOT infer negation or other semantic operators. Explicit, grounded operator
//! labels or phase-producing encoders are required to add that information.

use std::fmt;

/// A validated, dense complex waveform. Embedding dimensions are not time or
/// physical frequencies: this is a complex feature-space representation.
#[derive(Debug, Clone, PartialEq)]
pub struct WavePattern {
    real: Vec<f32>,
    imag: Vec<f32>,
    energy: f64,
}

/// Optional semantic transformation, selected by a caller with reliable
/// operator metadata. Never infer an operator from embedding coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseOperator {
    Identity,
    /// Shift every component by pi; use only for an explicitly identified
    /// logical negation of this SAME proposition or atomic memory.
    Negate,
    /// Apply an externally supplied phase offset (radians).
    Rotate(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveError {
    Empty,
    DimensionMismatch,
    NonFinite,
    NegativeAmplitude,
    InvalidWeight,
    InvalidCosine,
}

impl fmt::Display for WaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("wave pattern cannot be empty"),
            Self::DimensionMismatch => f.write_str("wave pattern dimensions do not match"),
            Self::NonFinite => f.write_str("wave pattern contains a non-finite number"),
            Self::NegativeAmplitude => f.write_str("wave amplitudes must be non-negative"),
            Self::InvalidWeight => f.write_str("resonance weight must be within [0, 1]"),
            Self::InvalidCosine => f.write_str("cosine similarity must be finite and within [-1, 1]"),
        }
    }
}

impl std::error::Error for WaveError {}

impl WavePattern {
    /// Sign-phase initialization: real-valued embeddings remain exactly real.
    /// Alone, this only gives a cosine-like score, not negation awareness.
    pub fn from_embedding(embedding: &[f32]) -> Result<Self, WaveError> {
        Self::from_cartesian(embedding.to_vec(), vec![0.0; embedding.len()])
    }

    /// Construct complex coordinates with non-negative amplitudes and phases
    /// expressed in radians. This permits externally generated phase channels.
    pub fn from_polar(amplitudes: &[f32], phases: &[f32]) -> Result<Self, WaveError> {
        if amplitudes.is_empty() {
            return Err(WaveError::Empty);
        }
        if amplitudes.len() != phases.len() {
            return Err(WaveError::DimensionMismatch);
        }
        let mut real = Vec::with_capacity(amplitudes.len());
        let mut imag = Vec::with_capacity(amplitudes.len());
        for (&amplitude, &phase) in amplitudes.iter().zip(phases) {
            if !amplitude.is_finite() || !phase.is_finite() {
                return Err(WaveError::NonFinite);
            }
            if amplitude < 0.0 {
                return Err(WaveError::NegativeAmplitude);
            }
            let (sin, cos) = phase.sin_cos();
            real.push(amplitude * cos);
            imag.push(amplitude * sin);
        }
        Self::from_cartesian(real, imag)
    }

    /// Construct from explicit real and imaginary feature channels.
    pub fn from_cartesian(real: Vec<f32>, imag: Vec<f32>) -> Result<Self, WaveError> {
        if real.is_empty() {
            return Err(WaveError::Empty);
        }
        if real.len() != imag.len() {
            return Err(WaveError::DimensionMismatch);
        }
        let mut energy = 0.0_f64;
        for (&re, &im) in real.iter().zip(&imag) {
            if !re.is_finite() || !im.is_finite() {
                return Err(WaveError::NonFinite);
            }
            energy += f64::from(re).powi(2) + f64::from(im).powi(2);
        }
        Ok(Self { real, imag, energy })
    }

    pub fn dimensions(&self) -> usize {
        self.real.len()
    }

    pub fn energy(&self) -> f64 {
        self.energy
    }

    pub fn apply(&self, operator: PhaseOperator) -> Result<Self, WaveError> {
        match operator {
            PhaseOperator::Identity => Ok(self.clone()),
            PhaseOperator::Negate => Self::from_cartesian(
                self.real.iter().map(|v| -v).collect(),
                self.imag.iter().map(|v| -v).collect(),
            ),
            PhaseOperator::Rotate(theta) => {
                if !theta.is_finite() {
                    return Err(WaveError::NonFinite);
                }
                let (sin, cos) = theta.sin_cos();
                let real = self
                    .real
                    .iter()
                    .zip(&self.imag)
                    .map(|(&re, &im)| re * cos - im * sin)
                    .collect();
                let imag = self
                    .real
                    .iter()
                    .zip(&self.imag)
                    .map(|(&re, &im)| re * sin + im * cos)
                    .collect();
                Self::from_cartesian(real, imag)
            }
        }
    }

    /// Published normalized constructive-interference score in [0, 1].
    ///
    /// S = 0.5 * (E1 + E2 + 2 Re<q,d>) / (E1 + E2)
    ///     * (2 sqrt(E1 E2) / (E1 + E2))
    ///
    /// Complexity is O(dimensions); this kernel does not build an ANN index.
    pub fn resonance(&self, other: &Self) -> Result<f64, WaveError> {
        if self.dimensions() != other.dimensions() {
            return Err(WaveError::DimensionMismatch);
        }
        if self.energy == 0.0 || other.energy == 0.0 {
            return Ok(0.0);
        }

        let dot: f64 = self
            .real
            .iter()
            .zip(&self.imag)
            .zip(other.real.iter().zip(&other.imag))
            .map(|((&ar, &ai), (&br, &bi))| {
                f64::from(ar) * f64::from(br) + f64::from(ai) * f64::from(bi)
            })
            .sum();

        let total = self.energy + other.energy;
        let interference = (total + 2.0 * dot) / total;
        let balance = 2.0 * (self.energy * other.energy).sqrt() / total;
        Ok((0.5 * interference * balance).clamp(0.0, 1.0))
    }
}

/// A bounded reranking candidate. Dense cosine is calculated on the original
/// embeddings before adding semantic phase, typically from the ANN short list.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    pub id: &'a str,
    pub pattern: &'a WavePattern,
    pub dense_cosine: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit<'a> {
    pub id: &'a str,
    pub score: f64,
    pub resonance: f64,
    pub dense_cosine: f64,
}

/// Hybrid rerank an existing dense/lexical shortlist using a phase-aware score.
/// Nothing here replaces a vector index or discovers polarity automatically.
pub fn rerank_candidates<'a>(
    query: &WavePattern,
    candidates: impl IntoIterator<Item = Candidate<'a>>,
    resonance_weight: f64,
    top_k: usize,
) -> Result<Vec<RankedHit<'a>>, WaveError> {
    if !resonance_weight.is_finite() || !(0.0..=1.0).contains(&resonance_weight) {
        return Err(WaveError::InvalidWeight);
    }

    let mut matches = Vec::new();
    for candidate in candidates {
        if !candidate.dense_cosine.is_finite() || !(-1.0..=1.0).contains(&candidate.dense_cosine) {
            return Err(WaveError::InvalidCosine);
        }
        let resonance = query.resonance(candidate.pattern)?;
        let dense_unit = (1.0 + candidate.dense_cosine) / 2.0;
        let score = resonance_weight * resonance + (1.0 - resonance_weight) * dense_unit;
        matches.push(RankedHit {
            id: candidate.id,
            score,
            resonance,
            dense_cosine: candidate.dense_cosine,
        });
    }
    matches.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(b.id)));
    matches.truncate(top_k);
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn real_embedding_is_cosine_equivalent_at_equal_norm() {
        let x = WavePattern::from_embedding(&[1.0, 0.0]).unwrap();
        let same = WavePattern::from_embedding(&[1.0, 0.0]).unwrap();
        let orthogonal = WavePattern::from_embedding(&[0.0, 1.0]).unwrap();
        let opposite = WavePattern::from_embedding(&[-1.0, 0.0]).unwrap();
        near(x.resonance(&same).unwrap(), 1.0);
        near(x.resonance(&orthogonal).unwrap(), 0.5);
        near(x.resonance(&opposite).unwrap(), 0.0);
    }

    #[test]
    fn phase_negation_requires_explicit_operator() {
        let asserted = WavePattern::from_embedding(&[0.8, 0.6]).unwrap();
        let denied = asserted.apply(PhaseOperator::Negate).unwrap();
        near(asserted.resonance(&denied).unwrap(), 0.0);
        near(denied.resonance(&denied).unwrap(), 1.0);

        // Without operator metadata the wrapper cannot tell these apart.
        let unmodified = WavePattern::from_embedding(&[0.8, 0.6]).unwrap();
        near(asserted.resonance(&unmodified).unwrap(), 1.0);
    }

    #[test]
    fn unequal_magnitudes_are_penalized() {
        let a = WavePattern::from_embedding(&[1.0]).unwrap();
        let b = WavePattern::from_embedding(&[2.0]).unwrap();
        near(a.resonance(&b).unwrap(), 0.72);
    }

    #[test]
    fn complex_rotation_is_globally_invariant() {
        let a = WavePattern::from_polar(&[1.0, 2.0], &[0.4, -1.0]).unwrap();
        let b = WavePattern::from_polar(&[2.0, 1.0], &[-0.5, 0.7]).unwrap();
        let first = a.resonance(&b).unwrap();
        let ar = a.apply(PhaseOperator::Rotate(0.8)).unwrap();
        let br = b.apply(PhaseOperator::Rotate(0.8)).unwrap();
        near(ar.resonance(&br).unwrap(), first);
    }

    #[test]
    fn rejects_invalid_input() {
        assert_eq!(WavePattern::from_embedding(&[]).unwrap_err(), WaveError::Empty);
        assert_eq!(
            WavePattern::from_embedding(&[f32::NAN]).unwrap_err(),
            WaveError::NonFinite
        );
        assert_eq!(
            WavePattern::from_polar(&[-1.0], &[0.0]).unwrap_err(),
            WaveError::NegativeAmplitude
        );
        assert_eq!(
            WavePattern::from_polar(&[1.0], &[]).unwrap_err(),
            WaveError::DimensionMismatch
        );
        let a = WavePattern::from_embedding(&[1.0]).unwrap();
        let b = WavePattern::from_embedding(&[1.0, 0.0]).unwrap();
        assert_eq!(a.resonance(&b).unwrap_err(), WaveError::DimensionMismatch);
    }

    #[test]
    fn zero_vectors_are_not_valid_matches() {
        let z = WavePattern::from_embedding(&[0.0, 0.0]).unwrap();
        let a = WavePattern::from_embedding(&[1.0, 0.0]).unwrap();
        near(z.resonance(&a).unwrap(), 0.0);
        near(z.resonance(&z).unwrap(), 0.0);
    }

    #[test]
    fn hybrid_reranker_distinguishes_explicit_polarity() {
        let asserted = WavePattern::from_embedding(&[1.0, 0.0]).unwrap();
        let denied = asserted.apply(PhaseOperator::Negate).unwrap();
        let matches = rerank_candidates(
            &denied,
            [
                Candidate { id: "asserted", pattern: &asserted, dense_cosine: 1.0 },
                Candidate { id: "denied", pattern: &denied, dense_cosine: 1.0 },
            ],
            0.75,
            2,
        ).unwrap();
        assert_eq!(matches[0].id, "denied");
        near(matches[0].score, 1.0);
        near(matches[1].score, 0.25);
    }

    #[test]
    fn rejects_invalid_fusion_and_cosine() {
        let a = WavePattern::from_embedding(&[1.0]).unwrap();
        assert_eq!(
            rerank_candidates(&a, [Candidate { id: "a", pattern: &a, dense_cosine: f64::NAN }], 0.5, 1).unwrap_err(),
            WaveError::InvalidCosine
        );
        assert_eq!(
            rerank_candidates(&a, std::iter::empty(), 1.5, 1).unwrap_err(),
            WaveError::InvalidWeight
        );
    }
}
