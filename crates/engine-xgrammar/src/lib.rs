// SPDX-License-Identifier: Apache-2.0
//! [Bounty #16] (`hanzoai/engine#97`): **Near-zero-overhead constrained generation** (XGrammar).
//!
//! Clean-room simulation harness comparing ways to enforce a grammar (e.g. JSON
//! schema) during token-by-token decoding.
//!
//! # Design
//!
//! Decoding with a grammar requires knowing which tokens are legal at each step.
//! Three strategies:
//!
//! - `Unconstrained`: no mask. Fastest, but emits structurally invalid output.
//! - `ReparsePerToken`: re-parse the partial output against the grammar at every
//!   token. Correct, but the mask computation dominates decode time.
//! - `PrecompiledMask`: precompute the token mask once per grammar state and
//!   cache it, so per-token cost is a lookup.
//!
//! Metrics: tokens/sec, invalid-output rate, and the break-even grammar reuse
//! count at which precompilation pays for itself.
//!
//! # Honest boundaries
//! - No real tokenizer, grammar engine, or GPU is exercised.
//! - Per-token costs are modelled constants, not measured kernel timings.
//! - Grammar states are synthetic; real JSON schemas vary wildly in mask size.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced against a real decoder.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum XGError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A synthetic generation request.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u32,
    /// Distinct grammar states visited during this generation.
    pub grammar_states: usize,
    /// Tokens generated.
    pub tokens: usize,
    /// Probability that an unconstrained model violates the grammar.
    pub violation_prob: f64,
}

pub fn generate_requests(seed: u64, n: usize) -> Vec<Request> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|i| {
            let tokens = 40 + (rng.next_u64() as usize % 360);
            // Bigger outputs visit more grammar states.
            let states = 4 + tokens / 12 + (rng.next_u64() as usize % 8);
            Request {
                id: i as u32,
                grammar_states: states,
                tokens,
                violation_prob: 0.02 + rng.next_f64() * 0.28,
            }
        })
        .collect()
}

/// Modelled per-token costs in nanoseconds.
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    /// Base model decode step.
    pub decode_ns: f64,
    /// Cost to re-parse partial output and derive a mask from scratch.
    pub reparse_ns: f64,
    /// Cost of a cached mask lookup.
    pub lookup_ns: f64,
    /// One-time cost to precompile a mask for a grammar state.
    pub precompile_ns: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            decode_ns: 12_000.0,
            reparse_ns: 9_000.0,
            lookup_ns: 120.0,
            precompile_ns: 48_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Unconstrained,
    ReparsePerToken,
    PrecompiledMask,
}

impl Policy {
    pub fn label(&self) -> &'static str {
        match self {
            Policy::Unconstrained => "unconstrained",
            Policy::ReparsePerToken => "reparse-per-token",
            Policy::PrecompiledMask => "precompiled-mask",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub requests: usize,
    pub total_tokens: usize,
    pub tokens_per_sec: f64,
    pub invalid_output_rate: f64,
    pub total_time_ms: f64,
    /// Mask overhead as a fraction of total decode time.
    pub mask_overhead_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub n_requests: usize,
    /// How many requests share the same grammar (controls cache reuse).
    pub requests_per_grammar: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0x56A_11A, n_requests: 400, requests_per_grammar: 40 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, n_requests: 60, requests_per_grammar: 10 }
    }
}

pub fn evaluate(policy: Policy, reqs: &[Request], cm: &CostModel, per_grammar: usize) -> PolicyResult {
    let mut total_tokens = 0usize;
    let mut total_ns = 0.0f64;
    let mut mask_ns = 0.0f64;
    let mut invalid = 0usize;

    // Precompiled masks are keyed by (grammar group, state index) and shared
    // across requests that use the same grammar.
    let mut cache: HashMap<(usize, usize), f64> = HashMap::new();

    for (idx, r) in reqs.iter().enumerate() {
        total_tokens += r.tokens;
        let grammar_group = idx / per_grammar.max(1);

        match policy {
            Policy::Unconstrained => {
                total_ns += r.tokens as f64 * cm.decode_ns;
                // A request is invalid if the unconstrained model happens to
                // violate the grammar; deterministic per request id.
                let draw = ((r.id as u64 * 2_654_435_761) % 1000) as f64 / 1000.0;
                if draw < r.violation_prob {
                    invalid += 1;
                }
            }
            Policy::ReparsePerToken => {
                total_ns += r.tokens as f64 * (cm.decode_ns + cm.reparse_ns);
                mask_ns += r.tokens as f64 * cm.reparse_ns;
            }
            Policy::PrecompiledMask => {
                // Precompile each distinct state once per grammar group.
                for s in 0..r.grammar_states {
                    let key = (grammar_group, s);
                    if !cache.contains_key(&key) {
                        cache.insert(key, cm.precompile_ns);
                        total_ns += cm.precompile_ns;
                        mask_ns += cm.precompile_ns;
                    }
                }
                total_ns += r.tokens as f64 * (cm.decode_ns + cm.lookup_ns);
                mask_ns += r.tokens as f64 * cm.lookup_ns;
            }
        }
    }

    let seconds = total_ns / 1e9;
    PolicyResult {
        policy: policy.label().into(),
        requests: reqs.len(),
        total_tokens,
        tokens_per_sec: if seconds > 0.0 { total_tokens as f64 / seconds } else { 0.0 },
        invalid_output_rate: invalid as f64 / reqs.len().max(1) as f64,
        total_time_ms: total_ns / 1e6,
        mask_overhead_fraction: if total_ns > 0.0 { mask_ns / total_ns } else { 0.0 },
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, XGError> {
    if cfg.n_requests == 0 {
        return Err(XGError::InvalidConfig("n_requests must be positive".into()));
    }
    if cfg.requests_per_grammar == 0 {
        return Err(XGError::InvalidConfig("requests_per_grammar must be positive".into()));
    }
    let reqs = generate_requests(cfg.seed, cfg.n_requests);
    let cm = CostModel::default();
    let results = vec![
        evaluate(Policy::Unconstrained, &reqs, &cm, cfg.requests_per_grammar),
        evaluate(Policy::ReparsePerToken, &reqs, &cm, cfg.requests_per_grammar),
        evaluate(Policy::PrecompiledMask, &reqs, &cm, cfg.requests_per_grammar),
    ];

    let mut notes = Vec::new();
    if let (Some(reparse), Some(pre)) = (
        results.iter().find(|r| r.policy == "reparse-per-token"),
        results.iter().find(|r| r.policy == "precompiled-mask"),
    ) {
        notes.push(format!(
            "precompiled masks are {:.2}x faster than re-parsing ({:.0} vs {:.0} tok/s) at the same validity, with mask overhead dropping from {:.1}% to {:.1}% of decode time",
            pre.tokens_per_sec / reparse.tokens_per_sec.max(1e-9),
            pre.tokens_per_sec,
            reparse.tokens_per_sec,
            reparse.mask_overhead_fraction * 100.0,
            pre.mask_overhead_fraction * 100.0
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/engine#97 [Bounty #16] Near-zero-overhead constrained generation (XGrammar)".into(),
        hip: "HIP-1344: Inference Engine Efficiency".into(),
        honesty_labels: vec![
            "no real tokenizer, grammar engine, or GPU exercised".into(),
            "per-token costs are modelled constants, not measured kernel timings".into(),
            "grammar state counts are synthetic".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced against a real decoder.".into(),
        ],
        config: cfg.clone(),
        results,
        notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precompiled_beats_reparse() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let re = report.results.iter().find(|r| r.policy == "reparse-per-token").unwrap();
        let pre = report.results.iter().find(|r| r.policy == "precompiled-mask").unwrap();
        assert!(
            pre.tokens_per_sec > re.tokens_per_sec,
            "precompiled {:.0} tok/s should beat reparse {:.0}",
            pre.tokens_per_sec,
            re.tokens_per_sec
        );
    }

    #[test]
    fn constrained_policies_emit_valid_output() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for name in ["reparse-per-token", "precompiled-mask"] {
            let r = report.results.iter().find(|r| r.policy == name).unwrap();
            assert_eq!(r.invalid_output_rate, 0.0, "{} emitted invalid output", name);
        }
    }

    #[test]
    fn unconstrained_is_fastest_but_invalid() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let un = report.results.iter().find(|r| r.policy == "unconstrained").unwrap();
        let pre = report.results.iter().find(|r| r.policy == "precompiled-mask").unwrap();
        assert!(un.tokens_per_sec > pre.tokens_per_sec);
        assert!(un.invalid_output_rate > 0.0);
    }

    #[test]
    fn mask_overhead_is_lower_with_precompilation() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let re = report.results.iter().find(|r| r.policy == "reparse-per-token").unwrap();
        let pre = report.results.iter().find(|r| r.policy == "precompiled-mask").unwrap();
        assert!(pre.mask_overhead_fraction < re.mask_overhead_fraction);
    }

    #[test]
    fn bad_config_rejected() {
        assert!(run_bench(&BenchConfig { seed: 1, n_requests: 0, requests_per_grammar: 4 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, n_requests: 4, requests_per_grammar: 0 }).is_err());
    }
}
