// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic multi-trial benchmark harness, 16-vector adversarial verification
//! matrix, HIP-1350 5-axis evaluation, and 27B interactive-decode negative-regime
//! boundary analysis for `hanzo-zkml`.

use std::time::Instant;

use crate::circuit::{
    prove_circuit, prove_circuit_unchecked_for_adversarial_test, setup_circuit_keys,
    verify_circuit_proof, CircuitVersion, CompactClassifierV1, DeterministicEvalScorerV1,
    KaiPolicyGateV1,
};
use crate::envelope::{escape_json, HanzoProofEnvelope};
use crate::field::Fp;
use crate::merkle::LigeroParams;
use crate::sha256::{hex_encode, sha256};
use crate::zkvm::{
    build_zkvm_compact_classifier_program, build_zkvm_eval_scorer_program,
    build_zkvm_kai_policy_program, prove_zkvm, prove_zkvm_with_corrupted_trace_for_test,
    setup_zkvm_keys, verify_zkvm_receipt, ZkVmVersion,
};
use crate::{TrustTier, ZkmlError, PROTOCOL_VERSION};

/// Assumed commodity CPU rate (`$0.04 / vCPU-hour`) for transparent cost accounting.
pub const COMMODITY_VCPU_USD_PER_HOUR: f64 = 0.04;

/// Useful tokens amortized per single compact-classifier / Kai-gate / eval proof.
pub const TOKENS_PER_DECISION_REQUEST: f64 = 512.0;

/// Lower-bound MACs / R1CS constraints for one dense 27B-parameter decode token
/// (`2 * 27e9` multiply-adds, excluding nonlinear attention/norms).
pub const DENSE_27B_MACS_PER_TOKEN: f64 = 54_000_000_000.0;

/// Deterministic splitmix64 PRNG for seeded benchmark input generation (zero external deps).
#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn range_i64(&mut self, lo: i64, hi_inclusive: i64) -> i64 {
        let span = (hi_inclusive - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }
}

/// Measured latency, size, throughput, and cost metrics for one workload.
#[derive(Clone, Debug)]
pub struct WorkloadBenchmark {
    /// Stable workload name.
    pub name: String,
    /// Backend category (`"model_specific_circuit"` or `"general_zkvm"`).
    pub backend: String,
    /// Number of measured trials.
    pub trials: usize,
    /// Number of algebraic R1CS constraints verified.
    pub constraints_count: usize,
    /// Executed zkVM cycles (`0` for model-specific circuits).
    pub trace_cycles: usize,
    /// Exact serialized proof/receipt size in bytes.
    pub proof_bytes: usize,
    /// Prover p50 latency in milliseconds.
    pub prove_p50_ms: f64,
    /// Prover p95 latency in milliseconds.
    pub prove_p95_ms: f64,
    /// Prover p99 latency in milliseconds.
    pub prove_p99_ms: f64,
    /// Prover mean latency in milliseconds.
    pub prove_mean_ms: f64,
    /// Verifier p50 latency in milliseconds.
    pub verify_p50_ms: f64,
    /// Verifier p95 latency in milliseconds.
    pub verify_p95_ms: f64,
    /// Verifier p99 latency in milliseconds.
    pub verify_p99_ms: f64,
    /// Verifier mean latency in milliseconds.
    pub verify_mean_ms: f64,
    /// Prover throughput (proofs / second, derived from mean latency).
    pub prove_throughput_per_sec: f64,
    /// Verifier throughput (verifications / second, derived from mean latency).
    pub verify_throughput_per_sec: f64,
    /// Share of trials completing with valid verified proofs (`1.0` = 100%).
    pub completion_rate: f64,
    /// Amortized proving + verification CPU cost (`$/M useful tokens` at 512 tokens/request).
    pub amortized_cost_usd_per_million_tokens: f64,
    /// Canonical `HanzoProofEnvelope v1` digest for trial 0.
    pub sample_envelope_digest: String,
}

/// Result of one adversarial tampering attack vector.
#[derive(Clone, Debug)]
pub struct AdversarialCheck {
    /// Attack vector name.
    pub attack_id: String,
    /// Target path (`"circuit"` or `"zkvm"`).
    pub target_path: String,
    /// Whether the stateless verifier rejected the attack (`true` required).
    pub rejected: bool,
    /// Rejection error message returned by the verifier.
    pub rejection_reason: String,
}

/// Measured CPU extrapolation proving why full per-token zkML on 27B autoregressive
/// decode is rejected as an initial interactive deliverable.
#[derive(Clone, Debug)]
pub struct NegativeRegime27B {
    /// Reference compact circuit used for per-constraint CPU timing calibration.
    pub calibration_workload: String,
    /// Measured CPU nanoseconds per R1CS constraint (mean).
    pub measured_ns_per_constraint: f64,
    /// Lower-bound MACs/constraints per single 27B decode token.
    pub dense_27b_constraints_per_token_lower_bound: f64,
    /// Extrapolated single-token CPU prove latency in seconds.
    pub extrapolated_single_token_prove_sec: f64,
    /// Extrapolated single-token CPU prove latency in hours.
    pub extrapolated_single_token_prove_hours: f64,
    /// Extrapolated CPU proving cost in USD per 1M useful tokens.
    pub extrapolated_cost_usd_per_million_tokens: f64,
    /// Explicit architectural verdict (`"REJECTED_FOR_INTERACTIVE_27B_DECODE"`).
    pub verdict: String,
}

/// Complete deterministic benchmark + adversarial audit + HIP-1350 5-axis report.
#[derive(Clone, Debug)]
pub struct BenchmarkSuiteReport {
    /// Protocol version.
    pub protocol_version: String,
    /// Seed used for deterministic test input generation.
    pub seed: u64,
    /// Number of measured trials per workload.
    pub trials: usize,
    /// Ligero RS blowup factor.
    pub ligero_blowup: usize,
    /// Ligero column query count.
    pub ligero_queries: usize,
    /// Conservative Ligero proximity soundness bits.
    pub ligero_proximity_soundness_bits: f64,
    /// Sumcheck extension field soundness bits (`~128` over `F_{p^2}`).
    pub sumcheck_field_soundness_bits: f64,
    /// All measured workload benchmarks (Path A + Path B).
    pub workloads: Vec<WorkloadBenchmark>,
    /// All adversarial attack test results.
    pub adversarial_checks: Vec<AdversarialCheck>,
    /// Share of adversarial attacks rejected (`0.0..=1.0`).
    pub adversarial_detection_rate: f64,
    /// Explicit 27B negative regime analysis.
    pub negative_regime_27b: NegativeRegime27B,
    /// Share of verification work proceeding without Hanzo-operated infrastructure (`1.0`).
    pub decentralization_score: f64,
    /// Share of artifacts exportable/verifiable/forkable under Apache-2.0 (`1.0`).
    pub sovereignty_score: f64,
}

fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_amortized_usd_per_million_tokens(prove_ms: f64, verify_ms: f64) -> f64 {
    let total_cpu_hours_per_proof = ((prove_ms + verify_ms) / 1000.0) / 3600.0;
    let usd_per_proof = total_cpu_hours_per_proof * COMMODITY_VCPU_USD_PER_HOUR;
    (usd_per_proof / TOKENS_PER_DECISION_REQUEST) * 1_000_000.0
}

struct TimingSummaryInput<'a> {
    name: &'a str,
    backend: &'a str,
    constraints_count: usize,
    trace_cycles: usize,
    proof_bytes: usize,
    prove_ms: Vec<f64>,
    verify_ms: Vec<f64>,
    completed: usize,
    sample_envelope_digest: String,
}

fn summarize_timings(mut input: TimingSummaryInput<'_>) -> WorkloadBenchmark {
    let trials = input.prove_ms.len();
    let prove_mean = input.prove_ms.iter().sum::<f64>() / (trials.max(1) as f64);
    let verify_mean = input.verify_ms.iter().sum::<f64>() / (trials.max(1) as f64);
    input.prove_ms.sort_by(|a, b| a.total_cmp(b));
    input.verify_ms.sort_by(|a, b| a.total_cmp(b));

    WorkloadBenchmark {
        name: input.name.to_string(),
        backend: input.backend.to_string(),
        trials,
        constraints_count: input.constraints_count,
        trace_cycles: input.trace_cycles,
        proof_bytes: input.proof_bytes,
        prove_p50_ms: percentile_sorted(&input.prove_ms, 0.50),
        prove_p95_ms: percentile_sorted(&input.prove_ms, 0.95),
        prove_p99_ms: percentile_sorted(&input.prove_ms, 0.99),
        prove_mean_ms: prove_mean,
        verify_p50_ms: percentile_sorted(&input.verify_ms, 0.50),
        verify_p95_ms: percentile_sorted(&input.verify_ms, 0.95),
        verify_p99_ms: percentile_sorted(&input.verify_ms, 0.99),
        verify_mean_ms: verify_mean,
        prove_throughput_per_sec: if prove_mean > 0.0 {
            1000.0 / prove_mean
        } else {
            0.0
        },
        verify_throughput_per_sec: if verify_mean > 0.0 {
            1000.0 / verify_mean
        } else {
            0.0
        },
        completion_rate: (input.completed as f64) / (trials.max(1) as f64),
        amortized_cost_usd_per_million_tokens: compute_amortized_usd_per_million_tokens(
            prove_mean,
            verify_mean,
        ),
        sample_envelope_digest: input.sample_envelope_digest,
    }
}

/// Runs the complete 16-vector adversarial rejection suite across Path A and Path B.
pub fn run_adversarial_matrix(
    ligero_params: LigeroParams,
) -> Result<Vec<AdversarialCheck>, ZkmlError> {
    let mut checks = Vec::new();
    let mut record = |attack_id: &str, target_path: &str, res: Result<(), ZkmlError>| match res {
        Ok(()) => checks.push(AdversarialCheck {
            attack_id: attack_id.to_string(),
            target_path: target_path.to_string(),
            rejected: false,
            rejection_reason: "UNEXPECTED_ACCEPT".to_string(),
        }),
        Err(e) => checks.push(AdversarialCheck {
            attack_id: attack_id.to_string(),
            target_path: target_path.to_string(),
            rejected: true,
            rejection_reason: e.to_string(),
        }),
    };

    // --- Path A: Model-Specific Circuit Attacks (8 vectors) ---
    let clf = CompactClassifierV1::compile();
    let weights = CompactClassifierV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(clf.spec.clone(), &weights, ligero_params)?;
    let inputs = vec![5, -3, 8, 2, -4, 6, 1, -2];
    let (outputs, witness) = CompactClassifierV1::evaluate_witness(&inputs, &weights)?;
    let valid_proof = prove_circuit(&pk, &inputs, &outputs, &witness)?;

    // 1. Wrong weights (prove under tampered weights, verify against true VK).
    let mut wrong_weights = weights.clone();
    wrong_weights[20] += 1;
    let (bad_pk, _) = setup_circuit_keys(clf.spec.clone(), &wrong_weights, ligero_params)?;
    let (bad_out, bad_wit) = CompactClassifierV1::evaluate_witness(&inputs, &wrong_weights)?;
    let wrong_w_proof = prove_circuit(&bad_pk, &inputs, &bad_out, &bad_wit)?;
    record(
        "circuit_wrong_weights_vk",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &bad_out, &wrong_w_proof).map(|_| ()),
    );

    // 2. Wrong output logit (+1 perturbation).
    let mut bad_logit_out = outputs.clone();
    bad_logit_out[0] += 1;
    record(
        "circuit_wrong_output_logit",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &bad_logit_out, &valid_proof).map(|_| ()),
    );

    // 3. Wrong predicted class index.
    let mut bad_class_out = outputs.clone();
    bad_class_out[4] = (bad_class_out[4] + 1) % 4;
    record(
        "circuit_wrong_output_class",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &bad_class_out, &valid_proof).map(|_| ()),
    );

    // 4. Wrong circuit version.
    let mut bad_ver_proof = valid_proof.clone();
    bad_ver_proof.version = CircuitVersion::new(2, 0, 0);
    record(
        "circuit_wrong_version",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &outputs, &bad_ver_proof).map(|_| ()),
    );

    // 5. Wrong circuit digest.
    let mut bad_digest_proof = valid_proof.clone();
    bad_digest_proof.circuit_digest[0] ^= 0xFF;
    record(
        "circuit_wrong_digest",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &outputs, &bad_digest_proof).map(|_| ()),
    );

    // 6. Violated intermediate R1CS constraint (forged ReLU/hidden witness proved unchecked).
    let mut forged_witness = witness.clone();
    forged_witness[18] += Fp::ONE;
    let forged_proof =
        prove_circuit_unchecked_for_adversarial_test(&pk, &inputs, &outputs, &forged_witness)?;
    record(
        "circuit_forged_intermediate_witness",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &outputs, &forged_proof).map(|_| ()),
    );

    // 7. Wrong preprocessing / perturbed input feature.
    let mut bad_inputs = inputs.clone();
    bad_inputs[3] += 1;
    record(
        "circuit_wrong_preprocessing_input",
        "circuit",
        verify_circuit_proof(&vk, &bad_inputs, &outputs, &valid_proof).map(|_| ()),
    );

    // 8. Cross-domain proof replay.
    let mut bad_domain_proof = valid_proof.clone();
    bad_domain_proof.domain_tag = "hanzo.zkml.other_domain.v1".to_string();
    record(
        "circuit_cross_domain_replay",
        "circuit",
        verify_circuit_proof(&vk, &inputs, &outputs, &bad_domain_proof).map(|_| ()),
    );

    // --- Path B: General zkVM Attacks (8 vectors) ---
    let (vm_prog, vm_weights) = build_zkvm_compact_classifier_program();
    let max_cycles = vm_prog.instructions.len() + 8;
    let (vm_pk, vm_vk) = setup_zkvm_keys(vm_prog.clone(), &vm_weights, max_cycles, ligero_params)?;
    let (vm_out, vm_receipt) = prove_zkvm(&vm_pk, &inputs)?;

    // 9. Wrong expected ImageID.
    let mut bad_image_id = vm_vk.image_id;
    bad_image_id[0] ^= 0x01;
    record(
        "zkvm_wrong_image_id",
        "zkvm",
        verify_zkvm_receipt(&bad_image_id, &vm_vk, &inputs, &vm_out, &vm_receipt).map(|_| ()),
    );

    // 10. Wrong zkVM ISA version.
    let mut bad_vm_ver = vm_receipt.clone();
    bad_vm_ver.vm_version = ZkVmVersion::new(2, 0, 0);
    record(
        "zkvm_wrong_vm_version",
        "zkvm",
        verify_zkvm_receipt(&vm_vk.image_id, &vm_vk, &inputs, &vm_out, &bad_vm_ver).map(|_| ()),
    );

    // 11. Wrong weights table committed.
    let mut bad_vm_weights = vm_weights.clone();
    bad_vm_weights[12] += 1;
    let (bad_vm_pk, _) =
        setup_zkvm_keys(vm_prog.clone(), &bad_vm_weights, max_cycles, ligero_params)?;
    let (bad_vm_out, bad_vm_receipt) = prove_zkvm(&bad_vm_pk, &inputs)?;
    record(
        "zkvm_wrong_weights",
        "zkvm",
        verify_zkvm_receipt(
            &vm_vk.image_id,
            &vm_vk,
            &inputs,
            &bad_vm_out,
            &bad_vm_receipt,
        )
        .map(|_| ()),
    );

    // 12. Wrong journal output.
    let mut forged_vm_out = vm_out.clone();
    forged_vm_out[0] += 1;
    record(
        "zkvm_wrong_journal_output",
        "zkvm",
        verify_zkvm_receipt(
            &vm_vk.image_id,
            &vm_vk,
            &inputs,
            &forged_vm_out,
            &vm_receipt,
        )
        .map(|_| ()),
    );

    // 13. Wrong journal input.
    let mut forged_vm_in = inputs.clone();
    forged_vm_in[0] += 1;
    record(
        "zkvm_wrong_journal_input",
        "zkvm",
        verify_zkvm_receipt(&vm_vk.image_id, &vm_vk, &forged_vm_in, &vm_out, &vm_receipt)
            .map(|_| ()),
    );

    // 14. Corrupted internal execution trace witness.
    let (corr_out, corr_receipt) = prove_zkvm_with_corrupted_trace_for_test(&vm_pk, &inputs, 25)?;
    record(
        "zkvm_corrupted_trace_witness",
        "zkvm",
        verify_zkvm_receipt(&vm_vk.image_id, &vm_vk, &inputs, &corr_out, &corr_receipt).map(|_| ()),
    );

    // 15. Tampered Merkle spot-opened trace row.
    let mut bad_row_receipt = vm_receipt.clone();
    bad_row_receipt.trace_spot_openings[0].row.regs_after[0] += 1;
    record(
        "zkvm_tampered_merkle_trace_row",
        "zkvm",
        verify_zkvm_receipt(&vm_vk.image_id, &vm_vk, &inputs, &vm_out, &bad_row_receipt)
            .map(|_| ()),
    );

    // 16. Forged cycle bound exceeded.
    let mut bad_cycle_receipt = vm_receipt.clone();
    bad_cycle_receipt.cycles_executed = max_cycles + 1;
    record(
        "zkvm_cycle_bound_exceeded",
        "zkvm",
        verify_zkvm_receipt(
            &vm_vk.image_id,
            &vm_vk,
            &inputs,
            &vm_out,
            &bad_cycle_receipt,
        )
        .map(|_| ()),
    );

    Ok(checks)
}

/// Evaluates the explicit 27B interactive-decode negative regime from measured CPU timing.
pub fn evaluate_27b_negative_regime(calibration: &WorkloadBenchmark) -> NegativeRegime27B {
    let ns_per_constraint =
        (calibration.prove_mean_ms * 1_000_000.0) / (calibration.constraints_count.max(1) as f64);
    let single_token_sec = (ns_per_constraint * DENSE_27B_MACS_PER_TOKEN) / 1_000_000_000.0;
    let single_token_hours = single_token_sec / 3600.0;
    let usd_per_million_tokens = single_token_hours * COMMODITY_VCPU_USD_PER_HOUR * 1_000_000.0;

    NegativeRegime27B {
        calibration_workload: calibration.name.clone(),
        measured_ns_per_constraint: ns_per_constraint,
        dense_27b_constraints_per_token_lower_bound: DENSE_27B_MACS_PER_TOKEN,
        extrapolated_single_token_prove_sec: single_token_sec,
        extrapolated_single_token_prove_hours: single_token_hours,
        extrapolated_cost_usd_per_million_tokens: usd_per_million_tokens,
        verdict: "REJECTED_FOR_INTERACTIVE_27B_DECODE".to_string(),
    }
}

/// Runs the full multi-trial benchmark suite (3 Path A circuits + 3 Path B zkVM programs)
/// and the 16-vector adversarial verification matrix.
pub fn run_full_benchmark_suite(
    seed: u64,
    trials: usize,
    ligero_params: LigeroParams,
) -> Result<BenchmarkSuiteReport, ZkmlError> {
    let trials = trials.max(3);
    let mut rng = SplitMix64::new(seed);
    let mut workloads = Vec::new();

    // --- Path A.1: CompactClassifierV1 ---
    {
        let clf = CompactClassifierV1::compile();
        let weights = CompactClassifierV1::reference_weights();
        let (pk, vk) = setup_circuit_keys(clf.spec.clone(), &weights, ligero_params)?;
        let mut prove_ms = Vec::with_capacity(trials);
        let mut verify_ms = Vec::with_capacity(trials);
        let mut completed = 0usize;
        let mut proof_bytes = 0usize;
        let mut env_digest = String::new();

        for t in 0..(trials + 2) {
            let inputs: Vec<i64> = (0..8).map(|_| rng.range_i64(-16, 16)).collect();
            let t0 = Instant::now();
            let (outputs, witness) = CompactClassifierV1::evaluate_witness(&inputs, &weights)?;
            let proof = prove_circuit(&pk, &inputs, &outputs, &witness)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let report = verify_circuit_proof(&vk, &inputs, &outputs, &proof)?;
            let v_ms = t1.elapsed().as_secs_f64() * 1000.0;

            if t >= 2 {
                prove_ms.push(p_ms);
                verify_ms.push(v_ms);
                proof_bytes = report.proof_bytes;
                completed += 1;
                if t == 2 {
                    let env = HanzoProofEnvelope::from_circuit_proof(
                        &vk,
                        &proof,
                        &inputs,
                        &outputs,
                        "enso/route/classifier-v1",
                        "visor/job/bench-0",
                    );
                    env.verify_binding(&proof.to_bytes(), &inputs, &outputs)?;
                    env_digest = hex_encode(&env.envelope_digest);
                }
            }
        }
        workloads.push(summarize_timings(TimingSummaryInput {
            name: "compact_classifier_mlp_8x8x4",
            backend: "model_specific_circuit",
            constraints_count: clf.spec.constraints.len(),
            trace_cycles: 0,
            proof_bytes,
            prove_ms,
            verify_ms,
            completed,
            sample_envelope_digest: env_digest,
        }));
    }

    // --- Path A.2: KaiPolicyGateV1 ---
    {
        let gate = KaiPolicyGateV1::compile();
        let weights = KaiPolicyGateV1::reference_weights();
        let (pk, vk) = setup_circuit_keys(gate.spec.clone(), &weights, ligero_params)?;
        let mut prove_ms = Vec::with_capacity(trials);
        let mut verify_ms = Vec::with_capacity(trials);
        let mut completed = 0usize;
        let mut proof_bytes = 0usize;
        let mut env_digest = String::new();

        for t in 0..(trials + 2) {
            let inputs = vec![
                rng.range_i64(0, 150),
                rng.range_i64(20, 200),
                rng.range_i64(64, 512),
                rng.range_i64(1, 3),
                rng.range_i64(10, 100),
                rng.range_i64(0, 100),
            ];
            let t0 = Instant::now();
            let (outputs, witness) = KaiPolicyGateV1::evaluate_witness(&inputs, &weights)?;
            let proof = prove_circuit(&pk, &inputs, &outputs, &witness)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let report = verify_circuit_proof(&vk, &inputs, &outputs, &proof)?;
            let v_ms = t1.elapsed().as_secs_f64() * 1000.0;

            if t >= 2 {
                prove_ms.push(p_ms);
                verify_ms.push(v_ms);
                proof_bytes = report.proof_bytes;
                completed += 1;
                if t == 2 {
                    let env = HanzoProofEnvelope::from_circuit_proof(
                        &vk,
                        &proof,
                        &inputs,
                        &outputs,
                        "enso/route/kai-gate-v1",
                        "visor/job/bench-1",
                    );
                    env.verify_binding(&proof.to_bytes(), &inputs, &outputs)?;
                    env_digest = hex_encode(&env.envelope_digest);
                }
            }
        }
        workloads.push(summarize_timings(TimingSummaryInput {
            name: "kai_policy_gate_6f",
            backend: "model_specific_circuit",
            constraints_count: gate.spec.constraints.len(),
            trace_cycles: 0,
            proof_bytes,
            prove_ms,
            verify_ms,
            completed,
            sample_envelope_digest: env_digest,
        }));
    }

    // --- Path A.3: DeterministicEvalScorerV1 ---
    {
        let eval = DeterministicEvalScorerV1::compile();
        let weights = DeterministicEvalScorerV1::reference_weights();
        let (pk, vk) = setup_circuit_keys(eval.spec.clone(), &weights, ligero_params)?;
        let mut prove_ms = Vec::with_capacity(trials);
        let mut verify_ms = Vec::with_capacity(trials);
        let mut completed = 0usize;
        let mut proof_bytes = 0usize;
        let mut env_digest = String::new();

        for t in 0..(trials + 2) {
            let inputs: Vec<i64> = (0..16).map(|_| rng.range_i64(40, 100)).collect();
            let t0 = Instant::now();
            let (outputs, witness) =
                DeterministicEvalScorerV1::evaluate_witness(&inputs, &weights)?;
            let proof = prove_circuit(&pk, &inputs, &outputs, &witness)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let report = verify_circuit_proof(&vk, &inputs, &outputs, &proof)?;
            let v_ms = t1.elapsed().as_secs_f64() * 1000.0;

            if t >= 2 {
                prove_ms.push(p_ms);
                verify_ms.push(v_ms);
                proof_bytes = report.proof_bytes;
                completed += 1;
                if t == 2 {
                    let env = HanzoProofEnvelope::from_circuit_proof(
                        &vk,
                        &proof,
                        &inputs,
                        &outputs,
                        "enso/route/eval-scorer-v1",
                        "visor/job/bench-2",
                    );
                    env.verify_binding(&proof.to_bytes(), &inputs, &outputs)?;
                    env_digest = hex_encode(&env.envelope_digest);
                }
            }
        }
        workloads.push(summarize_timings(TimingSummaryInput {
            name: "deterministic_eval_scorer_8r",
            backend: "model_specific_circuit",
            constraints_count: eval.spec.constraints.len(),
            trace_cycles: 0,
            proof_bytes,
            prove_ms,
            verify_ms,
            completed,
            sample_envelope_digest: env_digest,
        }));
    }

    // --- Path B.1: zkVM Compact Linear Classifier ---
    {
        let (prog, weights) = build_zkvm_compact_classifier_program();
        let max_cycles = prog.instructions.len() + 8;
        let (pk, vk) = setup_zkvm_keys(prog, &weights, max_cycles, ligero_params)?;
        let mut prove_ms = Vec::with_capacity(trials);
        let mut verify_ms = Vec::with_capacity(trials);
        let mut completed = 0usize;
        let mut proof_bytes = 0usize;
        let mut trace_cycles = 0usize;
        let mut env_digest = String::new();

        for t in 0..(trials + 2) {
            let inputs: Vec<i64> = (0..8).map(|_| rng.range_i64(-16, 16)).collect();
            let t0 = Instant::now();
            let (outputs, receipt) = prove_zkvm(&pk, &inputs)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let report = verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &outputs, &receipt)?;
            let v_ms = t1.elapsed().as_secs_f64() * 1000.0;

            if t >= 2 {
                prove_ms.push(p_ms);
                verify_ms.push(v_ms);
                proof_bytes = report.receipt_bytes;
                trace_cycles = report.cycles_executed;
                completed += 1;
                if t == 2 {
                    let env = HanzoProofEnvelope::from_zkvm_receipt(
                        &vk,
                        &receipt,
                        &inputs,
                        &outputs,
                        "enso/route/zkvm-classifier-v1",
                        "visor/job/bench-3",
                    );
                    env.verify_binding(&receipt.to_bytes(), &inputs, &outputs)?;
                    env_digest = hex_encode(&env.envelope_digest);
                }
            }
        }
        workloads.push(summarize_timings(TimingSummaryInput {
            name: "zkvm_compact_classifier_linear_8x4",
            backend: "general_zkvm",
            constraints_count: vk.trace_vk.spec.constraints.len(),
            trace_cycles,
            proof_bytes,
            prove_ms,
            verify_ms,
            completed,
            sample_envelope_digest: env_digest,
        }));
    }

    // --- Path B.2: zkVM Kai Policy Gate ---
    {
        let (prog, weights) = build_zkvm_kai_policy_program();
        let max_cycles = prog.instructions.len() + 8;
        let (pk, vk) = setup_zkvm_keys(prog, &weights, max_cycles, ligero_params)?;
        let mut prove_ms = Vec::with_capacity(trials);
        let mut verify_ms = Vec::with_capacity(trials);
        let mut completed = 0usize;
        let mut proof_bytes = 0usize;
        let mut trace_cycles = 0usize;
        let mut env_digest = String::new();

        for t in 0..(trials + 2) {
            let inputs = vec![
                rng.range_i64(0, 150),
                rng.range_i64(20, 200),
                rng.range_i64(64, 512),
                rng.range_i64(1, 3),
                rng.range_i64(10, 100),
                rng.range_i64(0, 100),
            ];
            let t0 = Instant::now();
            let (outputs, receipt) = prove_zkvm(&pk, &inputs)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let report = verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &outputs, &receipt)?;
            let v_ms = t1.elapsed().as_secs_f64() * 1000.0;

            if t >= 2 {
                prove_ms.push(p_ms);
                verify_ms.push(v_ms);
                proof_bytes = report.receipt_bytes;
                trace_cycles = report.cycles_executed;
                completed += 1;
                if t == 2 {
                    let env = HanzoProofEnvelope::from_zkvm_receipt(
                        &vk,
                        &receipt,
                        &inputs,
                        &outputs,
                        "enso/route/zkvm-kai-gate-v1",
                        "visor/job/bench-4",
                    );
                    env.verify_binding(&receipt.to_bytes(), &inputs, &outputs)?;
                    env_digest = hex_encode(&env.envelope_digest);
                }
            }
        }
        workloads.push(summarize_timings(TimingSummaryInput {
            name: "zkvm_kai_policy_gate_6f",
            backend: "general_zkvm",
            constraints_count: vk.trace_vk.spec.constraints.len(),
            trace_cycles,
            proof_bytes,
            prove_ms,
            verify_ms,
            completed,
            sample_envelope_digest: env_digest,
        }));
    }

    // --- Path B.3: zkVM Deterministic Eval Scorer ---
    {
        let (prog, weights) = build_zkvm_eval_scorer_program();
        let max_cycles = prog.instructions.len() + 8;
        let (pk, vk) = setup_zkvm_keys(prog, &weights, max_cycles, ligero_params)?;
        let mut prove_ms = Vec::with_capacity(trials);
        let mut verify_ms = Vec::with_capacity(trials);
        let mut completed = 0usize;
        let mut proof_bytes = 0usize;
        let mut trace_cycles = 0usize;
        let mut env_digest = String::new();

        for t in 0..(trials + 2) {
            let inputs: Vec<i64> = (0..16).map(|_| rng.range_i64(40, 100)).collect();
            let t0 = Instant::now();
            let (outputs, receipt) = prove_zkvm(&pk, &inputs)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let report = verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &outputs, &receipt)?;
            let v_ms = t1.elapsed().as_secs_f64() * 1000.0;

            if t >= 2 {
                prove_ms.push(p_ms);
                verify_ms.push(v_ms);
                proof_bytes = report.receipt_bytes;
                trace_cycles = report.cycles_executed;
                completed += 1;
                if t == 2 {
                    let env = HanzoProofEnvelope::from_zkvm_receipt(
                        &vk,
                        &receipt,
                        &inputs,
                        &outputs,
                        "enso/route/zkvm-eval-scorer-v1",
                        "visor/job/bench-5",
                    );
                    env.verify_binding(&receipt.to_bytes(), &inputs, &outputs)?;
                    env_digest = hex_encode(&env.envelope_digest);
                }
            }
        }
        workloads.push(summarize_timings(TimingSummaryInput {
            name: "zkvm_deterministic_eval_scorer_8r",
            backend: "general_zkvm",
            constraints_count: vk.trace_vk.spec.constraints.len(),
            trace_cycles,
            proof_bytes,
            prove_ms,
            verify_ms,
            completed,
            sample_envelope_digest: env_digest,
        }));
    }

    let adversarial_checks = run_adversarial_matrix(ligero_params)?;
    let rejected_count = adversarial_checks.iter().filter(|c| c.rejected).count();
    let adversarial_detection_rate =
        (rejected_count as f64) / (adversarial_checks.len().max(1) as f64);

    let negative_regime_27b = evaluate_27b_negative_regime(&workloads[0]);

    Ok(BenchmarkSuiteReport {
        protocol_version: PROTOCOL_VERSION.to_string(),
        seed,
        trials,
        ligero_blowup: ligero_params.blowup,
        ligero_queries: ligero_params.num_queries,
        ligero_proximity_soundness_bits: ligero_params.proximity_soundness_bits(),
        sumcheck_field_soundness_bits: 128.0,
        workloads,
        adversarial_checks,
        adversarial_detection_rate,
        negative_regime_27b,
        decentralization_score: 1.0,
        sovereignty_score: 1.0,
    })
}

impl BenchmarkSuiteReport {
    /// Serializes the benchmark report into canonical JSON and returns `(json, sha256_hex)`.
    pub fn to_canonical_json_with_sha256(&self) -> (String, String) {
        let mut body = String::new();
        body.push_str("{\n");
        body.push_str(&format!(
            "  \"protocol_version\": \"{}\",\n",
            escape_json(&self.protocol_version)
        ));
        body.push_str(&format!(
            "  \"trust_tier\": \"{}\",\n",
            TrustTier::CryptographicProof.as_str()
        ));
        body.push_str("  \"hip\": \"HIP-1350\",\n");
        body.push_str("  \"issue\": \"hanzoai/engine#68\",\n");
        body.push_str(&format!("  \"seed\": {},\n", self.seed));
        body.push_str(&format!("  \"trials\": {},\n", self.trials));
        body.push_str("  \"cryptographic_parameters\": {\n");
        body.push_str("    \"base_field\": \"Goldilocks (2^64 - 2^32 + 1)\",\n");
        body.push_str("    \"challenge_field\": \"F_{p^2} = F_p[u]/(u^2 - 7)\",\n");
        body.push_str(&format!("    \"ligero_blowup\": {},\n", self.ligero_blowup));
        body.push_str(&format!(
            "    \"ligero_queries\": {},\n",
            self.ligero_queries
        ));
        body.push_str(&format!(
            "    \"ligero_proximity_soundness_bits\": {:.2},\n",
            self.ligero_proximity_soundness_bits
        ));
        body.push_str(&format!(
            "    \"sumcheck_field_soundness_bits\": {:.2}\n",
            self.sumcheck_field_soundness_bits
        ));
        body.push_str("  },\n");

        body.push_str("  \"workloads\": [\n");
        for (idx, w) in self.workloads.iter().enumerate() {
            let comma = if idx + 1 == self.workloads.len() {
                ""
            } else {
                ","
            };
            body.push_str("    {\n");
            body.push_str(&format!("      \"name\": \"{}\",\n", escape_json(&w.name)));
            body.push_str(&format!(
                "      \"backend\": \"{}\",\n",
                escape_json(&w.backend)
            ));
            body.push_str(&format!("      \"trials\": {},\n", w.trials));
            body.push_str(&format!(
                "      \"constraints_count\": {},\n",
                w.constraints_count
            ));
            body.push_str(&format!("      \"trace_cycles\": {},\n", w.trace_cycles));
            body.push_str(&format!("      \"proof_bytes\": {},\n", w.proof_bytes));
            body.push_str(&format!("      \"prove_p50_ms\": {:.4},\n", w.prove_p50_ms));
            body.push_str(&format!("      \"prove_p95_ms\": {:.4},\n", w.prove_p95_ms));
            body.push_str(&format!("      \"prove_p99_ms\": {:.4},\n", w.prove_p99_ms));
            body.push_str(&format!(
                "      \"prove_mean_ms\": {:.4},\n",
                w.prove_mean_ms
            ));
            body.push_str(&format!(
                "      \"verify_p50_ms\": {:.4},\n",
                w.verify_p50_ms
            ));
            body.push_str(&format!(
                "      \"verify_p95_ms\": {:.4},\n",
                w.verify_p95_ms
            ));
            body.push_str(&format!(
                "      \"verify_p99_ms\": {:.4},\n",
                w.verify_p99_ms
            ));
            body.push_str(&format!(
                "      \"verify_mean_ms\": {:.4},\n",
                w.verify_mean_ms
            ));
            body.push_str(&format!(
                "      \"prove_throughput_per_sec\": {:.2},\n",
                w.prove_throughput_per_sec
            ));
            body.push_str(&format!(
                "      \"verify_throughput_per_sec\": {:.2},\n",
                w.verify_throughput_per_sec
            ));
            body.push_str(&format!(
                "      \"completion_rate\": {:.4},\n",
                w.completion_rate
            ));
            body.push_str(&format!(
                "      \"amortized_cost_usd_per_million_tokens\": {:.6},\n",
                w.amortized_cost_usd_per_million_tokens
            ));
            body.push_str(&format!(
                "      \"sample_envelope_digest\": \"{}\"\n",
                escape_json(&w.sample_envelope_digest)
            ));
            body.push_str(&format!("    }}{comma}\n"));
        }
        body.push_str("  ],\n");

        body.push_str("  \"adversarial_matrix\": {\n");
        let rejected = self
            .adversarial_checks
            .iter()
            .filter(|c| c.rejected)
            .count();
        body.push_str(&format!(
            "    \"total_vectors\": {},\n",
            self.adversarial_checks.len()
        ));
        body.push_str(&format!("    \"rejected_vectors\": {rejected},\n"));
        body.push_str(&format!(
            "    \"detection_rate\": {:.4},\n",
            self.adversarial_detection_rate
        ));
        body.push_str("    \"checks\": [\n");
        for (idx, c) in self.adversarial_checks.iter().enumerate() {
            let comma = if idx + 1 == self.adversarial_checks.len() {
                ""
            } else {
                ","
            };
            body.push_str(&format!(
                "      {{\"attack_id\": \"{}\", \"target_path\": \"{}\", \"rejected\": {}, \"rejection_reason\": \"{}\"}}{comma}\n",
                escape_json(&c.attack_id),
                escape_json(&c.target_path),
                c.rejected,
                escape_json(&c.rejection_reason),
            ));
        }
        body.push_str("    ]\n");
        body.push_str("  },\n");

        let n = &self.negative_regime_27b;
        body.push_str("  \"negative_regime_27b\": {\n");
        body.push_str(&format!(
            "    \"calibration_workload\": \"{}\",\n",
            escape_json(&n.calibration_workload)
        ));
        body.push_str(&format!(
            "    \"measured_ns_per_constraint\": {:.2},\n",
            n.measured_ns_per_constraint
        ));
        body.push_str(&format!(
            "    \"dense_27b_constraints_per_token_lower_bound\": {:.0},\n",
            n.dense_27b_constraints_per_token_lower_bound
        ));
        body.push_str(&format!(
            "    \"extrapolated_single_token_prove_sec\": {:.2},\n",
            n.extrapolated_single_token_prove_sec
        ));
        body.push_str(&format!(
            "    \"extrapolated_single_token_prove_hours\": {:.2},\n",
            n.extrapolated_single_token_prove_hours
        ));
        body.push_str(&format!(
            "    \"extrapolated_cost_usd_per_million_tokens\": {:.2},\n",
            n.extrapolated_cost_usd_per_million_tokens
        ));
        body.push_str(&format!(
            "    \"verdict\": \"{}\"\n",
            escape_json(&n.verdict)
        ));
        body.push_str("  },\n");

        body.push_str("  \"hip_1350_five_axis\": {\n");
        body.push_str(&format!(
            "    \"cost_model_vcpu_usd_per_hour\": {:.4},\n",
            COMMODITY_VCPU_USD_PER_HOUR
        ));
        body.push_str(&format!(
            "    \"tokens_per_decision_request\": {:.0},\n",
            TOKENS_PER_DECISION_REQUEST
        ));
        body.push_str(&format!(
            "    \"trust_adversarial_detection_rate\": {:.4},\n",
            self.adversarial_detection_rate
        ));
        body.push_str(&format!(
            "    \"decentralization_hanzo_independent_share\": {:.4},\n",
            self.decentralization_score
        ));
        body.push_str(&format!(
            "    \"sovereignty_export_verify_fork_share\": {:.4}\n",
            self.sovereignty_score
        ));
        body.push_str("  }\n");
        body.push_str("}\n");

        let digest_hex = hex_encode(&sha256(body.as_bytes()));
        (body, digest_hex)
    }
}
