// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive acceptance, adversarial rejection, determinism, and HIP-1350
//! invariant test suite for `hanzo-zkml` (`hanzoai/engine#68`).

use hanzo_zkml::bench_harness::{
    evaluate_27b_negative_regime, run_adversarial_matrix, run_full_benchmark_suite,
};
use hanzo_zkml::circuit::{
    prove_circuit, prove_circuit_unchecked_for_adversarial_test, setup_circuit_keys,
    verify_circuit_proof, CircuitVersion, CompactClassifierV1, DeterministicEvalScorerV1,
    KaiPolicyGateV1,
};
use hanzo_zkml::envelope::{HanzoProofEnvelope, ProofBackend, ENVELOPE_SCHEMA_VERSION};
use hanzo_zkml::field::{Fp, MAX_SIGNED_MAGNITUDE};
use hanzo_zkml::merkle::LigeroParams;
use hanzo_zkml::zkvm::{
    build_zkvm_compact_classifier_program, build_zkvm_eval_scorer_program,
    build_zkvm_kai_policy_program, prove_zkvm, prove_zkvm_with_corrupted_trace_for_test,
    setup_zkvm_keys, verify_zkvm_receipt, Instruction, LoadSource, ZkVmProgram, ZkVmVersion,
};
use hanzo_zkml::{TrustTier, ZkmlError};

// ============================================================================
// Path A: Model-Specific Circuits — Positive & Determinism Tests
// ============================================================================

#[test]
fn circuit_compact_classifier_proves_verifies_and_is_deterministic() {
    let clf = CompactClassifierV1::compile();
    let weights = CompactClassifierV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(clf.spec, &weights, LigeroParams::default()).unwrap();

    let inputs = vec![6, -4, 11, 3, -7, 5, 2, -3];
    let (outputs, witness) = CompactClassifierV1::evaluate_witness(&inputs, &weights).unwrap();
    let proof1 = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();
    let proof2 = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();

    // Deterministic canonical wire encoding.
    assert_eq!(proof1.to_bytes(), proof2.to_bytes());
    assert_eq!(proof1.to_bytes().len(), proof1.byte_len());

    let report = verify_circuit_proof(&vk, &inputs, &outputs, &proof1).unwrap();
    assert_eq!(report.circuit_id, "hanzo.compact_classifier.mlp_8x8x4");
    assert_eq!(report.version, CircuitVersion::new(1, 0, 0));
    assert!(report.constraints_verified > 300);

    // Check predicted class is genuine argmax of the 4 logits.
    let logits = &outputs[0..4];
    let pred = outputs[4] as usize;
    assert!(logits.iter().all(|&l| logits[pred] >= l));
}

#[test]
fn circuit_kai_policy_gate_covers_all_three_routing_decisions() {
    let gate = KaiPolicyGateV1::compile();
    let weights = KaiPolicyGateV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(gate.spec, &weights, LigeroParams::default()).unwrap();

    // Case 0: ROUTE_LOCAL_FAST (score < 1200)
    let low_risk = vec![10, 100, 128, 1, 50, 80];
    let (out0, wit0) = KaiPolicyGateV1::evaluate_witness(&low_risk, &weights).unwrap();
    assert_eq!(out0[1], 0);
    let p0 = prove_circuit(&pk, &low_risk, &out0, &wit0).unwrap();
    verify_circuit_proof(&vk, &low_risk, &out0, &p0).unwrap();

    // Case 1: ROUTE_VERIFIED_ESCROW (1200 <= score < 2800)
    let med_risk = vec![80, 50, 300, 2, 20, 20];
    let (out1, wit1) = KaiPolicyGateV1::evaluate_witness(&med_risk, &weights).unwrap();
    assert_eq!(out1[1], 1);
    let p1 = prove_circuit(&pk, &med_risk, &out1, &wit1).unwrap();
    verify_circuit_proof(&vk, &med_risk, &out1, &p1).unwrap();

    // Case 2: REJECT_POLICY_VIOLATION (score >= 2800)
    let high_risk = vec![200, 10, 512, 3, 5, 0];
    let (out2, wit2) = KaiPolicyGateV1::evaluate_witness(&high_risk, &weights).unwrap();
    assert_eq!(out2[1], 2);
    let p2 = prove_circuit(&pk, &high_risk, &out2, &wit2).unwrap();
    verify_circuit_proof(&vk, &high_risk, &out2, &p2).unwrap();
}

#[test]
fn circuit_deterministic_eval_scorer_passes_and_fails_correctly() {
    let eval = DeterministicEvalScorerV1::compile();
    let weights = DeterministicEvalScorerV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(eval.spec, &weights, LigeroParams::default()).unwrap();

    // Passing rubric (high scores close to targets).
    let pass_in = vec![
        90, 88, 92, 95, 85, 91, 89, 93, 90, 90, 90, 95, 85, 90, 90, 92,
    ];
    let (out_pass, wit_pass) =
        DeterministicEvalScorerV1::evaluate_witness(&pass_in, &weights).unwrap();
    assert_eq!(out_pass[3], 1);
    let p_pass = prove_circuit(&pk, &pass_in, &out_pass, &wit_pass).unwrap();
    verify_circuit_proof(&vk, &pass_in, &out_pass, &p_pass).unwrap();

    // Failing rubric (low scores far from targets -> large L2 penalty).
    let fail_in = vec![
        30, 40, 25, 35, 20, 45, 30, 35, 90, 90, 90, 90, 90, 90, 90, 90,
    ];
    let (out_fail, wit_fail) =
        DeterministicEvalScorerV1::evaluate_witness(&fail_in, &weights).unwrap();
    assert_eq!(out_fail[3], 0);
    let p_fail = prove_circuit(&pk, &fail_in, &out_fail, &wit_fail).unwrap();
    verify_circuit_proof(&vk, &fail_in, &out_fail, &p_fail).unwrap();
}

// ============================================================================
// Path A: Adversarial Rejection Tests
// ============================================================================

#[test]
fn circuit_rejects_wrong_weights() {
    let clf = CompactClassifierV1::compile();
    let weights = CompactClassifierV1::reference_weights();
    let (_, vk) = setup_circuit_keys(clf.spec.clone(), &weights, LigeroParams::default()).unwrap();

    let mut wrong_weights = weights.clone();
    wrong_weights[25] += 1;
    let (bad_pk, _) =
        setup_circuit_keys(clf.spec, &wrong_weights, LigeroParams::default()).unwrap();

    let inputs = vec![3, -2, 7, 1, -5, 4, 2, -1];
    let (bad_out, bad_wit) =
        CompactClassifierV1::evaluate_witness(&inputs, &wrong_weights).unwrap();
    let bad_proof = prove_circuit(&bad_pk, &inputs, &bad_out, &bad_wit).unwrap();

    let err = verify_circuit_proof(&vk, &inputs, &bad_out, &bad_proof).unwrap_err();
    assert_eq!(err, ZkmlError::WeightsCommitmentMismatch);
}

#[test]
fn circuit_rejects_wrong_outputs_and_wrong_argmax() {
    let clf = CompactClassifierV1::compile();
    let weights = CompactClassifierV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(clf.spec, &weights, LigeroParams::default()).unwrap();

    let inputs = vec![5, -3, 8, 2, -4, 6, 1, -2];
    let (outputs, witness) = CompactClassifierV1::evaluate_witness(&inputs, &weights).unwrap();
    let proof = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();

    // Alter logit 1 by +1.
    let mut bad_logit = outputs.clone();
    bad_logit[1] += 1;
    assert!(verify_circuit_proof(&vk, &inputs, &bad_logit, &proof).is_err());

    // Flip predicted class.
    let mut bad_class = outputs.clone();
    bad_class[4] = (bad_class[4] + 1) % 4;
    assert!(verify_circuit_proof(&vk, &inputs, &bad_class, &proof).is_err());
}

#[test]
fn circuit_rejects_wrong_version_and_wrong_digest() {
    let gate = KaiPolicyGateV1::compile();
    let weights = KaiPolicyGateV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(gate.spec, &weights, LigeroParams::default()).unwrap();

    let inputs = vec![80, 50, 300, 2, 20, 20];
    let (outputs, witness) = KaiPolicyGateV1::evaluate_witness(&inputs, &weights).unwrap();
    let proof = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();

    let mut wrong_ver = proof.clone();
    wrong_ver.version = CircuitVersion::new(1, 1, 0);
    assert!(matches!(
        verify_circuit_proof(&vk, &inputs, &outputs, &wrong_ver),
        Err(ZkmlError::CircuitVersionMismatch { .. })
    ));

    let mut wrong_digest = proof;
    wrong_digest.circuit_digest[31] ^= 0x01;
    assert_eq!(
        verify_circuit_proof(&vk, &inputs, &outputs, &wrong_digest),
        Err(ZkmlError::CircuitDigestMismatch)
    );
}

#[test]
fn circuit_rejects_violated_intermediate_constraint_and_forged_witness() {
    let clf = CompactClassifierV1::compile();
    let weights = CompactClassifierV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(clf.spec, &weights, LigeroParams::default()).unwrap();

    let inputs = vec![4, -1, 6, -3, 2, 5, -2, 3];
    let (outputs, mut witness) = CompactClassifierV1::evaluate_witness(&inputs, &weights).unwrap();

    // Corrupt an internal ReLU/hidden witness wire.
    witness[20] += Fp::ONE;

    // Honest prover refuses unsatisfied witness.
    assert!(matches!(
        prove_circuit(&pk, &inputs, &outputs, &witness),
        Err(ZkmlError::UnsatisfiedConstraint { .. })
    ));

    // Malicious prover skipping local satisfaction check is caught cryptographically by verifier.
    let forged =
        prove_circuit_unchecked_for_adversarial_test(&pk, &inputs, &outputs, &witness).unwrap();
    assert!(matches!(
        verify_circuit_proof(&vk, &inputs, &outputs, &forged),
        Err(ZkmlError::SumcheckFailed { .. })
    ));
}

#[test]
fn circuit_rejects_wrong_preprocessing_input_and_cross_domain_replay() {
    let clf = CompactClassifierV1::compile();
    let weights = CompactClassifierV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(clf.spec, &weights, LigeroParams::default()).unwrap();

    let inputs = vec![4, -1, 6, -3, 2, 5, -2, 3];
    let (outputs, witness) = CompactClassifierV1::evaluate_witness(&inputs, &weights).unwrap();
    let proof = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();

    let mut wrong_inputs = inputs.clone();
    wrong_inputs[0] += 1;
    assert!(verify_circuit_proof(&vk, &wrong_inputs, &outputs, &proof).is_err());

    let mut replayed = proof;
    replayed.domain_tag = "hanzo.zkml.forged_domain.v9".to_string();
    assert!(matches!(
        verify_circuit_proof(&vk, &inputs, &outputs, &replayed),
        Err(ZkmlError::DomainSeparationMismatch { .. })
    ));
}

#[test]
fn circuit_rejects_tampered_merkle_opening_and_sumcheck_round() {
    let gate = KaiPolicyGateV1::compile();
    let weights = KaiPolicyGateV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(gate.spec, &weights, LigeroParams::default()).unwrap();

    let inputs = vec![80, 50, 300, 2, 20, 20];
    let (outputs, witness) = KaiPolicyGateV1::evaluate_witness(&inputs, &weights).unwrap();
    let proof = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();

    // Tamper one Sumcheck evaluation.
    let mut bad_sc = proof.clone();
    bad_sc.outer_sumcheck_rounds[0][2].c0 += Fp::ONE;
    assert!(matches!(
        verify_circuit_proof(&vk, &inputs, &outputs, &bad_sc),
        Err(ZkmlError::SumcheckFailed { .. })
    ));

    // Tamper one Merkle sibling inside witness PCS opening.
    let mut bad_merkle = proof;
    bad_merkle.witness_opening.opened_columns[0]
        .merkle_proof
        .siblings[0][0] ^= 0x80;
    assert_eq!(
        verify_circuit_proof(&vk, &inputs, &outputs, &bad_merkle),
        Err(ZkmlError::PcsVerificationFailed { slice: "witness" })
    );
}

#[test]
fn circuit_rejects_out_of_field_range_inputs() {
    let gate = KaiPolicyGateV1::compile();
    let weights = KaiPolicyGateV1::reference_weights();
    let (_, vk) = setup_circuit_keys(gate.spec, &weights, LigeroParams::default()).unwrap();
    let (pk, _) = setup_circuit_keys(
        KaiPolicyGateV1::compile().spec,
        &weights,
        LigeroParams::default(),
    )
    .unwrap();

    let valid_inputs = vec![10, 20, 30, 1, 10, 10];
    let (outputs, witness) = KaiPolicyGateV1::evaluate_witness(&valid_inputs, &weights).unwrap();
    let proof = prove_circuit(&pk, &valid_inputs, &outputs, &witness).unwrap();

    let overflow_inputs = vec![MAX_SIGNED_MAGNITUDE + 1, 20, 30, 1, 10, 10];
    assert!(matches!(
        verify_circuit_proof(&vk, &overflow_inputs, &outputs, &proof),
        Err(ZkmlError::FieldRangeExceeded { .. })
    ));
}

// ============================================================================
// Path B: General Deterministic Register zkVM — Positive & Equivalence Tests
// ============================================================================

#[test]
fn zkvm_kai_policy_matches_circuit_output_and_verifies() {
    let (prog, vm_weights) = build_zkvm_kai_policy_program();
    let max_cycles = prog.instructions.len() + 4;
    let (vm_pk, vm_vk) =
        setup_zkvm_keys(prog, &vm_weights, max_cycles, LigeroParams::default()).unwrap();

    let circuit_weights = KaiPolicyGateV1::reference_weights();

    for inputs in [
        vec![10, 100, 128, 1, 50, 80],
        vec![80, 50, 300, 2, 20, 20],
        vec![200, 10, 512, 3, 5, 0],
    ] {
        let (c_out, _) = KaiPolicyGateV1::evaluate_witness(&inputs, &circuit_weights).unwrap();
        let (vm_out, receipt) = prove_zkvm(&vm_pk, &inputs).unwrap();

        // Path A and Path B agree bit-for-bit on both score and routing decision!
        assert_eq!(vm_out, c_out);

        let report =
            verify_zkvm_receipt(&vm_vk.image_id, &vm_vk, &inputs, &vm_out, &receipt).unwrap();
        assert_eq!(report.image_id, vm_vk.image_id);
        assert_eq!(report.cycles_executed, vm_vk.program.instructions.len());
    }
}

#[test]
fn zkvm_eval_scorer_and_compact_classifier_verify_deterministically() {
    // 1. Deterministic Eval Scorer
    let (eval_prog, eval_weights) = build_zkvm_eval_scorer_program();
    let (e_pk, e_vk) = setup_zkvm_keys(
        eval_prog.clone(),
        &eval_weights,
        eval_prog.instructions.len() + 8,
        LigeroParams::default(),
    )
    .unwrap();
    let eval_in = vec![
        85, 90, 88, 92, 80, 86, 84, 89, 85, 88, 90, 90, 82, 85, 85, 90,
    ];
    let (e_out, e_receipt) = prove_zkvm(&e_pk, &eval_in).unwrap();
    verify_zkvm_receipt(&e_vk.image_id, &e_vk, &eval_in, &e_out, &e_receipt).unwrap();
    assert_eq!(e_out[3], 1);

    // 2. Compact Linear Classifier exercising all 8 opcodes including SRA + CMP_GT + SELECT.
    let (clf_prog, clf_weights) = build_zkvm_compact_classifier_program();
    let (c_pk, c_vk) = setup_zkvm_keys(
        clf_prog.clone(),
        &clf_weights,
        clf_prog.instructions.len() + 8,
        LigeroParams::default(),
    )
    .unwrap();
    let clf_in = vec![7, -5, 9, 2, -6, 4, 3, -1];
    let (c_out1, c_receipt1) = prove_zkvm(&c_pk, &clf_in).unwrap();
    let (c_out2, c_receipt2) = prove_zkvm(&c_pk, &clf_in).unwrap();
    assert_eq!(c_out1, c_out2);
    assert_eq!(c_receipt1.to_bytes(), c_receipt2.to_bytes());
    assert_eq!(c_receipt1.to_bytes().len(), c_receipt1.byte_len());
    verify_zkvm_receipt(&c_vk.image_id, &c_vk, &clf_in, &c_out1, &c_receipt1).unwrap();

    // Verify argmax class matches max logit.
    let pred = c_out1[4] as usize;
    assert!(c_out1[0..4].iter().all(|&l| c_out1[pred] >= l));
}

#[test]
fn zkvm_sra_handles_negative_values_algebraically_exact() {
    // Program computing (-5 >> 2 == -2) and (-1 >> 3 == -1) via SRA.
    let prog = ZkVmProgram {
        program_id: "test.sra_negative".to_string(),
        num_inputs: 2,
        num_outputs: 2,
        num_weights: 1,
        instructions: vec![
            Instruction::Load {
                dst: 0,
                src: LoadSource::Input(0),
            },
            Instruction::Load {
                dst: 1,
                src: LoadSource::Input(1),
            },
            Instruction::Sra {
                dst: 2,
                a: 0,
                shift: 2,
            },
            Instruction::Sra {
                dst: 3,
                a: 1,
                shift: 3,
            },
            Instruction::Halt {
                out_regs: vec![2, 3],
            },
        ],
    };
    let (pk, vk) = setup_zkvm_keys(prog, &[0], 16, LigeroParams::default()).unwrap();
    let (out, receipt) = prove_zkvm(&pk, &[-5, -1]).unwrap();
    assert_eq!(out, vec![-2, -1]);
    verify_zkvm_receipt(&vk.image_id, &vk, &[-5, -1], &out, &receipt).unwrap();
}

// ============================================================================
// Path B: General zkVM Adversarial Rejection Tests
// ============================================================================

#[test]
fn zkvm_rejects_wrong_image_id_and_modified_program() {
    let (prog, weights) = build_zkvm_kai_policy_program();
    let max_cycles = prog.instructions.len() + 8;
    let (pk, vk) =
        setup_zkvm_keys(prog.clone(), &weights, max_cycles, LigeroParams::default()).unwrap();
    let inputs = vec![80, 50, 300, 2, 20, 20];
    let (out, receipt) = prove_zkvm(&pk, &inputs).unwrap();

    // Mutate expected_image_id.
    let mut wrong_id = vk.image_id;
    wrong_id[0] ^= 0xFF;
    assert_eq!(
        verify_zkvm_receipt(&wrong_id, &vk, &inputs, &out, &receipt),
        Err(ZkmlError::ImageIdMismatch)
    );

    // Mutate an instruction in a forked program -> different ImageID.
    let mut modified_prog = prog;
    modified_prog.instructions[0] = Instruction::Load {
        dst: 0,
        src: LoadSource::Imm(999),
    };
    let (_, mod_vk) =
        setup_zkvm_keys(modified_prog, &weights, max_cycles, LigeroParams::default()).unwrap();
    assert_ne!(mod_vk.image_id, vk.image_id);
    assert_eq!(
        verify_zkvm_receipt(&mod_vk.image_id, &mod_vk, &inputs, &out, &receipt),
        Err(ZkmlError::ImageIdMismatch)
    );
}

#[test]
fn zkvm_rejects_wrong_weights_and_wrong_journal() {
    let (prog, weights) = build_zkvm_kai_policy_program();
    let max_cycles = prog.instructions.len() + 8;
    let (pk, vk) =
        setup_zkvm_keys(prog.clone(), &weights, max_cycles, LigeroParams::default()).unwrap();
    let inputs = vec![80, 50, 300, 2, 20, 20];
    let (out, receipt) = prove_zkvm(&pk, &inputs).unwrap();

    // Wrong journal output.
    let mut bad_out = out.clone();
    bad_out[1] = 2;
    assert!(verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &bad_out, &receipt).is_err());

    // Wrong journal input.
    let mut bad_in = inputs.clone();
    bad_in[0] += 1;
    assert!(verify_zkvm_receipt(&vk.image_id, &vk, &bad_in, &out, &receipt).is_err());

    // Wrong weights table committed into another prover.
    let mut bad_w = weights;
    bad_w[0] += 1;
    let (bad_pk, _) = setup_zkvm_keys(prog, &bad_w, max_cycles, LigeroParams::default()).unwrap();
    let (other_out, other_receipt) = prove_zkvm(&bad_pk, &inputs).unwrap();
    assert_eq!(
        verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &other_out, &other_receipt),
        Err(ZkmlError::ImageIdMismatch)
    );
}

#[test]
fn zkvm_rejects_corrupted_trace_step_merkle_tamper_and_cycle_bound() {
    let (prog, weights) = build_zkvm_compact_classifier_program();
    let max_cycles = prog.instructions.len() + 8;
    let (pk, vk) = setup_zkvm_keys(prog, &weights, max_cycles, LigeroParams::default()).unwrap();
    let inputs = vec![5, -3, 8, 2, -4, 6, 1, -2];
    let (out, receipt) = prove_zkvm(&pk, &inputs).unwrap();

    // Corrupted internal trace witness.
    let (corr_out, corr_receipt) =
        prove_zkvm_with_corrupted_trace_for_test(&pk, &inputs, 20).unwrap();
    assert!(verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &corr_out, &corr_receipt).is_err());

    // Tampered Merkle trace row opening.
    let mut bad_row = receipt.clone();
    bad_row.trace_spot_openings[0].row.regs_after[1] += 1;
    assert!(matches!(
        verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &out, &bad_row),
        Err(ZkmlError::TraceVerificationFailed { .. })
    ));

    // Exceeded cycle bound.
    let mut bad_cycles = receipt.clone();
    bad_cycles.cycles_executed = max_cycles + 1;
    assert!(matches!(
        verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &out, &bad_cycles),
        Err(ZkmlError::CycleBoundExceeded { .. })
    ));

    // Wrong VM version.
    let mut bad_ver = receipt;
    bad_ver.vm_version = ZkVmVersion::new(9, 0, 0);
    assert!(matches!(
        verify_zkvm_receipt(&vk.image_id, &vk, &inputs, &out, &bad_ver),
        Err(ZkmlError::VmVersionMismatch { .. })
    ));
}

// ============================================================================
// Versioned HanzoProofEnvelope v1 & HIP-1350 Sovereign Interop Tests
// ============================================================================

#[test]
fn envelope_v1_roundtrips_json_and_enforces_hip1350_invariants() {
    let gate = KaiPolicyGateV1::compile();
    let weights = KaiPolicyGateV1::reference_weights();
    let (pk, vk) = setup_circuit_keys(gate.spec, &weights, LigeroParams::default()).unwrap();
    let inputs = vec![80, 50, 300, 2, 20, 20];
    let (outputs, witness) = KaiPolicyGateV1::evaluate_witness(&inputs, &weights).unwrap();
    let proof = prove_circuit(&pk, &inputs, &outputs, &witness).unwrap();
    let proof_bytes = proof.to_bytes();

    let env = HanzoProofEnvelope::from_circuit_proof(
        &vk,
        &proof,
        &inputs,
        &outputs,
        "enso/route/self-hosted-kai",
        "visor/job/42",
    );
    assert_eq!(env.schema_version, ENVELOPE_SCHEMA_VERSION);
    assert_eq!(env.trust_tier, TrustTier::CryptographicProof);
    assert_eq!(env.proof_backend, ProofBackend::ModelSpecificCircuit);
    assert!(!env.routing_context.hanzo_service_required);
    assert!(env.artifact_refs.artifact_cid.starts_with("vfs://sha256/"));
    assert!(env.artifact_refs.weights_cid.starts_with("vfs://sha256/"));

    // Canonical JSON roundtrip.
    let json = env.to_canonical_json();
    let parsed = HanzoProofEnvelope::from_canonical_json(&json).unwrap();
    assert_eq!(parsed, env);
    parsed
        .verify_binding(&proof_bytes, &inputs, &outputs)
        .unwrap();

    // Tampering with payload bytes fails binding check.
    let mut bad_bytes = proof_bytes.clone();
    bad_bytes[0] ^= 0x01;
    assert!(parsed
        .verify_binding(&bad_bytes, &inputs, &outputs)
        .is_err());

    // Conflating TrustTier (e.g., claiming HardwareAttestation) is rejected.
    let mut conflated = parsed.clone();
    conflated.trust_tier = TrustTier::HardwareAttestation;
    conflated.envelope_digest = conflated.compute_envelope_digest();
    assert!(conflated
        .verify_binding(&proof_bytes, &inputs, &outputs)
        .is_err());

    // Requiring Hanzo-controlled service is rejected.
    let mut centralized = parsed;
    centralized.routing_context.hanzo_service_required = true;
    centralized.envelope_digest = centralized.compute_envelope_digest();
    assert!(centralized
        .verify_binding(&proof_bytes, &inputs, &outputs)
        .is_err());
}

#[test]
fn envelope_v1_wraps_zkvm_receipt_independently() {
    let (prog, weights) = build_zkvm_kai_policy_program();
    let (pk, vk) = setup_zkvm_keys(
        prog.clone(),
        &weights,
        prog.instructions.len() + 4,
        LigeroParams::default(),
    )
    .unwrap();
    let inputs = vec![10, 100, 128, 1, 50, 80];
    let (outputs, receipt) = prove_zkvm(&pk, &inputs).unwrap();

    let env = HanzoProofEnvelope::from_zkvm_receipt(
        &vk,
        &receipt,
        &inputs,
        &outputs,
        "enso/route/zkvm-open",
        "visor/job/99",
    );
    assert_eq!(env.proof_backend, ProofBackend::GeneralZkVm);
    let parsed = HanzoProofEnvelope::from_canonical_json(&env.to_canonical_json()).unwrap();
    parsed
        .verify_binding(&receipt.to_bytes(), &inputs, &outputs)
        .unwrap();
}

// ============================================================================
// Benchmark Suite, 16-Vector Adversarial Matrix & 27B Negative Regime Tests
// ============================================================================

#[test]
fn full_16_vector_adversarial_matrix_achieves_100_percent_rejection() {
    let checks = run_adversarial_matrix(LigeroParams::default()).unwrap();
    assert_eq!(checks.len(), 16);
    for c in &checks {
        assert!(
            c.rejected,
            "adversarial attack '{}' on '{}' unexpectedly accepted!",
            c.attack_id, c.target_path
        );
    }
}

#[test]
fn benchmark_harness_and_27b_negative_regime_are_sound() {
    let report = run_full_benchmark_suite(1350, 3, LigeroParams::default()).unwrap();
    assert_eq!(report.workloads.len(), 6);
    assert_eq!(report.adversarial_checks.len(), 16);
    assert!((report.adversarial_detection_rate - 1.0).abs() < f64::EPSILON);
    assert!((report.decentralization_score - 1.0).abs() < f64::EPSILON);
    assert!((report.sovereignty_score - 1.0).abs() < f64::EPSILON);

    for w in &report.workloads {
        assert!((w.completion_rate - 1.0).abs() < f64::EPSILON);
        assert!(w.proof_bytes > 0);
        assert!(w.constraints_count > 0);
        assert!(w.prove_mean_ms > 0.0);
        assert!(w.verify_mean_ms > 0.0);
    }

    let neg = evaluate_27b_negative_regime(&report.workloads[0]);
    assert_eq!(neg.verdict, "REJECTED_FOR_INTERACTIVE_27B_DECODE");
    assert!(neg.extrapolated_single_token_prove_hours > 1.0);

    let (json, sha) = report.to_canonical_json_with_sha256();
    assert_eq!(sha.len(), 64);
    assert!(json.contains("\"verdict\": \"REJECTED_FOR_INTERACTIVE_27B_DECODE\""));
    assert!(!json.contains("/home/") && !json.contains("/Users/"));
}
