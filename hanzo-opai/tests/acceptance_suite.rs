// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive unit and integration acceptance suite for `hanzo-opai` (`hanzoai/engine#67`).

use hanzo_opai::{
    ct_eq_digest, digest_to_hex, execute_trace, execute_trace_with_fault, fixed_point_silu_q12,
    hex_to_digest, isqrt_u128, max_bisection_rounds, rescale_i128, round_div_i128,
    run_dispute_to_completion, verify_single_step, verify_transcript_sovereign, ActorStrategy,
    Claim, ClaimStatus, DisputeGame, DisputeMove, DisputePhase, DisputeTranscript, DyadicScale,
    ExecutionProfile, FaultSpec, GameConfig, InputCommitment, NumericDomain, OpAiError,
    OptimisticCoordinator, OverflowMode, Party, QuantizedModel, ResolutionReason, RoundingMode,
    Sha256, TraceMerkleTree, SIGMOID_LUT_Q15,
};

fn test_fixture() -> (
    ExecutionProfile,
    QuantizedModel,
    InputCommitment,
    GameConfig,
) {
    let profile = ExecutionProfile::canonical();
    let model = QuantizedModel::synthesize(1350, 32, 16, 32, 4).expect("model");
    let prompt_bytes = b"Hanzo HIP-1350 optimistic AI fraud-proof fixture prompt";
    let prompt_tokens = vec![1, 5, 11, 17, 23, 29, 3, 9];
    let input = InputCommitment::new(
        prompt_bytes,
        prompt_tokens,
        2, // T = 2 * (8*4 + 4) = 72 steps
        profile.digest(),
        model.digest(),
    )
    .expect("input");
    let config = GameConfig::default();
    (profile, model, input, config)
}

#[test]
fn sha256_nist_vectors_and_merkle_inclusion_proofs() {
    assert_eq!(
        digest_to_hex(&Sha256::digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        digest_to_hex(&Sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        digest_to_hex(&Sha256::digest(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );

    let d = Sha256::digest(b"hanzo-opai");
    let parsed = hex_to_digest(&digest_to_hex(&d)).expect("hex roundtrip");
    assert!(ct_eq_digest(&d, &parsed));

    let leaves: Vec<[u8; 32]> = (0u64..13)
        .map(|i| Sha256::digest(&i.to_le_bytes()))
        .collect();
    let tree = TraceMerkleTree::build(&leaves).expect("merkle build");
    let root = tree.root();
    for (idx, leaf) in leaves.iter().enumerate() {
        let proof = tree.prove(idx).expect("prove");
        assert!(TraceMerkleTree::verify_proof(&root, leaf, &proof));
        let mut forged = *leaf;
        forged[0] ^= 1;
        assert!(!TraceMerkleTree::verify_proof(&root, &forged, &proof));
    }
}

#[test]
fn fixed_point_math_and_floating_point_exclusion_gate() {
    // Floating-point profile must be unconditionally rejected.
    let mut fp_profile = ExecutionProfile::canonical();
    fp_profile.numeric_domain = NumericDomain::FloatingPointDisallowed;
    assert!(matches!(
        fp_profile.validate(),
        Err(OpAiError::UnsupportedFloatingPointProfile { .. })
    ));

    // Sigmoid LUT exact symmetry & monotonicity.
    for i in 0..=256 {
        assert_eq!(SIGMOID_LUT_Q15[i] + SIGMOID_LUT_Q15[256 - i], 32768);
        if i < 256 {
            assert!(SIGMOID_LUT_Q15[i] <= SIGMOID_LUT_Q15[i + 1]);
        }
    }
    // SiLU(0) == 0, SiLU(4096) ~= 4096 * sigmoid(1.0) ~= 2994.
    assert_eq!(
        fixed_point_silu_q12(0, RoundingMode::RoundHalfToEven, OverflowMode::Saturating).unwrap(),
        0
    );
    let silu_one = fixed_point_silu_q12(
        4096,
        RoundingMode::RoundHalfToEven,
        OverflowMode::Saturating,
    )
    .unwrap();
    assert!((2990..=3000).contains(&silu_one), "silu(1.0)={silu_one}");

    // Exact banker's vs half-away-from-zero rounding on ties.
    let half_scale = DyadicScale::new(1, 1).unwrap();
    assert_eq!(
        rescale_i128(3, half_scale, RoundingMode::RoundHalfToEven).unwrap(),
        2
    );
    assert_eq!(
        rescale_i128(5, half_scale, RoundingMode::RoundHalfToEven).unwrap(),
        2
    );
    assert_eq!(
        rescale_i128(5, half_scale, RoundingMode::RoundHalfAwayFromZero).unwrap(),
        3
    );
    assert_eq!(
        rescale_i128(-5, half_scale, RoundingMode::RoundHalfAwayFromZero).unwrap(),
        -3
    );

    // Overflow-free round_div_i128 at extreme i128::MIN / u128 boundary.
    assert_eq!(
        round_div_i128(i128::MIN, 2, RoundingMode::RoundHalfToEven).unwrap(),
        i128::MIN / 2
    );
    assert!(matches!(
        round_div_i128(10, 0, RoundingMode::RoundHalfToEven),
        Err(OpAiError::ArithmeticOverflow(_))
    ));

    // Exact bitwise integer square root.
    assert_eq!(isqrt_u128(0), 0);
    assert_eq!(isqrt_u128(1), 1);
    assert_eq!(isqrt_u128(16_777_216), 4096);
    let huge = (1u128 << 120) + 12345;
    let r = isqrt_u128(huge);
    assert!(r * r <= huge && (r + 1) * (r + 1) > huge);
}

#[test]
fn pinpoints_exact_single_step_at_step_0_middle_and_t_minus_1() {
    let (profile, model, input, config) = test_fixture();
    let total_steps = model.total_steps(input.max_new_tokens);
    assert_eq!(total_steps, 72);
    let honest = execute_trace(&profile, &model, &input, None).expect("honest trace");
    let honest_again = execute_trace(&profile, &model, &input, None).expect("deterministic replay");
    assert_eq!(honest.state_roots, honest_again.state_roots);
    assert_eq!(honest.output_tokens, honest_again.output_tokens);

    let cases = [
        (
            0u64,
            FaultSpec::PerturbHidden {
                step: 0,
                index: 2,
                delta: 1,
            },
        ),
        (
            total_steps / 2,
            FaultSpec::CorruptStepOutput {
                step: total_steps / 2,
                delta: 512,
            },
        ),
        (
            total_steps - 1,
            FaultSpec::CorruptOutputToken {
                step: total_steps - 1,
                xor_mask: 3,
            },
        ),
        (17, FaultSpec::ForgePostRootOnly { step: 17 }),
    ];

    for (idx, (expected_step, fault)) in cases.into_iter().enumerate() {
        let dishonest =
            execute_trace_with_fault(&profile, &model, &input, None, Some(fault)).expect("bad");
        assert_ne!(dishonest.end_root(), honest.end_root());

        let claim = Claim::from_trace(
            "dishonest-provider",
            idx as u64,
            &dishonest,
            config.provider_bond,
            1_000,
        )
        .expect("claim");
        let mut game = DisputeGame::open(
            claim,
            "watchtower-alpha",
            config.challenger_base_bond,
            config.clone(),
            1_010,
            0,
        )
        .expect("open");

        let res = run_dispute_to_completion(
            &mut game,
            &profile,
            &model,
            &input,
            &dishonest,
            &honest,
            ActorStrategy::Honest,
            ActorStrategy::Honest,
            3,
        )
        .expect("dispute run");

        assert_eq!(res.winner, Party::Challenger);
        assert_eq!(res.reason, ResolutionReason::SingleStepFraudProven);
        assert_eq!(res.pinpointed_step, Some(expected_step));
        assert!(game.rounds_completed <= max_bisection_rounds(total_steps));
        assert!(res
            .settlement
            .verify_conservation(config.provider_bond, config.challenger_base_bond));
        assert_eq!(res.settlement.provider_slashed, config.provider_bond);
        assert!(res.settlement.burned > 0);
        assert!(res.settlement.challenger_payout > config.challenger_base_bond);

        // Sovereign export & third-party replay (0 Hanzo calls).
        let transcript = DisputeTranscript::from_resolved_game(&game).expect("export");
        let report = verify_transcript_sovereign(&transcript, &profile, &model, &input)
            .expect("sovereign replay");
        assert_eq!(report.control_plane_calls, 0);
        assert_eq!(report.pinpointed_step, Some(expected_step));
        assert!(report.single_step_judge_executed);
        assert!(report.bond_conservation_valid);
    }
}

#[test]
fn start_root_forgery_and_false_start_root_accusation() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");
    let forged_start = execute_trace_with_fault(
        &profile,
        &model,
        &input,
        None,
        Some(FaultSpec::ForgeStartRoot),
    )
    .expect("forged start");

    // 1. Forged start_root is caught in 0 bisection rounds.
    let bad_claim =
        Claim::from_trace("forger", 1, &forged_start, config.provider_bond, 100).expect("claim");
    let mut game = DisputeGame::open(
        bad_claim,
        "watchtower",
        config.challenger_base_bond,
        config.clone(),
        105,
        0,
    )
    .expect("open");
    let res = run_dispute_to_completion(
        &mut game,
        &profile,
        &model,
        &input,
        &forged_start,
        &honest,
        ActorStrategy::Honest,
        ActorStrategy::Honest,
        2,
    )
    .expect("run");
    assert_eq!(game.rounds_completed, 0);
    assert_eq!(res.winner, Party::Challenger);
    assert_eq!(res.reason, ResolutionReason::InvalidStartRootProven);
    assert_eq!(res.pinpointed_step, Some(0));

    let tr = DisputeTranscript::from_resolved_game(&game).expect("tr");
    let rep = verify_transcript_sovereign(&tr, &profile, &model, &input).expect("replay");
    assert_eq!(rep.rounds_replayed, 0);
    assert_eq!(rep.winner, Party::Challenger);

    // 2. False start_root accusation against an honest claim immediately slashes challenger.
    let good_claim =
        Claim::from_trace("honest", 2, &honest, config.provider_bond, 100).expect("claim");
    let mut g2 = DisputeGame::open(
        good_claim,
        "griefer",
        config.challenger_base_bond,
        config.clone(),
        105,
        0,
    )
    .expect("open");
    let r2 = g2
        .assert_invalid_start_root(&profile, &model, &input, 107)
        .expect("check");
    assert_eq!(r2.winner, Party::Defender);
    assert_eq!(r2.reason, ResolutionReason::ValidStartRootVerified);
    assert_eq!(
        r2.settlement.challenger_slashed,
        config.challenger_base_bond
    );
}

#[test]
fn invalid_pre_state_witness_slashes_defender() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");
    let claim = Claim::from_trace("bad-witness-prov", 1, &honest, config.provider_bond, 500)
        .expect("claim");
    let mut game = DisputeGame::open(
        claim,
        "watchtower",
        config.challenger_base_bond,
        config.clone(),
        505,
        0,
    )
    .expect("open");
    let res = run_dispute_to_completion(
        &mut game,
        &profile,
        &model,
        &input,
        &honest,
        &honest,
        ActorStrategy::SubmitCorruptPreState,
        ActorStrategy::MaliciousGrief { prefer_left: true },
        2,
    )
    .expect("run");
    assert_eq!(res.winner, Party::Challenger);
    assert_eq!(res.reason, ResolutionReason::InvalidPreStateWitness);

    let tr = DisputeTranscript::from_resolved_game(&game).expect("tr");
    verify_transcript_sovereign(&tr, &profile, &model, &input).expect("replay");
}

#[test]
fn malicious_challenger_slashed_and_honest_provider_compensated() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");
    let claim =
        Claim::from_trace("honest-prov", 1, &honest, config.provider_bond, 1_000).expect("claim");

    for &(prefer_left, slot) in &[(true, 0usize), (false, 1usize)] {
        let bond = config.required_challenger_bond(slot);
        let mut game = DisputeGame::open(
            claim.clone(),
            format!("malicious-{slot}"),
            bond,
            config.clone(),
            1_010,
            slot,
        )
        .expect("open");
        let res = run_dispute_to_completion(
            &mut game,
            &profile,
            &model,
            &input,
            &honest,
            &honest,
            ActorStrategy::Honest,
            ActorStrategy::MaliciousGrief { prefer_left },
            2,
        )
        .expect("run");
        assert_eq!(res.winner, Party::Defender);
        assert_eq!(res.reason, ResolutionReason::SingleStepVerifiedHonest);
        assert_eq!(res.settlement.challenger_slashed, bond);
        assert!(res.settlement.provider_payout > config.provider_bond);
        assert!(res.settlement.burned > 0);
        assert!(res
            .settlement
            .verify_conservation(config.provider_bond, bond));

        let tr = DisputeTranscript::from_resolved_game(&game).expect("export");
        let wire = tr.to_wire_bytes();
        let json = tr.to_json_string();
        assert_eq!(DisputeTranscript::from_wire_bytes(&wire).unwrap(), tr);
        assert_eq!(DisputeTranscript::from_json_str(&json).unwrap(), tr);
        verify_transcript_sovereign(&tr, &profile, &model, &input).expect("replay");
    }
}

#[test]
fn defender_and_challenger_timeout_forfeits_and_deadline_guards() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");
    let claim = Claim::from_trace("prov", 1, &honest, config.provider_bond, 2_000).expect("claim");

    // 1. Premature timeout claim rejected; backdated or expired move rejected.
    let mut g = DisputeGame::open(
        claim.clone(),
        "ch-1",
        config.challenger_base_bond,
        config.clone(),
        2_010,
        0,
    )
    .expect("open");
    assert!(matches!(
        g.claim_timeout_forfeit(2_010 + config.turn_deadline_secs),
        Err(OpAiError::TurnNotExpired { .. })
    ));
    let mid = g.range.midpoint();
    let mid_root = honest.root_at(mid).unwrap();
    // Backdated move rejected.
    assert!(matches!(
        g.submit_defender_midpoint(mid, mid_root, 2_009),
        Err(OpAiError::DeadlineViolation(_))
    ));
    // Expired move rejected.
    assert!(matches!(
        g.submit_defender_midpoint(mid, mid_root, 2_010 + config.turn_deadline_secs + 1),
        Err(OpAiError::DeadlineViolation(_))
    ));
    // Valid timeout forfeit slashes defender.
    let res = g
        .claim_timeout_forfeit(2_010 + config.turn_deadline_secs + 1)
        .expect("forfeit")
        .clone();
    assert_eq!(res.winner, Party::Challenger);
    assert_eq!(res.reason, ResolutionReason::DefenderTimeoutForfeit);
    let tr = DisputeTranscript::from_resolved_game(&g).expect("tr");
    verify_transcript_sovereign(&tr, &profile, &model, &input).expect("replay");

    // 2. Defender stalls at single-step witness -> DefenderTimeoutForfeit.
    let mut g_ss = DisputeGame::open(
        claim.clone(),
        "ch-2",
        config.challenger_base_bond,
        config.clone(),
        2_010,
        0,
    )
    .expect("open");
    let res_ss = run_dispute_to_completion(
        &mut g_ss,
        &profile,
        &model,
        &input,
        &honest,
        &honest,
        ActorStrategy::StallAtSingleStep,
        ActorStrategy::MaliciousGrief { prefer_left: true },
        2,
    )
    .expect("run");
    assert_eq!(res_ss.winner, Party::Challenger);
    assert_eq!(res_ss.reason, ResolutionReason::DefenderTimeoutForfeit);
    let tr_ss = DisputeTranscript::from_resolved_game(&g_ss).expect("tr");
    verify_transcript_sovereign(&tr_ss, &profile, &model, &input).expect("replay");

    // 3. Challenger stalls mid-bisection -> ChallengerTimeoutForfeit.
    let mut g_ch = DisputeGame::open(
        claim,
        "ch-3",
        config.challenger_base_bond,
        config.clone(),
        2_010,
        0,
    )
    .expect("open");
    let res_ch = run_dispute_to_completion(
        &mut g_ch,
        &profile,
        &model,
        &input,
        &honest,
        &honest,
        ActorStrategy::Honest,
        ActorStrategy::StallAtRound(2),
        2,
    )
    .expect("run");
    assert_eq!(res_ch.winner, Party::Defender);
    assert_eq!(res_ch.reason, ResolutionReason::ChallengerTimeoutForfeit);
    let tr_ch = DisputeTranscript::from_resolved_game(&g_ch).expect("tr");
    verify_transcript_sovereign(&tr_ch, &profile, &model, &input).expect("replay");
}

#[test]
fn shallow_one_step_game_resolves_in_zero_bisection_rounds() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, Some(1)).expect("1-step honest");
    let dishonest = execute_trace_with_fault(
        &profile,
        &model,
        &input,
        Some(1),
        Some(FaultSpec::PerturbHidden {
            step: 0,
            index: 0,
            delta: 42,
        }),
    )
    .expect("1-step dishonest");
    let claim =
        Claim::from_trace("shallow-prov", 1, &dishonest, config.provider_bond, 100).expect("claim");
    let mut game = DisputeGame::open(
        claim,
        "shallow-ch",
        config.challenger_base_bond,
        config.clone(),
        101,
        0,
    )
    .expect("open");
    assert!(matches!(game.phase, DisputePhase::AwaitDefenderSingleStep));
    let res = run_dispute_to_completion(
        &mut game,
        &profile,
        &model,
        &input,
        &dishonest,
        &honest,
        ActorStrategy::Honest,
        ActorStrategy::Honest,
        1,
    )
    .expect("run");
    assert_eq!(game.rounds_completed, 0);
    assert_eq!(res.winner, Party::Challenger);
    assert_eq!(res.pinpointed_step, Some(0));

    // Also verify out-of-range total_steps is rejected by verify_single_step.
    let overflow_steps = model.total_steps(input.max_new_tokens) + 1;
    assert!(matches!(
        verify_single_step(
            &profile,
            &model,
            &input,
            overflow_steps,
            0,
            honest.state_at(0).unwrap(),
            &honest.start_root(),
            &honest.end_root()
        ),
        Err(OpAiError::InvalidClaim(_))
    ));
}

#[test]
fn multi_challenger_tournament_anti_griefing_and_multi_tenant_isolation() {
    let (profile, model, input, mut config) = test_fixture();
    config.max_concurrent_challenges_per_claim = 4;
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");
    let dishonest = execute_trace_with_fault(
        &profile,
        &model,
        &input,
        None,
        Some(FaultSpec::PerturbLogit {
            step: 35,
            index: 1,
            delta: 9,
        }),
    )
    .expect("dishonest");

    let mut coord = OptimisticCoordinator::new(config.clone()).expect("coord");

    // Tenant A: Honest claim attacked by 4 concurrent Sybil griefers with escalating bonds.
    let claim_a =
        Claim::from_trace("tenant-a-prov", 1, &honest, config.provider_bond, 1_000).expect("a");
    let cid_a = coord.submit_claim(claim_a.clone()).expect("submit a");

    // Tenant B: Honest unchallenged claim settles optimistically.
    let claim_b =
        Claim::from_trace("tenant-b-prov", 2, &honest, config.provider_bond, 1_000).expect("b");
    let cid_b = coord.submit_claim(claim_b.clone()).expect("submit b");

    // Tenant C: Dishonest claim challenged concurrently by 3 honest watchtowers.
    let claim_c =
        Claim::from_trace("tenant-c-bad", 3, &dishonest, config.provider_bond, 1_000).expect("c");
    let cid_c = coord.submit_claim(claim_c.clone()).expect("submit c");

    // Open 4 escalating challenges on Tenant A.
    let mut a_disputes = Vec::new();
    for slot in 0..4usize {
        let req = config.required_challenger_bond(slot);
        if slot > 0 {
            assert!(req > config.required_challenger_bond(slot - 1));
            // Under-bonded griefer is rejected!
            assert!(matches!(
                coord.open_challenge(&cid_a, format!("sybil-{slot}"), req - 1, 1_005),
                Err(OpAiError::InsufficientBond { .. })
            ));
        }
        let did = coord
            .open_challenge(&cid_a, format!("sybil-{slot}"), req, 1_005)
            .expect("open sybil");
        a_disputes.push(did);
    }
    // 5th concurrent challenger rejected by concurrency cap.
    assert!(matches!(
        coord.open_challenge(
            &cid_a,
            "sybil-overflow",
            config.required_challenger_bond(4),
            1_006
        ),
        Err(OpAiError::MaxConcurrentChallengesExceeded { limit: 4 })
    ));
    // Claim A cannot finalize while active disputes exist, even after window closes.
    assert!(matches!(
        coord.finalize_claim(&cid_a, 1_000 + config.challenge_window_secs + 10),
        Err(OpAiError::ChallengeWindowViolation(_))
    ));

    // Resolve all 4 Sybil challenges on Tenant A in favor of honest provider.
    for (slot, did) in a_disputes.iter().enumerate() {
        let g = coord.dispute_mut(did).unwrap();
        let res = run_dispute_to_completion(
            g,
            &profile,
            &model,
            &input,
            &honest,
            &honest,
            ActorStrategy::Honest,
            ActorStrategy::MaliciousGrief {
                prefer_left: slot % 2 == 0,
            },
            2,
        )
        .unwrap();
        assert_eq!(res.winner, Party::Defender);
        coord.settle_dispute(did).expect("settle a");
    }
    coord
        .finalize_claim(&cid_a, 1_000 + config.challenge_window_secs)
        .expect("finalize a");
    assert_eq!(coord.claim(&cid_a).unwrap().status, ClaimStatus::Finalized);
    assert!(coord.claim(&cid_a).unwrap().provider_compensation_earned > 0);

    // Tenant B: cannot finalize early; finalizes cleanly at window close; rejects late challenge.
    assert!(matches!(
        coord.finalize_claim(&cid_b, 1_000 + config.challenge_window_secs - 1),
        Err(OpAiError::ChallengeWindowViolation(_))
    ));
    coord
        .finalize_claim(&cid_b, 1_000 + config.challenge_window_secs)
        .expect("finalize b");
    assert_eq!(coord.claim(&cid_b).unwrap().status, ClaimStatus::Finalized);
    assert!(matches!(
        coord.open_challenge(&cid_b, "late-ch", config.challenger_base_bond, 1_500),
        Err(OpAiError::ClaimAlreadySettled)
    ));

    // Tenant C: 3 concurrent honest watchtowers challenge dishonest claim; watchtower-0 finishes first.
    let mut c_disputes = Vec::new();
    for slot in 0..3usize {
        let req = config.required_challenger_bond(slot);
        let did = coord
            .open_challenge(&cid_c, format!("honest-wt-{slot}"), req, 1_010)
            .expect("open wt");
        c_disputes.push((did, req));
    }
    // Duplicate active challenger ID rejected.
    assert!(matches!(
        coord.open_challenge(
            &cid_c,
            "honest-wt-0",
            config.required_challenger_bond(3),
            1_011
        ),
        Err(OpAiError::ChallengerAlreadyActive(_))
    ));
    {
        let g0 = coord.dispute_mut(&c_disputes[0].0).unwrap();
        let r0 = run_dispute_to_completion(
            g0,
            &profile,
            &model,
            &input,
            &dishonest,
            &honest,
            ActorStrategy::Honest,
            ActorStrategy::Honest,
            2,
        )
        .unwrap();
        assert_eq!(r0.winner, Party::Challenger);
        assert_eq!(r0.pinpointed_step, Some(35));
    }
    coord.settle_dispute(&c_disputes[0].0).expect("settle c0");
    assert_eq!(coord.claim(&cid_c).unwrap().status, ClaimStatus::Slashed);
    // Sibling watchtowers 1 and 2 must be cleanly refunded!
    for &(sib_id, sib_bond) in &c_disputes[1..] {
        let sib = coord.dispute(&sib_id).unwrap();
        let res = sib.resolution().expect("refunded");
        assert_eq!(res.reason, ResolutionReason::RefundedClaimAlreadySlashed);
        assert_eq!(res.settlement.challenger_payout, sib_bond);
        let tr = DisputeTranscript::from_resolved_game(sib).expect("export refund");
        verify_transcript_sovereign(&tr, &profile, &model, &input).expect("replay refund");
    }

    // Exact global bond conservation across all 3 tenants & all concurrent disputes!
    assert_eq!(coord.total_locked, 0);
    assert!(coord.verify_conservation());
}

#[test]
fn transcript_tampering_and_wire_malformation_rejected() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");
    let bad = execute_trace_with_fault(
        &profile,
        &model,
        &input,
        None,
        Some(FaultSpec::CorruptStepOutput {
            step: 22,
            delta: 512,
        }),
    )
    .expect("bad");
    assert_ne!(bad.end_root(), honest.end_root());
    let claim = Claim::from_trace("prov", 7, &bad, config.provider_bond, 500).expect("claim");
    let mut game = DisputeGame::open(
        claim,
        "watchtower",
        config.challenger_base_bond,
        config.clone(),
        510,
        0,
    )
    .expect("open");
    let res = run_dispute_to_completion(
        &mut game,
        &profile,
        &model,
        &input,
        &bad,
        &honest,
        ActorStrategy::Honest,
        ActorStrategy::Honest,
        2,
    )
    .expect("run");
    assert_eq!(res.winner, Party::Challenger);
    assert_eq!(res.pinpointed_step, Some(22));

    let tr = DisputeTranscript::from_resolved_game(&game).expect("export");
    let wire = tr.to_wire_bytes();
    let json = tr.to_json_string();

    // Valid round-trip.
    assert_eq!(DisputeTranscript::from_wire_bytes(&wire).unwrap(), tr);
    assert_eq!(DisputeTranscript::from_json_str(&json).unwrap(), tr);

    // 1. Truncated wire rejected.
    assert!(DisputeTranscript::from_wire_bytes(&wire[..wire.len() - 1]).is_err());
    // 2. Trailing byte rejected.
    let mut trailing = wire.clone();
    trailing.push(0x00);
    assert!(DisputeTranscript::from_wire_bytes(&trailing).is_err());
    // 3. Bad magic rejected.
    let mut bad_magic = wire.clone();
    bad_magic[0] = b'X';
    assert!(DisputeTranscript::from_wire_bytes(&bad_magic).is_err());
    // 4. Flipped byte rejected.
    let mut flipped = wire;
    flipped[40] ^= 0x80;
    assert!(DisputeTranscript::from_wire_bytes(&flipped).is_err());

    // 5. Semantic tampering WITH recomputed digest rejected by sovereign replay.
    let mut forged_winner = tr.clone();
    forged_winner.resolution.winner = Party::Defender;
    forged_winner.transcript_digest = forged_winner.compute_digest();
    assert!(verify_transcript_sovereign(&forged_winner, &profile, &model, &input).is_err());

    let mut forged_bond = tr.clone();
    forged_bond.resolution.settlement.burned += 10;
    forged_bond.transcript_digest = forged_bond.compute_digest();
    assert!(verify_transcript_sovereign(&forged_bond, &profile, &model, &input).is_err());

    let mut extra_move = tr.clone();
    extra_move
        .moves
        .push(DisputeMove::ConcurrentRefund { at_secs: 999 });
    extra_move.transcript_digest = extra_move.compute_digest();
    assert!(verify_transcript_sovereign(&extra_move, &profile, &model, &input).is_err());

    // 6. Cross-profile replay rejected.
    let mut other_profile = profile.clone();
    other_profile.rounding_mode = RoundingMode::RoundHalfAwayFromZero;
    assert!(verify_transcript_sovereign(&tr, &other_profile, &model, &input).is_err());
}

#[test]
fn protocol_state_machine_and_config_validation_guards() {
    let (profile, model, input, config) = test_fixture();
    let honest = execute_trace(&profile, &model, &input, None).expect("honest");

    // GameConfig rejects burn == 0 (challenger_reward_bps == 10_000 prevents self-challenge).
    let mut bad_cfg = config.clone();
    bad_cfg.challenger_reward_bps = 10_000;
    assert!(bad_cfg.validate().is_err());
    let mut zero_bond_cfg = config.clone();
    zero_bond_cfg.provider_bond = 0;
    assert!(zero_bond_cfg.validate().is_err());

    // WrongTurn, InvalidBisectionMidpoint, NotReadyForSingleStep.
    let claim =
        Claim::from_trace("prov-guard", 1, &honest, config.provider_bond, 100).expect("claim");
    let mut game = DisputeGame::open(
        claim.clone(),
        "ch-guard",
        config.challenger_base_bond,
        config.clone(),
        105,
        0,
    )
    .expect("open");
    // Challenger cannot move before defender submits midpoint.
    assert!(matches!(
        game.submit_challenger_branch(hanzo_opai::BranchChoice::Left, 106),
        Err(OpAiError::WrongTurn(_))
    ));
    // Defender cannot submit wrong midpoint.
    assert!(matches!(
        game.submit_defender_midpoint(1, honest.root_at(1).unwrap(), 106),
        Err(OpAiError::InvalidBisectionMidpoint { .. })
    ));
    // Single-step proof cannot be submitted before hi == lo + 1.
    assert!(matches!(
        game.submit_single_step_proof(
            &profile,
            &model,
            &input,
            honest.state_at(0).unwrap().clone(),
            106
        ),
        Err(OpAiError::NotReadyForSingleStep { .. })
    ));

    // Coordinator rejects duplicate claims and unknown claim IDs.
    let mut coord = OptimisticCoordinator::new(config).expect("coord");
    let cid = coord.submit_claim(claim.clone()).expect("submit");
    assert!(matches!(
        coord.submit_claim(claim),
        Err(OpAiError::DuplicateClaim)
    ));
    assert!(matches!(
        coord.finalize_claim(&[0u8; 32], 9_999),
        Err(OpAiError::ClaimNotFound)
    ));
    // Challenge after window closed is rejected.
    assert!(matches!(
        coord.open_challenge(&cid, "too-late", 1_000_000, 100 + 601),
        Err(OpAiError::ChallengeWindowViolation(_))
    ));
}

#[test]
fn every_step_op_category_verified_individually_by_single_step_judge() {
    let (profile, model, input, _) = test_fixture();
    let trace = execute_trace(&profile, &model, &input, None).expect("trace");
    let mut seen_tags = std::collections::BTreeSet::new();

    for step in 0..trace.total_steps {
        let pre = trace.state_at(step).unwrap();
        let pre_root = trace.root_at(step).unwrap();
        let post_root = trace.root_at(step + 1).unwrap();
        let verdict = verify_single_step(
            &profile,
            &model,
            &input,
            trace.total_steps,
            step,
            pre,
            &pre_root,
            &post_root,
        )
        .expect("verify step");
        assert!(verdict.pre_root_valid);
        assert!(verdict.defender_valid);
        seen_tags.insert(verdict.op_tag);
    }
    // All 12 distinct atomic transformer/MLP step operation categories must be exercised.
    assert_eq!(seen_tags.len(), 12, "seen tags: {seen_tags:?}");
}

#[test]
fn malformed_json_transcripts_rejected_cleanly() {
    assert!(DisputeTranscript::from_json_str("").is_err());
    assert!(DisputeTranscript::from_json_str("{}").is_err());
    assert!(DisputeTranscript::from_json_str("{\"schema\":\"wrong.v9\"}").is_err());
    assert!(
        DisputeTranscript::from_json_str("{\"schema\":\"hanzo.opai.transcript.v1\"} trailing")
            .is_err()
    );
    assert!(DisputeTranscript::from_json_str("{\"unterminated").is_err());
}

#[test]
fn checked_overflow_mode_aborts_deterministically() {
    assert!(matches!(
        hanzo_opai::add_i32(i32::MAX, 1, OverflowMode::CheckedError),
        Err(OpAiError::ArithmeticOverflow(_))
    ));
    assert_eq!(
        hanzo_opai::add_i32(i32::MAX, 1, OverflowMode::Saturating).unwrap(),
        i32::MAX
    );
    assert!(matches!(
        hanzo_opai::narrow_i32(i128::from(i32::MAX) + 1, OverflowMode::CheckedError),
        Err(OpAiError::ArithmeticOverflow(_))
    ));
}
