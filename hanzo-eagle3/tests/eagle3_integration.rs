#![forbid(unsafe_code)]
//! Comprehensive integration & mathematical property test suite for `hanzo-eagle3`
//! (`hanzoai/engine#55`, HIP-1346).

use hanzo_eagle3::bench::{
    format_report_json, run_full_benchmark, run_workload_strategy, verify_stochastic_equivalence,
    BaselineRef, BenchEnvironment, BenchStrategy,
};
use hanzo_eagle3::corpus::{find_repo_root, GroundedCorpus, WorkloadKind};
use hanzo_eagle3::eagle3::Eagle3DraftHead;
use hanzo_eagle3::lookup::PromptLookupDrafter;
use hanzo_eagle3::matrix::{
    evaluate_acceptance_gate, evaluate_large_batch_negative_regime, HybridModelSpec,
};
use hanzo_eagle3::rng::{argmax_f32, sha256_hex, softmax_temp, SplitMix64};
use hanzo_eagle3::selector::{
    DraftStrategy, SelectorConfig, StepFeedback, StepSignals, WorkloadAwareSelector,
};
use hanzo_eagle3::target::{TargetKvCache, TargetModel};
use hanzo_eagle3::verify::{
    residual_distribution, verify_linear_greedy, verify_linear_stochastic, verify_tree_greedy,
    verify_tree_stochastic, DraftTree, LinearDraft, TreeNode,
};

fn load_corpus() -> GroundedCorpus {
    let root = find_repo_root().expect("repository corpus files must be present");
    GroundedCorpus::load_from_repo(&root).expect("corpus files must load cleanly")
}

#[test]
fn test_01_sha256_fips180_4_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn test_02_grounded_corpus_manifest_deterministic() {
    let c1 = load_corpus();
    let c2 = load_corpus();
    assert_eq!(c1.files.len(), 3);
    assert_eq!(c1.total_bytes, 140_742);
    assert_eq!(c1.combined_sha256, c2.combined_sha256);
    assert_eq!(
        c1.files[0].sha256,
        "aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0"
    );
    assert_eq!(
        c1.files[1].sha256,
        "67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a"
    );
    assert_eq!(
        c1.files[2].sha256,
        "8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a"
    );
}

#[test]
fn test_03_tree_attention_forward_is_bit_exact_vs_sequential() {
    let corpus = load_corpus();
    let target = TargetModel::from_grounded_corpus(&corpus, 1346);
    let turns = corpus.build_workload_turns(WorkloadKind::MixedProseCode);
    let prompt = &turns[0].prompt;

    // Prefill identical prefix into two independent caches
    let mut seq_cache = TargetKvCache::new();
    let last = target.prefill_prompt(prompt, &mut seq_cache, 0.0);
    let tree_cache = seq_cache.clone();

    let anchor = argmax_f32(&last.logits) as u32;
    // Build a multi-branch tree with 2 children at depth 1 and 2 grandchildren at depth 2
    let nodes = vec![
        TreeNode {
            id: 0,
            parent: None,
            depth: 0,
            token: anchor,
            cumulative_log_prob: 0.0,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 1,
            parent: Some(0),
            depth: 1,
            token: 32,
            cumulative_log_prob: -0.5,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 2,
            parent: Some(0),
            depth: 1,
            token: 101,
            cumulative_log_prob: -0.7,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 3,
            parent: Some(1),
            depth: 2,
            token: 116,
            cumulative_log_prob: -1.1,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 4,
            parent: Some(2),
            depth: 2,
            token: 114,
            cumulative_log_prob: -1.3,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
    ];
    let tree = DraftTree::with_computed_mask(nodes);
    let tree_out = target.forward_tree(&tree, &tree_cache, 0.0);

    // Compare path [0 -> 2 -> 4] node-by-node against sequential `forward_step_raw`!
    for &nid in &[0usize, 2, 4] {
        let seq_step = target.forward_step_raw(tree.nodes[nid].token, &mut seq_cache, 0.0);
        let max_diff = seq_step
            .logits
            .iter()
            .zip(tree_out.logits[nid].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(
            max_diff, 0.0,
            "tree attention node {nid} must be bit-identical (0.0 diff) to sequential decode"
        );
    }
}

#[test]
fn test_04_greedy_linear_and_tree_verification_exactness() {
    // Linear verify: accept 2, reject 3rd -> emit [d0, d1, correction]
    let draft = vec![5u32, 9, 12, 3];
    let mut logits = vec![vec![0.0_f32; 16]; 5];
    logits[0][5] = 10.0;
    logits[1][9] = 10.0;
    logits[2][7] = 10.0; // mismatch vs draft 12
    logits[3][3] = 10.0;
    logits[4][1] = 10.0;

    let out = verify_linear_greedy(&draft, &logits);
    assert_eq!(out.accepted_draft_count, 2);
    assert_eq!(out.emitted_tokens, vec![5, 9, 7]);
    assert_eq!(out.accepted_path, vec![0, 1, 2]);

    // All accepted -> emit bonus token from logits[k]
    logits[2][12] = 15.0;
    let out_all = verify_linear_greedy(&draft, &logits);
    assert_eq!(out_all.accepted_draft_count, 4);
    assert_eq!(out_all.emitted_tokens, vec![5, 9, 12, 3, 1]);

    // Multi-branch tree verify: root 0 -> child 1 (tok 5) or child 2 (tok 8); child 2 -> child 3 (tok 11)
    let tree = DraftTree::with_computed_mask(vec![
        TreeNode {
            id: 0,
            parent: None,
            depth: 0,
            token: 0,
            cumulative_log_prob: 0.0,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 1,
            parent: Some(0),
            depth: 1,
            token: 5,
            cumulative_log_prob: -0.2,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 2,
            parent: Some(0),
            depth: 1,
            token: 8,
            cumulative_log_prob: -0.4,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
        TreeNode {
            id: 3,
            parent: Some(2),
            depth: 2,
            token: 11,
            cumulative_log_prob: -0.6,
            draft_probs: Vec::new(),
            children: Vec::new(),
        },
    ]);
    let mut t_logits = vec![vec![0.0_f32; 16]; 4];
    t_logits[0][8] = 10.0; // matches child 2 (second branch!)
    t_logits[2][11] = 10.0; // matches grandchild 3!
    t_logits[3][14] = 10.0; // bonus token at leaf
    let t_out = verify_tree_greedy(&tree, &t_logits);
    assert_eq!(t_out.accepted_draft_count, 2);
    assert_eq!(t_out.accepted_path, vec![0, 2, 3]);
    assert_eq!(t_out.emitted_tokens, vec![8, 11, 14]);
}

#[test]
fn test_05_stochastic_rejection_sampling_preserves_exact_distribution() {
    let receipt = verify_stochastic_equivalence(1346, 60_000);
    assert!(
        receipt.passed,
        "stochastic verification failed: linear TV={:.5} chi2={:.2}, tree TV={:.5} chi2={:.2}",
        receipt.linear_tv_distance,
        receipt.linear_chi_square,
        receipt.tree_tv_distance,
        receipt.tree_chi_square
    );
    assert!(receipt.linear_tv_distance < 0.01);
    assert!(receipt.tree_tv_distance < 0.01);
    assert!(
        receipt.tree_acceptance_rate > receipt.linear_acceptance_rate,
        "2-branch stochastic tree must achieve strictly higher acceptance rate than 1-branch linear"
    );
}

#[test]
fn test_06_residual_distribution_and_zero_prob_edge_cases() {
    let p = vec![0.5_f32, 0.3, 0.2];
    let q = vec![0.1_f32, 0.8, 0.1];
    let r = residual_distribution(&p, &q);
    let sum: f32 = r.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!((r[0] - 0.8).abs() < 1e-5);
    assert_eq!(r[1], 0.0);
    assert!((r[2] - 0.2).abs() < 1e-5);

    // Identical p == q residual falls back gracefully to p
    let r_eq = residual_distribution(&p, &p);
    assert!((r_eq[0] - 0.5).abs() < 1e-5);

    // Stochastic verify when draft token has q_d == 0 rejects safely without NaN
    let mut rng = SplitMix64::new(42);
    let draft = LinearDraft {
        tokens: vec![1],
        draft_probs: vec![vec![1.0, 0.0, 0.0]],
    };
    let out = verify_linear_stochastic(&draft, &[p.clone(), p.clone()], &mut rng);
    assert_eq!(out.accepted_draft_count, 0);
    assert_eq!(out.emitted_tokens.len(), 1);
}

#[test]
fn test_07_eagle3_stochastic_tree_draft_and_verify_non_empty() {
    let corpus = load_corpus();
    let target = TargetModel::from_grounded_corpus(&corpus, 1346);
    let eagle3 = Eagle3DraftHead::new_untrained(&target, 1346);
    let mut cache = TargetKvCache::new();
    let turns = corpus.build_workload_turns(WorkloadKind::MixedProseCode);
    let last = target.prefill_prompt(&turns[0].prompt, &mut cache, 0.0);
    let anchor = argmax_f32(&last.logits) as u32;

    let mut rng = SplitMix64::new(99);
    let stree = eagle3.draft_tree_stochastic(anchor, &last.features, &target, 2, 2, &mut rng);
    assert_eq!(stree.num_nodes(), 1 + 2 + 4);
    let tree_out = target.forward_tree(&stree, &cache, 0.0);
    let target_probs: Vec<Vec<f32>> = tree_out
        .logits
        .iter()
        .map(|l| softmax_temp(l, 1.0))
        .collect();
    let outcome = verify_tree_stochastic(&stree, &target_probs, &mut rng);
    assert!(!outcome.emitted_tokens.is_empty());
    assert_eq!(
        outcome.emitted_tokens.len(),
        1 + outcome.accepted_draft_count
    );
}

#[test]
fn test_08_prompt_lookup_sam_recency_and_tail_non_shadowing() {
    let mut lookup = PromptLookupDrafter::new(3, 8);
    // Repeated code pattern: [10, 20, 30, 40, 50, 60, 70] followed later by [10, 20, 30]
    let seq = vec![10u32, 20, 30, 40, 50, 60, 70, 99, 98, 97, 10, 20, 30];
    lookup.extend_slice(&seq);
    assert!(lookup.num_sam_states() >= seq.len());

    let prop = lookup.propose(&seq, 4);
    assert_eq!(prop.matched_ngram_len, 3);
    assert_eq!(prop.draft.tokens, vec![40, 50, 60, 70]);

    // After accepting 40, tail becomes [..., 10, 20, 30, 40]; lookup must NOT shadow pos 3
    let mut next_seq = seq.clone();
    next_seq.push(40);
    lookup.push_token(40);
    let prop2 = lookup.propose(&next_seq, 3);
    assert_eq!(prop2.matched_ngram_len, 4);
    assert_eq!(prop2.draft.tokens, vec![50, 60, 70]);

    // Non-matching tail returns empty draft cleanly
    let unseen = vec![120u32, 121, 122, 123];
    let prop_none = lookup.propose(&unseen, 4);
    assert_eq!(prop_none.matched_ngram_len, 0);
    assert!(prop_none.draft.tokens.is_empty());
}

#[test]
fn test_09_eagle3_ttt_training_measurably_improves_loss_and_tau() {
    let corpus = load_corpus();
    let target = TargetModel::from_grounded_corpus(&corpus, 1346);
    let mut eagle3 = Eagle3DraftHead::new_untrained(&target, 1346);
    let mixed = corpus.build_workload_turns(WorkloadKind::MixedProseCode);

    let (untrained_lin, _) = run_workload_strategy(
        &target,
        &eagle3,
        &mixed,
        "mixed_untrained",
        BenchStrategy::Eagle3LinearFixed,
        None,
    );
    let (untrained_tree, _) = run_workload_strategy(
        &target,
        &eagle3,
        &mixed,
        "mixed_untrained",
        BenchStrategy::Eagle3TreeFixed,
        None,
    );

    let seqs = corpus.training_sequences(30, 48);
    let report = eagle3.train_ttt(&target, &seqs, 24, 3, 0.015);
    assert!(
        report.trained_loss < report.untrained_loss * 0.35,
        "TTT cross-entropy must drop sharply: {:.4} -> {:.4}",
        report.untrained_loss,
        report.trained_loss
    );
    assert!(
        report.trained_top1_acc > report.untrained_top1_acc + 0.50,
        "TTT step-1 top-1 accuracy must improve substantially"
    );

    let (trained_lin, _) = run_workload_strategy(
        &target,
        &eagle3,
        &mixed,
        "mixed_trained",
        BenchStrategy::Eagle3LinearFixed,
        None,
    );
    let (trained_tree, _) = run_workload_strategy(
        &target,
        &eagle3,
        &mixed,
        "mixed_trained",
        BenchStrategy::Eagle3TreeFixed,
        None,
    );

    assert!(
        trained_lin.mean_tau > untrained_lin.mean_tau + 0.6,
        "held-out linear tau must improve measurably: {:.3} -> {:.3}",
        untrained_lin.mean_tau,
        trained_lin.mean_tau
    );
    assert!(
        trained_tree.mean_tau > trained_lin.mean_tau,
        "EAGLE-3 tree verification must beat linear chain on held-out mixed prose/code: tree={:.3} vs lin={:.3}",
        trained_tree.mean_tau,
        trained_lin.mean_tau
    );
    assert!(
        trained_tree.mean_tau > untrained_tree.mean_tau + 0.8,
        "held-out tree tau must improve measurably: {:.3} -> {:.3}",
        untrained_tree.mean_tau,
        trained_tree.mean_tau
    );
}

#[test]
fn test_10_all_strategies_are_100_percent_bit_identical_to_plain_decode() {
    let corpus = load_corpus();
    let target = TargetModel::from_grounded_corpus(&corpus, 1346);
    let mut eagle3 = Eagle3DraftHead::new_untrained(&target, 1346);
    let seqs = corpus.training_sequences(16, 48);
    let _ = eagle3.train_ttt(&target, &seqs, 12, 3, 0.015);

    for wk in [
        WorkloadKind::CodeEditRepetitive,
        WorkloadKind::MixedProseCode,
        WorkloadKind::CreativeHighEntropyProse,
    ] {
        let turns = corpus.build_workload_turns(wk);
        let (plain_m, plain_out) = run_workload_strategy(
            &target,
            &eagle3,
            &turns,
            wk.as_str(),
            BenchStrategy::PlainDecodeBaseline,
            None,
        );
        let base_ref = BaselineRef {
            outputs: &plain_out,
            wall_ms: plain_m.cpu_wall_ms,
            single_verify_ms: plain_m.cpu_p50_itl_ms,
        };
        for &strat in &BenchStrategy::ALL[1..] {
            let (m, out) =
                run_workload_strategy(&target, &eagle3, &turns, wk.as_str(), strat, Some(base_ref));
            assert_eq!(
                m.output_bit_exact_pct,
                100.0,
                "strategy {} on {} must be 100.0% bit-exact",
                strat.name(),
                wk.as_str()
            );
            assert_eq!(m.quality_regression_pp, 0.0);
            assert_eq!(out, plain_out);
        }
    }
}

#[test]
fn test_11_selector_stand_down_hysteresis_and_compute_bound_guard() {
    let mut sel = WorkloadAwareSelector::new(SelectorConfig::default());

    // 1. Large-batch compute-bound guard must immediately choose PlainDecode
    let big_batch = sel.select(StepSignals {
        batch_size: 32,
        lookup_match_len: 8,
        lookup_draft_avail: 6,
        recent_entropy_nats: 0.5,
    });
    assert_eq!(big_batch, DraftStrategy::PlainDecode);
    assert_eq!(sel.telemetry.compute_bound_stand_downs, 1);

    // 2. Calibrate PlainDecode at 1.0 ms/tok
    let s0 = sel.select(StepSignals {
        batch_size: 1,
        lookup_match_len: 0,
        lookup_draft_avail: 0,
        recent_entropy_nats: 1.0,
    });
    assert_eq!(s0, DraftStrategy::PlainDecode);
    sel.observe(StepFeedback {
        strategy: DraftStrategy::PlainDecode,
        accepted_drafts: 0,
        draft_ms: 0.0,
        verify_ms: 1.0,
        target_entropy_nats: 1.0,
    });

    // 3. Feed 2 consecutive high-entropy zero-accept steps -> must trigger stand-down cooldown!
    for _ in 0..2 {
        sel.observe(StepFeedback {
            strategy: DraftStrategy::Eagle3Tree {
                depth: 4,
                fanout: 2,
                max_nodes: 10,
            },
            accepted_drafts: 0,
            draft_ms: 0.4,
            verify_ms: 1.5,
            target_entropy_nats: 3.1,
        });
    }
    assert_eq!(sel.telemetry.stand_down_events, 1);

    // During cooldown without strong suffix match, selector must stay on PlainDecode (no thrashing)
    for _ in 0..5 {
        let chosen = sel.select(StepSignals {
            batch_size: 1,
            lookup_match_len: 0,
            lookup_draft_avail: 0,
            recent_entropy_nats: 3.0,
        });
        assert_eq!(chosen, DraftStrategy::PlainDecode);
    }
}

#[test]
fn test_12_selector_prefers_simpler_prompt_lookup_on_code_edits() {
    let mut sel = WorkloadAwareSelector::new(SelectorConfig::default());
    // Seed plain baseline
    let _ = sel.select(StepSignals {
        batch_size: 1,
        lookup_match_len: 0,
        lookup_draft_avail: 0,
        recent_entropy_nats: 0.8,
    });
    sel.observe(StepFeedback {
        strategy: DraftStrategy::PlainDecode,
        accepted_drafts: 0,
        draft_ms: 0.0,
        verify_ms: 1.0,
        target_entropy_nats: 0.8,
    });

    // Live 6-token suffix match in code edit diff -> must pick simpler PromptLookup, not Eagle3!
    let chosen = sel.select(StepSignals {
        batch_size: 1,
        lookup_match_len: 6,
        lookup_draft_avail: 6,
        recent_entropy_nats: 0.4,
    });
    assert!(matches!(chosen, DraftStrategy::PromptLookup { .. }));
}

#[test]
fn test_13_negative_regime_1_high_entropy_slowdown_and_stand_down() {
    let corpus = load_corpus();
    let target = TargetModel::from_grounded_corpus(&corpus, 1346);
    let mut eagle3 = Eagle3DraftHead::new_untrained(&target, 1346);
    let seqs = corpus.training_sequences(20, 48);
    let _ = eagle3.train_ttt(&target, &seqs, 16, 3, 0.015);

    let creative = corpus.build_workload_turns(WorkloadKind::CreativeHighEntropyProse);
    let (plain_m, plain_out) = run_workload_strategy(
        &target,
        &eagle3,
        &creative,
        "creative_high_entropy_prose",
        BenchStrategy::PlainDecodeBaseline,
        None,
    );
    let base_ref = BaselineRef {
        outputs: &plain_out,
        wall_ms: plain_m.cpu_wall_ms,
        single_verify_ms: plain_m.cpu_p50_itl_ms,
    };
    let (tree_m, _) = run_workload_strategy(
        &target,
        &eagle3,
        &creative,
        "creative_high_entropy_prose",
        BenchStrategy::Eagle3TreeFixed,
        Some(base_ref),
    );
    let (sel_m, _) = run_workload_strategy(
        &target,
        &eagle3,
        &creative,
        "creative_high_entropy_prose",
        BenchStrategy::WorkloadAwareSelector,
        Some(base_ref),
    );

    assert!(
        tree_m.cpu_speedup_vs_plain < 0.85,
        "fixed Eagle3Tree must honestly regress on high-entropy creative prose (got {:.2}x)",
        tree_m.cpu_speedup_vs_plain
    );
    let tel = sel_m.selector_telemetry.expect("selector telemetry");
    assert!(
        tel.stand_down_events >= 1,
        "selector must trigger stand-down"
    );
    assert!(
        tel.plain_steps > tel.linear_steps + tel.tree_steps,
        "selector must spend majority of high-entropy steps in PlainDecode stand-down"
    );
    assert!(
        sel_m.cpu_wall_ms < tree_m.cpu_wall_ms * 0.70,
        "selector stand-down must drastically beat fixed Eagle3Tree on high-entropy prose"
    );
}

#[test]
fn test_14_negative_regime_2_large_batch_compute_bound_rejection() {
    let specs = HybridModelSpec::shared_matrix();
    assert_eq!(specs.len(), 3);
    for spec in &specs {
        let rows_b64 = evaluate_large_batch_negative_regime(spec, 64, 2.04, 2.29);
        let lin = rows_b64
            .iter()
            .find(|r| r.strategy == "eagle3_linear_fixed")
            .unwrap();
        let tree = rows_b64
            .iter()
            .find(|r| r.strategy == "eagle3_tree_fixed")
            .unwrap();
        let sel = rows_b64
            .iter()
            .find(|r| r.strategy == "workload_aware_selector")
            .unwrap();

        assert!(
            lin.speedup_vs_plain < 1.0,
            "{} B=64 linear must regress",
            spec.name
        );
        assert_eq!(lin.gate_verdict, "REJECTED");
        assert!(
            tree.speedup_vs_plain < 0.70,
            "{} B=64 tree must regress",
            spec.name
        );
        assert_eq!(tree.gate_verdict, "REJECTED");
        assert_eq!(sel.speedup_vs_plain, 1.0);
        assert_eq!(sel.gate_verdict, "ACCEPTED");
    }
}

#[test]
fn test_15_negative_regime_3_prompt_lookup_beats_eagle3_on_code_edit() {
    let corpus = load_corpus();
    let target = TargetModel::from_grounded_corpus(&corpus, 1346);
    let mut eagle3 = Eagle3DraftHead::new_untrained(&target, 1346);
    let seqs = corpus.training_sequences(20, 48);
    let _ = eagle3.train_ttt(&target, &seqs, 16, 3, 0.015);

    let code_edit = corpus.build_workload_turns(WorkloadKind::CodeEditRepetitive);
    let (plain_m, plain_out) = run_workload_strategy(
        &target,
        &eagle3,
        &code_edit,
        "code_edit_repetitive",
        BenchStrategy::PlainDecodeBaseline,
        None,
    );
    let base_ref = BaselineRef {
        outputs: &plain_out,
        wall_ms: plain_m.cpu_wall_ms,
        single_verify_ms: plain_m.cpu_p50_itl_ms,
    };
    let (lookup_m, _) = run_workload_strategy(
        &target,
        &eagle3,
        &code_edit,
        "code_edit_repetitive",
        BenchStrategy::PromptLookupOnly,
        Some(base_ref),
    );
    let (tree_m, _) = run_workload_strategy(
        &target,
        &eagle3,
        &code_edit,
        "code_edit_repetitive",
        BenchStrategy::Eagle3TreeFixed,
        Some(base_ref),
    );

    assert!(
        lookup_m.mean_tau > tree_m.mean_tau + 1.5,
        "simpler PromptLookup must beat Eagle3Tree tau on repetitive code edits: {:.2} vs {:.2}",
        lookup_m.mean_tau,
        tree_m.mean_tau
    );
    assert!(
        lookup_m.cpu_wall_ms < tree_m.cpu_wall_ms,
        "simpler PromptLookup must beat Eagle3Tree wall time on repetitive code edits"
    );
}

#[test]
fn test_16_acceptance_gate_rejects_quality_and_end_to_end_regressions() {
    let (v_exact, _) = evaluate_acceptance_gate("eagle3_tree", 99.5, -0.5, 10.0, 20.0, 2.0, 4.0);
    assert_eq!(v_exact, "REJECTED");

    let (v_slow, _) = evaluate_acceptance_gate("eagle3_tree", 100.0, 0.0, 22.0, 20.0, 3.8, 4.0);
    assert_eq!(v_slow, "REJECTED");

    let (v_tail, _) = evaluate_acceptance_gate("eagle3_tree", 100.0, 0.0, 18.0, 20.0, 4.5, 4.0);
    assert_eq!(v_tail, "REJECTED");

    let (v_ok, _) =
        evaluate_acceptance_gate("workload_aware_selector", 100.0, 0.0, 10.0, 20.0, 2.1, 4.0);
    assert_eq!(v_ok, "ACCEPTED");
}

#[test]
fn test_17_full_benchmark_json_receipt_and_4b_9b_27b_selector_dominance() {
    let corpus = load_corpus();
    let env = BenchEnvironment {
        cpu_arch: std::env::consts::ARCH.to_string(),
        cpu_model: "test-cpu".to_string(),
        os_kernel: std::env::consts::OS.to_string(),
        gpu_driver: "CPU-only test harness".to_string(),
        git_commit: "test".to_string(),
        rustc_version: "rustc".to_string(),
    };
    let report = run_full_benchmark(corpus, env, 1346);
    assert!(report.stochastic_verification.passed);
    assert_eq!(report.cpu_workloads.len(), 20); // 4 workloads x 5 strategies
    assert_eq!(report.matrix_scorecard.len(), 15); // 3 models x 5 strategies

    for w in &report.cpu_workloads {
        assert_eq!(w.output_bit_exact_pct, 100.0);
        assert_eq!(w.quality_regression_pp, 0.0);
    }

    // Workload-aware selector must win end-to-end 20-turn wall time on 4B, 9B, and 27B!
    for model in ["4B-Hybrid", "9B-Hybrid", "27B-Hybrid"] {
        let sel = report
            .matrix_scorecard
            .iter()
            .find(|r| r.model == model && r.strategy == "workload_aware_selector")
            .unwrap();
        let tree = report
            .matrix_scorecard
            .iter()
            .find(|r| r.model == model && r.strategy == "eagle3_tree_fixed")
            .unwrap();
        let lookup = report
            .matrix_scorecard
            .iter()
            .find(|r| r.model == model && r.strategy == "prompt_lookup_only")
            .unwrap();
        assert_eq!(sel.gate_verdict, "ACCEPTED");
        assert!(sel.agent_loop_20_turn_s < tree.agent_loop_20_turn_s);
        assert!(sel.agent_loop_20_turn_s < lookup.agent_loop_20_turn_s);
        assert!(sel.speedup_vs_plain > 1.80);
    }

    let json = format_report_json(&report);
    assert!(json.contains("\"receipt_sha256\""));
    assert!(json.contains("\"hanzoai/engine#55\""));
    assert!(json.contains("\"neg_high_entropy_spec_slowdown\""));
    assert!(json.contains("\"neg_large_batch_verify_overhead\""));
    assert!(json.contains("\"neg_eagle3_vs_prompt_lookup_code_edit\""));
}
