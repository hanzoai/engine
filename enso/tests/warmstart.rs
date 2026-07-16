//! Warm-start validation: fit `W` + arms from bench-shaped per-item data, persist it
//! in the shipped format, load it back, and prove the learned policy is real — it
//! prefers the measured-best arm on knowledge tasks and is not degenerate.
//!
//! Mirrors the `enso fit` binary's pipeline (ingest → fit_base → persist::save_base →
//! persist::load) with a fixture reflecting the measured GPQA-Diamond accuracies
//! (gpt-5.5 0.904, opus-4.8 0.869, fable-5 0.813, deepseek-v4-pro 0.763), so the
//! assertion tracks reality without reading external files.

use enso::{fit_base, persist, Enso, Featurizer, HashFeaturizer, Learner, Policy};
use hanzo_router::classify::Request;
use hanzo_router::registry::{Level, Modality, Registry, Task};
use hanzo_router::{Slo, User};

const KNOWLEDGE_TASKS: [Task; 3] = [Task::General, Task::Reasoning, Task::Math];

/// Measured GPQA-Diamond per-arm accuracies (the deployed data). deepseek also carries
/// a LiveCodeBench (code) result — so the fixture mirrors the real cross-task corpus.
const KNOWLEDGE: [(&str, f64); 4] = [
    ("gpt-5.5", 0.904),
    ("opus-4.8", 0.869),
    ("fable-5", 0.813),
    ("deepseek-v4-pro", 0.763),
];
const CODE: [(&str, f64); 1] = [("deepseek-v4-pro", 0.48)];
const N_ITEMS: usize = 198;

/// The enso cloud default: maximize measured quality subject to hard ceilings — the
/// soft cost/latency weights are zeroed (the gateway sends no soft trade).
fn quality_slo() -> Slo {
    Slo { lambda_cost: 0.0, mu_latency: 0.0, ..Default::default() }
}

fn push(samples: &mut Vec<enso::EvalSample>, model: &str, task: Task, acc: f64, n: usize) {
    let n_correct = (acc * n as f64).round() as usize;
    for i in 0..n {
        samples.push(enso::EvalSample {
            task,
            modality: Modality::Text,
            approx_tokens: 512,
            text: String::new(),
            model: model.into(),
            level: Level::Balanced,
            quality: if i < n_correct { 1.0 } else { 0.0 },
            latency_ms: 5000.0,
            cost: 0.02,
            vram_gb: 0.0,
            max_context: 200_000,
        });
    }
}

/// Mirror the `enso fit` binary EXACTLY: bench records → tasks, ingest → arms, ridge
/// over (one-hot task x, single-column quality p, per-item reward), persist.
fn fit_and_persist(path: &std::path::Path) {
    let mut samples = Vec::new();
    for (m, acc) in KNOWLEDGE {
        for t in KNOWLEDGE_TASKS {
            push(&mut samples, m, t, acc, N_ITEMS);
        }
    }
    for (m, acc) in CODE {
        push(&mut samples, m, Task::Code, acc, 175);
    }
    let table = enso::ingest(&samples);
    let tuples: Vec<(Vec<f64>, Vec<f64>, f64)> = samples
        .iter()
        .map(|s| {
            let ti = s.task.index();
            let mut x = vec![0.0f64; enso::featurize::FEAT_DIM];
            x[ti] = 1.0;
            let mut p = vec![0.0f64; enso::PROFILE_DIM];
            p[ti] = table.by_model(&s.model).unwrap().quality[ti];
            (x, p, s.quality)
        })
        .collect();
    let policy: Policy = fit_base(tuples.iter().map(|(x, p, y)| (x.as_slice(), p.as_slice(), *y)), 0.01);
    persist::save_base(path, &Learner::new(policy, 1.0, 0.5), &table).unwrap();
}

fn pick(enso: &Enso, text: &str, tokens: usize) -> String {
    let req = Request {
        text: text.into(),
        approx_tokens: tokens,
        has_media: false,
        task_hint: None,
        modality_hint: None,
    };
    let (route, _) = enso.route_explained(&req, &User::new("org/tester"), &quality_slo(), &Registry::new(vec![]));
    route.model
}

#[test]
fn warm_start_prefers_gpt55_on_knowledge_and_routes_code_elsewhere() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heads-base.safetensors");
    fit_and_persist(&path);

    // Load exactly as serving does.
    let (learner, table) = persist::load(&path, 1.0, 0.5).unwrap();
    assert_eq!(table.profiles.len(), 4, "all four arms must be baked in");
    let enso = Enso::new(table, learner);

    // Knowledge → the measured-best arm gpt-5.5 (90.4 vs deepseek 76.3). Real deployed
    // behavior fit from the GPQA-Diamond per-item accuracies, at the quality-first SLO.
    for q in [
        "Explain why the reaction rate depends on temperature; reason step by step.",
        "Which quantum number determines the shape of an atomic orbital? Analyze.",
        "Prove that the given integral converges and solve for its value.",
    ] {
        let arm = pick(&enso, q, 400);
        assert_eq!(arm, "gpt-5.5", "knowledge must route to the measured-best arm, got {arm} for {q:?}");
        assert_ne!(arm, "deepseek-v4-pro", "must not pick the weakest arm on knowledge");
    }

    // Non-degenerate: a code request routes to the arm with code eval data (deepseek),
    // NOT the knowledge winner — the policy is a real function of the task.
    let code = pick(&enso, "fix this ```rust``` compile bug in the function", 300);
    assert_eq!(code, "deepseek-v4-pro", "code must route to the arm with code eval data, got {code}");
    assert_ne!(code, "gpt-5.5", "degenerate: code and knowledge both routed to gpt-5.5");
}

/// Non-degeneracy of the MACHINERY: given eval data where different arms win different
/// tasks, the fitted policy routes each task to its measured-best arm — the policy is a
/// real function of the request, not a constant. A code-specialist and a
/// reasoning-specialist, each dominant in its own task.
#[test]
fn warm_start_routes_different_tasks_to_different_arms() {
    let mut samples = Vec::new();
    let arm_task_quality = [
        ("reasoner", Task::Reasoning, 0.95),
        ("reasoner", Task::Code, 0.30),
        ("coder", Task::Reasoning, 0.30),
        ("coder", Task::Code, 0.95),
    ];
    for (model, task, acc) in arm_task_quality {
        let n_correct = (acc * 100.0) as usize;
        for i in 0..100 {
            samples.push(enso::EvalSample {
                task,
                modality: Modality::Text,
                approx_tokens: 256,
                text: String::new(),
                model: model.into(),
                level: Level::Balanced,
                quality: if i < n_correct { 1.0 } else { 0.0 },
                latency_ms: 2000.0,
                cost: 0.01,
                vram_gb: 0.0,
                max_context: 100_000,
            });
        }
    }
    let table = enso::ingest(&samples);
    let tuples: Vec<(Vec<f64>, Vec<f64>, f64)> = samples
        .iter()
        .map(|s| {
            let ti = s.task.index();
            let mut x = vec![0.0f64; enso::featurize::FEAT_DIM];
            x[ti] = 1.0;
            let mut p = vec![0.0f64; enso::PROFILE_DIM];
            p[ti] = table.by_model(&s.model).unwrap().quality[ti];
            (x, p, s.quality)
        })
        .collect();
    let policy = fit_base(tuples.iter().map(|(x, p, y)| (x.as_slice(), p.as_slice(), *y)), 0.01);
    let enso = Enso::new(table, Learner::new(policy, 1.0, 0.5));

    let reasoning = pick(&enso, "why does this hold? reason step by step and explain how", 200);
    let code = pick(&enso, "fix this ```rust``` compile bug in the function", 200);
    assert_eq!(reasoning, "reasoner", "reasoning task must route to the reasoning specialist, got {reasoning}");
    assert_eq!(code, "coder", "code task must route to the code specialist, got {code}");
    assert_ne!(reasoning, code, "policy is degenerate — both tasks routed to the same arm");
}

#[test]
fn online_observe_moves_the_user_policy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heads-base.safetensors");
    fit_and_persist(&path);
    let (learner, table) = persist::load(&path, 1.0, 0.5).unwrap();
    let mut enso = Enso::new(table, learner);

    // A user who repeatedly rewards deepseek on a code-like request drifts toward it —
    // proves the online loop mutates per-user theta (feature-keyed observe path).
    let x = HashFeaturizer::default().featurize(&Request {
        text: "write a quick python one-liner".into(),
        approx_tokens: 20,
        has_media: false,
        task_hint: None,
        modality_hint: None,
    });
    let before = enso.learner().delta_norm("org/alice");
    for _ in 0..25 {
        assert!(enso.observe_features("org/alice", &x, "deepseek-v4-pro", 1.0));
    }
    let after = enso.learner().delta_norm("org/alice");
    assert!(after > before, "online observe must move theta off the base prior");

    // An unknown arm is a no-op (reported false so the caller can 404).
    assert!(!enso.observe_features("org/alice", &x, "no-such-model", 1.0));
}
