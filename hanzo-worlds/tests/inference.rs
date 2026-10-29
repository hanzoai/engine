use std::thread;

use hanzo_worlds::{
    AdmissibilityPolicy, Counts, GenerativeModel, RegistryError, Request, WorldInference,
    WorldRegistry,
};

fn identity_2() -> Vec<Vec<f64>> {
    vec![vec![1., 0.], vec![0., 1.]]
}

fn swap_2() -> Vec<Vec<f64>> {
    vec![vec![0., 1.], vec![1., 0.]]
}

fn world() -> WorldInference {
    let id = identity_2();
    let m = GenerativeModel::new(
        vec![id.clone(), id.clone()],
        vec![id, swap_2()],
        vec![0.9, 0.1],
    )
    .unwrap();
    WorldInference::with_version(m, 4.0, "two-state-sim-v1").unwrap()
}

fn epistemic_world() -> WorldInference {
    let m = GenerativeModel::new(
        vec![identity_2(), vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
        vec![identity_2(), identity_2()],
        vec![0.5, 0.5],
    )
    .unwrap();
    WorldInference::with_version(m, 4.0, "epistemic-v1").unwrap()
}

#[test]
fn infer_observe_then_plan_separates_inference_recommendation_and_execution() {
    let r = world()
        .infer(
            &Request {
                belief: vec![0.5, 0.5],
                observation: Some((0, 0)),
                policies: vec![vec![0], vec![1]],
            },
            &[0.5, 0.5],
            &[true, true],
        )
        .unwrap();

    assert_eq!(r.model_version, "two-state-sim-v1");
    let post = r.inference.posterior.as_ref().unwrap();
    assert_eq!(post.belief, vec![1.0, 0.0]);
    assert!((post.free_energy + 0.5f64.ln()).abs() < 1e-12);
    assert_eq!(r.inference.policies.selected, 0);
    assert_eq!(r.recommendation.policy_index, 0);
    assert_eq!(r.recommendation.actions, vec![0]);
    assert_eq!(
        r.recommendation.probability,
        r.inference.policies.probabilities[0]
    );
    assert_eq!(
        r.recommendation.expected_free_energy,
        r.inference.policies.terms[0].expected_free_energy
    );
    assert!(r.recommendation.advisory_only);
    assert!(!r.execution_authorized);
}

#[test]
fn efe_decomposition_and_epistemic_preference_hold() {
    let w = epistemic_world();
    let r = w
        .infer(
            &Request {
                belief: vec![0.5, 0.5],
                observation: None,
                policies: vec![vec![0], vec![1]],
            },
            &[0.5, 0.5],
            &[true, true],
        )
        .unwrap();
    assert_eq!(r.recommendation.policy_index, 0);
    assert!(r.recommendation.probability > 0.9);
    for t in &r.inference.policies.terms {
        assert!((t.risk + t.ambiguity - t.expected_free_energy).abs() < 1e-12);
        assert!((t.extrinsic - t.information_gain - t.expected_free_energy).abs() < 1e-12);
    }
    assert!((r.inference.policies.terms[0].information_gain - 2f64.ln()).abs() < 1e-12);
    assert!(r.inference.policies.terms[1].information_gain.abs() < 1e-12);
}

#[test]
fn host_mask_and_strict_schema_cannot_be_overridden_by_request() {
    let req = Request {
        belief: vec![1.0, 0.0],
        observation: None,
        policies: vec![vec![0], vec![1]],
    };
    let masked = world().infer(&req, &[0.5, 0.5], &[false, true]).unwrap();
    assert_eq!(masked.inference.policies.probabilities, vec![0.0, 1.0]);
    assert_eq!(masked.recommendation.policy_index, 1);
    assert!(masked.recommendation.advisory_only);
    assert!(!masked.execution_authorized);
    assert!(world().infer(&req, &[0.5, 0.5], &[false, false]).is_err());

    for malicious in [
        r#"{"belief":[1,0],"policies":[[0]],"observation":null,"admissible":[true]}"#,
        r#"{"belief":[1,0],"policies":[[0]],"allowed":[true]}"#,
        r#"{"belief":[1,0],"policies":[[0]],"habits":[1.0]}"#,
        r#"{"belief":[1,0],"policies":[[0]],"preferences":[0.99,0.01]}"#,
        r#"{"belief":[1,0],"policies":[[0]],"precision":100.0}"#,
        r#"{"belief":[1,0],"policies":[[0]],"tenant_id":"victim"}"#,
        r#"{"belief":[1,0],"policies":[[0]],"execution_authorized":true}"#,
        r#"{"belief":[1,0],"policies":[[0]],"model_version":"override"}"#,
    ] {
        assert!(
            serde_json::from_str::<Request>(malicious).is_err(),
            "malicious override payload must be rejected: {malicious}"
        );
    }
}

#[test]
fn registry_enforces_tenant_isolation_versioning_and_host_admissibility() {
    let reg = WorldRegistry::new();
    // Tenant A prefers action 0 (stay in state 0), but host forbids action 0 via prefix rule.
    reg.register(
        "tenant-a",
        "world_1",
        world(),
        AdmissibilityPolicy::allow_actions([0, 1]).with_forbidden_prefix(vec![0]),
    )
    .unwrap();
    // Tenant B registers the same world_id with an epistemic model under a distinct revision.
    reg.register(
        "tenant-b",
        "world_1",
        epistemic_world(),
        AdmissibilityPolicy::allow_actions([0, 1]),
    )
    .unwrap();

    let req = Request {
        belief: vec![0.5, 0.5],
        observation: None,
        policies: vec![vec![0], vec![1]],
    };

    // Tenant A gets its own model and host-enforced safety mask (action 0 forbidden -> action 1).
    let ra = reg.infer("tenant-a", "world_1", &req).unwrap();
    assert_eq!(ra.world_id.as_deref(), Some("world_1"));
    assert_eq!(ra.model_version, "two-state-sim-v1");
    assert_eq!(ra.inference.policies.probabilities, vec![0.0, 1.0]);
    assert_eq!(ra.recommendation.policy_index, 1);
    assert!(ra.recommendation.advisory_only);
    assert!(!ra.execution_authorized);

    // Tenant B gets its distinct epistemic model and selects informative action 0.
    let rb = reg.infer("tenant-b", "world_1", &req).unwrap();
    assert_eq!(rb.model_version, "epistemic-v1");
    assert_eq!(rb.recommendation.policy_index, 0);
    assert!(rb.recommendation.advisory_only);
    assert!(!rb.execution_authorized);

    // Cross-tenant lookup returns NotFound (zero existence leakage).
    assert_eq!(
        reg.infer("tenant-c", "world_1", &req).unwrap_err(),
        RegistryError::NotFound
    );
    // Missing/blank tenant fails closed with Unauthorized.
    assert!(matches!(
        reg.infer("", "world_1", &req),
        Err(RegistryError::Unauthorized(_))
    ));
    assert!(matches!(
        reg.infer("   ", "world_1", &req),
        Err(RegistryError::Unauthorized(_))
    ));
    // All-inadmissible policies fail closed with Inadmissible (422).
    let blocked_req = Request {
        belief: vec![1.0, 0.0],
        observation: None,
        policies: vec![vec![0], vec![0, 1]],
    };
    assert!(matches!(
        reg.infer("tenant-a", "world_1", &blocked_req),
        Err(RegistryError::Inadmissible(_))
    ));
}

#[test]
fn rejects_malformed_inputs_and_path_traversal() {
    let reg = WorldRegistry::new();
    reg.register(
        "tenant-a",
        "world_1",
        world(),
        AdmissibilityPolicy::allow_actions([0, 1]).with_max_horizon(4),
    )
    .unwrap();

    let valid_req = Request {
        belief: vec![0.5, 0.5],
        observation: None,
        policies: vec![vec![0]],
    };
    for bad_id in ["", "../weights", "a/b", "world id", "world?x=1"] {
        assert!(
            matches!(
                reg.infer("tenant-a", bad_id, &valid_req),
                Err(RegistryError::MalformedInput(_))
            ),
            "bad world id must fail with MalformedInput: {bad_id:?}"
        );
    }

    let bad_requests = vec![
        Request {
            belief: vec![0.2, 0.7],
            observation: None,
            policies: vec![vec![0]],
        },
        Request {
            belief: vec![f64::NAN, 0.5],
            observation: None,
            policies: vec![vec![0]],
        },
        Request {
            belief: vec![f64::INFINITY, 0.0],
            observation: None,
            policies: vec![vec![0]],
        },
        Request {
            belief: vec![-0.1, 1.1],
            observation: None,
            policies: vec![vec![0]],
        },
        Request {
            belief: vec![1.0],
            observation: None,
            policies: vec![vec![0]],
        },
        // Impossible zero-evidence observation: belief [0, 1], identity A, observing outcome 0.
        Request {
            belief: vec![0.0, 1.0],
            observation: Some((0, 0)),
            policies: vec![vec![0]],
        },
        // Out-of-bounds observation action.
        Request {
            belief: vec![0.5, 0.5],
            observation: Some((9, 0)),
            policies: vec![vec![0]],
        },
        // Empty candidate policies.
        Request {
            belief: vec![0.5, 0.5],
            observation: None,
            policies: vec![],
        },
        // Empty policy sequence.
        Request {
            belief: vec![0.5, 0.5],
            observation: None,
            policies: vec![vec![]],
        },
        // Exceeds host max_horizon = 4.
        Request {
            belief: vec![0.5, 0.5],
            observation: None,
            policies: vec![vec![0, 0, 0, 0, 0]],
        },
    ];
    for req in bad_requests {
        assert!(
            matches!(
                reg.infer("tenant-a", "world_1", &req),
                Err(RegistryError::MalformedInput(_))
            ),
            "malformed request must fail: {req:?}"
        );
    }
}

#[test]
fn numerical_stability_extreme_precision_and_horizon() {
    let w = world();
    for precision in [0.0, 1e-9, 1.0, 25.0, 100.0] {
        let m = GenerativeModel::new(
            vec![identity_2(), identity_2()],
            vec![identity_2(), swap_2()],
            vec![0.9, 0.1],
        )
        .unwrap();
        let wi = WorldInference::with_version(m, precision, "stab-v1").unwrap();
        let req = Request {
            belief: vec![1e-15, 1.0 - 1e-15],
            observation: None,
            policies: vec![vec![0; 16], vec![1; 16]],
        };
        let r = wi.infer(&req, &[0.5, 0.5], &[true, true]).unwrap();
        let sum: f64 = r.inference.policies.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "precision={precision} sum={sum}");
        assert!(r
            .inference
            .policies
            .probabilities
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0));
        assert!(r.recommendation.expected_free_energy.is_finite());
        assert!(!r.execution_authorized);
    }
    // Dirichlet count learning preserves normalization & immutability.
    let counts = Counts {
        a: vec![vec![vec![1.0, 1.0], vec![1.0, 1.0]]],
        b: vec![vec![vec![1.0, 1.0], vec![1.0, 1.0]]],
    };
    let next = counts
        .learned(&[0.2, 0.8], 0, 0, 1.0, vec![0.5, 0.5])
        .unwrap();
    assert!(next.model(vec![0.5, 0.5]).is_ok());
    assert_eq!(w.model_version(), "two-state-sim-v1");
}

#[test]
fn concurrent_multi_tenant_isolation() {
    let reg = WorldRegistry::new();
    for t in 0..16usize {
        let pref = if t % 2 == 0 {
            vec![0.95, 0.05]
        } else {
            vec![0.05, 0.95]
        };
        let m = GenerativeModel::new(
            vec![identity_2(), identity_2()],
            vec![identity_2(), swap_2()],
            pref,
        )
        .unwrap();
        let wi = WorldInference::with_version(m, 4.0, format!("rev-{t}")).unwrap();
        reg.register(
            format!("tenant-{t}"),
            "shared-name",
            wi,
            AdmissibilityPolicy::allow_actions([0, 1]),
        )
        .unwrap();
    }

    let handles: Vec<_> = (0..16usize)
        .map(|t| {
            let reg = reg.clone();
            thread::spawn(move || {
                let req = Request {
                    belief: vec![1.0, 0.0],
                    observation: None,
                    policies: vec![vec![0], vec![1]],
                };
                for _ in 0..100 {
                    let r = reg
                        .infer(&format!("tenant-{t}"), "shared-name", &req)
                        .unwrap();
                    assert_eq!(r.model_version, format!("rev-{t}"));
                    // Even tenants prefer state 0 (action 0); odd tenants prefer state 1 (action 1).
                    assert_eq!(r.recommendation.policy_index, t % 2);
                    assert!(r.recommendation.advisory_only);
                    assert!(!r.execution_authorized);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
