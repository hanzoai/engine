//! Benchmark & qualification suite for `hanzo-worlds`:
//! mathematical correctness, numerical stability, latency, memory footprint,
//! malformed input rejection, and multi-tenant isolation.

use std::mem::size_of;
use std::thread;
use std::time::Instant;

use hanzo_worlds::{
    AdmissibilityPolicy, GenerativeModel, RegistryError, Request, Response, WorldInference,
    WorldRegistry,
};

fn build_cyclic_model(states: usize, actions: usize) -> GenerativeModel {
    let mut a = Vec::with_capacity(actions);
    let mut b = Vec::with_capacity(actions);
    for act in 0..actions {
        let mut a_mat = vec![vec![0.0; states]; states];
        let mut b_mat = vec![vec![0.0; states]; states];
        let noise = 0.05 + 0.05 * ((act % 3) as f64);
        let stay = 1.0 - noise;
        let spread = if states > 1 {
            noise / ((states - 1) as f64)
        } else {
            0.0
        };
        for s in 0..states {
            let next_s = (s + act) % states;
            for row in 0..states {
                a_mat[row][s] = if row == s { stay } else { spread };
                b_mat[row][s] = if row == next_s { stay } else { spread };
            }
        }
        a.push(a_mat);
        b.push(b_mat);
    }
    let mut c = vec![0.1 / (states as f64); states];
    c[0] += 0.9;
    let sum: f64 = c.iter().sum();
    for x in &mut c {
        *x /= sum;
    }
    GenerativeModel::new(a, b, c).expect("valid synthetic model")
}

fn percentile_us(sorted_ns: &[u128], pct: f64) -> f64 {
    let idx = ((sorted_ns.len().saturating_sub(1)) as f64 * pct).round() as usize;
    sorted_ns[idx.min(sorted_ns.len() - 1)] as f64 / 1_000.0
}

fn bench_latency(
    label: &str,
    states: usize,
    actions: usize,
    policies_n: usize,
    horizon: usize,
    iters: usize,
) {
    let model = build_cyclic_model(states, actions);
    let engine = WorldInference::with_version(model, 4.0, format!("bench-{states}s")).unwrap();
    let reg = WorldRegistry::new();
    reg.register(
        "bench-tenant",
        "bench-world",
        engine,
        AdmissibilityPolicy::allow_actions(0..actions),
    )
    .unwrap();

    let belief = vec![1.0 / (states as f64); states];
    let policies: Vec<Vec<usize>> = (0..policies_n)
        .map(|i| (0..horizon).map(|h| (i + h) % actions).collect())
        .collect();
    let req = Request {
        belief,
        observation: Some((0, 0)),
        policies,
    };

    // Warmup
    for _ in 0..5 {
        let r = reg.infer("bench-tenant", "bench-world", &req).unwrap();
        assert!(!r.execution_authorized && r.recommendation.advisory_only);
    }

    let mut samples_ns = Vec::with_capacity(iters);
    let mut max_decomp_err = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = reg.infer("bench-tenant", "bench-world", &req).unwrap();
        samples_ns.push(t0.elapsed().as_nanos());
        for t in &r.inference.policies.terms {
            let e1 = (t.risk + t.ambiguity - t.expected_free_energy).abs();
            let e2 = (t.extrinsic - t.information_gain - t.expected_free_energy).abs();
            max_decomp_err = max_decomp_err.max(e1).max(e2);
        }
        assert!(!r.execution_authorized);
        assert!(r.recommendation.advisory_only);
    }
    samples_ns.sort_unstable();
    let mean_us = (samples_ns.iter().sum::<u128>() as f64) / (iters as f64) / 1_000.0;
    println!(
        "latency[{label}] states={states} actions={actions} policies={policies_n} horizon={horizon} iters={iters}: mean={mean_us:.2}us p50={:.2}us p95={:.2}us p99={:.2}us max_efe_err={max_decomp_err:.3e}",
        percentile_us(&samples_ns, 0.50),
        percentile_us(&samples_ns, 0.95),
        percentile_us(&samples_ns, 0.99),
    );
    assert!(
        max_decomp_err < 1e-11,
        "EFE decomposition drifted: {max_decomp_err}"
    );
}

fn bench_numerical_stability() {
    let model = build_cyclic_model(16, 4);
    let mut checks = 0usize;
    for precision in [0.0, 1e-12, 0.5, 4.0, 32.0, 100.0] {
        let engine = WorldInference::with_version(model.clone(), precision, "stab-v1").unwrap();
        let mut belief = vec![1e-15 / 15.0; 16];
        belief[0] = 1.0 - 1e-15;
        let policies = vec![vec![0; 16], vec![1; 16], vec![2; 16], vec![3; 16]];
        let r = engine
            .infer(
                &Request {
                    belief,
                    observation: Some((0, 0)),
                    policies,
                },
                &[0.25, 0.25, 0.25, 0.25],
                &[true, true, true, true],
            )
            .unwrap();
        let sum: f64 = r.inference.policies.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(r
            .inference
            .policies
            .probabilities
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0));
        assert!(r.recommendation.expected_free_energy.is_finite());
        assert!(!r.execution_authorized);
        checks += 1;
    }
    println!(
        "numerical_stability: PASS ({checks} extreme precision/belief/horizon sweeps, 0 NaN/Inf)"
    );
}

fn bench_malformed_rejection() {
    let reg = WorldRegistry::new();
    let engine = WorldInference::with_version(build_cyclic_model(4, 2), 4.0, "val-v1").unwrap();
    reg.register(
        "t1",
        "w1",
        engine,
        AdmissibilityPolicy::allow_actions([0, 1]),
    )
    .unwrap();

    let bad_json = [
        r#"{"belief":[0.25,0.25,0.25,0.25],"policies":[[0]],"admissible":[true]}"#,
        r#"{"belief":[0.25,0.25,0.25,0.25],"policies":[[0]],"preferences":[1,0,0,0]}"#,
        r#"{"belief":[0.25,0.25,0.25,0.25],"policies":[[0]],"execution_authorized":true}"#,
        r#"{"belief":[0.25,0.25,0.25,0.25],"policies":[[0]],"tenant_id":"other"}"#,
    ];
    let t0 = Instant::now();
    let mut rejected = 0usize;
    for _ in 0..1000 {
        for raw in &bad_json {
            assert!(serde_json::from_str::<Request>(raw).is_err());
            rejected += 1;
        }
        assert!(matches!(
            reg.infer(
                "t1",
                "../escape",
                &Request {
                    belief: vec![0.25; 4],
                    observation: None,
                    policies: vec![vec![0]],
                }
            ),
            Err(RegistryError::MalformedInput(_))
        ));
        rejected += 1;
    }
    let elapsed_us = t0.elapsed().as_micros();
    println!(
        "malformed_rejection: PASS ({rejected} rejected in {elapsed_us}us, {:.2}us/check)",
        elapsed_us as f64 / rejected as f64
    );
}

fn bench_memory_and_multitenant() {
    let reg = WorldRegistry::new();
    let base_model = build_cyclic_model(8, 2);
    let n_worlds = 1000usize;
    let t0 = Instant::now();
    for i in 0..n_worlds {
        let engine =
            WorldInference::with_version(base_model.clone(), 4.0, format!("rev-{i}")).unwrap();
        reg.register(
            format!("tenant-{}", i % 32),
            format!("world-{i}"),
            engine,
            AdmissibilityPolicy::allow_actions([0, 1]),
        )
        .unwrap();
    }
    let reg_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(reg.len(), n_worlds);

    // Concurrent multi-tenant stress & isolation verification.
    let t1 = Instant::now();
    let handles: Vec<_> = (0..8usize)
        .map(|tid| {
            let reg = reg.clone();
            thread::spawn(move || {
                let req = Request {
                    belief: vec![0.125; 8],
                    observation: Some((0, 0)),
                    policies: vec![vec![0], vec![1]],
                };
                for k in 0..250usize {
                    let idx = (tid * 32 + k) % n_worlds;
                    let owner = format!("tenant-{}", idx % 32);
                    let intruder = format!("tenant-{}", (idx + 1) % 32);
                    let wid = format!("world-{idx}");
                    let r = reg.infer(&owner, &wid, &req).unwrap();
                    assert_eq!(r.model_version, format!("rev-{idx}"));
                    assert!(!r.execution_authorized && r.recommendation.advisory_only);
                    assert_eq!(
                        reg.infer(&intruder, &wid, &req).unwrap_err(),
                        RegistryError::NotFound
                    );
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let stress_ms = t1.elapsed().as_secs_f64() * 1000.0;
    // Approximate heap payload per 8-state 2-action model: 2*(2*8*8 + 8) f64s = 2048 B.
    let est_model_bytes = 2 * 2 * 8 * 8 * size_of::<f64>() + 8 * size_of::<f64>();
    println!(
        "memory_and_multitenant: PASS worlds={n_worlds} reg_time={reg_ms:.2}ms concurrent_4000_ops={stress_ms:.2}ms sizeof(WorldInference)={}B sizeof(Response)={}B est_8state_model_heap={}B",
        size_of::<WorldInference>(),
        size_of::<Response>(),
        est_model_bytes,
    );
}

fn main() {
    println!("=== hanzo-worlds qualification benchmarks ===");
    bench_latency("2s-2a-4p-h2", 2, 2, 4, 2, 2000);
    bench_latency("16s-4a-16p-h4", 16, 4, 16, 4, 500);
    bench_latency("64s-4a-16p-h4", 64, 4, 16, 4, 100);
    bench_latency("128s-4a-8p-h2", 128, 4, 8, 2, 50);
    bench_numerical_stability();
    bench_malformed_rejection();
    bench_memory_and_multitenant();
}
