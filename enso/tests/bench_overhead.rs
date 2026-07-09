//! End-to-end decision-latency benchmark for the enso path: `featurize` +
//! bilinear `select`, over the trained profile table with a ~1k-char prompt.
//! No criterion in the workspace, so this is a timing-loop test: run with
//!   cargo test -p enso --release --test bench_overhead -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use enso::profile::PROFILE_DIM;
use enso::{
    fit_base, ingest, synth, Enso, Featurizer, HashFeaturizer, Learner, SelectCtx, Selector,
};
use hanzo_router::classify::Request;
use hanzo_router::registry::{Backend, Modality, ModelCard, Registry, Task};
use hanzo_router::{RoutePolicy, Slo, User};

const ITERS: usize = 10_000;
const WARMUP: usize = 1_000;
const SEED_TRAIN: u64 = 7;
const N_TRAIN: usize = 1500;
const GAMMA: f64 = 1.0;
const ALPHA: f64 = 0.5;

fn trained() -> (Enso, Registry) {
    let lineup = synth::lineup();
    let samples = synth::gen_eval_samples(&lineup, N_TRAIN, SEED_TRAIN);
    let table = ingest(&samples);
    let feat = HashFeaturizer::default();
    let train: Vec<(Vec<f64>, [f64; PROFILE_DIM], f64)> = samples
        .iter()
        .filter_map(|s| {
            let x = feat.featurize(&s.to_request());
            let p = table.get(&s.model, s.level, s.modality)?.features();
            Some((x, p, s.quality))
        })
        .collect();
    let base = fit_base(
        train
            .iter()
            .map(|(x, p, y)| (x.as_slice(), p.as_slice(), *y)),
        GAMMA,
    );
    let learner = Learner::new(base, GAMMA, ALPHA);
    let mut cards: Vec<ModelCard> = Vec::new();
    for tp in &lineup {
        if cards.iter().any(|c| c.id == tp.model) {
            continue;
        }
        cards.push(ModelCard {
            id: tp.model.to_string(),
            backend: Backend::Local {
                est_bytes: (tp.vram_gb as u64).max(1) << 30,
            },
            tasks: Task::ALL
                .into_iter()
                .filter(|t| {
                    lineup
                        .iter()
                        .any(|o| o.model == tp.model && o.quality[t.index()] > 0.0)
                })
                .collect(),
            max_context: lineup
                .iter()
                .filter(|o| o.model == tp.model)
                .map(|o| o.max_context)
                .max()
                .unwrap_or(0),
            vision: tp.modality == Modality::Vision,
            cost_per_1k: tp.cost,
        });
    }
    (Enso::new(table, learner), Registry::new(cards))
}

fn prompt_1k() -> String {
    let base = "Please review this Rust function for correctness and refactor it. \
        The compile error appears in the loop where we borrow the buffer mutably \
        while iterating. Explain step by step why the borrow checker rejects it, \
        then propose a fix that avoids the extra allocation. Here is the code: \
        ```rust fn process(items: &mut Vec<Item>) { for i in items.iter() {} } ``` \
        Also add a unit test that exercises the dirty path and note edge cases. ";
    let mut s = String::new();
    while s.len() < 1000 {
        s.push_str(base);
    }
    s.truncate(1024);
    s
}

fn stats(mut ns: Vec<u128>) -> (f64, f64, f64) {
    ns.sort_unstable();
    let mean = ns.iter().sum::<u128>() as f64 / ns.len() as f64;
    (
        mean / 1000.0,
        ns[ns.len() / 2] as f64 / 1000.0,
        ns[(ns.len() as f64 * 0.99) as usize] as f64 / 1000.0,
    )
}

fn time<F: FnMut()>(mut f: F) -> (f64, f64, f64) {
    for _ in 0..WARMUP {
        f();
    }
    let mut s = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        s.push(t.elapsed().as_nanos());
    }
    stats(s)
}

#[test]
fn bench_enso_path_end_to_end() {
    let (enso, reg) = trained();
    let slo = Slo::default();
    let user = User::new("alice");
    let req = Request {
        text: prompt_1k(),
        approx_tokens: 256,
        has_media: false,
        task_hint: None,
        modality_hint: None,
    };

    let feat = HashFeaturizer::default();
    let selector = Selector;

    // (1) featurize only.
    let (f_mean, f_p50, f_p99) = time(|| {
        black_box(feat.featurize(black_box(&req)));
    });

    // (2) bilinear select only (featurization + weight lookup hoisted out).
    let x = feat.featurize(&req);
    let task = feat.task_of(&req);
    let modality = req.target_modality();
    let w = enso.learner().effective_w(&user.id).to_vec();
    let table = enso.table();
    let (s_mean, s_p50, s_p99) = time(|| {
        let ctx = SelectCtx {
            x: &x,
            w: &w,
            table,
            want: modality,
            task,
            slo: &slo,
        };
        black_box(selector.select(black_box(&ctx), None));
    });

    // (3) full enso decision: featurize + effective_w + guard + bilinear select.
    let (e_mean, e_p50, e_p99) = time(|| {
        black_box(enso.route(black_box(&req), &user, &slo, &reg));
    });

    println!(
        "\n[enso learned path] over trained {}-row table | ~1k-char prompt",
        table.profiles.len()
    );
    println!("  featurize          iters={ITERS}  mean={f_mean:.3}us  p50={f_p50:.3}us  p99={f_p99:.3}us");
    println!("  bilinear select    iters={ITERS}  mean={s_mean:.3}us  p50={s_p50:.3}us  p99={s_p99:.3}us");
    println!(
        "  featurize+select   (sum of means) = {:.3}us",
        f_mean + s_mean
    );
    println!("  full route (e2e)   iters={ITERS}  mean={e_mean:.3}us  p50={e_p50:.3}us  p99={e_p99:.3}us");
    println!(
        "  route = {:?}\n",
        enso.route(&req, &user, &slo, &reg).model
    );
    assert!(
        e_mean < 100.0,
        "enso end-to-end mean {e_mean}us regressed above 100us"
    );
}
