//! End-to-end decision-latency benchmark for the rules path:
//! `Heuristic::classify` and `Policy::select` over a realistic ~20-card
//! registry with a ~1k-char prompt. No criterion in the workspace, so this is
//! a timing-loop test: run with
//! `cargo test -p hanzo-router --release --test bench_overhead -- --nocapture`

use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Instant;

use hanzo_router::classify::{Classifier, Heuristic, Request};
use hanzo_router::memory::MemSnapshot;
use hanzo_router::policy::{Context, Policy};
use hanzo_router::registry::{Backend, ModelCard, Registry, Task};

const ITERS: usize = 10_000;
const WARMUP: usize = 1_000;
const GB: u64 = 1 << 30;

fn local(id: &str, gb: u64, tasks: Vec<Task>, ctx: usize, vision: bool) -> ModelCard {
    ModelCard {
        id: id.into(),
        backend: Backend::Local { est_bytes: gb * GB },
        tasks,
        max_context: ctx,
        vision,
        cost_per_1k: 0.0,
    }
}
fn cloud(
    id: &str,
    provider: &str,
    tasks: Vec<Task>,
    ctx: usize,
    vision: bool,
    cost: f64,
) -> ModelCard {
    ModelCard {
        id: id.into(),
        backend: Backend::Cloud {
            provider: provider.into(),
        },
        tasks,
        max_context: ctx,
        vision,
        cost_per_1k: cost,
    }
}

/// A realistic 20-model pool: local zen tiers + cloud providers across every task.
fn registry() -> Registry {
    use Task::*;
    Registry::new(vec![
        local("zen-nano-0.6b", 1, vec![CheapChat, General], 32_768, false),
        local("zen-eco-1.7b", 2, vec![CheapChat, General], 32_768, false),
        local("zen-agent-4b", 5, vec![General, Reasoning], 65_536, false),
        local("zen-coder-8b", 9, vec![Code, General], 131_072, false),
        local("zen-coder-24b", 24, vec![Code, Reasoning], 131_072, false),
        local("zen-omni-7b", 20, vec![Vision, General], 32_768, true),
        local("zen-reasoner-14b", 15, vec![Reasoning, Math], 65_536, false),
        local(
            "zen-ultra-72b",
            80,
            vec![Reasoning, Math, LongContext, General],
            262_144,
            false,
        ),
        local("zen-creative-8b", 9, vec![Creative, General], 32_768, false),
        local("zen-math-7b", 8, vec![Math], 32_768, false),
        cloud(
            "claude-haiku",
            "anthropic",
            vec![CheapChat, General],
            200_000,
            false,
            0.8,
        ),
        cloud(
            "claude-sonnet",
            "anthropic",
            vec![Code, Reasoning, Creative, General],
            200_000,
            true,
            3.0,
        ),
        cloud(
            "claude-opus",
            "anthropic",
            vec![Reasoning, Math, LongContext],
            200_000,
            true,
            15.0,
        ),
        cloud(
            "gpt-5-mini",
            "openai",
            vec![CheapChat, General],
            128_000,
            false,
            0.6,
        ),
        cloud(
            "gpt-5",
            "openai",
            vec![Code, Reasoning, General],
            256_000,
            true,
            5.0,
        ),
        cloud("gpt-5-codex", "openai", vec![Code], 256_000, false, 4.0),
        cloud(
            "gemini-flash",
            "google",
            vec![CheapChat, General, LongContext],
            1_000_000,
            true,
            0.4,
        ),
        cloud(
            "gemini-pro",
            "google",
            vec![Reasoning, LongContext, Vision],
            2_000_000,
            true,
            4.0,
        ),
        cloud(
            "grok-4",
            "xai",
            vec![Reasoning, Code, General],
            256_000,
            false,
            4.0,
        ),
        cloud(
            "deepseek-v3",
            "deepseek",
            vec![Code, Math, Reasoning],
            128_000,
            false,
            0.3,
        ),
    ])
}

/// A realistic ~1k-char code-review prompt (the kind hanzo-node routes).
fn prompt_1k() -> String {
    let base = "Please review this Rust function for correctness and refactor it. \
        The compile error appears in the loop where we borrow the buffer mutably \
        while iterating. Explain step by step why the borrow checker rejects it, \
        then propose a fix that avoids the extra allocation. Here is the code: \
        ```rust fn process(items: &mut Vec<Item>) -> Result<(), Error> { \
        for item in items.iter() { if item.dirty { items.push(item.clone()); } } \
        Ok(()) } ``` Also add a unit test that exercises the dirty path and the \
        clean path, and note any edge cases around empty input or capacity growth. ";
    let mut s = String::new();
    while s.len() < 1000 {
        s.push_str(base);
    }
    s.truncate(1024);
    s
}

fn realistic_policy() -> Policy {
    let mut prefer = std::collections::BTreeMap::new();
    prefer.insert(
        "code".into(),
        vec![
            "zen-coder-24b".into(),
            "zen-coder-8b".into(),
            "claude-sonnet".into(),
        ],
    );
    prefer.insert(
        "cheap_chat".into(),
        vec!["zen-nano-0.6b".into(), "gemini-flash".into()],
    );
    prefer.insert(
        "reasoning".into(),
        vec![
            "zen-reasoner-14b".into(),
            "zen-ultra-72b".into(),
            "claude-opus".into(),
        ],
    );
    Policy {
        prefer,
        memory_fraction: None,
        cost_ceiling: Some(10.0),
    }
}

fn stats(mut ns: Vec<u128>) -> (f64, f64, f64) {
    ns.sort_unstable();
    let mean = ns.iter().sum::<u128>() as f64 / ns.len() as f64;
    let p99 = ns[(ns.len() as f64 * 0.99) as usize] as f64;
    let p50 = ns[ns.len() / 2] as f64;
    (mean / 1000.0, p50 / 1000.0, p99 / 1000.0)
}

#[test]
fn bench_rules_path_end_to_end() {
    let reg = registry();
    let policy = realistic_policy();
    let classifier = Heuristic;
    let running: BTreeSet<String> = ["zen-agent-4b".into()].into_iter().collect();
    let req = Request {
        text: prompt_1k(),
        approx_tokens: 256,
        has_media: false,
        task_hint: None,
        modality_hint: None,
    };
    let mem = MemSnapshot {
        available_bytes: 48 * GB,
        total_bytes: 128 * GB,
        unified: true,
    };

    let decide = |req: &Request| {
        let task = classifier.classify(black_box(req));
        let ctx = Context {
            task,
            registry: &reg,
            mem,
            running: &running,
            vision_required: req.has_media,
            min_context: req.approx_tokens,
        };
        policy.select(black_box(&ctx))
    };

    for _ in 0..WARMUP {
        black_box(decide(&req));
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        black_box(decide(&req));
        samples.push(t.elapsed().as_nanos());
    }
    // whole-loop wall time too (removes per-iter Instant overhead from the mean).
    let t = Instant::now();
    for _ in 0..ITERS {
        black_box(decide(&req));
    }
    let loop_us = t.elapsed().as_nanos() as f64 / 1000.0 / ITERS as f64;

    let (mean, p50, p99) = stats(samples);
    println!("\n[hanzo-router rules path] classify(Heuristic)+Policy::select | 20-card registry | ~1k-char prompt");
    println!("  iters={ITERS}  mean={mean:.3}us  p50={p50:.3}us  p99={p99:.3}us  loop_amortized={loop_us:.3}us");
    let decision = decide(&req);
    println!("  decision = {decision:?}\n");
    assert!(mean < 50.0, "rules-path mean {mean}us regressed above 50us");
}
