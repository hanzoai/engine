//! # hanzo-router
//!
//! Memory- and engine-aware model router — the native companion to
//! [`hanzo-ml`](https://github.com/hanzoai/ml) and
//! [`hanzo-engine`](https://github.com/hanzoai/engine) that **hanzo-node** uses to
//! route each request across a swappable pool of **local + cloud** models.
//!
//! It is **pure logic over value inputs** (Rich Hickey style): hanzo-node fills a
//! [`MemSnapshot`] from the engine's `MemoryUsage`/`DeepSeekV2…DeviceMemory`
//! (`available`/`is_unified` — the same signal the auto device-mapper uses),
//! supplies the [`Registry`] of available models and the set of currently-loaded
//! ones, and the router decides — with **no hardware access of its own**, so it is
//! deterministic and unit-testable.
//!
//! The decision ([`Policy::select`]) prefers, in order:
//! 1. **Reuse** a model already loaded in a connectable engine (zero load cost),
//! 2. **Load** a local model whose resident footprint *fits* available memory
//!    (the "run it locally if the RAM is there" path),
//! 3. **Cloud** — fall through to the cheapest usable provider (OpenAI/Anthropic/…).
//!
//! ```
//! use hanzo_router::{classify::{Heuristic, Request, Classifier}, memory::MemSnapshot,
//!     policy::{Policy, Context, Decision}, registry::{Registry, ModelCard, Backend, Task}};
//! use std::collections::BTreeSet;
//!
//! let registry = Registry::new(vec![
//!     ModelCard { id: "deepseek-v4-flash".into(), backend: Backend::Local { est_bytes: 93 << 30 },
//!         tasks: vec![Task::Code, Task::Reasoning, Task::General], max_context: 1_048_576, vision: false, cost_per_1k: 0.0 },
//!     ModelCard { id: "claude-sonnet-4-5".into(), backend: Backend::Cloud { provider: "anthropic".into() },
//!         tasks: vec![Task::Code, Task::General], max_context: 200_000, vision: true, cost_per_1k: 3.0 },
//! ]);
//! let policy = Policy::default();
//! let task = Heuristic.classify(&Request { text: "fix this ```rust``` bug".into(), ..Default::default() });
//! // 121GB unified box, ~115GB free -> V4 (93GB) fits and is preferred for code.
//! let mem = MemSnapshot { available_bytes: 115 << 30, total_bytes: 121 << 30, unified: true };
//! let ctx = Context { task, registry: &registry, mem, running: &BTreeSet::new(),
//!     vision_required: false, min_context: 0 };
//! assert!(matches!(policy.select(&ctx), Decision::LoadLocal { .. }));
//! ```

pub mod classify;
pub mod memory;
pub mod policy;
pub mod registry;
pub mod route;

pub use classify::{Classifier, Heuristic, Request};
pub use memory::MemSnapshot;
pub use policy::{Context, Decision, Policy};
pub use registry::{Backend, Level, Modality, ModelCard, Registry, Task};
pub use route::{Route, RoutePolicy, Slo, User, REFUSED_MODEL};

/// Load a [`Policy`] from a YAML string (the declarative policy file).
pub fn load_policy(yaml: &str) -> Result<Policy, serde_yaml::Error> {
    serde_yaml::from_str(yaml)
}

/// One-shot convenience: classify `req`, then select. `running` is the set of
/// model ids currently loaded in connectable engines.
pub fn route(
    req: &Request,
    classifier: &dyn Classifier,
    policy: &Policy,
    registry: &Registry,
    mem: MemSnapshot,
    running: &std::collections::BTreeSet<String>,
) -> Decision {
    let task = classifier.classify(req);
    let ctx = Context {
        task,
        registry,
        mem,
        running,
        vision_required: req.has_media,
        min_context: req.approx_tokens,
    };
    policy.select(&ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn pool() -> Registry {
        Registry::new(vec![
            ModelCard {
                id: "deepseek-v4-flash".into(),
                backend: Backend::Local {
                    est_bytes: 93 << 30,
                },
                tasks: vec![Task::Code, Task::Reasoning, Task::General],
                max_context: 1_048_576,
                vision: false,
                cost_per_1k: 0.0,
            },
            ModelCard {
                id: "zen-omni".into(),
                backend: Backend::Local {
                    est_bytes: 18 << 30,
                },
                tasks: vec![Task::Vision, Task::General],
                max_context: 32_768,
                vision: true,
                cost_per_1k: 0.0,
            },
            ModelCard {
                id: "claude-sonnet-4-5".into(),
                backend: Backend::Cloud {
                    provider: "anthropic".into(),
                },
                tasks: vec![Task::Code, Task::General],
                max_context: 200_000,
                vision: true,
                cost_per_1k: 3.0,
            },
            ModelCard {
                id: "gpt-4o-mini".into(),
                backend: Backend::Cloud {
                    provider: "openai".into(),
                },
                tasks: vec![Task::CheapChat, Task::General],
                max_context: 128_000,
                vision: true,
                cost_per_1k: 0.5,
            },
        ])
    }

    fn ctx<'a>(
        task: Task,
        reg: &'a Registry,
        avail_gb: u64,
        running: &'a BTreeSet<String>,
    ) -> Context<'a> {
        Context {
            task,
            registry: reg,
            mem: MemSnapshot {
                available_bytes: avail_gb << 30,
                total_bytes: 121 << 30,
                unified: true,
            },
            running,
            vision_required: false,
            min_context: 0,
        }
    }

    #[test]
    fn loads_local_v4_for_code_when_ram_available() {
        let reg = pool();
        let d = Policy::default().select(&ctx(Task::Code, &reg, 115, &BTreeSet::new()));
        assert_eq!(
            d,
            Decision::LoadLocal {
                model: "deepseek-v4-flash".into(),
                est_bytes: 93 << 30
            }
        );
    }

    #[test]
    fn falls_to_cloud_when_v4_does_not_fit() {
        let reg = pool();
        // Only 60GB free -> V4 (93GB) can't load -> cheapest usable cloud for code = claude.
        let d = Policy::default().select(&ctx(Task::Code, &reg, 60, &BTreeSet::new()));
        assert_eq!(
            d,
            Decision::Cloud {
                provider: "anthropic".into(),
                model: "claude-sonnet-4-5".into()
            }
        );
    }

    #[test]
    fn reuses_running_model_even_if_smaller_one_fits() {
        let reg = pool();
        let running: BTreeSet<String> = ["deepseek-v4-flash".to_string()].into_iter().collect();
        // V4 already loaded -> reuse it (zero load cost) rather than reload.
        let d = Policy::default().select(&ctx(Task::Code, &reg, 60, &running));
        assert_eq!(
            d,
            Decision::Reuse {
                model: "deepseek-v4-flash".into()
            }
        );
    }

    #[test]
    fn vision_requires_a_vision_model() {
        let reg = pool();
        let running = BTreeSet::new();
        let mut c = ctx(Task::Vision, &reg, 115, &running);
        c.vision_required = true;
        // V4 isn't vision; zen-omni (18GB, vision) fits and is chosen.
        let d = Policy::default().select(&c);
        assert_eq!(
            d,
            Decision::LoadLocal {
                model: "zen-omni".into(),
                est_bytes: 18 << 30
            }
        );
    }

    #[test]
    fn cheap_chat_prefers_cheapest_cloud_via_policy() {
        let reg = pool();
        // Policy steers cheap_chat to the cheap cloud model explicitly.
        let policy: Policy = load_policy("prefer:\n  cheap_chat: [gpt-4o-mini]\n").unwrap();
        let running = BTreeSet::new();
        let d = policy.select(&ctx(Task::CheapChat, &reg, 115, &running));
        assert_eq!(
            d,
            Decision::Cloud {
                provider: "openai".into(),
                model: "gpt-4o-mini".into()
            }
        );
    }

    #[test]
    fn no_fit_when_nothing_usable() {
        let reg = Registry::new(vec![ModelCard {
            id: "huge".into(),
            backend: Backend::Local {
                est_bytes: 500 << 30,
            },
            tasks: vec![Task::General],
            max_context: 0,
            vision: false,
            cost_per_1k: 0.0,
        }]);
        let running = BTreeSet::new();
        assert_eq!(
            Policy::default().select(&ctx(Task::Code, &reg, 60, &running)),
            Decision::NoFit
        );
    }

    #[test]
    fn classifier_routes_by_task() {
        assert_eq!(
            Heuristic.classify(&Request {
                text: "```py```".into(),
                ..Default::default()
            }),
            Task::Code
        );
        assert_eq!(
            Heuristic.classify(&Request {
                text: "prove sqrt(2) irrational".into(),
                ..Default::default()
            }),
            Task::Math
        );
        assert_eq!(
            Heuristic.classify(&Request {
                has_media: true,
                ..Default::default()
            }),
            Task::Vision
        );
        assert_eq!(
            Heuristic.classify(&Request {
                approx_tokens: 100_000,
                ..Default::default()
            }),
            Task::LongContext
        );
    }
}
