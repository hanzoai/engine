//! Verifies the declarative `router_policy.yaml` drop-in path: `load_policy`
//! parses a complete policy (per-task prefer lists, memory_fraction,
//! cost_ceiling) and `Policy::select` honors it with zero code changes.
//! Mirrors the file documented in zen-router/docs/config.md.

use std::collections::BTreeSet;

use hanzo_router::memory::MemSnapshot;
use hanzo_router::policy::Context;
use hanzo_router::registry::{Backend, ModelCard, Registry, Task};
use hanzo_router::{load_policy, Decision};

const GB: u64 = 1 << 30;

const POLICY_YAML: &str = r#"
prefer:
  cheap_chat:   [zen-nano-0.6b, zen5-flash, claude-haiku-4-5-20251001]
  general:      [zen-agent-4b, zen5, claude-haiku-4-5-20251001]
  code:         [zen-coder-24b, zen5-coder, gpt-5.1-codex-max, claude-sonnet-4-6]
  reasoning:    [zen-agent-4b, zen5-max, claude-opus-4-8, gpt-5.5]
memory_fraction: 0.85
cost_ceiling: 0.02
"#;

fn registry() -> Registry {
    use Task::*;
    Registry::new(vec![
        ModelCard {
            id: "zen-nano-0.6b".into(),
            backend: Backend::Local { est_bytes: GB },
            tasks: vec![CheapChat, General],
            max_context: 32_768,
            vision: false,
            cost_per_1k: 0.0,
        },
        ModelCard {
            id: "zen-coder-24b".into(),
            backend: Backend::Local { est_bytes: 24 * GB },
            tasks: vec![Code, Reasoning],
            max_context: 131_072,
            vision: false,
            cost_per_1k: 0.0,
        },
        ModelCard {
            id: "zen-agent-4b".into(),
            backend: Backend::Local { est_bytes: 5 * GB },
            tasks: vec![General, Reasoning],
            max_context: 65_536,
            vision: false,
            cost_per_1k: 0.0,
        },
        ModelCard {
            id: "claude-sonnet-4-6".into(),
            backend: Backend::Cloud {
                provider: "anthropic".into(),
            },
            tasks: vec![Code, General],
            max_context: 200_000,
            vision: true,
            cost_per_1k: 0.009,
        },
        ModelCard {
            id: "claude-opus-4-8".into(),
            backend: Backend::Cloud {
                provider: "anthropic".into(),
            },
            tasks: vec![Reasoning, Math],
            max_context: 200_000,
            vision: false,
            cost_per_1k: 0.045,
        },
        ModelCard {
            id: "gpt-5.5".into(),
            backend: Backend::Cloud {
                provider: "openai".into(),
            },
            tasks: vec![Reasoning, Code],
            max_context: 256_000,
            vision: false,
            cost_per_1k: 0.01,
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
            available_bytes: avail_gb * GB,
            total_bytes: 128 * GB,
            unified: true,
        },
        running,
        vision_required: false,
        min_context: 0,
    }
}

#[test]
fn full_policy_yaml_parses() {
    let p = load_policy(POLICY_YAML).expect("router_policy.yaml must parse");
    assert_eq!(p.memory_fraction, Some(0.85));
    assert_eq!(p.cost_ceiling, Some(0.02));
    assert_eq!(p.prefer["code"][0], "zen-coder-24b");
    assert_eq!(
        p.prefer["cheap_chat"],
        vec!["zen-nano-0.6b", "zen5-flash", "claude-haiku-4-5-20251001"]
    );
}

#[test]
fn empty_policy_is_valid() {
    let p = load_policy("{}").expect("empty policy must parse");
    assert!(p.prefer.is_empty() && p.cost_ceiling.is_none() && p.memory_fraction.is_none());
}

#[test]
fn zen_local_first_when_ram_available() {
    let p = load_policy(POLICY_YAML).unwrap();
    let reg = registry();
    // 40GB free: zen-coder-24b (24GB) fits -> served on-device, cost 0.
    let d = p.select(&ctx(Task::Code, &reg, 40, &BTreeSet::new()));
    assert_eq!(
        d,
        Decision::LoadLocal {
            model: "zen-coder-24b".into(),
            est_bytes: 24 * GB
        }
    );
}

#[test]
fn cost_ceiling_blocks_expensive_cloud() {
    let p = load_policy(POLICY_YAML).unwrap();
    let reg = registry();
    let running = BTreeSet::new();
    // Only 4GB free (budget 3.4GB): zen-agent-4b (5GB local) does not fit.
    // Preference order [zen-agent-4b(skip), zen5-max(absent), claude-opus-4-8(0.045 > ceiling,
    // filtered), gpt-5.5(0.01 <= 0.02)] -> the ceiling forces the affordable cloud model.
    let d = p.select(&ctx(Task::Reasoning, &reg, 4, &running));
    assert_eq!(
        d,
        Decision::Cloud {
            provider: "openai".into(),
            model: "gpt-5.5".into()
        }
    );
}
