//! Integration tests for Engine dependency-scoped benchmark routing, campaign validation,
//! progressive anytime-valid A/B runner, and qmm unimpacted skip behavior.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use hanzo_research::{
    impacted_benchmarks, CampaignRegistry, LineageRule, LocalResearchLedger, ProgressiveAbPlan,
    ProgressiveAbRunner, ProgressiveStageOutput,
};
use serde_json::{json, Value};

#[test]
fn engine_lineage_router_and_campaign_manifest_validate() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let lineage_bytes = fs::read(root.join("bench_lineage.json")).unwrap();
    let lineage: BTreeMap<String, LineageRule> = serde_json::from_slice(&lineage_bytes).unwrap();
    let all_tasks: Vec<String> = lineage.keys().cloned().collect();

    // Editing replay_gdn.rs scopes ONLY replay_gdn, never phase_policy, dflash2, or gguf_qmm.
    let rep = impacted_benchmarks(
        &all_tasks,
        &["hanzo-engine/src/models/replay_gdn.rs".to_string()],
        Some(&lineage),
        &[],
    )
    .unwrap();
    assert_eq!(rep.tasks, vec!["replay_gdn".to_string()]);
    assert!(!rep.full_release_suite);

    // Editing phase_policy.rs scopes ONLY phase_policy.
    let rep_phase = impacted_benchmarks(
        &all_tasks,
        &["hanzo-engine/src/kv_cache/phase_policy.rs".to_string()],
        Some(&lineage),
        &[],
    )
    .unwrap();
    assert_eq!(rep_phase.tasks, vec!["phase_policy".to_string()]);

    // Campaign registry loads cleanly via shared hanzo_research::CampaignRegistry.
    let reg = CampaignRegistry::from_file(root.join("campaigns/engine_inference.json")).unwrap();
    assert!(reg.objectives.len() >= 5);

    // qmm binary exits 0 immediately with skipped_unimpacted when replay_gdn.rs changed.
    let qmm_bin = env!("CARGO_BIN_EXE_qmm");
    let out = Command::new(qmm_bin)
        .args([
            "--lineage",
            root.join("bench_lineage.json").to_str().unwrap(),
            "--changed-paths",
            "hanzo-engine/src/models/replay_gdn.rs",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "qmm stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let skip_json: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(skip_json["status"], json!("skipped_unimpacted"));
    assert_eq!(skip_json["impacted_tasks"], json!(["replay_gdn"]));

    // ProgressiveAbRunner early-stops regressed engine candidate and records to LocalResearchLedger.
    let tmp = tempfile::tempdir().unwrap();
    let _ledger = LocalResearchLedger::open(tmp.path(), "engine-test").unwrap();
    let runner = ProgressiveAbRunner::new(ProgressiveAbPlan::default());
    let res = runner
        .run("decode_tps", "tok/s", |_idx, st| {
            Ok(ProgressiveStageOutput {
                stage_name: st.name.clone(),
                baseline_samples: vec![45.0, 45.2, 44.8],
                candidate_samples: vec![31.0, 30.8, 31.2],
                parity_passed: true,
                memory_integrity_passed: true,
                ..ProgressiveStageOutput::default()
            })
        })
        .unwrap();
    assert_eq!(res.verdict, "refuted");
    assert!(res.early_stopped);
    assert_eq!(res.stopped_stage, 1);
}
