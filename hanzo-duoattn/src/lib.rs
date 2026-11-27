// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-duoattn`
//!
//! Clean-room Rust implementation of **DuoAttention** (Xiao et al., ICLR 2025,
//! `arXiv:2410.10819`) for `hanzoai/engine#52` (HIP-1346):
//!
//! - **Per-Model Offline Head Calibration** ([`calibrate`]): learns continuous per-KV-head
//!   importance gates `alpha_{l,h} in [0, 1]` via projected Adam minimizing teacher-forced
//!   L2 attention-output divergence plus L1 sparsity over passkey & distant-reference sequences.
//! - **Fail-Closed Model-Bound Profiles** ([`profile`]): binds every calibrated retrieval vs.
//!   streaming head map to `(model_checkpoint_id, num_layers, num_attn_layers, num_q_heads,
//!   num_kv_heads, head_dim, sink_tokens, window_tokens, retrieval_ratio)` with a canonical
//!   FIPS 180-4 SHA-256 digest ([`sha256`]). Cross-model transfer and hybrid recurrent GDN
//!   layer eviction both fail closed.
//! - **Dual-Cache Multi-Head GQA Kernel** ([`cache`], [`attention`]): executes real `f32`
//!   causal scaled dot-product attention comparing `full_kv_baseline`,
//!   `uniform_lsa_streaming`, `random_head_selection_control`, and `duoattention_calibrated`.
//! - **HIP-1346 `4B/9B/27B` Hybrid Matrix & 5-Task Workloads** ([`model_matrix`], [`workload`]):
//!   benchmarks repository-level QA, distant-reference retrieval, needle-in-a-haystack,
//!   old-context code edits, and a 20-turn coding-agent loop alongside existing
//!   `hanzo-engine` LSA and three explicit negative regimes.
//!
//! Zero external crate dependencies (`std`-only), `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod attention;
pub mod cache;
pub mod calibrate;
pub mod error;
pub mod model_matrix;
pub mod profile;
pub mod sha256;
pub mod workload;

pub use attention::{compute_fidelity, DualAttentionModel, FidelityMetrics};
pub use cache::{CacheRegime, DualKvCache, HeadKvStorage, KvDtype};
pub use calibrate::{calibrate_model, CalibrationConfig, CalibrationReport, CalibrationSequence};
pub use error::DuoAttnError;
pub use model_matrix::{
    evaluate_agent_loop_scorecard, AgentLoopScorecard, HybridModelSpec, MatrixRegime,
};
pub use profile::{
    HeadCalibrationProfile, HeadRole, HeadScore, LayerStateKind, ModelIdentity,
    PROFILE_FORMAT_VERSION,
};
pub use workload::{
    build_calibration_dataset, build_five_workloads, evaluate_workload_regime,
    run_cross_model_negative_regime, run_short_context_negative_regime, CorpusEmbedder,
    CorpusProvenance, CrossModelNegativeResult, EvalEpisode, RepoCorpus,
    ShortContextNegativeResult, WorkloadRegimeResult,
};
