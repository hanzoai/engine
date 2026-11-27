// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! # `hanzo-poai` — Proof of AI: CPU-Verifiable GPU Matrix Products (HIP-1350)
//!
//! Clean-room, zero-dependency Rust (`std`-only, Apache-2.0) execution profile and
//! Freivalds probabilistic verifier for packed low-bit quantized integer matrix products
//! (`poai.matmul.v1`: 32-element `i16`-scaled `Int8Q8_0`, `PackedInt4Q4_0`, and custom
//! `PackedTernaryI2S` block layouts).
//!
//! ## Scope Boundary (`hanzoai/engine#66` / `zoo-apps/papers#13`)
//!
//! > A matrix-product proof verifies only the declared integer matrix product `C = A * B`
//! > under profile `poai.matmul.v1`; attention, normalization (`RMSNorm`/`LayerNorm`),
//! > softmax, sampling, KV-cache/recurrent state updates, and overall LLM response
//! > truthfulness or scientific value are separate operations and are **NOT** covered
//! > by this proof.
//!
//! ## Soundness Summary
//!
//! - **Finite field**: Mersenne prime `GF(p)` with `p = 2^61 - 1 = 2_305_843_009_213_693_951`.
//! - **Exact signed bijection**: All integer outputs must satisfy `|C[i, j]| <= (p - 1)/2 = 2^60 - 1`,
//!   eliminating modular wraparound ambiguity between `i64` accumulation and `GF(p)`.
//! - **Per-round soundness error**: For uniform non-zero challenge `r in GF(p)^n \ {0}`,
//!   any non-zero error matrix `D = A*B - C (mod p)` satisfies
//!   `Pr[D*r = 0 | r != 0] <= (p^(n-1) - 1) / (p^n - 1) < 1/p ~= 4.3368e-19`.
//! - **Multi-round soundness**: For `k = 2` independent rounds (default verifier minimum),
//!   error `< 1/p^2 < 2^-121`; for `k = 4`, error `< 1/p^4 < 2^-243`.

pub mod error;
pub mod field;
pub mod freivalds;
pub mod profile;
pub mod quant;
pub mod transcript;

pub use error::PoaiError;
pub use field::{Fp, MAX_SAFE_SIGNED, MODULUS_P};
pub use freivalds::{
    prove_matrix_product, prove_matrix_product_with_cached_b_commit,
    verify_freivalds_algebraic_only, verify_matrix_product,
    verify_matrix_product_with_cached_b_commit,
    verify_matrix_product_with_cached_b_commit_and_policy, verify_matrix_product_with_mode,
    verify_matrix_product_with_policy, MatrixProofReceipt, VerificationPolicy,
};
pub use profile::{
    OperatorKind, VerificationProfile, DEFAULT_FREIVALDS_ROUNDS, MAX_FREIVALDS_ROUNDS,
    MIN_FREIVALDS_ROUNDS, PROFILE_VERSION_V1, PROOF_SCOPE_DISCLAIMER,
};
pub use quant::{
    prover_matmul, ActivationMatrix, OutputMatrix, PackedQuantFormat, PackedWeightMatrix,
    MAX_SAFE_DIM, QUANT_BLOCK_SIZE,
};
pub use transcript::{
    commit_activation_matrix, commit_output_matrix, commit_round_projection, commit_weight_matrix,
    derive_challenge_vector, derive_transcript_seed, digest_to_hex, os_random_seed, ChallengeMode,
    Sha256,
};
