// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Versioned, model-bound DuoAttention head calibration profiles.
//!
//! Each [`HeadCalibrationProfile`] binds learned per-KV-head importance gates
//! (`alpha_{l,h} in [0, 1]`) and binary [`HeadRole`] assignments (`Retrieval` vs
//! `Streaming`) to a specific model checkpoint, architecture geometry, and
//! streaming window configuration via a canonical SHA-256 checksum.
//!
//! Loading or attaching a profile against a different checkpoint or mismatched
//! architecture fails closed with [`DuoAttnError::ModelMismatch`] or
//! [`DuoAttnError::ArchitectureMismatch`], enforcing the core invariant of
//! `hanzoai/engine#52`: **head calibration is model-specific and is NOT a safe
//! universal transformation**.

use std::fmt::Write as _;

use crate::error::DuoAttnError;
use crate::sha256::Sha256;

/// Current canonical binary/JSON schema version for [`HeadCalibrationProfile`].
pub const PROFILE_FORMAT_VERSION: u32 = 1;

/// Layer state kind in hybrid Hanzo architectures (matching `hanzo-paged-attn::kivi`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerStateKind {
    /// Standard paged key-value attention layer eligible for DuoAttention head partitioning.
    AttentionKv,
    /// Recurrent Gated DeltaNet (GDN) state layer — MUST remain exempt from KV eviction.
    RecurrentGdn,
}

/// Assigned KV-cache role for a single `(attn_layer, kv_head)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadRole {
    /// Retrieval head: retains the full `O(L)` KV history across the entire context.
    Retrieval,
    /// Streaming head: retains only `sink_tokens + window_tokens` (`O(1)` constant memory).
    Streaming,
}

impl HeadRole {
    /// Canonical lowercase string tag.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Retrieval => "retrieval",
            Self::Streaming => "streaming",
        }
    }
}

/// Immutable architectural and checkpoint identity of a target model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelIdentity {
    /// Canonical checkpoint identifier (e.g., `"hanzo-zen6-9b-hybrid-v1@sha256:..."`).
    pub model_checkpoint_id: String,
    /// Total transformer/hybrid layers (`num_attn_layers + num_gdn_layers`).
    pub num_layers: usize,
    /// Number of paged attention layers that maintain KV caches.
    pub num_attn_layers: usize,
    /// Number of recurrent GDN layers (exempt from DuoAttention).
    pub num_gdn_layers: usize,
    /// Query heads per attention layer.
    pub num_q_heads: usize,
    /// Key-Value heads per attention layer (`num_q_heads % num_kv_heads == 0`).
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Ordered per-layer state kinds of length `num_layers`.
    pub layer_kinds: Vec<LayerStateKind>,
}

impl ModelIdentity {
    /// Creates a pure-attention model identity (zero recurrent GDN layers).
    pub fn pure_attention(
        model_checkpoint_id: impl Into<String>,
        num_layers: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, DuoAttnError> {
        let layer_kinds = vec![LayerStateKind::AttentionKv; num_layers];
        let id = Self {
            model_checkpoint_id: model_checkpoint_id.into(),
            num_layers,
            num_attn_layers: num_layers,
            num_gdn_layers: 0,
            num_q_heads,
            num_kv_heads,
            head_dim,
            layer_kinds,
        };
        id.validate()?;
        Ok(id)
    }

    /// Creates a hybrid Attention + GDN model identity matching Hanzo's `4B/9B/27B` layouts.
    ///
    /// Interleaves attention layers evenly across the stack (e.g., 1 attention layer per
    /// 4 layers in `9+27`, `10+30`, `16+48` hybrid models).
    pub fn hybrid(
        model_checkpoint_id: impl Into<String>,
        num_attn_layers: usize,
        num_gdn_layers: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, DuoAttnError> {
        let num_layers = num_attn_layers + num_gdn_layers;
        if num_attn_layers == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "num_attn_layers must be >= 1".to_string(),
            ));
        }
        let mut layer_kinds = Vec::with_capacity(num_layers);
        let mut placed_attn = 0usize;
        for i in 0..num_layers {
            // Place attention layers evenly so the final layer is always an attention layer.
            let target_attn = ((i + 1) * num_attn_layers) / num_layers;
            if target_attn > placed_attn {
                layer_kinds.push(LayerStateKind::AttentionKv);
                placed_attn += 1;
            } else {
                layer_kinds.push(LayerStateKind::RecurrentGdn);
            }
        }
        debug_assert_eq!(placed_attn, num_attn_layers);
        let id = Self {
            model_checkpoint_id: model_checkpoint_id.into(),
            num_layers,
            num_attn_layers,
            num_gdn_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            layer_kinds,
        };
        id.validate()?;
        Ok(id)
    }

    /// Validates all structural invariants of the model identity.
    pub fn validate(&self) -> Result<(), DuoAttnError> {
        if self.model_checkpoint_id.trim().is_empty() {
            return Err(DuoAttnError::InvalidConfig(
                "model_checkpoint_id must be non-empty".to_string(),
            ));
        }
        if self.num_attn_layers == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "num_attn_layers must be >= 1".to_string(),
            ));
        }
        if self.num_attn_layers + self.num_gdn_layers != self.num_layers {
            return Err(DuoAttnError::InvalidConfig(format!(
                "num_attn_layers ({}) + num_gdn_layers ({}) != num_layers ({})",
                self.num_attn_layers, self.num_gdn_layers, self.num_layers
            )));
        }
        if self.layer_kinds.len() != self.num_layers {
            return Err(DuoAttnError::InvalidConfig(format!(
                "layer_kinds length ({}) != num_layers ({})",
                self.layer_kinds.len(),
                self.num_layers
            )));
        }
        let attn_count = self
            .layer_kinds
            .iter()
            .filter(|&&k| k == LayerStateKind::AttentionKv)
            .count();
        if attn_count != self.num_attn_layers {
            return Err(DuoAttnError::InvalidConfig(format!(
                "layer_kinds contains {attn_count} AttentionKv entries, expected {}",
                self.num_attn_layers
            )));
        }
        if self.num_kv_heads == 0 || self.num_q_heads == 0 || self.head_dim == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "num_q_heads, num_kv_heads, and head_dim must all be >= 1".to_string(),
            ));
        }
        if !self.num_q_heads.is_multiple_of(self.num_kv_heads) {
            return Err(DuoAttnError::InvalidConfig(format!(
                "num_q_heads ({}) must be a multiple of num_kv_heads ({})",
                self.num_q_heads, self.num_kv_heads
            )));
        }
        Ok(())
    }

    /// Number of query heads sharing each KV head (GQA replication factor).
    #[must_use]
    pub const fn gqa_group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    /// Total number of `(attn_layer, kv_head)` slots across all attention layers.
    #[must_use]
    pub const fn total_kv_heads(&self) -> usize {
        self.num_attn_layers * self.num_kv_heads
    }

    /// Verifies that `global_layer_idx` is an attention layer and returns its
    /// zero-based attention-layer index, failing closed on recurrent GDN layers.
    pub fn check_attention_layer(&self, global_layer_idx: usize) -> Result<usize, DuoAttnError> {
        let kind = self.layer_kinds.get(global_layer_idx).ok_or_else(|| {
            DuoAttnError::InvalidConfig(format!(
                "global_layer_idx {global_layer_idx} out of range (num_layers = {})",
                self.num_layers
            ))
        })?;
        if *kind == LayerStateKind::RecurrentGdn {
            return Err(DuoAttnError::RecurrentLayerExempt {
                layer_idx: global_layer_idx,
            });
        }
        let attn_idx = self.layer_kinds[..global_layer_idx]
            .iter()
            .filter(|&&k| k == LayerStateKind::AttentionKv)
            .count();
        Ok(attn_idx)
    }
}

/// Learned importance gate and assigned role for one `(attn_layer, kv_head)` pair.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadScore {
    /// Zero-based index among attention layers (`0..num_attn_layers`).
    pub attn_layer_idx: usize,
    /// Zero-based KV head index within the attention layer (`0..num_kv_heads`).
    pub kv_head_idx: usize,
    /// Learned DuoAttention gate `alpha_{l,h} in [0.0, 1.0]`.
    pub alpha: f32,
    /// Assigned cache role (`Retrieval` if above threshold, `Streaming` otherwise).
    pub role: HeadRole,
}

/// Versioned, SHA-256 checksummed, model-bound DuoAttention calibration artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadCalibrationProfile {
    /// Schema format version ([`PROFILE_FORMAT_VERSION`]).
    pub version: u32,
    /// Exact model checkpoint ID for which this calibration was produced.
    pub model_checkpoint_id: String,
    /// Total layers in the calibrated model.
    pub num_layers: usize,
    /// Number of attention KV layers in the calibrated model.
    pub num_attn_layers: usize,
    /// Number of query heads per attention layer.
    pub num_q_heads: usize,
    /// Number of KV heads per attention layer.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of initial attention sink tokens retained by streaming heads.
    pub sink_tokens: usize,
    /// Number of recent sliding-window tokens retained by streaming heads.
    pub window_tokens: usize,
    /// Target fraction of attention KV heads assigned to [`HeadRole::Retrieval`] (`0.0..=1.0`).
    pub retrieval_ratio: f32,
    /// Learned alpha cutoff separating retrieval heads from streaming heads.
    pub threshold: f32,
    /// Deterministic calibration seed.
    pub calibration_seed: u64,
    /// Number of projected gradient steps executed during offline calibration.
    pub calibration_steps: usize,
    /// L1 sparsity regularization weight `lambda`.
    pub l1_lambda: f32,
    /// Initial calibration objective value before optimization.
    pub calibration_loss_initial: f64,
    /// Final calibration objective value after optimization.
    pub calibration_loss_final: f64,
    /// Ordered per-KV-head scores (`len == num_attn_layers * num_kv_heads`).
    pub heads: Vec<HeadScore>,
    /// Canonical lowercase hex SHA-256 digest binding all fields above.
    pub checksum_sha256: String,
}

impl HeadCalibrationProfile {
    /// Constructs a new checksummed calibration profile from learned per-head `alpha`s.
    ///
    /// Classifies the top `ceil(retrieval_ratio * total_kv_heads)` heads (or `0` when
    /// `retrieval_ratio == 0.0`) by descending `alpha` (with deterministic index tie-breaking)
    /// as [`HeadRole::Retrieval`], and the rest as [`HeadRole::Streaming`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_alphas(
        model: &ModelIdentity,
        sink_tokens: usize,
        window_tokens: usize,
        retrieval_ratio: f32,
        calibration_seed: u64,
        calibration_steps: usize,
        l1_lambda: f32,
        calibration_loss_initial: f64,
        calibration_loss_final: f64,
        alphas: &[f32],
    ) -> Result<Self, DuoAttnError> {
        model.validate()?;
        if sink_tokens == 0 || window_tokens == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "sink_tokens and window_tokens must both be >= 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&retrieval_ratio) || retrieval_ratio.is_nan() {
            return Err(DuoAttnError::InvalidConfig(format!(
                "retrieval_ratio must be in [0.0, 1.0], got {retrieval_ratio}"
            )));
        }
        let total = model.total_kv_heads();
        if alphas.len() != total {
            return Err(DuoAttnError::DimensionMismatch {
                context: "HeadCalibrationProfile::from_alphas",
                expected: total,
                actual: alphas.len(),
            });
        }

        let (heads, threshold) = classify_heads(
            model.num_attn_layers,
            model.num_kv_heads,
            alphas,
            retrieval_ratio,
        );

        let mut profile = Self {
            version: PROFILE_FORMAT_VERSION,
            model_checkpoint_id: model.model_checkpoint_id.clone(),
            num_layers: model.num_layers,
            num_attn_layers: model.num_attn_layers,
            num_q_heads: model.num_q_heads,
            num_kv_heads: model.num_kv_heads,
            head_dim: model.head_dim,
            sink_tokens,
            window_tokens,
            retrieval_ratio,
            threshold,
            calibration_seed,
            calibration_steps,
            l1_lambda,
            calibration_loss_initial,
            calibration_loss_final,
            heads,
            checksum_sha256: String::new(),
        };
        profile.checksum_sha256 = profile.compute_checksum();
        Ok(profile)
    }

    /// Computes the canonical domain-separated SHA-256 checksum over every profile field.
    #[must_use]
    pub fn compute_checksum(&self) -> String {
        let mut h = Sha256::new();
        h.update_len_prefixed(b"hanzo-duoattn/profile/v1");
        h.update_u32(self.version);
        h.update_len_prefixed(self.model_checkpoint_id.as_bytes());
        h.update_u64(self.num_layers as u64);
        h.update_u64(self.num_attn_layers as u64);
        h.update_u64(self.num_q_heads as u64);
        h.update_u64(self.num_kv_heads as u64);
        h.update_u64(self.head_dim as u64);
        h.update_u64(self.sink_tokens as u64);
        h.update_u64(self.window_tokens as u64);
        h.update_f32(self.retrieval_ratio);
        h.update_f32(self.threshold);
        h.update_u64(self.calibration_seed);
        h.update_u64(self.calibration_steps as u64);
        h.update_f32(self.l1_lambda);
        h.update_f64(self.calibration_loss_initial);
        h.update_f64(self.calibration_loss_final);
        h.update_u64(self.heads.len() as u64);
        for head in &self.heads {
            h.update_u64(head.attn_layer_idx as u64);
            h.update_u64(head.kv_head_idx as u64);
            h.update_f32(head.alpha);
            h.update(&[match head.role {
                HeadRole::Retrieval => 1u8,
                HeadRole::Streaming => 0u8,
            }]);
        }
        h.finalize_hex()
    }

    /// Verifies schema version, internal dimensions, and SHA-256 integrity checksum.
    pub fn verify_integrity(&self) -> Result<(), DuoAttnError> {
        if self.version != PROFILE_FORMAT_VERSION {
            return Err(DuoAttnError::UnsupportedProfileVersion {
                version: self.version,
            });
        }
        let expected_heads = self.num_attn_layers.saturating_mul(self.num_kv_heads);
        if self.heads.len() != expected_heads {
            return Err(DuoAttnError::DimensionMismatch {
                context: "HeadCalibrationProfile::verify_integrity(heads)",
                expected: expected_heads,
                actual: self.heads.len(),
            });
        }
        let computed = self.compute_checksum();
        if computed != self.checksum_sha256 {
            return Err(DuoAttnError::ProfileChecksumMismatch {
                recorded: self.checksum_sha256.clone(),
                computed,
            });
        }
        Ok(())
    }

    /// Fail-closed verification that this profile is authentic AND bound to `target`.
    ///
    /// Rejects any cross-model reuse (`ModelMismatch`), architectural disagreement
    /// (`ArchitectureMismatch`), or corrupted checksum (`ProfileChecksumMismatch`).
    pub fn verify_for_model(&self, target: &ModelIdentity) -> Result<(), DuoAttnError> {
        target.validate()?;
        self.verify_integrity()?;

        if self.model_checkpoint_id != target.model_checkpoint_id {
            return Err(DuoAttnError::ModelMismatch {
                profile_model: self.model_checkpoint_id.clone(),
                target_model: target.model_checkpoint_id.clone(),
            });
        }
        if self.num_layers != target.num_layers {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "num_layers",
                profile_value: self.num_layers,
                target_value: target.num_layers,
            });
        }
        if self.num_attn_layers != target.num_attn_layers {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "num_attn_layers",
                profile_value: self.num_attn_layers,
                target_value: target.num_attn_layers,
            });
        }
        if self.num_q_heads != target.num_q_heads {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "num_q_heads",
                profile_value: self.num_q_heads,
                target_value: target.num_q_heads,
            });
        }
        if self.num_kv_heads != target.num_kv_heads {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "num_kv_heads",
                profile_value: self.num_kv_heads,
                target_value: target.num_kv_heads,
            });
        }
        if self.head_dim != target.head_dim {
            return Err(DuoAttnError::ArchitectureMismatch {
                field: "head_dim",
                profile_value: self.head_dim,
                target_value: target.head_dim,
            });
        }
        Ok(())
    }

    /// Re-thresholds the SAME calibrated `alpha` scores at a different retrieval-head budget
    /// `new_retrieval_ratio` without re-running offline optimization, re-signing the SHA-256 digest.
    pub fn rethreshold(&self, new_retrieval_ratio: f32) -> Result<Self, DuoAttnError> {
        self.verify_integrity()?;
        if !(0.0..=1.0).contains(&new_retrieval_ratio) || new_retrieval_ratio.is_nan() {
            return Err(DuoAttnError::InvalidConfig(format!(
                "new_retrieval_ratio must be in [0.0, 1.0], got {new_retrieval_ratio}"
            )));
        }
        let alphas: Vec<f32> = self.heads.iter().map(|h| h.alpha).collect();
        let (heads, threshold) = classify_heads(
            self.num_attn_layers,
            self.num_kv_heads,
            &alphas,
            new_retrieval_ratio,
        );
        let mut cloned = self.clone();
        cloned.retrieval_ratio = new_retrieval_ratio;
        cloned.threshold = threshold;
        cloned.heads = heads;
        cloned.checksum_sha256 = cloned.compute_checksum();
        Ok(cloned)
    }

    /// Returns the assigned [`HeadRole`] for `(attn_layer_idx, kv_head_idx)`.
    pub fn role_for(
        &self,
        attn_layer_idx: usize,
        kv_head_idx: usize,
    ) -> Result<HeadRole, DuoAttnError> {
        if attn_layer_idx >= self.num_attn_layers || kv_head_idx >= self.num_kv_heads {
            return Err(DuoAttnError::InvalidConfig(format!(
                "head coordinate ({attn_layer_idx}, {kv_head_idx}) out of bounds ({}x{})",
                self.num_attn_layers, self.num_kv_heads
            )));
        }
        let idx = attn_layer_idx * self.num_kv_heads + kv_head_idx;
        Ok(self.heads[idx].role)
    }

    /// Number of KV heads classified as [`HeadRole::Retrieval`].
    #[must_use]
    pub fn retrieval_head_count(&self) -> usize {
        self.heads
            .iter()
            .filter(|h| h.role == HeadRole::Retrieval)
            .count()
    }

    /// Number of KV heads classified as [`HeadRole::Streaming`].
    #[must_use]
    pub fn streaming_head_count(&self) -> usize {
        self.heads.len() - self.retrieval_head_count()
    }

    /// Actual realized retrieval head fraction (`retrieval_head_count / total_kv_heads`).
    #[must_use]
    pub fn realized_retrieval_ratio(&self) -> f64 {
        if self.heads.is_empty() {
            0.0
        } else {
            self.retrieval_head_count() as f64 / self.heads.len() as f64
        }
    }

    /// Serializes this profile to deterministic pretty JSON without external dependencies.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(1024 + self.heads.len() * 64);
        out.push_str("{\n");
        let _ = writeln!(out, "  \"version\": {},", self.version);
        let _ = writeln!(
            out,
            "  \"model_checkpoint_id\": \"{}\",",
            escape_json_str(&self.model_checkpoint_id)
        );
        let _ = writeln!(out, "  \"num_layers\": {},", self.num_layers);
        let _ = writeln!(out, "  \"num_attn_layers\": {},", self.num_attn_layers);
        let _ = writeln!(out, "  \"num_q_heads\": {},", self.num_q_heads);
        let _ = writeln!(out, "  \"num_kv_heads\": {},", self.num_kv_heads);
        let _ = writeln!(out, "  \"head_dim\": {},", self.head_dim);
        let _ = writeln!(out, "  \"sink_tokens\": {},", self.sink_tokens);
        let _ = writeln!(out, "  \"window_tokens\": {},", self.window_tokens);
        let _ = writeln!(out, "  \"retrieval_ratio\": {},", self.retrieval_ratio);
        let _ = writeln!(out, "  \"threshold\": {},", self.threshold);
        let _ = writeln!(out, "  \"calibration_seed\": {},", self.calibration_seed);
        let _ = writeln!(out, "  \"calibration_steps\": {},", self.calibration_steps);
        let _ = writeln!(out, "  \"l1_lambda\": {},", self.l1_lambda);
        let _ = writeln!(
            out,
            "  \"calibration_loss_initial\": {},",
            self.calibration_loss_initial
        );
        let _ = writeln!(
            out,
            "  \"calibration_loss_final\": {},",
            self.calibration_loss_final
        );
        let _ = writeln!(
            out,
            "  \"checksum_sha256\": \"{}\",",
            escape_json_str(&self.checksum_sha256)
        );
        out.push_str("  \"heads\": [\n");
        for (i, head) in self.heads.iter().enumerate() {
            let comma = if i + 1 < self.heads.len() { "," } else { "" };
            let _ = writeln!(
                out,
                "    {{\"attn_layer_idx\": {}, \"kv_head_idx\": {}, \"alpha\": {}, \"role\": \"{}\"}}{comma}",
                head.attn_layer_idx,
                head.kv_head_idx,
                head.alpha,
                head.role.as_str()
            );
        }
        out.push_str("  ]\n}\n");
        out
    }

    /// Parses a serialized profile JSON and verifies its SHA-256 integrity checksum.
    pub fn from_json(raw: &str) -> Result<Self, DuoAttnError> {
        let version = extract_json_u64(raw, "version")? as u32;
        let model_checkpoint_id = extract_json_string(raw, "model_checkpoint_id")?;
        let num_layers = extract_json_u64(raw, "num_layers")? as usize;
        let num_attn_layers = extract_json_u64(raw, "num_attn_layers")? as usize;
        let num_q_heads = extract_json_u64(raw, "num_q_heads")? as usize;
        let num_kv_heads = extract_json_u64(raw, "num_kv_heads")? as usize;
        let head_dim = extract_json_u64(raw, "head_dim")? as usize;
        let sink_tokens = extract_json_u64(raw, "sink_tokens")? as usize;
        let window_tokens = extract_json_u64(raw, "window_tokens")? as usize;
        let retrieval_ratio = extract_json_f64(raw, "retrieval_ratio")? as f32;
        let threshold = extract_json_f64(raw, "threshold")? as f32;
        let calibration_seed = extract_json_u64(raw, "calibration_seed")?;
        let calibration_steps = extract_json_u64(raw, "calibration_steps")? as usize;
        let l1_lambda = extract_json_f64(raw, "l1_lambda")? as f32;
        let calibration_loss_initial = extract_json_f64(raw, "calibration_loss_initial")?;
        let calibration_loss_final = extract_json_f64(raw, "calibration_loss_final")?;
        let checksum_sha256 = extract_json_string(raw, "checksum_sha256")?;

        let heads_start = raw
            .find("\"heads\"")
            .ok_or_else(|| DuoAttnError::SerializationError("missing \"heads\" array".into()))?;
        let arr_open = raw[heads_start..]
            .find('[')
            .ok_or_else(|| DuoAttnError::SerializationError("missing '[' for heads".into()))?
            + heads_start;
        let arr_close = raw[arr_open..]
            .find(']')
            .ok_or_else(|| DuoAttnError::SerializationError("missing ']' for heads".into()))?
            + arr_open;
        let heads_slice = &raw[arr_open + 1..arr_close];

        let mut heads = Vec::new();
        let mut cursor = 0usize;
        while let Some(obj_start_rel) = heads_slice[cursor..].find('{') {
            let obj_start = cursor + obj_start_rel;
            let obj_end_rel = heads_slice[obj_start..].find('}').ok_or_else(|| {
                DuoAttnError::SerializationError("unterminated head object".into())
            })?;
            let obj = &heads_slice[obj_start..=obj_start + obj_end_rel];
            let attn_layer_idx = extract_json_u64(obj, "attn_layer_idx")? as usize;
            let kv_head_idx = extract_json_u64(obj, "kv_head_idx")? as usize;
            let alpha = extract_json_f64(obj, "alpha")? as f32;
            let role_str = extract_json_string(obj, "role")?;
            let role = match role_str.as_str() {
                "retrieval" => HeadRole::Retrieval,
                "streaming" => HeadRole::Streaming,
                other => {
                    return Err(DuoAttnError::SerializationError(format!(
                        "invalid head role '{other}'"
                    )));
                }
            };
            heads.push(HeadScore {
                attn_layer_idx,
                kv_head_idx,
                alpha,
                role,
            });
            cursor = obj_start + obj_end_rel + 1;
        }

        let profile = Self {
            version,
            model_checkpoint_id,
            num_layers,
            num_attn_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            sink_tokens,
            window_tokens,
            retrieval_ratio,
            threshold,
            calibration_seed,
            calibration_steps,
            l1_lambda,
            calibration_loss_initial,
            calibration_loss_final,
            heads,
            checksum_sha256,
        };
        profile.verify_integrity()?;
        Ok(profile)
    }
}

fn classify_heads(
    num_attn_layers: usize,
    num_kv_heads: usize,
    alphas: &[f32],
    retrieval_ratio: f32,
) -> (Vec<HeadScore>, f32) {
    let total = num_attn_layers * num_kv_heads;
    let clamped: Vec<f32> = alphas
        .iter()
        .map(|&a| {
            if a.is_finite() {
                a.clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();

    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&i, &j| clamped[j].total_cmp(&clamped[i]).then_with(|| i.cmp(&j)));

    let target_retrieval = if retrieval_ratio <= 0.0 {
        0
    } else if retrieval_ratio >= 1.0 {
        total
    } else {
        ((retrieval_ratio as f64) * (total as f64))
            .round()
            .clamp(1.0, total as f64) as usize
    };

    let mut is_retrieval = vec![false; total];
    for &idx in order.iter().take(target_retrieval) {
        is_retrieval[idx] = true;
    }

    let threshold = if target_retrieval == 0 {
        1.0
    } else if target_retrieval == total {
        0.0
    } else {
        let lowest_ret = clamped[order[target_retrieval - 1]];
        let highest_str = clamped[order[target_retrieval]];
        0.5 * (lowest_ret + highest_str)
    };

    let mut heads = Vec::with_capacity(total);
    for layer in 0..num_attn_layers {
        for kv_head in 0..num_kv_heads {
            let idx = layer * num_kv_heads + kv_head;
            heads.push(HeadScore {
                attn_layer_idx: layer,
                kv_head_idx: kv_head,
                alpha: clamped[idx],
                role: if is_retrieval[idx] {
                    HeadRole::Retrieval
                } else {
                    HeadRole::Streaming
                },
            });
        }
    }
    (heads, threshold)
}

/// Escapes special characters for JSON string literals.
#[must_use]
pub fn escape_json_str(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn find_json_value<'a>(raw: &'a str, key: &str) -> Result<&'a str, DuoAttnError> {
    let needle = format!("\"{key}\"");
    let pos = raw
        .find(&needle)
        .ok_or_else(|| DuoAttnError::SerializationError(format!("missing JSON field '{key}'")))?;
    let after_key = &raw[pos + needle.len()..];
    let colon = after_key.find(':').ok_or_else(|| {
        DuoAttnError::SerializationError(format!("missing ':' after JSON field '{key}'"))
    })?;
    Ok(after_key[colon + 1..].trim_start())
}

fn extract_json_string(raw: &str, key: &str) -> Result<String, DuoAttnError> {
    let rest = find_json_value(raw, key)?;
    let rest = rest.strip_prefix('"').ok_or_else(|| {
        DuoAttnError::SerializationError(format!("expected string for field '{key}'"))
    })?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Ok(out),
            '\\' => {
                let esc = chars.next().ok_or_else(|| {
                    DuoAttnError::SerializationError(format!("unterminated escape in '{key}'"))
                })?;
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    other => out.push(other),
                }
            }
            c => out.push(c),
        }
    }
    Err(DuoAttnError::SerializationError(format!(
        "unterminated string for field '{key}'"
    )))
}

fn extract_json_number_token<'a>(raw: &'a str, key: &str) -> Result<&'a str, DuoAttnError> {
    let rest = find_json_value(raw, key)?;
    let end = rest
        .find(|c: char| {
            !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E'
        })
        .unwrap_or(rest.len());
    if end == 0 {
        return Err(DuoAttnError::SerializationError(format!(
            "expected numeric value for field '{key}'"
        )));
    }
    Ok(&rest[..end])
}

fn extract_json_u64(raw: &str, key: &str) -> Result<u64, DuoAttnError> {
    let tok = extract_json_number_token(raw, key)?;
    tok.parse::<u64>().map_err(|e| {
        DuoAttnError::SerializationError(format!("invalid u64 for field '{key}' ('{tok}'): {e}"))
    })
}

fn extract_json_f64(raw: &str, key: &str) -> Result<f64, DuoAttnError> {
    let tok = extract_json_number_token(raw, key)?;
    tok.parse::<f64>().map_err(|e| {
        DuoAttnError::SerializationError(format!("invalid f64 for field '{key}' ('{tok}'): {e}"))
    })
}
