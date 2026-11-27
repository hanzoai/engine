// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Strict 7-field composite prefix cache keys and zero-leak tenant authorization.
//!
//! Every [`PrefixCacheKey`] binds all seven compatibility and security dimensions
//! through domain-separated FIPS 180-4 SHA-256 hashing:
//! 1. `model_checkpoint_id`
//! 2. `tokenizer_hash`
//! 3. `quant_format`
//! 4. `rope_pos_encoding`
//! 5. `runtime_config_hash`
//! 6. `tenant_scope_acl`
//! 7. `token_prefix_chain_hash`

use crate::crypto::{Digest256, Sha256};
use crate::error::PrefixCacheError;

/// Domain tag for compatibility profile digest calculation.
const DOMAIN_COMPAT_V1: &[u8] = b"hanzo.prefixcache.compat.v1";
/// Domain tag for tenant scope namespace digest calculation.
const DOMAIN_SCOPE_V1: &[u8] = b"hanzo.prefixcache.scope.v1";
/// Domain tag for incremental token prefix chain calculation.
const DOMAIN_TOKEN_CHAIN_V1: &[u8] = b"hanzo.prefixcache.token_chain.v1";
/// Domain tag for full 7-field composite cache key calculation.
const DOMAIN_PREFIX_KEY_V1: &[u8] = b"hanzo.prefixcache.key.v1";

/// Unique identity of a model checkpoint (model identifier + immutable weight digest).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelCheckpointId {
    /// Canonical model identifier (e.g., `"hanzo-zen-9b-hybrid"`).
    pub model_id: String,
    /// Immutable SHA-256 digest of checkpoint weights/manifest.
    pub checkpoint_digest: Digest256,
}

impl ModelCheckpointId {
    /// Construct a checkpoint identifier from a model name and weight manifest string.
    #[must_use]
    pub fn from_manifest(model_id: impl Into<String>, manifest_bytes: &[u8]) -> Self {
        Self {
            model_id: model_id.into(),
            checkpoint_digest: Sha256::digest(manifest_bytes),
        }
    }
}

/// Unique identity of the tokenizer and vocabulary table.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TokenizerIdentity {
    /// Tokenizer family or identifier (e.g., `"llama3-tiktoken-128k"`).
    pub tokenizer_id: String,
    /// SHA-256 digest of tokenizer vocabulary and special-token map.
    pub vocab_hash: Digest256,
}

impl TokenizerIdentity {
    /// Construct a tokenizer identity by hashing raw tokenizer definition bytes.
    #[must_use]
    pub fn from_vocab_bytes(tokenizer_id: impl Into<String>, vocab_bytes: &[u8]) -> Self {
        Self {
            tokenizer_id: tokenizer_id.into(),
            vocab_hash: Sha256::digest(vocab_bytes),
        }
    }
}

/// KV cache quantization format bound into cache keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// Unquantized IEEE-754 half-precision (`FP16`) KV cache.
    Fp16,
    /// Unquantized brain-float (`BF16`) KV cache.
    Bf16,
    /// `FP8 E4M3` KV cache.
    Fp8E4M3,
    /// Symmetric per-token `INT8` KV cache (`hanzo-paged-attn::quant`).
    Int8Symmetric,
    /// Asymmetric per-channel K / per-token V 4-bit KIVI (`hanzo-paged-attn::kivi`).
    KiviInt4 {
        /// Key quantization group size `G`.
        group_size: u16,
        /// Sliding unquantized residual length `R`.
        residual_len: u16,
    },
    /// Asymmetric per-channel K / per-token V 2-bit KIVI (`hanzo-paged-attn::kivi`).
    KiviInt2 {
        /// Key quantization group size `G`.
        group_size: u16,
        /// Sliding unquantized residual length `R`.
        residual_len: u16,
    },
    /// Custom quantization tag for experimental formats.
    Custom(String),
}

impl QuantFormat {
    /// Bytes per scalar element for KV attention storage (nominal).
    #[must_use]
    pub fn kv_bits_per_elem(&self) -> f64 {
        match self {
            Self::Fp16 | Self::Bf16 => 16.0,
            Self::Fp8E4M3 | Self::Int8Symmetric => 8.25,
            Self::KiviInt4 { .. } => 4.38,
            Self::KiviInt2 { .. } => 2.25,
            Self::Custom(_) => 16.0,
        }
    }

    /// Canonical label for reports and JSON output.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Fp16 => "fp16".to_string(),
            Self::Bf16 => "bf16".to_string(),
            Self::Fp8E4M3 => "fp8-e4m3".to_string(),
            Self::Int8Symmetric => "int8-sym".to_string(),
            Self::KiviInt4 {
                group_size,
                residual_len,
            } => format!("kivi-int4(G={group_size},R={residual_len})"),
            Self::KiviInt2 {
                group_size,
                residual_len,
            } => format!("kivi-int2(G={group_size},R={residual_len})"),
            Self::Custom(tag) => format!("custom({tag})"),
        }
    }

    fn feed_hash(&self, h: &mut Sha256) {
        match self {
            Self::Fp16 => h.update_u32(1),
            Self::Bf16 => h.update_u32(2),
            Self::Fp8E4M3 => h.update_u32(3),
            Self::Int8Symmetric => h.update_u32(4),
            Self::KiviInt4 {
                group_size,
                residual_len,
            } => {
                h.update_u32(5);
                h.update_u32(u32::from(*group_size));
                h.update_u32(u32::from(*residual_len));
            }
            Self::KiviInt2 {
                group_size,
                residual_len,
            } => {
                h.update_u32(6);
                h.update_u32(u32::from(*group_size));
                h.update_u32(u32::from(*residual_len));
            }
            Self::Custom(name) => {
                h.update_u32(7);
                h.update_len_prefixed(name.as_bytes());
            }
        }
    }
}

/// Rotary position encoding scaling variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeScheme {
    /// Standard NeoX/Llama RoPE without context scaling.
    Standard,
    /// Linear frequency interpolation scaling.
    Linear,
    /// Dynamic NTK-aware scaling.
    DynamicNtk,
    /// YaRN extended-context scaling.
    Yarn,
    /// Llama-3 smoothstep frequency scaling.
    Llama3,
}

impl RopeScheme {
    const fn tag(self) -> u32 {
        match self {
            Self::Standard => 1,
            Self::Linear => 2,
            Self::DynamicNtk => 3,
            Self::Yarn => 4,
            Self::Llama3 => 5,
        }
    }
}

/// Position encoding parameters bound into cache keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RopePosEncoding {
    /// RoPE scaling variant.
    pub scheme: RopeScheme,
    /// IEEE-754 bit pattern of `rope_theta` (`f64::to_bits`).
    pub theta_bits: u64,
    /// IEEE-754 bit pattern of scaling factor (`f64::to_bits`).
    pub scaling_factor_bits: u64,
    /// Maximum sequence length configured for RoPE tables.
    pub max_position_embeddings: u32,
}

impl RopePosEncoding {
    /// Construct a RoPE configuration from floating-point parameters.
    #[must_use]
    pub fn new(
        scheme: RopeScheme,
        theta: f64,
        scaling_factor: f64,
        max_position_embeddings: u32,
    ) -> Self {
        Self {
            scheme,
            theta_bits: theta.to_bits(),
            scaling_factor_bits: scaling_factor.to_bits(),
            max_position_embeddings,
        }
    }

    fn feed_hash(&self, h: &mut Sha256) {
        h.update_u32(self.scheme.tag());
        h.update_u64(self.theta_bits);
        h.update_u64(self.scaling_factor_bits);
        h.update_u32(self.max_position_embeddings);
    }
}

/// Runtime execution configuration affecting KV/GDN memory layout and numerics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeConfigIdentity {
    /// Fixed number of tokens per prefix cache block (e.g., `64`).
    pub block_size: u32,
    /// Number of paged attention layers in the model.
    pub attn_layers: u16,
    /// Number of recurrent Gated DeltaNet (GDN) layers in the model.
    pub gdn_layers: u16,
    /// Number of KV heads per attention layer.
    pub kv_heads: u16,
    /// Attention head dimension.
    pub head_dim: u16,
    /// Recurrent GDN state dimension per head.
    pub gdn_state_dim: u16,
    /// Hash of additional runtime flags (e.g., deterministic math, sliding window, GDN precision).
    pub runtime_flags_hash: Digest256,
}

impl RuntimeConfigIdentity {
    fn feed_hash(&self, h: &mut Sha256) {
        h.update_u32(self.block_size);
        h.update_u32(u32::from(self.attn_layers));
        h.update_u32(u32::from(self.gdn_layers));
        h.update_u32(u32::from(self.kv_heads));
        h.update_u32(u32::from(self.head_dim));
        h.update_u32(u32::from(self.gdn_state_dim));
        h.update(self.runtime_flags_hash.as_bytes());
    }
}

/// Five model/runtime compatibility dimensions required for safe KV+GDN reuse across workers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompatibilityProfile {
    /// Field 1: Model & checkpoint identity.
    pub model_checkpoint: ModelCheckpointId,
    /// Field 2: Tokenizer & vocabulary identity.
    pub tokenizer: TokenizerIdentity,
    /// Field 3: KV quantization format.
    pub quant_format: QuantFormat,
    /// Field 4: Position encoding parameters.
    pub rope_pos_encoding: RopePosEncoding,
    /// Field 5: Runtime geometry and configuration hash.
    pub runtime_config: RuntimeConfigIdentity,
}

impl CompatibilityProfile {
    /// Compute the canonical SHA-256 digest of all five model/runtime compatibility fields.
    #[must_use]
    pub fn canonical_digest(&self) -> Digest256 {
        let mut h = Sha256::new();
        h.update_len_prefixed(DOMAIN_COMPAT_V1);
        h.update_len_prefixed(self.model_checkpoint.model_id.as_bytes());
        h.update(self.model_checkpoint.checkpoint_digest.as_bytes());
        h.update_len_prefixed(self.tokenizer.tokenizer_id.as_bytes());
        h.update(self.tokenizer.vocab_hash.as_bytes());
        self.quant_format.feed_hash(&mut h);
        self.rope_pos_encoding.feed_hash(&mut h);
        self.runtime_config.feed_hash(&mut h);
        h.finalize()
    }
}

/// Authorization namespace for a prefix cache block.
///
/// Private tenant prefixes (`TenantPrivate`) are strictly isolated by tenant namespace.
/// Shared scopes (`PublicSystem`, `OrgShared`) require explicit ACL authorization and
/// can never absorb mutable private turns.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TenantScope {
    /// Immutable public system prompts shared across authorized callers.
    PublicSystem,
    /// Organization-shared prefixes (e.g., shared repository index or org coding rubric).
    OrgShared {
        /// Unique organization identifier.
        org_id: String,
    },
    /// Strictly isolated private prefix owned by a single tenant.
    TenantPrivate {
        /// Unique tenant identifier.
        tenant_id: String,
    },
}

impl TenantScope {
    /// Returns `true` if this scope is shared across more than one tenant.
    #[must_use]
    pub const fn is_shared(&self) -> bool {
        matches!(self, Self::PublicSystem | Self::OrgShared { .. })
    }

    /// Compute the canonical domain-separated SHA-256 digest of this tenant scope.
    #[must_use]
    pub fn canonical_digest(&self) -> Digest256 {
        let mut h = Sha256::new();
        h.update_len_prefixed(DOMAIN_SCOPE_V1);
        match self {
            Self::PublicSystem => {
                h.update_u32(1);
            }
            Self::OrgShared { org_id } => {
                h.update_u32(2);
                h.update_len_prefixed(org_id.as_bytes());
            }
            Self::TenantPrivate { tenant_id } => {
                h.update_u32(3);
                h.update_len_prefixed(tenant_id.as_bytes());
            }
        }
        h.finalize()
    }
}

/// Caller authorization credentials evaluated on every read, write, and routing decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TenantAuthorization {
    /// Calling tenant identifier.
    pub tenant_id: String,
    /// Organizations for which the caller holds read membership.
    pub org_read_memberships: Vec<String>,
    /// Organizations for which the caller holds explicit publish/write rights.
    pub org_write_grants: Vec<String>,
    /// Whether the caller is permitted to reuse `PublicSystem` prefixes.
    pub allow_public_system_read: bool,
    /// Whether the caller is permitted to publish new `PublicSystem` prefixes.
    pub allow_public_system_write: bool,
}

impl TenantAuthorization {
    /// Standard developer tenant with private read/write, optional org read, and public system read.
    #[must_use]
    pub fn standard_tenant(
        tenant_id: impl Into<String>,
        org_read_memberships: Vec<String>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            org_read_memberships,
            org_write_grants: Vec::new(),
            allow_public_system_read: true,
            allow_public_system_write: false,
        }
    }

    /// Organization publisher authorized to populate `PublicSystem` and `OrgShared` prefixes.
    #[must_use]
    pub fn publisher(
        tenant_id: impl Into<String>,
        org_ids: Vec<String>,
        allow_public_system_write: bool,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            org_read_memberships: org_ids.clone(),
            org_write_grants: org_ids,
            allow_public_system_read: true,
            allow_public_system_write,
        }
    }

    /// Enforce read authorization for `scope`.
    pub fn authorize_read(&self, scope: &TenantScope) -> Result<(), PrefixCacheError> {
        match scope {
            TenantScope::PublicSystem => {
                if self.allow_public_system_read {
                    Ok(())
                } else {
                    Err(PrefixCacheError::UnauthorizedTenantAccess {
                        caller_tenant: self.tenant_id.clone(),
                        reason: "caller lacks PublicSystem read grant".to_string(),
                    })
                }
            }
            TenantScope::OrgShared { org_id } => {
                if self.org_read_memberships.iter().any(|o| o == org_id) {
                    Ok(())
                } else {
                    Err(PrefixCacheError::UnauthorizedTenantAccess {
                        caller_tenant: self.tenant_id.clone(),
                        reason: format!("caller is not a member of OrgShared('{org_id}')"),
                    })
                }
            }
            TenantScope::TenantPrivate { tenant_id } => {
                if &self.tenant_id == tenant_id {
                    Ok(())
                } else {
                    Err(PrefixCacheError::UnauthorizedTenantAccess {
                        caller_tenant: self.tenant_id.clone(),
                        reason: format!(
                            "cross-tenant read denied for TenantPrivate('{tenant_id}')"
                        ),
                    })
                }
            }
        }
    }

    /// Enforce write authorization for `scope`.
    pub fn authorize_write(&self, scope: &TenantScope) -> Result<(), PrefixCacheError> {
        match scope {
            TenantScope::PublicSystem => {
                if self.allow_public_system_write {
                    Ok(())
                } else {
                    Err(PrefixCacheError::UnauthorizedTenantAccess {
                        caller_tenant: self.tenant_id.clone(),
                        reason: "caller lacks PublicSystem publish grant".to_string(),
                    })
                }
            }
            TenantScope::OrgShared { org_id } => {
                if self.org_write_grants.iter().any(|o| o == org_id) {
                    Ok(())
                } else {
                    Err(PrefixCacheError::UnauthorizedTenantAccess {
                        caller_tenant: self.tenant_id.clone(),
                        reason: format!("caller lacks write grant for OrgShared('{org_id}')"),
                    })
                }
            }
            TenantScope::TenantPrivate { tenant_id } => {
                if &self.tenant_id == tenant_id {
                    Ok(())
                } else {
                    Err(PrefixCacheError::UnauthorizedTenantAccess {
                        caller_tenant: self.tenant_id.clone(),
                        reason: format!(
                            "cross-tenant write denied for TenantPrivate('{tenant_id}')"
                        ),
                    })
                }
            }
        }
    }
}

/// Enforce monotonic scope boundaries along a prefix block chain.
///
/// Allowed transitions:
/// - `PublicSystem` -> `PublicSystem`, `OrgShared(_)`, `TenantPrivate(_)`
/// - `OrgShared(A)` -> `OrgShared(A)`, `TenantPrivate(_)`
/// - `TenantPrivate(T)` -> `TenantPrivate(T)`
///
/// Disallowed (always rejected):
/// - Any private block transitioning back into a shared scope (prevents private-state blending).
/// - Cross-org transition `OrgShared(A) -> OrgShared(B)`.
/// - Cross-tenant transition `TenantPrivate(T1) -> TenantPrivate(T2)`.
pub fn validate_scope_transition(
    parent: Option<&TenantScope>,
    child: &TenantScope,
) -> Result<(), PrefixCacheError> {
    let Some(parent_scope) = parent else {
        return Ok(());
    };
    match (parent_scope, child) {
        (TenantScope::PublicSystem, _) => Ok(()),
        (TenantScope::OrgShared { org_id: p }, TenantScope::OrgShared { org_id: c }) if p == c => {
            Ok(())
        }
        (TenantScope::OrgShared { .. }, TenantScope::TenantPrivate { .. }) => Ok(()),
        (
            TenantScope::TenantPrivate { tenant_id: p },
            TenantScope::TenantPrivate { tenant_id: c },
        ) if p == c => Ok(()),
        (TenantScope::TenantPrivate { tenant_id }, TenantScope::PublicSystem)
        | (TenantScope::TenantPrivate { tenant_id }, TenantScope::OrgShared { .. }) => {
            Err(PrefixCacheError::ScopeBoundaryViolation {
                reason: format!(
                    "cannot blend mutable private turn of tenant '{tenant_id}' into shared scope {child:?}"
                ),
            })
        }
        (TenantScope::OrgShared { org_id: p }, TenantScope::PublicSystem) => {
            Err(PrefixCacheError::ScopeBoundaryViolation {
                reason: format!(
                    "cannot widen OrgShared('{p}') prefix back into PublicSystem scope"
                ),
            })
        }
        (TenantScope::OrgShared { org_id: p }, TenantScope::OrgShared { org_id: c }) => {
            Err(PrefixCacheError::ScopeBoundaryViolation {
                reason: format!("cross-organization prefix transition '{p}' -> '{c}' denied"),
            })
        }
        (
            TenantScope::TenantPrivate { tenant_id: p },
            TenantScope::TenantPrivate { tenant_id: c },
        ) => Err(PrefixCacheError::ScopeBoundaryViolation {
            reason: format!("cross-tenant prefix transition '{p}' -> '{c}' denied"),
        }),
    }
}

/// Incremental hash chain over block-aligned token sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenPrefixChain {
    /// Current chained SHA-256 digest after all preceding blocks.
    pub digest: Digest256,
    /// Number of full blocks folded into this chain.
    pub block_index: u32,
}

impl Default for TokenPrefixChain {
    fn default() -> Self {
        Self::root()
    }
}

impl TokenPrefixChain {
    /// Empty root chain before block 0.
    #[must_use]
    pub const fn root() -> Self {
        Self {
            digest: Digest256::ZERO,
            block_index: 0,
        }
    }

    /// Fold the next token block into the chain, returning the updated chain state.
    #[must_use]
    pub fn extend(&self, block_tokens: &[u32]) -> Self {
        let mut h = Sha256::new();
        h.update_len_prefixed(DOMAIN_TOKEN_CHAIN_V1);
        h.update(self.digest.as_bytes());
        h.update_u32(self.block_index);
        h.update_u32(block_tokens.len() as u32);
        for &tok in block_tokens {
            h.update_u32(tok);
        }
        Self {
            digest: h.finalize(),
            block_index: self.block_index + 1,
        }
    }
}

/// Canonical 7-field composite prefix cache key.
///
/// Uniquely identifies a single block position in the prefix tree while binding
/// model/checkpoint, tokenizer, quantization, RoPE encoding, runtime config,
/// tenant scope ACL, and full preceding token prefix chain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrefixCacheKey {
    /// Digest of the 5-field compatibility profile.
    pub compat_digest: Digest256,
    /// Digest of the tenant authorization scope.
    pub scope_digest: Digest256,
    /// Chained digest of all tokens from position 0 through the end of this block.
    pub token_chain_digest: Digest256,
    /// Zero-based block index along the sequence.
    pub block_index: u32,
    /// Final canonical SHA-256 key digest binding all 7 dimensions.
    pub key_digest: Digest256,
}

impl PrefixCacheKey {
    /// Derive a canonical 7-field [`PrefixCacheKey`] from its compatibility profile,
    /// tenant scope, and token prefix chain state.
    #[must_use]
    pub fn derive(
        compat: &CompatibilityProfile,
        scope: &TenantScope,
        chain: &TokenPrefixChain,
    ) -> Self {
        let compat_digest = compat.canonical_digest();
        let scope_digest = scope.canonical_digest();
        let mut h = Sha256::new();
        h.update_len_prefixed(DOMAIN_PREFIX_KEY_V1);
        h.update(compat_digest.as_bytes());
        h.update(scope_digest.as_bytes());
        h.update(chain.digest.as_bytes());
        h.update_u32(chain.block_index);
        Self {
            compat_digest,
            scope_digest,
            token_chain_digest: chain.digest,
            block_index: chain.block_index,
            key_digest: h.finalize(),
        }
    }

    /// Verify that this key matches the expected compatibility profile and that
    /// the caller is authorized to read `claimed_scope`.
    pub fn verify_access(
        &self,
        expected_compat: &CompatibilityProfile,
        claimed_scope: &TenantScope,
        auth: &TenantAuthorization,
    ) -> Result<(), PrefixCacheError> {
        if !self
            .compat_digest
            .ct_eq(&expected_compat.canonical_digest())
        {
            return Err(PrefixCacheError::CompatibilityMismatch);
        }
        if !self.scope_digest.ct_eq(&claimed_scope.canonical_digest()) {
            return Err(PrefixCacheError::UnauthorizedTenantAccess {
                caller_tenant: auth.tenant_id.clone(),
                reason: "key scope digest does not match claimed tenant scope".to_string(),
            });
        }
        auth.authorize_read(claimed_scope)
    }
}

/// Per-block specification pairing a token block slice with its intended tenant scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedTokenBlock<'a> {
    /// Exact `block_size` tokens belonging to this block.
    pub tokens: &'a [u32],
    /// Authorization scope for this block.
    pub scope: TenantScope,
}

/// Partition a flat token sequence and scope boundaries into aligned [`ScopedTokenBlock`]s,
/// validating monotonic scope transitions at every block boundary.
///
/// `public_prefix_blocks` blocks are tagged [`TenantScope::PublicSystem`], followed by
/// `org_prefix_blocks` tagged [`TenantScope::OrgShared`] (if `org_id` is `Some`), and
/// all remaining full blocks tagged [`TenantScope::TenantPrivate`].
pub fn partition_scoped_blocks<'a>(
    tokens: &'a [u32],
    block_size: usize,
    public_prefix_blocks: usize,
    org_prefix_blocks: usize,
    org_id: Option<&str>,
    tenant_id: &str,
) -> Result<Vec<ScopedTokenBlock<'a>>, PrefixCacheError> {
    if block_size == 0 {
        return Err(PrefixCacheError::InvalidBlockAlignment {
            reason: "block_size must be non-zero".to_string(),
        });
    }
    let full_blocks = tokens.len() / block_size;
    let mut out = Vec::with_capacity(full_blocks);
    let mut prev_scope: Option<TenantScope> = None;

    for idx in 0..full_blocks {
        let start = idx * block_size;
        let end = start + block_size;
        let scope = if idx < public_prefix_blocks {
            TenantScope::PublicSystem
        } else if idx < public_prefix_blocks + org_prefix_blocks {
            if let Some(org) = org_id {
                TenantScope::OrgShared {
                    org_id: org.to_string(),
                }
            } else {
                TenantScope::TenantPrivate {
                    tenant_id: tenant_id.to_string(),
                }
            }
        } else {
            TenantScope::TenantPrivate {
                tenant_id: tenant_id.to_string(),
            }
        };
        validate_scope_transition(prev_scope.as_ref(), &scope)?;
        prev_scope = Some(scope.clone());
        out.push(ScopedTokenBlock {
            tokens: &tokens[start..end],
            scope,
        });
    }
    Ok(out)
}
